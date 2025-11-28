"""High-level whole-slide inference utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from tiatoolbox.wsicore import WSIReader
from utils import get_logger, _generate_colors, _prepare_ome_xml

from stitcher import ChunkedStitcher, PredictionCanvas
from tiler import PatchInfo, WSITiler

DEFAULT_TIFF_COMPRESSION = {"compression": "jpeg", "compressionargs": {"level": 87}}

__all__ = ["predict_wsi", "save_prediction"]

logger = get_logger(__name__)


def _ensure_tuple(value: float | Sequence[float]) -> Tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            msg = "Microns-per-pixel values must have exactly two entries."
            raise ValueError(msg)
        return float(value[0]), float(value[1])
    arr = np.asarray(value, dtype=float).flatten()
    if arr.size == 1:
        scalar = float(arr[0])
        return scalar, scalar
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    msg = "Microns-per-pixel values must have one or two entries"
    raise ValueError(msg)


def _save_bounds_extent(
    patch_infos: Sequence[PatchInfo],
) -> Tuple[int, int, int, int] | None:
    if not patch_infos:
        return None
    min_x0 = min(info.save_bounds[0] for info in patch_infos)
    min_y0 = min(info.save_bounds[1] for info in patch_infos)
    max_x1 = max(info.save_bounds[2] for info in patch_infos)
    max_y1 = max(info.save_bounds[3] for info in patch_infos)
    return min_x0, min_y0, max_x1, max_y1


def predict_wsi(
    model: nn.Module,
    reader: WSIReader,
    tile_size: int,
    *,
    overlap: int = 0,
    read_mpp: float | Sequence[float] = 0.5,
    save_mpp: float | Sequence[float] = 0.5,
    batch_size: int = 1,
    mixed_precision: bool = False,
    device: str | torch.device | None = None,
    memmap_dir: str | Path | None = None,
    chunk_size: int = 8192,
    tissue_mask: str | np.array | None = "otsu",
    min_mask_ratio: float = 0.2,
    background_colour: str = "black",
    crop: bool = False,
) -> PredictionCanvas:
    """Run tiled inference on a whole-slide image and stitch results.

    Returns
    -------
    PredictionCanvas
        Lazy wrapper around the stitched prediction memmap. Use
        :meth:`PredictionCanvas.to_array` to materialize the result or pass the
        object directly to :func:`save_prediction` for zero-copy TIFF writing.
    """

    device = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device).eval()
    logger.info("Running inference on slide")

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=mixed_precision):
        dummy = torch.zeros(1, 3, tile_size, tile_size, device=device)
        dummy_out = model(dummy)
        if isinstance(dummy_out, tuple):
            # model returning (stains, rgb) - we want the latter
            out_ch = int(dummy_out[1].shape[1]) # rgb only
        else:
            out_ch = int(dummy_out.shape[1])

    read_mpp_tuple = _ensure_tuple(read_mpp)
    save_mpp_tuple = _ensure_tuple(save_mpp)

    tiler = WSITiler(
        reader,
        tile_size,
        overlap=overlap,
        read_mpp=read_mpp_tuple,
        save_mpp=save_mpp_tuple,
        tissue_mask=tissue_mask,
        min_mask_ratio=min_mask_ratio,
    )

    crop_to_mask = bool(crop and tissue_mask is not None)
    if crop_to_mask:
        extent = _save_bounds_extent(tiler.patch_infos)
        if extent is None:
            logger.warning(
                "Cropping requested but no patches were generated; using full canvas."
            )
            crop_to_mask = False
        else:
            min_x0, min_y0, max_x1, max_y1 = extent
            width = max_x1 - min_x0
            height = max_y1 - min_y0
            if width <= 0 or height <= 0:
                logger.warning(
                    "Cropping requested but computed extent is empty; using full canvas."
                )
                crop_to_mask = False
            else:
                logger.info(
                    "Cropping output canvas to masked bounds at (%d, %d) with size %dx%d",
                    min_x0,
                    min_y0,
                    width,
                    height,
                )
                tiler.rebase_save_bounds(x_offset=min_x0, y_offset=min_y0)

    if not crop_to_mask:
        slide_w, slide_h = reader.info.slide_dimensions
        base_mpp = getattr(reader.info, "mpp", read_mpp_tuple)
        base_mpp_tuple = _ensure_tuple(base_mpp)
        scale_x = base_mpp_tuple[0] / save_mpp_tuple[0]
        scale_y = base_mpp_tuple[1] / save_mpp_tuple[1]
        width = int(math.ceil(slide_w * scale_x))
        height = int(math.ceil(slide_h * scale_y))

    stitcher = ChunkedStitcher(
        shape=(height, width),
        n_channels=out_ch,
        chunk_size=chunk_size,
        memmap_dir=Path(memmap_dir) if memmap_dir else None,
        background_colour=background_colour,
    )

    chunks_x = math.ceil(width / chunk_size)
    chunks_y = math.ceil(height / chunk_size)
    total_chunks = chunks_x * chunks_y

    for bounds in tqdm(stitcher.iter_chunk_bounds(), total=total_chunks, desc="Chunks"):
        patch_indices = tiler.query_patch_indices(bounds)
        chunk_canvas, chunk_weights = stitcher.allocate_chunk_arrays(bounds)
        if not patch_indices:
            stitcher.finalize_chunk(bounds, chunk_canvas, chunk_weights)
            continue
        for start in range(0, len(patch_indices), batch_size):
            batch_indices = patch_indices[start : start + batch_size]
            batch_imgs = tiler.load_batch(batch_indices)
            if batch_imgs.size == 0:
                continue
            imgs = torch.from_numpy(batch_imgs).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=mixed_precision):
                preds = model(imgs)
                if isinstance(preds, tuple):
                    preds = preds[1]  # rgb only
            preds = preds.detach().cpu()
            for idx, pred in zip(batch_indices, preds):
                patch_info = tiler.patch_infos[int(idx)]
                target_h = patch_info.save_bounds[3] - patch_info.save_bounds[1]
                target_w = patch_info.save_bounds[2] - patch_info.save_bounds[0]
                if pred.shape[-2:] != (target_h, target_w):
                    pred = F.interpolate(
                        pred.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                # clip to ensure between 0-1
                pred = pred.clamp(0, 1)
                pred_np = pred.numpy()
                stitcher.accumulate(
                    chunk_canvas,
                    chunk_weights,
                    pred_np,
                    patch_info.save_bounds,
                    bounds,
                )
        stitcher.finalize_chunk(bounds, chunk_canvas, chunk_weights)

    canvas = stitcher.finalize()
    logger.info("Inference finished")
    return canvas


def save_prediction(
    arr: PredictionCanvas | np.ndarray,
    path: str | Path,
    *,
    save_mpp: float | Sequence[float] | None = None,
    pyramid_min_dim: int = 512,
    channels: Sequence[str] | None = None,
    fliplr: bool = False,
) -> None:
    """Save prediction array to ``.npy`` or pyramidal TIFF.

    Parameters
    ----------
    arr:
        Array-like prediction data or a :class:`~spatx.inference.stitcher.PredictionCanvas`
        returned by :func:`predict_wsi`. Channel-first (``CxHxW``) and channel-last
        (``HxWxC``) layouts are both supported.
    path:
        Destination file path. ``.npy``/``.npz`` are stored as NumPy arrays
        while ``.tif`` outputs a pyramidal OME-TIFF.
    save_mpp:
        Microns-per-pixel resolution for the saved TIFF. When provided the
        corresponding TIFF resolution tags and OME metadata are populated.
    pyramid_min_dim:
        Minimum dimension (height/width) of the smallest pyramid level.
    channels:
        Optional list of channel names to embed in the OME-XML description.
        When omitted, generic channel identifiers (``channel_0``...) are used.
    """

    path = Path(path)
    canvas: PredictionCanvas | None = None
    if isinstance(arr, PredictionCanvas):
        canvas = arr
        canvas.flush()
        data_obj: np.ndarray = canvas.channel_last_view()
    else:
        data_obj = np.asarray(arr)

    if path.suffix.lower() in {".npy", ".npz"}:
        cf_data = canvas.to_array() if canvas is not None else np.asarray(data_obj)
        if cf_data.dtype != np.uint8:
            cf_data = np.clip(np.rint(cf_data), 0, 255).astype(np.uint8)
        np.save(path, cf_data)
        return

    data = data_obj
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.ndim != 3:
        msg = "Prediction array must be 2D or 3D"
        raise ValueError(msg)
    
    if fliplr:
        data = np.fliplr(data)

    def _is_channel_first(candidate: np.ndarray) -> bool:
        return (
            candidate.shape[0] <= 32
            and candidate.shape[0] < candidate.shape[1]
            and candidate.shape[0] < candidate.shape[2]
        )

    if _is_channel_first(data):
        channel_last = np.moveaxis(data, 0, -1)
    elif (
        data.shape[-1] <= 32
        and data.shape[-1] < data.shape[0]
        and data.shape[-1] < data.shape[1]
    ):
        channel_last = data
    else:
        msg = "Unable to infer channel dimension for prediction array"
        raise ValueError(msg)

    cf_data = np.moveaxis(channel_last, -1, 0)

    if cf_data.dtype != np.uint8:
        cf_data = np.clip(np.rint(cf_data), 0, 255).astype(np.uint8)
        channel_last = np.moveaxis(cf_data, 0, -1)

    n_channels = int(cf_data.shape[0])
    height_cf = int(cf_data.shape[1])
    width_cf = int(cf_data.shape[2])

    base_resolution = None
    pixel_size = (1.0, 1.0)
    if save_mpp is not None:
        sx, sy = _ensure_tuple(save_mpp)
        base_resolution = (10000.0 / sx, 10000.0 / sy)
        pixel_size = (float(sx), float(sy))

    channel_names: list[str]
    if channels is None:
        channel_names = [f"channel{idx}" for idx in range(n_channels)]
    else:
        channel_names = [str(g) for g in channels]
        if len(channel_names) != n_channels:
            msg = "Number of gene names must match channel count"
            raise ValueError(msg)

    colors = _generate_colors(n_channels)
    ome_xml = _prepare_ome_xml(
        (n_channels, height_cf, width_cf),
        cf_data.dtype,
        channel_names,
        colors,
        pixel_size,
    )

    def _save_with_tifffile(cf_array: np.ndarray) -> None:
        try:
            import tifffile as tiff
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "pyvips unavailable and tifffile import failed for TIFF output"
            ) from exc

        levels = [cf_array]
        current_level = cf_array
        while min(current_level.shape[1:]) > pyramid_min_dim:
            tensor = torch.from_numpy(current_level.astype(np.float32) / 255.0)
            down = F.interpolate(
                tensor.unsqueeze(0),
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            next_level = np.clip(np.rint(down.numpy() * 255.0), 0, 255).astype(np.uint8)
            if next_level.shape[1] < 2 or next_level.shape[2] < 2:
                break
            current_level = next_level
            levels.append(current_level)

        resolution_unit = "CENTIMETER" if base_resolution is not None else None

        def _prepare(level: np.ndarray) -> np.ndarray:
            if n_channels == 1:
                return np.ascontiguousarray(level[0])
            return np.ascontiguousarray(np.moveaxis(level, 0, -1))

        def _scaled_resolution(idx: int) -> Tuple[float, float] | None:
            if base_resolution is None:
                return None
            scale = 2**idx
            return (base_resolution[0] / scale, base_resolution[1] / scale)

        with tiff.TiffWriter(path, bigtiff=True) as writer:
            writer.write(
                _prepare(levels[0]),
                subifds=len(levels) - 1,
                photometric="minisblack",
                metadata=None,
                resolution=base_resolution,
                resolutionunit=resolution_unit,
                planarconfig="contig",
                description=ome_xml,
                **DEFAULT_TIFF_COMPRESSION,
            )
            for idx, level in enumerate(levels[1:], start=1):
                writer.write(
                    _prepare(level),
                    subfiletype=1,
                    photometric="minisblack",
                    metadata=None,
                    resolution=_scaled_resolution(idx),
                    resolutionunit=resolution_unit,
                    planarconfig="contig",
                    **DEFAULT_TIFF_COMPRESSION,
                )

    try:
        import pyvips
    except Exception:
        logger.warning(
            "pyvips is unavailable; falling back to tifffile-based pyramid writer."
        )
        _save_with_tifffile(cf_data)
        return

    vips_data = channel_last
    if vips_data.ndim == 2:
        vips_data = vips_data[:, :, np.newaxis]
    if not vips_data.flags["C_CONTIGUOUS"]:
        vips_data = np.ascontiguousarray(vips_data)

    height, width = int(vips_data.shape[0]), int(vips_data.shape[1])
    bands = int(vips_data.shape[2]) if vips_data.ndim == 3 else 1

    vips_img = pyvips.Image.new_from_memory(
        vips_data,
        width,
        height,
        bands,
        pyvips.BandFormat.UCHAR,
    )
    vips_img = vips_img.copy()
    vips_img.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml)

    compression = DEFAULT_TIFF_COMPRESSION.get("compression", "deflate")
    compression_args = DEFAULT_TIFF_COMPRESSION.get("compressionargs", {})

    def _tile_extent(dim: int) -> int:
        if dim <= 0:
            return 16
        if dim < 16:
            return 16
        extent = min(256, dim)
        remainder = extent % 16
        if remainder:
            extent = max(16, extent - remainder)
        return extent

    tile_width = _tile_extent(width)
    tile_height = _tile_extent(height)

    enable_pyramid = max(height, width) > max(1, pyramid_min_dim)

    save_kwargs = {
        "tile": True,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "bigtiff": True,
        "compression": compression,
    }
    if enable_pyramid:
        save_kwargs.update({"pyramid": True, "depth": "onepixel", "subifd": True})
    if compression == "jpeg":
        save_kwargs["Q"] = compression_args.get("level", 90)
    elif compression_args:
        save_kwargs.update(compression_args)
    if base_resolution is not None:
        mm_resolution = (base_resolution[0] / 10.0, base_resolution[1] / 10.0)
        save_kwargs.update(
            {
                "xres": mm_resolution[0],
                "yres": mm_resolution[1],
                "resunit": pyvips.enums.ForeignTiffResunit.CM,
            }
        )

    vips_img.tiffsave(str(path), **save_kwargs)
