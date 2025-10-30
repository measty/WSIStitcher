"""Command-line interface for whole-slide inference."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
import yaml
import numpy as np

from tiatoolbox.wsicore import WSIReader
from predictor import predict_wsi, save_prediction

def _is_pattern(path_str: str | None) -> bool:
    if not path_str:
        return False
    return any(ch in path_str for ch in "*?[]")


def _expand_paths(spec: str | None) -> list[Path]:
    if spec is None:
        return []
    expanded = os.path.expanduser(spec)
    if _is_pattern(expanded):
        matches = sorted(Path(p) for p in glob.glob(expanded, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {spec}")
        return matches
    return [Path(expanded)]


def _derive_output_path(pattern: str, wsi_path: Path) -> Path:
    expanded = os.path.expanduser(pattern)
    if not _is_pattern(expanded):
        return Path(expanded)
    stem = wsi_path.stem
    replaced = expanded.replace("*", stem)
    if _is_pattern(replaced):
        raise ValueError(
            "Output pattern must resolve to a concrete filename after substitution"
        )
    return Path(replaced)

class DummyModel(torch.nn.Module):
    """A dummy 'pass through' model that returns the input as output."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
def load_model(checkpoint_path: Path, config: dict) -> torch.nn.Module:
    """Add your model loading code here.
    
    Args:
        checkpoint_path: Path to model weights.
        config: Configuration dictionary (from YAML) if needed by your model.
    Returns:
        torch.nn.Module: The loaded model.

    Example:
        from your_code import ModelClass
        model = ModelClass(**config['model_params'])
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model
    """
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run model inference on a WSI")
    p.add_argument("--checkpoint", type=str, help="Path to trained model weights, or none will make a dummy 'pass through' model for format conversion/masked rebuilding etc.")
    p.add_argument(
        "--wsi",
        type=str,
        help="Path or glob pattern to whole-slide image(s)",
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Model configuration YAML (if needed by your model). Uses one in checkpt dir if not provided.",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size in pixels (defaults to training config)",
    )
    p.add_argument("--overlap", type=int, default=0, help="Tile overlap in pixels")
    p.add_argument("--read-mpp", type=float, help="Microns-per-pixel to read at")
    p.add_argument("--save-mpp", type=float, help="Microns-per-pixel to save output WSI at. If omitted, uses slide baseline mpp.")
    p.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for inference"
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="prediction.npy",
        help="Output file (.npy or .tif) or glob pattern for batch inference",
    )
    p.add_argument("--device", default=None, help="cpu or cuda, default auto-select")
    p.add_argument(
        "--memmap-dir", type=Path, help="Directory for memmap intermediate storage"
    )
    p.add_argument(
        "--channels",
        nargs="+",
        help="Optional list of channel names for multi-channel WSI inputs",
    )
    p.add_argument(
        "--background-colour",
        type=str,
        default="white",
        help="Background colour for the output image ('white' or 'black')",
    )
    p.add_argument(
        "--mask", type=str, default='otsu', help="tiatoolbox Tissue mask method (e.g., 'otsu'), path to a mask file, or none for no masking; passed to tiatoolbox SlidingWindowPatchExtractor.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.checkpoint is None or args.checkpoint.lower() == "none":
        # using dummy passthrough model
        args.checkpoint = None
    else:
        args.checkpoint = Path(args.checkpoint)
        ckpt_config = args.checkpoint.parent / "config.yaml"
        if args.config is None:
            args.config = ckpt_config
    try:
        with open(args.config, "r") as fh:
            cfg = yaml.safe_load(fh)
    except Exception:
        cfg = {}
        print("Warning: could not load configuration file")
    if args.mask.lower() == "none":
        args.mask = None

    wsi_paths = _expand_paths(str(args.wsi))

    if args.checkpoint is None:
        model = DummyModel()
    else:
        model = load_model(args.checkpoint, cfg)

    read_mpp = args.read_mpp
    if read_mpp is None:
        read_cfg = cfg.get("data", {}).get("mpp")
        if read_cfg is None:
            raise ValueError("Unable to determine read_mpp from configuration, provide in CLI")
        read_vals = np.asarray(read_cfg, dtype=float).flatten()
        if read_vals.size == 0:
            raise ValueError("Configured read_mpp is empty")
        read_mpp = float(read_vals[0])

    output_pattern = args.output
    multiple_slides = len(wsi_paths) > 1
    if multiple_slides and not _is_pattern(output_pattern):
        raise ValueError(
            "When processing multiple slides, --output must include a glob pattern"
        )

    channel_names = args.channels
    if channel_names is None:
        inferred = cfg.get("data", {}).get("channels")
        if isinstance(inferred, str):
            if inferred.lower() != "all":
                channel_names = [inferred]
        elif isinstance(inferred, (list, tuple)):
            channel_names = [str(g) for g in inferred]

    for wsi_path in wsi_paths:
        print(f"Running inference on {wsi_path}")
        reader = WSIReader.open(wsi_path)

        save_mpp = args.save_mpp
        if save_mpp is None:
            mpp_meta = getattr(reader.info, "mpp", None)
            if mpp_meta is None:
                raise ValueError("Slide is missing baseline mpp metadata")
            save_vals = np.asarray(mpp_meta, dtype=float).flatten()
            if save_vals.size == 0:
                raise ValueError("Slide mpp metadata is empty")
            print("using slide baseline mpp for save_mpp")
            save_mpp = float(save_vals[0])

        tile_size = args.tile_size
        if tile_size is None:
            tile_cfg = cfg.get("data", {}).get("tile_size")
            if tile_cfg is None:
                print(
                    "Tile size missing from configuration; defaulting to 512 pixels"
                )
                tile_cfg = 512
            if isinstance(tile_cfg, (tuple, list)):
                if not tile_cfg:
                    raise ValueError("tile_size configuration is empty")
                tile_cfg = tile_cfg[0]
            tile_size = int(tile_cfg)

        prediction = predict_wsi(
            model,
            reader,
            tile_size,
            overlap=args.overlap,
            read_mpp=read_mpp,
            save_mpp=save_mpp,
            batch_size=args.batch_size,
            device=args.device,
            memmap_dir=args.memmap_dir,
            background_colour=args.background_colour,
            tissue_mask=args.mask,
        )

        output_path = _derive_output_path(output_pattern, wsi_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_prediction(
                prediction, output_path, save_mpp=save_mpp, channels=channel_names
            )
            print(f"Saved prediction to {output_path}")
        finally:
            prediction.close()
            close_fn = getattr(reader, "close", None)
            if callable(close_fn):
                close_fn()

    print("Inference complete")


if __name__ == "__main__":
    main()
