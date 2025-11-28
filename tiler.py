"""Utilities for planning tiled inference using :mod:`tiatoolbox`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import sqlite3
from shapely.geometry import Polygon
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor

from tiatoolbox.wsicore import WSIReader

__all__ = ["PatchInfo", "WSITiler"]


def _ensure_tuple(value: float | Sequence[float]) -> Tuple[float, float]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            msg = "Microns-per-pixel values must have exactly two entries."
            raise ValueError(msg)
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


@dataclass(frozen=True)
class PatchInfo:
    """Record describing a patch extracted from a WSI."""

    index: int
    read_bounds: Tuple[int, int, int, int]
    save_bounds: Tuple[int, int, int, int]

    @property
    def save_size(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.save_bounds
        return y1 - y0, x1 - x0


class WSITiler:
    """Plan tiled inference using :class:`SlidingWindowPatchExtractor`."""

    def __init__(
        self,
        reader: WSIReader,
        tile_size: int,
        *,
        overlap: int = 0,
        read_mpp: float | Sequence[float] = 0.5,
        save_mpp: float | Sequence[float] = 0.5,
        tissue_mask: str | None = "otsu",
        min_mask_ratio: float = 0.2,
    ) -> None:
        self.reader = reader
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        if self.tile_size <= 0:
            msg = "tile_size must be positive"
            raise ValueError(msg)
        if self.overlap < 0 or self.overlap >= self.tile_size:
            msg = "overlap must be non-negative and smaller than tile_size"
            raise ValueError(msg)

        self.read_mpp = _ensure_tuple(read_mpp)
        self.save_mpp = _ensure_tuple(save_mpp)
        stride = self.tile_size - self.overlap
        extractor_kwargs = {
            "input_img": reader,
            "patch_size": (self.tile_size, self.tile_size),
            "stride": (stride, stride),
            "resolution": self.read_mpp,
            "units": "mpp",
            "within_bound": False,
            "min_mask_ratio": float(min_mask_ratio),
        }
        if tissue_mask is not None:
            extractor_kwargs["input_mask"] = tissue_mask
        self.extractor = SlidingWindowPatchExtractor(**extractor_kwargs)

        self._patch_infos = self._build_patch_infos()
        self._store = self._build_patch_index()

    # ------------------------------------------------------------------
    def _build_patch_infos(self) -> List[PatchInfo]:
        scale_x = self.read_mpp[0] / self.save_mpp[0]
        scale_y = self.read_mpp[1] / self.save_mpp[1]
        infos: List[PatchInfo] = []
        for idx, (x0, y0, x1, y1) in enumerate(self.extractor.coordinate_list):
            save_bounds = (
                int(round(x0 * scale_x)),
                int(round(y0 * scale_y)),
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
            )
            read_bounds = (int(x0), int(y0), int(x1), int(y1))
            infos.append(
                PatchInfo(index=idx, read_bounds=read_bounds, save_bounds=save_bounds)
            )
        return infos

    # ------------------------------------------------------------------
    def _build_patch_index(self) -> SQLiteStore | None:
        try:
            conn = sqlite3.connect(":memory:")
            has_attr = hasattr(conn, "enable_load_extension")
            conn.close()
            if not has_attr:
                return None
            store = SQLiteStore(":memory:")
        except (
            AttributeError,
            OSError,
        ):  # pragma: no cover - sqlite lacking extensions
            return None
        for info in self._patch_infos:
            geometry = Polygon.from_bounds(*info.save_bounds)
            store.append(
                Annotation(geometry=geometry, properties={"index": info.index})
            )
        return store

    # ------------------------------------------------------------------
    @property
    def patch_infos(self) -> Sequence[PatchInfo]:
        return self._patch_infos

    # ------------------------------------------------------------------
    @property
    def patch_store(self) -> SQLiteStore | None:
        return self._store

    # ------------------------------------------------------------------
    def rebase_save_bounds(self, *, x_offset: int = 0, y_offset: int = 0) -> None:
        """Shift patch save bounds by the provided offsets and rebuild the index."""

        if x_offset == 0 and y_offset == 0:
            return
        rebased: List[PatchInfo] = []
        for info in self._patch_infos:
            sx0, sy0, sx1, sy1 = info.save_bounds
            rebased_bounds = (
                sx0 - x_offset,
                sy0 - y_offset,
                sx1 - x_offset,
                sy1 - y_offset,
            )
            rebased.append(
                PatchInfo(
                    index=info.index,
                    read_bounds=info.read_bounds,
                    save_bounds=rebased_bounds,
                )
            )
        self._patch_infos = rebased
        self._store = self._build_patch_index()

    # ------------------------------------------------------------------
    def query_patch_indices(self, bounds: Tuple[int, int, int, int]) -> List[int]:
        if self._store is None:
            x0, y0, x1, y1 = bounds
            indices = []
            for info in self._patch_infos:
                sx0, sy0, sx1, sy1 = info.save_bounds
                ix0 = max(x0, sx0)
                iy0 = max(y0, sy0)
                ix1 = min(x1, sx1)
                iy1 = min(y1, sy1)
                if ix0 < ix1 and iy0 < iy1:
                    indices.append(info.index)
            return indices
        geometry = Polygon.from_bounds(*bounds)
        annotations = self._store.query(geometry)
        return [int(ann.properties["index"]) for ann in annotations.values()]

    # ------------------------------------------------------------------
    def load_batch(self, indices: Sequence[int]) -> np.ndarray:
        images: List[np.ndarray] = []
        for idx in indices:
            patch = self.extractor[int(idx)]
            arr = np.asarray(patch, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            arr = arr.transpose(2, 0, 1) / 255.0
            images.append(arr)
        if not images:
            return np.empty((0, 3, self.tile_size, self.tile_size), dtype=np.float32)
        return np.stack(images, axis=0)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._patch_infos)
