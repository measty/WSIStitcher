"""Chunk-based stitching utilities for whole-slide inference."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

__all__ = ["ChunkBounds", "ChunkedStitcher", "PredictionCanvas"]


ChunkBounds = Tuple[int, int, int, int]


@dataclass
class PredictionCanvas:
    """Represents the stitched uint8 canvas stored on disk."""

    memmap: np.memmap
    path: Path
    shape: Tuple[int, int, int]
    is_temporary: bool
    _tmp_dir: tempfile.TemporaryDirectory[str] | None
    _closed: bool = False

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Flush pending writes to disk."""

        if self._closed:
            return
        self.memmap.flush()

    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """Return the canvas as a channel-first ``uint8`` array."""

        self.flush()
        arr = np.asarray(self.memmap)
        cf_arr = np.moveaxis(arr, -1, 0).copy()
        return cf_arr

    # ------------------------------------------------------------------
    def channel_last_view(self) -> np.ndarray:
        """Return a channel-last view of the canvas without copying."""

        return self.memmap

    # ------------------------------------------------------------------
    def close(self, *, remove: bool | None = None) -> None:
        """Close the underlying memmap and optionally delete the file."""

        if self._closed:
            return
        remove_file = self.is_temporary if remove is None else remove
        self.flush()
        if hasattr(self.memmap, "_mmap"):
            self.memmap._mmap.close()  # type: ignore[attr-defined]
        if remove_file:
            Path(self.path).unlink(missing_ok=True)
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
        self._closed = True

    # ------------------------------------------------------------------
    def __enter__(self) -> PredictionCanvas:
        return self

    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
        self.close()

    # ------------------------------------------------------------------
    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def __array__(self) -> np.ndarray:
        return self.to_array()


@dataclass
class ChunkedStitcher:
    """Manage chunk-wise aggregation onto a uint8 canvas."""

    shape: Tuple[int, int]
    n_channels: int
    chunk_size: int
    memmap_dir: Path | None = None
    background_colour: str = "black"

    def __post_init__(self) -> None:
        height, width = self.shape
        if height <= 0 or width <= 0:
            msg = "Canvas dimensions must be positive"
            raise ValueError(msg)
        if self.n_channels <= 0:
            msg = "Number of channels must be positive"
            raise ValueError(msg)
        if self.chunk_size <= 0:
            msg = "chunk_size must be positive"
            raise ValueError(msg)

        self.height = int(height)
        self.width = int(width)
        self.chunk_size = int(self.chunk_size)
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        if self.memmap_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            canvas_dir = Path(self._tmp_dir.name)
        else:
            canvas_dir = Path(self.memmap_dir)
            canvas_dir.mkdir(parents=True, exist_ok=True)
        self._canvas_path = canvas_dir / "prediction.uint8"
        self.canvas = np.memmap(
            self._canvas_path,
            dtype=np.uint8,
            mode="w+",
            shape=(self.height, self.width, self.n_channels),
        )
        self._weight_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._finalized = False

    # ------------------------------------------------------------------
    def iter_chunk_bounds(self) -> Iterator[ChunkBounds]:
        """Yield non-overlapping chunk bounds covering the canvas."""

        for y0 in range(0, self.height, self.chunk_size):
            y1 = min(self.height, y0 + self.chunk_size)
            for x0 in range(0, self.width, self.chunk_size):
                x1 = min(self.width, x0 + self.chunk_size)
                yield (x0, y0, x1, y1)

    # ------------------------------------------------------------------
    def allocate_chunk_arrays(
        self, bounds: ChunkBounds
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = bounds
        chunk_h = y1 - y0
        chunk_w = x1 - x0
        canvas = np.zeros((self.n_channels, chunk_h, chunk_w), dtype=np.float32)
        weight = np.zeros((chunk_h, chunk_w), dtype=np.float32)
        return canvas, weight

    # ------------------------------------------------------------------
    def _weight_mask(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        mask = self._weight_cache.get(key)
        if mask is None:
            if height <= 1 or width <= 1:
                mask = np.ones((height, width), dtype=np.float32)
            else:
                wy = np.hanning(height + 2)[1:-1]
                wx = np.hanning(width + 2)[1:-1]
                mask = np.outer(wy, wx).astype(np.float32)
            self._weight_cache[key] = mask
        return mask

    # ------------------------------------------------------------------
    def accumulate(
        self,
        chunk_canvas: np.ndarray,
        chunk_weights: np.ndarray,
        prediction: np.ndarray,
        patch_bounds: ChunkBounds,
        chunk_bounds: ChunkBounds,
    ) -> None:
        """Add a single patch prediction into a chunk accumulator."""

        px0, py0, px1, py1 = patch_bounds
        cx0, cy0, cx1, cy1 = chunk_bounds
        ix0 = max(px0, cx0)
        iy0 = max(py0, cy0)
        ix1 = min(px1, cx1)
        iy1 = min(py1, cy1)
        if ix0 >= ix1 or iy0 >= iy1:
            return

        patch_x0 = ix0 - px0
        patch_y0 = iy0 - py0
        patch_x1 = patch_x0 + (ix1 - ix0)
        patch_y1 = patch_y0 + (iy1 - iy0)

        chunk_x0 = ix0 - cx0
        chunk_y0 = iy0 - cy0
        chunk_x1 = chunk_x0 + (ix1 - ix0)
        chunk_y1 = chunk_y0 + (iy1 - iy0)

        mask = self._weight_mask(prediction.shape[1], prediction.shape[2])
        mask_slice = mask[patch_y0:patch_y1, patch_x0:patch_x1]
        pred_slice = prediction[:, patch_y0:patch_y1, patch_x0:patch_x1]

        chunk_canvas[:, chunk_y0:chunk_y1, chunk_x0:chunk_x1] += pred_slice * mask_slice
        chunk_weights[chunk_y0:chunk_y1, chunk_x0:chunk_x1] += mask_slice

    # ------------------------------------------------------------------
    def finalize_chunk(
        self,
        bounds: ChunkBounds,
        chunk_canvas: np.ndarray,
        chunk_weights: np.ndarray,
    ) -> None:
        black_mask = chunk_weights <= 1e-8
        weight = np.maximum(chunk_weights, 1e-8)[None, ...]
        chunk = np.divide(chunk_canvas, weight, dtype=np.float32)
        if self.background_colour == "white":
            chunk[:, black_mask] = 1.0  # set black pixels to white
        chunk = np.clip(chunk, 0.0, 1.0)
        chunk_uint8 = np.rint(chunk * 255.0).astype(np.uint8)
        x0, y0, x1, y1 = bounds
        chunk_cl = np.moveaxis(chunk_uint8, 0, -1)
        self.canvas[y0:y1, x0:x1, :] = chunk_cl

    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """Return the stitched canvas as a numpy array."""

        canvas = self.finalize()
        try:
            return canvas.to_array()
        finally:
            canvas.close()

    # ------------------------------------------------------------------
    def finalize(self) -> PredictionCanvas:
        """Flush buffers and return a :class:`PredictionCanvas`."""

        if self._finalized:
            msg = "ChunkedStitcher.finalize() may only be called once"
            raise RuntimeError(msg)
        self.canvas.flush()
        canvas = PredictionCanvas(
            memmap=self.canvas,
            path=Path(self._canvas_path),
            shape=(self.height, self.width, self.n_channels),
            is_temporary=self._tmp_dir is not None,
            _tmp_dir=self._tmp_dir,
        )
        self._tmp_dir = None
        self._finalized = True
        return canvas
