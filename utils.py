import logging
import os
from rich.logging import RichHandler
import colorsys
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence
from tifffile import OmeXml
import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a Rich console handler.

    Parameters
    ----------
    name:
        Name of the logger to retrieve.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.getenv("SPATX_LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = RichHandler(rich_tracebacks=True, markup=True)
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

def _generate_colors(n: int) -> list[tuple[int, int, int]]:
    if n <= 0:
        return []
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    colors: list[tuple[int, int, int]] = []
    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(float(hue), 0.75, 1.0)
        colors.append(
            (
                int(round(r * 255)),
                int(round(g * 255)),
                int(round(b * 255)),
            )
        )
    return colors


def _prepare_ome_xml(
    shape: tuple[int, int, int],
    dtype: np.dtype,
    genes: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
    pixel_size: tuple[float, float],
) -> str:
    """
    Generate OME-XML for channel-last interleaved TIFF storage (Y, X, S).

    This matches pyvips writing an image with `bands=C` (SamplesPerPixel=C,
    PlanarConfiguration=CONTIG), i.e. a single 2D plane with interleaved samples.
    """
    if len(shape) != 3:
        raise ValueError("shape must be a 3-tuple")

    # Accept either legacy (C, H, W) or channel-last (H, W, C) inputs.
    a, b, c = shape
    if a == len(genes) and b > 32 and c > 32:
        # Likely (C, H, W)
        C, H, W = shape
    else:
        # Likely (H, W, C)
        H, W, C = shape

    dtype_name = np.dtype(dtype).name

    ox = OmeXml()
    ox.addimage(
        dtype=dtype_name,
        # Channel-last interleaved: (Y, X, S)
        shape=(H, W, C),
        # storedshape = (planecount, separate, depth, length(Y), width(X), contig(samples))
        storedshape=(1, 1, 1, H, W, C),
        axes="YXS",
        Name="GeneExpression",
        PhysicalSizeX=float(pixel_size[0]),
        PhysicalSizeY=float(pixel_size[1]),
    )

    root = ET.fromstring(ox.tostring())
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    # In this representation there is typically ONE <Channel> with SamplesPerPixel=C.
    # (OME doesn't have a standard way to name each interleaved sample as a separate Channel.)
    ch = root.find(".//ome:Channel", ns)
    if ch is not None:
        ch.set("Name", "GeneExpression (interleaved samples)")
        # Optional: leave Color unset or set a neutral value; per-gene colors go below.

    # Keep your per-gene metadata in your custom table
    scan = ET.SubElement(root, "ScanColorTable", {"ref": "gene_colors"})
    for gene, color in zip(genes, colors):
        ET.SubElement(scan, "ScanColorTable-k").text = gene
        ET.SubElement(scan, "ScanColorTable-v").text = ", ".join(map(str, color))

    return ET.tostring(root, encoding="unicode", xml_declaration=True)
