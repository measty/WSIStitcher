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
    c, h, w = shape
    dtype_name = np.dtype(dtype).name
    ox = OmeXml()
    ox.addimage(
        dtype=dtype_name,
        shape=(c, h, w),
        storedshape=(c, 1, 1, h, w, 1),
        axes="CYX",
        Name="GeneExpression",
        PhysicalSizeX=float(pixel_size[0]),
        PhysicalSizeY=float(pixel_size[1]),
    )
    root = ET.fromstring(ox.tostring())
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    for channel, gene, color in zip(root.findall(".//ome:Channel", ns), genes, colors):
        channel.set("Name", gene)
        channel.set("Color", "#%02X%02X%02X" % color)

    scan = ET.SubElement(root, "ScanColorTable", {"ref": "gene_colors"})
    for gene, color in zip(genes, colors):
        ET.SubElement(scan, "ScanColorTable-k").text = gene
        ET.SubElement(scan, "ScanColorTable-v").text = ", ".join(map(str, color))

    return ET.tostring(root, encoding="unicode", xml_declaration=True)