"""
Scene generation and loading module.

Handles SVG to high-resolution raster conversion with proper physical scale mapping.
"""

from .svg_loader import SVGLoader, load_svg
from .scene_units import SceneUnits, PhysicalExtent

__all__ = [
    "SVGLoader",
    "load_svg",
    "SceneUnits",
    "PhysicalExtent",
]
