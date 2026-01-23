"""
SVG to high-resolution raster conversion with physical scale mapping.

Uses CairoSVG to render SVG files at arbitrary resolutions while maintaining
the physical scale relationship defined by the scene units.
"""

import io
from pathlib import Path
from typing import Optional, Union, Tuple

import cairosvg
import numpy as np
from PIL import Image

from .scene_units import SceneUnits, PhysicalExtent


class SVGLoader:
    """
    Loads and rasterizes SVG files with physical scale awareness.
    
    The loader extracts the viewBox dimensions from the SVG and maps them
    to physical coordinates, then renders at the resolution required by
    the internal oversampled representation.
    """
    
    def __init__(
        self,
        svg_path: Union[str, Path],
        physical_width_mm: Optional[float] = None,
        physical_height_mm: Optional[float] = None,
    ):
        """
        Initialize the SVG loader.
        
        Args:
            svg_path: Path to the SVG file
            physical_width_mm: Physical width of the SVG content in mm.
                              If None, uses SVG viewBox width as mm.
            physical_height_mm: Physical height of the SVG content in mm.
                               If None, computed from aspect ratio.
        """
        self.svg_path = Path(svg_path)
        self._svg_content = self.svg_path.read_text()
        
        # Extract SVG dimensions from viewBox or width/height attributes
        self._svg_width, self._svg_height = self._parse_svg_dimensions()
        
        # Determine physical extent
        if physical_width_mm is None:
            physical_width_mm = self._svg_width  # Assume SVG units are mm
        
        if physical_height_mm is None:
            aspect = self._svg_height / self._svg_width
            physical_height_mm = physical_width_mm * aspect
        
        self.physical_extent = PhysicalExtent(
            width_mm=physical_width_mm,
            height_mm=physical_height_mm,
        )
    
    def _parse_svg_dimensions(self) -> Tuple[float, float]:
        """Extract width and height from SVG viewBox or dimensions."""
        import re
        
        # Try viewBox first
        viewbox_match = re.search(
            r'viewBox\s*=\s*["\']([^"\']+)["\']',
            self._svg_content
        )
        if viewbox_match:
            parts = viewbox_match.group(1).split()
            if len(parts) >= 4:
                return float(parts[2]), float(parts[3])
        
        # Fall back to width/height attributes
        width_match = re.search(
            r'width\s*=\s*["\']?([\d.]+)',
            self._svg_content
        )
        height_match = re.search(
            r'height\s*=\s*["\']?([\d.]+)',
            self._svg_content
        )
        
        width = float(width_match.group(1)) if width_match else 100.0
        height = float(height_match.group(1)) if height_match else 100.0
        
        return width, height
    
    def render(
        self,
        scene_units: SceneUnits,
        background_value: float = 1.0,
    ) -> np.ndarray:
        """
        Render the SVG at the resolution specified by scene units.
        
        Args:
            scene_units: Defines the internal oversampled resolution
            background_value: Value for transparent regions (0-1 range)
        
        Returns:
            Grayscale image as float32 array normalized to [0, 1]
        """
        width_px, height_px = scene_units.internal_resolution
        
        # Render SVG to PNG in memory
        png_data = cairosvg.svg2png(
            bytestring=self._svg_content.encode('utf-8'),
            output_width=width_px,
            output_height=height_px,
        )
        
        # Load as PIL Image and convert to numpy
        img = Image.open(io.BytesIO(png_data))
        
        # Handle transparency - composite onto background
        if img.mode == 'RGBA':
            # Create background
            background = Image.new('RGBA', img.size, 
                                   (int(background_value * 255),) * 3 + (255,))
            img = Image.alpha_composite(background, img)
        
        # Convert to grayscale
        img_gray = img.convert('L')
        
        # Convert to float32 normalized array
        arr = np.array(img_gray, dtype=np.float32) / 255.0
        
        return arr
    
    def render_rgb(
        self,
        scene_units: SceneUnits,
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Render the SVG as RGB at the resolution specified by scene units.
        
        Args:
            scene_units: Defines the internal oversampled resolution
            background_color: RGB tuple for transparent regions
        
        Returns:
            RGB image as float32 array normalized to [0, 1], shape (H, W, 3)
        """
        width_px, height_px = scene_units.internal_resolution
        
        png_data = cairosvg.svg2png(
            bytestring=self._svg_content.encode('utf-8'),
            output_width=width_px,
            output_height=height_px,
        )
        
        img = Image.open(io.BytesIO(png_data))
        
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, background_color + (255,))
            img = Image.alpha_composite(background, img)
        
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        
        return arr


def load_svg(
    svg_path: Union[str, Path],
    scene_units: SceneUnits,
    physical_width_mm: Optional[float] = None,
    as_rgb: bool = False,
) -> np.ndarray:
    """
    Convenience function to load and render an SVG in one step.
    
    Args:
        svg_path: Path to SVG file
        scene_units: Scene units defining resolution and scale
        physical_width_mm: Physical width of SVG content in mm
        as_rgb: If True, return RGB image; otherwise grayscale
    
    Returns:
        Image array normalized to [0, 1]
    """
    loader = SVGLoader(svg_path, physical_width_mm=physical_width_mm)
    
    if as_rgb:
        return loader.render_rgb(scene_units)
    else:
        return loader.render(scene_units)
