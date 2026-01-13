"""
Example script for super-resolution prototyping.

This script demonstrates basic image loading, downsampling, and
upsampling operations for testing SR algorithms.

Usage:
    uv run --package sr-prototyping python SR_prototyping/scripts/example.py
"""

import numpy as np
from pathlib import Path

# Optional imports - check availability
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Note: opencv-python not available, using PIL fallback")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_image(path: Path) -> np.ndarray:
    """Load an image as numpy array."""
    if HAS_CV2:
        img = cv2.imread(str(path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if HAS_PIL:
        return np.array(Image.open(path))
    
    raise RuntimeError("No image loading library available")


def downsample(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample image by given factor."""
    if HAS_CV2:
        h, w = image.shape[:2]
        return cv2.resize(image, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    
    # Simple numpy fallback
    return image[::factor, ::factor]


def upsample_bilinear(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Upsample image using bilinear interpolation."""
    if HAS_CV2:
        h, w = image.shape[:2]
        return cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_LINEAR)
    
    if HAS_PIL:
        pil_img = Image.fromarray(image)
        h, w = image.shape[:2]
        return np.array(pil_img.resize((w * factor, h * factor), Image.BILINEAR))
    
    raise RuntimeError("No upsampling library available")


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between images."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    print("Super Resolution Prototyping Example")
    print("=" * 40)
    print()
    
    # Check for test image
    resources_dir = Path(__file__).parent.parent / "resources"
    media_dir = resources_dir / "media"
    
    print(f"Resources directory: {resources_dir}")
    print(f"Media directory: {media_dir}")
    print()
    
    # Create a synthetic test image if no real image available
    print("Creating synthetic test pattern...")
    
    # Generate a test pattern (checkerboard with gradient)
    size = 256
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Checkerboard pattern
    checker_size = 32
    checkerboard = ((xx * size // checker_size + yy * size // checker_size) % 2)
    
    # Add gradient
    gradient = xx * 0.3 + yy * 0.3
    
    # Combine
    test_image = (checkerboard * 0.7 + gradient) * 255
    test_image = test_image.astype(np.uint8)
    
    # Convert to RGB
    test_image_rgb = np.stack([test_image] * 3, axis=-1)
    
    print(f"Test image shape: {test_image_rgb.shape}")
    print()
    
    # Demonstrate downsampling and upsampling
    print("Downsampling by 2x...")
    downsampled = downsample(test_image_rgb, factor=2)
    print(f"Downsampled shape: {downsampled.shape}")
    
    print("Upsampling with bilinear interpolation...")
    upsampled = upsample_bilinear(downsampled, factor=2)
    print(f"Upsampled shape: {upsampled.shape}")
    
    # Compute quality metric
    psnr = compute_psnr(test_image_rgb, upsampled)
    print()
    print(f"PSNR (bilinear upsampling): {psnr:.2f} dB")
    print()
    
    # Note about SR algorithms
    print("Next steps for super-resolution:")
    print("  1. Capture multiple beam-shifted images")
    print("  2. Register/align the shifted frames")
    print("  3. Combine frames using SR reconstruction")
    print("  4. Compare against single-frame upsampling")
    print()
    
    # Display if matplotlib available
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(test_image_rgb)
        axes[0].set_title("Original (256x256)")
        axes[0].axis("off")
        
        axes[1].imshow(downsampled)
        axes[1].set_title("Downsampled (128x128)")
        axes[1].axis("off")
        
        axes[2].imshow(upsampled)
        axes[2].set_title(f"Bilinear Upsampled\nPSNR: {psnr:.2f} dB")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Save figure
        output_path = media_dir / "example_output.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to: {output_path}")
        
        # plt.show()  # Uncomment to display interactively
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
