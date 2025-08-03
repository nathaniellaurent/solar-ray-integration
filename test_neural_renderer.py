"""
Test script for NeuralSolarRenderer with random NeRF weights.

This script tests the neural rendering pipeline using randomly initialized
NeRF weights to verify that the integration and rendering process works
correctly without requiring trained models.
"""

import os
import sys
import torch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.fits import PrimaryHDU
import matplotlib.pyplot as plt
from typing import Optional

# Add the package to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from solar_ray_integration.rendering.neural_renderer import NeuralSolarRenderer
from solar_ray_integration.model.model import NeRF


def create_dummy_fits_hdu(
    image_size: tuple = (512, 512),
    hgln_obs: float = 0.0,
    hglt_obs: float = 0.0,
    dsun_obs: float = 1.496e11
) -> PrimaryHDU:
    """
    Create a dummy FITS HDU with proper WCS for testing.
    
    Args:
        image_size: (height, width) of the image
        hgln_obs: Heliographic longitude of observer
        hglt_obs: Heliographic latitude of observer  
        dsun_obs: Distance from observer to Sun center
        
    Returns:
        FITS HDU with proper WCS headers
    """
    height, width = image_size
    
    # Create dummy image data
    dummy_data = np.ones((height, width), dtype=np.float32)
    
    # Create basic WCS header
    header = fits.Header()
    
    # Image dimensions
    header['NAXIS'] = 2
    header['NAXIS1'] = width
    header['NAXIS2'] = height
    
    # WCS parameters for heliographic coordinates
    header['CTYPE1'] = 'HPLN-TAN'
    header['CTYPE2'] = 'HPLT-TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    
    # Reference pixel (center of image)
    header['CRPIX1'] = width / 2.0
    header['CRPIX2'] = height / 2.0
    
    # Reference coordinate (Sun center)
    header['CRVAL1'] = 0.0  # Heliographic longitude
    header['CRVAL2'] = 0.0  # Heliographic latitude
    
    # Pixel scale (degrees per pixel) - roughly 2 arcsec/pixel
    arcsec_per_pixel = 2.0
    deg_per_pixel = arcsec_per_pixel / 3600.0
    header['CDELT1'] = deg_per_pixel
    header['CDELT2'] = deg_per_pixel
    
    # Coordinate matrix (identity matrix for simple case)
    header['CD1_1'] = deg_per_pixel
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = deg_per_pixel
    
    # Observer position
    header['HGLN_OBS'] = float(hgln_obs)
    header['HGLT_OBS'] = float(hglt_obs)
    header['DSUN_OBS'] = float(dsun_obs)
    
    # Solar radius in arcseconds (typical value)
    header['RSUN_REF'] = 6.96e8  # Solar radius in meters
    header['RSUN_OBS'] = 960.0   # Solar radius in arcseconds
    
    # Additional observer coordinates that might be needed
    header['CRLN_OBS'] = float(hgln_obs)  # Carrington longitude of observer
    header['CRLT_OBS'] = float(hglt_obs)  # Carrington latitude of observer
    
    # Date/time information (optional but can help)
    header['DATE-OBS'] = '2023-01-01T12:00:00'
    header['MJD-OBS'] = 59945.5
    
    # Create HDU
    hdu = PrimaryHDU(dummy_data, header=header)
    
    return hdu


def test_neural_renderer_random_weights(
    integration_method: str = "linear",
    image_size: tuple = (256, 256),  # Smaller for faster testing
    device: str = "cpu",
    nerf_config: Optional[dict] = None
):
    """
    Test NeuralSolarRenderer with random weights.
    
    Args:
        integration_method: Integration method to test
        image_size: Size of test image
        device: Device to run on
        nerf_config: NeRF configuration (None for default)
    """
    print(f"Testing NeuralSolarRenderer with {integration_method} integration")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    
    try:
        # Create smaller NeRF config for faster testing
        if nerf_config is None:
            nerf_config = {
                'd_input': 3,
                'd_output': 1,
                'n_layers': 4,      # Reduced from 8
                'd_filter': 128,    # Reduced from 512
                'encoding': 'positional'
            }
        
        # Create neural renderer
        renderer = NeuralSolarRenderer(
            nerf_config=nerf_config,
            dx=2e7,  # Larger step size for faster computation
            device=device,  # Let it auto-detect device
            integration_method=integration_method
        )
        
        # Move to device
        renderer = renderer.to(device)
        
        print(f"✓ Renderer created successfully")
        print(f"  NeRF parameters: {sum(p.numel() for p in renderer.nerf.parameters()):,}")
        
        # Use dummy HDU to avoid byte order issues with real FITS files
        test_hdu = create_dummy_fits_hdu(
            image_size=(1024, 1024),  # Use full resolution for testing
            hgln_obs=0.0,
            hglt_obs=0.0
        )
        print(f"✓ Test HDU created")
        print(f"  Image shape: {test_hdu.data.shape}")
        print(f"  DSUN_OBS from header: {test_hdu.header['DSUN_OBS']}")
        wcs = WCS(test_hdu.header)
        print(f"  WCS created with {wcs.wcs.ctype[0]}, {wcs.wcs.ctype[1]}")
        # Test forward pass
        print("Running forward pass...")
        with torch.no_grad():  # No gradients needed for testing
            start_time = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
            end_time = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
            
            if start_time:
                start_time.record()
            
            rendered_image = renderer(test_hdu, requires_grad=False)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                elapsed_time = None
        
        print(f"✓ Forward pass completed")
        print(f"  Output shape: {rendered_image.shape}")
        print(f"  Output dtype: {rendered_image.dtype}")
        print(f"  Output range: [{rendered_image.min().item():.6f}, {rendered_image.max().item():.6f}]")
        print(f"  Output mean: {rendered_image.mean().item():.6f}")
        print(f"  Output std: {rendered_image.std().item():.6f}")
        
        if elapsed_time:
            print(f"  Computation time: {elapsed_time:.2f} seconds")
        
        # Check for NaN or inf values
        if torch.isnan(rendered_image).any():
            print("⚠️  Warning: Output contains NaN values")
        if torch.isinf(rendered_image).any():
            print("⚠️  Warning: Output contains infinite values")
        
        return rendered_image, renderer
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_different_viewing_angles(renderer, base_hdu, device="cpu"):
    """Test renderer with different viewing angles."""
    print("\nTesting different viewing angles...")
    
    viewing_angles = [
        (0.0, 0.0),     # Front view
        (30.0, 0.0),    # 30° longitude
        (0.0, 30.0),    # 30° latitude
        (-30.0, -15.0), # Different angle
    ]
    
    results = []
    
    for hgln, hglt in viewing_angles:
        print(f"  Testing HGLN={hgln}°, HGLT={hglt}°")
        
        # Create HDU with different viewing angle
        test_hdu = create_dummy_fits_hdu(
            image_size=base_hdu.data.shape,
            hgln_obs=hgln,
            hglt_obs=hglt
        )
        
        with torch.no_grad():
            rendered = renderer(test_hdu, requires_grad=False)
            results.append((hgln, hglt, rendered))
            
        print(f"    Output range: [{rendered.min().item():.6f}, {rendered.max().item():.6f}]")
    
    return results


def visualize_results(results, save_path="neural_renderer_test_output.png"):
    """Visualize test results."""
    print(f"\nSaving visualization to {save_path}")
    
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(4*n_results, 4))
    
    if n_results == 1:
        axes = [axes]
    
    for i, (hgln, hglt, image) in enumerate(results):
        ax = axes[i]
        
        # Convert to numpy and handle device
        if isinstance(image, torch.Tensor):
            if image.device.type == 'cuda':
                image_np = image.cpu().numpy()
            else:
                image_np = image.numpy()
        else:
            image_np = image
            
        im = ax.imshow(image_np, cmap='viridis', origin='lower')
        ax.set_title(f'HGLN={hgln}°, HGLT={hglt}°')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Field Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved")


def main():
    """Run comprehensive tests of the NeuralSolarRenderer."""
    print("="*60)
    print("Neural Solar Renderer Test with Random Weights")
    print("="*60)
    
    # Test configuration
    device = "cpu"  # Use CPU for testing to avoid GPU memory issues
    image_size = (128, 128)  # Full resolution for testing
    
    # Test all integration methods
    integration_methods = ["linear"]
    
    all_results = []
    
    for method in integration_methods:
        print(f"\n{'='*40}")
        print(f"Testing {method.upper()} Integration")
        print(f"{'='*40}")
        
        # Test basic functionality
        rendered_image, renderer = test_neural_renderer_random_weights(
            integration_method=method,
            image_size=image_size,
            device=device
        )
        
        if rendered_image is not None and renderer is not None:
            # Create base HDU for viewing angle tests
            base_hdu = create_dummy_fits_hdu(image_size=image_size)
            
            # Test different viewing angles
            angle_results = test_different_viewing_angles(renderer, base_hdu, device)
            all_results.extend([(method, *result) for result in angle_results])
            
            print(f"✓ {method} integration test passed")
        else:
            print(f"✗ {method} integration test failed")
    
    # Visualize results from first method
    if all_results:
        first_method_results = [(hgln, hglt, img) for method, hgln, hglt, img in all_results 
                               if method == integration_methods[0]]
        if first_method_results:
            visualize_results(first_method_results[:4])  # Show first 4 angles
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
