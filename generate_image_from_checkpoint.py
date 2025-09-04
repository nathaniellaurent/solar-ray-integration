"""Generate rendered image(s) from a trained Solar NeRF checkpoint.

Usage examples:
    # Render a single image (index 0):
    python generate_image_from_checkpoint.py \
        --checkpoint outputs/final_model.ckpt \
        --output-dir rendered_outputs

    # Render first 5 images:
    python generate_image_from_checkpoint.py \
        --checkpoint outputs/final_model.ckpt \
        --num-images 5 \
        --output-dir rendered_outputs

    # Render specific indices:
    python generate_image_from_checkpoint.py \
        --checkpoint outputs/final_model.ckpt \
        --indices 0 5 10 15 \
        --output-dir rendered_outputs

    # Override HGLN/HGLT values:
    python generate_image_from_checkpoint.py \
        --checkpoint outputs/final_model.ckpt \
        --hgln 0.0 --hglt 0.0 \
        --num-images 3 \
        --output-dir rendered_outputs

Outputs:
  - Individual PNG images for each rendered sample
  - Summary PNG showing all images in a grid (if multiple images)
  - Numpy arrays of predictions (if --save-npy)
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits import PrimaryHDU
from astropy.io import fits
from astropy.wcs import WCS

# Ensure package import
sys.path.append(os.path.dirname(__file__))
from solar_ray_integration.training.ray_samples_module import RayWiseLightningModule
from solar_ray_integration.training.dataset import SolarPerspectiveDataset


def create_dummy_fits_hdu(
    image_size=(64, 64),
    hgln_obs: float = 0.0,
    hglt_obs: float = 0.0,
    dsun_obs: float = 151693705216
) -> PrimaryHDU:
    """Create a minimal FITS HDU with WCS headers for rendering."""
    height, width = image_size
    data = np.ones((height, width), dtype=np.float32)
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = width
    header['NAXIS2'] = height
    header['CTYPE1'] = 'HPLN-TAN'
    header['CTYPE2'] = 'HPLT-TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CRPIX1'] = width / 2.0
    header['CRPIX2'] = height / 2.0
    header['CRVAL1'] = 0.0
    header['CRVAL2'] = 0.0
    arcsec_per_pixel = 2.0
    deg_per_pixel = arcsec_per_pixel / 3600.0
    header['CDELT1'] = deg_per_pixel
    header['CDELT2'] = deg_per_pixel
    header['CD1_1'] = deg_per_pixel
    header['CD1_2'] = 0.0
    header['CD2_1'] = 0.0
    header['CD2_2'] = deg_per_pixel
    header['HGLN_OBS'] = float(hgln_obs)
    header['HGLT_OBS'] = float(hglt_obs)
    header['DSUN_OBS'] = float(dsun_obs)
    header['RSUN_REF'] = 6.96e8
    header['RSUN_OBS'] = 960.0
    header['CRLN_OBS'] = float(hgln_obs)
    header['CRLT_OBS'] = float(hglt_obs)
    header['DATE-OBS'] = '2025-01-01T00:00:00'
    header['MJD-OBS'] = 60500.0
    return PrimaryHDU(data, header=header)


def create_summary_plot(rendered_images, indices, output_dir):
    """Create a summary plot showing all rendered images in a grid."""
    num_images = len(rendered_images)
    
    # Calculate grid dimensions
    cols = min(4, num_images)  # Maximum 4 columns
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (arr, idx) in enumerate(zip(rendered_images, indices)):
        ax = axes[i]
        im = ax.imshow(arr, origin='lower', cmap='inferno')
        ax.set_title(f'Index {idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'summary_all_rendered.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot: {summary_path}")


def render_from_checkpoint(
    checkpoint: str,
    hgln: float = None,
    hglt: float = None,
    image_size=(64, 64),
    device: str = 'cpu',
    output_dir: str = 'outputs',
    save_npy: bool = False,
    indices: list = None,
    num_images: int = 1
):
   
    print(f"Loading checkpoint: {checkpoint}")
    model = RayWiseLightningModule.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    dataset = SolarPerspectiveDataset(
            data_dir="perspective_data/perspective_data_linear_fits_64x64",  # Dummy, won't be used
            dx=1e7,
            device=device,
            normalize=True,
            normalization_value=139.0,
        )
    
    # Determine which indices to render
    if indices is None:
        # Generate indices based on num_images
        max_index = min(len(dataset), num_images)
        indices = list(range(max_index))
    else:
        # Validate provided indices
        max_dataset_index = len(dataset) - 1
        indices = [i for i in indices if 0 <= i <= max_dataset_index]
        if not indices:
            raise ValueError(f"No valid indices found. Dataset has {len(dataset)} samples (0-{max_dataset_index})")
    
    print(f"Rendering {len(indices)} images from indices: {indices}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    rendered_images = []
    
    for i, index in enumerate(indices):
        print(f"\nRendering image {i+1}/{len(indices)} (dataset index {index})...")
        
        sample = dataset[index]
        header = sample['header']
        
        # Extract actual hgln, hglt from sample if not provided
        actual_hgln = sample.get('hgln', hgln if hgln is not None else 0.0)
        actual_hglt = sample.get('hglt', hglt if hglt is not None else 0.0)

        with torch.no_grad():
            rendered = model.render_sample_image(header, device=device)

        rendered_cpu = rendered.detach().cpu().float()
        arr = rendered_cpu.numpy()
        rendered_images.append(arr)

        # Normalize for visualization (optional; comment out if not desired)
        vmin, vmax = float(arr.min()), float(arr.max())
        print(f"Rendered image stats: min={vmin:.5f} max={vmax:.5f} mean={arr.mean():.5f} std={arr.std():.5f}")

        # Generate output filename
        output_png = os.path.join(output_dir, f'rendered_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.png')
        
        plt.figure(figsize=(5,5))
        plt.imshow(arr, origin='lower', cmap='inferno')
        plt.colorbar(label='Intensity')
        plt.title(f'Rendered | Index {index} | HGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PNG: {output_png}")

        if save_npy:
            npy_path = os.path.splitext(output_png)[0] + '.npy'
            np.save(npy_path, arr.astype(np.float32))
            print(f"Saved NumPy array: {npy_path}")
    
    # Create a summary plot if multiple images
    if len(rendered_images) > 1:
        create_summary_plot(rendered_images, indices, output_dir)

    return rendered_images


def parse_args():
    p = argparse.ArgumentParser(description='Render image(s) from Solar NeRF checkpoint')
    p.add_argument('--checkpoint', type=str, default='outputs/final_model.ckpt', help='Path to .ckpt file')
    p.add_argument('--hgln', type=float, default=None, help='Heliographic longitude (optional, will use dataset values if not provided)')
    p.add_argument('--hglt', type=float, default=None, help='Heliographic latitude (optional, will use dataset values if not provided)')
    p.add_argument('--image-size', type=int, nargs=2, default=[64, 64], help='Image size H W')
    p.add_argument('--device', type=str, default='cpu', help='cpu | cuda:0 | auto')
    p.add_argument('--output-dir', type=str, default='rendered_outputs', help='Output directory for images')
    p.add_argument('--save-npy', action='store_true', help='Also save raw array .npy files')
    p.add_argument('--num-images', type=int, default=1, help='Number of images to render (starting from index 0)')
    p.add_argument('--indices', type=int, nargs='+', default=None, help='Specific dataset indices to render (overrides --num-images)')
    return p.parse_args()


def main():
    args = parse_args()
    render_from_checkpoint(
        checkpoint=args.checkpoint,
        hgln=args.hgln,
        hglt=args.hglt,
        image_size=tuple(args.image_size),
        device=args.device,
        output_dir=args.output_dir,
        save_npy=args.save_npy,
        indices=args.indices,
        num_images=args.num_images,
    )


if __name__ == '__main__':
    main()
