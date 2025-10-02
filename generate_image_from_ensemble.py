"""Generate rendered image(s) from an ensemble of Solar NeRF checkpoints.

Usage examples:
    # Render from all checkpoints in directory:
    python generate_image_from_ensemble.py \
        --checkpoint-dir SWA_checkpoints \
        --output-dir ensemble_outputs

    # Render first 5 images with target comparison:
    python generate_image_from_ensemble.py \
        --checkpoint-dir SWA_checkpoints \
        --num-images 5 \
        --compare-target \
        --output-dir ensemble_outputs

    # Render specific indices:
    python generate_image_from_ensemble.py \
        --checkpoint-dir SWA_checkpoints \
        --indices 0 5 10 15 \
        --output-dir ensemble_outputs

    # Render specific coordinates (HGLN=0°, HGLT=0°,30°,60°,90°):
    python generate_image_from_ensemble.py \
        --checkpoint-dir SWA_checkpoints \
        --coordinates 0 0 0 30 0 60 0 90 \
        --output-dir ensemble_outputs

Outputs:
  - Mean PNG images for each rendered sample (ensemble average)
  - Standard deviation PNG images showing uncertainty at each pixel
  - If --compare-target: comparison plots with target images
  - Summary PNG showing all mean images in a grid
  - Numpy arrays of means, stds, and targets (if --save-npy)
"""
import os
import sys
import argparse
from pathlib import Path
import glob
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



def create_summary_plot(rendered_images, indices, output_dir, target_images=None, hgln_values=None, hglt_values=None, std_images=None):
    """Create a summary plot showing all rendered images in a grid."""
    num_images = len(rendered_images)
    
    if target_images is not None:
        # Comparison mode: each image gets 5 subplots (target, predicted, abs diff, % diff, uncertainty)
        # Arrange them in rows of 5 columns per image
        cols_per_image = 5
        total_cols = cols_per_image
        rows = num_images
        
        fig, axes = plt.subplots(rows, total_cols, figsize=(5*total_cols, 5*rows))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08,
                    hspace=0.08, wspace=0.16)  # Make images closer together
        plot_title = 'summary_comparison_all.png'
        
        # Handle single image case
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        # Add column headers at the top
        column_titles = ['Target', 'Predicted', 'Absolute Diff', 'Percent Diff (%)', 'Uncertainty']
        for j, title in enumerate(column_titles):
            axes[0, j].text(0.5, 1.1, title, transform=axes[0, j].transAxes, 
                           ha='center', va='bottom', fontsize=38, fontweight='bold')
        
        # Pre-compute all difference arrays and find global min/max for consistent colorbars
        all_targets = []
        all_predicted = []
        all_abs_diffs = []
        all_percent_diffs = []
        all_uncertainties = []
        
        for i, (arr, idx) in enumerate(zip(rendered_images, indices)):
            target_arr = target_images[i]
            abs_diff = np.abs(arr - target_arr)
            percent_diff = np.where(target_arr != 0, 100 * abs_diff / np.abs(target_arr), 0)
            std_arr = std_images[i] if std_images else np.zeros_like(arr)
            
            all_targets.append(target_arr)
            all_predicted.append(arr)
            all_abs_diffs.append(abs_diff)
            all_percent_diffs.append(percent_diff)
            all_uncertainties.append(std_arr)
        
        # Calculate global min/max for each image type using 90th percentile / 0.9 for max
        # target_vmin, target_vmax = np.min(all_targets), np.percentile(all_targets, 95) / 0.95
        # predicted_vmin, predicted_vmax = np.min(all_predicted), np.percentile(all_predicted, 95) / 0.95
        # abs_diff_vmin, abs_diff_vmax = np.min(all_abs_diffs), np.percentile(all_abs_diffs, 95) / 0.95
        # percent_diff_vmin, percent_diff_vmax = np.min(all_percent_diffs), np.percentile(all_percent_diffs, 95) / 0.95
        # uncertainty_vmin, uncertainty_vmax = np.min(all_uncertainties), np.percentile(all_uncertainties, 95) / 0.95

        target_vmin, target_vmax = 0.0, 1.0
        predicted_vmin, predicted_vmax = 0.0, 1.0
        abs_diff_vmin, abs_diff_vmax = 0.0, 0.114
        percent_diff_vmin, percent_diff_vmax = 0.0, 100.0
        uncertainty_vmin, uncertainty_vmax = 0.0, 0.1

        print("Target min/max: ", target_vmin, target_vmax)
        print("Predicted min/max: ", predicted_vmin, predicted_vmax)
        print("Abs diff min/max: ", abs_diff_vmin, abs_diff_vmax)
        print("Percent diff min/max: ", percent_diff_vmin, percent_diff_vmax)
        print("Uncertainty min/max: ", uncertainty_vmin, uncertainty_vmax)
        
        
        for i, (arr, idx) in enumerate(zip(rendered_images, indices)):
            target_arr = all_targets[i]
            abs_diff = all_abs_diffs[i]
            percent_diff = all_percent_diffs[i]
            std_arr = all_uncertainties[i]
            
            # Get lat/lon for this image
            hgln = hgln_values[i] if hgln_values else 0.0
            hglt = hglt_values[i] if hglt_values else 0.0
            
            # Add row label on the left (without index)
            axes[i, 0].text(-0.1, 0.5, f'HGLN={hgln:.1f}°\nHGLT={hglt:.1f}°', 
                           transform=axes[i, 0].transAxes, ha='right', va='center', 
                           fontsize=32, fontweight='bold', rotation=90)
            
            # Target
            ax_target = axes[i, 0]
            im1 = ax_target.imshow(target_arr, origin='lower', cmap='inferno', vmin=target_vmin, vmax=target_vmax)
            ax_target.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax_target, fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=16)
            
            # Predicted
            ax_predicted = axes[i, 1]
            im2 = ax_predicted.imshow(arr, origin='lower', cmap='inferno', vmin=predicted_vmin, vmax=predicted_vmax)
            ax_predicted.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax_predicted, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=16)
            
            # Absolute difference
            ax_abs = axes[i, 2]
            im3 = ax_abs.imshow(abs_diff, origin='lower', cmap='viridis', vmin=abs_diff_vmin, vmax=abs_diff_vmax)
            ax_abs.axis('off')
            cbar3 = plt.colorbar(im3, ax=ax_abs, fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=16)
            
            # Percent difference
            ax_pct = axes[i, 3]
            im4 = ax_pct.imshow(percent_diff, origin='lower', cmap='plasma', vmin=percent_diff_vmin, vmax=percent_diff_vmax)
            ax_pct.axis('off')
            cbar4 = plt.colorbar(im4, ax=ax_pct, fraction=0.046, pad=0.04)
            cbar4.ax.tick_params(labelsize=16)
            
            # Uncertainty (standard deviation)
            ax_std = axes[i, 4]
            im5 = ax_std.imshow(std_arr, origin='lower', cmap='plasma', vmin=uncertainty_vmin, vmax=uncertainty_vmax)
            ax_std.axis('off')
            cbar5 = plt.colorbar(im5, ax=ax_std, fraction=0.046, pad=0.04)
            cbar5.ax.tick_params(labelsize=16)
            
    else:
        # Simple mode: just predicted images
        cols = min(4, num_images)  # Maximum 4 columns
        rows = (num_images + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        fig.subplots_adjust(hspace=0.08, wspace=0.05)  # Make images closer together
        plot_title = 'summary_all_predicted.png'
        
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Add column header at the top
        if num_images == 1:
            axes[0].text(0.5, 1.25, 'Predicted', transform=axes[0].transAxes, 
                        ha='center', va='bottom', fontsize=28, fontweight='bold')
        else:
            # For multiple images, add "Predicted" header above the first row
            for j in range(min(cols, num_images)):
                if j < len(axes):
                    axes[j].text(0.5, 1.25, 'Predicted' if j == cols//2 else '', 
                               transform=axes[j].transAxes, ha='center', va='bottom', 
                               fontsize=28, fontweight='bold')
        
        for i, (arr, idx) in enumerate(zip(rendered_images, indices)):
            ax = axes[i]
            im = ax.imshow(arr, origin='lower', cmap='inferno')
            
            # Get lat/lon for this image
            hgln = hgln_values[i] if hgln_values else 0.0
            hglt = hglt_values[i] if hglt_values else 0.0
            
            # Add coordinates as text below the image (without index)
            ax.text(0.5, -0.18, f'HGLN={hgln:.1f}° HGLT={hglt:.1f}°', 
                   transform=ax.transAxes, ha='center', va='top', 
                   fontsize=18, fontweight='bold')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=16)
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].set_visible(False)
    
    
    summary_path = os.path.join(output_dir, plot_title)
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot: {summary_path}")


def find_indices_by_coordinates(dataset, target_coordinates, tolerance=1.0):
    """
    Find dataset indices that match the specified HGLN/HGLT coordinates.
    
    Args:
        dataset: SolarPerspectiveDataset instance
        target_coordinates: List of tuples [(hgln1, hglt1), (hgln2, hglt2), ...]
        tolerance: Maximum allowed difference in degrees for coordinate matching
    
    Returns:
        List of indices matching the target coordinates
    """
    found_indices = []
    
    for target_hgln, target_hglt in target_coordinates:
        best_idx = None
        best_distance = float('inf')
        
        # Search through dataset to find closest matching coordinates
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_hgln = sample['hgln'].item()
            sample_hglt = sample['hglt'].item()
            
            # Calculate distance between target and sample coordinates
            distance = ((sample_hgln - target_hgln) ** 2 + (sample_hglt - target_hglt) ** 2) ** 0.5
            
            if distance < best_distance and distance <= tolerance:
                best_distance = distance
                best_idx = idx
        
        if best_idx is not None:
            found_indices.append(best_idx)
            sample = dataset[best_idx]
            actual_hgln = sample['hgln'].item()
            actual_hglt = sample['hglt'].item()
            print(f"Target (HGLN={target_hgln:.1f}°, HGLT={target_hglt:.1f}°) -> Index {best_idx} (HGLN={actual_hgln:.1f}°, HGLT={actual_hglt:.1f}°)")
        else:
            print(f"Warning: No match found for coordinates (HGLN={target_hgln:.1f}°, HGLT={target_hglt:.1f}°) within tolerance {tolerance}°")
    
    return found_indices


def render_from_ensemble(
    checkpoint_dir: str,
    hgln: float = None,
    hglt: float = None,
    image_size=(64, 64),
    device: str = 'cpu',
    output_dir: str = 'outputs',
    save_npy: bool = False,
    indices: list = None,
    num_images: int = 1,
    compare_target: bool = False,
    coordinates: list = None
):
   
    # Find all checkpoint files starting with "solar-nerf" - use ** to match any number of characters
    checkpoint_pattern = os.path.join(checkpoint_dir, "solar-nerf*.ckpt")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern, recursive=False))
    
    # Alternative: also try finding any .ckpt files containing "solar-nerf" in the name
    if not checkpoint_files:
        alt_pattern = os.path.join(checkpoint_dir, "*solar-nerf*.ckpt")
        checkpoint_files = sorted(glob.glob(alt_pattern, recursive=False))
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
    
    # Limit to 10 checkpoints
    checkpoint_files = checkpoint_files[:10]
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for i, ckpt in enumerate(checkpoint_files):
        print(f"  {i+1}: {os.path.basename(ckpt)}")

    dataset = SolarPerspectiveDataset(
            data_dir="perspective_data/perspective_data_linear_fits_64x64",  # Dummy, won't be used
            dx=1e7,
            device=device,
            normalize=True,
            normalization_value=139.0,
        )
    
    # Determine which indices to render
    if coordinates is not None:
        # Convert coordinates to list of tuples if needed
        if isinstance(coordinates[0], (int, float)):
            # Single coordinate pair given as [hgln, hglt]
            coordinates = [tuple(coordinates)]
        elif isinstance(coordinates[0], list):
            # Multiple coordinate pairs given as [[hgln1, hglt1], [hgln2, hglt2], ...]
            coordinates = [tuple(coord) for coord in coordinates]
        # coordinates is already a list of tuples
        
        print(f"Finding dataset indices for coordinates: {coordinates}")
        indices = find_indices_by_coordinates(dataset, coordinates, tolerance=2.0)
        
        if not indices:
            raise ValueError(f"No dataset samples found matching coordinates: {coordinates}")
            
    elif indices is None:
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
    
    mean_images = []
    std_images = []
    target_images = [] if compare_target else None
    hgln_values = []
    hglt_values = []
    
    for i, index in enumerate(indices):
        print(f"\nRendering image {i+1}/{len(indices)} (dataset index {index})...")
        
        sample = dataset[index]
        header = sample['header']
        
        # Extract actual hgln, hglt from sample if not provided
        actual_hgln = sample.get('hgln', hgln if hgln is not None else 0.0)
        actual_hglt = sample.get('hglt', hglt if hglt is not None else 0.0)

        # Collect predictions from all checkpoints
        ensemble_predictions = []
        
        for j, checkpoint_file in enumerate(checkpoint_files):
            print(f"  Loading checkpoint {j+1}/{len(checkpoint_files)}: {os.path.basename(checkpoint_file)}")
            
            model = RayWiseLightningModule.load_from_checkpoint(checkpoint_file, map_location=device)
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                rendered = model.render_sample_image(header, device=device)
            
            rendered_cpu = rendered.detach().cpu().float().numpy()
            ensemble_predictions.append(rendered_cpu)
            
            # Clean up model to free memory
            del model
            torch.cuda.empty_cache() if device.startswith('cuda') else None
        
        # Calculate ensemble statistics
        ensemble_predictions = np.array(ensemble_predictions)  # Shape: (num_models, H, W)
        mean_prediction = np.mean(ensemble_predictions, axis=0)
        std_prediction = np.std(ensemble_predictions, axis=0)
        
        mean_images.append(mean_prediction)
        std_images.append(std_prediction)
        hgln_values.append(actual_hgln)
        hglt_values.append(actual_hglt)
        
        print(f"Ensemble stats:")
        print(f"  Mean image: min={mean_prediction.min():.5f} max={mean_prediction.max():.5f} mean={mean_prediction.mean():.5f}")
        print(f"  Std image: min={std_prediction.min():.5f} max={std_prediction.max():.5f} mean={std_prediction.mean():.5f}")

        # Get target image if comparison is requested
        if compare_target:
            target = sample['image'].detach().cpu().float().numpy()
            target_images.append(target)
            
            # Calculate comparison metrics with ensemble mean
            abs_diff = np.abs(mean_prediction - target)
            percent_diff = np.where(target != 0, 100 * abs_diff / np.abs(target), 0)
            
            print(f"Comparison metrics (ensemble mean vs target):")
            print(f"  Mean absolute difference: {abs_diff.mean():.5f}")
            print(f"  Max absolute difference: {abs_diff.max():.5f}")
            print(f"  Mean percent difference: {percent_diff.mean():.2f}%")
            print(f"  Max percent difference: {percent_diff.max():.2f}%")

        # Save ensemble mean image
        plt.figure(figsize=(5,5))
        plt.imshow(mean_prediction, origin='lower', cmap='inferno')
        plt.colorbar(label='Intensity')
        plt.title(f'Ensemble Mean | Index {index} | HGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
        mean_output_png = os.path.join(output_dir, f'ensemble_mean_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.png')
        plt.savefig(mean_output_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ensemble mean PNG: {mean_output_png}")

        # Save ensemble standard deviation image
        plt.figure(figsize=(5,5))
        plt.imshow(std_prediction, origin='lower', cmap='plasma')
        plt.colorbar(label='Standard Deviation')
        plt.title(f'Ensemble Std Dev | Index {index} | HGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
        std_output_png = os.path.join(output_dir, f'ensemble_std_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.png')
        plt.savefig(std_output_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ensemble std PNG: {std_output_png}")

        # Create comparison plot if target is available
        if compare_target:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 1: Mean, Target, Absolute Difference
            im1 = axes[0,0].imshow(mean_prediction, origin='lower', cmap='inferno')
            axes[0,0].set_title(f'Ensemble Mean | Index {index}\nHGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
            
            im2 = axes[0,1].imshow(target, origin='lower', cmap='inferno')
            axes[0,1].set_title(f'Target | Index {index}\nHGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
            axes[0,1].axis('off')
            plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
            
            abs_diff = np.abs(mean_prediction - target)
            im3 = axes[0,2].imshow(abs_diff, origin='lower', cmap='viridis')
            axes[0,2].set_title(f'Absolute Difference\nHGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
            axes[0,2].axis('off')
            plt.colorbar(im3, ax=axes[0,2], fraction=0.046, pad=0.04)
            
            # Row 2: Std Dev, Percent Difference, Uncertainty vs Error
            im4 = axes[1,0].imshow(std_prediction, origin='lower', cmap='plasma')
            axes[1,0].set_title(f'Ensemble Std Dev\nHGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
            axes[1,0].axis('off')
            plt.colorbar(im4, ax=axes[1,0], fraction=0.046, pad=0.04)
            
            percent_diff = np.where(target != 0, 100 * abs_diff / np.abs(target), 0)
            im5 = axes[1,1].imshow(percent_diff, origin='lower', cmap='plasma', vmin=0, vmax=100)
            axes[1,1].set_title(f'Percent Difference (%)\nHGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
            axes[1,1].axis('off')
            plt.colorbar(im5, ax=axes[1,1], fraction=0.046, pad=0.04)
            
            # Enhanced uncertainty vs error comparison with 2D histogram and marginals
            std_flat = std_prediction.flatten()
            err_flat = abs_diff.flatten()
            
            # Remove invalid points for correlation analysis
            valid_mask = (std_flat > 1e-8) & (err_flat >= 0) & np.isfinite(std_flat) & np.isfinite(err_flat)
            std_clean = std_flat[valid_mask]
            err_clean = err_flat[valid_mask]
            
            if len(std_clean) > 10:
                # Replace the single subplot with a gridspec for marginal distributions
                from matplotlib.gridspec import GridSpec
                
                # Remove the existing subplot and create a new gridspec
                axes[1,2].remove()
                
                # Create a 3x3 grid in the space of the removed subplot
                gs = GridSpec(3, 3, left=0.67, right=0.99, bottom=0.08, top=0.47, 
                             hspace=0.02, wspace=0.02)
                
                # Main 2D histogram plot
                ax_main = fig.add_subplot(gs[1:, :-1])
                
                # Marginal distribution plots
                ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
                ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
                
                # Create 2D histogram
                bins = 50
                h, xedges, yedges = np.histogram2d(std_clean, err_clean, bins=bins)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                # Plot 2D histogram
                im = ax_main.imshow(h.T, origin='lower', extent=extent, aspect='auto', 
                                   cmap='Blues', interpolation='bilinear')
                
                # Add colorbar for the 2D histogram
                cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
                cbar.set_label('Count', fontsize=8)
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(std_clean, err_clean)[0, 1]
                
                # Add line of best fit
                if not np.isnan(correlation) and len(std_clean) > 1:
                    # Linear regression
                    coeffs = np.polyfit(std_clean, err_clean, 1)
                    poly_func = np.poly1d(coeffs)
                    
                    # Generate line points
                    x_line = np.linspace(std_clean.min(), std_clean.max(), 100)
                    y_line = poly_func(x_line)
                    
                    # Plot line of best fit
                    ax_main.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2, 
                               label=f'Best fit (slope={coeffs[0]:.3f})')
                    
                    # Add legend
                    ax_main.legend(fontsize=8, loc='upper left')
                    
                    # Print correlation analysis to console
                    print(f"Uncertainty-Error Analysis:")
                    print(f"  Correlation coefficient: {correlation:.4f}")
                    print(f"  Line of best fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
                    print(f"  Valid data points: {len(std_clean)}/{len(std_flat)} ({100*len(std_clean)/len(std_flat):.1f}%)")
                    
                    # Calculate R-squared
                    y_pred = poly_func(std_clean)
                    ss_res = np.sum((err_clean - y_pred) ** 2)
                    ss_tot = np.sum((err_clean - np.mean(err_clean)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    print(f"  R-squared: {r_squared:.4f}")
                
                # Top marginal (std dev distribution)
                ax_top.hist(std_clean, bins=bins//2, alpha=0.7, color='blue', density=True)
                ax_top.set_ylabel('Density', fontsize=8)
                ax_top.tick_params(labelbottom=False, labelsize=8)
                ax_top.set_title(f'Uncertainty vs Error\nρ = {correlation:.3f}', fontsize=9)
                
                # Right marginal (error distribution)
                ax_right.hist(err_clean, bins=bins//2, alpha=0.7, color='red', 
                             orientation='horizontal', density=True)
                ax_right.set_xlabel('Density', fontsize=8)
                ax_right.tick_params(labelleft=False, labelsize=8)
                
                # Main plot labels
                ax_main.set_xlabel('Ensemble Std Dev', fontsize=9)
                ax_main.set_ylabel('Absolute Error', fontsize=9)
                ax_main.tick_params(labelsize=8)
                ax_main.grid(True, alpha=0.3)
                
            else:
                # Fallback for insufficient data
                axes[1,2].text(0.5, 0.5, 'Insufficient data\nfor 2D histogram', 
                              transform=axes[1,2].transAxes, ha='center', va='center', 
                              fontsize=12)
                axes[1,2].set_title('Uncertainty vs Error\n(insufficient data)')
                axes[1,2].set_xlabel('Ensemble Std Dev')
                axes[1,2].set_ylabel('Absolute Error')
                print(f"Uncertainty-Error Analysis: Insufficient data points ({len(std_clean)})")
            
            plt.suptitle(f'Ensemble Analysis | HGLN={actual_hgln:.1f}° HGLT={actual_hglt:.1f}°')
            comparison_output_png = os.path.join(output_dir, f'ensemble_comparison_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.png')
            plt.savefig(comparison_output_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved ensemble comparison PNG: {comparison_output_png}")

        if save_npy:
            # Save ensemble statistics
            mean_npy_path = os.path.join(output_dir, f'ensemble_mean_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.npy')
            np.save(mean_npy_path, mean_prediction.astype(np.float32))
            print(f"Saved ensemble mean NumPy array: {mean_npy_path}")
            
            std_npy_path = os.path.join(output_dir, f'ensemble_std_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.npy')
            np.save(std_npy_path, std_prediction.astype(np.float32))
            print(f"Saved ensemble std NumPy array: {std_npy_path}")
            
            if compare_target:
                target_npy_path = os.path.join(output_dir, f'target_idx{index}_hgln{actual_hgln:.1f}_hglt{actual_hglt:.1f}.npy')
                np.save(target_npy_path, target.astype(np.float32))
                print(f"Saved target NumPy array: {target_npy_path}")
    
    # Create a summary plot if multiple images
    if len(mean_images) > 1:
        create_summary_plot(mean_images, indices, output_dir, target_images, hgln_values, hglt_values, std_images)

    return mean_images, std_images, target_images


def parse_args():
    p = argparse.ArgumentParser(description='Render image(s) from ensemble of Solar NeRF checkpoints')
    p.add_argument('--checkpoint-dir', type=str, default='SWA_checkpoints', help='Directory containing checkpoint files')
    p.add_argument('--hgln', type=float, default=None, help='Heliographic longitude (optional, will use dataset values if not provided)')
    p.add_argument('--hglt', type=float, default=None, help='Heliographic latitude (optional, will use dataset values if not provided)')
    p.add_argument('--image-size', type=int, nargs=2, default=[64, 64], help='Image size H W')
    p.add_argument('--device', type=str, default='cuda:0', help='cpu | cuda:0 | auto')
    p.add_argument('--output-dir', type=str, default='ensemble_outputs', help='Output directory for images')
    p.add_argument('--save-npy', action='store_true', help='Also save raw array .npy files')
    p.add_argument('--num-images', type=int, default=1, help='Number of images to render (starting from index 0)')
    p.add_argument('--indices', type=int, nargs='+', default=None, help='Specific dataset indices to render (overrides --num-images)')
    p.add_argument('--coordinates', type=float, nargs='+', default=None, 
                   help='Specify images by HGLN/HGLT coordinates. Format: --coordinates hgln1 hglt1 hgln2 hglt2 ... (overrides --indices and --num-images)')
    p.add_argument('--compare-target', action='store_true', help='Compare rendered images with target images (absolute and percent difference)')
    return p.parse_args()


def main():
    args = parse_args()
    
    # Parse coordinates if provided
    coordinates = None
    if args.coordinates is not None:
        if len(args.coordinates) % 2 != 0:
            raise ValueError("Coordinates must be provided in pairs (hgln, hglt). Got odd number of values.")
        
        # Group coordinates into pairs
        coordinates = []
        for i in range(0, len(args.coordinates), 2):
            hgln, hglt = args.coordinates[i], args.coordinates[i+1]
            coordinates.append((hgln, hglt))
        
        print(f"Requested coordinates: {coordinates}")
    
    render_from_ensemble(
        checkpoint_dir=args.checkpoint_dir,
        hgln=args.hgln,
        hglt=args.hglt,
        image_size=tuple(args.image_size),
        device=args.device,
        output_dir=args.output_dir,
        save_npy=args.save_npy,
        indices=args.indices,
        num_images=args.num_images,
        compare_target=args.compare_target,
        coordinates=coordinates,
    )


if __name__ == '__main__':
    main()
