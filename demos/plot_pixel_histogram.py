"""
Plot histogram of pixel values from perspective_data_linear_fits_small FITS files.

This script loads all FITS files from the perspective_data_linear_fits_small directory
and creates histograms of their pixel values to analyze the distribution of 
integrated field values across different viewing angles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob

def load_fits_data(data_dir):
    """
    Load all FITS files from the specified directory.
    
    Args:
        data_dir: Path to directory containing FITS files
        
    Returns:
        list: List of tuples (filename, data_array, header)
    """
    fits_files = glob.glob(os.path.join(data_dir, "*.fits"))
    fits_files.sort()  # Sort for consistent ordering
    
    data_list = []
    
    print(f"Found {len(fits_files)} FITS files in {data_dir}")
    
    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
                # Extract viewing angles from header if available
                hgln = header.get('HGLN_OBS', 'Unknown')
                hglt = header.get('HGLT_OBS', 'Unknown')
                
                filename = os.path.basename(fits_file)
                data_list.append((filename, data, header, hgln, hglt))
                
        except Exception as e:
            print(f"Error loading {fits_file}: {e}")
    
    return data_list


def plot_combined_histogram(data_list, bins=100, log_scale=False):
    """
    Plot histogram of all pixel values combined.
    
    Args:
        data_list: List of (filename, data, header, hgln, hglt) tuples
        bins: Number of histogram bins
        log_scale: Whether to use log scale on y-axis
    """
    all_pixels = []
    
    for filename, data, header, hgln, hglt in data_list:
        # Flatten the 2D image to 1D array and remove any NaN values
        pixels = data.flatten()
        pixels = pixels[~np.isnan(pixels)]
        all_pixels.extend(pixels)
    
    all_pixels = np.array(all_pixels)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    counts, bin_edges, patches = plt.hist(all_pixels, bins=bins, alpha=0.7, 
                                         edgecolor='black', linewidth=0.5)
    
    # Add statistics text
    stats_text = f"""Statistics (All Files):
    Total pixels: {len(all_pixels):,}
    Min: {all_pixels.min():.6f}
    Max: {all_pixels.max():.6f}
    Mean: {all_pixels.mean():.6f}
    Std: {all_pixels.std():.6f}
    Median: {np.median(all_pixels):.6f}
    Non-zero pixels: {np.count_nonzero(all_pixels):,} ({100*np.count_nonzero(all_pixels)/len(all_pixels):.1f}%)"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.title(f'Histogram of All Pixel Values\n{len(data_list)} FITS files from perspective_data_linear_fits_small')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Count (log scale)')
    
    plt.tight_layout()
    return plt.gcf()


def plot_individual_histograms(data_list, max_plots=9):
    """
    Plot histograms for individual files in a grid.
    
    Args:
        data_list: List of (filename, data, header, hgln, hglt) tuples
        max_plots: Maximum number of individual plots to show
    """
    n_plots = min(len(data_list), max_plots)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        filename, data, header, hgln, hglt = data_list[i]
        pixels = data.flatten()
        pixels = pixels[~np.isnan(pixels)]
        
        ax.hist(pixels, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'HGLN={hgln}°, HGLT={hglt}°', fontsize=10)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add basic stats
        stats_text = f'Mean: {pixels.mean():.4f}\nStd: {pixels.std():.4f}\nNon-zero: {np.count_nonzero(pixels)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_viewing_angle_statistics(data_list):
    """
    Plot statistics as a function of viewing angles.
    
    Args:
        data_list: List of (filename, data, header, hgln, hglt) tuples
    """
    hgln_values = []
    hglt_values = []
    means = []
    stds = []
    non_zero_fractions = []
    
    for filename, data, header, hgln, hglt in data_list:
        if hgln != 'Unknown' and hglt != 'Unknown':
            pixels = data.flatten()
            pixels = pixels[~np.isnan(pixels)]
            
            hgln_values.append(float(hgln))
            hglt_values.append(float(hglt))
            means.append(pixels.mean())
            stds.append(pixels.std())
            non_zero_fractions.append(np.count_nonzero(pixels) / len(pixels))
    
    hgln_values = np.array(hgln_values)
    hglt_values = np.array(hglt_values)
    means = np.array(means)
    stds = np.array(stds)
    non_zero_fractions = np.array(non_zero_fractions)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean vs HGLN
    axes[0, 0].scatter(hgln_values, means, c=hglt_values, cmap='coolwarm', alpha=0.7)
    axes[0, 0].set_xlabel('Heliographic Longitude (degrees)')
    axes[0, 0].set_ylabel('Mean Pixel Value')
    axes[0, 0].set_title('Mean Pixel Value vs HGLN')
    axes[0, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar1.set_label('HGLT (degrees)')
    
    # Plot 2: Mean vs HGLT
    axes[0, 1].scatter(hglt_values, means, c=hgln_values, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('Heliographic Latitude (degrees)')
    axes[0, 1].set_ylabel('Mean Pixel Value')
    axes[0, 1].set_title('Mean Pixel Value vs HGLT')
    axes[0, 1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar2.set_label('HGLN (degrees)')
    
    # Plot 3: Std vs viewing angles
    axes[1, 0].scatter(hgln_values, stds, c=hglt_values, cmap='coolwarm', alpha=0.7)
    axes[1, 0].set_xlabel('Heliographic Longitude (degrees)')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_title('Pixel Value Std vs HGLN')
    axes[1, 0].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar3.set_label('HGLT (degrees)')
    
    # Plot 4: Non-zero fraction
    axes[1, 1].scatter(hgln_values, non_zero_fractions, c=hglt_values, cmap='coolwarm', alpha=0.7)
    axes[1, 1].set_xlabel('Heliographic Longitude (degrees)')
    axes[1, 1].set_ylabel('Fraction of Non-zero Pixels')
    axes[1, 1].set_title('Non-zero Pixel Fraction vs HGLN')
    axes[1, 1].grid(True, alpha=0.3)
    cbar4 = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar4.set_label('HGLT (degrees)')
    
    plt.tight_layout()
    return fig


def main():
    """Main function to generate all plots."""
    # Define data directory
    data_dir = "../perspective_data/perspective_data_linear_fits_small"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        print("Make sure you've run generate_perspectives.py to create the data first.")
        return
    
    # Load all FITS data
    print("Loading FITS files...")
    data_list = load_fits_data(data_dir)
    
    if not data_list:
        print("No FITS files found!")
        return
    
    print(f"Loaded {len(data_list)} FITS files successfully.")
    
    # Create plots
    print("Creating combined histogram...")
    fig1 = plot_combined_histogram(data_list, bins=100, log_scale=False)
    plt.savefig('pixel_histogram_combined.png', dpi=150, bbox_inches='tight')
    
    print("Creating combined histogram (log scale)...")
    fig2 = plot_combined_histogram(data_list, bins=100, log_scale=True)
    plt.savefig('pixel_histogram_combined_log.png', dpi=150, bbox_inches='tight')
    
    print("Creating individual histograms...")
    fig3 = plot_individual_histograms(data_list, max_plots=9)
    plt.savefig('pixel_histogram_individual.png', dpi=150, bbox_inches='tight')
    
    print("Creating viewing angle statistics...")
    fig4 = plot_viewing_angle_statistics(data_list)
    plt.savefig('pixel_statistics_vs_angles.png', dpi=150, bbox_inches='tight')
    
    print("All plots saved!")
    print("Files saved:")
    print("  - pixel_histogram_combined.png")
    print("  - pixel_histogram_combined_log.png") 
    print("  - pixel_histogram_individual.png")
    print("  - pixel_statistics_vs_angles.png")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
