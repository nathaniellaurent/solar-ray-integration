"""
Utility script for testing and inspecting the training pipeline.

This script provides functions to test data loading, visualize samples,
and validate the training setup.
"""

import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

from .dataset import SolarPerspectiveDataset, SolarPerspectiveDataModule
from .lightning_module import SolarNerfLightningModule
from .config import get_config, create_config_template


def test_dataset(data_dir: str = "perspective_data/perspective_data_linear_fits_small"):
    """Test the dataset loading."""
    print(f"Testing dataset loading from: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return False
    
    try:
        # Create dataset
        dataset = SolarPerspectiveDataset(data_dir, normalize=True)
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
        print(f"  HGLN: {sample['hgln'].item():.1f}°")
        print(f"  HGLT: {sample['hglt'].item():.1f}°")
        print(f"  WCS params shape: {sample['wcs_params'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        return False


def visualize_samples(data_dir: str = "perspective_data/perspective_data_linear_fits_small", 
                     num_samples: int = 4):
    """Visualize sample images from the dataset."""
    try:
        dataset = SolarPerspectiveDataset(data_dir, normalize=True)
        
        # Select samples to visualize
        indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            
            im = axes[i].imshow(sample['image'].numpy(), origin='lower', cmap='inferno')
            axes[i].set_title(f"HGLN={sample['hgln'].item():.1f}°, HGLT={sample['hglt'].item():.1f}°")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.suptitle("Sample Solar Perspective Images", y=1.02)
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error visualizing samples: {e}")
        return False


def test_datamodule(data_dir: str = "perspective_data/perspective_data_linear_fits_small"):
    """Test the PyTorch Lightning data module."""
    print("Testing DataModule...")
    
    try:
        # Create data module
        data_module = SolarPerspectiveDataModule(
            data_dir=data_dir,
            batch_size=2,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        # Setup
        data_module.setup()
        print(f"✓ DataModule setup successfully")
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✓ Train batch loaded successfully")
        print(f"  Batch size: {len(batch['image'])}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  HGLN range: [{batch['hgln'].min():.1f}, {batch['hgln'].max():.1f}]")
        print(f"  HGLT range: [{batch['hglt'].min():.1f}, {batch['hglt'].max():.1f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing DataModule: {e}")
        return False


def test_model():
    """Test model initialization and forward pass."""
    print("Testing model...")
    
    try:
        # Create a simple model
        model = SolarNerfLightningModule(
            integration_method="linear",
            learning_rate=1e-4
        )
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False


def run_full_test(data_dir: str = "perspective_data/perspective_data_linear_fits_small"):
    """Run complete test suite."""
    print("="*50)
    print("Running Solar NeRF Training Pipeline Tests")
    print("="*50)
    
    # Test 1: Dataset
    print("\n1. Testing Dataset...")
    dataset_ok = test_dataset(data_dir)
    
    # Test 2: DataModule
    print("\n2. Testing DataModule...")
    datamodule_ok = test_datamodule(data_dir)
    
    # Test 3: Model
    print("\n3. Testing Model...")
    model_ok = test_model()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"  Dataset: {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"  DataModule: {'✓ PASS' if datamodule_ok else '✗ FAIL'}")
    print(f"  Model: {'✓ PASS' if model_ok else '✗ FAIL'}")
    
    all_pass = dataset_ok and datamodule_ok and model_ok
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    
    if all_pass:
        print("\n Training pipeline is ready!")
        print("\nTo start training, run:")
        print("  python -m solar_ray_integration.training train")
        print("\nOr for a quick test:")
        print("  python -m solar_ray_integration.training train --max-epochs 5 --batch-size 2")
    
    return all_pass


def main():
    """Main function for utility script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Solar NeRF training pipeline")
    parser.add_argument("--data-dir", type=str, 
                       default="perspective_data/perspective_data_linear_fits_small",
                       help="Data directory")
    parser.add_argument("--action", type=str, default="test",
                       choices=["test", "visualize", "config"],
                       help="Action to perform")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    if args.action == "test":
        run_full_test(args.data_dir)
    elif args.action == "visualize":
        visualize_samples(args.data_dir, args.num_samples)
    elif args.action == "config":
        create_config_template()


if __name__ == "__main__":
    main()
