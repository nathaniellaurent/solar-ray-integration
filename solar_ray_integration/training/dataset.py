"""
Dataset module for Solar Perspective data.

This module provides PyTorch Dataset classes for loading and processing
solar perspective data stored as FITS files.
"""

import os
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import warnings
from astropy.utils.exceptions import AstropyWarning

# Suppress Astropy warnings
warnings.filterwarnings("ignore", category=AstropyWarning)


class SolarPerspectiveDataset(Dataset):
    """
    Dataset for loading solar perspective data from FITS files.
    
    Each sample contains:
    - Viewing angles (hgln, hglt)
    - WCS information for ray generation
    - Target image data
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalization_value: float = 1.0
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing FITS files
            transform: Optional transform to apply to images
            normalize: Whether to normalize image values
            normalization_value: Fixed value to divide pixel values by for normalization
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        self.normalization_value = normalization_value
        
        # Find all FITS files
        self.fits_files = glob.glob(os.path.join(data_dir, "*.fits"))
        self.fits_files.sort()
        
        if len(self.fits_files) == 0:
            raise ValueError(f"No FITS files found in {data_dir}")
        
        # Parse metadata from filenames
        self.metadata = self._parse_metadata()
        
        print(f"Found {len(self.fits_files)} FITS files in {data_dir}")
        
    def _parse_metadata(self) -> List[Dict[str, float]]:
        """Parse viewing angles from filenames."""
        metadata = []
        pattern = r"output_tensor_(\d+)_hgln_(-?\d+\.?\d*)_hglt_(-?\d+\.?\d*)\.fits"
        
        for fits_file in self.fits_files:
            filename = os.path.basename(fits_file)
            match = re.match(pattern, filename)
            
            if match:
                idx, hgln, hglt = match.groups()
                metadata.append({
                    'idx': int(idx),
                    'hgln': float(hgln),
                    'hglt': float(hglt),
                    'filename': fits_file
                })
            else:
                raise ValueError(f"Could not parse filename: {filename}")
                
        return metadata
    
    def __len__(self) -> int:
        return len(self.fits_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
            - 'image': Target image tensor [H, W]
            - 'hgln': Heliographic longitude
            - 'hglt': Heliographic latitude
            - 'wcs_params': WCS parameters as tensor
        """
        metadata = self.metadata[idx]
        fits_file = metadata['filename']
        
        # Load FITS file
        with fits.open(fits_file) as hdul:
            hdu = hdul[0]
            image_data = hdu.data.astype(np.float32)
            header = hdu.header
        
        # Extract WCS
        wcs = WCS(header)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_data)
        
        # Note: We don't resize images to preserve WCS accuracy
        # Image resizing would invalidate pixel scale and WCS parameters
        
        # Normalize if requested
        if self.normalize:
            # Normalize by dividing by the specified normalization value
            image_tensor = image_tensor / self.normalization_value
        
        # Extract WCS parameters (simplified)
        wcs_params = torch.tensor([
            header.get('CRPIX1', 0.0),
            header.get('CRPIX2', 0.0),
            header.get('CDELT1', 0.0),
            header.get('CDELT2', 0.0),
            header.get('CRVAL1', 0.0),
            header.get('CRVAL2', 0.0),
            header.get('HGLN_OBS', metadata['hgln']),
            header.get('HGLT_OBS', metadata['hglt']),
        ], dtype=torch.float32)
        
        sample = {
            'image': image_tensor,
            'hgln': torch.tensor(metadata['hgln'], dtype=torch.float32),
            'hglt': torch.tensor(metadata['hglt'], dtype=torch.float32),
            'wcs_params': wcs_params,
            'header': header  # Keep original header for WCS reconstruction
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class SolarPerspectiveDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Solar Perspective data.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        normalize: bool = True,
        normalization_value: float = 139.0,
        seed: int = 42
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing FITS files
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            normalize: Whether to normalize images
            normalization_value: Fixed value to divide pixel values by for normalization
            seed: Random seed for data splitting
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.normalization_value = normalization_value
        self.seed = seed
        
        # Ensure splits sum to 1
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        # Create full dataset
        full_dataset = SolarPerspectiveDataset(
            data_dir=self.data_dir,
            normalize=self.normalize,
            normalization_value=self.normalization_value
        )
        
        # Split dataset
        dataset_size = len(full_dataset)
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Use torch generator for reproducible splits
        generator = torch.Generator().manual_seed(self.seed)
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for handling variable-sized data.
    """
    # Standard collation for tensors
    collated = {}
    
    for key in batch[0].keys():
        if key == 'header':
            # Keep headers as list
            collated[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            # Keep as list for non-tensor items
            collated[key] = [item[key] for item in batch]
    
    return collated
