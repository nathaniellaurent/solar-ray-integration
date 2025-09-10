"""
Ray sample dataset: provides (ray_along_steps, pixel_value) pairs instead of full images.

For each FITS image in data_dir, a RayIntegrator is used to generate a tensor of shape (H, W, S, 3)
containing per-step 3D field samples (e.g. vector components) along each pixel ray. Each dataset
sample corresponds to one pixel (h, w):
  - ray: Tensor[S,3]  (per-step 3-component values retained, NOT flattened)
  - pixel: Tensor[] (observed pixel intensity from the FITS image)
  - hgln, hglt: viewing angles
  - meta indices (image_idx, h, w)

This allows training models directly on ray -> pixel mappings while preserving the step and
component structure (S,3) needed for sequence / volumetric models.

Notes:
- Computing (H,W,S,3) for every image can be expensive. This implementation computes rays lazily
  and caches only the most recently accessed image (LRU of size 1) to control memory usage.
- Assumes constant number of steps S across images; if it changes, an error is raised.
"""

from __future__ import annotations
import os, glob, re
from typing import List, Dict, Any, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dfreproject import calculate_rays



from ..ray_integration.integrate_field import RayIntegrator

RSUN = int(6.9634e8)


class SolarRayPixelDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dx: float = 1e7,
        device: str = 'cuda:0',
        requires_grad: bool = False,
        normalize: bool = True,
        normalization_value: float = 1.0,
    ):
        self.data_dir = data_dir
        self.dx = dx
        self.device = device
        self.requires_grad = requires_grad
        self.normalize = normalize
        self.normalization_value = normalization_value

        self.fits_files = sorted(glob.glob(os.path.join(data_dir, '*.fits')))
        if not self.fits_files:
            raise ValueError(f"No FITS files found in {data_dir}")

        # Parse metadata (hgln, hglt) from filenames: output_tensor_{idx}_hgln_{lon}_hglt_{lat}.fits
        self.meta: List[Dict[str, Any]] = []
        pattern = r"output_tensor_(\d+)_hgln_(-?\d+\.?\d*)_hglt_(-?\d+\.?\d*)\.fits"
        for f in self.fits_files:
            m = re.match(pattern, os.path.basename(f))
            if not m:
                raise ValueError(f"Filename format not recognized: {f}")
            idx, hgln, hglt = m.groups()
            self.meta.append({'filename': f, 'hgln': float(hgln), 'hglt': float(hglt), 'idx': int(idx)})

        # Collect per-image shapes and cumulative pixel counts
        self.image_shapes: List[tuple[int, int]] = []
        self.cumulative_counts: List[int] = []
        total = 0
        for m in self.meta:
            with fits.open(m['filename']) as hdul:
                hdu = hdul[0]
                H, W = hdu.data.shape
            self.image_shapes.append((H, W))
            total += H * W
            self.cumulative_counts.append(total)
        self.total_pixels = total

        # Preload rays and pixels for all images (flattened)
        self.rays_all = []  # List of (H*W, 3)
        self.obs_all = []   # List of (H*W, 3)
        self.pixels = []    # List of (H*W,)

        self.sample_image = None
        self.sample_header = None

        # Set a single sample image and header for reference
        sample_meta = self.meta[0]
        with fits.open(sample_meta['filename']) as hdul:
            hdu = hdul[0]
            self.sample_image = torch.from_numpy(hdu.data.astype(np.float32))
            self.sample_header = hdu.header.copy()


        for m in self.meta:
            with fits.open(m['filename']) as hdul:
                hdu = hdul[0].copy()
                img = torch.from_numpy(hdu.data.astype(np.float32))
            if self.normalize:
                img = img / self.normalization_value

            wcs = WCS(hdu.header)
            obs, rays = calculate_rays(
                source_wcs=wcs,
                shape_out=img.shape,
                requires_grad=True
            ) # (H,W,3)
            # rays: (H,W,3) -> flatten to (H*W,3)
            H, W = img.shape
            rays = rays.reshape(-1, 3)

            img_flatten = img.reshape(-1)
            if img_flatten.shape[0] != rays.shape[0]:
                raise ValueError(f"Image pixel count {img_flatten.shape[0]} does not match rays {rays.shape[0]} for {m['filename']}")

            for i in range(H * W):
                self.rays_all.append(rays[i])
                self.obs_all.append(obs)
                self.pixels.append(img_flatten[i])

        print("Number of images:", len(self.meta))
        print("Length of rays_all:", len(self.rays_all))
        print("Length of obs_all:", len(self.obs_all))
        print("Length of pixels:", len(self.pixels))



    def __len__(self) -> int:
        return self.total_pixels

    def __getitem__(self, index: int) -> Dict[str, Any]:
        
        
        # Get the ray, obs, and pixel directly from flattened arrays
        ray_val = self.rays_all[index].clone()  # (3,)
        obs_val = self.obs_all[index].clone()   # (3,)
        pixel_val = self.pixels[index].clone()  # scalar


        sample = {
            'ray': ray_val,       # (3,)
            'obs': obs_val,       # (3,)
            'pixel': pixel_val,   # scalar
            'sample_image': self.sample_image,
            'sample_header': self.sample_header
        }
        return sample
    
    def get_image(self, image_idx: int) -> tuple[Any, np.ndarray]:
        """
        Returns the FITS header and image data for the given image index.
        """
        if image_idx < 0 or image_idx >= len(self.meta):
            raise IndexError(f"image_idx {image_idx} out of range")
        filename = self.meta[image_idx]['filename']
        with fits.open(filename) as hdul:
            hdu = hdul[0]
            header = hdu.header.copy()
            image = hdu.data.copy()
        return header, image




   


class SolarRayPixelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        dx: float = 1e7,
        device: str = 'cuda:0',
        normalize: bool = True,
        normalization_value: float = 139.0,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        seed: int = 42,
        dsun: float = 151693705216.0,
        rsun: float = 696000000.0

    ):
        super().__init__()
        if train_fraction + val_fraction > 1.0:
            raise ValueError("train_fraction + val_fraction must be <= 1.0")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dx = dx
        self.device = device
        self.normalize = normalize
        self.normalization_value = normalization_value
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.dsun = dsun
        self.rsun = rsun

    def setup(self, stage: Optional[str] = None):
        full = SolarRayPixelDataset(
            data_dir=self.data_dir,
            dx=self.dx,
            device=self.device,
            normalize=self.normalize,
            normalization_value=self.normalization_value,
        )
        N = len(full)
        n_train = int(self.train_fraction * N)
        n_val = int(self.val_fraction * N)
        n_test = N - n_train - n_val
        g = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(full, [n_train, n_val, n_test], generator=g)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.ray_pixel_collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.ray_pixel_collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.ray_pixel_collate)

    def ray_pixel_collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        rays = torch.stack([b['ray'] for b in batch])          # (B,3)
        obs = torch.stack([b['obs'] for b in batch])           # (B,3)
        pixels = torch.stack([b['pixel'] for b in batch])      # (B,)

        # Ensure inputs are float32 and on the correct device
        obs = obs.to(dtype=torch.float32, device=self.device)
        rays = rays.to(dtype=torch.float32, device=self.device)
        
        # Batch size
        B = obs.shape[0]
        
        # Create steps (S,)
        dsun = self.dsun
        rsun = self.rsun
        dx = self.dx
        steps = torch.arange((dsun - 2*rsun), (dsun + 2*rsun), dx, device=self.device, dtype=torch.float32)
        S = steps.shape[0]
        
        # Reshape steps to (S, 1) and then expand for batch: (B, S, 1)
        steps_reshaped = steps.unsqueeze(1)  # (S, 1)
        steps_expanded = steps_reshaped.unsqueeze(0).expand(B, S, 1)  # (B, S, 1)
        # Expand obs and ray for broadcasting: (B, 1, 3)
        obs_expanded = obs.unsqueeze(1)  # (B, 1, 3)
        rays_expanded = rays.unsqueeze(1)  # (B, 1, 3)
        
        # Compute ray positions: (B, S, 3)
        rays_with_steps = obs_expanded + steps_expanded * rays_expanded

        return {
            'rays_with_steps': rays_with_steps,  # (B,S,3)
            'pixels': pixels,
            'sample_header': batch[0]['sample_header'],
            'sample_image': batch[0]['sample_image']
       }
