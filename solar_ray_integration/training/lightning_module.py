"""
PyTorch Lightning module for training Solar NeRF models.

This module defines the training, validation, and testing procedures
for neural radiance fields applied to solar perspective data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os

from ..rendering.neural_renderer import NeuralSolarRenderer
from ..model.model import NeRF
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS


class SolarNerfLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Solar NeRF models.
    
    This module handles the training loop, loss computation, and logging
    for neural radiance fields applied to solar perspective rendering.
    """
    
    def __init__(
        self,
        nerf_config: Optional[Dict] = None,
        integration_method: str = "linear",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-6,
        scheduler_type: str = "cosine",
        loss_type: str = "mse",
        dx: float = 1e7,
        device_type: str = "cuda:0",
        log_images_every_n_epochs: int = 10,
        **kwargs
    ):
        """
        Initialize the Lightning module.
        
        Args:
            nerf_config: Configuration for the NeRF model
            integration_method: Ray integration method
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler_type: Type of learning rate scheduler
            loss_type: Type of loss function
            dx: Step size for ray integration
            device_type: Device for computation
            log_images_every_n_epochs: How often to log sample images
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Default NeRF configuration
        if nerf_config is None:
            nerf_config = {
                'd_input': 3,
                'd_output': 1,
                'n_layers': 8,
                'd_filter': 512,
                'encoding': 'positional'
            }
        
        # Initialize the neural renderer
        self.neural_renderer = NeuralSolarRenderer(
            nerf_config=nerf_config,
            dx=dx,
            device=None,  # Will be set dynamically
            integration_method=integration_method
        )
        
        # Loss function
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch of data containing headers and other info
            
        Returns:
            Rendered images
        """
        batch_size = len(batch['header'])
        rendered_images = []
        
        for i in range(batch_size):
            # Reconstruct HDU from header and viewing angles
            header = batch['header'][i]
            
            # Create a dummy image for ray calculation (we only need the header/WCS)
            dummy_data = np.ones((header['NAXIS2'], header['NAXIS1']), dtype=np.float32)
            source_hdu = PrimaryHDU(dummy_data, header=header)
            
            # Render using neural renderer
            rendered = self.neural_renderer(source_hdu, requires_grad=True)
            rendered_images.append(rendered)
        
        return torch.stack(rendered_images)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        predicted = self.forward(batch)
        target = batch['image']
        
        # Ensure same shape
        if predicted.shape != target.shape:
            predicted = F.interpolate(
                predicted.unsqueeze(1) if len(predicted.shape) == 3 else predicted,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Compute loss
        loss = self.loss_fn(predicted, target)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        # Store for epoch-level statistics
        self.train_losses.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        predicted = self.forward(batch)
        target = batch['image']
        
        # Ensure same shape
        if predicted.shape != target.shape:
            predicted = F.interpolate(
                predicted.unsqueeze(1) if len(predicted.shape) == 3 else predicted,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Compute loss
        loss = self.loss_fn(predicted, target)
        
        # Additional metrics
        mae = F.l1_loss(predicted, target)
        mse = F.mse_loss(predicted, target)
        
        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_mse', mse, on_epoch=True)
        
        # Store for epoch-level statistics
        self.val_losses.append(loss.detach())
        
        # Log sample images periodically
        if (batch_idx == 0 and 
            self.current_epoch % self.hparams.log_images_every_n_epochs == 0):
            self._log_sample_images(predicted, target, batch)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        # Forward pass
        predicted = self.forward(batch)
        target = batch['image']
        
        # Ensure same shape
        if predicted.shape != target.shape:
            predicted = F.interpolate(
                predicted.unsqueeze(1) if len(predicted.shape) == 3 else predicted,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Compute metrics
        loss = self.loss_fn(predicted, target)
        mae = F.l1_loss(predicted, target)
        mse = F.mse_loss(predicted, target)
        
        # Logging
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Scheduler
        if self.hparams.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
    
    def _log_sample_images(self, predicted: torch.Tensor, target: torch.Tensor, batch: Dict[str, torch.Tensor]):
        """Log sample images to tensorboard."""
        # Take first image from batch
        pred_img = predicted[0].detach().cpu().numpy()
        target_img = target[0].detach().cpu().numpy()
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target image
        im1 = axes[0].imshow(target_img, origin='lower', cmap='inferno')
        axes[0].set_title('Target')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted image
        im2 = axes[1].imshow(pred_img, origin='lower', cmap='inferno')
        axes[1].set_title('Predicted')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = np.abs(target_img - pred_img)
        im3 = axes[2].imshow(diff, origin='lower', cmap='viridis')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        # Add viewing angles to title
        hgln = batch['hgln'][0].item()
        hglt = batch['hglt'][0].item()
        fig.suptitle(f'Epoch {self.current_epoch}, HGLN={hgln:.1f}°, HGLT={hglt:.1f}°')
        
        # Log to tensorboard
        self.logger.experiment.add_figure(
            'sample_images',
            fig,
            global_step=self.current_epoch
        )
        
        plt.close(fig)
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_losses:
            avg_train_loss = torch.stack(self.train_losses).mean()
            self.log('train_loss_epoch', avg_train_loss)
            self.train_losses.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_losses:
            avg_val_loss = torch.stack(self.val_losses).mean()
            self.log('val_loss_epoch', avg_val_loss)
            self.val_losses.clear()
