"""
PyTorch Lightning module for training Solar NeRF models.

This module defines the training, validation, and testing procedures
for neural radiance fields applied to solar perspective data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import os
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from ..rendering.ray_renderer import RayRenderer
from ..model.model import NeRF
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS
from ..ray_integration import RayIntegrator
from ..ray_integration.integrate_field import calculate_rays

# Solar radius constant (meters)


class RayWiseLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Solar NeRF models.
    
    This module handles the training loop, loss computation, and logging
    for neural radiance fields applied to solar perspective rendering.
    """
    
    def __init__(
        self,
        nerf_config: Optional[Dict] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-6,
        scheduler_type: str = "cosine",
        loss_type: str = "mse",
        dx: float = 1e7,
        device: str = "cuda:0",
        checkpoint_dir: str = "checkpoints/solar_nerf",
        checkpoint_monitor: str = "val_loss",
        save_top_k: int = 3,
        image_save_dir: str = "outputs/images",
        accumulate_grad_batches: int = 1,
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
            accumulate_grad_batches: Gradient accumulation steps (mirrors Trainer setting for logging purposes)
        """
        super().__init__()
        self.save_hyperparameters()
        # Ensure checkpoint and image save directories exist early
        os.makedirs(self.hparams.checkpoint_dir, exist_ok=True)
        os.makedirs(self.hparams.image_save_dir, exist_ok=True)
        
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
        print(f"Using device: {device} (used for NeuralSolarRenderer and ray integration)")
        self.ray_renderer = RayRenderer(
            nerf_config = nerf_config,
            dx = 1e7,
            device = device
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
        self.dx = dx
    
    def configure_callbacks(self):
        """Configure callbacks: model checkpointing, LR monitoring."""
        ckpt_cb = ModelCheckpoint(
            dirpath=self.hparams.checkpoint_dir,
            filename="solar-nerf-{epoch:02d}-{val_loss:.4f}",
            monitor=self.hparams.checkpoint_monitor,
            mode="min" if "loss" in self.hparams.checkpoint_monitor else "max",
            save_top_k=self.hparams.save_top_k,
            save_last=True,
            auto_insert_metric_name=False
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return [ckpt_cb, lr_monitor]
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch of data containing rays, obs, and other info
            
        Returns:
            Rendered pixels as a 1D tensor (B,)
        """
        # Process entire batch at once instead of looping
        # obs = batch['obs']      # (B, 3)
        # rays = batch['rays']    # (B, 3)
        
        # # Process batch through ray renderer
        # # Detach and clone obs and rays to ensure fresh tensors
        # obs_batch = obs.detach().clone().requires_grad_(True)
        # rays_batch = rays.detach().clone().requires_grad_(True)
        rays_with_steps = batch['rays_with_steps']  # (B, S, 3)
        # Pass the whole batch to ray_renderer (assumes ray_renderer supports batch processing)
        rays_with_steps = rays_with_steps.detach().clone().requires_grad_(True)
        rendered_pixels = self.ray_renderer(rays_with_steps=rays_with_steps, requires_grad=True)
        return rendered_pixels
    
    def render_sample_image(self, header: Any, device: str) -> torch.Tensor:
        """
        Render a full sample image from the provided header using the current model.
        
        Args:
            header: FITS header containing WCS information for ray calculation.
            device: Device to perform rendering on.
        
        Returns:
            Rendered image as a 2D tensor.
        """
        # Calculate rays and observer positions from WCS header
        wcs = WCS(header)
        shape_out = (header['NAXIS2'], header['NAXIS1'])
        obs, rays = calculate_rays(
            source_wcs=wcs,
            shape_out=shape_out,
            device=device
        )
        

        obs = obs.expand(rays.shape[1], rays.shape[2], 3)
        # Flatten obs and rays for batch processing
        obs_flat = obs.reshape(-1, 3)
        rays_flat = rays.reshape(-1, 3)

        rays_with_steps = self.obs_rays_to_rays_with_steps(obs=obs_flat, rays=rays_flat, wcs=wcs)  # (B, S, 3)

        
        rendered_pixels = self.ray_renderer(rays_with_steps=rays_with_steps, requires_grad=False)
        # Unflatten rendered_pixels to image shape
        rendered_image = rendered_pixels.reshape(shape_out)
        return rendered_image

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        predicted = self.forward(batch)
        target = batch['pixels']

        # Ensure same shape
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")
        
        # Compute loss
        loss = self.loss_fn(predicted, target)

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")
        
        
        # Store for epoch-level statistics
        self.train_losses.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        predicted = self.forward(batch)
        target = batch['pixels']


        # Ensure same shape
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape} in validation_step")
        
        # Compute loss
        loss = self.loss_fn(predicted, target)
        
        # Additional metrics
        mae = F.l1_loss(predicted, target)
        mse = F.mse_loss(predicted, target)

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")



        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")

        
        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_mse', mse, on_epoch=True)

        # Log example images for first batch only
        if batch_idx == 0:
            # Retrieve sample image and header from batch
            sample_image = batch['sample_image']  
            header = batch['sample_header']       

            obs, rays = calculate_rays(
                source_wcs=WCS(header),
                shape_out=(sample_image.shape[0], sample_image.shape[1]),
                device=self.hparams.device
            )
            obs = obs.expand(rays.shape[1], rays.shape[2], 3)

            # Flatten obs and rays for batch processing
            obs_flat = obs.reshape(-1, 3)
            rays_flat = rays.reshape(-1, 3)
            rays_with_steps = self.obs_rays_to_rays_with_steps(obs=obs_flat, rays=rays_flat, wcs=WCS(header))  # (B, S, 3)

            rendered_pixels = self.ray_renderer(rays_with_steps)

            # Unflatten rendered_pixels to image shape
            rendered_image = rendered_pixels.reshape(sample_image.shape)

            self._log_step_images(
                predicted=rendered_image.unsqueeze(0),
                target=torch.tensor(sample_image).unsqueeze(0),
                stage='train'
            )
        
        
        
        
        # Store for epoch-level statistics
        self.val_losses.append(loss.detach())
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        # Forward pass
        predicted = self.forward(batch)
        target = batch['pixels']

        # Ensure same shape
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape} in test_step")
        
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
        print(f"Using weight decay: {self.hparams.weight_decay}")

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # optimizer = SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        #     momentum=0.05,
        #     dampening=0.5
        # )
        
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

    def _log_step_images(self, predicted: torch.Tensor, target: torch.Tensor, stage: str):
        """Log per-step images (target, prediction, diff) to TensorBoard only using a Matplotlib figure.
        Args:
            predicted: (B,H,W) or (B,1,H,W)
            target: same shape as predicted
            batch: original batch (unused, reserved for metadata)
            stage: 'train' | 'val' | 'test'
        NOTE: High frequency logging can make event files large; adjust frequency if needed.
        """
        if self.logger is None or not hasattr(self.logger, 'experiment'):
            return
        with torch.no_grad():
            pred = predicted[0].detach().float().cpu()
            tgt = target[0].detach().float().cpu()
            if pred.dim() == 3:  # (C,H,W) keep first channel
                pred = pred[0]
            if tgt.dim() == 3:
                tgt = tgt[0]
            diff = (tgt - pred).abs()
            # Build figure
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            im0 = axes[0].imshow(tgt.numpy(), origin='lower', cmap='inferno')
            axes[0].set_title('Target')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(pred.numpy(), origin='lower', cmap='inferno')
            axes[1].set_title('Predicted')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            im2 = axes[2].imshow(diff.numpy(), origin='lower', cmap='viridis')
            axes[2].set_title('Abs Diff')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            fig.suptitle(f"{stage.capitalize()} Step {self.global_step}")
            self.logger.experiment.add_figure(f'step/{stage}_comparison', fig, global_step=self.global_step)
            plt.close(fig)

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_losses:
            avg_train_loss = torch.stack(self.train_losses).mean()
            self.log('train_loss_epoch', avg_train_loss)
            self.train_losses.clear()
        # Access the dataset via self.trainer.datamodule
        
        
        
        # Optionally log or save the rendered image

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_losses:
            avg_val_loss = torch.stack(self.val_losses).mean()
            self.log('val_loss_epoch', avg_val_loss)
            self.val_losses.clear()
    
    def on_fit_start(self):
        """Log the effective batch size once training begins."""
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            per_device_bs = getattr(dm, 'batch_size', None)
            if per_device_bs is not None:
                num_devices = max(1, getattr(self.trainer, 'num_devices', 1))
                eff_bs = per_device_bs * self.hparams.accumulate_grad_batches * num_devices
                self.log('effective_batch_size', eff_bs, prog_bar=True)
                print(f"Effective batch size (per step after accumulation across {num_devices} device(s)): {eff_bs}")
        except Exception as e:
            print(f"Could not compute effective batch size: {e}")

    def obs_rays_to_rays_with_steps(self, obs: torch.Tensor, rays: torch.Tensor, wcs: WCS) -> torch.Tensor:
        """
        Convert obs and rays to rays_with_steps for ray integration.
        
        Args:
            obs: Observer positions (B, 3)
            rays: Ray directions (B, 3)
        """
        obs = obs.to(dtype=torch.float32, device=self.device)
        rays = rays.to(dtype=torch.float32, device=self.device)

        # Batch size
        B = obs.shape[0]

        # Create steps (S,)
        dsun = wcs.wcs.aux.dsun_obs
        rsun = wcs.wcs.aux.rsun_ref
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

        return rays_with_steps  # (B, S, 3)
    

    
