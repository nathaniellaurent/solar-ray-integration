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

from ..rendering.neural_renderer import NeuralSolarRenderer
from ..model.model import NeRF
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS
from ..ray_integration import RayIntegrator

# Solar radius constant (meters)


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
        device: str = "cuda:0",
        log_images_every_n_epochs: int = 1,
        checkpoint_dir: str = "checkpoints/solar_nerf",
        checkpoint_monitor: str = "val_loss",
        save_top_k: int = 3,
        image_save_dir: str = "outputs/images",
        accumulate_grad_batches: int = 1,
        compare_ground_truth: bool = False,
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
        self.neural_renderer = NeuralSolarRenderer(
            nerf_config=nerf_config,
            dx=dx,
            device=device,
            integration_method=integration_method,
            return_per_step= compare_ground_truth

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
        if not self.hparams.compare_ground_truth:
            target = batch['image']
        else:
            target = batch['ground_truth']

        # Ensure same shape
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")
        
        # Compute loss
        loss = self.loss_fn(predicted, target)

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")

        if self.hparams.compare_ground_truth:
            predicted = self.neural_renderer.return_collapsed_output(predicted[0])
            predicted = predicted.unsqueeze(0)
            target = batch['image']
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")
        
        # Per-step TB image logging (no disk I/O)
        self._log_step_images(predicted, target, batch, stage='train')
        
        # Store for epoch-level statistics
        self.train_losses.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        predicted = self.forward(batch)
        if not self.hparams.compare_ground_truth:
            target = batch['image']
        else:
            target = batch['ground_truth']

        # Ensure same shape
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape} in validation_step")
        
        # Compute loss
        loss = self.loss_fn(predicted, target)
        
        # Additional metrics
        mae = F.l1_loss(predicted, target)
        mse = F.mse_loss(predicted, target)

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")


        if self.hparams.compare_ground_truth:
            predicted = self.neural_renderer.return_collapsed_output(predicted[0])
            predicted = predicted.unsqueeze(0)
            target = batch['image']

        print(f"Predicted shape: {predicted.shape}, Target shape: {target.shape}")

        
        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True)
        self.log('val_mse', mse, on_epoch=True)
        
        # Per-step TB image logging
        self._log_step_images(predicted, target, batch, stage='val')
        
        # Log sample images periodically
        if (batch_idx == 0 and 
            self.current_epoch % self.hparams.log_images_every_n_epochs == 0):
            self._log_sample_images(predicted, target, batch)
        
        # Store for epoch-level statistics
        self.val_losses.append(loss.detach())
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        # Forward pass
        predicted = self.forward(batch)
        if not self.hparams.compare_ground_truth:
            target = batch['image']
        else:
            target = batch['ground_truth']


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
        if self.hparams.compare_ground_truth:
            predicted = self.neural_renderer.return_collapsed_output(predicted[0])
            predicted = predicted.unsqueeze(0)
            target = batch['image']
    
        # Save / log sample images for first batch of test
        if batch_idx == 0:
            self._log_sample_images(predicted, target, batch)
        
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
    
    def _log_sample_images(self, predicted: torch.Tensor, target: torch.Tensor, batch: Dict[str, torch.Tensor]):
        """Log sample images to tensorboard and save to disk."""
        # Take first image from batch
        pred_img = predicted[0].detach().cpu().float().numpy()
        target_img = target[0].detach().cpu().float().numpy()
        
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
        
        # Save figure & raw arrays to disk
        save_base = os.path.join(self.hparams.image_save_dir, f"epoch{self.current_epoch:04d}_hgln{hgln:+05.1f}_hglt{hglt:+05.1f}")
        fig_path = save_base + "_comparison.png"
        pred_path = save_base + "_pred.npy"
        target_path = save_base + "_target.npy"
        diff_path = save_base + "_diff.npy"
        os.makedirs(self.hparams.image_save_dir, exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        # Save arrays (float32)
        np.save(pred_path, pred_img.astype(np.float32))
        np.save(target_path, target_img.astype(np.float32))
        np.save(diff_path, diff.astype(np.float32))
        
        # Log to tensorboard (only if logger is available)
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_figure(
                'sample_images',
                fig,
                global_step=self.current_epoch
            )
        
        plt.close(fig)
    
    def _log_step_images(self, predicted: torch.Tensor, target: torch.Tensor, batch: Dict[str, torch.Tensor], stage: str):
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

    

    
