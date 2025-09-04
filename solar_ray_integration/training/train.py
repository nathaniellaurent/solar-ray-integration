import os
import sys
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger  # Re-enabled
import torch

from .dataset import SolarPerspectiveDataModule
from .lightning_module import SolarNerfLightningModule
from .ray_samples_module import RayWiseLightningModule
from .ray_samples_dataset import SolarRayPixelDataModule



def train_model(
    data_dir: str = "perspective_data/perspective_data_linear_fits_small",
    output_dir: str = "outputs",
    experiment_name: str = "solar_nerf",
    batch_size: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-6,
    integration_method: str = "linear",
    num_workers: int = 4,
    gpus: int = 1,
    precision: str = "16-mixed",
    seed: int = 42,
    resume_from_checkpoint: str = None,
    accumulate_grad_batches: int = 1,
    compare_ground_truth: bool = False,
    single_ray: bool = False,
    **kwargs
):
    """
    Train a Solar NeRF model.
    
    Args:
        data_dir: Directory containing FITS files
        output_dir: Directory to save outputs
        experiment_name: Name for the experiment
        batch_size: Training batch size
        max_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        integration_method: Ray integration method
        num_workers: Number of data loading workers
        gpus: Number of GPUs to use
        precision: Training precision
        seed: Random seed
        resume_from_checkpoint: Path to checkpoint to resume from
        accumulate_grad_batches: Number of batches to accumulate before optimizer step
    """
    # Set seed for reproducibility
    pl.seed_everything(seed)
    device = "cpu" if gpus < 1 else "cuda:0"

    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure log directory exists
    log_root = output_path / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    if not single_ray:
        data_module = SolarPerspectiveDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            device=device,
            compare_ground_truth=compare_ground_truth
        )
    else:
        data_module = SolarRayPixelDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            device=device,
            normalize=True,
            normalization_value=139.0
        )

    
    # Smaller NeRF configuration for reduced memory usage
    nerf_config = {
        'd_input': 3,
        'd_output': 1,
        'n_layers': 4,  # Reduced from 8 to 4
        'd_filter': 64,  # Reduced from 512 to 256
        'encoding': 'positional' #'positional'
    }
    
    # Initialize model
    print(f"Using device: {device}")
    if not single_ray:
        model = SolarNerfLightningModule(
            nerf_config=nerf_config,
            integration_method=integration_method,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=None,
            loss_type="mse",
            log_images_every_n_epochs=1,
            device=device,
            accumulate_grad_batches=accumulate_grad_batches,
            compare_ground_truth=compare_ground_truth
        )
    else:
        model = RayWiseLightningModule(
            nerf_config=nerf_config,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=None,
            loss_type="mse",
            device=device,
            accumulate_grad_batches=accumulate_grad_batches,
        )
    
    # Callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            filename="solar_nerf-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=20,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # TensorBoard logger (fixed). If protobuf issues arise, pin protobuf < 5.0: pip install 'protobuf<5'
    logger = TensorBoardLogger(
        save_dir=str(log_root),  # root logs directory
        name=experiment_name,    # experiment subfolder
        version=None,            # auto-increment version
        default_hp_metric=False  # disable automatic hp metric
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else "auto",
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=accumulate_grad_batches
    )
    
    # Print model summary
    print(f"Training Solar NeRF model:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Logs directory: {log_root}")
    print(f"  Experiment name: {experiment_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Integration method: {integration_method}")
    print(f"  GPUs: {gpus}")
    print(f"  Workers: {num_workers}")
    print(f"  Precision: {precision}")
    print(f"  Accumulate grad batches: {accumulate_grad_batches}")
    print(f"  Model layers: {nerf_config['n_layers']}")
    print(f"  Model width: {nerf_config['d_filter']}")
    
    # Train model
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint
    )
    
    # Test model
    print("Testing model...")
    trainer.test(model, datamodule=data_module)
    
    # Save final model
    final_model_path = output_path / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return trainer, model


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Train Solar NeRF model")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, 
                       default="perspective_data/perspective_data_linear_fits_small",
                       help="Directory containing FITS files")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--experiment-name", type=str, default="solar_nerf",
                       help="Experiment name")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--integration-method", type=str, default="linear",
                       choices=["linear", "volumetric", "volumetric_correction"],
                       help="Ray integration method")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1,
                       help="Number of batches to accumulate before optimizer step")
    
    # Data arguments
    parser.add_argument("--num-workers", type=int, default=4, help="Data loading workers")
    
    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="32",
                       choices=["32", "16-mixed", "bf16-mixed"],
                       help="Training precision")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Checkpoint path to resume from")
    
    parser.add_argument("--compare-ground-truth", action='store_true',
                       help="Compare predictions to ground truth instead of collapsed output")
    parser.add_argument("--single-ray", action='store_true',
                       help="Use ray-wise training (single ray per pixel)")
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        integration_method=args.integration_method,
        num_workers=args.num_workers,
        gpus=args.gpus,
        precision=args.precision,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        accumulate_grad_batches=args.accumulate_grad_batches,
        compare_ground_truth=args.compare_ground_truth,
        single_ray=args.single_ray
    )


if __name__ == "__main__":
    main()
