"""
Configuration file for Solar NeRF training.

This file contains default configurations and utility functions
for setting up training parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import yaml


@dataclass
class NeRFConfig:
    """Configuration for NeRF model."""
    d_input: int = 3
    d_output: int = 1
    n_layers: int = 8
    d_filter: int = 512
    encoding: str = 'positional'
    skip_connections: Tuple[int, ...] = ()


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "perspective_data/perspective_data_linear_fits"
    batch_size: int = 4
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    normalize: bool = True
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for training."""
    max_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    scheduler_type: str = "cosine"  # "cosine", "plateau", "none"
    loss_type: str = "mse"  # "mse", "l1", "huber"
    integration_method: str = "linear"  # "linear", "volumetric", "volumetric_correction"
    dx: float = 1e7
    log_images_every_n_epochs: int = 10


@dataclass
class HardwareConfig:
    """Configuration for hardware."""
    gpus: int = 1
    precision: str = "16-mixed"  # "32", "16-mixed", "bf16-mixed"
    device_type: str = "cuda:0"


@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    output_dir: str = "outputs"
    experiment_name: str = "solar_nerf"
    resume_from_checkpoint: Optional[str] = None
    save_top_k: int = 3
    early_stopping_patience: int = 20


@dataclass
class TrainingPipelineConfig:
    """Complete configuration for training pipeline."""
    nerf: NeRFConfig = field(default_factory=NeRFConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nerf': self.nerf.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'hardware': self.hardware.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingPipelineConfig':
        """Create from dictionary."""
        return cls(
            nerf=NeRFConfig(**config_dict.get('nerf', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def save(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingPipelineConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


# Default configurations for different scenarios
DEFAULT_CONFIGS = {
    'quick_test': TrainingPipelineConfig(
        data=DataConfig(
            batch_size=2,
            num_workers=2
        ),
        training=TrainingConfig(
            max_epochs=10,
            log_images_every_n_epochs=2
        ),
        experiment=ExperimentConfig(
            experiment_name="quick_test"
        )
    ),
    
    'full_training': TrainingPipelineConfig(
        data=DataConfig(
            batch_size=4,
            num_workers=4
        ),
        training=TrainingConfig(
            max_epochs=200,
            learning_rate=1e-4,
            log_images_every_n_epochs=10
        ),
        experiment=ExperimentConfig(
            experiment_name="full_training"
        )
    ),
    
    'high_res': TrainingPipelineConfig(
        data=DataConfig(
            batch_size=2,
            num_workers=4
        ),
        training=TrainingConfig(
            max_epochs=300,
            learning_rate=5e-5,
            log_images_every_n_epochs=5
        ),
        experiment=ExperimentConfig(
            experiment_name="high_res"
        )
    )
}


def get_config(name: str = 'default') -> TrainingPipelineConfig:
    """Get a configuration by name."""
    if name == 'default':
        return TrainingPipelineConfig()
    elif name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[name]
    else:
        raise ValueError(f"Unknown config name: {name}. Available: {list(DEFAULT_CONFIGS.keys())}")


def create_config_template(filepath: str = "training_config.yaml"):
    """Create a template configuration file."""
    config = TrainingPipelineConfig()
    config.save(filepath)
    print(f"Configuration template saved to: {filepath}")
    return config
