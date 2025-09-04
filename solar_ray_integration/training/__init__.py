"""
Training module for Solar Ray Integration models.

This module provides lazy imports to avoid circular dependencies.
Import the specific components only when needed.
"""

# Lazy imports to avoid circular dependencies
def _lazy_import():
    from .dataset import SolarPerspectiveDataset, SolarPerspectiveDataModule
    from .lightning_module import SolarNerfLightningModule
    from .train import train_model
    from .config import TrainingPipelineConfig, get_config, create_config_template
    from .utils import test_dataset, visualize_samples, run_full_test
    
    return {
        'SolarPerspectiveDataset': SolarPerspectiveDataset,
        'SolarPerspectiveDataModule': SolarPerspectiveDataModule,
        'SolarNerfLightningModule': SolarNerfLightningModule,
        'train_model': train_model,
        'TrainingPipelineConfig': TrainingPipelineConfig,
        'get_config': get_config,
        'create_config_template': create_config_template,
        'test_dataset': test_dataset,
        'visualize_samples': visualize_samples,
        'run_full_test': run_full_test
    }

def __getattr__(name):
    """Lazy import mechanism."""
    _components = _lazy_import()
    if name in _components:
        return _components[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'SolarPerspectiveDataset', 
    'SolarPerspectiveDataModule', 
    'SolarNerfLightningModule',
    'train_model',
    'TrainingPipelineConfig',
    'get_config',
    'create_config_template',
    'test_dataset',
    'visualize_samples',
    'run_full_test'
]
