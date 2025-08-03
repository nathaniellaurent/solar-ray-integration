"""
Solar Ray Integration Package

A Python package for integrating scalar fields along rays through solar volumes,
with support for neural radiance fields and traditional field functions.
"""

from .model import NeRF, EmissionModel, Sine, TrainablePositionalEncoding, PositionalEncoding
from .ray_integration import RayIntegrator, integrate_field_linear, integrate_field_volumetric, integrate_field_volumetric_correction
from .rendering import NeuralSolarRenderer


__all__ = [
    # Model components
    'NeRF', 'EmissionModel', 'Sine', 'TrainablePositionalEncoding', 'PositionalEncoding',
    # Ray integration
    'RayIntegrator', 'integrate_field_linear', 'integrate_field_volumetric', 'integrate_field_volumetric_correction',
    # Rendering
    'NeuralSolarRenderer'
]
