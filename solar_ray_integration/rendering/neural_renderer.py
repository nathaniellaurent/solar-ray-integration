"""
Neural rendering module for solar ray field integration.

This module provides a PyTorch nn.Module that combines a NeRF model with 
ray integration to render solar field data.
"""

import torch
import torch.nn as nn
from typing import Any, Optional

from ..model.model import NeRF
from ..ray_integration.integrate_field import (
            integrate_field_linear, 
            integrate_field_volumetric, 
            integrate_field_volumetric_correction
        )


class NeuralSolarRenderer(nn.Module):
    """
    Neural renderer that combines a NeRF model with ray integration for solar field rendering.
    
    This module uses a Neural Radiance Field (NeRF) as the scalar field function
    and integrates it along rays through the solar volume to produce rendered images.
    """
    
    def __init__(
        self,
        nerf_config: dict = None,
        dx: float = 1e7,
        device: str = "cuda:0",
        integration_method: str = "linear"
    ):
        """
        Initialize the neural solar renderer.
        
        Args:
            nerf_config: Configuration dictionary for the NeRF model.
            dx: Step size along the ray in meters.
            device: Device for computation ('cuda:0' for GPU, 'cpu' for CPU).
            integration_method: Integration method ('linear', 'volumetric', 'volumetric_correction').
        """
        super().__init__()
        
        # Default NeRF configuration
        if nerf_config is None:
            nerf_config = {
                'd_input': 3,
                'd_output': 1,
                'n_layers': 8,
                'd_filter': 512,
                'encoding': 'positional'
            }
        
        self.nerf = NeRF(**nerf_config)
        self.dx = dx
        self.device = device
        self.integration_method = integration_method
        
    def forward(self, source_hdu: Any, requires_grad: bool = True) -> torch.Tensor:
        """
        Forward pass: render the solar field using the NeRF model.
        
        Args:
            source_hdu: FITS HDU containing image data and header for ray calculation.
            requires_grad: If True, enables autograd for PyTorch tensors.
            
        Returns:
            Rendered image as a 2D tensor.
        """
        
        
        # Integrate based on method
        if self.integration_method == "linear":
            output_tensor = integrate_field_linear(
                field=self.neural_field,
                source_hdu=source_hdu,
                dx=self.dx,
                requires_grad=requires_grad,
                device=self.device
            )
            
        elif self.integration_method == "volumetric":
            output_tensor = integrate_field_volumetric(
                field=self.neural_field,
                source_hdu=source_hdu,
                dx=self.dx,
                requires_grad=requires_grad,
                device=self.device
            )
            
        elif self.integration_method == "volumetric_correction":
            output_tensor = integrate_field_volumetric_correction(
                field=self.neural_field,
                source_hdu=source_hdu,
                dx=self.dx,
                requires_grad=requires_grad,
                device=self.device
            )
            
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        
        return output_tensor
    
    def neural_field(self, coords: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Neural field function that evaluates the NeRF at given coordinates.
        
        Args:
            coords: Tensor of shape (..., 3) representing (x, y, z) coordinates.
            **kwargs: Additional arguments (unused, for compatibility).
            
        Returns:
            Field values at the given coordinates.
        """
        # Reshape coordinates for NeRF input
        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 3)
        
        # Normalize coordinates (optional - you may want to adjust this based on your data)
        # coords_normalized = coords_flat / 7e8  # Normalize by solar radius
        
        # Evaluate NeRF
        field_values = self.nerf(coords_flat)
        
        # Reshape back to original spatial dimensions
        field_values = field_values.reshape(*original_shape, -1)
        
        # If NeRF outputs multiple channels, you might want to select one or combine them
        if field_values.shape[-1] > 1:
            field_values = field_values[..., 0]  # Take first channel
        else:
            field_values = field_values.squeeze(-1)
        
        # Apply activation to ensure positive values (optional)
        field_values = torch.relu(field_values)
        
        return field_values
    
    def set_integration_method(self, method: str):
        """Set the integration method."""
        valid_methods = ["linear", "volumetric", "volumetric_correction"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.integration_method = method
    
    def get_nerf_parameters(self):
        """Get the NeRF model parameters for optimization."""
        return self.nerf.parameters()
