"""
Neural rendering module for solar ray field integration.

This module provides a PyTorch nn.Module that combines a NeRF model with 
ray integration to render solar field data.
"""

import torch
import torch.nn as nn
from typing import Any, Optional
import matplotlib.pyplot as plt  # Added for visualization

from ..model.model import NeRF
from ..ray_integration.integrate_field import (
    RayIntegrator,
    collapse_output
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
        integration_method: str = "linear",
        return_per_step: bool = False
    ):
        """
        Initialize the neural solar renderer.
        
        Args:
            nerf_config: Configuration dictionary for the NeRF model.
            dx: Step size along the ray in meters.
            device: Device for computation ('cuda:0' for GPU, 'cpu' for CPU).
            integration_method: Integration method ('linear', 'volumetric', 'volumetric_correction').
            return_per_step: If True, returns the output at each integration step.
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
        
        self.nerf = NeRF(device=device, **nerf_config)
        self.dx = dx
        self.device = device
        self.integration_method = integration_method
        self.return_per_step = return_per_step
        
    def forward(self, source_hdu: Any, requires_grad: bool = True) -> torch.Tensor:
        """
        Forward pass: render the solar field using the NeRF model.
        
        Args:
            source_hdu: FITS HDU containing image data and header for ray calculation.
            requires_grad: If True, enables autograd for PyTorch tensors.
            
        Returns:
            Rendered image as a 2D tensor.
        """

        print("source_hdu data shape:", source_hdu.data.shape)
        if hasattr(source_hdu, 'header') and 'NAXIS1' in source_hdu.header and 'NAXIS2' in source_hdu.header:
            print("WCS shape from header: (NAXIS2, NAXIS1) =", (source_hdu.header['NAXIS2'], source_hdu.header['NAXIS1']))
        else:
            print("WCS shape info not found in header.")
        
        
        # Integrate based on method using RayIntegrator architecture
        integrator = RayIntegrator(
            field=self.neural_field,
            source_hdu=source_hdu,
            dx=self.dx,
            requires_grad=requires_grad,
            device=self.device
        )
        if self.integration_method == "linear":

            per_step = integrator.calculate_field_linear()

        elif self.integration_method == "volumetric":

            per_step = integrator.calculate_field_volumetric()

        elif self.integration_method == "volumetric_trapezoidal":

            per_step = integrator.calculate_field_volumetric_trapezoidal()

        elif self.integration_method == "volumetric_correction":

            per_step = integrator.calculate_field_volumetric_correction()

        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        

        if self.return_per_step:
            return per_step  # shape (H,W,S) or (H,W,S-1)
        # Collapse steps to final 2D image
        output_tensor = collapse_output(per_step)
        
        # Visualize output tensor with matplotlib (non-blocking)
        # try:
        #     img = output_tensor.detach().cpu().float().numpy()
        #     plt.figure(figsize=(4,4))
        #     plt.imshow(img, origin='lower', cmap='inferno')
        #     plt.colorbar(fraction=0.046, pad=0.04)
        #     plt.title('NeuralSolarRenderer Output')
        #     plt.tight_layout()
        #     plt.show(block=True)
        #     plt.pause(0.001)
        #     plt.close()
        # except Exception as e:
        #     print(f"Visualization failed: {e}")
        
        return output_tensor

    def return_collapsed_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the last collapsed output if available."""
        return collapse_output(input_tensor)

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
        # if field_values.shape[-1] > 1:
        #     field_values = field_values[..., 0]  # Take first channel
        # else:
        field_values = field_values.squeeze(-1)
        
        # Apply activation to ensure positive values (optional)
        # field_values = torch.relu(field_values)
        
        return field_values
    
    def set_integration_method(self, method: str):
        """Set the integration method."""
        valid_methods = [
            "linear", "volumetric", "volumetric_trapezoidal", "volumetric_correction"
        ]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.integration_method = method
    
    def get_nerf_parameters(self):
        """Get the NeRF model parameters for optimization."""
        return self.nerf.parameters()
