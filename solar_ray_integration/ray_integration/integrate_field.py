"""
Integration routines for solar ray field calculations.

This module provides functions to integrate a scalar field along rays through a solar volume,
using PyTorch tensors and Astropy WCS for coordinate transformations. All calculations are performed
on the GPU by default.

Functions:
    - integrate_field_linear: Integrate a field along rays without area correction.
    - integrate_field_volumetric: Integrate a field along rays with pixel area correction.
    - integrate_field_volumetric_correction: Integrate a field along rays with pixel area and 1/r^2 correction.

Dependencies:
    - torch
    - numpy
    - astropy
    - einops
    - dfreproject (for ray calculation)
"""

from astropy.wcs import WCS
from dfreproject import calculate_rays
import matplotlib.pyplot as plt
import torch
from dfreproject import TensorHDU
from einops import rearrange, reduce
import numpy as np
from typing import Callable, Any


class RayIntegrator:

    def __init__(
        self,
        field: Callable[[torch.Tensor, Any], torch.Tensor],
        source_hdu: Any,
        dx: float = 1e7,
        requires_grad: bool = False,
        device: str = "cuda:0"
    ):
        self.field = field
        self.source_hdu = source_hdu
        self.dx = dx
        self.requires_grad = requires_grad
        self.device = device
        self.shape = source_hdu.data.shape

        self.source_wcs = WCS(source_hdu.header)

        if hasattr(self.source_wcs.wcs.aux, "dsun_obs") and self.source_wcs.wcs.aux.dsun_obs is not None:
            self.dsun = torch.tensor(self.source_wcs.wcs.aux.dsun_obs, dtype=torch.float32, requires_grad=self.requires_grad, device=self.device)
        else:
            self.dsun = torch.tensor(1.496e11, dtype=torch.float32, requires_grad=self.requires_grad, device=self.device)

        if hasattr(self.source_wcs.wcs.aux, "rsun_ref") and self.source_wcs.wcs.aux.rsun_ref is not None:
            self.rsun = torch.tensor(self.source_wcs.wcs.aux.rsun_ref, dtype=torch.float32, requires_grad=self.requires_grad, device=self.device)
        else:
            self.rsun = torch.tensor(6.957e8, dtype=torch.float32, requires_grad=self.requires_grad, device=self.device)

        self.steps = self.create_steps()

    def create_steps(self):
        dsun = self.dsun
        rsun = self.rsun
        dx = self.dx

        steps = torch.arange((dsun - 2*rsun).item(), (dsun + 2*rsun).item(), dx, device=self.device, dtype=torch.float32, requires_grad=self.requires_grad)
        steps = rearrange(steps, "s -> 1 1 1 s")

        return steps
    
        

    def generate_ray_tensor(self):
        

        source_wcs = self.source_wcs


        obs, rays = calculate_rays(
            source_wcs=source_wcs,
            shape_out=self.shape,
            requires_grad=self.requires_grad,
        )

       



        rays = rays.to(dtype=torch.float32, device=self.device)
        obs = obs.to(dtype=torch.float32, device=self.device)

        dsun = torch.norm(obs)  # Distance from observer to Sun in meters
        dsun = dsun.to(dtype=torch.int64, device=self.device)

        rays_no_batch = rearrange(rays, "1 h w c -> h w c 1")  # shape (H, W, 3)

        steps = self.steps

        # Estimate memory usage of rays_with_steps
        rays_no_batch_shape = rays_no_batch.shape  # (H, W, 3, 1)
        steps_shape = steps.shape  # (1, 1, 1, S)

        # Resulting shape of rays_with_steps
        rays_with_steps_shape = (
            rays_no_batch_shape[0],  # H
            rays_no_batch_shape[1],  # W
            steps_shape[3],          # S
            rays_no_batch_shape[2],  # C (3)
        )

        # Calculate memory usage
        element_size = rays_no_batch.element_size()  # Size of each element in bytes
        num_elements = torch.prod(torch.tensor(rays_with_steps_shape))  # Total number of elements
        memory_bytes = num_elements.item() * element_size  # Total memory in bytes
        memory_megabytes = memory_bytes / (1024 ** 2)  # Convert to MB

        print(f"rays_with_steps will use approximately {memory_megabytes:.2f} MB of memory.")

        rays_with_steps = rays_no_batch * steps

        # Debugging: Check the shape and memory usage of rays_with_steps
        print(f"Shape of rays_with_steps: {rays_with_steps.shape}")
        print(f"Memory usage of rays_with_steps: {memory_megabytes:.2f} MB")

        del rays_no_batch, steps
        obs = rearrange(obs, "c -> 1 1 c 1")

        rays_with_obs = rays_with_steps + obs

        del rays_with_steps

        torch.cuda.empty_cache()

        rays_with_obs = rearrange(rays_with_obs, "h w c s -> h w s c")
        print("Debug: Before applying field function")
        print(f"Shape of rays_with_obs: {rays_with_obs.shape}")
        
        output_tensor = self.field(rays_with_obs, radius=6.9634e8, value=1.0)
        # Debugging: Check the shape and type of output_tensor after applying the field function
        print("Debug: After applying field function")
        print(f"Shape of output_tensor: {output_tensor.shape}")
        print(f"Type of output_tensor: {type(output_tensor)}")

        # Debugging: Check memory usage of output_tensor
        tensor_bytes = output_tensor.element_size() * output_tensor.nelement()
        tensor_megabytes = tensor_bytes / (1024 ** 2)
        print(f"output_tensor uses {tensor_megabytes:.2f} MB of memory")

        del rays_with_obs

      

        return output_tensor

    def generate_area_tensor(self):

        source_hdu = TensorHDU(torch.tensor(self.source_hdu.data, requires_grad=True, device=self.device), self.source_hdu.header)
        source_wcs = WCS(source_hdu.header)

        if hasattr(source_wcs.wcs.aux, "rsun_ref") and source_wcs.wcs.aux.rsun_ref is not None:
            rsun = torch.tensor(source_wcs.wcs.aux.rsun_ref, dtype=torch.float32, requires_grad=self.requires_grad, device=self.device)
        else:
            rsun = torch.tensor(6.957e8, dtype=torch.float32, requires_grad=self.requires_grad, device=self.device)

        obs, rays_corners = calculate_rays(
            source_wcs=source_wcs,
            shape_out=self.shape,
            corners=True,
            requires_grad=False
        )

        rays_corners = rays_corners.to(dtype=torch.float32, device=self.device)
        obs = obs.to(dtype=torch.float32, device=self.device)

        dsun = torch.norm(obs)  # Distance from observer to Sun in meters
        dsun = dsun.to(dtype=torch.int64, device=self.device)

        rays_corners_no_batch = rearrange(rays_corners, "1 h w c -> h w c 1")  # shape (H, W, 3)
        del rays_corners

        steps = torch.arange(dsun - 2*rsun, dsun + 2*rsun, self.dx, device=self.device, dtype=torch.float32)
        steps = rearrange(steps, "s -> 1 1 1 s")

        H, W = rays_corners_no_batch.shape[0], rays_corners_no_batch.shape[1]
        h1, w1 = H // 2, W // 2
        w2 = H // 2 + 1

        ref_pix1 = rays_corners_no_batch[h1, w1, :]  # Reference pixel at (h1, w1)
        ref_pix2 = rays_corners_no_batch[h1, w2, :]  # Reference pixel at (h1, w2)

        del rays_corners_no_batch

        ref_pix1 = rearrange(ref_pix1, "c 1 -> 1 1 c 1")
        ref_pix2 = rearrange(ref_pix2, "c 1 -> 1 1 c 1")

        ref_pix1_with_steps = ref_pix1 * steps
        ref_pix2_with_steps = ref_pix2 * steps
        del ref_pix1, ref_pix2, steps

        obs = rearrange(obs, "c -> 1 1 c 1")

        ref_pix1_with_obs = ref_pix1_with_steps + obs
        ref_pix2_with_obs = ref_pix2_with_steps + obs

        del obs, ref_pix1_with_steps, ref_pix2_with_steps

        diff = ref_pix2_with_obs - ref_pix1_with_obs
        norm = torch.norm(diff, dim=2)
        area = norm ** 2

        return area

    def generate_correction(self):
        return 1 / (self.steps ** 2)


def integrate_field_linear(
    field: Callable[[torch.Tensor, Any], torch.Tensor],
    source_hdu: Any,
    dx: float = 1e7,
    requires_grad: bool = False,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    Integrate a scalar field along rays from the observer through the solar volume.

    Args:
        field: Function that evaluates the scalar field at given coordinates.
        source_hdu: FITS HDU containing image data and header.
        dx: Step size along the ray in meters.
        requires_grad: If True, enables autograd for PyTorch tensors.
        device: Device for computation ('cuda:0' for GPU, 'cpu' for CPU).

    Returns:
        Integrated field image (2D tensor).
    """

    print(f"Integrating field with dx={dx}, requires_grad={requires_grad}, device={device}")


    integration = RayIntegrator(
        field=field,
        source_hdu=source_hdu,
        dx=dx,
        requires_grad=requires_grad,
        device=device
    )

    output_tensor = integration.generate_ray_tensor()
        
    
    
    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')

    torch.cuda.empty_cache()

    return output_tensor


def integrate_field_volumetric(
    field: Callable[[torch.Tensor, Any], torch.Tensor],
    source_hdu: Any,
    dx: float = 1e7,
    requires_grad: bool = False,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    Integrate a scalar field along rays, including pixel area correction.

    Args:
        field: Function that evaluates the scalar field at given coordinates.
        source_hdu: FITS HDU containing image data and header.
        dx: Step size along the ray in meters.
        requires_grad: If True, enables autograd for PyTorch tensors.
        device: Device for computation ('cuda:0' for GPU, 'cpu' for CPU).

    Returns:
        Integrated field image (2D tensor) with pixel area correction.
    """

    integration = RayIntegrator(
        field=field,
        source_hdu=source_hdu,
        dx=dx,
        requires_grad=requires_grad,
        device=device
    )

    output_tensor = integration.generate_ray_tensor()
    
    area = integration.generate_area_tensor()

    output_tensor = output_tensor * area
    
    del area


    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')
    torch.cuda.empty_cache()

    return output_tensor

def integrate_field_volumetric_trapezoidal(
    field: Callable[[torch.Tensor, Any], torch.Tensor],
    source_hdu: Any,
    dx: float = 1e7,
    requires_grad: bool = False,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    Integrate a scalar field along rays, including pixel area correction.

    Args:
        field: Function that evaluates the scalar field at given coordinates.
        source_hdu: FITS HDU containing image data and header.
        dx: Step size along the ray in meters.
        requires_grad: If True, enables autograd for PyTorch tensors.
        device: Device for computation ('cuda:0' for GPU, 'cpu' for CPU).

    Returns:
        Integrated field image (2D tensor) with pixel area correction.
    """

    integration = RayIntegrator(
        field=field,
        source_hdu=source_hdu,
        dx=dx,
        requires_grad=requires_grad,
        device=device
    )

    output_tensor = integration.generate_ray_tensor()

    # Pad zeros to the front of the third dimension (s) of output_tensor
    output_tensor_1 = torch.cat(
        [torch.zeros_like(output_tensor[:, :, :1]), output_tensor], dim=2
    )

    output_tensor_2 = torch.cat(
        [output_tensor, torch.zeros_like(output_tensor[:, :, :1])], dim=2
    )

    output_tensor = (output_tensor_1 + output_tensor_2) / 2

    del output_tensor_1, output_tensor_2

    output_tensor = output_tensor[:, :, 1:-1]

    area = integration.generate_area_tensor()
    area = area[:, :, :-1]

    # Check memory usage of output_tensor
    tensor_bytes = output_tensor.element_size() * output_tensor.nelement()
    tensor_megabytes = tensor_bytes / (1024 ** 2)
    print(f"output_tensor uses {tensor_megabytes:.2f} MB of memory")

    output_tensor = output_tensor * area
    
    
    del area


    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')
    torch.cuda.empty_cache()

    return output_tensor

def integrate_field_volumetric_correction(
    field: Callable[[torch.Tensor, Any], torch.Tensor],
    source_hdu: Any,
    dx: float = 1e7,
    requires_grad: bool = False,
    device: str = "cuda:0"
) -> torch.Tensor:
    """
    Integrate a scalar field along rays, including pixel area and 1/r^2 correction.

    Args:
        field: Function that evaluates the scalar field at given coordinates.
        source_hdu: FITS HDU containing image data and header.
        dx: Step size along the ray in meters.
        requires_grad: If True, enables autograd for PyTorch tensors.
        device: Device for computation ('cuda:0' for GPU, 'cpu' for CPU).

    Returns:
        Integrated field image (2D tensor) with pixel area and 1/r^2 correction.
    """

    integration = RayIntegrator(
        field=field,
        source_hdu=source_hdu,
        dx=dx,
        requires_grad=requires_grad,
        device=device
    )

    output_tensor = integration.generate_ray_tensor()
    area = integration.generate_area_tensor()
    output_tensor = output_tensor * area

    correction = integration.generate_correction()

    correction = rearrange(correction, "1 1 1 s -> 1 1 s ")  # Reshape to match output_tensor shape


    output_tensor = output_tensor * correction

    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')



    del area, correction
    
    torch.cuda.empty_cache()

    return output_tensor