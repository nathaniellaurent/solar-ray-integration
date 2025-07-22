from astropy.io import fits
from astropy.wcs import WCS
from dfreproject import calculate_rays
from sunpy.data.sample import AIA_193_JUN2012, STEREO_A_195_JUN2012
import matplotlib.pyplot as plt
import torch
from astropy.io.fits import PrimaryHDU
from dfreproject import TensorHDU
from einops import rearrange, reduce, repeat
import numpy as np

def integrate_field_linear(field, source_hdu, dx = 1e7, requires_grad=False):
    shape = source_hdu.data.shape

    device = torch.device("cuda:0")
    source_hdu = TensorHDU(torch.tensor(source_hdu.data, requires_grad=True, device=device), source_hdu.header)
    source_wcs = WCS(source_hdu.header)


    if hasattr(source_wcs.wcs.aux, "rsun_ref") and source_wcs.wcs.aux.rsun_ref is not None:
        rsun =  torch.tensor(source_wcs.wcs.aux.rsun_ref, dtype=torch.float32, requires_grad=requires_grad, device=device)
    else:
        rsun = torch.tensor(6.957e8, dtype=torch.float32, requires_grad=requires_grad, device=device)

    obs, rays = calculate_rays(
        source_wcs = source_wcs,
        shape_out=shape,
        requires_grad=requires_grad,
    )  

    rays = rays.to(dtype=torch.float32, device=device)
    obs = obs.to(dtype=torch.float32, device=device)

    dsun = torch.norm(obs)  # Distance from observer to Sun in meters
    dsun = dsun.to(dtype=torch.int64, device=device)

    rays_no_batch = rearrange(rays, "1 h w c -> h w c 1")  # shape (H, W, 3)



    steps = torch.arange(dsun - 2*rsun, dsun + 2*rsun, dx, device=device, dtype=torch.float32)
    steps = rearrange(steps, "s -> 1 1 1 s")


    rays_with_steps = rays_no_batch*steps


    del rays_no_batch, steps
    obs = rearrange(obs, "c -> 1 1 c 1")

    rays_with_obs = rays_with_steps + obs


    del rays_with_steps

    torch.cuda.empty_cache()

    rays_with_obs = rearrange(rays_with_obs, "h w c s -> h w s c")
    output_tensor = field(rays_with_obs, radius=6.9634e8, value=1.0)
    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')

    torch.cuda.empty_cache()
    del rays_with_obs

    return output_tensor


def integrate_field_volumetric(field, source_hdu, dx = 1e7, requires_grad=False):
    shape = source_hdu.data.shape

    device = torch.device("cuda:0")
    source_hdu = TensorHDU(torch.tensor(source_hdu.data, requires_grad=True, device=device), source_hdu.header)
    source_wcs = WCS(source_hdu.header)


    if hasattr(source_wcs.wcs.aux, "rsun_ref") and source_wcs.wcs.aux.rsun_ref is not None:
        rsun =  torch.tensor(source_wcs.wcs.aux.rsun_ref, dtype=torch.float32, requires_grad=requires_grad, device=device)
    else:
        rsun = torch.tensor(6.957e8, dtype=torch.float32, requires_grad=requires_grad, device=device)

    obs, rays = calculate_rays(
        source_wcs = source_wcs,
        shape_out=shape,
        corners=True,
        requires_grad=False
    )
    _, rays_centered = calculate_rays(
        source_wcs = source_wcs,
        shape_out=shape,
        corners=False,
        requires_grad=False
    )

    rays = rays.to(dtype=torch.float32, device=device)
    obs = obs.to(dtype=torch.float32, device=device)
    rays_centered = rays_centered.to(dtype=torch.float32, device=device)

    dsun = torch.norm(obs)  # Distance from observer to Sun in meters
    dsun = dsun.to(dtype=torch.int64, device=device)

    rays_centered_no_batch = rearrange(rays_centered, "1 h w c -> h w c 1")  # shape (H, W, 3)
    rays_no_batch = rearrange(rays, "1 h w c -> h w c 1")  # shape (H, W, 3)
    del rays_centered, rays


    steps = torch.arange(dsun - 2*rsun, dsun + 2*rsun, dx, device=device, dtype=torch.float32)
    steps = rearrange(steps, "s -> 1 1 1 s")


    rays_centered_with_steps = rays_centered_no_batch*steps



    del rays_centered_no_batch
    obs = rearrange(obs, "c -> 1 1 c 1")

    rays_centered_with_obs = rays_centered_with_steps + obs

    del rays_centered_with_steps


    torch.cuda.empty_cache()


    H, W = rays_no_batch.shape[0], rays_no_batch.shape[1]
    h1, w1 = H // 2, W // 2
    w2 = H // 2 + 1


    ref_pix1 = rays_no_batch[h1, w1, :]  # Reference pixel at (h1, w1)
    ref_pix2 = rays_no_batch[h1, w2, :]  # Reference pixel at (h1, w2)


    del rays_no_batch

    ref_pix1 = rearrange(ref_pix1, "c 1 -> 1 1 c 1")  # Reshape to match rays_with_obs shape
    ref_pix2 = rearrange(ref_pix2, "c 1 -> 1 1 c 1")


    ref_pix1_with_steps = ref_pix1 * steps
    ref_pix2_with_steps = ref_pix2 * steps
    del ref_pix1, ref_pix2, steps

    ref_pix1_with_obs = ref_pix1_with_steps + obs
    ref_pix2_with_obs = ref_pix2_with_steps + obs
    del obs, ref_pix1_with_steps, ref_pix2_with_steps


    diff = ref_pix2_with_obs - ref_pix1_with_obs
    norm = torch.norm(diff, dim=2)
    area = norm ** 2

    del diff, norm

    rays_centered_with_obs = rearrange(rays_centered_with_obs, "h w c s -> h w s c")



    output_tensor = field(rays_centered_with_obs, radius=6.9634e8, value=1.0)

    output_tensor = output_tensor * area

    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')




    
    del rays_centered_with_obs, area
    torch.cuda.empty_cache()

    return output_tensor

def integrate_field_volumetric_correction(field, source_hdu, dx = 1e7, requires_grad=False):
    shape = source_hdu.data.shape

    device = torch.device("cuda:0")
    source_hdu = TensorHDU(torch.tensor(source_hdu.data, requires_grad=True, device=device), source_hdu.header)
    source_wcs = WCS(source_hdu.header)


    if hasattr(source_wcs.wcs.aux, "rsun_ref") and source_wcs.wcs.aux.rsun_ref is not None:
        rsun =  torch.tensor(source_wcs.wcs.aux.rsun_ref, dtype=torch.float32, requires_grad=requires_grad, device=device)
    else:
        rsun = torch.tensor(6.957e8, dtype=torch.float32, requires_grad=requires_grad, device=device)

    obs, rays = calculate_rays(
        source_wcs = source_wcs,
        shape_out=shape,
        corners=True,
        requires_grad=False
    )
    _, rays_centered = calculate_rays(
        source_wcs = source_wcs,
        shape_out=shape,
        corners=False,
        requires_grad=False
    )

    rays = rays.to(dtype=torch.float32, device=device)
    obs = obs.to(dtype=torch.float32, device=device)
    rays_centered = rays_centered.to(dtype=torch.float32, device=device)

    dsun = torch.norm(obs)  # Distance from observer to Sun in meters
    dsun = dsun.to(dtype=torch.int64, device=device)

    rays_centered_no_batch = rearrange(rays_centered, "1 h w c -> h w c 1")  # shape (H, W, 3)
    rays_no_batch = rearrange(rays, "1 h w c -> h w c 1")  # shape (H, W, 3)
    del rays_centered, rays


    steps = torch.arange(dsun - 2*rsun, dsun + 2*rsun, dx, device=device, dtype=torch.float32)
    steps = rearrange(steps, "s -> 1 1 1 s")

    correction  = 1/(steps**2)


    rays_centered_with_steps = rays_centered_no_batch*steps



    del rays_centered_no_batch
    obs = rearrange(obs, "c -> 1 1 c 1")

    rays_centered_with_obs = rays_centered_with_steps + obs

    del rays_centered_with_steps


    torch.cuda.empty_cache()


    H, W = rays_no_batch.shape[0], rays_no_batch.shape[1]
    h1, w1 = H // 2, W // 2
    w2 = H // 2 + 1


    ref_pix1 = rays_no_batch[h1, w1, :]  # Reference pixel at (h1, w1)
    ref_pix2 = rays_no_batch[h1, w2, :]  # Reference pixel at (h1, w2)


    del rays_no_batch

    ref_pix1 = rearrange(ref_pix1, "c 1 -> 1 1 c 1")  # Reshape to match rays_with_obs shape
    ref_pix2 = rearrange(ref_pix2, "c 1 -> 1 1 c 1")


    ref_pix1_with_steps = ref_pix1 * steps
    ref_pix2_with_steps = ref_pix2 * steps
    del ref_pix1, ref_pix2, steps

    ref_pix1_with_obs = ref_pix1_with_steps + obs
    ref_pix2_with_obs = ref_pix2_with_steps + obs
    del obs, ref_pix1_with_steps, ref_pix2_with_steps


    diff = ref_pix2_with_obs - ref_pix1_with_obs
    norm = torch.norm(diff, dim=2)
    area = norm ** 2

    del diff, norm

    rays_centered_with_obs = rearrange(rays_centered_with_obs, "h w c s -> h w s c")



    output_tensor = field(rays_centered_with_obs, radius=6.9634e8, value=1.0)

    output_tensor = output_tensor * area

    correction = rearrange(correction, "1 1 1 s -> 1 1 s")

    output_tensor = output_tensor * correction

    output_tensor = reduce(output_tensor, 'h w s -> h w', 'sum')




    del rays_centered_with_obs, area, correction
    
    torch.cuda.empty_cache()

    return output_tensor