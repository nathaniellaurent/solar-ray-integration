import argparse
import torch
import numpy as np
from pathlib import Path
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS

from solar_ray_integration.training.dataset import SolarPerspectiveDataModule
from solar_ray_integration.ray_integration.integrate_field import RayIntegrator, collapse_output, calculate_rays

# Analytic field (duplicated so we don't import sunpy-heavy demo module)
RSUN = int(6.9634e8)

def sun_sphere_scalar(coords, radius=RSUN, radius2=RSUN//2, radius3=RSUN//3, value=1.0):
    """Piecewise constant + shapes analytic field used to generate synthetic data.
    coords: (...,3) tensor (meters).
    Returns emission value tensor of shape (...,).
    """
    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords must be torch.Tensor")
    dist2 = (coords[..., 0]**2 + coords[..., 1]**2 + coords[..., 2]**2)
    mask = dist2 <= radius**2

    center_2 = torch.tensor([radius, radius, 0.0], dtype=coords.dtype, device=coords.device)
    dist2_new = (coords[..., 0] - center_2[0])**2 + (coords[..., 1] - center_2[1])**2 + (coords[..., 2] - center_2[2])**2
    mask_new = dist2_new <= radius2**2
    mask_combined = mask | mask_new

    center_3 = torch.tensor([-radius, -radius, radius/2], dtype=coords.dtype, device=coords.device)
    dist2_new = (coords[..., 0] - center_3[0])**2 + (coords[..., 1] - center_3[1])**2 + (coords[..., 2] - center_3[2])**2
    mask_new = dist2_new <= radius3**2

    torus_center = center_3
    torus_major_radius = radius3 * 1.5
    torus_minor_radius = radius3 / 4
    x = coords[..., 0] - torus_center[0]
    y = coords[..., 1] - torus_center[1]
    z = coords[..., 2] - torus_center[2]
    torus_r = torch.sqrt(x**2 + z**2)
    torus_mask = ((torus_r - torus_major_radius)**2 + y**2) <= torus_minor_radius**2

    mask_combined = mask_combined | mask_new | torus_mask
    return (torch.where(mask_combined, torch.tensor(value, dtype=coords.dtype, device=coords.device), torch.zeros((), dtype=coords.dtype, device=coords.device)))/139.0


def evaluate(obs: torch.Tensor , ray: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """
        Forward pass: render the solar field using the NeRF model.
        
        Args:
            ray: Tensor representing the rays to be rendered. Shape (B, 3) for a batch of rays.
            obs: Tensor representing the observer positions. Shape (B, 3) for a batch of observers.
            requires_grad: If True, enables autograd for PyTorch tensors.
            
        Returns:
            Rendered rays as a 1D tensor of shape (B,).
        """
        # Ensure inputs are float32 and on the correct device
        obs = obs.to(dtype=torch.float32)
        ray = ray.to(dtype=torch.float32)
        
        # Batch size
        B = obs.shape[0]
        
        # Create steps (S,)
        dsun: float = 151693705216.0
        rsun: float = 696000000.0
        dx = 1e7
        steps = torch.arange(dsun - 2*rsun, dsun + 2*rsun, dx, dtype=torch.float32)
        S = steps.shape[0]
        
        # Reshape steps to (S, 1) and then expand for batch: (B, S, 1)
        steps_reshaped = steps.unsqueeze(1)  # (S, 1)
        steps_expanded = steps_reshaped.unsqueeze(0).expand(B, S, 1)  # (B, S, 1)
        # Expand obs and ray for broadcasting: (B, 1, 3)
        obs_expanded = obs.unsqueeze(1)  # (B, 1, 3)
        ray_expanded = ray.unsqueeze(1)  # (B, 1, 3)
        # print(obs_expanded.shape, ray_expanded.shape, steps_expanded.shape)
        # Compute ray positions: (B, S, 3)
        ray_with_steps = obs_expanded + steps_expanded * ray_expanded
        
        # Evaluate neural field: (B, S)
        output_tensor = sun_sphere_scalar(ray_with_steps)
        
        # Collapse along steps dimension: (B,)
        pixel_value = output_tensor.sum(dim=1)

        return pixel_value

def compute_analytic_loss(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    dx: float = 1e7,
    device: str = 'cpu',
    loss_type: str = 'mse',
    limit_batches: int | None = None,
    save_example: str | None = None
):
    dm = SolarPerspectiveDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=42
    )
    dm.setup('validate')
    loader = dm.val_dataloader()

    if loss_type == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif loss_type == 'l1':
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError('loss_type must be mse or l1')

    losses = []
    example_saved = False

    for b_idx, batch in enumerate(loader):
        preds = []
        headers = batch['header']
        target = batch['image']  # (B,H,W)
        for hdr in headers:
            # Create dummy image just to carry header/WCS through integrator path
            dummy = np.ones((hdr['NAXIS2'], hdr['NAXIS1']), dtype=np.float32)
            hdu = PrimaryHDU(dummy, header=hdr)
            print("Dummy HDU created with shape:", dummy.shape)

            obs, rays = calculate_rays(source_wcs=WCS(hdu.header), shape_out=(dummy.shape[0], dummy.shape[1]), device=device)
            # Flatten rays and obs for batch processing
            obs_expanded = obs.expand(rays.shape[1], rays.shape[2], 3)
            print("Rays shape:", rays.shape)
            print("Obs expanded shape:", obs_expanded.shape)
            obs_flat = obs_expanded.reshape(-1, 3)
            rays_flat = rays.reshape(-1, 3)
            img_flat = evaluate(obs_flat, rays_flat, requires_grad=False)
            img = img_flat.reshape(dummy.shape[0], dummy.shape[1])
            print(img)
            preds.append(img)
        pred_batch = torch.stack(preds)  # (B,H,W)
        target = target.to(pred_batch.device)
        if pred_batch.shape != target.shape:
            pred_batch = torch.nn.functional.interpolate(
                pred_batch.unsqueeze(1), size=target.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(1)
        loss = loss_fn(pred_batch, target)
        losses.append(loss.detach())

        if save_example and not example_saved:
            try:
                import matplotlib.pyplot as plt
                diff = (target[0] - pred_batch[0]).abs().cpu().numpy()
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(target[0].cpu().numpy(), origin='lower', cmap='inferno'); axes[0].set_title('Target'); axes[0].axis('off')
                axes[1].imshow(pred_batch[0].cpu().numpy(), origin='lower', cmap='inferno'); axes[1].set_title('Analytic'); axes[1].axis('off')
                im = axes[2].imshow(diff, origin='lower', cmap='viridis'); axes[2].set_title('Abs Diff'); axes[2].axis('off')
                fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                Path(save_example).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_example, dpi=120, bbox_inches='tight')
                plt.close(fig)
                example_saved = True
            except Exception as e:
                print(f"Could not save example figure: {e}")

        if limit_batches is not None and (b_idx + 1) >= limit_batches:
            break

    mean_loss = torch.stack(losses).mean() if losses else torch.tensor(float('nan'))
    print(f"Analytic {loss_type.upper()} loss over {len(losses)} batch(es): {mean_loss.item():.6e}")
    return mean_loss


def main():
    p = argparse.ArgumentParser(description='Compute loss using analytic field instead of neural network')
    p.add_argument('--data-dir', type=str, required=True, help='Directory with prepared FITS dataset')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--dx', type=float, default=1e7)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--loss-type', type=str, default='mse', choices=['mse','l1'])
    p.add_argument('--limit-batches', type=int, default=None)
    p.add_argument('--save-example', type=str, default=None, help='Path to save one comparison figure')
    args = p.parse_args()
    compute_analytic_loss(**vars(args))

if __name__ == '__main__':
    main()
