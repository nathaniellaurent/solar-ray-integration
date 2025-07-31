# Solar NeRF Training

This directory contains the training pipeline for Solar Neural Radiance Fields (NeRF) models using PyTorch Lightning.

## Files

- `dataset.py`: Dataset and DataModule classes for loading solar perspective data
- `lightning_module.py`: PyTorch Lightning module defining the training/validation loops
- `train.py`: Main training script with command-line interface
- `config.py`: Configuration management with predefined settings
- `utils.py`: Utility functions for testing and visualization

## Quick Start

The training module uses a unified command interface:

1. **Test the pipeline:**
   ```bash
   python -m solar_ray_integration.training utils --action test
   ```

2. **Visualize sample data:**
   ```bash
   python -m solar_ray_integration.training utils --action visualize
   ```

3. **Start training:**
   ```bash
   python -m solar_ray_integration.training train
   ```

4. **Quick test run:**
   ```bash
   python -m solar_ray_integration.training train --max-epochs 5 --batch-size 2
   ```

> **Note**: The training module can also be run with just `python -m solar_ray_integration.training` (defaults to training) or `python -m solar_ray_integration.training --help` for help.

## Configuration

The training can be configured using command-line arguments or YAML configuration files:

```bash
# Create a configuration template
python -m solar_ray_integration.training utils --action config

# Edit the generated training_config.yaml file, then use it
python -m solar_ray_integration.training train --config training_config.yaml
```

## Data Format

The training expects FITS files with the naming convention:
```
output_tensor_{idx:03d}_hgln_{longitude}_hglt_{latitude}.fits
```

Each FITS file should contain:
- Image data in the primary HDU
- WCS header information
- HGLN_OBS and HGLT_OBS keywords for viewing angles

## Training Arguments

Key training arguments:

- `--data-dir`: Directory containing FITS files (default: perspective_data/perspective_data_linear_fits)
- `--batch-size`: Training batch size (default: 4)
- `--max-epochs`: Maximum training epochs (default: 100)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--integration-method`: Ray integration method (linear, volumetric, volumetric_correction)
- `--gpus`: Number of GPUs to use (default: 1)

## Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

This will show:
- Training and validation losses
- Learning rate schedules
- Sample rendered images
- Model metrics

## Output

Training outputs are saved to `outputs/` by default:
- `checkpoints/`: Model checkpoints
- `logs/`: TensorBoard logs
- `final_model.ckpt`: Final trained model

## Requirements

- PyTorch Lightning
- PyTorch
- Astropy
- Matplotlib
- NumPy
- TensorBoard

Install with:
```bash
pip install pytorch-lightning torch astropy matplotlib numpy tensorboard
```
