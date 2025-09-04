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

## Debugging

If training gets killed without clear error messages, try these debugging approaches:

### 1. Enable Verbose Logging
```bash
# Run with verbose output and save logs
python -m solar_ray_integration.training train --batch-size 1 --max-epochs 1 --gpus 0 2>&1 | tee training.log
```

### 2. Monitor System Resources
```bash
# In another terminal, monitor memory and CPU usage
htop
# or
watch -n 1 'free -h && echo "CPU:" && top -bn1 | grep "Cpu(s)"'
```

### 3. Check for OOM (Out of Memory) Issues
```bash
# Check system logs for OOM killer messages
sudo dmesg | grep -i "killed\|memory\|oom"
# or
journalctl -f | grep -i "killed\|memory\|oom"
```

### 4. Run with Reduced Resources
```bash
# Use smaller batch size and fewer workers
python -m solar_ray_integration.training train --batch-size 1 --num-workers 1 --max-epochs 1
```

### 5. Profile Memory Usage
```bash
# Install memory profiler
pip install memory-profiler

# Run with memory profiling
python -m memory_profiler -m solar_ray_integration.training train --batch-size 1 --max-epochs 1
```

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



### 1. Memory Optimization
```bash
# Use minimal memory settings
python -m solar_ray_integration.training train \
    --batch-size 1 \
    --num-workers 0 \
    --max-epochs 1 \
    --gpus 0 \
    --precision 32

# Monitor memory usage while training
free -h && python -m solar_ray_integration.training train --batch-size 1 --num-workers 0 --gpus 0
```

### 2. Reduce Model Size
Edit the `nerf_config` in `train.py`:
```python
nerf_config = {
    'd_input': 3,
    'd_output': 1,
    'n_layers': 2,    # Reduce from 8
    'd_filter': 128,  # Reduce from 512
    'encoding': 'positional'
}
```

### 3. Enable Verbose Logging
```bash
# Run with verbose output and save logs
python -m solar_ray_integration.training train --batch-size 1 --max-epochs 1 --gpus 0 2>&1 | tee training.log
```

### 4. Run with Reduced Resources
```bash
# Use minimal settings for testing
python -m solar_ray_integration.training train \
    --batch-size 1 \
    --num-workers 0 \
    --max-epochs 1 \
    --gpus 0 \
    --precision 32

# Set memory limits (optional)
ulimit -v 4000000  # Limit virtual memory to ~4GB
python -m solar_ray_integration.training train --batch-size 1 --num-workers 0 --gpus 0
```

### 5. Profile Memory Usage
```bash
# Install memory profiler
pip install memory-profiler

# Run with memory profiling
python -m memory_profiler -m solar_ray_integration.training train --batch-size 1 --max-epochs 1
```

## Memory Requirements

Typical memory usage by configuration:

| Batch Size | Model Size | Workers | Estimated RAM |
|------------|------------|---------|---------------|
| 1          | Small (2 layers, 128 width) | 0 | ~1-2 GB |
| 1          | Medium (4 layers, 256 width) | 0 | ~2-4 GB |
| 4          | Large (8 layers, 512 width) | 4 | ~8-16 GB |

For systems with limited memory:
- Use `--batch-size 1`
- Use `--num-workers 0`
- Use `--precision 32` (CPU only)
- Reduce model size in code

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
