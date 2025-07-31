"""
Main entry point for solar ray integration training.

Usage:
    python -m solar_ray_integration.training [subcommand] [args...]

Subcommands:
    train       - Start training (default)
    utils       - Run utilities (test, visualize, config)
    
Examples:
    python -m solar_ray_integration.training train --max-epochs 100
    python -m solar_ray_integration.training utils --action test
"""

import sys

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['train', '--help', '-h']:
        # Default to training
        from .train import main as train_main
        if len(sys.argv) >= 2 and sys.argv[1] == 'train':
            # Remove 'train' from args
            sys.argv = sys.argv[:1] + sys.argv[2:]
        train_main()
    elif sys.argv[1] == 'utils':
        # Remove 'utils' from args
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from .utils import main as utils_main
        utils_main()
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
