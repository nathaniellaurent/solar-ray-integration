# Solar Ray Integration

A Python package for neural radiance field (NeRF) modeling of solar atmospheric phenomena through ray integration techniques. This project applies machine learning to solar physics by training neural networks to learn 3D scalar field representations of the solar atmosphere and render realistic solar observations from arbitrary viewing perspectives.

## Overview

This project implements Solar NeRF, a specialized neural radiance field architecture designed for modeling the solar corona and chromosphere. The system integrates scalar fields along rays cast from observer positions to generate synthetic solar images that match real observational data. The approach enables 3D reconstruction of solar atmospheric structures from 2D observational datasets with different heliographic viewing angles (HGLN/HGLT coordinates).

## Key Features

**Neural Architecture**: Implements a custom NeRF model with positional encoding for learning continuous 3D solar field representations from sparse observational data.

**Ray Integration Methods**: Supports multiple integration techniques (linear, volumetric, volumetric_correction) for computing line-of-sight intensities through the solar volume.

**Ensemble Modeling**: Includes tools for training ensembles of models using Stochastic Weight Averaging (SWA) and generating uncertainty quantification through ensemble statistics.

**Training Pipeline**: Complete PyTorch Lightning-based training infrastructure with configurable parameters, automatic checkpointing, TensorBoard logging, and comprehensive validation metrics.

**Visualization Tools**: Advanced plotting capabilities for comparing predictions with ground truth, analyzing uncertainty distributions, and generating publication-quality comparison plots with coordinated color scales.

## Applications

This framework enables researchers to reconstruct 3D solar atmospheric structures from multi-perspective observations, predict solar appearances from unobserved viewing angles, and quantify prediction uncertainties for scientific analysis. The ensemble capabilities support robust uncertainty estimation critical for space weather modeling and solar physics research.
