"""
Training module for VAE models.

Exports:
    - train_variational_autoencoder: Main training function with noise augmentation
    - get_device: Auto-detect best available device (CUDA/MPS/CPU)
    - save_checkpoint: Save model checkpoint with training state
    - load_checkpoint: Load checkpoint and restore training state
    - save_model_for_inference: Save model optimized for inference (NEW!)
    - add_noise: Add Gaussian noise to data (for augmentation)
    - count_parameters: Count model parameters
    - print_model_summary: Print model architecture summary
    - set_seed: Set random seed for reproducibility
"""

from .trainer import train_variational_autoencoder
from .utils import (
    get_device,
    save_checkpoint,
    load_checkpoint,
    save_model_for_inference,
    add_noise,
    count_parameters,
    print_model_summary,
    set_seed
)

__all__ = [
    'train_variational_autoencoder',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'save_model_for_inference',
    'add_noise',
    'count_parameters',
    'print_model_summary',
    'set_seed'
]
