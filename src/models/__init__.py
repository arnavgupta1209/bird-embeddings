"""
Models module for bird embeddings VAE.

Exports:
    - VariationalAutoencoder: Main VAE model class
    - variational_autoencoder_loss_function: VAE loss computation
"""

from .vae import VariationalAutoencoder, variational_autoencoder_loss_function

__all__ = [
    'VariationalAutoencoder',
    'variational_autoencoder_loss_function'
]
