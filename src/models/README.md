# VAE Model Module

This module contains the Variational Autoencoder (VAE) architecture for learning embeddings from eBird checklist data.

## Overview

The VAE learns to compress high-dimensional bird species presence-absence data (e.g., 500+ species) into a lower-dimensional continuous latent space (e.g., 16 dimensions). This creates meaningful embeddings that capture ecological patterns and relationships between different checklists.

## Components

### `VariationalAutoencoder` Class

**Purpose**: Neural network that learns to encode and decode bird checklist data.

**Architecture**:
```
Input (e.g., 500 species)
    ↓
Encoder (3-layer MLP: 500 → 512 → 512 → 512)
    ↓
Latent Parameters (μ and log(σ²), each 16-dim)
    ↓
Reparameterization (sample z ~ N(μ, σ²))
    ↓
Decoder (3-layer MLP: 16 → 512 → 512 → 512 → 500)
    ↓
Reconstructed Output (sigmoid, 0-1 probabilities)
```

**Parameters**:
- `input_dimension` (int): Number of input features (number of species in dataset)
- `latent_dimension` (int, default=16): Size of the embedding space
- `hidden_dimension` (int, default=512): Number of neurons in hidden layers

**Methods**:
- `encode(input_tensor)`: Maps input to latent distribution parameters (μ, log(σ²))
- `reparameterize(mean, logvar)`: Samples from latent distribution using reparameterization trick
- `decode(latent_sample)`: Maps latent vector back to reconstructed input
- `forward(input_tensor)`: Complete encoding-decoding pass

### `variational_autoencoder_loss_function`

**Purpose**: Computes the VAE training loss (negative ELBO).

**Loss Components**:
1. **Reconstruction Loss** (Binary Cross Entropy): Measures how well the model reconstructs the input
2. **KL Divergence**: Regularizes the latent space to follow N(0,1) distribution

**Formula**:
```
Total Loss = BCE(reconstruction, input) + KL(N(μ, σ²) || N(0, 1))
```

**Returns**: `(total_loss, reconstruction_loss, kl_divergence_loss)`

## Usage Example

```python
import torch
from src.models.vae import VariationalAutoencoder, variational_autoencoder_loss_function

# Create model for 500 species with 16-dimensional embeddings
vae = VariationalAutoencoder(
    input_dimension=500,
    latent_dimension=16,
    hidden_dimension=512
)

# Forward pass
input_data = torch.randn(32, 500)  # batch of 32 checklists
reconstructed, mu, logvar = vae(input_data)

# Compute loss
total_loss, recon_loss, kl_loss = variational_autoencoder_loss_function(
    reconstructed_input=reconstructed,
    original_input=input_data,
    latent_mean=mu,
    latent_log_variance=logvar
)

# Get embeddings (mean of latent distribution)
with torch.no_grad():
    embeddings, _ = vae.encode(input_data)  # Shape: [32, 16]
```

## Technical Details

### Reparameterization Trick

To allow backpropagation through the stochastic sampling process, we use the reparameterization trick:

```
z = μ + σ * ε,  where ε ~ N(0, 1)
```

This transforms a non-differentiable sampling operation into a differentiable one.

### Why Sigmoid Output?

The decoder uses sigmoid activation because:
- eBird checklist data is binary (species present/absent) or can be treated as probabilities
- Sigmoid maps to [0, 1] range, matching the data distribution
- Works well with Binary Cross Entropy loss

### Why Log-Variance?

We predict `log(σ²)` instead of `σ²` because:
- Ensures variance is always positive (σ² = exp(log(σ²)))
- More numerically stable during training
- Prevents numerical overflow/underflow

## Model Size

For a typical configuration:
- Input: 500 species
- Latent: 16 dimensions
- Hidden: 512 neurons

**Parameters**:
- Encoder: ~1.3M parameters
- Decoder: ~1.3M parameters
- **Total: ~2.6M parameters**

## References

- Kingma & Welling (2013): "Auto-Encoding Variational Bayes" - Original VAE paper
- Doersch (2016): "Tutorial on Variational Autoencoders" - Excellent tutorial
