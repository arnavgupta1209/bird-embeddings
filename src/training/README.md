# Training Module

This module provides training functionality for Variational Autoencoder (VAE) models, including noise augmentation, checkpoint management, and training utilities.

---

## Overview

The training module implements a **denoising VAE training strategy**:
- Add Gaussian noise to input during training
- Model learns to reconstruct clean data from noisy input
- No noise during validation
- This acts as regularization and helps learn robust representations

---

## Quick Start

```python
from src.models import VariationalAutoencoder
from src.data import load_ebird_data, create_species_matrix, split_train_val, create_dataloaders
from src.training import train_variational_autoencoder, get_device, set_seed

# Set random seed for reproducibility
set_seed(42)

# Load and prepare data
data = load_ebird_data("kerala.txt")
matrix, species = create_species_matrix(data, min_species_observations=30)
train_ds, val_ds = split_train_val(matrix, species, val_size=0.2)
train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=128)

# Create model
model = VariationalAutoencoder(
    input_dimension=len(species),
    latent_dimension=16,
    hidden_dimension=512
)

# Auto-detect device
device = get_device()

# Train!
history = train_variational_autoencoder(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    noise_std=0.1,
    device=device,
    checkpoint_dir='checkpoints',
    save_every=10
)

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.legend()
plt.show()
```

---

## Main Training Function

### `train_variational_autoencoder()`

**Purpose:** Train a VAE model with noise augmentation and automatic checkpointing.

**Key Features:**
- Denoising autoencoder strategy (add noise during training)
- Separate tracking of reconstruction loss and KL divergence
- Automatic checkpoint saving every N epochs
- Progress printing with configurable frequency
- Returns complete training history

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | - | VAE model to train |
| `train_loader` | `DataLoader` | - | Training data loader |
| `val_loader` | `DataLoader` | - | Validation data loader |
| `num_epochs` | `int` | `50` | Number of training epochs |
| `learning_rate` | `float` | `1e-3` | Learning rate for Adam optimizer |
| `noise_std` | `float` | `0.1` | Std dev of Gaussian noise (0.0 = no noise) |
| `device` | `str` | `'cpu'` | Device to train on ('cpu', 'cuda', 'mps') |
| `print_every` | `int` | `1` | Print progress every N epochs |
| `checkpoint_dir` | `str` | `None` | Directory to save checkpoints (None = no saving) |
| `save_every` | `int` | `10` | Save checkpoint every N epochs |

**Returns:**

Dictionary with training history:
```python
{
    'train_loss': [...],        # Total training loss per epoch
    'train_recon_loss': [...],  # Training reconstruction loss
    'train_kl_loss': [...],     # Training KL divergence
    'val_loss': [...],          # Total validation loss per epoch
    'val_recon_loss': [...],    # Validation reconstruction loss
    'val_kl_loss': [...]        # Validation KL divergence
}
```

---

## Utility Functions

### Device Management

#### `get_device(prefer_cuda=True)`

Automatically detect the best available device.

```python
from src.training import get_device

# Auto-detect (prefers CUDA if available)
device = get_device()

# Prefer MPS over CUDA (for Apple Silicon)
device = get_device(prefer_cuda=False)
```

**Priority:** CUDA → MPS → CPU

---

### Checkpoint Management

#### `save_checkpoint()`

Save model and training state to disk.

```python
from src.training import save_checkpoint

save_checkpoint(
    model=vae_model,
    optimizer=optimizer,
    epoch=50,
    train_loss=120.5,
    val_loss=125.3,
    filepath='checkpoints/best_model.pt',
    history=training_history,
    metadata={'latent_dim': 16, 'noise_std': 0.1}
)
```

**Saved data:**
- Model state dict (weights and biases)
- Optimizer state dict (for resuming training)
- Current epoch number
- Training and validation losses
- Full training history (optional)
- Custom metadata (optional)

---

#### `load_checkpoint()`

Load checkpoint and restore training state.

```python
from src.training import load_checkpoint

# Create fresh model and optimizer
model = VariationalAutoencoder(input_dimension=467, latent_dimension=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load checkpoint
checkpoint = load_checkpoint(
    filepath='checkpoints/checkpoint_epoch_50.pt',
    model=model,
    optimizer=optimizer,
    device='cuda'
)

# Resume training from checkpoint
start_epoch = checkpoint['epoch'] + 1
previous_history = checkpoint['history']

# Continue training
history = train_variational_autoencoder(
    model,
    train_loader,
    val_loader,
    num_epochs=100,  # Train to epoch 100
    # ... other params
)
```

---

### Model Utilities

#### `print_model_summary(model)`

Print detailed model summary.

```python
from src.training import print_model_summary

model = VariationalAutoencoder(467, 16, 512)
print_model_summary(model)
```

**Output:**
```
================================================================================
Model Summary
================================================================================
VariationalAutoencoder(
  (encoder_network): Sequential(...)
  (latent_mean_layer): Linear(in_features=512, out_features=16)
  ...
)
--------------------------------------------------------------------------------
Total parameters: 1,554,931
Trainable parameters: 1,554,931
Model size: 5.93 MB
================================================================================
```

---

#### `count_parameters(model)`

Count total and trainable parameters.

```python
from src.training import count_parameters

total, trainable = count_parameters(model)
print(f"Total: {total:,}, Trainable: {trainable:,}")
# Total: 1,554,931, Trainable: 1,554,931
```

---

### Reproducibility

#### `set_seed(seed)`

Set random seed for reproducible results.

```python
from src.training import set_seed

set_seed(42)  # Now all random operations are deterministic
```

Sets seeds for:
- Python's `random` module
- NumPy
- PyTorch (CPU and CUDA)

⚠️ **Note:** Call this at the start of your script, before creating models or datasets.

---

### Noise Augmentation

#### `add_noise(data, noise_std=0.1)`

Add Gaussian noise to data (used internally by trainer).

```python
from src.training import add_noise
import torch

clean = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
noisy = add_noise(clean, noise_std=0.1)
# Output: tensor([[0.05, 0.98, 0.02, 1.00]])
```

---

## Training Strategies

### Standard Training

```python
history = train_variational_autoencoder(
    model, train_loader, val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    noise_std=0.1,
    device='cuda'
)
```

### No Noise (Pure VAE)

```python
history = train_variational_autoencoder(
    model, train_loader, val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    noise_std=0.0,  # No noise augmentation
    device='cuda'
)
```

### Heavy Regularization

```python
history = train_variational_autoencoder(
    model, train_loader, val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    noise_std=0.2,  # More noise = more regularization
    device='cuda'
)
```

### With Checkpointing

```python
history = train_variational_autoencoder(
    model, train_loader, val_loader,
    num_epochs=100,
    checkpoint_dir='checkpoints/experiment_1',
    save_every=5,  # Save every 5 epochs
    device='cuda'
)
```

---

## Hyperparameter Guidelines

### Learning Rate

- **Default:** `1e-3` (0.001)
- **Too high:** Loss oscillates or diverges
- **Too low:** Training is very slow
- **Typical range:** `1e-4` to `1e-2`

### Noise Standard Deviation

- **Default:** `0.1`
- **No noise:** `0.0` (pure VAE, less regularization)
- **Light noise:** `0.05` to `0.1`
- **Heavy noise:** `0.15` to `0.3` (strong regularization)
- **Effect:** Higher noise → more robust representations, but slower convergence

### Batch Size

- **Typical range:** `64` to `256`
- **Larger batches:** More stable gradients, faster training (if GPU has memory)
- **Smaller batches:** More noise in gradients, may generalize better

### Number of Epochs

- **Small datasets:** `30-50` epochs
- **Large datasets:** `50-100` epochs
- **Monitor:** Stop when validation loss plateaus or starts increasing

---

## Monitoring Training

### Loss Components

The trainer tracks three loss components:

1. **Total Loss** = Reconstruction Loss + KL Divergence
   - This is what the optimizer minimizes
   
2. **Reconstruction Loss** (Binary Cross-Entropy)
   - How well model reconstructs input
   - Should decrease steadily
   
3. **KL Divergence**
   - Regularization term (latent space → standard normal)
   - Usually small compared to reconstruction loss
   - Prevents overfitting in latent space

### Expected Behavior

**Healthy training:**
- Both train and val losses decrease
- Val loss follows train loss closely
- Losses stabilize after some epochs

**Overfitting:**
- Train loss keeps decreasing
- Val loss stops decreasing or increases
- **Solution:** Add more noise, reduce model size, or stop training earlier

**Underfitting:**
- Both losses plateau at high values
- **Solution:** Increase model capacity, train longer, reduce noise

---

## Complete Training Pipeline Example

```python
from src.models import VariationalAutoencoder
from src.data import (
    load_ebird_data, 
    create_species_matrix, 
    split_train_val, 
    create_dataloaders
)
from src.training import (
    train_variational_autoencoder,
    get_device,
    set_seed,
    print_model_summary,
    save_checkpoint
)
import matplotlib.pyplot as plt

# 1. Set random seed for reproducibility
set_seed(42)

# 2. Load and preprocess data
print("Loading data...")
data = load_ebird_data("kerala_data.txt", nrows=500000)

print("Creating species matrix...")
matrix, species = create_species_matrix(
    data,
    min_species_observations=50
)

print("Splitting into train/val...")
train_ds, val_ds = split_train_val(matrix, species, val_size=0.2)

print("Creating DataLoaders...")
train_loader, val_loader = create_dataloaders(
    train_ds, val_ds,
    batch_size=128,
    num_workers=4,
    pin_memory=True  # For GPU training
)

# 3. Create model
print("\nCreating model...")
model = VariationalAutoencoder(
    input_dimension=len(species),
    latent_dimension=16,
    hidden_dimension=512
)

print_model_summary(model)

# 4. Setup training
device = get_device()

# 5. Train
print("\nStarting training...")
history = train_variational_autoencoder(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    noise_std=0.1,
    device=device,
    checkpoint_dir='checkpoints/kerala_vae',
    save_every=10,
    print_every=5
)

# 6. Save final model
import torch
torch.save(model.state_dict(), 'vae_kerala_final.pt')

# 7. Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total loss
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Total Loss')
axes[0].legend()

# Reconstruction loss
axes[1].plot(history['train_recon_loss'], label='Train')
axes[1].plot(history['val_recon_loss'], label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Reconstruction Loss')
axes[1].set_title('Reconstruction Loss (BCE)')
axes[1].legend()

# KL divergence
axes[2].plot(history['train_kl_loss'], label='Train')
axes[2].plot(history['val_kl_loss'], label='Val')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('KL Divergence')
axes[2].set_title('KL Divergence')
axes[2].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()

print("\n✅ Training complete!")
```

---

## Checkpoint File Format

Checkpoints are saved as PyTorch `.pt` files containing:

```python
{
    'epoch': 50,                          # Epoch number
    'model_state_dict': {...},            # Model weights
    'optimizer_state_dict': {...},        # Optimizer state
    'train_loss': 120.5,                  # Training loss at this epoch
    'val_loss': 125.3,                    # Validation loss at this epoch
    'history': {                          # Full training history (optional)
        'train_loss': [...],
        'val_loss': [...],
        ...
    },
    'metadata': {                         # Custom metadata (optional)
        'latent_dim': 16,
        'hidden_dim': 512,
        'noise_std': 0.1,
        ...
    }
}
```

---

## Troubleshooting

### Loss is NaN

**Causes:**
- Learning rate too high
- Numerical instability

**Solutions:**
- Reduce learning rate (try `1e-4`)
- Check data for NaN values
- Ensure data is in valid range [0, 1]

### Training is very slow

**Causes:**
- Using CPU instead of GPU
- Batch size too small
- Too many workers for DataLoader

**Solutions:**
- Use `device = get_device()` to use GPU
- Increase batch size (if GPU has memory)
- Set `num_workers=0` on Windows

### Model not improving

**Causes:**
- Learning rate too low
- Model too small
- Too much noise

**Solutions:**
- Increase learning rate
- Increase `hidden_dimension`
- Reduce `noise_std`

### Overfitting

**Causes:**
- Model too large
- Not enough regularization

**Solutions:**
- Increase `noise_std` (more regularization)
- Reduce model size
- Stop training earlier

---

## Module Structure

```
src/training/
├── __init__.py           # Module exports
├── trainer.py            # Main training function
├── utils.py              # Utility functions
└── README.md             # This file
```

---

**Module Status:** ✅ Complete (Phase 4 - Tasks 1-2)

**Next Steps:**
- Create test notebook → `notebooks/test_training.ipynb`
- Extract inference utilities → `src/inference/` (Phase 5)
