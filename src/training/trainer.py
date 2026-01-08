"""
VAE Training Module

This module provides training functionality for Variational Autoencoders (VAEs).
Includes noise augmentation, loss tracking, and checkpoint management.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Dict
import os
from pathlib import Path


def train_variational_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    noise_std: float = 0.1,
    device: str = 'cpu',
    print_every: int = 1,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10
) -> Dict[str, list]:
    """
    Train a Variational Autoencoder (VAE) with noise augmentation.
    
    Training strategy:
    1. Add Gaussian noise to input during training (denoising autoencoder)
    2. Clamp noisy input to [0, 1] to keep valid range
    3. Model reconstructs the clean (non-noisy) input
    4. No noise during validation
    
    This noise augmentation acts as regularization and helps the model learn
    more robust representations.
    
    Args:
        model (nn.Module): VAE model to train (e.g., VariationalAutoencoder)
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs. Default: 50
        learning_rate (float): Learning rate for Adam optimizer. Default: 1e-3
        noise_std (float): Standard deviation of Gaussian noise added to training
            data. Higher values = more regularization. Default: 0.1
        device (str): Device to train on ('cpu', 'cuda', 'mps'). Default: 'cpu'
        print_every (int): Print progress every N epochs. Default: 1
        checkpoint_dir (str, optional): Directory to save checkpoints.
            If None, no checkpoints are saved. Default: None
        save_every (int): Save checkpoint every N epochs. Default: 10
            
    Returns:
        dict: Training history with keys:
            - 'train_loss': List of average training losses per epoch
            - 'train_recon_loss': List of training reconstruction losses
            - 'train_kl_loss': List of training KL divergence losses
            - 'val_loss': List of average validation losses per epoch
            - 'val_recon_loss': List of validation reconstruction losses
            - 'val_kl_loss': List of validation KL divergence losses
            
    Example:
        >>> from src.models import VariationalAutoencoder
        >>> from src.data import create_dataloaders, split_train_val
        >>> from src.training import train_variational_autoencoder
        >>> 
        >>> # Setup model and data
        >>> model = VariationalAutoencoder(input_dim=467, latent_dim=16, hidden_dim=512)
        >>> train_ds, val_ds = split_train_val(matrix, species)
        >>> train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=128)
        >>> 
        >>> # Train
        >>> history = train_variational_autoencoder(
        ...     model,
        ...     train_loader,
        ...     val_loader,
        ...     num_epochs=50,
        ...     learning_rate=1e-3,
        ...     noise_std=0.1,
        ...     device='cuda',
        ...     checkpoint_dir='checkpoints',
        ...     save_every=10
        ... )
        >>> 
        >>> # Plot losses
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history['train_loss'], label='Train')
        >>> plt.plot(history['val_loss'], label='Val')
        >>> plt.legend()
        >>> plt.show()
        
    Notes:
        - Model is automatically moved to specified device
        - Uses Adam optimizer
        - Loss function should be imported from src.models
        - Training uses model.train() mode, validation uses model.eval() mode
        - Gradients are disabled during validation for efficiency
    """
    
    # Import loss function (assumes it's available from src.models)
    from src.models import variational_autoencoder_loss_function
    
    # Initialize tracking lists
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    print(f"\nStarting training on {device}...")
    print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}, Noise std: {noise_std}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 80)
    
    # Training loop
    for epoch in range(num_epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        
        # Initialize epoch metrics
        epoch_train_loss = 0.0
        epoch_train_recon_loss = 0.0
        epoch_train_kl_loss = 0.0
        
        for batch in train_loader:
            # Get clean data and move to device
            clean = batch.to(device)
            
            # Add Gaussian noise to input (denoising autoencoder strategy)
            # This acts as regularization and helps learn robust representations
            noisy = clean + noise_std * torch.randn_like(clean)
            
            # Clamp to valid range [0, 1] since our data is binary presence/absence
            noisy = torch.clamp(noisy, 0.0, 1.0)
            
            # Forward pass: encode noisy input, decode to reconstruct clean input
            reconstructed, latent_mean, latent_logvar = model(noisy)
            
            # Compute loss: model tries to reconstruct clean data from noisy input
            loss, recon_loss, kl_loss = variational_autoencoder_loss_function(
                reconstructed_input=reconstructed,
                original_input=clean,  # Target is clean data, not noisy
                latent_mean=latent_mean,
                latent_log_variance=latent_logvar
            )
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate batch losses
            epoch_train_loss += loss.item()
            epoch_train_recon_loss += recon_loss.item()
            epoch_train_kl_loss += kl_loss.item()
        
        # Calculate average training losses for this epoch
        # Divide by number of samples (not batches) for per-sample loss
        num_train_samples = len(train_loader.dataset)
        avg_train_loss = epoch_train_loss / num_train_samples
        avg_train_recon_loss = epoch_train_recon_loss / num_train_samples
        avg_train_kl_loss = epoch_train_kl_loss / num_train_samples
        
        history['train_loss'].append(avg_train_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        history['train_kl_loss'].append(avg_train_kl_loss)
        
        # ========== VALIDATION PHASE ==========
        model.eval()
        
        # Initialize epoch metrics
        epoch_val_loss = 0.0
        epoch_val_recon_loss = 0.0
        epoch_val_kl_loss = 0.0
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for batch in val_loader:
                # Get clean data and move to device
                clean = batch.to(device)
                
                # NO NOISE during validation - evaluate on clean data
                reconstructed, latent_mean, latent_logvar = model(clean)
                
                # Compute validation loss
                val_loss, val_recon_loss, val_kl_loss = variational_autoencoder_loss_function(
                    reconstructed_input=reconstructed,
                    original_input=clean,
                    latent_mean=latent_mean,
                    latent_log_variance=latent_logvar
                )
                
                # Accumulate batch losses
                epoch_val_loss += val_loss.item()
                epoch_val_recon_loss += val_recon_loss.item()
                epoch_val_kl_loss += val_kl_loss.item()
        
        # Calculate average validation losses for this epoch
        num_val_samples = len(val_loader.dataset)
        avg_val_loss = epoch_val_loss / num_val_samples
        avg_val_recon_loss = epoch_val_recon_loss / num_val_samples
        avg_val_kl_loss = epoch_val_kl_loss / num_val_samples
        
        history['val_loss'].append(avg_val_loss)
        history['val_recon_loss'].append(avg_val_recon_loss)
        history['val_kl_loss'].append(avg_val_kl_loss)
        
        # ========== LOGGING ==========
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} "
                f"(Recon: {avg_train_recon_loss:.4f}, KL: {avg_train_kl_loss:.4f}) | "
                f"Val Loss: {avg_val_loss:.4f} "
                f"(Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f})"
            )
        
        # ========== CHECKPOINTING ==========
        if checkpoint_dir is not None and (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}")
    
    print("-" * 80)
    print(f"✓ Training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    
    return history
