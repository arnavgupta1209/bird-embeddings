"""
Training Utilities

Helper functions for training VAE models including device management,
checkpoint saving/loading, and noise augmentation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import os
from pathlib import Path


def get_device(prefer_cuda: bool = True) -> str:
    """
    Automatically detect and return the best available device.
    
    Priority order:
    1. CUDA (NVIDIA GPU) if available and prefer_cuda=True
    2. MPS (Apple Silicon GPU) if available
    3. CPU as fallback
    
    Args:
        prefer_cuda (bool): If True, prefer CUDA over MPS when both available.
            Default: True
            
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
        
    Example:
        >>> device = get_device()
        >>> print(f"Training on: {device}")
        Training on: cuda
        >>> 
        >>> model = model.to(device)
        >>> data = data.to(device)
    """
    if prefer_cuda and torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print("✓ Using CPU")
    
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    filepath: str,
    history: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with training state.
    
    Saves:
    - Model state dict (weights and biases)
    - Optimizer state dict (for resuming training)
    - Training metadata (epoch, losses, history)
    - Optional custom metadata
    
    Args:
        model (nn.Module): PyTorch model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current epoch number
        train_loss (float): Training loss at this epoch
        val_loss (float): Validation loss at this epoch
        filepath (str): Path where checkpoint will be saved
        history (dict, optional): Full training history dict.
            Default: None
        metadata (dict, optional): Additional metadata to save
            (e.g., hyperparameters, model config). Default: None
            
    Example:
        >>> save_checkpoint(
        ...     model=vae_model,
        ...     optimizer=optimizer,
        ...     epoch=50,
        ...     train_loss=120.5,
        ...     val_loss=125.3,
        ...     filepath='checkpoints/best_model.pt',
        ...     history=training_history,
        ...     metadata={'latent_dim': 16, 'hidden_dim': 512}
        ... )
        ✓ Saved checkpoint to: checkpoints/best_model.pt
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # Add optional data
    if history is not None:
        checkpoint['history'] = history
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Save to disk
    torch.save(checkpoint, filepath)
    print(f"✓ Saved checkpoint to: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint and restore training state.
    
    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.
            If None, optimizer state is not restored. Default: None
        device (str): Device to load checkpoint onto. Default: 'cpu'
            
    Returns:
        dict: Checkpoint dictionary containing:
            - epoch: Epoch number when checkpoint was saved
            - train_loss: Training loss at checkpoint
            - val_loss: Validation loss at checkpoint
            - history: Training history (if available)
            - metadata: Additional metadata (if available)
            
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        
    Example:
        >>> # Load checkpoint and continue training
        >>> model = VariationalAutoencoder(input_dimension=467, latent_dimension=16)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> 
        >>> checkpoint = load_checkpoint(
        ...     'checkpoints/checkpoint_epoch_50.pt',
        ...     model,
        ...     optimizer,
        ...     device='cuda'
        ... )
        >>> 
        >>> start_epoch = checkpoint['epoch'] + 1
        >>> print(f"Resuming from epoch {start_epoch}")
        >>> print(f"Previous val loss: {checkpoint['val_loss']:.4f}")
        
    Notes:
        - Model and optimizer states are loaded in-place
        - Returns checkpoint dict for accessing history/metadata
        - Use device parameter to load checkpoint to specific device
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Loaded checkpoint from: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint


def add_noise(
    data: torch.Tensor,
    noise_std: float = 0.1,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0
) -> torch.Tensor:
    """
    Add Gaussian noise to data and clamp to valid range.
    
    This is the noise augmentation strategy used during VAE training.
    Adding noise acts as regularization and helps the model learn more
    robust representations.
    
    Args:
        data (torch.Tensor): Input data tensor
        noise_std (float): Standard deviation of Gaussian noise.
            Higher values = more noise = more regularization.
            Default: 0.1
        clamp_min (float): Minimum value after clamping. Default: 0.0
        clamp_max (float): Maximum value after clamping. Default: 1.0
            
    Returns:
        torch.Tensor: Noisy data clamped to [clamp_min, clamp_max]
        
    Example:
        >>> clean_data = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        >>> noisy_data = add_noise(clean_data, noise_std=0.1)
        >>> print(noisy_data)
        tensor([[0.0234, 0.9876, 0.0123, 1.0000]])
        
    Notes:
        - Noise is sampled from N(0, noise_std^2)
        - Clamping ensures data stays in valid range
        - For binary data, clamping to [0, 1] is typical
    """
    # Add Gaussian noise
    noisy = data + noise_std * torch.randn_like(data)
    
    # Clamp to valid range
    noisy = torch.clamp(noisy, clamp_min, clamp_max)
    
    return noisy


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            
    Example:
        >>> model = VariationalAutoencoder(
        ...     input_dimension=467,
        ...     latent_dimension=16,
        ...     hidden_dimension=512
        ... )
        >>> total, trainable = count_parameters(model)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
        Total: 718,835, Trainable: 718,835
        
    Notes:
        - Useful for understanding model size
        - Trainable params may be less than total if some layers are frozen
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model (nn.Module): PyTorch model
        
    Example:
        >>> model = VariationalAutoencoder(467, 16, 512)
        >>> print_model_summary(model)
        ================================================================================
        Model Summary
        ================================================================================
        VariationalAutoencoder(
          (encoder_hidden): Linear(in_features=467, out_features=512, bias=True)
          (encoder_mean): Linear(in_features=512, out_features=16, bias=True)
          (encoder_log_variance): Linear(in_features=512, out_features=16, bias=True)
          (decoder_hidden): Linear(in_features=16, out_features=512, bias=True)
          (decoder_output): Linear(in_features=512, out_features=467, bias=True)
        )
        --------------------------------------------------------------------------------
        Total parameters: 718,835
        Trainable parameters: 718,835
        Model size: 2.74 MB
        ================================================================================
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(model)
    print("-" * 80)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Calculate model size in MB (assuming float32 = 4 bytes)
    size_mb = (total * 4) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")
    print("=" * 80)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed (int): Random seed value
        
    Example:
        >>> set_seed(42)
        ✓ Random seed set to: 42
        >>> # Now all random operations will be reproducible
        
    Notes:
        - Call this at the start of your script for reproducible results
        - CUDA operations may still have some randomness
        - For complete reproducibility, also set CUBLAS workspace config
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    print(f"✓ Random seed set to: {seed}")


def save_model_for_inference(
    model: nn.Module,
    filepath: str,
    input_dim: int,
    latent_dim: int,
    hidden_dims: list,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save model in a format optimized for inference.
    
    This function saves the model state dict along with architecture parameters,
    making it easy to reload the model without needing to know the architecture
    beforehand. This is the recommended format for deployed models.
    
    Args:
        model (nn.Module): Trained model to save
        filepath (str): Path where to save the model
        input_dim (int): Input dimension of the model
        latent_dim (int): Latent dimension of the model
        hidden_dims (list): List of hidden layer dimensions
        metadata (dict, optional): Additional metadata to save (e.g., training info)
        
    Example:
        >>> model = VariationalAutoencoder(
        ...     input_dim=467,
        ...     latent_dim=16,
        ...     hidden_dims=[512, 256]
        ... )
        >>> # After training...
        >>> save_model_for_inference(
        ...     model,
        ...     'models/vae_kerala_final.pth',
        ...     input_dim=467,
        ...     latent_dim=16,
        ...     hidden_dims=[512, 256],
        ...     metadata={'dataset': 'kerala', 'species': 244}
        ... )
        ✓ Model saved for inference: models/vae_kerala_final.pth
        
    Notes:
        - Saves state_dict (not full model object) for portability
        - Includes architecture params for easy reloading
        - Compatible with src.inference.EmbeddingExtractor
        - Does NOT save optimizer state (use save_checkpoint for that)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims
    }
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"✓ Model saved for inference: {filepath}")
    print(f"  Input dim: {input_dim}, Latent dim: {latent_dim}")
    print(f"  Hidden dims: {hidden_dims}")
