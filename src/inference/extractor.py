"""
Inference module for extracting embeddings from eBird checklists using trained VAE.

This module provides functionality to:
- Load a trained VAE model
- Generate embeddings (latent representations) from bird checklist data
- Support batch processing and different sampling strategies
"""

import torch
import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path

from ..models import VariationalAutoencoder


class EmbeddingExtractor:
    """
    Extracts embeddings from eBird checklist data using a trained VAE model.
    
    The extractor can generate embeddings in two modes:
    1. Deterministic: Uses mean (μ) of latent distribution only
    2. Stochastic: Samples from latent distribution using reparameterization
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to the trained VAE checkpoint (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
    def _load_model(self, model_path: str) -> VariationalAutoencoder:
        """
        Load the trained VAE model from checkpoint.
        
        Supports two checkpoint formats:
        1. Legacy: Entire model object saved with torch.save(model, path)
        2. Modern: State dict saved with torch.save({'model_state_dict': ..., ...}, path)
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded VAE model
            
        Raises:
            ValueError: If checkpoint format is not recognized
        """
        # Load checkpoint with weights_only=False to support legacy pickled models
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Detect checkpoint format
        if isinstance(checkpoint, VariationalAutoencoder):
            # Format 1: Legacy - entire model object was saved
            # This happens when using: torch.save(model, path)
            model = checkpoint
            model.to(self.device)
            
            print(f"✓ Model loaded from {model_path} (legacy format)")
            print(f"  Device: {self.device}")
            
        elif isinstance(checkpoint, dict):
            # Format 2: Modern - state dict with metadata
            # This happens when using: torch.save({'model_state_dict': ..., ...}, path)
            
            # Verify required keys exist
            required_keys = ['model_state_dict', 'input_dim', 'latent_dim', 'hidden_dims']
            missing_keys = [k for k in required_keys if k not in checkpoint]
            
            if missing_keys:
                raise ValueError(
                    f"Checkpoint dict is missing required keys: {missing_keys}. "
                    f"Found keys: {list(checkpoint.keys())}"
                )
            
            # Extract model architecture parameters
            input_dim = checkpoint['input_dim']
            latent_dim = checkpoint['latent_dim']
            hidden_dims = checkpoint['hidden_dims']
            
            # hidden_dims should be a single int (the VAE uses only one hidden dimension)
            # If it's a list, take the first element
            if isinstance(hidden_dims, list):
                hidden_dim = hidden_dims[0]
            else:
                hidden_dim = hidden_dims
            
            # Create model with same architecture
            # Note: VAE constructor uses singular parameter names
            model = VariationalAutoencoder(
                input_dimension=input_dim,
                latent_dimension=latent_dim,
                hidden_dimension=hidden_dim
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            print(f"✓ Model loaded from {model_path}")
            print(f"  Input dim: {input_dim}, Latent dim: {latent_dim}, Hidden dim: {hidden_dim}, Device: {self.device}")
            
        else:
            raise ValueError(
                f"Unsupported checkpoint format. Expected either:\n"
                f"  1. VariationalAutoencoder object (legacy), or\n"
                f"  2. Dict with keys: model_state_dict, input_dim, latent_dim, hidden_dims\n"
                f"  Got: {type(checkpoint)}"
            )
        
        return model
    
    def extract_embeddings(
        self,
        data: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        use_mean: bool = True,
        batch_size: int = 256
    ) -> np.ndarray:
        """
        Extract embeddings from checklist data.
        
        Args:
            data: Input data (tensor, numpy array, or DataFrame)
            use_mean: If True, uses mean (μ) only; if False, samples from distribution
            batch_size: Batch size for processing (larger = faster but more memory)
            
        Returns:
            Numpy array of embeddings, shape (n_samples, latent_dim)
        """
        # Convert input to tensor
        if isinstance(data, pd.DataFrame):
            data_tensor = torch.FloatTensor(data.values)
        elif isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data)
        else:
            data_tensor = data
            
        data_tensor = data_tensor.to(self.device)
        
        embeddings = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i + batch_size]
                
                # Get latent representation
                mu, log_var = self.model.encode(batch)
                
                if use_mean:
                    # Use mean directly (deterministic)
                    z = mu
                else:
                    # Sample from distribution (stochastic)
                    z = self.model.reparameterize(mu, log_var)
                
                embeddings.append(z.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def extract_with_reconstruction(
        self,
        data: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        use_mean: bool = True,
        batch_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and reconstructions from checklist data.
        
        Useful for analyzing reconstruction quality or detecting anomalies.
        
        Args:
            data: Input data (tensor, numpy array, or DataFrame)
            use_mean: If True, uses mean (μ) only; if False, samples from distribution
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings, reconstructions), both as numpy arrays
        """
        # Convert input to tensor
        if isinstance(data, pd.DataFrame):
            data_tensor = torch.FloatTensor(data.values)
        elif isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data)
        else:
            data_tensor = data
            
        data_tensor = data_tensor.to(self.device)
        
        embeddings = []
        reconstructions = []
        
        with torch.no_grad():
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i + batch_size]
                
                # Get full forward pass
                mu, log_var = self.model.encode(batch)
                
                if use_mean:
                    z = mu
                else:
                    z = self.model.reparameterize(mu, log_var)
                
                recon = self.model.decode(z)
                
                embeddings.append(z.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())
        
        return np.vstack(embeddings), np.vstack(reconstructions)


def load_and_extract(
    model_path: str,
    data: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    use_mean: bool = True,
    batch_size: int = 256,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to load model and extract embeddings in one call.
    
    Args:
        model_path: Path to trained VAE checkpoint
        data: Input checklist data
        use_mean: If True, uses mean (μ) only; if False, samples from distribution
        batch_size: Batch size for processing
        device: Device to run on ('cuda', 'cpu', or None for auto)
        
    Returns:
        Numpy array of embeddings
    """
    extractor = EmbeddingExtractor(model_path, device)
    return extractor.extract_embeddings(data, use_mean, batch_size)


def save_embeddings(
    embeddings: np.ndarray,
    save_path: str,
    checklist_ids: Optional[pd.Series] = None,
    metadata: Optional[dict] = None
):
    """
    Save embeddings to file with optional metadata.
    
    Args:
        embeddings: Numpy array of embeddings
        save_path: Path to save file (.npz or .csv)
        checklist_ids: Optional checklist IDs to include
        metadata: Optional dictionary of metadata to save
    """
    save_path = Path(save_path)
    
    if save_path.suffix == '.npz':
        # Save as compressed numpy format
        save_dict = {'embeddings': embeddings}
        if checklist_ids is not None:
            save_dict['checklist_ids'] = checklist_ids.values
        if metadata is not None:
            save_dict.update(metadata)
        np.savez_compressed(save_path, **save_dict)
        
    elif save_path.suffix == '.csv':
        # Save as CSV
        df = pd.DataFrame(embeddings)
        if checklist_ids is not None:
            df.insert(0, 'checklist_id', checklist_ids.values)
        df.to_csv(save_path, index=False)
        
    else:
        raise ValueError(f"Unsupported file format: {save_path.suffix}. Use .npz or .csv")
    
    print(f"✓ Embeddings saved to {save_path}")


def load_embeddings(file_path: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load embeddings from file.
    
    Args:
        file_path: Path to embeddings file (.npz or .csv)
        
    Returns:
        If .npz: tuple of (embeddings, checklist_ids) if IDs exist, else just embeddings
        If .csv: numpy array of embeddings (without checklist IDs column if present)
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(file_path)
        embeddings = data['embeddings']
        if 'checklist_ids' in data:
            return embeddings, data['checklist_ids']
        return embeddings
        
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # If first column is 'checklist_id', exclude it
        if df.columns[0] == 'checklist_id':
            return df.iloc[:, 1:].values
        return df.values
        
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
