"""
PyTorch Dataset for eBird Checklist Embeddings

This module provides PyTorch Dataset classes for working with bird checklist
species presence-absence data in machine learning pipelines.
"""

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union


class BirdChecklistDataset(Dataset):
    """
    PyTorch Dataset for bird checklist species presence-absence data.
    
    Converts species matrix (pandas DataFrame) into PyTorch tensors suitable
    for training neural networks like VAEs.
    
    Args:
        species_matrix (pd.DataFrame): Species presence-absence matrix from
            create_species_matrix(). Shape: [num_checklists, num_species + 1]
            First column should be checklist ID, rest are species (0/1 values)
        species_columns (list of str): List of species column names to use
            as features. If None, uses all numeric columns.
        
    Example:
        >>> from src.data import load_ebird_data, create_species_matrix
        >>> from src.data import BirdChecklistDataset
        >>> 
        >>> # Load and preprocess data
        >>> data = load_ebird_data("kerala.txt", nrows=100000)
        >>> matrix, species = create_species_matrix(data)
        >>> 
        >>> # Create dataset
        >>> dataset = BirdChecklistDataset(matrix, species)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> print(f"Feature dimension: {dataset[0].shape}")
        >>> 
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=128, shuffle=True)
    """
    
    def __init__(
        self,
        species_matrix: pd.DataFrame,
        species_columns: Optional[list] = None
    ):
        super().__init__()
        
        # If species columns not provided, use all numeric columns
        # (assumes first column is checklist ID, rest are species)
        if species_columns is None:
            # Get all numeric columns (species are 0/1 floats)
            numeric_cols = species_matrix.select_dtypes(include=[np.number]).columns.tolist()
            species_columns = numeric_cols
        
        # Extract species data and convert to tensor
        # Shape: [num_checklists, num_species]
        species_data = species_matrix[species_columns].values
        self.data = torch.tensor(species_data, dtype=torch.float32)
        
        # Store metadata
        self.num_checklists = len(self.data)
        self.num_species = self.data.shape[1]
        self.species_columns = species_columns
    
    def __len__(self) -> int:
        """Return the number of checklists in the dataset."""
        return self.num_checklists
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single checklist as a tensor.
        
        Args:
            idx (int): Index of the checklist to retrieve
            
        Returns:
            torch.Tensor: Binary species presence-absence vector
                Shape: [num_species]
        """
        return self.data[idx]
    
    def get_batch(self, indices: list) -> torch.Tensor:
        """
        Get multiple checklists at once.
        
        Args:
            indices (list of int): Indices of checklists to retrieve
            
        Returns:
            torch.Tensor: Batch of checklist vectors
                Shape: [batch_size, num_species]
        """
        return self.data[indices]


def split_train_val(
    species_matrix: pd.DataFrame,
    species_columns: Optional[list] = None,
    val_size: Union[float, int] = 0.2,
    random_seed: Optional[int] = 42,
    shuffle: bool = True
) -> Tuple[BirdChecklistDataset, BirdChecklistDataset]:
    """
    Split species matrix into train and validation datasets.
    
    Args:
        species_matrix (pd.DataFrame): Species presence-absence matrix
        species_columns (list of str, optional): Species column names.
            If None, uses all numeric columns.
        val_size (float or int): If float (0-1), proportion of data for validation.
            If int, absolute number of validation examples. Default: 0.2 (20%)
        random_seed (int, optional): Random seed for reproducibility.
            If None, split is not reproducible. Default: 42
        shuffle (bool): Whether to shuffle data before splitting. Default: True
        
    Returns:
        tuple: (train_dataset, val_dataset)
            - train_dataset: BirdChecklistDataset for training
            - val_dataset: BirdChecklistDataset for validation
            
    Example:
        >>> # 80/20 train/val split
        >>> train_ds, val_ds = split_train_val(matrix, species, val_size=0.2)
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        >>> 
        >>> # Specific number of validation examples
        >>> train_ds, val_ds = split_train_val(matrix, species, val_size=1000)
        >>> 
        >>> # Deterministic split (always same)
        >>> train_ds, val_ds = split_train_val(matrix, species, random_seed=42)
        >>> 
        >>> # Non-reproducible split
        >>> train_ds, val_ds = split_train_val(matrix, species, random_seed=None)
    """
    
    num_examples = len(species_matrix)
    
    # Convert val_size to number of examples
    if isinstance(val_size, float):
        if not 0 < val_size < 1:
            raise ValueError(f"val_size as float must be between 0 and 1, got {val_size}")
        val_count = int(val_size * num_examples)
    else:
        val_count = val_size
        if not 0 < val_count < num_examples:
            raise ValueError(
                f"val_size as int must be between 0 and {num_examples}, got {val_count}"
            )
    
    train_count = num_examples - val_count
    
    # Create indices for splitting
    if shuffle:
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            generator = torch.Generator().manual_seed(random_seed)
            indices = torch.randperm(num_examples, generator=generator)
        else:
            indices = torch.randperm(num_examples)
    else:
        indices = torch.arange(num_examples)
    
    # Split indices
    train_indices = indices[:train_count]
    val_indices = indices[train_count:]
    
    # Create train and validation DataFrames
    train_matrix = species_matrix.iloc[train_indices.numpy()].reset_index(drop=True)
    val_matrix = species_matrix.iloc[val_indices.numpy()].reset_index(drop=True)
    
    # Create datasets
    train_dataset = BirdChecklistDataset(train_matrix, species_columns)
    val_dataset = BirdChecklistDataset(val_matrix, species_columns)
    
    print(f"✓ Split data: {len(train_dataset):,} train, {len(val_dataset):,} val")
    print(f"  Train/Val ratio: {len(train_dataset)/len(val_dataset):.2f}")
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: BirdChecklistDataset,
    val_dataset: BirdChecklistDataset,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train and validation datasets.
    
    Convenience function to create DataLoaders with sensible defaults
    for training VAE models.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Number of examples per batch. Default: 128
        num_workers (int): Number of subprocesses for data loading.
            0 means data will be loaded in main process. Default: 0
        pin_memory (bool): If True, DataLoader will copy tensors into
            CUDA pinned memory before returning them. Useful for GPU training.
            Default: False
            
    Returns:
        tuple: (train_loader, val_loader)
            - train_loader: DataLoader for training (shuffled)
            - val_loader: DataLoader for validation (not shuffled)
            
    Example:
        >>> train_ds, val_ds = split_train_val(matrix, species)
        >>> train_loader, val_loader = create_dataloaders(
        ...     train_ds, val_ds,
        ...     batch_size=128,
        ...     num_workers=4,
        ...     pin_memory=True  # For GPU training
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in train_loader:
        ...     # batch shape: [batch_size, num_species]
        ...     pass
    """
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"✓ Created DataLoaders:")
    print(f"  Train: {len(train_loader)} batches of size {batch_size}")
    print(f"  Val: {len(val_loader)} batches of size {batch_size}")
    
    return train_loader, val_loader


def get_full_tensor(dataset: BirdChecklistDataset) -> torch.Tensor:
    """
    Get all data from a dataset as a single tensor.
    
    Useful for inference where you want to process all data at once
    without batching.
    
    Args:
        dataset: BirdChecklistDataset instance
        
    Returns:
        torch.Tensor: All checklists stacked into single tensor
            Shape: [num_checklists, num_species]
            
    Example:
        >>> dataset = BirdChecklistDataset(matrix, species)
        >>> all_data = get_full_tensor(dataset)
        >>> print(all_data.shape)  # [num_checklists, num_species]
    """
    return dataset.data
