"""
Data loading and preprocessing module for eBird embeddings.

Exports:
    - load_ebird_data: Load eBird TSV files into pandas DataFrame
    - get_ebird_columns: Get column names from eBird file
    - validate_ebird_data: Validate eBird data structure
    - create_species_matrix: Convert observations to species presence-absence matrix
    - filter_columns: Filter eBird data to keep only relevant columns
    - get_species_statistics: Calculate species prevalence statistics
    - BirdChecklistDataset: PyTorch Dataset for checklist data
    - split_train_val: Split data into train and validation datasets
    - create_dataloaders: Create PyTorch DataLoaders
    - get_full_tensor: Get all data as a single tensor
"""

from .loader import load_ebird_data, get_ebird_columns, validate_ebird_data
from .preprocessor import create_species_matrix, filter_columns, get_species_statistics, EBirdPreprocessor
from .dataset import BirdChecklistDataset, split_train_val, create_dataloaders, get_full_tensor

__all__ = [
    'load_ebird_data',
    'get_ebird_columns',
    'validate_ebird_data',
    'create_species_matrix',
    'filter_columns',
    'get_species_statistics',
    'EBirdPreprocessor',
    'BirdChecklistDataset',
    'split_train_val',
    'create_dataloaders',
    'get_full_tensor'
]
