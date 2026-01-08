"""
eBird Data Preprocessor

This module provides functions for transforming raw eBird observation data
into species presence-absence matrices suitable for machine learning models.

The main transformation converts observation-level data (one row per species
per checklist) into checklist-level data (one row per checklist with binary
indicators for each species).
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union


def create_species_matrix(
    data: pd.DataFrame,
    checklist_id_col: str = 'SAMPLING EVENT IDENTIFIER',
    species_col: str = 'COMMON NAME',
    min_species_observations: Optional[int] = None,
    min_checklist_species: Optional[int] = None,
    keep_checklist_metadata: bool = False,
    metadata_cols: Optional[List[str]] = None,
    apply_quality_filters: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert eBird observation data to species presence-absence matrix.
    
    Transforms observation-level data (multiple rows per checklist, one per species)
    into checklist-level data (one row per checklist with binary columns for each species).
    
    Process:
    1. Apply quality filters (CATEGORY='species', etc.)
    2. Create binary presence indicator (1 = species present in checklist)
    3. Pivot data: checklists as rows, species as columns
    4. Fill missing values with 0 (species not observed)
    5. Optionally filter rare species and sparse checklists
    
    Args:
        data (pd.DataFrame): Raw eBird observation data with columns for
            checklist IDs and species names
        checklist_id_col (str): Name of column containing unique checklist IDs.
            Default: 'SAMPLING EVENT IDENTIFIER'
        species_col (str): Name of column containing species names.
            Default: 'COMMON NAME'
        min_species_observations (int, optional): Minimum number of checklists
            a species must appear in to be included. Filters out rare species.
            If None, all species are kept. Default: None
        min_checklist_species (int, optional): Minimum number of species a
            checklist must have to be included. Filters out sparse checklists.
            If None, all checklists are kept. Default: None
        keep_checklist_metadata (bool): If True, merges checklist metadata
            (location, date, etc.) back into the result. Default: False
        metadata_cols (list of str, optional): Specific metadata columns to keep
            if keep_checklist_metadata=True. If None, keeps common columns.
            Default: None
        apply_quality_filters (bool): If True, applies data quality filters:
            - CATEGORY == 'species' (excludes hybrids, unidentified birds)
            - OBSERVATION TYPE in ['Traveling', 'Stationary'] (standard protocols)
            - ALL SPECIES REPORTED == 1 (complete checklists only)
            Default: True
            
    Returns:
        tuple: (species_matrix, species_list)
            - species_matrix (pd.DataFrame): Binary matrix where:
                - Each row = one checklist
                - Each column = one species (1 if present, 0 if absent)
                - First column = checklist ID
                - Shape: [num_checklists, num_species + 1]
            - species_list (list of str): Ordered list of species names
                (column names excluding checklist ID)
                
    Raises:
        ValueError: If required columns are missing from input data
        
    Examples:
        >>> # Basic usage: just get species matrix
        >>> matrix, species = create_species_matrix(data)
        >>> print(f"Matrix shape: {matrix.shape}")
        >>> print(f"Number of species: {len(species)}")
        
        >>> # Filter rare species (must appear in at least 10 checklists)
        >>> matrix, species = create_species_matrix(
        ...     data,
        ...     min_species_observations=10
        ... )
        
        >>> # Skip quality filters (use all data)
        >>> matrix, species = create_species_matrix(
        ...     data,
        ...     apply_quality_filters=False
        ... )
        
        >>> # Filter sparse checklists and keep metadata
        >>> matrix, species = create_species_matrix(
        ...     data,
        ...     min_checklist_species=5,
        ...     keep_checklist_metadata=True,
        ...     metadata_cols=['COUNTY', 'OBSERVATION DATE', 'LATITUDE', 'LONGITUDE']
        ... )
        
    Notes:
        - Input data should have one row per species per checklist
        - Quality filters ensure high-quality, complete checklists
        - Output matrix is sparse (mostly 0s) but stored as dense DataFrame
        - For large datasets, consider filtering rare species to reduce size
        - Checklist ID column will be first column in output matrix
    """
    
    # Validate input columns exist
    if checklist_id_col not in data.columns:
        raise ValueError(
            f"Checklist ID column '{checklist_id_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )
    if species_col not in data.columns:
        raise ValueError(
            f"Species column '{species_col}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )
    
    print(f"Input data: {len(data):,} observations")
    print(f"  Unique checklists: {data[checklist_id_col].nunique():,}")
    print(f"  Unique species: {data[species_col].nunique():,}")
    
    # Apply quality filters if requested
    if apply_quality_filters:
        print("\nApplying quality filters...")
        data_filtered = data.copy()
        
        initial_count = len(data_filtered)
        
        # Filter 1: CATEGORY == 'species'
        # Excludes hybrids, slashes (uncertain IDs), spuhs, etc.
        if 'CATEGORY' in data_filtered.columns:
            data_filtered = data_filtered[data_filtered['CATEGORY'] == 'species']
            print(f"  After CATEGORY='species': {len(data_filtered):,} observations "
                  f"({initial_count - len(data_filtered):,} removed)")
        else:
            print("  Warning: 'CATEGORY' column not found, skipping filter")
        
        # Filter 2: OBSERVATION TYPE in ['Traveling', 'Stationary']
        # Excludes incidental, historical, and other non-standard observations
        if 'OBSERVATION TYPE' in data_filtered.columns:
            pre_filter = len(data_filtered)
            data_filtered = data_filtered[
                data_filtered['OBSERVATION TYPE'].isin(['Traveling', 'Stationary'])
            ]
            print(f"  After OBSERVATION TYPE filter: {len(data_filtered):,} observations "
                  f"({pre_filter - len(data_filtered):,} removed)")
        else:
            print("  Warning: 'OBSERVATION TYPE' column not found, skipping filter")
        
        # Filter 3: ALL SPECIES REPORTED == 1
        # Only keep complete checklists where observer reported all species seen
        if 'ALL SPECIES REPORTED' in data_filtered.columns:
            pre_filter = len(data_filtered)
            data_filtered = data_filtered[data_filtered['ALL SPECIES REPORTED'] == 1]
            print(f"  After ALL SPECIES REPORTED=1: {len(data_filtered):,} observations "
                  f"({pre_filter - len(data_filtered):,} removed)")
        else:
            print("  Warning: 'ALL SPECIES REPORTED' column not found, skipping filter")
        
        print(f"\n✓ Quality filters applied: {initial_count:,} → {len(data_filtered):,} observations")
        print(f"  Removed: {initial_count - len(data_filtered):,} ({(initial_count - len(data_filtered))/initial_count*100:.1f}%)")
        
        data = data_filtered
        
        print(f"\nFiltered data:")
        print(f"  Unique checklists: {data[checklist_id_col].nunique():,}")
        print(f"  Unique species: {data[species_col].nunique():,}")
    
    # Create a binary presence indicator column
    # Every observation row means that species was present in that checklist
    data_copy = data.copy()
    data_copy['presence'] = 1
    
    # Pivot table to create species matrix
    # Rows = checklists (indexed by checklist ID)
    # Columns = species (from species_col)
    # Values = presence (1 if present, 0 if absent via fill_value)
    # observed=True: only include species actually observed (faster for categorical data)
    print("\nPivoting data to create species matrix...")
    species_matrix = data_copy.pivot_table(
        index=checklist_id_col,
        columns=species_col,
        values='presence',
        fill_value=0,  # Species not observed = 0
        observed=True,
        aggfunc='max'  # If species appears multiple times in checklist, just mark as present
    ).reset_index()
    
    print(f"✓ Initial matrix: {species_matrix.shape[0]:,} checklists × {species_matrix.shape[1]-1:,} species")
    
    # Filter rare species if requested
    if min_species_observations is not None:
        print(f"\nFiltering species with < {min_species_observations} observations...")
        
        # Calculate how many checklists each species appears in
        # (sum of each column, excluding checklist ID column)
        species_counts = species_matrix.iloc[:, 1:].sum(axis=0)
        
        # Keep species that appear in at least min_species_observations checklists
        species_to_keep = species_counts[species_counts >= min_species_observations].index.tolist()
        
        # Select checklist ID + kept species columns
        species_matrix = species_matrix[[checklist_id_col] + species_to_keep]
        
        print(f"✓ Removed {len(species_counts) - len(species_to_keep):,} rare species")
        print(f"  Kept {len(species_to_keep):,} species")
    
    # Filter sparse checklists if requested
    if min_checklist_species is not None:
        print(f"\nFiltering checklists with < {min_checklist_species} species...")
        
        # Calculate how many species each checklist has
        # (sum of each row, excluding checklist ID column)
        checklist_species_counts = species_matrix.iloc[:, 1:].sum(axis=1)
        
        # Keep checklists that have at least min_checklist_species species
        checklists_to_keep = checklist_species_counts >= min_checklist_species
        species_matrix = species_matrix[checklists_to_keep]
        
        removed_count = (~checklists_to_keep).sum()
        print(f"✓ Removed {removed_count:,} sparse checklists")
        print(f"  Kept {len(species_matrix):,} checklists")
    
    # Add checklist metadata if requested
    if keep_checklist_metadata:
        print("\nAdding checklist metadata...")
        
        # Define default metadata columns if none specified
        if metadata_cols is None:
            metadata_cols = [
                'COUNTY',
                'OBSERVATION DATE',
                'LATITUDE',
                'LONGITUDE',
                'DURATION MINUTES',
                'EFFORT DISTANCE KM',
                'NUMBER OBSERVERS',
                'ALL SPECIES REPORTED'
            ]
        
        # Only keep metadata columns that exist in the data
        available_metadata = [col for col in metadata_cols if col in data.columns]
        if not available_metadata:
            print("  Warning: No requested metadata columns found in data")
        else:
            # Get unique checklist metadata (drop duplicates by checklist ID)
            metadata = data[[checklist_id_col] + available_metadata].drop_duplicates(
                subset=[checklist_id_col]
            )
            
            # Merge metadata into species matrix
            species_matrix = species_matrix.merge(
                metadata,
                on=checklist_id_col,
                how='left'
            )
            print(f"✓ Added {len(available_metadata)} metadata columns: {available_metadata}")
    
    # Extract species list (all columns except checklist ID and metadata)
    species_columns = [col for col in species_matrix.columns 
                      if col != checklist_id_col 
                      and (not keep_checklist_metadata or col not in metadata_cols)]
    
    print(f"\n✓ Final matrix: {species_matrix.shape[0]:,} checklists × {len(species_columns):,} species")
    
    return species_matrix, species_columns


def filter_columns(
    data: pd.DataFrame,
    columns_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter eBird data to keep only relevant columns.
    
    Reduces memory usage by dropping unnecessary columns before preprocessing.
    
    Args:
        data (pd.DataFrame): Raw eBird data with all 53 columns
        columns_to_keep (list of str, optional): Specific columns to keep.
            If None, keeps a default set of useful columns.
            
    Returns:
        pd.DataFrame: Filtered data with only specified columns
        
    Example:
        >>> # Keep only essential columns
        >>> filtered = filter_columns(data)
        >>> print(f"Reduced from {data.shape[1]} to {filtered.shape[1]} columns")
        
        >>> # Keep custom columns
        >>> cols = ['COMMON NAME', 'SAMPLING EVENT IDENTIFIER', 'OBSERVATION DATE']
        >>> filtered = filter_columns(data, columns_to_keep=cols)
    """
    
    if columns_to_keep is None:
        # Default columns for creating species matrix + basic metadata
        columns_to_keep = [
            'CATEGORY',
            'COMMON NAME',
            'SCIENTIFIC NAME',
            'OBSERVATION COUNT',
            'COUNTRY',
            'COUNTRY CODE',
            'STATE',
            'STATE CODE',
            'COUNTY',
            'COUNTY CODE',
            'LATITUDE',
            'LONGITUDE',
            'OBSERVATION DATE',
            'TIME OBSERVATIONS STARTED',
            'SAMPLING EVENT IDENTIFIER',
            'OBSERVATION TYPE',
            'DURATION MINUTES',
            'EFFORT DISTANCE KM',
            'NUMBER OBSERVERS',
            'ALL SPECIES REPORTED',
            'GROUP IDENTIFIER'
        ]
    
    # Only keep columns that exist in the data
    available_cols = [col for col in columns_to_keep if col in data.columns]
    missing_cols = [col for col in columns_to_keep if col not in data.columns]
    
    if missing_cols:
        print(f"Warning: {len(missing_cols)} requested columns not found: {missing_cols[:5]}...")
    
    filtered_data = data[available_cols].copy()
    
    print(f"✓ Filtered columns: {len(data.columns)} -> {len(filtered_data.columns)}")
    print(f"  Memory usage reduced: {data.memory_usage(deep=True).sum() / 1e9:.2f} GB "
          f"-> {filtered_data.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    return filtered_data


def get_species_statistics(species_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics about species prevalence in the dataset.
    
    Args:
        species_matrix (pd.DataFrame): Species presence-absence matrix
            (output from create_species_matrix)
            
    Returns:
        pd.DataFrame: Statistics for each species with columns:
            - species_name: Species common name
            - num_checklists: Number of checklists species appears in
            - prevalence: Proportion of checklists with this species (0-1)
            
    Example:
        >>> matrix, species = create_species_matrix(data)
        >>> stats = get_species_statistics(matrix)
        >>> print(stats.head(10))
        >>> print(f"Most common species: {stats.iloc[0]['species_name']}")
    """
    
    # Exclude first column (checklist ID) and any metadata columns
    # Assume species columns are numeric (0/1)
    species_cols = species_matrix.select_dtypes(include=[np.number]).columns
    
    stats = []
    num_checklists = len(species_matrix)
    
    for species in species_cols:
        count = species_matrix[species].sum()
        prevalence = count / num_checklists
        stats.append({
            'species_name': species,
            'num_checklists': int(count),
            'prevalence': prevalence
        })
    
    stats_df = pd.DataFrame(stats)
    # Sort by prevalence (most common first)
    stats_df = stats_df.sort_values('prevalence', ascending=False).reset_index(drop=True)
    
    return stats_df


class EBirdPreprocessor:
    """
    Scikit-learn style preprocessor for eBird data.
    
    Wraps the preprocessing functions into a class with fit/transform interface.
    """
    
    def __init__(
        self,
        min_species_observations: Optional[int] = None,
        min_checklist_species: Optional[int] = None,
        apply_quality_filters: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            min_species_observations: Minimum observations required to keep a species
            min_checklist_species: Minimum species required to keep a checklist
            apply_quality_filters: Whether to apply quality filters
        """
        self.min_species_observations = min_species_observations
        self.min_checklist_species = min_checklist_species
        self.apply_quality_filters = apply_quality_filters
        self.species_list_ = None
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data in one step.
        
        Args:
            data: Raw eBird observation data
            
        Returns:
            Species presence-absence matrix
        """
        matrix, species_list = create_species_matrix(
            data,
            min_species_observations=self.min_species_observations,
            min_checklist_species=self.min_checklist_species,
            apply_quality_filters=self.apply_quality_filters
        )
        
        self.species_list_ = species_list
        
        # Drop checklist ID column for model input
        if 'SAMPLING EVENT IDENTIFIER' in matrix.columns:
            matrix = matrix.drop(columns=['SAMPLING EVENT IDENTIFIER'])
        
        return matrix
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted species list.
        
        Args:
            data: Raw eBird observation data
            
        Returns:
            Species presence-absence matrix
        """
        if self.species_list_ is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
        
        # Create matrix without filtering
        matrix, _ = create_species_matrix(
            data,
            apply_quality_filters=False
        )
        
        # Ensure same species columns
        for species in self.species_list_:
            if species not in matrix.columns:
                matrix[species] = 0
        
        # Keep only species from training
        keep_cols = ['SAMPLING EVENT IDENTIFIER'] + self.species_list_
        matrix = matrix[[col for col in keep_cols if col in matrix.columns]]
        
        # Drop checklist ID
        if 'SAMPLING EVENT IDENTIFIER' in matrix.columns:
            matrix = matrix.drop(columns=['SAMPLING EVENT IDENTIFIER'])
        
        return matrix
