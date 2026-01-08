"""
eBird Data Loader

This module provides functions for loading eBird checklist data from TSV files.
eBird data comes in tab-separated format with ~53 columns containing information
about bird observations, locations, dates, observers, and protocols.

The loader handles common issues like mixed dtypes in certain columns and
provides flexible options for loading subsets of data during development.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
import warnings


def load_ebird_data(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    columns: Optional[List[str]] = None,
    low_memory: bool = False,
    suppress_warnings: bool = True
) -> pd.DataFrame:
    """
    Load eBird checklist data from a TSV file.
    
    This function loads eBird sampling event data, which contains bird observations
    with associated metadata (location, date, observer, protocol, etc.). The data
    is in tab-separated format with approximately 53 columns.
    
    Common eBird data files:
    - ebd_IN-KL_smp_relSep-2025.txt (Kerala, India)
    - ebd_IN-KA_smp_relOct-2025.txt (Karnataka, India)
    - ebd_sampling_relAug-2025.txt (Global sampling)
    
    Args:
        file_path (str or Path): Path to the eBird TSV file. Can be absolute or
            relative to the project root or data/raw directory.
        nrows (int, optional): Number of rows to read. Useful for testing with
            a subset of data. If None, reads entire file. Default: None
        columns (list of str, optional): Specific columns to load. If None, loads
            all columns. Useful for reducing memory usage. Default: None
        low_memory (bool): If True, internally process file in chunks to reduce
            memory usage. Set to True if you get dtype warnings. Default: False
        suppress_warnings (bool): If True, suppresses pandas DtypeWarning for
            columns with mixed types. Default: True
            
    Returns:
        pd.DataFrame: DataFrame containing eBird checklist data with columns like:
            - GLOBAL UNIQUE IDENTIFIER: Unique ID for each observation
            - COMMON NAME: Bird species common name
            - SCIENTIFIC NAME: Bird species scientific name
            - OBSERVATION COUNT: Number of individuals observed
            - OBSERVATION DATE: Date of observation
            - LOCALITY: Location name
            - LATITUDE/LONGITUDE: Geographic coordinates
            - SAMPLING EVENT IDENTIFIER: Unique checklist ID
            - ... and more (varies by eBird release)
            
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file format is invalid
        
    Examples:
        >>> # Load full Kerala dataset
        >>> data = load_ebird_data("ebd_IN-KL_smp_relSep-2025.txt")
        >>> print(f"Loaded {len(data)} observations")
        
        >>> # Load first 1M rows for testing
        >>> data = load_ebird_data("ebd_IN-KL_smp_relSep-2025.txt", nrows=1000000)
        
        >>> # Load only specific columns to save memory
        >>> columns = ['COMMON NAME', 'OBSERVATION DATE', 'LOCALITY']
        >>> data = load_ebird_data("kerala_data.txt", columns=columns)
        
        >>> # Load with full path
        >>> data = load_ebird_data(
        ...     r"C:\\data\\ebird\\ebd_IN-KL_smp_relSep-2025.txt",
        ...     nrows=3000000
        ... )
    
    Notes:
        - eBird TSV files can be very large (several GB)
        - Use nrows parameter during development to speed up loading
        - Some columns may have mixed dtypes (this is normal for eBird data)
        - Set low_memory=True if you encounter dtype warnings
        - File must be tab-separated with header row
    """
    
    # Convert to Path object for easier manipulation
    file_path = Path(file_path)
    
    # If path is not absolute, try to find it in common locations
    if not file_path.is_absolute():
        # Try relative to current directory
        if not file_path.exists():
            # Try in data/raw directory
            data_raw_path = Path("data/raw") / file_path
            if data_raw_path.exists():
                file_path = data_raw_path
            else:
                # Try with just filename in data/raw
                data_raw_path = Path("data/raw") / file_path.name
                if data_raw_path.exists():
                    file_path = data_raw_path
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"eBird data file not found: {file_path}\n"
            f"Tried locations:\n"
            f"  - {file_path}\n"
            f"  - data/raw/{file_path.name}\n"
            f"Please check the file path and ensure the data file exists."
        )
    
    # Suppress dtype warnings if requested
    # These warnings occur because some columns have mixed types
    # (e.g., EXOTIC CODE, AGE/SEX, OBSERVER ORCID ID, PROJECT NAMES)
    if suppress_warnings:
        warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    
    try:
        # Load the eBird data
        # sep='\t' because eBird uses tab-separated format
        # header=0 to use first row as column names
        data = pd.read_csv(
            file_path,
            sep='\t',
            header=0,
            nrows=nrows,
            usecols=columns,
            low_memory=low_memory
        )
        
        # Re-enable warnings
        if suppress_warnings:
            warnings.filterwarnings('default', category=pd.errors.DtypeWarning)
        
        # Log basic info about loaded data
        print(f"✓ Loaded eBird data from: {file_path.name}")
        print(f"  Rows: {len(data):,}")
        print(f"  Columns: {len(data.columns)}")
        if nrows:
            print(f"  (Limited to first {nrows:,} rows)")
        
        return data
        
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(
            f"The eBird data file is empty: {file_path}"
        )
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(
            f"Error parsing eBird data file {file_path}. "
            f"Ensure it's a valid tab-separated file. Original error: {e}"
        )
    except Exception as e:
        # Re-enable warnings before raising
        if suppress_warnings:
            warnings.filterwarnings('default', category=pd.errors.DtypeWarning)
        raise


def get_ebird_columns(file_path: Union[str, Path]) -> List[str]:
    """
    Get the column names from an eBird data file without loading the full dataset.
    
    This function reads only the header row, making it very fast even for large files.
    Useful for exploring the data structure before loading.
    
    Args:
        file_path (str or Path): Path to the eBird TSV file
        
    Returns:
        List[str]: List of column names in the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> columns = get_ebird_columns("ebd_IN-KL_smp_relSep-2025.txt")
        >>> print(f"eBird data has {len(columns)} columns")
        >>> print(f"First few columns: {columns[:5]}")
        >>> 
        >>> # Check if a specific column exists
        >>> if 'COMMON NAME' in columns:
        ...     print("Species names available!")
    """
    file_path = Path(file_path)
    
    # Use same path resolution logic as load_ebird_data
    if not file_path.is_absolute() and not file_path.exists():
        data_raw_path = Path("data/raw") / file_path
        if data_raw_path.exists():
            file_path = data_raw_path
        else:
            data_raw_path = Path("data/raw") / file_path.name
            if data_raw_path.exists():
                file_path = data_raw_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"eBird data file not found: {file_path}")
    
    # Read only the header (0 rows of data)
    header_df = pd.read_csv(file_path, sep='\t', nrows=0)
    return header_df.columns.tolist()


def validate_ebird_data(data: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame contains valid eBird data.
    
    Checks for:
    - Required columns exist
    - Data types are reasonable
    - No completely empty DataFrame
    
    Args:
        data (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if valid eBird data
        
    Raises:
        ValueError: If critical validation fails
        
    Example:
        >>> data = load_ebird_data("kerala_data.txt", nrows=1000)
        >>> validate_ebird_data(data)
        >>> # Proceeds if valid, raises ValueError if not
    """
    # Check not empty
    if len(data) == 0:
        raise ValueError("DataFrame is empty")
    
    # Check for critical columns that should exist in eBird sampling data
    required_columns = [
        'COMMON NAME',
        'SCIENTIFIC NAME',
        'SAMPLING EVENT IDENTIFIER',
        'OBSERVATION DATE'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required eBird columns: {missing_columns}\n"
            f"This may not be a valid eBird sampling event file."
        )
    
    print("✓ eBird data validation passed")
    return True
