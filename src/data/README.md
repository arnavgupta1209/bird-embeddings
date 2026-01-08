# Data Loading Module

This module handles loading and validating eBird checklist data from TSV files.

## Functions

### `load_ebird_data()`

Main function for loading eBird data files.

**Usage:**
```python
from src.data import load_ebird_data

# Load full dataset
data = load_ebird_data("ebd_IN-KL_smp_relSep-2025.txt")

# Load subset for testing
data = load_ebird_data("ebd_IN-KL_smp_relSep-2025.txt", nrows=100000)

# Load specific columns only
data = load_ebird_data(
    "kerala_data.txt",
    columns=['COMMON NAME', 'OBSERVATION DATE', 'LOCALITY']
)
```

**Parameters:**
- `file_path`: Path to eBird TSV file (auto-searches in `data/raw/`)
- `nrows`: Limit number of rows (useful for testing)
- `columns`: List of specific columns to load (saves memory)
- `low_memory`: Process in chunks (use if you get dtype warnings)
- `suppress_warnings`: Hide mixed dtype warnings (default: True)

**Returns:** `pd.DataFrame` with eBird observation data

---

### `get_ebird_columns()`

Get column names from an eBird file without loading the full data.

**Usage:**
```python
from src.data import get_ebird_columns

# See what columns are available
columns = get_ebird_columns("ebd_IN-KL_smp_relSep-2025.txt")
print(f"File has {len(columns)} columns")
print(columns[:10])  # First 10 columns

# Check if a column exists
if 'BREEDING CODE' in columns:
    print("Breeding data available!")
```

**Why use this:** Very fast (only reads header), useful for exploring data structure before loading GB-sized files.

---

### `validate_ebird_data()`

Validate that a DataFrame contains valid eBird data.

**Usage:**
```python
from src.data import load_ebird_data, validate_ebird_data

data = load_ebird_data("my_data.txt")
validate_ebird_data(data)  # Raises ValueError if invalid
```

**Checks:**
- DataFrame is not empty
- Required columns exist (COMMON NAME, SCIENTIFIC NAME, etc.)

---

## eBird Data Format

eBird sampling event data comes in TSV (tab-separated) format with ~53 columns:

**Key Columns:**
- `GLOBAL UNIQUE IDENTIFIER`: Unique observation ID
- `COMMON NAME`: Species common name (e.g., "House Sparrow")
- `SCIENTIFIC NAME`: Species scientific name (e.g., "Passer domesticus")
- `OBSERVATION COUNT`: Number of individuals seen
- `SAMPLING EVENT IDENTIFIER`: Unique checklist ID (groups observations)
- `OBSERVATION DATE`: When the observation was made
- `LOCALITY`: Location name
- `LATITUDE` / `LONGITUDE`: Geographic coordinates
- `PROTOCOL NAME`: Survey protocol used
- `DURATION MINUTES`: How long the checklist took
- `ALL SPECIES REPORTED`: Whether all species seen were reported

**Data Quirks:**
- Some columns have mixed types (strings and numbers) - this is normal
- Large files (multi-GB) - use `nrows` parameter for testing
- Tab-separated, not comma-separated

---

## Common Workflows

### Development/Testing
```python
# Load small subset for quick iteration
data = load_ebird_data("kerala_data.txt", nrows=10000)
```

### Production
```python
# Load full dataset
data = load_ebird_data("kerala_data.txt")

# Or load specific columns to save memory
important_cols = [
    'SAMPLING EVENT IDENTIFIER',
    'COMMON NAME',
    'OBSERVATION DATE',
    'LOCALITY',
    'LATITUDE',
    'LONGITUDE'
]
data = load_ebird_data("kerala_data.txt", columns=important_cols)
```

### Exploring New Dataset
```python
# First, see what columns it has
columns = get_ebird_columns("new_data.txt")
print(columns)

# Then load a sample
data = load_ebird_data("new_data.txt", nrows=1000)
print(data.info())
```

---

## File Organization

Place eBird data files in:
```
data/
└── raw/
    ├── ebd_IN-KL_smp_relSep-2025.txt    # Kerala
    ├── ebd_IN-KA_smp_relOct-2025.txt    # Karnataka
    └── ...
```

The loader will automatically find files in `data/raw/` even if you just provide the filename.

---

## Preprocessing

### `EBirdPreprocessor` Class

**NEW!** Scikit-learn style preprocessor for easier data preprocessing workflow.

**Usage**:
```python
from src.data import load_ebird_data
from src.data.preprocessor import EBirdPreprocessor

# Load data
data = load_ebird_data("kerala_data.txt")

# Create and fit preprocessor
preprocessor = EBirdPreprocessor(
    min_species_observations=10,  # Filter rare species
    min_checklist_species=5,      # Filter sparse checklists
    apply_quality_filters=True    # Apply eBird quality filters
)

# Transform data
processed_matrix = preprocessor.fit_transform(data)
print(f"Processed shape: {processed_matrix.shape}")
print(f"Species list: {len(preprocessor.species_list_)} species")

# Transform new data with same species list
new_data = load_ebird_data("new_data.txt")
new_matrix = preprocessor.transform(new_data)  # Uses fitted species list
```

**Benefits**:
- Consistent interface with scikit-learn
- Remembers fitted species list for transforming new data
- Automatically handles checklist ID column removal
- Ideal for inference on new data

---

### `create_species_matrix()` Function

**Purpose**: Convert observation-level data into species presence-absence matrix.

**What it does**:
- Input: Multiple rows per checklist (one row per species observed)
- Output: One row per checklist with binary columns for each species

**Usage**:
```python
from src.data import load_ebird_data, create_species_matrix

# Load data
data = load_ebird_data("kerala_data.txt", nrows=100000)

# Create basic species matrix (applies quality filters by default)
matrix, species_list = create_species_matrix(data)
print(f"Matrix: {matrix.shape[0]} checklists × {len(species_list)} species")

# Quality filters applied by default:
# - CATEGORY == 'species' (excludes hybrids, unidentified birds)
# - OBSERVATION TYPE in ['Traveling', 'Stationary']
# - ALL SPECIES REPORTED == 1 (complete checklists only)

# Skip quality filters if needed
matrix, species_list = create_species_matrix(data, apply_quality_filters=False)

# Filter rare species (must appear in ≥10 checklists)
matrix, species_list = create_species_matrix(
    data,
    min_species_observations=10
)

# Filter sparse checklists (must have ≥5 species)
matrix, species_list = create_species_matrix(
    data,
    min_checklist_species=5
)

# Include metadata columns
matrix, species_list = create_species_matrix(
    data,
    keep_checklist_metadata=True,
    metadata_cols=['COUNTY', 'OBSERVATION DATE', 'LATITUDE', 'LONGITUDE']
)
```

**Returns**: `(species_matrix, species_list)`
- `species_matrix`: DataFrame with checklists as rows, species as columns (0/1 values)
- `species_list`: Ordered list of species names

---

### `filter_columns()`

Reduce memory by keeping only necessary columns before preprocessing.

```python
from src.data import filter_columns

# Keep default columns
filtered = filter_columns(data)

# Keep custom columns
cols = ['COMMON NAME', 'SAMPLING EVENT IDENTIFIER']
filtered = filter_columns(data, columns_to_keep=cols)
```

---

### `get_species_statistics()`

Calculate prevalence of each species in the matrix.

```python
from src.data import get_species_statistics

matrix, species_list = create_species_matrix(data)
stats = get_species_statistics(matrix)

print(stats.head(10))  # Top 10 most common species
print(f"Rarest species: {stats.iloc[-1]['species_name']}")
```

---

## PyTorch Dataset

### `BirdChecklistDataset`

PyTorch Dataset that wraps species matrix for training.

**Usage**:
```python
from src.data import BirdChecklistDataset

# Create dataset from species matrix
dataset = BirdChecklistDataset(matrix, species_list)
print(f"Dataset size: {len(dataset)}")
print(f"Feature dim: {dataset.num_species}")

# Access single checklist
checklist = dataset[0]  # Shape: [num_species]
```

---

### `split_train_val()`

Split species matrix into train and validation datasets.

**Usage**:
```python
from src.data import split_train_val

# 80/20 split (default)
train_ds, val_ds = split_train_val(matrix, species_list, val_size=0.2)

# Specific number of validation examples
train_ds, val_ds = split_train_val(matrix, species_list, val_size=1000)

# Reproducible split
train_ds, val_ds = split_train_val(matrix, species_list, random_seed=42)

# Different random split each time
train_ds, val_ds = split_train_val(matrix, species_list, random_seed=None)
```

---

### `create_dataloaders()`

Create PyTorch DataLoaders for training.

**Usage**:
```python
from src.data import create_dataloaders

train_loader, val_loader = create_dataloaders(
    train_ds, val_ds,
    batch_size=128,
    num_workers=4,      # For parallel data loading
    pin_memory=True     # For GPU training
)

# Training loop
for batch in train_loader:
    # batch shape: [batch_size, num_species]
    pass
```

---

## Complete Pipeline Example

```python
from src.data import (
    load_ebird_data,
    create_species_matrix,
    split_train_val,
    create_dataloaders
)

# 1. Load raw data
data = load_ebird_data("kerala.txt", nrows=100000)

# 2. Create species matrix
matrix, species_list = create_species_matrix(
    data,
    min_species_observations=10  # Filter rare species
)

# 3. Split into train/val
train_ds, val_ds = split_train_val(
    matrix,
    species_list,
    val_size=0.2,
    random_seed=42
)

# 4. Create DataLoaders
train_loader, val_loader = create_dataloaders(
    train_ds, val_ds,
    batch_size=128
)

# 5. Ready for training!
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

---

## Next Steps

After loading and preprocessing data:
1. **Create species matrix** → `create_species_matrix()` ✅ **DONE**
2. **Create PyTorch datasets** → `split_train_val()` + `BirdChecklistDataset` ✅ **DONE**
2. Create PyTorch datasets → `src/data/dataset.py` (TODO)
3. Train VAE model → `src/training/` (TODO)

---

**Module Status:** ✅ Complete (Phase 3 - All Tasks Complete!)
