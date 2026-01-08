# Bird Embeddings with Variational Autoencoders

A PyTorch implementation of Variational Autoencoders (VAEs) for learning compressed embeddings from eBird checklist data. This project transforms high-dimensional bird species presence-absence data into meaningful low-dimensional representations that capture ecological patterns.

## 🎯 Project Overview

**What it does**: Converts eBird checklists (e.g., 500+ species) into compact embeddings (e.g., 16 dimensions) that preserve ecological information.

**Why it's useful**:
- **Dimensionality reduction**: Compress large species lists into manageable vectors
- **Similarity search**: Find similar checklists based on species composition
- **Visualization**: Plot checklists in 2D/3D space
- **Feature extraction**: Use embeddings for downstream ML tasks (classification, clustering, etc.)

**Important Note**: This repository contains **code and workflows only**. Large data files and trained models are excluded due to GitHub size limits. Follow the [Getting Started](#-getting-started) section to generate them.

## 📁 Project Structure

```
bird-embeddings/
├── src/                        # ✅ Core modules (reusable code)
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py          # Load eBird TSV files
│   │   ├── preprocessor.py    # Create species matrices
│   │   ├── dataset.py         # PyTorch Dataset classes
│   │   └── README.md
│   ├── models/                 # VAE model architecture
│   │   ├── vae.py             # VariationalAutoencoder class
│   │   └── README.md
│   ├── training/               # Training utilities
│   │   ├── trainer.py         # Training loop with noise augmentation
│   │   ├── utils.py           # Checkpointing, device management
│   │   └── README.md
│   └── inference/              # Embedding extraction
│       ├── extractor.py       # EmbeddingExtractor class
│       └── README.md
│
├── data/                       # ⚠️ NOT INCLUDED (too large)
│   ├── raw/                    # Place eBird data here
│   │   └── ebd_IN-KL_smp_relSep-2025.txt  # Download from eBird
│   └── processed/              # Generated embeddings (created by you)
│       └── kerala_embeddings.npz
│
├── models/                     # ⚠️ NOT INCLUDED (too large)
│   └── vae_kerala.pth         # Trained models (created by you)
│
├── checkpoints/                # Training checkpoints (created during training)
│
├── scripts/                    # ✅ Utility scripts
│   ├── train_new_model.py     # Train and save compatible models
│   ├── check_model_compatibility.py  # Verify model format
│   └── test_extractor_fix.py  # Quick inference test
│
├── notebooks/                  # ✅ Training notebooks
│   └── train_vae_kerala.ipynb # Example training workflow
│
├── test_notebooks/             # ✅ Test suite
│   ├── test_vae_module.ipynb
│   ├── test_data_pipeline.ipynb
│   ├── test_training.ipynb
│   └── test_inference.ipynb
│
├── projects/                   # ✅ Example downstream tasks
│   ├── district_prediction/   # Predict district from embeddings
│   ├── wetland_prediction/    # Classify wetland habitats
│   └── _template/             # Project template for new analyses
│
├── docs/                       # ✅ Documentation
│   ├── INFERENCE_FIX_SUMMARY.md
│   ├── TEST_INFERENCE_READY.md
│   ├── ORGANIZATION_SUMMARY.md
│   └── PHASE_2_SUMMARY.md
│
├── requirements.txt            # ✅ Python dependencies
├── .gitignore                  # ✅ Ignore large files and old notebooks
└── README.md                   # This file
```

**Legend:**
- ✅ Included in repository
- ⚠️ Not included (generate yourself following the workflow)


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Pandas, NumPy, scikit-learn
- Jupyter Notebook (for interactive analysis)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd bird-embeddings

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

**Required:** Download Kerala eBird data from [eBird](https://ebird.org/data/download)

1. Request eBird Basic Dataset (EBD) for **Kerala, India** (region code: `IN-KL`)
2. Download the sampling event file: `ebd_IN-KL_smp_relSep-2025.txt` (or latest)
3. Place the file in `data/raw/`:

```bash
data/
└── raw/
    └── ebd_IN-KL_smp_relSep-2025.txt  # Your eBird data file
```

**Note:** The actual data and trained models are NOT included in this repository due to size constraints. You need to generate them following the workflow below.

---

## 📋 Complete Workflow

### Step 1: Prepare Your Data

Load and explore the eBird data:

```python
from src.data import load_ebird_data, get_ebird_columns

# Check what columns are available
columns = get_ebird_columns("ebd_IN-KL_smp_relSep-2025.txt")
print(f"Dataset has {len(columns)} columns")

# Load full dataset (or use nrows=100000 for testing)
data = load_ebird_data("ebd_IN-KL_smp_relSep-2025.txt")
print(f"Loaded {len(data)} observations")
```

### Step 2: Create Species Matrix

Convert observation-level data to checklist-level species matrix:

```python
from src.data import create_species_matrix

# Create binary species presence-absence matrix
matrix, species_list = create_species_matrix(
    data,
    min_species_observations=30,  # Filter rare species
    min_checklist_species=5,      # Filter sparse checklists
    apply_quality_filters=True    # Use complete checklists only
)

print(f"Matrix shape: {matrix.shape[0]} checklists × {len(species_list)} species")
```

### Step 3: Train the VAE Model

Train a Variational Autoencoder to learn embeddings:

```python
from src.models import VariationalAutoencoder
from src.data import split_train_val, create_dataloaders
from src.training import train_variational_autoencoder, get_device, set_seed

# Set random seed for reproducibility
set_seed(42)

# Split data into train/validation
train_ds, val_ds = split_train_val(matrix, species_list, val_size=0.2)

# Create DataLoaders
train_loader, val_loader = create_dataloaders(
    train_ds, val_ds,
    batch_size=128,
    num_workers=4
)

# Create model
model = VariationalAutoencoder(
    input_dimension=len(species_list),
    latent_dimension=16,  # Embedding size
    hidden_dimension=512
)

# Train the model
device = get_device()  # Auto-detect CUDA/MPS/CPU
history = train_variational_autoencoder(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    noise_std=0.1,
    device=device,
    checkpoint_dir='checkpoints/my_vae',
    save_every=10
)
```

**Or use the training script:**

```bash
python scripts/train_new_model.py
```

### Step 4: Save Model for Inference

Save the trained model in a compatible format:

```python
from src.training import save_model_for_inference

save_model_for_inference(
    model=model,
    filepath='models/vae_kerala.pth',
    input_dim=len(species_list),
    latent_dim=16,
    hidden_dims=[512]
)
```

### Step 5: Extract Embeddings

Use the trained model to extract embeddings from checklists:

```python
from src.inference import EmbeddingExtractor

# Load the trained model
extractor = EmbeddingExtractor('models/vae_kerala.pth', device='cpu')

# Extract embeddings (deterministic mode)
embeddings = extractor.extract_embeddings(matrix, use_mean=True)
print(f"Embeddings shape: {embeddings.shape}")  # [num_checklists, 16]

# Save embeddings for downstream tasks
extractor.save_embeddings(
    embeddings,
    'data/processed/kerala_embeddings.npz',
    checklist_ids=matrix.index.tolist()
)
```

### Step 6: Use Embeddings for Downstream Tasks

See example projects in `projects/`:

- **District Prediction** (`projects/district_prediction/`): Predict Kerala district from species composition
- **Wetland Classification** (`projects/wetland_prediction/`): Identify wetland habitats using embeddings

---

## 🧪 Testing & Validation

### Quick Test

Verify everything is working:

```bash
# Test inference module
python scripts/test_extractor_fix.py

# Test data pipeline
jupyter notebook test_notebooks/test_data_pipeline.ipynb

# Test training
jupyter notebook test_notebooks/test_training.ipynb

# Test inference
jupyter notebook test_notebooks/test_inference.ipynb
```

### Check Model Compatibility

```bash
python scripts/check_model_compatibility.py models/vae_kerala.pth
```

## 📊 Data Format

**Input**: eBird checklist data in TSV format
- Columns include species observations, location, date, etc.
- Each row represents a single species observation in a checklist

**Preprocessed**: Binary/one-hot encoded species presence-absence matrix
- Rows = checklists
- Columns = species
- Values = 0 (absent) or 1 (present)

**Output**: Continuous embeddings
- Each checklist → 16-dimensional vector (configurable)

## 🧠 Model Architecture

```
Input: [batch_size, num_species] (e.g., [32, 500])
    ↓
Encoder: 3-layer MLP (500 → 512 → 512 → 512)
    ↓
Latent Parameters: μ and log(σ²) (each [batch_size, 16])
    ↓
Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
    ↓
Decoder: 3-layer MLP (16 → 512 → 512 → 512 → 500)
    ↓
Output: [batch_size, num_species] (reconstructed, sigmoid)
```

**Loss Function**:
```
Total Loss = BCE(reconstruction, input) + KL(N(μ, σ²) || N(0, 1))
```

- **Reconstruction Loss**: Binary cross-entropy between input and reconstruction
- **KL Divergence**: Regularizes latent space to follow standard normal distribution

## 📈 Current Status

### ✅ Complete Modules

**Phase 1-5: Core Infrastructure - COMPLETE**
- ✅ Project structure and organization
- ✅ VAE model architecture (`src/models/`)
- ✅ Data loading and preprocessing (`src/data/`)
- ✅ Training pipeline with noise augmentation (`src/training/`)
- ✅ Inference and embedding extraction (`src/inference/`)
- ✅ All test notebooks passing
- ✅ Utility scripts for training and testing
- ✅ Comprehensive documentation

**Example Projects - COMPLETE**
- ✅ District prediction (Random Forest + Neural Network)
- ✅ Wetland habitat classification (three approaches)

### 🎯 What You Can Do

With this repository, you can:

1. **Train VAE models** on your own eBird data
2. **Extract embeddings** from bird checklists
3. **Use embeddings** for downstream ML tasks:
   - Geographic prediction (district, region)
   - Habitat classification (wetland, forest, etc.)
   - Species distribution modeling
   - Checklist similarity search
   - Clustering analysis

### ⚠️ What's NOT Included

Due to GitHub file size limits:
- ❌ Raw eBird data files (download from eBird.org)
- ❌ Trained model checkpoints (generate using workflow)
- ❌ Processed embeddings (extract using inference module)
- ❌ Old/legacy notebooks (analysis.ipynb, main.ipynb)

### 🔄 Repository Philosophy

This repository provides **code and workflows**, not data or pre-trained models. 

**Why?** 
- eBird data files are multi-GB in size
- Models are specific to your dataset
- Following the workflow teaches you the full pipeline
- You maintain control over data quality and preprocessing choices

## 🔧 Development

### Test Notebooks
All test notebooks are passing:
- `test_notebooks/test_vae_module.ipynb` - Tests VAE model architecture
- `test_notebooks/test_data_pipeline.ipynb` - Tests data loading and preprocessing
- `test_notebooks/test_training.ipynb` - Tests training pipeline
- `test_notebooks/test_inference.ipynb` - Tests embedding extraction

### Training Notebooks
- `notebooks/train_vae_kerala.ipynb` - Example VAE training workflow

### Utility Scripts
Run from project root:
```bash
python scripts/train_new_model.py            # Train new VAE model
python scripts/check_model_compatibility.py   # Verify model format
python scripts/test_extractor_fix.py         # Quick inference test
```

### Model Compatibility

⚠️ **Important**: Always save models using `save_model_for_inference()` to ensure compatibility with the inference module.

**Correct way:**
```python
from src.training import save_model_for_inference

save_model_for_inference(
    model=vae_model,
    filepath='models/my_model.pth',
    input_dim=467,
    latent_dim=16,
    hidden_dims=[512]
)
```

**Wrong way (will not work with inference module):**
```python
torch.save(vae_model, 'model.pth')  # ❌ Don't do this
```

## 📝 Documentation

- **Main README** (this file): Project overview and quick start
- **Module READMEs**: Detailed docs in each `src/` subdirectory
  - `src/models/README.md` - VAE architecture details
  - `src/data/README.md` - Data pipeline documentation  
  - `src/training/README.md` - Training utilities
  - `src/inference/README.md` - Embedding extraction guide
- **Fix Documentation**: `docs/` folder
  - `INFERENCE_FIX_SUMMARY.md` - Technical details of the inference fix
  - `TEST_INFERENCE_READY.md` - How to use test_inference.ipynb
  - `PHASE_2_SUMMARY.md` - Phase 2 development summary
  - `ORGANIZATION_SUMMARY.md` - Project organization details

## ❓ FAQ & Troubleshooting

### Where is the data and models?

**Not included** due to GitHub size limits. Follow these steps:

1. Download eBird data from https://ebird.org/data/download
2. Run the complete workflow in [Getting Started](#-getting-started) to:
   - Create species matrices
   - Train your VAE model
   - Extract embeddings

### What eBird data file do I need?

Request the **eBird Basic Dataset (EBD)** for your region of interest. For Kerala, India:
- Region code: `IN-KL`
- File format: Sampling event data (TSV)
- Example: `ebd_IN-KL_smp_relSep-2025.txt`



### Model is too large / running out of memory?

Reduce model size:
```python
model = VariationalAutoencoder(
    input_dimension=len(species_list),
    latent_dimension=8,      # Reduce from 16
    hidden_dimension=256     # Reduce from 512
)
```

Or filter data more aggressively:
```python
matrix, species = create_species_matrix(
    data,
    min_species_observations=100,  # Increase from 30
    min_checklist_species=10       # Increase from 5
)
```

### How do I choose hyperparameters?

**Start with defaults**, then tune:
- `latent_dimension`: 8-32 (smaller = more compression, larger = more info retained)
- `hidden_dimension`: 256-512 (larger = more capacity)
- `noise_std`: 0.05-0.2 (higher = more regularization)
- `learning_rate`: 1e-4 to 1e-3

### Inference module not loading my model?

Make sure you saved the model correctly:
```python
from src.training import save_model_for_inference

save_model_for_inference(
    model=model,
    filepath='models/my_model.pth',
    input_dim=len(species_list),
    latent_dim=16,
    hidden_dims=[512]
)
```

**Don't use** `torch.save(model, ...)` directly!

## 📚 References

- **Kingma & Welling (2013)**: "Auto-Encoding Variational Bayes" - Original VAE paper
- **Doersch (2016)**: "Tutorial on Variational Autoencoders" - Excellent introduction
- **eBird**: https://ebird.org/ - Bird observation data source

---

## 🗂️ Example Projects

### District Prediction

Predict Kerala district from bird checklist embeddings using Random Forest classifier.

**Location:** `projects/district_prediction/`

**Goal:** Train a classifier to predict which of 14 Kerala districts a checklist comes from using only the 16-dimensional embedding.

**Key Results:** ~70-80% accuracy on district prediction

**Workflow:**
1. `01_exploration.ipynb` - Load data, extract embeddings, add district labels
2. `02_analysis.ipynb` - Train Random Forest classifier
3. `02b_neural_network.ipynb` - Train neural network classifier
4. `03_results.ipynb` - Visualize confusion matrix, feature importance

---

### Wetland Habitat Classification

Predict whether a checklist location is near wetland habitat using species composition embeddings.

**Location:** `projects/wetland_prediction/`

**Goal:** Determine if VAE embeddings capture habitat information using three labeling approaches:

1. **Bird Proportion Heuristic:** Label based on % of wetland-indicator species (94.6% accuracy)
2. **Hotspot Name Heuristic:** Label based on location name keywords (87.9% accuracy)
3. **Intersection Approach:** Combine both heuristics for stricter labels (97.0% accuracy)

**Key Insight:** High accuracy on hotspot-based labels (independent of species data) proves embeddings learned meaningful habitat associations.

**Workflow:**
1. `00_preprocessing_proportion_heuristic.ipynb` - Create bird-based labels
2. `00b_preprocessing_hotspot_heuristic.ipynb` - Create location-based labels
3. `01_exploration.ipynb` - Load embeddings and labels
4. `02_analysis.ipynb` - Train classifiers and compare approaches

---

**Last Updated**: January 8, 2026  
**Status**: ✅ Core modules complete, tested, and documented  
**Note**: Data and models not included - generate using provided workflows
