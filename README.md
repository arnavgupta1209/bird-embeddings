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

1. Clone repository: `git clone <repository-url>`
2. Install dependencies: `pip install -r requirements.txt`

### Data Setup

**Required:** Download Kerala eBird data from [eBird](https://ebird.org/data/download)

1. Request eBird Basic Dataset (EBD) for **Kerala, India** (region code: `IN-KL`)
2. Download the sampling event file: `ebd_IN-KL_smp_relSep-2025.txt` (or latest)
3. Place the file in `data/raw/ebd_IN-KL_smp_relSep-2025.txt`

**Note:** The actual data and trained models are NOT included in this repository due to size constraints. You need to generate them following the workflow below.

---

## 📋 Complete Workflow

### Step 1: Train the VAE Model

**Option A - Use training script (recommended):**
```bash
python scripts/train_new_model.py
```

**Option B - Use training notebook:**
- Open and run `notebooks/train_vae_kerala.ipynb`
- Follow the cells to load data, create species matrix, and train model
- Model will be saved to `models/` directory

**What this does:**
- Loads eBird data from `data/raw/`
- Creates species presence-absence matrix
- Trains VAE model (50 epochs, ~30 min on GPU)
- Saves trained model for inference

### Step 2: Extract Embeddings

**Run inference notebook:**
- Open `test_notebooks/test_inference.ipynb`
- Load your trained model
- Extract embeddings from checklists
- Save embeddings to `data/processed/`

### Step 3: Use Embeddings for Downstream Tasks

**Example projects in `projects/` folder:**

- **District Prediction:** `projects/district_prediction/`
  - Run notebooks 01 → 02 → 03 in order
  - Predicts Kerala district from embeddings
  
- **Wetland Classification:** `projects/wetland_prediction/`
  - Run preprocessing notebooks (00, 00b)
  - Then run 01 → 02 for analysis
  - Classifies wetland vs non-wetland habitats

---

## 🧪 Testing & Validation

### Quick Test

Verify everything is working:

```bash
python scripts/test_extractor_fix.py
```

### Run Test Notebooks

Test individual components:
- `test_notebooks/test_data_pipeline.ipynb` - Data loading
- `test_notebooks/test_training.ipynb` - Training pipeline
- `test_notebooks/test_inference.ipynb` - Embedding extraction
- `test_notebooks/test_vae_module.ipynb` - VAE model

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
- `python scripts/train_new_model.py` - Train new VAE model
- `python scripts/check_model_compatibility.py` - Verify model format
- `python scripts/test_extractor_fix.py` - Quick inference test

### Model Compatibility

⚠️ **Important**: Always save models using `save_model_for_inference()` to ensure compatibility with the inference module. See `src/training/README.md` for details.

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

Options to reduce memory usage:
- Reduce `latent_dimension` from 16 to 8
- Reduce `hidden_dimension` from 512 to 256
- Increase `min_species_observations` to filter more species
- Use smaller batch size

See module READMEs in `src/` for detailed parameter tuning.

### How do I choose hyperparameters?

**Start with defaults**, then tune:
- `latent_dimension`: 8-32 (smaller = more compression)
- `hidden_dimension`: 256-512 (larger = more capacity)
- `noise_std`: 0.05-0.2 (higher = more regularization)
- `learning_rate`: 1e-4 to 1e-3

See `src/training/README.md` for comprehensive hyperparameter guide.

### Inference module not loading my model?

Make sure you saved the model using `save_model_for_inference()` from `src.training`. Don't use `torch.save(model, ...)` directly. See `src/training/README.md` for proper model saving.

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
