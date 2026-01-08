# District Prediction from Bird Checklists

Predict Kerala district from bird checklist embeddings using a Random Forest classifier.

## Overview

This project uses VAE-generated embeddings from bird checklists to predict which district in Kerala the observation was made. The embeddings capture species composition patterns that vary by location.

**Goal**: Train a classifier to predict district from a 16-dimensional embedding vector.

## Project Structure

```
district_prediction/
├── notebooks/           # Analysis notebooks
│   ├── 01_exploration.ipynb    # Load data, extract embeddings, add district labels
│   ├── 02_analysis.ipynb       # Train Random Forest classifier
│   └── 03_results.ipynb        # Visualize confusion matrix, feature importance
├── data/
│   ├── raw/            # (empty - use shared data)
│   └── processed/      # embeddings_with_districts.npz
├── models/             # district_classifier.pkl
└── results/            # Confusion matrices, metrics
    ├── figures/
    └── tables/
```

## Workflow

### 1. Data Preparation (01_exploration.ipynb)
- Load Kerala eBird data
- Extract district from locality names
- Generate embeddings using shared VAE model
- Save embeddings + district labels

### 2. Model Training (02_analysis.ipynb)
- Train Random Forest classifier
- Evaluate on test set
- Save trained model

### 3. Results (03_results.ipynb)
- Confusion matrix visualization
- Per-district accuracy
- Feature importance analysis

## Kerala Districts

The model predicts one of 14 districts:
1. Thiruvananthapuram
2. Kollam
3. Pathanamthitta
4. Alappuzha
5. Kottayam
6. Idukki
7. Ernakulam
8. Thrissur
9. Palakkad
10. Malappuram
11. Kozhikode
12. Wayanad
13. Kannur
14. Kasaragod

## How to Run

```bash
# Start from project folder
cd projects/district_prediction

# Run notebooks in order
jupyter notebook notebooks/01_exploration.ipynb
jupyter notebook notebooks/02_analysis.ipynb
jupyter notebook notebooks/03_results.ipynb
```

## Expected Results

- **Input**: 16-dimensional embedding from checklist
- **Output**: Predicted district + confidence scores
- **Expected Accuracy**: ~70-80% (depends on data quality and species coverage per district)

## Dependencies

Uses the backbone modules:
- `src.data` - Data loading and preprocessing
- `src.inference` - Embedding extraction
- Standard ML libraries: scikit-learn, pandas, numpy

---

**Created**: December 6, 2025  
**Author**: Arnav  
**Status**: In Progress
