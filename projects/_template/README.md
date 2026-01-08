# Project Template

This is a template for new analysis projects using the bird embeddings backbone.

## Project Structure

```
project_name/
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_exploration.ipynb
│   ├── 02_analysis.ipynb
│   └── 03_results.ipynb
├── data/
│   ├── raw/            # Project-specific raw data
│   └── processed/      # Processed data (embeddings + labels, etc.)
├── models/             # Project-specific trained models
├── results/            # Analysis outputs
│   ├── figures/        # Plots and visualizations
│   └── tables/         # CSV/Excel results
├── scripts/            # Optional production scripts
└── README.md           # Project documentation
```

## How to Use This Template

### 1. Copy Template to Start New Project

```bash
# From project root
cp -r projects/_template projects/your_project_name
cd projects/your_project_name
```

### 2. Update README.md

Edit this file to describe:
- What analysis you're doing
- What data you're using
- Key findings
- How to reproduce results

### 3. Start Working in Notebooks

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

## Common Workflow

### In your notebooks:

```python
# Setup paths to access the backbone
import sys
from pathlib import Path

# Add backbone to path
project_root = Path.cwd().parent.parent  # Goes to bird-embeddings/
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
```

### Use the backbone modules:

```python
# Load eBird data
from src.data import load_ebird_data, EBirdPreprocessor

data_path = project_root / 'data' / 'raw' / 'ebd_IN-KL_smp_relSep-2025.txt'
df = load_ebird_data(str(data_path))

# Preprocess
preprocessor = EBirdPreprocessor()
processed_df = preprocessor.fit_transform(df)

# Extract embeddings
from src.inference import EmbeddingExtractor

model_path = project_root / 'models' / 'vae_model_inference_ready.pth'
extractor = EmbeddingExtractor(str(model_path))
embeddings = extractor.extract_embeddings(processed_df)
```

### Save project-specific data:

```python
import numpy as np

# Save to project's data folder
np.savez(
    '../data/processed/my_embeddings.npz',
    embeddings=embeddings,
    labels=labels
)
```

## Tips

1. **Keep it self-contained**: All project files should be in this folder
2. **Use relative paths**: Reference backbone with `project_root`
3. **Save intermediate results**: Store processed data to avoid re-running expensive operations
4. **Document as you go**: Update README with findings
5. **Clean up**: Delete unused notebooks/data before archiving

## Shared Resources

These are shared across all projects (don't copy them):

- **Backbone code**: `src/` (data loading, models, training, inference)
- **Trained VAE models**: `models/vae_model_inference_ready.pth`
- **Raw eBird data**: `data/raw/`
- **Utility scripts**: `scripts/`

## Example Projects

See other projects in `projects/` folder for examples:
- `district_prediction/` - Predict district from embeddings
- `species_clustering/` - Cluster species by co-occurrence patterns

---

**Created**: [Date]  
**Author**: [Your Name]  
**Status**: Template
