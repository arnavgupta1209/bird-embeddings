# Inference Module

This module provides functionality for extracting embeddings from eBird checklist data using a trained VAE model.

## ⚠️ Important: Model Requirements

**Models must be saved using `save_model_for_inference()` from the training module.**

Legacy models saved directly with `torch.save(model, path)` from notebooks are **NOT compatible** with this inference module. 

To create a compatible model:
```python
from src.training import save_model_for_inference

save_model_for_inference(
    model=model,
    filepath='model.pth',
    input_dim=467,
    latent_dim=16,
    hidden_dims=[512]
)
```

Or use the provided training script:
```bash
python train_new_model.py
```

## Overview

The inference module allows you to:
- Load a trained VAE model
- Extract latent embeddings from bird checklist data
- Choose between deterministic (mean-based) or stochastic (sampled) embeddings
- Generate reconstructions for quality analysis
- Save and load embeddings in multiple formats

## Quick Start

```python
from src.inference import EmbeddingExtractor

# Initialize extractor with trained model
extractor = EmbeddingExtractor(
    model_path='vae_bird_embeddings_kerala_good.pth',
    device='cpu'  # or 'cuda' for GPU
)

# Extract embeddings from preprocessed data
embeddings = extractor.extract_embeddings(
    data=processed_df,
    use_mean=True,  # Deterministic mode
    batch_size=256
)

print(f"Extracted embeddings shape: {embeddings.shape}")
```

## Classes and Functions

### `EmbeddingExtractor`

Main class for extracting embeddings from trained VAE models.

#### Initialization

```python
extractor = EmbeddingExtractor(
    model_path: str,      # Path to .pth checkpoint file
    device: Optional[str] # 'cpu', 'cuda', or None for auto-detect
)
```

#### Methods

**`extract_embeddings(data, use_mean=True, batch_size=256)`**

Extract embeddings from checklist data.

- **Parameters:**
  - `data`: Input data (pandas DataFrame, numpy array, or torch Tensor)
  - `use_mean`: If True, uses mean (μ) of latent distribution (deterministic). If False, samples from distribution (stochastic)
  - `batch_size`: Number of samples to process at once

- **Returns:** Numpy array of embeddings, shape `(n_samples, latent_dim)`

- **Example:**
  ```python
  # Deterministic embeddings (recommended for most use cases)
  embeddings_mean = extractor.extract_embeddings(data, use_mean=True)
  
  # Stochastic embeddings (includes sampling noise)
  embeddings_sampled = extractor.extract_embeddings(data, use_mean=False)
  ```

**`extract_with_reconstruction(data, use_mean=True, batch_size=256)`**

Extract both embeddings and reconstructions.

- **Returns:** Tuple of `(embeddings, reconstructions)`, both as numpy arrays

- **Example:**
  ```python
  embeddings, reconstructions = extractor.extract_with_reconstruction(data)
  
  # Compute reconstruction error
  import numpy as np
  recon_error = np.mean((data.values - reconstructions) ** 2)
  print(f"Mean reconstruction error: {recon_error:.6f}")
  ```

### Convenience Functions

**`load_and_extract(model_path, data, use_mean=True, batch_size=256, device=None)`**

One-liner to load model and extract embeddings.

```python
from src.inference import load_and_extract

embeddings = load_and_extract(
    model_path='vae_model.pth',
    data=processed_df,
    use_mean=True
)
```

**`save_embeddings(embeddings, save_path, checklist_ids=None, metadata=None)`**

Save embeddings to file (supports .npz and .csv formats).

```python
from src.inference import save_embeddings
import pandas as pd

# Save as compressed NPZ (recommended)
save_embeddings(
    embeddings,
    save_path='embeddings/kerala_embeddings.npz',
    checklist_ids=df['SAMPLING EVENT IDENTIFIER'],
    metadata={'model': 'vae_kerala', 'latent_dim': 128}
)

# Save as CSV (human-readable)
save_embeddings(
    embeddings,
    save_path='embeddings/kerala_embeddings.csv',
    checklist_ids=df['SAMPLING EVENT IDENTIFIER']
)
```

**`load_embeddings(file_path)`**

Load embeddings from file.

```python
from src.inference import load_embeddings

# Load from NPZ
embeddings, checklist_ids = load_embeddings('embeddings.npz')

# Load from CSV
embeddings = load_embeddings('embeddings.csv')
```

## Complete Workflow Example

```python
import pandas as pd
from src.data.loader import load_ebird_data
from src.data.preprocessor import EBirdPreprocessor
from src.inference import EmbeddingExtractor, save_embeddings

# 1. Load and preprocess data
df = load_ebird_data('data/ebd_IN-KL_smp_relSep-2025/ebd_IN-KL_smp_relSep-2025.txt')
preprocessor = EBirdPreprocessor()
processed_df = preprocessor.fit_transform(df)

# 2. Initialize extractor
extractor = EmbeddingExtractor(
    model_path='vae_bird_embeddings_kerala_good.pth',
    device='cuda'
)

# 3. Extract embeddings
embeddings = extractor.extract_embeddings(
    data=processed_df,
    use_mean=True,
    batch_size=512
)

# 4. Save embeddings
save_embeddings(
    embeddings,
    save_path='embeddings/kerala_checklist_embeddings.npz',
    checklist_ids=df['SAMPLING EVENT IDENTIFIER'],
    metadata={
        'model': 'vae_kerala_good',
        'latent_dim': embeddings.shape[1],
        'n_samples': len(embeddings),
        'extraction_mode': 'deterministic'
    }
)

print(f"✓ Extracted and saved {len(embeddings)} embeddings")
```

## Deterministic vs Stochastic Mode

### Deterministic Mode (`use_mean=True`)
- Uses the mean (μ) of the latent distribution directly
- **Recommended for most applications**
- Produces consistent, reproducible embeddings
- Better for downstream tasks like clustering, classification, similarity search
- Same input always produces same embedding

### Stochastic Mode (`use_mean=False`)
- Samples from the latent distribution using reparameterization trick
- Introduces sampling noise
- Useful for generating variations or augmentation
- Same input produces slightly different embeddings each time

**When to use each:**
- **Use deterministic** for: similarity search, clustering, classification, visualization
- **Use stochastic** for: data augmentation, generative tasks, uncertainty estimation

## Input Data Formats

The extractor accepts multiple input formats:

```python
# 1. Pandas DataFrame (most common)
embeddings = extractor.extract_embeddings(processed_df)

# 2. Numpy array
embeddings = extractor.extract_embeddings(processed_df.values)

# 3. PyTorch Tensor
import torch
tensor_data = torch.FloatTensor(processed_df.values)
embeddings = extractor.extract_embeddings(tensor_data)
```

All formats produce identical results.

## File Formats

### NPZ Format (Recommended)
- Compressed binary format
- Efficient storage
- Can store embeddings + metadata
- Fast loading

### CSV Format
- Human-readable text format
- Easy to inspect
- Compatible with Excel, R, etc.
- Larger file size

## Best Practices

1. **Use deterministic mode** (`use_mean=True`) for consistent embeddings
2. **Batch size**: Increase for faster processing (if you have enough memory)
   - CPU: 128-256
   - GPU: 512-1024+
3. **Save in NPZ format** for efficient storage and loading
4. **Include metadata** when saving (model version, parameters, etc.)
5. **Keep checklist IDs** with embeddings for traceability

## Testing

Run the test notebook to verify the module works correctly:

```bash
# Open and run notebooks/test_inference.ipynb
```

The test notebook covers:
- Basic embedding extraction
- Deterministic vs stochastic modes
- Reconstruction quality
- Save/load functionality
- Different input formats

## Troubleshooting

**Problem:** Model fails to load
- **Solution:** Ensure the checkpoint file exists and is a valid .pth file from the training module

**Problem:** Out of memory errors
- **Solution:** Reduce `batch_size` parameter

**Problem:** Embeddings don't match expected dimensions
- **Solution:** Ensure input data has been preprocessed with the same preprocessor used during training

**Problem:** Different results each time (deterministic mode)
- **Solution:** Ensure `use_mean=True` and the model is in eval mode (automatically set by EmbeddingExtractor)

## See Also

- `src/models.py` - VAE model architecture
- `src/data/preprocessor.py` - Data preprocessing
- `src/training/` - Training utilities
- `notebooks/test_inference.ipynb` - Testing and examples
