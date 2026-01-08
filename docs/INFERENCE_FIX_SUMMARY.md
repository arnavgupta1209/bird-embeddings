# Fix Summary: test_inference.ipynb

## Problem
The `test_inference.ipynb` notebook was failing with error:
```
AttributeError: Can't get attribute 'VariationalAutoencoder'
```

## Root Cause
The model file `vae_bird_embeddings_kerala_good.pth` was saved using the legacy method:
```python
torch.save(vae_model, 'vae_bird_embeddings.pth')
```

This pickles the entire model object from the notebook's `__main__` namespace. When loading from a different context (the inference module), Python cannot find the `VariationalAutoencoder` class because it's looking in the wrong namespace.

## Solution
**Use properly saved models with the training module's `save_model_for_inference()` function.**

### What was fixed:
1. ✅ Fixed parameter naming mismatch in `src/inference/extractor.py`:
   - VAE constructor expects: `input_dimension`, `latent_dimension`, `hidden_dimension`
   - Extractor was using: `input_dim`, `latent_dim`, `hidden_dims`
   - Now correctly maps checkpoint keys to constructor parameters

2. ✅ Updated `notebooks/test_inference.ipynb`:
   - Changed model path from legacy model to new properly-saved model
   - Added comment explaining the requirement

3. ✅ Created `train_new_model.py`:
   - Script to train a fresh model and save it correctly
   - Uses `save_model_for_inference()` to ensure compatibility

4. ✅ Updated `src/inference/README.md`:
   - Added clear warning about model requirements
   - Documented the proper workflow

## How to Use

### Option 1: Train a new model (Recommended)
```bash
python train_new_model.py
```

This will:
- Train a VAE model on Kerala eBird data
- Save it properly to `vae_model_inference_ready.pth`
- Be compatible with the inference module

### Option 2: Re-save an existing model
If you have a trained model loaded in memory:
```python
from src.training import save_model_for_inference

save_model_for_inference(
    model=your_model,
    filepath='model_inference_ready.pth',
    input_dim=467,  # Your model's input dimension
    latent_dim=16,  # Your model's latent dimension
    hidden_dims=[512]  # Your model's hidden dimensions
)
```

### Then run the test notebook
```bash
jupyter notebook notebooks/test_inference.ipynb
```

The notebook will now work correctly with the properly-saved model.

## Technical Details

### Proper model format (checkpoint dict):
```python
{
    'model_state_dict': model.state_dict(),
    'input_dim': 467,
    'latent_dim': 16,
    'hidden_dims': [512],
    'metadata': {...}  # optional
}
```

### What the extractor does:
1. Loads checkpoint file
2. Checks if it's a dict (modern format) or object (legacy format)
3. For dict format:
   - Extracts architecture parameters
   - Creates new VAE with those parameters
   - Loads weights into the model
4. Returns ready-to-use model

## Files Modified
- `src/inference/extractor.py` - Fixed parameter naming
- `notebooks/test_inference.ipynb` - Updated model path
- `src/inference/README.md` - Added documentation
- `train_new_model.py` - Created training script (NEW)
