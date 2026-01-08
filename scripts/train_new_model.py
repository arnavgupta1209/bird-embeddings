"""
Train a new VAE model and save it in the proper format for inference.
Run from project root: python scripts/train_new_model.py
"""
import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_ebird_data
from src.data.preprocessor import EBirdPreprocessor
from src.models import VariationalAutoencoder
from src.training import train_variational_autoencoder, get_device, save_model_for_inference
import torch

print("="*80)
print("Training New VAE Model")
print("="*80)

# 1. Load and preprocess data
print("\n1. Loading and preprocessing data...")
data_path = project_root / 'data' / 'raw' / 'ebd_IN-KL_smp_relSep-2025.txt'
df = load_ebird_data(str(data_path), nrows=50000)
print(f"   Loaded {len(df)} checklists")

preprocessor = EBirdPreprocessor()
processed_df = preprocessor.fit_transform(df)
print(f"   Processed to {processed_df.shape[1]} features")

# 2. Create model
print("\n2. Creating model...")
device = get_device()
input_dim = processed_df.shape[1]
latent_dim = 16
hidden_dim = 512

model = VariationalAutoencoder(
    input_dimension=input_dim,
    latent_dimension=latent_dim,
    hidden_dimension=hidden_dim
)
model = model.to(device)
print(f"   Model: input={input_dim}, latent={latent_dim}, hidden={hidden_dim}")

# 3. Prepare data
print("\n3. Preparing data loaders...")
from torch.utils.data import DataLoader
import numpy as np

# Split train/val
train_size = int(0.8 * len(processed_df))
train_data = torch.FloatTensor(processed_df[:train_size].values)
val_data = torch.FloatTensor(processed_df[train_size:].values)

# Create simple dataset class that returns tensor directly
class SimpleDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = SimpleDataset(train_data)
val_dataset = SimpleDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# 4. Train
print("\n4. Training model (10 epochs)...")

history = train_variational_autoencoder(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    learning_rate=1e-3,
    device=device,
    noise_std=0.1,
    print_every=2
)

# 5. Save model
print("\n5. Saving model...")
save_path = project_root / 'models' / 'vae_model_inference_ready.pth'

save_model_for_inference(
    model=model,
    filepath=str(save_path),
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dims=[hidden_dim],  # Save as list for compatibility
    metadata={
        'dataset': 'Kerala eBird',
        'n_samples': len(processed_df),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1]
    }
)

print("\n" + "="*80)
print("✓ Training complete! Model saved to:")
print(f"  {save_path}")
print("="*80)
