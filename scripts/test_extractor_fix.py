"""
Quick test to verify the inference module fix.
Run from project root: python scripts/test_extractor_fix.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import EmbeddingExtractor
import torch

print("Testing EmbeddingExtractor fix...")
print("-" * 50)

# Test loading the new model
model_path = project_root / 'models' / 'vae_model_inference_ready.pth'

try:
    print(f"\n1. Loading model from: {model_path}")
    extractor = EmbeddingExtractor(str(model_path), device='cpu')
    print("   ✓ Model loaded successfully!")
    
    # Test with dummy data
    print("\n2. Testing with dummy data...")
    dummy_data = torch.randn(5, 373)  # 5 samples, 373 features (matches trained model)
    embeddings = extractor.extract_embeddings(dummy_data, use_mean=True, batch_size=5)
    print(f"   ✓ Extracted embeddings with shape: {embeddings.shape}")
    
    print("\n" + "=" * 50)
    print("SUCCESS! All tests passed.")
    print("=" * 50)
    
except Exception as e:
    print(f"\n   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 50)
    print("FAILED - See error above")
    print("=" * 50)
