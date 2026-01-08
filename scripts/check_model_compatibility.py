"""
Check if a model file is compatible with the inference module.
Run from project root: python scripts/check_model_compatibility.py [model_path]
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

def check_model_compatibility(model_path):
    """Check if a model file is compatible with the inference module."""
    
    print("="*80)
    print("Model Compatibility Checker")
    print("="*80)
    print(f"\nChecking: {model_path}")
    
    if not Path(model_path).exists():
        print("❌ ERROR: File does not exist!")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"\nFile type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("✓ Model is a dictionary (modern format)")
            print(f"\nKeys found: {list(checkpoint.keys())}")
            
            # Check required keys
            required_keys = ['model_state_dict', 'input_dim', 'latent_dim', 'hidden_dims']
            missing_keys = [k for k in required_keys if k not in checkpoint]
            
            if missing_keys:
                print(f"\n❌ INCOMPATIBLE: Missing required keys: {missing_keys}")
                print("\nTo fix, re-save using save_model_for_inference():")
                print("  from src.training import save_model_for_inference")
                print("  save_model_for_inference(model, 'model.pth', input_dim, latent_dim, [hidden_dim])")
                return False
            else:
                print(f"\n✅ COMPATIBLE: All required keys present")
                print(f"\nModel architecture:")
                print(f"  Input dim: {checkpoint['input_dim']}")
                print(f"  Latent dim: {checkpoint['latent_dim']}")
                print(f"  Hidden dims: {checkpoint['hidden_dims']}")
                if 'metadata' in checkpoint:
                    print(f"  Metadata: {checkpoint['metadata']}")
                return True
                
        else:
            print("❌ INCOMPATIBLE: Model is a pickled object (legacy format)")
            print("\nLegacy models saved with torch.save(model, path) from notebooks")
            print("are NOT compatible with the inference module.")
            print("\nOptions:")
            print("  1. Train a new model using train_new_model.py")
            print("  2. Re-save the model using save_model_for_inference()")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR loading file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_model_compatibility.py <path_to_model.pth>")
        print("\nChecking default models in models/ folder...")
        
        models_dir = project_root / 'models'
        models_to_check = [
            'vae_bird_embeddings_kerala_good.pth',
            'vae_bird_embeddings.pth',
            'vae_model_inference_ready.pth'
        ]
        
        for model_name in models_to_check:
            model_path = models_dir / model_name
            if model_path.exists():
                check_model_compatibility(str(model_path))
                print()
    else:
        model_path = sys.argv[1]
        check_model_compatibility(model_path)
