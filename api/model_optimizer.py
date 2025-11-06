"""
Model optimization utilities for production deployment
"""
import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize trained models for production deployment"""
    
    def __init__(self, model_dir: str = "."):
        self.model_dir = Path(model_dir)
        
    def compress_models(self, compression_level: int = 3) -> Dict[str, str]:
        """
        Compress model files to reduce size
        
        Args:
            compression_level: 0-9, higher = more compression
            
        Returns:
            Dict mapping original -> compressed filenames
        """
        model_files = list(self.model_dir.glob("*.pkl"))
        compressed_files = {}
        
        for model_file in model_files:
            try:
                # Load model
                model = joblib.load(model_file)
                
                # Compress and save
                compressed_name = model_file.stem + "_compressed.pkl"
                compressed_path = self.model_dir / compressed_name
                
                joblib.dump(model, compressed_path, compress=compression_level)
                
                # Compare sizes
                original_size = model_file.stat().st_size
                compressed_size = compressed_path.stat().st_size
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                logger.info(f"Compressed {model_file.name}: "
                          f"{original_size:,} → {compressed_size:,} bytes "
                          f"({compression_ratio:.1f}% reduction)")
                
                compressed_files[str(model_file)] = str(compressed_path)
                
            except Exception as e:
                logger.error(f"Failed to compress {model_file}: {e}")
                
        return compressed_files
    
    def validate_models(self) -> Dict[str, bool]:
        """Validate that all models can be loaded successfully"""
        model_files = list(self.model_dir.glob("*.pkl"))
        validation_results = {}
        
        for model_file in model_files:
            try:
                model = joblib.load(model_file)
                
                # Basic validation - try to get model info
                if hasattr(model, 'predict'):
                    # Try a dummy prediction if possible
                    if hasattr(model, 'n_features_in_'):
                        dummy_input = np.random.rand(1, model.n_features_in_)
                        _ = model.predict(dummy_input)
                    
                validation_results[str(model_file)] = True
                logger.info(f"✓ {model_file.name} validated successfully")
                
            except Exception as e:
                validation_results[str(model_file)] = False
                logger.error(f"✗ {model_file.name} validation failed: {e}")
                
        return validation_results
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all models"""
        model_files = list(self.model_dir.glob("*.pkl"))
        model_info = {}
        
        for model_file in model_files:
            try:
                model = joblib.load(model_file)
                file_size = model_file.stat().st_size
                
                info = {
                    'file_size': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 2),
                    'model_type': type(model).__name__,
                    'has_predict': hasattr(model, 'predict'),
                    'has_predict_proba': hasattr(model, 'predict_proba'),
                }
                
                # Try to get additional info
                if hasattr(model, 'n_features_in_'):
                    info['n_features'] = model.n_features_in_
                    
                if hasattr(model, 'classes_'):
                    info['n_classes'] = len(model.classes_)
                    info['classes'] = list(model.classes_)
                    
                if hasattr(model, 'estimators_'):
                    info['n_estimators'] = len(model.estimators_)
                    
                model_info[model_file.name] = info
                
            except Exception as e:
                model_info[model_file.name] = {'error': str(e)}
                
        return model_info
    
    def create_model_manifest(self) -> Dict[str, Any]:
        """Create a manifest file with model metadata"""
        manifest = {
            'models': self.get_model_info(),
            'validation': self.validate_models(),
            'total_size_mb': sum(
                info.get('file_size_mb', 0) 
                for info in self.get_model_info().values() 
                if 'file_size_mb' in info
            )
        }
        
        # Save manifest
        manifest_path = self.model_dir / 'model_manifest.json'
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
            
        logger.info(f"Model manifest saved to {manifest_path}")
        return manifest

def optimize_for_deployment():
    """Main function to optimize models for deployment"""
    optimizer = ModelOptimizer()
    
    print("🔍 Analyzing models...")
    model_info = optimizer.get_model_info()
    
    print(f"\n📊 Found {len(model_info)} model files:")
    for name, info in model_info.items():
        if 'error' not in info:
            print(f"  • {name}: {info['file_size_mb']} MB ({info['model_type']})")
        else:
            print(f"  • {name}: ERROR - {info['error']}")
    
    total_size = sum(info.get('file_size_mb', 0) for info in model_info.values())
    print(f"  Total size: {total_size:.2f} MB")
    
    print("\n✅ Validating models...")
    validation_results = optimizer.validate_models()
    valid_models = sum(validation_results.values())
    print(f"  {valid_models}/{len(validation_results)} models validated successfully")
    
    print("\n🗜️ Compressing models...")
    compressed_files = optimizer.compress_models(compression_level=3)
    print(f"  Compressed {len(compressed_files)} models")
    
    print("\n📄 Creating model manifest...")
    manifest = optimizer.create_model_manifest()
    
    print(f"\n✨ Optimization complete!")
    print(f"  • Total models: {len(model_info)}")
    print(f"  • Valid models: {valid_models}")
    print(f"  • Compressed models: {len(compressed_files)}")
    print(f"  • Total size: {manifest['total_size_mb']:.2f} MB")
    
    return manifest

if __name__ == "__main__":
    optimize_for_deployment()