"""
Model Manager — Singleton for handling multiple active ML models
===============================================================
This manager loads all available trained model pairs from trained_models/
and allows routes to fetch a specific model by type.
"""

import os
import logging
from typing import Dict, List, Optional
from models.combined_model_loader import CombinedModelLoader, load_best_available_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.default_model_type = None
        return cls._instance

    def _find_model_files(self, trained_models_dir: str, model_type: str):
        """
        Find ASD and emotion model files for a given model_type.
        Supports both exact names (resnet50v2_asd_model.h5) and versioned
        names (resnet50v2_v3_asd_model.h5).
        Returns (asd_path, emo_path) or (None, None) if not found.
        """
        import glob

        def _find(suffix):
            # 1. Exact match (no version)
            exact = os.path.join(trained_models_dir, f"{model_type}_{suffix}.h5")
            if os.path.exists(exact):
                return exact
            # 2. Versioned match: {model_type}_v*_{suffix}.h5
            # Use explicit _v prefix so resnet50* never matches resnet50v2* files
            pattern = os.path.join(trained_models_dir, f"{model_type}_v*_{suffix}.h5")
            matches = sorted(glob.glob(pattern))  # sort so highest version is last
            if matches:
                # Extra guard: ensure the matched filename starts with exactly model_type + "_v"
                # (glob on Windows can be case-insensitive, so do a str check too)
                safe = [m for m in matches
                        if os.path.basename(m).startswith(f"{model_type}_v")]
                if safe:
                    return safe[-1]
            return None

        return _find("asd_model"), _find("emotion_model")

    def initialize(self, trained_models_dir: str):
        """Load all available model pairs from the directory."""
        logger.info(f"Initializing ModelManager with directory: {trained_models_dir}")
        self.models = {}
        
        # Priority for the default model - Consolidating to ResNet50V2 only
        priority = ["resnet50v2"]
        
        for model_type in priority:
            asd_path, emo_path = self._find_model_files(trained_models_dir, model_type)

            if asd_path and emo_path:
                logger.info(f"  Found {model_type.upper()} files:")
                logger.info(f"    ASD:     {os.path.basename(asd_path)}")
                logger.info(f"    Emotion: {os.path.basename(emo_path)}")
                try:
                    logger.info(f"Loading {model_type.upper()}...")
                    img_size = 224 # Standard size for ResNet50V2
                    # Look for label files with the same prefix as the found model file
                    base_prefix = os.path.basename(asd_path).replace("_asd_model.h5", "")
                    asd_lbl = os.path.join(trained_models_dir, f"{base_prefix}_asd_labels.json")
                    emo_lbl = os.path.join(trained_models_dir, f"{base_prefix}_emotion_labels.json")
                    # Also try generic (non-versioned) label files
                    if not os.path.exists(asd_lbl):
                        asd_lbl = os.path.join(trained_models_dir, f"{model_type}_asd_labels.json")
                    if not os.path.exists(emo_lbl):
                        emo_lbl = os.path.join(trained_models_dir, f"{model_type}_emotion_labels.json")
                    
                    loader = CombinedModelLoader(
                        asd_model_path=asd_path,
                        emotion_model_path=emo_path,
                        model_type=model_type,
                        asd_labels_path=asd_lbl if os.path.exists(asd_lbl) else None,
                        emotion_labels_path=emo_lbl if os.path.exists(emo_lbl) else None,
                        img_size=img_size
                    )
                    
                    if loader.is_ready:
                        self.models[model_type] = loader
                        if self.default_model_type is None:
                            self.default_model_type = model_type
                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {e}")
            else:
                logger.info(f"  No files found for {model_type.upper()}, skipping.")

        if not self.models:
            logger.warning("No models loaded. Please train models and place them in trained_models/")

    def get_model(self, model_type: Optional[str] = None) -> Optional[CombinedModelLoader]:
        """Get a specific model by type, or the default if not found/specified."""
        requested = model_type.lower() if model_type else "None"
        if not model_type or model_type.lower() not in self.models:
            resolved = self.default_model_type if self.default_model_type else "None"
            logger.info(f"ModelManager: Requested '{requested}' not found/None, falling back to '{resolved}'")
            if self.default_model_type:
                return self.models[self.default_model_type]
            return None
        logger.info(f"ModelManager: Returning requested model '{requested}'")
        return self.models[model_type.lower()]

    def list_available_models(self) -> List[str]:
        """Return list of loaded model types."""
        return list(self.models.keys())

# Global instance
model_manager = ModelManager()
