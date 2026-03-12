"""
FYP Model Loader - High Accuracy Model Integration
Supports VGG16, VGG19, ResNet50, ResNet50V2, InceptionV3
Based on FYP implementation achieving 93-94% accuracy
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional, List
import logging
import io
import base64
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Known last conv layers per architecture (for reliable Grad-CAM) ---
LAST_CONV_LAYERS = {
    'vgg16':      'block5_conv3',
    'vgg19':      'block5_conv4',
    'resnet50':   'conv5_block3_out',
    'resnet50v2': 'conv5_block3_out',
    'inceptionv3': 'mixed10',
}

GRADCAM_COLORMAP = cv2.COLORMAP_JET


class FYPModelLoader:
    """
    Loads and manages high-accuracy FYP models for ASD detection
    """
    
    def __init__(self, model_path: str, model_type: str = "vgg19"):
        """
        Initialize FYP Model Loader
        
        Args:
            model_path: Path to the trained model file (.h5)
            model_type: Type of model (vgg16, vgg19, resnet50, resnet50v2, inceptionv3)
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.img_size = 224  # Aligned with original research
        self.model = None
        self.class_names = ['ASD', 'NON_ASD']
        
        # XAI target layer
        self._xai_layer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Using existing trained model instead")
                return
            
            logger.info(f"Loading FYP {self.model_type.upper()} model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully! Input shape: {self.model.input_shape}")
            
            # Resolve XAI layer
            self._xai_layer = self._find_xai_layer(self.model)
            if self._xai_layer:
                logger.info(f"  [SUCCESS] XAI layer: {self._xai_layer}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _find_xai_layer(self, model: keras.Model) -> Optional[str]:
        """Find the best layer for Grad-CAM."""
        known = LAST_CONV_LAYERS.get(self.model_type)
        if known:
            found = self._get_layer_from_model(model, known)
            if found:
                return known

        # Fallback: scan for last conv-like layer
        for layer in reversed(model.layers):
            name = layer.name.lower()
            if any(x in name for x in ['conv', 'mixed', 'block']):
                if hasattr(layer, 'output_shape'):
                    out = layer.output_shape
                    if isinstance(out, (list, tuple)) and len(out) == 4:
                        return layer.name
        return None
    
    def preprocess_image(self, image_input) -> Tuple[np.ndarray, Image.Image]:
        """Preprocess image for FYP model prediction"""
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) \
                      if image_input.shape[-1] == 3 else image_input
        else:
            raise ValueError("image_input must be bytes or numpy array")

        pil_image = Image.fromarray(img_rgb)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, pil_image
    
    def predict(self, image_input) -> Dict:
        """Predict ASD classification (no XAI)."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        processed_image, _ = self.preprocess_image(image_input)
        prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        asd_prob = float(prediction)
        non_asd_prob = 1.0 - asd_prob
        predicted_class = "ASD" if asd_prob > 0.5 else "NON_ASD"
        confidence = max(asd_prob, non_asd_prob)
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "ASD": round(asd_prob, 4),
                "NON_ASD": round(non_asd_prob, 4)
            },
            "model_type": self.model_type
        }

    def predict_with_xai(self, image_input) -> Dict:
        """Predict ASD + Grad-CAM XAI."""
        if self.model is None:
            raise ValueError("Model not loaded.")

        processed_image, pil_image = self.preprocess_image(image_input)
        raw_pred = self.model.predict(processed_image, verbose=0)[0][0]
        
        asd_prob = float(raw_pred)
        non_asd_prob = 1.0 - asd_prob
        predicted_class = "ASD" if asd_prob > 0.5 else "NON_ASD"
        confidence = max(asd_prob, non_asd_prob)

        # XAI
        try:
            heatmap = self._generate_gradcam(processed_image)
            heatmap_b64, overlay_b64 = self._heatmap_to_outputs(heatmap, pil_image)
            xai_data = {
                "heatmap": heatmap_b64,
                "overlay": overlay_b64,
                "original": self._original_to_b64(pil_image),
                "explanation": f"The model detected {predicted_class} with {confidence*100:.1f}% confidence.",
                "method": "Grad-CAM",
                "target_layer": self._xai_layer
            }
        except Exception as e:
            logger.error(f"Grad-CAM error: {e}")
            xai_data = {"error": str(e)}

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "ASD": round(asd_prob, 4),
                "NON_ASD": round(non_asd_prob, 4)
            },
            "model_type": self.model_type,
            "xai": xai_data
        }

    def _generate_gradcam(self, img_array: np.ndarray) -> np.ndarray:
        if not self._xai_layer:
            raise ValueError("No XAI layer identified.")

        target_layer = self._get_layer_from_model(self.model, self._xai_layer)
        grad_model = tf.keras.Model([self.model.inputs], [target_layer.output, self.model.output])

        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            # Binary prediction: target the probability directly
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.nn.relu(heatmap)
        
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            return np.zeros(heatmap.shape.as_list(), dtype=np.float32)
        return (heatmap / max_val).numpy()

    def _get_layer_from_model(self, model, name):
        for layer in model.layers:
            if layer.name == name: return layer
            if isinstance(layer, keras.Model):
                res = self._get_layer_from_model(layer, name)
                if res: return res
        return None

    def _heatmap_to_outputs(self, heatmap, pil_image):
        orig_arr = np.array(pil_image)
        h, w = orig_arr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, GRADCAM_COLORMAP)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(orig_arr.astype(np.float32), 0.6, 
                                 heatmap_colored.astype(np.float32), 0.4, 0).astype(np.uint8)
        
        def to_b64(arr):
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            
        return to_b64(heatmap_colored), to_b64(overlay)

    def _original_to_b64(self, pil_image):
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict ASD classification for multiple images (legacy support)"""
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
    
    def aggregate_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Aggregate multiple predictions (for gamification multi-frame analysis)
        Uses majority voting and average confidence
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Aggregated prediction result
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        # Count votes
        asd_votes = sum(1 for p in predictions if p["prediction"] == "ASD")
        non_asd_votes = len(predictions) - asd_votes
        
        # Determine final prediction
        final_prediction = "ASD" if asd_votes > non_asd_votes else "NON_ASD"
        
        # Calculate average probabilities
        avg_asd_prob = np.mean([p["probabilities"]["ASD"] for p in predictions])
        avg_non_asd_prob = np.mean([p["probabilities"]["NON_ASD"] for p in predictions])
        
        # Calculate confidence (how consistent are the predictions)
        consistency = max(asd_votes, non_asd_votes) / len(predictions)
        
        result = {
            "prediction": final_prediction,
            "confidence": float(max(avg_asd_prob, avg_non_asd_prob)),
            "consistency": float(consistency),
            "probabilities": {
                "ASD": float(avg_asd_prob),
                "NON_ASD": float(avg_non_asd_prob)
            },
            "total_frames": len(predictions),
            "votes": {
                "ASD": asd_votes,
                "NON_ASD": non_asd_votes
            },
            "model_type": self.model_type
        }
        
        return result


class EnsembleModelLoader:
    """
    Ensemble of multiple FYP models for maximum accuracy
    Combines predictions from VGG16, VGG19, ResNet50, etc.
    """
    
    def __init__(self, model_configs: List[Dict[str, str]]):
        """
        Initialize ensemble of models
        
        Args:
            model_configs: List of dicts with 'path' and 'type' keys
                          Example: [{'path': 'vgg19.h5', 'type': 'vgg19'}, ...]
        """
        self.models = []
        
        for config in model_configs:
            try:
                model = FYPModelLoader(config['path'], config['type'])
                self.models.append(model)
                logger.info(f"Added {config['type']} to ensemble")
            except Exception as e:
                logger.warning(f"Could not load {config['type']}: {e}")
        
        if not self.models:
            raise ValueError("No models loaded in ensemble")
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Predict using ensemble (majority voting)
        
        Args:
            image: Input image
            
        Returns:
            Aggregated prediction from all models
        """
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(image)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {model.model_type} prediction failed: {e}")
        
        if not predictions:
            raise ValueError("All models failed to predict")
        
        # Aggregate predictions
        asd_votes = sum(1 for p in predictions if p["prediction"] == "ASD")
        non_asd_votes = len(predictions) - asd_votes
        
        final_prediction = "ASD" if asd_votes > non_asd_votes else "NON_ASD"
        
        # Average probabilities
        avg_asd_prob = np.mean([p["probabilities"]["ASD"] for p in predictions])
        avg_non_asd_prob = np.mean([p["probabilities"]["NON_ASD"] for p in predictions])
        
        result = {
            "prediction": final_prediction,
            "confidence": float(max(avg_asd_prob, avg_non_asd_prob)),
            "probabilities": {
                "ASD": float(avg_asd_prob),
                "NON_ASD": float(avg_non_asd_prob)
            },
            "ensemble_size": len(predictions),
            "votes": {
                "ASD": asd_votes,
                "NON_ASD": non_asd_votes
            },
            "model_types": [p["model_type"] for p in predictions]
        }
        
        return result


# Convenience function
def load_fyp_model(model_path: str, model_type: str = "vgg19") -> FYPModelLoader:
    """
    Load a single FYP model
    
    Args:
        model_path: Path to model file
        model_type: Model architecture type
        
    Returns:
        FYPModelLoader instance
    """
    return FYPModelLoader(model_path, model_type)
