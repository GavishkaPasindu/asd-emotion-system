"""
Combined Model Loader ΓÇö WITH BUILT-IN XAI (Grad-CAM)
====================================================
Loads both ASD and Emotion .h5 files trained in Google Colab.
Supports: VGG16, VGG19, ResNet50, ResNet50V2, InceptionV3

XAI is a CORE FEATURE ΓÇö every prediction includes a Grad-CAM heatmap.
predict_asd_with_xai()       ΓåÆ ASD prediction + heatmap + overlay
predict_emotion_with_xai()   ΓåÆ Emotion prediction + heatmap + overlay
predict_combined_with_xai()  ΓåÆ Both, one call
"""

import os
import json
import io
import base64
import logging

import numpy as np
import cv2

# ΓöÇΓöÇ Keras version compatibility fix ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Old models saved with Keras < 2.12 use 'batch_shape' in InputLayer config.
# New Keras uses 'shape' instead. This shim makes both work transparently.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import InputLayer as _BaseInputLayer

    class CompatInputLayer(_BaseInputLayer):
        """Accepts both `batch_shape` (old Keras) and `shape` (new Keras)."""
        def __init__(self, **kwargs):
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                kwargs['shape'] = batch_shape[1:]   # strip batch dim
            super().__init__(**kwargs)

    _COMPAT_OBJECTS = {'InputLayer': CompatInputLayer}
except Exception:
    _COMPAT_OBJECTS = {}

from tensorflow import keras
from PIL import Image
from typing import Dict, List, Optional, Tuple

# Preprocessing is now a simple 1/255 scaling to match Colab training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ΓöÇΓöÇ Default class labels ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Emotion model trained on ASD children faces with 4 reaction classes.
# fear removed (insufficient samples); disgust/happy/neutral not in scope.
DEFAULT_EMOTION_CLASSES = ['anger', 'joy', 'sadness', 'surprise']

# ASD classes follow Keras flow_from_directory alphabetical order:
# ASD=0, Non_ASD=1
# Sigmoid output: value > 0.5  ΓåÆ predicted class index 1 ΓåÆ Non_ASD
#                 value <= 0.5 ΓåÆ predicted class index 0 ΓåÆ ASD
DEFAULT_ASD_CLASSES = ['ASD', 'Non_ASD']

# ΓöÇΓöÇ Known last conv layers per architecture (for reliable Grad-CAM) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# These are the last convolutional / feature layer in each backbone.
# Grad-CAM targets this layer to produce the attention heatmap.
LAST_CONV_LAYERS = {
    'vgg16':          'block5_conv3',
    'vgg19':          'block5_conv4',
    'resnet50':       'conv5_block3_out',
    # ResNet50V2 uses pre-activation, so the last meaningful spatial feature
    # layer is conv5_block3_3_relu (or post_bn if weights were saved that way)
    'resnet50v2':     'conv5_block3_3_relu',
    'inceptionv3':    'mixed10',
}

# ΓöÇΓöÇ XAI colormap (JET = standard Grad-CAM coloring) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
GRADCAM_COLORMAP = cv2.COLORMAP_JET


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
class CombinedModelLoader:
    """
    Loads a pair of .h5 models (ASD + Emotion) trained in Google Colab.
    Every predict_*_with_xai() call ALWAYS returns XAI output.
    """

    def __init__(
        self,
        asd_model_path: str,
        emotion_model_path: str,
        model_type: str = "resnet50v2",
        asd_labels_path:     Optional[str] = None,
        emotion_labels_path: Optional[str] = None,
        img_size: int = 224,
    ):
        self.asd_model_path     = asd_model_path
        self.emotion_model_path = emotion_model_path
        self.model_type         = model_type.lower()
        self.img_size           = img_size

        # InceptionV3 minimum input size
        if 'inception' in self.model_type and self.img_size < 139:
            self.img_size = 139

        self.asd_model     = None
        self.emotion_model = None

        # Class labels
        self.asd_classes     = self._load_labels(asd_labels_path,     DEFAULT_ASD_CLASSES)
        self.emotion_classes = self._load_labels(emotion_labels_path, DEFAULT_EMOTION_CLASSES)

        # XAI target layer names (auto-detected or from LAST_CONV_LAYERS)
        self._asd_xai_layer     = None
        self._emotion_xai_layer = None

        self._load_models()

    # ΓöÇΓöÇΓöÇ Label helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _load_labels(self, path: Optional[str], defaults: List[str]) -> List[str]:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                max_idx = max(int(k) for k in data.keys())
                labels = [data[str(i)] for i in range(max_idx + 1)]
                logger.info(f"  [SUCCESS] Loaded labels from {os.path.basename(path)}: {labels}")
                return labels
            except Exception as e:
                logger.warning(f"Label load failed ({path}): {e}")
        logger.info(f"  [INFO] Using default labels: {defaults}")
        return defaults

    # ΓöÇΓöÇΓöÇ Model loading ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _load_models(self):
        for attr, path, label in [
            ('asd_model',     self.asd_model_path,     'ASD'),
            ('emotion_model', self.emotion_model_path, 'Emotion'),
        ]:
            if os.path.exists(path):
                m = self._load_model_compat(path, label)
                if m is not None:
                    setattr(self, attr, m)
                    logger.info(f"  [SUCCESS] {label} input shape: {m.input_shape}")
                    # Detect the best Grad-CAM target layer for this model
                    xai_layer = self._find_xai_layer(m, label)
                    if attr == 'asd_model':
                        self._asd_xai_layer = xai_layer
                    else:
                        self._emotion_xai_layer = xai_layer
                    logger.info(f"  [XAI] {label} target layer: {xai_layer}")
                else:
                    logger.error(f"Failed to load {label} model: {path}")
                    raise RuntimeError(f"Could not load {label} model from {path}")
            else:
                logger.warning(f"{label} model not found: {path}")

    def _load_model_compat(self, path: str, label: str):
        """Load a Keras model, auto-patching old 'batch_shape' configs for new Keras."""
        import h5py, json, tempfile, shutil

        def _fix_config(cfg):
            """Recursively fix Keras 3 ΓåÆ Keras 2 config incompatibilities:
            1. InputLayer 'batch_shape' -> 'batch_input_shape'
            2. DTypePolicy dict objects -> plain string (e.g. {'class_name':'DTypePolicy','config':{'name':'float32'}} -> 'float32')
            """
            if isinstance(cfg, dict):
                # Fix 1: InputLayer batch_shape
                if cfg.get('class_name') == 'InputLayer':
                    c = cfg.get('config', {})
                    if 'batch_shape' in c:
                        c['batch_input_shape'] = c.pop('batch_shape')

                # Fix 2: DTypePolicy dict ΓåÆ plain string, applied to any key
                for key in list(cfg.keys()):
                    val = cfg[key]
                    if (isinstance(val, dict) and
                            val.get('class_name') == 'DTypePolicy' and
                            isinstance(val.get('config'), dict)):
                        cfg[key] = val['config'].get('name', 'float32')
                    else:
                        _fix_config(val)
            elif isinstance(cfg, list):
                for item in cfg:
                    _fix_config(item)

        # ΓöÇΓöÇ Strategy 1: plain load (works if Keras versions match) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        try:
            logger.info(f"Loading {label} model (strategy 1 - direct): {path}")
            return keras.models.load_model(path, compile=False)
        except Exception as e1:
            logger.warning(f"  Strategy 1 failed: {e1}")

        # ΓöÇΓöÇ Strategy 2: patch the H5 model_config JSON and reload ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        try:
            logger.info(f"Loading {label} model (strategy 2 - H5 config patch): {path}")
            # Copy to a temp file so we don't modify the original
            tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
            tmp.close()
            shutil.copy2(path, tmp.name)

            with h5py.File(tmp.name, 'r+') as f:
                if 'model_config' in f.attrs:
                    raw = f.attrs['model_config']
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8')
                    cfg = json.loads(raw)
                    _fix_config(cfg)
                    f.attrs['model_config'] = json.dumps(cfg).encode('utf-8')

            model = keras.models.load_model(tmp.name, compile=False)
            os.unlink(tmp.name)
            return model
        except Exception as e2:
            logger.warning(f"  Strategy 2 failed: {e2}")
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        # ΓöÇΓöÇ Strategy 3: build model from JSON config + load weights separately
        try:
            logger.info(f"Loading {label} model (strategy 3 - from_json + weights): {path}")
            with h5py.File(path, 'r') as f:
                raw = f.attrs.get('model_config', '{}')
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                cfg = json.loads(raw)
                _fix_config(cfg)
                model = keras.models.model_from_json(json.dumps(cfg))
                if 'model_weights' in f:
                    model.load_weights(path)
            return model
        except Exception as e3:
            logger.warning(f"  Strategy 3 failed: {e3}")


        # ΓöÇΓöÇ Strategy 4: Rebuild architecture + load weights (Keras 3 ΓåÆ Keras 2) ΓöÇΓöÇ
        # This works because weights are always cross-version compatible.
        try:
            logger.info(f"Loading {label} model (strategy 4 - rebuild+weights): {path}")
            model = self._build_model_by_type(label)
            if model is not None:
                model.load_weights(path, by_name=True, skip_mismatch=True)
                logger.info(f"  [SUCCESS] Rebuilt {label} model and loaded weights by name")
                return model
        except Exception as e4:
            logger.warning(f"  Strategy 4 failed: {e4}")

        return None


    def _build_model_by_type(self, label: str):
        """
        Rebuild a CNN model architecture locally matching the Colab notebook design.
        Used as fallback when H5 config cannot be deserialized (Keras 3 vs 2 mismatch).
        """
        is_asd = (label.upper() == 'ASD')
        img_size   = self.img_size
        model_type = self.model_type.lower()

        # Pick the backbone
        BACKBONES = {
            'resnet50v2':  (keras.applications.ResNet50V2,  'post_bn'),
            'resnet50':    (keras.applications.ResNet50,    'conv5_block3_out'),
            'vgg16':       (keras.applications.VGG16,       'block5_conv3'),
            'vgg19':       (keras.applications.VGG19,       'block5_conv4'),
            'inceptionv3': (keras.applications.InceptionV3, 'mixed10'),
        }
        if model_type not in BACKBONES:
            logger.warning(f"Unknown model type '{model_type}' for rebuild ΓÇö defaulting to resnet50v2")
            model_type = 'resnet50v2'

        BackboneCls, _ = BACKBONES[model_type]
        try:
            base = BackboneCls(
                include_top=False, weights=None,
                input_shape=(img_size, img_size, 3)
            )
        except TypeError:
            # Some backbones don't take input_shape like that
            inp = keras.layers.Input(shape=(img_size, img_size, 3))
            base = BackboneCls(include_top=False, weights=None)(inp)

        x = base.output
        x = keras.layers.GlobalAveragePooling2D()(x)

        # ── All "Improved" Colab notebooks use this 2-layer Dense head ──────
        # ASD head  : Dense(512,relu,name=asd_dense1) → BN(asd_bn1) → Drop(0.5)
        #             Dense(256,relu,name=asd_dense2) → BN(asd_bn2) → Drop(0.4)
        #             Dense(1,sigmoid,name=asd_output)
        # Emotion   : same pattern with emo_ prefix, softmax output with NC classes
        # Using EXACT names is CRITICAL for load_weights(by_name=True) to work.
        p = 'asd' if is_asd else 'emo'

        x = keras.layers.Dense(512, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(1e-4),
                               name=f'{p}_dense1')(x)
        x = keras.layers.BatchNormalization(name=f'{p}_bn1')(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(256, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(1e-4),
                               name=f'{p}_dense2')(x)
        x = keras.layers.BatchNormalization(name=f'{p}_bn2')(x)
        x = keras.layers.Dropout(0.4)(x)

        if is_asd:
            out = keras.layers.Dense(1, activation='sigmoid', name='asd_output')(x)
        else:
            num_cls = len(self.emotion_classes) if hasattr(self, 'emotion_classes') else 4
            out = keras.layers.Dense(num_cls, activation='softmax', name='emo_output')(x)

        model = keras.Model(base.input, out, name=f'{label}_rebuilt')
        logger.info(f"  Rebuilt {label} model ({model_type}) with {model.count_params():,} params")
        return model

    def _find_xai_layer(self, model: keras.Model, label: str) -> Optional[str]:
        """
        Find the best layer for Grad-CAM.
        Priority:
          1. Try each known candidate for this architecture (multiple fallbacks)
          2. Scan nested sub-models for last spatial conv/activation layer
          3. Scan top-level layers in reverse for last conv/spatial layer
        """
        # Architecture-specific candidate lists ΓÇö ordered from best to fallback
        CANDIDATE_LAYERS = {
            'vgg16':            ['block5_conv3', 'block5_conv2', 'block5_conv1'],
            'vgg19':            ['block5_conv4', 'block5_conv3'],
            'resnet50':         ['conv5_block3_out', 'conv5_block3_3_bn', 'conv5_block3_3_relu'],
            'resnet50v2':       ['conv5_block3_3_relu', 'post_bn', 'post_relu',
                                 'conv5_block3_out', 'conv5_block3_preact'],
            'inceptionv3':      ['mixed10', 'mixed9', 'mixed8'],
        }

        candidates = CANDIDATE_LAYERS.get(self.model_type, [])
        if not candidates and self.model_type in LAST_CONV_LAYERS:
            candidates = [LAST_CONV_LAYERS[self.model_type]]

        # 1. Try each candidate name (searches nested sub-models too)
        for name in candidates:
            found = self._recursive_find_layer(model, name)
            if found:
                logger.info(f"  [SUCCESS] {label} XAI layer (known candidate '{name}'): {found}")
                return found

        # 2. Scan inside nested sub-models (backbone is usually a nested Model)
        for layer in reversed(model.layers):
            if isinstance(layer, keras.Model):
                for sub_layer in reversed(layer.layers):
                    name = sub_layer.name.lower()
                    if any(x in name for x in ['conv', 'mixed', 'block', 'relu', 'activation']):
                        try:
                            out = sub_layer.output_shape
                            if isinstance(out, (list, tuple)) and len(out) == 4:
                                logger.info(f"  [SUCCESS] {label} XAI layer (sub-model scan): {sub_layer.name}")
                                return sub_layer.name
                        except Exception:
                            pass

        # 3. Scan top-level layers in reverse for last conv/spatial layer
        for layer in reversed(model.layers):
            name = layer.name.lower()
            if any(x in name for x in ['conv', 'mixed', 'block', 'relu']):
                try:
                    out = layer.output_shape
                    if isinstance(out, (list, tuple)) and len(out) == 4:
                        logger.info(f"  [SUCCESS] {label} XAI layer (auto-scan): {layer.name}")
                        return layer.name
                except Exception:
                    pass

        logger.warning(f"  [WARNING] No suitable XAI layer found for {label} model")
        return None


    def _recursive_find_layer(self, model: keras.Model, name: str) -> Optional[str]:
        """Search model and nested sub-models for a layer by name."""
        for layer in model.layers:
            if layer.name == name:
                return name
            if isinstance(layer, keras.Model):
                result = self._recursive_find_layer(layer, name)
                if result:
                    return result
        return None

    # ΓöÇΓöÇΓöÇ Image preprocessing ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def preprocess(self, image_input) -> Tuple[np.ndarray, Image.Image]:
        """
        Returns:
            img_array (np.ndarray): shape (1, H, W, 3), float32 [0,1]
            pil_image (PIL.Image): original image for overlay
        """
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
        resized   = cv2.resize(img_rgb, (self.img_size, self.img_size))
        
        # ── Preprocessing (Matching Colab Notebooks) ──────────────────────────
        # All models were trained with simple 1/255 scaling and RGB color space.
        # We MUST match this exactly to get identical results to Colab.
        
        # 1. Convert to float array (0-255 range initially)
        arr = resized.astype(np.float32)
        
        # 2. Rescale to [0, 1] range as per ImageDataGenerator(rescale=1./255)
        arr = arr / 255.0
        
        # 3. Add batch dimension
        arr = np.expand_dims(arr, axis=0)
            
        return arr, pil_image

    # ΓöÇΓöÇΓöÇ Grad-CAM XAI core ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _generate_gradcam(
        self,
        model: keras.Model,
        img_array: np.ndarray,
        layer_name: Optional[str],
        is_binary: bool,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        Works for both binary (sigmoid) and multi-class (softmax) models.
        Raises ValueError if heatmap cannot be generated.
        """
        if layer_name is None:
            logger.warning(f"No XAI layer for {self.model_type} ΓÇö returning blank heatmap")
            # Return a neutral grey heatmap so predictions still work
            return np.full((self.img_size, self.img_size), 0.5, dtype=np.float32)

        # Build a grad-model: input ΓåÆ (conv output, final predictions)
        # We need to handle the case where the target layer is inside a nested sub-model.
        target_layer = self._get_layer_from_model(model, layer_name)
        if target_layer is None:
            logger.warning(f"Layer '{layer_name}' not found ΓÇö returning blank heatmap")
            return np.full((self.img_size, self.img_size), 0.5, dtype=np.float32)

        # Strategy A: Build grad model directly if layer is in outer model
        try:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )
        except Exception:
            # Strategy B: The layer is inside a nested sub-model.
            # Build: input ΓåÆ nested_sub_model ΓåÆ target layer output; parallel final output
            try:
                nested_model = None
                for layer in model.layers:
                    if isinstance(layer, keras.Model):
                        try:
                            layer.get_layer(layer_name)
                            nested_model = layer
                            break
                        except (ValueError, AttributeError):
                            pass
                if nested_model is None:
                    raise ValueError("Cannot find nested model containing target layer")
                inner_target = nested_model.get_layer(layer_name)
                grad_model = tf.keras.Model(
                    inputs=model.inputs,
                    outputs=[inner_target.output, model.output]
                )
            except Exception as e2:
                logger.warning(f"Grad-CAM sub-model build failed: {e2} ΓÇö returning blank heatmap")
                return np.full((self.img_size, self.img_size), 0.5, dtype=np.float32)

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            results = grad_model(img_tensor, training=False)
            
            # Robust unpacking (Keras 3 can return outputs as a list)
            if isinstance(results, (list, tuple)):
                conv_outputs = results[0]
                predictions  = results[1]
            else:
                conv_outputs = results
                predictions  = results

            # If individual outputs are still lists/tuples, take the first element
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            if isinstance(conv_outputs, (list, tuple)):
                conv_outputs = conv_outputs[0]

            if is_binary:
                # Binary: gradient w.r.t. ASD probability
                loss = predictions[:, 0]
            else:
                # Multi-class: gradient w.r.t. predicted class
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]

        # Gradients of loss w.r.t. conv feature maps
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("Gradient is None ΓÇö model may not be differentiable at target layer")

        # Average gradients over spatial dims ΓåÆ per-channel weights
        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])

        # Weighted sum of feature maps
        # Explicit reshaping to ensure broadcasting works for all architectures (VGG, ResNet, Inception)
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # ReLU + normalize to [0, 1]
        heatmap = tf.nn.relu(heatmap)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            logger.warning("Grad-CAM: zero heatmap ΓÇö model may be overfit or image is unusual")
            heatmap_np = np.zeros(heatmap.shape.as_list(), dtype=np.float32)
        else:
            heatmap_np = (heatmap / max_val).numpy()

        return heatmap_np

    def _get_layer_from_model(self, model: keras.Model, name: str):
        """Get a layer by name, searching nested sub-models too."""
        for layer in model.layers:
            if layer.name == name:
                return layer
            if isinstance(layer, keras.Model):
                found = self._get_layer_from_model(layer, name)
                if found:
                    return found
        return None

    def _heatmap_to_outputs(
        self, heatmap: np.ndarray, pil_image: Image.Image
    ) -> Tuple[str, str]:
        """
        Convert Grad-CAM heatmap to:
          - heatmap_b64: colored heatmap as base64 PNG
          - overlay_b64: heatmap overlaid on original image as base64 PNG
        """
        # Resize heatmap to match original image
        orig_arr = np.array(pil_image)
        h, w = orig_arr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap (JET)
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, GRADCAM_COLORMAP)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay (70% original, 30% heatmap for clear visibility)
        overlay = cv2.addWeighted(
            orig_arr.astype(np.float32),     0.6,
            heatmap_colored.astype(np.float32), 0.4,
            0
        ).astype(np.uint8)

        def to_b64(arr: np.ndarray) -> str:
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return to_b64(heatmap_colored), to_b64(overlay)



    # ΓöÇΓöÇΓöÇ MediaPipe face-region definitions ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Landmark indices per region (MediaPipe 468-point face mesh)
    _MP_REGIONS = {
        'Forehead':      [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377],
        'Left Eye':      [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
        'Right Eye':     [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
        'Left Eyebrow':  [70,63,105,66,107,55,65,52,53,46],
        'Right Eyebrow': [300,293,334,296,336,285,295,282,283,276],
        'Nose':          [1,2,5,4,197,195,196,174,48,64,98,97,326,327,294,278],
        'Mouth/Lips':    [61,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
        'Left Cheek':    [116,111,117,118,119,120,121,128,126,142,36,205,206,207,187,147,123],
        'Right Cheek':   [345,340,346,347,348,349,350,451,452,350,277,329,330,280,411,376,352],
        'Chin':          [152,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338],
    }
    # Fallback proportional bounding boxes (y0,y1,x0,x1) when no face detected
    _FALLBACK_BOXES = {
        'Forehead':      (0.00, 0.22, 0.15, 0.85),
        'Left Eye':      (0.22, 0.42, 0.05, 0.48),
        'Right Eye':     (0.22, 0.42, 0.52, 0.95),
        'Left Eyebrow':  (0.16, 0.30, 0.05, 0.48),
        'Right Eyebrow': (0.16, 0.30, 0.52, 0.95),
        'Nose':          (0.38, 0.65, 0.30, 0.70),
        'Mouth/Lips':    (0.58, 0.80, 0.20, 0.80),
        'Left Cheek':    (0.35, 0.65, 0.00, 0.35),
        'Right Cheek':   (0.35, 0.65, 0.65, 1.00),
        'Chin':          (0.78, 1.00, 0.20, 0.80),
    }
    # Distinct BGR colors for each region overlay
    _REGION_COLORS = {
        'Forehead':      (255, 200,  80),
        'Left Eye':      ( 80, 200, 255),
        'Right Eye':     ( 80, 200, 255),
        'Left Eyebrow':  ( 80, 255, 160),
        'Right Eyebrow': ( 80, 255, 160),
        'Nose':          (255, 100, 100),
        'Mouth/Lips':    (255,  80, 200),
        'Left Cheek':    (200, 160, 255),
        'Right Cheek':   (200, 160, 255),
        'Chin':          (100, 220, 255),
    }

    def _get_mp_masks(self, img_rgb: np.ndarray, h: int, w: int) -> dict:
        """Use MediaPipe FaceLandmarker to build per-region binary masks."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request

            # Download the face landmark model if not cached
            model_path = '/tmp/face_landmarker.task'
            if not os.path.exists(model_path):
                logger.info('Downloading MediaPipe face_landmarker.task (~29 MB)...')
                urllib.request.urlretrieve(
                    'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
                    'face_landmarker/float16/1/face_landmarker.task',
                    model_path
                )
                logger.info('  face_landmarker.task downloaded Γ£ô')

            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            options   = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.40,
                min_face_presence_confidence=0.40,
                min_tracking_confidence=0.40,
            )
            detector = mp_vision.FaceLandmarker.create_from_options(options)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=img_rgb.astype(np.uint8))
            result   = detector.detect(mp_image)

            if not result.face_landmarks:
                logger.info('  [MediaPipe] No face detected ΓåÆ fallback boxes')
                return self._make_fallback_masks(h, w)

            lm     = result.face_landmarks[0]
            pts_all = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)

            masks = {}
            for rn, idx_list in self._MP_REGIONS.items():
                valid = [i for i in idx_list if i < len(pts_all)]
                pts   = pts_all[valid]
                m     = np.zeros((h, w), dtype=np.uint8)
                if len(pts) >= 3:
                    cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
                elif len(pts) > 0:
                    for px, py in pts:
                        cv2.circle(m, (px, py), 8, 255, -1)
                masks[rn] = cv2.GaussianBlur(m, (15, 15), 0)
            return masks

        except Exception as e:
            logger.warning(f'  [MediaPipe] Error: {e} ΓåÆ fallback boxes')
            return self._make_fallback_masks(h, w)

    def _make_fallback_masks(self, h: int, w: int) -> dict:
        """Build proportional bounding-box masks when MediaPipe unavailable."""
        masks = {}
        for nm, (y0, y1, x0, x1) in self._FALLBACK_BOXES.items():
            m = np.zeros((h, w), dtype=np.uint8)
            m[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 255
            masks[nm] = m
        return masks

    def _generate_face_region_attention(
        self,
        pil_image: Image.Image,
        cam_heatmap: np.ndarray,
    ) -> dict:
        """
        Score each facial region by how much of the Grad-CAM attention falls
        on it, using MediaPipe face landmarks (same approach as training notebooks).

        Returns:
            {
              'region_scores': {region_name: float, ...},   # sorted descending
              'face_overlay':  'data:image/png;base64,...', # annotated face image
            }
        """
        try:
            img_rgb  = np.array(pil_image)          # H├ùW├ù3  uint8
            h, w     = img_rgb.shape[:2]

            # Resize heatmap to match image
            heatmap  = cv2.resize(cam_heatmap, (w, h))

            masks    = self._get_mp_masks(img_rgb, h, w)

            # ΓöÇΓöÇ Score each region ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            cam_mean = float(heatmap.mean()) + 1e-8
            scores   = {}
            for rn, mask in masks.items():
                m_float = mask.astype(np.float32) / 255.0
                if m_float.sum() < 1:
                    scores[rn] = 0.0
                else:
                    scores[rn] = round(float((heatmap * m_float).mean() / cam_mean), 3)

            scores_sorted = dict(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)
            )

            # ΓöÇΓöÇ Build color-annotated face overlay ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            overlay = img_rgb.copy()
            THR     = 0.80   # highlight regions above this threshold

            for rn, mask in masks.items():
                score = scores.get(rn, 0.0)
                color = self._REGION_COLORS.get(rn, (200, 200, 200))
                # Semi-transparent fill for high-attention regions
                if score >= THR:
                    colored = np.zeros_like(overlay)
                    colored[mask > 0] = color
                    overlay = cv2.addWeighted(overlay, 0.65, colored, 0.35, 0)
                # Draw region outline
                contours, _ = cv2.findContours(
                    (mask > 50).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1,
                                 color if score >= THR else (160, 160, 160), 1)

            # Annotate top-3 region names
            top3 = list(scores_sorted.items())[:3]
            font  = cv2.FONT_HERSHEY_SIMPLEX
            y_off = 18
            for i, (rn, sc) in enumerate(top3):
                color = self._REGION_COLORS.get(rn, (255, 255, 255))
                label = f'{rn}: {sc:.2f}'
                cv2.putText(overlay, label, (6, y_off + i * 16),
                            font, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, label, (6, y_off + i * 16),
                            font, 0.40, color,   1, cv2.LINE_AA)

            # Encode to base64 PNG
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format='PNG')
            face_overlay_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

            return {
                'region_scores': scores_sorted,
                'face_overlay':  face_overlay_b64,
            }

        except Exception as e:
            logger.warning(f'Face-region attention failed: {e}')
            return {'region_scores': {}, 'face_overlay': None}

    def _original_to_b64(self, pil_image: Image.Image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ΓöÇΓöÇΓöÇ XAI explanation text ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _asd_explanation(self, predicted: str, confidence: float) -> str:
        conf_pct = f"{confidence * 100:.1f}%"
        if predicted == 'ASD':
            level = "high" if confidence > 0.85 else "moderate"
            return (
                f"The model detected ASD traits with {conf_pct} confidence. "
                f"The heatmap highlights facial regions that contributed most "
                f"to this {level}-confidence classification. "
                f"Key areas typically include eye contact patterns and facial symmetry."
            )
        else:
            level = "high" if confidence > 0.85 else "moderate"
            return (
                f"The model classified this as Non-ASD with {conf_pct} confidence. "
                f"The heatmap shows {level}-salience facial features analyzed. "
                f"No strong ASD-related facial patterns were detected."
            )

    def _emotion_explanation(self, emotion: str, confidence: float, top_emotions: list) -> str:
        conf_pct = f"{confidence * 100:.1f}%"
        top_str  = ', '.join(f"{e['emotion']} ({e['probability']*100:.0f}%)"
                              for e in top_emotions[:3])
        return (
            f"Primary emotion detected: {emotion.upper()} ({conf_pct} confidence). "
            f"Top emotions: {top_str}. "
            f"The heatmap highlights the facial regions ΓÇö particularly eyes, brows, "
            f"and mouth ΓÇö that most strongly influenced this emotion classification."
        )

    # ΓöÇΓöÇΓöÇ Public prediction API ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def predict_asd(self, image_input) -> Dict:
        """ASD prediction only (no XAI)."""
        if self.asd_model is None:
            raise ValueError(f"ASD model not loaded: {self.asd_model_path}")

        img_array, _ = self.preprocess(image_input)
        raw          = self.asd_model.predict(img_array, verbose=0)

        # Sigmoid: output is P(class_index_1) because keras binary_crossentropy
        # trains toward label=1 (Non_ASD when flow_from_directory sorts alphabetically:
        # ASD=0, Non_ASD=1).  So raw[0][0] > 0.5 ΓåÆ Non_ASD.
        sigmoid_val  = float(raw[0][0])
        prob_non_asd = sigmoid_val
        prob_asd     = 1.0 - sigmoid_val
        predicted    = self.asd_classes[1] if sigmoid_val > 0.5 else self.asd_classes[0]
        confidence   = max(prob_asd, prob_non_asd)

        return {
            "success": True,
            "predicted_class": predicted,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.asd_classes[0]: round(prob_asd, 4),      # ASD
                self.asd_classes[1]: round(prob_non_asd, 4),  # Non_ASD
            },
            "model_type": self.model_type,
        }

    def predict_emotion(self, image_input) -> Dict:
        """Emotion prediction only (no XAI)."""
        if self.emotion_model is None:
            raise ValueError(f"Emotion model not loaded: {self.emotion_model_path}")

        img_array, _ = self.preprocess(image_input)
        raw          = self.emotion_model.predict(img_array, verbose=0)[0]

        pred_idx    = int(np.argmax(raw))
        pred_emo    = (self.emotion_classes[pred_idx]
                       if pred_idx < len(self.emotion_classes)
                       else f"class_{pred_idx}")
        confidence  = float(raw[pred_idx])

        probs = {
            (self.emotion_classes[i] if i < len(self.emotion_classes) else f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(raw)
        }
        top_emotions = [
            {"emotion": e, "probability": round(p, 4)}
            for e, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        return {
            "success": True,
            "predicted_emotion": pred_emo,
            "confidence": round(confidence, 4),
            "probabilities": probs,
            "top_emotions": top_emotions,
            "model_type": self.model_type,
        }

    # ΓöÇΓöÇ WITH XAI ΓÇö these are the primary methods for the API ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def predict_asd_with_xai(self, image_input) -> Dict:
        """
        ASD prediction + Grad-CAM XAI heatmap.
        XAI output is ALWAYS present in the result.
        """
        if self.asd_model is None:
            raise ValueError(f"ASD model not loaded: {self.asd_model_path}")

        img_array, pil_image = self.preprocess(image_input)

        # ΓöÇΓöÇ Prediction
        raw          = self.asd_model.predict(img_array, verbose=0)
        # Sigmoid > 0.5 ΓåÆ Non_ASD (class index 1); <= 0.5 ΓåÆ ASD (class index 0)
        sigmoid_val  = float(raw[0][0])
        prob_non_asd = sigmoid_val
        prob_asd     = 1.0 - sigmoid_val
        predicted    = self.asd_classes[1] if sigmoid_val > 0.5 else self.asd_classes[0]
        confidence   = max(prob_asd, prob_non_asd)

        # ΓöÇΓöÇ XAI: Grad-CAM
        heatmap = self._generate_gradcam(
            self.asd_model, img_array, self._asd_xai_layer, is_binary=True
        )
        heatmap_b64, overlay_b64 = self._heatmap_to_outputs(heatmap, pil_image)

        # ΓöÇΓöÇ XAI: MediaPipe face-region attention
        face_region = self._generate_face_region_attention(pil_image, heatmap)

        return {
            "success": True,
            "predicted_class": predicted,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.asd_classes[0]: round(prob_asd, 4),
                self.asd_classes[1]: round(prob_non_asd, 4),
            },
            "model_type": self.model_type,
            "xai": {
                "heatmap":      heatmap_b64,
                "overlay":      overlay_b64,
                "face_regions": face_region,
                "original":     self._original_to_b64(pil_image),
                "explanation":  self._asd_explanation(predicted, confidence),
                "method":       "Grad-CAM + MediaPipe Face Regions",
                "target_layer": self._asd_xai_layer,
            },
        }

    def predict_emotion_with_xai(self, image_input) -> Dict:
        """
        Emotion prediction + Grad-CAM XAI heatmap.
        XAI output is ALWAYS present in the result.
        """
        if self.emotion_model is None:
            raise ValueError(f"Emotion model not loaded: {self.emotion_model_path}")

        img_array, pil_image = self.preprocess(image_input)

        # ΓöÇΓöÇ Prediction
        raw = self.emotion_model.predict(img_array, verbose=0)[0]

        pred_idx    = int(np.argmax(raw))
        pred_emo    = (self.emotion_classes[pred_idx]
                       if pred_idx < len(self.emotion_classes)
                       else f"class_{pred_idx}")
        confidence  = float(raw[pred_idx])

        probs = {
            (self.emotion_classes[i] if i < len(self.emotion_classes) else f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(raw)
        }
        top_emotions = [
            {"emotion": e, "probability": round(p, 4)}
            for e, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        # ΓöÇΓöÇ XAI: Grad-CAM
        heatmap = self._generate_gradcam(
            self.emotion_model, img_array, self._emotion_xai_layer, is_binary=False
        )
        heatmap_b64, overlay_b64 = self._heatmap_to_outputs(heatmap, pil_image)

        # ΓöÇΓöÇ XAI: MediaPipe face-region attention
        face_region = self._generate_face_region_attention(pil_image, heatmap)

        return {
            "success": True,
            "predicted_emotion": pred_emo,
            "confidence": round(confidence, 4),
            "probabilities": probs,
            "top_emotions": top_emotions,
            "model_type": self.model_type,
            "xai": {
                "heatmap":      heatmap_b64,
                "overlay":      overlay_b64,
                "face_regions": face_region,
                "original":     self._original_to_b64(pil_image),
                "explanation":  self._emotion_explanation(pred_emo, confidence, top_emotions),
                "method":       "Grad-CAM + MediaPipe Face Regions",
                "target_layer": self._emotion_xai_layer,
            },
        }

    # Lazy-load Haar cascade once per instance
    _face_cascade = None

    @property
    def face_cascade(self):
        if CombinedModelLoader._face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            CombinedModelLoader._face_cascade = cv2.CascadeClassifier(cascade_path)
        return CombinedModelLoader._face_cascade

    def _crop_face(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Detect the largest face and return a tight crop (with 20% padding).
        Falls back to the full image if no face is detected.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return img_bgr  # no face found ΓÇö use full frame as fallback

        # Pick the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.20)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        return img_bgr[y1:y2, x1:x2]

    def predict_emotion_fast(self, image_input) -> Dict:
        """
        Lightweight emotion-only prediction with face detection ΓÇö NO XAI, NO Grad-CAM.
        Crops the face before inference so the model sees only the face, not the background.
        Use this for real-time frame capture (gamification) where speed matters.
        """
        if self.emotion_model is None:
            raise ValueError("Emotion model not loaded")

        # Decode to BGR numpy (needed for face detection)
        if isinstance(image_input, bytes):
            nparr  = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise ValueError("image_input must be bytes or numpy array")

        # ΓöÇΓöÇ Crop face before feeding to model ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        face_bgr  = self._crop_face(img_bgr)
        face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(face_rgb, (self.img_size, self.img_size))
        arr       = resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(arr, axis=0)

        # ΓöÇΓöÇ Inference ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        raw = self.emotion_model.predict(img_array, verbose=0)[0]

        pred_idx   = int(np.argmax(raw))
        pred_emo   = (self.emotion_classes[pred_idx]
                      if pred_idx < len(self.emotion_classes)
                      else f"class_{pred_idx}")
        confidence = float(raw[pred_idx])
        probs = {
            (self.emotion_classes[i] if i < len(self.emotion_classes) else f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(raw)
        }
        top_emotions = [
            {"emotion": e, "probability": round(p, 4)}
            for e, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        return {
            "success": True,
            "predicted_emotion": pred_emo,
            "confidence": round(confidence, 4),
            "probabilities": probs,
            "top_emotions": top_emotions,
            "model_type": self.model_type,
        }

    def predict_combined_with_xai(self, image_input) -> Dict:
        """
        Run ASD + Emotion prediction together, both with Grad-CAM XAI.
        Preprocesses the image ONCE for efficiency.
        XAI is ALWAYS included in both sub-results.
        """
        logger.info(f"CombinedModelLoader: Running prediction using model type '{self.model_type}'")
        if self.asd_model is None or self.emotion_model is None:
            raise ValueError("Both ASD and Emotion models must be loaded")

        img_array, pil_image = self.preprocess(image_input)

        # ΓöÇΓöÇ ASD prediction (STEP 1)
        asd_raw      = self.asd_model.predict(img_array, verbose=0)
        # Sigmoid > 0.5 ΓåÆ Non_ASD (class index 1 in alphabetical sort); <= 0.5 ΓåÆ ASD
        sigmoid_val   = float(asd_raw[0][0])
        prob_non_asd  = sigmoid_val
        prob_asd      = 1.0 - sigmoid_val
        asd_predicted = self.asd_classes[1] if sigmoid_val > 0.5 else self.asd_classes[0]
        asd_confidence = max(prob_asd, prob_non_asd)

        # ΓöÇΓöÇ Emotion prediction (STEP 2 ΓÇö runs on same image regardless of ASD result)
        emo_raw      = self.emotion_model.predict(img_array, verbose=0)[0]
        pred_idx     = int(np.argmax(emo_raw))
        pred_emo     = (self.emotion_classes[pred_idx]
                        if pred_idx < len(self.emotion_classes)
                        else f"class_{pred_idx}")
        emo_confidence = float(emo_raw[pred_idx])
        probs = {
            (self.emotion_classes[i] if i < len(self.emotion_classes) else f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(emo_raw)
        }
        top_emotions = [
            {"emotion": e, "probability": round(p, 4)}
            for e, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        ]

        # ΓöÇΓöÇ XAI: Grad-CAM for both models
        asd_heatmap = self._generate_gradcam(
            self.asd_model, img_array, self._asd_xai_layer, is_binary=True
        )
        emo_heatmap = self._generate_gradcam(
            self.emotion_model, img_array, self._emotion_xai_layer, is_binary=False
        )

        asd_hm_b64,  asd_ov_b64  = self._heatmap_to_outputs(asd_heatmap, pil_image)
        emo_hm_b64,  emo_ov_b64  = self._heatmap_to_outputs(emo_heatmap, pil_image)

        # ΓöÇΓöÇ Face region attention (MediaPipe) for both models
        asd_face_region = self._generate_face_region_attention(pil_image, asd_heatmap)
        emo_face_region = self._generate_face_region_attention(pil_image, emo_heatmap)

        original_b64 = self._original_to_b64(pil_image)

        return {
            "success": True,
            "model_type": self.model_type,
            "original_image": original_b64,
            "asd": {
                "success": True,
                "predicted_class": asd_predicted,
                "confidence": round(asd_confidence, 4),
                "probabilities": {
                    self.asd_classes[0]: round(prob_asd, 4),
                    self.asd_classes[1]: round(prob_non_asd, 4),
                },
                "model_type": self.model_type,
                "xai": {
                    "heatmap":      asd_hm_b64,
                    "overlay":      asd_ov_b64,
                    "original":     original_b64,
                    "face_regions": asd_face_region,
                    "explanation":  self._asd_explanation(asd_predicted, asd_confidence),
                    "method":       "Grad-CAM",
                    "target_layer": self._asd_xai_layer,
                },
            },
            "emotion": {
                "success": True,
                "predicted_emotion": pred_emo,
                "confidence": round(emo_confidence, 4),
                "probabilities": probs,
                "top_emotions": top_emotions,
                "model_type": self.model_type,
                "xai": {
                    "heatmap":      emo_hm_b64,
                    "overlay":      emo_ov_b64,
                    "original":     original_b64,
                    "face_regions": emo_face_region,
                    "explanation":  self._emotion_explanation(pred_emo, emo_confidence, top_emotions),
                    "method":       "Grad-CAM",
                    "target_layer": self._emotion_xai_layer,
                },
            },
        }

    # ΓöÇΓöÇ Legacy compatibility ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def predict_asd(self, image_input) -> Dict:       # noqa: F811
        return self.predict_asd_with_xai(image_input)

    def predict_emotion(self, image_input) -> Dict:   # noqa: F811
        return self.predict_emotion_with_xai(image_input)

    def predict_combined(self, image_input) -> Dict:  # noqa: F811
        return self.predict_combined_with_xai(image_input)

    @property
    def is_ready(self) -> bool:
        return self.asd_model is not None and self.emotion_model is not None


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
def load_best_available_model(
    trained_models_dir: str = "trained_models",
) -> Optional[CombinedModelLoader]:
    """
    Auto-detect and load the best available trained model pair.
    Priority: resnet50v2 > inceptionv3 > resnet50 > vgg19 > vgg16
    """
    priority = [
        ("resnet50v2",  "resnet50v2_asd_model.h5",   "resnet50v2_emotion_model.h5",  224),
        ("inceptionv3", "inceptionv3_asd_model.h5",  "inceptionv3_emotion_model.h5", 224),
        ("resnet50",    "resnet50_asd_model.h5",      "resnet50_emotion_model.h5",    224),
        ("vgg19",       "vgg19_asd_model.h5",         "vgg19_emotion_model.h5",       224),
        ("vgg16",       "vgg16_asd_model.h5",         "vgg16_emotion_model.h5",       224),
    ]

    for model_type, asd_file, emo_file, img_size in priority:
        asd_path = os.path.join(trained_models_dir, asd_file)
        emo_path = os.path.join(trained_models_dir, emo_file)

        if not (os.path.exists(asd_path) and os.path.exists(emo_path)):
            continue

        asd_lbl = os.path.join(trained_models_dir, f"{model_type}_asd_labels.json")
        emo_lbl = os.path.join(trained_models_dir, f"{model_type}_emotion_labels.json")

        try:
            logger.info(f"Auto-loading {model_type.upper()} model pairΓÇª")
            loader = CombinedModelLoader(
                asd_model_path=asd_path,
                emotion_model_path=emo_path,
                model_type=model_type,
                asd_labels_path=asd_lbl     if os.path.exists(asd_lbl) else None,
                emotion_labels_path=emo_lbl if os.path.exists(emo_lbl) else None,
                img_size=img_size,
            )
            if loader.is_ready:
                logger.info(f"Γ£à Loaded {model_type.upper()} ΓÇö "
                            f"ASD XAI layer: {loader._asd_xai_layer} | "
                            f"Emotion XAI layer: {loader._emotion_xai_layer}")
                return loader
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            continue

    logger.warning("ΓÜá No trained model pairs found in trained_models/")
    return None
