import tensorflow as tf
from typing import Tuple
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import io
import os
import base64

class ResNetEmotionDetector:
    def __init__(self, model_path=None):
        """Initialize the ResNet50V2 emotion detector"""
        self.img_size = (224, 224)
        self.class_names = ['anger', 'fear', 'joy', 'sadness']
        
        # Load architecture
        self.base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self._xai_layer = 'conv5_block3_out' # Target last conv of ResNet50V2
        
        # Define the classification head
        self.emotion_model = Sequential([
            Flatten(input_shape=self.base_model.output_shape[1:]),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        if model_path and os.path.exists(model_path):
            try:
                self.emotion_model.load_weights(model_path)
                print(f"ResNet Emotion Model weights loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        
        # Build a full functional model for Grad-CAM
        # This connects the base model and the head
        self.full_model = Model(
            inputs=self.base_model.input, 
            outputs=self.emotion_model(self.base_model.output)
        )

    def preprocess_image(self, image_data) -> Tuple[np.ndarray, Image.Image]:
        """Preprocess image for model input. Returns (img_array, pil_image)"""
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif hasattr(image_data, 'convert'): # PIL Image
            img_rgb = np.array(image_data.convert('RGB'))
        else:
            img_rgb = image_data
        
        pil_image = Image.fromarray(img_rgb)
        img_resized = cv2.resize(img_rgb, self.img_size)
        
        # Convert to numpy array and preprocess for ResNetV2 (standardizes to [-1, 1])
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array, pil_image

    def predict(self, image_data, **kwargs):
        """Predict emotion using ResNet50V2 + Dense head"""
        try:
            img_array, _ = self.preprocess_image(image_data)
            prediction = self.full_model.predict(img_array, verbose=0)[0]
            
            predicted_idx = np.argmax(prediction)
            predicted_class = self.class_names[predicted_idx]
            confidence_score = float(prediction[predicted_idx])
            
            prob_dict = {name: float(prediction[i]) for i, name in enumerate(self.class_names)}
            top_emotions = [
                {'emotion': self.class_names[idx], 'probability': float(prediction[idx])}
                for idx in np.argsort(prediction)[::-1]
            ]
            
            return {
                'success': True,
                'predicted_emotion': predicted_class,
                'confidence': round(confidence_score, 4),
                'probabilities': prob_dict,
                'top_emotions': top_emotions
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_with_xai(self, image_data) -> dict:
        """Predict emotion + Grad-CAM XAI"""
        try:
            img_array, pil_image = self.preprocess_image(image_data)
            prediction = self.full_model.predict(img_array, verbose=0)[0]
            
            predicted_idx = np.argmax(prediction)
            pred_emo = self.class_names[predicted_idx]
            confidence = float(prediction[predicted_idx])

            # Generate Heatmap
            target_layer = self.base_model.get_layer(self._xai_layer)
            grad_model = Model([self.full_model.inputs], [target_layer.output, self.full_model.output])
            
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_array)
                loss = preds[:, predicted_idx]
            
            grads = tape.gradient(loss, conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
            conv_out = conv_out[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
            heatmap = tf.nn.relu(heatmap)
            
            if tf.reduce_max(heatmap) > 0:
                heatmap = (heatmap / tf.reduce_max(heatmap)).numpy()
            else:
                heatmap = heatmap.numpy()

            # Encode outputs
            heatmap_b64, overlay_b64 = self._heatmap_to_outputs(heatmap, pil_image)
            
            return {
                'success': True,
                'predicted_emotion': pred_emo,
                'confidence': round(confidence, 4),
                'probabilities': {n: float(prediction[i]) for i, n in enumerate(self.class_names)},
                'xai': {
                    'heatmap': heatmap_b64,
                    'overlay': overlay_b64,
                    'original': self._original_to_b64(pil_image),
                    'explanation': f"Detected {pred_emo} ({confidence*100:.1f}%). Heatmap shows high-salience areas.",
                    'method': 'Grad-CAM',
                    'target_layer': self._xai_layer
                }
            }
        except Exception as e:
            return {'success': False, 'error': f"XAI prediction failed: {e}"}

    def _heatmap_to_outputs(self, heatmap, pil_image):
        orig_arr = np.array(pil_image)
        h, w = orig_arr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
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
