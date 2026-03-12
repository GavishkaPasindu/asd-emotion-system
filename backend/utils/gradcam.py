"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
For Explainable AI visualization of model decisions
"""

import tensorflow as tf
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64


class GradCAM:
    """Grad-CAM implementation for both TensorFlow and PyTorch models"""
    
    @staticmethod
    def generate_gradcam_tensorflow(model, img_array, layer_name=None):
        """
        Generate Grad-CAM heatmap for TensorFlow model
        
        Args:
            model: TensorFlow model
            img_array: Preprocessed image array
            layer_name: Target layer name (auto-detect if None)
        
        Returns:
            Heatmap as numpy array
        """
        try:
            # Find last conv layer if not specified
            if layer_name is None:
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        layer_name = layer.name
                        break
            
            if layer_name is None:
                raise ValueError("No convolutional layer found")
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(layer_name).output, model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                # For binary classification
                if predictions.shape[-1] == 1:
                    loss = predictions[:, 0]
                else:
                    # For multi-class
                    class_idx = tf.argmax(predictions[0])
                    loss = predictions[:, class_idx]
            
            # Get gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            
            
            # Normalize
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            else:
                print("Warning: Grad-CAM heatmap has 0 max value (TF)")
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Grad-CAM TF error: {e}")
            return None
    
    @staticmethod
    def generate_gradcam_pytorch(model, img_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for PyTorch model
        
        Args:
            model: PyTorch model
            img_tensor: Preprocessed image tensor
            target_class: Target class index (auto-detect if None)
        
        Returns:
            Heatmap as numpy array
        """
        try:
            model.eval()
            
            # Find last conv layer
            last_conv = None
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            
            if last_conv is None:
                raise ValueError("No convolutional layer found")
            
            # Hook to capture gradients and activations
            gradients = []
            activations = []
            
            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])
            
            def forward_hook(module, input, output):
                activations.append(output)
            
            # Register hooks
            backward_handle = last_conv.register_full_backward_hook(backward_hook)
            forward_handle = last_conv.register_forward_hook(forward_hook)
            
            # Forward pass
            img_tensor.requires_grad = True
            output = model(img_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            model.zero_grad()
            class_score = output[0, target_class]
            class_score.backward()
            
            # Remove hooks
            backward_handle.remove()
            forward_handle.remove()
            
            # Get gradients and activations
            grads = gradients[0].cpu().data.numpy()[0]
            acts = activations[0].cpu().data.numpy()[0]
            
            # Global average pooling of gradients
            weights = np.mean(grads, axis=(1, 2))
            
            # Weighted combination
            heatmap = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * acts[i]
            
            # ReLU and normalize
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            else:
                print("Warning: Grad-CAM heatmap has 0 max value (PyTorch)")
            
            return heatmap
            
        except Exception as e:
            print(f"Grad-CAM PyTorch error: {e}")
            return None
    
    @staticmethod
    def create_heatmap_overlay(original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Create overlay of heatmap on original image
        
        Args:
            original_image: Original image (PIL or numpy)
            heatmap: Grad-CAM heatmap
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            Overlayed image as numpy array
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(original_image, Image.Image):
                original_image = np.array(original_image)
            
            # Ensure RGB
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            elif original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
            
            # Resize heatmap to match image
            h, w = original_image.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # Convert to uint8
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlayed = cv2.addWeighted(
                original_image.astype(np.float32), 1 - alpha,
                heatmap_colored.astype(np.float32), alpha,
                0
            ).astype(np.uint8)
            
            return overlayed
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return original_image
    
    @staticmethod
    def heatmap_to_base64(heatmap, colormap=cv2.COLORMAP_JET):
        """Convert heatmap to base64 encoded image"""
        try:
            # Convert to uint8
            heatmap_uint8 = np.uint8(255 * heatmap)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL
            pil_image = Image.fromarray(heatmap_colored)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"Base64 conversion error: {e}")
            return None
    
    @staticmethod
    def image_to_base64(image):
        """Convert numpy array or PIL image to base64"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"Image to base64 error: {e}")
            return None
