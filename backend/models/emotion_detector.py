"""
Improved Emotion Recognition Model with Enhanced Accuracy
Supports PyTorch models with ensemble predictions and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import os


class EmotionCNN(nn.Module):
    """CNN architecture for emotion recognition (matches training script)"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EmotionDetector:
    def __init__(self, model_path, device=None):
        """Initialize emotion detector"""
        self.model_path = model_path
        self.img_size = 128
        self.class_names = ['Natural', 'anger', 'fear', 'joy', 'sadness', 'surprise']
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = EmotionCNN(len(self.class_names)).to(self.device)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state'])
                    if 'classes' in checkpoint:
                        self.class_names = checkpoint['classes']
                    print(f"Emotion Model loaded from {model_path}")
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"Emotion Model loaded (state dict)")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using newly initialized model")
        else:
            print(f"Model not found at {model_path}, using new model")
        
        self.model.eval()
    
    def preprocess_image(self, image_data, augment=False):
        """Preprocess image for model input"""
        # Convert to PIL Image if bytes
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize
        img_resized = cv2.resize(img_array, (self.img_size, self.img_size))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.5) / 0.5  # [-1, 1]
        
        # Data augmentation for ensemble
        if augment:
            augmented_images = [img_normalized]
            
            # Horizontal flip
            augmented_images.append(np.fliplr(img_normalized).copy())
            
            # Slight rotations
            center = (self.img_size // 2, self.img_size // 2)
            for angle in [-3, 3]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img_normalized, M, (self.img_size, self.img_size))
                augmented_images.append(rotated)
            
            # Convert to tensors
            tensors = []
            for img in augmented_images:
                tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                tensors.append(tensor)
            
            return torch.stack(tensors).to(self.device)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        return img_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_data, use_ensemble=True):
        """
        Predict emotion with improved accuracy
        
        Args:
            image_data: Image bytes or PIL Image
            use_ensemble: Use test-time augmentation
        
        Returns:
            dict with prediction results
        """
        try:
            self.model.eval()
            
            with torch.no_grad():
                if use_ensemble:
                    img_batch = self.preprocess_image(image_data, augment=True)
                    outputs = self.model(img_batch)
                    probabilities = F.softmax(outputs, dim=1)
                    # Average predictions
                    avg_probs = probabilities.mean(dim=0)
                    confidence, predicted_idx = torch.max(avg_probs, 0)
                else:
                    img_batch = self.preprocess_image(image_data, augment=False)
                    outputs = self.model(img_batch)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    avg_probs = probabilities.squeeze()
            
            # Get results
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            all_probs = avg_probs.cpu().numpy()
            
            # Create probability dictionary
            prob_dict = {
                self.class_names[i]: float(all_probs[i])
                for i in range(len(self.class_names))
            }
            
            # Get top 3 emotions
            top_indices = np.argsort(all_probs)[-3:][::-1]
            top_emotions = [
                {
                    'emotion': self.class_names[idx],
                    'probability': float(all_probs[idx])
                }
                for idx in top_indices
            ]
            
            return {
                'success': True,
                'predicted_emotion': predicted_class,
                'confidence': float(confidence_score),
                'probabilities': prob_dict,
                'top_emotions': top_emotions,
                'all_probabilities': all_probs.tolist()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_attention_map(self, image_data):
        """Get attention map for visualization"""
        try:
            self.model.eval()
            img_batch = self.preprocess_image(image_data, augment=False)
            
            # Get feature maps from last conv layer
            features = None
            def hook_fn(module, input, output):
                nonlocal features
                features = output
            
            # Register hook on last conv layer
            last_conv = None
            for module in self.model.features.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            
            if last_conv:
                handle = last_conv.register_forward_hook(hook_fn)
                
                with torch.no_grad():
                    _ = self.model(img_batch)
                
                handle.remove()
                
                if features is not None:
                    # Average across channels
                    attention = features.mean(dim=1).squeeze().cpu().numpy()
                    
                    return {
                        'success': True,
                        'attention_map': attention
                    }
            
            return {
                'success': False,
                'error': 'Could not extract attention map'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
