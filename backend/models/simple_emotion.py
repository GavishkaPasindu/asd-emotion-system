
import cv2
import numpy as np
from fer.fer import FER
import logging
import io
from PIL import Image

logger = logging.getLogger(__name__)

class SimpleEmotionDetector:
    def __init__(self):
        """Initialize the FER (Face Emotion Recognition) detector"""
        try:
            # Initialize FER with default mtcnn=False for speed (uses OpenCV haarcascade)
            # or mtcnn=True for better accuracy properly
            self.detector = FER(mtcnn=True) 
            logger.info("FER Emotion Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FER: {e}")
            self.detector = None

    def predict(self, image_input):
        """
        Predict emotion from image
        Args:
            image_input: bytes or numpy array
        Returns:
            dict: {success, predicted_emotion, confidence, emotions_score}
        """
        if self.detector is None:
            return {"success": False, "error": "Model not initialized"}

        try:
            # Convert bytes to numpy array if needed
            if isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image_input

            if img is None:
                return {"success": False, "error": "Invalid image data"}

            # FER requires RGB
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # FER uses OpenCV which is BGR usually, but let's double check usage. 
            # FER internal implementation often expects array. 
            # safe to pass the image read by opencv directly usually.

            # Get top emotion
            # detect_emotions returns list of dicts: [{'box': (x, y, w, h), 'emotions': {'angry': 0.0, ...}}]
            analysis = self.detector.detect_emotions(img)

            if not analysis:
                return {
                    "success": True,
                    "predicted_emotion": "Neutral", # Fallback
                    "confidence": 0.0,
                    "emotions_score": {}
                }

            # Get the first face found
            first_face = analysis[0]
            emotions = first_face['emotions']
            
            # Find dominant emotion
            dominant_emotion, confidence = max(emotions.items(), key=lambda item: item[1])

            return {
                "success": True,
                "predicted_emotion": dominant_emotion,
                "confidence": float(confidence),
                "emotions_score": emotions
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"success": False, "error": str(e)}
