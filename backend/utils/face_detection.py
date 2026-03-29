import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def has_face(image_bytes: bytes) -> bool:
    """
    Multi-stage face detection for high reliability:
    1. MediaPipe AI (Primary - high accuracy)
    2. Strict Frontal Haar Cascade (Secondary - fast)
    3. Profile/Tilt Haar Cascade (Tertiary - handles angles)
    
    Includes Pattern Rejection to eliminate repetitive circular patterns (mats, etc.)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- Stage 1: Stricter MediaPipe Face Detection ---
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = os.path.join(os.getcwd(), 'trained_models', 'face_landmarker.task')
            if not os.path.exists(model_path):
                # Fallback to /tmp if not in project folder
                model_path = '/tmp/face_landmarker.task'

            if os.path.exists(model_path):
                base_opts = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    num_faces=1,
                    min_face_detection_confidence=0.55, # High confidence required
                )
                with mp_vision.FaceLandmarker.create_from_options(options) as detector:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    detection_result = detector.detect(mp_image)
                    if detection_result.face_landmarks:
                        logger.info("Face detected by MediaPipe AI.")
                        return True
        except Exception as e:
            logger.warning(f"MediaPipe detection skipped: {e}")

        # --- Stage 2: Stricter Haar Cascade + Pattern Rejection ---
        # Frontal face cascade
        cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Stricter parameters: higher minNeighbors and larger minSize (100x100)
        faces = cc.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6,   # Require higher consensus
            minSize=(100, 100) # Require decent face size
        )
        
        # --- Pattern Rejection Logic ---
        # If too many repetitive objects are found, it's likely noise/mat circles.
        if len(faces) > 4:
            logger.info(f"Pattern Rejection Triggered! Found {len(faces)} repetitive objects.")
            return False

        if len(faces) > 0:
            logger.info(f"Frontal face detected by Haar Cascade ({len(faces)} found).")
            return True

        # --- Stage 3: Strict Profile Cascade ---
        # Profile face cascade (for side views or tilts)
        pc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        profiles = pc.detectMultiScale(gray, 1.1, 6, (100, 100))
        if len(profiles) > 0:
            logger.info(f"Profile/Tilted face detected by Haar Cascade.")
            return True

        logger.info("Image rejected - no clear face detected in any stage.")
        return False

    except Exception as e:
        logger.error(f"Critical face detection error: {e}")
        return False
