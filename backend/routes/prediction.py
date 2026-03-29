"""
Prediction API Routes -- XAI ALWAYS INCLUDED
============================================
XAI (Grad-CAM heatmaps) is included in EVERY prediction response.
It is NOT optional and NOT silently dropped.
"""

from flask import Blueprint, request, jsonify
import numpy as np
import cv2

prediction_bp = Blueprint('prediction', __name__)

from models.manager import model_manager

# CombinedModelLoader instance -- fallback if manager not used correctly
combined_model = None

def init_combined_model(model):
    """Fallback initialization."""
    global combined_model
    combined_model = model

def _get_active_model():
    """Dynamically resolve the model based on X-Model-Type header."""
    requested_model = request.headers.get('X-Model-Type')
    model = model_manager.get_model(requested_model)
    print(f"DEBUG: Prediction route - Requested: {requested_model}, Resolved: {model.model_type if model else 'None'}")
    if model is None:
        # Final fallback to whatever was initialized
        return combined_model
    return model

def _require_model():
    model = _get_active_model()
    if model is None:
        return jsonify({
            'success': False,
            'error': 'No model loaded. Please check backend/trained_models/'
        }), 503
    return None


def _read_image():
    """Read and validate image from request. Returns (image_bytes, error_response)."""
    if 'image' not in request.files:
        print("DEBUG: No 'image' key in request.files")
        return None, (jsonify({'success': False, 'error': 'No image file provided'}), 400)
    
    file = request.files['image']
    if file.filename == '':
        print("DEBUG: Empty filename in request.files['image']")
        return None, (jsonify({'success': False, 'error': 'Empty filename'}), 400)
    
    image_bytes = file.read()
    has_face = _has_face(image_bytes)
    if not has_face:
        print("DEBUG: Face detection found 0 faces -- rejecting request")
        return None, (jsonify({
            'success': False, 
            'error': 'No face detected in the image. Please upload a clear photo of a person\'s face for accurate analysis.'
        }), 400)
    
    print(f"DEBUG: Image read successfully, size: {len(image_bytes)} bytes")
    return image_bytes, None


def _has_face(image_bytes: bytes) -> bool:
    """Strict face detection: MediaPipe (Primary) + Stricter Haar Fallback + Pattern Rejection."""
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
            import os

            model_path = os.path.join(os.getcwd(), 'trained_models', 'face_landmarker.task')
            if not os.path.exists(model_path):
                model_path = '/tmp/face_landmarker.task'

            if os.path.exists(model_path):
                base_opts = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    num_faces=1,
                    min_face_detection_confidence=0.55, # Increased from 0.35 to avoid patterns
                )
                with mp_vision.FaceLandmarker.create_from_options(options) as detector:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    detection_result = detector.detect(mp_image)
                    if detection_result.face_landmarks:
                        print("DEBUG: MediaPipe confirmed high-confidence face.")
                        return True
        except Exception as e:
            print(f"DEBUG: MediaPipe stage failed: {e}")

        # --- Stage 2: Stricter Haar Cascade + Pattern Rejection ---
        cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Stricter parameters: higher minNeighbors and larger minSize
        faces = cc.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6,   # Require higher consensus (increased from 4)
            minSize=(100, 100) # Require decent face size for clinical analysis
        )
        
        # --- Pattern Rejection Logic ---
        # If too many small repetitive objects are found (like mat circles), reject the image.
        if len(faces) > 4:
            print(f"DEBUG: Pattern Rejection Triggered! Found {len(faces)} repetitive objects.")
            return False

        if len(faces) > 0:
            print(f"DEBUG: Strict Haar Cascade detected {len(faces)} face(s).")
            return True

        # --- Stage 3: Strict Profile Cascade ---
        pc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        profiles = pc.detectMultiScale(gray, 1.1, 6, (100, 100))
        if len(profiles) > 0:
            print(f"DEBUG: Strict Profile Cascade detected {len(profiles)} face(s).")
            return True

        print("DEBUG: Image rejected - no clear face detected.")
        return False

    except Exception as e:
        print(f"Critical face detection error: {e}")
        return False


# ------------------------------------------------------------------------------
@prediction_bp.route('/api/predict/asd', methods=['POST'])
def predict_asd():
    """
    ASD Detection Endpoint
    --------------------─
    POST /api/predict/asd
    Form-data: image (file)

    Response always includes:
      - predicted_class, confidence, probabilities
      - xai.heatmap   (base64 PNG -- JET colormap Grad-CAM)
      - xai.overlay   (base64 PNG -- heatmap overlaid on face)
      - xai.explanation (human-readable text)
      - xai.target_layer (which conv layer Grad-CAM used)
    """
    err = _require_model()
    if err:
        return err

    image_bytes, read_err = _read_image()
    if read_err:
        return read_err

    try:
        model = _get_active_model()
        result = model.predict_asd_with_xai(image_bytes)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500


# ------------------------------------------------------------------------------
@prediction_bp.route('/api/predict/emotion', methods=['POST'])
def predict_emotion():
    """
    Emotion Detection Endpoint
    --------------------------
    POST /api/predict/emotion
    Form-data: image (file)

    Response always includes:
      - predicted_emotion, confidence, probabilities, top_emotions
      - xai.heatmap, xai.overlay, xai.explanation, xai.target_layer
    """
    err = _require_model()
    if err:
        return err

    image_bytes, read_err = _read_image()
    if read_err:
        return read_err

    try:
        model = _get_active_model()
        result = model.predict_emotion_with_xai(image_bytes)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500


# ------------------------------------------------------------------------------
@prediction_bp.route('/api/predict/combined', methods=['POST'])
def predict_combined():
    """
    Combined ASD + Emotion Detection Endpoint  ← MAIN ENDPOINT
    ------------------------------------------
    POST /api/predict/combined
    Form-data: image (file)

    Response structure:
    {
      "success": true,
      "model_type": "resnet50v2",
      "original_image": "data:image/png;base64,...",
      "asd": {
        "predicted_class": "ASD",
        "confidence": 0.93,
        "probabilities": { "ASD": 0.93, "Non_ASD": 0.07 },
        "xai": {
          "heatmap":      "data:image/png;base64,...",   ← always present
          "overlay":      "data:image/png;base64,...",   ← always present
          "original":     "data:image/png;base64,...",
          "explanation":  "The model detected ASD traits...",
          "method":       "Grad-CAM",
          "target_layer": "conv5_block3_out"
        }
      },
      "emotion": {
        "predicted_emotion": "happy",
        "confidence": 0.88,
        "probabilities": { ... },
        "top_emotions": [ ... ],
        "xai": { ... }    ← always present
      }
    }
    """
    err = _require_model()
    if err:
        return err

    image_bytes, read_err = _read_image()
    if read_err:
        return read_err

    try:
        model = _get_active_model()
        result = model.predict_combined_with_xai(image_bytes)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Combined prediction error: {str(e)}'}), 500


# /api/models/metrics is handled by app.py (get_model_metrics)


def validate_face(image_bytes):
    """
    Check if image contains a face using Haar Cascade
    Returns: True if face detected, False otherwise
    """
    try:
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return len(faces) > 0
    except Exception as e:
        print(f"Face detection error: {e}")
        # Fail safe: return True if detection fails (assume it's a valid image that we just couldn't process)
        return True



# ----------------------------------------------------------------------------─
# Helper explanation functions are already defined in CombinedModelLoader,
# so these redundant ones can be removed or kept as local utilities if needed.
# For simplicity and to avoid confusion, we consolidate everything into the class.
