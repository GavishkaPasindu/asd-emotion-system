"""
Gamification Routes
Handles specific gamification features for ASD screening and emotion regulation
"""

from flask import Blueprint, request, jsonify
import random
import uuid
import base64
import cv2
import numpy as np
import os
import json
from datetime import datetime
import logging
from PIL import Image
import io

from models.manager import model_manager
from utils.face_detection import has_face

# Configure logger
logger = logging.getLogger(__name__)

gamification_bp = Blueprint('gamification', __name__)

# Note: asd_detector and emotion_detector globals are deprecated in favor of model_manager

# File for session persistence
SESSIONS_FILE = 'sessions.json'

def init_models(asd_model, emotion_model):
    """Legacy compatibility placeholder."""
    pass

def load_sessions():
    """Load sessions from file"""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return {}
    return {}

def save_sessions(sessions):
    """Save sessions to file"""
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving sessions: {e}")

# Load sessions on startup
active_sessions = load_sessions()

# Curated calming YouTube videos
CALMING_VIDEOS = {
    'anger': [
        {
            'id': 'lFcSrYw-ARY',
            'title': 'Peaceful Piano Music - Relaxing Music for Stress Relief',
            'duration': '3:00:00',
            'description': 'Calm piano melodies to reduce anger and stress'
        },
        {
            'id': 'jfKfPfyJRdk',
            'title': 'Relaxing Music with Nature Sounds - Waterfall',
            'duration': '3:00:00',
            'description': 'Soothing nature sounds for emotional regulation'
        },
        {
            'id': '1ZYbU82GVz4',
            'title': 'Deep Breathing Exercise - Calm Down',
            'duration': '5:00',
            'description': 'Guided breathing to manage anger'
        },
        {
            'id': 'inpok4MKVLM',
            'title': 'Ocean Waves - Calming Sounds',
            'duration': '1:00:00',
            'description': 'Peaceful ocean waves for relaxation'
        }
    ],
    'sadness': [
        {
            'id': 'lTRiuFIWV54',
            'title': 'Uplifting Music - Positive Energy',
            'duration': '1:00:00',
            'description': 'Cheerful music to lift your mood'
        },
        {
            'id': 'Jyy0ra2WcQQ',
            'title': 'Happy Background Music',
            'duration': '30:00',
            'description': 'Bright, positive instrumental music'
        }
    ],
    'fear': [
        {
            'id': 'ZToicYcHIOU',
            'title': 'Meditation for Anxiety Relief',
            'duration': '10:00',
            'description': 'Guided meditation to reduce fear and anxiety'
        },
        {
            'id': 'O-6f5wQXSu8',
            'title': 'Peaceful Music - Safe Space',
            'duration': '1:00:00',
            'description': 'Gentle music creating a sense of safety'
        }
    ],
    'neutral': [
        {
            'id': 'jfKfPfyJRdk',
            'title': 'Ambient Music for Focus',
            'duration': '2:00:00',
            'description': 'Background music for concentration'
        }
    ]
}

# -- ASD-friendly CARTOON videos ----------------------------------------------
# To customize: replace the 'id' string with the 11-character YouTube video ID
# For example, in https://www.youtube.com/watch?v=XqZsoesa55w, the ID is 'XqZsoesa55w'
CARTOON_VIDEOS = [
    {'id': 'XqZsoesa55w', 'title': 'Baby Shark Dance 🦈',        'category': 'cartoon'},
    {'id': 'FDxk0bmShBU', 'title': 'Peppa Pig Episodes 🐷',      'category': 'cartoon'},
]

# -- MUSIC videos ------------------------------------------------------------─
# To customize: replace the 'id' string with the 11-character YouTube video ID
MUSIC_VIDEOS = [
    {'id': 'jbBbRjs_niM', 'title': 'Wheels on the Bus 🚌',          'category': 'music'},
    {'id': 'n38kGst16sI', 'title': 'Twinkle Twinkle Little Star ⭐', 'category': 'music'},
]

# -- Puzzle themes (handled client-side) --------------------------------------
PUZZLE_THEMES = [
    {'id': 'animals', 'title': 'Animal Match 🐶',  'description': 'Match the animal grid!'},
    {'id': 'faces',   'title': 'Find the Face 😊', 'description': 'Find the correct emotion!'},
    {'id': 'fruits',  'title': 'Fruit Hunt 🍎',    'description': 'Find the hidden fruits!'},
]

# Legacy alias so existing /api/gamification/asd-videos still works
ASD_FRIENDLY_VIDEOS = CARTOON_VIDEOS


@gamification_bp.route('/api/gamification/check', methods=['POST'])
def check_emotion_trigger():
    """
    Check if emotion requires intervention (e.g., anger detected)
    Returns: Suggested action and video recommendations
    """
    try:
        data = request.get_json()
        
        if not data or 'emotion' not in data:
            return jsonify({
                'success': False,
                'error': 'Emotion data required'
            }), 400
        
        emotion = data['emotion'].lower()
        confidence = data.get('confidence', 0)
        
        # Determine if intervention is needed
        needs_intervention = False
        intervention_type = None
        message = ""
        
        # Robust mapping for custom labels
        # if the user has 'train' as happy or 'test' as sad, this would eventually go here
        
        if emotion in ['anger', 'angry'] and confidence > 0.5:
            needs_intervention = True
            intervention_type = 'calming_video'
            message = "We detected some frustration. Would you like to watch a calming video?"
        
        elif emotion in ['sadness', 'sad'] and confidence > 0.6:
            needs_intervention = True
            intervention_type = 'uplifting_video'
            message = "Feeling down? Let's watch something uplifting together!"
        
        elif emotion in ['fear', 'scared', 'anxious'] and confidence > 0.6:
            needs_intervention = True
            intervention_type = 'relaxation_video'
            message = "Take a deep breath. Would you like some relaxation exercises?"
        
        # Get video suggestions
        suggested_videos = []
        if needs_intervention:
            emotion_key = emotion if emotion in CALMING_VIDEOS else 'neutral'
            suggested_videos = CALMING_VIDEOS.get(emotion_key, [])
        elif confidence > 0.7:
             # Fallback: if high confidence but unknown emotion, suggest relaxing content anyway if requested
             pass

        return jsonify({
            'success': True,
            'needs_intervention': needs_intervention,
            'intervention_type': intervention_type,
            'message': message,
            'suggested_videos': suggested_videos,
            'emotion': emotion,
            'confidence': confidence
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gamification_bp.route('/api/gamification/videos', methods=['GET'])
def get_calming_videos():
    """
    Get list of calming videos by emotion category
    Query params: emotion (optional)
    """
    try:
        emotion = request.args.get('emotion', 'all').lower()
        
        if emotion == 'all':
            # Return all videos
            all_videos = []
            for category, videos in CALMING_VIDEOS.items():
                for video in videos:
                    video_copy = video.copy()
                    video_copy['category'] = category
                    all_videos.append(video_copy)
            
            return jsonify({
                'success': True,
                'videos': all_videos,
                'total': len(all_videos)
            }), 200
        
        elif emotion in CALMING_VIDEOS:
            videos = CALMING_VIDEOS[emotion]
            return jsonify({
                'success': True,
                'emotion': emotion,
                'videos': videos,
                'total': len(videos)
            }), 200
        
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown emotion category: {emotion}'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gamification_bp.route('/api/gamification/asd-videos', methods=['GET'])
def get_asd_friendly_videos():
    """
    Get ASD-friendly videos for screening sessions
    Query params: category (optional)
    """
    try:
        category = request.args.get('category', 'all').lower()
        
        if category == 'all':
            return jsonify({
                'success': True,
                'videos': ASD_FRIENDLY_VIDEOS,
                'total': len(ASD_FRIENDLY_VIDEOS),
                'categories': list(set(v['category'] for v in ASD_FRIENDLY_VIDEOS))
            }), 200
        
        # Filter by category
        filtered_videos = [v for v in ASD_FRIENDLY_VIDEOS if v['category'] == category]
        
        return jsonify({
            'success': True,
            'category': category,
            'videos': filtered_videos,
            'total': len(filtered_videos)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gamification_bp.route('/api/gamification/activities', methods=['GET'])
def get_activities():
    """
    Return all 4 gamification categories with their items.
    Used by the category picker on the frontend.
    """
    return jsonify({
        'success': True,
        'categories': [
            {
                'id': 'puzzle',
                'title': 'Puzzle Game',
                'emoji': '🧩',
                'description': 'Fun swap-puzzle -- match the pictures!',
                'color': 'from-indigo-500 to-purple-600',
                'items': PUZZLE_THEMES,
            },
            {
                'id': 'cartoon',
                'title': 'Cartoon Videos',
                'emoji': '📺',
                'description': 'Watch fun cartoons while we learn!',
                'color': 'from-pink-500 to-rose-500',
                'items': CARTOON_VIDEOS,
            },
            {
                'id': 'music',
                'title': 'Music Videos',
                'emoji': '🎵',
                'description': 'Sing along to favourite nursery rhymes!',
                'color': 'from-green-500 to-teal-500',
                'items': MUSIC_VIDEOS,
            },
            {
                'id': 'social',
                'title': 'Social Interaction',
                'emoji': '👤',
                'description': 'Meet a friendly virtual friend!',
                'color': 'from-sky-500 to-violet-500',
                'items': [{'id': 'default', 'title': 'Virtual Friend', 'description': 'Interactive social prompts'}],
            },
        ]
    })


@gamification_bp.route('/api/gamification/suggest', methods=['POST'])
def suggest_video():
    """
    Get a single suggested video based on emotion
    """
    try:
        data = request.get_json()
        
        if not data or 'emotion' not in data:
            return jsonify({
                'success': False,
                'error': 'Emotion required'
            }), 400
        
        emotion = data['emotion'].lower()
        
        # Get videos for emotion
        emotion_key = emotion if emotion in CALMING_VIDEOS else 'neutral'
        videos = CALMING_VIDEOS.get(emotion_key, [])
        
        if not videos:
            return jsonify({
                'success': False,
                'error': 'No videos available for this emotion'
            }), 404
        
        # Pick random video
        suggested_video = random.choice(videos)
        
        return jsonify({
            'success': True,
            'video': suggested_video,
            'emotion': emotion
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gamification_bp.route('/api/gamification/track-engagement', methods=['POST'])
def track_engagement():
    """
    Track user engagement with calming videos
    Stores: video watched, duration, emotion before/after
    """
    try:
        data = request.get_json()
        
        required_fields = ['video_id', 'watch_duration', 'emotion_before']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # In production, save to database
        # For now, just acknowledge
        engagement_data = {
            'video_id': data['video_id'],
            'watch_duration': data['watch_duration'],
            'emotion_before': data['emotion_before'],
            'emotion_after': data.get('emotion_after'),
            'completed': data.get('completed', False),
            'helpful': data.get('helpful', None)
        }
        
        # Calculate improvement score
        improvement_score = 0
        if engagement_data['emotion_after']:
            # Simple scoring: if emotion changed from negative to positive
            negative_emotions = ['anger', 'sadness', 'fear']
            positive_emotions = ['joy', 'Natural']
            
            if (engagement_data['emotion_before'] in negative_emotions and 
                engagement_data['emotion_after'] in positive_emotions):
                improvement_score = 100
            elif engagement_data['emotion_before'] != engagement_data['emotion_after']:
                improvement_score = 50
        
        return jsonify({
            'success': True,
            'message': 'Engagement tracked successfully',
            'improvement_score': improvement_score,
            'data': engagement_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@gamification_bp.route('/api/gamification/start-session', methods=['POST'])
def start_session():
    """Initialize a new screening session"""
    try:
        data = request.get_json()
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session data
        session_data = {
            "id": session_id,
            "started_at": datetime.now().isoformat(),
            "duration": data.get('duration', 10),  # Default 10 seconds
            "frames": [],
            "predictions": [],
            "status": "active",
            "metadata": {
                "child_age": data.get('child_age'),
                "session_type": data.get('session_type', 'screening'),
                "selected_video": data.get('selected_video')
            }
        }
        
        # Store session
        active_sessions[session_id] = session_data
        save_sessions(active_sessions)
        
        logger.info(f"Started new session: {session_id}")
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "duration": session_data["duration"],
            "message": "Session started successfully"
        })
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@gamification_bp.route('/api/gamification/capture-frame', methods=['POST'])
def capture_frame():
    """Process a video frame during screening session"""
    global asd_detector, emotion_detector
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({"success": False, "error": "Invalid session ID"}), 400
        
        session_data = active_sessions[session_id]
        
        if session_data["status"] != "active":
            return jsonify({"success": False, "error": "Session is not active"}), 400
        
        # Get frame data (base64 encoded image)
        frame_data = data.get('frame')
        if not frame_data:
            return jsonify({"success": False, "error": "No frame data provided"}), 400
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_bytes = base64.b64decode(frame_data)
            
            # Additional processing for analysis
            image = Image.open(io.BytesIO(img_bytes))
            
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return jsonify({"success": False, "error": "Failed to decode image"}), 400
        
        # Run predictions if models are available
        asd_result = None
        emotion_result = None
        
        # Get requested model
        requested_model = data.get('model_type')
        model = model_manager.get_model(requested_model)
        
        if model:
            try:
                # Validate face presence before prediction
                if not has_face(img_bytes):
                    emotion_result = {
                        "success": True, 
                        "predicted_emotion": "No Face Detected",
                        "confidence": 0.0,
                        "probabilities": {}
                    }
                else:
                    # Use fast emotion-only prediction (no XAI/Grad-CAM) for real-time responsiveness
                    emotion_result = model.predict_emotion_fast(img_bytes)
                    if not emotion_result.get('success'):
                        emotion_result = None
            except Exception as e:
                logger.error(f"Prediction error in gamification: {e}")
            
        # Store prediction data
        prediction_entry = {
            "timestamp": datetime.now().isoformat(),
            "frame_index": len(session_data["frames"]),
            "asd_analysis": asd_result if asd_result and asd_result['success'] else None,
            "emotion_analysis": emotion_result if emotion_result and emotion_result['success'] else None
        }
        
        session_data["predictions"].append(prediction_entry)
        session_data["frames"].append({
            "timestamp": datetime.now().isoformat(),
            "has_prediction": prediction_entry["asd_analysis"] is not None
        })
        
        # Save session periodically (or just at end, but better safe here)
        if len(session_data["frames"]) % 5 == 0:
            save_sessions(active_sessions)
        
        return jsonify({
            "success": True,
            "frame_number": len(session_data["frames"]),
            "total_frames": len(session_data["frames"]),
            "asd_result": asd_result,
            "emotion_result": emotion_result,
            "message": "Frame captured and analyzed"
        })
        
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@gamification_bp.route('/api/gamification/end-session', methods=['POST'])
def end_session():
    """Finalize session and generate results"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({"success": False, "error": "Invalid session ID"}), 400
        
        session_data = active_sessions[session_id]
        
        # Mark session as completed
        session_data["status"] = "completed"
        session_data["ended_at"] = datetime.now().isoformat()
        
        # Calculate session duration
        started = datetime.fromisoformat(session_data["started_at"])
        ended = datetime.fromisoformat(session_data["ended_at"])
        actual_duration = (ended - started).total_seconds()
        
        # Persist final state
        save_sessions(active_sessions)
        
        result = {
            "success": True,
            "session_id": session_id,
            "total_frames": len(session_data["frames"]),
            "total_predictions": len(session_data["predictions"]),
            "duration": actual_duration,
            "status": "completed",
            "message": "Session completed successfully"
        }
        
        logger.info(f"Session {session_id} completed with {len(session_data['frames'])} frames")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@gamification_bp.route('/api/gamification/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Retrieve session results"""
    try:
        # Reload sessions to ensure latest data
        current_sessions = load_sessions()
        
        if session_id not in current_sessions:
            return jsonify({"success": False, "error": "Session not found"}), 404
        
        session_data = current_sessions[session_id]
        
        # Return session summary
        summary = {
            "id": session_data["id"],
            "status": session_data["status"],
            "started_at": session_data["started_at"],
            "ended_at": session_data.get("ended_at"),
            "total_frames": len(session_data["frames"]),
            "metadata": session_data["metadata"],
            "predictions": session_data.get("predictions", [])
        }

        # Calculate emotion statistics
        emotion_counts = {}
        total_predictions = 0
        
        for pred in session_data.get("predictions", []):
            if pred.get("emotion_analysis") and pred["emotion_analysis"].get("predicted_emotion"):
                emotion = pred["emotion_analysis"]["predicted_emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_predictions += 1
                
        # Calculate percentages
        emotion_stats = []
        for emotion, count in emotion_counts.items():
            percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
            emotion_stats.append({
                "emotion": emotion,
                "count": count,
                "percentage": round(percentage, 1)
            })
            
        summary["emotion_stats"] = emotion_stats
        
        return jsonify({
            "success": True,
            "session": summary
        })
        
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@gamification_bp.route('/api/gamification/sessions', methods=['GET'])
def get_all_sessions():
    """Get all sessions (for history/dashboard)"""
    try:
        current_sessions = load_sessions()
        sessions_list = []
        
        for session_id, session_data in current_sessions.items():
            sessions_list.append({
                "id": session_data["id"],
                "status": session_data["status"],
                "started_at": session_data["started_at"],
                "total_frames": len(session_data["frames"]),
                "metadata": session_data["metadata"]
            })
        
        # Sort by start time (most recent first)
        sessions_list.sort(key=lambda x: x["started_at"], reverse=True)
        
        return jsonify({
            "success": True,
            "sessions": sessions_list,
            "count": len(sessions_list)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

