"""
Real-time Emotion Tracking Routes
WebSocket-based live facial emotion recognition during video playback
"""

from flask import Blueprint, request
from flask_socketio import emit, join_room, leave_room
import base64
import io
from PIL import Image
import numpy as np
from datetime import datetime

from models.manager import model_manager

realtime_bp = Blueprint('realtime', __name__)

# Store active sessions (in production, use Redis or database)
active_sessions = {}


def init_socketio(socketio):
    """Initialize SocketIO event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        """Client connected"""
        print(f'Client connected: {request.sid}')
        emit('connection_response', {
            'status': 'connected',
            'session_id': request.sid
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Client disconnected"""
        print(f'Client disconnected: {request.sid}')
        if request.sid in active_sessions:
            del active_sessions[request.sid]
    
    @socketio.on('start_tracking')
    def handle_start_tracking(data):
        """Start emotion tracking session"""
        session_id = request.sid
        
        active_sessions[session_id] = {
            'started_at': datetime.now().isoformat(),
            'video_id': data.get('video_id'),
            'emotion_timeline': [],
            'frame_count': 0
        }
        
        emit('tracking_started', {
            'session_id': session_id,
            'message': 'Emotion tracking started'
        })
    
    @socketio.on('stop_tracking')
    def handle_stop_tracking():
        """Stop emotion tracking session"""
        session_id = request.sid
        
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            session_data['ended_at'] = datetime.now().isoformat()
            
            emit('tracking_stopped', {
                'session_id': session_id,
                'summary': {
                    'total_frames': session_data['frame_count'],
                    'timeline': session_data['emotion_timeline']
                }
            })
    
    @socketio.on('process_frame')
    def handle_process_frame(data):
        """
        Process video frame for emotion detection
        Data: { image: base64_string, timestamp: float }
        """
        try:
            session_id = request.sid
            
            if session_id not in active_sessions:
                emit('error', {'message': 'No active tracking session'})
                return
            
            # Decode base64 image
            image_data = data.get('image', '')
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Resolve model to use
            requested_model = data.get('model_type')
            detector = model_manager.get_model(requested_model)
            
            if not detector:
                emit('error', {'message': 'Emotion detector not available'})
                return

            # Predict emotion
            result = detector.predict_emotion_with_xai(image_bytes)
            
            if result['success']:
                # Add to timeline
                timestamp = data.get('timestamp', 0)
                emotion_data = {
                    'timestamp': timestamp,
                    'emotion': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                }
                
                active_sessions[session_id]['emotion_timeline'].append(emotion_data)
                active_sessions[session_id]['frame_count'] += 1
                
                # Emit emotion update
                emit('emotion_update', {
                    'emotion': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'timestamp': timestamp,
                    'top_emotions': result.get('top_emotions', [])
                })
            else:
                emit('error', {'message': result.get('error', 'Prediction failed')})
        
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @socketio.on('get_timeline')
    def handle_get_timeline():
        """Get emotion timeline for current session"""
        session_id = request.sid
        
        if session_id in active_sessions:
            timeline = active_sessions[session_id]['emotion_timeline']
            emit('timeline_data', {
                'timeline': timeline,
                'total_frames': len(timeline)
            })
        else:
            emit('error', {'message': 'No active session'})


# REST API endpoints for analytics

@realtime_bp.route('/api/realtime/session/<session_id>', methods=['GET'])
def get_session_data(session_id):
    """Get session data by ID"""
    if session_id in active_sessions:
        return {
            'success': True,
            'session': active_sessions[session_id]
        }, 200
    else:
        return {
            'success': False,
            'error': 'Session not found'
        }, 404


@realtime_bp.route('/api/realtime/sessions', methods=['GET'])
def get_all_sessions():
    """Get all active sessions"""
    return {
        'success': True,
        'sessions': active_sessions,
        'total': len(active_sessions)
    }, 200
