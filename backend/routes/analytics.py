"""
Analytics Dashboard Routes
Provides aggregated statistics and insights
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import random

analytics_bp = Blueprint('analytics', __name__)

# In-memory storage (use database in production)
analytics_data = {
    'predictions': [],
    'sessions': [],
    'engagement': []
}


@analytics_bp.route('/api/analytics/summary', methods=['GET'])
def get_summary():
    """
    Get overall analytics summary
    Returns: Total predictions, ASD/Non-ASD ratio, emotion distribution
    """
    try:
        model_type = request.args.get('model_type')
        
        # Filter predictions by model if provided
        predictions = analytics_data['predictions']
        if model_type:
            predictions = [p for p in predictions if p.get('model_type') == model_type]
            
        # Calculate statistics
        total_predictions = len(predictions)
        
        # ASD statistics
        asd_count = sum(1 for p in predictions 
                       if p.get('type') == 'asd' and p.get('result') == 'ASD')
        non_asd_count = sum(1 for p in predictions 
                           if p.get('type') == 'asd' and p.get('result') == 'NON_ASD')
        
        # Emotion distribution
        emotion_counts = {}
        for p in predictions:
            if p.get('type') == 'emotion':
                emotion = p.get('result', 'Unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Session statistics (Sessions aren't necessarily model-specific in this simple storage)
        total_sessions = len(analytics_data['sessions'])
        avg_session_duration = 0
        if total_sessions > 0:
            durations = [s.get('duration', 0) for s in analytics_data['sessions']]
            avg_session_duration = sum(durations) / len(durations)
        
        return jsonify({
            'success': True,
            'summary': {
                'total_predictions': total_predictions,
                'asd_detection': {
                    'total': asd_count + non_asd_count,
                    'asd': asd_count,
                    'non_asd': non_asd_count,
                    'ratio': asd_count / max(asd_count + non_asd_count, 1)
                },
                'emotion_distribution': emotion_counts,
                'sessions': {
                    'total': total_sessions,
                    'avg_duration_seconds': avg_session_duration
                }
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/api/analytics/emotion-timeline', methods=['GET'])
def get_emotion_timeline():
    """
    Get emotion changes over time
    Query params: session_id (optional), limit (default: 100)
    """
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 100))
        
        # Filter by session if provided
        if session_id:
            session = next((s for s in analytics_data['sessions'] 
                          if s.get('id') == session_id), None)
            if session:
                timeline = session.get('emotion_timeline', [])
            else:
                timeline = []
        else:
            # Get all emotion predictions
            timeline = [p for p in analytics_data['predictions'] 
                       if p.get('type') == 'emotion']
        
        # Limit results
        timeline = timeline[-limit:]
        
        return jsonify({
            'success': True,
            'timeline': timeline,
            'total': len(timeline)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/api/analytics/engagement', methods=['GET'])
def get_engagement_stats():
    """
    Get user engagement statistics
    Returns: Video watch time, completion rate, emotion improvement
    """
    try:
        total_engagements = len(analytics_data['engagement'])
        
        if total_engagements == 0:
            return jsonify({
                'success': True,
                'engagement': {
                    'total_videos_watched': 0,
                    'total_watch_time': 0,
                    'completion_rate': 0,
                    'improvement_rate': 0
                }
            }), 200
        
        # Calculate metrics
        total_watch_time = sum(e.get('watch_duration', 0) 
                              for e in analytics_data['engagement'])
        
        completed = sum(1 for e in analytics_data['engagement'] 
                       if e.get('completed', False))
        completion_rate = completed / total_engagements
        
        improved = sum(1 for e in analytics_data['engagement'] 
                      if e.get('improvement_score', 0) > 50)
        improvement_rate = improved / total_engagements
        
        # Average ratings
        ratings = [e.get('helpful') for e in analytics_data['engagement'] 
                  if e.get('helpful') is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return jsonify({
            'success': True,
            'engagement': {
                'total_videos_watched': total_engagements,
                'total_watch_time_seconds': total_watch_time,
                'completion_rate': completion_rate,
                'improvement_rate': improvement_rate,
                'average_rating': avg_rating
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/api/analytics/record-prediction', methods=['POST'])
def record_prediction():
    """Record a prediction for analytics"""
    try:
        data = request.get_json()
        
        prediction_record = {
            'id': len(analytics_data['predictions']) + 1,
            'timestamp': datetime.now().isoformat(),
            'type': data.get('type'),  # 'asd' or 'emotion'
            'result': data.get('result'),
            'confidence': data.get('confidence'),
            'model_type': data.get('model_type')
        }
        
        analytics_data['predictions'].append(prediction_record)
        
        return jsonify({
            'success': True,
            'message': 'Prediction recorded',
            'id': prediction_record['id']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/api/analytics/record-session', methods=['POST'])
def record_session():
    """Record a tracking session"""
    try:
        data = request.get_json()
        
        session_record = {
            'id': data.get('session_id', len(analytics_data['sessions']) + 1),
            'timestamp': datetime.now().isoformat(),
            'duration': data.get('duration', 0),
            'emotion_timeline': data.get('emotion_timeline', []),
            'video_id': data.get('video_id')
        }
        
        analytics_data['sessions'].append(session_record)
        
        return jsonify({
            'success': True,
            'message': 'Session recorded',
            'id': session_record['id']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/api/analytics/record-engagement', methods=['POST'])
def record_engagement():
    """Record video engagement"""
    try:
        data = request.get_json()
        
        engagement_record = {
            'id': len(analytics_data['engagement']) + 1,
            'timestamp': datetime.now().isoformat(),
            'video_id': data.get('video_id'),
            'watch_duration': data.get('watch_duration', 0),
            'completed': data.get('completed', False),
            'emotion_before': data.get('emotion_before'),
            'emotion_after': data.get('emotion_after'),
            'improvement_score': data.get('improvement_score', 0),
            'helpful': data.get('helpful')
        }
        
        analytics_data['engagement'].append(engagement_record)
        
        return jsonify({
            'success': True,
            'message': 'Engagement recorded',
            'id': engagement_record['id']
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
