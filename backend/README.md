# ASD Emotion Recognition Backend

Flask backend for ASD detection and emotion recognition with real-time tracking.

## Setup

1. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Add your trained models:**
   - Place `best_asd_model.h5` in `trained_models/`
   - Place `emotion_model_complete.pt` in `trained_models/`

4. **Configure environment:**
```bash
copy .env.example .env
# Edit .env with your settings
```

5. **Run the server:**
```bash
python app.py
```

Server will start at `http://localhost:5000`

## API Endpoints

### Prediction
- `POST /api/predict/asd` - ASD detection
- `POST /api/predict/emotion` - Emotion recognition
- `POST /api/predict/combined` - Both predictions with XAI

### Gamification
- `POST /api/gamification/check` - Check emotion triggers
- `GET /api/gamification/videos` - Get calming videos
- `POST /api/gamification/suggest` - Get video suggestion

### Analytics
- `GET /api/analytics/summary` - Overall statistics
- `GET /api/analytics/emotion-timeline` - Emotion timeline
- `GET /api/analytics/engagement` - Engagement metrics

### Real-time (WebSocket)
- Connect to `ws://localhost:5000/socket.io/`
- Events: `start_tracking`, `process_frame`, `emotion_update`

## Model Files

Place your trained models in `trained_models/`:
- `best_asd_model.h5` - TensorFlow/Keras ASD model
- `emotion_model_complete.pt` - PyTorch emotion model

## Features

✅ Improved accuracy with ensemble predictions
✅ Explainable AI with Grad-CAM heatmaps
✅ Real-time emotion tracking via WebSocket
✅ Gamification with calming video suggestions
✅ Analytics dashboard
✅ CORS enabled for frontend integration
