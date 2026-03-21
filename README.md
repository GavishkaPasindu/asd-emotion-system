# ASD Emotion Detection System - Updated README

## Overview

Complete full-stack application for ASD detection and emotion recognition with explainable AI visualizations.

## Project Structure

```
asd-emotion-system/
├-- backend/                    # Flask API server
│   ├-- app.py                 # Main application
│   ├-- models/                # ML model classes
│   ├-- routes/                # API endpoints
│   ├-- utils/                 # Utilities (Grad-CAM)
│   ├-- trained_models/        # Trained model files (you need to create these)
│   ├-- train_asd_model.py     # Training script for ASD model
│   ├-- train_emotion_model.py # Training script for emotion model
│   ├-- TRAINING_GUIDE.md      # Detailed training instructions
│   └-- requirements.txt       # Python dependencies
└-- frontend/                   # Next.js React app
    ├-- app/                   # Application pages
    ├-- lib/                   # API and WebSocket clients
    ├-- components/            # Reusable UI components
    └-- package.json           # Node dependencies
```

## Quick Start

### Option 1: Train Your Own Models (Recommended)

#### Step 1: Download Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/hasibur013/autism-facial-emotion-recognition

#### Step 2: Prepare Dataset Structure

For ASD Detection:
```
dataset/
├-- ASD/          # ASD images
└-- NON_ASD/      # Non-ASD images
```

For Emotion Recognition:
```
emotion_dataset/
├-- train/
│   ├-- anger/
│   ├-- fear/
│   ├-- joy/
│   ├-- sadness/
│   ├-- surprise/
│   └-- Natural/
└-- test/
    └-- (same structure)
```

#### Step 3: Train Models

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Train ASD model (~5-15 minutes)
python train_asd_model.py

# Train emotion model (~2-5 minutes)
python train_emotion_model.py
```

This will create:
- `trained_models/best_asd_model.h5`
- `trained_models/emotion_model_complete.pt`

See [TRAINING_GUIDE.md](backend/TRAINING_GUIDE.md) for detailed instructions.

### Option 2: Use Pre-trained Models

If you have pre-trained models, place them in:
- `backend/trained_models/best_asd_model.h5`
- `backend/trained_models/emotion_model_complete.pt`

## Running the Application

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (optional)
copy .env.example .env

# Run server
python app.py
```

Server runs at: `http://localhost:5000`

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs at: `http://localhost:3000`

## Features

### ✅ ASD Detection
- Upload facial images
- AI-powered ASD screening
- Confidence scores
- Grad-CAM explainable AI visualizations

### ✅ Emotion Recognition
- Detect 6 emotions: anger, fear, joy, sadness, surprise, neutral
- Probability distribution for all emotions
- Attention heatmaps showing model focus

### ✅ Combined Analysis
- Single upload for both ASD and emotion detection
- Side-by-side results
- Comprehensive insights

### ✅ Real-time Tracking
- WebSocket-powered live emotion monitoring
- Emotion timeline history
- Color-coded emotion display

### ✅ Analytics Dashboard
- Total predictions statistics
- Emotion distribution charts (bar + pie)
- Average confidence metrics

## API Endpoints

### Prediction
- `POST /api/predict/asd` - ASD detection
- `POST /api/predict/emotion` - Emotion recognition
- `POST /api/predict/combined` - Combined analysis

### Analytics
- `GET /api/analytics/summary` - Overall statistics
- `GET /api/analytics/emotion-timeline` - Emotion timeline
- `GET /api/analytics/engagement` - Engagement metrics

### Gamification
- `GET /api/gamification/videos` - Get calming videos
- `POST /api/gamification/suggest` - Get video suggestion

### Real-time
- WebSocket: `ws://localhost:5000/socket.io/`
- Events: `start_tracking`, `process_frame`, `emotion_update`

## Technology Stack

### Backend
- **Framework**: Flask 3.0
- **ML**: TensorFlow 2.15, PyTorch 2.1
- **Computer Vision**: OpenCV
- **Real-time**: Flask-SocketIO
- **XAI**: Grad-CAM implementation

### Frontend
- **Framework**: Next.js 16.1
- **UI**: React 19, TypeScript
- **Styling**: Tailwind CSS 4
- **Charts**: Recharts
- **Icons**: Lucide React
- **Real-time**: Socket.io-client

## Model Information

### ASD Detection Model
- **Architecture**: 4-layer CNN with BatchNorm and Dropout
- **Input**: 128x128 RGB images
- **Output**: Binary classification (ASD/Non-ASD)
- **Framework**: TensorFlow/Keras
- **Expected Accuracy**: 85-95%

### Emotion Recognition Model
- **Architecture**: 3-layer CNN with BatchNorm
- **Input**: 128x128 RGB images
- **Output**: 6-class classification
- **Classes**: anger, fear, joy, sadness, surprise, Natural
- **Framework**: PyTorch
- **Expected Accuracy**: 80-90%

## Dataset

**Source**: [Autism Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/hasibur013/autism-facial-emotion-recognition)

The dataset contains:
- ASD and Non-ASD facial images
- 6 emotion categories
- Augmented training data
- Separate train/test splits

## Training Your Models

See [TRAINING_GUIDE.md](backend/TRAINING_GUIDE.md) for:
- Detailed training instructions
- Dataset preparation
- Configuration options
- Troubleshooting tips
- Expected training times

## Environment Variables

Create `backend/.env`:
```env
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
HOST=0.0.0.0
PORT=5000
CORS_ORIGINS=http://localhost:3000
MAX_CONTENT_LENGTH=16777216
ASD_MODEL_PATH=trained_models/best_asd_model.h5
EMOTION_MODEL_PATH=trained_models/emotion_model_complete.pt
```

Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Development

### Backend Development
```bash
cd backend
python app.py  # Runs with auto-reload in debug mode
```

### Frontend Development
```bash
cd frontend
npm run dev  # Runs with hot-reload
```

## Production Build

### Frontend
```bash
cd frontend
npm run build
npm start
```

## Troubleshooting

### Models Not Loading
- Ensure model files exist in `backend/trained_models/`
- Check file paths in `.env`
- Verify models were trained successfully

### CORS Errors
- Check `CORS_ORIGINS` in backend `.env`
- Ensure frontend URL matches CORS settings

### WebSocket Connection Failed
- Verify backend is running
- Check firewall settings
- Ensure correct WebSocket URL in frontend

### Low Model Accuracy
- Check dataset quality
- Verify correct data augmentation
- Try training for more epochs
- Ensure proper train/test split

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: Hasibur Rahman (Kaggle)
- Original Colab notebooks provided by user
- TensorFlow and PyTorch communities

## Support

For issues or questions:
1. Check [TRAINING_GUIDE.md](backend/TRAINING_GUIDE.md)
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify dataset structure is correct
