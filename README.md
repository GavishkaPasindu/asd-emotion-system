# ASD Emotion Detection System

## Overview

A comprehensive diagnostic support system that integrates AI models to detect Autism Spectrum Disorder (ASD) traits and recognize emotional states from facial imagery. The system provides transparency through explainable AI (XAI) visualizations including Grad-CAM heatmaps and MediaPipe-based facial landmark analysis.

## Project Structure

---
asd-emotion-system/
--- backend/                    # Flask API Server
    --- app.py                 # Application Entry Point
    --- models/                # Model Loaders and Logic
    --- routes/                # Prediction and Analytics Routes
    --- utils/                 # Grad-CAM and XAI Utilities
    --- trained_models/        # Directory for .h5 Model Files
    --- colab_notebooks/       # Jupyter Notebooks for Model Training
    --- requirements.txt       # Python Dependencies
--- frontend/                   # Next.js Web Application
    --- app/                   # Frontend Pages and Components
    --- lib/                   # API and WebSocket Service Layers
    --- context/               # State Management
    --- package.json           # Node.js Dependencies

## Implementation Details

### ASD Detection Model
- Framework: TensorFlow/Keras
- Backbone Support: ResNet50V2 (Default), VGG16, VGG19, ResNet50, InceptionV3
- Input Dimensions: 224x224 RGB
- Output: Binary Classification (ASD / Non-ASD)
- Performance: 85-95% Validation Accuracy

### Emotion Recognition Model
- Framework: TensorFlow/Keras
- Architecture: Optimized CNN for facial feature extraction
- Emotion Categories: 4 classes (Anger, Joy, Sadness, Surprise)
- Input Dimensions: 224x224 RGB
- Output: Multi-class Probability Distribution

## Features

### Computer Vision Validation
- Robust multi-stage face detection using MediaPipe and OpenCV cascades.
- Rejection of non-face images or repetitive patterns to ensure diagnostic accuracy.

### Explainable AI (XAI)
- Grad-CAM (Gradient-weighted Class Activation Mapping): Highlights localized facial regions contributing to the classification.
- MediaPipe Face Mesh Integration: Quantifies model attention on specific facial landmarks (Eyes, Mouth, Brows).

### Real-time Monitoring
- WebSocket-based streaming for live emotion tracking.
- Dynamic timeline visualization of emotional states over time.

### Analytics Dashboard
- Comprehensive statistical overview of prediction history.
- Distribution analysis of detected emotional states.

## Setup and Installation

### Backend

1. Navigate to the backend directory:
   cd backend

2. Create and activate a virtual environment:
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Place your trained .h5 models in backend/trained_models/

5. Start the server:
   python app.py

### Frontend

1. Navigate to the frontend directory:
   cd frontend

2. Install dependencies:
   npm install

3. Start the development server:
   npm run dev

## Model Training

Models were developed using the Autism Facial Emotion Recognition Dataset via Google Colab. Training notebooks for all supported architectures are located in:
backend/colab_notebooks/

Refer to backend/TRAINING_GUIDE.md for detailed instructions on reproducing the training results.

## System Requirements
- Python 3.9+
- Node.js 18+
- TensorFlow 2.15
- MediaPipe 0.10.9

## Credits
- Dataset: Hasibur Rahman (Kaggle)
- Explainability: Grad-CAM implementation based on Selvaraju et al.
