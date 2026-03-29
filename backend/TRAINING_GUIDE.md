# Model Training Guide

This guide describes the process of preparing and training the machine learning models for the ASD emotion detection system.

## Prerequisites

### 1. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

#### ASD Detection Dataset
Source: [Kaggle - Autism Facial Emotion Recognition](https://www.kaggle.com/datasets/hasibur013/autism-facial-emotion-recognition)

Structure:
--- asd_data/
    --- ASD/          (ASD positive facial images)
    --- NON_ASD/      (Control group facial images)

#### Emotion Recognition Dataset
Structure:
--- emotion_data/
    --- train/
        --- anger/
        --- joy/
        --- sadness/
        --- surprise/
    --- test/        (same structure)

## Training Process

The models are developed using the "Improved" architecture set (ResNet50V2, InceptionV3, etc.) as documented in the Google Colab notebooks.

### Supported Architectures
- ResNet50V2 (Recommended for best accuracy)
- InceptionV3
- VGG16 / VGG19
- ResNet50

### Training via Google Colab
The primary training workflow is managed through the standalone notebooks in:
backend/colab_notebooks/

1. Upload the notebooks to Google Colab.
2. Link your preferred dataset source.
3. Execute the training cells to generate the .h5 model files.
4. Download the resulting .h5 files to the backend/trained_models/ directory.

## Model Specifications

### ASD Detection
- Input: 224x224 RGB
- Framework: TensorFlow/Keras
- Output: Binary (Sigmoid)
- Key Layers: GlobalAveragePooling2D, Dense(512), BatchNorm, Dropout(0.5)

### Emotion Recognition
- Input: 224x224 RGB
- Framework: TensorFlow/Keras
- Output: 4-class Softmax
- Classes: Anger, Joy, Sadness, Surprise

## Troubleshooting

### Low Detection Accuracy
- Ensure images are cropped specifically to the face region.
- Verify that the data augmentation parameters (rotation, zoom, flip) are active during training.
- Check for class imbalance in the training set.

### Real-time Performance Issues
- If using CPU, consider using the VGG16 architecture for faster inference.
- Ensure the MediaPipe face detection model is correctly cached in backend/trained_models/.

## Post-Training Deployment

Once .h5 files are generated:
1. Place them in: backend/trained_models/
2. Update the environment variables in .env if custom filenames are used.
3. Restart the Flask server (python app.py) to load the new models.
