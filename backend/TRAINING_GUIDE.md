# Model Training Guide

This directory contains training scripts to generate the ML models for the ASD emotion detection system.

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

#### ASD Detection Dataset
Download from Kaggle: https://www.kaggle.com/datasets/hasibur013/autism-facial-emotion-recognition

Standardized Extraction Path: `C:\Users\pc\Desktop\datasetnew\asd_data`
```
asd_data/
├── ASD/          (ASD images)
└── NON_ASD/      (Non-ASD images)
```

#### Emotion Recognition Dataset
Standardized Extraction Path: `C:\Users\pc\Desktop\datasetnew\emotion_data`
```
emotion_data/
├── train/
│   ├── anger/
│   ├── fear/
│   ├── joy/
│   ├── sadness/
│   ├── surprise/
│   └── Natural/
└── test/
    └── (same structure)
```

## Training the Models

### Train ASD Detection Model

```bash
python train_asd_model.py
```

This will:
- Load the ASD/NON_ASD dataset
- Train a CNN model for ~15 epochs
- Save the best model to `trained_models/best_asd_model.h5`
- Generate training history plots

**Expected Output:**
- Model file: `trained_models/best_asd_model.h5`
- Training plot: `trained_models/asd_training_history.png`
- Accuracy: ~85-95% (depends on dataset quality)

### Train Emotion Recognition Model

```bash
python train_emotion_model.py
```

This will:
- Load the emotion dataset (6 classes)
- Train a PyTorch CNN model for ~8 epochs
- Save the model to `trained_models/emotion_model_complete.pt`
- Generate training history plots

**Expected Output:**
- Model file: `trained_models/emotion_model_complete.pt`
- Training plot: `trained_models/emotion_training_history.png`
- Accuracy: ~80-90% (depends on dataset quality)

## Training Configuration

### ASD Model
- Image Size: 128x128
- Batch Size: 16
- Epochs: 15 (with early stopping)
- Architecture: 4-layer CNN with BatchNorm and Dropout
- Framework: TensorFlow/Keras

### Emotion Model
- Image Size: 128x128
- Batch Size: 64
- Epochs: 8 (with early stopping)
- Architecture: 3-layer CNN with BatchNorm
- Framework: PyTorch

## Training Time

- **ASD Model**: ~5-15 minutes (depending on dataset size and hardware)
- **Emotion Model**: ~2-5 minutes (depending on dataset size and hardware)

With GPU: Much faster (~2-5 minutes total)
With CPU: Slower but still manageable (~10-20 minutes total)

## Troubleshooting

### Dataset Not Found
Make sure your dataset is extracted in the correct location:
- ASD: `C:\Users\pc\Desktop\datasetnew\asd_data\ASD\`
- Emotion: `C:\Users\pc\Desktop\datasetnew\emotion_data\train\`

### Out of Memory
Reduce batch size in the training scripts:
- ASD: Change `BATCH_SIZE = 16` to `BATCH_SIZE = 8`
- Emotion: Change `BATCH_SIZE = 64` to `BATCH_SIZE = 32`

### Low Accuracy
- Ensure dataset quality is good
- Check if images are properly labeled
- Try training for more epochs
- Increase data augmentation

## After Training

Once both models are trained, you can:

1. **Test the backend:**
```bash
python app.py
```

2. **Test predictions:**
The backend will automatically load the models from `trained_models/`

3. **Use with frontend:**
Start the frontend and upload images to test the full system

## Model Files

After successful training, you should have:
```
trained_models/
├── best_asd_model.h5              # ASD detection model (TensorFlow)
├── emotion_model_complete.pt       # Emotion recognition model (PyTorch)
├── asd_training_history.png        # Training plots
└── emotion_training_history.png    # Training plots
```

## Notes

- The training scripts are adapted from the original Colab notebooks
- Models use the same architecture as the original code
- Training includes data augmentation for better generalization
- Early stopping prevents overfitting
- Best models are automatically saved during training
