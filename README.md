# CNN-Transformer Architecture for Deepfake Detection

## Project Overview
This project implements an advanced deepfake detection system using a hybrid CNN-Transformer architecture. The model leverages the spatial feature extraction capabilities of EfficientNetB0 combined with the temporal modeling power of Transformers to analyze video sequences and identify manipulated content.

## Dataset
This project uses the "1000 Videos Split" dataset from Kaggle:
- **Dataset**: [1000 Videos Split](https://www.kaggle.com/datasets/nanduncs/1000-videos-split)
- The dataset contains real and manipulated (deepfake) videos split into training, validation, and test sets.

## Model Architecture
The detection system consists of three key components:

1. **Feature Extraction**: Pre-trained EfficientNetB0 CNN for extracting spatial features from individual video frames
2. **Temporal Modeling**: Transformer with multi-head attention to capture temporal relationships between frames
3. **Classification**: Fully connected layers with dropout for regularization

Key features of the architecture:
- Time-distributed CNN to process multiple frames from a video
- Multi-head attention mechanism for modeling frame relationships
- Residual connections and layer normalization for stable training
- Data augmentation pipeline for improved generalization

## Implementation Details

### Face Detection Preprocessing
- Uses MTCNN (Multi-task Cascaded Convolutional Networks) to detect and extract faces from video frames
- Processes a consistent number of frames per video (10 frames by default)
- Normalizes extracted face regions to standardized dimensions (224×224)

### Data Augmentation
- Horizontal flipping
- Random rotation (±10%)
- Brightness adjustment (±20%)

### Training Strategy
- Two-phase training approach:
  1. Initial training with frozen EfficientNetB0 base
  2. Fine-tuning of selected layers for domain adaptation
- Early stopping to prevent overfitting
- Model checkpointing to save best-performing model

## Requirements
- TensorFlow 2.x
- OpenCV
- MTCNN
- NumPy
- Zipfile (for dataset extraction)

## Installation
```bash
pip install tensorflow opencv-python mtcnn numpy
```

## Usage

### Dataset Preparation
```python
# Extract the dataset
dataset_dir = '/path/to/dataset'
os.makedirs(dataset_dir, exist_ok=True)

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)
```

### Training the Model
```python
# Load dataset
X, y = load_dataset(dataset_dir)

# Split into train/validation sets
train_X, val_X = X[:train_size], X[train_size:]
train_y, val_y = y[:train_size], y[train_size:]

# Build and train model
model = build_cnn_transformer_model()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[checkpoint, early_stopping]
)
```

### Evaluating Performance
```python
# Evaluate on test set
test_loss, test_acc, test_auc = model.evaluate(test_X, test_y)
print(f"Test Accuracy: {test_acc}, Test AUC: {test_auc}")
```

## Performance Metrics
The model is evaluated using:
- Binary accuracy
- Area Under the ROC Curve (AUC)
- Additional metrics can be added such as precision, recall, and F1-score

## Advantages Over Traditional Approaches
- **Temporal Context**: Unlike frame-by-frame classification, captures relationships between frames
- **Attention Mechanism**: Focuses on the most relevant parts of the video for detecting manipulation
- **Face-Focused**: Concentrates on facial regions where manipulations are most common
- **Transfer Learning**: Leverages pre-trained models for efficient feature extraction

## Future Work
- Implement cross-dataset evaluation to test generalization capabilities
- Explore different backbone architectures (ResNet, Vision Transformer)
- Add explainability features to visualize which parts of videos are flagged as manipulated
- Develop a lightweight version for real-time detection on edge devices
