# -*- coding: utf-8 -*-
"""Cnn-Lstm algorithm 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mVBOQgWniOjmFqOutJOZRZ1VaWcmj4QB

# **1. Dataset Preparation**
**Datasets**: Use publicly available datasets such as FaceForensics++ or Celeb-DF.


---


**Preprocessing**:

---



*   Extract frames from videos (e.g., 10 frames per video).
*   LResize images to match the Xception model’s input size (299x299 pixels).
*   Normalize pixel values to the range [-1, 1] (as required by Xception).
*   Label frames as real or fake.
"""

import zipfile
import os

# Path where the zip file is uploaded
zip_path = '/content/archive.zip'

# Output directory for the dataset
dataset_dir = '/content/dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

print(f"Dataset extracted to {dataset_dir}")

import os

# Check the directory structure
for root, dirs, files in os.walk(dataset_dir):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files[:5]}")  # Print only the first 5 files
    print("-" * 50)

"""
# **2. Model Architecture**
The CNN-LSTM model consists of:


*   CNN Layers: To extract spatial features.
*   LSTM Layers: To capture temporal dependencies across frames.
*   Fully Connected Layers: For classification



"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Callbacks
checkpoint = ModelCheckpoint('cnn_lstm_model.h5', save_best_only=True, monitor='val_loss')
early_stopping = EarlyStopping(patience=10, monitor='val_loss')

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[checkpoint, early_stopping]
)

"""# **3. Training**


*   Data Generator: Use a generator to handle batches of video frames for efficient memory usage.
*   Augmentation: Apply augmentations such as horizontal flips, rotations, or brightness adjustments to improve generalization.




"""

import tensorflow as tf
from tensorflow.keras import layers, models

# CNN-LSTM Architecture
def build_cnn_lstm_model(input_shape=(10, 224, 224, 3)):  # Example for 10 frames of 224x224 RGB images
    model = models.Sequential()

    # CNN Layers
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))

    # LSTM Layers
    model.add(layers.LSTM(128, return_sequences=False))

    # Fully Connected Layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_lstm_model()
model.summary()

"""# **4. Evaluation**
Evaluate the model on the test set
"""

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")