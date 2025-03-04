# -*- coding: utf-8 -*-
import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from mtcnn import MTCNN  # For face detection
import cv2
import numpy as np

# --- 1. Dataset Preparation ---
# Extract dataset
zip_path = '/content/archive.zip'
dataset_dir = '/content/dataset'
os.makedirs(dataset_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)
print(f"Dataset extracted to {dataset_dir}")

# Preprocessing function with face detection
detector = MTCNN()

def preprocess_video_frames(video_path, num_frames=10, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)
    
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)
            face = (face / 127.5) - 1.0  # Normalize to [-1, 1]
            frames.append(face)
        if len(frames) >= num_frames:
            break
    
    cap.release()
    while len(frames) < num_frames:  # Pad if necessary
        frames.append(np.zeros((target_size[0], target_size[1], 3)))
    return np.array(frames[:num_frames])

# Example: Assume a function to load your dataset
def load_dataset(dataset_dir, num_frames=10):
    real_videos = [...]  # List of paths to real videos
    fake_videos = [...]  # List of paths to fake videos
    X, y = [], []
    for video in real_videos:
        frames = preprocess_video_frames(video, num_frames)
        X.append(frames)
        y.append(0)  # Real label
    for video in fake_videos:
        frames = preprocess_video_frames(video, num_frames)
        X.append(frames)
        y.append(1)  # Fake label
    return np.array(X), np.array(y)

# --- 2. Updated Model Architecture ---
def build_cnn_transformer_model(input_shape=(10, 224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Pretrained CNN (EfficientNetB0)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape[1:], weights='imagenet')
    base_model.trainable = False  # Freeze for initial training
    
    # Extract features for each frame
    cnn_features = layers.TimeDistributed(base_model)(inputs)
    cnn_features = layers.TimeDistributed(layers.GlobalAveragePooling2D())(cnn_features)
    
    # Transformer for temporal modeling
    transformer = layers.MultiHeadAttention(num_heads=4, key_dim=128)(cnn_features, cnn_features)
    transformer = layers.LayerNormalization()(transformer + cnn_features)  # Residual connection
    transformer = layers.GlobalAveragePooling1D()(transformer)
    
    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(transformer)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Compile model
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

model = build_cnn_transformer_model()
model.summary()

# --- 3. Data Generator with Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2),
])

def data_generator(X, y, batch_size=8):
    while True:
        indices = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            augmented_X = data_augmentation(batch_X)
            yield augmented_X, batch_y

# --- 4. Training ---
# Load your dataset (replace with actual paths)
X, y = load_dataset(dataset_dir)
train_size = int(0.8 * len(X))
train_X, val_X = X[:train_size], X[train_size:]
train_y, val_y = y[:train_size], y[train_size:]

batch_size = 8
train_gen = data_generator(train_X, train_y, batch_size)
val_gen = data_generator(val_X, val_y, batch_size)

checkpoint = ModelCheckpoint('cnn_transformer_model.h5', save_best_only=True, monitor='val_loss')
early_stopping = EarlyStopping(patience=10, monitor='val_loss')

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_X) // batch_size,
    validation_data=val_gen,
    validation_steps=len(val_X) // batch_size,
    epochs=50,
    callbacks=[checkpoint, early_stopping]
)

# --- 5. Evaluation ---
# Assuming test_X, test_y are prepared similarly
test_loss, test_acc, test_auc = model.evaluate(val_X, val_y)
print(f"Test Accuracy: {test_acc}, Test AUC: {test_auc}")
