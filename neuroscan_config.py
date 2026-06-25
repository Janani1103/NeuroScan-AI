"""
Configuration file for NeuroScanAI
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32

# Training parameters
EPOCHS = 20
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

# Classes
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
NUM_CLASSES = len(CLASSES)

# Model parameters
PRETRAINED_MODEL = 'MobileNetV2'
FREEZE_BASE_MODEL = True
DROPOUT_RATE = 0.5
DENSE_UNITS = 128
