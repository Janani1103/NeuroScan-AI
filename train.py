"""
Training script for NeuroScanAI
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import neuroscan_config as config
from data_preprocessing import get_data_generators, get_class_mapping
from model import build_mobilenetv2_model, unfreeze_top_layers


def train_model():
    """
    Train the MobileNetV2 model with callbacks and save the best model.
    """
    
    # Set up data generators
    print("Setting up data generators...")
    train_generator, val_generator, test_generator = get_data_generators()
    
    # Build model
    print("\nBuilding MobileNetV2 model...")
    model = build_mobilenetv2_model()
    
    # Callbacks
    checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    csv_logger = CSVLogger(
        os.path.join(config.OUTPUT_DIR, 'training_log.csv'),
        append=True
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr, csv_logger]
    
    # Initial training
    print("\nStarting initial training (frozen base model)...")
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning
    print("\nFine-tuning the model (unfreezing top layers)...")
    model = unfreeze_top_layers(model, num_layers_to_unfreeze=20)
    
    fine_tune_history = model.fit(
        train_generator,
        epochs=config.EPOCHS // 2,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, 'final_model.keras')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    return model, history, fine_tune_history


def plot_training_history(history, fine_tune_history):
    """
    Plot and save training history graphs.
    
    Args:
        history: Initial training history
        fine_tune_history: Fine-tuning history
    """
    # Combine histories
    acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    loss = history.history['loss'] + fine_tune_history.history['loss']
    val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history.history['accuracy']) + 1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history.history['loss']) + 1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Train the model
    model, history, fine_tune_history = train_model()
    
    print("\nTraining completed successfully!")
