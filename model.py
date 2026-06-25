"""
MobileNetV2 transfer learning model for NeuroScanAI
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import neuroscan_config as config


def build_mobilenetv2_model():
    """
    Build a MobileNetV2-based transfer learning model for brain tumor classification.
    
    Returns:
        model: Compiled Keras model
    """
    
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    )
    
    # Freeze the base model
    if config.FREEZE_BASE_MODEL:
        base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(config.DENSE_UNITS, activation='relu')(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def unfreeze_top_layers(model, num_layers_to_unfreeze=20):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
        
    Returns:
        model: Model with unfrozen layers
    """
    # Make the base model trainable
    model.layers[0].trainable = True
    
    # Freeze all layers except the last N layers
    for layer in model.layers[0].layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Building MobileNetV2 model...")
    model = build_mobilenetv2_model()
    
    print(f"\nModel summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
