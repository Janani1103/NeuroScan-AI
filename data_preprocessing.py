"""
Data preprocessing and augmentation module for NeuroScanAI
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import neuroscan_config as config


def get_data_generators():
    """
    Create data generators for training, validation, and testing with augmentation.
    
    Returns:
        train_generator: Training data generator with augmentation
        val_generator: Validation data generator without augmentation
        test_generator: Test data generator without augmentation
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=config.VALIDATION_SPLIT
    )
    
    # Validation and test data generator (only rescaling)
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def get_class_mapping(generator):
    """
    Get the class mapping from the generator.
    
    Args:
        generator: Keras data generator
        
    Returns:
        dict: Class index to class name mapping
    """
    return {v: k for k, v in generator.class_indices.items()}


if __name__ == "__main__":
    # Test the data generators
    print("Setting up data generators...")
    train_gen, val_gen, test_gen = get_data_generators()
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    print(f"\nClass mapping: {get_class_mapping(train_gen)}")
    
    print("\nData generators ready!")
