"""
Test script for NeuroScanAI prediction system
Tests the model with sample images from each category
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import neuroscan_config as config
from data_preprocessing import get_data_generators, get_class_mapping

def load_model():
    """Load the trained model."""
    model_path = os.path.join('models', 'best_model.keras')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

def predict_image(model, image_path, class_mapping):
    """Make prediction on a single image."""
    if not os.path.exists(image_path):
        return None, None, None
    
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((config.IMG_HEIGHT, config.IMG_WIDTH))
    
    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    pred_class = class_mapping[pred_index]
    
    return pred_class, confidence, predictions[0]

def test_sample_images():
    """Test the prediction system with sample images from each category."""
    print("=" * 60)
    print("NeuroScanAI - Prediction System Test")
    print("=" * 60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get class mapping
    print("\nSetting up data generator for class mapping...")
    _, _, test_generator = get_data_generators()
    class_mapping = get_class_mapping(test_generator)
    print(f"Classes: {class_mapping}")
    
    # Test images from each category
    test_cases = [
        ('glioma', 'Testing/glioma/Te-gl_1.jpg'),
        ('meningioma', 'Testing/meningioma/Te-aug-me_1.jpg'),
        ('pituitary', 'Testing/pituitary/Te-pi_1.jpg'),
        ('notumor', 'Testing/notumor/Te-no_1.jpg'),
    ]
    
    print("\n" + "=" * 60)
    print("Testing Sample Images")
    print("=" * 60)
    
    results = []
    
    for true_class, image_path in test_cases:
        print(f"\nTesting: {image_path}")
        print(f"True Class: {true_class}")
        
        pred_class, confidence, predictions = predict_image(model, image_path, class_mapping)
        
        if pred_class is not None:
            is_correct = "[CORRECT]" if pred_class.lower() == true_class.lower() else "[INCORRECT]"
            print(f"Predicted Class: {pred_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"Result: {is_correct}")
            
            # Show probability distribution
            print("Probability Distribution:")
            for i, class_name in enumerate(class_mapping):
                print(f"  {class_name}: {predictions[i]*100:.2f}%")
            
            results.append({
                'image': image_path,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'correct': pred_class.lower() == true_class.lower()
            })
        else:
            print(f"Error: Could not load image {image_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "[OK]" if result['correct'] else "[FAIL]"
        print(f"{i}. {status} {result['image']}")
        print(f"   True: {result['true_class']} | Predicted: {result['predicted_class']} | Confidence: {result['confidence']:.2%}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_sample_images()