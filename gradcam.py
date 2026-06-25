"""
Grad-CAM visualization for explainable AI in NeuroScanAI
"""
import os
import numpy as np
import tensorflow as tf
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import matplotlib.pyplot as plt
import neuroscan_config as config
from data_preprocessing import get_class_mapping


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        img_array: Preprocessed image array
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the predicted class (optional)
        
    Returns:
        heatmap: Grad-CAM heatmap
    """
    
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of the top predicted class w.r.t. the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron (top predicted) w.r.t. the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def get_last_conv_layer_name(model):
    """
    Find the name of the last convolutional layer in the model.
    
    Args:
        model: Keras model
        
    Returns:
        str: Name of the last convolutional layer
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def preprocess_image(img_path):
    """
    Preprocess an image for Grad-CAM visualization.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        img_array: Preprocessed image array
        original_img: Original image for display
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
    )
    original_img = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    return img_array, original_img


def save_and_display_gradcam(img_path, heatmap, alpha=0.4, save_path=None):
    """
    Superimpose the heatmap on the original image and save it.
    
    Args:
        img_path: Path to the original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor for heatmap overlay
        save_path: Path to save the visualization (optional)
        
    Returns:
        superimposed_img: Image with heatmap overlay
    """
    if not CV2_AVAILABLE:
        print("Warning: cv2 not available. Grad-CAM overlay not generated.")
        return None
    
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Save if path provided
    if save_path:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    return superimposed_img


def generate_gradcam_visualization(model_path, img_path, output_path):
    """
    Generate complete Grad-CAM visualization for an image.
    
    Args:
        model_path: Path to the trained model
        img_path: Path to the input image
        output_path: Path to save the visualization
    """
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get last conv layer name
    last_conv_layer_name = get_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        print("Could not find convolutional layer in the model")
        return
    
    # Preprocess image
    img_array, original_img = preprocess_image(img_path)
    
    # Generate prediction
    preds = model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds[0])
    pred_confidence = np.max(preds[0])
    
    # Get class name
    _, _, test_generator = get_data_generators()
    class_mapping = get_class_mapping(test_generator)
    class_name = class_mapping[pred_class]
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class)
    
    # Save visualization
    save_and_display_gradcam(img_path, heatmap, alpha=0.4, save_path=output_path)
    
    print(f"Grad-CAM visualization saved to: {output_path}")
    print(f"Prediction: {class_name} (confidence: {pred_confidence*100:.2f}%)")


if __name__ == "__main__":
    if not CV2_AVAILABLE:
        print("Error: cv2 module is required for this script. Please install opencv-python.")
        print("Run: pip install opencv-python")
        exit(1)
    
    # Example usage
    model_path = os.path.join(config.MODEL_DIR, 'best_model.keras')
    
    # Find a test image
    test_dir = os.path.join(config.TEST_DIR, 'glioma')
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    if test_images:
        img_path = os.path.join(test_dir, test_images[0])
        output_path = os.path.join(config.OUTPUT_DIR, 'gradcam_example.png')
        
        generate_gradcam_visualization(model_path, img_path, output_path)
    else:
        print("No test images found")
