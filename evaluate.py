"""
Evaluation module for NeuroScanAI
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import neuroscan_config as config
from data_preprocessing import get_data_generators, get_class_mapping


def evaluate_model(model_path):
    """
    Evaluate the trained model on test data and compute metrics.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Get test data generator
    _, _, test_generator = get_data_generators()
    
    # Get predictions
    print("Generating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get class names
    class_mapping = get_class_mapping(test_generator)
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-score  : {f1:.4f} ({f1*100:.2f}%)")
    print("="*50)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Save metrics to file
    save_metrics(accuracy, precision, recall, f1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def plot_confusion_matrix(cm, class_names):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()


def save_metrics(accuracy, precision, recall, f1):
    """
    Save evaluation metrics to a text file.
    
    Args:
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    metrics_path = os.path.join(config.OUTPUT_DIR, 'evaluation_metrics.txt')
    
    with open(metrics_path, 'w') as f:
        f.write("NeuroScanAI - Evaluation Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision : {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall    : {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1-score  : {f1:.4f} ({f1*100:.2f}%)\n")
        f.write("="*50 + "\n")
    
    print(f"Evaluation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    # Evaluate the best model
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.keras')
    
    if os.path.exists(best_model_path):
        metrics = evaluate_model(best_model_path)
    else:
        print(f"Model not found at: {best_model_path}")
        print("Please train the model first using train.py")
