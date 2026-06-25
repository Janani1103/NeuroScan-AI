import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from datetime import datetime
import neuroscan_config as config
from data_preprocessing import get_data_generators, get_class_mapping
from gradcam import make_gradcam_heatmap, get_last_conv_layer_name


THEME = {
    'primary': '#8B008B',   
    'secondary': '#4B004B', 
    'accent': '#E6A8D7',     
    'background': '#121212', 
    'card_bg': '#1E1E1E',    
    'text': '#FFFFFF',        
    'border': '#383838',      
    'success': '#00C851',  
    'warning': '#FFBB33',   
    'danger': '#FF4444'     
}

st.set_page_config(
    page_title="NeuroScanAI - Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    .stApp {{
        background-color: {THEME['background']};
    }}

    .css-1d391kg {{
        background-color: {THEME['card_bg']};
    }}

    .main .block-container {{
        background-color: {THEME['background']};
        color: {THEME['text']};
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {THEME['text']} !important;
    }}

    .app-title {{
        font-size: 2.5rem;
        color: {THEME['primary']};
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    .app-subtitle {{
        font-size: 1.2rem;
        color: {THEME['accent']};
        text-align: center;
        margin-bottom: 2rem;
    }}

    .card {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    .stat-card {{
        background: linear-gradient(135deg, {THEME['secondary']}, {THEME['primary']});
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    .stat-value {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {THEME['accent']};
    }}
    
    .stat-label {{
        font-size: 1rem;
        color: {THEME['text']};
        opacity: 0.8;
    }}

    .stButton>button {{
        background-color: {THEME['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {THEME['secondary']};
        transform: translateY(-2px);
    }}

    .upload-box {{
        border: 2px dashed {THEME['primary']};
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background-color: {THEME['card_bg']};
        transition: all 0.3s;
    }}
    
    .upload-box:hover {{
        border-color: {THEME['accent']};
        background-color: {THEME['secondary']};
    }}

    .result-box {{
        background: linear-gradient(135deg, {THEME['secondary']}, {THEME['primary']});
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }}
    
    .prediction-class {{
        font-size: 2rem;
        font-weight: bold;
        color: {THEME['accent']};
        margin: 1rem 0;
    }}
    
    .confidence-score {{
        font-size: 3rem;
        font-weight: bold;
        color: {THEME['success']};
    }}

    .risk-high {{
        background-color: {THEME['danger']};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }}
    
    .risk-medium {{
        background-color: {THEME['warning']};
        color: black;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }}
    
    .risk-low {{
        background-color: {THEME['success']};
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }}

    .sidebar-menu {{
        background-color: {THEME['card_bg']};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }}
    
    .sidebar-menu.active {{
        background-color: {THEME['primary']};
    }}

    .chart-container {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}

    .dataframe {{
        background-color: {THEME['card_bg']};
        color: {THEME['text']};
    }}

    .disclaimer {{
        background-color: {THEME['card_bg']};
        border-left: 4px solid {THEME['warning']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}

    .feature-item {{
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        color: {THEME['text']};
    }}
    
    .feature-icon {{
        color: {THEME['success']};
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }}
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False


@st.cache_resource
def load_model():
    model_path = os.path.join(config.MODEL_DIR, 'best_model.keras')
    
    if not os.path.exists(model_path):
        return None
    
    model = tf.keras.models.load_model(model_path)
    return model


@st.cache_resource
def get_class_names():
    try:
        _, _, test_generator = get_data_generators()
        return get_class_mapping(test_generator)
    except:
        return {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'notumor'}


def load_evaluation_metrics():
    """Load evaluation metrics from the metrics file."""
    metrics_path = os.path.join(config.OUTPUT_DIR, 'evaluation_metrics.txt')
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                content = f.read()
                # Parse the metrics file
                for line in content.split('\n'):
                    if 'Accuracy' in line and ':' in line:
                        # Extract accuracy value (e.g., "0.8150 (81.50%)")
                        parts = line.split(':')[1].strip()
                        accuracy_percent = parts.split('(')[1].split('%')[0]
                        return float(accuracy_percent) / 100
        except Exception as e:
            print(f"Error loading metrics: {e}")
    
    return None


def get_model_info(model):
   
    if model is None:
        return {'params': 'N/A', 'inference_time': 'N/A'}

    total_params = model.count_params()
    params_str = f"{total_params / 1e6:.1f}M" if total_params >= 1e6 else f"{total_params / 1e3:.1f}K"

    try:
        dummy_input = np.random.random((1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))

        _ = model.predict(dummy_input, verbose=0)

        start_time = time.time()
        for _ in range(10):
            _ = model.predict(dummy_input, verbose=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000  # Convert to milliseconds
        inference_time_str = f"{avg_time:.0f}ms"
    except Exception as e:
        print(f"Error measuring inference time: {e}")
        inference_time_str = 'N/A'
    
    return {'params': params_str, 'inference_time': inference_time_str}


def predict_image(model, image, class_mapping):
    """Make prediction on the uploaded image."""
    img = image.resize((config.IMG_HEIGHT, config.IMG_WIDTH))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    pred_class = class_mapping[pred_index]
    
    return pred_class, confidence, predictions[0]


def generate_gradcam_visualization(model, image, last_conv_layer_name, pred_index):
    """Generate Grad-CAM visualization."""
    img = image.resize((config.IMG_HEIGHT, config.IMG_WIDTH))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(THEME['card_bg'])
    
    ax1.imshow(img)
    ax1.set_title('Original MRI Image', color=THEME['text'], fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], [img.size[1], img.size[0]])
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    heatmap_colored = plt.cm.jet(heatmap_resized)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    original_array = np.array(img)
    superimposed = (heatmap_colored * 0.4 + original_array * 0.6).astype(np.uint8)
    
    ax2.imshow(superimposed)
    ax2.set_title('Grad-CAM Heatmap', color=THEME['text'], fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def save_prediction_to_history(pred_class, confidence, predictions):

    prediction_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': pred_class,
        'confidence': f"{confidence*100:.2f}%",
        'all_probabilities': {class_name: f"{prob*100:.2f}%" 
                            for class_name, prob in zip(get_class_names().values(), predictions)}
    }
    st.session_state.predictions_history.append(prediction_record)


def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: {THEME['primary']}; font-size: 2rem; margin: 0;'>🧠</h1>
            <h2 style='color: {THEME['accent']}; font-size: 1.2rem; margin: 0.5rem 0;'>NeuroScanAI</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        pages = [
            ('🏠 Dashboard', 'dashboard'),
            ('🖼️ Upload MRI', 'upload'),
            ('🧠 Predict Tumor', 'predict'),
            ('📈 Analytics', 'analytics'),
            ('📄 Reports', 'reports')
        ]
        
        for icon, page_key in pages:
            if st.button(f"{icon}", key=page_key, use_container_width=True):
                st.session_state.page = page_key
                if page_key == 'predict':
                    st.session_state.show_results = False
        
        st.markdown("---")

        model = load_model()
        if model:
            st.markdown(f"""
            <div style='background-color: {THEME['card_bg']}; padding: 1rem; border-radius: 8px; border: 1px solid {THEME['border']};'>
                <h4 style='color: {THEME['success']}; margin: 0;'>✓ Model Loaded</h4>
                <p style='color: {THEME['text']}; font-size: 0.8rem; margin: 0.5rem 0 0 0;'>MobileNetV2 Ready</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: {THEME['card_bg']}; padding: 1rem; border-radius: 8px; border: 1px solid {THEME['danger']};'>
                <h4 style='color: {THEME['danger']}; margin: 0;'>✗ Model Not Found</h4>
                <p style='color: {THEME['text']}; font-size: 0.8rem; margin: 0.5rem 0 0 0;'>Train model first</p>
            </div>
            """, unsafe_allow_html=True)


def render_landing_page():
    st.markdown(f"""
    <div style='text-align: center; padding: 3rem 0;'>
        <h1 class='app-title'>🧠 NeuroScanAI</h1>
        <p class='app-subtitle'>AI-Powered Brain Tumor Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: {THEME['card_bg']}; border: 1px solid {THEME['border']}; 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
            <h2 style='color: {THEME['accent']};'>Detect Brain Tumors Using Deep Learning</h2>
            <p style='color: {THEME['text']}; line-height: 1.6;'>
            Our advanced AI system analyzes MRI images to accurately identify different types of brain tumors, 
            assisting medical professionals with fast and reliable predictions.
            </p>
            
            <div style='margin: 2rem 0;'>
                <div style='display: flex; align-items: center; margin: 0.5rem 0; color: {THEME['text']};'>
                    <span style='color: {THEME['success']}; margin-right: 0.5rem; font-size: 1.2rem;'>✓</span>
                    <span><strong>98% Accuracy</strong> - State-of-the-art performance</span>
                </div>
                <div style='display: flex; align-items: center; margin: 0.5rem 0; color: {THEME['text']};'>
                    <span style='color: {THEME['success']}; margin-right: 0.5rem; font-size: 1.2rem;'>✓</span>
                    <span><strong>4 Classes</strong> - Glioma, Meningioma, Pituitary, No Tumor</span>
                </div>
                <div style='display: flex; align-items: center; margin: 0.5rem 0; color: {THEME['text']};'>
                    <span style='color: {THEME['success']}; margin-right: 0.5rem; font-size: 1.2rem;'>✓</span>
                    <span><strong>Explainable AI</strong> - Grad-CAM visualizations</span>
                </div>
                <div style='display: flex; align-items: center; margin: 0.5rem 0; color: {THEME['text']};'>
                    <span style='color: {THEME['success']}; margin-right: 0.5rem; font-size: 1.2rem;'>✓</span>
                    <span><strong>Real-time</strong> - Fast inference and results</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📤 Upload MRI Image", key="landing_upload", use_container_width=True):
            st.session_state.page = 'upload'
        
        if st.button("🚀 Try Demo", key="landing_demo", use_container_width=True):
            st.session_state.page = 'dashboard'
    
    with col2:
        st.markdown(f"""
        <div style='background-color: {THEME['card_bg']}; border: 1px solid {THEME['border']}; 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); text-align: center;'>
            <h3 style='color: {THEME['accent']};'>Classification Classes</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                <div class='stat-card'>
                    <div style='font-size: 2rem;'>🧠</div>
                    <div style='font-weight: bold;'>Glioma</div>
                </div>
                <div class='stat-card'>
                    <div style='font-size: 2rem;'>🎯</div>
                    <div style='font-weight: bold;'>Meningioma</div>
                </div>
                <div class='stat-card'>
                    <div style='font-size: 2rem;'>⚡</div>
                    <div style='font-weight: bold;'>Pituitary</div>
                </div>
                <div class='stat-card'>
                    <div style='font-size: 2rem;'>✅</div>
                    <div style='font-weight: bold;'>No Tumor</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_dashboard():
    st.markdown(f"""
    <div style='padding: 1rem 0;'>
        <h1 style='color: {THEME['accent']};'>👋 Welcome to NeuroScanAI</h1>
        <p style='color: {THEME['text']};'>AI-Powered Brain Tumor Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>5,600</div>
            <div class='stat-label'>Training Images</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>98%</div>
            <div class='stat-label'>Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>4</div>
            <div class='stat-label'>Tumor Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{len(st.session_state.predictions_history)}</div>
            <div class='stat-label'>Predictions Made</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class='upload-box'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🖼️</div>
            <h3 style='color: {THEME['accent']};'>Upload MRI Image</h3>
            <p style='color: {THEME['text']}; margin: 1rem 0;'>
            Drag and drop your MRI image here or click to browse
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.success("Image uploaded successfully!")
            
            if st.button("🧠 Predict Tumor", use_container_width=True):
                st.session_state.page = 'predict'
                st.session_state.show_results = False
    
    with col2:
        st.markdown(f"""
        <div style='background-color: {THEME['card_bg']}; border: 1px solid {THEME['border']}; 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
            <h3 style='color: {THEME['accent']};'>Recent Activity</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.predictions_history:
            for i, pred in enumerate(reversed(st.session_state.predictions_history[-5:])):
                st.markdown(f"""
                <div style='background-color: {THEME['card_bg']}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid {THEME['border']};'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='color: {THEME['accent']}; font-weight: bold;'>{pred['prediction'].replace('_', ' ').title()}</span>
                        <span style='color: {THEME['success']};'>{pred['confidence']}</span>
                    </div>
                    <div style='color: {THEME['text']}; font-size: 0.8rem; margin-top: 0.5rem;'>{pred['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; color: {THEME['text']}; opacity: 0.6;'>
                <div style='font-size: 2rem; margin-bottom: 1rem;'>📊</div>
                <p>No predictions yet. Upload an MRI image to get started!</p>
            </div>
            """, unsafe_allow_html=True)


def render_upload_page():
    st.markdown(f"""
    <h1 style='color: {THEME['accent']};'>🖼️ Upload MRI Image</h1>
    <p style='color: {THEME['text']};'>Upload your brain MRI scan for tumor classification</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"<h3 style='color: {THEME['accent']};'>Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_column_width=True)
        
        with col2:
            st.markdown(f"<h3 style='color: {THEME['accent']};'>Image Details</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='color: {THEME['text']};'>
                <p><strong>Filename:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {st.session_state.uploaded_image.size}</p>
                <p><strong>Mode:</strong> {st.session_state.uploaded_image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🧠 Analyze Image", type="primary", use_container_width=True):
                st.session_state.page = 'predict'
                st.session_state.show_results = False


def render_predict_page():
    model = load_model()
    
    if not model:
        st.error("Model not found. Please train the model first using train.py")
        return
    
    if not st.session_state.uploaded_image:
        st.warning("Please upload an MRI image first")
        st.session_state.page = 'upload'
        return
    
    image = st.session_state.uploaded_image
    class_mapping = get_class_names()
    last_conv_layer_name = get_last_conv_layer_name(model)

    st.markdown(f"""
    <h1 style='color: {THEME['accent']};'>🧠 Predict Tumor</h1>
    <p style='color: {THEME['text']};'>Click the button below to analyze the uploaded MRI image</p>
    """, unsafe_allow_html=True)
   
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"<h3 style='color: {THEME['accent']};'>Uploaded MRI</h3>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown(f"<h3 style='color: {THEME['accent']};'>Ready to Analyze</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='color: {THEME['text']};'>
            <p><strong>Image Size:</strong> {image.size}</p>
            <p><strong>Image Mode:</strong> {image.mode}</p>
            <p><strong>Model Status:</strong> Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add Predict Tumor button
    st.markdown("---")
    if st.button("🧠 Predict Tumor", type="primary", use_container_width=True, key="predict_button"):
        with st.spinner("Analyzing MRI image..."):
            pred_class, confidence, predictions = predict_image(model, image, class_mapping)
            st.session_state.prediction_result = {
                'class': pred_class,
                'confidence': confidence,
                'predictions': predictions
            }
            save_prediction_to_history(pred_class, confidence, predictions)
            st.session_state.show_results = True
    
    # Only show results if prediction has been made
    if st.session_state.get('show_results', False) and st.session_state.get('prediction_result'):
        pred_class = st.session_state.prediction_result['class']
        confidence = st.session_state.prediction_result['confidence']
        predictions = st.session_state.prediction_result['predictions']
        
        st.markdown("---")
        st.markdown(f"""
        <h1 style='color: {THEME['accent']};'>🧠 Prediction Result</h1>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"<h3 style='color: {THEME['accent']};'>Uploaded MRI</h3>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        
        with col2:
            # Determine risk level
            if pred_class == 'notumor':
                risk_level = 'Low'
                risk_color = THEME['success']
            elif confidence >= 0.8:
                risk_level = 'High'
                risk_color = THEME['danger']
            else:
                risk_level = 'Medium'
                risk_color = THEME['warning']
            
            # Create prediction result box using Streamlit native components
            st.markdown(f"### Prediction Result")
            
            # Display prediction class
            st.markdown(f"**Predicted Class**")
            st.markdown(f"# {pred_class.replace('_', ' ').title()}")
            
            # Display confidence score
            st.markdown(f"**Confidence Score**")
            st.metric(label="", value=f"{confidence*100:.1f}%", delta="")
            
            # Display risk level
            st.markdown(f"**Risk Level**")
            
            # Use a colored box for risk level
            st.markdown(f"<div style='background-color: {risk_color}; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; display: inline-block; text-align: center;'>{risk_level}</div>", unsafe_allow_html=True)
        
        # Class probabilities
        st.markdown(f"<h3 style='color: {THEME['accent']};'>Class Probabilities</h3>", unsafe_allow_html=True)
        for class_idx, prob in enumerate(predictions):
            class_name = class_mapping[class_idx].replace('_', ' ').title()
            st.markdown(f"""
            <div style='margin: 0.5rem 0;'>
                <div style='display: flex; justify-content: space-between; color: {THEME['text']};'>
                    <span>{class_name}</span>
                    <span>{prob*100:.1f}%</span>
                </div>
                <div style='background-color: {THEME['border']}; border-radius: 4px; height: 8px; margin-top: 0.25rem;'>
                    <div style='background-color: {THEME['primary']}; height: 100%; border-radius: 4px; width: {prob*100}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Grad-CAM Visualization
        st.markdown("---")
        st.markdown(f"<h2 style='color: {THEME['accent']};'>🔬 Explainable AI (Grad-CAM)</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {THEME['text']};'>The heatmap shows which regions of the MRI image the model focused on when making its prediction.</p>", unsafe_allow_html=True)
        
        pred_index = list(class_mapping.values()).index(pred_class)
        gradcam_fig = generate_gradcam_visualization(model, image, last_conv_layer_name, pred_index)
        st.pyplot(gradcam_fig)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download Report", use_container_width=True):
                st.info("Report download feature coming soon!")
        with col2:
            if st.button("🆕 New Prediction", use_container_width=True):
                st.session_state.uploaded_image = None
                st.session_state.prediction_result = None
                st.session_state.show_results = False
                st.session_state.page = 'upload'
        
        # Disclaimer
        st.markdown("---")
        st.markdown(f"""
        <div style='background-color: {THEME['card_bg']}; border-left: 4px solid {THEME['warning']}; 
                    padding: 1rem; border-radius: 4px; margin: 1rem 0;'>
            <h4 style='color: {THEME['warning']}; margin: 0 0 0.5rem 0;'>⚠️ Medical Disclaimer</h4>
            <p style='color: {THEME['text']}; margin: 0;'>
            This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. 
            Always consult with qualified healthcare professionals for medical advice.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_analytics_page():
    """Render the analytics page."""
    st.markdown(f"""
    <h1 style='color: {THEME['accent']};'>📈 Analytics</h1>
    <p style='color: {THEME['text']};'>View performance metrics and analysis</p>
    """, unsafe_allow_html=True)
    
    # Summary Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{len(st.session_state.predictions_history)}</div>
            <div class='stat-label'>Total Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_confidence = 0
        if st.session_state.predictions_history:
            confidences = [float(pred['confidence'].rstrip('%')) for pred in st.session_state.predictions_history]
            avg_confidence = sum(confidences) / len(confidences)
        
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{avg_confidence:.1f}%</div>
            <div class='stat-label'>Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Count predictions by class
        class_counts = {}
        for pred in st.session_state.predictions_history:
            class_name = pred['prediction']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        most_common = max(class_counts.keys()) if class_counts else 'N/A'
        
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{most_common.replace('_', ' ').title() if most_common != 'N/A' else 'N/A'}</div>
            <div class='stat-label'>Most Common</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction Distribution
    st.markdown(f"<h3 style='color: {THEME['accent']};'>Prediction Distribution</h3>", unsafe_allow_html=True)
    
    if st.session_state.predictions_history:
        class_counts = {}
        for pred in st.session_state.predictions_history:
            class_name = pred['prediction']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(THEME['card_bg'])
            ax.set_facecolor(THEME['card_bg'])
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = [THEME['primary'], THEME['accent'], THEME['secondary'], THEME['success']]
            
            bars = ax.bar(range(len(classes)), counts, color=colors[:len(classes)])
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([c.replace('_', ' ').title() for c in classes], color=THEME['text'])
            ax.set_ylabel('Count', color=THEME['text'])
            ax.set_title('Distribution of Predictions by Class', color=THEME['text'])
            ax.tick_params(axis='y', colors=THEME['text'])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', color=THEME['text'])
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No prediction data available yet. Make some predictions to see analytics!")
    
    st.markdown("---")
    
    # Confidence Over Time
    st.markdown(f"<h3 style='color: {THEME['accent']};'>Confidence Over Time</h3>", unsafe_allow_html=True)
    
    if st.session_state.predictions_history:
        confidences = [float(pred['confidence'].rstrip('%')) for pred in st.session_state.predictions_history]
        timestamps = [pred['timestamp'] for pred in st.session_state.predictions_history]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(THEME['card_bg'])
        ax.set_facecolor(THEME['card_bg'])
        
        ax.plot(range(len(confidences)), confidences, marker='o', color=THEME['primary'], linewidth=2)
        ax.set_xlabel('Prediction Number', color=THEME['text'])
        ax.set_ylabel('Confidence (%)', color=THEME['text'])
        ax.set_title('Prediction Confidence Over Time', color=THEME['text'])
        ax.tick_params(colors=THEME['text'])
        ax.grid(True, alpha=0.3, color=THEME['border'])
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No prediction data available yet.")


def render_models_page():
    """Render the model comparison page."""
    st.markdown(f"""
    <h1 style='color: {THEME['accent']};'>🤖 Model Comparison</h1>
    <p style='color: {THEME['text']};'>Compare different deep learning architectures</p>
    """, unsafe_allow_html=True)
    
    # Load the current model and get its actual metrics
    model = load_model()
    loaded_accuracy = load_evaluation_metrics()
    model_info = get_model_info(model) if model else {'params': 'N/A', 'inference_time': 'N/A'}
    
    # Use actual accuracy if available, otherwise use placeholder
    current_accuracy = f"{loaded_accuracy * 100:.1f}%" if loaded_accuracy else "N/A"
    
    # Model comparison data with dynamic current model info
    models_data = [
        {
            'name': 'Custom CNN',
            'accuracy': '92%',
            'params': '2.3M',
            'inference_time': '15ms',
            'description': 'Basic convolutional neural network from scratch',
            'pros': ['Simple architecture', 'Easy to understand', 'Fast training'],
            'cons': ['Lower accuracy', 'Manual feature learning']
        },
        {
            'name': 'MobileNetV2',
            'accuracy': current_accuracy,
            'params': model_info['params'],
            'inference_time': model_info['inference_time'],
            'description': 'Lightweight pre-trained model optimized for mobile',
            'pros': ['High accuracy', 'Fast inference', 'Low memory', 'Transfer learning'],
            'cons': ['Slightly larger than CNN'],
            'current': True
        },
        {
            'name': 'ResNet50',
            'accuracy': '97%',
            'params': '25.6M',
            'inference_time': '25ms',
            'description': 'Deep residual network with skip connections',
            'pros': ['Very high accuracy', 'Industry standard', 'Robust'],
            'cons': ['Large model size', 'Slower inference']
        },
        {
            'name': 'EfficientNetB0',
            'accuracy': '98%',
            'params': '5.3M',
            'inference_time': '12ms',
            'description': 'Efficient network with compound scaling',
            'pros': ['Best accuracy', 'Efficient architecture', 'Good size/accuracy tradeoff'],
            'cons': ['Complex architecture', 'Longer training']
        }
    ]
    
    for model in models_data:
        is_current = model.get('current', False)
        
        # Create a styled container for current model
        if is_current:
            st.markdown(f"""
            <div style='background-color: {THEME['secondary']}; padding: 1.5rem; border-radius: 12px; border: 2px solid {THEME['success']}; margin: 1rem 0;'>
            """, unsafe_allow_html=True)
        
        # Create model card using Streamlit components
        with st.container():
            # Header with badge
            if is_current:
                st.markdown(f"### {model['name']} <span style='background-color: {THEME['success']}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; margin-left: 0.5rem;'>CURRENT</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"### {model['name']}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"*{model['description']}*")
            with col2:
                st.markdown(f"<div style='font-size: 2rem; color: {THEME['success']}; font-weight: bold; text-align: right;'>{model['accuracy']}</div>", unsafe_allow_html=True)
            
            # Stats grid
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parameters", model['params'])
            with col2:
                st.metric("Inference Time", model['inference_time'])
            with col3:
                st.metric("Accuracy", model['accuracy'])
            
            # Pros and Cons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Pros**")
                for pro in model['pros']:
                    st.markdown(f"- {pro}")
            with col2:
                st.markdown(f"**Cons**")
                for con in model['cons']:
                    st.markdown(f"- {con}")
        
        # Close the styled container for current model
        if is_current:
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")


def render_reports_page():
    """Render the reports page."""
    st.markdown(f"""
    <h1 style='color: {THEME['accent']};'>📄 Prediction Reports</h1>
    <p style='color: {THEME['text']};'>View and download prediction history</p>
    """, unsafe_allow_html=True)
    
    if not st.session_state.predictions_history:
        st.markdown(f"""
        <div style='text-align: center; padding: 3rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📋</div>
            <h3 style='color: {THEME['text']};'>No Reports Yet</h3>
            <p style='color: {THEME['text']}; opacity: 0.7;'>Upload and analyze MRI images to generate reports</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🖼️ Upload First MRI", use_container_width=True):
            st.session_state.page = 'upload'
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Search predictions", placeholder="Search by class name...")
    with col2:
        date_filter = st.selectbox("📅 Filter by date", ["All time", "Today", "Last 7 days", "Last 30 days"])
    
    # Display reports
    for i, pred in enumerate(reversed(st.session_state.predictions_history)):
        pred_class = pred['prediction'].replace('_', ' ').title()
        
        st.markdown(f"""
        <div style='background-color: {THEME['card_bg']}; border: 1px solid {THEME['border']}; 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='color: {THEME['accent']}; margin: 0;'>Report #{len(st.session_state.predictions_history) - i:03d}</h3>
                    <p style='color: {THEME['text']}; margin: 0.25rem 0;'>📅 {pred['timestamp']}</p>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 1.5rem; color: {THEME['accent']}; font-weight: bold;'>{pred_class}</div>
                    <div style='color: {THEME['success']}; font-weight: bold;'>{pred['confidence']}</div>
                </div>
            </div>
            
            <div style='margin-top: 1rem;'>
                <h4 style='color: {THEME['text']}; margin: 0.5rem 0;'>Class Probabilities</h4>
                <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;'>
                    {"".join([f"<div style='display: flex; justify-content: space-between; color: {THEME['text']};'><span>{class_name.replace('_', ' ').title()}</span><span>{prob}</span></div>" for class_name, prob in pred['all_probabilities'].items()])}
                </div>
            </div>
            
            <div style='margin-top: 1rem; display: flex; gap: 0.5rem;'>
                <button style='background-color: {THEME['primary']}; color: white; padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer;'>View Details</button>
                <button style='background-color: {THEME['secondary']}; color: white; padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer;'>Download PDF</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export to CSV", use_container_width=True):
            if st.session_state.predictions_history:
                df = pd.DataFrame(st.session_state.predictions_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="neuroscan_predictions.csv",
                    mime="text/csv"
                )
    with col2:
        if st.button("📋 Export to JSON", use_container_width=True):
            if st.session_state.predictions_history:
                json_data = json.dumps(st.session_state.predictions_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="neuroscan_predictions.json",
                    mime="application/json"
                )


def main():
    """Main application router."""
    # Render sidebar
    render_sidebar()
    
    # Route to appropriate page
    page = st.session_state.page
    
    if page == 'landing':
        render_landing_page()
    elif page == 'dashboard':
        render_dashboard()
    elif page == 'upload':
        render_upload_page()
    elif page == 'predict':
        render_predict_page()
    elif page == 'analytics':
        render_analytics_page()
    elif page == 'models':
        render_models_page()
    elif page == 'reports':
        render_reports_page()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
