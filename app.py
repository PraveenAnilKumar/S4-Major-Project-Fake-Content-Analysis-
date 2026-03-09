# app.py (fully integrated with enhanced Fake News Detection and fixed Sentiment Analysis)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import time
import subprocess
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import hashlib
from pathlib import Path
import glob  # Make sure to import glob

# Fix imports - use consistent naming
from deepfake_detector_advanced import deepfake_detector
from fake_news_detector import FakeNewsDetector
from sentiment_analyzer import SentimentAnalyzer
from utils import utils

# Import optional sentiment extensions
try:
    from aspect_sentiment import AspectSentimentAnalyzer
    ASPECT_AVAILABLE = True
except:
    ASPECT_AVAILABLE = False

try:
    from batch_sentiment import BatchSentimentProcessor
    BATCH_PROCESSOR_AVAILABLE = True
except:
    BATCH_PROCESSOR_AVAILABLE = False

try:
    from sentiment_viz import SentimentVisualizer
    VIZ_AVAILABLE = True
except:
    VIZ_AVAILABLE = False

# Initialize sentiment analyzer with ensemble
sentiment_analyzer = SentimentAnalyzer(use_ensemble=True)

# Initialize optional analyzers
if ASPECT_AVAILABLE:
    aspect_analyzer = AspectSentimentAnalyzer()
if BATCH_PROCESSOR_AVAILABLE:
    batch_processor = BatchSentimentProcessor()
if VIZ_AVAILABLE:
    viz = SentimentVisualizer()

# ---------- Initialize fake news detector ----------
fake_news_detector = FakeNewsDetector(use_transformer=True)

# Check if model exists and load it - UPDATED to use new auto-loading
model_path = 'models/fake_news/'
st.session_state['fake_news_loaded'] = fake_news_detector.is_trained

# ---------- Page config ----------
st.set_page_config(page_title="TruthGuard AI", page_icon="🛡️", layout="wide")

# ---------- User database file ----------
USER_DB = "users.json"

def hash_password(password):
    """Return SHA-256 hash of password."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load user database from JSON file."""
    if Path(USER_DB).exists():
        with open(USER_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save user database to JSON file."""
    with open(USER_DB, 'w') as f:
        json.dump(users, f, indent=2)

# ---------- Initialize session state ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

# Initialize session state for sentiment
if 'last_sentiment' not in st.session_state:
    st.session_state.last_sentiment = ('NEUTRAL', 0.5)
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'aspect_text' not in st.session_state:
    st.session_state.aspect_text = ""

# Initialize session state for sentiment text input - FIXED: Don't set value here
if 'sentiment_text_input' not in st.session_state:
    st.session_state.sentiment_text_input = ""

# Initialize session state for fake news text input
if 'fn_text' not in st.session_state:
    st.session_state.fn_text = ""

# Initialize session state for deepfake model selection
if 'selected_df_model' not in st.session_state:
    st.session_state.selected_df_model = "Ensemble (All Models)"
if 'df_model_weights' not in st.session_state:
    st.session_state.df_model_weights = {}

# Initialize session state for fake news model selection
if 'selected_fn_model' not in st.session_state:
    st.session_state.selected_fn_model = None

# Initialize session state for clear flags
if 'clear_sentiment' not in st.session_state:
    st.session_state.clear_sentiment = False
if 'clear_fn' not in st.session_state:
    st.session_state.clear_fn = False

# ---------- Custom CSS (enhanced) ----------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font Change */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc !important; /* Update global text to light color */
    }

    /* General Dark/Light Background Tweaks for Premium Feel */
    .stApp {
        background-color: #0f172a; /* Dark background */
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); /* Lighter gradient for dark bg */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #cbd5e1; /* Lighter slate for subheader */
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* Cards & Containers - Glassmorphism & Shadow */
    .metric-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1.8rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }

    /* Results styling with glowing and pulsing effects */
    .result-safe {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
        animation: pulse-green 3s infinite;
    }
    .result-fake {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
        animation: pulse-red 3s infinite;
    }
    .result-suspicious {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
        animation: pulse-orange 3s infinite;
    }

    /* Animations */
    @keyframes pulse-green {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 0 35px rgba(16, 185, 129, 0.6); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
    }
    @keyframes pulse-red {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 0 35px rgba(239, 68, 68, 0.6); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    }
    @keyframes pulse-orange {
        0% { transform: scale(1); box-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 0 35px rgba(245, 158, 11, 0.6); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
    }

    /* Sidebar User Card */
    .sidebar-user-card {
        background: rgba(30, 41, 59, 0.8); /* Darker glass effect */
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        border-left: 5px solid #818cf8; /* Lighter indigo */
        transition: transform 0.2s;
    }
    .sidebar-user-card:hover {
        transform: translateX(5px);
    }
    .sidebar-user-card h4 {
        margin: 0;
        color: #f8fafc; /* Light text */
        font-size: 1.2rem;
        font-weight: 700;
    }
    .sidebar-user-card p {
        margin: 0.4rem 0 0;
        color: #cbd5e1; /* Lighter slate text */
        font-size: 0.95rem;
    }

    /* Streamlit Button Restyling */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1); /* Light border */
        background: #1e293b; /* Dark slate background */
        color: #f8fafc; /* Light text */
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:hover {
        background: #334155; /* Slightly lighter slate */
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
        color: #818cf8; /* Light indigo text on hover */
        border-color: #6366f1;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary buttons (e.g., forms, analyzing) often get kind="primary" in standard Streamlit styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%); /* Adjusted primary gradient */
        color: white;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }

    /* Container padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Model Badges */
    .model-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-right: 0.5rem;
        text-transform: uppercase;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .model-badge-primary { background: linear-gradient(135deg, #3b82f6, #60a5fa); color: white; }
    .model-badge-ensemble { background: linear-gradient(135deg, #8b5cf6, #a78bfa); color: white; }
    .model-badge-single { background: linear-gradient(135deg, #10b981, #34d399); color: white; }
    .model-badge-transformer { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: white; }
    .model-badge-rf { background: linear-gradient(135deg, #10b981, #34d399); color: white; }

    /* Weight Sliders container */
    .weight-slider {
        padding: 1rem;
        background: #1e293b; /* Darker bg */
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        border: 1px solid #334155;
        transition: box-shadow 0.3s ease;
    }
    .weight-slider:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
        color: #f8fafc; /* Light text */
        background-color: #1e293b; /* Darker background */
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .streamlit-expanderHeader:hover {
        background-color: #334155;
        color: #818cf8; /* Hover color */
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5); /* Darker tab list bg */
        padding: 5px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        color: #94a3b8; /* Dimmer text for inactive tabs */
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #818cf8;
        background: rgba(129, 140, 248, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important; /* Dark bg for selected tab */
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        color: #818cf8 !important; /* Indigo text for selected */
        border-bottom: 3px solid #6366f1 !important;
    }
    
    /* Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #334155; /* Darker border */
        background-color: #0f172a; /* Darker input bg */
        color: #f8fafc; /* Light text */
        transition: all 0.3s;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #818cf8;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Login / Registration ----------
def login_tab():
    st.sidebar.markdown("## 🔐 Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login", key="login_button"):
        if not username or not password:
            st.sidebar.error("Please enter both username and password.")
        else:
            users = load_users()
            # Also allow hardcoded admin for fallback
            if username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.session_state.username = "admin"
                st.session_state.role = "admin"
                st.rerun()
            elif username in users and users[username]["password"] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = users[username].get("role", "user")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")

def register_tab():
    st.sidebar.markdown("## 📝 Register")
    with st.sidebar.form("register_form"):
        new_user = st.text_input("Choose a username")
        new_pass = st.text_input("Choose a password", type="password")
        confirm_pass = st.text_input("Confirm password", type="password")
        # Optional admin key – in a real system, keep this secret.
        admin_key = st.text_input("Admin key (optional)", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            if not new_user or not new_pass:
                st.sidebar.error("Username and password required.")
            elif new_pass != confirm_pass:
                st.sidebar.error("Passwords do not match.")
            else:
                users = load_users()
                if new_user in users:
                    st.sidebar.error("Username already exists.")
                else:
                    # Determine role: if admin_key matches a secret, set admin; otherwise user.
                    ADMIN_SECRET = "truthguard2024"  # change this!
                    role = "admin" if admin_key == ADMIN_SECRET else "user"
                    users[new_user] = {
                        "password": hash_password(new_pass),
                        "role": role,
                        "created": datetime.now().isoformat()
                    }
                    save_users(users)
                    st.sidebar.success("Registration successful! You can now log in.")

# ---------- Show login/register if not authenticated ----------
if not st.session_state.authenticated:
    st.markdown('<h1 class="main-header">🛡️ TruthGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Please log in or register to continue</p>', unsafe_allow_html=True)
    login_tab()
    register_tab()
    st.stop()

# ---------- Sidebar for authenticated users ----------
with st.sidebar:
    st.image("https://via.placeholder.com/300x80/4a90e2/ffffff?text=TruthGuard+AI", use_container_width=True)
    
    st.markdown(f"""
    <div class="sidebar-user-card">
        <h4>👤 {st.session_state.username}</h4>
        <p>Role: <strong>{st.session_state.role.upper()}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    mode = st.radio(
        "Select Feature",
        ["🏠 Home", "📸 Deepfake Detection", "📰 Fake News Detection", "😊 Sentiment Analysis"],
        label_visibility="collapsed"
    )
    mode_map = {
        "🏠 Home": "Home",
        "📸 Deepfake Detection": "Deepfake Detection",
        "📰 Fake News Detection": "Fake News Detection",
        "😊 Sentiment Analysis": "Sentiment Analysis"
    }
    mode = mode_map[mode]
    
    st.markdown("---")
    
    # ========== ENHANCED ADMIN TOOLS SECTION ==========
    if st.session_state.role == "admin":
        with st.expander("🔧 Admin Tools", expanded=False):
            st.markdown("#### Train Models")
            
            # ===== DEEPFAKE TRAINING SECTION =====
            st.markdown("##### 🎭 Deepfake Detection Models")
            
            # Get available models for fine-tuning
            try:
                model_info = deepfake_detector.get_model_info()
                available_models = model_info['model_names']
            except:
                available_models = []
            
            # Training options
            model_to_train = st.selectbox(
                "Select model to train",
                ["Train New Model"] + available_models,
                key="deepfake_model_select"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox(
                    "Model architecture",
                    ["efficientnet", "mobilenet", "resnet", "custom"],
                    key="model_architecture"
                )
            with col2:
                training_mode = st.selectbox(
                    "Training mode",
                    ["Quick (5-10 epochs)", "Standard (20 epochs)", "Deep (50 epochs)"],
                    key="training_mode"
                )
            
            # Map training mode to epochs
            epoch_map = {
                "Quick (5-10 epochs)": 10,
                "Standard (20 epochs)": 20,
                "Deep (50 epochs)": 50
            }
            epochs = epoch_map[training_mode]
            
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.selectbox("Batch size", [16, 32, 64], index=1, key="df_batch_size")
            with col2:
                use_augmentation = st.checkbox("Use data augmentation", value=True, key="use_aug")
            
            # Advanced options expander
            with st.expander("⚙️ Advanced Options"):
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    value=1e-4,
                    key="df_lr"
                )
                validation_split = st.slider("Validation split", 0.1, 0.3, 0.2, key="df_val_split")
                early_stopping = st.checkbox("Use early stopping", value=True, key="df_early_stop")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Training", use_container_width=True, key="start_df_training"):
                    # Build command
                    cmd = [
                        "python", "train_deepfake.py",
                        f"--model-type={model_type}",
                        f"--epochs={epochs}",
                        f"--batch-size={batch_size}",
                        f"--learning-rate={learning_rate}",
                        f"--validation-split={validation_split}"
                    ]
                    
                    if use_augmentation:
                        cmd.append("--use-augmentation")
                    
                    if early_stopping:
                        cmd.append("--early-stopping")
                    
                    # Add fine-tuning path if existing model selected
                    if model_to_train != "Train New Model":
                        # Try different possible extensions
                        possible_paths = [
                            f"models/{model_to_train}.h5",
                            f"models/{model_to_train}",
                            f"models/deepfake_{model_to_train}.h5"
                        ]
                        
                        model_path = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                model_path = path
                                break
                        
                        if model_path:
                            cmd.append(f"--fine-tune={model_path}")
                            st.info(f"Fine-tuning {model_to_train} for {epochs//2} epochs...")
                        else:
                            st.warning(f"Model file not found. Training from scratch instead.")
                    
                    # Show command in info
                    st.info(f"Training started: {' '.join(cmd)}")
                    
                    # Run training in background
                    try:
                        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                        st.success("✅ Training process launched! Check console for progress.")
                    except Exception as e:
                        st.error(f"Error launching training: {e}")
            
            with col2:
                if st.button("📊 View Training History", use_container_width=True, key="view_history"):
                    # Show latest training plots
                    history_files = glob.glob("models/training_history_*.png")
                    eval_files = glob.glob("models/evaluation_*.json")
                    
                    if history_files:
                        latest = max(history_files, key=os.path.getctime)
                        st.image(latest, caption=f"Latest Training History: {os.path.basename(latest)}")
                    else:
                        st.warning("No training history found")
                    
                    # Show latest evaluation results
                    if eval_files:
                        latest_eval = max(eval_files, key=os.path.getctime)
                        try:
                            with open(latest_eval, 'r') as f:
                                eval_data = json.load(f)
                            st.json(eval_data)
                        except:
                            pass
            
            st.markdown("---")
            
            # ===== SENTIMENT ANALYSIS FINE-TUNING SECTION =====
            st.markdown("##### 😊 Sentiment Analysis Fine-Tuning")
            
            with st.expander("Fine-tune Sentiment Models", expanded=False):
                st.info("Upload a CSV with 'text' and 'sentiment' columns (POSITIVE, NEUTRAL, NEGATIVE)")
                
                # Check if transformers are available
                try:
                    from transformers import pipeline
                    transformers_available = True
                except:
                    transformers_available = False
                
                if not transformers_available:
                    st.warning("⚠️ Transformers library not available. Install with: pip install transformers torch")
                
                sentiment_file = st.file_uploader(
                    "Upload training data (CSV)", 
                    type=['csv'], 
                    key="sentiment_train_upload"
                )
                
                if sentiment_file:
                    try:
                        df = pd.read_csv(sentiment_file)
                        st.write("📊 Preview:", df.head())
                        
                        # Show basic stats
                        st.write(f"Total rows: {len(df)}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            text_col = st.selectbox("Text column", df.columns, key="sent_text_col")
                        with col2:
                            label_col = st.selectbox("Label column", df.columns, key="sent_label_col")
                        with col3:
                            # Show label distribution
                            if label_col in df.columns:
                                label_counts = df[label_col].value_counts()
                                st.write("Label distribution:")
                                st.dataframe(label_counts)
                        
                        # Training parameters
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            epochs = st.slider("Epochs", 1, 10, 3, key="sent_epochs")
                        with col2:
                            batch_size = st.selectbox("Batch size", [8, 16, 32], index=1, key="sent_batch_size")
                        with col3:
                            learning_rate = st.select_slider(
                                "Learning rate",
                                options=[1e-5, 2e-5, 3e-5, 5e-5],
                                value=2e-5,
                                key="sent_lr"
                            )
                        
                        validation_split = st.slider("Validation split", 0.1, 0.3, 0.1, key="sent_val_split")
                        
                        if st.button("🚀 Start Fine-Tuning", key="start_sentiment_training", use_container_width=True):
                            texts = df[text_col].astype(str).tolist()
                            labels = df[label_col].astype(str).tolist()
                            
                            # Validate labels
                            valid_labels = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
                            invalid_labels = set(labels) - set(valid_labels)
                            if invalid_labels:
                                st.error(f"Invalid labels found: {invalid_labels}. Must be POSITIVE, NEUTRAL, or NEGATIVE")
                            else:
                                with st.spinner("Fine-tuning sentiment model... This may take several minutes..."):
                                    try:
                                        history = sentiment_analyzer.fine_tune(
                                            texts=texts,
                                            labels=labels,
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size,
                                            validation_split=validation_split,
                                            save_model=True
                                        )
                                        
                                        if 'error' in history:
                                            st.error(f"Error: {history['error']}")
                                        else:
                                            st.success("✅ Fine-tuning completed successfully!")
                                            
                                            # Display results
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Accuracy", f"{history.get('eval_accuracy', 0):.2%}")
                                            with col2:
                                                st.metric("Precision", f"{history.get('eval_precision', 0):.2%}")
                                            with col3:
                                                st.metric("Recall", f"{history.get('eval_recall', 0):.2%}")
                                            
                                            st.metric("F1 Score", f"{history.get('eval_f1', 0):.2%}")
                                            
                                            with st.expander("📊 Detailed Results"):
                                                st.json(history)
                                            
                                            st.info("The fine-tuned model is now active for sentiment analysis!")
                                            
                                    except Exception as e:
                                        st.error(f"Error during fine-tuning: {e}")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {e}")
            
            st.markdown("---")
            
            # ===== UPDATED FAKE NEWS TRAINING SECTION =====
            st.markdown("##### 📰 Fake News Training")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Train Random Forest", use_container_width=True, key="train_rf"):
                    cmd = [
                        "python", "train_fakenews_simple.py", 
                        "--dataset", "datasets/fake_news/all_fake_news_combined.csv",
                        "--model-type", "random_forest",
                        "--test-size", "0.2"
                    ]
                    st.info(f"Training started: {' '.join(cmd)}")
                    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    st.success("✅ Random Forest training launched! This will take 5-15 minutes.")

            with col2:
                if st.button("🧠 Train Transformer", use_container_width=True, key="train_transformer"):
                    cmd = [
                        "python", "train_fakenews_transformer.py",
                        "--dataset", "datasets/fake_news/all_fake_news_combined.csv",
                        "--model-name", "distilbert-base-uncased",
                        "--epochs", "2",
                        "--batch-size", "8",
                        "--max-length", "128"
                    ]
                    st.info(f"Transformer training started. This will take 1-2 hours.")
                    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    st.success("✅ Transformer training launched! Check console for progress.")

            # View trained models
            if st.button("📋 View Trained Models", use_container_width=True, key="view_fake_news_models"):
                models = fake_news_detector.get_available_models()
                if models:
                    st.write("### Available Models")
                    for model in models:
                        badge = "🟢" if model['type'] == 'transformer' else "🟡"
                        st.write(f"{badge} **{model['name'][:50]}**")
                        st.caption(f"  Type: {model['type']}, Accuracy: {model.get('accuracy', 'N/A')}")
                else:
                    st.warning("No trained models found. Train one first!")
            
            st.markdown("---")
            
            # ===== DATASET UPLOAD SECTION =====
            st.markdown("#### Upload Datasets")
            st.caption("Place files in the appropriate folders:")
            st.code("""
datasets/deepfake/train/real/
datasets/deepfake/train/fake/
datasets/deepfake/test/real/
datasets/deepfake/test/fake/
datasets/fake_news/all_fake_news_combined.csv
datasets/sentiment/train.csv
            """)
            uploaded_file = st.file_uploader("Upload dataset file", type=['csv','zip','jpg','png'], key="admin_upload")
            if uploaded_file is not None:
                os.makedirs("temp", exist_ok=True)
                with open(os.path.join("temp", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved to temp/{uploaded_file.name} (admin can move to correct folder)")

# ---------- Header ----------
st.markdown('<h1 class="main-header">🛡️ TruthGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Media Authenticity & Analysis</p>', unsafe_allow_html=True)

# -------------------- Home Page --------------------
if mode == "Home":
    st.markdown("## 🌟 Welcome to TruthGuard AI")
    st.markdown("TruthGuard AI is your comprehensive platform for detecting digital deception and analyzing media authenticity. Explore our powerful tools below:")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="metric-card" style="height: 100%;">
            <h3 style="margin-top: 0; font-size: 1.5rem;">📸 Deepfake Detection</h3>
            <p style="font-size: 1rem; opacity: 0.9;">Analyze images and videos using advanced AI models to detect manipulation and synthetic media generation.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown('''
        <div class="metric-card" style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); height: 100%;">
            <h3 style="margin-top: 0; font-size: 1.5rem;">📰 Fake News Detection</h3>
            <p style="font-size: 1rem; opacity: 0.9;">Verify the authenticity of text articles with NLP models like DistilBERT and Random Forest to combat misinformation.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col3:
        st.markdown('''
        <div class="metric-card" style="background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); height: 100%;">
            <h3 style="margin-top: 0; font-size: 1.5rem;">😊 Sentiment Analysis</h3>
            <p style="font-size: 1rem; opacity: 0.9;">Understand the emotional tone behind text with our ensemble-based sentiment classification system.</p>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 Select a feature from the sidebar navigation to get started.")

# -------------------- Deepfake Detection with Model Switching --------------------
if mode == "Deepfake Detection":
    st.header("📸 Deepfake Detection")
    
    # Get available models from detector
    model_info = deepfake_detector.get_model_info()
    available_models = model_info['model_names']
    
    # Create model selection options
    model_options = ["Ensemble (All Models)"] + available_models
    
    # Model selection section
    with st.expander("🎯 Model Selection", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select Detection Model",
                options=model_options,
                index=0,
                key="model_selector",
                help="Choose which model to use for detection"
            )
            st.session_state.selected_df_model = selected_model
        
        with col2:
            st.metric("Available Models", len(available_models))
            if selected_model == "Ensemble (All Models)":
                st.info("🧠 Using all models for better accuracy")
            else:
                st.success(f"✅ Using single model: {selected_model}")
        
        # Show model info
        if selected_model != "Ensemble (All Models)":
            model_idx = available_models.index(selected_model)
            st.caption(f"Model path: {model_info['model_paths'][model_idx]}")
        
        # Advanced ensemble settings (only when ensemble is selected)
        if selected_model == "Ensemble (All Models)" and st.session_state.role == "admin":
            st.markdown("---")
            st.markdown("#### ⚙️ Ensemble Weight Configuration")
            st.caption("Adjust weights for each model in the ensemble")
            
            weights = {}
            total_weight = 0
            
            # Create sliders for each model
            cols = st.columns(2)
            for i, model_name in enumerate(available_models):
                with cols[i % 2]:
                    default_weight = st.session_state.df_model_weights.get(model_name, 1.0)
                    weight = st.slider(
                        f"{model_name}",
                        min_value=0.0,
                        max_value=2.0,
                        value=default_weight,
                        step=0.1,
                        key=f"weight_{model_name}"
                    )
                    weights[model_name] = weight
                    total_weight += weight
            
            # Show total weight
            st.metric("Total Weight", f"{total_weight:.1f}")
            
            if st.button("Apply Weights", key="apply_weights"):
                st.session_state.df_model_weights = weights
                deepfake_detector.set_model_weights(weights)
                st.success("✅ Model weights updated!")
    
    # Main detection interface
    col1, col2 = st.columns([1, 1])
    with col1:
        thresh = st.slider(
            "Detection Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.65, 
            step=0.05,
            help="Higher threshold = more strict detection"
        )
        deepfake_detector.threshold = thresh
    
    with col2:
        if selected_model == "Ensemble (All Models)":
            st.metric("Active Models", len(available_models))
        else:
            st.metric("Active Model", "1 (Single)")
    
    # Upload tabs
    upload_tab1, upload_tab2 = st.tabs(["📷 Image", "🎥 Video"])
    
    with upload_tab1:
        img_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'bmp'], key="df_img_upload")
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded", use_container_width=True)
            
            if st.button("Analyze Image", key="analyze_img"):
                with st.spinner(f"Analyzing with {selected_model}..."):
                    arr = np.array(img)
                    if len(arr.shape) == 2:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                    elif arr.shape[2] == 4:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                    
                    # Pass selected model to detector
                    if selected_model == "Ensemble (All Models)":
                        res = deepfake_detector.detect_deepfake_ensemble(arr)
                    else:
                        res = deepfake_detector.detect_with_single_model(arr, selected_model)
                    
                    res['filename'] = img_file.name
                    res['model_used'] = selected_model
                    st.session_state['df_result'] = res
    
    with upload_tab2:
        vid_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="df_vid_upload")
        if vid_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(vid_file.read())
            tfile.close()
            st.video(tfile.name)
            
            if st.button("Analyze Video", key="analyze_vid"):
                with st.spinner(f"Analyzing video with {selected_model}..."):
                    if selected_model == "Ensemble (All Models)":
                        res = deepfake_detector.detect_deepfake_video_advanced(tfile.name)
                    else:
                        res = deepfake_detector.detect_video_with_single_model(tfile.name, selected_model)
                    
                    res['filename'] = vid_file.name
                    res['model_used'] = selected_model
                    st.session_state['df_result'] = res
            
            os.unlink(tfile.name)
    
    # Display results
    if 'df_result' in st.session_state:
        r = st.session_state.df_result
        st.markdown("---")
        
        # Show which model was used
        model_used = r.get('model_used', 'Ensemble')
        st.markdown(f"**Model Used:** <span class='model-badge model-badge-ensemble'>{model_used}</span>", 
                   unsafe_allow_html=True)
        
        st.subheader("Result")
        is_df = r['is_deepfake']
        conf = r['confidence']
        
        if is_df and conf > 80:
            st.markdown(f'<div class="result-fake">🚨 DEEPFAKE DETECTED ({conf:.1f}%)</div>', 
                       unsafe_allow_html=True)
        elif not is_df and conf < 30:
            st.markdown(f'<div class="result-safe">✅ AUTHENTIC ({(100-conf):.1f}%)</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-suspicious">⚠️ SUSPICIOUS ({conf:.1f}%)</div>', 
                       unsafe_allow_html=True)
        
        # Display metrics
        colA, colB, colC = st.columns(3)
        colA.metric("Faces", r.get('face_count', 0))
        colB.metric("Confidence Score", f"{r.get('ensemble_score', r.get('confidence', 0))/100:.3f}")
        colC.metric("Consistency", f"{r.get('consistency', 0)*100:.1f}%")
        
        # Show individual model scores if available
        if 'model_scores' in r and r['model_scores']:
            with st.expander("📊 Individual Model Scores"):
                scores_df = pd.DataFrame({
                    'Model': list(r['model_scores'].keys()),
                    'Score': list(r['model_scores'].values())
                }).sort_values('Score', ascending=False)
                
                # Create bar chart
                fig = px.bar(scores_df, x='Model', y='Score', 
                           title="Model Predictions",
                           color='Score',
                           color_continuous_scale=['green', 'yellow', 'red'])
                st.plotly_chart(fig, use_container_width=True)
        
        st.info(r.get('message', ''))

# -------------------- UPDATED FAKE NEWS DETECTION SECTION --------------------
elif mode == "Fake News Detection":
    st.header("📰 Fake News Detection")
    
    # Get available models
    available_models = fake_news_detector.get_available_models()
    model_info = fake_news_detector.get_model_info()
    
    # Model selection and info
    with st.expander("🤖 Model Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if model_info['is_trained']:
                model_type = model_info.get('model_type', 'Unknown')
                badge_class = "model-badge-transformer" if model_info.get('use_transformer') else "model-badge-rf"
                st.markdown(f"**Currently Loaded:** <span class='model-badge {badge_class}'>{model_type}</span>", 
                           unsafe_allow_html=True)
                
                if model_info.get('metadata') and 'accuracy' in model_info['metadata']:
                    st.metric("Model Accuracy", f"{model_info['metadata']['accuracy']:.2%}")
                elif model_info.get('metadata') and 'eval_accuracy' in model_info['metadata']:
                    st.metric("Model Accuracy", f"{model_info['metadata']['eval_accuracy']:.2%}")
            else:
                st.warning("⚠️ No trained model found. Using fallback predictions.")
                st.info("Admin can train models in the sidebar.")
        
        with col2:
            if available_models:
                st.write(f"📊 **Available Models:** {len(available_models)}")
                
                # Create a selectbox for model selection (optional)
                model_names = [m['name'] for m in available_models]
                selected_model_name = st.selectbox(
                    "Switch Model (auto-loads best by default)",
                    ["Auto (Best)"] + model_names,
                    key="fn_model_selector"
                )
                
                if selected_model_name != "Auto (Best)":
                    # Find the selected model
                    selected_model = next((m for m in available_models if m['name'] == selected_model_name), None)
                    if selected_model:
                        if selected_model['type'] == 'transformer':
                            # Load transformer model
                            if fake_news_detector.load_transformer_model(selected_model['path']):
                                st.success(f"✅ Loaded {selected_model['type']} model")
                                st.rerun()
                        else:
                            # Load traditional model
                            if fake_news_detector.load_traditional_model(selected_model['path']):
                                st.success(f"✅ Loaded {selected_model['type']} model")
                                st.rerun()
            else:
                st.info("No trained models available")
    
    # Main detection interface
    method = st.radio("Input", ["Text", "File"], key="fn_method")
    
    if method == "Text":
        # Check if we need to clear
        if st.session_state.clear_fn:
            st.session_state.clear_fn = False
            st.session_state.fn_text = ""
        
        text = st.text_area("Enter news text", height=150, key="fn_text", 
                           placeholder="Paste news article text here...")
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            analyze_clicked = st.button("🔍 Analyze", key="analyze_fake", use_container_width=True)
        with col2:
            clear_clicked = st.button("🗑️ Clear", key="clear_fake", use_container_width=True)
        
        if clear_clicked:
            st.session_state.clear_fn = True
            st.rerun()
        
        if analyze_clicked and text.strip():
            with st.spinner("Analyzing with trained model..."):
                try:
                    # Get prediction
                    label, conf = fake_news_detector.predict(text)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if label == 'FAKE':
                            st.markdown(f'<div class="result-fake">🚨 FAKE NEWS DETECTED</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="result-safe">✅ REAL NEWS</div>', 
                                      unsafe_allow_html=True)
                        
                        # Show confidence meter
                        st.metric("Confidence Score", f"{conf:.2%}")
                        
                        # Show probability bar
                        st.progress(conf, text=f"Fake probability: {conf*100:.1f}%")
                    
                    with col2:
                        # Create confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=conf * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Confidence"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "red" if label == 'FAKE' else "green"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 100], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': conf * 100
                                }
                            }
                        ))
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show model info used
                    model_type = model_info.get('model_type', 'Fallback')
                    if model_info.get('use_transformer'):
                        st.caption(f"🤖 Predicted by: **Transformer Model** ({model_type})")
                    else:
                        st.caption(f"🤖 Predicted by: **Random Forest Model** ({model_type})")
                    
                    # Show word count
                    word_count = len(text.split())
                    st.caption(f"📝 Text length: {word_count} words")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    else:  # File upload
        file = st.file_uploader("Upload file", type=['txt', 'csv'], key="fn_file")
        if file:
            # Load and preview data
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                st.write("📊 File Preview:")
                st.dataframe(df.head())
                
                # Select text column
                text_col = st.selectbox("Select text column", df.columns, key="fn_col")
                texts = df[text_col].astype(str).tolist()
            else:
                content = file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
                st.write(f"📊 Loaded {len(texts)} texts")
                if texts:
                    st.write("Preview:", texts[:3])
            
            # Analysis options
            col1, col2 = st.columns(2)
            with col1:
                max_texts = st.number_input("Max texts to analyze", 
                                           min_value=1, 
                                           max_value=min(500, len(texts)), 
                                           value=min(50, len(texts)),
                                           key="fn_max_texts")
            with col2:
                show_progress = st.checkbox("Show progress", value=True, key="fn_show_progress")
            
            if st.button("🚀 Analyze File", key="analyze_file_fake", use_container_width=True):
                results = []
                progress_bar = st.progress(0) if show_progress else None
                
                for i, t in enumerate(texts[:max_texts]):
                    try:
                        label, conf = fake_news_detector.predict(t)
                        results.append({
                            'text': t[:100] + '...' if len(t) > 100 else t,
                            'label': label,
                            'confidence': f"{conf:.2%}",
                            'confidence_score': conf
                        })
                    except Exception as e:
                        results.append({
                            'text': t[:100] + '...' if len(t) > 100 else t,
                            'label': 'ERROR',
                            'confidence': 'N/A',
                            'confidence_score': 0.0
                        })
                    
                    if show_progress:
                        progress_bar.progress((i + 1) / max_texts)
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    fake_count = len(results_df[results_df['label'] == 'FAKE'])
                    st.metric("Fake News", fake_count)
                with col2:
                    real_count = len(results_df[results_df['label'] == 'REAL'])
                    st.metric("Real News", real_count)
                with col3:
                    error_count = len(results_df[results_df['label'] == 'ERROR'])
                    st.metric("Errors", error_count)
                
                # Average confidence
                valid_results = results_df[results_df['label'] != 'ERROR']
                if len(valid_results) > 0:
                    avg_conf = valid_results['confidence_score'].mean()
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"fakenews_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_fn_btn"
                )
                
                # Show distribution pie chart
                if len(valid_results) > 0:
                    fig = px.pie(valid_results, names='label', 
                                title='Prediction Distribution',
                                color='label',
                                color_discrete_map={'FAKE':'red', 'REAL':'green'})
                    st.plotly_chart(fig, use_container_width=True, key="fn_dist_chart")

# -------------------- FIXED SENTIMENT ANALYSIS SECTION --------------------
elif mode == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    
    # Check if sentiment analyzer is available
    if not hasattr(sentiment_analyzer, 'is_trained') or not sentiment_analyzer.is_trained:
        st.warning("Sentiment analyzer using default model.")
    
    # Show ensemble info in header
    if hasattr(sentiment_analyzer, 'ensemble_names') and sentiment_analyzer.ensemble_names:
        st.info(f"🎯 Using ensemble of {len(sentiment_analyzer.ensemble_names) + 1} models for better accuracy")
    
    # Create tabs for different sentiment analysis modes
    sent_tab1, sent_tab2, sent_tab3, sent_tab4 = st.tabs([
        "📝 Text Analysis", 
        "📁 File Analysis", 
        "🔬 Advanced", 
        "🎯 Aspect Analysis"
    ])
    
    # Tab 1: Text Analysis - FIXED with form
    with sent_tab1:
        st.markdown("### Single Text Analysis")
        
        # Check if we need to clear
        if st.session_state.clear_sentiment:
            st.session_state.clear_sentiment = False
            st.session_state.sentiment_text_input = ""
        
        # Use a form to handle clearing properly
        with st.form(key="sentiment_form"):
            text = st.text_area("Enter text to analyze", height=150, key="sentiment_text_input",
                               placeholder="Type or paste text here...")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                analyze_btn = st.form_submit_button("🔍 Analyze Text", use_container_width=True)
            with col2:
                clear_btn = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
        # Handle clear button
        if clear_btn:
            st.session_state.clear_sentiment = True
            st.rerun()
        
        # Handle analyze button
        if analyze_btn and text.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Get sentiment
                    label, conf = sentiment_analyzer.analyze(text)
                    st.session_state.last_sentiment = (label, conf)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if label == 'POSITIVE':
                            st.success(f"😊 POSITIVE")
                        elif label == 'NEGATIVE':
                            st.error(f"😠 NEGATIVE")
                        else:
                            st.info(f"😐 NEUTRAL")
                        
                        st.metric("Confidence", f"{conf:.2%}")
                        
                        # Show probability bar
                        st.progress(conf, text=f"Confidence: {conf:.2%}")
                    
                    with col2:
                        # Show confidence gauge
                        fig = sentiment_analyzer.create_gauge(conf, label)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show probability distribution
                    try:
                        probs = sentiment_analyzer.predict_proba(text)
                        if len(probs) >= 3:
                            df_probs = pd.DataFrame({
                                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                                'Probability': probs[:3]
                            })
                            fig = px.bar(df_probs, x='Sentiment', y='Probability', 
                                       color='Sentiment',
                                       color_discrete_map={'Negative':'red', 
                                                          'Neutral':'gray', 
                                                          'Positive':'green'},
                                       title="Probability Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                    
                    # Show word count
                    word_count = len(text.split())
                    st.caption(f"📝 Text length: {word_count} words")
                    
                    # Aspect analysis if available
                    if ASPECT_AVAILABLE and text.strip():
                        with st.expander("🔍 Aspect-Based Analysis"):
                            aspects = aspect_analyzer.analyze_aspects(text)
                            for aspect, result in aspects.items():
                                st.write(f"**{aspect.title()}:** {result['label']} ({result['confidence']:.2%})")
                    
                except Exception as e:
                    st.error(f"Analysis error: {e}")
    
    # Tab 2: File Analysis
    with sent_tab2:
        st.markdown("### Batch File Analysis")
        st.info("Upload a CSV or TXT file with texts to analyze")
        
        file = st.file_uploader("Choose a file", type=['txt', 'csv'], key="sentiment_file_upload")
        
        if file is not None:
            # Load and preview data
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                st.write("📊 File Preview:")
                st.dataframe(df.head())
                
                # Select text column
                text_col = st.selectbox("Select text column", df.columns, key="file_text_col")
                texts = df[text_col].tolist()
            else:
                content = file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
                st.write(f"📊 Loaded {len(texts)} texts")
                if texts:
                    st.write("Preview:", texts[:3])
            
            # Analysis options
            col1, col2 = st.columns(2)
            with col1:
                max_texts = st.number_input("Max texts to analyze", 
                                           min_value=1, 
                                           max_value=min(500, len(texts)), 
                                           value=min(100, len(texts)),
                                           key="max_texts_input")
            with col2:
                show_progress = st.checkbox("Show progress", value=True, key="show_progress_checkbox")
            
            if st.button("🚀 Analyze File", key="analyze_file_btn", use_container_width=True):
                results = []
                progress_bar = st.progress(0) if show_progress else None
                
                for i, t in enumerate(texts[:max_texts]):
                    try:
                        label, conf = sentiment_analyzer.analyze(t)
                        results.append({
                            'text': t[:100] + '...' if len(t) > 100 else t,
                            'sentiment': label,
                            'confidence': f"{conf:.2%}",
                            'confidence_score': conf
                        })
                    except:
                        results.append({
                            'text': t[:100] + '...' if len(t) > 100 else t,
                            'sentiment': 'ERROR',
                            'confidence': 'N/A',
                            'confidence_score': 0.0
                        })
                    
                    if show_progress:
                        progress_bar.progress((i + 1) / max_texts)
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    pos_count = len(results_df[results_df['sentiment'] == 'POSITIVE'])
                    st.metric("Positive", pos_count)
                with col2:
                    neu_count = len(results_df[results_df['sentiment'] == 'NEUTRAL'])
                    st.metric("Neutral", neu_count)
                with col3:
                    neg_count = len(results_df[results_df['sentiment'] == 'NEGATIVE'])
                    st.metric("Negative", neg_count)
                
                # Average confidence
                valid_results = results_df[results_df['sentiment'] != 'ERROR']
                if len(valid_results) > 0:
                    avg_conf = valid_results['confidence_score'].mean()
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_file_btn"
                )
                
                # Show distribution
                if len(valid_results) > 0:
                    fig = px.pie(valid_results, names='sentiment', 
                                title='Sentiment Distribution',
                                color='sentiment',
                                color_discrete_map={'POSITIVE':'green', 
                                                   'NEGATIVE':'red', 
                                                   'NEUTRAL':'gray'})
                    st.plotly_chart(fig, use_container_width=True, key="file_distribution_chart")
    
    # Tab 3: Advanced Features
    with sent_tab3:
        st.markdown("### 🔬 Advanced Sentiment Analysis")
        
        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["🤖 Model Info", "📊 Batch Processing", "📈 Visualizations"])
        
        with adv_tab1:
            st.markdown("#### Ensemble Model Details")
            if hasattr(sentiment_analyzer, 'ensemble_names') and sentiment_analyzer.ensemble_names:
                st.write(f"**Using {len(sentiment_analyzer.ensemble_names) + 1} models:**")
                
                # Primary model
                st.success("✅ **Primary:** DistilBERT (base model)")
                
                # Ensemble models
                for i, name in enumerate(sentiment_analyzer.ensemble_names):
                    st.info(f"✅ **Ensemble {i+1}:** {name}")
                
                # VADER fallback
                if hasattr(sentiment_analyzer, 'vader') and sentiment_analyzer.vader:
                    st.info("✅ **Fallback:** VADER (rule-based)")
                
                st.markdown("---")
                st.markdown("""
                **How Ensemble Voting Works:**
                1. Each model analyzes the text independently
                2. Models vote on the sentiment
                3. Majority vote determines final sentiment
                4. Confidence is average of voting models
                
                This approach is **5-10% more accurate** than single models!
                """)
            else:
                st.warning("Ensemble models not loaded. Using single model.")
        
        with adv_tab2:
            st.markdown("#### Batch Processing")
            st.info("Enter multiple texts (one per line) for batch analysis")
            
            batch_text = st.text_area("Enter texts", height=150, key="batch_text_area")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Analyze Batch", key="analyze_batch_btn", use_container_width=True):
                    if batch_text.strip():
                        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                        
                        if len(texts) > 100:
                            st.warning(f"Processing {len(texts)} texts. This may take a moment...")
                        
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            results = sentiment_analyzer.analyze_batch(texts)
                            st.session_state.batch_results = results
                            st.success(f"✅ Analysis complete!")
            
            with col2:
                if st.button("🗑️ Clear Results", key="clear_batch_btn"):
                    st.session_state.batch_results = None
                    st.rerun()
            
            # Display results if available
            if st.session_state.batch_results is not None:
                results = st.session_state.batch_results
                
                batch_tab1, batch_tab2, batch_tab3 = st.tabs(["📋 Results", "📊 Distribution", "📈 Statistics"])
                
                with batch_tab1:
                    st.dataframe(results, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_batch_btn"
                    )
                
                with batch_tab2:
                    fig = sentiment_analyzer.create_distribution(results)
                    st.plotly_chart(fig, use_container_width=True, key="batch_distribution_chart")
                
                with batch_tab3:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pos_pct = len(results[results['sentiment']=='POSITIVE']) / len(results) * 100
                        st.metric("Positive", f"{pos_pct:.1f}%")
                    with col2:
                        neu_pct = len(results[results['sentiment']=='NEUTRAL']) / len(results) * 100
                        st.metric("Neutral", f"{neu_pct:.1f}%")
                    with col3:
                        neg_pct = len(results[results['sentiment']=='NEGATIVE']) / len(results) * 100
                        st.metric("Negative", f"{neg_pct:.1f}%")
                    
                    st.metric("Average Confidence", f"{results['confidence'].mean():.2%}")
        
        with adv_tab3:
            st.markdown("#### Confidence Gauge")
            st.info("Shows confidence for the last analyzed text")
            
            last_label, last_conf = st.session_state.last_sentiment
            fig = sentiment_analyzer.create_gauge(last_conf, last_label)
            st.plotly_chart(fig, use_container_width=True, key="viz_gauge_chart")
            
            if st.button("🎯 Test with Samples", key="test_samples_btn"):
                sample_texts = [
                    "I love this product! It's amazing!",
                    "This is terrible, worst experience ever.",
                    "The movie was okay, nothing special.",
                    "Customer service was excellent!",
                    "The quality is poor and disappointing."
                ]
                
                for i, sample in enumerate(sample_texts):
                    label, conf = sentiment_analyzer.analyze(sample)
                    st.write(f"**Text:** {sample}")
                    
                    if label == 'POSITIVE':
                        st.success(f"Sentiment: {label} ({conf:.2%})")
                    elif label == 'NEGATIVE':
                        st.error(f"Sentiment: {label} ({conf:.2%})")
                    else:
                        st.info(f"Sentiment: {label} ({conf:.2%})")
                    st.markdown("---")
    
    # Tab 4: Aspect Analysis
    with sent_tab4:
        st.markdown("### 🎯 Aspect-Based Sentiment Analysis")
        st.info("Analyze sentiment for specific aspects like product quality, price, service, etc.")
        
        if ASPECT_AVAILABLE:
            aspect_text = st.text_area("Enter text for aspect analysis", 
                                      height=150, 
                                      key="aspect_text_area",
                                      value=st.session_state.aspect_text)
            
            if st.button("🔍 Analyze Aspects", key="analyze_aspects_btn", use_container_width=True):
                if aspect_text.strip():
                    with st.spinner("Analyzing aspects..."):
                        aspects = aspect_analyzer.analyze_aspects(aspect_text)
                        
                        if aspects:
                            st.success("✅ Aspect Analysis Complete!")
                            
                            # Create columns for aspects
                            cols = st.columns(min(3, len(aspects)))
                            for i, (aspect, result) in enumerate(aspects.items()):
                                with cols[i % 3]:
                                    if result['label'] == 'POSITIVE':
                                        st.markdown(f"""
                                        <div style="padding:1rem; background:rgba(0,255,0,0.1); 
                                                  border-radius:10px; margin:0.5rem 0;">
                                            <h4>{aspect.title()}</h4>
                                            <h2 style="color:green;">😊</h2>
                                            <p><strong>{result['label']}</strong></p>
                                            <p>Confidence: {result['confidence']:.2%}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif result['label'] == 'NEGATIVE':
                                        st.markdown(f"""
                                        <div style="padding:1rem; background:rgba(255,0,0,0.1); 
                                                  border-radius:10px; margin:0.5rem 0;">
                                            <h4>{aspect.title()}</h4>
                                            <h2 style="color:red;">😠</h2>
                                            <p><strong>{result['label']}</strong></p>
                                            <p>Confidence: {result['confidence']:.2%}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style="padding:1rem; background:rgba(128,128,128,0.1); 
                                                  border-radius:10px; margin:0.5rem 0;">
                                            <h4>{aspect.title()}</h4>
                                            <h2 style="color:gray;">😐</h2>
                                            <p><strong>{result['label']}</strong></p>
                                            <p>Confidence: {result['confidence']:.2%}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Store in session state
                            st.session_state.aspect_text = aspect_text
                        else:
                            st.warning("No specific aspects detected. Showing general sentiment.")
                            label, conf = sentiment_analyzer.analyze(aspect_text)
                            if label == 'POSITIVE':
                                st.success(f"😊 General Sentiment: {label} ({conf:.2%})")
                            elif label == 'NEGATIVE':
                                st.error(f"😠 General Sentiment: {label} ({conf:.2%})")
                            else:
                                st.info(f"😐 General Sentiment: {label} ({conf:.2%})")
        else:
            st.warning("Aspect analysis module not available. Install required dependencies.")
            
            # Fallback to regular sentiment
            aspect_text = st.text_area("Enter text", height=150, key="aspect_text_fallback")
            if st.button("Analyze", key="analyze_aspects_fallback_btn"):
                if aspect_text.strip():
                    label, conf = sentiment_analyzer.analyze(aspect_text)
                    if label == 'POSITIVE':
                        st.success(f"😊 {label} ({conf:.2%})")
                    elif label == 'NEGATIVE':
                        st.error(f"😠 {label} ({conf:.2%})")
                    else:
                        st.info(f"😐 {label} ({conf:.2%})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; padding:1rem;'>
    <p>🛡️ TruthGuard AI - Advanced Media Authenticity & Analysis</p>
    <p style='font-size:0.8rem;'>Powered by Ensemble Learning, Computer Vision, and NLP</p>
</div>
""", unsafe_allow_html=True)
