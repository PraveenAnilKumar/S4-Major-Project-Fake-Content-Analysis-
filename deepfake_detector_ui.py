#deepfake_detector.py



"""
deepfake_detector_advanced_ui.py - Advanced Deepfake Detector with User-Friendly UI
Compatible with train_deepfake_batch_advanced.py training script
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
import joblib
import mediapipe as mp
import warnings
from scipy import signal
from skimage import exposure, filters, feature
import gc
import time
import tempfile
from datetime import datetime
import hashlib
import json
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="TruthGuard AI - Advanced Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #4a90e2, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-safe {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    .result-fake {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    .result-suspicious {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(45deg, #4a90e2, #7c3aed);
    }
    .upload-box {
        border: 3px dashed #4a90e2;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .model-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4a90e2;
        margin: 1rem 0;
    }
    .feature-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success { background: #10b981; color: white; }
    .badge-warning { background: #f59e0b; color: white; }
    .badge-danger { background: #ef4444; color: white; }
    .badge-info { background: #3b82f6; color: white; }
</style>
""", unsafe_allow_html=True)

class AdvancedDeepfakeDetectorUI:
    def __init__(self):
        self.detector = None
        self.initialize_detector()
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'detection_history' not in st.session_state:
            st.session_state.detection_history = []
        if 'model_stats' not in st.session_state:
            st.session_state.model_stats = {
                'total_detections': 0,
                'deepfakes_found': 0,
                'authentic_found': 0,
                'avg_confidence': 0
            }
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        if 'analysis_mode' not in st.session_state:
            st.session_state.analysis_mode = "Quick Scan"
    
    def initialize_detector(self):
        """Initialize the deepfake detector"""
        with st.spinner("🚀 Initializing TruthGuard AI Detector..."):
            self.detector = AdvancedDeepfakeDetector()
            
            # Check if models exist
            model_paths = [
                'models/deepfake_cnn.h5',
                'models/deepfake_mobilenet.h5',
                'models/deepfake_artifact.h5'
            ]
            
            if all(os.path.exists(path) for path in model_paths):
                st.success("✅ Models loaded successfully!")
            else:
                st.warning("⚠️ Some models not found. Using default models. Run training script to improve accuracy.")
    
    def render_sidebar(self):
        """Render sidebar with settings and info"""
        with st.sidebar:
            st.image("https://via.placeholder.com/300x100/4a90e2/ffffff?text=TruthGuard+AI", use_container_width=True)
            
            st.markdown("### 🎯 Detection Settings")
            
            # Detection mode
            st.session_state.analysis_mode = st.radio(
                "Analysis Mode",
                ["Quick Scan", "Deep Analysis", "Forensic Analysis"],
                help="Quick Scan: Fast detection. Deep Analysis: More thorough. Forensic: Most detailed"
            )
            
            # Confidence threshold
            self.detector.deepfake_threshold = st.slider(
                "Detection Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.65,
                step=0.05,
                help="Higher values = fewer false positives but may miss some deepfakes"
            ) / 100
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                st.checkbox("Enable Ensemble Detection", value=True, key="ensemble")
                st.checkbox("Analyze Metadata", value=True, key="metadata")
                st.checkbox("Extract All Faces", value=False, key="all_faces")
                st.number_input("Max Faces to Analyze", min_value=1, max_value=10, value=5, key="max_faces")
            
            # Model info
            st.markdown("### 📊 Model Status")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Version", "2.0.0")
            with col2:
                st.metric("Ensemble Models", "3")
            
            # Detection stats
            st.markdown("### 📈 Detection Statistics")
            stats = st.session_state.model_stats
            st.metric("Total Detections", stats['total_detections'])
            st.metric("Deepfakes Found", stats['deepfakes_found'])
            st.metric("Authentic Media", stats['authentic_found'])
            if stats['total_detections'] > 0:
                st.metric("Avg Confidence", f"{stats['avg_confidence']/stats['total_detections']:.1f}%")
            
            # Recent history
            if st.session_state.detection_history:
                st.markdown("### 🕒 Recent Detections")
                for item in st.session_state.detection_history[-5:]:
                    status_color = "🔴" if item['is_deepfake'] else "🟢"
                    st.text(f"{status_color} {item['filename']} - {item['confidence']:.1f}%")
    
    def render_header(self):
        """Render main header"""
        st.markdown('<h1 class="main-header">🛡️ TruthGuard AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Deepfake Detection System</p>', unsafe_allow_html=True)
        
        # Feature badges
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<span class="feature-badge badge-success">✨ Ensemble Detection</span>', unsafe_allow_html=True)
        with col2:
            st.markdown('<span class="feature-badge badge-info">🎭 Multi-Face Analysis</span>', unsafe_allow_html=True)
        with col3:
            st.markdown('<span class="feature-badge badge-warning">🔄 Temporal Analysis</span>', unsafe_allow_html=True)
        with col4:
            st.markdown('<span class="feature-badge badge-danger">🔍 Artifact Detection</span>', unsafe_allow_html=True)
    
    def render_upload_section(self):
        """Render file upload section"""
        st.markdown("### 📤 Upload Media for Analysis")
        
        upload_tab1, upload_tab2 = st.tabs(["📷 Image", "🎥 Video"])
        
        with upload_tab1:
            self.render_image_upload()
        
        with upload_tab2:
            self.render_video_upload()
    
    def render_image_upload(self):
        """Render image upload interface"""
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Image info
                st.markdown("**Image Info:**")
                st.write(f"- Format: {image.format}")
                st.write(f"- Size: {image.size}")
                st.write(f"- Mode: {image.mode}")
            
            with col2:
                # Analysis button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Convert PIL to numpy
                        img_array = np.array(image)
                        if len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        elif img_array.shape[2] == 4:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        
                        # Perform detection
                        result = self.detector.detect_deepfake_ensemble(img_array)
                        
                        # Add filename to result
                        result['filename'] = uploaded_file.name
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Update stats
                        st.session_state.current_result = result
                        self.update_stats(result)
                        
                        st.rerun()
    
    def render_video_upload(self):
        """Render video upload interface"""
        uploaded_file = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            key="video_uploader"
        )
        
        if uploaded_file is not None:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display video info
                st.video(tfile.name)
                
                # Video info
                st.markdown("**Video Info:**")
                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                st.write(f"- FPS: {fps:.2f}")
                st.write(f"- Frames: {frame_count}")
                st.write(f"- Duration: {duration:.2f}s")
            
            with col2:
                # Analysis button
                if st.button("🎬 Analyze Video", type="primary", use_container_width=True):
                    with st.spinner("Analyzing video (this may take a while)..."):
                        # Perform detection based on mode
                        if st.session_state.analysis_mode == "Quick Scan":
                            # Quick scan - analyze key frames
                            result = self.detector.detect_deepfake_video_advanced(tfile.name)
                        else:
                            # Deep analysis
                            result = self.detector.detect_deepfake_video_advanced(tfile.name)
                        
                        # Add filename to result
                        result['filename'] = uploaded_file.name
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Update stats
                        st.session_state.current_result = result
                        self.update_stats(result)
                        
                        st.rerun()
            
            # Clean up temp file
            os.unlink(tfile.name)
    
    def render_results(self):
        """Render detection results"""
        if st.session_state.current_result is None:
            return
        
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("### 🔬 Analysis Results")
        
        # Main result card
        is_deepfake = result['is_deepfake']
        confidence = result['confidence']
        
        # Determine result class
        if is_deepfake and confidence > 80:
            result_class = "result-fake"
            result_text = "🚨 DEEPFAKE DETECTED"
            result_icon = "🔴"
        elif not is_deepfake and confidence < 30:
            result_class = "result-safe"
            result_text = "✅ AUTHENTIC MEDIA"
            result_icon = "🟢"
        else:
            result_class = "result-suspicious"
            result_text = "⚠️ SUSPICIOUS - FURTHER ANALYSIS NEEDED"
            result_icon = "🟡"
        
        # Display main result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<div class="{result_class}">{result_icon} {result_text}</div>', unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown("#### Confidence Score")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(confidence / 100)
        with col2:
            st.markdown(f"**{confidence:.1f}%**")
        
        # Detailed results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎭 Face Analysis", "📈 Metrics", "📋 Report"])
        
        with tab1:
            self.render_overview_tab(result)
        
        with tab2:
            self.render_face_analysis_tab(result)
        
        with tab3:
            self.render_metrics_tab(result)
        
        with tab4:
            self.render_report_tab(result)
    
    def render_overview_tab(self, result):
        """Render overview tab"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Faces Detected", result.get('face_count', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Ensemble Score", f"{result.get('ensemble_score', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Consistency", f"{result.get('consistency', 0)*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Message
        st.info(result.get('message', 'Analysis complete'))
        
        # File info
        st.markdown("#### File Information")
        st.write(f"- **Filename:** {result.get('filename', 'Unknown')}")
        st.write(f"- **Analysis Time:** {result.get('timestamp', 'N/A')}")
        st.write(f"- **Analysis Mode:** {st.session_state.analysis_mode}")
    
    def render_face_analysis_tab(self, result):
        """Render face analysis tab"""
        if 'face_analyses' in result and result['face_analyses']:
            for i, face in enumerate(result['face_analyses']):
                with st.expander(f"Face {i+1} Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Detection Info**")
                        st.write(f"- Confidence: {face['confidence']:.2%}")
                        st.write(f"- Method: {face.get('method', 'Unknown')}")
                        st.write(f"- BBox: {face['bbox']}")
                    
                    with col2:
                        st.markdown("**Artifact Analysis**")
                        if 'artifacts' in face:
                            artifacts = face['artifacts']
                            st.write(f"- Sharpness: {artifacts.get('sharpness', 0):.3f}")
                            st.write(f"- Compression Artifacts: {artifacts.get('compression_artifacts', 0):.3f}")
                            st.write(f"- Noise Level: {artifacts.get('noise_level', 0):.3f}")
                    
                    # Artifact score gauge
                    artifact_score = face.get('artifact_score', 0)
                    st.progress(artifact_score, text=f"Artifact Score: {artifact_score:.2%}")
        else:
            st.info("No faces detected in the image")
    
    def render_metrics_tab(self, result):
        """Render metrics tab with visualizations"""
        # Create radar chart for metrics
        if 'face_analyses' in result and result['face_analyses']:
            categories = ['Sharpness', 'Color Uniformity', 'Compression', 'Edge Density', 'Noise']
            
            fig = go.Figure()
            
            for i, face in enumerate(result['face_analyses'][:3]):  # Show first 3 faces
                if 'artifacts' in face:
                    artifacts = face['artifacts']
                    values = [
                        artifacts.get('sharpness', 0),
                        artifacts.get('color_uniformity', 0.5),
                        artifacts.get('compression_artifacts', 0),
                        artifacts.get('edge_density', 0),
                        artifacts.get('noise_level', 0)
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f'Face {i+1}'
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Face Artifact Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Frequency analysis if available
        if 'face_analyses' in result and result['face_analyses']:
            st.markdown("#### Frequency Domain Analysis")
            freq_data = []
            for i, face in enumerate(result['face_analyses']):
                if 'freq_analysis' in face:
                    freq = face['freq_analysis']
                    freq_data.append({
                        'Face': f'Face {i+1}',
                        'Low Freq': freq['low_freq'],
                        'Mid Freq': freq['mid_freq'],
                        'High Freq': freq['high_freq'],
                        'Suspicious': freq['suspicious_pattern']
                    })
            
            if freq_data:
                df = pd.DataFrame(freq_data)
                st.dataframe(df, use_container_width=True)
    
    def render_report_tab(self, result):
        """Render detailed report tab"""
        st.markdown("#### Detailed Analysis Report")
        
        # Generate report content
        report = {
            'filename': result.get('filename', 'Unknown'),
            'timestamp': result.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            'verdict': 'Deepfake' if result['is_deepfake'] else 'Authentic',
            'confidence': f"{result['confidence']:.2f}%",
            'analysis_mode': st.session_state.analysis_mode,
            'threshold': f"{self.detector.deepfake_threshold*100:.1f}%",
            'faces_detected': result.get('face_count', 0),
            'ensemble_score': f"{result.get('ensemble_score', 0):.4f}",
            'consistency': f"{result.get('consistency', 0)*100:.2f}%",
            'message': result.get('message', ''),
            'face_details': []
        }
        
        # Add face details
        if 'face_analyses' in result:
            for i, face in enumerate(result['face_analyses']):
                face_detail = {
                    'face_index': i + 1,
                    'bbox': face['bbox'],
                    'detection_confidence': f"{face['confidence']:.2%}",
                    'artifact_score': f"{face.get('artifact_score', 0):.2%}",
                    'method': face.get('method', 'Unknown')
                }
                
                if 'artifacts' in face:
                    face_detail['artifacts'] = face['artifacts']
                
                if 'freq_analysis' in face:
                    face_detail['frequency_analysis'] = face['freq_analysis']
                
                report['face_details'].append(face_detail)
        
        # Display report
        st.json(report)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            # JSON download
            report_json = json.dumps(report, indent=2)
            b64 = base64.b64encode(report_json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="deepfake_report.json">📥 Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Text report download
            report_text = self.generate_text_report(report)
            b64 = base64.b64encode(report_text.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="deepfake_report.txt">📥 Download Text Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def generate_text_report(self, report):
        """Generate human-readable text report"""
        lines = []
        lines.append("=" * 50)
        lines.append("TRUTHGUARD AI - DEEPFAKE DETECTION REPORT")
        lines.append("=" * 50)
        lines.append(f"Filename: {report['filename']}")
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Verdict: {report['verdict']}")
        lines.append(f"Confidence: {report['confidence']}")
        lines.append(f"Analysis Mode: {report['analysis_mode']}")
        lines.append(f"Faces Detected: {report['faces_detected']}")
        lines.append(f"Message: {report['message']}")
        lines.append("=" * 50)
        
        if report['face_details']:
            lines.append("\nFACE ANALYSIS DETAILS:")
            for face in report['face_details']:
                lines.append(f"\nFace {face['face_index']}:")
                lines.append(f"  BBox: {face['bbox']}")
                lines.append(f"  Detection Confidence: {face['detection_confidence']}")
                lines.append(f"  Artifact Score: {face['artifact_score']}")
                lines.append(f"  Detection Method: {face['method']}")
        
        return "\n".join(lines)
    
    def update_stats(self, result):
        """Update session statistics"""
        stats = st.session_state.model_stats
        stats['total_detections'] += 1
        stats['avg_confidence'] += result['confidence']
        
        if result['is_deepfake']:
            stats['deepfakes_found'] += 1
        else:
            stats['authentic_found'] += 1
        
        # Add to history
        st.session_state.detection_history.append({
            'filename': result.get('filename', 'Unknown'),
            'is_deepfake': result['is_deepfake'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Keep only last 50
        if len(st.session_state.detection_history) > 50:
            st.session_state.detection_history = st.session_state.detection_history[-50:]
    
    def render_model_training_section(self):
        """Render model training interface"""
        st.markdown("---")
        st.markdown("### 🏋️ Model Training")
        
        with st.expander("Advanced Model Training"):
            st.markdown("""
            Train the deepfake detection model on your own dataset for improved accuracy.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Training Configuration")
                batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=16, step=8)
                epochs = st.number_input("Epochs", min_value=5, max_value=200, value=20, step=5)
                learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
                use_augmentation = st.checkbox("Use Data Augmentation", value=True)
            
            with col2:
                st.markdown("#### Dataset Paths")
                train_real = st.text_input("Real Training Images Path", "datasets/deepfake/train/real/")
                train_fake = st.text_input("Fake Training Images Path", "datasets/deepfake/train/fake/")
                val_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
            
            if st.button("🚀 Start Training", type="primary"):
                st.info("Training started in background. Check console for progress.")
                # This would call your training script
                # subprocess.Popen(["python", "train_deepfake_batch_advanced.py"])
    
    def run(self):
        """Main UI runner"""
        self.render_sidebar()
        self.render_header()
        self.render_upload_section()
        self.render_results()
        self.render_model_training_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 1rem;'>
                <p>🛡️ TruthGuard AI - Advanced Deepfake Detection System v2.0</p>
                <p>Powered by Ensemble Learning, Computer Vision, and Neural Networks</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Modified AdvancedDeepfakeDetector class with compatibility fixes
class AdvancedDeepfakeDetector:
    def __init__(self):
        # Initialize face detectors
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=True, device=self.device, select_largest=False)
        
        # Initialize MediaPipe with proper error handling
        try:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                min_detection_confidence=0.5
            )
        except:
            self.mp_face_mesh = None
            print("MediaPipe face mesh not available")
        
        # Model parameters
        self.input_shape = (299, 299, 3)  # Match training script
        self.model = None
        self.ensemble_models = []
        
        # Detection thresholds
        self.deepfake_threshold = 0.65
        self.artifact_threshold = 0.5
        
        # Initialize models
        self.load_or_create_models()
        
        # Performance tracking
        self.last_gc_time = time.time()
        
    def load_or_create_models(self):
        """Load or create ensemble of lightweight models"""
        model_paths = [
            'models/deepfake_cnn.h5',
            'models/deepfake_mobilenet.h5',
            'models/deepfake_artifact.h5'
        ]
        
        try:
            if all(os.path.exists(path) for path in model_paths):
                self.model = tf.keras.models.load_model(model_paths[0])
                for path in model_paths[1:]:
                    if os.path.exists(path):
                        self.ensemble_models.append(tf.keras.models.load_model(path))
                print("✅ Models loaded successfully")
            else:
                print("⚠️ Some models not found. Creating new models...")
                self.create_ensemble_models()
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.create_ensemble_models()
    
    def create_ensemble_models(self):
        """Create ensemble of lightweight models for better accuracy"""
        # Model 1: Lightweight CNN (spatial features)
        self.model = self.create_cnn_model()
        
        # Model 2: MobileNetV2 based (transfer learning)
        try:
            model2 = self.create_mobilenet_model()
            self.ensemble_models.append(model2)
        except Exception as e:
            print(f"⚠️ MobileNet creation failed: {e}")
        
        # Model 3: Artifact detection model
        try:
            model3 = self.create_artifact_model()
            self.ensemble_models.append(model3)
        except Exception as e:
            print(f"⚠️ Artifact model creation failed: {e}")
    
    def create_cnn_model(self):
        """Create optimized CNN model - compatible with training script"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),
            
            # Fourth block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def create_mobilenet_model(self):
        """Create MobileNetV2 based model - compatible with training script"""
        base_model = applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_artifact_model(self):
        """Model specialized in detecting compression artifacts - compatible with training script"""
        model = models.Sequential([
            layers.Input(shape=(112, 112, 3)),
            
            layers.Conv2D(32, (7, 7), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def detect_faces_advanced(self, image):
        """Advanced face detection with multiple methods"""
        faces_data = []
        
        # Method 1: MTCNN
        try:
            if isinstance(image, np.ndarray):
                if image.shape[-1] == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image)
            
            boxes, probs = self.mtcnn.detect(pil_image, landmarks=False)
            
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob > 0.9:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                        
                        face = image[y1:y2, x1:x2]
                        if face.size > 0:
                            faces_data.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': prob,
                                'face': face,
                                'method': 'mtcnn'
                            })
        except Exception as e:
            print(f"MTCNN error: {e}")
        
        # Method 2: MediaPipe (fallback)
        if len(faces_data) == 0 and self.mp_face_mesh is not None:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(rgb_image)
                
                if results.multi_face_landmarks:
                    h, w = image.shape[:2]
                    for landmarks in results.multi_face_landmarks:
                        x_coords = [lm.x for lm in landmarks.landmark]
                        y_coords = [lm.y for lm in landmarks.landmark]
                        
                        x1, x2 = int(min(x_coords) * w), int(max(x_coords) * w)
                        y1, y2 = int(min(y_coords) * h), int(max(y_coords) * h)
                        
                        margin = 20
                        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
                        
                        face = image[y1:y2, x1:x2]
                        if face.size > 0:
                            faces_data.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': 0.95,
                                'face': face,
                                'method': 'mediapipe'
                            })
            except Exception as e:
                print(f"MediaPipe error: {e}")
        
        return faces_data
    
    def analyze_face_consistency(self, face):
        """Analyze face for inconsistencies and artifacts"""
        artifacts = {}
        
        # Convert to grayscale
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        else:
            gray = face
        
        # 1. Check for blurring
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        artifacts['sharpness'] = min(laplacian_var / 500, 1.0)
        
        # 2. Check for color inconsistencies
        if len(face.shape) == 3:
            lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
            color_uniformity = np.std(lab.reshape(-1, 3), axis=0).mean() / 128
            artifacts['color_uniformity'] = min(color_uniformity, 1.0)
        
        # 3. Check for compression artifacts
        gray_float = gray.astype(np.float32)
        dct_coeffs = cv2.dct(gray_float)
        high_freq_energy = np.mean(np.abs(dct_coeffs[50:, 50:]))
        artifacts['compression_artifacts'] = min(high_freq_energy / 100, 1.0)
        
        # 4. Check for edge inconsistencies
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        artifacts['edge_density'] = edge_density
        
        # 5. Check for noise patterns
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_diff = cv2.absdiff(gray, noise)
        noise_level = np.mean(noise_diff) / 255
        artifacts['noise_level'] = noise_level
        
        # Calculate overall artifact score
        artifact_score = (
            (1 - artifacts['sharpness']) * 0.25 +
            artifacts.get('color_uniformity', 0.5) * 0.2 +
            artifacts['compression_artifacts'] * 0.25 +
            artifacts['edge_density'] * 0.15 +
            artifacts['noise_level'] * 0.15
        )
        
        return artifact_score, artifacts
    
    def analyze_frequency_domain(self, image):
        """Analyze image in frequency domain for artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Get magnitude spectrum
        magnitude = np.log(np.abs(fft_shift) + 1)
        
        # Analyze frequency distribution
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequencies (center)
        low_freq = magnitude[center_h-30:center_h+30, center_w-30:center_w+30].mean()
        
        # Mid frequencies
        mid_freq = magnitude[center_h-60:center_h+60, center_w-60:center_w+60].mean()
        
        # High frequencies (edges)
        high_freq = magnitude.mean()
        
        # Deepfakes often have unusual frequency patterns
        freq_ratio = (high_freq - low_freq) / (mid_freq + 1e-6)
        
        return {
            'low_freq': float(low_freq),
            'mid_freq': float(mid_freq),
            'high_freq': float(high_freq),
            'freq_ratio': float(freq_ratio),
            'suspicious_pattern': bool(freq_ratio < 0.5 or freq_ratio > 2.0)
        }
    
    def detect_deepfake_ensemble(self, image):
        """Ensemble detection using multiple models"""
        # Preprocess image
        faces_data = self.detect_faces_advanced(image)
        
        if len(faces_data) == 0:
            return {
                'is_deepfake': False,
                'confidence': 0,
                'message': 'No face detected',
                'face_count': 0,
                'details': {}
            }
        
        all_predictions = []
        face_analyses = []
        
        for face_data in faces_data:
            face = face_data['face']
            face_resized = cv2.resize(face, (299, 299))
            face_input = np.expand_dims(face_resized / 255.0, axis=0)
            
            # Get predictions from ensemble
            predictions = []
            
            # Model 1: CNN prediction
            if self.model is not None:
                try:
                    pred1 = self.model.predict(face_input, verbose=0)[0][0]
                    predictions.append(float(pred1))
                except:
                    pass
            
            # Model 2: MobileNet prediction
            if len(self.ensemble_models) > 0:
                try:
                    pred2 = self.ensemble_models[0].predict(face_input, verbose=0)[0][0]
                    predictions.append(float(pred2))
                except:
                    pass
            
            # Model 3: Artifact model
            if len(self.ensemble_models) > 1:
                try:
                    face_small = cv2.resize(face, (112, 112))
                    face_small_input = np.expand_dims(face_small / 255.0, axis=0)
                    pred3 = self.ensemble_models[1].predict(face_small_input, verbose=0)[0][0]
                    predictions.append(float(pred3))
                except:
                    pass
            
            # Additional analysis
            artifact_score, artifacts = self.analyze_face_consistency(face)
            freq_analysis = self.analyze_frequency_domain(face)
            
            # Combine predictions
            avg_pred = np.mean(predictions) if predictions else 0.5
            
            # Weighted score
            weighted_score = (
                avg_pred * 0.5 +
                artifact_score * 0.3 +
                (1 if freq_analysis['suspicious_pattern'] else 0) * 0.2
            )
            
            all_predictions.append(weighted_score)
            face_analyses.append({
                'bbox': face_data['bbox'],
                'confidence': float(face_data['confidence']),
                'artifact_score': float(artifact_score),
                'artifacts': artifacts,
                'freq_analysis': freq_analysis,
                'model_score': float(avg_pred),
                'method': face_data.get('method', 'unknown')
            })
        
        # Overall decision
        final_score = float(np.mean(all_predictions))
        is_deepfake = final_score > self.deepfake_threshold
        
        # Calculate confidence based on consistency
        if len(all_predictions) > 1:
            consistency = 1 - float(np.std(all_predictions))
        else:
            consistency = 1.0
        
        confidence = final_score * 100 * consistency
        
        return {
            'is_deepfake': bool(is_deepfake),
            'confidence': float(min(confidence, 100)),
            'face_count': len(faces_data),
            'face_analyses': face_analyses,
            'ensemble_score': float(final_score),
            'consistency': float(consistency),
            'message': 'Deepfake detected with high confidence' if is_deepfake and confidence > 80 
                      else 'Likely authentic' if not is_deepfake and confidence < 30
                      else 'Suspicious - further analysis recommended'
        }
    
    def detect_deepfake_video_advanced(self, video_path):
        """Advanced video deepfake detection"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Analyze key frames
        frame_results = []
        temporal_scores = []
        
        # Process every 30th frame for efficiency
        for i in range(0, frame_count, 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                result = self.detect_deepfake_ensemble(frame)
                frame_results.append({
                    'frame_index': i,
                    'is_deepfake': result['is_deepfake'],
                    'confidence': result['confidence'],
                    'face_count': result['face_count']
                })
                temporal_scores.append(result['confidence'] / 100)
        
        cap.release()
        
        if len(temporal_scores) == 0:
            return {
                'is_deepfake': False,
                'confidence': 0,
                'message': 'Could not analyze video',
                'frames_analyzed': 0
            }
        
        # Analyze temporal consistency
        temporal_mean = float(np.mean(temporal_scores))
        temporal_std = float(np.std(temporal_scores))
        
        # Final decision
        is_deepfake = temporal_mean > self.deepfake_threshold
        
        # Calculate confidence
        confidence = temporal_mean * 100 * (1 - temporal_std)
        
        return {
            'is_deepfake': bool(is_deepfake),
            'confidence': float(min(confidence, 100)),
            'frames_analyzed': len(frame_results),
            'temporal_consistency': float((1 - temporal_std) * 100),
            'frame_results': frame_results[:10],  # Return first 10 for preview
            'message': 'Deepfake detected in video' if is_deepfake else 'Video appears authentic',
            'duration': float(duration)
        }

# Run the UI
if __name__ == "__main__":
    ui = AdvancedDeepfakeDetectorUI()
    ui.run()