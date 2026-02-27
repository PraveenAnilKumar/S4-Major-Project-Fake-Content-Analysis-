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

from deepfake_detector_advanced import deepfake_detector
from fake_news_detector import fake_news_detector
from sentiment_analyzer import sentiment_analyzer
from utils_advanced import utils

# ---------- Load pre‑trained models ----------
if not fake_news_detector.is_trained:
    fake_news_detector.load_models('models/fake_news/')

# ---------- Page config ----------
st.set_page_config(page_title="TruthGuard AI", page_icon="🛡️", layout="wide")

# ---------- Hardcoded credentials (for demo only) ----------
ADMIN_CREDENTIALS = {"admin": "admin123"}
USER_CREDENTIALS = {"user": "user123"}

# ---------- Initialize session state ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

# ---------- Custom CSS (enhanced) ----------
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
    .sidebar-user-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #4a90e2;
    }
    .sidebar-user-card h4 {
        margin: 0;
        color: #1e293b;
        font-size: 1.1rem;
    }
    .sidebar-user-card p {
        margin: 0.2rem 0 0;
        color: #64748b;
        font-size: 0.9rem;
    }
    .sidebar .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: white;
        color: #1e293b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    .sidebar .stButton>button:hover {
        background: #f1f5f9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Login / Logout ----------
def login():
    st.sidebar.markdown("## 🔐 Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login", key="login_button"):
        if not username or not password:
            st.sidebar.error("Please enter both username and password.")
        elif username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = "admin"
            st.rerun()
        elif username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = "user"
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()

# ---------- Show login if not authenticated ----------
if not st.session_state.authenticated:
    st.markdown('<h1 class="main-header">🛡️ TruthGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Please log in to continue</p>', unsafe_allow_html=True)
    login()
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
        logout()
    
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    mode = st.radio(
        "Select Feature",
        ["📸 Deepfake Detection", "📰 Fake News Detection", "😊 Sentiment Analysis"],
        label_visibility="collapsed"
    )
    mode_map = {
        "📸 Deepfake Detection": "Deepfake Detection",
        "📰 Fake News Detection": "Fake News Detection",
        "😊 Sentiment Analysis": "Sentiment Analysis"
    }
    mode = mode_map[mode]
    
    st.markdown("---")
    
    if st.session_state.role == "admin":
        with st.expander("🔧 Admin Tools", expanded=False):
            st.markdown("#### Train Models")
            if st.button("🤖 Train Deepfake Model", use_container_width=True):
                st.info("Deepfake training started in background. Check console for progress.")
                subprocess.Popen(["python", "train_deepfake_batch_advanced.py"])
            
            if st.button("📊 Train Fake News (Traditional)", use_container_width=True):
                st.info("Fake news training started (traditional ML).")
                subprocess.Popen(["python", "train_fakenews.py", "--dataset", "datasets/fake_news/train.csv"])
            
            if st.button("🧠 Train Fake News (Transformer)", use_container_width=True):
                st.info("Fake news training started (transformer). This may take a while.")
                subprocess.Popen(["python", "train_fakenews.py", "--dataset", "datasets/fake_news/train.csv", "--transformer"])
            
            if st.button("😶 Train Sentiment Model", use_container_width=True):
                st.info("Sentiment fine-tuning started.")
                subprocess.Popen(["python", "train_sentiment.py", "--dataset", "datasets/sentiment/train.csv"])
            
            st.markdown("#### Upload Datasets")
            st.caption("Place files in the appropriate folders:")
            st.code("""
datasets/deepfake/train/real/
datasets/deepfake/train/fake/
datasets/deepfake/test/real/
datasets/deepfake/test/fake/
datasets/fake_news/train.csv
datasets/sentiment/train.csv
            """)
            uploaded_file = st.file_uploader("Upload dataset file", type=['csv','zip','jpg','png'])
            if uploaded_file is not None:
                os.makedirs("temp", exist_ok=True)
                with open(os.path.join("temp", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved to temp/{uploaded_file.name} (admin can move to correct folder)")

# ---------- Header ----------
st.markdown('<h1 class="main-header">🛡️ TruthGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Media Authenticity & Analysis</p>', unsafe_allow_html=True)

# -------------------- Deepfake --------------------
if mode == "Deepfake Detection":
    st.header("📸 Deepfake Detection")
    col1, col2 = st.columns([1,1])
    with col1:
        thresh = st.slider("Threshold", 0.1, 0.9, 0.65, 0.05)
        deepfake_detector.deepfake_threshold = thresh
    with col2:
        st.metric("Ensemble Models", len(deepfake_detector.ensemble_models)+1)

    upload_tab1, upload_tab2 = st.tabs(["📷 Image", "🎥 Video"])
    with upload_tab1:
        img_file = st.file_uploader("Upload image", type=['jpg','jpeg','png','bmp'])
        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded", use_container_width=True)
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    arr = np.array(img)
                    if len(arr.shape)==2:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                    elif arr.shape[2]==4:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                    res = deepfake_detector.detect_deepfake_ensemble(arr)
                    res['filename'] = img_file.name
                    st.session_state['df_result'] = res
    with upload_tab2:
        vid_file = st.file_uploader("Upload video", type=['mp4','avi','mov'])
        if vid_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(vid_file.read())
            tfile.close()
            st.video(tfile.name)
            if st.button("Analyze Video"):
                with st.spinner("Analyzing video..."):
                    res = deepfake_detector.detect_deepfake_video_advanced(tfile.name)
                    res['filename'] = vid_file.name
                    st.session_state['df_result'] = res
            os.unlink(tfile.name)

    if 'df_result' in st.session_state:
        r = st.session_state.df_result
        st.markdown("---")
        st.subheader("Result")
        is_df = r['is_deepfake']
        conf = r['confidence']
        if is_df and conf>80:
            st.markdown(f'<div class="result-fake">🚨 DEEPFAKE DETECTED ({conf:.1f}%)</div>', unsafe_allow_html=True)
        elif not is_df and conf<30:
            st.markdown(f'<div class="result-safe">✅ AUTHENTIC ({(1-conf):.1f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-suspicious">⚠️ SUSPICIOUS ({conf:.1f}%)</div>', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        colA.metric("Faces", r.get('face_count',0))
        colB.metric("Ensemble Score", f"{r.get('ensemble_score',0):.3f}")
        colC.metric("Consistency", f"{r.get('consistency',0)*100:.1f}%")
        st.info(r.get('message',''))

# -------------------- Fake News --------------------
elif mode == "Fake News Detection":
    st.header("📰 Fake News Detection")
    
    # Show model status
    if not fake_news_detector.is_trained:
        st.warning("⚠️ No trained model found. Please train the model first (admin only).")
        st.stop()
    else:
        model_type = "transformer" if fake_news_detector.use_transformer else "traditional ML"
        st.success(f"✅ Model loaded ({model_type})")
    
    method = st.radio("Input", ["Text", "File"])
    
    if method == "Text":
        text = st.text_area("Enter news text", height=150)
        if st.button("Analyze", key="analyze_fake"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        label, conf = fake_news_detector.predict(text)
                        prob = fake_news_detector.predict_proba(text)
                        if label == 1:
                            st.error(f"🚨 FAKE NEWS ({conf*100:.1f}%)")
                        else:
                            st.success(f"✅ REAL NEWS ({(1-conf)*100:.1f}%)")
                        st.progress(prob, text=f"Fake probability: {prob*100:.1f}%")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        print("Error in prediction:", e)
    else:  # File upload
        file = st.file_uploader("Upload file", type=['txt','csv'])
        if file:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                col = st.selectbox("Select text column", df.columns)
                texts = df[col].tolist()
            else:
                content = file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            st.write(f"Loaded {len(texts)} entries.")
            if st.button("Analyze File", key="analyze_file_fake"):
                results = []
                progress_bar = st.progress(0)
                for i, t in enumerate(texts):
                    label, conf = fake_news_detector.predict(t)
                    results.append({'text': t[:50], 'label': 'FAKE' if label else 'REAL', 'confidence': f"{conf*100:.1f}%"})
                    progress_bar.progress((i+1)/len(texts))
                st.dataframe(pd.DataFrame(results))

# -------------------- Sentiment --------------------
elif mode == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    if not sentiment_analyzer.is_trained:
        st.warning("Sentiment analyzer not loaded. Install transformers/vader.")
    else:
        method = st.radio("Input", ["Text", "File"])
        if method == "Text":
            text = st.text_area("Enter text", height=150)
            if st.button("Analyze"):
                label, conf = sentiment_analyzer.analyze(text)
                probs = sentiment_analyzer.predict_proba(text)
                if label == 'POSITIVE':
                    st.success(f"😊 POSITIVE ({conf:.2%})")
                elif label == 'NEGATIVE':
                    st.error(f"😠 NEGATIVE ({conf:.2%})")
                else:
                    st.info(f"😐 NEUTRAL")
                if len(probs)==3:
                    df = pd.DataFrame({'Sentiment':['Negative','Neutral','Positive'], 'Prob':probs})
                    fig = px.bar(df, x='Sentiment', y='Prob', color='Sentiment',
                                 color_discrete_map={'Negative':'red','Neutral':'gray','Positive':'green'})
                    st.plotly_chart(fig)
        else:
            file = st.file_uploader("Upload file", type=['txt','csv'])
            if file:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    col = st.selectbox("Select text column", df.columns)
                    texts = df[col].tolist()
                else:
                    content = file.read().decode('utf-8')
                    texts = [line.strip() for line in content.split('\n') if line.strip()]
                if st.button("Analyze File"):
                    results = []
                    for t in texts:
                        label, conf = sentiment_analyzer.analyze(t)
                        results.append({'text': t[:50], 'sentiment': label, 'confidence': f"{conf:.2%}"})
                    st.dataframe(pd.DataFrame(results))

st.markdown("---")
st.markdown("<div style='text-align:center; color:#666'>TruthGuard AI - Unified Media Analysis</div>", unsafe_allow_html=True)