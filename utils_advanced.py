import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import hashlib
import json
from datetime import datetime
import os
import psutil
import gc
from cachetools import TTLCache
import base64
from io import BytesIO

class AdvancedUtils:
    def __init__(self):
        self.result_cache = TTLCache(maxsize=100, ttl=3600)
        self.image_cache = TTLCache(maxsize=50, ttl=1800)

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {'rss': mem.rss / 1024 / 1024, 'vms': mem.vms / 1024 / 1024, 'percent': process.memory_percent()}

    def optimize_memory(self):
        if self.get_memory_usage()['rss'] > 3500:
            gc.collect()
            self.image_cache.clear()
            return True
        return False

    @st.cache_data(ttl=3600, max_entries=50)
    def load_image_cached(self, image_file):
        return np.array(Image.open(image_file))

    def preprocess_frame_advanced(self, frame, target_size=(224,224)):
        frame = cv2.resize(frame, target_size)
        if len(frame.shape)==3 and frame.shape[2]==3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        frame_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        frame_denoised = cv2.fastNlMeansDenoisingColored(frame_eq, None, 10,10,7,21)
        return frame_denoised.astype(np.float32)/255.0

    def extract_frames_adaptive(self, video_path, max_frames=15):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total/fps if fps>0 else 0
        if duration<10: num = min(max_frames, total)
        elif duration<30: num = min(12, total)
        else: num = min(8, total)
        indices = np.linspace(0, total-1, num, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                if len(frames)==0 or self.is_keyframe(frame, frames[-1]):
                    frames.append(frame)
        cap.release()
        return frames, duration

    def is_keyframe(self, curr, prev, thresh=30):
        if prev is None: return True
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        return np.mean(cv2.absdiff(curr_gray, prev_gray)) > thresh

    def create_advanced_gauge(self, value, title, subtext=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': title},
            delta={'reference': 50},
            gauge={'axis': {'range': [0,100]}, 'steps': [
                {'range':[0,30],'color':'green'},
                {'range':[30,60],'color':'yellow'},
                {'range':[60,80],'color':'orange'},
                {'range':[80,100],'color':'red'}
            ]}
        ))
        fig.update_layout(height=350)
        return fig

    def create_comparison_chart(self, real, fake, uncertain=0):
        return px.bar(x=['Authentic','Fake','Uncertain'], y=[real*100,fake*100,uncertain*100],
                      color=['green','red','gray'], labels={'x':'','y':'Confidence %'})

    def extract_news_from_url(self, url):
        try:
            from newspaper import Article
            a = Article(url)
            a.download(); a.parse(); a.nlp()
            return {'title':a.title,'text':a.text,'summary':a.summary,'keywords':a.keywords,
                    'publish_date':a.publish_date,'authors':a.authors,'success':True}
        except Exception as e:
            return {'success':False,'error':str(e)}

    def compute_perceptual_hash(self, image):
        small = cv2.resize(image, (8,8))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        return ''.join(['1' if p>mean else '0' for row in gray for p in row])

    def image_to_base64(self, image):
        buffered = BytesIO()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def save_detection_result(self, result_data):
        history_file = 'detection_history.json'
        history = []
        if os.path.exists(history_file):
            with open(history_file) as f:
                history = json.load(f)
        result_data['timestamp'] = datetime.now().isoformat()
        result_data['memory_usage'] = self.get_memory_usage()
        history.append(result_data)
        if len(history) > 100:
            history = history[-100:]
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

utils = AdvancedUtils()