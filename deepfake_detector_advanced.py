"""
deepfake_detector_advanced.py - Advanced Deepfake Detector with compatibility fixes
Compatible with train_deepfake_batch_advanced.py (224x224 input)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import os
import joblib
import mediapipe as mp
import warnings
import gc
import time

warnings.filterwarnings('ignore')

class AdvancedDeepfakeDetector:
    def __init__(self, input_size=224):  # match training script
        self.input_size = input_size
        self.input_shape = (input_size, input_size, 3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=True, device=self.device, select_largest=False)

        try:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                min_detection_confidence=0.5
            )
        except:
            self.mp_face_mesh = None
            print("MediaPipe face mesh not available")

        self.model = None
        self.ensemble_models = []
        self.deepfake_threshold = 0.65
        self.artifact_threshold = 0.5

        self.load_or_create_models()
        self.last_gc_time = time.time()

    # ------------------------------------------------------------------
    # Model loading / creation (FIXED)
    # ------------------------------------------------------------------
    def load_or_create_models(self):
        """Load any existing models; create only those that are missing."""
        model_paths = {
            'cnn': 'models/deepfake_cnn.h5',
            'mobilenet': 'models/deepfake_mobilenet.h5',
            'artifact': 'models/deepfake_artifact.h5'
        }

        # Load or create CNN model (primary)
        if os.path.exists(model_paths['cnn']):
            try:
                self.model = tf.keras.models.load_model(model_paths['cnn'])
                print("✅ Loaded CNN model from", model_paths['cnn'])
            except Exception as e:
                print(f"⚠️ Failed to load CNN model: {e}. Creating new one.")
                self.model = self.create_cnn_model()
        else:
            print("⚠️ CNN model not found. Creating new one.")
            self.model = self.create_cnn_model()

        # Load or create MobileNet model (ensemble)
        if os.path.exists(model_paths['mobilenet']):
            try:
                self.ensemble_models.append(tf.keras.models.load_model(model_paths['mobilenet']))
                print("✅ Loaded MobileNet model")
            except Exception as e:
                print(f"⚠️ Failed to load MobileNet model: {e}. Creating new one.")
                self.ensemble_models.append(self.create_mobilenet_model())
        else:
            print("⚠️ MobileNet model not found. Creating new one.")
            self.ensemble_models.append(self.create_mobilenet_model())

        # Load or create artifact model (ensemble)
        if os.path.exists(model_paths['artifact']):
            try:
                self.ensemble_models.append(tf.keras.models.load_model(model_paths['artifact']))
                print("✅ Loaded artifact model")
            except Exception as e:
                print(f"⚠️ Failed to load artifact model: {e}. Creating new one.")
                self.ensemble_models.append(self.create_artifact_model())
        else:
            print("⚠️ Artifact model not found. Creating new one.")
            self.ensemble_models.append(self.create_artifact_model())

    def create_cnn_model(self):
        """Create optimized CNN model - compatible with training script."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),

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
        """Create MobileNetV2 based model."""
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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_artifact_model(self):
        """Model specialized in detecting compression artifacts."""
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

    # ------------------------------------------------------------------
    # Face detection and analysis
    # ------------------------------------------------------------------
    def detect_faces_advanced(self, image):
        faces_data = []
        # MTCNN
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

        # MediaPipe fallback
        if len(faces_data) == 0 and self.mp_face_mesh is not None:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    h, w = image.shape[:2]
                    for landmarks in results.multi_face_landmarks:
                        x_coords = [lm.x for lm in landmarks.landmark]
                        y_coords = [lm.y for lm in landmarks.landmark]
                        x1 = int(min(x_coords) * w) - 20
                        y1 = int(min(y_coords) * h) - 20
                        x2 = int(max(x_coords) * w) + 20
                        y2 = int(max(y_coords) * h) + 20
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
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
        artifacts = {}
        if len(face.shape) == 3:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        else:
            gray = face

        # Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        artifacts['sharpness'] = min(laplacian_var / 500, 1.0)

        # Color uniformity
        if len(face.shape) == 3:
            lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
            color_uniformity = np.std(lab.reshape(-1, 3), axis=0).mean() / 128
            artifacts['color_uniformity'] = min(color_uniformity, 1.0)

        # Compression artifacts (DCT)
        gray_float = gray.astype(np.float32)
        dct_coeffs = cv2.dct(gray_float)
        high_freq_energy = np.mean(np.abs(dct_coeffs[50:, 50:]))
        artifacts['compression_artifacts'] = min(high_freq_energy / 100, 1.0)

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        artifacts['edge_density'] = np.sum(edges > 0) / edges.size

        # Noise level
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_diff = cv2.absdiff(gray, noise)
        artifacts['noise_level'] = np.mean(noise_diff) / 255

        artifact_score = (
            (1 - artifacts['sharpness']) * 0.25 +
            artifacts.get('color_uniformity', 0.5) * 0.2 +
            artifacts['compression_artifacts'] * 0.25 +
            artifacts['edge_density'] * 0.15 +
            artifacts['noise_level'] * 0.15
        )
        return artifact_score, artifacts

    def analyze_frequency_domain(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        h, w = magnitude.shape
        ch, cw = h // 2, w // 2

        low_freq = magnitude[ch-30:ch+30, cw-30:cw+30].mean()
        mid_freq = magnitude[ch-60:ch+60, cw-60:cw+60].mean()
        high_freq = magnitude.mean()
        freq_ratio = (high_freq - low_freq) / (mid_freq + 1e-6)

        return {
            'low_freq': float(low_freq),
            'mid_freq': float(mid_freq),
            'high_freq': float(high_freq),
            'freq_ratio': float(freq_ratio),
            'suspicious_pattern': bool(freq_ratio < 0.5 or freq_ratio > 2.0)
        }

    # ------------------------------------------------------------------
    # Main detection methods
    # ------------------------------------------------------------------
    def detect_deepfake_ensemble(self, image):
        faces = self.detect_faces_advanced(image)
        if len(faces) == 0:
            return {
                'is_deepfake': False,
                'confidence': 0,
                'message': 'No face detected',
                'face_count': 0
            }

        all_preds = []
        face_analyses = []

        for face_data in faces:
            face = face_data['face']
            face_resized = cv2.resize(face, (self.input_size, self.input_size))
            face_input = np.expand_dims(face_resized / 255.0, axis=0)

            predictions = []
            if self.model is not None:
                try:
                    pred1 = self.model.predict(face_input, verbose=0)[0][0]
                    predictions.append(float(pred1))
                except:
                    pass

            if len(self.ensemble_models) > 0:
                try:
                    pred2 = self.ensemble_models[0].predict(face_input, verbose=0)[0][0]
                    predictions.append(float(pred2))
                except:
                    pass

            if len(self.ensemble_models) > 1:
                try:
                    face_small = cv2.resize(face, (112, 112))
                    face_small_input = np.expand_dims(face_small / 255.0, axis=0)
                    pred3 = self.ensemble_models[1].predict(face_small_input, verbose=0)[0][0]
                    predictions.append(float(pred3))
                except:
                    pass

            artifact_score, artifacts = self.analyze_face_consistency(face)
            freq_analysis = self.analyze_frequency_domain(face)
            avg_pred = np.mean(predictions) if predictions else 0.5

            weighted = avg_pred * 0.5 + artifact_score * 0.3 + (1 if freq_analysis['suspicious_pattern'] else 0) * 0.2
            all_preds.append(weighted)

            face_analyses.append({
                'bbox': face_data['bbox'],
                'confidence': float(face_data['confidence']),
                'artifact_score': float(artifact_score),
                'artifacts': artifacts,
                'freq_analysis': freq_analysis,
                'model_score': float(avg_pred),
                'method': face_data.get('method', 'unknown')
            })

        final_score = float(np.mean(all_preds))
        is_df = final_score > self.deepfake_threshold
        if len(all_preds) > 1:
            consistency = 1 - float(np.std(all_preds))
        else:
            consistency = 1.0

        confidence = final_score * 100 * consistency

        return {
            'is_deepfake': bool(is_df),
            'confidence': float(min(confidence, 100)),
            'face_count': len(faces),
            'face_analyses': face_analyses,
            'ensemble_score': float(final_score),
            'consistency': float(consistency),
            'message': 'Deepfake detected with high confidence' if is_df and confidence > 80
                      else 'Likely authentic' if not is_df and confidence < 30
                      else 'Suspicious - further analysis recommended'
        }

    def detect_deepfake_video_advanced(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        frame_results = []
        temporal_scores = []

        for i in range(0, frame_count, 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                res = self.detect_deepfake_ensemble(frame)
                frame_results.append({
                    'frame_index': i,
                    'is_deepfake': res['is_deepfake'],
                    'confidence': res['confidence'],
                    'face_count': res['face_count']
                })
                temporal_scores.append(res['confidence'] / 100)

        cap.release()
        if len(temporal_scores) == 0:
            return {
                'is_deepfake': False,
                'confidence': 0,
                'message': 'Could not analyze video',
                'frames_analyzed': 0
            }

        temporal_mean = float(np.mean(temporal_scores))
        temporal_std = float(np.std(temporal_scores))
        is_df = temporal_mean > self.deepfake_threshold
        confidence = temporal_mean * 100 * (1 - temporal_std)

        return {
            'is_deepfake': bool(is_df),
            'confidence': float(min(confidence, 100)),
            'frames_analyzed': len(frame_results),
            'temporal_consistency': float((1 - temporal_std) * 100),
            'frame_results': frame_results[:10],
            'message': 'Deepfake detected in video' if is_df else 'Video appears authentic',
            'duration': float(duration)
        }

# Global instance for easy import
deepfake_detector = AdvancedDeepfakeDetector()