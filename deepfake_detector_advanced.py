"""
Advanced Deepfake Detector with Ensemble Learning
Supports loading existing .h5 model files and model switching
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.preprocessing import image
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Any
import hashlib
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetectorAdvanced:
    """
    Advanced Deepfake Detector using ensemble of pre-trained models
    Supports loading existing .h5 model files and model switching
    """
    
    def __init__(self, threshold: float = 0.65, models_dir: str = "models"):
        """
        Initialize the deepfake detector
        
        Args:
            threshold: Confidence threshold for deepfake classification
            models_dir: Directory containing model files
        """
        self.threshold = threshold
        self.models_dir = models_dir
        self.ensemble_models = []
        self.model_names = []
        self.model_paths = []
        self.input_size = (224, 224)
        self.model_weights = {}  # Dictionary to store custom weights for ensemble
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load your existing models
        self._load_existing_models()
        
        # If no models loaded, fallback to pre-trained
        if len(self.ensemble_models) == 0:
            logger.warning("No existing models found. Loading pre-trained models as fallback.")
            self._load_pretrained_models()
    
    def _load_existing_models(self):
        """Load your existing .h5 model files"""
        try:
            # Look for all .h5 files in the models directory
            h5_files = glob.glob(os.path.join(self.models_dir, "*.h5"))
            
            if not h5_files:
                logger.info("No .h5 files found in models directory")
                return
            
            logger.info(f"Found {len(h5_files)} .h5 model files")
            
            for model_path in h5_files:
                try:
                    model_name = os.path.basename(model_path).replace('.h5', '')
                    logger.info(f"Loading model: {model_name} from {model_path}")
                    
                    # Load the model
                    model = load_model(model_path, compile=False)
                    
                    self.ensemble_models.append(model)
                    self.model_names.append(model_name)
                    self.model_paths.append(model_path)
                    
                    logger.info(f"Successfully loaded {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading model {model_path}: {e}")
            
            logger.info(f"Loaded {len(self.ensemble_models)} existing models successfully")
            
        except Exception as e:
            logger.error(f"Error scanning models directory: {e}")
    
    def _load_pretrained_models(self):
        """Load pre-trained models as fallback"""
        try:
            # Load ResNet50
            logger.info("Loading ResNet50...")
            resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.ensemble_models.append(resnet)
            self.model_names.append('ResNet50')
            self.model_paths.append('pretrained')
            
            # Load EfficientNet
            logger.info("Loading EfficientNet...")
            efficientnet = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            self.ensemble_models.append(efficientnet)
            self.model_names.append('EfficientNet')
            self.model_paths.append('pretrained')
            
            # Load Xception
            logger.info("Loading Xception...")
            xception = Xception(weights='imagenet', include_top=False, pooling='avg')
            self.ensemble_models.append(xception)
            self.model_names.append('Xception')
            self.model_paths.append('pretrained')
            
            logger.info(f"Loaded {len(self.ensemble_models)} pre-trained models successfully")
            
        except Exception as e:
            logger.error(f"Error loading pre-trained models: {e}")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models if primary models fail"""
        try:
            from tensorflow.keras.applications import MobileNetV2
            mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.ensemble_models.append(mobilenet)
            self.model_names.append('MobileNetV2')
            self.model_paths.append('pretrained')
            logger.info("Loaded MobileNetV2 as fallback")
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
    
    def set_model_weights(self, weights: Dict[str, float]):
        """
        Set custom weights for ensemble models
        
        Args:
            weights: Dictionary mapping model names to weights
        """
        self.model_weights = weights
        logger.info(f"Model weights updated: {weights}")
    
    def preprocess_for_model(self, image_array: np.ndarray, model_name: str) -> np.ndarray:
        """
        Preprocess image for specific model based on its expected input
        
        Args:
            image_array: Input image array
            model_name: Name of the model
            
        Returns:
            Preprocessed image batch
        """
        # Resize image
        img_resized = cv2.resize(image_array, self.input_size)
        
        # Convert to RGB if needed
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
        
        # Convert to float32 and normalize to [0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict_with_model(self, model: Any, image_batch: np.ndarray) -> float:
        """
        Get prediction from a model
        
        Args:
            model: Loaded Keras model
            image_batch: Preprocessed image batch
            
        Returns:
            Deepfake probability score
        """
        try:
            # Get prediction
            pred = model.predict(image_batch, verbose=0)
            
            # Handle different output shapes
            if isinstance(pred, list):
                pred = pred[0]
            
            # Flatten if needed
            if len(pred.shape) > 2:
                pred = pred.flatten()
            
            # Get the probability (assuming binary classification)
            if pred.shape[-1] == 2:  # Binary classification with 2 outputs
                score = float(pred[0][1])  # Probability of fake
            elif len(pred) == 1:  # Single output
                score = float(pred[0])
            else:
                # Take mean as score
                score = float(np.mean(pred))
            
            return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return 0.5  # Return neutral score on error
    
    def detect_faces(self, image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        
        Args:
            image_array: Input image
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def detect_deepfake_ensemble(self, image_array: np.ndarray) -> Dict:
        """
        Detect if an image is a deepfake using ensemble of loaded models
        
        Args:
            image_array: Input image array
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Detect faces
            faces = self.detect_faces(image_array)
            face_count = len(faces)
            
            if face_count == 0:
                return {
                    'is_deepfake': False,
                    'confidence': 0.0,
                    'face_count': 0,
                    'ensemble_score': 0.0,
                    'consistency': 0.0,
                    'message': 'No faces detected in image',
                    'model_scores': {},
                    'models_used': self.model_names
                }
            
            # Preprocess image once
            img_batch = self.preprocess_for_model(image_array, 'generic')
            
            # Get predictions from all models
            model_scores = {}
            valid_scores = []
            
            for i, model in enumerate(self.ensemble_models):
                model_name = self.model_names[i]
                try:
                    score = self.predict_with_model(model, img_batch)
                    model_scores[model_name] = float(score)
                    valid_scores.append(score)
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
                    model_scores[model_name] = 0.5  # Neutral score on error
            
            if not valid_scores:
                return {
                    'is_deepfake': False,
                    'confidence': 0.0,
                    'face_count': face_count,
                    'ensemble_score': 0.0,
                    'consistency': 0.0,
                    'message': 'All models failed to predict',
                    'model_scores': model_scores,
                    'models_used': self.model_names
                }
            
            # Calculate ensemble score (weighted average if weights are set)
            if self.model_weights and len(self.model_weights) > 0:
                # Calculate weighted average
                weighted_sum = 0
                total_weight = 0
                for model_name, score in model_scores.items():
                    weight = self.model_weights.get(model_name, 1.0)
                    weighted_sum += score * weight
                    total_weight += weight
                
                ensemble_score = weighted_sum / total_weight if total_weight > 0 else np.mean(valid_scores)
            else:
                # Default to simple average
                ensemble_score = np.mean(valid_scores)
            
            # Calculate consistency between models (lower std dev = more consistent)
            consistency = 1.0 - np.std(valid_scores) if len(valid_scores) > 1 else 1.0
            
            # Final decision
            is_deepfake = ensemble_score > self.threshold
            confidence = ensemble_score * 100 if is_deepfake else (1 - ensemble_score) * 100
            
            result = {
                'is_deepfake': is_deepfake,
                'confidence': float(confidence),
                'face_count': face_count,
                'ensemble_score': float(ensemble_score),
                'consistency': float(consistency),
                'model_scores': model_scores,
                'models_used': self.model_names,
                'message': self._get_message(is_deepfake, confidence, face_count, ensemble_score)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in deepfake detection: {e}")
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'face_count': 0,
                'ensemble_score': 0.0,
                'consistency': 0.0,
                'message': f'Error during analysis: {str(e)}',
                'model_scores': {},
                'models_used': self.model_names
            }
    
    def detect_with_single_model(self, image_array: np.ndarray, model_name: str) -> Dict:
        """
        Detect deepfake using a single specific model
        
        Args:
            image_array: Input image array
            model_name: Name of the model to use
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Find model index
            if model_name not in self.model_names:
                return {
                    'is_deepfake': False,
                    'confidence': 0.0,
                    'face_count': 0,
                    'ensemble_score': 0.0,
                    'consistency': 1.0,
                    'message': f'Model {model_name} not found',
                    'model_scores': {},
                    'models_used': [model_name]
                }
            
            model_idx = self.model_names.index(model_name)
            model = self.ensemble_models[model_idx]
            
            # Detect faces
            faces = self.detect_faces(image_array)
            face_count = len(faces)
            
            if face_count == 0:
                return {
                    'is_deepfake': False,
                    'confidence': 0.0,
                    'face_count': 0,
                    'ensemble_score': 0.0,
                    'consistency': 1.0,
                    'message': 'No faces detected in image',
                    'model_used': model_name,
                    'models_used': [model_name]
                }
            
            # Preprocess and predict
            img_batch = self.preprocess_for_model(image_array, model_name)
            score = self.predict_with_model(model, img_batch)
            
            # Final decision
            is_deepfake = score > self.threshold
            confidence = score * 100 if is_deepfake else (1 - score) * 100
            
            result = {
                'is_deepfake': is_deepfake,
                'confidence': float(confidence),
                'face_count': face_count,
                'ensemble_score': float(score),
                'consistency': 1.0,
                'model_scores': {model_name: float(score)},
                'model_used': model_name,
                'models_used': [model_name],
                'message': self._get_message(is_deepfake, confidence, face_count, score)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single model detection: {e}")
            return {
                'is_deepfake': False,
                'confidence': 0.0,
                'face_count': 0,
                'ensemble_score': 0.0,
                'consistency': 1.0,
                'message': f'Error during analysis: {str(e)}',
                'model_used': model_name,
                'models_used': [model_name] if model_name else []
            }
    
    def detect_deepfake_video_advanced(self, video_path: str, sample_rate: int = 30) -> Dict:
        """
        Analyze video for deepfakes using ensemble
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame
            
        Returns:
            Dictionary with video analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            frame_results = []
            frame_count = 0
            processed_frames = 0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Analyze frame with ensemble
                    result = self.detect_deepfake_ensemble(frame)
                    frame_results.append({
                        'frame': frame_count,
                        'is_deepfake': result['is_deepfake'],
                        'confidence': result['confidence'],
                        'ensemble_score': result['ensemble_score']
                    })
                    processed_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            if not frame_results:
                return {'error': 'No frames could be processed'}
            
            # Aggregate results
            deepfake_frames = sum(1 for r in frame_results if r['is_deepfake'])
            avg_confidence = np.mean([r['confidence'] for r in frame_results])
            avg_ensemble_score = np.mean([r['ensemble_score'] for r in frame_results])
            
            # Determine video authenticity
            deepfake_ratio = deepfake_frames / len(frame_results)
            is_deepfake = deepfake_ratio > 0.3  # Threshold for video
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': float(avg_confidence),
                'deepfake_frames': deepfake_frames,
                'total_frames_analyzed': len(frame_results),
                'deepfake_ratio': float(deepfake_ratio),
                'avg_ensemble_score': float(avg_ensemble_score),
                'frame_results': frame_results[:10],  # Return first 10 for preview
                'models_used': self.model_names,
                'message': self._get_video_message(is_deepfake, deepfake_ratio, avg_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            return {'error': str(e)}
    
    def detect_video_with_single_model(self, video_path: str, model_name: str, sample_rate: int = 30) -> Dict:
        """
        Analyze video using a single specific model
        
        Args:
            video_path: Path to video file
            model_name: Name of the model to use
            sample_rate: Sample every Nth frame
            
        Returns:
            Dictionary with video analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            frame_results = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Analyze frame with single model
                    result = self.detect_with_single_model(frame, model_name)
                    frame_results.append({
                        'frame': frame_count,
                        'is_deepfake': result['is_deepfake'],
                        'confidence': result['confidence'],
                        'ensemble_score': result['ensemble_score']
                    })
                    processed_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            if not frame_results:
                return {'error': 'No frames could be processed'}
            
            # Aggregate results
            deepfake_frames = sum(1 for r in frame_results if r['is_deepfake'])
            avg_confidence = np.mean([r['confidence'] for r in frame_results])
            avg_ensemble_score = np.mean([r['ensemble_score'] for r in frame_results])
            
            deepfake_ratio = deepfake_frames / len(frame_results)
            is_deepfake = deepfake_ratio > 0.3
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': float(avg_confidence),
                'deepfake_frames': deepfake_frames,
                'total_frames_analyzed': len(frame_results),
                'deepfake_ratio': float(deepfake_ratio),
                'avg_ensemble_score': float(avg_ensemble_score),
                'frame_results': frame_results[:10],
                'model_used': model_name,
                'models_used': [model_name],
                'message': self._get_video_message(is_deepfake, deepfake_ratio, avg_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        return {
            'total_models': len(self.ensemble_models),
            'model_names': self.model_names,
            'model_paths': self.model_paths,
            'threshold': self.threshold,
            'weights': self.model_weights
        }
    
    def _get_message(self, is_deepfake: bool, confidence: float, face_count: int, ensemble_score: float) -> str:
        """Generate user-friendly message"""
        if is_deepfake:
            if confidence > 80:
                return f"🚨 High confidence deepfake detected! ({confidence:.1f}%) - {face_count} face(s) analyzed"
            else:
                return f"⚠️ Potential deepfake detected ({confidence:.1f}%) - Further analysis recommended"
        else:
            if confidence < 30:
                return f"✅ Image appears authentic ({(100-confidence):.1f}% confidence) - {face_count} face(s) analyzed"
            else:
                return f"ℹ️ No clear signs of manipulation ({(100-confidence):.1f}% confidence)"
    
    def _get_video_message(self, is_deepfake: bool, ratio: float, confidence: float) -> str:
        """Generate message for video analysis"""
        if is_deepfake:
            return f"🚨 Deepfake detected in {ratio*100:.1f}% of frames (avg confidence: {confidence:.1f}%)"
        else:
            return f"✅ Video appears authentic (deepfake in only {ratio*100:.1f}% of frames)"

# Create singleton instance with models directory pointing to your models folder
deepfake_detector = DeepfakeDetectorAdvanced(models_dir="models")
