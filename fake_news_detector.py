"""
Fake News Detection Module
Supports both traditional ML and transformer-based approaches
Now with support for loading trained models
"""

import numpy as np
import pandas as pd
import os
import pickle
import logging
import glob
from typing import Union, Tuple, List, Dict, Optional
import re
import string
from datetime import datetime
import json

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Try importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class FakeNewsDetector:
    """
    Fake News Detector with multiple model support
    Can load both traditional ML and transformer models
    """
    
    def __init__(self, use_transformer: bool = False, model_path: str = 'models/fake_news/'):
        """
        Initialize the fake news detector
        
        Args:
            use_transformer: Whether to use transformer model
            model_path: Path to saved models
        """
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.transformer_model = None
        self.transformer_pipeline = None
        self.is_trained = False
        self.model_metadata = {}
        self.available_models = []
        
        # Text preprocessing components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing models
        self.load_best_model()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_best_model(self):
        """Automatically load the best available model"""
        logger.info(f"Scanning for models in {self.model_path}")
        
        # First, get all available models
        self.get_available_models()
        
        if not self.available_models:
            logger.warning("⚠️ No models found in directory")
            return
        
        # Try to load transformer models first (usually more accurate)
        transformer_models = [m for m in self.available_models if m['type'] == 'transformer']
        if transformer_models and TRANSFORMERS_AVAILABLE:
            # Get the most recent transformer model
            latest_transformer = transformer_models[0]  # Already sorted by timestamp
            logger.info(f"Attempting to load transformer model: {latest_transformer['name']}")
            if self.load_transformer_model(latest_transformer['path']):
                self.use_transformer = True
                self.is_trained = True
                logger.info("✅ Successfully loaded transformer model")
                return
            else:
                logger.warning(f"Failed to load transformer model, trying next...")
        
        # Try Random Forest models
        rf_models = [m for m in self.available_models if m['type'] == 'random_forest']
        if rf_models:
            latest_rf = rf_models[0]
            logger.info(f"Attempting to load Random Forest model: {latest_rf['name']}")
            if self.load_traditional_model(latest_rf['path']):
                self.use_transformer = False
                self.is_trained = True
                logger.info("✅ Successfully loaded Random Forest model")
                return
            else:
                logger.warning(f"Failed to load Random Forest model")
        
        # Try pipeline model as last resort
        model_file = os.path.join(self.model_path, 'model.pkl')
        vectorizer_file = os.path.join(self.model_path, 'vectorizer.pkl')
        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            logger.info("Attempting to load pipeline model")
            if self.load_pipeline_model(model_file, vectorizer_file):
                self.use_transformer = False
                self.is_trained = True
                logger.info("✅ Successfully loaded pipeline model")
                return
        
        logger.warning("⚠️ No models could be loaded. Using fallback predictions.")
        self.is_trained = False
    
    def load_traditional_model(self, model_path: str) -> bool:
        """
        Load a trained traditional ML model (Random Forest, etc.)
        
        Args:
            model_path: Path to the .pkl model file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            logger.info(f"Loading traditional model from {model_path}")
            
            # Try to load with joblib first
            self.model = joblib.load(model_path)
            
            # Try to find corresponding vectorizer
            base_name = os.path.basename(model_path).replace('.pkl', '')
            vectorizer_patterns = [
                os.path.join(self.model_path, f'vectorizer_{base_name.replace("random_forest_", "")}.pkl'),
                os.path.join(self.model_path, 'vectorizer.pkl')
            ]
            
            vectorizer_loaded = False
            for vec_path in vectorizer_patterns:
                if os.path.exists(vec_path):
                    self.vectorizer = joblib.load(vec_path)
                    logger.info(f"Loaded vectorizer from {vec_path}")
                    vectorizer_loaded = True
                    break
            
            # If model is a pipeline, extract vectorizer
            if hasattr(self.model, 'named_steps') and 'tfidf' in self.model.named_steps:
                self.vectorizer = self.model.named_steps['tfidf']
                vectorizer_loaded = True
            
            if not vectorizer_loaded:
                logger.warning("No vectorizer found, but model may still work")
            
            self.is_trained = True
            self.use_transformer = False
            
            # Load metadata if exists
            meta_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded metadata: {self.model_metadata.get('accuracy', 'N/A')}")
            
            logger.info(f"✅ Traditional model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading traditional model: {e}")
            return False
    
    def load_pipeline_model(self, model_file: str, vectorizer_file: str) -> bool:
        """
        Load pipeline model and vectorizer
        
        Args:
            model_file: Path to model.pkl
            vectorizer_file: Path to vectorizer.pkl
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            logger.info(f"Loading pipeline model from {model_file}")
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.is_trained = True
            self.use_transformer = False
            logger.info("✅ Pipeline model loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading pipeline model: {e}")
            return False
    
    def load_transformer_model(self, model_path: str) -> bool:
        """
        Load a trained transformer model
        
        Args:
            model_path: Path to transformer model directory
            
        Returns:
            bool: True if loaded successfully
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available")
            return False
        
        try:
            logger.info(f"Loading transformer model from {model_path}")
            
            # Check if it's a directory with final_model
            if os.path.isdir(model_path):
                model_dir = model_path
                # Check for final_model subdirectory
                final_model_dir = os.path.join(model_path, 'final_model')
                if os.path.exists(final_model_dir):
                    model_dir = final_model_dir
                    logger.info(f"Using final_model subdirectory: {model_dir}")
            else:
                model_dir = model_path
            
            # Try to load with pipeline first (easier)
            try:
                self.transformer_pipeline = pipeline(
                    "text-classification",
                    model=model_dir,
                    tokenizer=model_dir
                )
                logger.info(f"✅ Transformer pipeline loaded from {model_dir}")
                
                # Test the pipeline
                test_result = self.transformer_pipeline("test")[0]
                logger.info(f"Pipeline test successful: {test_result}")
                
            except Exception as e:
                logger.warning(f"Pipeline loading failed: {e}, trying manual load...")
                # Fall back to manual loading
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    logger.info(f"✅ Transformer model loaded manually from {model_dir}")
                except Exception as e2:
                    logger.error(f"Manual loading also failed: {e2}")
                    return False
            
            self.use_transformer = True
            self.is_trained = True
            
            # Load metadata
            meta_path = os.path.join(model_path, 'metrics.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded metadata with accuracy: {self.model_metadata.get('accuracy', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            return False
    
    def get_available_models(self) -> List[Dict]:
        """
        Get list of all available trained models
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        logger.info(f"Scanning directory: {self.model_path}")
        
        # Check transformer models
        transformer_dirs = glob.glob(os.path.join(self.model_path, "transformer_*"))
        logger.info(f"Found {len(transformer_dirs)} transformer directories")
        
        for trans_dir in transformer_dirs:
            model_info = {
                'name': os.path.basename(trans_dir),
                'type': 'transformer',
                'path': trans_dir,
                'timestamp': os.path.getctime(trans_dir)
            }
            
            # Try to load metrics
            meta_path = os.path.join(trans_dir, 'metrics.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        metrics = json.load(f)
                        model_info['accuracy'] = metrics.get('accuracy', 'N/A')
                        model_info['f1'] = metrics.get('f1', 'N/A')
                except:
                    model_info['accuracy'] = 'N/A'
            
            models.append(model_info)
            logger.info(f"Found transformer model: {model_info['name']}")
        
        # Check Random Forest models
        rf_files = glob.glob(os.path.join(self.model_path, "random_forest_*.pkl"))
        logger.info(f"Found {len(rf_files)} Random Forest files")
        
        for rf_file in rf_files:
            model_info = {
                'name': os.path.basename(rf_file),
                'type': 'random_forest',
                'path': rf_file,
                'timestamp': os.path.getctime(rf_file)
            }
            
            # Try to load metadata
            meta_path = rf_file.replace('.pkl', '_metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        metrics = json.load(f)
                        model_info['accuracy'] = metrics.get('accuracy', 'N/A')
                except:
                    model_info['accuracy'] = 'N/A'
            
            models.append(model_info)
            logger.info(f"Found RF model: {model_info['name']}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        self.available_models = models
        
        logger.info(f"Total models found: {len(models)}")
        return models
    
    def train(self, texts: List[str], labels: List[int], model_type: str = 'logistic'):
        """
        Train the fake news detector
        
        Args:
            texts: List of text samples
            labels: List of labels (0=real, 1=fake)
            model_type: Type of model ('logistic', 'random_forest', or 'transformer')
        """
        logger.info(f"Training {model_type} model on {len(texts)} samples")
        
        if model_type == 'transformer' and TRANSFORMERS_AVAILABLE:
            self._train_transformer(texts, labels)
        else:
            self._train_traditional(texts, labels, model_type)
        
        self.is_trained = True
        logger.info("Training completed")
    
    def _train_traditional(self, texts: List[str], labels: List[int], model_type: str):
        """Train traditional ML model"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create pipeline
        if model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:  # random_forest
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', classifier)
        ])
        
        # Train
        self.model.fit(processed_texts, labels)
        
        # Save vectorizer separately for inference
        self.vectorizer = self.model.named_steps['tfidf']
        
        # Save model
        self._save_model()
    
    def _train_transformer(self, texts: List[str], labels: List[int]):
        """Train transformer model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available")
            return
        
        # This would require the full training script
        logger.warning("Full transformer training should use train_fakenews_transformer.py")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict if text is fake news using the loaded model
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (label, confidence)
            label: 'FAKE' or 'REAL'
            confidence: Confidence score (0-1)
        """
        # Check if we have any model loaded
        model_available = (self.model is not None or 
                          self.transformer_pipeline is not None or 
                          self.transformer_model is not None)
        
        if not self.is_trained or not model_available:
            logger.warning("No trained model loaded. Using fallback prediction.")
            return self._fallback_predict(text)
        
        try:
            if self.use_transformer:
                return self._predict_transformer(text)
            else:
                return self._predict_traditional(text)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_predict(text)
    
    def _predict_traditional(self, text: str) -> Tuple[str, float]:
        """Predict using traditional ML model"""
        try:
            # Preprocess text
            processed = self.preprocess_text(text)
            
            # If model is a pipeline, use it directly
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([processed])[0]
                
                # Handle different output shapes
                if len(proba) == 2:
                    fake_prob = proba[1]
                    real_prob = proba[0]
                    
                    if fake_prob > real_prob:
                        return ('FAKE', float(fake_prob))
                    else:
                        return ('REAL', float(real_prob))
                else:
                    # Multi-class or single output
                    pred = self.model.predict([processed])[0]
                    confidence = 0.95
                    return ('FAKE' if pred == 1 else 'REAL', confidence)
            
            # Fallback
            pred = self.model.predict([processed])[0]
            return ('FAKE' if pred == 1 else 'REAL', 0.95)
            
        except Exception as e:
            logger.error(f"Traditional prediction error: {e}")
            return self._fallback_predict(text)
    
    def _predict_transformer(self, text: str) -> Tuple[str, float]:
        """Predict using transformer model"""
        try:
            # Use pipeline if available
            if self.transformer_pipeline is not None:
                result = self.transformer_pipeline(text[:512])[0]
                label = result['label']
                score = result['score']
                
                # Map labels
                if 'LABEL_1' in label or 'FAKE' in label.upper() or label.lower() == 'fake':
                    return ('FAKE', score)
                elif 'LABEL_0' in label or 'REAL' in label.upper() or label.lower() == 'real':
                    return ('REAL', score)
                else:
                    # Try to infer from score
                    if score > 0.5:
                        return ('FAKE', score)
                    else:
                        return ('REAL', 1 - score)
            
            # Manual inference
            elif self.tokenizer and self.transformer_model:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Assuming binary classification (0=real, 1=fake)
                if probs.shape[-1] == 2:
                    fake_prob = probs[0][1].item()
                    real_prob = probs[0][0].item()
                    
                    if fake_prob > real_prob:
                        return ('FAKE', fake_prob)
                    else:
                        return ('REAL', real_prob)
                else:
                    # Multi-class
                    pred_class = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred_class].item()
                    return ('FAKE' if pred_class == 1 else 'REAL', confidence)
            
            else:
                return self._fallback_predict(text)
                
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return self._fallback_predict(text)
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of (label, confidence) tuples
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def _fallback_predict(self, text: str) -> Tuple[str, float]:
        """
        Fallback prediction using simple heuristics
        """
        # Simple keyword-based detection
        text_lower = text.lower()
        
        fake_indicators = [
            'breaking', 'shocking', 'you won\'t believe', 'viral',
            'they don\'t want you to know', 'secret', 'conspiracy',
            'miracle', 'cure', 'hidden truth', 'what happened next'
        ]
        
        real_indicators = [
            'according to', 'source', 'report', 'study', 'research',
            'official', 'government', 'university', 'published'
        ]
        
        fake_score = sum(1 for word in fake_indicators if word in text_lower)
        real_score = sum(1 for word in real_indicators if word in text_lower)
        
        total = fake_score + real_score
        if total == 0:
            return ('REAL', 0.6)  # Default to real with moderate confidence
        
        fake_ratio = fake_score / total
        
        if fake_ratio > 0.6:
            return ('FAKE', fake_ratio)
        elif fake_ratio < 0.3:
            return ('REAL', 1 - fake_ratio)
        else:
            return ('REAL', 0.55)
    
    def load_model(self, path: str):
        """
        Load trained model from disk (legacy method)
        
        Args:
            path: Path to model directory
        """
        self.load_best_model()
    
    def _save_model(self):
        """Save model to disk"""
        os.makedirs(self.model_path, exist_ok=True)
        
        model_file = os.path.join(self.model_path, 'model.pkl')
        vectorizer_file = os.path.join(self.model_path, 'vectorizer.pkl')
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the currently loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'is_trained': self.is_trained,
            'use_transformer': self.use_transformer,
            'model_path': self.model_path,
            'available_models': len(self.available_models) if self.available_models else 0,
            'metadata': self.model_metadata
        }
        
        if self.is_trained:
            if self.use_transformer:
                info['model_type'] = 'transformer'
                if hasattr(self, 'transformer_pipeline') and self.transformer_pipeline:
                    try:
                        info['model_name'] = str(self.transformer_pipeline.model.config._name_or_path)
                    except:
                        info['model_name'] = 'transformer'
            else:
                if self.model:
                    if hasattr(self.model, 'named_steps'):
                        classifier = self.model.named_steps.get('classifier')
                        if classifier:
                            info['model_type'] = type(classifier).__name__
                        else:
                            info['model_type'] = type(self.model).__name__
                    else:
                        info['model_type'] = type(self.model).__name__
        else:
            info['model_type'] = 'fallback'
        
        return info

# Create singleton instance - will automatically load the best model
fake_news_detector = FakeNewsDetector()
