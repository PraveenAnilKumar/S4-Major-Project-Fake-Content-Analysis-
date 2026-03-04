"""
Fake News Detection Module
Supports both traditional ML and transformer-based approaches
"""

import numpy as np
import pandas as pd
import os
import pickle
import logging
from typing import Union, Tuple, List, Dict
import re
import string

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Try importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        self.is_trained = False
        
        # Text preprocessing components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Try to load existing model
        self.load_model(model_path)
    
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
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
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
        
        # This is a placeholder - actual transformer training requires more code
        logger.warning("Full transformer training not implemented. Loading pre-trained model.")
        self.load_transformer_model('distilbert-base-uncased-finetuned-sst-2-english')
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict if text is fake news
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (label, confidence)
            label: 'FAKE' or 'REAL'
            confidence: Confidence score (0-1)
        """
        if not self.is_trained and self.model is None:
            logger.warning("Model not trained. Using fallback prediction.")
            return self._fallback_predict(text)
        
        try:
            if self.use_transformer and self.transformer_model is not None:
                return self._predict_transformer(text)
            else:
                return self._predict_traditional(text)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_predict(text)
    
    def _predict_traditional(self, text: str) -> Tuple[str, float]:
        """Predict using traditional ML model"""
        processed = self.preprocess_text(text)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([processed])[0]
            
            # Assuming binary classification with classes [0, 1]
            if len(proba) == 2:
                fake_prob = proba[1]
                real_prob = proba[0]
                
                if fake_prob > real_prob:
                    return ('FAKE', float(fake_prob))
                else:
                    return ('REAL', float(real_prob))
        
        # Fallback to simple prediction
        pred = self.model.predict([processed])[0]
        confidence = 0.95  # Default confidence
        return ('FAKE' if pred == 1 else 'REAL', confidence)
    
    def _predict_transformer(self, text: str) -> Tuple[str, float]:
        """Predict using transformer model"""
        if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
            return self._fallback_predict(text)
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Assuming binary classification
            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()
            
            if fake_prob > real_prob:
                return ('FAKE', fake_prob)
            else:
                return ('REAL', real_prob)
                
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return self._fallback_predict(text)
    
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
            return ('REAL', 0.55)  # Slight lean to real
    
    def load_model(self, path: str):
        """
        Load trained model from disk
        
        Args:
            path: Path to model directory
        """
        model_file = os.path.join(path, 'model.pkl')
        vectorizer_file = os.path.join(path, 'vectorizer.pkl')
        
        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            try:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.is_trained = True
                logger.info(f"Model loaded from {path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        # Try loading transformer model
        transformer_path = os.path.join(path, 'transformer')
        if os.path.exists(transformer_path):
            self.load_transformer_model(transformer_path)
    
    def load_transformer_model(self, model_name_or_path: str):
        """
        Load transformer model
        
        Args:
            model_name_or_path: Model name or path
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path
            )
            self.use_transformer = True
            logger.info(f"Transformer model loaded from {model_name_or_path}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
    
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

# Create singleton instance
fake_news_detector = FakeNewsDetector()
