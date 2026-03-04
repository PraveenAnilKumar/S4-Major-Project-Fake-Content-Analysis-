"""
Sentiment Analysis Module with Ensemble Learning
Combines multiple models for accurate sentiment detection
Includes fine-tuning capability for domain-specific adaptation
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Union, Optional
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import string
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Try importing transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import Trainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try importing textblob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

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
    nltk.download('vader_lexicon')

class SentimentDataset(Dataset):
    """Custom Dataset for sentiment fine-tuning"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalyzer:
    """
    Sentiment Analyzer with ensemble of multiple models
    Supports fine-tuning for domain-specific adaptation
    """
    
    def __init__(self, use_ensemble: bool = True, models_dir: str = "models/sentiment"):
        """
        Initialize the sentiment analyzer
        
        Args:
            use_ensemble: Whether to use ensemble of models
            models_dir: Directory to save fine-tuned models
        """
        self.use_ensemble = use_ensemble
        self.is_trained = False
        self.ensemble_models = []
        self.ensemble_names = []
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize tokenizers and models for fine-tuning
        self.tokenizer = None
        self.transformer_model = None
        
        # Initialize VADER
        try:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER initialized")
        except:
            self.vader = None
            logger.warning("VADER not available")
        
        # Initialize transformer models if available
        if TRANSFORMERS_AVAILABLE and use_ensemble:
            self._init_transformer_models()
        
        # Initialize TextBlob
        if TEXTBLOB_AVAILABLE:
            logger.info("TextBlob available")
        
        # Always add VADER as base model
        if self.vader:
            self.ensemble_models.append('vader')
            self.ensemble_names.append('VADER')
        
        logger.info(f"Initialized with {len(self.ensemble_models)} models")
    
    def _init_transformer_models(self):
        """Initialize transformer-based models"""
        try:
            # DistilBERT for sentiment
            self.distilbert = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.ensemble_models.append('distilbert')
            self.ensemble_names.append('DistilBERT')
            logger.info("DistilBERT loaded")
        except Exception as e:
            logger.warning(f"Could not load DistilBERT: {e}")
        
        try:
            # RoBERTa for sentiment
            self.roberta = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.ensemble_models.append('roberta')
            self.ensemble_names.append('RoBERTa')
            logger.info("RoBERTa loaded")
        except Exception as e:
            logger.warning(f"Could not load RoBERTa: {e}")
        
        # Initialize tokenizer for fine-tuning
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=3  # POSITIVE, NEUTRAL, NEGATIVE
            )
            logger.info("Fine-tuning tokenizer and model initialized")
        except Exception as e:
            logger.warning(f"Could not initialize fine-tuning models: {e}")
    
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
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of text using ensemble
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, confidence)
            label: 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
            confidence: Confidence score (0-1)
        """
        if not text or not text.strip():
            return ('NEUTRAL', 0.5)
        
        if self.use_ensemble and len(self.ensemble_models) > 1:
            return self._analyze_ensemble(text)
        else:
            return self._analyze_single(text)
    
    def _analyze_ensemble(self, text: str) -> Tuple[str, float]:
        """
        Analyze using ensemble of models
        """
        results = []
        
        # Get predictions from each model
        if 'vader' in self.ensemble_models:
            vader_result = self._analyze_vader(text)
            if vader_result:
                results.append(vader_result)
        
        if 'distilbert' in self.ensemble_models:
            bert_result = self._analyze_distilbert(text)
            if bert_result:
                results.append(bert_result)
        
        if 'roberta' in self.ensemble_models:
            roberta_result = self._analyze_roberta(text)
            if roberta_result:
                results.append(roberta_result)
        
        if TEXTBLOB_AVAILABLE:
            blob_result = self._analyze_textblob(text)
            if blob_result:
                results.append(blob_result)
        
        # If no results, use fallback
        if not results:
            return self._analyze_heuristic(text)
        
        # Count votes
        votes = [r[0] for r in results]
        vote_counts = Counter(votes)
        
        # Get majority sentiment
        if vote_counts:
            majority_sentiment = vote_counts.most_common(1)[0][0]
            
            # Calculate average confidence for majority
            confidences = [r[1] for r in results if r[0] == majority_sentiment]
            avg_confidence = np.mean(confidences) if confidences else 0.7
            
            return (majority_sentiment, float(avg_confidence))
        
        return ('NEUTRAL', 0.5)
    
    def _analyze_single(self, text: str) -> Tuple[str, float]:
        """
        Analyze using single best available model
        """
        # Try transformer first
        if 'distilbert' in self.ensemble_models:
            result = self._analyze_distilbert(text)
            if result:
                return result
        
        # Then VADER
        if self.vader:
            result = self._analyze_vader(text)
            if result:
                return result
        
        # Then TextBlob
        if TEXTBLOB_AVAILABLE:
            result = self._analyze_textblob(text)
            if result:
                return result
        
        # Finally heuristic
        return self._analyze_heuristic(text)
    
    def _analyze_vader(self, text: str) -> Optional[Tuple[str, float]]:
        """Analyze using VADER"""
        try:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                return ('POSITIVE', abs(compound))
            elif compound <= -0.05:
                return ('NEGATIVE', abs(compound))
            else:
                return ('NEUTRAL', 0.5 + abs(compound) * 0.5)
        except:
            return None
    
    def _analyze_distilbert(self, text: str) -> Optional[Tuple[str, float]]:
        """Analyze using DistilBERT"""
        try:
            result = self.distilbert(text[:512])[0]
            label = result['label'].upper()
            score = result['score']
            
            # Map labels
            if 'POS' in label:
                return ('POSITIVE', score)
            elif 'NEG' in label:
                return ('NEGATIVE', score)
            else:
                return ('NEUTRAL', score)
        except:
            return None
    
    def _analyze_roberta(self, text: str) -> Optional[Tuple[str, float]]:
        """Analyze using RoBERTa"""
        try:
            result = self.roberta(text[:512])[0]
            label = result['label'].upper()
            score = result['score']
            
            # Map labels (RoBERTa uses LABEL_0, LABEL_1, LABEL_2)
            if 'LABEL_2' in label:  # Positive
                return ('POSITIVE', score)
            elif 'LABEL_0' in label:  # Negative
                return ('NEGATIVE', score)
            else:  # LABEL_1 is Neutral
                return ('NEUTRAL', score)
        except:
            return None
    
    def _analyze_textblob(self, text: str) -> Optional[Tuple[str, float]]:
        """Analyze using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return ('POSITIVE', polarity)
            elif polarity < -0.1:
                return ('NEGATIVE', abs(polarity))
            else:
                return ('NEUTRAL', 0.5 + abs(polarity) * 0.5)
        except:
            return None
    
    def _analyze_heuristic(self, text: str) -> Tuple[str, float]:
        """
        Simple heuristic-based analysis as fallback
        """
        text_lower = text.lower()
        
        # Positive words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best',
                         'fantastic', 'wonderful', 'awesome', 'happy', 'glad']
        
        # Negative words
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible',
                         'disappointing', 'poor', 'sad', 'angry', 'upset']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return ('NEUTRAL', 0.5)
        
        pos_ratio = pos_count / total
        
        if pos_ratio > 0.6:
            return ('POSITIVE', pos_ratio)
        elif pos_ratio < 0.4:
            return ('NEGATIVE', 1 - pos_ratio)
        else:
            return ('NEUTRAL', 0.6)
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for text in texts:
            try:
                label, conf = self.analyze(text)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': label,
                    'confidence': conf
                })
            except:
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': 'ERROR',
                    'confidence': 0.0
                })
        
        return pd.DataFrame(results)
    
    # ========== NEW FINE-TUNING METHOD ==========
    def fine_tune(self, texts: List[str], labels: List[str], 
                  epochs: int = 3, 
                  learning_rate: float = 2e-5,
                  batch_size: int = 16,
                  validation_split: float = 0.1,
                  save_model: bool = True) -> Dict:
        """
        Fine-tune transformer models on domain-specific data
        
        Args:
            texts: List of training texts
            labels: List of sentiment labels ('POSITIVE', 'NEUTRAL', 'NEGATIVE')
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            save_model: Whether to save the fine-tuned model
            
        Returns:
            Dictionary with training history and metrics
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available for fine-tuning")
            return {'error': 'Transformers not available'}
        
        if self.tokenizer is None or self.transformer_model is None:
            logger.warning("Fine-tuning models not initialized")
            return {'error': 'Fine-tuning models not initialized'}
        
        try:
            logger.info(f"Starting fine-tuning on {len(texts)} examples...")
            
            # Convert labels to IDs
            label_map = {'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0}
            reverse_label_map = {2: 'POSITIVE', 1: 'NEUTRAL', 0: 'NEGATIVE'}
            
            # Validate and convert labels
            valid_labels = []
            valid_texts = []
            
            for text, label in zip(texts, labels):
                if label in label_map:
                    valid_texts.append(text)
                    valid_labels.append(label_map[label])
            
            if len(valid_texts) == 0:
                logger.error("No valid labels found")
                return {'error': 'No valid labels provided'}
            
            logger.info(f"Processing {len(valid_texts)} valid examples")
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                valid_texts, valid_labels, 
                test_size=validation_split, 
                random_state=42,
                stratify=valid_labels if len(set(valid_labels)) > 1 else None
            )
            
            # Create datasets
            train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.models_dir, 'fine_tuned'),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.models_dir, 'logs'),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                learning_rate=learning_rate,
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.transformer_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self._compute_metrics,
            )
            
            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Evaluate
            logger.info("Evaluating on validation set...")
            eval_results = trainer.evaluate()
            
            # Save the fine-tuned model
            if save_model:
                model_path = os.path.join(self.models_dir, f'fine_tuned_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}')
                trainer.save_model(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"Fine-tuned model saved to {model_path}")
                
                # Update the pipeline to use fine-tuned model
                self.distilbert = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    tokenizer=model_path
                )
                logger.info("Updated inference pipeline with fine-tuned model")
            
            # Prepare results
            history = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_results.get('eval_loss', 0),
                'eval_accuracy': eval_results.get('eval_accuracy', 0),
                'eval_precision': eval_results.get('eval_precision', 0),
                'eval_recall': eval_results.get('eval_recall', 0),
                'eval_f1': eval_results.get('eval_f1', 0),
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'training_examples': len(train_texts),
                'validation_examples': len(val_texts)
            }
            
            logger.info(f"Fine-tuning completed. Validation accuracy: {history['eval_accuracy']:.4f}")
            self.is_trained = True
            
            return history
            
        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Load a previously fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available")
            return False
        
        try:
            self.distilbert = pipeline(
                "sentiment-analysis",
                model=model_path,
                tokenizer=model_path
            )
            logger.info(f"Loaded fine-tuned model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            return False
    
    # ========== EXISTING METHODS (unchanged) ==========
    def predict_proba(self, text: str) -> List[float]:
        """
        Get probability distribution across sentiments
        
        Args:
            text: Input text
            
        Returns:
            List of [negative, neutral, positive] probabilities
        """
        # Default distribution
        default = [0.33, 0.34, 0.33]
        
        try:
            label, conf = self.analyze(text)
            
            if label == 'POSITIVE':
                return [max(0.1, 1 - conf - 0.1), 0.1, min(0.9, conf)]
            elif label == 'NEGATIVE':
                return [min(0.9, conf), 0.1, max(0.1, 1 - conf - 0.1)]
            else:  # NEUTRAL
                return [0.2, 0.6, 0.2]
        except:
            return default
    
    def create_gauge(self, confidence: float, label: str) -> go.Figure:
        """
        Create a gauge chart for confidence visualization
        
        Args:
            confidence: Confidence score
            label: Sentiment label
            
        Returns:
            Plotly figure
        """
        # Color mapping
        colors = {
            'POSITIVE': 'green',
            'NEGATIVE': 'red',
            'NEUTRAL': 'gray'
        }
        
        color = colors.get(label, 'blue')
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Confidence - {label}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig
    
    def create_distribution(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create distribution chart for batch results
        
        Args:
            results_df: DataFrame with results
            
        Returns:
            Plotly figure
        """
        if results_df.empty:
            return go.Figure()
        
        counts = results_df['sentiment'].value_counts()
        
        colors = {
            'POSITIVE': 'green',
            'NEGATIVE': 'red',
            'NEUTRAL': 'gray'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[colors.get(s, 'blue') for s in counts.index]
            )
        ])
        
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        return {
            'ensemble_models': self.ensemble_names,
            'total_models': len(self.ensemble_models),
            'use_ensemble': self.use_ensemble,
            'is_fine_tuned': self.is_trained,
            'vader_available': self.vader is not None,
            'textblob_available': TEXTBLOB_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE
        }

# Create singleton instance
sentiment_analyzer = SentimentAnalyzer()
