from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import os
import re
import requests
from urllib.parse import urlparse
import whois
from datetime import datetime, timedelta
import warnings
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from newspaper import Article
import feedparser
import time
import json
from bs4 import BeautifulSoup
import validators
import tldextract
import hashlib
import pickle

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

class AdvancedFakeNewsDetector:
    def __init__(self):
        # Initialize models
        self.text_model = None
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.classifier = None
        self.gb_classifier = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Knowledge base for known fake news sources
        self.fake_news_domains = self.load_fake_news_domains()
        self.reliable_domains = self.load_reliable_domains()
        
        # Cache for news analysis
        self.analysis_cache = {}
        
        # Load or create models
        self.load_or_create_models()
        
        # Initialize live news sources
        self.news_sources = self.init_news_sources()
        
    def load_fake_news_domains(self):
        """Load known fake news domains"""
        # This would ideally be loaded from a file
        # For now, return a sample list
        return set([
            'infowars.com',
            'breitbart.com',
            'naturalnews.com',
            'beforeitsnews.com',
            'theonion.com'  # Satire, not fake but should be flagged
        ])
    
    def load_reliable_domains(self):
        """Load known reliable news domains"""
        return set([
            'reuters.com',
            'apnews.com',
            'bbc.com',
            'bbc.co.uk',
            'nytimes.com',
            'washingtonpost.com',
            'wsj.com',
            'economist.com',
            'theguardian.com',
            'npr.org',
            'cnn.com',
            'aljazeera.com'
        ])
    
    def init_news_sources(self):
        """Initialize RSS feeds for live news"""
        return {
            'reuters': 'http://feeds.reuters.com/reuters/topNews',
            'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
            'ap': 'https://www.apnews.com/apf-topnews',
            'aljazeera': 'https://www.aljazeera.com/xml/rss/all.xml'
        }
    
    def load_or_create_models(self):
        """Load or create ensemble of models"""
        model_path = 'models/fake_news_rf.pkl'
        gb_model_path = 'models/fake_news_gb.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        count_vectorizer_path = 'models/count_vectorizer.pkl'
        
        if all(os.path.exists(p) for p in [model_path, vectorizer_path]):
            self.classifier = joblib.load(model_path)
            self.tfidf_vectorizer = joblib.load(vectorizer_path)
            if os.path.exists(gb_model_path):
                self.gb_classifier = joblib.load(gb_model_path)
            if os.path.exists(count_vectorizer_path):
                self.count_vectorizer = joblib.load(count_vectorizer_path)
        else:
            # Initialize lightweight models
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )
            self.count_vectorizer = CountVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )
            self.gb_classifier = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        
        # Load transformer model for deep text analysis
        try:
            self.text_model = pipeline(
                "text-classification",
                model="roberta-base-openai-detector",
                device=-1,
                max_length=512,
                truncation=True
            )
        except:
            try:
                # Fallback to smaller model
                self.text_model = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
            except:
                self.text_model = None
    
    def preprocess_text_advanced(self, text):
        """Advanced text preprocessing"""
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs but keep domain for analysis
        text = re.sub(r'http\S+', '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_advanced_features(self, text):
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic statistics
        words = text.split()
        sentences = nltk.sent_tokenize(text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Readability scores
        if sentences and words:
            # Flesch-Kincaid grade level
            syllables = sum([self.count_syllables(word) for word in words])
            features['fk_grade'] = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        else:
            features['fk_grade'] = 0
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_compound'] = sentiment['compound']
        features['sentiment_positive'] = sentiment['pos']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_neutral'] = sentiment['neu']
        
        # Subjectivity using TextBlob
        blob = TextBlob(text)
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Linguistic features
        features['exclamation_ratio'] = text.count('!') / len(words) if words else 0
        features['question_ratio'] = text.count('?') / len(words) if words else 0
        features['quote_ratio'] = text.count('"') / len(words) if words else 0
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Named entity density (simplified)
        entities = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
        features['entity_density'] = len([chunk for chunk in entities if hasattr(chunk, 'label')]) / len(words) if words else 0
        
        # Part of speech distribution (simplified)
        pos_tags = nltk.pos_tag(words)
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        features['noun_ratio'] = pos_counts.get('NN', 0) / len(words) if words else 0
        features['verb_ratio'] = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0)) / len(words) if words else 0
        features['adjective_ratio'] = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)) / len(words) if words else 0
        
        # Clickbait detection
        clickbait_phrases = [
            'you won\'t believe', 'shocking', 'mind blowing', 'incredible',
            'what happens next', 'the reason why', 'this is what happens',
            'doctors hate this', 'secret', 'miracle', 'will make you',
            'changed my life', 'you need to see', 'gone wrong', 'amazing'
        ]
        
        features['clickbait_score'] = sum(1 for phrase in clickbait_phrases if phrase in text.lower()) / len(clickbait_phrases)
        
        return features
    
    def count_syllables(self, word):
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count += 1
        return count
    
    def analyze_source_credibility_advanced(self, url):
        """Advanced source credibility analysis"""
        if not url or not validators.url(url):
            return {
                'credibility_score': 0,
                'domain': 'invalid',
                'issues': ['Invalid URL']
            }
        
        try:
            # Extract domain
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"
            full_domain = f"{extracted.subdomain}.{domain}" if extracted.subdomain else domain
            
            # Check against known lists
            if domain in self.fake_news_domains or full_domain in self.fake_news_domains:
                return {
                    'credibility_score': 0,
                    'domain': full_domain,
                    'issues': ['Known fake news source'],
                    'warning': 'HIGH_RISK'
                }
            
            if domain in self.reliable_domains or full_domain in self.reliable_domains:
                return {
                    'credibility_score': 90,
                    'domain': full_domain,
                    'issues': [],
                    'warning': 'RELIABLE'
                }
            
            # Check domain age
            try:
                domain_info = whois.whois(domain)
                creation_date = None
                
                if domain_info.creation_date:
                    if isinstance(domain_info.creation_date, list):
                        creation_date = domain_info.creation_date[0]
                    else:
                        creation_date = domain_info.creation_date
                
                if creation_date:
                    domain_age = (datetime.now() - creation_date).days
                    is_new_domain = domain_age < 180  # Less than 6 months
                else:
                    domain_age = 0
                    is_new_domain = True
            except:
                domain_age = 0
                is_new_domain = True
            
            # Check SSL certificate
            has_ssl = url.startswith('https')
            
            # Check for suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club', '.online']
            is_suspicious_tld = any(domain.endswith(tld) for tld in suspicious_tlds)
            
            # Calculate credibility score
            credibility_score = 50  # Base score
            issues = []
            
            if is_new_domain:
                credibility_score -= 20
                issues.append('New domain (less than 6 months old)')
            
            if not has_ssl:
                credibility_score -= 10
                issues.append('No SSL certificate')
            
            if is_suspicious_tld:
                credibility_score -= 15
                issues.append('Suspicious domain extension')
            
            # Try to get page info
            try:
                response = requests.get(url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code != 200:
                    credibility_score -= 10
                    issues.append(f'HTTP error: {response.status_code}')
                
                # Check for contact/about pages
                soup = BeautifulSoup(response.text, 'html.parser')
                has_about = bool(soup.find('a', href=re.compile('about', re.I)))
                has_contact = bool(soup.find('a', href=re.compile('contact', re.I)))
                
                if not has_about:
                    credibility_score -= 5
                    issues.append('No about page found')
                
                if not has_contact:
                    credibility_score -= 5
                    issues.append('No contact page found')
                    
            except:
                credibility_score -= 15
                issues.append('Could not access website')
            
            return {
                'credibility_score': max(0, min(100, credibility_score)),
                'domain': full_domain,
                'domain_age_days': domain_age,
                'has_ssl': has_ssl,
                'is_new_domain': is_new_domain,
                'is_suspicious_tld': is_suspicious_tld,
                'issues': issues,
                'warning': 'CAUTION' if credibility_score < 50 else 'VERIFY' if credibility_score < 70 else 'OK'
            }
            
        except Exception as e:
            return {
                'credibility_score': 30,
                'domain': url,
                'issues': [f'Analysis error: {str(e)}'],
                'warning': 'ERROR'
            }
    
    def detect_fake_news_advanced(self, text, url=None, analyze_sources=True):
        """Advanced fake news detection with ensemble methods"""
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{url}".encode()).hexdigest()
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Preprocess
        clean_text = self.preprocess_text_advanced(text)
        
        # Extract features
        text_features = self.extract_advanced_features(clean_text)
        
        # Get transformer model prediction
        transformer_score = 0.5
        if self.text_model and len(clean_text) > 50:
            try:
                # Split long text
                chunks = [clean_text[i:i+512] for i in range(0, len(clean_text), 512)]
                chunk_scores = []
                
                for chunk in chunks[:3]:  # Max 3 chunks
                    result = self.text_model(chunk)[0]
                    if result['label'] == 'FAKE' or result['label'] == 'NEGATIVE':
                        chunk_scores.append(result['score'])
                    else:
                        chunk_scores.append(1 - result['score'])
                
                transformer_score = np.mean(chunk_scores)
            except:
                pass
        
        # Get ML model predictions
        ml_score = 0.5
        gb_score = 0.5
        
        if self.classifier and hasattr(self.classifier, 'classes_'):
            try:
                # Vectorize text
                text_vector = self.tfidf_vectorizer.transform([clean_text])
                
                # Random Forest prediction
                rf_proba = self.classifier.predict_proba(text_vector)[0]
                ml_score = rf_proba[1] if len(rf_proba) > 1 else rf_proba[0]
                
                # Gradient Boosting prediction
                if self.gb_classifier and hasattr(self.gb_classifier, 'classes_'):
                    gb_proba = self.gb_classifier.predict_proba(text_vector)[0]
                    gb_score = gb_proba[1] if len(gb_proba) > 1 else gb_proba[0]
            except:
                pass
        
        # Source credibility analysis
        source_score = 50
        source_analysis = None
        if url and analyze_sources:
            source_analysis = self.analyze_source_credibility_advanced(url)
            source_score = source_analysis['credibility_score']
        
        # Combine scores with weights
        final_score = (
            transformer_score * 0.25 +
            ml_score * 0.20 +
            gb_score * 0.15 +
            (source_score / 100) * 0.25 +
            self.analyze_text_style_risk(text_features) * 0.15
        ) * 100
        
        # Determine risk level
        risk_level = 'LOW'
        if final_score > 70:
            risk_level = 'HIGH'
        elif final_score > 40:
            risk_level = 'MEDIUM'
        
        # Prepare result
        result = {
            'is_fake_news': final_score > 60,
            'confidence': final_score,
            'risk_level': risk_level,
            'text_features': text_features,
            'source_analysis': source_analysis,
            'model_scores': {
                'transformer': transformer_score * 100,
                'random_forest': ml_score * 100,
                'gradient_boosting': gb_score * 100,
                'source_credibility': source_score
            },
            'clickbait_score': text_features['clickbait_score'] * 100,
            'sensationalism_score': (
                text_features['exclamation_ratio'] * 100 +
                text_features['caps_ratio'] * 50
            ) / 2,
            'message': self.generate_feedback_message(final_score, text_features, source_analysis),
            'recommendations': self.generate_recommendations(final_score, text_features, source_analysis)
        }
        
        # Cache result
        self.analysis_cache[cache_key] = result
        
        return result
    
    def analyze_text_style_risk(self, features):
        """Analyze text style for fake news indicators"""
        risk_score = 0
        weights = {
            'exclamation_ratio': 0.2,
            'caps_ratio': 0.15,
            'subjectivity': 0.15,
            'clickbait_score': 0.3,
            'sentiment_extremeness': 0.2
        }
        
        # Extremely positive or negative sentiment
        sentiment_extremeness = abs(features.get('sentiment_compound', 0))
        
        # Calculate weighted score
        risk_score += features.get('exclamation_ratio', 0) * weights['exclamation_ratio']
        risk_score += features.get('caps_ratio', 0) * weights['caps_ratio']
        risk_score += features.get('subjectivity', 0.5) * weights['subjectivity']
        risk_score += features.get('clickbait_score', 0) * weights['clickbait_score']
        risk_score += sentiment_extremeness * weights['sentiment_extremeness']
        
        return min(risk_score, 1.0)
    
    def generate_feedback_message(self, score, features, source_analysis):
        """Generate user-friendly feedback message"""
        if score > 80:
            return "⚠️ HIGH RISK: This content shows strong indicators of being fake or misleading news."
        elif score > 60:
            return "⚠️ MEDIUM RISK: Multiple fake news indicators detected. Please verify with reliable sources."
        elif score > 40:
            return "ℹ️ LOW RISK: Some suspicious elements found, but content may be legitimate."
        else:
            return "✅ LOW RISK: Content appears to be credible based on our analysis."
    
    def generate_recommendations(self, score, features, source_analysis):
        """Generate specific recommendations"""
        recommendations = []
        
        if score > 60:
            recommendations.append("• Verify this information with multiple reliable news sources")
            recommendations.append("• Check if mainstream news outlets are reporting the same story")
            
        if features.get('clickbait_score', 0) > 0.3:
            recommendations.append("• Be cautious of sensational headlines designed to attract clicks")
            
        if features.get('exclamation_ratio', 0) > 0.05:
            recommendations.append("• Excessive exclamation marks often indicate misleading content")
            
        if source_analysis:
            if source_analysis.get('warning') == 'HIGH_RISK':
                recommendations.append("• This source is known for publishing fake news")
            elif source_analysis.get('is_new_domain'):
                recommendations.append("• This website was created recently - verify its credibility")
            elif source_analysis.get('issues'):
                recommendations.append(f"• Source issues: {', '.join(source_analysis['issues'][:2])}")
        
        if features.get('sentiment_extremeness', 0) > 0.8:
            recommendations.append("• Extreme emotional language may indicate bias or manipulation")
        
        return recommendations if recommendations else ["• Content appears normal - still verify with trusted sources"]
    
    def get_live_news(self, source='all', limit=10):
        """Fetch live news from RSS feeds"""
        all_news = []
        
        sources_to_check = self.news_sources.items() if source == 'all' else [(source, self.news_sources.get(source))]
        
        for src_name, feed_url in sources_to_check:
            if not feed_url:
                continue
                
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:limit]:
                    news_item = {
                        'source': src_name,
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'analysis': None
                    }
                    
                    # Analyze if we have enough text
                    if news_item['title'] and news_item['summary']:
                        text = f"{news_item['title']} {news_item['summary']}"
                        news_item['analysis'] = self.detect_fake_news_advanced(
                            text, 
                            news_item['link'],
                            analyze_sources=False  # Skip source analysis for live news
                        )
                    
                    all_news.append(news_item)
                    
            except Exception as e:
                print(f"Error fetching {src_name}: {e}")
        
        return all_news
    
    def train_advanced(self, texts, labels):
        """Advanced training with cross-validation"""
        # Preprocess texts
        clean_texts = [self.preprocess_text_advanced(text) for text in texts]
        
        # Vectorize texts
        X_tfidf = self.tfidf_vectorizer.fit_transform(clean_texts)
        X_count = self.count_vectorizer.fit_transform(clean_texts)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_count])
        
        # Train Random Forest
        self.classifier.fit(X_combined, labels)
        
        # Train Gradient Boosting
        self.gb_classifier.fit(X_combined, labels)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.classifier, X_combined, labels, cv=5)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.classifier, 'models/fake_news_rf.pkl')
        joblib.dump(self.gb_classifier, 'models/fake_news_gb.pkl')
        joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.count_vectorizer, 'models/count_vectorizer.pkl')
        
        return {
            'accuracy': self.classifier.score(X_combined, labels),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

# Create singleton instance
fake_news_detector = AdvancedFakeNewsDetector()