"""
Aspect-Based Sentiment Analysis
Analyzes sentiment for specific aspects in text
"""

import re
import logging
from typing import Dict, List, Tuple
from sentiment_analyzer import sentiment_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AspectSentimentAnalyzer:
    """
    Analyze sentiment for specific aspects like product quality, price, service, etc.
    """
    
    def __init__(self):
        """Initialize aspect analyzer"""
        self.aspect_keywords = {
            'quality': ['quality', 'build', 'material', 'durable', 'sturdy', 
                       'well-made', 'craftsmanship', 'construction'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 
                     'value', 'worth', 'overpriced', 'budget'],
            'service': ['service', 'support', 'customer service', 'help', 
                       'assistance', 'response', 'staff', 'representative'],
            'delivery': ['delivery', 'shipping', 'arrived', 'package', 
                        'shipped', 'shipping time', 'delivery time'],
            'usability': ['easy', 'difficult', 'interface', 'user-friendly', 
                         'intuitive', 'complicated', 'setup', 'installation'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient',
                           'battery', 'power', 'response time'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful',
                      'ugly', 'aesthetic', 'color', 'finish']
        }
        
    def analyze_aspects(self, text: str) -> Dict[str, Dict]:
        """
        Analyze sentiment for each aspect
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with aspect sentiments
        """
        text_lower = text.lower()
        aspects_found = {}
        
        for aspect, keywords in self.aspect_keywords.items():
            # Check if aspect is mentioned
            mentioned = any(keyword in text_lower for keyword in keywords)
            
            if mentioned:
                # Extract sentences containing aspect keywords
                sentences = self._extract_aspect_sentences(text, keywords)
                
                if sentences:
                    # Analyze sentiment of these sentences
                    aspect_sentiment = self._analyze_sentences(sentences)
                    aspects_found[aspect] = aspect_sentiment
                else:
                    # Analyze full text as fallback
                    label, conf = sentiment_analyzer.analyze(text)
                    aspects_found[aspect] = {
                        'label': label,
                        'confidence': conf,
                        'sentences': []
                    }
        
        return aspects_found
    
    def _extract_aspect_sentences(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extract sentences containing aspect keywords
        
        Args:
            text: Input text
            keywords: Aspect keywords
            
        Returns:
            List of relevant sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        relevant = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant.append(sentence.strip())
        
        return relevant
    
    def _analyze_sentences(self, sentences: List[str]) -> Dict:
        """
        Analyze sentiment of sentences
        
        Args:
            sentences: List of sentences
            
        Returns:
            Aggregated sentiment
        """
        if not sentences:
            return {'label': 'NEUTRAL', 'confidence': 0.5, 'sentences': []}
        
        results = []
        for sentence in sentences:
            label, conf = sentiment_analyzer.analyze(sentence)
            results.append({
                'sentence': sentence,
                'label': label,
                'confidence': conf
            })
        
        # Aggregate
        labels = [r['label'] for r in results]
        from collections import Counter
        most_common = Counter(labels).most_common(1)[0][0]
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        return {
            'label': most_common,
            'confidence': avg_confidence,
            'sentences': results
        }