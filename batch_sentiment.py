"""
Batch Sentiment Processing
Process multiple texts efficiently
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sentiment_analyzer import sentiment_analyzer
from tqdm import tqdm

class BatchSentimentProcessor:
    """
    Process multiple texts for sentiment analysis
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of texts to process at once
        """
        self.batch_size = batch_size
    
    def process_file(self, filepath: str, text_column: str = None) -> pd.DataFrame:
        """
        Process file with multiple texts
        
        Args:
            filepath: Path to file (CSV or TXT)
            text_column: Column name for CSV files
            
        Returns:
            DataFrame with results
        """
        # Load file
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if text_column:
                texts = df[text_column].tolist()
            else:
                # Assume first column is text
                texts = df.iloc[:, 0].tolist()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        # Process in batches
        results = self.process_texts(texts)
        
        # Add original text if from CSV
        if filepath.endswith('.csv') and text_column:
            results['original_text'] = texts
        
        return results
    
    def process_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Process multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing"):
            batch = texts[i:i + self.batch_size]
            
            for text in batch:
                try:
                    label, conf = sentiment_analyzer.analyze(text)
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': label,
                        'confidence': conf,
                        'length': len(text)
                    })
                except Exception as e:
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': 'ERROR',
                        'confidence': 0.0,
                        'length': len(text),
                        'error': str(e)
                    })
        
        return pd.DataFrame(results)
    
    def get_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics from results
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if results_df.empty:
            return stats
        
        # Counts
        stats['total'] = len(results_df)
        stats['positive'] = len(results_df[results_df['sentiment'] == 'POSITIVE'])
        stats['negative'] = len(results_df[results_df['sentiment'] == 'NEGATIVE'])
        stats['neutral'] = len(results_df[results_df['sentiment'] == 'NEUTRAL'])
        stats['errors'] = len(results_df[results_df['sentiment'] == 'ERROR'])
        
        # Percentages
        stats['positive_pct'] = stats['positive'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['negative_pct'] = stats['negative'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['neutral_pct'] = stats['neutral'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        # Confidence
        valid_results = results_df[results_df['sentiment'] != 'ERROR']
        if not valid_results.empty:
            stats['avg_confidence'] = valid_results['confidence'].mean()
            stats['min_confidence'] = valid_results['confidence'].min()
            stats['max_confidence'] = valid_results['confidence'].max()
        else:
            stats['avg_confidence'] = 0
        
        # Text length statistics
        if 'length' in results_df.columns:
            stats['avg_length'] = results_df['length'].mean()
            stats['min_length'] = results_df['length'].min()
            stats['max_length'] = results_df['length'].max()
        
        return stats