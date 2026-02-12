"""
Sentiment Analysis for Reviews
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os


class SentimentAnalyzer:
    """Analyze sentiment of medicine/treatment reviews."""
    
    def __init__(self):
        self.model = None
        self.labels = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        self.positive_words = {
            'effective', 'works', 'helped', 'excellent', 'great', 'good',
            'recommend', 'relief', 'improved', 'better', 'amazing', 'wonderful',
            'perfect', 'cured', 'healed', 'recovered', 'satisfied', 'happy'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'horrible', 'awful', 'worse', 'useless',
            'ineffective', 'painful', 'side effects', 'nausea', 'headache',
            'allergic', 'dangerous', 'harmful', 'waste', 'disappointed'
        }
    
    def train(self):
        """Train sentiment model."""
        training_texts = [
            "This medicine worked great for me!",
            "Excellent results highly recommend",
            "Very effective treatment",
            "Doctor was helpful and caring",
            "Quick recovery thanks to this",
            "I feel so much better now",
            "Best medication ever",
            "Completely cured my symptoms",
            "Amazing results",
            "No side effects at all",
            "Affordable and effective",
            "Would use again",
            "Terrible side effects",
            "Did not work at all",
            "Waste of money",
            "Made my condition worse",
            "Poor experience",
            "Caused severe nausea",
            "Allergic reaction",
            "Would not recommend",
            "Completely useless",
            "Expensive and ineffective",
            "Dangerous medication",
            "Horrible experience",
            "Average results nothing special",
            "It was okay I guess",
            "Not sure if it helped",
            "Some improvement",
            "Takes time to work",
            "Standard medication",
            "As expected",
            "Moderate effectiveness",
        ]
        
        training_labels = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Positive
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Negative
            2, 2, 2, 2, 2, 2, 2, 2               # Neutral
        ]
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        
        self.model.fit(training_texts, training_labels)
    
    def analyze(self, text: str):
        """Analyze sentiment of text."""
        if self.model is None:
            self.train()
        
        cleaned = self._preprocess(text)
        
        prediction = self.model.predict([cleaned])[0]
        probabilities = self.model.predict_proba([cleaned])[0]
        
        # Keyword analysis
        keyword_score = self._keyword_sentiment(cleaned)
        
        # Combine scores
        ml_score = (prediction - 1)  # Convert 0,1,2 to -1,0,1
        combined = (ml_score * 0.7 + keyword_score * 0.3)
        
        if combined > 0.2:
            sentiment = 'positive'
        elif combined < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': round(max(probabilities), 3),
            'score': round(combined, 3),
            'probabilities': {
                'negative': round(probabilities[0], 3),
                'positive': round(probabilities[1], 3),
                'neutral': round(probabilities[2], 3)
            }
        }
    
    def analyze_batch(self, reviews: list):
        """Analyze multiple reviews."""
        results = []
        counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_score = 0
        
        for review in reviews:
            result = self.analyze(review)
            results.append({
                'text': review[:100] + '...' if len(review) > 100 else review,
                **result
            })
            counts[result['sentiment']] += 1
            total_score += result['score']
        
        n = len(reviews)
        avg_score = total_score / n if n > 0 else 0
        
        return {
            'results': results,
            'summary': {
                'total': n,
                'distribution': {k: {'count': v, 'percent': round(v/n*100, 1)} for k, v in counts.items()},
                'average_score': round(avg_score, 3),
                'overall': 'positive' if avg_score > 0.2 else 'negative' if avg_score < -0.2 else 'neutral'
            }
        }
    
    def _preprocess(self, text: str):
        """Clean text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
    def _keyword_sentiment(self, text: str):
        """Keyword-based sentiment."""
        words = set(text.split())
        pos = len(words.intersection(self.positive_words))
        neg = len(words.intersection(self.negative_words))
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0
    
    def save(self, path: str):
        """Save model."""
        joblib.dump({'model': self.model, 'labels': self.labels}, path)
    
    def load(self, path: str):
        """Load model."""
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.labels = data['labels']
            return True
        return False