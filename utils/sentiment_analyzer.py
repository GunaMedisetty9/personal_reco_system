"""
Sentiment Analysis for Reviews (3-class)
0 = negative, 1 = neutral, 2 = positive
"""

import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class SentimentAnalyzer:
    MODEL_VERSION = 3  # increment to invalidate old pickles

    def __init__(self):
        self.model = None
        self.labels = {0: "negative", 1: "neutral", 2: "positive"}

        self.positive_words = {
            "effective", "works", "helped", "excellent", "great", "good",
            "recommend", "relief", "improved", "better", "amazing", "wonderful",
            "perfect", "cured", "healed", "recovered", "satisfied", "happy",
        }
        self.negative_words = {
            "bad", "terrible", "horrible", "awful", "worse", "useless",
            "ineffective", "painful", "nausea", "headache",
            "allergic", "dangerous", "harmful", "waste", "disappointed",
        }

    def train(self) -> None:
        positive_texts = [
            "good", "great", "excellent",
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
        ]

        negative_texts = [
            "bad", "terrible", "awful",
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
        ]

        neutral_texts = [
            "ok", "fine",
            "Average results nothing special",
            "It was okay I guess",
            "Not sure if it helped",
            "Some improvement",
            "Takes time to work",
            "Standard medication",
            "As expected",
            "Moderate effectiveness",
        ]

        # Build (text, label) pairs => cannot mismatch
        samples = []
        samples += [(t, 2) for t in positive_texts]
        samples += [(t, 0) for t in negative_texts]
        samples += [(t, 1) for t in neutral_texts]

        training_texts = [t for (t, y) in samples]
        training_labels = [y for (t, y) in samples]

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ])
        self.model.fit(training_texts, training_labels)

    def analyze(self, text: str) -> dict:
        if self.model is None:
            self.train()

        cleaned = self._preprocess(text)
        pred = int(self.model.predict([cleaned])[0])  # 0/1/2
        probs = self.model.predict_proba([cleaned])[0]  # order: [0,1,2]

        keyword_score = self._keyword_sentiment(cleaned)

        # pred 0,1,2 -> -1,0,+1
        ml_score = pred - 1
        combined = (ml_score * 0.7 + keyword_score * 0.3)

        if combined > 0.2:
            sentiment = "positive"
        elif combined < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": round(float(max(probs)), 3),
            "score": round(float(combined), 3),
            "probabilities": {
                "negative": round(float(probs[0]), 3),
                "neutral": round(float(probs[1]), 3),
                "positive": round(float(probs[2]), 3),
            },
        }

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return " ".join(text.split())

    def _keyword_sentiment(self, text: str) -> float:
        words = set(text.split())
        pos = len(words.intersection(self.positive_words))
        neg = len(words.intersection(self.negative_words))
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {"model": self.model, "labels": self.labels, "version": self.MODEL_VERSION},
            path,
        )
