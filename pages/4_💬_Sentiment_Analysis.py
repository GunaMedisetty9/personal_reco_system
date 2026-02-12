"""
Sentiment Analysis for Reviews
"""

import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class SentimentAnalyzer:
    """Analyze sentiment of medicine/treatment reviews."""

    MODEL_VERSION = 2  # bump this when you change labels/training logic

    def __init__(self):
        self.model = None

        # IMPORTANT: keep this order consistent with ml_score = prediction - 1
        # 0 -> negative, 1 -> neutral, 2 -> positive
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
        """Train sentiment model."""
        training_texts = [
            # Positive
            "good",
            "great",
            "excellent",
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

            # Negative
            "bad",
            "terrible",
            "awful experience",
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

            # Neutral
            "ok",
            "fine",
            "Average results nothing special",
            "It was okay I guess",
            "Not sure if it helped",
            "Some improvement",
            "Takes time to work",
            "Standard medication",
            "As expected",
            "Moderate effectiveness",
        ]

        # Label mapping:
        # 0=negative, 1=neutral, 2=positive
        training_labels = (
            [2] * 14 +   # positive samples count above
            [0] * 15 +   # negative samples count above
            [1] * 10     # neutral samples count above
        )

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ])
        self.model.fit(training_texts, training_labels)

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text."""
        if self.model is None:
            self.train()

        cleaned = self._preprocess(text)
        prediction = int(self.model.predict([cleaned])[0])
        probabilities = self.model.predict_proba([cleaned])[0]  # order matches classes [0,1,2]

        keyword_score = self._keyword_sentiment(cleaned)

        # Works ONLY if prediction encoding is: 0 neg, 1 neutral, 2 pos
        ml_score = (prediction - 1)  # -> -1, 0, +1
        combined = (ml_score * 0.7 + keyword_score * 0.3)

        if combined > 0.2:
            sentiment = "positive"
        elif combined < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": round(float(max(probabilities)), 3),
            "score": round(float(combined), 3),
            "probabilities": {
                "negative": round(float(probabilities[0]), 3),
                "neutral": round(float(probabilities[1]), 3),
                "positive": round(float(probabilities[2]), 3),
            },
        }

    def analyze_batch(self, reviews: list) -> dict:
        """Analyze multiple reviews."""
        results = []
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_score = 0.0

        for review in reviews:
            result = self.analyze(review)
            results.append({
                "text": (review[:100] + "...") if len(review) > 100 else review,
                **result,
            })
            counts[result["sentiment"]] += 1
            total_score += float(result["score"])

        n = len(reviews)
        avg_score = total_score / n if n > 0 else 0.0

        return {
            "results": results,
            "summary": {
                "total": n,
                "distribution": {
                    k: {"count": v, "percent": round((v / n * 100), 1) if n else 0.0}
                    for k, v in counts.items()
                },
                "average_score": round(float(avg_score), 3),
                "overall": "positive" if avg_score > 0.2 else "negative" if avg_score < -0.2 else "neutral",
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
        """Save model."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {"model": self.model, "labels": self.labels, "version": self.MODEL_VERSION},
            path
        )

    def load(self, path: str) -> bool:
        """Load model. Returns False if incompatible/old pickle."""
        if not os.path.exists(path):
            return False
        try:
            data = joblib.load(path)

            # Force retrain if old file (no version) or wrong version
            if data.get("version") != self.MODEL_VERSION:
                self.model = None
                return False

            self.model = data.get("model")
            self.labels = data.get("labels", self.labels)
            return self.model is not None
        except Exception:
            return False
