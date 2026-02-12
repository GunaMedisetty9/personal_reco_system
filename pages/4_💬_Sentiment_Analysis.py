import os, sys
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
st.title("💬 Sentiment Analysis")

os.makedirs("models", exist_ok=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import plotly.express as px
    import pandas as pd
    from utils.sentiment_analyzer import SentimentAnalyzer
except Exception as e:
    st.error("Import error:")
    st.exception(e)
    st.stop()

@st.cache_resource
def get_model():
    m = SentimentAnalyzer()
    ok = False
    try:
        ok = m.load("models/sentiment_model.pkl")
    except Exception:
        ok = False
    if (not ok) or (m.model is None):
        m.train()
        m.save("models/sentiment_model.pkl")
    return m

try:
    model = get_model()
except Exception as e:
    st.error("Model train/load error:")
    st.exception(e)
    st.stop()

text = st.text_area("Enter text", "good")
if st.button("Analyze"):
    r = model.analyze(text)
    st.write(r)
    df = pd.DataFrame(
        {"label": ["negative","neutral","positive"],
         "p": [r["probabilities"]["negative"], r["probabilities"]["neutral"], r["probabilities"]["positive"]]}
    )
    st.plotly_chart(px.bar(df, x="label", y="p"), use_container_width=True)
# """
# Sentiment Analysis for Reviews
# """

# import os
# import re
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline


# class SentimentAnalyzer:
#     MODEL_VERSION = 2  # bump when you change label mapping/training

#     def __init__(self):
#         self.model = None
#         # Must match prediction encoding used in scoring: 0 neg, 1 neutral, 2 pos
#         self.labels = {0: "negative", 1: "neutral", 2: "positive"}

#         self.positive_words = {
#             "effective", "works", "helped", "excellent", "great", "good",
#             "recommend", "relief", "improved", "better", "amazing", "wonderful",
#             "perfect", "cured", "healed", "recovered", "satisfied", "happy",
#         }

#         self.negative_words = {
#             "bad", "terrible", "horrible", "awful", "worse", "useless",
#             "ineffective", "painful", "nausea", "headache",
#             "allergic", "dangerous", "harmful", "waste", "disappointed",
#         }

#     def train(self) -> None:
#         """Train sentiment model."""
#         positive_texts = [
#             "good", "great", "excellent",
#             "This medicine worked great for me!",
#             "Excellent results highly recommend",
#             "Very effective treatment",
#             "Doctor was helpful and caring",
#             "Quick recovery thanks to this",
#             "I feel so much better now",
#             "Best medication ever",
#             "Completely cured my symptoms",
#             "Amazing results",
#             "No side effects at all",
#             "Affordable and effective",
#             "Would use again",
#         ]

#         negative_texts = [
#             "bad", "terrible", "awful",
#             "Terrible side effects",
#             "Did not work at all",
#             "Waste of money",
#             "Made my condition worse",
#             "Poor experience",
#             "Caused severe nausea",
#             "Allergic reaction",
#             "Would not recommend",
#             "Completely useless",
#             "Expensive and ineffective",
#             "Dangerous medication",
#             "Horrible experience",
#         ]

#         neutral_texts = [
#             "ok", "fine",
#             "Average results nothing special",
#             "It was okay I guess",
#             "Not sure if it helped",
#             "Some improvement",
#             "Takes time to work",
#             "Standard medication",
#             "As expected",
#             "Moderate effectiveness",
#         ]

#         training_texts = positive_texts + negative_texts + neutral_texts
#         training_labels = (
#             [2] * len(positive_texts) +
#             [0] * len(negative_texts) +
#             [1] * len(neutral_texts)
#         )

#         # Safety check (prevents the exact error you got)
#         assert len(training_texts) == len(training_labels), (len(training_texts), len(training_labels))

#         self.model = Pipeline([
#             ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
#             ("clf", MultinomialNB()),
#         ])
#         self.model.fit(training_texts, training_labels)

#     def analyze(self, text: str) -> dict:
#         if self.model is None:
#             self.train()

#         cleaned = self._preprocess(text)
#         prediction = int(self.model.predict([cleaned])[0])   # 0,1,2
#         probabilities = self.model.predict_proba([cleaned])[0]

#         keyword_score = self._keyword_sentiment(cleaned)
#         ml_score = prediction - 1  # 0,1,2 -> -1,0,+1
#         combined = (ml_score * 0.7 + keyword_score * 0.3)

#         if combined > 0.2:
#             sentiment = "positive"
#         elif combined < -0.2:
#             sentiment = "negative"
#         else:
#             sentiment = "neutral"

#         return {
#             "sentiment": sentiment,
#             "confidence": round(float(max(probabilities)), 3),
#             "score": round(float(combined), 3),
#             "probabilities": {
#                 "negative": round(float(probabilities[0]), 3),
#                 "neutral": round(float(probabilities[1]), 3),
#                 "positive": round(float(probabilities[2]), 3),
#             },
#         }

#     def _preprocess(self, text: str) -> str:
#         text = text.lower()
#         text = re.sub(r"[^a-zA-Z\s]", "", text)
#         return " ".join(text.split())

#     def _keyword_sentiment(self, text: str) -> float:
#         words = set(text.split())
#         pos = len(words.intersection(self.positive_words))
#         neg = len(words.intersection(self.negative_words))
#         total = pos + neg
#         return (pos - neg) / total if total > 0 else 0.0

#     def save(self, path: str) -> None:
#         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#         joblib.dump(
#             {"model": self.model, "labels": self.labels, "version": self.MODEL_VERSION},
#             path
#         )

#     def load(self, path: str) -> bool:
#         if not os.path.exists(path):
#             return False
#         try:
#             data = joblib.load(path)
#             if data.get("version") != self.MODEL_VERSION:
#                 self.model = None
#                 return False
#             self.model = data.get("model")
#             self.labels = data.get("labels", self.labels)
#             return self.model is not None
#         except Exception:
#             return False
