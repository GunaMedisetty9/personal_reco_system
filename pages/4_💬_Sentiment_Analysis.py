import os, sys, traceback
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
st.title("💬 Sentiment Analysis")
st.caption("If anything fails, the error will show here.")

# Ensure folder exists
os.makedirs("models", exist_ok=True)

# Make root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Button to clear Streamlit cache (VERY useful when debugging)
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Reset cache"):
        st.cache_resource.clear()
        st.rerun()

try:
    import pandas as pd
    import plotly.express as px
    from utils.sentiment_analyzer import SentimentAnalyzer
except Exception as e:
    st.error("Import failed:")
    st.code(traceback.format_exc())
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
except Exception:
    st.error("Model train/load failed:")
    st.code(traceback.format_exc())
    st.stop()

text = st.text_area("Enter a medicine review:", "good", height=120)

if st.button("Analyze Sentiment", type="primary"):
    try:
        r = model.analyze(text)
        st.success(f"Sentiment: {r['sentiment'].upper()}  |  Confidence: {r['confidence']}")
        df = pd.DataFrame({
            "label": ["negative", "neutral", "positive"],
            "p": [r["probabilities"]["negative"], r["probabilities"]["neutral"], r["probabilities"]["positive"]],
        })
        st.plotly_chart(px.bar(df, x="label", y="p"), use_container_width=True)
        st.write(r)
    except Exception:
        st.error("Analyze failed:")
        st.code(traceback.format_exc())
