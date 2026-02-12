"""
Sentiment Analysis Page
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sentiment_analyzer import SentimentAnalyzer
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

st.title("💬 Medicine Review Sentiment Analysis")
st.markdown("Analyze the sentiment of medicine reviews and feedback")

# Load model
@st.cache_resource
def load_model():
    model = SentimentAnalyzer()
    model.load('models/sentiment_model.pkl')
    if model.model is None:
        model.train()
    return model

sentiment_model = load_model()

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📝 Single Review", "📊 Batch Analysis"])

with tab1:
    st.subheader("Analyze Single Review")
    
    review_text = st.text_area(
        "Enter a medicine review:",
        placeholder="e.g., This medicine worked great for my headaches. I noticed improvement within an hour!",
        height=150
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if review_text.strip():
            result = sentiment_model.analyze(review_text)
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_emoji = {
                    'positive': '😊',
                    'negative': '😞',
                    'neutral': '😐'
                }
                
                sentiment_color = {
                    'positive': 'success',
                    'negative': 'error',
                    'neutral': 'warning'
                }
                
                getattr(st, sentiment_color[result['sentiment']])(
                    f"### {sentiment_emoji[result['sentiment']]} {result['sentiment'].upper()}"
                )
                
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            with col2:
                st.markdown("**Probability Breakdown:**")
                
                probs = result['probabilities']
                
                # Create bar chart
                fig = px.bar(
                    x=['Negative', 'Positive', 'Neutral'],
                    y=[probs['negative'], probs['positive'], probs['neutral']],
                    color=['Negative', 'Positive', 'Neutral'],
                    color_discrete_map={
                        'Negative': '#FF6B6B',
                        'Positive': '#4CAF50',
                        'Neutral': '#FFC107'
                    }
                )
                fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a review to analyze.")

with tab2:
    st.subheader("Analyze Multiple Reviews")
    
    st.markdown("Enter multiple reviews (one per line):")
    
    sample_reviews = """This medicine worked great for me!
Terrible side effects, would not recommend.
It was okay, nothing special.
Best medication I've ever used for my condition.
Made my symptoms worse instead of better.
Average results, takes time to work."""
    
    reviews_text = st.text_area(
        "Reviews:",
        value=sample_reviews,
        height=200
    )
    
    if st.button("Analyze All Reviews", type="primary"):
        reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
        
        if reviews:
            with st.spinner("Analyzing reviews..."):
                result = sentiment_model.analyze_batch(reviews)
            
            # Summary
            st.markdown("### 📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            summary = result['summary']
            
            col1.metric("Total Reviews", summary['total'])
            col2.metric("Positive", f"{summary['distribution']['positive']['percent']:.1f}%")
            col3.metric("Negative", f"{summary['distribution']['negative']['percent']:.1f}%")
            col4.metric("Neutral", f"{summary['distribution']['neutral']['percent']:.1f}%")
            
            # Overall sentiment
            overall_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
            st.info(f"**Overall Sentiment:** {overall_emoji[summary['overall']]} {summary['overall'].upper()}")
            
            # Distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                dist_data = pd.DataFrame([
                    {'Sentiment': 'Positive', 'Count': summary['distribution']['positive']['count']},
                    {'Sentiment': 'Negative', 'Count': summary['distribution']['negative']['count']},
                    {'Sentiment': 'Neutral', 'Count': summary['distribution']['neutral']['count']}
                ])
                
                fig = px.pie(
                    dist_data, 
                    values='Count', 
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Negative': '#FF6B6B',
                        'Neutral': '#FFC107'
                    },
                    title='Sentiment Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Individual results
                st.markdown("**Individual Results:**")
                
                for r in result['results']:
                    emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                    st.write(f"{emoji[r['sentiment']]} **{r['sentiment']}** ({r['confidence']*100:.0f}%): {r['text']}")
        else:
            st.warning("Please enter at least one review.")