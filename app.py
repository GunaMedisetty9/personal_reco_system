"""
Healthcare Recommendation System - Main Streamlit App
"""

import streamlit as st
import os

st.set_page_config(
    page_title="Healthcare Recommendation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Healthcare Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Personalized Health Recommendations</p>', unsafe_allow_html=True)
    
    # Check if models exist
    models_exist = os.path.exists('models/disease_model.pkl')
    
    if not models_exist:
        st.warning("⚠️ Models not trained yet!")
        st.info("Run `py train_models.py` to train the models first.")
        
        if st.button("🔧 Train Models Now"):
            with st.spinner("Training models... This may take a minute..."):
                import train_models
                train_models.main()
            st.success("✅ Models trained successfully! Please refresh the page.")
            st.rerun()
        return
    
    st.success("✅ All models loaded and ready!")
    
    # Dashboard Overview
    st.markdown("---")
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍</h3>
            <h4>Disease Prediction</h4>
            <p>AI-powered diagnosis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💊</h3>
            <h4>Medicine Recommendation</h4>
            <p>Personalized suggestions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚠️</h3>
            <h4>Drug Interactions</h4>
            <p>Safety checks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h3>💬</h3>
            <h4>Sentiment Analysis</h4>
            <p>Review insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Diseases Covered", "10+", "ML Model")
    with col2:
        st.metric("Medicines Database", "40+", "Content-Based")
    with col3:
        st.metric("Interactions Tracked", "30+", "Safety First")
    with col4:
        st.metric("ML Algorithms", "5+", "Hybrid System")
    
    st.markdown("---")
    
    # Navigation
    st.subheader("🚀 Quick Navigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.page_link("pages/1_🔍_Disease_Prediction.py", label="🔍 Disease Prediction", icon="🔍")
        st.caption("Enter symptoms and get AI-powered diagnosis")
        
        st.page_link("pages/2_💊_Medicine_Recommendation.py", label="💊 Medicine Recommendations", icon="💊")
        st.caption("Get personalized medicine suggestions")
    
    with col2:
        st.page_link("pages/3_⚠️_Drug_Interactions.py", label="⚠️ Drug Interactions", icon="⚠️")
        st.caption("Check medicine safety and interactions")
        
        st.page_link("pages/4_💬_Sentiment_Analysis.py", label="💬 Review Analysis", icon="💬")
        st.caption("Analyze medicine reviews and feedback")
    
    # About section
    st.markdown("---")
    st.subheader("ℹ️ About This System")
    
    st.markdown("""
    This **AI-Powered Healthcare Recommendation System** uses multiple Machine Learning algorithms:
    
    - **Content-Based Filtering**: TF-IDF vectorization with cosine similarity
    - **Collaborative Filtering**: User-item matrix factorization
    - **Hybrid Recommendations**: Combining multiple approaches
    - **Gradient Boosting**: For disease prediction
    - **NLP/Sentiment Analysis**: For review processing
    
    > ⚠️ **Disclaimer**: This is a demonstration system for educational purposes. 
    > Always consult healthcare professionals for medical advice.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        🏥 Healthcare Recommendation System | Built with Streamlit & Machine Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()