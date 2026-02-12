"""
Medicine Recommendation Page
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.medicine_recommender import MedicineRecommender

st.set_page_config(page_title="Medicine Recommendations", page_icon="💊", layout="wide")

st.title("💊 Medicine Recommendations")
st.markdown("Find medicines and personalized recommendations")

# Load model
@st.cache_resource
def load_model():
    model = MedicineRecommender()
    model.load('models/medicine_model.pkl')
    if model.medicines_df is None:
        model.load_medicines()
    return model

medicine_model = load_model()

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🎯 By Condition", "🔍 Find Similar"])

with tab1:
    st.subheader("Get Recommendations by Condition")
    
    conditions = list(medicine_model.disease_medicine_map.keys())
    selected_condition = st.selectbox("Select Condition", conditions)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Your Age", min_value=1, max_value=120, value=35, key="age1")
    with col2:
        allergies = st.text_input("Any Allergies? (comma-separated)", "")
    
    if st.button("Get Recommendations", type="primary"):
        patient_data = {
            'age': age,
            'allergies': [a.strip() for a in allergies.split(',') if a.strip()],
            'medical_conditions': []
        }
        
        recommendations = medicine_model.recommend_by_disease(selected_condition, patient_data)
        
        st.markdown(f"### 💊 Recommendations for {selected_condition}")
        
        for i, med in enumerate(recommendations, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{i}. {med['medicine_name']}**")
                    st.write(f"📋 Dosage: {med['dosage']} | ⏰ {med['frequency']}")
                    if med.get('notes'):
                        st.caption(f"💡 {med['notes']}")
                
                with col2:
                    confidence_color = "🟢" if med['confidence'] > 0.7 else "🟡"
                    st.metric("Confidence", f"{confidence_color} {med['confidence']*100:.0f}%")
                
                if med['warnings']:
                    for warn in med['warnings']:
                        st.warning(warn)
                
                st.markdown("---")

with tab2:
    st.subheader("Find Similar Medicines")
    
    medicine_name = st.text_input("Enter Medicine Name", placeholder="e.g., Aspirin, Metformin")
    
    if medicine_name and st.button("Find Similar", type="primary"):
        similar = medicine_model.find_similar_medicines(medicine_name)
        
        if similar:
            st.markdown(f"### Medicines similar to {medicine_name}")
            
            for med in similar:
                col1, col2, col3 = st.columns([2, 2, 1])
                col1.write(f"**{med['name']}**")
                col2.write(f"Category: {med['category']}")
                col3.metric("Similarity", f"{med['similarity']*100:.0f}%")
        else:
            st.warning("No similar medicines found. Try a different name.")
    
    # Medicine Categories
    st.markdown("---")
    st.subheader("📚 Browse by Category")
    
    categories = medicine_model.medicines_df['category'].unique() if medicine_model.medicines_df is not None else []
    
    cols = st.columns(3)
    for i, cat in enumerate(categories):
        with cols[i % 3]:
            with st.expander(f"🏷️ {cat}"):
                meds = medicine_model.medicines_df[medicine_model.medicines_df['category'] == cat]
                for _, med in meds.iterrows():
                    st.write(f"• {med['name']}")