"""
Disease Prediction Page
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.disease_predictor import DiseasePredictionModel
from utils.medicine_recommender import MedicineRecommender

st.set_page_config(page_title="Disease Prediction", page_icon="🔍", layout="wide")

st.title("🔍 Disease Prediction")
st.markdown("Enter your symptoms and vitals to get AI-powered health insights")

# Load models
@st.cache_resource
def load_models():
    disease_model = DiseasePredictionModel()
    disease_model.load('models/disease_model.pkl')
    if not disease_model.is_trained:
        disease_model.train()
    
    medicine_model = MedicineRecommender()
    medicine_model.load('models/medicine_model.pkl')
    if medicine_model.medicines_df is None:
        medicine_model.load_medicines()
    
    return disease_model, medicine_model

disease_model, medicine_model = load_models()

st.markdown("---")

# Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    
    st.subheader("💉 Vital Signs")
    bp_systolic = st.number_input("Blood Pressure (Systolic)", min_value=60, max_value=250, value=120)
    bp_diastolic = st.number_input("Blood Pressure (Diastolic)", min_value=40, max_value=150, value=80)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=20.0, max_value=600.0, value=100.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=72)
    temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)

with col2:
    st.subheader("🩺 Symptoms")
    st.caption("Check all symptoms that apply")
    
    symptoms = {}
    symptom_cols = st.columns(2)
    
    symptom_list = [
        ('fever', 'Fever'), ('cough', 'Cough'), ('fatigue', 'Fatigue'),
        ('headache', 'Headache'), ('chest_pain', 'Chest Pain'),
        ('shortness_of_breath', 'Shortness of Breath'), ('nausea', 'Nausea'),
        ('dizziness', 'Dizziness'), ('joint_pain', 'Joint Pain'),
        ('skin_rash', 'Skin Rash'), ('weight_loss', 'Weight Loss'),
        ('excessive_thirst', 'Excessive Thirst'), ('frequent_urination', 'Frequent Urination'),
        ('blurred_vision', 'Blurred Vision'), ('numbness', 'Numbness'),
        ('muscle_weakness', 'Muscle Weakness')
    ]
    
    for i, (key, label) in enumerate(symptom_list):
        with symptom_cols[i % 2]:
            symptoms[key] = 1 if st.checkbox(label) else 0

st.markdown("---")

# Predict button
if st.button("🔮 Get Diagnosis", type="primary", use_container_width=True):
    
    # Prepare input data
    patient_data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'blood_pressure_systolic': bp_systolic,
        'blood_pressure_diastolic': bp_diastolic,
        'glucose_level': glucose,
        'heart_rate': heart_rate,
        'temperature': temperature,
        'bmi': bmi,
        'cholesterol': 180,
        'hemoglobin': 14,
        **symptoms
    }
    
    with st.spinner("Analyzing your symptoms..."):
        result = disease_model.predict(patient_data)
    
    st.markdown("---")
    
    # Display Results
    st.subheader("📋 Diagnosis Results")
    
    # Primary Diagnosis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['primary_diagnosis'] == 'Healthy':
            st.success(f"### ✅ {result['primary_diagnosis']}")
        else:
            st.warning(f"### ⚠️ {result['primary_diagnosis']}")
        
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    
    with col2:
        st.markdown("**Other Possibilities:**")
        for pred in result['all_predictions'][1:4]:
            st.write(f"• {pred['disease']}: {pred['probability']:.1f}%")
    
    # Risk Factors
    if result['risk_factors']:
        st.markdown("### ⚠️ Risk Factors Identified")
        for risk in result['risk_factors']:
            st.warning(risk)
    
    # Medicine Recommendations
    if result['primary_diagnosis'] != 'Healthy':
        st.markdown("### 💊 Recommended Medicines")
        
        medicines = medicine_model.recommend_by_disease(
            result['primary_diagnosis'],
            {'age': age, 'allergies': [], 'medical_conditions': []}
        )
        
        for med in medicines:
            with st.expander(f"💊 {med['medicine_name']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Dosage:** {med['dosage']}")
                col2.write(f"**Frequency:** {med['frequency']}")
                col3.write(f"**Confidence:** {med['confidence']*100:.0f}%")
                
                if med.get('notes'):
                    st.info(f"📝 {med['notes']}")
                
                if med['warnings']:
                    for warn in med['warnings']:
                        st.warning(warn)
    
    # Disclaimer
    st.markdown("---")
    st.info("""
    ⚠️ **Disclaimer:** This is an AI-based prediction for educational purposes only. 
    Please consult a healthcare professional for proper medical advice and treatment.
    """)