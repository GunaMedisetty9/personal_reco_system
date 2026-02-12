"""
Drug Interactions Checker Page
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.medicine_recommender import MedicineRecommender

st.set_page_config(page_title="Drug Interactions", page_icon="⚠️", layout="wide")

st.title("⚠️ Drug Interaction Checker")
st.markdown("Check for potential interactions between medications")

# Load model
@st.cache_resource
def load_model():
    model = MedicineRecommender()
    model.load_medicines()
    return model

medicine_model = load_model()

st.markdown("---")

# Medicine selection
st.subheader("Select Medicines to Check")

# Get medicine list
if medicine_model.medicines_df is not None:
    medicine_list = medicine_model.medicines_df['name'].unique().tolist()
else:
    medicine_list = ['Aspirin', 'Ibuprofen', 'Metformin', 'Lisinopril', 'Warfarin']

selected_medicines = st.multiselect(
    "Select medicines (at least 2)",
    options=medicine_list,
    default=[]
)

# Manual entry option
st.markdown("**Or enter manually:**")
manual_medicines = st.text_input("Enter medicine names (comma-separated)", "")

if manual_medicines:
    manual_list = [m.strip() for m in manual_medicines.split(',') if m.strip()]
    all_medicines = list(set(selected_medicines + manual_list))
else:
    all_medicines = selected_medicines

# Display selected medicines
if all_medicines:
    st.markdown("### Selected Medicines:")
    cols = st.columns(min(len(all_medicines), 4))
    for i, med in enumerate(all_medicines):
        with cols[i % 4]:
            st.info(f"💊 {med}")

st.markdown("---")

# Check interactions
if st.button("🔍 Check Interactions", type="primary", use_container_width=True):
    if len(all_medicines) < 2:
        st.warning("Please select at least 2 medicines to check for interactions.")
    else:
        interactions = medicine_model.check_interactions(all_medicines)
        
        if interactions:
            st.error(f"### ⚠️ Found {len(interactions)} Potential Interaction(s)")
            
            for interaction in interactions:
                severity_color = {
                    'Severe': '🔴',
                    'Moderate': '🟠',
                    'Low': '🟢'
                }.get(interaction['severity'], '🟡')
                
                with st.expander(f"{severity_color} {interaction['medicines'][0]} + {interaction['medicines'][1]}", expanded=True):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.metric("Severity", interaction['severity'])
                    
                    with col2:
                        st.write(f"**Description:** {interaction['description']}")
                    
                    if interaction['severity'] == 'Severe':
                        st.error("⚠️ **Warning:** This is a severe interaction. Consult your doctor immediately.")
                    elif interaction['severity'] == 'Moderate':
                        st.warning("⚠️ **Caution:** Monitor for side effects and consult your healthcare provider.")
        else:
            st.success("### ✅ No Known Interactions Found")
            st.write("The selected medicines have no known significant interactions in our database.")
            st.info("💡 However, always consult with a healthcare professional before combining medications.")

# Information section
st.markdown("---")
st.subheader("ℹ️ About Drug Interactions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔴 Severe
    - Life-threatening
    - Avoid combination
    - Seek immediate medical advice
    """)

with col2:
    st.markdown("""
    ### 🟠 Moderate
    - Potentially significant
    - Monitor closely
    - Consult healthcare provider
    """)

with col3:
    st.markdown("""
    ### 🟢 Low
    - Minor interaction
    - Usually manageable
    - Be aware of effects
    """)

st.markdown("---")
st.info("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a healthcare professional for medical advice.")