import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
st.title("📊 Analytics")

st.write("Basic project analytics (files, datasets, model artifacts).")

st.subheader("✅ Model files status")
model_dir = "models"
model_files = ["disease_model.pkl", "medicine_model.pkl", "sentiment_model.pkl"]

if not os.path.exists(model_dir):
    st.warning("models/ folder not found.")
else:
    rows = []
    for f in model_files:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            rows.append({"file": f, "exists": True, "size_kb": round(os.path.getsize(path)/1024, 2)})
        else:
            rows.append({"file": f, "exists": False, "size_kb": 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("📁 Data files preview")
data_dir = "data"
data_files = ["diseases.csv", "medicines.csv", "symptoms.csv", "interactions.csv"]

if not os.path.exists(data_dir):
    st.warning("data/ folder not found.")
else:
    for f in data_files:
        path = os.path.join(data_dir, f)
        st.markdown(f"### {f}")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            st.info("File missing or empty.")
            continue
        try:
            df = pd.read_csv(path)
            st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read {f}: {e}")