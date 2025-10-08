import streamlit as st
import pandas as pd
import joblib

# Load model and metadata
model = joblib.load("alzheimers_model.pkl")
metadata = joblib.load("alzheimers_metadata.pkl")

st.title("🧠 Alzheimer's Disease Prediction App")

feature_columns = metadata.get("feature_columns", [])
st.write(f"Model expects {len(feature_columns)} features.")

user_input = {}

# Create dynamic input fields for each feature
for feature in feature_columns:
    user_input[feature] = st.text_input(f"Enter {feature}")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Convert numeric columns properly
input_df = input_df.apply(pd.to_numeric, errors='ignore')

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Prediction result: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
