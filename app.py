import streamlit as st
import joblib
import json
import pandas as pd

# Load model
model = joblib.load("model_pipeline.pkl")

# Load metadata
with open("model_metadata.json") as f:
    metadata = json.load(f)

features = metadata["features"]

st.title("🎓 Student Dropout Predictor")
st.write("Enter student details to predict the risk of dropout.")

# Input form
user_input = {}
for feature in features:
    if feature == "gender":
        user_input[feature] = st.selectbox("Gender", ["Male", "Female"])
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict Dropout Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.write("### Prediction:", "🔴 High Risk of Dropout" if prediction == 1 else "🟢 Low Risk of Dropout")
    st.write("### Probability of Dropout:", f"{probability:.2%}")
