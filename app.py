
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/mental_health_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

st.title("🧠 Human Mental Health Prediction App")
st.write("Enter your details below:")

# Input fields
age = st.slider("Age", 15, 40, 21)
gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
depression = st.selectbox("Do you have Depression?", ["Yes", "No"])
anxiety = st.selectbox("Do you have Anxiety?", ["Yes", "No"])
panic = st.selectbox("Do you have Panic attacks?", ["Yes", "No"])

# Prepare input
input_dict = {
    'Age': age,
    'Choose your gender': gender,
    'Marital status': marital,
    'Do you have Depression?': depression,
    'Do you have Anxiety?': anxiety,
    'Do you have Panic attack?': panic
}

input_df = pd.DataFrame([input_dict])

# Encode inputs
for col in input_df.columns:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    output = target_encoder.inverse_transform(prediction)[0]
    st.success(f"🩺 Prediction: {output}")
