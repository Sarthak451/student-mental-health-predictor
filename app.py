import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/mental_health_model.pkl")

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = joblib.load(f)

with open("model/target_encoder.pkl", "rb") as f:
    target_encoder = joblib.load(f)

st.title("🧠 Human Mental Health Prediction")

st.markdown("### Please enter your information:")

# Input form
with st.form("mental_health_form"):
    gender = st.selectbox("Choose your gender", options=label_encoders['Choose your gender'].classes_)
    age = st.slider("Age", 18, 40, 20)
    course = st.selectbox("What is your course?", options=label_encoders['What is your course?'].classes_)
    year = st.selectbox("Your current year of Study", options=label_encoders['Your current year of Study'].classes_)
    cgpa = st.number_input("What is your CGPA?", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    marital_status = st.selectbox("Marital Status", options=label_encoders['Marital status'].classes_)
    depression = st.selectbox("Do you have Depression?", options=label_encoders['Do you have Depression?'].classes_)
    anxiety = st.selectbox("Do you have Anxiety?", options=label_encoders['Do you have Anxiety?'].classes_)
    panic_attack = st.selectbox("Do you have Panic attack?", options=label_encoders['Do you have Panic attack?'].classes_)
    treatment = st.selectbox("Did you seek any specialist for treatment?", options=label_encoders['Did you seek any specialist for a treatment?'].classes_)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build input DataFrame
    input_data = {
        'Choose your gender': [gender],
        'Age': [age],
        'What is your course?': [course],
        'Your current year of Study': [year],
        'What is your CGPA?': [cgpa],
        'Marital status': [marital_status],
        'Do you have Depression?': [depression],
        'Do you have Anxiety?': [anxiety],
        'Do you have Panic attack?': [panic_attack],
        'Did you seek any specialist for a treatment?': [treatment]
    }

    input_df = pd.DataFrame(input_data)

    # Apply label encoders
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)
    prediction_label = target_encoder.inverse_transform(prediction)[0]

    st.success(f"🧾 **Predicted Mental Health Condition:** {prediction_label}")
