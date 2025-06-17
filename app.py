
# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Student Mental Health Predictor")
st.markdown("Fill out the following details to predict your mental health status.")

with st.form("mental_health_form"):
    gender = st.selectbox("Choose your gender", label_encoders['gender'].classes_)
    age = st.slider("Enter your age", min_value=10, max_value=100, value=21)
    course = st.selectbox("What is your course?", label_encoders['course'].classes_)
    year = st.selectbox("Your current year of Study", label_encoders['year_of_study'].classes_)
    cgpa = st.selectbox("What is your CGPA?", label_encoders['cgpa'].classes_)
    marital_status = st.selectbox("Marital status", label_encoders['marital_status'].classes_)
    specialist = st.selectbox("Did you seek any specialist for a treatment?", label_encoders['sought_treatment'].classes_)
    anxiety = st.selectbox("Do you have Anxiety?", label_encoders['anxiety'].classes_)
    panic = st.selectbox("Do you have Panic attack?", label_encoders['panic_attack'].classes_)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        'gender': gender,
        'age': age,
        'course': course,
        'year_of_study': year,
        'cgpa': cgpa,
        'marital_status': marital_status,
        'sought_treatment': specialist,
        'anxiety': anxiety,
        'panic_attack': panic
    }

    input_df = pd.DataFrame([input_dict])

    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    input_df = input_df[model.feature_names_in_]
    input_df.columns = model.feature_names_in_

    prediction = model.predict(input_df)[0]
    result = "have" if prediction == 1 else "do not have"
    st.success(f"🎯 **Prediction:** You **{result.upper()}** signs of depression.")
