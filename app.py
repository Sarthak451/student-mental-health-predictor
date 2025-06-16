import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/mental_health_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Human Mental Health Predictor")
st.markdown("Use this tool to predict mental health condition based on student data.")

# 📝 Form for user input
with st.form("mental_health_form"):
    gender = st.selectbox("Choose your gender", options=label_encoders['Choose your gender'].classes_)
    age = st.number_input("Age", min_value=15, max_value=100)
    course = st.selectbox("What is your course?", options=label_encoders['What is your course?'].classes_)
    year = st.selectbox("Your current year of Study", options=label_encoders['Your current year of Study'].classes_)
    cgpa = st.number_input("What is your CGPA?", min_value=0.0, max_value=10.0, step=0.01)
    marital_status = st.selectbox("Marital status", options=label_encoders['Marital status'].classes_)

    submitted = st.form_submit_button("🔍 Predict")

# 🧠 Perform prediction
if submitted:
    input_dict = {
        'Choose your gender': gender,
        'Age': age,
        'What is your course?': course,
        'Your current year of Study': year,
        'What is your CGPA?': cgpa,
        'Marital status': marital_status
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    # Make prediction
    prediction = model.predict(input_df)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    st.success(f"🎯 Predicted Mental Health Condition: **{predicted_label}**")
