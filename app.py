import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Mental Health Prediction", layout="centered")
st.title("🧠 Mental Health Prediction App")

st.markdown("Fill out the following details to predict your mental health status.")

# Create a form
with st.form("mental_health_form"):
    gender = st.selectbox("Choose your gender", options=label_encoders['Choose your gender'].classes_)
    age = st.number_input("Enter your age", min_value=10, max_value=100, step=1)
    course = st.selectbox("What is your course?", options=label_encoders['What is your course?'].classes_)
    year = st.selectbox("Your current year of Study", options=label_encoders['Your current year of Study'].classes_)
    cgpa = st.selectbox("What is your CGPA?", options=label_encoders['What is your CGPA?'].classes_)
    marital_status = st.selectbox("Marital status", options=label_encoders['Marital status'].classes_)

    submitted = st.form_submit_button("🔍 Predict")

# Process after form submission
if submitted:
    # Create input DataFrame
    input_data = {
        'Choose your gender': [gender],
        'Age': [age],
        'What is your course?': [course],
        'Your current year of Study': [year],
        'What is your CGPA?': [cgpa],
        'Marital status': [marital_status]
    }

    input_df = pd.DataFrame(input_data)

    # Encode categorical features
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Prediction: {prediction}")
