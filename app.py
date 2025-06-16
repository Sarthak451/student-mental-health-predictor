import streamlit as st
import pandas as pd
import joblib

<<<<<<< HEAD
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
=======
# Load model and encoders
model = joblib.load("model/model.pkl")

label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Mental Health Predictor", layout="centered")
st.title("🧠 Student Mental Health Predictor")

with st.form("mental_health_form"):
    gender = st.selectbox("Choose your gender", label_encoders['Choose your gender'].classes_)
    age = st.slider("Select your age", min_value=15, max_value=50, value=21)
    course = st.selectbox("What is your course?", label_encoders['What is your course?'].classes_)
    year = st.selectbox("Your current year of Study", label_encoders['Your current year of Study'].classes_)
    cgpa = st.selectbox("What is your CGPA?", label_encoders['What is your CGPA?'].classes_)
    marital_status = st.selectbox("Marital status", label_encoders['Marital status'].classes_)
    specialist = st.selectbox("Did you seek any specialist for a treatment?", label_encoders['Did you seek any specialist for a treatment?'].classes_)
>>>>>>> 97ffa4e (🔄 Updated app.py and train_model.py with new model logic and input handling)

    submitted = st.form_submit_button("Predict")

<<<<<<< HEAD
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
=======
if submitted:
    input_dict = {
        'Choose your gender': gender,
        'Age': age,
        'What is your course?': course,
        'Your current year of Study': year,
        'What is your CGPA?': cgpa,
        'Marital status': marital_status,
        'Did you seek any specialist for a treatment?': specialist
>>>>>>> 97ffa4e (🔄 Updated app.py and train_model.py with new model logic and input handling)
    }

    input_df = pd.DataFrame(input_data)

    # Apply label encoding
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
<<<<<<< HEAD
    st.success(f"🎯 Prediction: {prediction}")
=======
    
    st.success(f"🎯 **Prediction:** The model predicts you **{prediction.upper()}** mental health condition.")
>>>>>>> 97ffa4e (🔄 Updated app.py and train_model.py with new model logic and input handling)
