
# 🧠 Student Mental Health Predictor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Sarthak451-student-mental-health-predictor.streamlit.app/)

This web app uses machine learning to predict whether a student shows signs of depression based on academic and lifestyle factors.

## 🚀 Try the App Live

 visit: https://student-mental-health-predictor-kyhwc7cmtzbf9kdstzgrmh.streamlit.app/

## 📦 How to Run Locally

```bash
git clone https://github.com/Sarthak451/student-mental-health-predictor.git
cd student-mental-health-predictor
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 📁 Folder Structure

```
student-mental-health-predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   └── Student Mental health.csv
├── model/
│   ├── model.pkl
│   └── label_encoders.pkl
```

## 💡 Features

- Predicts likelihood of student depression
- Built with Streamlit and scikit-learn
- Interactive form UI
- Deployable via Streamlit Cloud

## 📄 License

MIT License – use freely with attribution.
