
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and clean the dataset
df = pd.read_csv("Student Mental health.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces from column names

# Drop timestamp column if not needed
if 'Timestamp' in df.columns:
    df.drop('Timestamp', axis=1, inplace=True)

# Strip spaces from values and fill missing
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical features
label_encoders = {}
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and label (we'll predict 'Do you have Depression?')
X = df.drop(['Do you have Depression?'], axis=1)
y = df['Do you have Depression?']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoders
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model and encoders saved successfully.")
