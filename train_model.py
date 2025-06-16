import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("Student Mental health.csv")

# Rename columns
df.rename(columns={
    'Choose your gender': 'Gender',
    'What is your course?': 'Course',
    'Your current year of Study': 'Year',
    'What is your CGPA?': 'CGPA',
    'Marital status': 'Marital_Status',
    'Do you have Depression?': 'Depression',
    'Do you have Anxiety?': 'Anxiety',
    'Do you have Panic attack?': 'Panic_Attack',
    'Did you seek any specialist for a treatment?': 'Treatment'
}, inplace=True)

# 👇 Debug: Check CGPA values before conversion
print("Unique CGPA values before cleaning:", df['CGPA'].unique())

# Convert CGPA ranges to average floats
def convert_cgpa(value):
    if isinstance(value, str) and " - " in value:
        try:
            low, high = value.split(" - ")
            return round((float(low) + float(high)) / 2, 2)
        except:
            return None
    try:
        return float(value)
    except:
        return None

df['CGPA'] = df['CGPA'].apply(convert_cgpa)
df = df.dropna(subset=['CGPA'])  # Drop rows where CGPA conversion failed

# Drop timestamp if exists
if 'Timestamp' in df.columns:
    df.drop(columns=['Timestamp'], inplace=True)

# Features and target
features = ['Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital_Status']
target = 'Depression'

# Encode categorical variables
label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])

# Split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mental_health_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(target_encoder, "model/target_encoder.pkl")

print("✅ Model trained and saved successfully.")

