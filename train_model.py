# 🔧 train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/Student Mental health.csv")
df.columns = df.columns.str.strip()

# Rename columns for consistency
df.rename(columns={
    'Choose your gender': 'gender',
    'What is your course?': 'course',
    'Your current year of Study': 'year_of_study',
    'What is your CGPA?': 'cgpa',
    'Marital status': 'marital_status',
    'Do you have Depression?': 'depression',
    'Do you have Anxiety?': 'anxiety',
    'Do you have Panic attack?': 'panic_attack',
    'Did you seek any specialist for a treatment?': 'sought_treatment'
}, inplace=True)

# Drop timestamp
df.drop(columns=['Timestamp'], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Save encoders
joblib.dump(label_encoders, 'model/label_encoders.pkl')

# Define features and target
X = df.drop(columns=['depression'])
y = df['depression']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/model.pkl')
print("✅ Model and encoders saved.")
