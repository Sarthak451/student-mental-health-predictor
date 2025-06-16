import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/Student Mental health.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Select useful columns
features = [
    'Choose your gender',
    'Age',
    'Marital status',
    'Do you have Depression?',
    'Do you have Anxiety?',
    'Do you have Panic attack?'
]
target = 'Did you seek any specialist for a treatment?'

df = df[features + [target]].dropna()

# Encode categorical columns
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "model/mental_health_model.pkl")
joblib.dump({col: label_encoders[col] for col in features}, "model/label_encoders.pkl")
joblib.dump(label_encoders[target], "model/target_encoder.pkl")

print("✅ Model trained and saved successfully!")
