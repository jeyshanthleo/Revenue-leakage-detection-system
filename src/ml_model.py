import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Load data
telco = pd.read_csv(DATA_DIR / "telco.csv")

# Cleaning
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
telco.fillna(0, inplace=True)

# Target variable
telco['Churn'] = telco['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop customerID (not useful for model)
telco.drop(columns=['customerID'], inplace=True)

# Encode categorical columns
for col in telco.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    telco[col] = le.fit_transform(telco[col])

# Split data
X = telco.drop('Churn', axis=1)
y = telco['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))