import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Load data
telco = pd.read_csv(DATA_DIR / "telco.csv")

# Cleaning
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
telco.fillna(0, inplace=True)

telco['Churn'] = telco['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

telco.drop(columns=['customerID'], inplace=True)

# Encode categorical columns
for col in telco.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    telco[col] = le.fit_transform(telco[col])

X = telco.drop('Churn', axis=1)
y = telco['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(10)

# Plot
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Top Factors Driving Customer Churn")
plt.xlabel("Importance Score")

plt.show()