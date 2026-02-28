import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

telco = pd.read_csv(DATA_DIR / "telco.csv")

telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
telco.fillna(0, inplace=True)

telco['CLV'] = telco['MonthlyCharges'] * telco['tenure']
telco['ChurnRisk'] = telco['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

telco['RevenueLeakageScore'] = (
    telco['ChurnRisk'] * 0.6 +
    (1 / (telco['tenure'] + 1)) * 0.4
)

# Top risky customers
top_risk = telco.sort_values(by="RevenueLeakageScore", ascending=False).head(10)

print("\nTop 10 Revenue Leakage Risk Customers:")
print(top_risk[['customerID', 'MonthlyCharges', 'tenure', 'RevenueLeakageScore']])

# Estimated revenue loss
telco['PotentialLoss'] = telco['MonthlyCharges'] * telco['ChurnRisk']

total_loss = telco['PotentialLoss'].sum()

print("\nEstimated Monthly Revenue at Risk:", round(total_loss, 2))