import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

telco = pd.read_csv(DATA_DIR / "telco.csv")
ecom = pd.read_csv(DATA_DIR / "ecommerce.csv")

# Convert TotalCharges to numeric
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
telco.fillna(0, inplace=True)

# -----------------------------
# CUSTOMER LIFETIME VALUE (CLV)
# -----------------------------

telco['CLV'] = telco['MonthlyCharges'] * telco['tenure']

# -----------------------------
# CHURN RISK SCORE
# -----------------------------

telco['ChurnRisk'] = telco['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# -----------------------------
# REVENUE LEAKAGE SCORE
# -----------------------------

telco['RevenueLeakageScore'] = (
    telco['ChurnRisk'] * 0.6 +
    (1 / (telco['tenure'] + 1)) * 0.4
)

# Higher score = higher leakage risk

print("Top Leakage Risk Customers:")
print(telco[['customerID', 'CLV', 'RevenueLeakageScore']].sort_values(
    by='RevenueLeakageScore',
    ascending=False
).head())


# -----------------------------
# ECOMMERCE LOSS ANALYSIS
# -----------------------------

ecom['RevenueLoss'] = ecom['Price (Rs.)'] - ecom['Final_Price(Rs.)']

print("\nTop Revenue Loss Transactions:")
print(ecom[['User_ID', 'RevenueLoss']].sort_values(
    by='RevenueLoss',
    ascending=False
).head())


print("\nRevenue Metrics Created Successfully")