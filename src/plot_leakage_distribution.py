import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

plt.figure(figsize=(8,5))
sns.histplot(telco['RevenueLeakageScore'], bins=30, kde=True)
plt.title("Revenue Leakage Score Distribution")

plt.show()