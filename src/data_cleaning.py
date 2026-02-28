import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

telco = pd.read_csv(DATA_DIR / "telco.csv")
ecom = pd.read_csv(DATA_DIR / "ecommerce.csv")

print("Before Cleaning:")
print("Telco Missing Values:\n", telco.isnull().sum())
print("\nEcommerce Missing Values:\n", ecom.isnull().sum())


# -------- TELCO CLEANING -------- #

# Convert TotalCharges to numeric
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')

# Fill missing values
telco.fillna(0, inplace=True)


# -------- ECOMMERCE CLEANING -------- #

# Fill missing values
ecom.fillna(0, inplace=True)


print("\nAfter Cleaning:")
print("Telco Missing Values:\n", telco.isnull().sum())
print("\nEcommerce Missing Values:\n", ecom.isnull().sum())


print("\nCleaning Completed Successfully")