import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

telco = pd.read_csv(DATA_DIR / "telco.csv")
ecom = pd.read_csv(DATA_DIR / "ecommerce.csv")

print("Telco shape:", telco.shape)
print("Ecommerce shape:", ecom.shape)

print("\nTelco columns:", telco.columns.tolist())
print("\nEcommerce columns:", ecom.columns.tolist())