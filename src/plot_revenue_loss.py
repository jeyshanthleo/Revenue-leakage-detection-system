import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

ecom = pd.read_csv(DATA_DIR / "ecommerce.csv")

ecom['RevenueLoss'] = ecom['Price (Rs.)'] - ecom['Final_Price(Rs.)']

plt.figure(figsize=(8,5))
sns.histplot(ecom['RevenueLoss'], bins=30, kde=True)

plt.title("Ecommerce Revenue Loss Distribution")

plt.show()