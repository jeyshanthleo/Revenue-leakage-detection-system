import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # fixes blank plots on many Mac setups
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Load + clean
telco = pd.read_csv(DATA_DIR / "telco.csv")
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
telco.fillna(0, inplace=True)

telco["Churn"] = telco["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
telco.drop(columns=["customerID"], inplace=True)

for col in telco.select_dtypes(include=["object"]).columns:
    telco[col] = LabelEncoder().fit_transform(telco[col])

X = telco.drop("Churn", axis=1)
y = telco["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Sample for speed
X_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 1) Global summary (save to file)
plt.figure()
shap.summary_plot(shap_values[1], X_sample, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png", dpi=200)
plt.close()
print("Saved:", OUT_DIR / "shap_summary.png")

# 2) Local explanation for one customer (save to file)
i = 0
exp = shap.Explanation(
    values=shap_values[1][i],
    base_values=explainer.expected_value[1],
    data=X_sample.iloc[i],
    feature_names=X_sample.columns
)

plt.figure()
shap.waterfall_plot(exp, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_waterfall_customer0.png", dpi=200)
plt.close()
print("Saved:", OUT_DIR / "shap_waterfall_customer0.png")