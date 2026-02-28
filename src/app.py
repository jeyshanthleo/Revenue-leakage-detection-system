import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==============================
# PATH SETUP (FIXED)
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]   # project root
DATA_DIR = BASE_DIR / "data"

TELCO_PATH = DATA_DIR / "telco.csv"
ECOM_PATH = DATA_DIR / "ecommerce.csv"


# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    telco = pd.read_csv(TELCO_PATH)
    ecom = pd.read_csv(ECOM_PATH)

    # Cleaning
    telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
    telco.fillna(0, inplace=True)

    # Feature Engineering
    telco['CLV'] = telco['MonthlyCharges'] * telco['tenure']
    telco['ChurnRisk'] = telco['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    telco['RevenueLeakageScore'] = (
        telco['ChurnRisk'] * 0.6 +
        (1 / (telco['tenure'] + 1)) * 0.4
    )

    telco['PotentialLoss'] = telco['MonthlyCharges'] * telco['ChurnRisk']

    return telco, ecom


telco, ecom = load_data()


# ==============================
# DASHBOARD UI
# ==============================

st.set_page_config(page_title="Revenue Leakage Dashboard", layout="wide")

st.title("💰 Hidden Revenue Leakage Detection System")
st.markdown("AI-Driven Customer Revenue Risk Analytics")


# ==============================
# KPI METRICS
# ==============================

total_customers = len(telco)
total_revenue = telco['MonthlyCharges'].sum()
total_loss = telco['PotentialLoss'].sum()
avg_leakage_score = telco['RevenueLeakageScore'].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Total Monthly Revenue", f"${total_revenue:,.2f}")
col3.metric("Revenue At Risk", f"${total_loss:,.2f}")
col4.metric("Avg Leakage Score", round(avg_leakage_score, 3))


st.divider()


# ==============================
# GRAPH 1 — Leakage Score Distribution
# ==============================

fig1 = px.histogram(
    telco,
    x="RevenueLeakageScore",
    nbins=30,
    title="Revenue Leakage Score Distribution"
)

st.plotly_chart(fig1, use_container_width=True)


# ==============================
# GRAPH 2 — CLV vs Leakage
# ==============================

fig2 = px.scatter(
    telco,
    x="CLV",
    y="RevenueLeakageScore",
    color="Churn",
    title="Customer Value vs Leakage Risk"
)

st.plotly_chart(fig2, use_container_width=True)


# ==============================
# GRAPH 3 — Revenue Loss by Churn
# ==============================

loss_summary = telco.groupby("Churn")["PotentialLoss"].sum().reset_index()

fig3 = px.bar(
    loss_summary,
    x="Churn",
    y="PotentialLoss",
    color="Churn",
    title="Revenue Loss by Churn Category"
)

st.plotly_chart(fig3, use_container_width=True)


st.divider()


# ==============================
# TOP RISK CUSTOMERS TABLE
# ==============================

st.subheader("🚨 Top High Revenue Risk Customers")

top_risk = telco.sort_values(
    by="RevenueLeakageScore",
    ascending=False
).head(10)

st.dataframe(top_risk)


# ==============================
# DOWNLOAD OPTION
# ==============================

st.download_button(
    "Download Risk Data",
    telco.to_csv(index=False),
    file_name="revenue_leakage_results.csv"
)
# ================= KPI METRICS =================

total_customers = len(telco)
high_risk_customers = (telco["RevenueLeakageScore"] > 0.6).sum()
total_revenue_risk = telco["PotentialLoss"].sum()
avg_leakage_score = telco["RevenueLeakageScore"].mean()

st.markdown("## 📊 Revenue Leakage Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", total_customers)
c2.metric("High Risk Customers", high_risk_customers)
c3.metric("Monthly Revenue At Risk ($)", round(total_revenue_risk, 2))
c4.metric("Avg Leakage Score", round(avg_leakage_score, 3))
st.sidebar.header("Filters")

churn_filter = st.sidebar.selectbox(
    "Select Churn Category",
    ["All", "Yes", "No"]
)

if churn_filter != "All":
    telco = telco[telco["Churn"] == churn_filter]
def risk_segment(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    else:
        return "High"

telco["RiskSegment"] = telco["RevenueLeakageScore"].apply(risk_segment)
st.subheader("Customer Risk Segmentation")

seg_fig = px.histogram(
    telco,
    x="RiskSegment",
    color="RiskSegment",
    title="Customer Risk Distribution"
)

st.plotly_chart(seg_fig, use_container_width=True)
csv = telco.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Processed Data",
    csv,
    "revenue_leakage_data.csv",
    "text/csv"
)
st.markdown("""
---
### 💡 Business Impact

This system helps companies:

- Identify customers likely to churn
- Estimate monthly revenue leakage
- Prioritize retention campaigns
- Optimize customer lifetime value

Potential Use Cases:

- SaaS companies
- Telecom providers
- Subscription businesses
- E-commerce platforms
""")