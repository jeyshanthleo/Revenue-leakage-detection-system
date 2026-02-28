# 🚀 Hidden Revenue Leakage Detection System

An end-to-end machine learning and analytics system designed to identify customers at risk of churn and estimate potential revenue leakage for subscription-based businesses.

This project combines predictive modeling, financial impact estimation, and interactive visualization to help organizations proactively reduce revenue loss and improve customer retention strategies.

---

## 📊 Project Overview

Revenue leakage is a major hidden problem in subscription and SaaS businesses. Customers often churn silently without early warning signals, leading to financial losses.

This system solves that problem by:

✅ Predicting churn risk using machine learning  
✅ Estimating customer lifetime value (CLV)  
✅ Calculating potential revenue leakage  
✅ Identifying high-risk customers  
✅ Providing interactive dashboards for business decision-making  

---

## 🧠 Machine Learning Model

Model Used: **Random Forest Classifier**

Evaluation Metrics:

- Accuracy: ~79%
- Precision: 83%
- Recall: 91% (non-churn class)
- F1 Score: 0.87

The model demonstrates strong capability in identifying customer risk segments and potential churn behavior.

---

## 💰 Business Logic & Financial Metrics

### Customer Lifetime Value (CLV)

```
CLV = MonthlyCharges × Tenure
```

Used to estimate potential long-term business value of each customer.

### Revenue Leakage Risk Score

Risk score generated using machine learning probability outputs.

### Potential Financial Loss

```
PotentialLoss = MonthlyCharges × ChurnRisk
```

Helps prioritize high-value customers for retention campaigns.

---

## 📈 Dashboard Features

Interactive dashboard built using **Streamlit + Plotly** including:

- CLV vs Revenue Leakage Risk visualization
- Revenue Leakage Score Distribution
- Ecommerce Revenue Loss Distribution
- Feature Importance Analysis
- High-Risk Customer Table
- Business Insights Summary

---

## 🖼 Dashboard Preview
<img width="799" height="500" alt="Screenshot 2026-02-28 at 1 06 45 PM" src="https://github.com/user-attachments/assets/418ca8dc-63c7-4ae8-9855-fd76585c0af7" />
<img width="798" height="498" alt="Screenshot 2026-02-28 at 1 17 20 PM" src="https://github.com/user-attachments/assets/4ac2ec09-7c78-4a89-8289-cb0612b80f58" />
<img width="798" height="498" alt="Screenshot 2026-02-28 at 1 21 30 PM" src="https://github.com/user-attachments/assets/5a507a32-a1af-4735-9f8f-8e870040b9f2" />
<img width="998" height="599" alt="Screenshot 2026-02-28 at 2 02 46 PM" src="https://github.com/user-attachments/assets/b27be63f-354a-40d1-af96-b1ae1c6896cc" />



---

## 🏗 Project Architecture

```
Data Sources → Data Cleaning → Feature Engineering → Machine Learning Model
→ Risk Scoring → Revenue Impact Analysis → Interactive Dashboard
```

---

## 📂 Project Structure

```
Revenue_Leakage_Detection_System/
│
├── data/
│   ├── telco.csv
│   └── ecommerce.csv
│
├── src/
│   ├── load_data.py
│   ├── data_cleaning.py
│   ├── leakage_insights.py
│   ├── ml_model.py
│   ├── plot_clv_vs_leakage.py
│   ├── plot_leakage_distribution.py
│   ├── plot_revenue_loss.py
│   ├── plot_feature_importance.py
│   └── app.py
│
├── notebooks/
├── outputs/
├── .gitignore
└── README.md
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Git & GitHub

---

## 🎯 Business Impact

This system enables organizations to:

- Reduce churn-related revenue loss
- Prioritize high-value customer retention
- Improve customer lifetime value
- Make proactive financial decisions
- Detect hidden revenue leakage early

---

## ▶️ How to Run the Project

Clone repository:

```bash
git clone https://github.com/jeyshanthleo/Revenue-leakage-detection-system.git
cd Revenue-leakage-detection-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run src/app.py
```

---

## 👤 Author

**Jeyshanth Leo**  
MS in Business Analytics — Northeastern University  

GitHub: https://github.com/jeyshanthleo  
LinkedIn: https://www.linkedin.com/in/jeyshanth/

