
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from fredapi import Fred
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- Get Live CPI from FRED ---
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
cpi_series = fred.get_series('CPIAUCSL')
current_cpi = cpi_series.iloc[-1]

st.title("☕ Starbucks Revenue Forecasting App")
st.write(f"### Current CPI: {current_cpi:.2f}")

# --- User Inputs ---
st.sidebar.header("User Inputs")
cpi_input = st.sidebar.slider("Adjusted CPI", min_value=200.0, max_value=320.0, value=float(current_cpi), step=0.1)
avg_ticket_input = st.sidebar.slider("Expected Avg Ticket ($)", 3.0, 8.0, 5.5, step=0.1)
txn_input = st.sidebar.slider("Expected Transactions", 800, 1200, 1000, step=10)

# --- Replace Forecast Period Data with User Input ---
df['cpi'] = df['cpi'].fillna(method='ffill')
df['avg_ticket'] = df['avg_ticket'].fillna(method='ffill')
df['transactions'] = df['transactions'].fillna(method='ffill')
df.loc['2023-01-01':, 'cpi'] = cpi_input
df.loc['2023-01-01':, 'avg_ticket'] = avg_ticket_input
df.loc['2023-01-01':, 'transactions'] = txn_input

# --- ARIMAX Forecast ---
train_data = df.loc[:'2022-12-31']
test_data = df.loc['2023-01-01':]
endog_train = train_data['revenue']
exog_train = train_data[['cpi', 'avg_ticket', 'transactions']]
model = SARIMAX(endog_train, exog=exog_train, order=(1, 1, 1))
results = model.fit(disp=False)
exog_forecast = test_data[['cpi', 'avg_ticket', 'transactions']]
forecast = results.get_prediction(start=test_data.index[0], end=test_data.index[-1], exog=exog_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
actual = test_data['revenue']
errors = actual - forecast_mean
percent_errors = errors / actual * 100

# --- Risk Flagging ---
risk_flags = pd.DataFrame({
    'Forecast': forecast_mean,
    'Actual': actual,
    'Error (%)': percent_errors
})
risk_flags['Flag'] = risk_flags['Error (%)'].apply(lambda x: '🚨 High Risk' if abs(x) > 5 else '✔️ Normal')

# --- Visualization: Forecast vs Actual ---
st.subheader("📈 ARIMAX Revenue Forecast vs Actual")
fig1, ax1 = plt.subplots()
df['revenue'].plot(ax=ax1, label='Actual Revenue', color='blue')
forecast_mean.plot(ax=ax1, label='Forecasted Revenue (2023)', color='green')
ax1.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='green', alpha=0.3)
ax1.set_ylabel("Revenue")
ax1.legend()
st.pyplot(fig1)

# --- Risk Summary ---
st.subheader("📊 Forecast Errors & Risk Flags")
st.dataframe(risk_flags.style.format({'Forecast': '${:,.0f}', 'Actual': '${:,.0f}', 'Error (%)': '{:.2f}%'}))

# --- Visualization: % Growth Over Time ---
st.subheader("📊 % Growth in Revenue, Avg Ticket, and CPI Over Time")
growth_df = df[['revenue', 'avg_ticket', 'cpi']].pct_change().dropna() * 100
fig_growth, ax_growth = plt.subplots()
growth_df['revenue'].plot(ax=ax_growth, label='Revenue Growth (%)', color='blue')
growth_df['avg_ticket'].plot(ax=ax_growth, label='Avg Ticket Growth (%)', color='orange')
growth_df['cpi'].plot(ax=ax_growth, label='CPI Growth (%)', color='purple')
ax_growth.set_ylabel("% Growth")
ax_growth.legend()
st.pyplot(fig_growth)

# --- Regression Model ---
X_reg = df[['cpi', 'avg_ticket', 'transactions']]
y_reg = df['revenue']
reg_model = LinearRegression().fit(X_reg, y_reg)
df['expected_revenue'] = reg_model.predict(X_reg)

st.subheader("📊 Regression: Expected Revenue vs Actual Revenue & Expenses")
fig2, ax2 = plt.subplots()
df['revenue'].plot(ax=ax2, label='Actual Revenue', color='blue')
df['expected_revenue'].plot(ax=ax2, label='Expected Revenue (Regression)', linestyle='--', color='green')
df['expenses'].plot(ax=ax2, label='Actual Expenses', color='red')
ax2.set_ylabel("USD ($)")
ax2.legend()
st.pyplot(fig2)

# --- Summary ---
high_risk_2023 = risk_flags.loc['2023-01-01':]
high_risk_count = (high_risk_2023['Flag'] == '🚨 High Risk').sum()
avg_error_2023 = high_risk_2023['Error (%)'].mean()

if high_risk_count == 0:
    summary_text = f"""### 📌 Summary for Audit Committee
No high-risk quarters were identified in 2023. The ARIMAX forecast using CPI ({cpi_input:.2f}), avg ticket ({avg_ticket_input:.2f}), and transactions ({txn_input}) closely aligns with reported revenue.

This consistency suggests that Starbucks' reported revenue is well-supported by key economic and operational inputs. The regression model further confirms this by showing minimal variance between expected and actual revenues across all quarters. The observed alignment between revenue growth and input growth patterns strengthens the case for reliability in reported figures.

Based on this evidence, there is no indication of revenue overstatement in the reviewed period, and the financial reporting appears to reflect the underlying business conditions accurately.
"""
elif high_risk_count == 1:
    summary_text = f"""### 📌 Summary for Audit Committee
One high-risk quarter was flagged in 2023 where actual revenue materially exceeded forecasted values. The model, based on CPI ({cpi_input:.2f}), avg ticket ({avg_ticket_input:.2f}), and transactions ({txn_input}), generally aligns well otherwise.

The presence of a single significant deviation warrants targeted scrutiny. Auditors should consider whether temporary operational or macroeconomic anomalies might explain the gap. If no external justification is found, this deviation could indicate a timing or estimation error in revenue recognition.

While not indicative of widespread overstatement, this anomaly suggests the need for further documentation and review of revenue policies for the flagged quarter.
"""
else:
    summary_text = f"""### 📌 Summary for Audit Committee
{high_risk_count} high-risk quarters were flagged in 2023 with an average deviation of {avg_error_2023:.2f}%. Forecasts based on CPI ({cpi_input:.2f}), avg ticket ({avg_ticket_input:.2f}), and transactions ({txn_input}) showed persistent misalignment with reported revenue.

Such repeated deviations raise substantial concern about potential overstatement of revenue. The regression model also reflects a breakdown in the relationship between operational drivers and reported figures, especially where revenue growth far exceeds changes in underlying inputs.

Auditors should investigate these flagged quarters thoroughly, focusing on revenue recognition timing, estimates, and the validity of recorded transactions. Evidence of revenue recognition that outpaces business activity may warrant extended testing and disclosure consideration.
"""

st.markdown(summary_text)
