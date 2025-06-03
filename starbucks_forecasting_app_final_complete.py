
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

st.markdown(f"""### 📌 Summary for Audit Committee
This app applies an ARIMAX model using CPI (set to {cpi_input:.2f}), average ticket size ({avg_ticket_input:.2f}), and transaction volume ({txn_input}) to forecast Starbucks revenue in 2023. Forecasted values are benchmarked against reported revenue, with risk flags issued for deviations exceeding 5%.

In 2023, the model identified **{high_risk_count} quarter(s)** where forecasted revenue deviated materially from actuals. These variances may represent potential revenue overstatement risks and should be examined closely in the audit process.

The regression model, using the same input drivers, indicates that actual revenue generally tracks well with expectations. This suggests that revenue reporting is largely consistent with economic and operational inputs. However, if any flagged quarter shows revenue growth not supported by input growth (e.g., flat CPI or avg ticket), it warrants further audit scrutiny.

Auditors should focus on these flagged quarters and evaluate whether reported revenue aligns with sales activity and customer behavior. Pay particular attention to periods where revenue grows significantly faster than CPI or average ticket size, which could signal aggressive or premature revenue recognition.
""")
