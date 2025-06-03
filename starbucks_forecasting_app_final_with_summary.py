# --- Summary ---
st.markdown("""### 📌 Summary for Audit Committee
This app applies an ARIMAX model using CPI, average ticket size, and transaction volume to forecast Starbucks revenue in 2023. Forecasted values are benchmarked against reported revenue, with risk flags issued for deviations exceeding 5%.

The ARIMAX forecast shows generally strong alignment with reported revenue, reinforcing credibility in the majority of periods. However, specific quarters were flagged due to material deviations between the forecasted and actual revenue. These discrepancies present potential risks of revenue overstatement, particularly if unsupported by changes in the underlying inputs.

The regression model confirms that actual revenue correlates strongly with the predictive inputs—CPI, average ticket size, and transaction volume. While transaction volume appears to be the most influential factor, CPI had limited predictive power, suggesting that revenue is more influenced by Starbucks' operational performance than macroeconomic inflation.

Growth trends further validate this: periods of high revenue growth generally coincide with proportional increases in average ticket size and CPI. However, any divergence—where revenue grows disproportionately faster than inputs—raises red flags. For instance, if revenue increases sharply while average ticket and CPI remain flat, it could signal premature or unsupported revenue recognition.

Auditors should prioritize review of flagged quarters and consider whether marketing strategies, accounting policies, or estimation methods contributed to the mismatch. Where input trends do not justify revenue spikes, audit testing should focus on cut-off procedures, deferred revenue treatment, and sales return allowances.
""")
