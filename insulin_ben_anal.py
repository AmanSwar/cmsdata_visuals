import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

insulin_cost_file = "data/textdata/insulin beneficiary cost file  PPUF_2024Q4.txt"

insulin_df = pd.read_csv(insulin_cost_file, delimiter='|')

insulin_cost_cols = [
    'copay_amt_pref_insln', 'copay_amt_nonpref_insln',
    'copay_amt_mail_pref_insln', 'copay_amt_mail_nonpref_insln'
]

for col in insulin_cost_cols:
    insulin_df[col] = pd.to_numeric(insulin_df[col], errors='coerce')

insulin_df['AVG_COPAY_RETAIL'] = insulin_df[['copay_amt_pref_insln', 'copay_amt_nonpref_insln']].mean(axis=1)
insulin_df['AVG_COPAY_MAIL'] = insulin_df[['copay_amt_mail_pref_insln', 'copay_amt_mail_nonpref_insln']].mean(axis=1)

print("\nInsulin Beneficiary Cost Summary:")
print(insulin_df[['AVG_COPAY_RETAIL', 'AVG_COPAY_MAIL']].describe())


insulin_plan_summary = insulin_df.groupby('PLAN_ID')[['AVG_COPAY_RETAIL', 'AVG_COPAY_MAIL']].mean().reset_index()
print("\nAverage Insulin Copays by PLAN_ID:")
print(insulin_plan_summary.head())


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(insulin_df['AVG_COPAY_RETAIL'].dropna(), bins=30, color='coral', edgecolor='black')
plt.xlabel('Average Retail Copay')
plt.ylabel('Frequency')
plt.title('Histogram of Average Retail Copay for Insulin')

plt.subplot(1, 2, 2)
plt.hist(insulin_df['AVG_COPAY_MAIL'].dropna(), bins=30, color='mediumpurple', edgecolor='black')
plt.xlabel('Average Mail Copay')
plt.ylabel('Frequency')
plt.title('Histogram of Average Mail Copay for Insulin')

plt.tight_layout()
plt.show()