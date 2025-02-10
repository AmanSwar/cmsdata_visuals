import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ben_cost_file = "data/textdata/beneficiary cost file  PPUF_2024Q4.txt"
ben_df = pd.read_csv(ben_cost_file, delimiter='|')

cost_columns = [
    'COST_AMT_PREF', 'COST_AMT_NONPREF',
    'COST_AMT_MAIL_PREF', 'COST_AMT_MAIL_NONPREF'
]

for col in cost_columns:
    ben_df[col] = pd.to_numeric(ben_df[col], errors='coerce')

ben_df['DAYS_SUPPLY'] = pd.to_numeric(ben_df['DAYS_SUPPLY'], errors='coerce')

ben_df['AVG_COST'] = ben_df[cost_columns].mean(axis=1)

ben_df['COST_PER_DAY'] = ben_df.apply(
    lambda row: row['AVG_COST'] / row['DAYS_SUPPLY'] if row['DAYS_SUPPLY'] and row['DAYS_SUPPLY'] > 0 else np.nan,
    axis=1
)

print("\nGeneral Beneficiary Cost Summary:")
print(ben_df[['AVG_COST', 'COST_PER_DAY']].describe())

tier_summary = ben_df.groupby('TIER')['AVG_COST'].agg(['mean', 'median', 'min', 'max']).reset_index()
print("\nSummary Statistics by TIER (AVG_COST):")
print(tier_summary)


if 'COVERAGE_LEVEL' in ben_df.columns:
    coverage_summary = ben_df.groupby('COVERAGE_LEVEL')['AVG_COST'].agg(['mean', 'median', 'min', 'max']).reset_index()
    print("\nSummary Statistics by COVERAGE_LEVEL (AVG_COST):")
    print(coverage_summary)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(ben_df['AVG_COST'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Average Cost')
plt.ylabel('Frequency')
plt.title('Histogram of Average Cost')

plt.subplot(1, 2, 2)
plt.boxplot(ben_df['AVG_COST'].dropna(), vert=False)
plt.xlabel('Average Cost')
plt.title('Boxplot of Average Cost')

plt.tight_layout()
plt.show()

# Histogram and Boxplot for COST_PER_DAY
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(ben_df['COST_PER_DAY'].dropna(), bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Cost per Day')
plt.ylabel('Frequency')
plt.title('Histogram of Cost per Day')

plt.subplot(1, 2, 2)
plt.boxplot(ben_df['COST_PER_DAY'].dropna(), vert=False)
plt.xlabel('Cost per Day')
plt.title('Boxplot of Cost per Day')

plt.tight_layout()
plt.show()