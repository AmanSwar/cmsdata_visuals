import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


DATA_DIR = "data/textdata"
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



plan_info_file = os.path.join(DATA_DIR , "plan information  PPUF_2024Q4.txt")
ben_file = os.path.join(DATA_DIR, "beneficiary cost file  PPUF_2024Q4.txt")
# formulary_file = os.path.join(DATA_DIR , "basic drugs formulary file  PPUF_2024Q4.txt")

plan_info = pd.read_csv(plan_info_file , delimiter="|" , encoding="ISO-8859-1")
benef = pd.read_csv(ben_file , delimiter="|" , encoding="ISO-8859-1")
# basic_drug = pd.read_csv(formulary_file , delimiter="|" , encoding="ISO-8859-1")



cost_colm = [
    'COST_AMT_PREF', 'COST_AMT_NONPREF', 
    'COST_AMT_MAIL_PREF', 'COST_AMT_MAIL_NONPREF'

]

#pre proc
for col in cost_colm:
    benef[col] = pd.to_numeric(benef[col], errors='coerce')


#avg cost
benef['AVG_COST'] = benef[cost_colm].mean(axis=1)

merged_df = pd.merge(plan_info, benef, on=['CONTRACT_ID', 'PLAN_ID', 'SEGMENT_ID'], how='inner')


plan_avg_cost = merged_df.groupby('PLAN_ID')['AVG_COST'].mean().reset_index()
plan_avg_cost.rename(columns={'AVG_COST': 'PLAN_AVG_COST'}, inplace=True)

plan_cost_df = pd.merge(plan_info, plan_avg_cost, on='PLAN_ID', how='left')

print("Plan-Level Cost Metrics (first few rows):")
print(plan_cost_df[['PLAN_ID', 'PREMIUM', 'DEDUCTIBLE', 'PLAN_AVG_COST']].head())


plt.figure(figsize=(10, 6))
plt.scatter(plan_cost_df['PREMIUM'], plan_cost_df['PLAN_AVG_COST'], alpha=0.7, color='blue')
plt.xlabel('Plan Premium')
plt.ylabel('Average Drug Cost per Plan')
plt.title('Plan Premium vs. Average Drug Cost')
plt.grid(True)
plt.show()


tier_summary = benef.groupby('TIER')['AVG_COST'].agg(['mean', 'median', 'min', 'max']).reset_index()
print("\nSummary Statistics by TIER:")
print(tier_summary)

# B. If available, group by COVERAGE_LEVEL and compute summary statistics
if 'COVERAGE_LEVEL' in benef.columns:
    coverage_summary = benef.groupby('COVERAGE_LEVEL')['AVG_COST'].agg(['mean', 'median', 'min', 'max']).reset_index()
    print("\nSummary Statistics by COVERAGE_LEVEL:")
    print(coverage_summary)


benef['AVG_COST_RETAIL'] = benef[['COST_AMT_PREF', 'COST_AMT_NONPREF']].mean(axis=1)

# Compute average cost for mail order channels (preferred and non-preferred mail)
benef['AVG_COST_MAIL'] = benef[['COST_AMT_MAIL_PREF', 'COST_AMT_MAIL_NONPREF']].mean(axis=1)

# Group by TIER to get summary statistics for retail and mail costs
retail_summary = benef.groupby('TIER')['AVG_COST_RETAIL'].agg(['mean', 'median', 'min', 'max']).reset_index()
mail_summary = benef.groupby('TIER')['AVG_COST_MAIL'].agg(['mean', 'median', 'min', 'max']).reset_index()

print("\nRetail Cost Summary by TIER:")
print(retail_summary)
print("\nMail Order Cost Summary by TIER:")
print(mail_summary)