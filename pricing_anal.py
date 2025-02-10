import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

price_file = "data/textdata/pricing file PPUF_2024Q4.txt"
pricing_df = pd.read_csv(price_file, delimiter='|')

pricing_df['DAYS_SUPPLY'] = pd.to_numeric(pricing_df['DAYS_SUPPLY'], errors='coerce')
pricing_df['UNIT_COST'] = pd.to_numeric(pricing_df['UNIT_COST'], errors='coerce')

#unit cost calculation
avg_unit_cost_by_ndc = pricing_df.groupby('NDC')['UNIT_COST'].mean().reset_index()
avg_unit_cost_by_ndc.rename(columns={'UNIT_COST': 'AVG_UNIT_COST'}, inplace=True)
print("Average Unit Cost by NDC:")
print(avg_unit_cost_by_ndc.head())

avg_unit_cost_by_plan = pricing_df.groupby(['CONTRACT_ID', 'PLAN_ID'])['UNIT_COST'].mean().reset_index()
avg_unit_cost_by_plan.rename(columns={'UNIT_COST': 'PLAN_AVG_UNIT_COST'}, inplace=True)
print("\nAverage Unit Cost by PLAN (by CONTRACT_ID and PLAN_ID):")
print(avg_unit_cost_by_plan.head())
pricing_df['TOTAL_COST'] = pricing_df['UNIT_COST'] * pricing_df['DAYS_SUPPLY']


corr_matrix = pricing_df[['UNIT_COST', 'DAYS_SUPPLY', 'TOTAL_COST']].corr()
print("\nCorrelation Matrix for UNIT_COST, DAYS_SUPPLY, and TOTAL_COST:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(pricing_df['UNIT_COST'], pricing_df['DAYS_SUPPLY'], alpha=0.5, color='teal')
plt.xlabel('Unit Cost')
plt.ylabel('Days Supply')
plt.title('Scatter Plot: Unit Cost vs. Days Supply')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(pricing_df['UNIT_COST'], pricing_df['TOTAL_COST'], alpha=0.5, color='orange')
plt.xlabel('Unit Cost')
plt.ylabel('Total Cost (Unit Cost x Days Supply)')
plt.title('Scatter Plot: Unit Cost vs. Total Cost')
plt.grid(True)
plt.show()