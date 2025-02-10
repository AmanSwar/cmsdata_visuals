import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

geo_file = "data/textdata/geographic locator file PPUF_2024Q4.txt"
geo_df = pd.read_csv(geo_file, delimiter='|' , encoding="ISO-8859-1")

plan_info_file = "data/textdata/plan information  PPUF_2024Q4.txt"
plan_info_df = pd.read_csv(plan_info_file, delimiter='|' , encoding="ISO-8859-1")

plan_info_df['PREMIUM'] = pd.to_numeric(plan_info_df['PREMIUM'], errors='coerce')
plan_info_df['DEDUCTIBLE'] = pd.to_numeric(plan_info_df['DEDUCTIBLE'], errors='coerce')

geo_merged_df = pd.merge(plan_info_df, geo_df, on='COUNTY_CODE', how='left')

regional_summary = geo_merged_df.groupby('STATENAME')[['PREMIUM', 'DEDUCTIBLE']].mean().reset_index()
print("\nRegional Summary by STATENAME (Average PREMIUM and DEDUCTIBLE):")
print(regional_summary)

plt.figure(figsize=(12, 6))
plt.bar(regional_summary['STATENAME'], regional_summary['PREMIUM'], color='steelblue')
plt.xlabel('State')
plt.ylabel('Average Premium')
plt.title('Average Premium by State')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(regional_summary['STATENAME'], regional_summary['DEDUCTIBLE'], color='salmon')
plt.xlabel('State')
plt.ylabel('Average Deductible')
plt.title('Average Deductible by State')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
