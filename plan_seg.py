import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

basic_formulary_file = "data/textdata/basic drugs formulary file  PPUF_2024Q4.txt"
basic_df = pd.read_csv(basic_formulary_file, delimiter='|')

drug_features = basic_df[['TIER_LEVEL_VALUE', 'QUANTITY_LIMIT_AMOUNT']].copy()

drug_features['TIER_LEVEL_VALUE'] = pd.to_numeric(drug_features['TIER_LEVEL_VALUE'], errors='coerce')
drug_features['QUANTITY_LIMIT_AMOUNT'] = pd.to_numeric(drug_features['QUANTITY_LIMIT_AMOUNT'], errors='coerce')
drug_features.dropna(inplace=True)


scaler = StandardScaler()
drug_features_scaled = scaler.fit_transform(drug_features)

kmeans_drug = KMeans(n_clusters=3, random_state=42)
drug_clusters = kmeans_drug.fit_predict(drug_features_scaled)

drug_features_df = drug_features.copy()
drug_features_df['Cluster'] = drug_clusters

plt.figure(figsize=(8, 6))
scatter = plt.scatter(drug_features_df['TIER_LEVEL_VALUE'],
                      drug_features_df['QUANTITY_LIMIT_AMOUNT'],
                      c=drug_features_df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('TIER_LEVEL_VALUE')
plt.ylabel('QUANTITY_LIMIT_AMOUNT')
plt.title('Drug Clustering based on Tier Level and Quantity Limit')
plt.colorbar(scatter, label='Cluster')
plt.show()


centroids_scaled = kmeans_drug.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
print("Drug Clustering Centroids (Original Scale):")
print(centroids)


beneficiary_cost_file = "data/textdata/beneficiary cost file  PPUF_2024Q4.txt"
ben_df = pd.read_csv(beneficiary_cost_file, delimiter='|' , encoding="ISO-8859-1")


cost_columns = ['COST_AMT_PREF', 'COST_AMT_NONPREF', 'COST_AMT_MAIL_PREF', 'COST_AMT_MAIL_NONPREF']
for col in cost_columns:
    ben_df[col] = pd.to_numeric(ben_df[col], errors='coerce')


ben_df['AVG_COST'] = ben_df[cost_columns].mean(axis=1)
plan_avg_cost = ben_df.groupby('PLAN_ID')['AVG_COST'].mean().reset_index()
plan_avg_cost.rename(columns={'AVG_COST': 'PLAN_AVG_COST'}, inplace=True)

plan_info_file = "data/textdata/plan information  PPUF_2024Q4.txt"  
plan_df = pd.read_csv(plan_info_file, delimiter='|' , encoding="ISO-8859-1")

plan_df['PREMIUM'] = pd.to_numeric(plan_df['PREMIUM'], errors='coerce')
plan_df['DEDUCTIBLE'] = pd.to_numeric(plan_df['DEDUCTIBLE'], errors='coerce')



plan_segmentation_df = pd.merge(plan_df, plan_avg_cost, on='PLAN_ID', how='left')

segmentation_features = plan_segmentation_df[['PLAN_AVG_COST', 'PREMIUM', 'DEDUCTIBLE']].copy()

segmentation_features.dropna(inplace=True)

scaler_plan = StandardScaler()
seg_features_scaled = scaler_plan.fit_transform(segmentation_features)

kmeans_plan = KMeans(n_clusters=3, random_state=42)
plan_clusters = kmeans_plan.fit_predict(seg_features_scaled)

segmentation_features['Cluster'] = plan_clusters

plan_segmentation_df = plan_segmentation_df.loc[segmentation_features.index].copy()
plan_segmentation_df['Cluster'] = plan_clusters

print("Plan Segmentation Cluster Summary:")
cluster_summary = plan_segmentation_df.groupby('Cluster')[['PLAN_AVG_COST', 'PREMIUM', 'DEDUCTIBLE']].mean()
print(cluster_summary)

sns.pairplot(segmentation_features, hue='Cluster', diag_kind='kde')
plt.suptitle("Plan Segmentation Clusters", y=1.02)
plt.show()