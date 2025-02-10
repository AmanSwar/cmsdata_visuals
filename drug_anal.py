import pandas as pd
import matplotlib.pyplot as plt

formulary_file = "data/textdata/basic drugs formulary file  PPUF_2024Q4.txt"
exl_form_file = "data/textdata/excluded drugs formulary file  PPUF_2024Q4.txt"
indic_file = "data/textdata/Indication Based Coverage Formulary File  PPUF_2024Q4.txt"

basic_df = pd.read_csv(formulary_file, delimiter='|' , encoding="ISO-8859-1")
excluded_df = pd.read_csv(exl_form_file, delimiter='|' , encoding="ISO-8859-1")
indication_df = pd.read_csv(indic_file, delimiter='|' , encoding="ISO-8859-1")

tier_distribution = basic_df['TIER_LEVEL_VALUE'].value_counts().sort_index()


print("\nDistribution of TIER_LEVEL_VALUE in Basic Formulary:")
print(tier_distribution)

plt.figure(figsize=(8, 5))

tier_distribution.plot(kind='bar', color='skyblue')
plt.xlabel('Tier Level Value')
plt.ylabel('Number of Drugs')
plt.title('Distribution of Tier Level Value in Basic Drugs Formulary')
plt.xticks(rotation=0)
plt.tight_layout()


plt.show()


print("\nQuantity Limit (Y/N) Counts:")
print(basic_df['QUANTITY_LIMIT_YN'].value_counts())

print("\nPrior Authorization (Y/N) Counts:")
print(basic_df['PRIOR_AUTHORIZATION_YN'].value_counts())

print("\nStep Therapy (Y/N) Counts:")
print(basic_df['STEP_THERAPY_YN'].value_counts())


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
basic_df['QUANTITY_LIMIT_YN'].value_counts().plot(kind='bar', ax=axs[0], color='lightgreen')
axs[0].set_title('Quantity Limit (Y/N)')
axs[0].set_xlabel('')
basic_df['PRIOR_AUTHORIZATION_YN'].value_counts().plot(kind='bar', ax=axs[1], color='lightcoral')
axs[1].set_title('Prior Authorization (Y/N)')
axs[1].set_xlabel('')
basic_df['STEP_THERAPY_YN'].value_counts().plot(kind='bar', ax=axs[2], color='lightskyblue')
axs[2].set_title('Step Therapy (Y/N)')
axs[2].set_xlabel('')
plt.tight_layout()
plt.show()


print("\nSummary for QUANTITY_LIMIT_AMOUNT (Basic Formulary):")
print(basic_df['QUANTITY_LIMIT_AMOUNT'].describe())

print("\nSummary for QUANTITY_LIMIT_AMOUNT (Excluded Formulary):")
print(excluded_df['QUANTITY_LIMIT_AMOUNT'].describe())


print("\nSummary for TIER_LEVEL_VALUE (Basic Formulary):")
print(basic_df['TIER_LEVEL_VALUE'].describe())

print("\nValue counts for TIER (Excluded Formulary):")
print(excluded_df['TIER'].value_counts())

