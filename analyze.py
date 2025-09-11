import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assume you have multiple CSVs for each location
csv_files = [
    ('Kamenac', 'inference_results/Pinova/results_inference_on_Kamenac_2025-07-15_14-17_2021data_modelUsed_Kamenac_2025-07-15_11-18_tf.csv'),
    ('Nasice', 'inference_results/Pinova/results_inference_on_Nasice_OBZ_2025-07-15_14-37_2021data_modelUsed_Nasice_OBZ_2025-07-15_14-06_tf.csv'),
    ('Nedelisce', 'inference_results/Pinova/results_inference_on_Nedelisce_2025-07-15_14-19_2021data_modelUsed_Nedelisce_2025-07-15_11-41_tf.csv'),
    ('Oklaj', 'inference_results/Pinova/results_inference_on_Oklaj_2025-07-15_14-21_2021data_modelUsed_Oklaj_2025-07-15_11-32_tf.csv'),
    ('Potomje', 'inference_results/Pinova/results_inference_on_Potomje_2025-07-15_15-44_2021data_modelUsed_Potomje_2025-07-15_14-33_tf.csv'),
    ('Skenderovci', 'inference_results/Pinova/results_inference_on_Skenderovci_2025-07-15_14-22_2021data_modelUsed_Skenderovci_2025-07-15_11-12_tf.csv'),
    ('Suhopolje', 'inference_results/Pinova/results_inference_on_Suhopolje_2025-07-15_14-15_2021data_modelUsed_Suhopolje_2025-07-15_10-45_tf.csv'),

]

dfs = []
for location, file in csv_files:
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['location'] = location  # Add the location label
    df['abs_diff'] = abs(df['real_value'] - df['predicted_value'])
    dfs.append(df)

# Combine all into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: convert to Celsius for residual analysis
combined_df['real_celsius'] = combined_df['real_value']
combined_df['predicted_celsius'] = combined_df['predicted_value']
combined_df['residual_celsius'] = combined_df['real_celsius'] - combined_df['predicted_celsius']

# Define categories based on abs_diff
def categorize_diff(diff):
    if diff <= 0.5:
        return '≤ 0.5°C'
    elif diff <= 1.0:
        return '0.5°C - 1.0°C'
    else:
        return '> 1.0°C'

combined_df['error_category'] = combined_df['abs_diff'].apply(categorize_diff)

# Count number of predictions per category per location
category_counts = combined_df.groupby(['location', 'error_category']).size().unstack(fill_value=0)

# Optional: percentage instead of counts
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100

category_counts.plot(kind='bar', stacked=True, figsize=(12, 6))

plt.title('Prediction Error Categories per Location (Counts)')
plt.xlabel('Location')
plt.ylabel('Number of Predictions')
plt.legend(title='Error Category')
plt.grid(True)
plt.tight_layout()
plt.show()

category_percentages.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20c')

plt.title('Prediction Error Categories per Location (Percentages)')
plt.xlabel('Location')
plt.ylabel('Percentage of Predictions (%)')
plt.legend(title='Error Category')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
combined_df.boxplot(column='residual_celsius', by='location', grid=True)

plt.title('Residual Error Distribution per Location')
plt.suptitle('')  # Suppress the automatic title
plt.xlabel('Location')
plt.ylabel('Residual Error (°C)')
plt.tight_layout()
plt.show()

import plotly.express as px

fig = px.histogram(
    combined_df,
    x='abs_diff',
    color='location',
    nbins=50,
    title='Absolute Prediction Errors by Location',
    labels={'abs_diff': 'Absolute Error (°C)'},
    barmode='overlay'
)
fig.update_layout(bargap=0.1)
fig.show()


# Save the counts to CSV
category_counts.to_csv('error_category_counts_per_location_Pinova.csv')

# Save the percentages to CSV
category_percentages.to_csv('error_category_percentages_per_location_Pinova.csv')

print("CSV files have been saved successfully!")
combined_df.to_csv('combined_predictions_with_error_categories_Pinova.csv', index=False)

print("Combined predictions dataframe saved as CSV!")


