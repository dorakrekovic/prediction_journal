import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
csv_file = '../inference_results/Pinova/results_inference_on_Suhopolje_2025-07-15_14-15_2021data_modelUsed_Suhopolje_2025-07-15_10-45_tf.csv'
df = pd.read_csv(csv_file)

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Calculate the absolute difference
df['abs_diff'] = abs(df['real_value'] - df['predicted_value'])


# Define color based on difference thresholds
def color_diff(row):
    if row['abs_diff'] <= 0.5:
        return 'green'
    elif row['abs_diff'] <= 1.0:
        return 'orange'
    else:
        return 'red'

df['color'] = df.apply(color_diff, axis=1)

# Plot settings
plt.figure(figsize=(16, 8))

# Plot the real values as a line
plt.plot(df['datetime'], df['real_value'], label='Real Value (°C)', color='blue', linewidth=2)

# Plot the predicted values with color coding
plt.scatter(df['datetime'], df['predicted_value'], c=df['color'], label='Predicted Value (°C)', s=50)

# Add legends for thresholds
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='green', label='Diff ≤ 0.5°C'),
    mpatches.Patch(color='orange', label='0.5°C < Diff ≤ 1.0°C'),
    mpatches.Patch(color='red', label='Diff > 1.0°C')
]

plt.legend(handles=legend_patches + [plt.Line2D([], [], color='blue', label='Real Value (°C)')])

# Titles and labels
plt.title('Real vs Predicted Temperatures with Error Highlight')
plt.xlabel('Datetime')
plt.ylabel('Temperature (°C)')

# Improve readability
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# Calculate residuals
df['residual'] = df['real_value'] - df['predicted_value']
df['residual_celsius'] = df['residual']  # Kelvin and Celsius difference are the same

# Define custom bins: from -5°C to +5°C, every 0.5°C
custom_bins = np.arange(-5, 5.5, 0.5)

# Plot histogram with custom bins
plt.figure(figsize=(10, 6))
plt.hist(df['residual_celsius'], bins=custom_bins, color='skyblue', edgecolor='black')

# Add mean and std lines
mean_residual = df['residual_celsius'].mean()
std_residual = df['residual_celsius'].std()

plt.axvline(mean_residual, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_residual:.2f}°C')
plt.axvline(mean_residual + std_residual, color='green', linestyle='dashed', linewidth=1, label=f'+1 Std Dev: {mean_residual + std_residual:.2f}°C')
plt.axvline(mean_residual - std_residual, color='green', linestyle='dashed', linewidth=1, label=f'-1 Std Dev: {mean_residual - std_residual:.2f}°C')

# Labels and titles
plt.title('Residual Error Histogram with Custom Bins')
plt.xlabel('Residual Error (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Save or show
plt.tight_layout()
plt.show()

# Convert Kelvin to Celsius for better readability
df['real_c'] = df['real_value']
df['predicted_c'] = df['predicted_value']

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(df['real_c'], df['predicted_c'], alpha=0.5, color='blue', label='Predictions')

# Plot y = x line (perfect prediction)
plt.plot(df['real_c'], df['real_c'], color='red', linestyle='--', label='Perfect Prediction (y = x)')

plt.xlabel('Real Value (°C)')
plt.ylabel('Predicted Value (°C)')
plt.title('Real vs Predicted Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import plotly.express as px

fig = px.scatter(
    df,
    x='datetime',
    y='residual',
    title='Residual Error Over Time (Interactive)',
    labels={'residual': 'Residual Error (K / °C)', 'datetime': 'Datetime'},
    color=df['residual'].apply(lambda x: 'Green' if abs(x) < 0.5 else ('Orange' if abs(x) < 1.0 else 'Red'))
)

fig.update_traces(marker=dict(size=6))
fig.show()

# Create rolling averages (window = 24 for daily average on hourly data)
df['real_smoothed'] = df['real_value'].rolling(window=24).mean()
df['predicted_smoothed'] = df['predicted_value'].rolling(window=24).mean()

# Plot smoothed values
plt.figure(figsize=(16, 6))
plt.plot(df['time'], df['real_smoothed'], label='Real (Smoothed)', color='blue')
plt.plot(df['time'], df['predicted_smoothed'], label='Predicted (Smoothed)', color='green')

plt.title('Real vs Predicted Temperatures (Smoothed)')
plt.xlabel('Datetime')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()