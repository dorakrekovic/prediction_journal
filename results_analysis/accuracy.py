import pandas as pd
import glob

# Find all CSV files in a folder (adjust the pattern as needed)
#csv_files = [f for f in glob.glob('inference_results/Copernicus/*.csv') if 'modelUsed' in f]
csv_files = [f for f in glob.glob('inference_results/Pinova/*.csv') if 'modelUsed' in f]

summary_rows = []
threshold = 1.0  # Celsius

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['abs_error'] = abs(df['real_value'] - df['predicted_value'])
    df['correct'] = df['abs_error'] <= threshold

    total_values = len(df)
    correctly_predicted = df['correct'].sum()
    to_transmit = total_values - correctly_predicted
    reduction_percent = (1-(to_transmit / total_values)) * 100
    accuracy = correctly_predicted / total_values

    # Get location and model from dataframe if available; else, extract from filename
    try:
        location = df['location'].iloc[0]
        model = df['model'].iloc[0]
    except:
        location = 'unknown'
        model = 'unknown'

    summary_rows.append({
        'csv_file': csv_file,
        'location': location,
        'model': model,
        'total': total_values,
        'correctly_predicted': correctly_predicted,
        'to_transmit': to_transmit,
        'reduction_percent': reduction_percent,
        'accuracy': accuracy
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'summary_across_files_{threshold}.csv', index=False)
print("Summary saved to '_summary_a...'")
print(summary_df)
