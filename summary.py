import pandas as pd
import glob
import os

# Settings
thresholds = [0.5, 1.0]
input_folder = 'inference_results/Pinova/'
file_pattern = '*.csv'
output_excel = 'summary_with_mae.xlsx'


# Gather all CSV files
csv_files = glob.glob(os.path.join(input_folder, file_pattern))

all_summaries = []

for threshold in thresholds:
    print(f"\n--- Processing threshold: {threshold} ---")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        if 'real_value' not in df.columns or 'predicted_value' not in df.columns:
            print(f"Missing required columns in {csv_file}, skipping.")
            continue

        # Compute errors and transmission decision
        df['abs_error'] = abs(df['real_value'] - df['predicted_value'])
        df['within_threshold'] = df['abs_error'] <= threshold
        df['should_transmit'] = df['abs_error'] > threshold

        total = len(df)
        correctly_predicted = df['within_threshold'].sum()
        to_transmit = df['should_transmit'].sum()

        # Metrics
        prediction_accuracy = correctly_predicted / total  # % of correct predictions
        data_reduction = (1 - to_transmit / total) * 100    # % of avoided transmissions

        # Transmission precision: how many of the transmitted were truly needed
        if to_transmit > 0:
            transmission_precision = df['should_transmit'].sum() / to_transmit
        else:
            transmission_precision = None

        # Extract metadata
        try:
            location = df['location'].iloc[0]
            model = df['model'].iloc[0]
        except:
            parts = os.path.basename(csv_file).replace('.csv', '').split('_')
            location = parts[0] if len(parts) > 0 else 'unknown'
            model = parts[1] if len(parts) > 1 else 'unknown'

        all_summaries.append({
            'threshold': threshold,
            'csv_file': os.path.basename(csv_file),
            'location': location,
            'model': model,
            'total_predictions': total,
            'predicted_within_threshold': correctly_predicted,
            'prediction_accuracy': round(prediction_accuracy * 100, 2),  # as percentage
            'to_transmit': to_transmit,
            'data_reduction_percent': round(data_reduction, 2),
            'transmission_precision': round(transmission_precision * 100, 2) if transmission_precision is not None else 'N/A'
        })
# Create summary DataFrame
summary_df = pd.DataFrame(all_summaries)
with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, index=False, sheet_name='Summary')

print(f"\n✅ Evaluation complete. Results saved to '{output_excel}'")
print(summary_df.head())
