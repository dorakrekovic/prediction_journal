import pandas as pd
import glob
import os

# === Settings ===
thresholds = [0.5, 1.0]
input_folder = 'inference_results/Copernicus/'
output_excel = 'summary_with_mae_Copenicus.xlsx'


file_pattern = '*.csv'

# === Load Files ===
csv_files = glob.glob(os.path.join(input_folder, file_pattern))
if not csv_files:
    print("⚠️ No CSV files found. Check the path or folder name.")
    exit()

all_rows = []

for threshold in thresholds:
    print(f"\n--- Evaluating for threshold ±{threshold}°C ---")
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue

        # Check required columns
        if not {'real_value', 'predicted_value'}.issubset(df.columns):
            print(f"⚠️ Skipping {file_path}: missing required columns.")
            continue

        # Error calculations
        df['abs_error'] = abs(df['real_value'] - df['predicted_value'])
        mae = df['abs_error'].mean()
        df['within_threshold'] = df['abs_error'] <= threshold
        df['should_transmit'] = df['abs_error'] > threshold

        total = len(df)
        correct = df['within_threshold'].sum()
        to_transmit = df['should_transmit'].sum()

        accuracy = (correct / total) * 100
        data_reduction = (1 - to_transmit / total) * 100
        transmission_precision = (to_transmit / to_transmit) * 100 if to_transmit > 0 else None

        # Metadata extraction
        try:
            location = df['location'].iloc[0]
            model = df['model'].iloc[0]
        except:
            parts = os.path.basename(file_path).replace('.csv', '').split('_')
            location = parts[0] if len(parts) > 0 else 'unknown'
            model = parts[1] if len(parts) > 1 else 'unknown'

        all_rows.append({
            'threshold': threshold,
            'file': os.path.basename(file_path),
            'location': location,
            'model': model,
            'total_predictions': total,
            'predicted_within_threshold': correct,
            'prediction_accuracy (%)': round(accuracy, 2),
            'mean_absolute_error': round(mae, 3),
            'data_reduction (%)': round(data_reduction, 2),
            'transmission_precision (%)': round(transmission_precision, 2) if transmission_precision is not None else 'N/A'
        })

# === Save Summary ===
summary_df = pd.DataFrame(all_rows)

with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, index=False, sheet_name='Summary')

print(f"\n✅ Evaluation complete. Results saved to '{output_excel}'")
print(summary_df.head())

