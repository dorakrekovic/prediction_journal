thresholds = [0.5, 1.0, 2.0]
import pandas as pd
import glob
import os

folder_path = '../inference_results/Pinova/'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

all_accuracy_results = []

for file_path in csv_files:
    df = pd.read_csv(file_path)

    if 'real_value' not in df.columns or 'predicted_value' not in df.columns:
        print(f"Skipping file (missing required columns): {file_path}")
        continue

    df['abs_error'] = abs(df['real_value'] - df['predicted_value'])

    accuracy_results = []

    for thresh in thresholds:
        colname = f'correct_{thresh}C'
        df[colname] = df['abs_error'] <= thresh
        acc_table = (
            df.groupby(['location', 'model'])[colname]
                .mean()
                .reset_index()
                .rename(columns={colname: 'accuracy'})
        )
        acc_table['threshold_C'] = thresh
        acc_table['source_file'] = os.path.basename(file_path)
        accuracy_results.append(acc_table)

    file_accuracy = pd.concat(accuracy_results, ignore_index=True)
    all_accuracy_results.append(file_accuracy)

# Combine results for all files
final_accuracy = pd.concat(all_accuracy_results, ignore_index=True)

# Convert to percentage
final_accuracy['accuracy_%'] = final_accuracy['accuracy'] * 100

# Optional: drop 'accuracy' (the fraction)
final_accuracy = final_accuracy.drop(columns=['accuracy'])

# Print or save
print(final_accuracy)
final_accuracy.to_csv('combined_accuracy_results_Pinova_same_location.csv', index=False)