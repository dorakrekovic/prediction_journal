import os

import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
from tf_1h import create_X_Y, train_mean, train_std, n_ahead, lag
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

location = "Nasice"
saved_model_path = f'models/Nasice_2025-07-18_12-27_tf_1h.h5'
# Get filename
filename = os.path.basename(saved_model_path)

# Remove extension
name_without_ext = os.path.splitext(filename)[0]
print(name_without_ext)

# Load saved model
inference_model = load_model(saved_model_path)

# Load and preprocess 2021 data
data_2021 = pd.read_csv(f'data/{location}2021.csv')
data_2021['datetime'] = pd.to_datetime(data_2021['datetime'], format='%d/%m/%Y %H:%M')
data_2021.sort_values('datetime', inplace=True)

features_final = ['t2m']
data_2021 = data_2021.groupby('datetime', as_index=False)[features_final].mean()

# Scale using training mean and std
data_2021_scaled = (data_2021[features_final] - train_mean) / train_std

# Prepare X and Y for inference_results
X_infer, Y_real = create_X_Y(data_2021_scaled.values, lag=lag, n_ahead=n_ahead)
datetime_infer = data_2021['datetime'].values[lag + n_ahead : ]

# Run inference_results
predictions_scaled = inference_model.predict(X_infer)

# Invert scaling
predictions_unscaled = (predictions_scaled.flatten() * train_std['t2m']) + train_mean['t2m']
Y_real_unscaled = (Y_real.flatten() * train_std['t2m']) + train_mean['t2m']

# Save predictions and real values with timestamps
results_inference_df = pd.DataFrame({
    'location': location,
    'model': name_without_ext,
    'datetime': datetime_infer,
    'real_value': Y_real_unscaled,
    'predicted_value': predictions_unscaled
})

results_inference_df.to_csv(f'inference_results/results_inference_on_{location}_{time_stamp}_2021data_modelUsed_{name_without_ext}.csv', index=False)

print(f"Inference results saved successfully to results_inference_on_{location}_{time_stamp}_2021data_modelUsed_{name_without_ext}.csv'")
