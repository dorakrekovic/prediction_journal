import os
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
from train.tf_1h import create_X_Y, train_mean, train_std, n_ahead, lag
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

location = "Kamenac"
saved_model_path = f'../models/Copernicus/Kamenac_2025-07-15_17-56_tf_1h.h5'
# Get filename
filename = os.path.basename(saved_model_path)

# Remove extension
name_without_ext = os.path.splitext(filename)[0]
print(name_without_ext)

# Load saved model
inference_model = load_model(saved_model_path)

train_mean = train_mean.rename({'t2m': 'temperature'}) - 273.15
train_std = train_std.rename({'t2m': 'temperature'})

# Load and preprocess 2021 data
data_2021 = pd.read_excel(f'data/Pinova/{location}_2021_Pinova.xlsx')
data_2021['time'] = pd.to_datetime(data_2021['time'], format='%d/%m/%Y %H:%M')
data_2021.sort_values('time', inplace=True)

features_final = ['temperature']
# Set time as index
data_2021.set_index('time', inplace=True)

# Resample to hourly means
data_2021 = data_2021[features_final].resample('1H').mean().dropna().reset_index()


# Scale using training mean and std
data_2021_scaled = (data_2021[features_final] - train_mean) / train_std

# Prepare X and Y for inference_results
data_scaled_array = data_2021_scaled[['temperature']].values  # Ensures shape (n, 1)


X_infer, Y_real = create_X_Y(data_scaled_array, lag=lag, n_ahead=n_ahead)

datetime_infer = data_2021['time'].values[lag + n_ahead : ]
#X_infer = X_infer[:, :, 0:1]
#print("X_infer shape:", X_infer.shape)  # Should be (samples, time_steps, 1)
# Run inference_results

predictions_scaled = inference_model.predict(X_infer)


# Invert scaling
predictions_unscaled = (predictions_scaled.flatten() * train_std['temperature']) + train_mean['temperature']
Y_real_unscaled = (Y_real.flatten() * train_std['temperature']) + train_mean['temperature']

# Save predictions and real values with timestamps
results_inference_df = pd.DataFrame({
    'location': location,
    'model': name_without_ext,
    'datetime': datetime_infer,
    'real_value': Y_real_unscaled,
    'predicted_value': predictions_unscaled
})

results_inference_df.to_csv(f'inference_results/Copernicus_train_Pinova_inf/results_inference_on_{location}_{time_stamp}_2021data_modelUsed_{name_without_ext}.csv', index=False)

print(f"Inference results saved successfully to results_inference_on_{location}_{time_stamp}_2021data_modelUsed_{name_without_ext}.csv'")
