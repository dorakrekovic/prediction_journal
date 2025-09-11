import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime

# === Load and prepare data ===
location = "Suhopolje"
data_path = f"data/{location}_19_20.csv"
d = pd.read_csv(data_path)
d['datetime'] = [datetime.strptime(x, '%d/%m/%Y %H:%M') for x in d['datetime']]
d.sort_values('datetime', inplace=True)
d = d.groupby('datetime', as_index=False)[['t2m']].mean()

# Parameters
lag = 24
n_ahead = 1
test_share = 0.1

# Create sequences
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0):
    n_features = ts.shape[1]
    X, Y = [], []
    for i in range(len(ts) - lag - n_ahead):
        Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
        X.append(ts[i:(i + lag)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y
# Normalize and split
ts = d[['t2m']]
nrows = ts.shape[0]
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]
train_mean = train.mean()
train_std = train.std()
train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

ts_s = pd.concat([train, test])
X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
Xval = X[int(X.shape[0] * (1 - test_share)):]
Yval = Y[int(Y.shape[0] * (1 - test_share)):]

# === Load TFLite model ===
tflite_model_path = "../models/lstm.tflite"  # Update timestamp
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input type:", input_details[0]['dtype'])
print("Output type:", output_details[0]['dtype'])

# === Run inference_results ===
y_true = []
y_pred = []

for i in range(len(Xval)):
    input_data = Xval[i:i+1].astype(np.float32)  # no quantization
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_true.append(Yval[i][0])
    y_pred.append(output_data[0][0])

# Invert scaling
y_true_unscaled = [(x * train_std['t2m']) + train_mean['t2m'] for x in y_true]
y_pred_unscaled = [(x * train_std['t2m']) + train_mean['t2m'] for x in y_pred]

# Calculate RMSE
rmse = sqrt(np.mean([(a - b)**2 for a, b in zip(y_true_unscaled, y_pred_unscaled)]))
print(f"TFLite RMSE: {rmse:.4f} C")

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(y_true_unscaled[-100:], label="Original", linewidth=2)
plt.plot(y_pred_unscaled[-100:], label="TFLite Forecast", linestyle='--')
plt.title(f"Forecast vs Actual - Last 100 Samples ({location})")
plt.xlabel("Time Step")
plt.ylabel("t2m (\u00b0C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()