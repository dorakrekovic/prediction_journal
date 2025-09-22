# rnn_full_float32_tflite.py
import os
from math import sqrt
import json
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Config
# -----------------------------
location = "Suhopolje"
data_path = f'../data_management/{location}_19_20.xlsx'
features = ['t2m']
features_final = ['t2m']  # model uses just this feature
lag = 24
n_ahead = 1
test_share = 0.1
epochs = 50
batch_size = 32
lr = 0.001
n_layer = 50

# Output paths
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
saved_model_path = os.path.join(models_dir, f'WASM_{time_stamp}_RNN.h5')
artifacts_path = saved_model_path.replace(".h5", "_artifacts.json")
tflite_model_path = saved_model_path.replace(".h5", "_float32.tflite")

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# -----------------------------
# Utils
# -----------------------------
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    n_features = ts.shape[1]
    X, Y = [], []
    for i in range(len(ts) - lag - n_ahead + 1):
        Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
        X.append(ts[i:(i + lag)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y

def build_model(n_lag, n_ft, n_layer, n_outputs):
    rnn_input = Input(shape=(n_lag, n_ft))
    rnn_layer = SimpleRNN(n_layer, activation='relu', unroll=True)(rnn_input)
    x = Dense(n_outputs)(rnn_layer)
    model = Model(inputs=rnn_input, outputs=x)
    return model

# -----------------------------
# Load & prep data
# -----------------------------
d = pd.read_excel(data_path)
# Ensure datetime parsing matches your file format
d['datetime'] = pd.to_datetime(d['datetime'], format='%d/%m/%Y %H:%M')
d.sort_values('datetime', inplace=True)

# Resample/aggregate if needed (here: mean on duplicates)
d = d.groupby('datetime', as_index=False)[features].mean()

ts = d[features_final].copy()
nrows = ts.shape[0]
split_idx = int(nrows * (1 - test_share))
train = ts.iloc[:split_idx]
test  = ts.iloc[split_idx:]

train_mean = train.mean()
train_std = train.std(ddof=0)  # population-style; keep consistent later

print("Normalization stats")
for col in train_mean.index:
    print(f"{col}: mean={train_mean[col]:.6f}, std={train_std[col]:.6f}")

train_norm = (train - train_mean) / train_std.replace(0, 1.0)
test_norm  = (test  - train_mean) / train_std.replace(0, 1.0)

ts_s = pd.concat([train_norm, test_norm])  # keep chronology
X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
n_ft = X.shape[2]

Xtrain = X[:int(X.shape[0] * (1 - test_share))]
Ytrain = Y[:int(Y.shape[0] * (1 - test_share))]
Xval   = X[int(X.shape[0] * (1 - test_share)):]
Yval   = Y[int(Y.shape[0] * (1 - test_share)):]

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of validation data: {Xval.shape}")

# -----------------------------
# Train
# -----------------------------
model = build_model(lag, n_ft, n_layer, n_ahead)
optimizer = Adam(learning_rate=lr)
model.compile(loss=tf.losses.MeanSquaredError(), optimizer=optimizer)

ckpt = ModelCheckpoint(
    saved_model_path,
    monitor='val_loss',  # prefer val_loss to keep best generalization
    mode='min',
    save_best_only=True,
    save_weights_only=False
)
history = model.fit(
    Xtrain, Ytrain,
    validation_data=(Xval, Yval),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=False,
    callbacks=[ckpt],
    verbose=2
)

# Load best checkpoint before conversion
best_model = tf.keras.models.load_model(saved_model_path)

# -----------------------------
# Save artifacts (normalization, shapes, etc.)
# -----------------------------
artifacts = {
    "train_mean": {k: float(v) for k, v in train_mean.to_dict().items()},
    "train_std":  {k: float(v) for k, v in train_std.to_dict().items()},
    "lag": lag,
    "n_features": int(n_ft),
    "n_ahead": n_ahead,
    "feature_order": features_final,
    "input_shape": [None, lag, int(n_ft)],  # batch, time, features
    "output_shape": [None, n_ahead],
    "dtype": "float32",
    "note": "Inputs/outputs are normalized. De-normalize outputs using y = y_norm*std + mean on feature 't2m'."
}
with open(artifacts_path, "w") as f:
    json.dump(artifacts, f, indent=2)
print(f"Saved artifacts to: {artifacts_path}")

# -----------------------------
# Full-precision (float32) TFLite conversion (NO optimizations)
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
# KEEP FULL PRECISION: do not set converter.optimizations
# converter.optimizations = []  # (optional; default is already no-optimizations)

tflite_model = converter.convert()
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved full-precision TFLite model to: {tflite_model_path}")

# -----------------------------
# Parity check: Keras vs TFLite on validation windows
# -----------------------------
def tflite_predict_float32(interpreter, Xbatch):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    # Ensure float32 and correct shape
    Xbatch = Xbatch.astype(np.float32)
    preds = []
    for i in range(Xbatch.shape[0]):
        x = Xbatch[i:i+1]  # (1, lag, n_ft)
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details['index'])  # (1, n_ahead)
        preds.append(y[0])
    return np.array(preds, dtype=np.float32)

# Keras predictions (normalized)
y_keras = best_model.predict(Xval, batch_size=256, verbose=0)

# TFLite predictions (normalized)
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
y_tflite = tflite_predict_float32(interpreter, Xval)

# MSE in normalized space
mse_norm = np.mean((y_keras - y_tflite) ** 2)
rmse_norm = sqrt(mse_norm)
print(f"[Parity] Keras vs TFLite normalized RMSE: {rmse_norm:.6e}")

# Example: de-normalize first 5 predictions back to °C for sanity check
mean_t2m = artifacts["train_mean"]["t2m"]
std_t2m  = artifacts["train_std"]["t2m"]
y_keras_c  = y_keras[:, 0] * std_t2m + mean_t2m
y_tflite_c = y_tflite[:, 0] * std_t2m + mean_t2m
print("First 5 predictions (°C) — Keras vs TFLite:")
for i in range(min(5, len(y_keras_c))):
    print(f"{i:02d}: {y_keras_c[i]:.4f} °C   |   {y_tflite_c[i]:.4f} °C")

print("\nDone. Inference I/O (both Keras and TFLite):")
print(f"  Input  shape: (batch, {lag}, {n_ft})  dtype=float32  (standardized)")
print(f"  Output shape: (batch, {n_ahead})       dtype=float32  (standardized)")
print("Remember to de-normalize outputs back to °C using artifacts' mean/std.")
