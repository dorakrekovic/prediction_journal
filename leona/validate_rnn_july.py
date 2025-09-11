#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt

# -------------------------
# Config
# -------------------------
location = "Suhopolje"
excel_path = f"../{location}2021.xlsx"
tflite_path = f"models/RNN_temp_in_C_new.tflite"
lag = 24
n_ahead = 1
features_final = ["t2m"]

# Normalization: "train" = first 90% (recommended, no leakage), "all" = whole file
NORMALIZE_ON = "train"
train_stats_share = 0.9

# Filter month/year
MONTH = 4
YEAR_FILTER = None   # set to e.g. 2021 to filter July 2021 only

save_csv = True
csv_out = f"{location}_rnn_april_results.csv"

# -------------------------
# Helpers
# -------------------------
def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0):
    n_features = ts.shape[1]
    X, Y = [], []
    for i in range(len(ts) - lag - n_ahead):
        Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
        X.append(ts[i:(i + lag)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y

def load_latest_tflite(path_pattern: str) -> str:
    if os.path.exists(path_pattern):
        return path_pattern
    cand = sorted(glob.glob(path_pattern))
    if not cand:
        raise FileNotFoundError(f"No TFLite model found: {path_pattern}")
    return cand[-1]

def quantize_to_int8(x_norm: np.ndarray, in_details) -> np.ndarray:
    q = in_details["quantization_parameters"]
    scale = float(q["scales"][0]) if len(q["scales"]) else 1.0
    zp    = int(q["zero_points"][0]) if len(q["zero_points"]) else 0
    return np.rint(x_norm / scale + zp).astype(np.int8)

def dequantize_from_int8(y_q: np.ndarray, out_details) -> np.ndarray:
    q = out_details["quantization_parameters"]
    scale = float(q["scales"][0]) if len(q["scales"]) else 1.0
    zp    = int(q["zero_points"][0]) if len(q["zero_points"]) else 0
    return (y_q.astype(np.float32) - zp) * scale

# -------------------------
# 1) Load data
# -------------------------
d = pd.read_excel(excel_path)
d["datetime"] = pd.to_datetime(d["datetime"], format="%d/%m/%Y %H:%M")
d.sort_values("datetime", inplace=True)
d = d.groupby("datetime", as_index=False)[features_final].mean()

ts = d[features_final].copy()
nrows = ts.shape[0]

# -------------------------
# 2) Normalization stats
# -------------------------
if NORMALIZE_ON == "train":
    cut_stats = int(nrows * train_stats_share)
    train_for_stats = ts.iloc[:cut_stats].copy()
    train_mean = train_for_stats.mean()
    train_std  = train_for_stats.std(ddof=0).replace(0, 1.0)
elif NORMALIZE_ON == "all":
    train_mean = ts.mean()
    train_std  = ts.std(ddof=0).replace(0, 1.0)
else:
    raise ValueError("NORMALIZE_ON must be 'train' or 'all'.")

print("Normalization stats:")
for col in train_mean.index:
    print(f"{col}: mean={train_mean[col]:.6f}, std={train_std[col]:.6f}")

ts_norm = (ts - train_mean) / train_std

# -------------------------
# 3) Build supervised samples
# -------------------------
X_all, Y_all = create_X_Y(ts_norm.values, lag=lag, n_ahead=n_ahead)
ts_datetimes_all = d["datetime"].iloc[lag + n_ahead :].reset_index(drop=True)
assert len(X_all) == len(ts_datetimes_all)

# -------------------------
# 4) Filter to July
# -------------------------
mask = ts_datetimes_all.dt.month.eq(MONTH)
if YEAR_FILTER is not None:
    mask &= ts_datetimes_all.dt.year.eq(YEAR_FILTER)

idxs = np.where(mask.values)[0]
if len(idxs) == 0:
    raise ValueError("No July samples found.")

X_sub = X_all[idxs]
Y_sub = Y_all[idxs]
times_sub = ts_datetimes_all.iloc[idxs].reset_index(drop=True)

print(f"Total supervised samples: {len(X_all)}; July samples: {len(X_sub)}")

# -------------------------
# 5) Load TFLite model
# -------------------------
tflite_model_path = load_latest_tflite(tflite_path)
print(f"Using TFLite model: {tflite_model_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print("Input spec:", in_det["shape"], in_det["dtype"])
print("Output spec:", out_det["shape"], out_det["dtype"])

# -------------------------
# 6) Inference
# -------------------------
preds_real = []
for i in range(len(X_sub)):
    x_norm = X_sub[i:i+1].astype(np.float32)

    if in_det["dtype"] == np.float32:
        x_in = x_norm
    elif in_det["dtype"] == np.int8:
        x_in = quantize_to_int8(x_norm, in_det)
    else:
        raise TypeError(f"Unsupported input dtype {in_det['dtype']}")

    interpreter.set_tensor(in_det["index"], x_in)
    interpreter.invoke()
    y_out = interpreter.get_tensor(out_det["index"])

    if out_det["dtype"] == np.float32:
        y_norm = y_out.astype(np.float32)
    elif out_det["dtype"] == np.int8:
        y_norm = dequantize_from_int8(y_out, out_det)
    else:
        raise TypeError(f"Unsupported output dtype {out_det['dtype']}")

    y_real = y_norm * float(train_std["t2m"]) + float(train_mean["t2m"])
    preds_real.append(float(y_real[0, 0]))

preds_real = np.array(preds_real, dtype=np.float32)
ytrue_real = (Y_sub.astype(np.float32) * float(train_std["t2m"]) + float(train_mean["t2m"])).ravel()

# -------------------------
# 7) Metrics
# -------------------------
mae  = np.mean(np.abs(preds_real - ytrue_real))
rmse = np.sqrt(np.mean((preds_real - ytrue_real) ** 2))
mape = np.mean(np.abs((preds_real - ytrue_real) / np.clip(np.abs(ytrue_real), 1e-6, None))) * 100

print(f"\nJULY EVALUATION")
print(f"MAE :  {mae:.3f} °C")
print(f"RMSE:  {rmse:.3f} °C")
print(f"MAPE:  {mape:.2f} %\n")

print("First 10 July predictions:")
for k in range(min(10, len(preds_real))):
    print(f"{k:3d}: time={times_sub[k]} | true={ytrue_real[k]:6.2f} °C | pred={preds_real[k]:6.2f} °C | err={preds_real[k]-ytrue_real[k]:+6.2f} °C")

# -------------------------
# 8) Save CSV
# -------------------------
if save_csv:
    out_df = pd.DataFrame({
        "datetime": times_sub,
        "y_true_C": ytrue_real,
        "y_pred_C": preds_real,
        "error_C": preds_real - ytrue_real,
    })
    out_df.to_csv(csv_out, index=False)
    print(f"\nSaved July results to: {csv_out}")
