#!/usr/bin/env python3
import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt

# -------------------------
# Config
# -------------------------
location = "Suhopolje"
excel_path = f"../{location}2021.xlsx"

# Exact path or glob pattern; both work with load_latest_tflite()
tflite_path_pattern = "models/CNN_int8.tflite"

features_final = ["t2m"]
lag = 24
n_ahead = 1

# How to compute normalization stats:
# - "train": use first (1 - train_stats_share) fraction of rows (recommended, avoids leakage)
# - "all":   use the entire dataset (leaks info, but sometimes desired)
NORMALIZE_ON = "train"      # "train" or "all"
train_stats_share = 0.9     # used only when NORMALIZE_ON == "train"

save_csv = True
csv_out = f"{location}_cnn_full_results.csv"

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

def load_latest_tflite(pattern: str) -> str:
    cand = sorted(glob.glob(pattern))
    if not cand:
        # If exact path provided and not globbed, still allow it:
        if os.path.exists(pattern):
            return pattern
        raise FileNotFoundError(f"No TFLite files found for pattern/path: {pattern}")
    return cand[-1]

def quantize_to_int8(x_norm: np.ndarray, in_details) -> np.ndarray:
    q = in_details["quantization_parameters"]
    if len(q["scales"]) == 0:
        return x_norm.astype(np.int8)
    scale = float(q["scales"][0])
    zp    = int(q["zero_points"][0])
    return np.rint(x_norm / scale + zp).astype(np.int8)

def dequantize_from_int8(y_q: np.ndarray, out_details) -> np.ndarray:
    q = out_details["quantization_parameters"]
    if len(q["scales"]) == 0:
        return y_q.astype(np.float32)
    scale = float(q["scales"][0])
    zp    = int(q["zero_points"][0])
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
# 2) Compute normalization stats
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

print("Normalization stats used:")
for col in train_mean.index:
    print(f"{col}: mean={train_mean[col]:.6f}, std={train_std[col]:.6f}")

# Normalize whole dataset using chosen stats
ts_norm = (ts - train_mean) / train_std

# -------------------------
# 3) Build supervised samples OVER THE WHOLE FILE
# -------------------------
X_all, Y_all = create_X_Y(ts_norm.values, lag=lag, n_ahead=n_ahead)

# Timestamps aligned to supervised samples (create_X_Y drops first lag+n_ahead)
ts_datetimes_all = d["datetime"].iloc[lag + n_ahead :].reset_index(drop=True)

assert len(ts_datetimes_all) == len(X_all), "Timestamp alignment mismatch."

print(f"Total supervised samples: {X_all.shape} | Timestamps: {len(ts_datetimes_all)}")

# -------------------------
# 4) Load TFLite model
# -------------------------
tflite_model_path = load_latest_tflite(tflite_path_pattern)
print(f"Using TFLite model: {tflite_model_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print("Input spec:", in_det["shape"], in_det["dtype"])
print("Output spec:", out_det["shape"], out_det["dtype"])

# -------------------------
# 5) Inference over ALL samples + timing
# -------------------------
preds_real = []
times_ms = []  # per-invoke() times (ms)

# Optional warm-up for stability
if len(X_all) > 0:
    xw = X_all[0:1].astype(np.float32)
    xw = xw if in_det["dtype"] == np.float32 else quantize_to_int8(xw, in_det)
    interpreter.set_tensor(in_det["index"], xw)
    interpreter.invoke()
    _ = interpreter.get_tensor(out_det["index"])

start_total = time.perf_counter()

for i in range(len(X_all)):
    x_norm = X_all[i:i+1].astype(np.float32)  # [1, lag, n_features]

    # Input dtype handling
    if in_det["dtype"] == np.float32:
        x_in = x_norm
    elif in_det["dtype"] == np.int8:
        x_in = quantize_to_int8(x_norm, in_det)
    else:
        raise TypeError(f"Unsupported input dtype: {in_det['dtype']}")

    interpreter.set_tensor(in_det["index"], x_in)

    # Measure pure model time
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000.0)

    y_out = interpreter.get_tensor(out_det["index"])  # [1, n_ahead]

    # Output dtype handling
    if out_det["dtype"] == np.float32:
        y_norm = y_out.astype(np.float32)
    elif out_det["dtype"] == np.int8:
        y_norm = dequantize_from_int8(y_out, out_det)
    else:
        raise TypeError(f"Unsupported output dtype: {out_det['dtype']}")

    # Denormalize to °C using the same stats used for normalization
    y_real = y_norm * float(train_std["t2m"]) + float(train_mean["t2m"])
    preds_real.append(float(y_real[0, 0]))

end_total = time.perf_counter()
total_s = end_total - start_total
preds_real = np.array(preds_real, dtype=np.float32)

# Timing summary
print(f"\nTotal inference loop duration (all {len(X_all)} samples): {total_s:.4f} s")
if len(X_all) > 0 and total_s > 0:
    print(f"Average per-sample (invoke only): {total_s/len(X_all)*1000:.4f} ms")
    print(f"Throughput: {len(X_all)/total_s:.4f} samples/sec")
    print(f"Per-inference (invoke) — Avg: {np.mean(times_ms):.4f} ms | "
          f"Min: {np.min(times_ms):.4f} ms | Max: {np.max(times_ms):.4f} ms")

# -------------------------
# Ground truth in °C (denormalize all Y)
# -------------------------
ytrue_real = (Y_all.astype(np.float32) * float(train_std["t2m"]) + float(train_mean["t2m"])).ravel()

# -------------------------
# 6) Metrics (over the whole dataset’s supervised samples)
# -------------------------
mae  = float(np.mean(np.abs(preds_real - ytrue_real)))
rmse = float(sqrt(np.mean((preds_real - ytrue_real) ** 2)))
mape = float(np.mean(np.abs((preds_real - ytrue_real) / np.clip(np.abs(ytrue_real), 1e-6, None))) * 100.0)

print(f"\nFULL-DATASET EVALUATION")
print(f"MAE :  {mae:.3f} °C")
print(f"RMSE:  {rmse:.3f} °C")
print(f"MAPE:  {mape:.2f} %\n")

# Show first 10 comparisons
print("First 10 comparisons:")
for k in range(min(10, len(preds_real))):
    print(f"{k:3d}: time={ts_datetimes_all[k]} | y_true={ytrue_real[k]:6.2f} °C | "
          f"y_pred={preds_real[k]:6.2f} °C | err={preds_real[k]-ytrue_real[k]:+6.2f} °C")

# -------------------------
# 7) Save CSV (whole dataset)
# -------------------------
if save_csv:
    out_df = pd.DataFrame({
        "datetime": ts_datetimes_all,
        "y_true_C": ytrue_real,
        "y_pred_C": preds_real,
        "error_C": preds_real - ytrue_real,
    })
    out_df.to_csv(csv_out, index=False)
    print(f"\nSaved results for ALL samples to: {csv_out}")
