#!/usr/bin/env python3
import os
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

# -------------------------
# Config
# -------------------------
location = "Suhopolje"
excel_path = f"../../{location}2021.xlsx"
tflite_path = f"../models/RNN_temp_in_C_new.tflite"  # exact path or glob pattern
inf_type = "invoke"  # label saved to CSV
lag = 24
n_ahead = 1
features_final = ["t2m"]

# Normalization stats source:
#   "train" -> first 90% of rows (recommended; avoids leakage)
#   "all"   -> all rows (leaks info, but sometimes desired)
NORMALIZE_ON = "train"      # "train" or "all"
train_stats_share = 0.9

save_csv = True
csv_out = f"{location}_rnn_full_results.csv"  # (unused here but kept)
summary_csv = "summary_inference.csv"         # <--- summary file you asked for

# Optional benchmarking knobs
WARMUP_RUNS = 5         # do a few warmups before timing (0 to disable)
PRINT_PER_SAMPLE = False  # set True to print each inference time

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
    # Accept exact file OR glob pattern
    if os.path.exists(path_pattern):
        return path_pattern
    cand = sorted(glob.glob(path_pattern))
    if not cand:
        raise FileNotFoundError(f"No TFLite files found for: {path_pattern}")
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

# Normalize entire dataset with chosen stats
ts_norm = (ts - train_mean) / train_std

# -------------------------
# 3) Build supervised samples for the WHOLE file
# -------------------------
X_all, Y_all = create_X_Y(ts_norm.values, lag=lag, n_ahead=n_ahead)

# Timestamps aligned to supervised samples (create_X_Y drops first lag + n_ahead)
ts_datetimes_all = d["datetime"].iloc[lag + n_ahead :].reset_index(drop=True)
assert len(ts_datetimes_all) == len(X_all), "Timestamp alignment mismatch."

print(f"Total supervised samples: {X_all.shape[0]} (windows {X_all.shape[1]}×{X_all.shape[2]})")

# -------------------------
# 4) Load TFLite model
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
# 5) Inference over ALL timestamps + timing (invoke-only per-sample)
# -------------------------
preds_real = []
times_ms = []  # collect per-sample invoke-only times

# Warm-up (optional)
if WARMUP_RUNS and len(X_all) > 0:
    x_warm = X_all[0:1].astype(np.float32)
    x_warm = x_warm if in_det["dtype"] == np.float32 else quantize_to_int8(x_warm, in_det)
    for _ in range(WARMUP_RUNS):
        interpreter.set_tensor(in_det["index"], x_warm)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_det["index"])

# Total loop timing
start_total = time.perf_counter()

for i in range(len(X_all)):
    x_norm = X_all[i:i+1].astype(np.float32)  # [1, lag, 1]

    # Input dtype handling
    if in_det["dtype"] == np.float32:
        x_in = x_norm
    elif in_det["dtype"] == np.int8:
        x_in = quantize_to_int8(x_norm, in_det)
    else:
        raise TypeError(f"Unsupported input dtype: {in_det['dtype']}")

    interpreter.set_tensor(in_det["index"], x_in)

    # Measure pure model inference time
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    times_ms.append(ms)
    if PRINT_PER_SAMPLE:
        print(f"Inference {i}: {ms:.4f} ms")

    y_out = interpreter.get_tensor(out_det["index"])  # [1, n_ahead]

    # Output dtype handling
    if out_det["dtype"] == np.float32:
        y_norm = y_out.astype(np.float32)
    elif out_det["dtype"] == np.int8:
        y_norm = dequantize_from_int8(y_out, out_det)
    else:
        raise TypeError(f"Unsupported output dtype: {out_det['dtype']}")

    # Denormalize to °C
    y_real = y_norm * float(train_std["t2m"]) + float(train_mean["t2m"])
    preds_real.append(float(y_real[0, 0]))

end_total = time.perf_counter()
total_s = end_total - start_total
preds_real = np.array(preds_real, dtype=np.float32)

# -------------------------
# 6) Print loop-level summary
# -------------------------
if len(times_ms) > 0:
    a = np.array(times_ms, dtype=np.float64)
    avg_ms = (total_s / len(times_ms)) * 1000.0
    thr = len(times_ms) / total_s if total_s > 0 else float("inf")

    print(f"\nTotal inference loop duration (all {len(times_ms)} samples): {total_s:.4f} s")
    print(f"Average per-sample (invoke only): {avg_ms:.4f} ms")
    print(f"Throughput: {thr:.4f} samples/sec")
else:
    a = np.array([], dtype=np.float64)
    print("\nNo samples to run inference on.")

# -------------------------
# 7) Save SUMMARY CSV (one row appended)
# -------------------------
if save_csv:
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_row = {
        "model_name": os.path.basename(tflite_model_path),
        "run_timestamp": run_timestamp,
        "inference_type": inf_type,
        "total_samples": int(len(preds_real)),
        "total_wall_ms": round(total_s * 1000.0, 4),
        "avg_ms": round(a.mean(), 4) if len(a) else None,
        "min_ms": round(a.min(), 4) if len(a) else None,
        "p50_ms": round(np.percentile(a, 50), 4) if len(a) else None,
        "p90_ms": round(np.percentile(a, 90), 4) if len(a) else None,
        "p95_ms": round(np.percentile(a, 95), 4) if len(a) else None,
        "p99_ms": round(np.percentile(a, 99), 4) if len(a) else None,
        "max_ms": round(a.max(), 4) if len(a) else None,
    }

    if os.path.exists(summary_csv):
        df_summary = pd.read_csv(summary_csv)
        df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([summary_row])

    df_summary.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"\nSaved summary row to: {summary_csv}")
