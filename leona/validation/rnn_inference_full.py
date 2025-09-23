#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

script_start = time.perf_counter()
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# ======================
# Config
# ======================
hardware = "jetson"
location = "Suhopolje"
excel_path = f"../../{location}2021.xlsx"
tflite_path = "../models/RNN_temp_in_C_new.tflite"   # exact path or glob pattern
inf_type = "e2e"                                     # label in the summary CSV

features_final = ["t2m"]
lag = 24
n_ahead = 1

# Normalization stats source (match training!)
NORMALIZE_ON = "train"     # "train" or "all"
train_stats_share = 0.9

# CSV outputs
save_csv = True
csv_out_detailed = f"{location}_rnn_full_e2e_results.csv"   # per-sample details
csv_out_summary  = "summary_inference_NEW.csv"                  # one row per run

# Optional benchmarking knobs
WARMUP_RUNS = 5           # warmups before timing (0 to disable)
PRINT_PER_SAMPLE = False  # True = print each sample's E2E time

# ======================
# Helpers
# ======================
def load_latest_tflite(path_pattern: str) -> str:
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

def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else None

# ======================
# 1) Load RAW data
# ======================
d = pd.read_excel(excel_path)
d["datetime"] = pd.to_datetime(d["datetime"], format="%d/%m/%Y %H:%M")
d.sort_values("datetime", inplace=True)
d = d.groupby("datetime", as_index=False)[features_final].mean()

ts = d[features_final].copy()  # RAW °C
nrows = ts.shape[0]

# ======================
# 2) Compute normalization stats (match training)
# ======================
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

# Also build normalized series for aligning targets
ts_norm = (ts - train_mean) / train_std

# ======================
# 3) Supervised indexing (no X build to avoid extra memory)
# ======================
num_samples = nrows - lag - n_ahead
if num_samples <= 0:
    raise ValueError("Not enough rows for the chosen lag and n_ahead.")

# Targets (normalized)
Y_all_norm = ts_norm.values[lag : nrows - n_ahead, 0][:, None]  # (num_samples, 1)
ts_datetimes_all = d["datetime"].iloc[lag + n_ahead :].reset_index(drop=True)

# Sanity checks
assert Y_all_norm.shape[0] == num_samples
assert len(ts_datetimes_all) == num_samples

# ======================
# 4) Load TFLite model
# ======================
tflite_model_path = load_latest_tflite(tflite_path)
print(f"Using TFLite model: {tflite_model_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print("Input spec :", in_det["shape"], in_det["dtype"])
print("Output spec:", out_det["shape"], out_det["dtype"])

# ======================
# 5) E2E per-sample timing from RAW → prediction
#    (raw -> normalize -> (quantize?) -> set/invoke/get -> (dequantize?) -> denormalize)
# ======================
preds_real = []
t_prep_ms, t_io_ms, t_post_ms, t_total_ms = [], [], [], []

# Warm-up (optional)
if WARMUP_RUNS and nrows > lag:
    x_raw_w = ts.values[:lag].astype(np.float32).reshape(1, lag, 1)
    x_norm_w = (x_raw_w - float(train_mean["t2m"])) / float(train_std["t2m"])
    x_in_w = x_norm_w if in_det["dtype"] == np.float32 else quantize_to_int8(x_norm_w, in_det)
    for _ in range(WARMUP_RUNS):
        interpreter.set_tensor(in_det["index"], x_in_w)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_det["index"])

# Total wall-clock around the entire loop
loop_t0 = time.perf_counter_ns()

for i in range(num_samples):
    # ---------- start full E2E ----------
    t0 = time.perf_counter_ns()

    # RAW window (°C)
    x_raw = ts.values[i : i + lag].astype(np.float32).reshape(1, lag, 1)

    # normalize per-sample (prep)
    x_norm = (x_raw - float(train_mean["t2m"])) / float(train_std["t2m"])
    # prepare input dtype
    if in_det["dtype"] == np.float32:
        x_in = x_norm.astype(np.float32)
    elif in_det["dtype"] == np.int8:
        x_in = quantize_to_int8(x_norm.astype(np.float32), in_det)
    else:
        raise TypeError(f"Unsupported input dtype: {in_det['dtype']}")
    t1 = time.perf_counter_ns()
    t_prep_ms.append((t1 - t0) / 1e6)

    # io + compute
    t2 = time.perf_counter_ns()
    interpreter.set_tensor(in_det["index"], x_in)
    interpreter.invoke()
    y_out = interpreter.get_tensor(out_det["index"])
    t3 = time.perf_counter_ns()
    t_io_ms.append((t3 - t2) / 1e6)

    # postprocess (dequantize if needed; then denormalize to °C)
    if out_det["dtype"] == np.float32:
        y_norm = y_out.astype(np.float32)
    elif out_det["dtype"] == np.int8:
        y_norm = dequantize_from_int8(y_out, out_det)
    else:
        raise TypeError(f"Unsupported output dtype: {out_det['dtype']}")

    y_real = y_norm * float(train_std["t2m"]) + float(train_mean["t2m"])
    preds_real.append(float(y_real[0, 0]))
    t4 = time.perf_counter_ns()
    t_post_ms.append((t4 - t3) / 1e6)

    # total per-sample
    e2e = (t4 - t0) / 1e6
    t_total_ms.append(e2e)

    if PRINT_PER_SAMPLE:
        print(f"{i:6d}  total_e2e={e2e:.4f} ms  (prep={t_prep_ms[-1]:.4f}, "
              f"io={t_io_ms[-1]:.4f}, post={t_post_ms[-1]:.4f})")

loop_t1 = time.perf_counter_ns()
total_wall_ms = (loop_t1 - loop_t0) / 1e6

preds_real = np.array(preds_real, dtype=np.float32)
ytrue_real = (Y_all_norm.astype(np.float32) * float(train_std["t2m"]) + float(train_mean["t2m"])).ravel()

# Sanity checks
assert len(preds_real) == num_samples
assert len(ytrue_real) == num_samples
assert len(ts_datetimes_all) == num_samples

# ======================
# Timing summary
# ======================
def stats(name, arr_ms):
    if not arr_ms:
        print(f"{name}: no samples")
        return
    a = np.array(arr_ms, dtype=np.float64)
    print(f"{name:14s} avg={a.mean():8.4f} ms | min={a.min():8.4f} | "
          f"p50={np.percentile(a,50):8.4f} | p90={np.percentile(a,90):8.4f} | "
          f"p95={np.percentile(a,95):8.4f} | p99={np.percentile(a,99):8.4f} | "
          f"max={a.max():8.4f}")

print(f"\nTotal wall time (ALL samples): {total_wall_ms:.4f} ms for {len(t_total_ms)} samples")
if total_wall_ms > 0 and len(t_total_ms) > 0:
    print(f"Throughput (E2E): {len(t_total_ms) / (total_wall_ms/1000.0):.2f} samples/sec")
stats("preprocess", t_prep_ms)
stats("io+compute", t_io_ms)
stats("postprocess", t_post_ms)
stats("total_e2e", t_total_ms)

# ======================
# 6) Save CSVs (rounded to 4 decimals)
# ======================

if save_csv and len(preds_real) > 0:
    # per-sample detailed
    detailed_df = pd.DataFrame({
        "datetime": ts_datetimes_all,
        "y_true_C": np.round(ytrue_real, 4),
        "y_pred_C": np.round(preds_real, 4),
        "error_C":  np.round(preds_real - ytrue_real, 4),
        "prep_ms":       np.round(t_prep_ms, 4),
        "io_compute_ms": np.round(t_io_ms, 4),
        "post_ms":       np.round(t_post_ms, 4),
        "total_e2e_ms":  np.round(t_total_ms, 4),
    })
    detailed_df.to_csv(csv_out_detailed, index=False, float_format="%.4f")
    print(f"Saved detailed results to: {csv_out_detailed}")

    # summary (appendable one row)
    a_tot = np.array(t_total_ms, dtype=np.float64)
    script_end = time.perf_counter()
    summary_row = {
        "model_name": os.path.basename(tflite_model_path),
        "run_timestamp": run_timestamp,
        "inference_type": inf_type,                     # "e2e"
        "total_samples": int(len(a_tot)),
        "total_wall_ms": round(total_wall_ms, 4),       # total time for ALL samples
        "avg_ms": round(a_tot.mean(), 4) if len(a_tot) else None,
        "min_ms": round(a_tot.min(), 4) if len(a_tot) else None,
        "p50_ms": round(pct(a_tot, 50), 4) if len(a_tot) else None,
        "p90_ms": round(pct(a_tot, 90), 4) if len(a_tot) else None,
        "p95_ms": round(pct(a_tot, 95), 4) if len(a_tot) else None,
        "p99_ms": round(pct(a_tot, 99), 4) if len(a_tot) else None,
        "max_ms": round(a_tot.max(), 4) if len(a_tot) else None,
        "end_time": f"{script_end - script_start:.4f}s",
        "end_timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "hw": hardware,
    }

    if os.path.exists(csv_out_summary):
        df_sum = pd.read_csv(csv_out_summary)
        df_sum = pd.concat([df_sum, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df_sum = pd.DataFrame([summary_row])

    df_sum.to_csv(csv_out_summary, index=False, float_format="%.4f")
    print(f"Saved summary to: {csv_out_summary}")


script_end2 = time.perf_counter()

print(f"\n[Script Runtime] {script_end2 - script_start:.4f} seconds total")

