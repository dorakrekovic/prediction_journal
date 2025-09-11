#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf

# -------------------------
# Config
# -------------------------
location = "Suhopolje"
excel_path = f"../{location}2021.xlsx"
tflite_model_path = f"models/RNN_temp_in_C_new.tflite"  # glob pattern OK (will pick latest)
lag = 24
n_ahead = 1
test_share = 0.1
features_final = ["t2m"]

# The 24 latest measured temperatures (°C) for one-step-ahead prediction:
last_24_raw = [
    5.32, 5.35, 5.48, 5.9, 6.39, 6.41, 6.44, 6.5, 6.5, 6.73, 7.6, 8.8,
    8.38, 7.5, 7.3, 6.5, 6.28, 5.82, 5.32, 4.68, 4.5, 3.99, 4.29, 4.5
]
assert len(last_24_raw) == lag, "Need exactly 24 values for a 24-lag model."

# The real next value to compare against:
y_true_next_c = 5.3

# -------------------------
# Helpers
# -------------------------
def load_latest_tflite(path_pattern: str) -> str:
    import glob
    cand = sorted(glob.glob(path_pattern))
    if not cand:
        raise FileNotFoundError(f"No TFLite files found for pattern: {path_pattern}")
    return cand[-1]

def quantize_to_int8(x_norm: np.ndarray, in_details) -> np.ndarray:
    q = in_details["quantization_parameters"]
    if len(q["scales"]) == 0:
        # fallback: treat as unquantized
        return x_norm.astype(np.int8)
    scale = float(q["scales"][0])
    zp    = int(q["zero_points"][0])
    x_q = np.rint(x_norm / scale + zp).astype(np.int8)
    return x_q

def dequantize_from_int8(y_q: np.ndarray, out_details) -> np.ndarray:
    q = out_details["quantization_parameters"]
    if len(q["scales"]) == 0:
        # fallback: treat as unquantized
        return y_q.astype(np.float32)
    scale = float(q["scales"][0])
    zp    = int(q["zero_points"][0])
    return (y_q.astype(np.float32) - zp) * scale

# -------------------------
# 1) Load data to recompute TRAIN normalization stats (same as training)
# -------------------------
d = pd.read_excel(excel_path)
d["datetime"] = pd.to_datetime(d["datetime"], format="%d/%m/%Y %H:%M")
d.sort_values("datetime", inplace=True)
d = d.groupby("datetime", as_index=False)[features_final].mean()

ts = d[features_final]
nrows = ts.shape[0]
train = ts.iloc[: int(nrows * (1 - test_share))].copy()
# use ddof=0 and protect against 0 std to match robust normalization
train_mean = train.mean()
train_std  = train.std(ddof=0).replace(0, 1.0)

t_mean = float(train_mean["t2m"])
t_std  = float(train_std["t2m"])

# -------------------------
# 2) Prepare input window
# -------------------------
x_raw = np.array(last_24_raw, dtype=np.float32).reshape(1, lag, 1)   # shape [1, 24, 1]
x_norm = (x_raw - t_mean) / t_std                                    # normalize

# -------------------------
# 3) Load TFLite model and infer
# -------------------------
tflite_path = load_latest_tflite(tflite_model_path)
print(f"Using TFLite model: {tflite_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print("Input spec:", in_det["shape"], in_det["dtype"])
print("Output spec:", out_det["shape"], out_det["dtype"])

# Prepare input according to I/O type
if in_det["dtype"] == np.float32:
    x_in = x_norm.astype(np.float32)
elif in_det["dtype"] == np.int8:
    x_in = quantize_to_int8(x_norm.astype(np.float32), in_det)
else:
    raise TypeError(f"Unsupported input dtype: {in_det['dtype']}")

interpreter.set_tensor(in_det["index"], x_in)
interpreter.invoke()
y_out = interpreter.get_tensor(out_det["index"])   # shape [1, 1] for n_ahead=1

# Convert output back to normalized float if needed
if out_det["dtype"] == np.float32:
    y_norm = y_out.astype(np.float32)
elif out_det["dtype"] == np.int8:
    y_norm = dequantize_from_int8(y_out, out_det)
else:
    raise TypeError(f"Unsupported output dtype: {out_det['dtype']}")

# Denormalize to °C
y_pred_c = float(y_norm[0, 0] * t_std + t_mean)

# -------------------------
# 4) Compare to the provided true value and print difference
# -------------------------
diff = y_pred_c - y_true_next_c

print(f"\nOne-step ahead prediction from the 24 inputs:")
print(f"Predicted next value: {y_pred_c:.3f} °C")
print(f"Actual next value:    {y_true_next_c:.3f} °C")
print(f"Difference (pred-true): {diff:+.3f} °C")
