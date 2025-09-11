import numpy as np
import tensorflow as tf
import pandas as pd
import datetime

TFLITE_MODEL_PATH = "../temp_predictor_int8_new.tflite"
DATA_PATH = "../Suhopolje2021.xlsx"

X_MEAN = 1.577878e+09
X_STD  = 1.823226e+07
Y_MEAN = 12.727435
Y_STD  = 8.472656

# === Load data ===
df = pd.read_excel(DATA_PATH)

# Convert column names if needed
df.columns = df.columns.str.strip().str.lower()
assert 'datetime' in df.columns and 't2m' in df.columns, "Columns must include 'datetime' and 't2m'"


# Convert datetime to normalized timestamp
df['timestamp'] = pd.to_datetime(df['datetime'], dayfirst=True).astype(np.int64) / 1e9
df['timestamp_norm'] = (df['timestamp'] - X_MEAN) / X_STD

# === Load TFLite model ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero = input_details[0]['quantization']
output_scale, output_zero = output_details[0]['quantization']
print("Input scale:", input_scale, "zero:", input_zero)
print("Output scale:", output_scale, "zero:", output_zero)

# === Run inference_results ===
def predict(x_norm):
    x_int8 = np.round(x_norm / input_scale + input_zero).astype(np.int8).reshape(1, 1)
    interpreter.set_tensor(input_details[0]['index'], x_int8)
    interpreter.invoke()
    y_int8 = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize and denormalize
    y_pred = (y_int8.astype(np.float32) - output_zero) * output_scale
    y_pred = y_pred * Y_STD + Y_MEAN
    return y_pred[0][0]

df['predicted_temp'] = df['timestamp_norm'].apply(predict)

# === Evaluation ===
from sklearn.metrics import mean_squared_error

y_true = df['t2m_celsius'].values
y_pred = df['predicted_temp'].values
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.3f} °C")

# === Optional: Save + Plot ===
df.to_excel("suhopolje_predictions.xlsx", index=False)
