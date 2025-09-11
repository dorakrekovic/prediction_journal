import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_excel("data_celsius.xlsx")
df['timestamp'] = pd.to_datetime(df['datetime']).astype(np.int64) / 1e9  # seconds
X = df[['timestamp']].values.astype(np.float32)
y = df['temp'].values.astype(np.float32)

# Normalize input
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std


# Normalize output
y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / y_std

print(f"X_mean = {X_mean:.6e}")
print(f"X_std  = {X_std:.6e}")
print(f"Y_mean = {y_mean:.6f}")
print(f"Y_std  = {y_std:.6f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# Build smallest possible model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), name='input'),
    tf.keras.layers.Dense(1, activation='linear', name='fc')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50)

# Representative dataset for int8 quantization
def rep_data():
    for x in X_train[:100]:
        yield [x.reshape(1, 1)]

# Convert to TFLite with full int8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("../temp_predictor_int8_new.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as temp_predictor_int8_new.tflite")
