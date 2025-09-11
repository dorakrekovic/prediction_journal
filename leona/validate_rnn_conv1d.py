import numpy as np
import pandas as pd
from datetime import datetime
from math import sqrt
import matplotlib.pyplot as plt
import tensorflow as tf


location = "Suhopolje"
d = pd.read_csv(f'{location}2021.csv')
d['datetime'] = [datetime.strptime(x, '%d/%m/%Y %H:%M') for x in d['datetime']]
d.sort_values('datetime', inplace=True)
d = d.groupby('datetime', as_index=False)[['t2m']].mean()

lag = 24
n_ahead = 1
test_share = 0.1

def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0):
    n_features = ts.shape[1]
    X, Y = [], []
    for i in range(len(ts) - lag - n_ahead):
        Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
        X.append(ts[i:(i + lag)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y

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

def evaluate_float_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true, y_pred = [], []

    for i in range(len(Xval)):
        input_data = Xval[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        y_true.append(Yval[i][0])
        y_pred.append(output_data[0][0])

    return y_true, y_pred

def evaluate_int8_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    y_true, y_pred = [], []

    for i in range(len(Xval)):
        input_data = Xval[i:i+1].astype(np.float32)
        quant_input = (input_data / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], quant_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        dequant_output = (output_data.astype(np.float32) - output_zero_point) * output_scale

        y_true.append(Yval[i][0])
        y_pred.append(dequant_output[0][0])

    return y_true, y_pred

def inverse_scale(data):
    return [(x * train_std['t2m']) + train_mean['t2m'] for x in data]

def evaluate_and_plot(model_path, is_int8=True, label=""):
    if is_int8:
        y_true, y_pred = evaluate_int8_model(model_path)
    else:
        y_true, y_pred = evaluate_float_model(model_path)

    y_true_unscaled = inverse_scale(y_true)
    y_pred_unscaled = inverse_scale(y_pred)

    rmse = sqrt(np.mean([(a - b) ** 2 for a, b in zip(y_true_unscaled, y_pred_unscaled)]))
    print(f"{label} RMSE: {rmse:.4f} C")

    plt.plot(y_true_unscaled[-100:], label=f"{label} - Actual", linewidth=2)
    plt.plot(y_pred_unscaled[-100:], label=f"{label} - Predicted", linestyle='--')


plt.figure(figsize=(12, 6))
evaluate_and_plot("rnn_selecttf.tflite", is_int8=False, label="RNN Float")
evaluate_and_plot("conv1D_int8.tflite", is_int8=True, label="Conv1D INT8")
plt.title("Comparison: RNN vs Conv1D INT8 (Last 100 samples)")
plt.xlabel("Time Step")
plt.ylabel("t2m (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()