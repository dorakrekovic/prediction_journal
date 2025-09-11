import os
from math import sqrt
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
location = "Suhopolje"
saved_model_path = f'models/{location}_{time_stamp}_RNN_temp_in_C.h5'
filename = os.path.basename(saved_model_path)
name_without_ext = os.path.splitext(filename)[0]

args = {
    'input_dim': 1,
    'past_observation': 24,
    'batch_size': 32,
    'optimizer': 'Adam',
    'loss_function': 'MSELoss',
    'num_epochs': 50,
}


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    n_features = ts.shape[1]
    X, Y = [], []
    for i in range(len(ts) - lag - n_ahead):
        Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
        X.append(ts[i:(i + lag)])
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    return X, Y

d = pd.read_excel(f'../data_management/{location}_19_20.xlsx')
d['datetime'] = pd.to_datetime(d['datetime'], format='%d/%m/%Y %H:%M')
d.sort_values('datetime', inplace=True)

features = ['t2m']
d = d.groupby('datetime', as_index=False)[features].mean()

lag = 24
n_ahead = 1
test_share = 0.1
epochs = 50
batch_size = 32
lr = 0.001
n_layer = 50
features_final = ['t2m']

ts = d[features_final]
nrows = ts.shape[0]
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]

train_mean = train.mean()
train_std = train.std()

print("Normalization stats")
for col in train_mean.index:
    print(f"{col}: mean={train_mean[col]:.6f}, std={train_std[col]:.6f}")

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

ts_s = pd.concat([train, test])
X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
n_ft = X.shape[2]

Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of validation data: {Xval.shape}")

def representative_dataset():
    for i in range(min(100, len(Xtrain))):
        yield [np.expand_dims(Xtrain[i], axis=0).astype(np.float32)]

class PredictionModel():
    def __init__(self, X, Y, n_outputs, n_lag, n_ft, n_layer, batch, epochs, lr, Xval=None, Yval=None):
        rnn_input = Input(shape=(n_lag, n_ft))
        rnn_layer = SimpleRNN(n_layer, activation='relu', unroll=True)(rnn_input)
        x = Dense(n_outputs)(rnn_layer)
        self.model = Model(inputs=rnn_input, outputs=x)
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y

    def modelSave(self):
        print(f"Model saved at {saved_model_path}")
        return ModelCheckpoint(saved_model_path, monitor='loss', mode='min', save_best_only=True)

    def train(self):
        optimizer = Adam(learning_rate=self.lr)
        self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=optimizer)
        callbacks = [self.modelSave()]
        self.model.fit(
            self.X, self.Y,
            epochs=self.epochs,
            batch_size=self.batch,
            validation_data=(self.Xval, self.Yval),
            shuffle=False,
            callbacks=callbacks
        )
        return self.model

def main():
    model_obj = PredictionModel(
        X=Xtrain, Y=Ytrain,
        n_outputs=n_ahead, n_lag=lag, n_ft=n_ft,
        n_layer=n_layer, batch=batch_size, epochs=epochs,
        lr=lr, Xval=Xval, Yval=Yval
    )
    model = model_obj.train()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    try:
        tflite_model = converter.convert()
        tflite_model_path = saved_model_path.replace(".h5", "_new.tflite")
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model with Select TF Ops saved at: {tflite_model_path}")
    except Exception as e:
        print("TFLite conversion failed:", e)


if __name__ == "__main__":
    main()
