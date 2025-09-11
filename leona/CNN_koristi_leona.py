import os
from math import sqrt
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
location = "Suhopolje"
os.makedirs("models", exist_ok=True)
saved_model_path = f'models/{location}_{time_stamp}_CNN.h5'
filename = os.path.basename(saved_model_path)
name_without_ext = os.path.splitext(filename)[0]


args = {
    'input_dim': 1,
    'past_observation': 24,
    'batch_size': 32,
    'optimizer': 'Adam',
    'loss_function': 'MSELoss',
    'num_epochs': 50,
    'model': 'CNN1D'
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
n_filters = 32
kernel_size = 3
dropout_rate = 0.1
features_final = ['t2m']

ts = d[features_final]
nrows = ts.shape[0]
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]


train_mean = train.mean()
train_std = train.std(ddof=0).replace(0, 1.0)
print("Normalization stats")
for col in train_mean.index:
    print(f"{col}: mean={train_mean[col]:.6f}, std={train_std[col]:.6f}")

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

ts_s = pd.concat([train, test])
X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
n_ft = X.shape[2]

cut = int(X.shape[0] * (1 - test_share))
Xtrain, Ytrain = X[:cut], Y[:cut]
Xval, Yval = X[cut:], Y[cut:]

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of validation data: {Xval.shape}")


def representative_dataset():
    for i in range(min(100, len(Xtrain))):
        yield [np.expand_dims(Xtrain[i].astype(np.float32), axis=0)]

# CNN model
def make_cnn(n_lag, n_ft, n_outputs, n_filters=32, kernel_size=3, dropout_rate=0.1):
    inp = Input(shape=(n_lag, n_ft))
    # causal padding keeps "future" from leaking into the past within each window
    x = Conv1D(n_filters, kernel_size, activation='relu', padding='causal')(inp)
    x = Conv1D(n_filters, kernel_size, activation='relu', padding='causal')(x)
    x = GlobalAveragePooling1D()(x)
    if dropout_rate and dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    out = Dense(n_outputs)(x)
    return Model(inp, out)

class PredictionModel():
    def __init__(
        self, X, Y, n_outputs, n_lag, n_ft,
        n_filters, kernel_size, dropout_rate,
        batch, epochs, lr, Xval=None, Yval=None
    ):
        self.model = make_cnn(
            n_lag=n_lag, n_ft=n_ft, n_outputs=n_outputs,
            n_filters=n_filters, kernel_size=kernel_size, dropout_rate=dropout_rate
        )
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y

    def modelSave(self):
        print(f"Model checkpoint path: {saved_model_path}")
        return ModelCheckpoint(saved_model_path, monitor='val_loss', mode='min', save_best_only=True)

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
            callbacks=callbacks,
            verbose=1
        )
        return self.model

def main():
    model_obj = PredictionModel(
        X=Xtrain, Y=Ytrain,
        n_outputs=n_ahead, n_lag=lag, n_ft=n_ft,
        n_filters=n_filters, kernel_size=kernel_size, dropout_rate=dropout_rate,
        batch=batch_size, epochs=epochs, lr=lr,
        Xval=Xval, Yval=Yval
    )
    model = model_obj.train()


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # enables post-training quant options
    converter.representative_dataset = representative_dataset


    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    #  Default (float32 I/O) with weight quant
    tflite_model = converter.convert()
    tflite_model_path = saved_model_path.replace(".h5", "_float32.tflite")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model (BUILTINS only, float I/O) saved at: {tflite_model_path}")

    # Full-INT8
    #converter.inference_input_type = tf.int8
    #converter.inference_output_type = tf.int8
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #tflite_int8 = converter.convert()
    #tflite_int8_path = saved_model_path.replace(".h5", "_int8.tflite")
    # with open(tflite_int8_path, "wb") as f:
    #    f.write(tflite_int8)
    #print(f"TFLite model (BUILTINS_INT8, int8 I/O) saved at: {tflite_int8_path}")

if __name__ == "__main__":
    main()
