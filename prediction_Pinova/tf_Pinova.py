import os
from math import sqrt
import pandas as pd
#import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import wandb

from wandb.integration.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
location = "Nasice_OBZ"
saved_model_path = f'../models/Pinova/{location}_{time_stamp}_tf.h5'

filename = os.path.basename(saved_model_path)
name_without_ext = os.path.splitext(filename)[0]
print(name_without_ext)
args = {
    'input_dim': 1,
    'past_observation': 24,
    'batch_size': 32,
    'optimizer': 'Adam',
    'loss_function': 'MSELoss',
    'num_epochs': 50,
}
wandb.init(project="AIOT_Pinova", config=args)

def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    # Izdvajanje značajki iz niza
    n_features = ts.shape[1]

    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y


d = pd.read_excel(f'../data/Pinova/{location}_19_20_Pinova.xlsx')
#print(f'Shape of data: {d.shape}')
#print(d.dtypes)

#d['time'] = [datetime.strptime(x, '%d/%m/%Y %H:%M:%S') for x in d['time']]
#d.sort_values('time', inplace=True)

d['time'] = pd.to_datetime(d['time'], dayfirst=True)  # Safely parses strings or timestamps
d['time'] = d['time'].dt.floor('min')


# Korištene značajke
features = ['temperature']

# Satna razina
#d = d.groupby('time', as_index=False)[features].mean()
#print(d[features].describe())

# Broj lagova koristenih u modelu
lag = 24

# Broj koraka unaprijed  koji se predvidaju
n_ahead = 1

# Udio skupa korišten za testiranje
test_share = 0.1

# Broj epoha za treniranje
epochs = 50

# Batch size
batch_size = 32

# Stopa ucenja
lr = 0.001

# Broj neurona u LSTM sloju
n_layer = 50

features_final = ['temperature']


# Koristeni stupci za predvidanje
ts = d[features_final]

nrows = ts.shape[0]


# Podjela u skup za treniranje i testiranje
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]

# Skaliranje podataka
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

ts_s = pd.concat([train, test])
X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)

n_ft = X.shape[2]

# Podjela u skup za treniranje i testiranje
Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of the target data: {Ytrain.shape}")

print(f"Shape of validation data: {Xval.shape}")
print(f"Shape of the validation target data: {Yval.shape}")

time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')



class PredictionModel():
    def __init__( self, X, Y, n_outputs, n_lag, n_ft, n_layer, batch, epochs, lr, Xval=None, Yval=None,
                  min_delta=0.001, patience=10):
        lstm_input = Input(shape=(n_lag, n_ft))
        #lstm_input = Input(shape=(24, 1))  # timesteps=24, features=1
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)
        x = Dense(n_outputs)(lstm_layer)
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch
        self.epochs = epochs
        self.n_layer = n_layer
        self.lr = lr
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.min_delta = min_delta
        self.patience = patience

    #def trainCallback(self):
        #return EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta)

    def modelSave(self):
        print(f"Model saved at {saved_model_path}")
        return ModelCheckpoint(saved_model_path, monitor='loss', mode='min', save_best_only=True)

    def train(self):
        empty_model = self.model
        optimizer = Adam(learning_rate=self.lr)
        empty_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=optimizer)
        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.Xval, self.Yval),
                shuffle=False,
                callbacks=[self.modelSave(), WandbMetricsLogger(), WandbModelCheckpoint("model-tensorflow-u-Pinova"), WandbCallback()])
        else:
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                shuffle=False,
                callbacks=[self.trainCallback()])
        self.model = empty_model
        return history

    def predict(self, X):
        return self.model.predict(X)

def main():
    model = PredictionModel(
        X=Xtrain,
        Y=Ytrain,
        n_outputs=n_ahead,
        n_lag=lag,
        n_ft=n_ft,
        n_layer=n_layer,
        batch=batch_size,
        epochs=epochs,
        lr=lr,
        Xval=Xval,
        Yval=Yval,
    )
    model.model.summary()

    history = model.train()

    WandbCallback(monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),log_weights=(True), log_gradients=(True), save_model=(True),training_data=(X,Y), validation_data=(X,Y), labels=None,generator=None, input_type=None, output_type=None, log_evaluation=(False), validation_steps=None, class_colors=None, log_batch_frequency=None,log_best_prefix="best_", save_graph=(True), validation_indexes=None,validation_row_processor=None, prediction_row_processor=None, predictions=24, infer_missing_processors=(True), log_evaluation_frequency=0,compute_flops=(False),)

    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    n_epochs = range(len(loss))

    # Usporedba predvidenih i stvarnih vrijednosti
    yhat = [x[0] for x in model.predict(Xval)]
    y = [y[0] for y in Yval]

    # Spremanje predikcija
    days = d['time'].values[-len(y):]

    frame = pd.concat([
    pd.DataFrame({'day': days, 'temperature': y, 'type': 'original'}),
    pd.DataFrame({'day': days, 'temperature': yhat, 'type': 'forecast'})
    ])

    # Invertirnanje skaliranja podataka
    frame['temperature_unscaled'] = [(x * train_std['temperature']) + train_mean['temperature'] for x in frame['temperature']]

    # Pivotiranje
    pivoted = frame.pivot_table(index='day', columns='type')
    pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
    pivoted['res'] = pivoted['temperature_unscaled_original'] - pivoted['temperature_unscaled_forecast']
    pivoted['res_unsc'] = [pow(x,2) for x in pivoted['res']]

    pivoted = frame.pivot_table(index='day', columns='type')
    pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
    pivoted['res'] = pivoted['temperature_unscaled_original'] - pivoted['temperature_unscaled_forecast']
    pivoted['res_mean'] = [pow(x,2) for x in pivoted['res']]
    #print(pivoted.tail(10))
    rmse=sqrt(pivoted['res_mean'].sum() / pivoted.shape[0])
    print(f"RMSE: {sqrt(pivoted['res_mean'].sum() / pivoted.shape[0])} C")

    print(pivoted.res_mean.describe())

    results_df = pd.DataFrame({
        "Location": [location],
        "Timestamp": [time_stamp],
        "Model": [name_without_ext],
        "RMSE": [rmse]
    })

    results_df.to_csv("results_Pinova.csv", mode='a', header=not pd.io.common.file_exists("results_Pinova.csv"), index=False)

if __name__ == "__main__":
    main()