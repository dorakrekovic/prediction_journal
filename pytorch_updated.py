#!/usr/bin/env python
# General design
# One-step-ahead
# Univariate

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns
import wandb

df = pd.read_csv("data/Nasice_19_20.csv")
final_location = "Nasice"
df.head(1)

df.describe()

df.info()

train_dates = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')

train_dates.head(1)

# variables for training
cols = list(df)[1:2]

df_for_training = df[cols].astype(float)

#df_for_training.plot.line()

scaler = StandardScaler()  # RobustScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

df_for_training_scaled.shape

#plt.plot(df_for_training_scaled)
#plt.show()

np.savetxt(f"scaled_new_{final_location}.csv", df_for_training_scaled, delimiter=',')

##splitting dataset into train and test split for time series data
training_size = int(len(df_for_training_scaled) * 0.65)
test_size = len(df_for_training_scaled) - training_size
train_data, test_data = df_for_training_scaled[0:training_size, :], df_for_training_scaled[
                                                                    training_size:len(df_for_training_scaled), :]

train_data.shape, test_data.shape

train_data[0], test_data[0]

from datetime import datetime

# Get current time and format it nicely
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

# Create dynamic filename with timestamp
saved_model_path = f'torch_univariate__{final_location}_{time_stamp}.pth'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

past_observation = 24


class SequenceDataset(Dataset):
    def __init__(self, data, past_len=past_observation):
        self.data = data
        self.data = torch.from_numpy(data).float().view(-1)
        self.past_len = past_len

    def __len__(self):
        return len(self.data) - self.past_len - 1

    def __getitem__(self, index):
        return self.data[index: index + self.past_len], self.data[index + self.past_len]


train_dataset = SequenceDataset(train_data, past_len=past_observation)
test_dataset = SequenceDataset(test_data, past_len=past_observation)

train_dataset[0]

train_dataset[1]

train_dataset[2]

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size, drop_last=True)


class Lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(Lstm_model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=0.2,
                            bidirectional=False)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.dropout(out[-1])  # Dropout after LSTM
        final_out = self.fc(out)
        return final_out, hn, cn

    def init(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0


input_dim = 1
hidden_size = 30
num_layers = 5

args = {
    'input_dim': 1,
    'hidden_size': 64,
    'num_layers': 5,
    'past_observation': 24,
    'batch_size': 128,
    'optimizer': 'AdamW',
    'loss_function': 'MSELoss',
    'num_epochs': 50,
}

wandb.init(project="IEEEEdge", config=args)

model = Lstm_model(input_dim, hidden_size, num_layers).to(device)

# Watch the model
wandb.watch(model, log_freq=100)
# from torchsummary import summary

# summary(model,)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def train(dataloader):
    hn, cn = model.init()
    model.train()
    losses = []
    for batch, item in enumerate(dataloader):
        x, y = item
        x = x.to(device)
        y = y.to(device)
        out, hn, cn = model(x.reshape(past_observation, batch_size, 1), hn, cn)
        loss = loss_fn(out.reshape(batch_size), y)
        losses.append(loss.item())
        hn = hn.detach()
        cn = cn.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train = np.sum(losses) / len(dataloader)
    # Logging the losses
    wandb.log({"epoch/epoch": epoch, "epoch/loss": loss_train})
    # print(f"train loss: {loss_train:>8f} ")
    return loss_train


torch.save(model.state_dict(), saved_model_path)
print(f"Model saved at {saved_model_path}")


def test(dataloader):
    hn, cn = model.init()
    model.eval()
    losses = []
    for batch, item in enumerate(dataloader):
        x, y = item
        x = x.to(device)
        y = y.to(device)
        out, hn, cn = model(x.reshape(past_observation, batch_size, 1), hn, cn)
        loss = loss_fn(out.reshape(batch_size), y)
        losses.append(loss.item())
    loss_test = np.sum(losses) / len(dataloader)
    wandb.log({"epoch/epoch": epoch, "epoch/val_loss": loss_test})
    # print(f"test loss: {loss_test:>8f} ")
    return loss_test


epochs = 50
train_losses = []
test_losses = []
for epoch in range(epochs):
    train_losses.append(train(train_dataloader))
    test_losses.append(test(test_dataloader))
    if epoch % 5 == 0:
        print(f"epoch {epoch} ")
        print(train_losses[epoch], test_losses[epoch])

#plt.plot(train_losses, label="train")
#plt.plot(test_losses, label="test")
#plt.legend()
#plt.show()

import math
from sklearn.metrics import mean_squared_error
import numpy as np


def calculate_metrics(data_loader):
    pred_arr = []
    y_arr = []
    with torch.no_grad():
        hn, cn = model.init()

        for batch, item in enumerate(data_loader):
            x, y = item
            x, y = x.to(device), y.to(device)
            x = x.view(past_observation, batch_size, 1)
            pred = model(x, hn, cn)[0]
            pred = scaler.inverse_transform(pred.detach().cpu().numpy()).reshape(-1)
            y = scaler.inverse_transform(y.detach().cpu().numpy().reshape(1, -1)).reshape(-1)
            pred_arr = pred_arr + list(pred)
            y_arr = y_arr + list(y)
        # print(pred_arr[21],y_arr[21])
        return math.sqrt(mean_squared_error(y_arr, pred_arr))


# calculating final loss metrics
print("final loss metrics after inversing scaling")
print(f"train mse loss {calculate_metrics(train_dataloader)}")
print(f"test mse loss {calculate_metrics(test_dataloader)}")

rmse_train = np.sqrt({calculate_metrics(train_dataloader)})
rmse_test = np.sqrt({calculate_metrics(test_dataloader)})
print(f" train RMSE: {rmse_train}")
print(f" test RMSE: {rmse_test}")