import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Reload model components and scaler
device = "cuda" if torch.cuda.is_available() else "cpu"
destination='Suhopolje'
saved_model_path = f'models/torch_univariate__2025-03-11_10-12.pth'  # example path

# Reload 2021 data
df_2021 = pd.read_csv(f"data/{destination}2021.csv")

# Convert datetime (adjust format if needed)
df_2021['datetime'] = pd.to_datetime(df_2021['datetime'], format='%d/%m/%Y %H:%M')

# Select the same feature column used for training
cols = list(df_2021)[1:2]
df_2021_for_inference = df_2021[cols].astype(float)

# IMPORTANT: Use the SAME scaler as in training. If you saved it, load it:
# scaler = joblib.load('scaler_Nasice.pkl')
# If not, RE-FIT the scaler on TRAIN data (2012-2020), not on 2021.
scaler = StandardScaler()
df_train = pd.read_csv(f"data/{destination}_19_20.csv")  # assuming 2012-2020 data
df_train_cols = df_train[cols].astype(float)
scaler.fit(df_train_cols)

# Scale the 2021 data
df_2021_scaled = scaler.transform(df_2021_for_inference)

class Lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(Lstm_model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=0,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn, cn))  # out shape: (seq_len, batch_size, hidden_size)
        final_out = self.fc(out[-1])  # final prediction from last time step
        return final_out, hn, cn

    def init(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0
# Your LSTM class should already be defined in the script
# Initialize the model with SAME hyperparameters
input_dim = 1
hidden_size = 30
num_layers = 5
past_observation = 24

model = Lstm_model(input_dim, hidden_size, num_layers).to(device)

# Load trained weights

model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()

print("Model loaded successfully for inference_results.")


predictions = []
real_values = []
datetimes = []

# Loop through the data starting from the past_observation index
for i in range(past_observation, len(df_2021_scaled)):
    # Get the past 24 observations
    input_seq = df_2021_scaled[i - past_observation:i]  # shape: (24, 1)

    # Convert to tensor and reshape: (seq_len, batch_size, input_dim)
    input_tensor = torch.tensor(input_seq).float().to(device)
    input_tensor = input_tensor.view(past_observation, 1, 1)

    with torch.no_grad():
        hn, cn = model.init(batch_size=1)
        pred_scaled, _, _ = model(input_tensor, hn, cn)

    # Inverse transform prediction back to original scale
    pred_value = scaler.inverse_transform(pred_scaled.cpu().numpy())[0][0]

    # Append real value and prediction
    datetimes.append(df_2021['datetime'].iloc[i])
    real_values.append(df_2021_for_inference.iloc[i, 0])  # real value
    predictions.append(pred_value)

    # Build result DataFrame
    results_df = pd.DataFrame({
        'datetime': datetimes,
        'real_value': real_values,
        'predicted_value': predictions
    })

    # Save to CSV
    output_csv_path = f'inference_results/inference_results_{saved_model_path}.csv'
    results_df.to_csv(output_csv_path, index=False)

print(f"Inference results saved to {output_csv_path}")