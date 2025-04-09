"""

Time: 2025/06/04
Author: Zhiyuan(Julian) Xie

Purpose:
Time series forecasting based on deep learning models (this example focuses on wind speed forecasting using multivariate single-step prediction)

Supported time series models:
- LSTM with multi-head attention
- BiLSTM
- Transformer
- Seq2Seq
- GRU
- LSTM

Workflow:
1) Data loading and reading
2) Feature engineering
   a) Data preprocessing
      - Handle duplicate entries and missing values
   b) Feature variables
      - Include first-order difference of wind power
      - Include components decomposed via EMD (Empirical Mode Decomposition)
3) Data normalization
4) Dataset splitting: train/test set preparation
5) Load model, define loss function and optimizer
6) Model training
7) Model prediction
   - Show predictions on the training set
   - Show predictions on the test set
8) Accuracy evaluation
   - RMSE (Root Mean Squared Error)
   - Pearson correlation coefficient

Usage Notes:
1) The `config` script is used for configuration. You can modify parameters such as data path, model type, input features, output target, etc.
2) The `datasplit` script handles dataset partitioning.
3) Once configured, this script can be run directly without modification.

Environment:
- PyTorch 1.8
- Python 3.6


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from model import LSTM_Attention,BiLSTM, TimeSeriesTransformer,Encoder,Decoder,Seq2Seq,GRU,LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from Config import config
from DataSplit import split_data
from train import fit
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import stats
from PyEMD import EMD

# 0.Load the parameters in the configuration file

config = config()

# 1. Load time series data

df = pd.read_csv(config.data_path, index_col=0)
df.head()
df = df[:30000]

# 2. Feature Engineering Data Cleaning (such as finding missing values and duplicate values)

print(df.isnull().sum())
df = df.fillna(method='bfill')
print(df.isnull().sum())
print(df.duplicated())
print(df.describe())
df = df[~df.index.duplicated()]

# 3. Data Normalization

scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))

scaler = MinMaxScaler()
df_label = df["ROUND(A.WS,1)"]  # which is normalized separately
label = scaler.fit_transform(np.array(df_label).reshape(-1, 1))

# 4. Divide the data set and obtain training and test data
df["YD15_lag1"] = df["YD15"].shift(1)
df["YD15_lag2"] = df["YD15"].shift(2)
df.dropna(inplace=True)

feature_cols = [
    "WINDSPEED", "PREPOWER", "WINDDIRECTION",
    "TEMPERATURE", "HUMIDITY", "PRESSURE",
    "YD15_lag1", "YD15_lag2"
]
target_col = "YD15"
timestep = 20

x_train, y_train, x_test, y_test = split_data(df, timestep, feature_cols, target_col)
#x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)

x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)

x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_data, config.batch_size,  False)
test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, False)

# 5. Load the model, define the loss, and define the optimizer
if config.model =='LSTM_Attention':
    Model = LSTM_Attention(config.feature_size, config.timestep, config.hidden_size, config.num_layers, config.num_heads,config.output_size)
elif config.model =='BiLSTM':
    Model = BiLSTM(input_dim=config.feature_size, hidden_dim=config.hidden_size, num_layers=config.num_layers, output_dim=config.output_size)
elif config.model =='TimeSeriesTransformer':
    Model = TimeSeriesTransformer(config.feature_size, config.num_heads, config.num_layers, config.output_size, hidden_space=32, dropout_rate=0.1)
elif config.model == 'GRU':
    Model = GRU(input_dim=config.feature_size, hidden_dim=config.hidden_size, num_layers=config.num_layers, output_dim=config.output_size)
elif config.model == 'LSTM':
    Model = LSTM(input_dim=config.feature_size, hidden_dim=config.hidden_size, num_layers=config.num_layers, output_dim=config.output_size)
elif config.model == 'Seq2Seq':
    Encoder = Encoder(input_size=config.feature_size, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_size=config.batch_size)
    Decoder = Decoder(input_size=config.feature_size, hidden_size=config.hidden_size, num_layers=config.num_layers, output_size=config.output_size, batch_size=config.batch_size)
    Model = Seq2Seq(input_size=config.feature_size, hidden_size=config.hidden_size, num_layers=config.num_layers, output_size=config.output_size, batch_size=config.batch_size)
else:
    print('please choose correct model name')


# Print model structure and total number of trainable parameters
for name, param in Model.named_parameters():
    print(f"Parameter name: {name}, shape: {param.size()}")
total_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")

loss_function = nn.MSELoss()                                                # Define the loss function
optimizer = torch.optim.AdamW(Model.parameters(), lr=config.learning_rate)  # Define the optimizer

# 6. Model training, loss saving and visualization

# Ensure plots directory exists

os.makedirs("../plots", exist_ok=True)

train_loss = []
test_loss = []
bst_loss = np.inf
for epoch in range(config.epochs):
    epoch_loss, epoch_test_loss = fit(epoch, Model, loss_function, optimizer, train_loader, test_loader, bst_loss)

    # Save loss to CSV file
    list = [epoch_loss, epoch_test_loss]
    data = pd.DataFrame([list])
    data.to_csv(config.loss_path, mode='a', header=False, index=False)
    train_loss.append(epoch_loss)
    test_loss.append(epoch_test_loss)

#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
# Plot training and testing loss curves
plt.title("Loss Curve")
plt.plot(range(1, config.epochs+1), train_loss, label='Train Loss')
plt.plot(range(1, config.epochs+1), test_loss, label='Test Loss')
plt.legend()
plt.savefig(f"../plots/{config.model}_loss_curve.png", dpi=300)
plt.show()

print('Finished Training')

# 7. Inference on training set — show prediction on first 200 samples
Model.eval()
plot_size = 200
plt.figure(figsize=(12, 8))

with torch.no_grad():
    x_train_pred = scaler.inverse_transform((Model(x_train_tensor).detach().numpy().reshape(-1, 1)[: plot_size]))
    y_train_true = scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size])

    plt.plot(x_train_pred, "b", label='Predicted (First 200 samples)')
    plt.plot(y_train_true, "r", label='Actual (First 200 samples)')
    plt.legend()
    plt.title("Test Set Prediction")
    plt.savefig(f"../plots/{config.model}_prediction.png", dpi=300)
    plt.show()

# 9. Accuracy evaluation on test set
with torch.no_grad():
    y_test_pred = Model(x_test_tensor).detach().numpy()
    y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))

    y_test_true = scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1))

y_true = y_test_true.tolist()
y_true_flattened_list = [item for sublist in y_true for item in sublist]

y_pred = y_test_pred.tolist()
y_pred_flattened_list = [item for sublist in y_pred for item in sublist]

print("Actual values:", y_true_flattened_list)
print("Predicted values:", y_pred_flattened_list)

# Calculate RMSE
mse = mean_squared_error(y_true_flattened_list, y_pred_flattened_list)
Rmse = np.sqrt(mse)
print(f"Rmse between y_true and y_pred:{Rmse} ")

# Calculate Pearson correlation coefficient
correlation, _ = stats.pearsonr(y_true_flattened_list, y_pred_flattened_list)
print(f"Pearson's Correlation Coefficient between y_true and y_pred: {correlation}")

plt.figure(figsize=(12, 6))
plt.plot(x_train_pred, "b", label='Predicted (Train)')
plt.plot(y_train_true, "r", label='Actual (Train)')
plt.legend(title=f"RMSE: {Rmse:.3f} | Corr: {correlation:.3f}")
plt.title("Train Prediction with Evaluation Metrics")
plt.savefig(f"../plots/{config.model}_training.png", dpi=300)
plt.show()

