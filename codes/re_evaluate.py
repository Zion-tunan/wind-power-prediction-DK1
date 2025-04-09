import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import os, sys

# 添加 codes 文件夹到系统路径
#sys.path.append(os.path.abspath("../codes"))

# 导入自定义模块
from model import LSTM_Attention
from Config import config
from DataSplit import split_data

# Step 1: Load config + data
config = config()

df = pd.read_csv(config.data_path, index_col=0)
df = df[:30000]
df = df.fillna(method='bfill')
df = df[~df.index.duplicated()]

scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))

scaler = MinMaxScaler()
df_label = df["ROUND(A.WS,1)"]
label = scaler.fit_transform(np.array(df_label).reshape(-1, 1))

x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)

x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# Step 2: Init model and load weights
#Model = LSTM_Attention(
#    input_size=config.feature_size,
#    seq_len=config.timestep,
#    hidden_size=config.hidden_size,
#    num_layers=config.num_layers,
#    num_heads=config.num_heads,
#    output_size=config.output_size
#)

Model = LSTM_Attention(
    feature_size=config.feature_size,
    timestep=config.timestep,
    hidden_size=config.hidden_size,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    output_size=config.output_size
)

# 加载路径注意用相对路径跳回 ../models/
Model.load_state_dict(torch.load("../models/LSTM_Attention_wind_15min.pth"))
Model.eval()

# Step 3: Predict & Evaluate
with torch.no_grad():
    y_test_pred = Model(x_test_tensor).detach().numpy()
    y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))
    y_test_true = scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1))

y_true = y_test_true.flatten().tolist()
y_pred = y_test_pred.flatten().tolist()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
corr, _ = stats.pearsonr(y_true, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ Pearson correlation: {corr:.4f}")

# Step 4: Plot (first 200 samples)
plt.figure(figsize=(12, 6))
plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.title(f"{config.model} Prediction | RMSE: {rmse:.3f} | Corr: {corr:.3f}")
plt.legend()
plt.savefig(f"../plots/{config.model}_eval.png", dpi=300)
plt.show()
