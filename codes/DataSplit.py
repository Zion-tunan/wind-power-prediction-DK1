"""
DataSpilt.py

Purpose:
This script provides the function `split_data()` that transforms multivariate time series data
into training and testing datasets suitable for supervised learning. Specifically, it constructs
single-step prediction sequences using a sliding window approach.

Functions:
- split_data(data, timestep, feature_size):
    Converts raw normalized data into (X, y) pairs for time series prediction.
    X is a sequence of `timestep` time steps with `feature_size` features,
    y is the wind speed value immediately following each sequence.
"""

import numpy as np

# Convert multivariate time series into single-step prediction training data
def split_data(df, timestep, feature_cols, target_col):

    dataX = []  # Store X (input sequences)
    dataY = []  # Store Y (target labels)

    features = df[feature_cols].values
    target = df[target_col].values

    for index in range(len(df) - timestep):
        dataX.append(features[index: index + timestep])
        dataY.append(target[index + timestep])

    # Convert to NumPy arrays
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # Split ratio: 80% training, 20% testing
    train_size = int(np.round(0.8 * dataX.shape[0]))

    x_train = dataX[:train_size]
    y_train = dataY[:train_size].reshape(-1, 1)

    x_test = dataX[train_size:]
    y_test = dataY[train_size:].reshape(-1, 1)

    return x_train, y_train, x_test, y_test



#测试下
# import pandas as pd
# data_path ="../Pro_wind_8D_LSTM_Attention/01.csv"
# #1.加载时间序列数据
# df = pd.read_csv(data_path, index_col=0)
# df = df[:30000]
#
# #2 查找缺失值、重复值
# print(df.isnull().sum())
# df = df.fillna(method='bfill')#把数据中的NA空值用后一个非空值进行填充，或者#df = df.fillna(0.0)
# print(df.isnull().sum())
# print(df.duplicated()) #查看是否有重复值
# print(df.describe()) #查看是否有异常值
# #去重
# df = df[~df.index.duplicated()]
#
# #差分
# yd15 = df["YD15"]
# yd15_diff = yd15.diff()  #一阶差分
# #将nan值变成0
# # 使用fillna方法将NaN值替换为0
# yd15_diff_filled = yd15_diff.fillna(0)
#
# power = df["ROUND(A.POWER,0)"]
# power_diff = power.diff()
# power_diff_filled = power_diff.fillna(0)
#
# df.insert(7, 'power_diff', power_diff_filled)
# df.insert(8, 'yd15_diff', yd15_diff_filled)
#
# data = np.array(df)
# # data = data[:, :8]
# x_train, y_train, x_test, y_test = split_data(data, 20, 9)

