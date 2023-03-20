import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

# 计算指标
def calculate_indicators(df):
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd - signal

    # EMA
    for n in [7, 25, 50, 144, 169]:
        df[f'EMA{n}'] = df['Close'].ewm(span=n, adjust=False).mean()

    # AO (Awesome Oscillator)
    median_price = (df['High'] + df['Low']) / 2
    df['AO'] = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()

    return df


# 转换时间
def convert_time(df):
    df['Time'] = pd.to_datetime(df['Time'], unit='ms') + timedelta(hours=8)
    return df


# 准备数据
def prepare_data(df, features, target, window_size):
    data = []
    labels = []

    for i in range(len(df) - window_size):
        data.append(df[features].iloc[i:i+window_size].values)
        labels.append(df[target].iloc[i+window_size])

    data, labels = np.array(data), np.array(labels)

    return data, labels


class StockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 加载数据
data_file = 'BTCUSDT_2017-08-18 to 2022-08-09.csv'
df = pd.read_csv(data_file)
df = calculate_indicators(df)
df = convert_time(df)
df.dropna(inplace=True)

# 特征和目标
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'EMA7', 'EMA25', 'EMA50', 'EMA144', 'EMA169', 'AO']
target = 'Close'
window_size = 10

data, labels = prepare_data(df, features, target, window_size)

# 划分训练集和测试集
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data, test_data = data[:train_size], data[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# 数据集和数据加载器
train_dataset = StockDataset(train_data, train_labels)
test_dataset = StockDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# MLP模型
input_size = len(features) * window_size
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size)
device = torch.device("mps")
model.to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device).float()
        batch_labels = batch_labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(batch_data.view(batch_data.size(0), -1))
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device).float()
            batch_labels = batch_labels.to(device).float()

            outputs = model(batch_data.view(batch_data.size(0), -1))
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

        test_loss /= len(test_loader)



# 预测
model.eval()
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device).float()
        batch_labels = batch_labels.to(device).float()

        outputs = model(batch_data.view(batch_data.size(0), -1))
        break

print(outputs)

# # 保存模型
# torch.save(model.state_dict(), 'your_model_file.pth')
#
# # 加载模型
# model = MLP(input_size, hidden_size, output_size)
# model.load_state_dict(torch.load('your_model_file.pth'))

