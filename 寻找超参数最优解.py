import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import itertools
from tqdm import tqdm


# 读取数据
df = pd.read_csv('BTCUSDT_5mdata.csv')

# 数据预处理
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume', 'dif', 'dea', 'macd', 'ema7', 'ema25', 'ema50', 'ema144', 'ema169',
    'ao']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', 'dif', 'dea', 'macd', 'ema7',
                                      'ema25', 'ema50', 'ema144', 'ema169', 'ao']])
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'dif', 'dea', 'macd', 'ema7', 'ema25', 'ema50', 'ema144', 'ema169',
        'ao']].values
y = df['Close'].values

# 将输入数据reshape为(N, 14)的形状
X = X[:len(X)-30]
X = X.reshape(-1, 14)
Y = y[:len(y)-30]
Y = Y.reshape(-1, 1)

# 将输入数据转换为torch tensor，并将设备选择为mps
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(Y)
device = torch.device("cuda")

# 定义超参数搜索空间
learning_rates = [0.01, 0.001, 0.0001, 0.00001]
hidden_sizes = [(512, 256, 128, 64, 32), (256, 128, 64, 32, 16), (128, 64, 32, 16, 8), (64, 32, 16, 8, 4),
                (32, 16, 8, 4, 2)]
num_epochs = [1000, 2000, 3000, 4000, 5000]

# 初始化最佳模型和最佳超参数组合
best_model = None
best_loss = float('inf')
best_params = None

# 插入计时
import time
start = time.time()

# 循环遍历所有超参数组合
for lr, hidden_size, epoch in itertools.product(learning_rates, hidden_sizes, num_epochs):

    # 定义模型
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(14, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
            self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
            self.fc5 = nn.Linear(hidden_size[3], hidden_size[4])
            self.fc6 = nn.Linear(hidden_size[4], 1)



        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            x = self.fc6(x)
            return x


    # 初始化模型
    model = Net().to(device)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型并添加进度条
    for epoch in tqdm(range(epoch), desc='Training model', leave=False):
        optimizer.zero_grad()
        output = model(X_train.to(device))
        loss = criterion(output, y_train.to(device))
        loss.backward()
        optimizer.step()
        tqdm.write(f"Epoch: {epoch + 1}\tLoss: {loss.item():.4f}")



    # 预测结果
    X_test = torch.FloatTensor(X[-30:])
    y_test = torch.FloatTensor(y[-30:])
    with torch.no_grad():
        output = model(X_test.to(device))

    # 使用验证集评估模型性能
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        val_loss = criterion(y_pred, y_train.to(device)).item()

    # 比较当前模型和最佳模型，并更新最佳模型和最佳超参数组合
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model
        best_params = (lr, hidden_size, epoch)

# 插入计时
end = time.time()
print(f"Time: {end - start:.2f} seconds")

print(f"Best hyperparameters: learning_rate={best_params[0]}, hidden_size={best_params[1]}, num_epochs={best_params[2]}")
print(f"Best validation loss: {best_loss:.4f}")



