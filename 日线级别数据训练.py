import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

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
X = X[:len(X)-1]
X = X.reshape(-1, 14)

# 将输入数据转换为torch tensor，并将设备选择为mps
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y)
device = torch.device("mps")


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 2)
        self.fc8 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x


# 初始化模型
model = Net()
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(X_train.to(device))
    loss = criterion(output, y_train.to(device))
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10000, loss.item()))

# 预测结果
X_test = torch.FloatTensor(X[-1:])
y_test = torch.FloatTensor(y[-1:])
with torch.no_grad():
    output = model(X_test.to(device))

# 将预测结果转换为numpy array，并进行逆归一化
output = output.cpu().numpy().reshape(-1, 1)
output = np.concatenate((output, X_test[:, 1:]), axis=1)
output = scaler.inverse_transform(output)
output = output[:, 0]

# 打印预测结果
print('Predicted Close price:')
print(output.flatten())

# 保存模型
