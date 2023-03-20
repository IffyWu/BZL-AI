import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt

# 读入数据
df = pd.read_csv('stock_data.csv')

# 将Time转换为datetime格式
df['Time'] = pd.to_datetime(df['Time'], unit='ms')

# 将Time转换为东八区时间
df['Time'] = df['Time'] + datetime.timedelta(hours=8)

# 对数据进行归一化处理
scaler = MinMaxScaler()
df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])

# 将数据分为训练集和测试集
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]


# 将数据转换为PyTorch的Dataset格式
class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'input': self.data[idx, :-1],
            'output': self.data[idx, -1]
        }
        return sample


train_dataset = StockDataset(train_df.values)
test_dataset = StockDataset(test_df.values)

# 将数据转换为PyTorch的DataLoader格式
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.relu(x)
        x = self.hidden_layer_2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


input_size = 11
hidden_size = 128
output_size = 1

# model = MLP(input_size, hidden_size, output_size)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

model = MLP(input_size, hidden_size, output_size)
device = torch.device("mps")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['input'], data['output']
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['input'], data['output']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), labels.float())

            running_loss += loss.item() * inputs.size(0)
            y_true += labels.cpu().numpy().tolist()
            y_pred += outputs.squeeze().cpu().numpy().tolist()

    return running_loss / len(dataloader.dataset), y_true, y_pred


epochs = 50

train_losses = []
test_losses = []

for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, y_true, y_pred = test(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, epochs, train_loss, test_loss))

# 使用模型进行预测
inputs = test_df.values[:, :-1]
inputs = torch.from_numpy(inputs).to(device)
outputs = model(inputs.float())

# 将预测结果和真实结果一起绘制成图表

plt.plot(y_true, label='True')
plt.plot(y_pred, label='Prediction')
plt.legend()
plt.show()
