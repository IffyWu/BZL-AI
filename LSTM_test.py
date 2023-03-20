import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 加载数据集
df = pd.read_csv('BTCUSDT_data.csv')
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'dif', 'dea', 'macd', 'ema7', 'ema25', 'ema50', 'ema144',
         'ema169', 'ao']]


# 将数据集分成训练集和测试集
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]


# 定义函数将数据转换为适当的格式
def create_dataset(df):
    data = df.values.astype('float32')
    input_seq = data[:-1]
    output_seq = data[1:]
    input_seq = torch.from_numpy(input_seq)
    output_seq = torch.from_numpy(output_seq)
    return input_seq, output_seq

# 将训练集和测试集转换为 PyTorch 张量
train_input_seq, train_output_seq = create_dataset(train_df)
test_input_seq, test_output_seq = create_dataset(test_df)

# 定义模型
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hx, cx):
        out, (hn, cn) = self.lstm(x, (hx, cx))
        out = self.fc(out[:, -1, :])
        return out, hn, cn

# 定义超参数
input_size = 14
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.01
device = torch.device('mps')

# 创建模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 初始化隐藏状态和细胞状态
batch_size = 1
hx = torch.zeros(num_layers, batch_size, hidden_size).to(device)
cx = torch.zeros(num_layers, batch_size, hidden_size).to(device)

# 训练模型
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs, hx, cx = model(train_input_seq.to(device), hx, cx)
    loss = criterion(outputs, train_output_seq.to(device))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # 每 10 个 epoch 打印一次预测结果
    if epoch % 10 == 0:
        with torch.no_grad():
            test_inputs = test_input_seq.to(device)
            test_hx = torch.zeros(num_layers, test_inputs.size(0), hidden_size).to(device)
            test_cx = torch.zeros(num_layers, test_inputs.size(0), hidden_size).to(device)
            test_outputs, _, _ = model(test_inputs, test_hx, test_cx)
            test_outputs = test_outputs.cpu().numpy()

            # 绘制预测结果
            import matplotlib.pyplot as plt
            plt.plot(test_output_seq.numpy(), label='True')
            plt.plot(test_outputs, label='Predicted')
            plt.legend()
            plt.show()
