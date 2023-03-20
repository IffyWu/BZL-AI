import time
import csv
import pandas as pd
import datetime
from binance.client import Client
from tqdm import tqdm  # 导入 tqdm 模块

api_key = 'pzpUU5CIycvfHFgx4pHn4SsTcBHLOkDI87jYMgjRyalCEQwEMYP89NmfALPGioC1'
api_secret = 'vQ0ADq4kf0QpEBbBbKKSdStmD7tH2wgZLDYJqEsFcdrctxJusdR472zhgopoUpy2'

client = Client(api_key, api_secret)


def get_day_data_to_csv(symbol, interval, lookback, day):
    # 初始化一个空的 DataFrame
    df = pd.DataFrame(columns=['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'DataTime'])
    pbar = tqdm(total=int(lookback) * int(day))  # 初始化进度条
    while True:
        # 调用 client.futures_historical_klines() 获取数据
        frame = pd.DataFrame(client.futures_historical_klines(symbol, interval, lookback + 'm ago UTC', day))  # 注意修改时间
        if frame.empty:
            continue
        frame = frame.iloc[:, :6]
        frame.columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        # 新增加一个列，用于把时间戳转换成可读时间
        frame['DataTime'] = pd.to_datetime(frame['TimeStamp'], unit='ms') + pd.Timedelta(hours=8)
        # 将新获取的数据拼接到 df 中
        df = pd.concat([df, frame], ignore_index=True)
        data_count = len(frame)  # 记录实际返回的数据条数
        pbar.update(data_count)  # 更新进度条
        # 获取最后一条数据的时间戳
        last_time = df.iloc[-1, 0]
        # 将时间戳转换为 datetime 格式
        last_time = datetime.datetime.fromtimestamp(last_time / 1000)
        # 获取当前时间
        now = int(round(time.time() * 1000))
        # 如果再次获取数据的时间戳大于当前时间减去一天的时间戳，说明数据已经获取完毕，跳出循环
        # 注意修改时间戳：86400000为一天的毫秒时间戳
        # 86400000*3为3d
        # 86400000/24为1h，86400000/24*2为2h，86400000/24*4为4h，86400000/24*6为6h，86400000/24*8为8h，86400000/24*12为12h
        # 86400000/24/60为1m,86400000/24/60*3为3m,86400000/24/60*5为5m,86400000/24/60*15为15m,86400000/24/60*30为30m
        if last_time > datetime.datetime.fromtimestamp((now - 86400000/24/60*5 * float(day)) / 1000):
            break

    # 将数据保存为 CSV 文件
    filename = f"{symbol}_day_{day}d.csv"
    df.to_csv(filename, index=False)

    pbar.close()  # 关闭进度条
    return filename


# 测试函数
symbol = 'BTCUSDT'
interval = '5m'
lookback = '40000'
day = '5'
filename = get_day_data_to_csv(symbol, interval, lookback, day)
print(f"数据已保存到文件：{filename}")




