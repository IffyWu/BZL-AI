import pandas as pd
import talib

# 读取csv文件
df = pd.read_csv('BTCUSDT_5min.csv')

# 计算指标
df['ema7'] = talib.EMA(df['Close'], timeperiod=7)
df['ema25'] = talib.EMA(df['Close'], timeperiod=25)
df['ema50'] = talib.EMA(df['Close'], timeperiod=50)
df['ema144'] = talib.EMA(df['Close'], timeperiod=144)
df['ema169'] = talib.EMA(df['Close'], timeperiod=169)
df['dif'], df['dea'], df['macd'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['ao'] = talib.SMA(df['Close'], timeperiod=5) - talib.SMA(df['Close'], timeperiod=34)

# 保存结果到csv文件
df.to_csv('BTCUSDT_5mdata.csv', index=False)
