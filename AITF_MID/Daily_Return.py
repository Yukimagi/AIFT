#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 第三方套件 yfinance, 可用來串接 Yahoo Finance API 下載股票的價量資訊. 而且拿到的資料就是 Pandas 的 DataFrame
import yfinance as yf
import pandas as pd
import csv
import matplotlib.pyplot as plt

# 設定股票代號範圍
start_symbol = 1101
end_symbol =9999

# 創建一個空的DataFrame，用於存儲所有股票的數據


dict_test={}#測試是否有此股票公司

# 迭代每個股票代號
for symbol in range(start_symbol, end_symbol + 1):
    try:
        for i in range(1950,2023):
            # 使用yfinance獲取股票數據
            symbol_str = f"{symbol}.TW"  # 在台灣股市中，股票代號需要加上.TW
            #抓取最小開始資料的時間到2023-10-29
            data = yf.download(symbol_str, start=f'{i}-01-01', end="2023-10-29")

            # 計算每日回報率並加到dictionary中
            data['Daily_Return'] = (data['Close']-data['Close'].shift(1)) / data['Close'].shift(1)

            # 添加股票代號列並加到dictionary中
            data['Symbol'] = symbol
            #建立稍後要存的數據pandas
            all_data = pd.DataFrame()
            # 將每支股票的數據添加到總數據中
            all_data = pd.concat([all_data, data])
            if not all_data.empty:
                # 將結果保存為CSV文件
                all_data.to_csv(f'price_data_all/{symbol}_price.csv')
                dict_test[symbol]=1
                #繪製Adj Close price的時間序列圖(老師說可以用adj close)
                symbol_data = all_data[all_data['Symbol'] == symbol]
                plt.figure(figsize=(10, 6))
                plt.plot(symbol_data.index, symbol_data['Adj Close'], label=f"Stock {symbol}")
                plt.title(f"Stock {symbol} Price Time Series")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                #存檔
                plt.savefig(f"price_time_series/stock_{symbol}_adjclose_time_series.png")
                plt.close()
            else:
                print(f'無{symbol}資料')
            break
    except:
        print('except')
        dict_test[symbol]=0
        pass


# In[ ]:




