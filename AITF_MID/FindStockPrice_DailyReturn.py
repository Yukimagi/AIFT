#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))
def find_stock_data(symbol, company_code, date):
    try:
        # 構建CSV檔案的路徑
        file_path = f'price_data_all/{symbol}_price.csv'

        # 嘗試讀取CSV檔案
        df = pd.read_csv(file_path)

        # 在讀取的DataFrame中找到符合條件的資料
        filtered_data = df[(df['Symbol'] == company_code) & (df['Date'] == date)]

        if not filtered_data.empty:
            # 資料存在，輸出找到的資料
            print(filtered_data)
        else:
            print(f'找不到{symbol}的{date}資料')
    except FileNotFoundError:
        print(f'無{symbol}的股票公司')

# 輸入股票代號、公司代號和日期
symbol = 2330
company_code = 2330
date = '2000-01-10'
'''
結果範例(會有所有股價與daily return)
Date       Open       High        Low      Close  Adj Close  \
0  2000-01-04  15.416665  15.781559  15.416665  15.781559   4.880246   

         Volume  Daily_Return  Symbol  
0  112224035437           NaN    1101  
'''
# 執行函數以查找資料
find_stock_data(symbol, company_code, date)

