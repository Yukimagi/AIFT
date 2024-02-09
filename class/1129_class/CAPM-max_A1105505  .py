#!/usr/bin/env python
# coding: utf-8

# In[76]:


#!pip install altair
# https://northfar.net/python4finance-5/
# Yu-Chi Lin, 2023/11/29, Kaohsiung,CAPM-max    


# In[1]:


#%matplotlib inline
from scipy import stats
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
# import pandas_datareader.data as pddata
import altair as alt
import statsmodels.api as sm


# In[2]:


plt.style.use('ggplot')
mpl.rcParams['figure.figsize']= [15, 9]


# In[3]:


ticker = '1101.TW'
beg_date = '2020-01-01'
end_date = '2023-11-28'

stock_0050 = yf.download(ticker,start=beg_date, end=end_date)


# In[4]:


stock_0050


# In[5]:


return_0050 = stock_0050['Adj Close'] / stock_0050['Adj Close'].shift(1) - 1
return_0050.dropna(inplace=True)
return_0050 = return_0050.to_frame()
return_0050.columns = ['ret']


# In[6]:


return_0050


# In[7]:


ticker = '0050.TW'
sp500 = yf.download(ticker,start=beg_date, end=end_date)
return_market = sp500['Adj Close'] / sp500['Adj Close'].shift(1) - 1
return_market.dropna(inplace=True)
return_market = return_market.to_frame()
return_market.columns = ['ret']
sp500.head()


# In[8]:


return_market


# In[9]:


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price', color=color)
ax1.plot(stock_0050.index, stock_0050['Adj Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('SP500 Index', color=color)  # we already handled the x-label with ax1
ax2.plot(sp500.index, sp500['Adj Close'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[10]:


#線性回歸output beta(apple對大盤)
fig, ax = plt.subplots()

color = 'tab:red'
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.plot(return_0050.index, return_0050.ret, label='0050 Return')
ax.plot(return_market.index, return_market.ret, label='Market Return')
ax.legend()
plt.show()


# In[11]:


results = stats.linregress(return_market['ret'], return_0050['ret'])
print(results)


# In[12]:


x = sm.add_constant(return_market['ret'])
ols_results = sm.OLS(return_0050['ret'], x).fit()
print(ols_results.summary())


# In[13]:


plt.plot(return_market['ret'], return_0050['ret'], '.')
x = np.linspace(-0.05, 0.05, 100)
y = results.slope * x + results.intercept
plt.plot(x, y)
plt.show()


# In[14]:


def ret_f(ticker, beg_date, end_date):
    p = yf.download(ticker,start=beg_date, end=end_date)
    ret = p / p.shift(1)
    return ret.dropna()


# In[16]:


# 將索引轉換為日期時間對象
return_0050.index = pd.to_datetime(return_0050.index)#因為會出錯，所以多加的

# 提取年份
years = return_0050.index.year.unique()

# 創建空列表來保存每年的 alpha 和 beta 值
alpha_values = []
beta_values = []
datas = pd.DataFrame()

for year in years:
    y = return_0050.loc[return_0050.index.year == year, 'ret']  # 使用 loc 屬性訪問列
    x = return_market.loc[return_market.index.year == year, 'ret']  # 使用 loc 屬性訪問列
    (beta, alpha, r_value, p_value, std_err) = stats.linregress(x, y)
    alpha = round(alpha, 6)
    alpha_values.append(alpha)
    beta = round(beta, 4)
    r_value = round(r_value, 4)
    p_value = round(p_value, 4)
    beta_values.append(beta)
    print(year, alpha, beta, r_value, p_value)
    datas = datas._append(pd.DataFrame([[year, alpha, beta, r_value, p_value]], columns=["year", "alpha", "beta", "r_value", "p_value"]), ignore_index=True)


# In[18]:


fig, ax1 = plt.subplots()
print(datas)

# 繪製alpha曲線
ax1.plot(datas.year, datas.alpha, label="alpha", color='blue')
ax1.set_ylabel('Alpha', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()

# 繪製beta曲線
ax2.plot(datas.year, datas.beta, label="beta", color='red')
ax2.set_ylabel('Beta', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 設定x軸標籤
plt.xticks(datas.year)

# 添加圖例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 顯示圖表
plt.show()


# In[ ]:




