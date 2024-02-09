#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pip', 'install scipy')
get_ipython().run_line_magic('pip', 'install altair')
get_ipython().run_line_magic('pip', 'install statsmodels')
get_ipython().run_line_magic('pip', 'install scikit-learn')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[21]:


# get the closing price of AMZN Stock
amzn = yf.download('1101.TW', start='2020-01-01', end='2023-11-28', progress=False)['Close']
amzn = pd.DataFrame(amzn)
amzn['amzn_return'] = amzn['Close'].pct_change()
amzn['amzn_log_return'] = np.log(amzn['Close']) - np.log(amzn['Close'].shift(1))
amzn.dropna(inplace=True)

# get the closing price of NASDAQ Index
nasdaq = yf.download('0050.TW', start='2020-01-01', end='2023-11-28', progress=False)['Close']
nasdaq = pd.DataFrame(nasdaq)
nasdaq['nasdaq_return'] = nasdaq['Close'].pct_change()
nasdaq['nasdaq_log_return'] = np.log(nasdaq['Close']) - np.log(nasdaq['Close'].shift(1))
nasdaq.dropna(inplace=True)
print(amzn.amzn_return)
print(nasdaq.nasdaq_return)


# In[22]:


def market_beta(X,Y,N):
    """ 
    X = The independent variable which is the Market
    Y = The dependent variable which is the Stock
    N = The length of the Window
     
    It returns the alphas and the betas of
    the rolling regression
    """
     
    # all the observations
    obs = len(X)
     
    # initiate the betas with null values
    betas = np.full(obs, np.nan)
     
    # initiate the alphas with null values
    alphas = np.full(obs, np.nan)
     
     
    for i in range((obs-N)):
        regressor = LinearRegression()
        regressor.fit(X.to_numpy()[i : i + N+1].reshape(-1,1), Y.to_numpy()[i : i + N+1])
         
        betas[i+N]  = regressor.coef_[0]
        alphas[i+N]  = regressor.intercept_
 
    return(alphas, betas)
  
results = market_beta(nasdaq.nasdaq_return,amzn.amzn_return, 60)#60天
 
results = pd.DataFrame(list(zip(*results)), columns = ['alpha', 'beta'])

# 計算避險口數
#現貨市值為2571177130500,期貨指數為11568.5,每點價值為50
#res = stats.linregress(return_apple.ret, return_market.ret)
results['sell'] = results['beta'] * 2571177130500 / (11568.5 * 50)

#印出所有results
for i in range(len(results)):
    print(results.iloc[i])


# In[23]:


results.index = amzn.index
plt.figure(figsize=(12,8))
results.beta.plot.line()
plt.title("Market Beta: 1101 vs 0050 with Rolling Window of 60 Days")


# In[24]:


results.index = amzn.index
plt.figure(figsize=(12,8))
results.sell.plot.line()
plt.title("Hedge Position: 1101 vs 0050 with Rolling Window of 60 Days")


# In[ ]:




