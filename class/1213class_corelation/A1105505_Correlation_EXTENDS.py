#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[27]:


import numpy as np
import scipy.stats
import pandas as pd


# In[28]:


#驗證相關係數
def Pearson_correlation(X,Y):

    if len(X)==len(Y):

        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))

        Sum_x_squared = sum((X-X.mean())**2)

        Sum_y_squared = sum((Y-Y.mean())**2)       

        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)

    return corr


# In[37]:


for i in range(1,6):
    # 讀取檔案
    df = df = pd.read_excel(f'correlationCoefficientExample/correlationCoefficientExample{i}.xlsx')

    # 提取X和Y列的數據
    x = df['X'].values
    y = df['Y'].values

    # 將x和y轉換為NumPy數組
    x = np.array(x)
    y = np.array(y)
    #x
    y
        #驗證相關係數    
    print(f'相關係數(x,y):{Pearson_correlation(x,y)}')    

    print(f'相關係數(x,y):{Pearson_correlation(x,x)}')
    
    #使用回歸線、其方程和 Pearson 相關係數創建 x-y 圖。您可以使用以下命令獲得回歸線的斜率和截距以及相關係數：linregress()
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    
    #取得包含回歸線方程和相關係數值的字串
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, correlation coefficient={r:.4f}'
    print(f'line:{line}')
    
    #使用 .plot（） 建立 x-y 圖|
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='s', label='Data points')
    ax.plot(x, intercept + slope * x, label=line)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(facecolor='white')
    plt.savefig(f'graph/correlationCoefficientExample{i}.png')  
    plt.show()

