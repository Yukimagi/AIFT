#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import scipy.stats
import pandas as pd

# 驗證相關係數
def Pearson_correlation(X, Y):
    if len(X) == len(Y):
        Sum_xy = sum((X - X.mean()) * (Y - Y.mean()))
        Sum_x_squared = sum((X - X.mean()) ** 2)
        Sum_y_squared = sum((Y - Y.mean()) ** 2)
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
        return corr

for i in range(1, 6):
    # 讀取CSV檔案
    df = pd.read_excel(f'correlationCoefficientExample/correlationCoefficientExample{i}.xlsx')

    # 提取X和Y列的數據
    x = df['X'].values
    y = df['Y'].values

    # 將x和y轉換為NumPy數組
    x = np.array(x)
    y = np.array(y)
    #x
    #y

    # 驗證相關係數
    correlation = Pearson_correlation(x, y)
    #驗證相關係數    
    print(f'相關係數(x,y):{Pearson_correlation(x,y)}')    

    # 使用回歸線、其方程和 Pearson 相關係數創建 x-y 圖。
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)

    # 取得包含回歸線方程和相關係數值的字串
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.4f}'
    print(f'line:{line}')

    # 創建 x-y 圖
    fig, ax = plt.subplots()

    # 如果具有相關性，則畫折線連接數據點
    if abs(correlation) > 0.5:
        ax.plot(x, y, label=line, marker='o', linestyle='-')
        ax.set_title('correlation\n'+line)
    else:
        # 如果不具相關性，則顯示點的散布位置
        ax.scatter(x, y, label=line, marker='o')
        ax.set_title('not correlation\n'+line)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(facecolor='white')

    # 保存圖形到檔案夾
    plt.savefig(f'graph/correlationCoefficientExample{i}.png')

    # 顯示當前圖形
    plt.show()


# In[ ]:




