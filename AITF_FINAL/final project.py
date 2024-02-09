#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_file_path='top200_1997.xlsx'

df = pd.read_excel(csv_file_path)
df.head()


# In[2]:


from sklearn.preprocessing import StandardScaler

columns_to_exclude = ['簡稱', '證券代碼', '年月','ReturnMean_year_Label']

# 排除指定的欄位
features_to_scale = df.drop(columns=columns_to_exclude)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
df_feat.head()


# In[ ]:


#選擇特定欄位分析
from sklearn.preprocessing import StandardScaler

# 指定要選擇的欄位
selected_columns = ['特定欄位1', '特定欄位2', '特定欄位3']

# 選擇指定的欄位
features_to_scale = df[selected_columns]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat = pd.DataFrame(scaled_features, columns=selected_columns)
df_feat.head()


# In[3]:


#將資料分成訓練組及測試組
from sklearn.model_selection import train_test_split

X = df_feat
y = df['ReturnMean_year_Label']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[4]:


#使用KNN演算法
from sklearn.neighbors import KNeighborsClassifier

clf=KNeighborsClassifier(n_neighbors=29,p=2,weights='distance',algorithm='brute')
clf.fit(X_train,y_train)


# In[29]:


clf.predict(X_test)


# In[30]:


clf.score(X_test,y_test)


# In[31]:


clf.score(X_train,y_train)


# In[27]:


error_rate = []

for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i, p=2, weights='distance', algorithm='brute')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# 將k=1~60的錯誤率製圖畫出。k=23之後，錯誤率就在5-6%之間震盪。
plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# In[35]:


csv_file_path = 'top200_1998.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# In[36]:


csv_file_path = 'top200_1999.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# In[37]:


csv_file_path = 'top200_2000.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# In[38]:


csv_file_path = 'top200_2001.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# In[62]:


csv_file_path = 'top200_2002.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# 選擇預測為1的股票
selected_stocks = new[predictions_new == 1]

# 計算return
stock_returns = selected_stocks['Return']

portfolio_returns = stock_returns.sum()

print(portfolio_returns)


# In[40]:


csv_file_path = 'top200_2003.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# In[41]:


csv_file_path = 'top200_2004.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')


# In[5]:


csv_file_path = 'top200_2005.xlsx'
new = pd.read_excel(csv_file_path)

# 排除指定的欄位
features_to_scale_new = new.drop(columns=columns_to_exclude)

# 使用之前訓練好的標準化物件進行標準化
scaled_features_new = scaler.transform(features_to_scale_new)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

# 使用已經訓練好的模型進行預測
predictions_new = clf.predict(df_feat_new)
#print(predictions_new)

# 比較預測結果
accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
print(f'新數據的預測準確率: {accuracy_new}')

predicted_positive_indices = (predictions_new == 1)

# 獲取股票名稱
predicted_positive_stock_names = new.loc[predicted_positive_indices, '簡稱']

# 預測要投資的股票名稱
print("選擇股票:")
print(predicted_positive_stock_names)


# 選擇預測為1的股票
selected_stocks = new[predictions_new == 1]

# 計算return
stock_returns = selected_stocks['Return']

portfolio_returns = stock_returns.sum()

print(portfolio_returns)


# In[ ]:




