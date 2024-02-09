#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from selenium import webdriver
#from webdriver_manager.chrome import ChromeDriverManager  #chrome時
from webdriver_manager.microsoft import EdgeChromiumDriverManager #edge時(預設瀏覽器)
from selenium.webdriver.edge.service import Service
#取得元素
#Selenium 4 不提供 find_element_by_XXX 的方法, 
#只提供取得第一個元素 find_element 或是所有元素的 find_elements 方法, 
#可以搭配 By 類別指定找尋元素的依據。
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from bs4 import BeautifulSoup
import zipfile
import time

#建立edge物件，並且發送請求到"證券交易所的上市公司季報"(可依照需求)網頁
domain_url = 'https://www.twse.com.tw/zh'

#如果你的電腦預設瀏覽器是google則要改成以下的方法，我目前是使用edge的方法
#browser = webdriver.Chrome(ChromeDriverManager().install()) #chrome時
browser = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))
#獲取此網站的資訊
browser.get(
    f'{domain_url}/trading/historical/stock-day-avg.html')
    #後接的網址是要下載區域的子網址連結ex:https://www.twse.com.tw/zh/statistics/statisticsList?type=05&subType=225
    
#要點擊查詢按鈕才會出現檔案資料表格
#所以就需要定位查詢按鈕元素
#可以在查詢按鈕的地方點擊滑鼠右鍵，選擇「檢查」來檢視原始碼
time.sleep(2)
test=1
#爬取從1101~9999的資料
for i in range (1101,9999):
    #並抓取1~12月的資料
    for j in range(1,13):
        try:
            #用selector的方法選擇年的時間(1999)
            select = Select(browser.find_element(By.CSS_SELECTOR, 'select#label0'))
            select.select_by_value("1999")
            #用name的方法選擇月的時間
            select2 = Select(browser.find_element(By.NAME, 'mm'))
            select2.select_by_value(str(j))
            #用selector的方法找到輸入框，搜尋對應的股票代碼，並輸入
            input_element = browser.find_element(By.CSS_SELECTOR, 'input#label1.stock-code-autocomplete')
            input_element.clear()
            input_element.send_keys(str(i))
            #點擊提交
            button = browser.find_element(By.CLASS_NAME,'submit')
            #點擊搜尋
            buttons = button.find_element(By.CLASS_NAME,'search')
            buttons.click()
            time.sleep(3)
            #找到下載csv的地方後點擊下載(資料會於下載區)
            button2 = browser.find_element(By.CLASS_NAME,'rwd-tools')
            button2s = button2.find_element(By.CLASS_NAME,'csv')
            button2s.click()
        except:
            pass
browser.close()


# In[ ]:




