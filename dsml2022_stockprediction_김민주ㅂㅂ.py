#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pymysql')
get_ipython().system('pip install sqlalchemy')


# In[2]:


import os
import time
import pandas as pd
import numpy as np
import datetime
import pymysql
from sqlalchemy import create_engine
import FinanceDataReader as fdr
from tqdm import tqdm


# In[3]:


code_data = pd.read_csv('./raw_data.csv', index_col = 0)


# code_date['상장일'] dtype을 object → datetime 변경

# In[4]:


code_data['상장일'] = pd.to_datetime(code_data['상장일'])
code_data['종목코드'] = code_data['종목코드'].apply(lambda x: str(x).zfill(6))


# In[5]:


code_data[:5]


# # 과제 1

# 데이터를 dataframe에서 list로 변환

# In[7]:


lst_stock = code_data.values.tolist()


# dictionary 사용

# In[8]:


dic_code2company = {}
for row in lst_stock:
    company, code, date = row[0], row[1], row[4]
    if date <= datetime.datetime(2018, 1, 2):
        dic_code2company[code] = company


# In[9]:


OF = open('assignment1_sql.txt', 'w', encoding = 'utf-8')
for code in dic_code2company.keys():
    data = '{}\t{}\n'.format(code, dic_code2company[code])
    OF.write(data)
OF.close()


# # 과제 2

# In[10]:


db_dsml = pymysql.connect(
    host = 'localhost', 
    port = 3306, 
    user = 'stock_user', 
    passwd = 'bigdata', 
    db = 'stock', 
    charset = 'utf8'
)

cursor = db_dsml.cursor()


# In[11]:


dic_code2date = {}

OF = open('assignment2_sql.txt', 'w', encoding = 'utf-8')
for code in tqdm(dic_code2company.keys()):
    sql_query = '''
                SELECT *
                FROM stock_{}
                WHERE Date
                BETWEEN '2018-01-01' AND '2021-12-31'
                '''.format(code)
    stock = pd.read_sql(sql = sql_query, con = db_dsml)
    lst_stock = stock.values.tolist()
    
    for row in lst_stock:
        if row[4] * row[5] >= 100000000000:
            data = '{}\t{}\n'.format(code, row[0].strftime('%Y%m%d'))
            OF.write(data)
            if code not in dic_code2date.keys():
                dic_code2date[code] = []
                dic_code2date[code].append(row[0])
            else:
                dic_code2date[code].append(row[0])
OF.close()


# # 과제 3

# In[11]:


OF = open('assignment3_sql.txt', 'w', encoding = 'utf-8')
for code in tqdm(dic_code2date.keys()):
    sql_query = '''
                SELECT *
                FROM stock_{}
                WHERE Date
                BETWEEN '2018-01-01' AND '2020-12-31'
                '''.format(code)
    stock = pd.read_sql(sql = sql_query, con = db_dsml)
    lst_stock = stock.values.tolist()
    
    for i, row_lst_stock in enumerate(lst_stock):
        # 예외 처리
        if (i < 9) or (i >= len(lst_stock)-1):
            continue  
        date = row_lst_stock[0]
        if date not in dic_code2date[code]:
            continue
        
        # 11 days data
        sub_stock = lst_stock[i-9:i+1]
        lst_data = []
        for row_sub_stock in sub_stock:
            open, high, low, close, volume = row_sub_stock[1:6]
            trading_value = close * volume
            lst_data += [open, high, low, close, trading_value]
            del open
        data = ','.join(map(str, lst_data))
        
        # label
        label = int(lst_stock[i+1][-1] >= 0.02)
        
        result = '{}\t{}\t{}\t{}\n'.format(code, date.strftime("%Y%m%d"), data, label)
        OF.write(result)
OF.close()


# # 과제 3-2

# In[12]:


OF = open('assignment3-2_sql.txt', 'w', encoding = 'utf-8')
for code in tqdm(dic_code2date.keys()):
    sql_query = '''
                SELECT *
                FROM stock_{}
                WHERE Date
                BETWEEN '2021-01-01' AND '2021-06-30'
                '''.format(code)
    stock = pd.read_sql(sql = sql_query, con = db_dsml)
    lst_stock = stock.values.tolist()
    
    for i, row_lst_stock in enumerate(lst_stock):
        # 예외 처리
        if (i < 9) or (i >= len(lst_stock)-1):
            continue  
        date = row_lst_stock[0]
        if date not in dic_code2date[code]:
            continue
        
        # 11 days data
        sub_stock = lst_stock[i-9:i+1]
        lst_data = []
        for row_sub_stock in sub_stock:
            open, high, low, close, volume = row_sub_stock[1:6]
            trading_value = close * volume
            lst_data += [open, high, low, close, trading_value]
            del open
        data = ','.join(map(str, lst_data))
        
        # label
        label = int(lst_stock[i+1][-1] >= 0.02)
        
        result = '{}\t{}\t{}\t{}\n'.format(code, date.strftime("%Y%m%d"), data, label)
        OF.write(result)
OF.close()


# In[ ]:




