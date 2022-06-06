#!/usr/bin/env python
# coding: utf-8

# # 과제 4-1
# 
# 학습 데이터셋(assignment3.txt)를 받아서 모델을 학습하여 저장(model_????.pickle)

# In[22]:


#from sklearn.preprocessing import MinMaxScaler

#scaler= MinMaxScaler()
#scaler.fit(trainX)
#X_scaled = scaler.fit_transform(trainX)   


# In[1]:


import numpy as np
import sklearn.metrics as metrics
import pickle
from sklearn.linear_model import LogisticRegression

IF=open("assignment3_sql.txt",'r')
lst_code_date=[]
trainX=[]
trainY=[]
for line in IF:
    code, date, x, y = line.strip().split("\t")
    lst_code_date.append([code, date])
    trainX.append(list(map(int, x.split(","))))
    trainY.append(int(y))

trainX=np.array(trainX)
trainY=np.array(trainY)
clf = LogisticRegression(C=0.01,penalty='l2')
clf.fit(trainX, trainY)

with open('model_logistic.pickle', 'wb') as f:
    pickle.dump(clf, f)


# In[25]:


from sklearn.model_selection import GridSearchCV
clf=LogisticRegression()
params={'penalty':['l2', 'l1'],
        'C':[0.01, 0.1, 1, 10,100]}

grid_clf = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=3 )
grid_clf.fit(trainX, trainY)

print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, 
                                                  grid_clf.best_score_))


# # 과제 4-2
# 
# 학습한 모델(model_????.pickle)과 시험 데이터셋(assignment3-2.txt)을 받아서 상승 여부 예측

# In[26]:


import numpy as np
import sklearn.metrics as metrics
import pickle
from sklearn.linear_model import LogisticRegression

IF=open("assignment3-2_sql.txt",'r')
lst_code_date=[]
testX=[]
testY=[]
for line in IF:
    code, date, x, y = line.strip().split("\t")
    lst_code_date.append([code, date])
    testX.append(list(map(int, x.split(","))))
    testY.append(int(y))
testX=np.array(testX)
testY=np.array(testY)

with open('model_logistic.pickle', 'rb') as f:
    clf = pickle.load(f)


# In[27]:


predY = clf.predict_proba(testX) # predict_proba 함수는 예측한 값을 확률 값으로 출력
predY2 = clf.predict(testX)  # predict 함수는 예측한 값을 이진 값(1 또는 0)으로 출력

print(predY)


# In[28]:


print(sum(predY >= 0.43))


# In[29]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testY, predY2)
print(accuracy)


# # 과제 4-3
# 
# 예측한 값으로부터 주문 요청 일지(assignment4.txt) 작성

# In[30]:


lst_output=[]
same_date=[]
for (code, date), y in zip(lst_code_date, predY):
    if y[1] >= 0.43: 
        lst_output.append([code, date, "buy", "b58"])  
        lst_output.append([code, date+"n", "sell", "all"])
    elif 0.4 <= y[1] <0.43:
        lst_output.append([code, date, "buy", "a28"])
        lst_output.append([code, date+"n", "sell", "all"])
        
    elif 0.37<= y[1] <0.4:
        lst_output.append([code, date, "buy", "r10"])
        lst_output.append([code, date+"n", "sell", "all"])


lst_output.sort(key=lambda x:x[1]) # date 기준으로 주문 요청 결과를 정렬

OF=open("trading2022firsthalf.txt",'w') # 주문 요청 일지를 파일로 기록
for row in lst_output:
    OF.write("\t".join(map(str, row))+"\n")
OF.close()


# In[31]:


import os
import time
import pandas as pd
import numpy as np
import datetime
import pymysql
from sqlalchemy import create_engine
import FinanceDataReader as fdr
from tqdm import tqdm



db_dsml = pymysql.connect(
    host = 'localhost', 
    port = 3306, 
    user = 'stock_user', 
    passwd = 'bigdata', 
    db = 'stock', 
    charset = 'utf8'
)

cursor = db_dsml.cursor()


# # 과제 5
# 
# 주문일지(assignment4.txt)를 받아서 수익률을 계산한다.

# In[32]:



start_money = 10000000 # 초기 현금 1천만원
money = start_money
dic_code2num ={}  # 보유 종목

IF=open("trading2022firsthalf.txt",'r')
for i, line in enumerate(tqdm(IF)): #주문 일지를 한 줄 읽어 옴
    code, date, request, amount = line.strip().split("\t")

    ##############################################################################################
    sql_query = '''
                SELECT *
                FROM stock_{}
                WHERE Date
                BETWEEN '2021-01-01' AND '2021-08-30'
                '''.format(code)
    stock = pd.read_sql(sql = sql_query, con = db_dsml)
    lst_stock = stock.values.tolist()

   
    
    if 'n' in date:
        for ii,row in enumerate(lst_stock):
            Date=row[0].strftime('%Y%m%d')
            if date == Date:
                close= lst_stock[ii+1][4]
                
    else:
        for row2 in lst_stock:
            Date=row2[0].strftime('%Y%m%d')
            if Date in date:
                close = row2[4]


    #close = 12345 # 종목(code)의 해당일(date) 또는 해당일 다음날(date+n)에 대한 종가를 읽어오도록 이 부분 코드를 수정할 것!!!
    ##############################################################################################
    
    if request == 'buy': # buy인 경우
        if amount.startswith('r'):#startswith 대소문자를 구분하고 인자값에 있는 문자열이 string에 있으면 true, 없으면 false를 반환한다.
            request_money = money * float(amount.lstrip("r")) / 100 #0.10 보유금액의 10프로 매수
        elif amount == 'all':
            request_money = money
        elif amount.isdigit():
            request_money = int(amount)
        elif amount.startswith('b'):
            request_money = money * float(amount.lstrip("b")) / 100

        elif amount.startswith('a'):
            request_money = money * float(amount.lstrip("a")) / 100
    
        # elif amount == ~~~~~    ##### 기타 필요한 매수 요청 옵션이 있을 시 작성
        else:
            raise Exception('Not permitted option')
        request_money = min(request_money, money)
        buy_num = int(request_money / close)
        money -= buy_num * close  # 현재 금액(money)을 실제 매수액을 뺀 만큼 업데이트
        if code not in dic_code2num:
            dic_code2num[code] = 0
        dic_code2num[code] += buy_num # 보유 종목 데이터에 구매 종목(code)를 매수 개수 만큼 증가
    if request == 'sell': # sell인 경우
        if amount == 'all':
            sell_num = dic_code2num[code]
        # elif amount == ~~~~~    ##### 기타 필요한 매도 요청 옵션이 있을 시 작성
        else:
            raise Exception('Not permitted option')            
        money += sell_num * close
        dic_code2num[code] -= sell_num
        if dic_code2num[code] == 0:
            del dic_code2num[code]
IF.close()            
            
if dic_code2num != {}: # 매매가 종료되었는데 보유 종목이 있으면
    raise Exception('Not empty stock') 

print("Final earning rate : {} %".format(str((money-start_money) / start_money * 100)))

