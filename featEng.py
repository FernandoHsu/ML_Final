#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:56:03 2019
由於特稱scale 落差太大！這份文件將所有變量做數學變換到更對稱的分配
# 1. 調整Y變量取log訓練，最後上傳時需要取exp還原
# 2. 將特徵作出有意義的組合
# 3. 根據observation的結果篩選掉不重要特徵

@author: cheweihsu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale, robust_scale, power_transform, normalize
from handle_missVal import handle_missVal

def trans(series, processor, log):
    twoD_array = series.values.reshape(-1,1)
    if log:
        twoD_array = np.log(twoD_array)
    twoD_array = processor(twoD_array)
    return pd.Series(twoD_array.reshape(1,-1)[0])

def trans_plot(df, col, log=False):
    fig, ax = plt.subplots(2,2, figsize=(8,8))
    for i,method in enumerate([minmax_scale, normalize, power_transform, robust_scale]):
        res = trans(df.loc[:,col], method, log)
        plt.subplot(2, 2, i+1)
        res.hist()
        plt.title(method.__name__)

def feature_enginner(df, target):
    # feature
    combine_feature(df)
    scaling_feature(df)
    # target
    transform_target(df, target)
    X, y = split_frame(df, target)
    return X, y

def combine_feature(data):
    data['diff_txn_complete']=data['txn_dt']-data['building_complete_dt']
    data.drop(columns=['txn_dt', 'building_complete_dt'])
    
    data['priceInc_ratio'] = data['parking_price']/data['village_income_median'] 
    data.drop(columns=['village_income_median'])
    
    data['highEduc_rate'] = data['bachelor_rate']+data['master_rate']+data['doc_rate']         
    data['midEduc_rate'] = data['highschool_rate']+data['jobschool_rate']    
    data['lowEduc_rate'] =  data['junior_rate']+data['elementary_rate'] 
    data.drop(columns=['bachelor_rate', 'master_rate', 'doc_rate',
                       'highschool_rate', 'jobschool_rate', 
                       'junior_rate', 'elementary_rate'
                       ])
    
    data['nat_pop'] = data['born_rate']-data['death_rate']
    data['soc_pop'] = data['marriage_rate']-data['divorce_rate']
    data.drop(columns=['born_rate', 'death_rate', 'marriage_rate', 'divorce_rate'])
    
def transform_target(df, target):
    pass

def scaling_feature(df):
    pass

#%%
DATASET = 'dataset/train.csv'
TARGET = 'total_price'
idx = pd.IndexSlice
df = pd.read_csv(DATASET)
data = handle_missVal(df)

#%% scale 落差太大！
#data.loc[:,['parking_price',TARGET]].plot(kind='box',subplots=True)

#data['Y'] = np.log(data.loc[:,TARGET]**1)
#data['PI'] = data['Y'] / data['village_income_median']
#g = sns.FacetGrid(data, row="city",height=1.7, aspect=4,)
#g.map(sns.distplot, TARGET, hist=False, rug=True);

#%%  create_newFeature
#col_1, col_2 = 'marriage_rate', 'divorce_rate'
#new_col = data[col_1]-data[col_2]
#data.loc[:,[col_1,col_2]].hist()
#plt.subplots()
#new_col.hist()

#%%
x = data.loc[:,'parking_area']
y = data.loc[:,'parking_price']
plt.figure(figsize=(8,7))
plt.scatter(x, y, c=Y, marker='v')
#plt.plot(x, make_pred(x, reg), lw=2.0)
plt.colorbar()
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
#%%

    
#%% TARGET
trans_plot(data, col='Y', log=True)

#%% PARKING PRICE
#trans_plot(df, col='parking_price', log=True)
#(data.loc[:,'parking_price']/data.loc[:,'parking_area']).hist()
#data.plot(kind='box')