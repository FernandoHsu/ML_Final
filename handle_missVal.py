#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:41:49 2019

@author: cheweihsu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

DATASET = 'dataset/train.csv'
TARGET = 'total_price'
df = pd.read_csv(DATASET)

#%% 4 col with missing value
def handle_missVal(df):
    data = df.copy(deep=True)
    show_miss(data)
    for proc in [proc_1, proc_2, proc_3]:
        proc(data)
    return data

def show_miss(df):
    missVal = df.isnull().mean()
    print(missVal[missVal!=0])
    
def proc_1(df):
    """
    處理這兩個特徵，都補0
    parking_area: 沒有停車位的房子
    parking_price: 沒有付出停車位的錢
    注意：找到新特徵： 18% 人沒付出了停車位錢，卻得到停車位
    """
    park_area, park_price = 'parking_area', 'parking_price'
    park_area_filter = df.loc[:,park_area].isnull()
    park_price_filter = df.loc[:,park_price].isnull()
    
    df.loc[park_area_filter, park_area] = 0
    df.loc[park_price_filter, park_price] = 0
    print('多少人沒付出了停車位錢？',park_price_filter.mean())
    print('多少人沒有得到停車位？',park_area_filter.mean())
    print('多少人沒付了錢，也沒有得到停車位？',(park_area_filter & park_price_filter).mean())
    print('多少人沒付了錢，卻得到停車位？',(~park_area_filter & park_price_filter).mean())
    print('多少人付了錢，卻沒有得到停車位？',(park_area_filter & ~park_price_filter).mean())
    df.loc[:,park_area].hist();plt.title(park_area);plt.subplots()
    df.loc[:,park_price].hist();plt.title(park_price);plt.show()

def proc_2(df):
    """
    處理這個特徵（26.5%），也是補0 (可能為空地)
    txn_floor：交易樓層
    """
    txn_floor_filter = df.txn_floor.isnull()
    df.loc[txn_floor_filter,'txn_floor'] = 0
    df.txn_floor.hist(bins=50)
    plt.title('building_floor')

def proc_3(df):
    """
    處理這個特徵(1.9%遺失)，補中位數，避免分配改變
    village_income_median：這個里的所得中位數
    """
    inc_med_filter = df.village_income_median.isnull()
    df.loc[inc_med_filter,'village_income_median'] = np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    twoD_array = df.village_income_median.values.reshape(-1,1)
    twoD_array = imp.fit_transform(twoD_array).reshape(1,-1)[0]
    df.loc[:,'village_income_median'] = twoD_array
    df.loc[:,'village_income_median'].hist(bins=10)
    plt.title('income_median')
    
    
#%% 
#data = handle_missVal(df)
#show_miss(data)

    
    