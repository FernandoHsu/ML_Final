#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:18:57 2019

@author: cheweihsu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

DATASET = 'dataset/train.csv'
TARGET = 'total_price'
idx = pd.IndexSlice


#%%
# 用log規模當作Y，訓練好後在還原
df = pd.read_csv(DATASET)

df['Y'] = np.log(df[TARGET])
#df.boxplot(column=TARGET,by='city')
#df.boxplot(column='Y',by='city')

#%%

input_df = np.log(df.loc[:,['parking_price', TARGET]])
pca = PCA(n_components=2)
res = pca.fit_transform(input_df.values)
res.hist()
#df['parking_price'].hist()
#(df[TARGET]/df['parking_price']).hist()

#%% 連續變量：過濾極端值 (1%, 99%) & 標準化


#%%
### 整個資料集
def caculate_MI(df):
    input_df = df.drop(columns=['Y', TARGET])
    max_MI = -np.Inf
    for n in [5,10,20,40,50,65,80,100]:
        encoder = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='quantile')
        res = encoder.fit_transform(df['Y'].values.reshape((-1,1)))
        res = res.reshape(1,-1)[0]
        imp = input_df.apply(lambda col:normalized_mutual_info_score(res, col), axis=0)
        imp = imp[imp>0]
        print('[discretizer Y , MI]: ({0},{1})'.format(n, imp.mean()))
        if imp.mean() > max_MI:
            max_MI = imp.mean()
            out = imp.copy(deep=True)
    out = out.sort_values()
    out.plot.barh(figsize=(80,100))
    plt.title('MI of discretized Y into ' + str(n))
    plt.savefig('impFeature.png')
    out.to_csv('feaImp.csv')

#%%
### 分各縣市
df = (df.set_index(['city','building_id'])
        .sort_index(axis=0)        
)

df_stat = {k:[] for k in df.index.levels[0]}
encoder = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')
for i,(city, frame) in enumerate(df.groupby(level=0)):
    label_true = encoder.fit_transform(df['Y'].values.reshape((-1,1)))
    label_true = label_true[:,0]
    res = frame.apply(lambda col:normalized_mutual_info_score(label_true,col), axis=0)
    res = res.sort_values(ascending=True)
    df_stat[city].append(res)
del i, city, frame, label_true, res


#%%
# plot and save
for city, s in df_stat.items():
    res = s[0]
    res[res<0] = 0
    print(city, res.shape)
    res.plot.barh(figsize=(80,100))
    plt.title('Mutual Info. of ' + str(city))
    plt.savefig('MI-' + str(city) + '.png')
del s, city, res









