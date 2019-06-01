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
from sklearn.preprocessing import minmax_scale, RobustScaler, power_transform, normalize, QuantileTransformer
from handle_missVal import handle_missVal
from constant import TRAINSET, TESTSET, TARGET
from constant import feat_bool, feat_count, feat_dist, feature_type

idx = pd.IndexSlice
#%%
def remove_feature(df):
    pass

def feature_enginner(df, target, y_trans=None):
    """
    Split data and do pre-processing.
    
    @ df: (data frame) entire .csv file
    @ target: (str), columns name of target
    @ y_trans: (str), log or quantile or robust
    """
    # feature
    combine_feature(df)
    #scaling_feature(df) ### not requie for tree-model
    remove_feature(df)
    # target
    y_transformer = transform_target(df, target, y_trans)
    # split 
    X, y = split_df(df, target)
    return X, y, y_transformer

def trans(series, processor):
    twoD_array = series.values.reshape(-1,1)
    try:
        twoD_array = processor(twoD_array)
    except ValueError as e:
        print(e)
    return pd.Series(twoD_array.reshape(1,-1)[0])

def trans_plot(df, col):
    nrow, ncol = 3, 2
    quantrans = QuantileTransformer(output_distribution='normal', random_state=0)
    robust_scale = RobustScaler(quantile_range=(5.0, 95.0))
    process = [np.log, minmax_scale, normalize, power_transform,
               robust_scale.fit_transform, quantrans.fit_transform]
    
    fig, ax = plt.subplots(nrow, ncol, figsize=(8,8))
    for i,method in enumerate(process):
        res = trans(df.loc[:,col], method)
        plt.subplot(nrow, ncol, i+1)
        try:
            sns.kdeplot(res)
        except:
            try:
                res.hist()
            except:
                print('Skip {} due to error'.format(method.__name__))
                continue
        if i == 4:
            title = 'RobustScaler'
        elif i == 5:
            title = 'QuantileTransformer'
        else:
            title = method.__name__
        plt.title(title)

def combine_feature(df):
    # 新屋舊屋
    df['diff_txn_complete']=df['txn_dt']-df['building_complete_dt']
    df.drop(columns=['txn_dt', 'building_complete_dt'], inplace=True)
    # 房價所得比
    #df['priceInc_ratio'] = df['parking_price']/df['village_income_median'] 
    df.drop(columns=['village_income_median'])
    # 簡化收入
    df['highEduc_rate'] = df['bachelor_rate']+df['master_rate']+df['doc_rate']         
    df['midEduc_rate'] = df['highschool_rate']+df['jobschool_rate']    
    df['lowEduc_rate'] =  df['junior_rate']+df['elementary_rate'] 
    df.drop(columns=['bachelor_rate', 'master_rate', 'doc_rate',
                       'highschool_rate', 'jobschool_rate', 
                       'junior_rate', 'elementary_rate'
                       ], inplace=True)
    # 人口自然/社會增加
    df['nat_pop'] = df['born_rate']-df['death_rate']
    df['soc_pop'] = df['marriage_rate']-df['divorce_rate']
    df.drop(columns=['born_rate', 'death_rate', 'marriage_rate',
                       'divorce_rate'], inplace=True)
    
    # 比率 (容積率等等)
    #df['r1'] = df.building_area / df.land_area
    #df['r2'] = df.parking_area / df.land_area
    #df['r3'] = df.parking_area / df.building_area
    #df['r4'] = df.building_area / df.town_area
    #df.loc[df.r2.isnull(),'r2'] = 0
    #df.loc[df.r3.isnull(),'r3'] = 0
    

def transform_target(df, target, method):
    if not method:
        return 
    
    if method == 'log':
        y_transformer = np.log
    elif method == 'quantile':
        y_transformer = QuantileTransformer(output_distribution='normal',
                                            random_state=0)       
    elif method == 'robust':
        y_transformer = RobustScaler(quantile_range=(5.0, 95.0))
    df[target] = trans(df[TARGET], y_transformer.fit_transform)
    return y_transformer

def scaling_feature(df):
    #num_features
    df.loc[:,feat_dist] /= 1000 #公尺變公里
    # 平移資料，以台灣為資料中心點，不伸縮因為無意義
    df.lon -= df.lon.mean()
    df.lat -= df.lat.mean()
    #min-max scaling
    df[feat_count] = minmax_scale(df[feat_count])
    df['town_area'] = trans(df['building_area'], minmax_scale) # 1d array 
    df['building_area'] = np.log(df['building_area'])
    df['land_area'] = np.log(df['land_area'])
    df['parking_area'] = np.log(df['parking_area'])
    

def split_df(df, target):
    y = df[target]
    X = df.drop(columns=[target])    
    return X, y



#%%
#df = pd.read_csv(TRAINSET)
#df = df.set_index(['building_id'])
#data = handle_missVal(df)
#X, y, y_transformer = feature_enginner(data, TARGET, y_trans=None)
#trans_plot(data, col=TARGET)


#%% (Y,X) scale 落差太大！ 但是 tree model 沒差
# <sol-1> 把 y 依照 parking_price 的規模 sclaing
# <sol-2> 用quantile transform，算完inverse回去
 
#data.plot(kind='box')


#g = sns.FacetGrid(data, row="city",height=1.7, aspect=4,)
#g.map(sns.distplot, TARGET, hist=False, rug=True);

#col_scale = data.apply(lambda col:np.quantile(col,.95)-np.quantile(col,.05))
#col_scale = col_scale.sort_values()


#%% <2> （高難度) 將各項設施col_index_距離 groupby(city)後依照各距離排名
#frame = data.groupby('city')
#frame.pivot(index='Item', columns='CType', values='USD')



