#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:50:12 2019

@author: cheweihsu
"""
import numpy as np

TRAINSET = 'dataset/train.csv'
TESTSET = 'dataset/test.csv'
TARGET = 'total_price'


# 該房屋方圓 ?? 公尺內 ?? 類別種類數
feat_count = np.array(['VI_50', 'XI_10', 'I_10', 'IV_10', 'N_10000',
       'IX_10', 'N_5000', 'VI_100', 'II_10', 'X_10', 'I_50', 'XI_50',
       'X_50', 'III_10', 'XIV_10', 'I_100', 'XI_100', 'V_10', 'IV_50',
       'VII_10', 'X_100', 'II_50', 'III_50', 'IX_50', 'XII_10', 'N_50',
       'N_1000', 'XIII_10', 'N_500', 'IV_100', 'VIII_10', 'V_50',
       'VI_250', 'I_250', 'II_100', 'IX_100', 'V_100', 'III_100', 'X_250',
       'XI_250', 'XIII_50', 'I_500', 'VI_500', 'VII_50', 'IV_250',
       'II_250', 'XIV_50', 'X_500', 'XI_500', 'XII_50', 'VI_1000',
       'III_250', 'I_1000', 'VIII_50', 'IV_500', 'V_250', 'XIII_100',
       'IX_250', 'VIII_100', 'II_500', 'VII_100', 'XII_100', 'XI_1000',
       'III_500', 'IV_1000', 'V_500', 'XIV_100', 'XIII_250', 'IX_500',
       'XIII_500', 'XIII_1000', 'X_1000', 'XII_250', 'VIII_250',
       'II_1000', 'VI_5000', 'IX_1000', 'XII_500', 'VII_250', 'VIII_500',
       'III_1000', 'IV_5000', 'V_1000', 'VI_10000', 'XIV_250', 'IV_10000',
       'I_5000', 'VII_500', 'XIII_5000', 'VIII_1000', 'XII_1000',
       'XIV_500', 'I_10000', 'XI_5000', 'VII_1000', 'X_5000', 'II_5000',
       'XIII_10000', 'XIV_1000', 'IX_5000', 'V_5000', 'X_10000',
       'XI_10000', 'III_5000', 'II_10000', 'IX_10000', 'V_10000',
       'III_10000', 'VIII_5000', 'XII_5000', 'VII_5000', 'XIV_5000',
       'VIII_10000', 'VII_10000', 'XIV_10000', 'XII_10000'])


# 該房屋方圓 ?? 公尺內有無 ?? 類別
feat_bool = np.array(['VI_index_10000', 'III_index_10000', 'XII_index_5000',
       'VIII_index_10000', 'X_index_5000', 'XII_index_10000',
       'VIII_index_5000', 'IV_index_10000', 'I_index_5000',
       'VI_index_5000', 'VII_index_5000', 'I_index_10000',
       'III_index_5000', 'IX_index_10000', 'V_index_10000',
       'XIV_index_10000', 'II_index_10000', 'XIV_index_5000',
       'VII_index_10000', 'V_index_5000', 'II_index_5000',
       'X_index_10000', 'VII_index_500', 'III_index_500', 'XI_index_500',
       'III_index_1000', 'VII_index_50', 'XIII_index_50', 'III_index_50',
       'VII_index_1000', 'II_index_50', 'X_index_50', 'VI_index_50',
       'XII_index_50', 'V_index_1000', 'V_index_500', 'VI_index_500',
       'VI_index_1000', 'XI_index_10000', 'IV_index_50', 'V_index_50',
       'XII_index_500', 'IV_index_5000', 'XII_index_1000',
       'IV_index_1000', 'XI_index_50', 'IV_index_500', 'XI_index_1000',
       'XI_index_5000', 'II_index_1000', 'II_index_500', 'XIV_index_1000',
       'XIV_index_500', 'IX_index_5000', 'XIV_index_50', 'X_index_500',
       'XIII_index_10000', 'IX_index_1000', 'XIII_index_5000',
       'XIII_index_1000', 'VIII_index_50', 'IX_index_50', 'I_index_50',
       'IX_index_500', 'X_index_1000', 'VIII_index_1000',
       'XIII_index_500', 'I_index_1000', 'I_index_500', 'VIII_index_500'])

# 該房屋與最近的 ?? 類別之距離
feat_dist = np.array(['XIV_MIN', 'VII_MIN', 'XII_MIN', 'II_MIN', 'I_MIN', 'X_MIN',
       'III_MIN', 'VIII_MIN', 'V_MIN', 'VI_MIN', 'IX_MIN', 'IV_MIN',
       'XI_MIN', 'XIII_MIN'])

feature_type = {
        'int':{'count':np.array(['txn_floor','total_floor']),
               'scale':feat_count},
        'bool':feat_bool,
        'category':np.array(['parking_way','building_type',
                    'building_use','building_material',
                    'city','town','village']),
        'float':
            {'dist':feat_dist,
             'price':np.array(['parking_price','total_price','village_income_median']),
             'area':np.array(['town_area','building_area','land_area','parking_area']),
             'scale':np.array(['lat','lon']),
             'density':np.array(['town_population_density'])
             }
        }