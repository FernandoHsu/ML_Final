#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:08:39 2019

@author: cheweihsu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, chi2
from sklearn.pipeline import Pipeline
from constant import DATASET, DATASET
idx = pd.IndexSlice

#%%
# 用log規模當作Y，訓練好後在還原
df = pd.read_csv(DATASET)
def split_XY(df, target, var_to_drop=['parking_price']):
    #df['lgParkPrice'] = np.log(df['parking_price'])
    X = df.drop(columns=var_to_drop + [TARGET])
    Y = np.log(df[TARGET])
    return X, Y

def rm_missing(X,y):
    missVal = X.isnull().mean()
    missVar = X.columns[missVal != 0]
    for var, val in zip(missVar, missVal):
        print('Missing Var on X', (var,val))
    print('Missing Pct on Y', y.isnull().mean())
    X = X.loc[:, ~X.columns.isin(missVar)]
    return X,y
    
#%%
df = (df.set_index(['city','building_id'])
        .sort_index(axis=0)        
)
X, y = split_XY(df, TARGET, var_to_drop=['parking_price'])
X, y = rm_missing(X,y)

#%% Build model

clf = Pipeline([
  ('feature_selection', SelectKBest(mutual_info_regression, k=20)),
  ('classification', RandomForestRegressor(n_jobs = -1))
])
clf.fit(X, y,)
#%%


pred = clf.transform(X)



#%% PCA + SGDClassifier: skip feature selection
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

scoring = {'AUC': 'roc_auc', 'Error': make_scorer(mean_squared_error)}
# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
rf = RandomForestRegressor(n_jobs = -1)
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])


# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': list(range(1,100,5)),
    'rf__n_estimators': [300],
    'rf__max_depth': [500]
}
search = GridSearchCV(pipe, param_grid, iid=False, cv=5, scoring=scoring)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#%%
from helper import plot_auc_error, plot_paramsCV_PCA

plot_paramsCV_PCA(params=[5, 20, 30, 40, 50, 64],cv_results=search.cv_results_)
plot_auc_error(results=search.cv_results_, scoring=scoring, paramCV='pca__n_components')
