#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:27:11 2019

@author: cheweihsu
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



def split_XY(df, target, var_to_drop=['parking_price']):
    #df['lgParkPrice'] = np.log(df['parking_price'])
    X = df.drop(columns=var_to_drop + [target])
    Y = np.log(df[target])
    return X, Y

def rm_missing(X,y):
    missVal = X.isnull().mean()
    missVar = X.columns[missVal != 0]
    for var, val in zip(missVar, missVal):
        print('Missing Var on X', (var,val))
    print('Missing Pct on Y', y.isnull().mean())
    X = X.loc[:, ~X.columns.isin(missVar)]
    return X,y

def cv_1sigma(cv_results, score='test'):
    """
    Calculate the bound within 1 standard deviation
    of the best `mean_test_scores`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`
    score: test or train

    Returns
    -------
    float
        Lower/Upper bound within 1 standard deviation of the
        best `mean_test/train_score`.
    """
    assert  score=='test' or score == 'train', 'Score must be train/test'
    best_score_idx = np.argmax(cv_results['mean_'+score+'_score'])
    best_score_idx = np.argmax(cv_results['mean_'+score+'_score'])
    best_score_mean = cv_results['mean_'+score+'_score'][best_score_idx]
    best_score_std = cv_results['mean_'+score+'_score'][best_score_idx]
    return (best_score_mean-best_score_std, best_score_mean+best_score_std)




def error_decomposion(y_true, y_pred):
    """
    Error = bias + var + noise
    see: https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html
    """
    y_error = mean_squared_error(y_true, y_pred)
    y_noise = np.var(y_true, axis=1)
    y_bias = (y_true - np.mean(y_pred, axis=1)) ** 2
    y_var = np.var(y_pred, axis=1)
    return y_error,y_noise, y_bias, y_var

def plot_error_decomposion(X_train, y_train, X_test, y_predict, name):
    y_error,y_noise, y_bias, y_var = error_decomposion(y_train, y_predict)
    # Plot train/pred data points
    plt.subplot(2, 1, 1)
    plt.plot(X_train, y_train, ".b", label="train data: y")
    plt.plot(X_test, y_predict, "r", label=r"test: $\^y(x)$")
    plt.plot(X_test, np.mean(y_predict, axis=1), "c", label=r"$\mathbb{E}_{LS} \^y(x)$")
    plt.xlim([-5, 5])
    plt.title(name)
    plt.legend(loc=(1.1, .5))
    # plot decompostion
    plt.subplot(2, 1, 2)
    plt.plot(X_test, y_error, "r", label="$error(x)$")
    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")
    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])
    plt.legend(loc=(1.1, .5))

    plt.subplots_adjust(right=.75)
    plt.show()
    
def plot_paramsCV_PCA(params, search):
    for dataset in ['train', 'test']:
        avg = search.cv_results_['mean_%s_score' % dataset]
        std = search.cv_results_['std_%s_score' % dataset]
        plt.plot(params,avg, linewidth=2, label='%s'%(dataset))
        plt.fill_between(params, avg-std, avg+std,                         
                         alpha=0.1 if dataset == 'test' else 0)
    plt.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend()
    plt.show()    
    
def plot_auc_error(results, scoring, paramCV='min_samples_split'):
    """
    This plot multi-scoring function during CV GridSearch.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
    """
    from sklearn.metrics import make_scorer
    paramCV = 'param_' + paramCV
    
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402);ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[paramCV].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()