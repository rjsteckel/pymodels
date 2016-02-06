# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import lars_path


def histogram(values):
    plt.hist(values)

def plot_fit(y, p):
    plt.plot(p, y,'ro')
    plt.plot([0,50],[0,50], 'g-')
    plt.xlabel('predicted')
    plt.ylabel('real')
    plt.show()
    plt.savefig('docs/foo.png') #, bbox_inches='tight')



def model_feature_importance(weights, feature_names, top_n=25, plotfile=None, title='Variable Importance'):
    model_weights = np.abs(weights)
    weight_importance = 100.0 * (model_weights / model_weights.max())
    sorted_idx = np.argsort(weight_importance)[::-1][:min(len(weights), top_n)]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 1, 1)
    barlist = plt.barh(pos, weight_importance[sorted_idx], align='center')
    #[ barlist[i].set_color('r') for i in sorted_idx if weights[i] < 0 ]
    plt.yticks(pos, [feature_names[i] for i in sorted_idx] )
    plt.xlabel('Relative Importance')
    plt.title(title)
    plt.show()
    if plotfile:
        plt.savefig(plotfile) #, bbox_inches='tight')


def regularized_model_features(X, y):
    alphas, _, coefs = lars_path(X.values, y, method='lasso', verbose=True)
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.show()
    plt.legend()


def plot_corr(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
