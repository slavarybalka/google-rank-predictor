# -*- coding: utf-8 -*-

# python 2.7
# using supervised learning to predict the position
# of the page in Google for a given keyword


'''
1. # of external backlinks
2. # of social signals
3. # page size
4. # page loading speed
5. # outbound backlinks
'''
import numpy as np
import pandas as pd
import matplotlib as plt

import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("C:\Data Science\pw_ml_dataset.csv")
cols = ['Rank',
        'Exact match in Title',
        'Exact Match Meta Desc',
        'Word Count',
        'Inlinks',
        'Outlinks',
        'External Outlinks',
        'Social Signals',
        'Response Time',
        'URL Length']

#cols = ['rank', 'test']
def compare_features(cols):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df[cols], size=2)

def show_heatmap(cols):
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)
    

#print df.head()
show_heatmap(cols)
compare_features(cols)

#plt.show()
'''
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

#TEST_FEATURES_TEMPLATE = [[external_backlinks, social_signals, page_size, page_loading_time, outbound_backlinks]]
#TEST_FEATURES = [[1, 18, 36, 23, 2]] #sample metrics of the page we are testing



X = df[['Size']].values
y = df[['Word Count']].values

tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('Page Size')
plt.ylabel('Word Count')
plt.show()

print X, Y
#print df.head()


def get_external_backlinks():
    end

def get_social_signals():
    end

def get_page_size():
    end


def get_page_loading_time():
    end

def get_outbound_backlinks():
    end
'''