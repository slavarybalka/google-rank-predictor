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
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


df = pd.read_csv("C:\Data Science\pw_ml_dataset-property-management-software.csv")
cols = ['Exact match in Title',
        'Exact Match Meta Desc',
        'Word Count',
        'Inlinks',
        'External Outlinks',
        'Majestic backlinks',
        'Facebook Likes',
        'Response Time',
        'URL Length',
        'Page Title Length',
        'Meta Desc Length',
        'Chars to keyword',
        'Exact Match H1',
        'com']

#WEBPAGE_FEATURES = [1, 1, 130, 34, 23, 22, 9, 0.123, 134, 18, 156, 2, 1, 1]
WEBPAGE_FEATURES = [1, 1, 1200, 40, 230, 220, 9, 0.323, 14, 18, 156, 24, 1, 1]

def compare_features(cols):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df[cols], size=2)

def show_heatmap(cols):
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols, xticklabels=cols)
    

#####


X = df[cols].values
y = df['Rank'].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)

forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

predicted_rank = forest.predict(WEBPAGE_FEATURES)

print "Predicted postion of the web page:", int(predicted_rank[0])

#show_heatmap(cols)



























