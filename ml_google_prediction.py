# -*- coding: utf-8 -*-

# python 2.7
# using supervised learning to predict the position
# of the page in Google for a given keyword

import numpy as np
import pandas as pd
import matplotlib as plt

import seaborn as sns

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score


#################### SETTING THE STAGE ###########################################

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
     

X = df[cols].values
y = df['Rank'].values


############################# DEFINING FUNCTIONS ####################################

def compare_features(cols):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df[cols], size=2)

def show_heatmap(cols):
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols, xticklabels=cols)
    


############################### SPLITTING THE DATA INTO TRAINING AND TESTING SETS ######################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


############################## IMPLEMENTING A RANDOM FOREST REGRESSOR #################################

forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)

forest = forest.fit(X_train, y_train)


y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

train_accuracy = forest.score(X_train, y_train)
test_accuracy = forest.score(X_test, y_test)

############################### PREDICT THE POSITION #####################################################

WEBPAGE_FEATURES = [1, 1, 130, 34, 23, 22, 9, 0.123, 134, 18, 156, 0, 1, 1]
predicted_rank = forest.predict(WEBPAGE_FEATURES)


############################### DISPLAYING THE RESULTS ###################################################

#show_heatmap(cols)
print "Predicted postion of the web page:", int(predicted_rank[0])
print "Training set accuracy: ", train_accuracy
print "Testing set accuracy: ", test_accuracy
if train_accuracy > test_accuracy:
    print "Overfitting! Add more data to the dataset."






























