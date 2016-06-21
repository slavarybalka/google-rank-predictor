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

df = pd.read_csv("C:\Data Science\pw_ml_dataset.csv")
#print df.columns

'''
cols = ['KW in Title',
        'Exact Match Meta Desc',
        'Word Count',
        'Inlinks',
        'External Outlinks',
        'Response Time',
        'Page Title Length',
        'Chars to keyword',
        'Exact Match H1',
        'on-page optimization',
        'Rank']
'''     
cols = ['on-page optimization',
        'Word Count',
        'Response Time',
        'Page Title Length',
        'Chars to keyword',   
        'Inlinks',
        'External Outlinks',
        'URL length'
        ]
        

X = df[cols].values
y = df['Rank'].values


############################# DEFINING FUNCTIONS ####################################

def compare_features(cols):
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(df[cols], size=2)

def show_heatmap(cols):
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=2.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)
    
# an idea: no need to use all possible returned result, can just focus on Top 100 and compare the metrics of the Top 10 against the rest (11-100).
# to see if there are any correlations between the results for different keywords.    

# add a function for reading_level test

############################### SPLITTING THE DATA INTO TRAINING AND TESTING SETS ######################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


############################## IMPLEMENTING A RANDOM FOREST REGRESSOR #################################

forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest = forest.fit(X_train, y_train)

############################## CALCULATING ACCURACY ################################################

train_accuracy = forest.score(X_train, y_train)
test_accuracy = forest.score(X_test, y_test)

############################### PREDICTING THE POSITION #####################################################

WEBPAGE_FEATURES = [3,1200,0.234,56,0,25,24,65]
predicted_rank = forest.predict(WEBPAGE_FEATURES)


############################### DISPLAYING THE RESULTS ###################################################

show_heatmap(cols)
#print "Predicted postion of the web page:", int(predicted_rank[0])

print "Training set accuracy: {:.1%}".format(train_accuracy)
print "Testing set accuracy: {:.1%}".format(test_accuracy)
if train_accuracy > test_accuracy:
    print "Overfitting! Add more data to the dataset."






























