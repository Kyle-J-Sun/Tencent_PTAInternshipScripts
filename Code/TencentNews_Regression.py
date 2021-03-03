#!/usr/bin/env python3

""" Try lightgbm for regression """

__author__ = 'Jinkai Sun (jingkai.sun20@imperial.ac.uk)'
__version__ = '0.0.1'

# importing libraries
import numpy as np
from collections import Counter
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score
pd.options.display.max_columns = 999

#loading the Boston Dataset
data = pd.read_csv('/Users/kyle/Documents/Virtual Intern/Tencent/data/total.csv', low_memory=False)
df = pd.DataFrame(data)
# print(df.head())
df = df.drop('Unnamed: 6', axis = 1)
df = df.drop('Unnamed: 7', axis = 1)
df = df.drop('id2', axis = 1)
df1 = df.drop('id1', axis = 1)
# print(data.head())
print(df.tail())
print(df1.head())
Y = list(df['id1'])

#Scaling the features using Standard Scaler
sc = StandardScaler()
sc.fit(df1)
X = pd.DataFrame(sc.fit_transform(df1))

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Converting the data into proper LGB Dataset Format
d_train = lgb.Dataset(X_train, label=y_train)

#Declaring the prarmeters
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt' #GradientBoostingDecisionTree
params['objective'] = 'regression' #regression task
params['n_estimators'] = 100
params['max_depth'] = 15
params['num_leaves'] = 120  #理论上 num_leaves < 2 ^ (max_depth)，否则容易导致 overfitting

#model creation and training
clf = lgb.train(params, d_train, 100)

#model prediction on X_test
y_pred = clf.predict(X_test)
print(y_pred)

#using RMSE error metric
RMSE = np.sqrt(mean_squared_error(y_pred, y_test))
print(RMSE)  #0.1772667172118889

y_pred = y_pred.round(0)
print(y_pred)

#roc_auc_score metric
print(roc_auc_score(y_pred, y_test)) #0.9567805221633807

import os
print(os.path())
