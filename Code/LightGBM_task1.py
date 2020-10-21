'''----------------------------------------------------Title------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------'''
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
df = df.drop('Unnamed: 6', axis = 1)
df = df.drop('Unnamed: 7', axis = 1)
df = df.drop('id2', axis = 1)
df1 = df.drop('id1', axis = 1)
# print(data.head())
print(df.tail())
print(df1.head())
Y = list(df['id1'])
# print(Y)

# #Scaling the features using Standard Scaler
sc = StandardScaler()
sc.fit(df1)
X = pd.DataFrame(sc.fit_transform(df1))

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#use 80% datasets as traning set and 20% as test sets
#converting the dataset into proper LGB format
d_train = lgb.Dataset(X_train, label=y_train)

#Specifying the parameter
params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt' #Gradient Boosting Decision Tree
params['objective'] = 'binary' #Binary target feature
params['metric'] = 'binary_logloss' #metric for binary classification
params['max_depth'] = 15
params['num_leaves'] = 120

#train the model
clf = lgb.train(params, d_train, 100)

#prediction on the test set
y_pred = clf.predict(X_test)
print(y_pred[0:5])

# if >= 0.5 ----> 1   else  ----> 0
#rounding the values
y_pred = y_pred.round(0)
print(y_pred[0:5])

#converting from float to integer
y_pred = y_pred.astype(int)
print(y_pred[0:5])

#roc_auc_score metric
print(roc_auc_score(y_pred, y_test)) #0.9590189951216723
print(y_pred[0:5])
