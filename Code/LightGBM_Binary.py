'''----------------------------------------------------LightGBM------------------------------------------------------------
LightGBM Binary Classification, Multi-Class Classification, Regression using Python

LightGBM is a gradient boosting framework that uses tree-based learning algorithms.

Types of Operation supported by LightGBM:
- Regression
- Binary Classification
- Multi-Class Classification
- Cross-Entropy
- Lambdrank

- propertie of Light GBM
Ads:
1. Faster training speed and higher efficiency: Light GBM use histogram based algorithm i.e it buckets continuous feature
values into discrete bins which fasten the training procedure.
更快的训练速度和更高的效率：轻型GBM使用基于直方图的算法，即它将连续的特征值存储到离散的仓中，从而加快了训练过程。

2. Lower memory usage: Replaces continuous values to discrete bins which result in lower memory usage.
较低的内存使用量：将连续的值替换为离散的bin，这会导致较低的内存使用量。

3. etter accuracy than any other boosting algorithm: It produces much more complex trees by following leaf wise split
approach rather than a level-wise approach which is the main factor in achieving higher accuracy. However, it can
sometimes lead to overfitting which can be avoided by setting the max_depth parameter.
比其他任何提升算法都更高的准确性：它是通过遵循逐叶拆分方法而不是逐级方法生成复杂得多的树，这是实现更高准确性的主要因素。但是，有时可能会导致过度拟合，
这可以通过设置max_depth参数来避免。

4. Compatibility with Large Datasets: It is capable of performing equally good with large datasets with a significant
reduction in training time as compared to XGBOOST.
与大型数据集的兼容性：与XGBOOST相比，它在大型数据集上的表现同样出色，并且训练时间大大减少

5. Parallel learning supported.
支持并行学习。

Disads:
1. Leading to overfitting (can be overcome by max_depth)
---------------------------------------------------------------------------------------------------------------------'''
# importing libraries
import numpy as np
from collections import Counter
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score
pd.options.display.max_columns = 999

# 1. Binary Classification using the Breast Cancer Dataset
# loading the breast cancer dataset
X = load_breast_cancer()
# print(X)
df = pd.DataFrame(X.data, columns = X.feature_names)
# print(df)
Y = X.target

#Scaling the features using Standard Scaler
sc = StandardScaler()
sc.fit(df)
X = pd.DataFrame(sc.fit_transform(df))

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#converting the dataset into proper LGB format
d_train = lgb.Dataset(X_train, label=y_train)

#Specifying the parameter
params = {}
params['learning_rate'] = 0.03
params['boosting_type'] = 'gbdt' #Gradient Boosting Decision Tree
params['objective'] = 'binary' #Binary target feature
params['metric'] = 'binary_logloss' #metric for binary classification
params['max_depth'] = 10

#train the model
clf = lgb.train(params, d_train, 100)

#prediction on the test set
y_pred = clf.predict(X_test)
print(y_pred[0:5])

# if >= 0.5 ----> 1   else  ----> 0
#rounding the values
y_pred = y_pred.round(0)

#converting from float to integer
y_pred = y_pred.astype(int)

#roc_auc_score metric
print(roc_auc_score(y_pred, y_test)) #0.965424739195231
print(y_pred[0:5])


