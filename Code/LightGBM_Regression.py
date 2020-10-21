'''----------------------------------------------------Regression-------------------------------------------------------
sklearn参数
sklearn本身的文档当中并没有LightGBM的描述，Github上面看到主要参数如下：

boosting_type : 提升类型，字符串，可选项 (default=gbdt)
gbdt, 传统梯度提升树
dart, 带Dropout的MART
goss, 单边梯度采样
rf, 随机森林

num_leaves : 基学习器的最大叶子树，整型，可选项 (default=31)
max_depth : 基学习器的最大树深度，小于等于0表示没限制，整型，可选项 (default=-1)
learning_rate : 提升学习率，浮点型，可选项 (default=0.1)
n_estimators : 提升次数，整型，可选项 (default=100)
subsample_for_bin : 构造分箱的样本个数，整型，可选项 (default=200000)
objective : 指定学习任务和相应的学习目标或者用户自定义的需要优化的目标损失函数，字符串， 可调用的或者None, 可选项 (default=None)，若不为None，则有:
regression for LGBMRegressor -binary or multiclass for LGBMClassifier
lambdarank for LGBMRanker
class_weight : 该参数仅在多分类的时候会用到，多分类的时候各个分类的权重，对于二分类任务，你可以使用is_unbalance 或 scale_pos_weight，字典数据, balanced or None, 可选项 (default=None)
min_split_gain : 在叶子节点上面做进一步分裂的最小损失减少值，浮点型，可选项 (default=0.)
min_child_weight : 在树的一个孩子或者叶子所需的最小样本权重和，浮点型，可选项 (default=1e-3)
min_child_samples : 在树的一个孩子或者叶子所需的最小样本，整型，可选项 (default=20)
subsample : 训练样本的子采样比例，浮点型，可选项 (default=1.)
subsample_freq : 子采样频率，小于等于0意味着不可用，整型，可选项 (default=0)
colsample_bytree : 构建单棵树时列采样比例，浮点型，可选项 (default=1.)
reg_alpha : $L_1$正则项，浮点型，可选项 (default=0.)
reg_lambda :$L_2$正则项，浮点型，可选项 (default=0.)
random_state : 随机数种子，整型或者None, 可选项 (default=None)
n_jobs : 线程数，整型，可选项 (default=-1)
silent : 运行时是否打印消息，布尔型，可选项 (default=True)
importance_type : 填入到feature_importances_的特征重要性衡量类型，如果是split，则以特征被用来分裂的次数，如果是gain，则以特征每次用于分裂的累积增益，字符串，可选项 (default=split)
---------------------------------------------------------------------------------------------------------------------'''
import bst as bst
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

#loading the Boston Dataset
X = load_boston()
print(X.feature_names)
df = pd.DataFrame(X.data, columns=X.feature_names)
print(df.head())
Y = X.target

#Scaling using the Standard Scaler
sc = StandardScaler()
sc.fit(df)
X = pd.DataFrame(sc.fit_transform(df))

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Converting the data into proper LGB Dataset Format
d_train = lgb.Dataset(X_train, label=y_train)

#Declaring the prarmeters
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt' #GradientBoostingDecisionTree
params['objective'] = 'regression' #regression task
params['n_estimators'] = 100
params['max_depth'] = 10
params['num_leaves'] = 13  #理论上 num_leaves < 2 ^ (max_depth)，否则容易导致 overfitting

#model creation and training
clf = lgb.train(params, d_train, 100)

#model prediction on X_test
y_pred = clf.predict(X_test)
print(y_pred)

#using RMSE error metric
RMSE = np.sqrt(mean_squared_error(y_pred, y_test))
print(RMSE)



