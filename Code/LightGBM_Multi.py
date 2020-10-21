'''---------------------------------------------Multi-Class Classification----------------------------------------------
Parameters:
For best fit:
1. num_leaves : This parameter is used to set the number of leaves to be formed in a tree. Theoretically relation
between num_leaves and max_depth is num_leaves= 2^(max_depth). However, this is not a good estimate in case of Light GBM
since splitting takes place leaf wise rather than depth wise. Hence num_leaves set must be smaller than 2^(max_depth)
otherwise it may lead to overfitting. Light GBM does not have a direct relation between num_leaves and max_depth and
hence the two must not be linked with each other.

2. min_data_in_leaf : It is also one of the important parameters in dealing with overfitting. Setting its value smaller
may cause overfitting and hence must be set accordingly. Its value should be hundreds to thousands of large datasets.

3. max_depth: It specifies the maximum depth or level up to which tree can grow.

For faster speed:
1. bagging_fraction : Is used to perform bagging for faster results

2. feature_fraction : Set fraction of the features to be used at each iteration

3. max_bin : Smaller value of max_bin can save much time as it buckets the feature values in discrete bins which is
computationally inexpensive.

For better accuracy:
1. Use bigger training data

2. num_leaves : Setting it to high value produces deeper trees with increased accuracy but lead to overfitting. Hence
its higher value is not preferred.

3. max_bin : Setting it to high values has similar effect as caused by increasing value of num_leaves and also slower
our training procedure.
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

#loading the dataset
X1 = load_wine()
# print(X1.feature_names)
df_1 = pd.DataFrame(X1.data, columns=X1.feature_names)
Y_1 = X1.target
print(Y_1)

#Scaling using the Standard Scaler
sc_1 = StandardScaler()
sc_1.fit(df_1)
X_1 = pd.DataFrame(sc_1.fit_transform(df_1))

#train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size=0.3, random_state=0)

#Converting the dataset in proper LGB format
d_train = lgb.Dataset(X_train, label = y_train)

#setting up the prarmeters
params = {}
params['learning_rate'] = 0.03
params['boosting_type'] = 'gbdt' # GradientBoostingDecisionTree
params['objective'] = 'multiclass' #Multi-class target feature
params['metric'] = 'multi_logloss' #metric for multi-class
params['max_depth'] = 10
params['num_class'] = 3 #no.of unique values in the target class not inclusive of the end value

# trainning the model
clf = lgb.train(params, d_train, 100) #targeting the model on 100 epocs

#prediction on the test dataset
y_pred_1 = clf.predict(X_test)

#printing the predctions
print(y_pred_1)

'''---------------------------------------------Multi-Class Classification----------------------------------------------
In a multi-class problem, the model produces num_class(3) probabilities as shown in the output above. We can use 
numpy.argmax() method to print the class which has the most reasonable result.
---------------------------------------------------------------------------------------------------------------------'''

#argmax() method
y_pred_2 = [np.argmax(line) for line in y_pred_1]
#printing the predictions
print(y_pred_2)

#using precision score for error metrics
prec_score = precision_score(y_pred_2, y_test, average=None).mean()
print(prec_score)  #0.9545454545454546

