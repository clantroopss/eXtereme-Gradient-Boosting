# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:31:14 2018

@author: clantroops
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import pandas as pd
from xgboost import plot_tree
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logregobj(preds, dtrain, i):
    """log likelihood loss"""
    labels = dtrain.get_label()
    if( i == 1):
        results['Labels_'+ str(i)] = labels
    results['Prediction_'+ str(i)] = preds
    preds = sigmoid(preds)
    results['Sigmoid_Pred_'+ str(i)] = preds
    grad = preds - labels
    results['Gradient_'+ str(i)] = grad
    hess = preds * (1.0-preds)
    results['Hessian_'+ str(i)] = hess
    return grad, hess


# Build a toy dataset.
X, Y = make_classification(n_samples=1000, n_features=5, n_redundant=0, n_informative=3,
                           random_state=1, n_clusters_per_class=1)

# Instantiate a Booster object to do the heavy lifting
dtrain = xgb.DMatrix(X, label=Y)
params = {'max_depth': 2, 'eta': 1, 'silent': 0, 'base_score': 0.5}
model = xgb.Booster(params, [dtrain])
model.dump_model('dump.raw.txt', with_stats= True)
results = pd.DataFrame()

fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(model, num_trees=0, ax=ax)
plt.show()
# Run 10 boosting iterations
# g and h can be monitored for gradient statistics
for i in range(1):
    pred = model.predict(dtrain)
    g, h = logregobj(pred, dtrain,i+1)    
    model.boost(dtrain, g, h)

model.dump_model('dump.raw.txt', with_stats= True)
#convert toy dataset into panda DataFrame object
df = pd.DataFrame(data =np.c_[X, Y])
#booster[0]:
#0:[f0<0.562576] yes=1,no=2,missing=1,gain=827.161,cover=235.004
#	1:[f0<0.489862] yes=3,no=4,missing=3,gain=8.27962,cover=106.457
#		3:leaf=1.57206,cover=103.637
#		4:leaf=-0.122908,cover=2.82004
#	2:[f2<-1.0492] yes=5,no=6,missing=5,gain=73.999,cover=128.547
#		5:leaf=0.649199,cover=7.99013
#		6:leaf=-2.42127,cover=120.557
#booster[1]:
#0:[f2<0.428415] yes=1,no=2,missing=1,gain=80.4231,cover=110.067
#	1:[f3<1.93018] yes=3,no=4,missing=3,gain=32.6623,cover=66.842
#		3:leaf=1.00711,cover=59.1609
#		4:leaf=-1.08557,cover=7.68103
#	2:[f0<0.468391] yes=5,no=6,missing=5,gain=6.98334,cover=43.2249
#		5:leaf=0.707158,cover=1.68867
#		6:leaf=-1.06264,cover=41.5363

print("Root Cover for the Booster 0:", sum(h))
criteria1 = df[0] < 0.562576
condition1  = results[criteria1]
print("Decision Node 1 - Cover for f0 < 0.562576 yes and missing:", sum(condition1['Hessian_1']))
criteria3_1 = df[0] < 0.562576
criteria3_2 = df[0] < 0.489862
criteria3 = criteria3_1 & criteria3_2
condition3 = results[criteria3]
print("Leaf Value 3:", -(sum(condition3['Gradient_1']) )/ (sum(condition3['Hessian_1']) +1))
print(" Leaf Value 1 - Cover for f0 < 0.562576 & f0 < 0.489862 yes and missing:", sum(condition3['Hessian_1']))

criteria4_1 = df[0] < 0.562576
criteria4_2 = df[0] >= 0.489862
criteria4 = criteria4_1 & criteria4_2
condition4 = results[criteria4]
print("Leaf Value 4:", -(sum(condition4['Gradient_1']) )/ (sum(condition4['Hessian_1']) +1))
print(" Leaf Value 2 - Cover for f0 < 0.562576 & f0 >= 0.489862 no:", sum(condition4['Hessian_1']))

criteria2 = df[0] >= 0.562576
condition2  = results[criteria2]
print("Decision Node 2 - Cover for f0 >= 0.562576 no", sum(condition2['Hessian_1']))

criteria5_1 = df[0] >= 0.562576
criteria5_2 = df[2] < -1.0492
criteria5 = criteria5_1 & criteria5_2
condition5 = results[criteria5]
print("Leaf Value 5:", -(sum(condition5['Gradient_1']) )/ (sum(condition5['Hessian_1']) +1))
print(" Leaf Value 1 - Cover for f0 >= 0.562576 & f2 < -1.0492 yes and missing:", sum(condition5['Hessian_1']))

criteria6_1 = df[0] >= 0.562576
criteria6_2 = df[2] >= -1.0492
criteria6 = criteria6_1 & criteria6_2
condition6 = results[criteria6]
print("Leaf Value 6:", -(sum(condition6['Gradient_1']) )/ (sum(condition6['Hessian_1']) +1))
print(" Leaf Value 2 - Cover for f0 < 0.562576 & f2 >= -1.0492 no:", sum(condition6['Hessian_1']))

gain = ( (np.square(sum(condition1['Gradient_1']))/(sum(condition1['Hessian_1'])+1))+ (np.square(sum(condition2['Gradient_1']))/(sum(condition2['Hessian_1'])+1)) - (np.square(sum(results['Gradient_1']))/(sum(results['Hessian_1'])+1)) )
print("Root Gain for the Booster 0:", gain)

decision_node1 = ( (np.square(sum(condition3['Gradient_1']))/(sum(condition3['Hessian_1'])+1))+ (np.square(sum(condition4['Gradient_1']))/(sum(condition4['Hessian_1'])+1)) - (np.square(sum(condition1['Gradient_1']))/(sum(condition1['Hessian_1'])+1)) )
print("Decision Node 1: Gain for the Booster 0:", decision_node1)

decision_node2 = ( (np.square(sum(condition5['Gradient_1']))/(sum(condition5['Hessian_1'])+1))+ (np.square(sum(condition6['Gradient_1']))/(sum(condition6['Hessian_1'])+1)) - (np.square(sum(condition2['Gradient_1']))/(sum(condition2['Hessian_1'])+1)) )
print("Decision Node 2: Gain for the Booster 0:", decision_node2)

# Evaluate predictions    
yhat = model.predict(dtrain)
yhat = 1.0 / (1.0 + np.exp(-yhat))
yhat_labels = np.round(yhat)
confusion_matrix(Y, yhat_labels)
