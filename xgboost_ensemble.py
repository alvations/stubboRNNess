import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator

types = {'m1': np.dtype(float), 'm2': np.dtype(float), 'm3': np.dtype(float), 'm4': np.dtype(float), 
         'm5': np.dtype(float), 'target': np.dtype(float)}

train_valid = pd.read_csv("ensemble.train",dtype=types, delimiter=' ')

#etas = [0.01, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.17, 0.16, 0.2]

params = {"objective": "binary:logistic",
          "booster" : "gbtree",
          "eta": 0.13,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.6,
          "silent": 1,
          "seed": 0,
          'eval_metric': 'error'
          }
num_boost_round = 200

features = ['m{}'.format(i) for i in range(1,6)]

X_train, X_valid = train_test_split(train_valid, test_size=0.20, random_state=10)
y_train = np.log1p(X_train.target)
y_valid = np.log1p(X_valid.target)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, verbose_eval=True)

#print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))

testtypes = {'f1': np.dtype(float), 'f2': np.dtype(float), 'f3': np.dtype(float), 'f4': np.dtype(float), 
         'f5': np.dtype(float), 'f6': np.dtype(float), 'f7': np.dtype(float), 'f8': np.dtype(float), 
         'f9': np.dtype(float), 'f10': np.dtype(float), 'f11': np.dtype(float), 'f12': np.dtype(float), 
         'f13': np.dtype(float), 'f14': np.dtype(float), 'c1': np.dtype(str), 't_id':np.dtype(str)}

testtypes = {'m1': np.dtype(float), 'm2': np.dtype(float), 'm3': np.dtype(float), 'm4': np.dtype(float), 
         'm5': np.dtype(float)}

test = pd.read_csv("ensemble.test",dtype=testtypes, delimiter=' ')

dtest = xgb.DMatrix(test[features])
results = gbm.predict(dtest)
# Make Submission
for r in results:
    print (int(r>=0.5))

