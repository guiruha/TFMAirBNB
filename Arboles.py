#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:03:49 2020

@author: Guillem Rochina y Helena Saigi
"""

cd ~/DadesAirBNB

# RANDOM FOREST REGRESSOR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import tqdm
import seaborn as sns

df = pd.read_pickle("~/DadesAirBNB/DatosModelar.pkl")

X = df[df.columns[df.columns.str.contains('PricePNight') == False]]
y = df['PricePNight']

X.drop(['id', 'date', 'goodprice'], axis = 1, inplace = True)

dummycols = ["host_response_time", "neighbourhood_group_cleansed", "property_type", "room_type",  "cancellation_policy"]

<<<<<<< HEAD
df.columns

df.isnull().sum()[df.isnull().sum()>0]

df.drop('TM', axis=1, inplace=True)
df.drop('PPT24H', axis=1, inplace=True)
df.drop('goodprice', axis=1, inplace=True)
df.drop('PricePNight', axis=1, inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Unnamed: 0.1', axis=1, inplace=True)
df.columns
=======
X = pd.get_dummies(X, columns = dummycols)
>>>>>>> 7ef347ea0576856c2d39718b8a1ebd11e4dbbd51

X = X[X.columns[X.dtypes != 'object']]

X.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()

#numcols = list(X.dtypes[((X.dtypes == 'float') | (X.dtypes == 'int')) & (X.dtypes.index.str.contains('rice') == False)].index)

#X_train[numcols] = sc.fit_transform(X_train[numcols])
#X_test[numcols] = sc.transform(X_test[numcols])
RandomForestRegressor?

rf = RandomForestRegressor(n_estimators = 20, n_jobs = -1, max_depth = 10, random_state = 1997, verbose = 10)

rf.fit(X_train, y_train)

rf.score(X_test, y_test)
rf.score(X_train, y_train)

for i, j in zip(df.columns[df.columns.str.contains('PricePNight') == False], rf.feature_importances_):
    if j > 0.01:
        print(i , '->', j)
    
# GRADIENT BOOSTING CON ARBOLES DE DECISIÓN

import xgboost as xgb
from hyperopt import hp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score


xgbreg = xgb.XGBRegressor(colsample_bytree=0.7, learning_rate=0.15, max_depth=30, 
                          n_estimators=30, subsample=0.7, random_state=1997, n_jobs = 8, verbose = 2) 

xgbreg.fit(X_train, y_train, early_stopping_rounds=3, 
           eval_set=[(X_train, y_train), (X_test, y_test)],
           eval_metric = 'rmse', verbose = 1)


fig, ax = plt.subplots(1, 1, figsize = (20, 15))
plt.plot(list(range(1, 31)), xgbreg.evals_result()['validation_0']['rmse'], color = 'maroon')
plt.plot(list(range(1, 31)), xgbreg.evals_result()['validation_1']['rmse'], color = 'navy')
plt.vlines(10, -5, 100, linestyle = '-', color = 'maroon')
plt.vlines(15, -5, 100, linestyle = '-', color = 'maroon')
ax.axvspan(10, 15, alpha=0.4, color='red')
plt.xticks(np.linspace(0 ,30, 11), rotation = 45)
plt.yticks(np.linspace(0, 100, 21))
plt.show()


y_pred = xgbreg.predict(X_test)
y_pred_train = xgbreg.predict(X_train)

explained_variance_score(y_test, y_pred)
r2_score(y_test, y_pred)
r2_score(y_train, y_pred_train)

xgbreg = xgb.XGBRegressor(colsample_bytree=0.7, learning_rate=0.13, max_depth=40, 
                          n_estimators=20, subsample=0.7, random_state=1997, n_jobs = 8, verbose = 2) 

xgbreg.fit(X, y, early_stopping_rounds=3, 
           eval_set=[(X_train, y_train), (X_test, y_test)],
           eval_metric = 'rmse', verbose = 1)

fig, ax = plt.subplots(1, 1, figsize = (20, 15))
plt.plot(list(range(1, 21)), xgbreg.evals_result()['validation_0']['rmse'], color = 'maroon')
plt.plot(list(range(1, 21)), xgbreg.evals_result()['validation_1']['rmse'], color = 'navy')
plt.vlines(10, -5, 100, linestyle = '-', color = 'maroon')
plt.vlines(15, -5, 100, linestyle = '-', color = 'maroon')
ax.axvspan(10, 15, alpha=0.4, color='red')
plt.xticks(np.linspace(0 ,30, 11), rotation = 45)
plt.yticks(np.linspace(0, 100, 21))
plt.show()

y_pred = xgbreg.predict(X_test)
y_pred_train = xgbreg.predict(X_train)

explained_variance_score(y_test, y_pred)
explained_variance_score(y_train, y_pred_train)
r2_score(y_test, y_pred)
r2_score(y_train, y_pred_train)


# Guardo el modelo para mañana
import pickle
filename = '~/users/helenasaigi/DadesAirBNB/ModeloXGBoost.sav'
pickle.dump(xgbreg, open(filename, 'wb'))

xgbmodel = pickle.load(open(filename, 'rb'))

y_pred_train = xgbreg.predict(X_train)
y_pred_test = xgbreg.predict(X_test)

r2_score(y_train, y_pred_train)
r2_score(y_test, y_pred_test)
explained_variance_score(y_test, y_pred)
explained_variance_score(y_test, y_pred)

# COARSE TO FINE TUNNING

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

rparameters = [{'learning_rate': [0.1, 0.13, 0.2], 'reg_alpha': [0.05, 0.1, 0.2], 'reg_lambda':[0.8, 1, 1.3],
                'max_depth': [30, 35, 40]}]

xgbcv = xgb.XGBRegressor(n_estimators=5, subsample=0.7, seed=1997, n_jobs = 8, verbose = 3) 

RandomSearch = RandomizedSearchCV(xgbcv, param_distributions = rparameters,  
                                  n_iter = 10, error_score = 'explained_variance', verbose = 10)

RandomSearch.fit(X, y)

RandomSearch.best_params_
RandomSearch.best_score_


gparameters = {'learning_rate': [0.2, 0.3], 'reg_alpha': [0.05, 0.01, 0.03], 
               'reg_lambda':[0.02, 0.05, 0.07]}

GridSearch = GridSearchCV(xgbcv, param_grid = gparameters, 
                          verbose = 10, cv = 5, iid = True)

GridSearch.fit(X_train, y_train)

GridSearch.best_params_

xgbfinal = xgb.XGBRegressor(n_estimators=10, subsample=0.7, seed=1997, n_jobs = 8, 
                            max_depth =  25, reg_lambda = 0.07, reg_alpha = 0.01, verbose = 3)

xgbfinal.fit(X_train, y_train, early_stopping_rounds=3, 
           eval_set=[(X_train, y_train), (X_test, y_test)],
           eval_metric = 'rmse', verbose = 1)

y_pred_train = xgbfinal.predict(X_train)
y_pred_test = xgbfinal.predict(X_test)

r2_score(y_train, y_pred_train)
r2_score(y_test, y_pred_test)

from xgboost import plot_importance

plot_importance(xgbfinal, max_num_features = 15, importance_type = 'weight')

from sklearn.model_selection import cross_val_score

CVScores = cross_val_score(xgbfinal, X, y, cv = 10, scoring = 'explained_variance', verbose = 3)

# Bayesiano
def objective(parametros):
    parametros = {'num_boost_round': hp.quniform('num_boost_round', 20, 60, 2), 'eta': hp.quniform('eta', 0.1, 0.5, 0.1),
                  'max_depth': hp.quniform('max_depth', 2, 10, 2)}
    xgbreg = xgb.XGBRegressor() 
