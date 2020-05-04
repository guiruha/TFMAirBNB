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

df = pd.read_csv("~/DadesAirBNB/DatosModelar.csv")

dummycols = ["host_response_time", "neighbourhood_group_cleansed", "property_type", "room_type",  "cancellation_policy"]

df = pd.get_dummies(df, columns = dummycols)

df = df[df.columns[df.dtypes != 'object']]

df.columns

df.isnull().sum()[df.isnull().sum()>0]

df.drop('TM', axis=1, inplace=True)
df.drop('PPT24H', axis=1, inplace=True)
df.drop('goodprice', axis=1, inplace=True)
df.drop('PricePNight', axis=1, inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Unnamed: 0.1', axis=1, inplace=True)
df.columns

X = df[df.columns[df.columns.str.contains('LogPricePNight') == False]]
y = df['LogPricePNight']

X.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1997)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()

#numcols = list(X.dtypes[((X.dtypes == 'float') | (X.dtypes == 'int')) & (X.dtypes.index.str.contains('rice') == False)].index)

#X_train[numcols] = sc.fit_transform(X_train[numcols])
#X_test[numcols] = sc.transform(X_test[numcols])

rf = RandomForestRegressor(n_estimators = 100, n_jobs = 4, max_depth = 10, random_state = 1997)
rf.fit(X_train, y_train)

rf.score(X_test, y_test)
rf.score(X_train, y_train)

# GRADIENT BOOSTING CON ARBOLES DE DECISIÓN

import xgboost as xgb
from hyperopt import hp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score


xgbreg = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0, learning_rate=0.07, max_depth=5, 
                          n_estimators=1000, subsample=0.7, seed=1997, n_jobs = 8) 

xgbreg.fit(X_train, y_train)

y_pred = xgbreg.predict(X_test)
y_pred_train = xgbreg.predict(X_train)

explained_variance_score(y_test, y_pred)
r2_score(y_test, y_pred)

# Guardo el modelo para mañana
import pickle
filename = '~/users/helenasaigi/DadesAirBNB/ModeloXGBoost.sav'
pickle.dump(xgbreg, open(filename, 'wb'))

xgbmodel = pickle.load(open(filename, 'rb'))

y_pred_train = xgbreg.predict(X_train)
y_pred_test = xgbreg.predict(X_test)

r2_score(y_train, y_pred_train)
r2_score(y_test, y_pred_test)


# COARSE TO FINE TUNNING

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

rparameters = {'colsample_bytree': [0.5, 0.7, 0.9], 'learning_rate': [0.05, 0.07, 0.1, 0.15], 'n_estimators': [500, 700, 1000, 1200], 
              'reg_alpha': [0.01, 0.05, 0.1], 'reg_lambda':[0.01, 0.05, 0.1]}

xgbcv = xgb.XGBRegressor(n_jobs = 4, seed = 1997)

RandomSearch = RandomizedSearchCV(xgbcv, param_distributions = rparameters, n_iter = 10, scoring = 'r2')

RandomSearch.fit(X_train, y_train)

RandomSearch.best_params_

result = pd.DataFrame({'Atributos': X_train.columns, 'Coeficiente': xgbmodel.feature_importances_})

result.sort_values(by = 'Coeficiente', ascending = False)[0:15]

"""
gparameters = {'colsample_bytree': [0.5, 0.7, 0.9], 'learning_rate': [0.05, 0.07, 0.1, 0.15], 'n_estimators': [500, 700, 1000, 1200], 
              'reg_alpha': [0.01, 0.05, 0.1], 'reg_lambda':[0.01, 0.05, 0.1]}
GridSearch = GridSearchCV(xgbcv, param_grid = gparameters, n_jobs = 4, scoring = 'r2')

GridSearch.fit(X_train, y_train)

finalparameters = GridSearch.best_params_

xgbfinal = xgb.XGBRegressor(**finalparameters)

# BAYESIANO
def objective(parametros):
    parametros = {'num_boost_round': hp.quniform('num_boost_round', 20, 60, 2), 'eta': hp.quniform('eta', 0.1, 0.5, 0.1),
                  'max_depth': hp.quniform('max_depth', 2, 10, 2)}
    xgbreg = xgb.XGBRegressor() 

"""
