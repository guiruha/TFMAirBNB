#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:02:04 2020

@author: guillem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

df = pd.read_csv("~/DadesAirBNB/DatosModelar.csv")

df.dtypes[df.dtypes == 'object']

dummycols = ["host_response_time", "neighbourhood_group_cleansed", "property_type", "room_type",  "cancellation_policy"]

X_raw = pd.get_dummies(df, columns = dummycols)

X_raw.drop('id', inplace = True, axis = 1)

corrcolumns = X_raw.dtypes[X_raw.dtypes != 'object'].index[X_raw.dtypes[X_raw.dtypes != 'object'].index.str.contains('rice') == False]

corrBis = []
corrPea = []

for column in corrcolumns:
    corrBis.append(round(scipy.stats.pointbiserialr(X_raw[column], X_raw['LogPricePNight'])[0], 3))
    corrPea.append(round(np.corrcoef(X_raw[column], X_raw['LogPricePNight'])[0, 1], 3))
    
templot = pd.DataFrame({'Atributo': corrcolumns, 'CorrBiserial': corrBis, 'CorrdePearson': corrPea})

templot['colors'] = templot['CorrBiserial'].apply(lambda x: 'positivo' if x > 0 else 'negativo')

fig, ax = plt.subplots(1, 1, figsize = (40, 10))
sns.barplot('Atributo', 'CorrBiserial', data = templot, hue = 'colors', palette = 'Set1')
ax.hlines(y = 0.5, color = "blue", linestyle = '--', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = -0.5, color = "red", linestyle = '--', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = 0.3, color = "blue", linestyle = '-.', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = -0.3, color = "red", linestyle = '-.', xmin = -1, xmax = templot.shape[0])
plt.xticks(rotation = 90)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (40, 10))
sns.barplot('Atributo', 'CorrdePearson', data = templot, hue = 'colors', palette = 'Set1')
ax.hlines(y = 0.5, color = "blue", linestyle = '--', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = -0.5, color = "red", linestyle = '--', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = 0.3, color = "blue", linestyle = '-.', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = -0.3, color = "red", linestyle = '-.', xmin = -1, xmax = templot.shape[0])
plt.xticks(rotation = 90)
plt.show()

np.corrcoef(X_raw['room_type_Private room'], X_raw['LogPricePNight'])[0, 1]
scipy.stats.pointbiserialr(X_raw['room_type_Private room'], X_raw['LogPricePNight'])[0]

corrcolumns = list(X_raw.dtypes[((X_raw.dtypes == 'float') | (X_raw.dtypes == 'int')) & (X_raw.dtypes.index.str.contains('rice') == False)].index)
correlations = X_raw[corrcolumns].corr()

colineales = []
for column in corrcolumns:
    for row in corrcolumns:
        if (np.abs(correlations.loc[row, column]) > 0.7) & (correlations.loc[row, column] !=  1):
            print(row, 'y', column, 'tienen una correlación de', correlations.loc[row, column])
            colineales.append('{} <-> {} == {}'.format(row, column, correlations.loc[row, column]))
            
selection = ['bedrooms', 'beds', 'accommodates', 'review_scores_value', 'review_scores_rating']
for column in selection:
    print(column, 'LogPricePNight', X_raw['LogPricePNight'].corr(X_raw[column]))

X_raw.drop(['beds', 'review_scores_rating'], axis = 1, inplace = True)

X_raw = X_raw[[x for x in X_raw.columns if x not in ['LogPricePNight']] + ['LogPricePNight']]

corrcolumns = list(X_raw.dtypes[((X_raw.dtypes == 'float') | (X_raw.dtypes == 'int')) & (X_raw.dtypes.index.str.contains('rice') == False)].index)

mask = np.zeros_like(X_raw[corrcolumns].corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(1, 1, figsize = (35, 30))
sns.heatmap(round(X_raw[corrcolumns].corr(), 3), vmin = -1, vmax = 1, center = 0, mask = mask, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()

atributos = [x for x in corrcolumns if np.abs(np.corrcoef(X_raw[x], X_raw['LogPricePNight'])[0, 1]) > 0.15]

factorcolumns = [x for x in list(X_raw.dtypes[(X_raw.dtypes == 'uint8')].index | X_raw.dtypes[(X_raw.dtypes == 'int')].index) 
                 if (0 in X_raw[x].value_counts().index)&(x.endswith('cercanos') == False)]

X_raw[factorcolumns] = X_raw[factorcolumns].astype('category')

X_model = X_raw[atributos]
y_model = X_raw['LogPricePNight']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size = 0.3, random_state = 1997)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

numcols = list(X_model.dtypes[((X_model.dtypes == 'float') | (X_model.dtypes == 'int')) & (X_model.dtypes.index.str.contains('rice') == False)].index)

X_train[numcols] = sc.fit_transform(X_train[numcols])
X_test[numcols] = sc.transform(X_test[numcols])

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)
lr.score(X_train, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y_pred = lr.predict(X_test)

mean_squared_error(np.exp(y_pred), np.exp(y_test))
mean_absolute_error(np.exp(y_pred), np.exp(y_test))

# PROBAMOS CON TODAS LAS VARIABLES
X_raw.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace = True)

X_model_tot = X_raw[list(X_raw.dtypes[((X_raw.dtypes != 'object')) & (X_raw.dtypes.index.str.contains('rice') == False)].index)]

X_train, X_test, y_train, y_test = train_test_split(X_model_tot, y_model, test_size = 0.3)

numcols = list(X_model_tot.dtypes[(X_model_tot.dtypes == 'float') | (X_model_tot.dtypes == 'int')].index)

X_train[numcols] = sc.fit_transform(X_train[numcols])
X_test[numcols] = sc.transform(X_test[numcols])

lrtotal = LinearRegression()

lrtotal.fit(X_train, y_train)

lrtotal.score(X_train, y_train)

lrtotal.score(X_test, y_test)

# APLICAMOS UNA LASSO

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.01)

lasso.fit(X_train, y_train)

lasso.score(X_train, y_train)

lasso.score(X_test, y_test)

lassoatr = []
for atr, coef in zip(X_model_tot.columns, lasso.coef_):
    if np.abs(coef) > 0:
        lassoatr.append(atr)

# REGRESIÓN LINEAL CON ATRIBUTOS SELECCIONADOS CON LASSO

X_final = X_model_tot[lassoatr]
y_final = X_raw['LogPricePNight']

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.3)

lrfinal = LinearRegression()

lrfinal.fit(X_train, y_train)

lrfinal.score(X_train, y_train)
lrfinal.score(X_test, y_test)

y_pred_final = lrfinal.predict(X_test)

mean_squared_error(y_test, y_pred_final)
mean_absolute_error(y_test, y_pred_final)

listafinal = []
for atr, coef in zip(X_final.columns, lrfinal.coef_):
    listafinal.append('El atributo {} tiene un coeficiente de {}'.format(atr, coef))

for i in  listafinal:
    print(i)

from sklearn.metrics import r2_score

print('El modelo final, utilizando {} atributos, ha conseguido un R-Cuadrado de {} % con un MSE de {}'\
      .format(X_final.shape[1], round(r2_score(y_test, y_pred_final)*100, 2), mean_squared_error(np.exp(y_test), np.exp(y_pred_final))))

