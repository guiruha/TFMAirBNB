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

corrs = []

for column in corrcolumns:
    corrs.append(scipy.stats.pointbiserialr(X_raw[column], X_raw['LogPricePNight'])[0])

templot = pd.DataFrame({'Atributo': corrcolumns, 'CorrCoef': corrs})

templot['colors'] = templot['CorrCoef'].apply(lambda x: 'positivo' if x > 0 else 'negativo')

fig, ax = plt.subplots(1, 1, figsize = (40, 10))
sns.barplot('Atributo', 'CorrCoef', data = templot, hue = 'colors', palette = 'Set1')
ax.hlines(y = 0.5, color = "blue", linestyle = '--', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = -0.5, color = "red", linestyle = '--', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = 0.3, color = "blue", linestyle = '-.', xmin = -1, xmax = templot.shape[0])
ax.hlines(y = -0.3, color = "red", linestyle = '-.', xmin = -1, xmax = templot.shape[0])
plt.xticks(rotation = 90)
plt.show()

np.corrcoef(X_raw['room_type_Private room'], X_raw['LogPricePNight'])[0, 1]
scipy.stats.pointbiserialr(X_raw['room_type_Private room'], X_raw['LogPricePNight'])

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

corrcolumns = list(X_raw.dtypes[((X_raw.dtypes == 'float') | (X_raw.dtypes == 'int'))].index)

mask = np.zeros_like(X_raw[corrcolumns].corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(1, 1, figsize = (35, 30))
sns.heatmap(round(X_raw[corrcolumns].corr(), 3), vmin = -1, vmax = 1, center = 0, mask = mask, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()

corrcolumns = list(X_raw.dtypes[((X_raw.dtypes == 'float') | (X_raw.dtypes == 'int')) & (X_raw.dtypes.index.str.contains('rice') == False)].index)

atributos = [x for x in corrcolumns if np.abs(np.corrcoef(X_raw[x], X_raw['LogPricePNight'])[0, 1]) > 0.20]

X_model = X_raw[atributos]
y_model = X_raw['LogPricePNight']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size = 0.3)

lr = LinearRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)

