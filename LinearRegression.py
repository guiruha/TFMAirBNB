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

factorcolumns = [x for x in list(X_raw.dtypes[(X_raw.dtypes == 'uint8')].index) if 0 in X_raw[x].value_counts().index]

X_raw[factorcolumns] = X_raw[factorcolumns].astype('category')

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

sns.barplot?
np.corrcoef(X_raw['room_type_Private room'], X_raw['LogPricePNight'])
scipy.stats.pointbiserialr(X_raw['room_type_Private room'], X_raw['LogPricePNight'])

X_raw['room_type_Shared room'].value_counts()

colineales = []
for column in correlations.columns:
    for row in correlations.index:
        if (np.abs(correlations.loc[row, column]) > 0.7) & (correlations.loc[row, column] !=  1):
            print(row, 'y', column, 'tienen una correlación de', correlations.loc[row, column])
            colineales.append(row)
selection = ['bedrooms', 'beds', 'accommodates', 'review_scores_accuracy', 'review_scores_value', 'review_scores_rating']
for column in selection:
    print(column, 'LogPricePNight', X_raw['LogPricePNight'].corr(X_raw[column]))


