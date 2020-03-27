#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:19:56 2020

@author: guillem
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



archives = !ls | grep 'df'

df = pd.read_csv(archives[0])

for i in range(1, len(archives)):
    df = df.append(pd.read_csv(archives[i]))
   
df.drop(['number_of_reviews_ltm', 'license', 'Unnamed: 0', 'requires_license'], axis = 1, inplace = True)
    
df.groupby(['date', 'id'])['goodprice'].count()[df.groupby(['date', 'id'])['goodprice'].count()>1]

df.drop_duplicates(subset = ['date', 'id'], inplace = True)

df.drop('price', axis = 1, inplace = True)

df.isnull().sum()[df.isnull().sum()>0]

agrupacion = df.groupby('date')['goodprice'].describe()

# La Luxury Villa es el último OUTLIER amb sentit en quant al preu (1200€)
df[df['goodprice']>1200]['id'].value_counts()

df = df[(df['goodprice']<1200)]

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
plt.plot(df.groupby('date')['goodprice'].mean().index, df.groupby('date')['goodprice'].mean(), color = "red")
plt.xticks(df.groupby('date')['goodprice'].mean().index, rotation = 75)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['goodprice'])
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(np.log(df['goodprice']), bins = 13)
plt.tight_layout()

df[['goodprice']].describe()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.boxplot(df['neighbourhood_group_cleansed'], df['goodprice'], whis = 10)
plt.tight_layout()

df[(df['goodprice']>300)&(df['minimum_nights']>1)][['id', 'goodprice', 'minimum_nights']]

df['PricePNight'] = [price/minnight if (price > 300)&(minnight > 1) else price for price, minnight in zip(df['goodprice'], df['minimum_nights'])]

df[(df['goodprice']>300)&(df['minimum_nights']>1)&(df['PricePNight']<30)][['id', 'goodprice', 'minimum_nights', 'PricePNight']].sort_values(['PricePNight'])

df[(df['goodprice']>300)&(df['minimum_nights']>1)&(df['PricePNight']<30)&(df['minimum_nights']<30)]\
    [['id', 'goodprice', 'minimum_nights', 'PricePNight']].sort_values(['PricePNight'])

# HE DESCOBERT QUE ELS QUE TENEN MÉS DE 28 DÍES DE MINIMUM_NIGHTS FIQUEN EL PREU MENSUAL

df[(df['minimum_nights']>28)&(df['goodprice']>20)&(df['PricePNight']< 10)][['goodprice', 'PricePNight']] # Estes files sobren

df.drop(df[(df['minimum_nights']>28)&(df['goodprice']>20)&(df['PricePNight']< 10)].index, inplace = True)

#Buscador per ficar al google chrome o firefox
def who(x):
    print('\n', df[df['id']==x]['name'].unique(), '\n')
    return 'https://www.airbnb.es/rooms/' + str(x)

who(10425744)

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['PricePNight'], bins = 13)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(np.log(df['PricePNight']), bins = 13)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
plt.plot(df.groupby('date')['PricePNight'].mean().index, df.groupby('date')['PricePNight'].mean(), color = "red")
plt.xticks(df.groupby('date')['PricePNight'].mean().index, rotation = 75)
plt.tight_layout()

df['LogPricePNight'] = np.log(df['PricePNight'])

# Pareix que a partir de 12 llits son outliers respecte al preu amb prou Leverage

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
plt.scatter(df['bedrooms'], df['LogPricePNight'], color = "navy", alpha = 0.05)
sns.regplot('bedrooms', 'LogPricePNight', data = df, scatter = False, color = "red")
plt.tight_layout()

# El mateix pasa a partir de 8 banys

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
plt.scatter(df['bathrooms'], df['LogPricePNight'], color = "navy", alpha = 0.05)
sns.regplot('bathrooms', 'LogPricePNight', data = df, scatter = False, color = "red")
plt.tight_layout()

# Crec que hi ha colinealitat entre bathrooms y bedrooms
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
plt.scatter(df['bathrooms'], df['bedrooms'], color = "navy", alpha = 0.05)
sns.regplot('bathrooms', 'bedrooms', data = df, scatter = False, color = "red")
plt.tight_layout()

np.corrcoef(df['bedrooms'], df['bathrooms'])

corrcolumns = list(df.dtypes[df.dtypes != 'object'].index)

matrixcorrA = df[corrcolumns[1:21]].corr()

sns.heatmap(matrixcorrA, vmin = -1, vmax = 1, cmap = "viridis", linecolor = "black")

matrixcorrB = df[corrcolumns[21:41]].corr()

sns.heatmap(matrixcorrB, vmin = -1, vmax = 1, cmap = "viridis", linecolor = "black")

matrixcorrC = df[corrcolumns[41:]].corr()

sns.heatmap(matrixcorrC, vmin = -1, vmax = 1, cmap = "viridis", linecolor = "black")

fig, ax = plt.subplots(1, 1, figsize = (30, 30))
sns.heatmap(df[corrcolumns[1:]].corr(), vmin = -1, vmax = 1, center = 0, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()

# EL DE BAIX ES PER FER UNA MATRIU TRIANGULAR INFERIOR PER ESTALVIAR LA PART SUPERIOR QUE ES INNECESARIA
mask = np.zeros_like(df[corrcolumns[1:]].corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(1, 1, figsize = (35, 30))
sns.heatmap(df[corrcolumns[1:]].corr(), vmin = -1, vmax = 1, center = 0, mask = mask, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['accommodates', 'instant_bookable', 'review_scores_rating', 'host_total_listings_count', 'Smoking allowed', 'bathrooms', 'beds', 'cleaning_fee', 'guests_included', 'availability_365', 'Air conditioning', 'Family/kid friendly']].values

y = df['LogPricePNight'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1997)

lineareg = LinearRegression()

lineareg.fit(X_train, y_train)

lineareg.score(X_test, y_test)

# HO FAIG AMB TOTES LES VARIABLES
X_total = df[corrcolumns[1:-3]].values

X_train, X_test, y_train, y_test = train_test_split(X_total, y, test_size = 0.3, random_state = 1997)

lineareg = LinearRegression()

lineareg.fit(X_train, y_train)

lineareg.score(X_test, y_test)

# PROBEM AMB UNA RIDGE
from sklearn.linear_model import Ridge

ridreg = Ridge(random_state = 1997)
ridreg.fit(X_train, y_train)

ridreg.score(X_test, y_test)

for n,v in zip(corrcolumns[1:-3], lassoreg.coef_):
    print('{} -> {}'.format(n, v))
    
# UTILITZE NOMÉS ELS ATRIBUTS DE LA RIDGE

X = df[['host_is_superhost', 'host_has_profile_pic', 'cleaning_fee', 'guests_included', 'minimum_nights', 'availability_365', 'number_of_reviews', 'review_scores_accuracy']]
y = df['LogPricePNight'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1997)

lineareg = LinearRegression()

lineareg.fit(X_train, y_train)

lineareg.score(X_test, y_test)
# No té massa sentit el que diu la RIDGE