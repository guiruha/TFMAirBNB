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
   
df.drop(['number_of_reviews_ltm', 'license', 'Unnamed: 0'], axis = 1, inplace = True)
    
df.groupby(['date', 'id'])['goodprice'].count()[df.groupby(['date', 'id'])['goodprice'].count()>1]

df.drop_duplicates(subset = ['date', 'id'], inplace = True)

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
sns.distplot(np.log(df['PricePNight']), bins = 13)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
plt.plot(df.groupby('date')['PricePNight'].mean().index, df.groupby('date')['PricePNight'].mean(), color = "red")
plt.xticks(df.groupby('date')['PricePNight'].mean().index, rotation = 75)
plt.tight_layout()