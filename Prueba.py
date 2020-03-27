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

df['adjusted_price']

for i in range(1, len(archives)):
    df = df.append(pd.read_csv(archives[i]))
   
df.drop(['number_of_reviews_ltm', 'host_acceptance_rate', 'license', 'Unnamed: 0'], axis = 1, inplace = True)
    
df.groupby(['date', 'id'])['adjusted_price'].count()[df.groupby(['date', 'id'])['adjusted_price'].count()>1]

df.drop_duplicates(subset = ['date', 'id'], inplace = True)

df.isnull().sum()[df.isnull().sum()>0]

agrupacion = df.groupby('date')['adjusted_price'].describe()

df[df['adjusted_price']>1350]['id'].value_counts()[df[df['adjusted_price']>400]['id'].value_counts()>10]

df = df[(df['adjusted_price']<1350)&(df['adjusted_price']>10)]

fig, ax = plt.subplots(1, 1, figsize = (16, 10))
plt.plot(df.groupby('date')['adjusted_price'].mean().index, df.groupby('date')['adjusted_price'].mean(), color = "red")
plt.xticks(df.groupby('date')['adjusted_price'].mean().index, rotation = 75)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (16, 10))
sns.distplot(df['adjusted_price'])
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (16, 10))
sns.distplot(np.log(df['adjusted_price']), bins = 10)
plt.tight_layout()

df[['adjusted_price']].describe()


df[df['id'] == 15593934]['name']
