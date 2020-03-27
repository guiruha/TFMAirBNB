#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:20:07 2020

@author: guillem
"""


archives = !ls | grep '.csv'
archives


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(archives[0])

for archive in range(1, len(archives)):
    df = df.append(pd.read_csv(archives[i]))


df.drop(df.columns[df.columns.str.endswith('url')], axis = 1, inplace = True)


nulls = df.isnull().sum() / df.shape[0]
nulls = nulls[nulls>0.05]

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.barplot(x = nulls.index, y = nulls)
plt.xticks(rotation = 90)
ax.hlines(xmin = 0, xmax = len(nulls), y = 0.4, color = "orange", linestyle = '--', label = "40%")
ax.hlines(xmin = 0, xmax = len(nulls), y = 0.5, color = "blue", linestyle = '-.', label = "50%")
ax.hlines(xmin = 0, xmax = len(nulls), y = 0.9, color = "red", linestyle = '-.', label = "90%")
plt.legend()
plt.title("Nulls por variable")
ax.set_xlabel("Variable")
ax.set_ylabel("% de Nulls")


fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.barplot(x = nulls[nulls>.5].index, y = nulls[nulls>.5])
plt.xticks(rotation = 90)
ax.hlines(xmin = 0, xmax = len(nulls[nulls>.5]), y = 0.4, color = "orange", linestyle = '--', label = "40%")
ax.hlines(xmin = 0, xmax = len(nulls[nulls>.5]), y = 0.5, color = "blue", linestyle = '-.', label = "50%")
ax.hlines(xmin = 0, xmax = len(nulls[nulls>.5]), y = 0.9, color = "red", linestyle = '-.', label = "90%")
plt.legend()
plt.title("Nulls por variable")
ax.set_xlabel("Variable")
ax.set_ylabel("% de Nulls")

df.drop(nulls[nulls>0.6].index, axis = 1, inplace = True)

nulls = df.isnull().sum() / df.shape[0]
nulls = nulls[nulls>0.05]

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.barplot(x = nulls.index, y = nulls)
plt.xticks(rotation = 90)
ax.hlines(xmin = 0, xmax = len(nulls), y = 0.4, color = "orange", linestyle = '--', label = "40%")
ax.hlines(xmin = 0, xmax = len(nulls), y = 0.5, color = "blue", linestyle = '-.', label = "50%")
ax.hlines(xmin = 0, xmax = len(nulls), y = 0.9, color = "red", linestyle = '-.', label = "90%")
plt.legend()
plt.title("Nulls por variable")
ax.set_xlabel("Variable")
ax.set_ylabel("% de Nulls")

df['city'].value_counts()

df['state'].value_counts()

df['country'].value_counts()

df.loc[df['state'] == 'Connecticut'][['city', 'neighbourhood']]

df.loc[df['city']=="L'Hospitalet de Llobregat"]['neighbourhood'].value_counts()

df.loc[df['city']=="."]['neighbourhood'].value_counts()

dropC = ['city', 'state', 'zipcode', 'country', 'country_code']
df.drop(dropC, axis = 1, inplace = True)

maxmincols = [x for x in df.columns if (x.startswith('maximum') | x.startswith('minimum'))]
maxmincols

maxmin = df[maxmincols]
maxmin.isnull().sum()/maxmin.shape[0]

for col in [x for x in maxmin.columns if 'minimum' in x]:
    print('La columna {} coincide con la de minimum_nigths un {:.2f}%'
          .format(col, ((maxmin.loc[:]['minimum_nights'] == maxmin.loc[:][col]).sum()/maxmin.shape[0])*100))
print('\n','='*50, '\n')
for col in [x for x in maxmin.columns if 'maximum' in x]:
    print('La columna {} coincide con la de minimum_nigths un {:.2f}%'
          .format(col, ((maxmin.loc[:]['maximum_nights'] == maxmin.loc[:][col]).sum()/maxmin.shape[0])*100))
    
DropC = [x for x in maxmincols if x not in ['minimum_nights', 'maximum_nights']]
print(len(maxmincols), len(DropC))
df.drop(DropC, axis = 1, inplace = True)

neighbourhoods = df[['neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']]
