#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:35:15 2020

@author: guillem
"""

cd ~/DadesAirBNB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

archives = !ls | grep 'df'

df = pd.read_csv(archives[0])

for i in range(1, len(archives)):
    df = df.append(pd.read_csv(archives[i]))
df.shape
df.drop(['Unnamed: 0', 'requires_license', 'availability_365', 'price'], axis = 1, inplace = True)
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month    
df['Day'] = df['date'].dt.day 
df['DayOfWeek'] = df['date'].dt.weekday
df['date'] = df['date'].dt.date
df = df.sort_values('date')

pricenulls = (df[df['goodprice'].isnull()]['date'].value_counts()/df['date'].value_counts()).sort_values(ascending = False)
df[df['goodprice'].isnull()]['available'].value_counts()

df.dropna(subset = ['goodprice'], axis = 0, inplace = True)

df.groupby(['id', 'date','goodprice'])['goodprice'].count()[df.groupby(['id', 'date','goodprice'])['goodprice'].count()>1]
df.groupby(['id', 'DayOfWeek', 'goodprice'])['goodprice'].count()[df.groupby(['id','DayOfWeek', 'goodprice'])['goodprice'].count()>1]
df.groupby(['id', 'DayOfWeek', 'date','goodprice'])['goodprice'].count()[df.groupby(['id', 'DayOfWeek', 'date','goodprice'])['goodprice'].count()>1]

df.drop_duplicates(subset = ['date', 'id'], inplace = True)

df.groupby(['Year', 'Month'])['id'].count()
df.isnull().sum()[df.isnull().sum()>0]/df.shape[0]

agrupacion = df.groupby('date')['goodprice'].describe()

df = df[df['Year']>2017]

df['PricePNight'] = [price/minnight if (price > 300)&(minnight > 1) else price for price, minnight in zip(df['goodprice'], df['minimum_nights'])]

df[(df['goodprice']>300)&(df['minimum_nights']>1)][['goodprice', 'minimum_nights', 'PricePNight']]

df[(df['goodprice']>1200)&(df['minimum_nights']>1)]['id'].value_counts()

def who(x):
    print('\n', df[df['id']==x]['name'].unique(), '\n')
    return 'https://www.airbnb.es/rooms/' + str(x)

who(2261714)

df = df[(df['goodprice']<1100)]

# EXPLORACIÓN DE LA VARIABLE DEPENDIENTE

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['PricePNight'], bins = 13)
plt.tight_layout()

df['PricePNight'].describe()

df = df[(df['PricePNight'] >= 8)]

df['LogPricePNight'] = np.log(df['PricePNight'])
df.drop(['goodprice'], axis = 1, inplace = True)

df['PricePNight'].describe()

normal = np.random.normal(loc = df['LogPricePNight'].mean(), scale = df['LogPricePNight'].std(), size = df['LogPricePNight'].shape[0])
normalstand = np.random.normal(loc = df['NormPricePNight'].mean(), scale = df['NormPricePNight'].std(), size = df['NormPricePNight'].shape[0])

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['LogPricePNight'], bins = 13,)
sns.distplot(normal, bins = 13)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df.sort_values(['id', 'date'])['LogPricePNight'], bins = 13,)
sns.distplot(normal, bins = 13)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df.sort_values(['id', 'date'])['NormPricePNight'], bins = 13,)
sns.distplot(normalstand, bins = 13)
plt.tight_layout()

import statsmodels.stats as st
import scipy.stats as sp

esta, pv = st._lilliefors(df['NormPricePNight'], dist = 'norm',   pvalmethod = 'table')

print("Estadisctico = {}, pvalue = {}".format(esta, pv))
if pv > 0.05:
    print("Es probablemente una muestra procedente de una Normal")
else:
    print("No parece que proceda de una Normal")
    
esta, pv = sp.anderson(df['NormPricePNight'])

print("Estadisctico = {}, pvalue = {}".format(esta, pv))
if pv > 0.05:
    print("Es probablemente una muestra procedente de una Normal")
else:
    print("No parece que proceda de una Normal")

esta, pv = sp.normaltest(df['NormPricePNight'])

print("Estadisctico = {}, pvalue = {}".format(esta, pv))
if pv > 0.05:
    print("Es probablemente una muestra procedente de una Normal")
else:
    print("No parece que proceda de una Normal")

# NO SE APROXIMA A UNA NORMAL SEGONS ESTOS TEST

fig, ax = plt.subplots(1, 1, figsize = (40, 15))
sns.pointplot(df['date'], df['PricePNight'])
plt.xticks(rotation = 90)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (40, 15))
sns.pointplot(df['date'], df['LogPricePNight'])
plt.xticks(rotation = 90)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (40, 15))
sns.pointplot(df['date'], df['LogPricePNight'].rolling(50).mean())
plt.xticks(rotation = 90)
plt.tight_layout()
# ANALITZEM HOST RESPONSE TIME
fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_response_time'].value_counts().index, df['host_response_time'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_response_time'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_response_time'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# ===

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_is_superhost'].value_counts().index, df['host_is_superhost'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_is_superhost'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_is_superhost'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# ===

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.distplot(df['host_total_listings_count'], ax = ax[0])
sns.scatterplot(df['host_total_listings_count'], df['PricePNight'], ax = ax[1])
sns.regplot(df['host_total_listings_count'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# ===

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_has_profile_pic'].value_counts().index, df['host_has_profile_pic'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_has_profile_pic'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_has_profile_pic'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# !!!

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_identity_verified'].value_counts().index, df['host_identity_verified'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_identity_verified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_identity_verified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# !!!

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['neighbourhood_group_cleansed'].value_counts().index, df['neighbourhood_group_cleansed'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['neighbourhood_group_cleansed'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['neighbourhood_group_cleansed'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# ===

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['is_location_exact'].value_counts().index, df['is_location_exact'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['is_location_exact'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['is_location_exact'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# !!!

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['property_type'].value_counts().index, df['property_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['property_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['property_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# ???

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['room_type'].value_counts().index, df['room_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['room_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['room_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# Només Private Room i Entire Home com a dummys?

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['accommodates'].value_counts().index, df['accommodates'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['accommodates'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['accommodates'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

