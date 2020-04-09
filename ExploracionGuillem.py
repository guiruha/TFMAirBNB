#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:19:56 2020

@author: guillem
"""

cd ~/DadesAirBNB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

archives = !ls | grep 'df'

df = pd.read_csv(archives[0])

for i in range(1, len(archives)):
    df = df.append(pd.read_csv(archives[i]))
   
df.drop(['Unnamed: 0', 'requires_license', 'price'], axis = 1, inplace = True)
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month    
df['Day'] = df['date'].dt.day 

df.groupby(['id', 'date','goodprice'])['goodprice'].count()[df.groupby(['id', 'date','goodprice'])['goodprice'].count()>1]

df.drop_duplicates(subset = ['date', 'id', 'goodprice'], inplace = True)


df[df['goodprice'].isnull()]['available'].value_counts()

df.dropna(subset = ['goodprice'], axis = 0, inplace = True)

df.isnull().sum()[df.isnull().sum()>0]

agrupacion = df.groupby('date')['goodprice'].describe()

# La Luxury Villa es el último OUTLIER amb sentit en quant al preu (1200€)
df[df['goodprice']>1200]['id'].value_counts()

df = df[(df['goodprice']<1200)]

fig, ax = plt.subplots(1, 1, figsize = (40, 10))
plt.plot(df.groupby('date')['goodprice'].mean().index, df.groupby('date')['goodprice'].mean(), color = "red")
plt.xticks(df.groupby('date')['goodprice'].mean().index, rotation = 75)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['goodprice'])
plt.tight_layout()

df['goodprice'] = df['goodprice'].apply(lambda x: x + 0.01 if x == 0 else x)

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(np.log(df['goodprice']), bins = 13)
plt.tight_layout()

df[['goodprice']].describe()

df[(df['goodprice']>300)&(df['minimum_nights']>1)][['id', 'goodprice', 'minimum_nights']]

# Calculem el PricePNight
df['PricePNight'] = [price/minnight if (price > 300)&(minnight > 1) else price for price, minnight in zip(df['goodprice'], df['minimum_nights'])]

df[(df['goodprice']>300)&(df['minimum_nights']>1)&(df['PricePNight']<30)][['id', 'goodprice', 'minimum_nights', 'PricePNight']].sort_values(['PricePNight'])

df[(df['goodprice']>300)&(df['minimum_nights']>1)&(df['PricePNight']<30)&(df['minimum_nights']<30)]\
    [['id', 'goodprice', 'minimum_nights', 'PricePNight']].sort_values(['PricePNight'])

# HE DESCOBERT QUE ELS QUE TENEN MÉS DE 28 DÍES DE MINIMUM_NIGHTS FIQUEN EL PREU MENSUAL

df[(df['minimum_nights']>28)&(df['goodprice']>20)&(df['PricePNight']< 20)][['goodprice', 'PricePNight']] # Estes files sobren

df.drop(df[(df['minimum_nights']>28)&(df['goodprice']>20)&(df['PricePNight']< 20)].index, inplace = True)

#Buscador per ficar al google chrome o firefox
def who(x):
    print('\n', df[df['id']==x]['name'].unique(), '\n')
    return 'https://www.airbnb.es/rooms/' + str(x)

who(199651)

df = df[df['PricePNight'] != 0.01]

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['PricePNight'], bins = 13)
plt.tight_layout()

df['PricePNight'].describe()
fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(np.log(df['PricePNight']), bins = 13)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
plt.plot(df.groupby('date')['PricePNight'].mean().index, df.groupby('date')['PricePNight'].mean(), color = "red")
plt.xticks(df.groupby('date')['PricePNight'].mean().index, rotation = 75)
plt.tight_layout()

# FINALMENT CREE EL LOGPRICEPERNIGHT
df['LogPricePNight'] = np.log(df['PricePNight'])
df.drop(['goodprice'], axis = 1, inplace = True)

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
plt.plot(df.groupby('date')['LogPricePNight'].mean().index, df.groupby('date')['LogPricePNight'].mean(), color = "red")
plt.xticks(df.groupby('date')['LogPricePNight'].mean().index, rotation = 75)
plt.tight_layout()

# Construim les categories
factorcolumns = [x for x in list(df.dtypes[df.dtypes == 'int'].index) if 0 in df[x].value_counts().index]

df[factorcolumns] = df[factorcolumns].astype('category')

# DEURÍEM AFEGIR ELS MESOS 1, 2 i 3?                             
sns.pointplot('Month', 'PricePNight', hue = 'Year', data = df)


# Pareix que a partir de 12 llits son outliers respecte al preu amb prou Leverage

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
plt.scatter(df['bedrooms'], df['LogPricePNight'], color = "navy", alpha = 0.05)
sns.regplot('bedrooms', 'LogPricePNight', data = df, scatter = False, color = "red")
plt.tight_layout()


fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.pairplot(
# El mateix pasa a partir de 8 banys

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
plt.scatter(df['bathrooms'], df['LogPricePNight'], color = "navy", alpha = 0.05)
sns.regplot('bathrooms', 'LogPricePNight', data = df, scatter = False, color = "red")


# Crec que hi ha colinealitat entre bathrooms y bedrooms
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
plt.scatter(df['bathrooms'], df['bedrooms'], color = "navy", alpha = 0.05)
sns.regplot('bathrooms', 'bedrooms', data = df, scatter = False, color = "red")
plt.tight_layout()

np.corrcoef(df['bedrooms'], df['bathrooms'])

# ARA FAIG UN POC DE FEATURE ENGINEERING I FEATURE SELECTION
df.dtypes[df.dtypes == 'object']
columnselection = ["host_response_time", "neighbourhood_group_cleansed", "property_type", "room_type", "bed_type", "cancellation_policy"]
X_raw = pd.get_dummies(df, columns = columnselection)

factorcolumns = [x for x in list(X_raw.dtypes[(X_raw.dtypes == 'uint8')].index) if 0 in X_raw[x].value_counts().index]

X_raw[factorcolumns] = X_raw[factorcolumns].astype('category')


# ELIMINEM LA COLUMNA ID QUE NO ES IMPORTANT
X_raw.drop('id', inplace = True, axis = 1)

corrcolumns = list(X_raw.dtypes[(X_raw.dtypes == 'float') | (X_raw.dtypes == 'int')].index)

correlations = X_raw[corrcolumns].corr()

colineales = []
for column in correlations.columns:
    for row in correlations.index:
        if (np.abs(correlations.loc[row, column]) > 0.7) & (correlations.loc[row, column] !=  1):
            print(row, 'y', column, 'tienen una correlación de', correlations.loc[row, column])
            colineales.append(row)
selection = ['bedrooms', 'beds', 'accommodates', 'review_scores_accuracy', 'review_scores_value', 'review_scores_rating']
for column in selection:
    print(column, 'LogPricePNight', X_raw['LogPricePNight'].corr(X_raw[column]))

np.corrcoef(X_raw['review_scores_value'], X_raw['review_scores_accuracy']) 
                  
X_raw.drop(['review_scores_rating', 'beds', 'bedrooms'], axis = 1, inplace = True)

corrcolumns = list(X_raw.dtypes[(X_raw.dtypes == 'float') | (X_raw.dtypes == 'int')].index)
fig, ax = plt.subplots(1, 1, figsize = (50, 35))
sns.heatmap(X_raw[corrcolumns[1:]].corr(), vmin = -1, vmax = 1, center = 0, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()

# EL DE BAIX ES PER FER UNA MATRIU TRIANGULAR INFERIOR PER ESTALVIAR LA PART SUPERIOR QUE ES INNECESARIA
mask = np.zeros_like(X_raw[corrcolumns[1:]].corr())
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(1, 1, figsize = (35, 30))
sns.heatmap(round(X_raw[corrcolumns[1:]].corr(), 3), vmin = -1, vmax = 1, center = 0, mask = mask, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()


# ARA CORRELACIONS ENTRE VARIABLES CATEGORIQUES
catpar = [(i,j) for i in X_raw.dtypes[X_raw.dtypes == "category"].index.values for j in X_raw.dtypes[X_raw.dtypes == "category"].index.values]

phi, p_values = [], []

import scipy

for c in catpar:
    if c[0] != c[1]:
        chistest = scipy.stats.chi2_contingency(pd.crosstab(X_raw[c[0]], X_raw[c[1]]))
        n = pd.crosstab(X_raw[c[0]], X_raw[c[1]]).sum().sum()
        phival = np.sqrt(chistest[0]/n)
        phi.append(phival)
    else:
        phi.append(1)

len(X_raw.dtypes[X_raw.dtypes == "category"].index.values)

phi2 = np.array(phi).reshape(58, 58)
phi2 = pd.DataFrame(phi2, index = X_raw.dtypes[X_raw.dtypes == "category"].index.values, columns = X_raw.dtypes[X_raw.dtypes == "category"].index.values)

fig, ax = plt.subplots(1, 1, figsize = (35, 30))
sns.heatmap(round(phi2, 3), vmin = -1, vmax = 1, center = 0, cmap = "RdBu", ax = ax, annot = True,  annot_kws = {"size": 8})
plt.show()

import scipy
for column in X_raw.dtypes[X_raw.dtypes == "category"].index.values:
    print(column, 'LogPricePNight', scipy.stats.pointbiserialr(X_raw[column], X_raw['LogPricePNight'])[0])

# REGRESIÓ LINEAL
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['accommodates', 'instant_bookable', 'review_scores_rating', 'host_total_listings_count', 'Smoking allowed', 'bathrooms', 'beds', 'cleaning_fee', 'guests_included', 'availability_365', 'Air conditioning', 'Family/kid friendly']].values

y = df['LogPricePNight'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1997)

lineareg = LinearRegression()

lineareg.fit(X_train, y_train)

lineareg.score(X_test, y_test)


# HO FAIG AMB TOTES LES VARIABLES
X_copy = X_raw.copy()

selection = list(X_raw.dtypes[(X_raw.dtypes != 'O') & (X_raw.columns != 'date')].index)
selectnum = list(X_raw.dtypes[(X_raw.dtypes != 'O') & (X_raw.columns != 'date') & (X_raw.dtypes != 'category')].index)

atrinum = pd.Series(selectnum)[pd.Series(selectnum).str.contains('rice') == False][1:-3].tolist()
atrinum
atricompl = pd.Series(selection)[pd.Series(selection).str.contains('rice') == False].tolist()

X = X_copy[atricompl]
y = df['LogPricePNight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1997)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train[atrinum] = ss.fit_transform(X_train[atrinum])
X_train = X_train.values
lineareg = LinearRegression()

lineareg.fit(X_train, y_train)

X_test[atrinum] = ss.transform(X_test[atrinum])

X_test = X_test.values

lineareg.score(X_test, y_test)

# PROBEM AMB UNA RIDGE
from sklearn.linear_model import Ridge

ridreg = Ridge(random_state = 1997)
ridreg.fit(X_train, y_train)

ridreg.score(X_test, y_test)

ridgeatributos = []
for n,v in zip(atricompl, ridreg.coef_):
    if np.abs(v) > 0:
        ridgeatributos.append(n)

serieatributos = pd.DataFrame(zip(ridgeatributos, ridreg.coef_, np.abs(ridreg.coef_)), columns = ['Atributos', 'Coeficiente', 'ValorAbsoluto']).set_index('Atributos').sort_values('ValorAbsoluto', ascending = False)
serieatributos[:30]

# VAIG A PROBRAR UNA ELASTIC NET (UNA MIERDA)

from sklearn.linear_model import ElasticNet

LinEN = ElasticNet()
LinEN.fit(X_train, y_train)

atributos = []
for atr, coef in zip(atricompl, LinEN.coef_):
    if np.abs(v) > 0:
        atributos.append(atr)

serieatributos = pd.DataFrame(zip(atributos, LinEN.user = coef_, np.abs(LinEN.coef_)), columns = ['Atributos', 'Coeficiente', 'ValorAbsoluto']).set_index('Atributos').sort_values('ValorAbsoluto', ascending = False)
serieatributos[:30]

# FINALMENT PROBE AMB UNA LASSO

from sklearn.linear_model import LassoCV

lcv = LassoCV()

lcv.fit(X_train, y_train)

lcv.score(X_test,y_test)

lastatributes = X_copy[atricompl].columns[np.abs(lcv.coef_) > 0].tolist()

X = X_copy[lastatributes]
y = df['LogPricePNight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1997)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_train = X_train
lineareg = LinearRegression()

lineareg.fit(X_train, y_train)
X_test = ss.fit_transform(X_test)

lineareg.score(X_test, y_test)

