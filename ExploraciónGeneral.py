#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:35:15 2020

@author: Guillem Rochina y Helena Saigi
"""
cd ~/DadesAirBNB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
plt.style.use('fivethirtyeight')


<<<<<<< HEAD
df = pd.read_csv('~/DadesAirBNB/DatosLimpios.csv') 
   
=======
df = pd.read_pickle('/home/guillem/DadesAirBNB/DatosLimpios.pkl') 
>>>>>>> 7ef347ea0576856c2d39718b8a1ebd11e4dbbd51
df.shape
df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month    
df['Day'] = df['date'].dt.day 
df['DayOfWeek'] = df['date'].dt.weekday
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

df = df[(df['Year']>2016)&(df['Year']<2020)]

df['PricePNight'] = [price/minnight if (price > 300)&(minnight > 1) else price for price, minnight in zip(df['goodprice'], df['minimum_nights'])]

df[(df['goodprice']>300)&(df['minimum_nights']>1)][['goodprice', 'minimum_nights', 'PricePNight']]
df[(df['goodprice']>= 1200)&(df['minimum_nights']>1)][['goodprice', 'minimum_nights', 'PricePNight']]

df[(df['goodprice']>= 1200)&(df['minimum_nights']>1)]['id'].value_counts()
df[(df['goodprice']>= 1200)&(df['minimum_nights']>1)][['id', 'PricePNight']]

def who(x):
    print('\n', df[df['id']==x]['name'].unique(), '\n')
    return 'https://www.airbnb.es/rooms/' + str(x)

who(18752257)

df = df[(df['goodprice']<9000)]
df = df[(df['PricePNight'] >= 8)]

#(df[df['goodprice']<= 600].shape[0] - df.shape[0])/df.shape[0]

#df = df[df['goodprice']<= 600]

# EXPLORACIÓN DE LA VARIABLE DEPENDIENTE

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['PricePNight'], bins = 13)
plt.tight_layout()

df['PricePNight'].describe()

df['LogPricePNight'] = np.log(df['PricePNight'])
df.drop(['goodprice'], axis = 1, inplace = True)

df['PricePNight'].describe()

normal = np.random.normal(loc = df['LogPricePNight'].mean(), scale = df['LogPricePNight'].std(), size = df['LogPricePNight'].shape[0])

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['LogPricePNight'], bins = 15)
sns.distplot(normal, bins = 15)
plt.tight_layout()

df['LogPricePNight'].describe()

import scipy.stats as sp

esta, pv = sp.normaltest(df['LogPricePNight'])

print("Estadisctico = {}, pvalue = {}".format(esta, pv))
if pv > 0.05:
    print("Es probablemente una muestra procedente de una Normal")
else:
    print("No parece que proceda de una Normal")

# NO SE APROXIMA A UNA NORMAL SEGONS ESTOS TEST

df.set_index('date', inplace = True)

fig, ax = plt.subplots(1, 1, figsize = (40, 15))
sns.pointplot(df.index.date, df['PricePNight'], ax = ax)
plt.xticks(rotation = 90)
plt.tight_layout()

# Detectamos outliers
df.groupby(df.index)['PricePNight'].describe()[df.groupby(df.index)['PricePNight'].mean() == df.groupby(df.index)['PricePNight'].mean().max()]
df['PricePNight'].idxmax()
df.loc['2017-10-28'][['id', 'PricePNight']].sort_values(by = 'PricePNight', ascending = False)[:10]

df = df[(df['goodprice']<1000)]

fig, ax = plt.subplots(1, 1, figsize = (20, 13))
plt.plot(df.resample('M')['PricePNight'].mean().index, df.resample('M')['PricePNight'].mean())
plt.xticks(rotation = 45)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (50, 13))
plt.plot(df.resample('W')['PricePNight'].mean().index, df.resample('W')['PricePNight'].mean())
<<<<<<< HEAD
plt.xticks(rotation = 45)
=======
plt.xticks(df.resample('W')['PricePNight'].mean().index, rotation = 45)
>>>>>>> 7ef347ea0576856c2d39718b8a1ebd11e4dbbd51
plt.tight_layout()

# Detectamos más outliers
df.loc['2019-02'][['id', 'PricePNight', 'minimum_nights', 'goodprice']]\
    .sort_values(by = 'PricePNight', ascending = False)[:20]

df = df[(df['goodprice']<500)]

fig, ax = plt.subplots(1, 1, figsize = (40, 15))
sns.pointplot(df.index.date, df['LogPricePNight'])
plt.xticks(rotation = 90)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (20, 13))
plt.plot(df.resample('M')['LogPricePNight'].mean().index, df.resample('M')['LogPricePNight'].mean())
plt.xticks(df.resample('M')['LogPricePNight'].mean().index, rotation = 45)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (40, 13))
plt.plot(df.resample('W')['LogPricePNight'].mean().index, df.resample('W')['LogPricePNight'].mean())
plt.xticks(df.resample('W')['LogPricePNight'].mean().index, rotation = 75)
plt.tight_layout()

# RESPONSE TIME !!! (DUMMY) !!!

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_response_time'].value_counts().index, df['host_response_time'].value_counts(normalize = True), ax = ax[0])
sns.pointplot(df['host_response_time'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_response_time'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

dummys = ['host_response_time']
# HOST IS SUPERHOST

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_is_superhost'].value_counts().index, df['host_is_superhost'].value_counts(normalize = True), ax = ax[0])
sns.pointplot(df['host_is_superhost'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_is_superhost'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('host_is_superhost', axis = 1, inplace = True)

# HOST TOTAL LISTINGS COUNT (FER REGRESIÓ)
df['host_total_listings_count'].value_counts(normalize = True).sort_index()
df['host_total_listings_count'].value_counts().sort_index()

df = df[df['host_total_listings_count'] != 0]

fig, ax = plt.subplots(1, 2, figsize = (20, 15))
sns.distplot(df['host_total_listings_count'], ax = ax[0])
sns.distplot(np.log(df['host_total_listings_count']), ax = ax[1], bins = 10)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(df[['host_total_listings_count']], df['LogPricePNight']).predict(df[['host_total_listings_count']])

fig, ax = plt.subplots(1, 1, figsize = (20, 15))
sns.scatterplot(np.log(df['host_total_listings_count']), df['LogPricePNight'],alpha = 0.005, size = 1, x_jitter = 20, color = "navy", marker = 'o')
plt.plot(np.log(df['host_total_listings_count']), 
         lr.fit(np.log(df[['host_total_listings_count']]), df['LogPricePNight']).predict(np.log(df[['host_total_listings_count']])),
         color = 'maroon')
plt.show()

np.corrcoef(np.log(df['host_total_listings_count']), df['LogPricePNight'])

np.corrcoef(df['host_total_listings_count'], df['LogPricePNight'])

df['Loghost_total_listings_count'] = np.log(df['host_total_listings_count'])

# PROFILE PIC

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_has_profile_pic'].value_counts().index, df['host_has_profile_pic'].value_counts(normalize = True), ax = ax[0])
sns.pointplot(df['host_has_profile_pic'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_has_profile_pic'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('host_has_profile_pic', axis = 1, inplace = True)

# IDENTITY VERIFIED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_identity_verified'].value_counts().index, df['host_identity_verified'].value_counts(normalize = True), ax = ax[0])
sns.pointplot(df['host_identity_verified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_identity_verified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# NEIGHBOURHOOD GROUP CLEANSED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['neighbourhood_group_cleansed'].value_counts().index, df['neighbourhood_group_cleansed'].value_counts(normalize = True), ax = ax[0])
sns.pointplot(df['neighbourhood_group_cleansed'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['neighbourhood_group_cleansed'], df['LogPricePNight'], ax = ax[2])
plt.show()

dummys.append('neighbourhood_group_cleansed')

# IS LOCATION EXACT

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['is_location_exact'].value_counts().index, df['is_location_exact'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['is_location_exact'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['is_location_exact'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# PROPERTY TYPE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['property_type'].value_counts().index, df['property_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['property_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['property_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['property_type'].value_counts(normalize = True)
df['property_type'].value_counts()

# MAL (CAMBIAR)
df['property_type'] = df['property_type'].apply(lambda x: 'other' if x != 'apartment' else 'apartment')

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['property_type'].value_counts().index, df['property_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['property_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['property_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['property_type'].value_counts(normalize = True)

temp = df[df['property_type'] == 'other']

# ROOM TYPE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['room_type'].value_counts().index, df['room_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['room_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['room_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['room_type'].value_counts(normalize = True)

df['room_type'] = df['room_type'].apply(lambda x: 'Entire_home' if x == 'Entire home/apt' else 'Single_room')

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['room_type'].value_counts().index, df['room_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['room_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['room_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.groupby('room_type')['PricePNight'].describe()

# NOMÉS ENTIRE HOME y PRIVATE ROOM COM A DUMMYS!!!!!!!!!!!!!!!!!!!
dummys.append('room_type')

# ACCOMMODATES

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['accommodates'].value_counts().index, df['accommodates'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['accommodates'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['accommodates'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(temp['accommodates'].value_counts().index, temp['accommodates'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(temp['accommodates'], temp['PricePNight'], ax = ax[1])
sns.boxplot(temp['accommodates'], temp['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

(df[df['accommodates']<11].shape[0] - df.shape[0])/df.shape[0]

df = df[df['accommodates']<11]

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['accommodates'].value_counts().index, df['accommodates'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['accommodates'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['accommodates'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# BATHROOMS

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['bathrooms'].apply(lambda x: np.floor(x)).value_counts().index, df['bathrooms'].apply(lambda x: np.floor(x)).value_counts(), ax = ax[0])
sns.pointplot(df['bathrooms'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['bathrooms'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['bathrooms'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

(df[df['bathrooms']<= 3].shape[0] - df.shape[0])/df.shape[0]

#df = df[df['bathrooms']<= 3]

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['bathrooms'].apply(lambda x: np.floor(x)).value_counts().index, df['bathrooms'].apply(lambda x: np.floor(x)).value_counts(), ax = ax[0])
sns.pointplot(df['bathrooms'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['bathrooms'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# AIR CONDITIONING

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Air conditioning'].value_counts().index, df['Air conditioning'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Air conditioning'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Air conditioning'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# SMOKING ALLOWED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Smoking allowed'].value_counts().index, df['Smoking allowed'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Smoking allowed'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Smoking allowed'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# FAMILY/KID FRIENDLY

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Family/kid friendly'].value_counts().index, df['Family/kid friendly'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Family/kid friendly'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Family/kid friendly'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# HOST GREETS YOU

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Host greets you'].value_counts().index, df['Host greets you'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Host greets you'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Host greets you'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# LAPTOP FRIENDLY WORKSPACE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Laptop friendly workspace'].value_counts().index, df['Laptop friendly workspace'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Laptop friendly workspace'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Laptop friendly workspace'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('Laptop friendly workspace', axis = 1, inplace = True)

# NUMBER OF REVIEWS

temp = df.drop_duplicates(subset = ['id'])

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['number_of_reviews'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

temp['number_of_reviews'].value_counts().sort_index()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.distplot(np.log(ss.fit_transform(temp[['number_of_reviews']])), bins = 10)
plt.xticks(rotation = 45)
plt.show()

plt.scatter(temp['number_of_reviews'], temp['PricePNight'])

plt.scatter(np.log(ss.fit_transform(temp[['number_of_reviews']])), temp['PricePNight'])

plt.scatter(np.log(ss.fit_transform(temp[['number_of_reviews']])), temp['LogPricePNight'])

np.corrcoef(df['number_of_reviews'], df['LogPricePNight'])

# REVIEW SCORE RATING

sns.distplot(temp['review_scores_rating'])
sns.distplot(np.log(temp['review_scores_rating']))

sns.scatterplot(np.log(ss.fit_transform(temp[['review_scores_rating']])).squeeze(), temp['LogPricePNight'], x_jitter = True, y_jitter = True)

np.corrcoef(df['review_scores_rating'], df['LogPricePNight'])

# REVIEW SCORE CLEANLINESS

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_cleanliness'].value_counts().index, df['review_scores_cleanliness'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_cleanliness'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_cleanliness'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['review_scores_cleanliness'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

(df[df['review_scores_cleanliness']>5].shape[0] - df.shape[0])/df.shape[0]

df = df[df['review_scores_cleanliness']>5]

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_cleanliness'].value_counts().index, df['review_scores_cleanliness'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_cleanliness'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_cleanliness'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['review_scores_cleanliness'], df['LogPricePNight'])
# REVIEW SCORE CHEKIN

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_checkin'].value_counts().index, df['review_scores_checkin'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_checkin'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_checkin'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['review_scores_checkin'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

df['review_scores_checkin'].value_counts(normalize = True)

np.corrcoef(df['review_scores_checkin'], df['LogPricePNight'])

df.drop('review_scores_checkin', axis = 1, inplace = True)

# REVIEW SCORE ACCURACY

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_accuracy'].value_counts().index, df['review_scores_accuracy'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_accuracy'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_accuracy'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['review_scores_accuracy'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

df['review_scores_accuracy'].value_counts(normalize = True)

np.corrcoef(df['review_scores_accuracy'], df['LogPricePNight'])

df.drop('review_scores_accuracy', axis = 1, inplace = True)

# REVIEW SCORE COMMUNICATION

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_communication'].value_counts().index, df['review_scores_communication'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_communication'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_communication'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['review_scores_communication'], df['LogPricePNight'])

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['review_scores_communication'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

df.drop('review_scores_communication', axis = 1, inplace = True)

# REVIEW SCORE LOCATION


fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_location'].value_counts().index, df['review_scores_location'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_location'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_location'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['review_scores_location'], df['LogPricePNight'])

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['review_scores_location'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

df.drop('review_scores_location', axis = 1, inplace = True)

# REVIEW SCORE VALUE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_value'].value_counts().index, df['review_scores_value'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_value'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_value'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['review_scores_value'], df['LogPricePNight'])

df.drop('review_scores_value', axis = 1, inplace = True)

# BEDROOMS

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
ax[0].hist(df['bedrooms'])
sns.pointplot(df['bedrooms'], df['PricePNight'], ax = ax[1])
sns.violinplot('bedrooms', 'LogPricePNight',  data = df, ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df = df[df['bedrooms']<8]

# BEDS

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
ax[0].hist(df['beds'])
sns.pointplot(df['beds'], df['PricePNight'], ax = ax[1])
sns.violinplot('beds', 'LogPricePNight',  data = df, ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(df['beds'].value_counts(normalize = True).sort_index().cumsum())
plt.show()

df['beds'].value_counts(normalize = True).sort_index()*100

df = df[df['beds']<8]

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
ax[0].hist(df['beds'])
sns.pointplot(df['beds'], df['PricePNight'], ax = ax[1])
sns.violinplot('beds', 'LogPricePNight',  data = df, ax = ax[2])
plt.xticks(rotation = 45)
plt.show()


# BED TYPE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['bed_type'].value_counts().index, df['bed_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['bed_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['bed_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('bed_type', axis = 1, inplace = True)

# INSTAN BOOKABLE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['instant_bookable'].value_counts().index, df['instant_bookable'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['instant_bookable'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['instant_bookable'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# IS BUSINESS TRAVEL READY

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['is_business_travel_ready'].value_counts().index, df['is_business_travel_ready'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['is_business_travel_ready'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['is_business_travel_ready'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['is_business_travel_ready'].value_counts(normalize = True)

df.drop('is_business_travel_ready', axis = 1, inplace = True)

# CANCELLATION POLICY

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['cancellation_policy'].value_counts().index, df['cancellation_policy'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['cancellation_policy'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['cancellation_policy'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# REQUIRE GUEST PROFILE PICTURE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['require_guest_profile_picture'].value_counts().index, df['require_guest_profile_picture'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['require_guest_profile_picture'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['require_guest_profile_picture'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('require_guest_profile_picture', axis = 1, inplace = True)

# REQUIRE GUEST PHONE VERIFICATION

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['require_guest_phone_verification'].value_counts().index, df['require_guest_phone_verification'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['require_guest_phone_verification'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['require_guest_phone_verification'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['require_guest_phone_verification'].value_counts(normalize = True)

df.drop('require_guest_phone_verification', axis = 1, inplace = True)

# REVIEW PER MONTH

plt.hist(df['reviews_per_month'])

df['reviews_per_month'].value_counts().sort_index()

df[df['reviews_per_month'] == 56.16][['id', 'date', 'host_since']]
df[df['reviews_per_month'] == 33.26][['id', 'date', 'host_since']]
df[df['reviews_per_month'] == 24.24][['id', 'date', 'host_since']]


df[df['reviews_per_month']>15]['id'].value_counts()

sns.pointplot(df['reviews_per_month'], df['PricePNight'])

df.groupby('id')[['reviews_per_month', 'LogPricePNight']].mean().plot(kind = 'scatter', x = 'reviews_per_month', y = 'LogPricePNight')

df = df[df['reviews_per_month']<12]

plt.hist(df['reviews_per_month'])

plt.hist(np.log(df['reviews_per_month']))
sns.scatterplot(np.log(df['reviews_per_month']), df['PricePNight'], x_jitter = True, y_jitter = True, alpha = 0.3, size = 8)

np.corrcoef(df['reviews_per_month'], df['LogPricePNight'])
np.corrcoef(np.log(df['reviews_per_month']), df['LogPricePNight'])

# PER A ELIMINAR SI NO HI HA COLINEALITAT

# HOST EMAIL VERIFIED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_emailverified'].value_counts().index, df['host_emailverified'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_emailverified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_emailverified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['host_emailverified'].value_counts(normalize = True)

np.corrcoef(df['host_emailverified'], df['LogPricePNight'])
scipy.stats.pointbiserialr(df['host_emailverified'], df['LogPricePNight'])[0]

df.drop('host_emailverified', axis = 1, inplace = True)

# HOST PHONE VERIFIED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_phoneverified'].value_counts().index, df['host_phoneverified'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_phoneverified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_phoneverified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df['host_phoneverified'].value_counts(normalize = True)

df.drop('host_phoneverified', axis = 1, inplace = True)

# HOST HAS JUMIO

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_hasjumio'].value_counts().index, df['host_hasjumio'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_hasjumio'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_hasjumio'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['host_hasjumio'], df['LogPricePNight'])

# HOST REVIEW VERIFIED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_reviewverified'].value_counts().index, df['host_reviewverified'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_reviewverified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_reviewverified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['host_reviewverified'], df['LogPricePNight'])

# HOST SELFIE VERIFIED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_selfieverified'].value_counts().index, df['host_selfieverified'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_selfieverified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_selfieverified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['host_selfieverified'], df['LogPricePNight'])
# PAID PARKING OFF PREMISES

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Paid parking off premises'].value_counts().index, df['Paid parking off premises'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Paid parking off premises'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Paid parking off premises'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['Paid parking off premises'], df['LogPricePNight'])

# PATIO OR BALCONY

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Patio or balcony'].value_counts().index, df['Patio or balcony'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Patio or balcony'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Patio or balcony'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['Patio or balcony'], df['LogPricePNight'])

# LUGGAGE DROPOFF ALLOWED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Luggage dropoff allowed'].value_counts().index, df['Luggage dropoff allowed'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Luggage dropoff allowed'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Luggage dropoff allowed'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['Luggage dropoff allowed'], df['LogPricePNight'])

# LONG TERM STAYS ALLOWED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Long term stays allowed'].value_counts().index, df['Long term stays allowed'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Long term stays allowed'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Long term stays allowed'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['Long term stays allowed'], df['LogPricePNight'])

# STEP - FREE ACCES

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Step-free access'].value_counts().index, df['Step-free access'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Step-free access'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Step-free access'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['Step-free access'], df['LogPricePNight'])
# ELIMINAR SI NO HI HA COLINEALITAT

# PAID PARKING ON PREMISES

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['Paid parking on premises'].value_counts().index, df['Paid parking on premises'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['Paid parking on premises'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['Paid parking on premises'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['Paid parking on premises'], df['LogPricePNight'])

# FINALIZAMOS POR AHORA DATOS GENERAL

df.reset_index(inplace = True)
df.to_pickle('~/DadesAirBNB/DatosGeneral.pkl')
df.to_csv('~/DadesAirBNB/DatosGeneral.csv', index = False)

map_df = df.reset_index().drop_duplicates(subset = ['id'])[['id', 'neighbourhood_group_cleansed', 'latitude', 'longitude']]

map_df.to_pickle('~/DadesAirBNB/Localizaciones.pkl')
map_df.to_csv('~/DadesAirBNB/Localizaciones.csv', index = False)


# AÑADIMOS LAS DISTANCIAS AL DATAFRAME ORIGINAL

df = pd.read_pickle('~/DadesAirBNB/DatosGeneral.pkl')
distancias = pd.read_pickle('~/DadesAirBNB/Distancias.pkl')

df.id.unique().shape[0] == distancias.id.count()

df = pd.merge(df, distancias, on = 'id')

df.isnull().sum()[df.isnull().sum()>0]

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(df['dist_metro']), df['LogPricePNight'], alpha = 0.005, x_jitter = True, y_jitter = True)
plt.show()

# PARA MEJORAR EL PROCESAMIENTO ELIMINAMOS DUPLICADOS

temp = df.drop_duplicates(subset = ['id', 'LogPricePNight'])

# DISTANCIA CON EL METRO

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_metro']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(temp['dist_metro'], temp['LogPricePNight'])
np.corrcoef(np.log(temp['dist_metro']), temp['LogPricePNight'])

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(np.log(temp['dist_metro']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_metro']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_metro']), B0 + B1*np.log(temp['dist_metro']), color = "maroon")
plt.show()

np.corrcoef(np.log(df['dist_metro']), df['LogPricePNight']) # Con el dataset total el corrcoef baja un poco

# DISTANCIA CON FERROCARIL CERCANIAS

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_fgc']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['dist_fgc']), temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['dist_fgc']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_fgc']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_fgc']), B0 + B1*np.log(temp['dist_fgc']), color = "maroon")
plt.show()

np.corrcoef(df['dist_fgc'], df['LogPricePNight'])
np.corrcoef(np.log(df['dist_fgc']), df['LogPricePNight'])

# DISTANCIA CON RENFE
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_renfe']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['dist_renfe']), temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['dist_renfe']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_renfe']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_renfe']), B0 + B1*np.log(temp['dist_renfe']), color = "maroon")
plt.show()

np.corrcoef(np.log(df['dist_renfe']), df['LogPricePNight'])

# DISTANCIA TREN A EL AEROPUERTO

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_trenaeropuerto']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['dist_trenaeropuerto']), temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['dist_trenaeropuerto']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_trenaeropuerto']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_trenaeropuerto']), B0 + B1*np.log(temp['dist_trenaeropuerto']), color = "maroon")
plt.show()

np.corrcoef(df['dist_trenaeropuerto'], df['LogPricePNight'])
np.corrcoef(np.log(df['dist_trenaeropuerto']), df['LogPricePNight'])

#  DISTANCIA TRANVIA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_tramvia']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(temp['dist_tramvia'], temp['LogPricePNight'])
np.corrcoef(np.log(temp['dist_tramvia']), temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['dist_tramvia']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_tramvia']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_tramvia']), B0 + B1*np.log(temp['dist_tramvia']), color = "maroon")
plt.show()

np.corrcoef(np.log(df['dist_tramvia']), df['LogPricePNight'])
np.corrcoef(df['dist_tramvia'], df['LogPricePNight'])

# DISTANCIA BUS DIURNO

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_bus']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['dist_bus']), temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['dist_bus']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_tramvia']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_tramvia']), B0 + B1*np.log(temp['dist_tramvia']), color = "maroon")
plt.show()

np.corrcoef(np.log(df['dist_bus']), df['LogPricePNight'])
np.corrcoef(df['dist_bus'], df['LogPricePNight'])

# DISTANCIA BUS AL AEROPUERTO

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_aerobus']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['dist_aerobus']), temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['dist_aerobus']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['dist_tramvia']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['dist_aerobus']), B0 + B1*np.log(temp['dist_aerobus']), color = "maroon")
plt.show()

np.corrcoef(np.log(df['dist_aerobus']), df['LogPricePNight'])
np.corrcoef(df['dist_aerobus'], df['LogPricePNight'])

# DISTANCIA A LA CATEDRA DE BARCELONA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(temp['Catedral de Barcelona_distance'], temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(temp['Catedral de Barcelona_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(temp['Catedral de Barcelona_distance'].values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(temp['Catedral de Barcelona_distance'], temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(temp['Catedral de Barcelona_distance'], B0 + B1*temp['Catedral de Barcelona_distance'], color = "maroon")
plt.show()

np.corrcoef(df['Catedral de Barcelona_distance'], df['LogPricePNight'])

# DISTANCIA A LA SAGRADA FAMILIA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Sagrada Familia_distance']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Sagrada Familia_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Sagrada Familia_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(temp['Sagrada Familia_distance'].values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_

B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(temp['Sagrada Familia_distance'], temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(temp['Sagrada Familia_distance'], B0 + B1*temp['Sagrada Familia_distance'], color = "maroon")
plt.show()

np.corrcoef(df['Sagrada Familia_distance'], df['LogPricePNight'])

# DISTANCIA A MONTJUIC

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Montjuic_distance']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Montjuic_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Montjuic_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(temp['Montjuic_distance'].values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_

B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(temp['Sagrada Familia_distance'], temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(temp['Sagrada Familia_distance'], B0 + B1*temp['Sagrada Familia_distance'], color = "maroon")
plt.show()

np.corrcoef(df['Montjuic_distance'], df['LogPricePNight'])

# DISTANCIA PARC GUELL

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Parc Guell_distance']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Parc Guell_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Parc Guell_distance'], temp['LoggPricePNight'])

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(np.log(temp['Parc Guell_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_

B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Parc Guell_distance']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Parc Guell_distance']), B0 + B1*np.log(temp['Parc Guell_distance']), color = "maroon")
plt.show()

np.corrcoef(df['Parc Guell_distance'], df['LogPricePNight'])

# DISTANCIA JARDINES DE GRACIA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Jardinets de Gràcia_distance']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Jardinets de Gràcia_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Jardinets de Gràcia_distance'], temp['LogPricePNight'])

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(np.log(temp['Parc Guell_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_

B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Jardinets de Gràcia_distance']), temp['LogPricePNight'], alpha = 0.01, x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Jardinets de Gràcia_distance']), B0 + B1*np.log(temp['Jardinets de Gràcia_distance']), color = "maroon")
plt.show()

np.corrcoef(df['Jardinets de Gràcia_distance'], df['LogPricePNight'])

# DISTANCIA VILA OLIMPICA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Vila Olimpica_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Vila Olimpica_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Vila Olimpica_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Vila Olimpica_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Vila Olimpica_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Vila Olimpica_distance']), B0 + B1*np.log(temp['Vila Olimpica_distance']), 
         color = "maroon")
plt.show()

# DISTANCIA COLON

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Colon_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Colon_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Colon_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Colon_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Colon_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Colon_distance']), B0 + B1*np.log(temp['Colon_distance']), 
         color = "maroon")
plt.show()

np.corrcoef(df['Colon_distance'], df['LogPricePNight'])

# DISTANCIA ARC DL TRIOMF

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Arc de Triomf_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Arc de Triomf_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Arc de Triomf_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Arc de Triomf_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Arc de Triomf_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Arc de Triomf_distance']), B0 + B1*np.log(temp['Arc de Triomf_distance']), 
         color = "maroon")
plt.show()

np.corrcoef(df['Arc de Triomf_distance'], df['LogPricePNight'])

# DISTANCIA GLORIES

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Glories_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Glories_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Glories_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Glories_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Glories_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Glories_distance']), B0 + B1*np.log(temp['Glories_distance']), 
         color = "maroon")
plt.show()

np.corrcoef(df['Glories_distance'], df['LogPricePNight'])

# DISTANCIA HOSPITAL SANT PAU

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Hospital de Sant Pau_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Hospital de Sant Pau_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Hospital de Sant Pau_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Hospital de Sant Pau_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Hospital de Sant Pau_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Hospital de Sant Pau_distance']), B0 + B1*np.log(temp['Hospital de Sant Pau_distance']), 
         color = "maroon")
plt.show()

np.corrcoef(df['Hospital de Sant Pau_distance'], df['LogPricePNight'])

# DISTANCIA PLAZA CATALUÑA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Pl. Catalunya_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Pl. Catalunya_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Pl. Catalunya_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Pl. Catalunya_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Pl. Catalunya_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Pl. Catalunya_distance']), B0 + B1*np.log(temp['Pl. Catalunya_distance']), 
         color = "maroon")
plt.show()

np.corrcoef(df['Pl. Catalunya_distance'], df['LogPricePNight'])

# DISTANCIA PASEO DE GRACIA

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Pg. de Gràcia_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.show()

np.corrcoef(np.log(temp['Pg. de Gràcia_distance']), temp['LogPricePNight'])
np.corrcoef(temp['Pg. de Gràcia_distance'], temp['LogPricePNight'])

lr = LinearRegression()

lr.fit(np.log(temp['Pg. de Gràcia_distance']).values.reshape(-1, 1), temp['LogPricePNight'].values)

B1 = lr.coef_
B0 = lr.intercept_

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
sns.scatterplot(np.log(temp['Pg. de Gràcia_distance']), temp['LogPricePNight'], alpha = 0.01, 
                x_jitter = 0.1, y_jitter = 0.1, color = "navy", marker = 'o')
plt.plot(np.log(temp['Pg. de Gràcia_distance']), B0 + B1*np.log(temp['Pg. de Gràcia_distance']), 
         color = "maroon")
plt.show()

np.corrcoef(df['Pg. de Gràcia_distance'], df['LogPricePNight'])

# AÑADIMOS DATASET DE TURISMO

turismo = pd.read_csv('~/DadesAirBNB/DistanciasTurismo.csv')

df = pd.merge(df, turismo, on = 'id')
df.isnull().sum()[df.isnull().sum()>0]

# AÑADIMOS DATASET DEL TIEMPO

meteo = pd.read_csv('~/DadesAirBNB/Meteo/Meteo.csv')

meteo = meteo[['DATA_LECTURA', 'TM', 'PPT24H']]

meteo['DATA_LECTURA'] = pd.to_datetime(meteo['DATA_LECTURA'])
df.index = pd.to_datetime(df.index)


meteo = meteo[[x in [2017, 2018, 2019] for x in meteo['DATA_LECTURA'].dt.year]].set_index('DATA_LECTURA')

<<<<<<< HEAD
type(df.index)
type(meteo.index)

=======
df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace = True)
>>>>>>> 7ef347ea0576856c2d39718b8a1ebd11e4dbbd51
df = df.join(meteo, on = df.index)
df.reset_index(inplace = True)

# TEMPERATURA MEDIA

fig, ax = plt.subplots(2, 1, figsize = (15, 10))
sns.distplot(df['TM'], bins = 30, ax = ax[0])
sns.scatterplot(df['TM'], df['PricePNight'],  ax = ax[1], x_jitter = 0.05, y_jitter = 0.05, alpha = 0.2, marker = 'o')
plt.show()

np.corrcoef(df['TM'], df['LogPricePNight'])

# PRECIPITACIONES

df['PPT24H'].idxmax()

fig, ax = plt.subplots(2, 1, figsize = (15, 10))
ax[0].hist(df['PPT24H'])
sns.scatterplot(df['PPT24H'], df['LogPricePNight'], ax = ax[1], x_jitter = 0.05, y_jitter = 0.05, alpha = 0.2, marker = 'o')
plt.show()

np.corrcoef(df['LogPricePNight'], df['PPT24H'])

# CREAMOS EL CSV FINAL PARA MODELAR

df.to_pickle('~/DadesAirBNB/DatosModelar.pkl')
df.to_csv('~/DadesAirBNB/DatosModelar.csv', index = False)

<<<<<<< HEAD
df.PPT24H.isnull().sum()
df.TM.isnull().sum()

df.shape

=======
>>>>>>> 7ef347ea0576856c2d39718b8a1ebd11e4dbbd51
# PART PER REVISAR I ELIMINAR
# VAIG A FER UN INTENT DE MAPA
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point
import contextily as ctx

bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

map_df = df.reset_index().drop_duplicates(subset = ['id'])[['id', 'neighbourhood_group_cleansed', 'latitude', 'longitude']]

map_df['geometry'] = map_df.apply(lambda x: Point(x.longitude, x.latitude), axis = 1)

map_df = gpd.GeoDataFrame(map_df, geometry = map_df.geometry, crs = bcn_df.crs)

fig, ax = plt.subplots(1, 1, figsize = (20, 15))
map_df.plot(column = 'neighbourhood_group_cleansed', cmap = 'Set3', ax = ax)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (20, 20))
bcn_df.plot(color = 'lightblue', ax = ax)
map_df.plot(column = "neighbourhood_group_cleansed", cmap = "hsv", ax = ax, legend = True, markersize = 50, 
            marker = '.')
plt.show()

ax = map_df.plot(column = "neighbourhood_group_cleansed", cmap = "hsv", legend = True, markersize = 50, 
            marker = '.', figsize = (20, 20))
ctx.add_basemap(ax)
plt.show()

AlPbar = map_df.groupby('neighbourhood_group_cleansed').size().reset_index()
AlPbar.columns = ['neighbourhood_group', 'count']

bcn_df = pd.merge(bcn_df, AlPbar, on = 'neighbourhood_group')

ax = bcn_df.plot(column = "count", cmap = "YlOrRd", legend = True, figsize = (20, 20), alpha = 0.7, scheme = 'maximumbreaks')
ctx.add_basemap(ax)
plt.show()

mapa = gpd.sjoin(map_df, bcn_df, op = "within")
mapa.columns
mapa = gpd.sjoin(bcn_df, mapa.drop(['neighbourhood_group_cleansed', 'index_right', 'neighbourhood', 'neighbourhood_group', 'count'], a), op = "within")

# DATASET DELS TRANSPORTS

transport = pd.read_csv("/home/guillem/DadesAirBNB/Metro/TRANSPORTS.csv")

transport.columns

transport = transport[['NOM_CAPA', 'LONGITUD', 'LATITUD', 'EQUIPAMENT', 'NOM_BARRI']]

# METRO
transport['NOM_CAPA'].value_counts()
metro = transport[transport['NOM_CAPA'] == 'Metro i línies urbanes FGC']

metro['geometry'] = metro.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)

metro = gpd.GeoDataFrame(metro, geometry = metro.geometry, crs = bcn_df.crs)

metro = metro.to_crs(epsg = 3857)
map_df = map_df.to_crs(epsg = 3857)


map_df['dist_metro'] = [min(i.distance(j) for j in metro.geometry) for i in map_df.geometry]

map_df[map_df.index == map_df['dist_metro'].idxmax()][['geometry', 'dist_metro']]

for i in metro.geometry:
    print(map_df[map_df.index == map_df['dist_metro'].idxmax()].geometry.distance(i))

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_metro'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_metro'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
metro.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

# FGC

fgc = transport[transport['NOM_CAPA'] == 'Ferrocarrils Generalitat (FGC)']

fgc['geometry'] = fgc.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)
fgc = gpd.GeoDataFrame(fgc, geometry = fgc.geometry, crs = bcn_df.crs)
fgc = fgc.to_crs(epsg = 3857)

map_df['dist_fgc'] = [min(i.distance(j) for j in fgc.geometry) for i in map_df.geometry]

map_df[map_df.index == map_df['dist_fgc'].idxmax()][['geometry', 'dist_fgc']]

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_fgc'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_fgc'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
fgc.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

map_df[map_df.index == map_df['dist_metro'].idxmax()][['geometry', 'dist_metro', 'dist_fgc']]

# RENFE

renfe = transport[transport['NOM_CAPA'] == 'RENFE']

renfe['geometry'] = renfe.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)
renfe = gpd.GeoDataFrame(renfe, geometry = renfe.geometry, crs = bcn_df.crs)
renfe = renfe.to_crs(epsg = 3857)

map_df['dist_renfe'] = [min(i.distance(j) for j in renfe.geometry) for i in map_df.geometry]

map_df[map_df.index == map_df['dist_renfe'].idxmax()][['geometry', 'dist_renfe']]

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_renfe'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_renfe'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
renfe.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

map_df[map_df.index == map_df['dist_metro'].idxmax()][['geometry', 'dist_metro', 'dist_fgc', 'dist_renfe']]

# TREN AEROPUERTO

trenaer = transport[transport['NOM_CAPA'] == "Tren a l'aeroport"]

trenaer['geometry'] = trenaer.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)
trenaer = gpd.GeoDataFrame(trenaer, geometry = trenaer.geometry, crs = bcn_df.crs)
trenaer = trenaer.to_crs(epsg = 3857)

map_df['dist_trenaeropuerto'] = [min(i.distance(j) for j in trenaer.geometry) for i in map_df.geometry]

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_trenaeropuerto'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_trenaeropuerto'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
trenaer.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

# TRAMVIA

tramvia = transport[transport['NOM_CAPA'] == 'Tramvia']

tramvia['geometry'] = tramvia.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)
tramvia = gpd.GeoDataFrame(tramvia, geometry = tramvia.geometry, crs = bcn_df.crs)
tramvia = tramvia.to_crs(epsg = 3857)

map_df['dist_tramvia'] = [min(i.distance(j) for j in tramvia.geometry) for i in map_df.geometry]

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_tramvia'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_tramvia'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
tramvia.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

# DATASET AUTOBUSOS

transportsbus = pd.read_csv("~/DadesAirBNB/ESTACIONS_BUS.csv")

transportsbus = transportsbus[['NOM_CAPA', 'LONGITUD', 'LATITUD', 'EQUIPAMENT', 'NOM_BARRI']]

transportsbus['NOM_CAPA'].value_counts()

# BUS DIURN

bus = transportsbus[transportsbus['NOM_CAPA'] == 'Autobusos diürns']

bus['geometry'] = bus.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)
bus = gpd.GeoDataFrame(bus, geometry = bus.geometry, crs = bcn_df.crs)
bus = bus.to_crs(epsg = 3857)

map_df['dist_bus'] = [min(i.distance(j) for j in bus.geometry) for i in map_df.geometry]

map_df['dist_bus']

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_bus'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_bus'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
bus.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

# AEROBUS

aerobus = transportsbus[transportsbus['NOM_CAPA'] == "Autobus a l'aeroport"]

aerobus['geometry'] = aerobus.apply(lambda x: Point(x.LONGITUD, x.LATITUD), axis = 1)
aerobus = gpd.GeoDataFrame(aerobus, geometry = aerobus.geometry, crs = bcn_df.crs)
aerobus = aerobus.to_crs(epsg = 3857)

map_df['dist_aerobus'] = [min(i.distance(j) for j in aerobus.geometry) for i in map_df.geometry]

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
map_df[map_df.index == map_df['dist_aerobus'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['dist_aerobus'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
aerobus.plot(ax = ax, color = "navy")
ctx.add_basemap(ax)
plt.show()

distances = map_df[map_df.columns[map_df.columns.str.startswith('dist')].to_list()]

distances = distances.apply(lambda x: round(x, 2), axis = 1)

df.to_csv('~/DadesAirBNB/DatosModelar.csv', index = False)

#distances.to_csv('~/DadesAirBNB/distanciastransporte.csv')


