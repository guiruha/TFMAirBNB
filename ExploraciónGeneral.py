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
plt.style.use('fivethirtyeight')

archives = !ls | grep 'df'

df = pd.read_csv(archives[0])

for i in range(1, len(archives)):
    df = df.append(pd.read_csv(archives[i]))
df.shape
df.drop(['Unnamed: 0', 'requires_license', 'price'], axis = 1, inplace = True)
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

df = df[(df['Year']>2017)&(df['Year']<2021)]


df['PricePNight'] = [price/minnight if (price > 300)&(minnight > 1) else price for price, minnight in zip(df['goodprice'], df['minimum_nights'])]

df[(df['goodprice']>300)&(df['minimum_nights']>1)][['goodprice', 'minimum_nights', 'PricePNight']]

df[(df['goodprice']>= 1200)&(df['minimum_nights']>1)]['id'].value_counts()

def who(x):
    print('\n', df[df['id']==x]['name'].unique(), '\n')
    return 'https://www.airbnb.es/rooms/' + str(x)

who(21290246)

df = df[(df['goodprice']<1200)]
df = df[(df['PricePNight'] >= 8)]

(df[df['goodprice']<= 600].shape[0] - df.shape[0])/df.shape[0]

df = df[df['goodprice']<= 600]

# EXPLORACIÓN DE LA VARIABLE DEPENDIENTE

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.distplot(df['PricePNight'], bins = 13)
plt.tight_layout()

df['PricePNight'].describe()

df['LogPricePNight'] = np.log(df['PricePNight'])
#df.drop(['goodprice'], axis = 1, inplace = True)

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

df.groupby(df.index)['PricePNight'].describe()[df.groupby(df.index)['PricePNight'].mean() == df.groupby(df.index)['PricePNight'].mean().max()]


fig, ax = plt.subplots(1, 1, figsize = (20, 13))
plt.plot(df[df.resample('M')['PricePNight'].mean().index, df[df.resample('M')['PricePNight'].mean())
plt.xticks(rotation = 45)
plt.tight_layout()

fig, ax = plt.subplots(1, 1, figsize = (50, 13))
plt.plot(df.resample('W')['PricePNight'].mean().index, df.resample('W')['PricePNight'].mean())
plt.xticks(df[df.resample('W')['PricePNight'].mean().index, rotation = 45)
plt.tight_layout()

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
sns.barplot(df['host_response_time'].value_counts().index, df['host_response_time'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_response_time'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_response_time'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

dummys = ['host_response_time']
# HOST IS SUPERHOST

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_is_superhost'].value_counts().index, df['host_is_superhost'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_is_superhost'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_is_superhost'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('host_is_superhost', axis = 1, inplace = True)

# HOST TOTAL LISTINGS COUNT (FER REGRESIÓ)

fig, ax = plt.subplots(1, 2, figsize = (20, 15))
sns.distplot(df['host_total_listings_count'], ax = ax[0])
sns.distplot(np.log(df['host_total_listings_count'].apply(lambda x: 0.01 if x == 0 else x)), ax = ax[1])
plt.show()

sns.regplot(df['host_total_listings_count'], df['LogPricePNight'])


# PROFILE PIC

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_has_profile_pic'].value_counts().index, df['host_has_profile_pic'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_has_profile_pic'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_has_profile_pic'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('host_has_profile_pic', axis = 1, inplace = True)

# IDENTITY VERIFIED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['host_identity_verified'].value_counts().index, df['host_identity_verified'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['host_identity_verified'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['host_identity_verified'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

df.drop('host_identity_verified', axis = 1, inplace = True)

# NEIGHBOURHOOD GROUP CLEANSED

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['neighbourhood_group_cleansed'].value_counts().index, df['neighbourhood_group_cleansed'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['neighbourhood_group_cleansed'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['neighbourhood_group_cleansed'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
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

df['property_type'] = df['property_type'].apply(lambda x: 'other' if x != 'apartment' else 'apartment')

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['property_type'].value_counts().index, df['property_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['property_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['property_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# ROOM TYPE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['room_type'].value_counts().index, df['room_type'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['room_type'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['room_type'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

# NOMÉS ENTIRE HOME y PRIVATE ROOM COM A DUMMYS!!!!!!!!!!!!!!!!!!!
dummys.append('room_type')

# ACCOMMODATES

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['accommodates'].value_counts().index, df['accommodates'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['accommodates'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['accommodates'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

temp = df[df['property_type'] == 'other']

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(temp['accommodates'].value_counts().index, temp['accommodates'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(temp['accommodates'], temp['PricePNight'], ax = ax[1])
sns.boxplot(temp['accommodates'], temp['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

temp = df[df['property_type'] != 'other']

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

df = df[df['bathrooms']<= 3]

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

# REVIEW SCORE LOCATION


fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_location'].value_counts().index, df['review_scores_location'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_location'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_location'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['review_scores_location'], df['LogPricePNight'])

# REVIEW SCORE VALUE

fig, ax = plt.subplots(3, 1, figsize = (20, 20))
sns.barplot(df['review_scores_value'].value_counts().index, df['review_scores_value'].value_counts()/df.shape[0], ax = ax[0])
sns.pointplot(df['review_scores_value'], df['PricePNight'], ax = ax[1])
sns.boxplot(df['review_scores_value'], df['LogPricePNight'], ax = ax[2])
plt.xticks(rotation = 45)
plt.show()

np.corrcoef(df['review_scores_value'], df['LogPricePNight'])

columnes = ['bedrooms', 'beds', 'bed_type', 'instant_bookable', 'is_business_travel_ready',
       'cancellation_policy', 'require_guest_profile_picture',
       'require_guest_phone_verification', 'reviews_per_month',
       'host_emailverified', 'host_phoneverified', 'host_hasjumio',
       'host_reviewverified', 'host_selfieverified', 
        'Paid parking off premises',
       'Patio or balcony', 'Luggage dropoff allowed',
       'Long term stays allowed',  'Step-free access',
       'Paid parking on premises',
       'host_acceptance_rate'


# VAIG A FER UN INTENT DE MAPA
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point
import contextily as ctx

bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

map_df = df.reset_index().drop_duplicates(subset = ['id'])[['neighbourhood_group_cleansed', 'latitude', 'longitude']]

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

map_df = map_df.to_crs(epsg=3857)
bcn_df = bcn_df.to_crs(epsg=3857)

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
