#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:11:17 2020

@author: guillem
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from datetime import date
plt.style.use('fivethirtyeight')

print('\n Importamos el dataset limpio')

df = pd.read_pickle('/home/guillem/DadesAirBNB/DatosLimpios.pkl') 

df = df[(df['year']>2016)&(df['year']<2021)]

print('\nEliminamos precios superiores a {} que suponen un {}% del dataframe'.format(1100, (df[df['goodprice']>1100].shape[0]/df.shape[0])*100))

df = df[(df['goodprice']<1100)]

df['LogGoodprice'] = np.log(df['goodprice'])

print('\nEmpezamos las transformaciones pertinentes a las variables del dataset principal')

df['LogAcommodates'] = np.log(df['accommodates'])
df.drop('accommodates', axis = 1, inplace = True)

df['availability_365_sqrt'] = np.sqrt(df['availability_365'])
df.drop('availability_365', axis = 1, inplace = True)

df['LogBathrooms'] = np.log(df['bathrooms'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('bathrooms', axis = 1, inplace = True)

df['LogBedrooms'] = np.log(df['bedrooms'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('bedrooms', axis = 1, inplace = True)

df['LogBeds'] = np.log(df['beds'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('beds', axis = 1, inplace = True)

df['SqrtCleaning_fee'] = np.sqrt(df['cleaning_fee'])
df.drop('cleaning_fee', axis = 1, inplace = True)

df['LogGuest_included'] = np.log(df['guests_included'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('guests_included', axis = 1, inplace = True)

df = df[df['minimum_nights']<367]

df['LogMaximum_nights'] = np.log(df['maximum_nights'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('maximum_nights', axis = 1, inplace = True)

df['LogNumber_of_reviews'] = np.log(df['number_of_reviews'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('number_of_reviews', axis = 1, inplace = True)

df['LogSecurity_deposit'] = np.log(df['security_deposit'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('security_deposit', axis = 1, inplace = True)

df['host_since'] = df['host_since'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
df['host_seniority'] = df['host_since'].apply(lambda x: ((date.today() - x).days/30))
df['CubicHost_seniority'] = (df['host_seniority'])**3
df.drop(['host_seniority', 'host_since'], axis = 1, inplace = True)

df['LogHost_listings_count'] = np.log(df['host_listings_count'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('host_listings_count', axis = 1, inplace = True)

df.drop('host_has_profile_pic', axis = 1, inplace = True)

df.drop('host_is_superhost', axis = 1, inplace = True)

df.drop('host_emailverified', axis = 1, inplace = True)

df.drop('host_phoneverified', axis = 1, inplace = True)

df.drop('host_selfieverified', axis = 1, inplace = True)

df.drop('host_idverified', axis = 1, inplace = True)

df.drop('Laptop friendly workspace', axis = 1, inplace = True)

df.drop('Paid parking off premises', axis = 1, inplace = True)

df.drop('Luggage dropoff allowed', axis = 1, inplace = True)

df.drop('Long term stays allowed', axis = 1, inplace = True)

df.drop('Pets allowed', axis = 1, inplace = True)

print('\nGeneramos los datasets para geoexploración')

map_df = df.reset_index().drop_duplicates(subset = ['id'])[['id', 'neighbourhood_group_cleansed', 'latitude', 'longitude']]

map_df.to_pickle('~/DadesAirBNB/Localizaciones.pkl')
map_df.to_csv('~/DadesAirBNB/Localizaciones.csv', index = False)
map_df.to_csv('~/DadesAirBNB/Localizaciones.zip', index = False)


print('\nImportamos archivos de distancias y transportes y procedemos a su tratamiento')

distancias = pd.read_pickle('/home/guillem/DadesAirBNB/Distancias.pkl')
turismo = pd.read_pickle('/home/guillem/DadesAirBNB/DistanciasTurismo.pkl')
df = pd.merge(df, distancias, on = 'id')
df = pd.merge(df, turismo, on = 'id')

df['LogLandmark_2_distance'] = np.log(df['Landmark_2_distance'])
df.drop('Landmark_2_distance', axis = 1, inplace = True)

df['CubicLandmark_3_distance'] = df['Landmark_3_distance']**3
df.drop('Landmark_3_distance', axis = 1, inplace = True)

df['CubicLandmark_4_distance'] = df['Landmark_4_distance']**3
df.drop('Landmark_4_distance', axis = 1, inplace = True)

df['SquaredLandmark_6_distance'] = df['Landmark_6_distance']**2
df.drop('Landmark_6_distance', axis = 1, inplace = True)

df['SquaredLandmark_8_distance'] = df['Landmark_8_distance']**2
df.drop('Landmark_8_distance', axis = 1, inplace = True)

df['SquaredLandmark_11_distance'] = df['Landmark_11_distance']**2
df.drop('Landmark_11_distance', axis = 1, inplace = True)

df['Cubicrenfe_distance'] = df['renfe_distance']**3
df.drop('renfe_distance', axis = 1, inplace = True)

df['Logtrenaer_distance'] = np.log(df['trenaer_distance'])
df.drop('trenaer_distance', axis = 1, inplace = True)

df['Logrestaurantes_cercanos'] = np.log(df['restaurantes_cercanos'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('restaurantes_cercanos', axis = 1, inplace = True)

df['Logmusica_cercanos'] = np.log(df['musica_cercanos'].apply(lambda x: 0.5 if x == 0 else x))
df.drop('musica_cercanos', axis = 1, inplace = True)

print('\nCreamos las features dummy que necesitaremos para la fase de modelado')
dummycols = ['host_response_time', 'neighbourhood_group_cleansed', 'property_type', 'room_type',  'cancellation_policy', 'review_scores_rating']
df = pd.get_dummies(df, columns = dummycols, drop_first = True)
df[df.columns[list(df.dtypes == 'uint8')]] = df[df.columns[list(df.dtypes == 'uint8')]].astype('int')

print('\nBuscamos correlaciones para eliminar posibles colinealidades')

corrcolumns = df.dtypes[df.dtypes != 'object'].index[df.dtypes[(df.dtypes != 'object')].index.str.contains('rice') == False]
corrcolumns = corrcolumns[corrcolumns != 'month_year']

colineales = []
correlations = df[corrcolumns].corr()
for column in corrcolumns:
    for row in corrcolumns:
        if (np.abs(correlations.loc[row, column]) > 0.75) & (correlations.loc[row, column] !=  1):
            colineales.append('{} <-> {} == {}'.format(row, column, correlations.loc[row, column]))
            
[x for x in colineales if 'dist' not in x]

for column in [x.split()[0] for x in colineales if 'dist' not in x]:
    print('{} tiene una correlación con goodprice de: {}'.format(column , np.corrcoef(df[column], df['LogGoodprice'])[0,1]))

print('\nEliminamos las columnas con colinealidad y menor correlación con LogGoodprice')

df.drop(['host_identity_verified', 'availability_365_sqrt', 'LogBeds', 'LogBedrooms', 'Logmusica_cercanos'],
        axis = 1, inplace = True)

print('\nGeneramos el dataset resultante para la etapa de modelado')

df.to_pickle('~/DadesAirBNB/DatosModelar.pkl')
df.to_csv('~/DadesAirBNB/DatosModelar.csv', index = False)