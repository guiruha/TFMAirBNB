#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:32:42 2020

@author: Guillem Rochina y Helena Saigí
"""

# Importamos librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_pickle('~/DadesAirBNB/Listings/April2017.pkl')
df = df.append(pd.read_pickle('~/DadesAirBNB/Listings/April2018.pkl'), ignore_index = True)
df = df.append(pd.read_pickle('~/DadesAirBNB/Listings/April2019.pkl'), ignore_index = True)
df = df.append(pd.read_pickle('~/DadesAirBNB/Listings/March2020.pkl'), ignore_index = True)

# Eliminamos la columnas con urls y los duplicados

df.drop(df.columns[df.columns.str.endswith('url')], axis = 1, inplace = True)
df = df.drop_duplicates('id')

# Eliminamos las columnas con más de un 60% de nulls

nulls = df.isnull().sum() / df.shape[0]
nulls = nulls[nulls>0.05]
df.drop(nulls[nulls>0.6].index, axis = 1, inplace = True)

# Eliminamos las columnas respecto al scraping o información no importante sobre el host

DropC = ['scrape_id', 'last_scraped', 'host_name', 'host_location', 'host_about',
         'host_neighbourhood', 'calendar_updated', 'calendar_last_scraped'] 
# Aunque parezca redundante hacer esto, ayuda a una mejor lectura del código.
df.drop(DropC, axis = 1, inplace = True)
    
# Eliminamos las columnas oportunas del análisis
# Si buscas la explicación la hallarás en el Google Colab o PDF del repositorio

df.drop(['experiences_offered'], axis = 1, inplace = True)

df['host_response_time'].fillna('undetermined', inplace = True)

df.drop('host_response_rate', axis = 1, inplace = True)

df['host_is_superhost'] = df['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

df['host_emailverified'] = df['host_verifications'].apply(lambda x: 1 if 'email' in x else 0)
df['host_phoneverified'] = df['host_verifications'].apply(lambda x: 1 if 'phone' in x else 0)
df['host_hasjumio'] = df['host_verifications'].apply(lambda x: 1 if 'jumio' in x else 0)
df['host_reviewverified'] = df['host_verifications'].apply(lambda x: 1 if 'review' in x else 0)
df['host_selfieverified'] = df['host_verifications'].apply(lambda x: 1 if 'selfie' in x else 0)
df['host_idverified'] = df['host_verifications'].apply(lambda x: 1 if 'government_id' in x else 0)

df.drop(['host_verifications'], axis = 1, inplace = True)

df['host_has_profile_pic'] = df['host_has_profile_pic'].apply(lambda x: 1 if x == 't' else 0)
df['host_identity_verified'] = df['host_identity_verified'].apply(lambda x: 1 if x == 't' else 0)

DropC = ['street', 'neighbourhood', 'neighbourhood_cleansed']
df.drop(DropC, axis = 1, inplace = True)

DropC = ['city', 'state', 'zipcode', 'market', 'smart_location', 'country_code', 'country']
df.drop(DropC, axis = 1, inplace = True)

df['is_location_exact'] = df['is_location_exact'].apply(lambda x: 1 if x == 't' else 0)

tempdict = {'Loft':'Apartment', 'Condominium':'House', 'Serviced apartment': 'Apartment', 
            'Bed & Breakfast':'Hotel', 'Guest suite': 'Hotel', 'Hostel': 'Hotel', 
            'Dorm': 'Hotel', 'Casa particular (Cuba)':'House', 'Aparthotel': 'Hotel',
            'Townhouse': 'House', 'Villa': 'House', 'Vaction home': 'House', 'Chalet': 'House', 
            'Dome house': 'House', 'Tiny house': 'House', 'Casa Particular': 'House', 
            'Apartment': 'Apartment', 'Hotel':'Hotel', 'House':'House'}

df['property_type'] = df['property_type'].map(tempdict).fillna('Other')

df.drop(['bed_type'], axis = 1, inplace = True)

columnselection = ['Air conditioning', 'Family/kid friendly', 'Host greets you', 
                   'Laptop friendly workspace', 'Paid parking off premises', 
                  'Patio or balcony', 'Luggage dropoff allowed', 'Long term stays allowed', 
                  'Smoking allowed', 'Step-free access', 'Pets allowed', '24-hour check-in']

for column in columnselection:
    df[column] = df['amenities'].apply(lambda x: 1 if column in x else 0)
df['Elevator'] = df['amenities'].apply(lambda x: 1 if ('Elevator' in x or 'Elevator in building' in x) else 0)
df.drop('amenities', axis = 1, inplace = True)

df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype('float')
df['security_deposit'] = df['security_deposit'].str.replace('$', '').str.replace(',','').fillna(0).astype('float')
df['cleaning_fee'] = df['cleaning_fee'].str.replace('$', '').str.replace(',','').fillna(0).astype('float')
df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',','').fillna(0).astype('float')

df.drop(['has_availability'], axis = 1, inplace = True)

df.drop(['first_review', 'last_review'], axis = 1, inplace = True)

tempdict = {'strict_14_with_grace_period': 'strict_less30', 'flexible':'flexible', 
            'moderate':'moderate', 'luxury_moderate': 'moderate', 'super_strict_30':'strict_30orMore', 
            'super_strict_60':'strict_30orMore', 'strict':'strict_less30'}

df['cancellation_policy'] = df['cancellation_policy'].map(tempdict)

df.drop(['require_guest_profile_picture'], axis = 1, inplace = True)

df.drop(['require_guest_phone_verification'], axis = 1, inplace = True)

df.drop(['is_business_travel_ready'], axis = 1, inplace = True)

df.drop(['host_id'], axis = 1, inplace = True)

df.drop(['host_total_listings_count'], axis = 1, inplace = True)
df.drop(['calculated_host_listings_count'], axis = 1, inplace = True)
df.dropna(subset = ['host_listings_count'], inplace = True)

df['bathrooms_imput'] = df['bathrooms'].isnull().astype('int')
df['bathrooms'].fillna(df['bathrooms'].median(), inplace = True)

df['bedrooms_imput'] = df['bedrooms'].isnull().astype('int')
df['bedrooms'].fillna(df['bedrooms'].median(), inplace = True)

df['beds_imput'] = df['beds'].isnull().astype('int')
df['beds'].fillna(df['beds'].median(), inplace = True)

df.drop(['availability_30', 'availability_60', 'availability_90'], axis = 1, inplace = True)

df.drop(['reviews_per_month'], axis = 1, inplace = True)

df.drop(['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
         'review_scores_communication', 'review_scores_location', 'review_scores_value'], 
        axis = 1, inplace = True)

df['review_scores_rating'] = df['review_scores_rating'].apply(lambda x: 'Excellent' if x >= 90 else ('NotExcellent' if x < 90 else 'Unavailable'))

cal = pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_April2016.pkl")
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_April2017.pkl"), ignore_index = True)
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_October2017.pkl"), ignore_index = True)
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_April2018.pkl"), ignore_index = True)
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_October2018.pkl"), ignore_index = True)
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_April2019.pkl"), ignore_index = True)
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_October2019.pkl"), ignore_index = True)
cal = cal.append(pd.read_pickle("/home/guillem/DadesAirBNB/Calendar/Calendar_March2020.pkl"), ignore_index = True)

cal = cal[['listing_id', 'date', 'price']]
cal['date'] = pd.to_datetime(cal['date'])

cal['month_year'] = cal['date'].dt.to_period('M')
cal['year'] = cal['date'].dt.year
cal['month'] = cal['date'].dt.month

cal['price'] = cal['price'].str.replace('$', '').str.replace(',', '').astype('float')

cal = cal.groupby(['month_year', 'year', 'month', 'listing_id']).mean().reset_index()
cal.columns = ['month_year', 'year', 'month', 'id', 'price_calendar']

dfclean = pd.merge(df, cal, how = 'inner', on = 'id')

dfclean['price_calendar'] = dfclean['price_calendar'].fillna(0)

def imputer(x):
  """Comprueba si la columna de price_calendar tiene un 0 y en ese caso
  añade el valor de price, en otro caso el valor de price_calendar no
  es alterado."""
  if x[0] == 0:
    return x[0] + x[1]
  else:
    return x[0]

dfclean['goodprice'] = dfclean[['price_calendar', 'price']].apply(imputer, axis = 1)

dfclean.drop(['price', 'price_calendar'], axis = 1, inplace = True)

print('Nos hemos quedado con un dataframe de {} filas y {} columnas\n Procedemos a guardarlo en un archivo'.format(dfclean.shape[0], dfclean.shape[1]))

#dfclean.to_csv('~/DadesAirBNB/DatosLimpios.csv', index = False)
dfclean.to_pickle('~/DadesAirBNB/DatosLimpios.pkl')