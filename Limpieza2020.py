#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:20:07 2020

@author: guillem
"""

import pandas as pd
import numpy as np

df = pd.read_csv('~/DadesAirBNB/Listings/April2019.csv')
df = df.append(pd.read_csv('~/DadesAirBNB/Listings/October2019.csv'))
df = df.append(pd.read_csv('~/DadesAirBNB/Listings/March2020.csv'))


df.drop(df.columns[df.columns.str.endswith('url')], axis = 1, inplace = True)

nulls = df.isnull().sum() / df.shape[0]
nulls = nulls[nulls>0.05]

df.drop(nulls[nulls>0.6].index, axis = 1, inplace = True)

dropC = ['city', 'state', 'zipcode', 'country', 'country_code']
df.drop(dropC, axis = 1, inplace = True)

maxmincols = [x for x in df.columns if (x.startswith('maximum') | x.startswith('minimum'))]
maxmincols

maxmin = df[maxmincols]

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

host_keys = [x for x in df.columns if x.startswith('host')]

df['host_emailverified'] = df['host_verifications'].apply(lambda x: 1 if 'email' in x else 0)
df['host_phoneverified'] = df['host_verifications'].apply(lambda x: 1 if 'phone' in x else 0)
df['host_hasjumio'] = df['host_verifications'].apply(lambda x: 1 if 'jumio' in x else 0)
df['host_reviewverified'] = df['host_verifications'].apply(lambda x: 1 if 'review' in x else 0)
df['host_selfieverified'] = df['host_verifications'].apply(lambda x: 1 if 'selfie' in x else 0)

dropC = ['host_name','host_about', 'host_neighbourhood', 'host_listings_count', 'host_verifications']
df.drop(dropC, axis = 1, inplace = True)

df.drop('experiences_offered', axis = 1, inplace = True)

df['host_response_time'].fillna('undetermined', inplace = True)

df['host_response_rate'] = df['host_response_rate'].str.replace('%', '')

df.drop('host_response_rate', axis = 1, inplace = True)

variablesdicotomicas = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']
for variable in variablesdicotomicas:
    df[variable] = df[variable].apply(lambda x: 1 if x == 't' else 0)
    
dropC = ['market', 'street', 'smart_location']
df.drop(dropC, axis = 1, inplace = True)

for column in ['bathrooms', 'bedrooms', 'beds']:
    df[column].fillna(df[column].median(), inplace = True)
    
df['price'] = df['price'].str.replace('$', '').str.replace(',','').astype('float')

columnselection = ['Air conditioning', 'Family/kid friendly', 'Host greets you', 'Laptop friendly workspace', 'Paid parking off premises', 
                  'Patio or balcony', 'Luggage dropoff allowed', 'Long term stays allowed', 'Smoking allowed', 'Step-free access',
                  'Paid parking on premises']
for column in columnselection:
    df[column] = df['amenities'].apply(lambda x: 1 if column in x else 0)
df.drop('amenities', axis = 1, inplace = True)

for column in ['security_deposit', 'cleaning_fee', 'extra_people']:
    df[column] = df[column].str.replace('$', '').str.replace(',','').astype('float')
    df[column].fillna(0, inplace = True)

dropC = ['has_availability', 'availability_30', 'availability_60', 'availability_90']
df.drop(dropC, axis = 1, inplace = True)

cancelpol = {'strict_14_with_grace_period': 'strict_less30', 'flexible':'flexible', 'moderate':'moderate', 'super_strict_30':'strict_30orMore',
           'super_strict_60':'strict_30orMore', 'strict':'strict_less30'}
df['cancellation_policy'] = df['cancellation_policy'].map(cancelpol)

df.dropna(subset = ['last_review'], inplace = True)

for column in df.columns[df.columns.str.startswith('review')]:
    df.dropna(subset = [column], inplace = True)

dropC = ['calendar_updated', 'calendar_last_scraped', 'host_location', 'last_review', 'last_scraped', 
         'neighbourhood', 'neighbourhood_cleansed']
df.drop(dropC, axis = 1, inplace = True)

df.dropna(subset = ['host_since'], inplace = True)

new_cat = {'Apartment':'apartment', 'Service apartment': 'hotel', 'Loft': 'apartment', 'House':'house', 'Condominium':'house',
          'Bed and breakfast':'hostel', 'Guest suite':'hotel', 'Hostel':'hostel', 'Boutique hotel':'hotel', 'Boat':'boat', 'Guest House':'hostel',
          'Hotel':'hotel', 'Townhouse':'house', 'Aparthotel':'hotel', 'Casa particular (Cuba)':'house', 'Villa':'villa', 'Chalet':'house', 
          'Houseboat':'boat', 'Resort':'hotel'}

df['property_type'] = df['property_type'].map(new_cat).fillna('other')

df.drop('scrape_id', axis = 1, inplace = True)
df.columns

df.drop(df.columns[df.columns.str.startswith('calculated')], axis = 1, inplace = True)

dicotmicol = ['instant_bookable', 'is_business_travel_ready', 'is_location_exact', 'require_guest_phone_verification',
    'require_guest_profile_picture', 'requires_license']
for column in dicotmicol:
    df[column] = df[column].apply(lambda x: 1 if x=='t' else 0)

DropC = ['host_id', 'first_review', 'license', 'number_of_reviews_ltm']
df.drop(DropC, axis = 1, inplace = True)

cal = pd.read_csv("/home/guillem/DadesAirBNB/Calendar/Calendar_April2019.csv")
cal = cal.append(pd.read_csv("/home/guillem/DadesAirBNB/Calendar/Calendar_October2019.csv"), ignore_index = True)
cal = cal.append(pd.read_csv("/home/guillem/DadesAirBNB/Calendar/Calendar_March2020.csv"))


cal['date'] = pd.to_datetime(cal['date'])

cal = cal[(cal['date'].dt.day == 1) | (cal['date'].dt.day == 7)| (cal['date'].dt.day == 13)| (cal['date'].dt.day == 19)| (cal['date'].dt.day == 25)| (cal['date'].dt.day == 30)]

cal = cal[['listing_id', 'date', 'price', 'available']]

cal.columns  = ['id', 'date', 'goodprice', 'available']

cal = cal.drop_duplicates(subset = ['date', 'goodprice', 'id'])

cal['goodprice'] = cal['goodprice'].str.replace('$', '').str.replace(',','').astype('float')

DF = pd.merge(df, cal, how = 'inner', on = 'id')

DF.drop_duplicates(subset = ['date', 'id', 'goodprice'], inplace = True)

DF.drop(['requires_license', 'price'], axis = 1, inplace = True)

DF.to_csv('~/DadesAirBNB/df2019.csv',  index = False)


# DECOMENTA PER COMPROBAR QUE LA REPARTICIÓ HA SIGUT EQUITATIVA
#DF.groupby('date')['adjusted_price'].count()
#DF.groupby('id')['adjusted_price'].count()
#DF.groupby(['id', 'date'])['adjusted_price'].count()
