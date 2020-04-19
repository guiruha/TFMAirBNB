#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:46:00 2020

@author: guillem
"""

import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

#  MUSEOS

mus_bib = pd.read_csv("~/DadesAirBNB/Turisme/C001_Biblioteques_i_museus.csv")

dfmus_bib = mus_bib[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]

dfmus_bib = mus_bib.reset_index().drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]

museos = dfmus_bib[dfmus_bib.EQUIPAMENT.str.contains('Museu')]

museos = gpd.GeoDataFrame(museos, geometry = gpd.points_from_xy(museos.LONGITUD, museos.LATITUD), crs = bcn_df.crs)

museos = museos.to_crs(epsg=3857)

map_df = pd.read_csv('~/DadesAirBNB/Localizaciones.csv')

map_df = gpd.GeoDataFrame(map_df, geometry = gpd.points_from_xy(map_df.longitude, map_df.latitude), crs = bcn_df.crs)

map_df = map_df.to_crs(epsg = 3857)

distance = 500

temp = map_df.iloc[:10]
temp['geometry'] = temp['geometry'].iloc[:10].buffer(distance)

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
temp.plot(ax = ax)
museos.plot(ax = ax, color = "red")
ctx.add_basemap(ax)
plt.show()

lista = [int(museos.iloc[i].geometry.within(temp.iloc[0].geometry)) for i in range(museos.shape[0])]

mapdistances = map_df.copy()
mapdistances['geometry'] = mapdistances['geometry'].buffer(distance)

map_df['museos_cercanos'] = [sum(i.within(j) for i in museos.geometry) for j in mapdistances.geometry]

map_df['museos_cercanos']

# TEATRE

dft_visual = pd.read_csv("~/DadesAirBNB/Turisme/C002_Cinemes_teatres_auditoris.csv")

dft_visual['LATITUD'] = dft_visual['LATITUD'].astype('str').apply(lambda x: x[:2]+'.'+ x[2:]).astype('float')
dft_visual['LONGITUD'] = dft_visual['LONGITUD'].astype('str').apply(lambda x: x[0]+'.'+x[1:]).astype('float')

dft_visual = dft_visual.reset_index().drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'SECCIO','NUM_BARRI', 'LONGITUD', 'LATITUD']]

teatro = dft_visual[dft_visual.SECCIO.str.contains('Teatre')]

teatro = gpd.GeoDataFrame(teatro, geometry = gpd.points_from_xy(teatro.LONGITUD, teatro.LATITUD), crs = bcn_df.crs)

teatro = teatro.to_crs(epsg=3857)

map_df['teatros_cercanos'] = [sum(i.within(j) for i in teatro.geometry) for j in mapdistances.geometry]

# CINE

cine = dft_visual[dft_visual.SECCIO.str.contains('Cinema')]

cine = gpd.GeoDataFrame(cine, geometry = gpd.points_from_xy(cine.LONGITUD, cine.LATITUD), crs = bcn_df.crs)

cine = cine.to_crs(epsg = 3857)

map_df['cines_cercanos'] = [sum(i.within(j) for i in cine.geometry) for j in mapdistances.geometry]

# AUDITORIOS

auditorio = dft_visual[dft_visual.SECCIO.str.contains('Auditori')]

auditorio = gpd.GeoDataFrame(auditorio, geometry = gpd.points_from_xy(auditorio.LONGITUD, auditorio.LATITUD), crs = bcn_df.crs)

auditorio = auditorio.to_crs(epsg = 3857)

map_df['auditorios_cercanos'] = [sum(i.within(j) for i in auditorio.geometry) for j in mapdistances.geometry]

# RESTAURANTES

restaurantes = pd.read_csv("~/DadesAirBNB/Turisme/H001_Restaurants.csv")

restaurantes = restaurantes.drop_duplicates(subset = ['EQUIPAMENT'])

restaurantes['NOM'] = (restaurantes['EQUIPAMENT'] + ' - ' + restaurantes['SECCIO']).str.replace(' - #','') 

rest = restaurantes[['NOM', 'LATITUD', 'LONGITUD', 'NUM_BARRI']]

rest = gpd.GeoDataFrame(rest, geometry = gpd.points_from_xy(rest.LONGITUD, rest.LATITUD), crs = bcn_df.crs)

rest = rest.to_crs(epsg = 3857)

map_df['restaurantes_cercanos'] = [sum(i.within(j) for i in rest.geometry) for j in mapdistances.geometry]

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
map_df.plot(column = 'neighbourhood_group_cleansed', cmap = 'rainbow', ax = ax, legend = True)
ctx.add_basemap(ax)

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
rest.plot(ax = ax, color = "navy")
map_df[map_df.index == map_df['restaurantes_cercanos'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['restaurantes_cercanos'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
ctx.add_basemap(ax)
plt.show()

# SALAS MUSICALES

musica = pd.read_csv("~/DadesAirBNB/Turisme/C004_Espais_de_musica_i_copes.csv")

musica = musica.drop_duplicates(subset = ['EQUIPAMENT'])

df_musica = musica[['EQUIPAMENT', 'LATITUD', 'LONGITUD', 'NUM_BARRI']]

df_musica = gpd.GeoDataFrame(df_musica, geometry = gpd.points_from_xy(df_musica.LONGITUD, df_musica.LATITUD), crs = bcn_df.crs)

df_musica = rest.to_crs(epsg = 3857)

map_df['musica_cercanos'] = [sum(i.within(j) for i in df_musica.geometry) for j in mapdistances.geometry]

fig, ax = plt.subplots(1, 1, figsize = (15, 10))
map_df.plot(column = 'neighbourhood_group_cleansed', cmap = 'rainbow', ax = ax, legend = True)
ctx.add_basemap(ax)

fig, ax = plt.subplots(1, 1, figsize = (25, 25))
rest.plot(ax = ax, color = "navy")
map_df[map_df.index == map_df['musica_cercanos'].idxmax()].plot(ax = ax, marker = "X", markersize = 500, color = "maroon")
map_df[map_df.index == map_df['musica_cercanos'].idxmin()].plot(ax = ax, marker = "X", markersize = 500, color = "green")
ctx.add_basemap(ax)
plt.show()


# Creamos csv de Turismo

turismodist = map_df[[x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'geometry', 'latitude', 'longitude']]]

turismodist.to_csv('~/DadesAirBNB/DistanciasTurismo.csv', index = False)
