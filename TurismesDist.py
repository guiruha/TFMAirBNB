#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:46:00 2020

@author: Guillem Rochina y Helena Saigí
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
map_df = pd.read_csv('~/DadesAirBNB/Localizaciones.csv')
map_df = gpd.GeoDataFrame(map_df, geometry = gpd.points_from_xy(map_df.longitude, map_df.latitude), crs = bcn_df.crs)
map_df = map_df.to_crs(epsg = 3857)

#  MUSEOS

mus_bib = pd.read_pickle("~/DadesAirBNB/Turisme/C001_Biblioteques_i_museus.pkl")
mus_bib = mus_bib[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]
mus_bib = mus_bib.drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]

museos = mus_bib[mus_bib.EQUIPAMENT.str.contains('Museu')]

museos = gpd.GeoDataFrame(museos, geometry = gpd.points_from_xy(museos.LONGITUD, museos.LATITUD), crs = bcn_df.crs)

museos = museos.to_crs(epsg=3857)

# CREAMOS LAS GEOMETRIAS DE EMJEMPLO CON BUFFER

distance = 600
temp = map_df.tail(10)
temp['geometry'] = temp['geometry'].buffer(distance)

# MAPA DE GEOMETRÍAS CON BUFFER Y MUSEOS

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
temp.plot(ax = ax, color = "blue", edgecolor = "black")
museos.plot(ax = ax, color = "maroon", label = 'Museos')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Geometrías con buffer de ejemplo y Museos', fontsize = 20)
ctx.add_basemap(ax)
ax.axis('off')
plt.show()

# CREAMOS LAS GEOMETRIAS CON BUFFER

mapbuffer = map_df.copy()
mapbuffer['geometry'] = mapdistances['geometry'].buffer(distance)

# CÁLCULAMOS MUSESOS CERCANOS

map_df['museos_cercanos'] = mapbuffer['geometry'].apply(lambda x: sum(x.distance(j) for j in museos.geometry))
#map_df['museos_cercanos'] = [sum(i.within(j) for i in museos.geometry) for j in mapdistances.geometry]


# TEATROS

visual = pd.read_pickle("~/DadesAirBNB/Turisme/C002_Cinemes_teatres_auditoris.pkl")

visual['LATITUD'] = visual['LATITUD'].astype('str').apply(lambda x: x[:2]+'.'+ x[2:]).astype('float')
visual['LONGITUD'] = visual['LONGITUD'].astype('str').apply(lambda x: x[0]+'.'+x[1:]).astype('float')

visual = visual.drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'SECCIO','NUM_BARRI', 'LONGITUD', 'LATITUD']]

teatro = visual[visual.SECCIO.str.contains('Teatre')]

teatro = gpd.GeoDataFrame(teatro, geometry = gpd.points_from_xy(teatro.LONGITUD, teatro.LATITUD), crs = bcn_df.crs)

teatro = teatro.to_crs(epsg=3857)

# MAPA DE GEOMETRÍAS CON BUFFER Y TEATROS

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
temp.plot(ax = ax, color = "blue", edgecolor = "black")
teatro.plot(ax = ax, color = "maroon", label = 'Teatros')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Geometrías con buffer de ejemplo y Teatros', fontsize = 20)
ctx.add_basemap(ax)
ax.axis('off')
plt.show()

# CÁLCULAMOS TEATROS CERCANOS

map_df['teatros_cercanos'] = mapbuffer['geometry'].apply(lambda x: sum(x.distance(j) for j in teatro.geometry))

# CINE

cine = visual[visual.SECCIO.str.contains('Cinema')]

cine = gpd.GeoDataFrame(cine, geometry = gpd.points_from_xy(cine.LONGITUD, cine.LATITUD), crs = bcn_df.crs)
cine = cine.to_crs(epsg = 3857)

# MAPA DE GEOMETRÍAS CON BUFFER Y CINES

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
temp.plot(ax = ax, color = "blue", edgecolor = "black")
cine.plot(ax = ax, color = "maroon", label = 'Cines')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Geometrías con buffer de ejemplo y Cines', fontsize = 20)
ctx.add_basemap(ax)
ax.axis('off')
plt.show()

# CÁLCULAMOS CINES CERCANOS

map_df['cines_cercanos'] = mapbuffer['geometry'].apply(lambda x: sum(x.distance(j) for j in cine.geometry))

# AUDITORIOS

auditorio = visual[visual.SECCIO.str.contains('Auditori')]

auditorio = gpd.GeoDataFrame(auditorio, geometry = gpd.points_from_xy(auditorio.LONGITUD, auditorio.LATITUD), crs = bcn_df.crs)
auditorio = auditorio.to_crs(epsg = 3857)

# MAPA DE GEOMETRÍAS CON BUFFER Y AUDITORIOS

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
temp.plot(ax = ax, color = "blue", edgecolor = "black")
cine.plot(ax = ax, color = "maroon", label = 'Auditorios')
plt.legend(fontsize = 20, loc = 'upper left')
plt.title('Geometrías con buffer de ejemplo y Auditorios', fontsize = 20)
ctx.add_basemap(ax)
ax.axis('off')
plt.show()

# CÁLCULAMOS AUDITORIOS CERCANOS

map_df['auditorios_cercanos'] = mapbuffer['geometry'].apply(lambda x: sum(x.distance(j) for j in auditorio.geometry))


# RESTAURANTES

restaurantes = pd.read_pickle("~/DadesAirBNB/Turisme/H001_Restaurants.pkl")

restaurantes = restaurantes.drop_duplicates(subset = ['EQUIPAMENT'])

restaurantes['NOM'] = (restaurantes['EQUIPAMENT'] + ' - ' + restaurantes['SECCIO']).str.replace(' - #','') 

rest = restaurantes[['NOM', 'LATITUD', 'LONGITUD', 'NUM_BARRI']]

rest = gpd.GeoDataFrame(rest, geometry = gpd.points_from_xy(rest.LONGITUD, rest.LATITUD), crs = bcn_df.crs)
rest = rest.to_crs(epsg = 3857)

# MAPA DE GEOMETRÍAS CON BUFFER Y RESTAURANTES
fig, ax = plt.subplots(1, 1, figsize = (15, 15))
temp.plot(ax = ax, color = "blue", edgecolor = "black")
rest.plot(ax = ax, color = "maroon", label = 'Restaurantes', markersize = 10)
plt.legend(fontsize = 20, loc = 'upper left')
plt.title('Geometrías con buffer de ejemplo y Restaurantes', fontsize = 20)
ctx.add_basemap(ax)
ax.axis('off')
plt.show()

# CÁLCULAMOS RESTAURANTES CERCANOS

map_df['restaurantes_cercanos'] = mapbuffer['geometry'].apply(lambda x: sum(x.distance(j) for j in rest.geometry))

# SALAS MUSICALES

musica = pd.read_pickle("~/DadesAirBNB/Turisme/C004_Espais_de_musica_i_copes.pkl")

musica = musica.drop_duplicates(subset = ['EQUIPAMENT'])

musica = musica[['EQUIPAMENT', 'LATITUD', 'LONGITUD', 'NUM_BARRI']]

musica = gpd.GeoDataFrame(df_musica, geometry = gpd.points_from_xy(df_musica.LONGITUD, df_musica.LATITUD), crs = bcn_df.crs)
musica = rest.to_crs(epsg = 3857)

# CÁLCULAMOS SALAS Y DISCOTECAS CERCANAS
map_df['musica_cercanos'] = mapbuffer['geometry'].apply(lambda x: sum(x.distance(j) for j in musica.geometry))

# MAPA DE GEOMETRÍAS CON BUFFER Y RESTAURANTES
fig, ax = plt.subplots(1, 1, figsize = (15, 15))
temp.plot(ax = ax, color = "blue", edgecolor = "black")
musica.plot(ax = ax, color = "maroon", label = 'Salas de Música', markersize = 10)
plt.legend(fontsize = 20, loc = 'upper left')
plt.title('Geometrías con buffer de ejemplo y Salas/Discotecas', fontsize = 20)
ctx.add_basemap(ax)
ax.axis('off')
plt.show()

# Creamos csv de Turismo

turismodist = map_df[[x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'geometry', 'latitude', 'longitude']]]

turismodist.to_pickle('~/DadesAirBNB/DistanciasTurismo.pkl')
turismodist.to_csv('~/DadesAirBNB/DistanciasTurismo.csv', index = False)