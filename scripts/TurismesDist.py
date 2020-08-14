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

print('\nImportamos los datos geoespaciales y los preparamos')
bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")
map_df = pd.read_pickle('~/DadesAirBNB/Localizaciones.pkl')
map_df = gpd.GeoDataFrame(map_df, geometry = gpd.points_from_xy(map_df.longitude, map_df.latitude), crs = bcn_df.crs)
map_df = map_df.to_crs(epsg = 3857)

print('\nCreamos los buffers con los que trabajaremos\n\n')
distance = 600
mapbuffer = map_df.copy()
mapbuffer['geometry'] = mapbuffer['geometry'].buffer(distance)

print('\nCalculamos museos dentro de cada buffer\n')
mus_bib = pd.read_pickle("~/DadesAirBNB/Turisme/C001_Biblioteques_i_museus.pkl")
mus_bib = mus_bib[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]
mus_bib = mus_bib.drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]

museos = mus_bib[mus_bib.EQUIPAMENT.str.contains('Museu')]

museos = gpd.GeoDataFrame(museos, geometry = gpd.points_from_xy(museos.LONGITUD, museos.LATITUD), crs = bcn_df.crs)

museos = museos.to_crs(epsg=3857)

map_df['museos_cercanos'] = [sum(i.within(j) for i in museos.geometry) for j in mapbuffer.geometry]


print('\nCalculamos teatros dentro de cada buffer\n')

visual = pd.read_pickle("~/DadesAirBNB/Turisme/C002_Cinemes_teatres_auditoris.pkl")

visual['LATITUD'] = visual['LATITUD'].astype('str').apply(lambda x: x[:2]+'.'+ x[2:]).astype('float')
visual['LONGITUD'] = visual['LONGITUD'].astype('str').apply(lambda x: x[0]+'.'+x[1:]).astype('float')

visual = visual.drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'SECCIO','NUM_BARRI', 'LONGITUD', 'LATITUD']]

teatro = visual[visual.SECCIO.str.contains('Teatre')]

teatro = gpd.GeoDataFrame(teatro, geometry = gpd.points_from_xy(teatro.LONGITUD, teatro.LATITUD), crs = bcn_df.crs)

teatro = teatro.to_crs(epsg=3857)

map_df['teatros_cercanos'] = [sum(i.within(j) for i in teatro.geometry) for j in mapbuffer.geometry]

print('\nCalculamos cines dentro de cada buffer\n')
cine = visual[visual.SECCIO.str.contains('Cinema')]

cine = gpd.GeoDataFrame(cine, geometry = gpd.points_from_xy(cine.LONGITUD, cine.LATITUD), crs = bcn_df.crs)
cine = cine.to_crs(epsg = 3857)

map_df['cines_cercanos'] = [sum(i.within(j) for i in cine.geometry) for j in mapbuffer.geometry]

print('\nCalculamos auditorios dentro de cada buffer\n')
auditorio = visual[visual.SECCIO.str.contains('Auditori')]

auditorio = gpd.GeoDataFrame(auditorio, geometry = gpd.points_from_xy(auditorio.LONGITUD, auditorio.LATITUD), crs = bcn_df.crs)
auditorio = auditorio.to_crs(epsg = 3857)

map_df['auditorios_cercanos'] = [sum(i.within(j) for i in auditorio.geometry) for j in mapbuffer.geometry]

print('\nCalculamos restaurantes dentro de cada buffer\n')
restaurantes = pd.read_pickle("~/DadesAirBNB/Turisme/H001_Restaurants.pkl")

restaurantes = restaurantes.drop_duplicates(subset = ['EQUIPAMENT'])

restaurantes['NOM'] = (restaurantes['EQUIPAMENT'] + ' - ' + restaurantes['SECCIO']).str.replace(' - #','') 

rest = restaurantes[['NOM', 'LATITUD', 'LONGITUD', 'NUM_BARRI']]

rest = gpd.GeoDataFrame(rest, geometry = gpd.points_from_xy(rest.LONGITUD, rest.LATITUD), crs = bcn_df.crs)
rest = rest.to_crs(epsg = 3857)

map_df['restaurantes_cercanos'] = [sum(i.within(j) for i in rest.geometry) for j in mapbuffer.geometry]

print('\nCalculamos salas de conciertos/discotecas dentro de cada buffer\n')
musica = pd.read_pickle("~/DadesAirBNB/Turisme/C004_Espais_de_musica_i_copes.pkl")

musica = musica.drop_duplicates(subset = ['EQUIPAMENT'])

musica = musica[['EQUIPAMENT', 'LATITUD', 'LONGITUD', 'NUM_BARRI']]

musica = gpd.GeoDataFrame(musica, geometry = gpd.points_from_xy(musica.LONGITUD, musica.LATITUD), crs = bcn_df.crs)
musica = musica.to_crs(epsg = 3857)

map_df['musica_cercanos'] = [sum(i.within(j) for i in musica.geometry) for j in mapbuffer.geometry]

turismodist = map_df[[x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'geometry', 'latitude', 'longitude']]]

print('\n Generamos el dataset de Turimos\n')
turismodist.to_pickle('~/DadesAirBNB/DistanciasTurismo.pkl')
turismodist.to_csv('~/DadesAirBNB/DistanciasTurismo.csv', index = False)


print('\nEjecución finalizada con éxito!')