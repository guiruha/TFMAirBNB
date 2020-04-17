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

mus_bib = pd.read_csv("~/DadesAirBNB/Turisme/C001_Biblioteques_i_museus.csv")

dfmus_bib = mus_bib[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]

dfmus_bib = mus_bib.reset_index().drop_duplicates(subset = ['EQUIPAMENT'])[['EQUIPAMENT', 'NUM_BARRI','LONGITUD', 'LATITUD']]

museos = dfmus_bib[dfmus_bib.EQUIPAMENT.str.contains('Museu')]

museos = gpd.GeoDataFrame(museos, geometry = gpd.points_from_xy(museos.LONGITUD, museos.LATITUD), crs = bcn_df.crs)

museos = museos.to_crs(epsg=3857)

df = pd.read_csv('~/DadesAirBNB/DatosGeneral.csv')

map_df = df[['id', 'latitude', 'longitude']]

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

