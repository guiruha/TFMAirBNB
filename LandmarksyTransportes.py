#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:14:10 2020

@author: guillem
"""


import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Point
import contextily as ctx

df = pd.read_csv('~/DadesAirBNB/DatosModelar.csv')

bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

map_df = df.reset_index().drop_duplicates(subset = ['id'])[['id', 'neighbourhood_group_cleansed', 'latitude', 'longitude']]

map_df['geometry'] = map_df.apply(lambda x: Point(x.longitude, x.latitude), axis = 1)

map_df = gpd.GeoDataFrame(map_df, geometry = map_df.geometry, crs = bcn_df.crs)

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