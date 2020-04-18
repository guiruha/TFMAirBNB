#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:14:10 2020

@author: guillem & helena
"""
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

map_df = pd.read_csv('~/DadesAirBNB/Localizaciones.csv')

bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

map_df['geometry'] = map_df.apply(lambda x: Point(x.longitude, x.latitude), axis = 1)

map_df = gpd.GeoDataFrame(map_df, geometry = gpd.points_from_xy(map_df.longitude, map_df.latitude), crs = bcn_df.crs)

# LANDMARKS

f = loadtxt("/home/guillem/DadesAirBNB/Flkr/Flickr_landmarks_geotags.txt", comments="#", delimiter=" ", unpack=False)

dff = pd.DataFrame(f)

dff

dff.columns = ['Latitude', 'Longitude']

dff.head()

## ??? dff = dff.reset_index()[['Latitude', 'Longitude']]

dff['geometry'] = dff.apply(lambda x: Point(x.Longitude, x.Latitude), axis = 1)

dff = gpd.GeoDataFrame(dff, geometry = gpd.points_from_xy(dff.Longitude, dff.Latitude), crs = bcn_df.crs)

plt.scatter(dff['Latitude'], dff['Longitude'])
plt.show()

dff = dff.to_crs(epsg=3857)
ax = dff.plot(cmap = "YlOrRd", legend = True, figsize = (20, 20), alpha = 0.7, scheme = 'maximumbreaks')
ctx.add_basemap(ax)
plt.show()

ax = dff.plot(cmap = "YlOrRd", legend = True, figsize = (20, 20), alpha = 0.7, scheme = 'maximumbreaks')
ctx.add_basemap(ax)
plt.show()

import scipy
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from collections import namedtuple

# ESTA PART NO FA FALTA
#Point = namedtuple("Point", "x y")

#dff.geometry.head()
#dff.geometry.x #Longitude
#dff.geometry.y #Latitude

#dff['Latitude_metros'] = dff.geometry.y
#dff['Longitude_metros'] = dff.geometry.x
#dff.head()

ax, fig = plt.subplots(1, 1, figsize=(15,10))
dendrogram = sch.dendrogram(sch.linkage(dff[['Latitude', 'Longitude']].values, method='complete', metric='euclidean'))

ax, fig = plt.subplots(1, 1, figsize=(15,10))
dendrogram = sch.dendrogram(sch.linkage(dff[['Latitude', 'Longitude']].values, method='complete', metric='cityblock'))

ax, fig = plt.subplots(1, 1, figsize=(15,10))
dendrogram = sch.dendrogram(sch.linkage(dff[['Latitude', 'Longitude']].values, method='average', metric='cityblock'))

hc = AgglomerativeClustering(n_clusters=11, affinity='cityblock', linkage='single')

clusters = hc.fit_predict(dff[['Longitude', 'Latitude']])

print(clusters[:10])

dff['clusters'] = hc.fit_predict(dff[['Longitude', 'Latitude']])

dff['clusters'] = dff['clusters'].astype('category')

dff.clusters.dtypes

dff = dff.to_crs(epsg=3857)
ax = dff.plot(column = 'clusters', cmap='Set3', legend = True, figsize = (20, 20), categorical=True, markersize = 400)
ctx.add_basemap(ax)
plt.show()

dff.groupby('clusters')['Latitude'].count()

col_clust = dff.groupby('clusters')['Latitude'].count()[dff.groupby('clusters')['Latitude'].count() > 5].index.tolist()

dff = dff.iloc[[i for i, x in zip(dff.index, dff['clusters']) if x in col_clust]]

dff.groupby('clusters')['Latitude'].count()

# Hierarchical clustering

dff = dff.to_crs(epsg = 4326)
dff['clusters'] = hc.fit_predict(dff[['Longitude', 'Latitude']])

dff = dff.to_crs(epsg = 3857)
ax = dff.plot(column = 'clusters', cmap='Set3', legend = True, figsize = (20, 20), categorical=True, markersize = 400)
ctx.add_basemap(ax)
plt.show()

dff = dff.to_crs(epsg = 4326)
hc = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')

dff['clusters_hc'] = hc.fit_predict(dff[['Longitude', 'Latitude']])

dff = dff.to_crs(epsg = 3857)
ax = dff.plot(column = 'clusters_hc', cmap='tab20b', legend = True, figsize = (20, 20), categorical=True, edgecolor='black', markersize=100)
ctx.add_basemap(ax)
plt.show()

col_clust = dff.groupby('clusters')['Latitude'].count()[dff.groupby('clusters')['Latitude'].count() < 5].index.tolist()

dff.drop([i for x, i in zip(dff['clusters'], dff.index) if x in col_clust], axis = 0, inplace = True)

dff['clusters'].value_counts()

np.random.seed(1997)
hc = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')

dff['clusters_hc'] = hc.fit_predict(dff[['Longitude', 'Latitude']])

dff = dff.to_crs(epsg = 3857)
ax = dff.plot(column = 'clusters_hc', cmap='tab20b', legend = True, figsize = (20, 20), categorical=True, edgecolor='black', markersize=100)
ctx.add_basemap(ax)
plt.show()


# K-MEANS

from sklearn.cluster import KMeans

km = KMeans()
km.get_params()
n_clusters = list(range(1, 20))

wcss = []
for cluster in n_clusters:
    km = KMeans(n_clusters=cluster)
    km.fit(dff[['Longitude', 'Latitude']])
    wcss.append(km.inertia_)

ax, fig = plt.subplots(1, 1, figsize=(10,7))
plt.plot(n_clusters,wcss)
plt.xticks(n_clusters)
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.title('Elbow Curve')
plt.show()

km = KMeans(n_clusters=12, random_state = 1997)

dff['clusters_km'] = km.fit_predict(dff[['Longitude', 'Latitude']])

dff[['clusters_hc', 'clusters_km']]

dff['clusters_hc'].value_counts()

dff['clusters_km'].value_counts()

km.cluster_centers_

km.cluster_centers_[:,0]
km.cluster_centers_[:,1]

dff = dff.to_crs(epsg=4326)
bcn_df = bcn_df.to_crs(epsg=4326)

ax = bcn_df.plot(figsize=(25,13))
dff.plot(column = 'clusters_km', cmap='tab20b', legend = True, figsize = (20, 20), categorical=True, edgecolor='black', markersize=100, ax=ax)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='*', s=300, color='yellow', edgecolor='black')
plt.tight_layout()

km.cluster_centers_

clusters = [ 'Arc de Triomf', 'Montjuic', 'Jardinets de Gràcia', 'Parc Guell', 'Sagrada Familia', 'Colon', 
            'Vila Olimpica', 'Pl. Catalunya', 'Catedral de Barcelona', 'Glories', 'Hospital de Sant Pau', 'Pg. de Gràcia']

centroids = km.cluster_centers_.tolist()

centroids_km = pd.DataFrame({'cluster': clusters, 'centroids': centroids})

centroids_km

centroids_km['Longitud'] = [centroids_km.centroids[i][0] for i in range(centroids_km.shape[0])]

centroids_km['Latitud'] = [centroids_km.centroids[i][1] for i in range(centroids_km.shape[0])]

centroids_km['geometry'] = centroids_km.apply(lambda x: Point(x.Longitud, x.Latitud), axis = 1)

landmarks = gpd.GeoDataFrame(centroids_km, geometry=gpd.points_from_xy(centroids_km.Longitud, centroids_km.Latitud), crs=bcn_df.crs)

landmarks = landmarks[['cluster', 'geometry']]

landmarks = landmarks.to_crs(epsg=3857)
map_df = map_df.to_crs(epsg=3857)

for i, landmark in enumerate(landmarks.cluster):
    map_df['{}_distance'.format(landmark)] = [j.distance(landmarks.geometry[i]) for j in map_df.geometry]

# DATASET DELS TRANSPORTS

transport = pd.read_csv("~/DadesAirBNB/Transports/METRO.csv")

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

transportsbus = pd.read_csv("~/DadesAirBNB/Transports/ESTACIONS_BUS.csv")

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

# Generamos el csv de las distancias para su posterior Join en el script de Exploración

cols_select = [x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'latitude','longitude', 'geometry']]

distances = map_df[cols_select]

distances.to_csv('~/DadesAirBNB/Distancias.csv')
