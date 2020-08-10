#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:57:53 2020

@author: guillem
"""


import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
plt.style.use('fivethirtyeight')

map_df = pd.read_pickle('~/DadesAirBNB/Localizaciones.pkl')

bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

map_df = gpd.GeoDataFrame(map_df, geometry = gpd.points_from_xy(map_df.longitude, map_df.latitude), crs = bcn_df.crs)

# LANDMARKS DATASET DE FLIKR

f = np.loadtxt("/home/guillem/DadesAirBNB/Flkr/Flickr_landmarks_geotags.txt", comments="#", delimiter=" ", unpack=False)

flickr = pd.DataFrame(f)

flickr.columns = ['Latitude', 'Longitude']

flickr = gpd.GeoDataFrame(flickr, geometry = gpd.points_from_xy(flickr.Longitude, flickr.Latitude), crs = bcn_df.crs)

# VISUALIZACIÓN GENERAL

flickr = flickr.to_crs(epsg=3857)
fig, ax = plt.subplots(1, 1, figsize = (15, 13))
flickr.plot(color = "Red", ax = ax)
ctx.add_basemap(ax)
plt.title('Landmarks más importantes de BCN', fontsize = 20)
ax.axis('off')
plt.tight_layout()

# DENDROGRAMA SINGLE Y EUCLÍDEA

fig, ax = plt.subplots(1, 1, figsize=(15,15))
dendrogram = sch.dendrogram(sch.linkage(flickr[['Latitude', 'Longitude']].values, 
                            method='single', metric='euclidean'))
_= ax.set_title('Dendrograma de distancias con método single y distancia euclídea', fontsize = 20)
_= ax.set_ylabel('Distancia (en Miles de Metros)')
_= ax.set_xticks([])

# DENDROGRAMA SINGLE Y MANHATTAN

fig, ax = plt.subplots(1, 1, figsize=(15,15))
dendrogram = sch.dendrogram(sch.linkage(flickr[['Latitude', 'Longitude']].values, 
                            method='single', metric='cityblock'))
_= ax.set_title('Dendrograma de distancias con método single y distancia de manhattan', fontsize = 20)
_= ax.set_ylabel('Distancia (en Miles de Metros)')
_= ax.set_xticks([])

# DENDROGRAMA AVERAGE Y MANHATTAN
   
fig, ax = plt.subplots(1, 1, figsize=(15,15))
dendrogram = sch.dendrogram(sch.linkage(flickr[['Latitude', 'Longitude']].values, 
                            method='average', metric='cityblock'))
_= ax.set_title('Dendrograma de distancias con método de la media y distancia de manhattan', fontsize = 20)
_= ax.set_ylabel('Distancia (en Miles de Metros)')
_= ax.set_xticks([])

# PRIMER CLUSTERING AGLOMERATIVO

np.random.seed(1997)
hc = AgglomerativeClustering(n_clusters=12, affinity='cityblock', linkage='single')
clusters = hc.fit_predict(flickr[['Longitude', 'Latitude']])

print(clusters[:10])

flickr['clusters_hc'] = hc.fit_predict(flickr[['Longitude', 'Latitude']])
flickr['clusters_hc'] = flickr['clusters_hc'].astype('category')

# MAPAS DE CLUSTERS CON CLUSTERING JERÁRQUICO

ax = flickr.plot(column = 'clusters_hc', cmap='rainbow', legend = True, figsize = (15, 13), 
             categorical=True, edgecolor='black', markersize=500)
ctx.add_basemap(ax)
plt.title('CLUSTERIZACIÓN DE LOS LANDMARKS A TRAVÉS DE CLUSTERING JERÁRQUICO', fontsize = 20)
ax.axis('off')
plt.tight_layout()

# CLUSTERS QUE SOBRAN

col_clust = flickr.groupby('clusters_hc')['Latitude'].count()[flickr.groupby('clusters_hc')['Latitude'].count() < 10].index.tolist()
col_clust

# ELIMINACIÓN DE LOS CLÚSTERS QUE SOBRAN
flickr.drop([i for x, i in zip(flickr['clusters_hc'], flickr.index) if x in col_clust], axis = 0, inplace = True)
flickr['clusters_hc'].value_counts()

# DENDROGRAMA WARD Y EUCLÍDEA

fig, ax = plt.subplots(1, 1, figsize=(15,15))
dendrogram = sch.dendrogram(sch.linkage(flickr[['Latitude', 'Longitude']].values, 
                            method='ward', metric='euclidean'))
_= ax.set_title('Dendrograma de distancias con método del ward y distancia de euclidea', fontsize = 20)
_= ax.set_ylabel('Distancia (en Miles de Metros)')
_= ax.set_xticks([])

# CLUSTERING JERÁRQUICO FINAL

hc = AgglomerativeClustering(n_clusters=11, affinity='euclidean', linkage='ward')
flickr['clusters_hc'] = hc.fit_predict(flickr[['Longitude', 'Latitude']])
flickr['clusters_hc'] = flickr['clusters_hc'].astype('category')

# MAPA CLUSTERING JERÁRQUICO FINAL

ax = flickr.plot(column = 'clusters_hc', cmap='rainbow', legend = True, figsize = (15, 13), 
             categorical=True, edgecolor='black', markersize=500)
ctx.add_basemap(ax)
plt.title('Clusterización de los landmarks a través del Clustering Jerárquico', fontsize = 20)
ax.axis('off')
plt.tight_layout()

# K-MEANS

km = KMeans()
km.get_params()
n_clusters = list(range(1, 20))

# MÉTODO DEL CODO (IRRELEVANTE PARA NUESTRO CASO)

wcss = []
for cluster in n_clusters:
    km = KMeans(n_clusters=cluster)
    km.fit(flickr[['Longitude', 'Latitude']])
    wcss.append(km.inertia_)

ax, fig = plt.subplots(1, 1, figsize=(10,7))
plt.plot(n_clusters,wcss)
plt.xticks(n_clusters)
plt.xlabel('Numero de Clusters')
plt.ylabel('WCSS')
plt.title('Método del Codo')
plt.show()

# K-MEANS FINAL

km = KMeans(n_clusters=11, random_state = 1997)
flickr['clusters_km'] = km.fit_predict(flickr[['Longitude', 'Latitude']])

# MAPA KMEANS FINAL

ax = flickr.plot(column = 'clusters_km', cmap='rainbow', legend = True, figsize = (15, 13), 
             categorical=True, edgecolor='black', markersize=500)
ctx.add_basemap(ax)
plt.title('CLUSTERIZACIÓN DE LOS LANDMARKS A TRAVÉS DE K-MEANS', fontsize = 20)
ax.axis('off')
plt.tight_layout()

# CENTROIDES

print('Los centroides de los clusters ajustados se encuentran en las coordenadas:\n {}'.format(km.cluster_centers_))
print('\nCon Longitudes de: \n{}'.format(km.cluster_centers_[:,0]))
print('\nY Latitudes de: \n{}'.format(km.cluster_centers_[:,1]))

# MAPA DE CENTROIDES Y CLUSTERS

flickr = flickr.to_crs(epsg=4326)
bcn_df = bcn_df.to_crs(epsg=4326)
ax = bcn_df.plot(figsize=(25,13))
flickr.plot(column = 'clusters_km', cmap='tab20b', legend = True, figsize = (20, 20), categorical=True, edgecolor='black', markersize=500, ax=ax)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='*', s=700, color='yellow', edgecolor='black')
ax.set_xlim([2.12, 2.23])
ax.set_ylim([41.36, 41.43])
plt.title('Visualización de clusters y sus centroides', fontsize = 20)
plt.xlabel('LATITUD')
plt.ylabel('LONGITUD')
ax.axis('off')
plt.tight_layout()

# CREACIÓN DE DATAFRAME DE CENTROIDES

clusters = ['Catedral de Barcelona', 'Parc Güell', 'Sagrada Familia', 'Montjuic', 'Pg. de Gràcia', 'Vila Olimpica',
            'Colon', 'Arc de Triomf', 'Glories', 'Hospital de Sant Pau', 'Pl. Catalunya']
centroids = km.cluster_centers_.tolist() # Los clusters cambian no importa la semilla que le pongas
# Por ello los nombres de los clúster pueden no ser correctos.

centroids_km = pd.DataFrame({'cluster': clusters, 'centroids': centroids})

# GEODATAFRAME DE CENTROIDES (LANDMARKS)

centroids_km['Longitud'] = [centroids_km.centroids[i][0] for i in range(centroids_km.shape[0])]
centroids_km['Latitud'] = [centroids_km.centroids[i][1] for i in range(centroids_km.shape[0])]
landmarks = gpd.GeoDataFrame(centroids_km, geometry=gpd.points_from_xy(centroids_km.Longitud, centroids_km.Latitud), crs=bcn_df.crs)
landmarks = landmarks[['cluster', 'geometry']]

landmarks = landmarks.to_crs(epsg=3857)
map_df = map_df.to_crs(epsg=3857)

# MAPA DE LANDMARKS Y CENTROIDES
ax = map_df.plot(marker = "X", markersize = 10, color = "navy",  figsize = (25, 13))
landmarks.plot(column = 'cluster', cmap = 'rainbow', legend = True, 
             categorical=True, edgecolor='black', marker = '*', markersize=1000, ax = ax)
ctx.add_basemap(ax)
plt.title('Visualización de los centroides y los alojamientos', fontsize = 20)
ax.axis('off')
plt.tight_layout()

# CREACIÓN DE LAS COLUMNAS DE DISTANCIAS

for i, landmark in enumerate(landmarks.cluster):
    map_df['{}_distance'.format(landmark)] = [j.distance(landmarks.geometry[i]) for j in map_df.geometry]

fig, ax = plt.subplots(1, 3, figsize = (30, 13))
map_df.sort_values(by = 'Sagrada Familia_distance').head(20).plot(ax = ax[0], marker = ".", markersize = 300, color = "maroon")
landmarks.plot(ax = ax[0],  marker = "*", markersize = 200, color = "gold")
map_df.sort_values(by = 'Colon_distance').head(20).plot(ax = ax[1], marker = ".", markersize = 300, color = "maroon")
landmarks.plot(ax = ax[1],  marker = "*", markersize = 200, color = "gold")
map_df.sort_values(by = 'Arc de Triomf_distance').head(20).plot(ax = ax[2], marker = ".", markersize = 300, color = "maroon")
landmarks.plot(ax = ax[2],  marker = "*", markersize = 200, color = "gold")
ctx.add_basemap(ax[0])
ctx.add_basemap(ax[1])
ctx.add_basemap(ax[2])
ax[0].set_title('Alojamientos más cercanos a Sagrada Familia')
ax[1].set_title('Alojamientos más cercanos a Estatua de Colón')
ax[2].set_title('Alojamientos más cercanos a Arc del Triomf')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
plt.show()

# DATASETS DE TRANSPORTES

transport = pd.read_pickle("~/DadesAirBNB/Transports/METRO.pkl")
transport = transport[['NOM_CAPA', 'LONGITUD', 'LATITUD', 'EQUIPAMENT', 'NOM_BARRI']]

# METRO
transport['NOM_CAPA'].value_counts()
metro = transport[transport['NOM_CAPA'] == 'Metro i línies urbanes FGC']

metro = gpd.GeoDataFrame(metro, geometry = gpd.points_from_xy(metro.LONGITUD, metro.LATITUD), crs = bcn_df.crs)

metro = metro.to_crs(epsg = 3857)
map_df = map_df.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE METRO

map_df['dist_metro'] = map_df['geometry'].apply(lambda x: min(x.distance(j) for j in metro.geometry))

# MAPA DE DISTANCIAS AL METRO

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_metro').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos cercanos al metro')
map_df.sort_values(by = 'dist_metro', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos al metro')
metro.plot(ax = ax, color = "navy", label = 'Paradas de Metro')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Cercanía a paradas de metro', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# FGC

fgc = transport[transport['NOM_CAPA'] == 'Ferrocarrils Generalitat (FGC)']

fgc = gpd.GeoDataFrame(fgc, geometry = gpd.points_from_xy(fgc.LONGITUD, fgc.LATITUD), crs = bcn_df.crs)
fgc = fgc.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS FGC

map_df['dist_fgc'] = map_df['geometry'].apply(lambda x: min(x.distance(j) for j in fgc.geometry))

map_df[map_df.index == map_df['dist_fgc'].idxmax()][['geometry', 'dist_fgc']]

# MAPA DE DISTANAS A PARADAS DE FGC

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_fgc').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos lejanos a los Ferrocarriles')
map_df.sort_values(by = 'dist_fgc', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos a los Ferrocarriles')
fgc.plot(ax = ax, color = "navy", label = 'Paradas de Ferrocarriles (FGC)')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Cercanía a paradas de Ferrocarril', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# RENFE

renfe = transport[(transport['EQUIPAMENT'] == 'RENFE - DE FRANÇA-')|(transport['EQUIPAMENT'] == 'RENFE - SANTS ESTACIÓ-')]

renfe = gpd.GeoDataFrame(renfe, geometry = gpd.points_from_xy(renfe.LONGITUD, renfe.LATITUD), crs = bcn_df.crs)
renfe = renfe.to_crs(epsg = 3857)

# CÁLCULO DISTANCIAS A ESTACIONES DE RENFE
map_df['dist_renfe'] = [min(i.distance(j) for j in renfe.geometry) for i in map_df.geometry]

map_df[map_df.index == map_df['dist_renfe'].idxmax()][['geometry', 'dist_renfe']]

# MAPA DE DISTANCIAS A PARADAS DE RENFE

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_renfe').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos cercanos a RENFE')
map_df.sort_values(by = 'dist_renfe', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos a RENFE')
renfe.plot(ax = ax, color = "navy", label = 'Paradas de RENFE')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Cercanía a paradas de RENFE', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# TREN AEROPUERTO

trenaer = transport[transport['NOM_CAPA'] == "Tren a l'aeroport"]

trenaer = gpd.GeoDataFrame(trenaer, geometry = gpd.points_from_xy(trenaer.LONGITUD, trenaer.LATITUD), crs = bcn_df.crs)
trenaer = trenaer.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE TREN AEROPUERTO

map_df['dist_trenaeropuerto'] = map_df['geometry'].apply(lambda x: min(x.distance(j) for j in trenaer.geometry))

# MAPA DE DISTANCIAS AL TREN AL AEROPUERTO

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_trenaeropuerto').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos cercanos al Tren del Aeropuerto')
map_df.sort_values(by = 'dist_trenaeropuerto', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos al Tren del Aeropuerto')
trenaer.plot(ax = ax, color = "navy", label = 'Paradas de Tren al Aeropuerto')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Cercanía a paradas del Tren al Aeropuerto', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# TRAMVIA

tramvia = transport[transport['NOM_CAPA'] == 'Tramvia']

tramvia = gpd.GeoDataFrame(tramvia, geometry = gpd.points_from_xy(tramvia.LONGITUD, tramvia.LATITUD), crs = bcn_df.crs)
tramvia = tramvia.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE TRAMVIA

map_df['dist_tramvia'] = map_df['geometry'].apply(lambda x: min(x.distance(j) for j in tramvia.geometry))

# MAPA DE DISTANCIAS A PARADAS DE TRAMVIA

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_tramvia').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos cercanos al Tramvia')
map_df.sort_values(by = 'dist_tramvia', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos al Tramvia')
tramvia.plot(ax = ax, color = "navy", label = 'Paradas de Tramvia')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Cercanía a paradas de Tramvia', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# DATASET AUTOBUSOS

transportsbus = pd.read_pickle("~/DadesAirBNB/Transports/ESTACIONS_BUS.pkl")
transportsbus = transportsbus[['NOM_CAPA', 'LONGITUD', 'LATITUD', 'EQUIPAMENT', 'NOM_BARRI']]


# BUS DIURN

bus = transportsbus[transportsbus['NOM_CAPA'] == 'Autobusos diürns']

bus = gpd.GeoDataFrame(bus, geometry = gpd.points_from_xy(bus.LONGITUD, bus.LATITUD), crs = bcn_df.crs)
bus = bus.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE BUS

map_df['dist_bus'] = map_df['geometry'].apply(lambda x: min(x.distance(j) for j in bus.geometry))

# MAPA DE DISTANCIAS A PARADAS DE BUS

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_bus').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos cercanos al Bus')
map_df.sort_values(by = 'dist_bus', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos al Bus')
bus.plot(ax = ax, color = "navy", markersize = 15, label = 'Paradas de Bus')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Cercanía a paradas de Autobús', fontsize = 20)
ctx.add_basemap(ax)
plt.show()

# AEROBUS

aerobus = transportsbus[transportsbus['NOM_CAPA'] == "Autobus a l'aeroport"]

aerobus = gpd.GeoDataFrame(aerobus, geometry = gpd.points_from_xy(aerobus.LONGITUD, aerobus.LATITUD), crs = bcn_df.crs)
aerobus = aerobus.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE BUS AL AEROPUERTO

map_df['dist_aerobus'] = map_df['geometry'].apply(lambda x: min(x.distance(j) for j in aerobus.geometry))

# MAPA DE DISTANCIAS A PARADAS DEL BUS AL AEROPUERTO

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'dist_aerobus').head(100).plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos cercanos al Bus del Aeropuerto')
map_df.sort_values(by = 'dist_aerobus', ascending=False).head(100).plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos lejanos al Bus del Aeropuerto')
aerobus.plot(ax = ax, color = "navy", label = 'Paradas de bus al aeropuerto')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Cercanía a paradas de Bus al Aeropuerto', fontsize = 20)
ctx.add_basemap(ax)
plt.show()

# Generamos el csv de las distancias para su posterior Join en el script de Exploración

cols_select = [x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'latitude','longitude', 'geometry']]

distances = map_df[cols_select]

distances.to_pickle('~/DadesAirBNB/Distancias.pkl')
distances.to_csv('~/DadesAirBNB/Distancias.csv', index = False)
