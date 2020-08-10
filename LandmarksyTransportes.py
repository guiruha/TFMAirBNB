#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:14:10 2020

@author: Guillem Rochina y Helena Saigí
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

clusters = ['Landmark_1', 'Landmark_2', 'Landmark_3', 'Landmark_4', 'Landmark_5', 'Landmark_6', 'Landmark_7',
            'Landmark_8', 'Landmark_9', 'Landmark_10', 'Landmark_11']
centroids = km.cluster_centers_.tolist() # Los clusters cambian no importa la semilla que le pongas
# Por ello los nombres de los clúster pueden no ser correctos.

centroids_km = pd.DataFrame({'cluster': clusters, 'centroids': centroids})

# GEODATAFRAME DE CENTROIDES (LANDMARKS)

centroids_km['Longitud'] = [centroids_km.centroids[i][0] for i in range(centroids_km.shape[0])]
centroids_km['Latitud'] = [centroids_km.centroids[i][1] for i in range(centroids_km.shape[0])]
landmarks = gpd.GeoDataFrame(centroids_km, geometry=gpd.points_from_xy(centroids_km.Longitud, centroids_km.Latitud), crs=bcn_df.crs)


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

def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6372800
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat1 - lat2)
   delta_lambda = np.radians(lon1 - lon2)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
   res = 2 * r * (np.arcsin(np.sqrt(a)))
   return np.round(res, 2)

for landlat, landlon, name in zip(landmarks.Latitud, landmarks.Longitud, landmarks.cluster):
    print(name)
    map_df['{}_distance'.format(name)] = [haversine_distance(landlat, landlon, listlat, listlon) \
                                          for listlat, listlon in zip(map_df.latitude, map_df.longitude)]

fig, ax = plt.subplots(1, 3, figsize = (30, 13))
map_df.sort_values(by = 'Landmark_1_distance').head(20).plot(ax = ax[0], marker = ".", markersize = 300, color = "maroon")
landmarks.plot(ax = ax[0],  marker = "*", markersize = 200, color = "gold")
map_df.sort_values(by = 'Landmark_5_distance').head(20).plot(ax = ax[1], marker = ".", markersize = 300, color = "maroon")
landmarks.plot(ax = ax[1],  marker = "*", markersize = 200, color = "gold")
map_df.sort_values(by = 'Landmark_10_distance').head(20).plot(ax = ax[2], marker = ".", markersize = 300, color = "maroon")
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

np.random.seed(207)
distance = 300
mapbuffer = map_df.copy()
mapbuffer['geometry'] = mapbuffer['geometry'].buffer(distance)

# CÁLCULO DE DISTANCIAS A PARADAS DE METRO

map_df['metros_cercanos'] = [sum(i.within(j) for i in metro.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANCIAS AL METRO

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'metros_cercanos', ascending = False).head(5000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", 
                        label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'metros_cercanos').head(5000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", 
                        label = 'Alojamientos con menor número de paradas cercanas')
metro.plot(ax = ax, color = "navy", label = 'Paradas de Metro')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Alojamiento con más paradas de metro cercanas', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# FGC

fgc = transport[transport['NOM_CAPA'] == 'Ferrocarrils Generalitat (FGC)']

fgc = gpd.GeoDataFrame(fgc, geometry = gpd.points_from_xy(fgc.LONGITUD, fgc.LATITUD), crs = bcn_df.crs)
fgc = fgc.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS FGC

map_df['fgc_cercanos'] = [sum(i.within(j) for i in fgc.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANAS A PARADAS DE FGC

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'fgc_cercanos', ascending = False).head(1000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'fgc_cercanos').head(1000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos con menor número de paradas cercanas')
fgc.plot(ax = ax, color = "navy", label = 'Paradas de Ferrocarriles (FGC)')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Alojamiento con más paradas de Ferrocarril cercanas', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# RENFE

renfe = transport[(transport['EQUIPAMENT'] == 'RENFE - DE FRANÇA-')|(transport['EQUIPAMENT'] == 'RENFE - SANTS ESTACIÓ-')]

renfe = gpd.GeoDataFrame(renfe, geometry = gpd.points_from_xy(renfe.LONGITUD, renfe.LATITUD), crs = bcn_df.crs)
renfe = renfe.to_crs(epsg = 3857)

# CÁLCULO DISTANCIAS A ESTACIONES DE RENFE
map_df['renfe_cercanos'] = [sum(i.within(j) for i in renfe.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANCIAS A PARADAS DE RENFE

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'renfe_cercanos', ascending=False).head(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'renfe_cercanos').head(1000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos con mayor número de paradas cercanas')
renfe.plot(ax = ax, color = "navy", label = 'Paradas de RENFE')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Alojamientos cercanos la estación de RENFE', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# TREN AEROPUERTO

trenaer = transport[transport['NOM_CAPA'] == "Tren a l'aeroport"]

trenaer = gpd.GeoDataFrame(trenaer, geometry = gpd.points_from_xy(trenaer.LONGITUD, trenaer.LATITUD), crs = bcn_df.crs)
trenaer = trenaer.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE TREN AEROPUERTO

map_df['trenaer_cercanos'] = [sum(i.within(j) for i in trenaer.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANCIAS AL TREN AL AEROPUERTO

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'trenaer_cercanos', ascending = False).head(500).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'trenaer_cercanos').head(300).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos con menor número de paradas cercanas')
trenaer.plot(ax = ax, color = "navy", label = 'Paradas de Tren al Aeropuerto')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Alojamientos con más estaciones cercanas de Tren al Aeropuerto', fontsize = 20)
ax.axis('off')
ctx.add_basemap(ax)
plt.show()

# TRAMVIA

tramvia = transport[transport['NOM_CAPA'] == 'Tramvia']

tramvia = gpd.GeoDataFrame(tramvia, geometry = gpd.points_from_xy(tramvia.LONGITUD, tramvia.LATITUD), crs = bcn_df.crs)
tramvia = tramvia.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE TRAMVIA

map_df['tranvia_cercanos'] = [sum(i.within(j) for i in tramvia.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANCIAS A PARADAS DE TRAMVIA

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'tranvia_cercanos', ascending=False).head(1000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'tranvia_cercanos').head(1000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos con menor número de paradas cercanas')
tramvia.plot(ax = ax, color = "navy", label = 'Paradas de Tranvía')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Alojamientos con más estaciones cercanas de Tranvía', fontsize = 20)
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

map_df['bus_cercanos'] = [sum(i.within(j) for i in bus.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANCIAS A PARADAS DE BUS

fig, ax = plt.subplots(1, 1, figsize = (15, 20))
map_df.sort_values(by = 'bus_cercanos', ascending=False).head(5000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'bus_cercanos').head(5000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos con menor número de paradas cercanas')
bus.plot(ax = ax, color = "navy", markersize = 15, label = 'Paradas de Bus')
plt.legend(fontsize = 20, loc = "lower right")
plt.title('Alojamientos con más estaciones cercanas de Autobús', fontsize = 20)
ctx.add_basemap(ax)
plt.show()

# AEROBUS

aerobus = transportsbus[transportsbus['NOM_CAPA'] == "Autobus a l'aeroport"]

aerobus = gpd.GeoDataFrame(aerobus, geometry = gpd.points_from_xy(aerobus.LONGITUD, aerobus.LATITUD), crs = bcn_df.crs)
aerobus = aerobus.to_crs(epsg = 3857)

# CÁLCULO DE DISTANCIAS A PARADAS DE BUS AL AEROPUERTO

map_df['aerobus_cercanos'] = [sum(i.within(j) for i in aerobus.geometry) for j in mapbuffer.geometry]

# MAPA DE DISTANCIAS A PARADAS DEL BUS AL AEROPUERTO

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
map_df.sort_values(by = 'aerobus_cercanos', ascending=False).head(100).head(5000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "green", label = 'Alojamientos con mayor número de paradas cercanas')
map_df.sort_values(by = 'aerobus_cercanos').head(5000).sample(100)\
                  .plot(ax = ax, marker = "X", markersize = 200, color = "maroon", label = 'Alojamientos con menor número de paradas cercanas')
aerobus.plot(ax = ax, color = "navy", label = 'Paradas de bus al aeropuerto')
plt.legend(fontsize = 20, loc = "upper left")
plt.title('Alojamientos con más estaciones cercanas de Autobús', fontsize = 20)
ctx.add_basemap(ax)
plt.show()

# Generamos el csv de las distancias para su posterior Join en el script de Exploración

cols_select = [x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'latitude','longitude', 'geometry']]

distances = map_df[cols_select]

distances.to_pickle('~/DadesAirBNB/Distancias.pkl')
distances.to_csv('~/DadesAirBNB/Distancias.csv', index = False)
