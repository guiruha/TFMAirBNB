#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:37:59 2020

@author: Guillem Rochina y Helena Saigí
"""


import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import scipy
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
plt.style.use('fivethirtyeight')

print('\nImportamos los datos geoespaciales y los preparamos para el  análisis')

map_df = pd.read_csv('~/DadesAirBNB/Localizaciones.csv')
bcn_df = gpd.read_file("/home/guillem/DadesAirBNB/neighbourhoods.geojson")

map_df = gpd.GeoDataFrame(map_df, geometry = gpd.points_from_xy(map_df.longitude, map_df.latitude), crs = bcn_df.crs)

f = np.loadtxt("/home/guillem/DadesAirBNB/Flkr/Flickr_landmarks_geotags.txt", comments="#", delimiter=" ", unpack=False)

flickr = pd.DataFrame(f)

flickr.columns = ['Latitude', 'Longitude']
flickr = gpd.GeoDataFrame(flickr, geometry = gpd.points_from_xy(flickr.Longitude, flickr.Latitude), crs = bcn_df.crs)

print('\nRealizamos Clustering Aglomerativo para detectar y eliminar outliers')
np.random.seed(1997)
hc = AgglomerativeClustering(n_clusters=12, affinity='cityblock', linkage='single')
clusters = hc.fit_predict(flickr[['Longitude', 'Latitude']])

flickr['clusters_hc'] = hc.fit_predict(flickr[['Longitude', 'Latitude']])
flickr['clusters_hc'] = flickr['clusters_hc'].astype('category')

col_clust = flickr.groupby('clusters_hc')['Latitude'].count()[flickr.groupby('clusters_hc')['Latitude'].count() < 10].index.tolist()

flickr.drop([i for x, i in zip(flickr['clusters_hc'], flickr.index) if x in col_clust], axis = 0, inplace = True)
flickr['clusters_hc'].value_counts()

hc = AgglomerativeClustering(n_clusters=12, affinity='euclidean', linkage='ward')
flickr['clusters_hc'] = hc.fit_predict(flickr[['Longitude', 'Latitude']])
flickr['clusters_hc'] = flickr['clusters_hc'].astype('category')

print('\nRealizamos K-Means para obtener los centroides')
km = KMeans(n_clusters=11, random_state = 1997)
flickr['clusters_km'] = km.fit_predict(flickr[['Longitude', 'Latitude']])

print('\nUna vez obtenidos los centroides pasamos las coordenadas a un dataframe')
clusters = ['Landmark_1', 'Landmark_2', 'Landmark_3', 'Landmark_4', 'Landmark_5', 'Landmark_6', 'Landmark_7',
            'Landmark_8', 'Landmark_9', 'Landmark_10', 'Landmark_11']
centroids = km.cluster_centers_.tolist()

centroids_km = pd.DataFrame({'cluster': clusters, 'centroids': centroids})
centroids_km['Longitud'] = [centroids_km.centroids[i][0] for i in range(centroids_km.shape[0])]
centroids_km['Latitud'] = [centroids_km.centroids[i][1] for i in range(centroids_km.shape[0])]

print('\nFinalmente generamos el dataset de landmarks a partir del cual calculamos distanicas')
landmarks = gpd.GeoDataFrame(centroids_km, geometry=gpd.points_from_xy(centroids_km.Longitud, centroids_km.Latitud), crs=bcn_df.crs)
landmarks = landmarks[['cluster', 'geometry', 'Latitud', 'Longitud']]

print('\nCalculamos la haversine distance entre landmarks y alojamientos')
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
        

map_df.columns = [x for x in map_df.columns if 'distance' not in x] + ['Landmark_{}_distance'.format(x) for x in range(1,12)]

print('\nTrabajamos con los datos de transporte')

transport = pd.read_csv('~/DadesAirBNB/Transports/METRO.csv')

transport = transport[['NOM_CAPA', 'LONGITUD', 'LATITUD', 'EQUIPAMENT', 'NOM_BARRI']]

transportsbus = pd.read_csv('~/DadesAirBNB/Transports/ESTACIONS_BUS.csv')

print('\nCalculamos distancias a paradas cercanas')

print('fgc')
fgc = transport[transport['NOM_CAPA'] == 'Ferrocarrils Generalitat (FGC)']
fgc = gpd.GeoDataFrame(fgc, geometry = gpd.points_from_xy(fgc.LONGITUD, fgc.LATITUD), crs = bcn_df.crs)
map_df['fgc_distance'] = [min(haversine_distance(tlat, tlon, listlat, listlon) for tlat,tlon in zip(fgc.LATITUD, fgc.LONGITUD))\
                          for listlat, listlon in zip(map_df.latitude, map_df.longitude)]

print('renfe')
renfe = transport[(transport['EQUIPAMENT'] == 'RENFE - DE FRANÇA-')|(transport['EQUIPAMENT'] == 'RENFE - SANTS ESTACIÓ-')]
renfe = gpd.GeoDataFrame(renfe, geometry = gpd.points_from_xy(renfe.LONGITUD, renfe.LATITUD), crs = bcn_df.crs)
map_df['renfe_distance'] = [min(haversine_distance(tlat, tlon, listlat, listlon) for tlat,tlon in zip(renfe.LATITUD, renfe.LONGITUD))\
                          for listlat, listlon in zip(map_df.latitude, map_df.longitude)]

print('tren aeropuerto')    
trenaer = transport[transport['NOM_CAPA'] == "Tren a l'aeroport"]
trenaer = gpd.GeoDataFrame(trenaer, geometry = gpd.points_from_xy(trenaer.LONGITUD, trenaer.LATITUD), crs = bcn_df.crs)
map_df['trenaer_distance'] = [min(haversine_distance(tlat, tlon, listlat, listlon) for tlat,tlon in zip(trenaer.LATITUD, trenaer.LONGITUD))\
                          for listlat, listlon in zip(map_df.latitude, map_df.longitude)]

print('bus aeropuerto')
aerobus = transportsbus[transportsbus['NOM_CAPA'] == "Autobus a l'aeroport"]
aerobus = gpd.GeoDataFrame(aerobus, geometry = gpd.points_from_xy(aerobus.LONGITUD, aerobus.LATITUD), crs = bcn_df.crs)
map_df['aerobus_distance'] = [min(haversine_distance(tlat, tlon, listlat, listlon) for tlat,tlon in zip(aerobus.LATITUD, aerobus.LONGITUD))\
                          for listlat, listlon in zip(map_df.latitude, map_df.longitude)]

print('\nCalculamos las paradas cercanas')
    
np.random.seed(1997)
distance = 300
map_df = map_df.to_crs(epsg = 3857)
mapbuffer = map_df.copy()
mapbuffer['geometry'] = mapbuffer['geometry'].buffer(distance)

print('metro')    
metro = transport[transport['NOM_CAPA'] == 'Metro i línies urbanes FGC']
metro = gpd.GeoDataFrame(metro, geometry = gpd.points_from_xy(metro.LONGITUD, metro.LATITUD), crs = bcn_df.crs)
metro = metro.to_crs(epsg = 3857)


map_df['metros_cercanos'] = [sum(i.within(j) for i in metro.geometry) for j in mapbuffer.geometry]


print('tramvia')
tramvia = transport[transport['NOM_CAPA'] == 'Tramvia']
tramvia = gpd.GeoDataFrame(tramvia, geometry = gpd.points_from_xy(tramvia.LONGITUD, tramvia.LATITUD), crs = bcn_df.crs)
tramvia = tramvia.to_crs(epsg = 3857)

map_df['tranvia_cercanos'] = [sum(i.within(j) for i in tramvia.geometry) for j in mapbuffer.geometry]

print('bus')
bus = transportsbus[transportsbus['NOM_CAPA'] == 'Autobusos diürns']
bus = gpd.GeoDataFrame(bus, geometry = gpd.points_from_xy(bus.LONGITUD, bus.LATITUD), crs = bcn_df.crs)
bus = bus.to_crs(epsg = 3857)

map_df['bus_cercanos'] = [sum(i.within(j) for i in bus.geometry) for j in mapbuffer.geometry]

cols_select = [x for x in map_df.columns if x not in ['neighbourhood_group_cleansed', 'latitude','longitude', 'geometry']]

distances = map_df[cols_select]

print('\n Generamos el dataset de Landmarks y de Transportes')
distances.to_csv('~/DadesAirBNB/Distancias.csv', index = False)
distances.to_pickle('~/DadesAirBNB/Distancias.pkl')

print('\n Ejecución finalizada con éxito!')









