# AirBNB analysis and price prediction
## TFM / Máster en Data Science / KSCHOOL
#### **Guillem Rochina Aguas y Helena Saigí Aguas**
#### 04/09/2020

# Introducción y Motivación

El sitio web AirBNB permite a propietarios de apartamentos convertirse en 'hosts' de la plataforma y ofertar sus propiedades, llamadas 'listings', online a fin de cobrar un alquiler vacacional. Determinar el precio del alojamiento puede ser una tarea difícil en ciudades con tanta competencia como Barcelona si lo que se busca es maximizar el beneficio/ocupación de tu listing. Además, a pesar de que ya existen plataformas como AirDNA ofreciendo asesoramiento en este aspecto, o bien son demasiado caros para el usuario medio u ofrecen predicciones poco convincentes.

![](/imagenes/AirBNBweb.png?raw=true)

Es por ello que el objetivo principal de este proyecto de TFM es el desarrollo de un modelo predictivo de los precios a través del uso de machine learning y deep learning, así como de data engineering para la limpieza y exploración de datos tanto de la web de AirBNB como de datos públicos de la ciudad de Barcelona.

De forma alternativo, otro de los usos adicionales que un usuario no propietario de alojamientos a este análisis, es el de 'buscador de chollos', que permita a usuarios interesados en alquilar habitaciones en la ciudad de Barcelona acceder a 'listings' que a priori deberían tener unos precios más elevados debido a sus características y predicciones por parte del modelo.

# Descripción de los Datasets

Los datos trabajados en este proyecto de TFM provienen de diversas fuentes, tratando desde datos propios de AirBNB como datasets de dominio público obtenidos de webs como Flickr y OpenData Barcelona.

## Dataset Principal

El dataset principal consta de csv y geojson obtenidos desde la web de [Inside AirBNB](http://insideairbnb.com/get-the-data.html), organización que se encarga de hacer web scrapping de todos los datos accesibles en la página web de AirBNB de forma mensual. Concretamente, los archivos utilizados constan de dos csv, **listings.csv** y **calendar.csv**, y un geojson correspondiente a **neighbourhood.geojson**, todos ellos presentados a continuación.

### Listings.csv

Es el archivo más importante de los tres, en este csv encontramos toda la información disponible en la página específica de cada uno de los listings, desde descripciones propias del host hasta cuantos servicios o comodidades adicionales ofrece el listing. A continuación se describen brevemente las principales columnas del dataset.

| Columna       | Descripción          
| ------------- |------------- | 
| host_since | fecha en la que se registró el host en AirBNB |
| host_response_time | tiempo medio que tarda el host en responder |
| host_response_rate | porcentaje de mensajes que responde el host |
| host_is_superhost | superhost es una señal de veteranía y experiencia en la plataforma|
| host_verifications | formas de contacto e identificación verificadas por el host y AirBNB|
| neighbourhood_group_cleansed | vecindario/distrito del listing |
| property_type | columna del tipo de propiedad generalizada a cuatro categorias (Apartment, House, Hotel y Other|
| room_type | tipo de habitación (Private, Hotel, Shared Room o bien Entrie Home) |
| amenities | diccionario de servicios/comodidades adicionales que ofrece el alojamiento |
| price | variable a predecir y (en su mayoría) precio por noche del listing |
| security_deposit | cantiad de depósito obligatoria a abonar durante la estancia (en caso de ser necesario) |
| cleaning_fee | tasa fija de limpieza |
| extra_people | cantidad adicional al precio por persona si se supera el nº de guest included|
| cancellation_policy | tipo de cancelación (flexible, moderate, strict...) |
| host_listings_count | nº de alojamientos en propiedad del host |
| accommodates | nº máximo de huéspedes permitidos |
| bathrooms | nº de baños |
| bedrooms | nº de dormitorios |
| beds | nº de camas |
| guests_included | nº de huéspedes incluídos en el precio base |
| minimum_nights | nº mínimo de noches para la reserva |
| maximum_nights | nº máximo de noches permitidas de reserva |
| availability_x | disponibilidad del listing los siguientes x días (30, 60, 90, 365)|
| reviews_per_month| media calculada de nº de reviews por mes |
| number_of_reviews| nº total de reviews de cada listing |
| reviews_scores_rating| calificación total del alojamiento proporcionada por los usuarios |

### Calendar.csv

El dataset de calendar nos proporciona información diaria sobre como se comportan los precios y la disponibilidad de los listings. A pesar de que encontramos el comportamiento de los precios a nivel diario, la capacidad de almacenamiento y procesamiento nos ha llevado a utilizar medias mensuales de cada listing para reducir el tamaño de los datos. Tan sólo nos centraremos en cuatro columnas relevantes de este csv.

| Columna       | Descripción          
| ------------- |------------- | 
| year | - |
| month | - |
| price_calendar | precio medio mensual por listing |
| year_availability | disponibilidad mensual de cada listing |

### Neighbourhood.geojson

**Neighbourhood.geojson** es un archivo utilizado para representar elementos geográficos de Barcelona, en este caso los vecindarios de la ciudad, a través de geometrías de tipo poligono y multipoligono. La principal utilidad de este archivo es la de ser la principal referencia para trabajar con las latitudes y longitudes de los demás datasets, a través de la librería **geopandas**.

## Dataset de Flickr

Dataset que recoge las coordenadas de la ciudad de Barcelona donde se han tomado fotos relacionadas con algún monumento o lugar de interés de la ciudad (los cuáles llamaremos **landmarks**). Este dataset tan solo presenta la latitud y longitud sin especificar de que landmark se trata ninguno de los puntos, estando sujeto a un problema de tipo no supervisado, concretamente **clustering**, y se trabajará con coordenadas en grados **(Código EPSG 4326)**.

## Datasets de Transportes

Los datasets de transportes provienen de la web [Open Data BCN](https://opendata-ajuntament.barcelona.cat/data/es/dataset), plataforma donde se almacena todo tipo de información pública gestionada por entidades municipales, y a la que tiene acceso público y gratuito cualquiera con interés en utilizar dichos datos.

Específicamente, en cuanto a transportes utilizamos dos csv distintos, uno de **Transportes Ferroviarios** (metro, ferrocarril, RENFE...) y otro de **Autobuses Urbanos**. En ambos datasets, se trabajará únicamente con coordenadas en metros **(Código EPSG 3847)** y tan sólo tendremos en cuenta 3 columnas relevantes:

| Columna       | Descripción          
| ------------- |------------- | 
| NOM_CAPA | Tipo y nombre del transporte |
| LATITUD | Latitud de la parada/estación |
| LONGITUD | Longitud de la parada/estación |

## Datasets de Sitios de Interés Turístico

Finalmente, los datasets relacionados con sitios de interés turístico se dividen en cuatro csv de nuevo procedentes de la web de [Open Data BCN](https://opendata-ajuntament.barcelona.cat/data/es/dataset): **Cinemes_teatres_auditoris.csv**, **Biblioteques_i_museus.csv**, **Restaurants.csv** y **Espais_de_musica_i_copes.csv**. Al igual que los datasets de transporte, tan solo nos hemos centrado en tres columnas de cada dataset:

| Columna       | Descripción          
| ------------- |------------- | 
| SECCIÓ | Tipo y nombre del sitio turístico |
| LATITUD | Latitud del sitio turístico |
| LONGITUD | Longitud del sitio turístico |
 
# Desarrollo del Proyecto

##  Paquetes y Prerequsitos

El desarrollo de los scripts y los notebooks se ha llevado a cabo mediante el lenguaje Python, por lo que será imprescindible para poder seguir este proyecto. No obstante, se ha optado por desarrollar el proyecto también en Google Colab a fin de que estas limitaciones dificulten lo mínimo posible el seguimiento de este proyecto.

En caso de querer seguir los scripts o los notebooks es necesario instalar los siguientes paquetes:

**numpy**, **scipy**, **pandas**, **scikit-learn**, **tensorflow**, **keras**, **geopandas**, **shapely**, **contextily**, **matplotlib**, **seaborn**, **statsmodels**, **datetime**.

En caso de no tener todos instalados recomendamos crear un environment a partir del archivo **TFMenvironment.yml** con el siguiente código:

```shell
$ conda env create -f environment.yml
```

**El proyecto ha sido divido en cuatro partes, y por tanto, aunque no es necesario se recomienda ejecutarlos en ese orden para tener un compresión global.**

##  Limpieza

[LINK A COLAB]

**INPUTS:** Listings.csv, Calendar.csv **OUTPUTS:** DatosLimpios.csv

La primera Fase de este proyecto consiste en la limpieza y análisis superficial de los datasets base para la evolución del TFM, listings.csv y calendar.csv. 

Un primer barrido de eliminación de columnas suprimió del proceso todas las variables relacionadas con urls, así como descripciones tanto del host como del alojamiento (se planteó el uso de NLP en estas columnas a fin de encontrar nuevos atributos útiles pero finalmente, dado que los algoritmos ya estabann dando muy buenos resultados, se decidió seguir un camino distinto). Por otro lado, también fueron eliminadas columnas con más de un **60%** de Nulls dada su relativamente baja importancia y el riesgo a introducir un sesgo grande por medio de la imputación de valores (tanto predichos a través de modelos lineales como medianas o medias).

```python
nulls = df.isnull().sum() / df.shape[0]
nulls[nulls > 0.5]
```

![](/imagenes/60Cleaning.png?raw=true)

```python
df.drop(nulls[nulls>0.6].index, axis = 1, inplace = True)
```

La limpieza se desarrolla a continuación con el siguiente procedimiento: eliminación de columnas poco útiles o repetidas, eliminación de filas repetidas o con datos anómalos, imputación de valores, etc. Destacamos los procedimientos de limpieza más relevantes y menos comunes a continuación:

- **Variables categóricas**

Existen columnas categóricas con un gran número de clases muy similares entre sí, es por ello que a fin de reducir las dimensiones de nuestro dataset lo máximo posible, hemos generalizado todos los valores en el menor número de categorías posible. Ejemplo de ello es la variable **cancellation_policy** que ha sido generalizada a cuatro alternativas **(flexible, moderate, strict_less30, strict_30orMore)**.

```python
df['cancellation_policy'].value_counts()
```
![](/imagenes/CatCleaning1.png?raw=true)


```python
tempdict = {'strict_14_with_grace_period': 'strict_less30', 'flexible':'flexible', 'moderate':'moderate', 'luxury_moderate': 'moderate', 'super_strict_30':'strict_30orMore', 'super_strict_60':'strict_30orMore', 'strict':'strict_less30'}
df['cancellation_policy'] = df['cancellation_policy'].map(tempdict)

df['cancellation_policy'].value_counts()
```

![](/imagenes/CatCleaning2.png?raw=true)

- **Variables string de precios**

Todas las columnas del dataset cuyo valor es el de un precio se presentan con un símbolo de dólar al principio y con comas a partir de los millares **E.G. $1,200.00**. La limpieza de estas variables ha sido abordada a través del method chaining de varias funciones **replace** ,para la eliminación de los símbolos anteriormente mencionados, y la transformación de tipo string a tipo float (en las columnas que presentaban Null debido a su naturaleza, E.G. existen listings sin tarifa de limpieza y en vez de ser codificado con 0 se presenta como un Null, se ha imputado valores de 0€).

```python
df[['price', 'security_deposit', 'cleaning_fee', 'extra_people']].sample(5)
```
![](/imagenes/ColPrice1.png?raw=true)

```python
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype('float')

df['security_deposit'] = df['security_deposit'].str.replace('$', '').str.replace(',','').fillna(0).astype('float')

df['cleaning_fee'] = df['cleaning_fee'].str.replace('$', '').str.replace(',','').fillna(0).astype('float')

df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',','').fillna(0).astype('float')

df[['price', 'security_deposit', 'cleaning_fee', 'extra_people']].sample(5)
``` 
![](/imagenes/ColPrice2.png?raw=true)

- **Caso Especial: Amenities**

La columna amenities ha resultado ser un caso especial, ya que cada registro se presenta en forma de lista (con llaves **{}** en vez de corchetes **[]**) y además de ser de tipo string. Por ello, en primer lugar para visualizar lo comunes que son cada uno de los amenities entre todos los alojamientos se utiliza de nuevo el method chaining para tratar con el string y transformarlo realmente en una lista, a continuación mediante un diccionario y una serie de pandas logramos el objetivo de visualizar el porcentaje total de aparición de cada amenity.

```python
amenities = df['amenities'].str.replace('{', '').str.replace('}', '').str\
                          .replace('"', '').str.split(",")
dict_amen= {}
for lista in amenities:
    for elemento in lista:
        if elemento in dict_amen:
            dict_amen[elemento] += 1
        else:
            dict_amen[elemento] = 1

ser_amen = pd.Series(dict_amen, index = dict_amen.keys())
(ser_amen[ser_amen>df.shape[0]*.10]/df.shape[0]).sort_values(ascending = False)
```

![](/imagenes/Amenities.png?raw=true)

Una vez visualizados, se seleccionaron los que por ser no tan comunes y, bajo nuestro criterio, relevantes para un huésped consideramos utiles para la determinación de un precio superior del alojamiento respecto a los que carecen de estos servicios. En concreto seleccionamos el siguiente conjunto a través de la creación de variables dummy:

```python
columnselection = ['Air conditioning', 'Family/kid friendly', 'Host greets you', 'Laptop friendly workspace', 'Paid parking off premises', 
                  'Patio or balcony', 'Luggage dropoff allowed', 'Long term stays allowed', 'Smoking allowed', 'Step-free access',
                  'Pets allowed', '24-hour check-in']
for column in columnselection:
    df[column] = df['amenities'].apply(lambda x: 1 if column in x else 0)
df['Elevator'] = df['amenities'].apply(lambda x: 1 if ('Elevator' in x or 'Elevator in building' in x) else 0)
df.drop('amenities', axis = 1, inplace = True)
```

Finalmente, para la última parte de esta sección se procedió al tratamiento de los datasets de **calendar.csv**, calculamos las columnas **price_calendar** y **year_availability** a través de groupbys y finalizamos con el merge de calendar y listings para la creación de **DatosLimpios.csv**.

##  Exploración Parte A

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv **OUTPUTS:** Localizaciones.csv

Esta primera fase de Exploración General se centra en el análisis, limpieza y transformación de la variable dependiente, en este caso **goodprice** obtenida a partir de las medias mensuales de precios calculadas en la fase de limpieza.

En primer lugar se procedió a investigar los motivos de la existencia de precios superiores a una cota de 2000€ y posteriormente 1200€, ya que excepto algún caso fuera de lo común superar esta barrera de precio por noche supone, bajo nuestro punto de vista, una anomalía producida por un error de registro o bien por un cálculo erróneo o fenómeno que no hemos tenido en cuenta. La investigación dió pié a descubrir un reducido número "outliers" en los que el propio alojamiento carecía de página propia en la web de AirBNB actualmente, o bien eran resultado de un cálculo erróneo del precio por noche (por parte del equipo de Inside AirBNB) a partir de los precios mensuales resultantes de alquilar un mínimo de 31 noches el alojamiento. Es por ello, que al tratarse de menos de un 1% de registros, estos fueron eliminados desde el principio del análisis a fin de evitar problemas futuros.

Resultado de este filtrado de precios obtenemos un histograma a priori visualmente muy semejante a una distribución **LogNormal** ![equation](https://latex.codecogs.com/png.latex?exp(X)\sim&space;LogN(\mu,&space;\sigma^2)).

![](/imagenes/LogNormal.png?raw=true)

Es por ello que aplicamos un logaritmo natural como medida típica para "normalizar" nuestra variable dependiente. Una comparativa de una distribución normal generada a partir del paquete random nos muestra bastante semejanza a una distribución normal  a pesar de la ligera asimetría positiva.

![](/imagenes/Normal.png?raw=true)

Concretamente, el **coeficiente de asimetría de Fisher** resulta ser de 0.404, como ya puede observarse en el gráfico superior, una pequeña asimetría positiva que nos aleja de una buena aproximación a una distribución normal. Por otra parte el **exceso de kurtosis** es de -0.168 causada principalmente por la cola izquierda, demasiado cercano al valor 0 como para considerarla platicurtica. Finalmente, graficar un QQ-Plot nos demuestra que el problema radica en las colas de la distribución (demasiados pocos registros de precios bajos respecto al grueso y el hecho de establecer un "cut-off" para eliminar mucho outlier nos dificulta que la distribución sea completamente "normal"). Los test **Kolmogorov-Smirnov** y **D'Agostino-Pearson** nos terminan de confirmar nuestras conclusiones.

![](/imagenes/QQplot.png?raw=true)

Para finalizar el análisis de la variable dependiente, se realizó un estudio de la evolución de los precios a fin de encontrar una posible tendencia o estacionalidad a lo largo de los meses y años. Para ello se procedió a gráficar los precios medios por mes junto a sus intervalos de confianza.

```python
def intervalosConfianza(agrupacion, variable):
  """Agrupa los datos del dataset según la columna que indiques y calcula el 
  intervalo de confianza de la segunda columna que especifiques"""

  intsup = (df.groupby(agrupacion)[variable].mean() + 1.96*(df.groupby(agrupacion)[variable].std()/np.sqrt(df.groupby(agrupacion)[variable].count()))).reset_index()
  intinf = (df.groupby(agrupacion)[variable].mean() - 1.96*(df.groupby(agrupacion)[variable].std()/np.sqrt(df.groupby(agrupacion)[variable].count()))).reset_index()
  return intsup, intinf

intsup, intinf = intervalosConfianza('month_year', 'goodprice')

fig, ax = plt.subplots(1, 1, figsize = (25, 8))
plt.plot(df.groupby('month_year')['goodprice'].mean().index, 
         df.groupby('month_year')['goodprice'].mean(), 
         color = 'navy', label = 'Precio Medio')
plt.fill_between(df.groupby('month_year')['goodprice'].mean().index,
                 intsup['goodprice'], intinf['goodprice'], 
                 color = 'red', alpha = 0.6, label = 'Intervalo de confianza')
ax.legend(edgecolor = 'black', facecolor = 'white', fancybox = True,
           bbox_to_anchor=(0.17, 0.96), fontsize = 'x-large')
plt.xticks(df['month_year'].unique(), rotation = 55)
plt.tight_layout()
```

![](/imagenes/EvoluciónMensual.png?raw=true)

Un primer vistazo nos muestra una clara estacionalidad de los precios, situandose los más altos en los meses de verano, con algunos picos en lo que suponemos vacaciones de primavera. Curiosamente, la tendencia positiva que encontramos desde 2017 a finales de 2020 se ve eclipsada por un año 2018 con precios especialmente altos en temporada de vacaciones de verano (a pesar de que no hemos encontrado ningún grupo de outliers que empuje los precios hacia arriba). La descomposición estacional de la variable precio nos permite ver claramente esta evolución.

![](/imagenes/Descomposicion.png?raw=true)

A partir de la descomposición estacional podemos ver de forma más evidente como los precios crecen hasta 14€ de media para luego volver a valores ligeramente superiores a 2017 el año siguiente. Además, ailsar la estacionalidad nos permite analizar más detenidamente que los picos más altos se encuentran en los mese de verano (Junio, Julio y Agosto presentan niveles similares) y a partir de los inicidos de Septiembre los precios caen en picado, siempre teniendo en cuenta lo que hemos comentado anteriormente de las vacaciones de primavera.

Por otra parte, los errores parecen ser completamente aleatorios, irregulares y centrados en 0, además de no presentar ningún patrón a priori aparente, por lo que lo que podemos asegurar que los residuos son producto de pequeñas causas impredecibles.

No obstante, de esta última parte del análisis solo podemos destacar la posible importancia de las variables **year** y **month** para la predicción de precios, ya que como pudimos ver en el apartado de limpieza, muchos registros empiezan en el año 2019 y otros desaparecen en los últimos años del dataset. Es más, como se puede observar en los notebooks cada registro presenta un comportamiento completamente distinto a los demás (algunos permanecen constantes, otros dan saltos arbitrarios a gusto del host y algunos si que se ven sujetos a la estacionalidad), por lo que abordar este problema como una serie temporal supondría dar un enfoque con muchas menos posiblidades de éxito que tratar cada registro como una predicción independiente.

##  Geoexploración

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv **OUTPUTS:** Distancias.csv, DistanciasTurismo.csv

La Geoexploración supone un interludio dentro la fase de exploración, una breve desviación que hemos decidido tratar en un notebook distinto debido a que tiene una temática distinta a la que tratamos en la exploración más general. Esta fase del proyecto se divide en dos enfoques distintos centrados en datos geoespaciales, en primer lugar la determinación de localizaciones de **Landmarks** a través del calculo de centroides y el ćalculo de distancias entre landmark y alojamiento, y por otro lado el cálculo, o bien de distancias a paradas de trasnporte cercanas, o bien el número de paradas cercanas a un alojamiento.

- **Centroides a partir de datos de Flickr**

A través de la recopilación de coordenadas de la ciudad de Barcelona donde se han tomado fotos relacionadas con algún monumento o sitio especialmente turístico, procedimos a realizar el cálculo de centroides como punto de referencia para la localización de un landmark. En un principio se presentaba un mapa con la siguiente distribución:

![](/imagenes/Landmarks1.png?raw=true)

A primera vista ya se observaron puntos bastante separados de lo que consideramos un landmark, los cuáles consideramos outliers o localizaciones con muy poco dato como para considerarlo un landmark. Para la eliminación de dichos datos anómalos se decidió probar con **Clustering Jerárquico**, utilizando varios métodos y métricas de distinto tipo, aunque finalmete se optó por utilizar el método **single** (crear clústers a partir de los puntos más cercanos entre sí) y la **distancia de manhattan** o **cityblock**, ![ecuation](https://latex.codecogs.com/png.latex?%5Cinline%20D%28x%2Cy%29%20%3D%20%5Csum_%7Bi%3D1%7D%5En%7Cx_%7Bi%7D-y_%7Bi%7D%7C), dado que discriminaba más fácilmente los puntos más alejados entre sí (y el cálculo de distancias por ciudad se lleva bien con la métrica de cityblock).

![](/imagenes/Dendrograma.png?raw=true)

Una vez eliminados los outliers a través de esta primera clusterización, se comprueba la desaparición de posibles puntos problemáticos a la hora de aplicar el algoritmo de **K-Means** a través del mismo clustering jerárquico utilizando la **distancia euclídea**, ![ecuation](https://latex.codecogs.com/png.latex?%5Cinline%20D%28x%2Cy%29%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%28x_%7Bi%7D-y_%7Bi%7D%29%5E2%7D
), y el método **ward** (busca minimizar la varianza total dentro de cada clúster), que asemejan los resultados a un k-means. Una vez demostrado que no debería haber problemas en el búsqueda de clusters y centroides se procede a aplicar y ajustar el k medias dando como resultado el siguiente mapa:

![](/imagenes/Kmeans.png?raw=true)

A partir de los centros de cada clúster, creamos un dataframe que recoge cada centroide y se le asigna un número del 1-11 debido a que la aleatoriedad en la asignación del número de clúster nos impide ser consistentes en la asignación de nombres a la agrupación.

```python
clusters = ['Landmark_1', 'Landmark_2', 'Landmark_3', 'Landmark_4', 'Landmark_5', 'Landmark_6', 'Landmark_7',
            'Landmark_8', 'Landmark_9', 'Landmark_10', 'Landmark_11']
centroids = km.cluster_centers_.tolist()

centroids_km = pd.DataFrame({'cluster': clusters, 'centroids': centroids})
```

![](/imagenes/Landmarks2.png?raw=true)

Finalizando este primer análisis de datos geográficos, cada uno de los centroides es utilizado para calcular las distancias con cada uno de los alojamientos de nuestro dataframe original. Para ello utilizamos la **fórmula del semiverseno** o **Haversine distance**, ![ecuation](https://latex.codecogs.com/png.latex?%5Cinline%20D%20%3D%202%5Ccdot%20r%20%5Ccdot%20arcsin%28%20%5Csqrt%7B%5Cfrac%7B%5Cvarphi_2%20-%20%5Cvarphi_1%7D%7B2%7D%20&plus;%20cos%28%5Cvarphi_1%29cos%28%5Cvarphi_2%29sin%5E2%28%5Cfrac%7B%5Clambda_2%20-%5Clambda_1%7D%7B2%7D%7D%29%20%29), dado que para el cálculo de distancias geográficos en la tierra es más "robusto" y fiable que la distancia euclídea. 

```python
for landlat, landlon, name in zip(landmarks.Latitud, landmarks.Longitud, landmarks.cluster):
    print(name)
    map_df['{}_Haversinedistance'.format(name)] = [haversine_distance(landlat, landlon, listlat, listlon) for listlat, listlon in zip(map_df.latitude, map_df.longitude)]```
```

Con esto finalizamos la primera parte de esta geoexploración.

- **Transportes y Sitios de Interés Turísico**

En la segunda fase de análisis geográfico el análisis de divide en dos vertientes:
 
    - Cálculo de distancias a la parada más cercana de cada medio de trasmporte para los listings del dataset.

    - Creación de **buffers** para el cálculo del número de elementos cercanos (principalmente sitios de interés turístico, aunque algún tipo de transporte ha sido incluido en este método) dentro del rango del buffer.

Para el cálculo de distancias utilizamos de nuevo la fórmula del semiverseno, sin embargo realizamos una pequeña modificación para que tan sólo se quede con la distancia más pequeña de todas las paradas.

```python
map_df['fgc_distance'] = [min(haversine_distance(tlat, tlon, listlat, listlon) for tlat,tlon in zip(fgc.LATITUD, fgc.LONGITUD)) for listlat, listlon in zip(map_df.latitude, map_df.longitude)]
```

![](/imagenes/TransporteCercanos.png?raw=true)

Para el método del buffer, transformamos las geometrias de todos los listings a fin de crear puntos con un diámetro mucho mayor, basándonos en un criterio distinto en caso de los transportes y los sitios turísticos (300 metros de radio para paradas de transporte y 600 para sitios turísticos). Por cada uno de las localizaciones se realiza el cálculo de booleanos (True/False) a través de la función de geopandas **within**, la cual marca con True los lugares que se encuentran dentro del *área* del alojamiento y con False los que no se encuentran en ella. Posteriormente, estos se suman (asumiendo que los valores True equivalen a 1 y los False a 0).

```python
map_df['tranvia_cercanos'] = [sum(i.within(j) for i in tramvia.geometry) for j in mapbuffer.geometry]

map_df['museos_cercanos'] = [sum(i.within(j) for i in museos.geometry) for j in mapbuffer.geometry]
```
![](/imagenes/ParadasCercanas.png?raw=true)

![](/imagenes/Museos.png?raw=true)

##  Exploración Parte B

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv, Distancias.csv, DistanciasTurismo.csv **OUTPUTS:** DatosModelar.csv



**Variables Numéricas**

Para este primer enfoque abordamos el análisis de las variables númericas a partir de **gráficos de colmena combinados con los histogramas de las distribuciones marginales**. Dado que nuestro baseline se basa en un modelo lineal básico, la **Regresión Lineal Múltiple**, decidimos realizar las transformaciones pertinentes basándonos en la maximización del **Coeficiente de correlación de Pearson**, ![equation](https://latex.codecogs.com/gif.latex?\inline&space;\rho_{xy}&space;=&space;\frac{S_{xy}}{S_{x}S_{y}}), ya sea a partir de features polinómicos, en escala logaritmica, raíces cuadradas o cúbicas etc.

![](/imagenes/Accommodates.png?raw=true)

**Variables Categóricas**

![](/imagenes/BarplotCat.png?raw=true)

![](/imagenes/EvoCat.png?raw=true)

**Variables Dicotómicas**

![](/imagenes/BarplotDic.png?raw=true)

![](/imagenes/EvoDic.png?raw=true)

**Landmarks**

En general las distancias a los landmarks se aproximan o bien a una forma uniforme o normal, acumulando la mayoría de registros en el intervalo de 1000 a 5000 metros. Para este tipo de distancia hemos seguido el mismo proceso que con las anteriores variables numéricas, combinar gráficos de colmena con una regresión y la distribución de sus marginales. Con ello observamos que todos y cada uno de los atributos relacionados con este tipo de distancias presentan en mayor o menor medida, una tendencia negativa, es de suponer ya que la proximidad a un Landmark debería influenciar al alza, en cierta medida, a los precios de un alojamiento turístico. No obstante, observamos que dicha relación lineal es muy débil. Calculada a partir de la **correlación de Pearson** ![ecuation](https://latex.codecogs.com/gif.latex?%5Crho_%7Bxy%7D%20%3D%20%5Cfrac%7BS_%7Bxy%7D%7D%7BS_%7Bx%7DS_%7By%7D%7D) , las distancias más relevantes a penas presentan una correlación lineal del 20%, incluso cuando hemos tratado de encontrar y aplicar transformaciones que aumentaran la relación lineal entre el precio y las distancias. A pesar de ello, dado que estas se tratan en un futuro con un Ánalisis de Componentes Principales o **PCA** en el momento de modelado, se esperaba una serie de combinaciones lineales que nos permitiera eliminar atributos innecesarios sin tener que perder varianza explicada por estos features.


**Transportes**

De nuevo los gráficos en colmena revelan unas cuantas distancias o features de paradas cercanas que carecen de importancia, a causa de la gran red de transportes públicos que ofrece Barcelona, especialmente en cuanto a metros y autobuses, por lo que cualquier alojamiento tiene una o varias paradas cercanas y la proximidad o el número de paradas próximas de este tipo de transportes no es relevante a la hora de elegir o no una residencia turística. A pesar de esta cuestión, si que encontramos distancias a transportes con una leve relación lineal negativa respecto a los precios, sobretodo si hablamos de distancia a las paradas de bus al aeropuerto o distancia a las estaciones de Ferrorcarril. Una vez más, hemos intentado encontrar alguna transformación que aplicada a los datos aumentara la relación con el precio, aunque solo se ha encontrado alguna mejora en unos pocos features aplicando una transformación logaritmica.

**Sitios Turísticos**

Curiosamente, la mayoría de atributos relacionados con el número de lugares de interés turístico carecen de importancia en términos de relación lineal, pese a que una última vez más se ha intentado aplicar transformaciones a los datos en busca de alguna señal más fuerte. No obstante, destacamos atributos como **restaurantes cercanos** que a través de una transformación a base logarítmica sí que presentan alguna señal suficiente para ser considerada 'relevante', en este caso se ha procedido a crear nuevos atributos que recogen esta transformación y eliminado los originales a fin de evitar el problema de la colinealidad innecesariamente.

##  Modelado

[LINK A COLAB]

**INPUTS:** DatosModelar.csv **OUTPUTS:** xxxxxx

##  Visualización y Dashboard

[LINK A COLAB]

**INPUTS:** xxxx **OUTPUTS:** []

# Conclusiones y Mejoras
