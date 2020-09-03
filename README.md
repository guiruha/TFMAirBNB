# Análisis y predicción de precios de alojamientos en AIRBNB

<p align="center">
  <img src="/imagenes/logoAirBNB.jpeg" />
</p>

## TFM / Máster en Data Science / KSCHOOL
#### **Guillem Rochina Aguas y Helena Saigí Aguas**
#### 04/09/2020

# Índice

* [Introducción y Motivación](https://github.com/guiruha/TFMAirBNB#introducci%C3%B3n-y-motivaci%C3%B3n)

* [Descripción de los Datasets](https://github.com/guiruha/TFMAirBNB#descripci%C3%B3n-de-los-datasets)

	- [Dataset Principal](https://github.com/guiruha/TFMAirBNB#dataset-principal)
	- [Dataset de Flickr](https://github.com/guiruha/TFMAirBNB#dataset-de-flickr)
	- [Datasets de Transportes](https://github.com/guiruha/TFMAirBNB#datasets-de-transportes)
	- [Datasets de Sitios de Interés Turístico](https://github.com/guiruha/TFMAirBNB#datasets-de-sitios-de-inter%C3%A9s-tur%C3%ADstico)

* [Desarrollo del Proyecto](https://github.com/guiruha/TFMAirBNB#desarrollo-del-proyecto)

	- [Paquetes y Prerequisitos](https://github.com/guiruha/TFMAirBNB#paquetes-y-prerequsitos)
	- [Limpieza](https://github.com/guiruha/TFMAirBNB#limpieza)
	- [Exploración Parte A](https://github.com/guiruha/TFMAirBNB#exploraci%C3%B3n-parte-a)
	- [Geoexploración](https://github.com/guiruha/TFMAirBNB#geoexploraci%C3%B3n)
	- [Exploración Parte B](https://github.com/guiruha/TFMAirBNB#exploraci%C3%B3n-parte-b)
	- [Modelado](https://github.com/guiruha/TFMAirBNB#modelado)

* [Visualización y Dashboard](https://github.com/guiruha/TFMAirBNB#visualizaci%C3%B3n-y-dashboard)

* [Conclusiones y Mejoras](https://github.com/guiruha/TFMAirBNB#conclusiones-y-mejoras)

# Introducción y Motivación

El sitio web AirBNB permite a propietarios de apartamentos convertirse en 'hosts' de la plataforma y ofertar sus propiedades, llamadas 'listings', online a fin de cobrar un alquiler vacacional. En un mercado de competencia perfecta como este (muchos ofertantes, productos muy similares, fácil acceso a información del mercado y pocas barreras de entrada), determinar el precio del alojamiento puede ser una tarea difícil, mucho más en ciudades con tanta competencia como Barcelona, si lo que se busca es maximizar el beneficio/ocupación de tu listing. Además, a pesar de que ya existen plataformas como AirDNA ofreciendo asesoramiento en este aspecto, son demasiado caros para el usuario medio y no es habitual el uso de estas plataformas por parte de particulares.

![](/imagenes/AirBNBweb.png?raw=true)

Es por ello que el objetivo principal de este proyecto de TFM es el desarrollo de un modelo predictivo de precios a través del uso de machine learning/deep learning, así como de data engineering para la limpieza y exploración de datos tanto de la web de AirBNB como de datos públicos de la ciudad de Barcelona.

De forma alternativa, otro de los usos adicionales a este análisis, sobretodo para usuarios no propietarios, es el de 'buscador de chollos', que permita a usuarios interesados en alquilar habitaciones en la ciudad de Barcelona acceder a 'listings' que a priori deberían tener unos precios más elevados debido a sus características y predicciones por parte del modelo.

# Descripción de los Datasets

Los datos trabajados en este proyecto de TFM provienen de diversas fuentes, tratando desde datos propios de AirBNB como datasets de dominio público obtenidos de webs como Flickr y OpenData Barcelona.

## Dataset Principal

El dataset principal consta de varios csv y un geojson obtenidos desde la web de [Inside AirBNB](http://insideairbnb.com/get-the-data.html), organización que se encarga de hacer web scrapping de todos los datos accesibles en la página web de AirBNB de forma mensual. Concretamente, los archivos utilizados constan de dos tipos de csv, **listings.csv** y **calendar.csv**, y un geojson correspondiente a **neighbourhood.geojson**, todos ellos presentados a continuación.

### Listings.csv

Es el archivo más importante de los tres. En este csv encontramos toda la información disponible en la página específica de cada uno de los listings, desde descripciones propias del host, hasta cuantos servicios o comodidades adicionales ofrece el listing. Para nuestro análisis se utilizaron datos de los meses de Abril y Octubre (para tener una visión general del año). A continuación se describen brevemente las principales columnas del dataset.

| Columna       | Descripción          
| ------------- |------------- | 
| host_since | Fecha en la que se registró el host en AirBNB |
| host_response_time | Tiempo medio que tarda el host en responder |
| host_response_rate | Porcentaje de mensajes que responde el host |
| host_is_superhost | Superhost es una señal de veterania y experiencia en la plataforma|
| host_verifications | Formas de contacto e identificación verificadas por el host y AirBNB|
| neighbourhood_group_cleansed | Vecindario/distrito del listing |
| property_type | Tipo de propiedad, generalizada a cuatro categorías (Apartment, House, Hotel y Other|
| room_type | Tipo de habitación (Private, Hotel, Shared Room o bien Entire Home) |
| ammenities | Lista de servicios/comodidades adicionales que ofrece el alojamiento |
| price | Variable a predecir y (en su mayoría) precio por noche del listing |
| security_deposit | Cantidad de depósito obligatoria a abonar durante la estancia (en caso de ser necesario) |
| cleaning_fee | Tasa fija de limpieza |
| extra_people | Cantidad adicional al precio por persona si se supera el nº de guest included|
| cancellation_policy | Tipo de cancelación (flexible, moderate, strict...) |
| host_listings_count | Nº de alojamientos en propiedad del host |
| accommodates | Nº máximo de huéspedes permitidos |
| bathrooms | Nº de baños |
| bedrooms | Nº de dormitorios |
| beds | Nº de camas |
| guests_included | Nº de huéspedes incluidos en el precio base |
| minimum_nights | Nº mínimo de noches para la reserva |
| maximum_nights | Nº máximo de noches permitidas de reserva |
| availability_x | Disponibilidad del listing los siguientes 'x' días (30, 60, 90, 365)|
| reviews_per_month| Media calculada de nº de reviews por mes |
| number_of_reviews| Nº total de reviews de cada listing |
| reviews_scores_rating| Calificación total del alojamiento proporcionada por los usuarios |

### Calendar.csv

El dataset de calendar nos proporciona información diaria sobre cómo se comportan los precios y la disponibilidad de los listings (de los meses de Abril y Octubre también). A pesar de que encontramos el comportamiento de los precios a nivel diario, la capacidad de almacenamiento y procesamiento nos ha llevado a utilizar medias mensuales de cada listing para reducir el tamaño de los datos. Tan sólo nos centraremos en cuatro columnas relevantes de este csv.

| Columna       | Descripción          
| ------------- |------------- | 
| year | - |
| month | - |
| price_calendar | Precio medio mensual por listing |
| year_availability | Disponibilidad mensual de cada listing |

### Neighbourhood.geojson

**Neighbourhood.geojson** es un archivo utilizado para representar elementos geográficos de Barcelona, en este caso los vecindarios de la ciudad, a través de geometrías de tipo polígono y multipoligono. La principal utilidad de este archivo es la de ser la principal referencia para trabajar con las latitudes y longitudes de los demás datasets, a través de la librería **geopandas**.

## Dataset de Flickr

Dataset que recoge las coordenadas de la ciudad de Barcelona donde se han tomado fotos relacionadas con algún monumento o lugar de interés de la ciudad (los cuáles llamaremos **landmarks**). Este dataset tan solo presenta la latitud y longitud sin especificar de qué landmark se trata ninguno de los puntos, estando sujeto a un problema de tipo no supervisado, concretamente **clustering**, y se trabajará con coordenadas en grados **(Código EPSG 4326)**.

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

En caso de querer ejecutar los scripts o los notebooks es necesario instalar los siguientes paquetes:

**numpy**, **scipy**, **pandas**, **scikit-learn**, **tensorflow**, **keras**, **geopandas**, **shapely**, **contextily**, **matplotlib**, **seaborn**, **statsmodels**, **datetime**. **keras-tuner**, **xgboost**, **catboost**, **shap**, **GpyOpt**, **livelossplot**.

A fin de no deber instalar todos uno a uno recomendamos crear un environment a partir del archivo **TFMenvironment.yml** con el siguiente código:

```shell
$ conda env create -f TFMenvironment.yml
```

**El proyecto ha sido divido en cuatro partes, y por tanto, aunque no es necesario se recomienda ejecutarlos en ese orden para tener un compresión global.**

##  Limpieza

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/guiruha/TFMAirBNB/blob/master/1_LimpiezaPrueba.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guiruha/TFMAirBNB/blob/master/1_LimpiezaPrueba.ipynb)

**INPUTS:** Listings.csv, Calendar.csv **OUTPUTS:** DatosLimpios.csv, Localizaciones.csv

La primera Fase de este proyecto consiste en la limpieza y análisis superficial de los datasets base para la evolución del TFM, listings.csv y calendar.csv. 

Un primer barrido de eliminación de columnas suprimió del proceso todas las variables relacionadas con urls, así como descripciones tanto del host como del alojamiento (se planteó el uso de NLP en estas columnas a fin de encontrar nuevos atributos útiles pero finalmente, dado que los algoritmos ya estaban dando muy buenos resultados, se decidió seguir un camino distinto). Por otro lado, también fueron eliminadas columnas con más de un **60%** de Nulls dada su relativa baja importancia y el riesgo a introducir un sesgo grande por medio de la imputación de valores (tanto predichos a través de modelos lineales como medianas o medias).

```python
nulls = df.isnull().sum() / df.shape[0]
nulls[nulls > 0.5]
```

![](/imagenes/60Cleaning.png?raw=true)

```python
df.drop(nulls[nulls>0.6].index, axis = 1, inplace = True)
```

La limpieza continua desarrollándose con el siguiente procedimiento: eliminación de columnas poco útiles o repetidas, eliminación de filas repetidas o con datos anómalos, imputación de valores, etc. Destacamos los procedimientos de limpieza más relevantes y menos comunes a continuación:

#### **Variables categóricas**

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

#### **Variables string de precios**

Todas las columnas del dataset cuyo valor es el de un precio se presentan con un símbolo de dólar al principio y con comas a partir de los millares **E.G. $1,200.00**. La limpieza de estas variables ha sido abordada a través del method chaining de varias funciones **replace**, para la eliminación de los símbolos anteriormente mencionados, y la transformación de tipo string a tipo float (en las columnas que presentaban Null debido a su naturaleza, E.G. existen listings sin tarifa de limpieza y en vez de ser codificado con 0 se presenta como un Null, se ha imputado valores de 0€).

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

#### **Discretización de variables**

El tratamiento de los valores continuos, como la columna **review_scores_rating** consitió discretizarla, concretamente en cuatro categorias, **Excellent** en el caso de que la score sea mayor a 90, **Good** en el caso de que esté entre 90 y 70, **NotGood** para las restantes, así como **Unavailable** en caso de no estar disponible.

```python

df['review_scores_rating'] = df['review_scores_rating'].apply(lambda x: 'Excellent' if x >= 90 else \
                                                              ('Good' if x < 90 and x > 70 else ('NotGood' if x > 0 else 'Unavailable')))
```

![](/imagenes/Discretizacion.png?raw=true)

#### **Caso Especial: Amenities**

La columna ammenities ha resultado ser un caso especial, ya que cada registro se presenta en forma de lista (con llaves **{ }** en vez de corchetes **[ ]**), además de ser de tipo string. Por ello, en primer lugar para visualizar lo comunes que son cada uno de los amenities entre todos los alojamientos se utiliza de nuevo el method chaining para tratar con el string y transformarlo realmente en una lista, a continuación mediante un diccionario y una serie de pandas logramos el objetivo de visualizar el porcentaje total de aparición de cada ammenity.

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

Una vez visualizados, se seleccionaron las que por no ser comunes y a su vez (bajo nuestro criterio) relevantes en la elección de un alojamiento, consideramos útiles para la determinación de un precio superior del alojamiento respecto a los que carecen de estos servicios. En concreto seleccionamos el siguiente conjunto a través de la creación de variables dummy:

```python
columnselection = ['Air conditioning', 'Family/kid friendly', 'Host greets you', 'Laptop friendly workspace', 'Paid parking off premises', 'Patio or balcony', 'Luggage dropoff allowed', 
'Long term stays allowed', 'Smoking allowed', 'Step-free access', 'Pets allowed', '24-hour check-in']

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

En primer lugar se procedió a investigar los motivos de la existencia de precios superiores a una cota de 2000€ y posteriormente 1200€, ya que excepto algún caso fuera de lo común superar esta barrera de precio por noche supone, bajo nuestro punto de vista, una anomalía producida por un error de registro o bien por un cálculo erróneo o fenómeno que no hemos tenido en cuenta. La investigación dio pie a descubrir un reducido número "outliers" en los que el propio alojamiento carecía de página propia en la web de AirBNB actualmente, o bien eran resultado de un cálculo erróneo del precio por noche (por parte del equipo de Inside AirBNB) a partir de los precios mensuales resultantes de alquilar un mínimo de 31 noches el alojamiento. Es por ello, que al tratarse de menos de un 1% de registros, estos fueron eliminados desde el principio del análisis a fin de evitar problemas futuros.

Resultado de este filtrado de precios obtenemos un histograma con una gran asimetría positiva debido a que la gran mayoría de precios se acumulan en rangos inferiores a 100 euros/noche a pesar de que existen un cantidad considerable de precios que superan esta cota con creces.

![](/imagenes/PreLog.png?raw=true)

Es por ello que aplicamos un logaritmo neperiano como transformación de datos típica para "normalizar" nuestra variable dependiente. Una comparativa de una distribución normal generada a partir del paquete random nos muestra bastante semejanza a una distribución normal, a pesar de la ligera asimetría positiva.

![](/imagenes/Normal.png?raw=true)

Concretamente, el **coeficiente de asimetría de Fisher** resulta ser de 0.404, como ya puede observarse en el gráfico superior, una pequeña asimetría positiva que nos aleja de una buena aproximación a una distribución normal. Por otra parte el **exceso de kurtosis** es de -0.168 causada principalmente por la cola izquierda, demasiado cercano al valor 0 como para considerarla platicúrtica. Finalmente, graficar un QQ-Plot nos demuestra que el problema radica en las colas de la distribución (demasiados pocos registros de precios bajos respecto al grueso y el hecho de establecer un "cut-off" para eliminar mucho outlier nos dificulta que la distribución sea completamente "normal"). Los test **Kolmogorov-Smirnov** y **D'Agostino-Pearson** nos terminan de confirmar nuestras conclusiones.

![](/imagenes/QQplot.png?raw=true)

Para finalizar el análisis de la variable dependiente, se realizó un estudio de la evolución de los precios medios a fin de encontrar una posible tendencia o estacionalidad a lo largo de los meses y años. Para ello se procedió a graficar los precios medios por mes junto a sus intervalos de confianza.

```python
def intervalosConfianza(agrupacion, variable):
  """Agrupa los datos del dataset según la columna que indiques y calcula el 
  intervalo de confianza de la segunda columna que especifiques"""

  intsup = (df.groupby(agrupacion)[variable].mean() 
  + 1.96*(df.groupby(agrupacion)[variable].std()/np.sqrt(df.groupby(agrupacion)[variable].count()))).reset_index()
  intinf = (df.groupby(agrupacion)[variable].mean() 
  - 1.96*(df.groupby(agrupacion)[variable].std()/np.sqrt(df.groupby(agrupacion)[variable].count()))).reset_index()
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

Un primer vistazo nos muestra una clara estacionalidad de los precios, situándose los más altos en los meses de verano, con algunos picos en lo que suponemos vacaciones de primavera. Curiosamente, la tendencia positiva que encontramos desde 2017 a finales de 2020 se ve eclipsada por un año 2018 con precios especialmente altos en temporada de vacaciones de verano (a pesar de que no hemos encontrado ningún grupo de outliers que empuje los precios hacia arriba). La descomposición estacional de la variable precio nos permite ver claramente esta evolución.

![](/imagenes/Descomposicion.png?raw=true)

A partir de la descomposición estacional podemos ver de forma más evidente como los precios crecen hasta 14€ de media para luego volver a valores ligeramente superiores a 2017 el año siguiente. Además, aislar la estacionalidad nos permite analizar más detenidamente que los picos más altos se encuentran en los meses de verano (Junio, Julio y Agosto presentan niveles similares) y a partir de los inicios de Septiembre los precios caen en picado, siempre teniendo en cuenta lo que hemos comentado anteriormente de las vacaciones de primavera.

Por otra parte, los errores parecen ser completamente aleatorios, irregulares y centrados en 0, además de no presentar ningún patrón a priori aparente, por lo que podemos asegurar que los residuos son producto de pequeñas causas impredecibles.

No obstante, de esta última parte del análisis solo podemos destacar la posible importancia de las variables **year** y **month** para la predicción de precios, ya que como pudimos ver en el apartado de limpieza, muchos registros empiezan en el año 2019 y otros desaparecen en los últimos años del dataset. Es más, como se puede observar en los notebooks cada registro presenta un comportamiento completamente distinto a los demás (algunos permanecen constantes, otros dan saltos arbitrarios a gusto del host y algunos si que se ven sujetos a la estacionalidad), por lo que abordar este problema como una serie temporal supondría dar un enfoque con muchas menos posibilidades de éxito que tratar cada registro como una predicción independiente.

##  Geoexploración

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv **OUTPUTS:** Distancias.csv, DistanciasTurismo.csv

La Geoexploración supone un interludio dentro la fase de exploración, una breve desviación que hemos decidido tratar en un notebook distinto debido a que tiene una temática distinta a la que tratamos en la exploración más general. Esta fase del proyecto se divide en dos enfoques distintos centrados en datos geoespaciales, en primer lugar la determinación de localizaciones de **Landmarks** a través del cálculo de centroides y el cálculo de distancias entre landmark y alojamiento, y por otro lado el cálculo, o bien de distancias a paradas de transporte cercanas, o bien el número de paradas cercanas a un alojamiento.

### **Centroides a partir de datos de Flickr**

A través de la recopilación de coordenadas de la ciudad de Barcelona donde se han tomado fotos relacionadas con algún monumento o sitio especialmente turístico, procedimos a realizar el cálculo de centroides como punto de referencia para la localización de un landmark. En un principio se presentaba un mapa con la siguiente distribución:

![](/imagenes/Landmarks1.png?raw=true)

A primera vista ya se observaron puntos bastante separados de lo que consideramos un landmark, los cuáles consideramos outliers o localizaciones con muy poco dato como para considerarlo un landmark. Para la eliminación de dichos datos anómalos se decidió probar con **Clustering Jerárquico**, utilizando varios métodos y métricas de distinto tipo, aunque finalmente se optó por utilizar el método **single** (crear clústeres a partir de los puntos más cercanos entre sí) y la **distancia de manhattan** o **cityblock**, ![ecuation](https://latex.codecogs.com/png.latex?%5Cinline%20D%28x%2Cy%29%20%3D%20%5Csum_%7Bi%3D1%7D%5En%7Cx_%7Bi%7D-y_%7Bi%7D%7C), dado que discriminaba más fácilmente los puntos más alejados entre sí (y el cálculo de distancias por ciudad se lleva bien con la métrica de cityblock).

![](/imagenes/Dendrograma.png?raw=true)

Una vez eliminados los outliers a través de esta primera clusterización, se comprueba la desaparición de posibles puntos problemáticos a la hora de aplicar el algoritmo de **K-Means** a través del mismo clustering jerárquico utilizando la **distancia euclídea**, ![ecuation](https://latex.codecogs.com/png.latex?%5Cinline%20D%28x%2Cy%29%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5En%28x_%7Bi%7D-y_%7Bi%7D%29%5E2%7D
), y el método **ward** (busca minimizar la varianza total dentro de cada clúster), que asemejan los resultados a un k-means. Una vez demostrado que no debería haber problemas en el búsqueda de clusters y centroides se procede a aplicar y ajustar el k medias dando como resultado el siguiente mapa:

![](/imagenes/Kmeans.png?raw=true)

A partir de los centros de cada clúster, creamos un dataframe que recoge cada centroide y se le asigna un número del 1-11 debido a que la aleatoriedad en la asignación del número de clúster nos impide ser consistentes en la asignación de nombres a la agrupación.

```python
clusters = ['Landmark_1', 'Landmark_2', 'Landmark_3', 'Landmark_4', 'Landmark_5', 'Landmark_6', 
	'Landmark_7', 'Landmark_8', 'Landmark_9', 'Landmark_10', 'Landmark_11']
centroids = km.cluster_centers_.tolist()

centroids_km = pd.DataFrame({'cluster': clusters, 'centroids': centroids})
```

![](/imagenes/Landmarks2.png?raw=true)

Finalizando este primer análisis de datos geográficos, cada uno de los centroides es utilizado para calcular las distancias con cada uno de los alojamientos de nuestro dataframe original. Para ello utilizamos la **fórmula del semiverseno** o **Haversine distance**, ![ecuation](https://latex.codecogs.com/png.latex?%5Cinline%20D%20%3D%202%5Ccdot%20r%20%5Ccdot%20arcsin%28%20%5Csqrt%7B%5Cfrac%7B%5Cvarphi_2%20-%20%5Cvarphi_1%7D%7B2%7D%20&plus;%20cos%28%5Cvarphi_1%29cos%28%5Cvarphi_2%29sin%5E2%28%5Cfrac%7B%5Clambda_2%20-%5Clambda_1%7D%7B2%7D%7D%29%20%29), dado que para el cálculo de distancias geográficos en la tierra es más "robusto" y fiable que la distancia euclídea. 

```python
for landlat, landlon, name in zip(landmarks.Latitud, landmarks.Longitud, landmarks.cluster):
    print(name)
    map_df['{}_Haversinedistance'.format(name)] = [haversine_distance(landlat, landlon, listlat, listlon) 
    for listlat, listlon in zip(map_df.latitude, map_df.longitude)]```
```

Con esto finalizamos la primera parte de esta geoexploración.

### **Transportes y Sitios de Interés Turísico**

En la segunda fase de análisis geográfico el análisis de divide en dos vertientes:

- Cálculo de distancias a la parada más cercana de cada medio de transporte para los listings del dataset.

- Creación de **buffers** para el cálculo del número de elementos cercanos (principalmente sitios de interés turístico, aunque algún tipo de transporte ha sido incluido en este método) dentro del rango del buffer.

Para el cálculo de distancias utilizamos de nuevo la fórmula del semiverseno, sin embargo realizamos una pequeña modificación para que tan sólo se quede con la distancia más pequeña de todas las paradas.

```python
map_df['fgc_distance'] = [min(haversine_distance(tlat, tlon, listlat, listlon) for tlat,tlon in zip(fgc.LATITUD, fgc.LONGITUD)) for listlat, listlon in zip(map_df.latitude, map_df.longitude)]
```

![](/imagenes/TransporteCercanos.png?raw=true)

Para el método del buffer, transformamos las geometrías de todos los listings a fin de crear puntos con un diámetro mucho mayor, basándonos en un criterio distinto en caso de los transportes y los sitios turísticos (300 metros de radio para paradas de transporte y 600 para sitios turísticos). Por cada uno de las localizaciones se realiza el cálculo de booleanos (True/False) a través de la función de geopandas **within**, la cual marca con True los lugares que se encuentran dentro del *área* del alojamiento y con False los que no se encuentran en ella. Posteriormente, estos se suman (asumiendo que los valores True equivalen a 1 y los False a 0).

```python
map_df['museos_cercanos'] = [sum(i.within(j) 
for i in museos.geometry) 
for j in mapbuffer.geometry]
```

![](/imagenes/Museos.png?raw=true)

```python
map_df['tranvia_cercanos'] = [sum(i.within(j) 
for i in tramvia.geometry) 
for j in mapbuffer.geometry]
```

![](/imagenes/ParadasCercanas.png?raw=true)

##  Exploración Parte B

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv, Distancias.csv, DistanciasTurismo.csv **OUTPUTS:** DatosModelar.csv

### **Variables Numéricas**

Para este primer enfoque abordamos el análisis de las variables numéricas a partir de **gráficos de colmena combinados con los histogramas de las distribuciones marginales**. Dado que nuestro baseline se basa en un modelo lineal básico, la **Regresión Lineal Múltiple**, decidimos realizar las transformaciones pertinentes basándonos en la maximización del **Coeficiente de correlación de Pearson**, ![equation](https://latex.codecogs.com/gif.latex?\inline&space;\rho_{xy}&space;=&space;\frac{S_{xy}}{S_{x}S_{y}}), ya sea a partir de features polinómicos, en escala logarítmica, raíces cuadradas o cúbicas etc.

![](/imagenes/Accommodates.png?raw=true)

### **Variables Categóricas**

Los atributos categóricos fueron tratados en primer lugar con **Boxplots** y **Barplots** para obtener una visión general de cómo se distribuían, así como influenciaban al precio en caso de tener algún peso sobre este. Esta vez en lugar de ser transformadas (se iban a transformar en variables dummy), nuestro objetivo era determinar si deberían incluirse en la etapa de modelado o no.

![](/imagenes/BarplotCat.png?raw=true)

Por otro lado, a fin de no quedarnos con una visión demasiado general de la influencia de las variables categóricas, graficamos la  evolución de precios medios respecto a cada una de las categorías. Con ello observamos que a pesar de que a nivel general las diferencias entre categorías es muy evidente en ciertos momentos del tiempo, dichas diferencias no son tan notables como a priori se podía suponer.

![](/imagenes/EvoCat.png?raw=true)

### **Variables Dicotómicas**

Un elevado número de variables de nuestro dataset presentan tan solo valores booleanos. Al igual que con las categóricas, estas han sido tratadas con **Barplots** para comprobar que proporción son True o False, ya que consideramos que no vale la pena mantener una variable con una proporción demasiado dispar (si el 90% de los hosts tienen un email verificado, esa variable tendrá muy poca influencia en los precios), y con **Boxplots**, para analizar que relación mantenía entre categoría dicotómica y precio.

![](/imagenes/BarplotDic.png?raw=true)

De nuevo, graficamos la evolución de precios medios respecto a las alternativas dicotómicas. Como resultado encontramos que la relación entre precios y dichos atributos varían dependiendo del año y mes, llegando incluso a igualarse la media de precios de ambas categorías, por lo que el peso de estos atributos puede no llegar a ser tan relevante como considerábamos al principio. No obstante, en algunos casos las diferencias en vez de igualarse, pueden incluso duplicarse, por lo que también podemos estar subestimando la importancia de ciertas variables.

![](/imagenes/EvoDic.png?raw=true)

### **Landmarks**

En general las distancias a los landmarks se aproximan o bien a una forma uniforme o normal, acumulando la mayoría de registros en el intervalo de 1000 a 5000 metros. Para este tipo de distancias hemos seguido el mismo proceso que con las anteriores variables numéricas, combinar gráficos de colmena con una regresión y la distribución de sus marginales. Con ello observamos que todos y cada uno de los atributos relacionados con este tipo de distancias presentan, en mayor o menor medida, una tendencia negativa. Es de suponer ya que la proximidad a un Landmark debería influenciar al alza, en cierta medida, a los precios de un alojamiento turístico. 

![](/imagenes/LandmarksDistancia.png?raw=true)

No obstante, es evidente que dicha relación lineal es muy débil. Calculada a partir de la **correlación de Pearson** ![equation](https://latex.codecogs.com/gif.latex?\inline&space;\rho_{xy}&space;=&space;\frac{S_{xy}}{S_{x}S_{y}}), las distancias más relevantes a penas presentan una correlación lineal del 20%, incluso cuando hemos tratado de encontrar y aplicar transformaciones que aumentaran la relación lineal entre el precio y las distancias. A pesar de ello, dado que estas se tratan en un futuro con un Análisis de Componentes Principales o **PCA** en el momento de modelado, se esperaba una serie de combinaciones lineales que nos permitiera eliminar atributos innecesarios sin tener que perder varianza explicada por estos features.


### **Transportes**

De nuevo los gráficos en colmena revelan unas cuantas distancias o features de paradas cercanas que carecen de importancia, a causa de la gran red de transportes públicos que ofrece Barcelona, especialmente en cuanto a metros y autobuses, por lo que cualquier alojamiento tiene una o varias paradas cercanas y la proximidad o el número de paradas próximas de este tipo de transportes no es relevante a la hora de elegir o no una residencia turística. A pesar de esta cuestión, si que encontramos distancias a transportes con una leve relación lineal negativa respecto a los precios, sobretodo si hablamos de distancia a las paradas de bus al aeropuerto o distancia a las estaciones de Ferrocarril. Una vez más, hemos intentado encontrar alguna transformación que aplicada a los datos aumentara la relación con el precio, aunque solo se ha encontrado alguna mejora en unos pocos features aplicando una transformación logarítmica.

![](/imagenes/FerrocarrilExploracion.png?raw=true)

### **Sitios Turísticos**

Curiosamente, la mayoría de atributos relacionados con el número de lugares de interés turístico carecen de importancia en términos de relación lineal, pese a que una última vez más se ha intentado aplicar transformaciones a los datos en busca de alguna señal más fuerte. No obstante, destacamos atributos como **restaurantes cercanos** que a través de una transformación a base logarítmica sí que presentan alguna señal suficiente para ser considerada 'relevante', en este caso se ha procedido a crear nuevos atributos que recogen esta transformación y eliminado los originales a fin de evitar el problema de la colinealidad innecesariamente.

![](/imagenes/RestaurantesExploracion.png?raw=true)

### **Correlaciones**

Una vez estudiadas todas las variables del dataset y su relación con el precio, se procedió a realizar por último un análisis de **correlaciones** entre variables dependientes a fin de encontrar posibles **colinealidaes** que pudieran ser evitadas con la eliminación de dichos atributos.

En primer lugar, se seleccionaron las variables categóricas y dicotómicas analizadas en apartados anteriores para transformarlas en **variables dummy** para que puedan ser introducidas en el proceso de ajuste y predicción. 

```python
dummycols = ['host_response_time', 'neighbourhood_group_cleansed', 'property_type', 'room_type',  'cancellation_policy', 'review_scores_rating']
df = pd.get_dummies(df, columns = dummycols, drop_first = True)
df[df.columns[list(df.dtypes == 'uint8')]] = df[df.columns[list(df.dtypes == 'uint8')]].astype('int')
```

A continuación, calculamos los Coeficiente de Correlación y los graficamos a modo de barplot para analizarlos de forma más visual. Si bien muchos de los coeficientes relevantes ya los habíamos identificado, esta vez han aparecido algunos entre las variables dummy muy dignas de mención, como **Room_type Private Room** o bien variables relacionadas con los ammenities como son **Air Conditioning** o  **Family Kid-Friendly**

![](/imagenes/CorrPearson.png?raw=true)

Continuando el tratamiento de correlaciones, graficamos una matriz de correlaciones, a través de la cual detectamos puntos calientes de interés. Uno de los principales focos donde se concentran las colinealidades elevadas se encuentran los atributos relacionados con las distancias, sin embargo, dada nuestra intención de ajustar estos atributos y reducir su dimensionalidad con un **PCA**, ignoramos estos features para más adelante.

![](/imagenes/MatrizCorr.png?raw=true)

Por otro lado, encontramos una correlación mayor de 80% entre las variables de **Beds, Accommmodates, Bedrooms**, como era de esperar debido a la naturaleza de su relación. Evidentemente, el número de huéspedes se presentó como el feature de mayor peso en la predicción de precios, por lo que a pesar de la clara influencia de **Beds** y **Bedrooms**, optamos por eliminarlas en favor de **LogAccommodates**. Por último, curiosamente los **Restaurantes Cercanos** y las **Salas de Música Cercanas** tiene una correlación de 86%, dada su relación más débil con la variable precio se eliminó la variable que recogía el número de salas de múscia cercanas.

##  Modelado

[LINK A COLAB]

**INPUTS:** DatosModelar.csv **OUTPUTS:** xxxxxx

La última sección del TFM consta del modelado y utilización de varios algoritmos de **Machine Learning**, donde se perseguirá la obtención de predicciones lo más aproximadas posibles respecto a la variable dependiente **goodprice**, o en su defecto **LogGoodprice**.

Los datos utilizados para esta parte final provienen del dataset previamente analizado y tratado en la **Sección 2- Exploración General**. A partir de este, se llevarán a cabo ajustes tanto con **Modelos de Regresión Lineal** como **Modelos de Árbol** y **Redes Neuronales**, así como un **PCA** y procesos de tuneado de hiperparámetros mediante varias metodologías, realizando tareas de **Estandarización** de las variables previamente al ajuste de los algoritmos.

Así mismo, como herramienta de comparación entre modelos se definió la función **evaluadorDeModelos**, la cual imprime la dos métricas elegidas para analizar el desempeño (**MAE**, como función de pérdida, y **R²**, como 'accuracy') además de graficar la relación lineal entre valores pronosticados vs reales y el valor de los residuos de cada predicción.

### **Baseline con Regresión Lineal y Lasso**

Como Baseline se decidió utilizar una **Regresión Lineal Múltiple** ya que la sección de análisis exploratorio se centró en estudiar las relaciones lineales entre las variables independientes y la dependiente, además de ser un algoritmo bastante rápido y sencillo de aplicar así como de comprender su funcionamiento.

### **Regresion Lineal Múltiple y Lasso**

Un primer ajuste del modelo nos revela un **Coeficiente de Determinación** de 66% y un **Mean Squared Error** de 0.351 tanto en Train como en Test, un buen punto de partida desde el que mejorar nuestras predicciones. 

![](/imagenes/RegresionLinealBase1.png?raw=true)

Haciendo uso del evaluador de modelos observamos como la Regresión Lineal Múltiple tiende a subestimar los precios a medida que nos acercamos a cotas más altas, mientras que, aunque en menor medida, encontramos precios ligeramente sobrestimados entre los valores reales más bajos. Un rápido vistazo a los errores de predicción muestra la existencia de posibles datos atípicos u **outliers** que no fueron detectados en la fase de Exploración, no obstante, más allá de este problema los residuos se distribuyen normalmente alrededor del valor 0.

![](/imagenes/RegresionLinealBase2.png?raw=true)

Como siguiente paso, se añadió a la función de coste de la regresión, ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20J%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%5Csum_%7Bi%3D1%7D%5Em%20%28y_%7Bi%7D-%20%5Chat%7By_%7Bi%7D%7D%29%5E2) una regurlarización L1 o Lasso, ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20&plus;%5Calpha%5Ccdot%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bj%3D1%7D%5En%7C%5Ctheta_j%7C) a fin de reducir el valor de los coeficientes y discriminar los coeficientes/atributos que no aportan apenas información a las predicciones del modelo. Un alfa curiosamente bajo es suficiente para detectar 9 atributos con pesos de valor igual a 0 manteniendo el mismo Coeficiente de Determinación (una pérdida de pocas décimas). 

![](/imagenes/RegresionLasso1.png?raw=true)

![](/imagenes/RegresionLasso2.png?raw=true)

La esquematización de los pesos de cada feature nos permite vislumbrar que las columnas eliminadas son aquellas que presentan colineanlidad entre ellas y no fueron eliminadas en su momento, es decir las distancias, así como otras que en principio ya sospechábamos poco importantes, como **latitude** (dada la disposición de la ciudad). Como era de esperar, atributos como **LogAccommodates** o la existencia de aire acondicionado en el listing (**Air conditioning**) presentan pesos positivos importantes en el modelo, mientras que **room_type_Private Room** o **room_type_Shared Room** tienen un peso negativo al ser contrapartida de **room_type_Entire Home**. Sorprendentemente, una gran porción de features relacionados con la Geoexploración juegan un papel relevante en las predicciones.

![](/imagenes/PesosRegresionLasso.png?raw=true)

A fin de reducir la dimensionalidad de los atributos eliminamos esos features de la matriz **X** y se volvió a ajustar la regresión lineal, logrando un R² prácticamente idéntico al conseguido con todos los atributos disponibles en el dataset, 66%. 

![](/imagenes/RegresionLinealFiltrada.png?raw=true)

![](/imagenes/RegresionFiltrada2.png?raw=true)

### **Regresión Lineal con PCA**

Como alternativa al **Feature Selection** anterior, ajustamos un **PCA** a las variables relacionadas con distancias para reducir la dimensionalidad de nuestro dataset. Este nos permitió más allá de esta reducción, la eliminación de la colinealidad en las variables que no se eliminaron en la fase de exploración (dadas las características del PCA).

```python
distancias = list(X_train.columns[X_train.columns.str.contains('dist')]) 
+ ['longitude', 'latitude']

X_trainpca = X_train[distancias]
X_testpca = X_test[distancias]
```

En este caso, seleccionamos de X_train, y X_test los atributos relacionados con las distancias. Escalamos la variables y realizamos el ajuste del PCA sin límite de componentes para analizar visualmente el número de atributos a seleccionar.

![](/imagenes/PCA.png?raw=true)

Una demostración gráfica del porcentaje de varianza de los datos acumulado por número de componentes, nos muestra que llegado el quinto componente se explica prácticamente el 99% de esta. Es por ello que volvemos a ajustar el modelo con tan sólo 5 componentes, eliminando 12 atributos que registran la poca varianza restante. Una vez hecho, eliminamos las columnas del **X_train** original, para posteriormente concatenar los 5 nuevos atributos obtenidos.

```python
pca = PCA(n_components = 5)
X_trainpca = pca.fit_transform(X_trainpca)
X_testpca = pca.transform(X_testpca)

X_train.drop(distancias, axis = 1, inplace = True)
X_test.drop(distancias, axis = 1, inplace = True)

atributosfin = X_train.columns.tolist() + ['PCA_Component_{}'.format(x) for x in range(1, X_trainpca.shape[1]+1)] # Esta variable se utiliza de referencia más adelante
X_train = np.concatenate((X_train, X_trainpca), axis = 1)
X_test = np.concatenate((X_test, X_testpca), axis = 1)

X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)
```

Volvemos a ajustar el modelo de Regresión Lineal Múltiple y observamos una caída del 5% del **Coeficiente de Determinación** respecto al ajuste inicial.

![](/imagenes/RegresionLinealPCA.png?raw=true)

![](/imagenes/RegresionLinealPCA2.png?raw=true)

Un nueva ojeada a los valores de los coeficientes manifiesta unos valores sorprendentemente bajos para los Componentes creados con el PCA. Por lo demás, los features presentan el mismo comportamiento observados en los ajustes anteriores.

![](/imagenes/PesosRegresionLinealPCA.png?raw=true)

No obstante, si llevamos a cabo una validación cruzada con todo el dataset observamos muy poca diferencia entre las dos alternativas, quedando esta nueva opción muy levemente por debajo a nivel de performance (Sin PCA obtenemos un 65.8  de media y con PCA 65.4), es por ello que optamos por las ventajas del **PCA** (eliminamos 13 features y aceleramos el ajuste del modelo) a pesar de que nos suponga un baseline más pobre en la parte de test. **Cabe destacar que para la validación cruzada  definimos una función propia para dadas las particularidades de nuestro enfoque**.

```python

def PCA_cross_validation(model, X, y, cv = 5, scoring = 'r2', standarization = True):
  kf = KFold(n_splits=cv, random_state=1997, shuffle=True)
  cvscore = []
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    distancias = list(X_train.columns[X_train.columns.str.contains('dist')]) + ['longitude', 'latitude']

    X_trainpca = X_train[distancias]
    X_testpca = X_test[distancias]

    sc = StandardScaler()
    X_trainpca = sc.fit_transform(X_trainpca)
    X_testpca = sc.transform(X_testpca)

    pca = PCA(n_components = 5)
    X_trainpca = pca.fit_transform(X_trainpca)
    X_testpca = pca.transform(X_testpca)

    X_train.drop(distancias, axis = 1, inplace = True)
    X_test.drop(distancias, axis = 1, inplace = True)

    X_train = np.concatenate((X_train, X_trainpca), axis = 1)
    X_test = np.concatenate((X_test, X_testpca), axis = 1)
    
    if standarization == True:
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)

    if scoring == 'r2':
      model.fit(X_train, y_train)
      cvscore.append(r2_score(y_test, model.predict(X_test)))

    elif scoring == 'mae':
      model.fit(X_train, y_train)
      cvscore.append(mean_absolute_error(y_test, model.predict(X_test)))

  return np.array(cvscore)

```

*Al decidir optar por la alternativa del PCA, definimos la función PCA_train_test_split para reducir la dimensionalidad y agilizar los siguientes ajustes y etapas del modelado.*

### **Algoritmos basados en Árboles**

#### **Comparativa CatBoost y XGBoost**

Existen varios modelos en estado del arte en cuanto a algoritmo de árboles. Particularmente, se escogió dos de las opciones estrella actualmente, **XGBoost** y **CatBoost**, ya que al construir árboles de forma secuencial, cada nuevo árbol ayuda a corregir los errores cometidos por el anterior.
XGBoost, a pesar de ser un algoritmo algo más antiguo que CatBoost sigue siendo un referente en las competiciones en páginas web como [Kaggle](https://www.kaggle.com/competitions), unos de los algoritmos más populares para aplicar a datos tabulares. No obstante, CatBoost presenta ciertas mejoras que lo hacen una alternativa bastante aconsejable, menor tiempo de entrenamiento, manejo de las variables categóricas y missing values. Con el objetivo de escoger una de las dos opciones, se determinaron manualmente los hiperparámetros hasta encontrar dos predicciones lo más acertadas posibles.

![](/imagenes/XGboost.png?raw=true)


![](/imagenes/XGBoost2.png?raw=true)

Aunque a priori el modelo **CatBoost** se mostraba como favorito en esta primera comparación, **XGBoost** demostró tener mayor facilidad para obtener predicciones mucho más acertadas tras varias pruebas y cambios manuales de hiperparámetros, llegando a superar en más de un 3% el performance en el R² de test (93.04 de Test frente al 96.7 de Train). Entre las ventajas de CatBoost, dado nuestro enfoque con los datasets en fases anteriores, la única interesante residía en la velocidad de ajuste, de lo cual no encontramos una mejora tan significativa como para considerarla relevante. Es por ello que se optó por XGBoost como modelo de árboles a optimizar. 

![](/imagenes/Catboost.png?raw=true)

![](/imagenes/CatBoost2.png?raw=true)


#### **Optimización de Hiperparámetros de XGBoost**

Tras haber determinado XGBoost como la alternativa más viable, se procedió a realizar el método de optimización conocido como **Coarse to Fine Tuning** para el tuneado de hiperparámetros. Esta metodología se basa en combinar el **Random Search** y el **Grid Search**, el primero utilizado para reconocer zonas 'prometedoras' donde hacer un búsqueda más exhaustiva de combinaciones de hiperarámetros, de lo cual se encarga el segundo.

En primer lugar, el Random Search realiza 10 **cross validations** combinando aleatoriamente los hiperparámetros **learning rate**, **gamma**, **max depth** y **min child weight**, los más recomendados para tunear. Como resultado, observamos que valores muy altos de gamma perjudica a test más que lo ayuda a pesar de ser un regularizador, min child weight no tiene importancia alguna en el score final, mientras que valores alrededor de 0.16 y 25 son bastante óptimos para el **learning rate** y para el **max depth** respectivamente.

```python
xgbcv = xgb.XGBRegressor(seed=1997, n_jobs = -1, verbose = 10, verbosity = 1, n_estimators = 50) 

param_dist = {"learning_rate": sp.stats.uniform(0.1, 0.5),
              "reg_lambda": sp.stats.uniform(0.3, 1.5),
              "max_depth": range(20, 30),
              "min_child_weight": range(1,10)}

rs = RandomizedSearchCV(xgbcv, param_distributions=param_dist, cv = 5,
                        scoring='neg_mean_absolute_error', n_iter = 10, verbose = 10)

rs.fit(X_train, y_train)
```

Una vez encontrado un rango de hiperparámetros viable, se establecen valores dentro de este para aplicar un **Grid Search** satisfactoriamente. Esta vez solo se utiliza un **3-fold cross validation** debido a cuestiones de velocidad de ajuste y recursos disponibles.

```python
xgbcv = xgb.XGBRegressor(seed=1997, n_jobs = -1, verbose = 10, verbosity = 1, n_estimators = 50, min_child_weight=5)

gparameters = {'learning_rate': [0.13, 0.17, 0.21], 'max_depth': [25, 27, 30],
               'reg_lambda': [0.5, 0.9, 1.3]}

GridSearch = GridSearchCV(xgbcv, param_grid = gparameters, scoring='neg_mean_absolute_error',
                          verbose = 10, cv = 3, iid = True)

GridSearch.fit(X_train, y_train)
```

El GridSearch nos deja con un modelo bastante similar al que de por sí ya habíamos obtenido en la comparativa entre CatBoost y XGBoost, esta vez alcanzando un Coeficiente de Determinación del 97%, lo cual es comprensible dado que de base ya tenemos un score muy difícil de superar.

Tras finalizar el Grid Search y por tanto el proceso de **Coarse to Fine Tunning**. optamos por la **Optimización Bayesiana**, una metodología mucho más eficiente utilizando un modelo probabilístico de la función de coste (concretamente hablamos de un proceso Gaussiano), el cual va actualizando los parámetros de forma automática, por lo que podríamos obtener unos parámetros mejor optimizados y un modelo más acertado en sus predicciones.

Para la implementación de este tipo de optimizadores debemos crear una función objetivo (en esta caso de coste) que llamaremos **crossVal_score**, siendo esta una validación cruzada de 3-Folds con función de pérdida **MAE**.

```python
domain = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 0.3)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (30, 50)},
        {'name': 'reg_lambda', 'type': 'continuous', 'domain': (0, 2)}]
 
def crossVal_score(parameters):
    parameters = parameters[0]
    score = -cross_val_score(
                xgb.XGBRegressor(n_estimators = 50,
                              learning_rate=parameters[0],
                              gamma=parameters[1],
                              max_depth=int(parameters[2]),
                              reg_lambda = parameters[3],
                              n_jobs = -1,
                              verbose = 10), 
                X_train, y_train, scoring='neg_mean_absolute_error', cv = 3).mean()
    score = np.array(score)
    return score


optimizer = BayesianOptimization(f=crossVal_score, 
                                 domain=domain,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 exact_eval=True, 
                                 maximize=False)

optimizer.run_optimization(max_iter=5)
```

![](/imagenes/XGBoostBayes.png?raw=true)

El modelo optimizado a través del método bayesiano nos dejó el mejor ajuste que se pudo encontrar, alcanzando casi la perfección en train, **99.93%**,  pero quedando un poco por detrás en la evaluación de test, **97.24%** (overfitting que no hemos sido capaces de evitar en pro de minimizar la función de pérdida en la medida de lo posible). Una validación cruzada de 5-folds nos dejó de media un **97.3%** de R².

![](/imagenes/XGBoostBayes2.png?raw=true)

Con el objetivo de asegurar la interpretabilidad del algoritmo de **XGBoost**, utilizamos los valores **Shap (Shapley Additive exPlanations)**, es decir, los valores medios de la contribución marginal a través de todas las permutaciones posibles, ya que la *interpretabilidad global* del algoritmo nos permite mostrar cuanto contribuye cada uno de los features (y no solo la importancia del feature sino su influencia positiva o negativa con la predicción).

![](/imagenes/PesosXGBoost.png?raw=true)

La importancia de los features en algoritmos basados en árboles nos indica el valor de cada uno de los features utilizados en las predicciones (ya que hablamos de contribuciones marginales, cuanto más se usa un atributo en la construcción secuencial de árboles, mayor será su importancia). A pesar de que tan sólo graficamos 30 de los 60 atributos disponibles, es evidente la tendencia a valores muy próximos a cero a partir de la última feature graficada, revelando la posibilidad de eliminar prácticamente la mitad de columnas de nuestro dataset sin perjudicar en gran medida las predicciones de nuestro modelo.

Entre los 7 atributos más importantes encontramos:

**Private Room**: La variable dummy que indica si el alojamiento es una habitación privada o no. Desde un principio esta variable era una evidente pieza clave, ya que la mayoría de los listings se dividen en Habitaciones Privadas y Casa Enteras, siendo este primero la alternativa de menor precio.

**LogAccommodates**: El atributo de huéspedes en escala logarítmica. Obviamente, cuanto mayor el número de huéspedes, mayor el precio por noche, como ya se observaba analizando las relaciones lineales.

**minimum_nights**: Aunque en un principio no presentaba una relación lineal muy evidente, ya se observó en el baseline su influencia negativa en el precio, suponemos que alquileres de mayor duración son menos atractivos y repercuten en unos precios menores por parte de lo host.

**extra_people**: Coste del alojamiento añadido por cada huésped más allá del establecido en **Accommodates**. De nuevo, esta variable es obviamente importante en la determinación del precio.

**Air Conditioning**: Variable dicotómica que indica la existencia o no de aire acondicionado en el alojamiento. Juega un papel muy importante en la determinación de precios tanto en los modelos lineales como de árboles, probablemente relacionado con la estacionalidad de la subida de precios en verano en los que tener A/C es un valor añadido buscado por los usuarios.

**PCA_component_1**: El primer componente que se obtiene del PCA sobre las variables de distancias. No volvemos a ver el siguiente componente hasta pasados unos cuantos atributos, por lo que nos preguntamos si ajustando de nuevo sin PCA las variables de distancia tendrían una influencia mayor.

**month**: De entre todas las variables esta es la más destacable de todas, ya que se demuestra como, a diferencia de los modelos lineales, el **XGBoost** ha sido capaz de capturar la estacionalidad que mostrábamos en la Fase de Exploración y el valor que tiene tener en cuenta el mes en el momento de determinar un precio.

![](/imagenes/ShapXGBoost.png?raw=true)

Un gráfico más elaborado nos permite confirmar lo que se comentaba anteriormente *(se realizó con una muestra aleatoria de 10000 samples debido a capacidad computacional)*. El eje horizontal muestra si el efecto del feature para esa predicción fue positivo (aumento el valor de la predicción) o viceversa, mientras que el color muestra el valor del feature en dicha predicción (para variables dicotómicas azul implica un valor de 0 y el rojo de 1). 

Advertimos como los valores de Private Room positivo empujan los precios hacia abajo confirmando nuestra hipótesis anterior, además de la relación positiva entre Accommodates y Price, así como la negativa entre minimum_nights y price. 

Features como **month** demuestran como valores intermedios (6, 7 y 8, es decir, meses de Verano) en su mayoría "empujan las predicciones hacia arriba" mientras que valores extremos relativos a los meses de Invierno por norma general tienen el efecto contrario. Por otra parte, **PCA_Component_1** recoge como esperábamos el comportamiento propio de variables relacionadas con distancias, menor distancia mayor precio.

### **Redes Neuronales**

#### **Creación de Redes Neuronales**

Por último, abordamos esta fase de modelado con redes neuronales. Concretamente, utilizaremos **Artificial Neural Networks** de poco tamaño (no más de 4/5 fully connected layers de 256/512 neuronas por capa) y jugaremos con las funciones de activación para encontrar una arquitectura que pueda competir con el XGBoost. Para la compilación de las redes utilizamos de nuevo el **MAE** como función de pérdida, así como el optimizador **ADAM** (el que consideramos más completo dadas las alternativas que disponemos).

Como primera arquitectura se creó una **Shallow Neural Network** de cuatro capas ocultas (128 nodos en las capas más cercanas a las de Input y Output, 256 para el resto) en las que utilizamos una función sigmoide de activación para las tres primeras y un **REctificador Lineal Unitario (RELU)** para la última. La capa del output utiliza una función de activación **Lineal** (no realiza ninguna transformación), ya que tratábamos un problema de regresión.
 
![](/imagenes/RedNeuronal1.png?raw=true)

Este primer ajuste exhibió unos resultados prometedores. Un **91.5%** en Test y **92.75%** en Train, así como una reducción del **MAE** bastante acorde entre los dos, demuestra una buena generalización por parte del modelo, por lo que la **regularización** no parecía precisa.

![](/imagenes/EvaluaRedNeuronal1.png?raw=true)

A fin de lograr reducir aún más el **MAE** y aproximar la performance al **XGBoost**, aumentamos la complejidad del modelo añadiendo una capa más en la arquitectura. Cambiando manualmente las funciones de activación encontramos que una combinación de **tangentes hiperbólicas** y **RELU**s presenta un mejor ajuste qu el anterior.

![](/imagenes/RedNeuronal2.png?raw=true)

Manualmente logramos mejorar el ajuste unas décimas. Sin embargo, si comparamos los resultados con el XGBoost observamos que las redes neuronales no llegan a alcanzar el mismo perfomance, aunque presentan un menor overfitting que dicho modelo. Por tanto, se plantea la posibilidad de que una Red Neuronal Artificial presente la oportunidad de conseguir mejores predicciones manteniendo un menor overfitting que el XGBoost, es decir, un modelo mejor generalizado. Es por ello que damos paso a la Optimización de Redes Neuronales.

![](/imagenes/EvaluaRedNeuronal2.png?raw=true)


#### **Optimización de Redes Neuronales**

Esta vez, en la búsqueda de los hiperparámetros óptimos, se utiliza directamente la **Optimización bayesiana** a través del nuevo paquete de Keras conocido como **Keras-Tuner**. A partir de la definición de una clase y la creación de un objeto (el **HyperModel**) conseguimos hacer uso de la función con alias **KerasBayesianOptimization**.

```python
class ANNRegressor(HyperModel):
  def __init__(self, input_shape):
    self.input_shape = input_shape

  def build(self, hp):
    model = keras.Sequential()
    
    model.add(
        Dense(units = hp.Int('units_1', min_value = 32, max_value = 256,
                             step = 16, default = 128),
              activation = hp.Choice('dense_activation_1', values = ['relu', 'sigmoid', 'tanh'],
                                     default = 'tanh'))
        )
    
    model.add(
        Dense(units = hp.Int('units_2', min_value = 32, max_value = 512,
                             step = 16, default = 256),
              activation = hp.Choice('dense_activation_2', values = ['relu', 'sigmoid', 'tanh'],
                                     default = 'relu'))
        )
    
    model.add(
        Dense(units = hp.Int('units_3', min_value = 32, max_value = 512,
                             step = 16, default = 256),
              activation = hp.Choice('dense_activation_3', values = ['relu', 'sigmoid', 'tanh'],
                                     default = 'tanh'))
        )
    
    model.add(
        Dense(units = hp.Int('units_4', min_value = 32, max_value = 256,
                             step = 16, default = 256),
              activation = hp.Choice('dense_activation_4', values = ['relu', 'sigmoid', 'tanh'],
                                     default = 'relu'))
        )
    
    model.add(
        Dense(units = 1, activation = 'linear')
      )
    
    model.compile(optimizer = keras.optimizers.Adam(
                  hp.Float('learning_rate', min_value = 1e-3, max_value = 1e-2,
                           default = 8e-3, step = 1e-3)),
                  loss = 'mae', metrics = ['mae'])
    return model

hypermodel = ANNRegressor(input_shape = (X_train.shape[1], ))
```

Una vez definida la clase y creado el objeto, la función KerasBayesianOptimization nos permite crear una función de hyperparameter tunning cuyo objetivo es minimizar el **MAE** con la mejor combinación posible de hiperparámetros. En la parte inferior podemos observar el resumen de nuestro tuner:
- Se seleccionará un número de nodos dentro del rango de 128 a 256 para las capas más cercanas a la de **Input** y **Output** y de 256 a 512 para las interiores, con saltos de 16 neuronas.

- Para cada una de las capas se escogerá entre las tres **funciones de activación** que mejor han funcionado en la sección anterior: **RELU, SIGMOIDE y TANGENTE HIPERBÓLICA**.

```python
tuner = KerasBayesianOptimization(
    hypermodel,
    max_trials = 20,
    objective = 'mae',
    executions_per_trial = 1,
    directory='TFMoptimization',
    project_name='AirBNB'
)
```

![](/imagenes/Summary.png?raw=true)

Tras un largo proceso de tuneado el resumen de resultados nos muestra como mejor candidato una combinación de tangentes hiperbólicas y rectificadores lineales en una arquitectura con el número máximo de nodos que se había planteado en un principio excepto para la capa conectada con la del output **(units_5)**. Una vez expuesto el mejor modelo, lo ajustamos a nuestros datos para elaborar las predicciones.

![](/imagenes/RedNeuronalOpt.png?raw=true)

Últimando esta fase alcanzamos una Red Neuronal bastante próxima a nivel de performance al **XGBoost**, con un **97.69%** de Coeficiente de Determinación en Train y **95.45%** en Test. Intentos de optimización con redes más complejas llevaron a peores resultados, por lo que se desestimó esta alternativa. 

![](/imagenes/EvaluaciónRedNeuronalOpt.png?raw=true)

### Discusión final de modelado

Acabado ya todo el proceso se dio paso a un pequeña comparativa entre los dos modelos predictivos principales de este proyecto.

En primer lugar, se aborda este apartado final con una comparativa de la evolución de precios pronosticada, se escoge de forma aleatoria 2 muestras (varias veces, enseñamos sólo una de ellas por resumir) de las que son graficadas su precio real y la predicción de cada uno de los modelos **[ Se debe tener en cuenta que cada predicción es independiente (en la parte A de la exploración general se explica el porqué de este enfoque) por lo que esta visualización se utiliza para tener una visión más global del comportamiento del algoritmo, ya que no podemos diferenciar entre predicciones de train (más precisas) y test. ]** En caso del **Baseline** queda demostrada su incapacidad para registrar la influencia de la estacionalidad en la evolución de los precios, así como la tendencia creciente que se presentaba a lo largo de los años. Además, encontramos mayor dificultad (y error) a medida que los precios a predecir son más elevados. 

![](/imagenes/LinearRegresionFinal.png?raw=true)

Con la **Red Neuronal** ya encontramos una mejora bastante significativa con respecto al **Baseline**. Esta ya logra registrar los efectos de la estacionalidad así como la tendencia de cada año, además de manejar las predicciones de precios sin mayor problema que con el resto. No obstante, los cambios más bruscos de precio, y como contrapartida los precios que se mantienen constantes a lo largo del tiempo, no logra acabar de registrarlos correctamente.

![](/imagenes/NNfinal.png?raw=true)

Finalmente, el modelo estrella de este proyecto, el **XGBoost** supera completamente los problemas para registrar la influencia del mes y el año a la hora de predecir los precios (como ya se observa con los Shap Values). En estos gráficos ya se vislumbra más fácilmente las predicciones provenientes de train en los puntos donde se solapan las curvas.

![](/imagenes/XGBoostFinal.png?raw=true)

Para esclarecer un poco mejor la captura de la influencia del feature **month** graficamos la media de precios de todos los precios pronosticados, así como el real. Queda definitivamente demostrado como el modelo de **Baseline**, la Regresión Lineal Múltiple es incapaz de registrar con éxito el peso que tiene el mes en la determinación de precios.

La Red Neuronal consigue captar la influencia de esta un poco mejor, pero no llega a registrar correctamente los picos ni los valores elevados en el mes de Febrero y Marzo. No obstante, los valores tan altos en meses como Febrero puede deberse a la existencia de outliers no detectados que han sido sobreajustados por el **XGBoost**, por lo que la Red Neuronal en realidad puede estar evitando este tipo de errores.

![](/imagenes/PredictMeses.png?raw=true)

Por otro lado, si hablamos de la variable **year**, observamos que tanto el XGBoost como la Red Neuronal han sido capaces en mayor o menor medida de capturar su influencia. El **Baseline**, a pesar de que ha podido capturar cierta influencia, su performance sigue siendo muy pobre en este aspecto también.

![](/imagenes/PredictAño.png?raw=true)

Un estudio más detallado de las diferencias entre predicción y valor real nos revelan más información de utilidad. El gráfico inferior visualiza la relación entre el precio real (eje horizontal), el precio pronosticado por **XGBoost** (eje vertical) y la diferencia entre el precio predicho por la **Red Neuronal** y el precio real (color y tamaño de los puntos). Como podemos observar, la Red Neuronal tiende a tener más problemas en la predicción de precios altos que el **XGBoost**, sobretodo a partir de los 800€.

Por otro lado, detectamos que establecer una cota de 1200€ como tope superior en la Fase de Exploración General permitió colarse unos cuantos outliers producidos por la introducción errónea de los precios por parte de la web Inside AirBNB, como se observa en los círculos rojos a lo largo de los valores de 1000 en el eje y (aunque algunos de ellos XGBoost los predice sin mucho margen de error respecto al real, por lo que en ese caso deberiamos comprobrar si es debido a un sobre-ajuste u **overfitting** en Train o de verdad son predicciones acertadas).

![](/imagenes/Errores.png?raw=true)

Destripando un poco el gráfico anterior podemos delimitar tres áreas claras:

**El Área Outlier**, capturando precios con una diferencia superior a los 100/120€, muestra precios sospechosamente distintos a las predicciones tanto del XGBoost como de la Red Neuronal. Los puntos existentes dentro de esta zona deberían recibir un análisis más detenido a fin de determinar el origen de tan gran diferencia y, en caso de ser erratas a causa de un mal registro, corregidos o bien eliminados para un nuevo ajuste de nuestros modelos.

**El Área Ineficiente** recoge los precios con diferencias positivas entre 20 y 100 euros. Seria recomendable que los host con listings situados en estas áreas replantearan la asignación de su precio, ya que las predicciones plantean unos valores inferiores para los pisos de sus características si pretenden mejorar la eficiencia de ocupación en sus alquileres. Por otra parte, los usuarios en busca de apartamentos deberían evitar estos alojamientos a toda costa.

**El Área Chollo** presenta los precios con valores pronosticados superiores a los precios reales de como mínimo 20€. Llamamos chollo a este área desde el punto de vista de un usuario no propietario, este tipo de consumidor deberia aprovechar este tipo de alojamientos a fin de pagar un precio inferior al valor real del alojamiento. Propietarios con listings en este área deberían elevar sus precios a fin de maximizar el beneficio lo máximo posible.

Los puntos inferiores al Área Chollo los consideramos también outliers, a pesar de ser un número muy reducido, los cuales deberían ser revisados de nuevo al igual que los que se sitúan en el Área Outlier.

![](/imagenes/erroresdestripado.png?raw=true)

##  Visualización y Dashboard

Finalizado el proceso de limpieza, exploración y modelización, la visualización y comunicación de resultados se presenta como la última fase del proyecto, un punto clave poder trasladar al usuario final el conocimiento e información obtenida durante el tratamiento de los datos a lo largo de las fases anteriores, utilizando siempre un contexto visual para facilitar su entendimiento.

Con el objetivo de acercar a los usuarios a una interfaz interactiva en la que puedan comprobar por cuenta propia los resultados presentados, se realizó una serie de **Dashboards** en **Tableau**. Con él pretendemos que el usuario tenga la posibilidad de contextualizar los datos que han sido incorporados a lo largo de todo el proyecto, e introducidos en los modelos presentados en la fase de modelado.

En caso de que se quiera interactuar por el dashboards, seran necesarios dos pasos, la de la descompresión de los datos en formato de hyper, así como la incorporación de la carpeta denominada landmarks en el repositorio local de Tableau correspondiente a los iconos usados en una parte de la visualización. Tanto los datos como la carpeta con los iconos han sido incorporados en el proyecto de GitHub.

Un rápido vistazo al dashboard base nos revela una división en 3 bloques principales.

![](/imagenes/TFMAirBNBTableau0.png?raw=true)

**El primero de ellos** presenta los datos obtenidos de la web de Inside Airbnb. Concretamente, las variables de todos aquellos registros con influencia relevante en la predicción de los modelos: 

En la parte superior, a modo de presentación podremos observar el número de pisos únicos dados de alta en AirBNB durante los años 2017 a 2020, así como la evolución de los precios medios a lo largo del tiempo.

![](/imagenes/SuperiorTableau.gif)

Bajo esta introducción, se muestra una selección de las principales características de los alojamientos, permitiéndonos analizar mediante filtros como cambia la evolución de precios dependiendo de distintos atributos: Tipo de propiedad, tipo de alquiler (E.G. toda la propiedad o solo una habitación), y otros aspectos relevantes que durante el proyecto han resultado ser relevantes en las predicciones, además de ser las primeras a ser consultadas por los usuarios cuando navegas por webs como **AirBNB**, como el número de habitaciones, camas o baños.

Junto a esta hallamos un mapa sensible también a los filtros anteriormente nombrados, cuya influencia puede ser analizada desde dos vertientes, un mapa de distribución de puntos **uno a uno**, donde se puede filtrar por barrio y otras variables continuas, o un mapa de densidad en forma de **colmena**, con el que visualizar los puntos "calientes" o de mayor densidad de cada una de las composiciones elegidas por el usuario final. En la parte superior de los mapas, y tal como se muestra en el gif a continuación, encontramos un botón que nos permite cambiar la visualización en función del mapa que queramos analizar.

![](/imagenes/MapaTableau.gif)

![](/imagenes/TFMAirBNBTableau1.png?raw=true)
![](/imagenes/TFMAirBNBTableau1b.png?raw=true)

**Un segundo bloque** sirve para complementar las características principales del alojamiento. En ella encontramos todas aquellas variables que aportan un valor añadido al precio de los alojamientos, como son los servicios que ofrecen, desde la existencia de **Aire Acondicionado** hasta si se **Permite Fumar**, el tipo de póliza de cancelación ofrecida así como el mínimo de noches necesarias para poder alquilar. Una vez más, estas tres alternativas también son utilizadas como filtro para la evolución de precios y mapas tal y como se muestra en el gif que podemos ver a continuación.

![](/imagenes/Bloque2Tableau.gif)

Destacar por último de este segundo bloque, que la visualización del mapa depende de un selector en la parte superior en la que poder escoger tres variables diferentes, cleaning fee, extra people o number of reviews, para su análisis.

![](/imagenes/TFMAirBNBTableau2.png?raw=true)

Finalmente, **en la tercera parte del Dashboard base**, encontramos una visualización que pretende mostrar el tipo de datos obtenidos a raíz del procesado de datasets externos, los cuáles nos han permitido nutrir de información adicional los modelos de predicción.

Específicamente podremos navegar por dos mapas, uno a raíz de datos extraídos de Open Data Barcelona a nivel de transportes y servicios, donde mostramos los servicios y oferta turística alrededor de cada uno de los alojamientos con un density map. Un parámetro en la parte superior del mapa nos permite cambiar la visualización en base al transporte o servicio que queramos visualizar y analizar.

![](/imagenes/Mapa3Tableau.gif)

Por otro lado encontramos el mapa donde podemos ver los landmarks de Barcelona. Esta visualización ha sido posible gracias al dataset de fotografías geolocalizadas de una API de Flickr y la aplicación del modelo de aprendizaje no supervisado de clusterización. El objetivo de este mapa es más ilustrativo y lo que pretende es aportar un contexto más visual de los puntos más relevantes con su icono asociado.

![](/imagenes/TFMAirBNBTableau3.png?raw=true)


La contextualización visual en el dashboard base de los diferentes datasets utilizados para el abordaje del objetivo principal del proyecto, el de desarrollar un modelo predictivo de los precios a través del uso de machine learning y deep learning, nos lleva a **una segunda parte ilustrativa del trabajo realizado, la visualización de los resultados obtenidos de los diferentes modelos de predicción trabajados durante la parte de modelado.**

A nivel de navegación, esta segunda parte tiene una primera parte común a las 4 visualizaciones que nos permite interacturar con los resultados obtenidos después de la fase de modealdo. En todas ellas encontraremos:

- 4 botones que nos permiten navegar por las 4 visualizaciones donde se muestran los resultados de los diferentes modelos trabajados en la fase de modelado.
- Dos selectores, uno que nos permitirá seleccionar el id de un alojamiento concreto si queremos ver los resultados de los modelos de forma individual o por un grupo concreto de identificadores, y el otro que nos da la opción de seleccionar si queremos visualizar los 4 años de histórico de datos o uno específico.

Empezamos por la primera imagen que podeis ver a continuación, la primera visualización en la que el primer gráfico que encontramos es el resultado de la predicción de los 4 modelos de forma comparativa.

Un segundo gráfico nos muestra un análisis más detallado de las diferencias entre predicción y valor real. La visualización muestra la relación entre el precio real (eje vertical) y el precio pronosticado (escogido en el primer parámetro superior del gráfico y eje horizontal) y, la leyenda de colores y tamaño de los puntos muestra la diferencia entre el precio predicho por el modelo seleccionado (en el segundo selector superior del gráfico) y el precio real.

Una visualización similar a la ya comentada en la fase de modelado pero que hemos considerado interesante incorporar en el dashboard para que el usuario pueda interactuar y así poder entender las diferencias entre las predicciones de los diferentes modelos.

![](/imagenes/TFMAirBNBTableau4.png?raw=true)

Una segunda visualización nos muestra los resultados del modelo de la Regresión lineal, la que hemos considerado nuestro modelo de baseline.

La estructura es muy similar a lo que ya hemos comentado anteriormente, lo que cambia es la visualización del primer gráfico en el que se mantienen únicamente los resultados del precio real y de la predicción de los precios obtenidos de la Regressión Lineal, y por otro lado, de cara a poder trasladar una mejor explicación del análisis obtenido del modelo, hemos incorporado el resultado de los coeficientes de las variables donde se puden observar cuáles han determindado en mayor y menor medidad la variabilidad de la predicción del precio.

![](/imagenes/TFMAirBNBTableau5.png?raw=true)

Llegamos a la tercera pestaña de visualización de esta segunda parte en la que los datos que se muestran son los resultados obtenidos del modelo de XGBoost. De la misma forma que anteriormente, en el primer gráfico se puede observar la comparativa entre el precio real y los obtenidos de la predicción con el modelo de XGBoost y por otro lado, con el objetivo de asegurar la interpretabilidad del algoritmo, se pueden visualizar los valores Shap (Shapley Additive exPlanations), es decir, los valores medios de la contribución marginal a través de todas las permutaciones posibles, de cara a mostrar cuanto contribuye cada una de las variables.

![](/imagenes/TFMAirBNBTableau6.png?raw=true)

Finalmente la última visualización muestra los resultados obtenidos en el último modelo trabajado, la Artificial Neural Network. Así como en las anteriores explicaciones una vez más dejamos en el primer gráfico la comparativa entre el precio real y en este caso con los resultados de los precios predichos obtenidos por la ANN.

En este caso, como no hay la opción de poder visualizar la importancia de las features en la contribución de la determinación de la predicción de los precios, podemos observar la estructura utilizada de la ANN

![](/imagenes/TFMAirBNBTableau7.png?raw=true)

Con todo, consideramos que es un Dashboard muy completo en el que quedan reflejadas las diferentes etapas del proyecto, en la que el usuario puede interacturar, analizar y sacar conclusiones y, que nos ayuda a comunicar de forma visual, óbviamente, todo nuestro trabajo.


# Conclusiones y Mejoras

Tras un largo proceso de limpieza, análisis y modelado de los datos hemos logrado un modelo capaz de predecir prácticamente un **97%** de la varianza de los precios con apenas un **0.07%** de **Mean Absolute Error**. El **XGBoost** se ha mostrado imbatible frente al gran alcance del campo del **Deep Learning**, lo cual demuestra por qué sigue siendo el rey de las competiciones de Kaggle a pesar de la existencia de alternativas más modernas y a priori más "potentes".

Dado los resultados de nuestro proyecto nos encontramos muy poco margen de mejora dados los datos que estamos trabajando actualmente, dentro del cuál costaría mucho esfuerzo superar por pocas décimas el Coeficiente de Determinación ya obtenido. No obstante, podemos plantear una serie de mejoras que aporten valor añadido:

- Mejorar la captura de datos de entrada desde el principio. A pesar de que la web Inside AirBNB realiza un trabajo excepcional de *Web Scrapping*, a menudo nos hemos encontrado con precios mal registrado (a pesar de haber realizado labores de limpieza con anterioridad). Realizar Web Scrapping de primera mano nos aseguraría unos registros más fiables (supondría "scrapear" datos cada mes para tener datos mensuales sobre el comportamiento de precios).

- No se ha trabajado con **NLP** en pro de concentrar los esfuerzos en las variables **geoespaciales**. Dado que el dataset original cuenta con una serie de columnas relacionadas con descripciones, situación y normas del alojamiento, la búsqueda de palabras clave, aplicación de **Redes Neuronales** para análisis de sentimientos o la simple creación de features como la longitud de la descripción del listing puede aportarnos variables de influencia para futuras predicciones. Además, la disponibilidad de un dataset de **reviews** nos permitiría desarrollar un modelo en el que explicar como las opiniones de los usuarios pueden influenciar la evolución del precio de un alojamiento.

- El proyecto se ha basado únicamente en alojamientos de la ciudad de Barcelona. Una alternativa atractiva podría ser la ampliación del foco del modelo a los alojamientos de toda España, con todas las consecuencias en la calidad de la predicción que esto repercutiría.

- Existen features con bastante influencia que un propietario nuevo no posee (como el Número de Reviews recibidas). Eliminar este tipo de variables e intentar mantener el mismo nivel en las predicciones plantea un desafío interesante.

- Por último, el desarrollo de una aplicación web que permitiese introducir al usuario final los parámetros necesarios y este devolviese una predicción de los precios aconsejados, todo ello a través de un script de Python sería una forma óptima de hacer uso del proyecto en su totalidad.





