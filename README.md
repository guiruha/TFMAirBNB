# AirBNB analysis and price prediction
## TFM / Máster en Data Science / KSCHOOL
#### **Guillem Rochina Aguas y Helena Saigí Aguas**
#### 04/09/2020

# Introducción y Motivación

El sitio web AirBNB permite a propietarios de apartamentos convertirse en 'hosts' de la plataforma y ofertar sus propiedades, llamadas 'listings', online a fin de cobrar un alquiler vacacional. Determinar el precio del alojamiento puede ser una tarea difícil en ciudades con tanta competencia como Barcelona si lo que se busca es maximizar el beneficio/ocupación de tu listing. Además, a pesar de que ya existen plataformas como AirDNA ofreciendo asesoramiento en este aspecto, o bien son demasiado caros para el usuario medio u ofrecen predicciones poco convicentes.

![](/imagenes/AirBNBweb.png?raw=true)

Es por ello que el objetivo principal de este proyecto de TFM es el desarrollo de un modelo predictivo a través del uso de machine learning y deep learning, así como la limpieza y exploración de datos tanto de la web de AirBNB como de datos públicos de la ciudad de Barcelona.

De forma alternativa, se puede abordar este proyecto como un 'buscador de chollos', que permita a usuarios interesados en alquilar habitaciones en la ciudad de Barcelona acceder a 'listings' que a priori deberían tener unos precios más elevados debido a sus características y predicciones por parte del modelo.

# Descripción de los Datasets

Los datos trabajados en este proyecto de TFM provienen de diversas fuentes, tratando desde datos propios de AirBNB como datasets de dominio público obtenidos de webs como Flickr y OpenData Barcelona.

## Dataset Principal

El dataset principal consta de csv y geojson obtenidos desde la web de [Inside AirBNB](http://insideairbnb.com/get-the-data.html), organización que se encarga de hacer web scrapping de todos los datos accesibles en la página web de AirBNB de forma mensual. Concretamente, los archivos utilizados constan de dos csv, **listings.csv** y **calendar.csv**, y un geojson **neighbourhood.geojson** que son presentados a continuación.

### Listings.csv

Siendo el archivo más importante de los tres, en este csv encontramos toda la información disponible en la página específica de cada uno de los listings, desde descripciones propias del host hasta cuantos servicios o comodidades adicionales ofrece el listing. A continuación se describen brevemente las principales columnas del dataset.

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

El proyecto ha sido divido en cuatro partes, y por tanto, aunque no es necesario se recomienda ejecutarlos en ese orden para tener un compresión global:

##  Limpieza

[LINK A COLAB]

**INPUTS:** Listings.csv, Calendar.csv **OUTPUTS:** DatosLimpios.csv

La primera Fase de este proyecto consiste en la limpieza y análisis superficial de los datasets base para la evolución del TFM, listings.csv y calendar.csv. 

Un primer barrido de eliminación de columnas suprimió del proceso todas las variables relacionadas con urls, así como descripciones tanto del host como del alojamiento (se planteó el uso de NLP en estas columnas a fin de encontrar nuevos atributos útiles pero finalmente se decidió seguir un camino distinto y enfocar los esfuerzos en otras alternativas). Por otro lado, también fueron eliminadas columnas con más de un **60%** de Nulls dada su relativamente baja importancia y el riesgo a introducir un sesgo grande por medio de la imputación de valores (tanto predichos a través de modelos lineales como medianas o medias).

![](/imagenes/60Cleaning.png?raw=true)

La limpieza se desarrolla a continuación con el procedimiento habitual: eliminación de columnas poco útiles o bien repetidas, eliminación de filas repetidas o con datos anómalos, imputación de valores etc. Destacamos los procedimientos de limpieza más relevantes y menos comunes a continuación:

- **Variables categóricas**

Existen columnas categóricas con un gran número de clases muy similares entre sí, es por ello que a fin de reducir las dimensiones de nuestro dataset lo máximo posible, hemos generalizado todos los valores en el menor número de categorías posible. Ejemplo de ello es la variable **cancellation_policy** que ha sido generalizada a cuatro alternativas **(flexible, moderate, strict_less30, strict_30orMore)**.

![](/imagenes/CatCleaning.png?raw=true)

- **Variables string de precios**

Todas las columnas de precios del dataset se presentan con un símbolo de dólar al principio y con comas a partir de los millares **E.G. $1,200.00**. La limpieza de estas variables ha sido abordada a través del method chaining de varias funciones **replace** ,para la eliminación de los símbolos anteriormente mencionados, y la transformación de tipo string a tipo float (en las columnas que presentaban Null debido a su naturaleza, E.G. existen listings sin tarifa de limpieza y en vez de ser codificado con 0 se presenta como un Null, se ha imputado valores de 0€).

![](/imagenes/ColPrice.png?raw=true) 

- **Caso Especial: Amenities**

La columna amenities ha resultado ser un caso especial, ya que cada registro se presenta en forma de lista (con llaves **{}** en vez de corchetes **[]**) y además de ser de tipo string. Por ello, en primer lugar para visualizar lo comunes que son cada uno de los amenities entre todos los alojamientos se utiliza de nuevo el method chaining para tratar con el string y transformarlo realmente en una lista, a continuación mediante un diccionario y una serie de pandas logramos el objetivo de visualizar el porcentaje total de aparición de cada amenity.

![](/imagenes/Amenities.png?raw=true)

Una vez visualizados, se seleccionaron los que por ser no tan comunes y, bajo nuestro criterio, relevantes para un huésped consideramos utiles para la determinación de un precio superior del alojamiento respecto a los que carecen de estos servicios. En concreto seleccionamos el siguiente conjunto a través de la creación de variables dummy:

![](/imagenes/AmenitiesDummy.png?raw=true)

Finalmente, para la última parte de esta sección se procedió al tratamiento de los datasets de **calendar.csv**, calculamos las columnas **price_calendar** y **year_availability** a través de groupbys y finalizamos con el merge de calendar y listings para la creación de **DatosLimpios.csv**.

##  Exploración Parte A

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv **OUTPUTS:** Localizaciones.csv

Esta primera fase de Exploración General se centra en el análisis, limpieza y transformación de la variable dependiente, en este caso **goodprice** obtenida a partir de las medias mensuales de precios calculadas en la fase de limpieza.

##  Geoexploración

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv **OUTPUTS:** Distancias.csv, DistanciasTurismo.csv

##  Exploración Parte B

[LINK A COLAB]

**INPUTS:** DatosLimpios.csv, Distancias.csv, DistanciasTurismo.csv **OUTPUTS:** DatosModelar.csv

**Variables Numéricas**

**Variables Categóricas**

**Variables Dicotómicas**

**Landmarks**

**Transportes**

**Sitios Turísticos**

##  Modelado

[LINK A COLAB]

**INPUTS:** DatosModelar.csv **OUTPUTS:** xxxxxx

##  Visualización y Dashboard

[LINK A COLAB]

**INPUTS:** xxxx **OUTPUTS:** []

# Conclusiones y Mejoras
