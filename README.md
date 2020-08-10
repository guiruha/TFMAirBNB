# AirBNB analysis and price prediction
## TFM / Master en data Science / Kschool
#### **Guillem Rochina Aguas y Helena Saigí Aguas**
#### 04/09/2020

# Introducción y Motivación

El sitio web AirBNB permite a propietarios de apartamentos convertirse en 'hosts' de la plataforma y ofertar sus propiedades, llamadas 'listings', online a fin de cobrar un alquiler vacacional. Determinar el precio del alojamiento puede ser una tarea difícil en ciudades con tanta competencia como Barcelona si lo que se busca es maximizar el beneficio/ocupación de tu listing. Además, a pesar de que ya existen plataformas como AirDNA ofreciendo asesoramiento en este aspecto, o bien son demasiado caros para el usuario medio u ofrecen predicciones poco convicentes.

![](/imagenes/AirBNBweb.png?raw=true)

Es por ello que el objetivo principal de este proyecto de TFM es el desarrollo de un modelo predictivo a través del uso de machine learning y deep learning, así como la limpieza y exploración de datos tanto de la web de AirBNB como de datos públicos de la ciudad de Barcelona.

De forma alternativa, se puede abordar este proyecto como un 'buscador de chollos', que permita a usuarios interesados en alquilar habitaciones en la ciudad de Barcelona acceder a 'listings' que a priori deberían tener unos precios más elevados debido a sus características y predicciones por parte del modello.

# Descripción de los Datasets

Los datos trabajados en este proyecto de TFM provienen de diversas fuentes, tratando desde datos propios de AirBNB como datasets de dominio público obtenidos de webs como Flickr y OpenData Barcelona.

## Dataset Principal

El dataset principal consta de csv y geojson obtenidos desde la web de [Inside AirBNB](http://insideairbnb.com/get-the-data.html), organización que se encarga de hacer web scrapping de todos los datos accesibles en la página web de AirBNB de forma mensual. Concretamente, los archivos utilizados constan de dos csv, **listings.csv** y **calendar.csv**, y un geojson **neighbourhood.geojson** que son presentados a continuación.

### Listings.csv

Siendo el archivo más importante de los tres, en este csv encontramos toda la información disponible en la página específica de cada uno de los listings, desde descripciones propias del host hasta cuantos servicios o comodidades adicionales ofrece el listing. A continuación se describen brevemente las principales columnas del dataset.

| Column        | Description          
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

El dataset de calendar nos proporciona información sobre los precios a nivel diario de cada uno de los listings (sin embargo se ha calculado la media mensual de cada listing a causa de capcadidad de almacenamiento y procesamiento), así como la disponibilidad diaria de cada listing, variable que utilizaremos para sustituir a las columnas availability_x debido a que nos fiamos más de la disponiblidad diaria que de los cálculos de las susodichas columnas.

### Neighbourhood.geojson

## Dataset de Flickr

## Datasets de Transportes

## Datasets de Sitios de Interés Turístico
 
# Desarrollo del Proyecto

##  Limpieza
##  Exploración Parte A
##  Geoexploración
##  Exploración Parte B
##  Modelado
