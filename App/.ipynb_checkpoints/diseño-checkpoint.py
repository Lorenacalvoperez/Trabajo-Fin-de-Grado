from shiny import App, reactive, render, ui
import plotly.express as px
import pandas as pd
from io import StringIO
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import pandas as pd
import requests

#Carga datos
def cargar_datos():
    '''
    Solicitud de las URLS de los datos y metadatos.
    Realiza dos peticiones HTTP GET:
        1. A  DATA_URL, que devuelve un JSON con los valores.
        2. A  METADATA_URL, que devuelve un JSON con información estructural.

    Returns
    -------
        - datos : Contiene los valores del conjunto de datos..
        - metadata : Contiene la información estructural de los datos (dimensiones como años y países).
    '''
   
    datos = requests.get(DATA_URL).json()  # Datos de casos de Parkinson
    metadata = requests.get(METADATA_URL).json()  # Información adicional (años y países)
    return datos, metadata
#Procesar datos
def procesar_datos(data_response, metadata_response):
    '''
    Procesa la respuesta de datos y metadata para extraer la información necesaria.

    Parámetros:
    ----------
    data_response : dict
        Diccionario JSON obtenido de la API que contiene los valores numéricos.
    
    metadata_response : dict
        Diccionario JSON obtenido de la API que contiene información adicional como los años disponibles
        y los países asociados a los datos.

    Returns:
    --------
        - valores (list): Lista de valores numéricos extraídos del campo "values".
        - años (list): Lista de identificadores de años extraídos de la metadata.
        - paises (dict): Diccionario que mapea IDs de país a nombres de país, obtenido de la metadata.

    Excepciones:
    -----------
    ValueError:
        Se lanza si no se encuentran las claves esperadas ("values", "years", "entities") en las estructuras 
        JSON correspondientes.
    '''
    # Extraer los valores numéricos (dependiendo de la estructura)
    if "values" in data_response:
        valores = data_response["values"]
    else:
        raise ValueError("No se encontró la clave 'values' en los datos")

    # Extraer la lista de años disponibles (independientemente de la estructura específica)
    if "dimensions" in metadata_response and "years" in metadata_response["dimensions"]:
        años = [item["id"] for item in metadata_response["dimensions"]["years"]["values"]]
    else:
        raise ValueError("No se encontró la clave 'years' en la metadata")

    # Extraer los países o entidades desde la metadata (independientemente de la estructura específica)
    if "dimensions" in metadata_response and "entities" in metadata_response["dimensions"]:
        paises = {item["id"]: item["name"] for item in metadata_response["dimensions"]["entities"]["values"]}
    else:
        raise ValueError("No se encontró la clave 'entities' en la metadata")
    
    return valores, años, paises
#Crear el DF

import pandas as pd

def crear_dataframe(valores, años, paises, nombre_columna):
    '''
    Crea un DataFrame de pandas combinando listas de valores, años y países.

    Esta función organiza los datos recibidos desde una API en un DataFrame estructurado,
    asignando los valores correspondientes a cada combinación de país y año. Si hay menos
    valores que combinaciones posibles, se rellena con None.

    Parámetros:
    ----------
    - valores : list
        Lista de valores numéricos, en orden secuencial.

    - años : list
        Lista de identificadores de años.

    - paises : dict
        Diccionario que mapea el ID del país al nombre del país, obtenido desde la metadata.

    - nombre_columna : str
        Nombre que se le dará a la columna de valores en el DataFrame resultante.

    Retorna:
    -------
    pd.DataFrame
        Un DataFrame con columnas: "Año", "País" y la columna con nombre "nombre_columna",
        que contiene los valores o "None" si no hay datos disponibles.
    '''
    # Crear la lista para almacenar las filas de datos
    datos_lista = []
    
    # Variable para iterar sobre los valores
    indice_valor = 0
    
    # Iterar sobre los países y años
    for id_pais, nombre_pais in paises.items():  # Iterar sobre los países
        for año in años:  # Iterar sobre los años disponibles
            if indice_valor < len(valores):
                # Si hay un valor disponible, añadirlo a la fila
                fila = {
                    "Año": año,
                    "País": nombre_pais,
                    nombre_columna: valores[indice_valor]
                }
                indice_valor += 1
            else:
                # Si no hay más valores, completar con None
                fila = {
                    "Año": año,
                    "País": nombre_pais,
                    nombre_columna: None
                }
            datos_lista.append(fila)  # Añadir la fila a la lista
    
    # Convertir la lista a un DataFrame
    df = pd.DataFrame(datos_lista)
    
    return df


def crear_dataframe_p(valores, años, paises, nombre_columna):
    '''
    Crea un DataFrame para los datos de las precipitaciones combinando países y años, asignando valores de forma secuencial.

    Para cada combinación (Año, País), se asigna un valor de la lista "valores". Si no hay
    suficientes valores para todas las combinaciones, se rellena con None.

    Parameters
    ----------
    - valores : list
        Lista de valores numéricos u observaciones.
    - años : list
        Lista de años disponibles.
    - paises : dict
        Diccionario con ID de país como claves y nombres de país como valores.
    - nombre_columna : str
        Nombre que se usará para la columna de los valores en el DataFrame.

    Returns
    -------
        DataFrame con columnas: "Año", "País" y "Precipitaciones", combinando todos los países
        y años con los valores correspondientes.
    '''
    datos_lista = []
    indice_valor = 0
    
    for año in años:
        for id_pais, nombre_pais in paises.items():
            if indice_valor < len(valores):
                fila = {
                    "Año": año,
                    "País": nombre_pais,
                    nombre_columna: valores[indice_valor]
                }
                indice_valor += 1
            else:
                fila = {
                    "Año": año,
                    "País": nombre_pais,
                    nombre_columna: None
                }
            datos_lista.append(fila)
    
    df = pd.DataFrame(datos_lista)
    return df

def construir_dataframe_pesticidas(datos, metadata):
    '''
    Construye un DataFrame con los valores de uso de pesticidas por país y año a partir
    de la respuesta de una API y su metadata asociada.

    Esta función toma los datos crudos de una fuente JSON, extrae los valores, años y países,
    y los organiza en un DataFrame con todas las combinaciones posibles de país y año.
    Si no hay datos disponibles para alguna combinación, se completa con NaN.

    Parámetros:
    ----------
    - datos : dict
        Diccionario JSON que contiene los valores principales bajo las claves "values",
        "entities" (IDs de países) y "years".

    - metadata : dict
        Diccionario JSON con información adicional sobre las entidades (países) y los años
        disponibles, utilizado para mapear los IDs a nombres reales.

    Retorna:
    -------
        DataFrame con columnas: "Año", "País" y "Pesticidas", donde cada fila representa
        una combinación de país y año con su valor correspondiente (o NaN si no hay dato).
    '''
    # Extraer arrays principales
    valores = datos["values"]
    entidades = datos["entities"]
    anios = datos["years"]

    # Crear diccionario ID -> Nombre de país
    id_to_country = {
        ent["id"]: ent["name"]
        for ent in metadata["dimensions"]["entities"]["values"]
    }

    # Lista completa de años del metadata
    lista_anios = [y["id"] for y in metadata["dimensions"]["years"]["values"]]

    # Lista completa de países (nombres)
    lista_paises = [ent["name"] for ent in metadata["dimensions"]["entities"]["values"]]

    # Crear todos los pares posibles País × Año
    df_base = pd.DataFrame(
        [(pais, anio) for pais in lista_paises for anio in lista_anios],
        columns=["País", "Año"]
    )

    # Crear DataFrame con datos reales desde el JSON
    registros = []
    for i in range(len(valores)):
        pais_id = entidades[i]
        anio = anios[i]
        valor = valores[i]
        pais = id_to_country.get(pais_id)
        if pais:
            registros.append((anio, pais, valor))  # Año primero

    df_valores = pd.DataFrame(registros, columns=["Año", "País", "Pesticidas"])

    # Merge con la tabla base (left join para conservar todos los países y años)
    df_final = pd.merge(df_base, df_valores, on=["Año", "País"], how="left")

    # Reordenar columnas
    df_final = df_final[["Año", "País", "Pesticidas"]]

    return df_final

import os
import pandas as pd

def cargar_csv(nombre_archivo):
    """
    Carga un archivo CSV desde la ruta 'Resultados/Archivos_csv/Datos' en un DataFrame de pandas.

    Parámetros
    ----------
    nombre_archivo : str
        Nombre del archivo CSV a cargar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados del archivo CSV.
    """
    # Construir la ruta completa
    ruta_archivo = os.path.join("..","Resultados", "Archivos_csv", "Predicciones", nombre_archivo)
    
    # Leer el CSV
    df = pd.read_csv(ruta_archivo, encoding="utf-8")
    print(f"Archivo '{ruta_archivo}' cargado correctamente.")
    
    return df

def cargar_csv_ranking(nombre_archivo):
    """
    Carga un archivo CSV desde la ruta 'Resultados/Archivos_csv/Datos' en un DataFrame de pandas.

    Parámetros
    ----------
    nombre_archivo : str
        Nombre del archivo CSV a cargar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados del archivo CSV.
    """
    # Construir la ruta completa
    ruta_archivo = os.path.join("..","Resultados", "Importancia_variables","CSV", nombre_archivo)
    
    # Leer el CSV
    df = pd.read_csv(ruta_archivo, encoding="utf-8")
    print(f"Archivo '{ruta_archivo}' cargado correctamente.")
    
    return df

# Parkison
DATA_URL = "https://api.ourworldindata.org/v1/indicators/916408.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/916408.metadata.json"
datos, metadata = cargar_datos()  
valores, años, paises = procesar_datos(datos, metadata) 
df_parkinson = crear_dataframe(valores, años, paises,"Parkinson").round(2) 

df_parkinson = df_parkinson[df_parkinson["País"] != "World"]

# Contaminación aire

# URLs de la API
DATA_URL = "https://api.ourworldindata.org/v1/indicators/939832.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/939832.metadata.json"
datos, metadata = cargar_datos()  
valores, años, paises = procesar_datos(datos, metadata) 
df_contaminacion= crear_dataframe(valores, años, paises,"Contaminacion_aire").round(2) 
# Agua
DATA_URL = "https://api.ourworldindata.org/v1/indicators/936533.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/936533.metadata.json"
datos, metadata = cargar_datos()  
valores, años, paises = procesar_datos(datos, metadata) 
df_agua = crear_dataframe(valores, años, paises,"Muertes_agua").round(2) 
# Plomo
DATA_URL = "https://api.ourworldindata.org/v1/indicators/941463.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/941463.metadata.json"
datos, metadata = cargar_datos()  
valores, años, paises = procesar_datos(datos, metadata) 
df_plomo = crear_dataframe(valores, años, paises,"Exp_plomo").round(2)
#pesticidad
DATA_URL = "https://api.ourworldindata.org/v1/indicators/1016584.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/1016584.metadata.json"
datos, metadata = cargar_datos()  
df_pepticidas = construir_dataframe_pesticidas(datos, metadata)

#precitioaciones
DATA_URL = "https://api.ourworldindata.org/v1/indicators/1005182.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/1005182.metadata.json"
datos, metadata = cargar_datos()  
valores, años, paises = procesar_datos(datos, metadata) 
df_precipitaciones = crear_dataframe_p(valores, años, paises,"Precipitaciones").round(2) 

# Cargar los archivos CSV en DataFrames
df_pred_promedio = cargar_csv("predicciones_modelos_promedio.csv").round(2)
df_pred_desviacion = cargar_csv("predicciones_modelos_desviacion.csv").round(2)
df_realesVSpredichos = cargar_csv("RealesVSPredichos.csv").round(2)
df_predicciones_GLM = cargar_csv("pred_GLM.csv").round(2)
df_predicciones_RF = cargar_csv("pred_RF.csv").round(2)
df_predicciones_XG = cargar_csv("pred_XG.csv").round(2)
df_predicciones_SVR = cargar_csv("pred_SVR.csv").round(2)
df_predicciones_KNN = cargar_csv('pred_KNN.csv').round(2)
df_predicciones_MLP = cargar_csv("pred_MLP.csv").round(2)
df_ranking = cargar_csv_ranking("Ranking_global_promedio.csv")

#Misma escala de distribicon para todos los mapas

min_parkinson = round(df_parkinson["Parkinson"].min(), 2)
q25_parkinson = round(df_parkinson["Parkinson"].quantile(0.25), 2)
q50_parkinson = round(df_parkinson["Parkinson"].quantile(0.50), 2)
q75_parkinson = round(df_parkinson["Parkinson"].quantile(0.75), 2)
q95_parkinson = round(df_parkinson["Parkinson"].quantile(0.95), 2)

min_contaminacion = round(df_contaminacion["Contaminacion_aire"].min(), 2)
q25_contaminacion = round(df_contaminacion["Contaminacion_aire"].quantile(0.25), 2)
q50_contaminacion = round(df_contaminacion["Contaminacion_aire"].quantile(0.50), 2)
q75_contaminacion = round(df_contaminacion["Contaminacion_aire"].quantile(0.75), 2)
q95_contaminacion = round(df_contaminacion["Contaminacion_aire"].quantile(0.95), 2)

#min_contaminacion = df_contaminacion["Contaminacion_aire"].min()
#max_contaminacion = df_contaminacion["Contaminacion_aire"].quantile(0.90)

# Percentiles para Exposición al Plomo
min_plomo = round(df_plomo["Exp_plomo"].min(), 2)
q25_plomo = round(df_plomo["Exp_plomo"].quantile(0.25), 2)
q50_plomo = round(df_plomo["Exp_plomo"].quantile(0.50), 2)
q75_plomo = round(df_plomo["Exp_plomo"].quantile(0.75), 2)
q95_plomo = round(df_plomo["Exp_plomo"].quantile(0.95), 2)


#min_plomo = df_plomo["Exp_plomo"].min()
#max_plomo = df_plomo["Exp_plomo"].quantile(0.90)

min_agua = round(df_agua["Muertes_agua"].min(), 2)
q25_agua = round(df_agua["Muertes_agua"].quantile(0.25), 2)
q50_agua = round(df_agua["Muertes_agua"].quantile(0.50), 2)
q75_agua = round(df_agua["Muertes_agua"].quantile(0.75), 2)
q95_agua = round(df_agua["Muertes_agua"].quantile(0.95), 2)


#min_agua = df_agua["Muertes_agua"].min()
#max_agua = df_agua["Muertes_agua"].quantile(0.75)

min_pesticidas = round(df_pepticidas["Pesticidas"].min(), 2)
q25_pesticidas = round(df_pepticidas["Pesticidas"].quantile(0.25), 2)
q50_pesticidas = round(df_pepticidas["Pesticidas"].quantile(0.50), 2)
q75_pesticidas = round(df_pepticidas["Pesticidas"].quantile(0.75), 2)
q95_pesticidas = round(df_pepticidas["Pesticidas"].quantile(0.95), 2)


#min_pepticidas = df_pepticidas["Pesticidas"].min()
#max_pepticidas = df_pepticidas["Pesticidas"].quantile(0.90)

min_precipitaciones = round(df_precipitaciones["Precipitaciones"].min(), 2)
q25_precipitaciones = round(df_precipitaciones["Precipitaciones"].quantile(0.25), 2)
q50_precipitaciones = round(df_precipitaciones["Precipitaciones"].quantile(0.50), 2)
q75_precipitaciones = round(df_precipitaciones["Precipitaciones"].quantile(0.75), 2)
q95_precipitaciones = round(df_precipitaciones["Precipitaciones"].quantile(0.95), 2)


#min_precipitaciones = df_precipitaciones["Precipitaciones"].min()
#max_precipitaciones = df_precipitaciones["Precipitaciones"].quantile(0.90)

# Calcula los cuantiles para el mapa de predicción
min_pred = round(df_pred_promedio["Parkinson_Predicho_Promedio"].min(), 2)
q25_pred = round(df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.25), 2)
q50_pred = round(df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.50), 2)
q75_pred = round(df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.75), 2)
q95_pred = round(df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.95), 2)

#min_val_glm = df_predicciones_GLM["Parkinson_Predicho"].min()
#max_val_glm =df_predicciones_GLM["Parkinson_Predicho"].quantile(0.95)

min_glm = round(df_predicciones_GLM["Parkinson_Predicho"].min(), 2)
q25_glm = round(df_predicciones_GLM["Parkinson_Predicho"].quantile(0.25), 2)
q50_glm = round(df_predicciones_GLM["Parkinson_Predicho"].quantile(0.50), 2)
q75_glm = round(df_predicciones_GLM["Parkinson_Predicho"].quantile(0.75), 2)
q95_glm = round(df_predicciones_GLM["Parkinson_Predicho"].quantile(0.95), 2)


#min_val_rf = df_predicciones_RF["Parkinson_Predicho"].min()
#max_val_rf =df_predicciones_RF["Parkinson_Predicho"].quantile(0.95)

min_rf = round(df_predicciones_RF["Parkinson_Predicho"].min(), 2)
q25_rf = round(df_predicciones_RF["Parkinson_Predicho"].quantile(0.25), 2)
q50_rf = round(df_predicciones_RF["Parkinson_Predicho"].quantile(0.50), 2)
q75_rf = round(df_predicciones_RF["Parkinson_Predicho"].quantile(0.75), 2)
q95_rf = round(df_predicciones_RF["Parkinson_Predicho"].quantile(0.95), 2)


#min_val_xg = df_predicciones_XG["Parkinson_Predicho"].min()
#max_val_xg = df_predicciones_XG["Parkinson_Predicho"].quantile(0.95)

min_xg = round(df_predicciones_XG["Parkinson_Predicho"].min(), 2)
q25_xg = round(df_predicciones_XG["Parkinson_Predicho"].quantile(0.25), 2)
q50_xg = round(df_predicciones_XG["Parkinson_Predicho"].quantile(0.50), 2)
q75_xg = round(df_predicciones_XG["Parkinson_Predicho"].quantile(0.75), 2)
q95_xg = round(df_predicciones_XG["Parkinson_Predicho"].quantile(0.95), 2)


#min_val_svr = df_predicciones_SVR["Parkinson_Predicho"].min()
#max_val_svr = df_predicciones_SVR["Parkinson_Predicho"].quantile(0.95)

min_svr = round(df_predicciones_SVR["Parkinson_Predicho"].min(), 2)
q25_svr = round(df_predicciones_SVR["Parkinson_Predicho"].quantile(0.25), 2)
q50_svr = round(df_predicciones_SVR["Parkinson_Predicho"].quantile(0.50), 2)
q75_svr = round(df_predicciones_SVR["Parkinson_Predicho"].quantile(0.75), 2)
q95_svr = round(df_predicciones_SVR["Parkinson_Predicho"].quantile(0.95), 2)


#min_val_knn = df_predicciones_KNN ["Parkinson_Predicho"].min()
#max_val_knn = df_predicciones_KNN ["Parkinson_Predicho"].quantile(0.95)

min_knn = round(df_predicciones_KNN["Parkinson_Predicho"].min(), 2)
q25_knn = round(df_predicciones_KNN["Parkinson_Predicho"].quantile(0.25), 2)
q50_knn = round(df_predicciones_KNN["Parkinson_Predicho"].quantile(0.50), 2)
q75_knn = round(df_predicciones_KNN["Parkinson_Predicho"].quantile(0.75), 2)
q95_knn = round(df_predicciones_KNN["Parkinson_Predicho"].quantile(0.95), 2)


#min_val_mlp = df_predicciones_MLP  ["Parkinson_Predicho"].min()
#max_val_mlp = df_predicciones_MLP  ["Parkinson_Predicho"].quantile(0.95)

min_mlp = round(df_predicciones_MLP["Parkinson_Predicho"].min(), 2)
q25_mlp = round(df_predicciones_MLP["Parkinson_Predicho"].quantile(0.25), 2)
q50_mlp = round(df_predicciones_MLP["Parkinson_Predicho"].quantile(0.50), 2)
q75_mlp = round(df_predicciones_MLP["Parkinson_Predicho"].quantile(0.75), 2)
q95_mlp = round(df_predicciones_MLP["Parkinson_Predicho"].quantile(0.95), 2)


#min_val = df_pred_promedio["Parkinson_Predicho_Promedio"].min()
#max_val = df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.95)

min_std = round(df_pred_desviacion["Desviacion"].min(), 2)
q25_std = round(df_pred_desviacion["Desviacion"].quantile(0.25), 2)
q50_std = round(df_pred_desviacion["Desviacion"].quantile(0.50), 2)
q75_std = round(df_pred_desviacion["Desviacion"].quantile(0.75), 2)
q95_std = round(df_pred_desviacion["Desviacion"].quantile(0.95), 2)


#min_std = df_pred_desviacion["Desviacion"].min()
#max_std = df_pred_desviacion["Desviacion"].quantile(0.95)


# 2. Calcular el error normalizado
df_realesVSpredichos["Error_Normalizado"] = (df_realesVSpredichos["Parkinson_Predicho_Promedio"] - df_realesVSpredichos["Parkinson_Real"]) 
df_realesVSpredichos["Error_Normalizado"] = df_realesVSpredichos["Error_Normalizado"].clip(-1, 1)
# Mínimo y máximo reales (no truncados)
        
real_min = df_realesVSpredichos['Error_Absoluto'].min()
real_max = df_realesVSpredichos['Error_Absoluto'].max()

years = df_parkinson['Año'].unique().tolist()

countries = df_parkinson['País'].unique().tolist()

# Definir la interfaz de usuario con CSS global
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style(""" 
            .sidebar {
                background-color: #007BFF !important;
                color: white !important;
                padding: 15px !important;
                height: 100vh !important;
                width: 250px !important;
                position: fixed;
                top: 0;
                left: 0;
                z-index: 1000;
            }
            .content-box {
                flex-direction: column;
                padding: 20px;
                border: none !important;
                background-color: transparent !important;
                margin-top: 10px;
                margin-left: 50px;
                margin-right: 60px;  
                padding-left: 20px;  
            }
            .nav-item {
                display: block;
                background-color: white;
                color: black;
                padding: 15px;
                margin: 10px;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                text-align: center;
                font-weight: bold;
                text-decoration: none;
            }
            .nav-item:hover {
                background-color: #e0e0e0;
            }
            .navset-pill .nav-link {
                border-radius: 0px !important;
            }
            #home_btn {
                background: none !important;
                border: none !important;
                color: white !important;
                font-size: 18px;
                cursor: pointer;
                text-align: left;
                padding: 10px 15px;
                display: block;
            }
            #home_btn:hover {
                text-decoration: underline;
            }
            .home-container {
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 20px;
            }
            .home-title {
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
            .home-subtitle {
                font-size: 18px;
                color: #666;
            }
            .map-container {
                margin-: 30px;
                padding-right: 30px;
                width: 90%;
                max-width: 1200px;
                margin: 0 auto;
                height: 600px;
            }
            #section1 .map-container h3 {
                font-size: 100px;
                font-weight: bold;
                color: #333;
                margin-left: 20px;
                margin-bottom: 20px;
            }
            .map-and-slider-container {
                display: flex;
                flex-direction: column;  
                align-items: flex-start;  
                width: 100%;  
            }
            .slider-box {
                margin-left: -40px;  
                width: 100%;  
            }
            /* Colores específicos para cada enlace del sidebar */
            .sidebar-link {
                display: block;
                padding: 20px 15px;  /* Más altura y espacio interno */
                margin: 15px 0;      /* Más separación entre enlaces */
                color: white;
                text-decoration: none;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                font-size: 18px;     /* Letra más grande */
                box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.2); /* Sombra sutil */
                transition: background-color 0.3s ease;
            }
            
            .home-link  { background-color: #1abc9c; }
            .park-link {background-color: #C0392B;  /* Rojo ladrillo para sección 1 */}
            .map-link   { background-color: #3498db; }
            .env-link   { background-color: #e67e22; }
            .graph-link { background-color: #9b59b6; }
            .analisis-link {background-color: #8A2BE2}
            .contact-link {background-color: #34495E }
            
            .sidebar-link:hover {
                opacity: 0.85;  /* Efecto al pasar el cursor */
                cursor: pointer;}

        """),
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.a("🏠 Home", class_="sidebar-link home-link", onclick="Shiny.setInputValue('page', 'home')"),
                ui.a("🧠 Enfermedad de Parkinson", class_="sidebar-link park-link", onclick="Shiny.setInputValue('page', 'section1')"),
                ui.a("🗺️ Mapa Mundial de párkinson", class_="sidebar-link map-link", onclick="Shiny.setInputValue('page', 'section2')"),
                ui.a("🌿 Variables Ambientales", class_="sidebar-link env-link", onclick="Shiny.setInputValue('page', 'section3')"),
                ui.a("📈 Predicciones", class_="sidebar-link graph-link", onclick="Shiny.setInputValue('page', 'section4')"),
                ui.a("🔍 Importancia de variables", class_="sidebar-link analisis-link", onclick="Shiny.setInputValue('page', 'section5')"),
                ui.a("📞 Más Información", class_="sidebar-link contact-link", onclick="Shiny.setInputValue('page', 'section6')"),

                class_="sidebar"
            )
        ),
        ui.output_ui("content_display")
    )
    
)

# Definir la lógica del servidor
def server(input, output, session):
    current_page = reactive.Value("home")
    @reactive.Effect
    def _():
        if input.page() is not None:
            current_page.set(input.page())


    @output
    @render.ui
    def content_display():
        page = current_page()
        if page == "home":
            return ui.div(
                # Franja de color con el título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
            
                # Imagen flotante a la izquierda y texto fluyendo a la derecha
                ui.div(
                    ui.img(
                        src="https://upload.wikimedia.org/wikipedia/commons/8/80/World_map_-_low_resolution.svg",
                        height="200px",
                        style="float: left; margin-right: 20px; margin-bottom: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Esta aplicación es una herramienta visual que explora la relación entre diferentes variables ambientales y la prevalencia de la enfermedad de Parkinson en diversas regiones del mundo. "
                        "El objetivo de esta app es proporcionar una visión comprensible y accesible sobre cómo factores ambientales, "
                        "pueden tener un impacto en la aparición y progresión de esta enfermedad neurodegenerativa.",
                        style="font-size: 18px; font-family: 'Verdana', sans-serif; color: #333333; line-height: 1.6;"
                    ),
                    ui.h3("¿Qué analizamos?", style="font-size: 24px; font-weight: bold; color: #2C3E50; margin-top: 20px;"),
                    ui.p(
                        "A través de esta aplicación, analizamos diferentes aspectos de la prevalencia de la enfermedad de Parkinson en función de los factores ambientales. "
                        "Entre los análisis realizados se incluyen la visualización geográfica de los países más afectados por la enfermedad, "
                        "y el uso de modelos predictivos entrenados con estos datos para predecir la prevalencia futura en distintas regiones del mundo.",
                        style="font-size: 18px; line-height: 1.6; color: #333333;"
                    ),
                    ui.h3("¿De dónde provienen los datos?",
                          style="font-size: 24px; font-weight: bold; color: #2C3E50; clear: both;"),
                    ui.p(
                        ui.HTML(
                            "Los datos utilizados provienen de <b><em>Our World in Data (OWID)</em></b>, una plataforma global que recopila y presenta datos de salud pública, sociales y ambientales de todo el mundo. "
                            "La misión de <b><em>OWID</em></b> es hacer que los datos sean accesibles para cualquier persona, con el fin de fomentar una mayor comprensión y toma de decisiones informadas. En nuestro caso, hemos utilizado información sobre la Tasa de mortalidad por contaminación del aire, "
                            "la Tasa de carga de enfermedad por exposición al plomo, muertes atribuidas a fuentes de agua inseguras, el uso de pesticidas y precipitaciones anuales."
                        ),
                        style="font-size: 18px; line-height: 1.6; color: #333333;"
                    ),

                    ui.HTML(
                        '<a href="https://ourworldindata.org/" target="_blank" '
                        'style="font-size: 18px; color: #3498db; text-decoration: none;">'
                        'Visita <b><em>Our World in Data</em></b> para más detalles</a>'
                    ),
                    ui.p(
                        "Al combinar estos datos con análisis estadísticos y modelos predictivos, se puede obtener una visión más clara de cómo estos factores ambientales pueden afectar la prevalencia de párkinson. "
                        "Además, este enfoque también ayuda a identificar posibles áreas geográficas donde el riesgo de párkinson es más alto, lo que puede llevar a una mejor planificación de políticas públicas y estrategias de salud.",
                        style="font-size: 18px; line-height: 1.6; color: #333333;"
                    ),

                ),
            )
        

        page = input.page()
        if page == "section1":
            return ui.div(
                # Franja de título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                # Información sobre la Enfermedad de Parkinson
                ui.div(
                    # Título de la sección
                    ui.h2("🧠 ¿Qué es la Enfermedad de Parkinson?", style="color: black; text-align: center; margin-top: 20px;"),
            
                    # Descripción general de la enfermedad
                    ui.p(
                        "La enfermedad de Parkinson es un trastorno neurodegenerativo crónico y progresivo que afecta el sistema nervioso central, "
                        "especialmente las áreas del cerebro encargadas del control del movimiento. Se caracteriza por síntomas como temblores, rigidez "
                        "muscular, lentitud de movimientos (bradicinesia) y problemas de equilibrio y coordinación.",
                         style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
            
                    ui.p(
                        "Aunque se desconoce la causa exacta de la enfermedad de Parkinson, se sabe que resulta de la degeneración de las neuronas dopaminérgicas en "
                        "una región del cerebro llamada sustancia negra. Sin embargo, investigaciones recientes sugieren que factores ambientales pueden "
                        "jugar un papel relevante en el desarrollo de la enfermedad, especialmente en personas con cierta predisposición genética.",
                         style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
            
                    # Título de síntomas comunes
                    ui.h3("🚶‍♂️ Síntomas Comunes de la enfermedad de Parkinson", style="color: black; text-align: center; margin-top: 20px;"),
            
                    # Descripción de los síntomas comunes
                   
                    ui.div(
                        ui.HTML("<strong>Temblores</strong>: Los temblores son uno de los síntomas más reconocibles. Aparecen en reposo y afectan típicamente las manos, brazos y piernas."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.div(
                        ui.HTML("<strong>Rigidez muscular</strong>: La rigidez en los músculos puede dificultar los movimientos y causar dolor."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.div(
                        ui.HTML("<strong>Bradicinesia (lentitud de movimientos)</strong>: La disminución de la velocidad al realizar movimientos, como caminar o escribir."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.div(
                        ui.HTML("<strong>Inestabilidad postural</strong>: Los pacientes pueden tener problemas para mantener el equilibrio, lo que aumenta el riesgo de caídas."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),



            
                    # Título de factores de riesgo
                    ui.h3("⚠️ Factores de Riesgo", style="color: black; text-align: center; margin-top: 20px;"),
            
                    # Descripción de los factores de riesgo
                    ui.div(
                        ui.HTML("<strong>Edad</strong>: La mayoría de las personas con párkinson son mayores de 60 años."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.div(
                        ui.HTML("<strong>Genética</strong>: Algunos casos tienen una predisposición genética, pero la mayoría de los casos son esporádicos (no hereditarios)."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.div(
                        ui.HTML("<strong>Sexo</strong>: Los hombres tienen un mayor riesgo de desarrollar párkinson que las mujeres."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.div(
                        ui.HTML("<strong>Factores Ambientales</strong>: Exposición a sustancias químicas, como pesticidas, y la contaminación del aire pueden aumentar el riesgo."),
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                )
            )


        
        
            



       
        elif page == "section2":
            return ui.div(
                # Franja de título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Descripción bajo el título
                ui.p(
                    ui.HTML(
                        'Visualiza la prevalencia estimada de la enfermedad de Parkinson: el número de personas afectadas por cada 100,000 habitantes en distintos países y años. '
                        '<a href="https://ourworldindata.org/grapher/parkinsons-disease-prevalence-ihme" target="_blank" style="color: #2980B9; text-decoration: underline;">Accede aquí</a>.'
                    ),
                    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_parkinson"),
        
                    # Slider de año
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año",
                                        min=df_parkinson["Año"].min(),
                                        max=df_parkinson["Año"].max(),
                                        value=df_parkinson["Año"].min(),
                                        step=1,
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
        
                    # Botón para ir al mapa europeo
                    ui.div(
                        ui.input_action_button("go_to_europe", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'europe_map')"),
                        style="margin-top: 10px;"
                    ),
        
                    # Sección de descarga de datos (sin botones)
                    ui.div(
                        ui.h3(
                            "📥 Descarga de datos",
                            style="text-align: center; margin-top: 30px; margin-bottom: 10px; font-family: 'Arial'"
                        ),
        
                        # Mensaje de restricción de licencia
                        ui.div(
                            ui.HTML(
                                '⚠️ La descarga de estos datos no está disponible debido a restricciones de licencia que no permiten su redistribución. '
                                'Para más información, visita este <a href="https://vizhub.healthdata.org/gbd-results/" '
                                'target="_blank" style="color: black; text-decoration: underline;">enlace</a>.'
                            ),
                            style="background-color: #AED6F1; color: black; padding: 15px; border-radius: 8px; text-align: center; font-family: \'Arial\', sans-serif; margin-top: 10px;"
                        )
                    ),
        
                    # Sección de citas añadida
                    ui.div(
                        ui.h3("📚 Citas", style="text-align: center; margin-top: 40px; font-family: 'Arial', sans-serif;"),
                    
                        # Cita original de IHME
                        ui.p(
                            ui.HTML(
                                'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2021 (GBD 2021). '
                                'Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. '
                                'Available from <a href="https://vizhub.healthdata.org/gbd-results/" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://vizhub.healthdata.org/gbd-results/</a>. '
                                '<br><strong>Attribution short:</strong> "IHME-GBD".'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                    
                        # Cita OWID extendida
                        ui.p(
                            ui.HTML(
                                '“Data Page: Rate of disease burden from lead exposure”, part of the following publication: '
                                'Esteban Ortiz-Ospina and Max Roser (2016) – “Global Health”. Data adapted from IHME, Global Burden of Disease. '
                                'Retrieved from <a href="https://ourworldindata.org/grapher/rate-disease-burden-lead" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://ourworldindata.org/grapher/parkinsons-disease-prevalence-ihme</a> [online resource].'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 20px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                    
                        # Cita OWID abreviada
                        ui.p(
                            "IHME, Global Burden of Disease (2024) – with minor processing by Our World in Data",
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;"
                        )
                    )
                )
            )







        elif page == "europe_map":
            return ui.div(
                # Franja de título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'section2')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_slider(
                        "year", "Selecciona el Año",
                        min=df_parkinson["Año"].min(),
                        max=df_parkinson["Año"].max(),
                        value=df_parkinson["Año"].min(),
                        step=1,
                        sep=""
                    ),
                    class_="slider-box"
                ),
                ui.output_ui("plot_europe"),
                class_="content-box"
            )


        elif page == "section3":
            return ui.div(
                # Título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                ui.div(
                    ui.h3("📊 Factores ambientales", style="color: #2C3E50; margin-top: 30px; text-align: center;"),
                    ui.p(
                        "En esta sección exploraremos distintos factores ambientales que podrían estar relacionados con la enfermedad de Parkinson. "
                        "Cada botón te llevará a una sección donde podrás ver datos específicos y visualizaciones que muestran cómo estos factores varían en diferentes regiones del mundo.",
                        style="font-size: 17px; margin: 10px 20px; text-align: justify; color: #333;"
                    ),
                    style="margin-bottom: 20px;"
                ),

        
                # Botones y enlaces separados
                ui.div(
                    # Contaminación
                    ui.div(
                        [
                            ui.input_action_button(
                                "show_contaminacion",
                                ui.HTML(
                                    "<strong>🌫️ Contaminación del Aire</strong><br><small>La exposición a PM2.5 y NO₂ se ha relacionado con un aumento del riesgo de Parkinson.</small>"
                                ),
                                class_="btn btn-primary",
                                onclick="Shiny.setInputValue('page', 'contaminacion')",
                                style="text-align: left; white-space: normal; padding: 15px; width: 100%;"
                            ),
                            ui.a("🔗 Accede aquí a los datos", href="https://ourworldindata.org/grapher/death-rates-from-air-pollution?tab=table", target="_blank", style="display: block; margin-top: 5px; color: #2980B9;")
                        ],
                        style="width: 18%;"
                    ),
                    
                    # Plomo
                    ui.div(
                        [
                            ui.input_action_button(
                                "show_plomo",
                                ui.HTML(
                                    "<strong>🔩 Exposición al Plomo</strong><br><small>La exposición prolongada a metales pesados como el plomo puede afectar el sistema nervioso central.</small>"
                                ),
                                class_="btn btn-primary",
                                onclick="Shiny.setInputValue('page', 'plomo')",
                                style="text-align: left; white-space: normal; padding: 15px; width: 100%;"
                            ),
                            ui.a("🔗 Accede aquí a los datos", href="https://ourworldindata.org/grapher/rate-disease-burden-lead?tab=table", target="_blank", style="display: block; margin-top: 5px; color: #2980B9;")
                        ],
                        style="width: 18%;"
                    ),
        
                    # Agua
                    ui.div(
                        [
                            ui.input_action_button(
                                "show_agua",
                                ui.HTML(
                                    "<strong>🚰 Aguas Inseguras</strong><br><small>El consumo de agua contaminada por metales pesados o tóxicos se ha vinculado con riesgo de párkinson.</small>"
                                ),
                                class_="btn btn-primary",
                                onclick="Shiny.setInputValue('page', 'agua')",
                                style="text-align: left; white-space: normal; padding: 15px; width: 100%;"
                            ),
                            ui.a("🔗 Accede aquí a los datos", href="https://ourworldindata.org/grapher/deaths-due-to-unsafe-water-sources?tab=table", target="_blank", style="display: block; margin-top: 5px; color: #2980B9;")
                        ],
                        style="width: 18%;"
                    ),
        
                    # Pesticidas
                    ui.div(
                        [
                            ui.input_action_button(
                                "show_pesticidas",
                                ui.HTML(
                                    "<strong>🌿 Uso de Pesticidas</strong><br><small>Sustancias como paraquat y maneb están asociadas con mayor riesgo de párkinson.</small>"
                                ),
                                class_="btn btn-primary",
                                onclick="Shiny.setInputValue('page', 'pesticidas')",
                                style="text-align: left; white-space: normal; padding: 15px; width: 100%;"
                            ),
                            ui.a("🔗 Accede aquí a los datos", href="https://ourworldindata.org/grapher/pesticide-use-tonnes?tab=table", target="_blank", style="display: block; margin-top: 5px; color: #2980B9;")
                        ],
                        style="width: 18%;"
                    ),
        
                    # Precipitaciones
                    ui.div(
                        [
                            ui.input_action_button(
                                "show_precipitaciones",
                                ui.HTML(
                                    "<strong>🌧️ Precipitaciones</strong><br><small>Cambios en la lluvia pueden afectar la exposición a contaminantes o pesticidas.</small>"
                                ),
                                class_="btn btn-primary",
                                onclick="Shiny.setInputValue('page', 'precipitaciones')",
                                style="text-align: left; white-space: normal; padding: 15px; width: 100%;"
                            ),
                            ui.a("🔗 Accede aquí a los datos", href="https://ourworldindata.org/grapher/average-precipitation-per-year?tab=table", target="_blank", style="display: block; margin-top: 5px; color: #2980B9;")
                        ],
                        style="width: 18%;"
                    ),
        
                    style="display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; margin: 30px 0 20px 0;"
                ),
        
                # Texto final
                ui.div(
                    ui.p(
                        "💡 Haz clic en los botones para ver más información. Usa los enlaces para acceder a los datos de cada variable.",
                        style="font-size: 16px; color: #555; text-align: center; margin-top: 20px; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    )
                )
            )





        elif page == "contaminacion":
            return ui.div(
                # Título principal
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                
                # Descripción
                ui.p(
                    "Visualiza la tasa estimada de muertes atribuidas a la contaminación del aire, expresadas por cada 100.000 habitantes. "
                    "Incluye contaminantes como el ozono en exteriores y puede reflejar múltiples factores de riesgo.",
                    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_contaminacion"),
        
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_contaminacion["Año"].min(), 
                                        max=df_contaminacion["Año"].max(), 
                                        value=df_contaminacion["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
        
                    ui.div(
                        ui.input_action_button("go_to_europe_aire", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_aire')")
                    ),
        
                    # Sección de descarga de datos (sin botones)
                    ui.div(
                        ui.h3(
                            "📥 Descarga de datos",
                            style="text-align: center; margin-top: 30px; margin-bottom: 10px; font-family: 'Arial', sans-serif;"
                        ),
                    
                        # Mensaje de restricción de licencia
                        ui.div(
                            ui.HTML(
                                '⚠️ La descarga de estos datos no está disponible debido a restricciones de licencia que no permiten su redistribución. '
                                'Para más información, visita este <a href="https://vizhub.healthdata.org/gbd-results/" '
                                'target="_blank" style="color: black; text-decoration: underline;">enlace</a>.'
                            ),
                            style="background-color: #AED6F1; color: black; padding: 15px; border-radius: 8px; text-align: center; font-family: \'Arial\', sans-serif; margin-top: 10px;"
                        )
                    ),
                    
                    # Sección de citas añadida
                    ui.div(
                        ui.h3("📚 Citas", style="text-align: center; margin-top: 40px; font-family: 'Arial', sans-serif;"),
                        ui.p(
                            ui.HTML(
                                'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2021 (GBD 2021). '
                                'Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. '
                                'Available from <a href="https://vizhub.healthdata.org/gbd-results/" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://vizhub.healthdata.org/gbd-results/</a>. '
                                '<br><strong>Attribution short:</strong> "IHME-GBD".'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                        # Cita OWID extendida
                        ui.p(
                            ui.HTML(
                                '“Data Page: Rate of disease burden from lead exposure”, part of the following publication: '
                                'Esteban Ortiz-Ospina and Max Roser (2016) – “Global Health”. Data adapted from IHME, Global Burden of Disease. '
                                'Retrieved from <a href="https://ourworldindata.org/grapher/rate-disease-burden-lead" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://ourworldindata.org/grapher/death-rates-from-air-pollution</a> [online resource].'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 20px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                        # Cita OWID abreviada
                        ui.p(
                            "IHME, Global Burden of Disease (2024) – with minor processing by Our World in Data",
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;"
                        )
                    ),
                    
                    # Botón volver atrás
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_contaminacion",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    )
                )
            )


        elif page == "plot_europe_aire":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'contaminacion')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_slider(
                        "year", "Selecciona el Año",
                        min=df_contaminacion["Año"].min(),
                        max=df_contaminacion["Año"].max(),
                        value=df_contaminacion["Año"].min(),
                        step=1,
                        sep=""
                    ),
                    class_="slider-box"
                ),
                ui.output_ui("plot_europe_aire"),
                class_="content-box"
            )


        elif page == "plomo":
            return ui.div(
                # Título principal
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Descripción
                ui.p(
                    "Visualiza la tasa de carga de enfermedad atribuida a la exposición al plomo. "
                    "Esta métrica representa el número estimado de años de vida perdidos debido a muerte prematura "
                    "o discapacidad causadas por dicha exposición, por cada 100.000 personas. "
                    "Se expresa en AVAD (Años de Vida Ajustados por Discapacidad) y está ajustada por edad, "
                    "lo que permite comparar países con diferentes estructuras demográficas.",
                    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_plomo"),
        
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_plomo["Año"].min(), 
                                        max=df_plomo["Año"].max(), 
                                        value=df_plomo["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
        
                    ui.div(
                        ui.input_action_button("go_to_europe_plomo", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_plomo')")
                    ),
        
                    # Sección de descarga de datos (sin botones)
                    ui.div(
                        ui.h3(
                            "📥 Descarga de datos",
                            style="text-align: center; margin-top: 30px; margin-bottom: 10px; font-family: 'Arial', sans-serif;"
                        ),
                    
                        # Mensaje de restricción de licencia
                        ui.div(
                            ui.HTML(
                                '⚠️ La descarga de estos datos no está disponible debido a restricciones de licencia que no permiten su redistribución. '
                                'Para más información, visita este <a href="https://vizhub.healthdata.org/gbd-results/" '
                                'target="_blank" style="color: black; text-decoration: underline;">enlace</a>.'
                            ),
                            style="background-color: #AED6F1; color: black; padding: 15px; border-radius: 8px; text-align: center; font-family: \'Arial\', sans-serif; margin-top: 10px;"
                        )
                    ),
                    
                    # Sección de citas añadida
                    ui.div(
                        ui.h3("📚 Citas", style="text-align: center; margin-top: 40px; font-family: 'Arial', sans-serif;"),
                        ui.p(
                            ui.HTML(
                                'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2021 (GBD 2021). '
                                'Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. '
                                'Available from <a href="https://vizhub.healthdata.org/gbd-results/" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://vizhub.healthdata.org/gbd-results/</a>. '
                                '<br><strong>Attribution short:</strong> "IHME-GBD".'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                        ui.p(
                            ui.HTML(
                                '“Data Page: Rate of disease burden from lead exposure”, part of the following publication: '
                                'Esteban Ortiz-Ospina and Max Roser (2016) – “Global Health”. Data adapted from IHME, Global Burden of Disease. '
                                'Retrieved from <a href="https://ourworldindata.org/grapher/rate-disease-burden-lead" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://ourworldindata.org/grapher/rate-disease-burden-lead</a> [online resource].'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 20px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                    
                        # Cita OWID abreviada
                        ui.p(
                            "IHME, Global Burden of Disease (2024) – with minor processing by Our World in Data",
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;"
                        )
                        
                    ),
                    
                    # Botón volver atrás
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_contaminacion",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    )
                )
            )


        elif page == "plot_europe_plomo":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'plomo')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_slider(
                        "year", "Selecciona el Año",
                        min=df_plomo["Año"].min(),
                        max=df_plomo["Año"].max(),
                        value=df_plomo["Año"].min(),
                        step=1,
                        sep=""
                    ),
                    class_="slider-box"
                ),
                ui.output_ui("plot_europe_plomo"),
                class_="content-box"
            )

        elif page == "agua":
            return ui.div(
                # Título principal
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Descripción
                ui.p(
                    "Visualiza el número estimado de muertes por cada 100.000 personas atribuibles a fuentes de agua insalubres. "
                    "Esto incluye el consumo de agua contaminada o la falta de acceso a instalaciones seguras de saneamiento e higiene. "
                    "Representa la carga de mortalidad que podría evitarse si toda la población tuviera acceso a agua potable y condiciones adecuadas de saneamiento.",
                    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_agua"),
        
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_agua["Año"].min(), 
                                        max=df_agua["Año"].max(), 
                                        value=df_agua["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
        
                    ui.div(
                        ui.input_action_button("go_to_europe_agua", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_agua')")
                    ),
        
                    # Sección de citas añadida
                    ui.div(
                        ui.h3("📚 Citas", style="text-align: center; margin-top: 40px; font-family: 'Arial', sans-serif;"),
                        ui.p(
                            ui.HTML(
                                'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2021 (GBD 2021). '
                                'Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. '
                                'Available from <a href="https://vizhub.healthdata.org/gbd-results/" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://vizhub.healthdata.org/gbd-results/</a>. '
                                '<br><strong>Attribution short:</strong> "IHME-GBD".'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                        ui.p(
                            ui.HTML(
                                '“Data Page: Rate of disease burden from lead exposure”, part of the following publication: '
                                'Esteban Ortiz-Ospina and Max Roser (2016) – “Global Health”. Data adapted from IHME, Global Burden of Disease. '
                                'Retrieved from <a href="https://ourworldindata.org/grapher/rate-disease-burden-lead" target="_blank" '
                                'style="color: black; text-decoration: underline;">https://ourworldindata.org/grapher/deaths-due-to-unsafe-water-sources</a> [online resource].'
                            ),
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 20px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                        ),
                    
                        # Cita OWID abreviada
                        ui.p(
                            "IHME, Global Burden of Disease (2024) – with minor processing by Our World in Data",
                            style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;"
                        )
                        
                    ),
                    
                    # Botón volver atrás
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_contaminacion",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    )
                )
            )


        elif page == "plot_europe_agua":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'agua')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_slider(
                        "year", "Selecciona el Año",
                        min=df_agua["Año"].min(),
                        max=df_agua["Año"].max(),
                        value=df_agua["Año"].min(),
                        step=1,
                        sep=""
                    ),
                    class_="slider-box"
                ),
                ui.output_ui("plot_europe_agua"),
                class_="content-box"
            )

        elif page == "pesticidas":
            return ui.div(
                # Encabezado principal
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Descripción
                ui.p(
                    "Visualiza el uso total de pesticidas en toneladas por país, entre 1990 y 2022. "
                    "Este valor representa la cantidad total de pesticidas utilizados anualmente, incluyendo insecticidas, herbicidas y fungicidas, "
                    "y refleja la intensidad del uso de productos químicos en la agricultura.",
                    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_pesticidas"),
        
                    ui.div(
                        ui.input_slider(
                            "year", "Selecciona el Año",
                            min=df_pepticidas["Año"].min(),
                            max=df_pepticidas["Año"].max(),
                            value=df_pepticidas["Año"].min(),
                            step=1,
                            sep=""
                        ),
                        style="margin-top: 10px;"
                    ),
        
                    ui.div(
                        ui.input_action_button(
                            "go_to_europe_pesticidas",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_pesticidas')"
                        )
                    ),
        
                    # Sección de descarga completa
                    ui.div(
                        ui.h3("📥 Descarga completa de datos",
                              style="text-align: center; margin-top: 30px; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
                        ui.div(
                            ui.download_button("downloadAll_uso_pesticidas", "Descargar CSV "),
                            ui.download_button("downloadAll_uso_pesticidas_json", "Descargar JSON"),
                            style="display: flex; flex-direction: column; gap: 10px; align-items: center;"
                        )
                    ),
        
                    # Sección de filtrado por año y país
                    ui.div(
                        ui.h3("📅 Filtra los datos por año y país",
                              style="text-align: center; margin-top: 40px; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
                        ui.div(
                            # Columna izquierda: selectores
                            ui.div(
                                ui.div(
                                    ui.input_select(
                                        "years_select",
                                        "Selecciona los años",
                                        choices=years,
                                        selected=[],
                                        multiple=True,
                                        selectize=True
                                    ),
                                    style="margin-bottom: 15px;"
                                ),
                                ui.input_select(
                                    "countries_select",
                                    "Selecciona los países",
                                    choices=countries,
                                    selected=[],
                                    multiple=True,
                                    selectize=True
                                ),
                                style="flex: 2; padding-right: 20px;"
                            ),
        
                            # Columna derecha: botones de descarga filtrada
                            ui.div(
                                ui.download_button("downloadData_uso_pesticidas", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_uso_pesticidas_json", "Descargar JSON Filtrado"),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
        
                    # Sección de citas y botón volver atrás
                    ui.div(
                            *[
                            ui.h3("📚 Citas", style="text-align: center; margin-top: 40px; font-family: 'Arial', sans-serif;"),
        
                            # Cita original de IHME
                            ui.p(
                                ui.HTML(
                                    'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2021 (GBD 2021). '
                                    'Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. '
                                    'Available from <a href="https://vizhub.healthdata.org/gbd-results/" target="_blank" '
                                    'style="color: black; text-decoration: underline;">https://vizhub.healthdata.org/gbd-results/</a>. '
                                    '<br><strong>Attribution short:</strong> "IHME-GBD".'
                                ),
                                style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                            ),
        
                            # Cita OWID extendida
                            ui.p(
                                ui.HTML(
                                    '“Data Page: Rate of disease burden from lead exposure”, part of the following publication: '
                                    'Esteban Ortiz-Ospina and Max Roser (2016) – “Global Health”. Data adapted from IHME, Global Burden of Disease. '
                                    'Retrieved from <a href="https://ourworldindata.org/grapher/pesticide-use-tonnes" target="_blank" '
                                    'style="color: black; text-decoration: underline;">https://ourworldindata.org/grapher/pesticide-use-tonnes</a> [online resource].'
                                ),
                                style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 20px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                            ),
        
                            # Cita OWID abreviada
                            ui.p(
                                "IHME, Global Burden of Disease (2024) – with minor processing by Our World in Data",
                                style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;"
                            ),
        
                            # Botón de volver atrás
                            ui.div(
                                ui.input_action_button(
                                    "volver_atras_pesticidas",
                                    "🔙 Volver Atrás",
                                    class_="btn btn-secondary",
                                    onclick="Shiny.setInputValue('page', 'section3')"
                                ),
                                style="text-align: center; margin-top: 30px;"
                            )
                        ]
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )

            


        elif page == "plot_europe_pesticidas":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'pesticidas')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_slider(
                        "year", "Selecciona el Año",
                        min=df_pepticidas["Año"].min(),
                        max=df_pepticidas["Año"].max(),
                        value=df_pepticidas["Año"].min(),
                        step=1,
                        sep=""
                    ),
                    class_="slider-box"
                ),
                ui.output_ui("plot_europe_pesticidas"),
                class_="content-box"
            )
            

        elif page == "precipitaciones":
            return ui.div(
                # Encabezado principal
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Descripción
                ui.p(
                    "Visualiza la cantidad total de precipitaciones anuales (lluvia y nieve) en cada país, medida como la profundidad del agua acumulada durante el año. "
                    "Este indicador refleja el volumen total de agua que cae sobre la superficie terrestre, excluyendo fenómenos como la niebla o el rocío.",
                    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_precipitaciones"),
        
                    # Slider de año
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_precipitaciones["Año"].min(), 
                                        max=df_precipitaciones["Año"].max(), 
                                        value=df_precipitaciones["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
        
                    # Botón para ir al mapa
                    ui.div(
                        ui.input_action_button("go_to_europe_precipitaciones", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_precipitaciones')")
                    ),
        
                    # Sección de descarga completa
                    ui.div(
                        ui.h3("📥 Descarga completa de datos", 
                              style="text-align: center; margin-top: 30px; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
                        ui.div(
                            ui.download_button("downloadAll_precipitaciones", "Descargar CSV "),
                            ui.download_button("downloadAll_precipitaciones_json", "Descargar JSON "),
                            style="display: flex; flex-direction: column; gap: 10px; align-items: center;"
                        )
                    ),
        
                    # Sección de filtrado por año y país
                    ui.div(
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-top: 40px; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
                        ui.div(
                            # Columna izquierda: selectores
                            ui.div(
                                ui.div(
                                    ui.input_select(
                                        "years_select",
                                        "Selecciona los años",
                                        choices=years,
                                        selected=[],
                                        multiple=True,
                                        selectize=True
                                    ),
                                    style="margin-bottom: 15px;"
                                ),
                                ui.input_select(
                                    "countries_select",
                                    "Selecciona los países",
                                    choices=countries,
                                    selected=[],
                                    multiple=True,
                                    selectize=True
                                ),
                                style="flex: 2; padding-right: 20px;"
                            ),
        
                            # Columna derecha: descarga de datos filtrados
                            ui.div(
                                ui.download_button("downloadData_precipitaciones", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_precipitaciones_json", "Descargar JSON Filtrado"),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
        
                    # Sección de citas y botón volver atrás
                    ui.div(
                            *[
                            ui.h3("📚 Citas", style="text-align: center; margin-top: 40px; font-family: 'Arial', sans-serif;"),
        
                            # Cita original de IHME
                            ui.p(
                                ui.HTML(
                                    'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2021 (GBD 2021). '
                                    'Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. '
                                    'Available from <a href="https://vizhub.healthdata.org/gbd-results/" target="_blank" '
                                    'style="color: black; text-decoration: underline;">https://vizhub.healthdata.org/gbd-results/</a>. '
                                    '<br><strong>Attribution short:</strong> "IHME-GBD".'
                                ),
                                style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                            ),
        
                            # Cita OWID extendida
                            ui.p(
                                ui.HTML(
                                    '“Data Page: Rate of disease burden from lead exposure”, part of the following publication: '
                                    'Esteban Ortiz-Ospina and Max Roser (2016) – “Global Health”. Data adapted from IHME, Global Burden of Disease. '
                                    'Retrieved from <a href="https://ourworldindata.org/grapher/pesticide-use-tonnes" target="_blank" '
                                    'style="color: black; text-decoration: underline;">https://ourworldindata.org/grapher/average-precipitation-per-year</a> [online resource].'
                                ),
                                style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 20px; text-align: justify; max-width: 800px; margin-left: auto; margin-right: auto;"
                            ),
        
                            # Cita OWID abreviada
                            ui.p(
                                "IHME, Global Burden of Disease (2024) – with minor processing by Our World in Data",
                                style="font-size: 14px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px; text-align: left; max-width: 800px; margin-left: auto; margin-right: auto;"
                            ),
        
                            # Botón de volver atrás
                            ui.div(
                                ui.input_action_button(
                                    "volver_atras_pesticidas",
                                    "🔙 Volver Atrás",
                                    class_="btn btn-secondary",
                                    onclick="Shiny.setInputValue('page', 'section3')"
                                ),
                                style="text-align: center; margin-top: 30px;"
                            )
                        ]
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )


        elif page == "plot_europe_precipitaciones":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'precipitaciones')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.div(
                    ui.input_slider(
                        "year", "Selecciona el Año",
                        min=df_precipitaciones["Año"].min(),
                        max=df_precipitaciones["Año"].max(),
                        value=df_precipitaciones["Año"].min(),
                        step=1,
                        sep=""
                    ),
                    class_="slider-box"
                ),
                ui.output_ui("plot_europe_precipitaciones"),
                class_="content-box"
            )

    
        
        elif page == "section4":
            return ui.div(
                # Franja de color con el título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.p(
    "Esta sección presenta visualizaciones geográficas relacionadas con la enfermedad de Parkinson. "
    "En primer lugar, se muestra un mapa con la prevalencia estimada por país según el modelo de predicción. "
    "A continuación, se visualiza la incertidumbre asociada a estas estimaciones, representada por la desviación estándar del modelo. "
    "Finalmente, se incluye un mapa de anomalías que refleja la diferencia entre los valores predichos y los reales, "
    "indicando posibles casos de sobreestimación o subestimación por parte del modelo.",
    style="max-width: 900px; text-align: justify; font-size: 16px; font-family: 'Arial', sans-serif; margin-bottom: 20px;"
),

        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_modelos_mapa"),
                    ui.output_ui("plot_modelos"),
                    ui.output_ui("plot_vs"),
                    style="display: flex; flex-direction: column; align-items: center; gap: 20px;"
                ),
                class_="content-box"
            )

        elif page == "section5":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.p(
                        ui.HTML(
                            "En esta sección se presenta un análisis global de la <strong>importancia de las variables</strong> utilizadas en los modelos de predicción de párkinson. "
                            "El gráfico que verás a continuación resume la influencia promedio de cada variable, calculada a partir de todos los modelos entrenados. "
                            "Esto proporciona una visión integral sobre qué factores tienen mayor peso en la predicción a nivel mundial. "
                            "Cuanto más bajo es el valor del ranking, mayor es la importancia de esa variable en los modelos. "
                            "Por ejemplo, la exposición al plomo muestra el ranking promedio más bajo, lo que indica que es una de las variables más influyentes y consistentes en la predicción de la enfermedad. "
                            "Este tipo de visualización permite identificar patrones comunes en los modelos y orientar futuras investigaciones o estrategias de intervención."
                        ),
                        style="font-size: 16px; font-family: 'Arial', sans-serif; text-align: justify; margin-bottom: 20px;"
                    ),
                    ui.output_ui("plot_ranking_global"),  # Aquí se renderiza el gráfico
                    ui.div(
                        ui.p(
                            "Para explorar los resultados de cada modelo de forma individual, selecciona una de las opciones disponibles a continuación.",
                            style="font-size: 16px; font-family: 'Arial', sans-serif; text-align: justify; margin: 0;"
                        ),
                        style="background-color: #f2f2f2; padding: 15px; border-radius: 8px; margin-top: 40px;"
                    ),
                ),
                ui.div(
                    ui.input_action_button("show_glm", "Modelo lineal", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelo GLM')"),
                    ui.input_action_button("show_tree_models", "Modelos basados en árboles", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelos_arboles')"),
                    ui.input_action_button("show_learning_models", "Otros modelos de Regression", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelos_learning')"),
                    style="display: flex; justify-content: space-around; margin: 30px 0 20px 0;"
                )
            )



        elif page == "modelo GLM":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.output_ui("plot_predict_glm"),
                    ui.div(
                        ui.input_action_button(
                            "plot_europe_predict_glm",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_predict_glm')"
                        ),
                        style="margin-top: 20px;"  # <-- AQUÍ ESTÁ EL ESPACIO
                    ),
                    ui.div(
                        ui.input_action_button(
                            "go_back",
                            "Volver atrás",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'section5')"
                        ),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )

        elif page == "plot_europe_predict_glm":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),


                
                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'modelo GLM')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.output_ui("plot_europe_predict_glm"),
                class_="content-box"
            )
        elif page == "modelos_arboles":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

        
                ui.div(  # Bloque explicativo justo debajo del título
                    ui.p(
                        "En esta sección puedes visualizar el comportamiento del modelo de predicción basado en bosques aleatorios (Random Forest). "
                        "Este modelo es capaz de capturar relaciones complejas entre las variables y ofrece un alto nivel de precisión. "
                        "También puedes explorar el modelo XGBoost, que es una técnica de boosting muy potente y ampliamente utilizada en competencias de ciencia de datos.",
                        style="font-size: 16px; font-family: 'Arial', sans-serif; text-align: justify; margin: 0;"
                    ),
                    style="background-color: #f2f2f2; padding: 15px; border-radius: 8px; margin-bottom: 30px;"
                ),
        
                ui.div(
                    ui.input_action_button("show_rf", "Random Forest", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelo RF')"),
                    ui.input_action_button("show_xg", "XGBoost Regressor", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelo XGBoost')"),
                    style="display: flex; justify-content: space-around; margin: 30px 0 20px 0;"
                ),
                
                ui.div(
                    ui.input_action_button("go_back", "🔙 Volver", class_="btn btn-secondary", onclick="Shiny.setInputValue('page', 'section5')"),
                    style="text-align: center; margin-top: 20px;"
                )
            )

        elif page == "modelo RF":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.output_ui("plot_predict_rf"),
                    ui.div(
                        ui.input_action_button(
                            "plot_europe_predict_rf",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_predict_rf')"
                        ),
                        style="margin-top: 20px;"  # <-- AQUÍ ESTÁ EL ESPACIO
                    ),
                    ui.div(
                        ui.input_action_button(
                            "go_back",
                            "Volver atrás",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'modelos_arboles')"
                        ),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )

        elif page == "plot_europe_predict_rf":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'modelo RF')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.output_ui("plot_europe_predict_rf"),
                class_="content-box"
            )
        elif page == "modelo XGBoost":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.output_ui("plot_predict_xg"),
                    ui.div(
                        ui.input_action_button(
                            "plot_europe_predict_xg",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_predict_xg')"
                        ),
                        style="margin-top: 20px;"  # <-- AQUÍ ESTÁ EL ESPACIO
                    ),
                    ui.div(
                        ui.input_action_button(
                            "go_back",
                            "Volver atrás",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'modelos_arboles')"
                        ),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )

        elif page == "plot_europe_predict_xg":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'modelo XGBoost')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.output_ui("plot_europe_predict_xg"),
                class_="content-box"
            )

        elif page == "modelos_learning":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(  # Descripción general
                    ui.p(
                        "En esta sección puedes explorar distintos modelos de regresión aplicados a la predicción de Parkinson, distintos de los enfoques basados en árboles. "
                        "Al hacer clic en los botones que aparecen a continuación, podrás visualizar cómo se comportan modelos como SVR, KNN y MLP, y comparar sus resultados. "
                        "Estos modelos capturan patrones en los datos mediante aproximaciones basadas en distancia, funciones kernel y redes neuronales, respectivamente, lo que te permitirá analizar su rendimiento desde diferentes perspectivas.",
                        style="font-size: 16px; font-family: 'Arial', sans-serif; text-align: justify; margin: 0;"
                    ),
                    style="background-color: #f2f2f2; padding: 15px; border-radius: 8px; margin-bottom: 20px;"
                ),
                        
                ui.div(  # Descripciones de cada modelo
                    ui.p(
                        "• SVR (Support Vector Regressor): utiliza los principios de las máquinas de soporte vectorial para realizar regresiones precisas, "
                        "siendo especialmente útil cuando existen relaciones no lineales entre las variables.\n\n"
                        "• KNN (K-Nearest Neighbors): predice el valor de un punto en función de sus 'k' vecinos más cercanos. "
                        "Es un modelo simple pero eficaz cuando los datos están bien distribuidos.\n\n"
                        "• MLP (Multi-Layer Perceptron): es una red neuronal con múltiples capas ocultas que permite aprender representaciones complejas, "
                        "lo que la hace poderosa para captar patrones no evidentes.",
                        style="font-size: 15px; font-family: 'Arial', sans-serif; white-space: pre-line; text-align: justify; margin-bottom: 30px;"
                    ),
                    style="background-color: #e8e8e8; padding: 15px; border-radius: 8px;"
                ),
                ui.div(
                    ui.input_action_button("show_svr", "SVR Regressor", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelo SVR')"),
                    ui.input_action_button("show_knn", "KNN Regressor", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelo KNN')"),
                    ui.input_action_button("show_mlp", "MLP Regressor", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelo MLP')"),
                    style="display: flex; justify-content: space-around; margin: 30px 0 20px 0;"
                ),
                ui.div(
                    ui.input_action_button("go_back", "🔙 Volver", class_="btn btn-secondary", onclick="Shiny.setInputValue('page', 'section5')"),
                    style="text-align: center; margin-top: 20px;"
                )
            )
        elif page == "modelo SVR":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.output_ui("plot_predict_svr"),
                    ui.div(
                        ui.input_action_button(
                            "plot_europe_predict_svr",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_predict_svr')"
                        ),
                        style="margin-top: 20px;"  # <-- AQUÍ ESTÁ EL ESPACIO
                    ),
                    ui.div(
                        ui.input_action_button(
                            "go_back",
                            "Volver atrás",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'modelos_learning')"
                        ),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )

        elif page == "plot_europe_predict_svr":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'modelo SVR')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.output_ui("plot_europe_predict_svr"),
                class_="content-box"
            )

        elif page == "modelo KNN":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.output_ui("plot_predict_knn"),
                    ui.div(
                        ui.input_action_button(
                            "plot_europe_predict_knn",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_predict_knn')"
                        ),
                        style="margin-top: 20px;"  # <-- AQUÍ ESTÁ EL ESPACIO
                    ),
                    ui.div(
                        ui.input_action_button(
                            "go_back",
                            "Volver atrás",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'modelos_learning')"
                        ),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )

        elif page == "plot_europe_predict_knn":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'modelo KNN')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.output_ui("plot_europe_predict_knn"),
                class_="content-box"
            )

        elif page == "modelo MLP":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.output_ui("plot_predict_mlp"),
                    ui.div(
                        ui.input_action_button(
                            "plot_europe_predict_mlp",
                            "🌍 Ver Mapa Europeo",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'plot_europe_predict_mlp')"
                        ),
                        style="margin-top: 20px;"  # <-- AQUÍ ESTÁ EL ESPACIO
                    ),
                    ui.div(
                        ui.input_action_button(
                            "go_back",
                            "Volver atrás",
                            class_="btn btn-primary",
                            onclick="Shiny.setInputValue('page', 'modelos_learning')"
                        ),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )

        elif page == "plot_europe_predict_mlp":
            return ui.div(
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                ui.div(
                    ui.input_action_button(
                        "go_back", 
                        "🔙 Volver al Mapa Global", 
                        class_="btn btn-secondary",
                        onclick="Shiny.setInputValue('page', 'modelo MLP')"
                    ),
                    style="margin-bottom: 20px;"
                ),
                ui.output_ui("plot_europe_predict_mlp"),
                class_="content-box"
            )       
    
            
        elif page == "section6":
            return ui.div(
                # Franja de color con el título
                ui.div(
                    ui.h1(
                        ui.HTML("🌍 Parkinson <em>Worldwide</em>"),
                        style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),

                
                # Cuerpo con información de contacto (Estilo atractivo)
                ui.div(
                    ui.p("¿Tienes preguntas, sugerencias o quieres colaborar? Ponte en contacto conmigo:",
                         style="font-size: 18px; margin: 20px; text-align: center; color: #333;"
                    ),
                    ui.div(
                        # Correo Electrónico
                        ui.div(
                            ui.h3("📧 Correo Electrónico", style="font-size: 20px; color: #8E44AD; text-align: center;"),
                            ui.p("Envíame un correo para cualquier consulta o inquietud.", style="text-align: center; font-size: 16px;"),
                            # Enlace mailto para el correo
                            ui.p(ui.tags.a("lcp1009@alu.ubu.es", href="mailto:tuemail@gmail.com", target="_blank", style="color: #2980B9; font-size: 18px;")),
                            style="background-color: #F2F3F4; padding: 20px; margin: 10px 0; border-radius: 8px;"
                        ),
                        
                        # GitHub
                        ui.div(
                            ui.h3("💻 GitHub", style="font-size: 20px; color: #8E44AD; text-align: center;"),
                            ui.p("Visita mi perfil de GitHub para ver otros proyectos y colaboraciones.", style="text-align: center; font-size: 16px;"),
                            ui.p(ui.tags.a("github.com/Lorenacalvoperez", href="https://github.com/Lorenacalvoperez/Trabajo-Fin-de-Grado", target="_blank", style="color: #2980B9; font-size: 18px;")),
                            style="background-color: #F2F3F4; padding: 20px; margin: 10px 0; border-radius: 8px;"
                        ),
                        
                        style="margin: 0 10px;"
                    ),
                    style="text-align: center; margin-top: 20px;"
                )
            )
    
    

    @output
    @render.download(filename="Parkinson_filtrado.csv")
    def downloadData():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_parkinson[df_parkinson['Año'].isin(selected_years) & df_parkinson['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_parkinson[df_parkinson['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_parkinson[df_parkinson['País'].isin(selected_countries)]
        else:
            filtered_df = df_parkinson  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer


    @output
    @render.download(filename="Parkinson_completo.csv")
    def downloadAll():
        buffer = io.StringIO()
        df_parkinson.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Parkinson_filtrado.json")
    def downloadData():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_parkinson[df_parkinson['Año'].isin(selected_years) & df_parkinson['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_parkinson[df_parkinson['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_parkinson[df_parkinson['País'].isin(selected_countries)]
        else:
            filtered_df = df_parkinson  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer


    @output
    @render.download(filename="Parkinson_filtrado.json")
    def downloadData_parkinson_json():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
    
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_parkinson[df_parkinson['Año'].isin(selected_years) & df_parkinson['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_parkinson[df_parkinson['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_parkinson[df_parkinson['País'].isin(selected_countries)]
        else:
            filtered_df = df_parkinson  # Si no se selecciona ningún filtro, usar el DataFrame completo
    
        # Convertir DataFrame a JSON (lista de registros)
        json_str = filtered_df.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

   


    @output
    @render.download(filename="Parkinson_filtrado_completo.json")
    def downloadAll_parkinson_json():
        json_str = df_parkinson.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer
    


    @output
    @render.download(filename="Tasa_contaminacion_aire_completo.csv")
    def downloadAll_json():
        buffer = io.StringIO()
        df_contaminacion.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
        

    @output
    @render.download(filename="Tasa_contaminacion_aire_filtrado.csv")
    def downloadData_contaminacion():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_contaminacion[df_contaminacion['Año'].isin(selected_years) & df_contaminacion['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_contaminacion[df_contaminacion['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_contaminacion[df_contaminacion['País'].isin(selected_countries)]
        else:
            filtered_df = df_contaminacion  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
    


    @output
    @render.download(filename="Tasa_contaminacion_aire_completo.csv")
    def downloadAll_contaminacion():
        buffer = io.StringIO()
        df_contaminacion.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
        
    @output
    @render.download(filename="Tasa_contaminacion_aire_filtrado.json")
    def downloadData_contaminacion_json():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
    
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_contaminacion[df_contaminacion['Año'].isin(selected_years) & df_contaminacion['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_contaminacion[df_contaminacion['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_contaminacion[df_contaminacion['País'].isin(selected_countries)]
        else:
            filtered_df = df_contaminacion  # Si no se selecciona ningún filtro, usar el DataFrame completo
    
        # Convertir DataFrame a JSON (lista de registros)
        json_str = filtered_df.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

   


    @output
    @render.download(filename="Tasa_contaminacion_aire_completo.json")
    def downloadAll_contaminacion_json():
        json_str = df_contaminacion.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer
        
    @output
    @render.download(filename="Exposicion_plomo_filtrado.csv")
    def downloadData_exposicion_plomo():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_plomo[df_plomo['Año'].isin(selected_years) & df_plomo['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_plomo[df_plomo['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_plomo[df_plomo['País'].isin(selected_countries)]
        else:
            filtered_df = df_plomo  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Exposicion_plomo_completo.csv")
    def downloadAll_exposicion_plomo():
        buffer = io.StringIO()
        df_plomo.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
        
    @output
    @render.download(filename="Exposicion_plomo_filtrado.json")
    def downloadData_exposicion_plomo_json():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
    
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_plomo[df_plomo['Año'].isin(selected_years) & df_plomo['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_plomo[df_plomo['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_plomo[df_plomo['País'].isin(selected_countries)]
        else:
            filtered_df = df_plomo  # Si no se selecciona ningún filtro, usar el DataFrame completo
    
        # Convertir DataFrame a JSON (lista de registros)
        json_str = filtered_df.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

   


    @output
    @render.download(filename="Exposicion_plomo_completo.json")
    def downloadAll_exposicion_plomo_json():
        json_str = df_plomo.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Muertes_agua_filtrado.csv")
    def downloadData_muertes_agua():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_agua[df_agua['Año'].isin(selected_years) & df_agua['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_agua[df_agua['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_agua[df_agua['País'].isin(selected_countries)]
        else:
            filtered_df = df_agua  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Muertes_agua_completo.csv")
    def downloadAll_muertes_agua():
        buffer = io.StringIO()
        df_agua.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Muertes_agua_filtrado.json")
    def downloadData_muertes_agua_json():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
    
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_agua[df_agua['Año'].isin(selected_years) & df_agua['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_agua[df_agua['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_agua[df_agua['País'].isin(selected_countries)]
        else:
            filtered_df = df_agua # Si no se selecciona ningún filtro, usar el DataFrame completo
    
        # Convertir DataFrame a JSON (lista de registros)
        json_str = filtered_df.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

   


    @output
    @render.download(filename="Muertes_agua_completo.json")
    def downloadAll_muertes_agua_json():
        json_str = df_agua.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer
    
    @output
    @render.download(filename="Uso_pesticidas_filtrado.csv")
    def downloadData_uso_pesticidas():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_pepticidas[df_pepticidas['Año'].isin(selected_years) & df_pepticidas['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_pepticidas[df_pepticidas['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_pepticidas[df_pepticidas['País'].isin(selected_countries)]
        else:
            filtered_df = df_pepticidas  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Uso_pesticidas_completo.csv")
    def downloadAll_uso_pesticidas():
        buffer = io.StringIO()
        df_pepticidas.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Uso_pesticidas_filtrado.json")
    def downloadData_uso_pesticidas_json():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
    
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_pepticidas[df_pepticidas['Año'].isin(selected_years) & df_pepticidas['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_pepticidas[df_pepticidas['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df =df_pepticidas[df_pepticidas['País'].isin(selected_countries)]
        else:
            filtered_df = df_pepticidas # Si no se selecciona ningún filtro, usar el DataFrame completo
    
        # Convertir DataFrame a JSON (lista de registros)
        json_str = filtered_df.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

   


    @output
    @render.download(filename="Uso_pesticidas_completo.json")
    def downloadAll_uso_pesticidas_json():
        json_str = df_pepticidas.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Precipitaciones_completo.csv")
    def downloadAll_precipitaciones():
        buffer = io.StringIO()
        df_precipitaciones.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Precipitaciones_filtrado.csv")
    def downloadData_precipitaciones():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
        
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_precipitaciones[df_precipitaciones['Año'].isin(selected_years) & df_precipitaciones['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_precipitaciones[df_precipitaciones['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df = df_precipitaciones[df_precipitaciones['País'].isin(selected_countries)]
        else:
            filtered_df = df_precipitaciones  # Si no se selecciona ningún filtro, usar el DataFrame completo
        
        buffer = io.StringIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Precipitaciones_filtrado.json")
    def downloadData_precipitaciones_json():
        selected_years = [int(year) for year in input.years_select()]
        selected_countries = input.countries_select()  # Obtener los países seleccionados
    
        # Filtrar los datos por los años y países seleccionados
        if selected_years and selected_countries:
            filtered_df = df_precipitaciones[df_precipitaciones['Año'].isin(selected_years) & df_precipitaciones['País'].isin(selected_countries)]
        elif selected_years:
            filtered_df = df_precipitaciones[df_pepticidas['Año'].isin(selected_years)]
        elif selected_countries:
            filtered_df =df_precipitaciones[df_precipitaciones['País'].isin(selected_countries)]
        else:
            filtered_df = df_precipitaciones # Si no se selecciona ningún filtro, usar el DataFrame completo
    
        # Convertir DataFrame a JSON (lista de registros)
        json_str = filtered_df.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

   


    @output
    @render.download(filename="Precipitaciones_completo.json")
    def downloadAll_precipitaciones_json():
        json_str = df_precipitaciones.to_json(orient="records", indent=2)
    
        buffer = io.StringIO()
        buffer.write(json_str)
        buffer.seek(0)
        return buffer

    
    @output
    @render.ui
    def plot_parkinson():
        año_seleccionado = input.year()  # Capturamos el año seleccionado en el slider
        df_filtrado = df_parkinson[df_parkinson["Año"] == año_seleccionado]
    
        fig_parkinson_filtrado = px.choropleth(
            df_filtrado,
            locations="País",
            locationmode="country names",
            color="Parkinson",
            hover_name="País",
            hover_data={"Parkinson": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_parkinson, q95_parkinson),
            title=f"Prevalencia del Parkinson por País y Año - {año_seleccionado}"
        )
    
        fig_parkinson_filtrado.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_parkinson_filtrado.update_layout(
            title={
                'text': f"<b>Prevalencia de párkinson por País y Año - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Número estimado de<br>casos de párkinson",
                tickvals=[min_parkinson, q25_parkinson, q50_parkinson, q75_parkinson, q95_parkinson],
                ticktext=[
                    f"Mín: {min_parkinson}",
                    f"Q25: {q25_parkinson}",
                    f"Q50: {q50_parkinson}",
                    f"Q75: {q75_parkinson}",
                    f"Q95: {q95_parkinson}"
                ]
            )
        )
    
        return ui.HTML(fig_parkinson_filtrado.to_html(full_html=False))

    @output
    @render.ui
    def plot_europe():
        año_seleccionado = input.year()
        
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar por Europa y año seleccionado
        df_europa = df_parkinson[(df_parkinson["País"].isin(paises_europa)) & (df_parkinson["Año"] == año_seleccionado)]
    
        # Calcular cuantiles para el campo Parkinson
        min_parkinson = round(df_europa["Parkinson"].min(), 2)
        q25 = round(df_europa["Parkinson"].quantile(0.25), 2)
        q50 = round(df_europa["Parkinson"].quantile(0.50), 2)
        q75 = round(df_europa["Parkinson"].quantile(0.75), 2)
        q95 = round(df_europa["Parkinson"].quantile(0.95), 2)
    
        # Crear el choropleth con rango hasta q95
        fig_europa = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson",
            hover_name="País",
            hover_data={"Parkinson": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_parkinson, q95),
            title=f"Prevalencia de párkinson en Europa por País y Año - {año_seleccionado}"
        )
    
        fig_europa.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        # Personalizar barra de color con cuantiles
        fig_europa.update_layout(
            title={
                'text': f"<b>Prevalencia de párkinson en Europa por País y Año - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,  # altura de barra de color (0.8 es acorde al mapa)
                thickness=25,
                y=0.5,
                title="Número estimado de<br>casos de párkinson",
                tickvals=[min_parkinson, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_parkinson}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa.to_html(full_html=False))


    @output
    @render.ui
    def plot_contaminacion():
        año_seleccionado = input.year()
        df_filtrado = df_contaminacion[df_contaminacion["Año"] == año_seleccionado]
    
        fig_contaminacion_filtrado = px.choropleth(
            df_filtrado,
            locations="País",
            locationmode="country names",
            color="Contaminacion_aire",
            hover_name="País",
            hover_data={"Contaminacion_aire": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_contaminacion, q95_contaminacion),
            labels={"Contaminacion_aire": "Tasa de mortalidad por<br>contaminación del aire"},
            title=f"Contaminación del Aire - {año_seleccionado}"
        )
    
        fig_contaminacion_filtrado.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_contaminacion_filtrado.update_layout(
            title={
                'text': f"<b>Contaminación del Aire - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Tasa de mortalidad<br>por contaminación",
                tickvals=[min_contaminacion, q25_contaminacion, q50_contaminacion, q75_contaminacion, q95_contaminacion],
                ticktext=[
                    f"Mín: {min_contaminacion}",
                    f"Q25: {q25_contaminacion}",
                    f"Q50: {q50_contaminacion}",
                    f"Q75: {q75_contaminacion}",
                    f"Q95: {q95_contaminacion}"
                ]
            )
        )
    
        return ui.HTML(fig_contaminacion_filtrado.to_html(full_html=False))
        

        fig_contaminacion_filtrado.update_layout(
        height=400,  # Hacerlo más grande
        margin={"r":0,"t":50,"l":0,"b":0}
    )

        
        return ui.HTML(fig_contaminacion_filtrado.to_html(full_html=False))

    @output
    @render.ui
    def plot_europe_aire():
        año_seleccionado = input.year()
    
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar datos por países y año seleccionado
        df_europa = df_contaminacion[(df_contaminacion["País"].isin(paises_europa)) & (df_contaminacion["Año"] == año_seleccionado)]
    
        # Calcular rango para el color basado en cuantiles, igual que con Parkinson
        min_contaminacion = round(df_europa["Contaminacion_aire"].min(), 2)
        q25 = round(df_europa["Contaminacion_aire"].quantile(0.25), 2)
        q50 = round(df_europa["Contaminacion_aire"].quantile(0.50), 2)
        q75 = round(df_europa["Contaminacion_aire"].quantile(0.75), 2)
        q95 = round(df_europa["Contaminacion_aire"].quantile(0.95), 2)
    
        fig_europa_Aire = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Contaminacion_aire",
            hover_name="País",
            hover_data={"Contaminacion_aire": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_contaminacion, q95),
            title=f"Contaminación del Aire en Europa - {año_seleccionado}"
        )
    
        fig_europa_Aire.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_Aire.update_layout(
            title={
                'text': f"<b>Contaminación de aire en Europa - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Tasa de mortalidad por<br>contaminación del aire",
                tickvals=[min_contaminacion, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_contaminacion}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_Aire.to_html(full_html=False))



    @output
    @render.ui
    def plot_plomo():
        año_seleccionado = input.year()
        df_filtrado = df_plomo[df_plomo["Año"] == año_seleccionado]
    
        fig_plomo_filtrado = px.choropleth(
            df_filtrado,
            locations="País",
            locationmode="country names",
            color="Exp_plomo",
            hover_name="País",
            hover_data={"Exp_plomo": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_plomo, q95_plomo),
            labels={"Exp_plomo": "Impacto en la salud<br>por exposición al plomo"},
            title=f"Exposición al Plomo - {año_seleccionado}"
        )
    
        fig_plomo_filtrado.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_plomo_filtrado.update_layout(
            title={
                'text': f"<b>Exposición al Plomo - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Carga de enfermedad<br>por plomo",
                tickvals=[min_plomo, q25_plomo, q50_plomo, q75_plomo, q95_plomo],
                ticktext=[
                    f"Mín: {min_plomo}",
                    f"Q25: {q25_plomo}",
                    f"Q50: {q50_plomo}",
                    f"Q75: {q75_plomo}",
                    f"Q95: {q95_plomo}"
                ]
            )
        )
    
        return ui.HTML(fig_plomo_filtrado.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_plomo():
        año_seleccionado = input.year()
    
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar datos por países y año
        df_europa = df_plomo[(df_plomo["País"].isin(paises_europa)) & (df_plomo["Año"] == año_seleccionado)]
    
        # Calcular cuantiles para Exp_plomo
        min_plomo = round(df_europa["Exp_plomo"].min(), 2)
        q25 = round(df_europa["Exp_plomo"].quantile(0.25), 2)
        q50 = round(df_europa["Exp_plomo"].quantile(0.50), 2)
        q75 = round(df_europa["Exp_plomo"].quantile(0.75), 2)
        q95 = round(df_europa["Exp_plomo"].quantile(0.95), 2)
    
        fig_europa_plomo = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Exp_plomo",
            hover_name="País",
            hover_data={"Exp_plomo": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_plomo, q95),
            title=f"Exposición al Plomo - {año_seleccionado}"
        )
    
        fig_europa_plomo.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_plomo.update_layout(
            title={
                'text': f"<b>Exposición al Plomo - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Impacto en la salud <br>por exposición al plomo",
                tickvals=[min_plomo, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_plomo}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_plomo.to_html(full_html=False))


    @output
    @render.ui
    def plot_agua():
        año_seleccionado = input.year()
    
        fig_agua_filtrado = px.choropleth(
            df_agua[df_agua["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Muertes_agua",
            hover_name="País",
            hover_data={"Muertes_agua": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_agua, q95_agua),
            labels={"Muertes_agua": "Muertes por fuentes<br>de agua inseguras"},
            title=f"Muertes por Agua Insalubre - {año_seleccionado}"
        )
    
        fig_agua_filtrado.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_agua_filtrado.update_layout(
            height=400,
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.97,
                thickness=25,
                y=0.5,
                title="Muertes por agua<br>insalubre",
                tickvals=[min_agua, q75_agua,q95_agua],
                ticktext=[f"Mín: {min_agua}", f"Q75: {q75_agua}",f"Q95: {q95_agua}"]
                
            )
        )
    
        return ui.HTML(fig_agua_filtrado.to_html(full_html=False))



    @output
    @render.ui
    def plot_europe_agua():
        año_seleccionado = input.year()
    
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar datos
        df_europa = df_agua[(df_agua["País"].isin(paises_europa)) & (df_agua["Año"] == año_seleccionado)]
    
        # Calcular cuantiles para Muertes_agua
        min_agua = round(df_europa["Muertes_agua"].min(), 2)
        q25 = round(df_europa["Muertes_agua"].quantile(0.25), 2)
        q50 = round(df_europa["Muertes_agua"].quantile(0.50), 2)
        q75 = round(df_europa["Muertes_agua"].quantile(0.75), 2)
        q95 = round(df_europa["Muertes_agua"].quantile(0.95), 2)
    
        fig_europa_agua = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Muertes_agua",
            hover_name="País",
            hover_data={"Muertes_agua": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_agua, q95),
            title=f"Muertes de agua - {año_seleccionado}"
        )
    
        fig_europa_agua.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_agua.update_layout(
            title={
                'text': f"<b>Muertes de agua - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Muertes por fuentes<br>de agua inseguras",
                tickvals=[min_agua,q50, q75, q95],
                ticktext=[
                    f"Mín: {min_agua}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_agua.to_html(full_html=False))

    
    
    @output
    @render.ui
    def plot_pesticidas():
        año_seleccionado = input.year()
        df_filtrado = df_pepticidas[df_pepticidas["Año"] == año_seleccionado]
    
        fig_pesticidas = px.choropleth(
            df_filtrado,
            locations="País",
            locationmode="country names",
            color="Pesticidas",
            hover_name="País",
            hover_data={"Pesticidas": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_pesticidas, q95_pesticidas),
            labels={"Pesticidas": "Uso de pesticidas<br>(Toneladas)"},
            title=f"Uso de Pesticidas - {año_seleccionado}"
        )
    
        fig_pesticidas.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_pesticidas.update_layout(
            title={
                'text': f"<b>Uso de Pesticidas por País - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Toneladas<br>de pesticidas",
                tickvals=[min_pesticidas,q75_pesticidas, q95_pesticidas],
                ticktext=[
                    f"Mín: {min_pesticidas}",
                    f"Q75: {q75_pesticidas}",
                    f"Q95: {q95_pesticidas}"
                ]
            )
        )
    
        return ui.HTML(fig_pesticidas.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_pesticidas():
        año_seleccionado = input.year()
    
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        df_europa = df_pepticidas[(df_pepticidas["País"].isin(paises_europa)) & (df_pepticidas["Año"] == año_seleccionado)]
    
        # Calcular cuantiles para el campo Pesticidas
        min_pest = round(df_europa["Pesticidas"].min(), 2)
        q25 = round(df_europa["Pesticidas"].quantile(0.25), 2)
        q50 = round(df_europa["Pesticidas"].quantile(0.50), 2)
        q75 = round(df_europa["Pesticidas"].quantile(0.75), 2)
        q95 = round(df_europa["Pesticidas"].quantile(0.95), 2)
    
        fig_europa_pesticidas = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Pesticidas",
            hover_name="País",
            hover_data={"Pesticidas": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_pest, q95),
            title=f"Exposición a pesticidas - {año_seleccionado}"
        )
    
        fig_europa_pesticidas.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_pesticidas.update_layout(
            title={
                'text': f"<b>Uso de pesticidas - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Uso de pesticidas<br>(Toneladas)",
                tickvals=[min_pest,q50, q75, q95],
                ticktext=[
                    f"Mín: {min_pest}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_pesticidas.to_html(full_html=False))

    
    @output
    @render.ui
    def plot_precipitaciones():
        año_seleccionado = input.year()
        df_filtrado = df_precipitaciones[df_precipitaciones["Año"] == año_seleccionado]
    
        fig_precipitaciones = px.choropleth(
            df_filtrado,
            locations="País",
            locationmode="country names",
            color="Precipitaciones",
            hover_name="País",
            hover_data={"Precipitaciones": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_precipitaciones, q95_precipitaciones),
            labels={"Precipitaciones": "Precipitación<br>(mm)"},
            title=f"Precipitaciones - {año_seleccionado}"
        )
    
        fig_precipitaciones.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_precipitaciones.update_layout(
            title={
                'text': f"<b>Precipitaciones por País - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.90,
                thickness=25,
                y=0.5,
                title="Precipitación<br>(mm)",
                tickvals=[
                    min_precipitaciones,
                    q25_precipitaciones,
                    q50_precipitaciones,
                    q75_precipitaciones,
                    q95_precipitaciones
                ],
                ticktext=[
                    f"Mín: {min_precipitaciones}",
                    f"Q25: {q25_precipitaciones}",
                    f"Q50: {q50_precipitaciones}",
                    f"Q75: {q75_precipitaciones}",
                    f"Q95: {q95_precipitaciones}"
                ]
            )
        )
    
        return ui.HTML(fig_precipitaciones.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_precipitaciones():
        año_seleccionado = input.year()
    
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        df_europa = df_precipitaciones[(df_precipitaciones["País"].isin(paises_europa)) & (df_precipitaciones["Año"] == año_seleccionado)]
    
        # Calcular cuantiles para el campo Precipitaciones
        min_prec = round(df_europa["Precipitaciones"].min(), 2)
        q25 = round(df_europa["Precipitaciones"].quantile(0.25), 2)
        q50 = round(df_europa["Precipitaciones"].quantile(0.50), 2)
        q75 = round(df_europa["Precipitaciones"].quantile(0.75), 2)
        q95 = round(df_europa["Precipitaciones"].quantile(0.95), 2)
    
        fig_europa_precipitaciones = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Precipitaciones",
            hover_name="País",
            hover_data={"Precipitaciones": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_prec, q95),
            title=f"Precipitaciones - {año_seleccionado}"
        )
    
        fig_europa_precipitaciones.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_precipitaciones.update_layout(
            title={
                'text': f"<b>Precipitaciones - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Cantidad de<br>Precipitación (mm)",
                tickvals=[min_prec, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_prec}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_precipitaciones.to_html(full_html=False))



    @output
    @render.ui
    def plot_predict_glm():
        fig_glm = px.choropleth(
            data_frame=df_predicciones_GLM,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_glm, q95_glm),  # 🔽 Recorte en Q95 como hiciste en el otro
            title=f"Predicción Prevalencia del Parkinson GLM"
        )
    
        fig_glm.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_glm.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia de párkinson GLM</b>",
                'font': {'size': 20},
                'x': 0.6,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Prevalencia<br> de párkinson",
                tickvals=[min_glm, q25_glm, q50_glm, q75_glm, q95_glm],
                ticktext=[
                    f"Mín: {min_glm}",
                    f"Q25: {q25_glm}",
                    f"Q50: {q50_glm}",
                    f"Q75: {q75_glm}",
                    f"Q95: {q95_glm}"
                ]
            )
        )
    
        return ui.HTML(fig_glm.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_predict_glm():
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar países europeos
        df_europa = df_predicciones_GLM[df_predicciones_GLM["País"].isin(paises_europa)]
    
        # Calcular cuantiles
        min_val = round(df_europa["Parkinson_Predicho"].min(), 2)
        q25 = round(df_europa["Parkinson_Predicho"].quantile(0.25), 2)
        q50 = round(df_europa["Parkinson_Predicho"].quantile(0.50), 2)
        q75 = round(df_europa["Parkinson_Predicho"].quantile(0.75), 2)
        q95 = round(df_europa["Parkinson_Predicho"].quantile(0.95), 2)
    
        # Crear mapa
        fig_europa_glm = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_val, q95),
            title="Predicción Prevalencia de párkinson GLM"
        )
    
        fig_europa_glm.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_glm.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson GLM</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Prevalencia<br> de párkinson",
                tickvals=[min_val, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_val}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_glm.to_html(full_html=False))


    @output
    @render.ui
    def plot_predict_rf():
        fig_rf = px.choropleth(
            data_frame=df_predicciones_RF,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_rf, q95_rf),  # Limita la escala hasta Q95
            title="Predicción Prevalencia de párkinson RF"
        )
    
        fig_rf.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_rf.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson RF</b>",
                'font': {'size': 20},
                'x': 0.6,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Prevalencia<br>de párkinson",
                tickvals=[min_rf, q25_rf, q50_rf, q75_rf, q95_rf],
                ticktext=[
                    f"Mín: {min_rf}",
                    f"Q25: {q25_rf}",
                    f"Q50: {q50_rf}",
                    f"Q75: {q75_rf}",
                    f"Q95: {q95_rf}"
                ]
            )
        )
    
        return ui.HTML(fig_rf.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_predict_rf():
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar países europeos
        df_europa = df_predicciones_RF[df_predicciones_RF["País"].isin(paises_europa)]
    
        # Calcular cuantiles para escala
        min_val = round(df_europa["Parkinson_Predicho"].min(), 4)
        q25 = round(df_europa["Parkinson_Predicho"].quantile(0.25), 4)
        q50 = round(df_europa["Parkinson_Predicho"].quantile(0.50), 4)
        q75 = round(df_europa["Parkinson_Predicho"].quantile(0.75), 4)
        q95 = round(df_europa["Parkinson_Predicho"].quantile(0.95), 4)
    
        # Crear choropleth
        fig_europa_rf = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_val, q95),
            title="Predicción Prevalencia de párkinson RF"
        )
    
        fig_europa_rf.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_rf.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson RF</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Prevalencia<br> de párkinson",
                tickvals=[min_val, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_val}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_rf.to_html(full_html=False))


    @output
    @render.ui
    def plot_predict_xg():
        fig_xg = px.choropleth(
            data_frame=df_predicciones_XG,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_xg, q95_xg),
            title="Predicción Prevalencia de párkinson XGBoost Regressor"
        )
    
        fig_xg.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_xg.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson XGBoost Regressor</b>",
                'font': {'size': 20},
                'x': 0.65,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Prevalencia<br> de párkinson",
                tickvals=[min_xg, q25_xg, q50_xg, q75_xg, q95_xg],
                ticktext=[
                    f"Mín: {min_xg}",
                    f"Q25: {q25_xg}",
                    f"Q50: {q50_xg}",
                    f"Q75: {q75_xg}",
                    f"Q95: {q95_xg}"
                ]
            )
        )
    
        return ui.HTML(fig_xg.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_predict_xg():
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        # Filtrar solo países europeos
        df_europa = df_predicciones_XG[df_predicciones_XG["País"].isin(paises_europa)]
    
        # Calcular cuantiles
        min_val = round(df_europa["Parkinson_Predicho"].min(), 4)
        q25 = round(df_europa["Parkinson_Predicho"].quantile(0.25), 4)
        q50 = round(df_europa["Parkinson_Predicho"].quantile(0.50), 4)
        q75 = round(df_europa["Parkinson_Predicho"].quantile(0.75), 4)
        q95 = round(df_europa["Parkinson_Predicho"].quantile(0.95), 4)
    
        # Crear choropleth
        fig_europa_xg = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_val, q95),
            title="Predicción Prevalencia de párkinson XGBoost Regressor"
        )
    
        fig_europa_xg.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_xg.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson XGBoost Regressor</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Prevalencia<br> de párkinson",
                tickvals=[min_val, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_val}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_xg.to_html(full_html=False))

        
    @output
    @render.ui
    def plot_predict_svr():
        fig_svr = px.choropleth(
            data_frame=df_predicciones_SVR,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_svr, q95_svr),
            title="Predicción Prevalencia de párkinson SVR Regressor"
        )
    
        fig_svr.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_svr.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson SVR Regressor</b>",
                'font': {'size': 20},
                'x': 0.6,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Prevalencia<br>de párkinson",
                tickvals=[min_svr, q25_svr, q50_svr, q75_svr, q95_svr],
                ticktext=[
                    f"Mín: {min_svr}",
                    f"Q25: {q25_svr}",
                    f"Q50: {q50_svr}",
                    f"Q75: {q75_svr}",
                    f"Q95: {q95_svr}"
                ]
            )
        )
    
        return ui.HTML(fig_svr.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_predict_svr():
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        df_europa = df_predicciones_SVR[df_predicciones_SVR["País"].isin(paises_europa)]
    
        # Calcular cuantiles
        min_val = round(df_europa["Parkinson_Predicho"].min(), 4)
        q25 = round(df_europa["Parkinson_Predicho"].quantile(0.25), 4)
        q50 = round(df_europa["Parkinson_Predicho"].quantile(0.50), 4)
        q75 = round(df_europa["Parkinson_Predicho"].quantile(0.75), 4)
        q95 = round(df_europa["Parkinson_Predicho"].quantile(0.95), 4)
    
        fig_europa_svr = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_val, q95),
            title="Predicción Prevalencia de párkinson SVR Regressor"
        )
    
        fig_europa_svr.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_svr.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson SVR Regressor</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Prevalencia de párkinson",
                tickvals=[min_val, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_val}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_svr.to_html(full_html=False))


    @output
    @render.ui
    def plot_predict_knn():
        fig_knn = px.choropleth(
            data_frame=df_predicciones_KNN,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_knn, q95_knn),
            title="Predicción Prevalencia de párkinson KNN Regressor"
        )
    
        fig_knn.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_knn.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson KNN Regressor</b>",
                'font': {'size': 20},
                'x': 0.5,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Prevalencia<br>de párkinson",
                tickvals=[min_knn, q25_knn, q50_knn, q75_knn, q95_knn],
                ticktext=[
                    f"Mín: {min_knn}",
                    f"Q25: {q25_knn}",
                    f"Q50: {q50_knn}",
                    f"Q75: {q75_knn}",
                    f"Q95: {q95_knn}"
                ]
            )
        )
    
        return ui.HTML(fig_knn.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_predict_knn():
    
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        df_europa = df_predicciones_KNN[df_predicciones_KNN["País"].isin(paises_europa)]
    
        # Calcular cuantiles
        min_val = round(df_europa["Parkinson_Predicho"].min(), 4)
        q25 = round(df_europa["Parkinson_Predicho"].quantile(0.25), 4)
        q50 = round(df_europa["Parkinson_Predicho"].quantile(0.50), 4)
        q75 = round(df_europa["Parkinson_Predicho"].quantile(0.75), 4)
        q95 = round(df_europa["Parkinson_Predicho"].quantile(0.95), 4)
    
        fig_europa_knn = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_val, q95),
            title="Predicción Prevalencia de párkinson KNN Regressor"
        )
    
        fig_europa_knn.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_knn.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson KNN Regressor</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Prevalencia de párkinson",
                tickvals=[min_val, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_val}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_knn.to_html(full_html=False))


    @output
    @render.ui
    def plot_predict_mlp():
        fig_mlp = px.choropleth(
            data_frame=df_predicciones_MLP,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_mlp, q95_mlp),
            title="Predicción Prevalencia de párkinson MLP Regressor"
        )
    
        fig_mlp.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_mlp.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson MLP Regressor</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Prevalencia<br>de párkinson",
                tickvals=[min_mlp, q25_mlp, q50_mlp, q75_mlp, q95_mlp],
                ticktext=[
                    f"Mín: {min_mlp}",
                    f"Q25: {q25_mlp}",
                    f"Q50: {q50_mlp}",
                    f"Q75: {q75_mlp}",
                    f"Q95: {q95_mlp}"
                ]
            )
        )
    
        return ui.HTML(fig_mlp.to_html(full_html=False))


    @output
    @render.ui
    def plot_europe_predict_mlp():
    
        # Lista de países de Europa
        paises_europa = [
            "Spain", "France", "Germany", "Italy", "United Kingdom", "Netherlands", 
            "Belgium", "Switzerland", "Portugal", "Sweden", "Norway", "Finland", "Denmark", 
            "Poland", "Austria", "Greece", "Hungary", "Ireland", "Czechia", "Slovakia", "Iceland",
            "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia", "Estonia", "Latvia", "Cyprus", 
            "Luxembourg", "Malta", "Lithuania", "Ukraine", "Bosnia and Herzegovina", 
            "North Macedonia", "Albania", "Montenegro", "Moldova", "Russia"
        ]
    
        df_europa = df_predicciones_MLP[df_predicciones_MLP["País"].isin(paises_europa)]
    
        # Calcular cuantiles
        min_val = round(df_europa["Parkinson_Predicho"].min(), 4)
        q25 = round(df_europa["Parkinson_Predicho"].quantile(0.25), 4)
        q50 = round(df_europa["Parkinson_Predicho"].quantile(0.50), 4)
        q75 = round(df_europa["Parkinson_Predicho"].quantile(0.75), 4)
        q95 = round(df_europa["Parkinson_Predicho"].quantile(0.95), 4)
    
        fig_europa_mlp = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_val, q95),
            title="Predicción Prevalencia de párkinson MLP Regressor"
        )
    
        fig_europa_mlp.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )
    
        fig_europa_mlp.update_layout(
            title={
                'text': "<b>Predicción Prevalencia de párkinson MLP Regressor</b>",
                'font': {'size': 20},
                'x': 0.65,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.95,
                thickness=25,
                y=0.5,
                title="Prevalencia de párkinson",
                tickvals=[min_val, q25, q50, q75, q95],
                ticktext=[
                    f"Mín: {min_val}",
                    f"Q25: {q25}",
                    f"Q50: {q50}",
                    f"Q75: {q75}",
                    f"Q95: {q95}"
                ],
            )
        )
    
        return ui.HTML(fig_europa_mlp.to_html(full_html=False))


    @output
    @render.ui
    def plot_modelos_mapa():
        fig_modelos = px.choropleth(
            df_pred_promedio,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho_Promedio",
            hover_name="País",
            hover_data={"Parkinson_Predicho_Promedio": True, "País": False},
            color_continuous_scale="Viridis",
            range_color=(min_pred, q95_pred),
            title="Predicción de prevalencia de la enfermedad de Parkinson"
        )
        fig_modelos.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_modelos.update_layout(
            title={
                'text': "<b>Predicción de prevalencia de la enfermedad de Parkinson</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            width=950,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.9,
                thickness=20,
                y=0.5,
                title="Prevalencia de párkinson",
                tickvals=[min_pred, q25_pred, q50_pred, q75_pred, q95_pred],
                ticktext=[
                    f"Mín: {min_pred}",
                    f"Q25: {q25_pred}",
                    f"Q50: {q50_pred}",
                    f"Q75: {q75_pred}",
                    f"Q95: {q95_pred}"
                ]
            )
        )
    
        return ui.HTML(fig_modelos.to_html(full_html=False))
        
    @output
    @render.ui
    def plot_modelos():
        fig_modelos_prueba = px.choropleth(
            data_frame=df_pred_desviacion,
            locations="País",
            locationmode="country names",
            color="Desviacion",
            hover_name="País",
            hover_data={"Desviacion": True, "País": False},
            color_continuous_scale="Reds",
            range_color=(min_std, q95_std),  # Limita la escala hasta Q95
            title=f"Incertidumbre del modelo de predicción"
        )
    
        fig_modelos_prueba.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_modelos_prueba.update_layout(
            title={
                'text': f"<b>Incertidumbre del modelo de predicción</b>",
                'font': {'size': 20},
                'x': 0.5,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,  # Más grande el mapa
            width=950,
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.99,
                thickness=20,
                y=0.5,
                title="Desviación estándar<br>del modelo",
                tickvals=[min_std, q25_std, q50_std, q75_std, q95_std],
                ticktext=[
                    f"Mín: {min_std}",
                    f"Q25: {q25_std}",
                    f"Q50: {q50_std}",
                    f"Q75: {q75_std}",
                    f"Q95: {q95_std}"
                ]
            )
        )
    
        return ui.HTML(fig_modelos_prueba.to_html(full_html=False))
    

    @output
    @render.ui
    def plot_vs():
        fig_modelos_prueba = px.choropleth(
        df_realesVSpredichos,
        locations="País",
        locationmode="country names",
        color="Error_Normalizado",
        color_continuous_scale=[
            [0.0, "red"],     # Sobreestimación (error negativo)
            [0.5, "white"],   # Sin error (cero)
            [1.0, "blue"]     # Subestimación (error positivo)
        ],
        range_color=[-1, 1],
        hover_name="País",
        hover_data={
            "Parkinson_Real": True,
            "Parkinson_Predicho_Promedio": True,
            "Error_Normalizado": True,
            "Error_Absoluto": True,
            "País": False
        },
        title="Mapa de Anomalías de párkinson"
    )

        fig_modelos_prueba.update_geos(
            projection_type="equirectangular",  # <- Mapa plano
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
         )
        fig_modelos_prueba.update_layout(
            title={
                'text': f"<b> Mapa de Anomalías de párkinson",
                'font': {'size': 20},
                'x': 0.4,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            width=950,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.9,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Anomalía Normalizada"
            )
        )
        return ui.HTML(fig_modelos_prueba.to_html(full_html=False))

    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    import numpy as np
    import matplotlib.colors as mcolors
    
    @output
    @render.ui
    def plot_ranking_global():
        # Mantener el orden original del DataFrame
        df_plot = df_ranking.copy()
    
        # Número de variables
        n = len(df_plot)
    
        # Crear gradiente desde azul oscuro a azul claro con menor opacidad
        base_color = np.array(mcolors.to_rgb("navy"))  # Azul oscuro
        colors = []
    
        for i in range(n):
            alpha = np.linspace(1.0, 0.2, n)[i]  # Transparencia de más opaco a más transparente
            mix = 0.3 + 0.7 * (i / (n - 1))      # Mezcla para hacerlo más claro
            color_rgb = base_color * (1 - mix) + np.array([1, 1, 1]) * mix  # Interpolación con blanco
            colors.append((*color_rgb, alpha))  # RGBA
    
        # Crear gráfico
        plt.figure(figsize=(10, max(4, n * 0.5)))
        bars = plt.barh(
            y=df_plot['Variable'],
            width=df_plot['Ranking_Promedio'],
            color=colors,
            edgecolor='black'
        )
    
        plt.xlabel("Ranking Promedio (menor = más importante)")
        plt.ylabel("")  # 🔇 Sin título del eje Y
        plt.title("Ranking Global Promedio de Variables")
        plt.tight_layout()
        plt.gca().invert_yaxis() 
    
        # Guardar en buffer PNG
        buf = io.BytesIO()
        plt.savefig(buf, format="png", transparent=True)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
    
        return ui.img(
            src=f"data:image/png;base64,{img_base64}",
            style="width: 100%; height: auto; border: 1px solid #ccc; padding: 10px;"
        )



# Crear y ejecutar la aplicación
app = App(app_ui, server)

