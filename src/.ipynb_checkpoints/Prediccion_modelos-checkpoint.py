# ULTIMO
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Modelo GLM
from patsy import dmatrix

def predecir_para_pais(modelos_por_pais, archivo_para_predecir):
    '''
    Realiza predicciones de enfermedad de Parkinson para cada país
    utilizando modelos previamente entrenados por país.

    Parámetros
    ----------
    modelos_por_pais : dict
        Diccionario donde cada clave es el nombre de un país y su valor es una tupla con:
        (modelo_glm, scaler_model, transformaciones, formula).
    archivo_para_predecir : str
        Ruta al archivo CSV que contiene los datos con una columna "País" y las variables necesarias
        para realizar la predicción.
        

    Retorno
    -------
    pandas.DataFrame
        DataFrame con dos columnas: "País" y "Parkinson_Predicho", donde cada fila contiene
        la predicción para ese país.
    '''
    
    df_pred = pd.read_csv(archivo_para_predecir)
    resultados = []

    for pais in df_pred["País"].unique():
        if pais not in modelos_por_pais:
            print(f"Modelo no disponible para {pais}.")
            continue

    
        # Extraer modelo y parámetros
        modelo, scaler, transformaciones, formula = modelos_por_pais[pais]

        # Datos del país
        df_pais = df_pred[df_pred["País"] == pais].copy()
        df_pais = df_pais.drop(columns=["País"])

        # Extraer parte derecha de la fórmula
        formula_rhs = formula.split("~")[1].strip()

        # Construir X con las transformaciones
        try:
            X_nuevo = dmatrix(formula_rhs, data=df_pais, return_type='dataframe')
        except Exception as e:
            print(f"Error procesando fórmula para {pais}: {e}")
            continue

        # Escalar si fue necesario
        if scaler:
            intercept = X_nuevo[['Intercept']]
            X_scaled = scaler.transform(X_nuevo.drop(columns='Intercept'))
            X_nuevo = pd.concat([
                intercept.reset_index(drop=True),
                pd.DataFrame(X_scaled, columns=X_nuevo.columns.drop('Intercept'))
            ], axis=1)

        # Predecir
        pred = modelo.predict(X_nuevo)

        
        # Calcular promedio (o suma, si prefieres)
        #promedio_pred = pred.mean()
        resultados.append({"País": pais, "Parkinson_Predicho": pred.values[0]})

    return pd.DataFrame(resultados)



def predecir_para_pais_RF_XG(modelos_por_pais, archivo_para_predecir):
    '''
    Realiza predicciones de enfermedad de Parkinson por país utilizando modelos tipo 
    Random Forest (RF) o XGBoost (XG) previamente entrenados.

    Parámetros
    ----------
    modelos_por_pais : dict
        Diccionario donde cada clave es el nombre de un país, y el valor es una tupla con:
        (modelo_entrenado, columnas_utilizadas)

    archivo_para_predecir : str
        Ruta al archivo CSV que contiene los datos de entrada, incluyendo una columna "País" 
        y las variables necesarias para la predicción.

    Retorna
    -------
    pandas.DataFrame
        DataFrame original con una columna adicional: "Parkinson_Predicho", que contiene 
        las predicciones generadas por el modelo correspondiente a cada país.

    '''
    df_para_predecir = pd.read_csv(archivo_para_predecir)
    df_para_predecir['Parkinson_Predicho'] = np.nan
 
    for pais in df_para_predecir["País"].unique():
        df_pais = df_para_predecir[df_para_predecir["País"] == pais].copy()

        if pais in modelos_por_pais:
            modelo, columnas_utilizadas = modelos_por_pais[pais]
            X_nuevo = df_pais[columnas_utilizadas]

            df_para_predecir.loc[df_para_predecir["País"] == pais, 'Parkinson_Predicho'] = modelo.predict(X_nuevo)

        else:
            print(f"⚠ Modelo no encontrado para {pais}.")

    print(df_para_predecir[['País', 'Parkinson_Predicho']])
    return df_para_predecir

import pandas as pd
import numpy as np

def predecir_para_pais_SVR_KNN(modelos_por_pais, archivo_para_predecir):
    '''
     """
    Realiza predicciones de enfermedad de Parkinson por país utilizando modelos previamente
    entrenados tipo SVR, KNN 

    Parámetros
    ----------
    modelos_por_pais : dict
        Diccionario donde cada clave es el nombre de un país y el valor es una tupla:
        (modelo_entrenado, columnas_utilizadas).

    archivo_para_predecir : str
        Ruta al archivo CSV con los datos de entrada, que debe incluir una columna "País"
        y las variables requeridas por los modelos.

    Retorna
    -------
    pandas.DataFrame
        DataFrame original con una nueva columna 'Parkinson_Predicho' que contiene las 
        predicciones realizadas para cada país (si el modelo está disponible).

    '''
    df_para_predecir = pd.read_csv(archivo_para_predecir).copy()
    df_para_predecir['Parkinson_Predicho'] = np.nan

    for pais in df_para_predecir["País"].unique():
        df_pais = df_para_predecir[df_para_predecir["País"] == pais].copy()

        if pais in modelos_por_pais:
            modelo, columnas_utilizadas = modelos_por_pais[pais]

            X_nuevo = df_pais[columnas_utilizadas]

            predicciones = modelo.predict(X_nuevo)
            df_para_predecir.loc[df_para_predecir["País"] == pais, 'Parkinson_Predicho'] = predicciones
        else:
            print(f"⚠ Modelo no encontrado para {pais}.")

    print(df_para_predecir[['País', 'Parkinson_Predicho']])
    return df_para_predecir

import pandas as pd
import numpy as np

def predecir_para_pais_MLP(modelos_por_pais, archivo_para_predecir):
    '''
    Realiza predicciones de enfermedad de Parkinson por país utilizando modelos tipo 
    MLP previamente entrenados.

    Parámetros
    ----------
    modelos_por_pais : dict
        Diccionario que contiene como clave el nombre del país y como valor una tupla:
        (modelo_entrenado, scaler, columnas_utilizadas).

    archivo_para_predecir : str
        Ruta al archivo CSV con los datos de entrada, incluyendo una columna "País" 
        y las variables requeridas para la predicción.

    Retorna
    -------
    pandas.DataFrame
        DataFrame original con una columna adicional 'Parkinson_Predicho', que contiene
        las predicciones generadas para cada fila, si existe un modelo para su país correspondiente.
    '''
    df_para_predecir = pd.read_csv(archivo_para_predecir).copy()
    df_para_predecir['Parkinson_Predicho'] = np.nan

    for pais in df_para_predecir["País"].unique():
        df_pais = df_para_predecir[df_para_predecir["País"] == pais].copy()

        if pais in modelos_por_pais:
            modelo, scaler, columnas_utilizadas = modelos_por_pais[pais]

            X_nuevo = df_pais[columnas_utilizadas]
            X_nuevo_scaled = scaler.transform(X_nuevo)

            # ✅ Convertimos el array escalado en DataFrame para mantener los nombres de columnas
            X_nuevo_scaled_df = pd.DataFrame(X_nuevo_scaled, columns=columnas_utilizadas)

            predicciones = modelo.predict(X_nuevo_scaled_df)
            df_para_predecir.loc[df_para_predecir["País"] == pais, 'Parkinson_Predicho'] = predicciones
        else:
            print(f"⚠ Modelo no encontrado para {pais}.")

    print(df_para_predecir[['País', 'Parkinson_Predicho']])
    return df_para_predecir
