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

def predecir_para_pais_SVR_KNN_MLP(modelos_por_pais, archivo_para_predecir):
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

def predecir_para_pais_MLP(modelos_mlp_por_pais, archivo_para_predecir):
    # Cargar los datos a predecir
    df_para_predecir = pd.read_csv(archivo_para_predecir)
    df_para_predecir = df_para_predecir.copy()

    # Aplicar las transformaciones necesarias a las variables
    df_para_predecir['Contaminacion_aire_2'] = df_para_predecir['Contaminacion_aire'] ** 2
    df_para_predecir['Muertes_agua_2'] = df_para_predecir['Muertes_agua'] ** 2
    df_para_predecir['Exp_plomo_2'] = df_para_predecir['Exp_plomo'] ** 2
    df_para_predecir['Pesticidas_log'] = np.log1p(df_para_predecir['Pesticidas'])

    # Crear columna para las predicciones
    df_para_predecir['Parkinson_Predicho'] = np.nan

    # Recorrer los países y hacer predicciones
    for pais in df_para_predecir["País"].unique():
        df_pais = df_para_predecir[df_para_predecir["País"] == pais].copy()

        if pais in modelos_mlp_por_pais:
            modelo, scaler, columnas_utilizadas = modelos_mlp_por_pais[pais]

            if all(col in df_para_predecir.columns for col in columnas_utilizadas):
                X_nuevo = df_pais[columnas_utilizadas]
                X_nuevo_scaled = scaler.transform(X_nuevo)
                predicciones = modelo.predict(X_nuevo_scaled)

                df_para_predecir.loc[df_para_predecir["País"] == pais, 'Parkinson_Predicho'] = predicciones
            else:
                print(f"⚠ Faltan columnas necesarias para predecir {pais}.")
        else:
            print(f"⚠ Modelo no encontrado para {pais}.")

    # Mostrar las predicciones
    print(df_para_predecir[['País', 'Parkinson_Predicho']])

    return df_para_predecir