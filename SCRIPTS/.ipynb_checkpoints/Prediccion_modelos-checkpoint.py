# ULTIMO
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Modelo GLM
def predecir_para_pais(modelos_por_pais, archivo_para_predecir):
    # Cargar los datos para predicción
    df_para_predecir = pd.read_csv(archivo_para_predecir)

    # Crear una columna para las predicciones
    df_para_predecir['Parkinson_Predicho'] = np.nan

    # Iterar sobre los países en el archivo de predicción
    for pais in df_para_predecir["País"].unique():
        #print(f"Haciendo predicciones para {pais}...")

        # Filtrar datos por país
        df_pais = df_para_predecir[df_para_predecir["País"] == pais].copy()

        # Comprobar si el modelo para el país está disponible
        if pais in modelos_por_pais:
            modelo, scaler, columnas_utilizadas = modelos_por_pais[pais]

            # Aplicar las mismas transformaciones que durante el entrenamiento
            if 'Muertes_agua' in columnas_utilizadas:
                df_pais.loc[:, 'Muertes_agua_2'] = df_pais['Muertes_agua'] ** 2
            if 'Exp_plomo' in columnas_utilizadas:
                df_pais.loc[:, 'Exp_plomo_2'] = df_pais['Exp_plomo'] ** 2
            if 'Pepticidas' in columnas_utilizadas:
                df_pais.loc[:, 'Pesticidas_log'] = np.log1p(df_pais['Pesticidas'])

            # Crear la matriz de características X con las variables necesarias para la predicción
            nuevas_variables = [var for var in columnas_utilizadas if var != 'País']
            X_nuevo = df_pais[nuevas_variables]

            # Escalar los datos si es necesario
            if scaler:
                X_nuevo = scaler.transform(X_nuevo)

            # Agregar constante (intercepto) al nuevo conjunto de datos
            X_nuevo = sm.add_constant(X_nuevo, has_constant='add')

            # Hacer las predicciones
            df_para_predecir.loc[df_para_predecir["País"] == pais, 'Parkinson_Predicho'] = modelo.predict(X_nuevo)

        else:
            print(f"No se encuentra el modelo para el país {pais}.")
    
    # Mostrar los resultados de las predicciones
    print(df_para_predecir[['País', 'Parkinson_Predicho']])

    return df_para_predecir

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

def predecir_para_pais_SVR_KNN_MLP(modelos_svr_por_pais, archivo_para_predecir):
    df_para_predecir = pd.read_csv(archivo_para_predecir)
    df_para_predecir = df_para_predecir.copy()

    # Aplicar las transformaciones necesarias
    df_para_predecir['Contaminacion_aire_2'] = df_para_predecir['Contaminacion_aire'] ** 2
    df_para_predecir['Muertes_agua_2'] = df_para_predecir['Muertes_agua'] ** 2
    df_para_predecir['Exp_plomo_2'] = df_para_predecir['Exp_plomo'] ** 2
    df_para_predecir['Pesticidas_log'] = np.log1p(df_para_predecir['Pesticidas'])

    # Crear columna para predicción
    df_para_predecir['Parkinson_Predicho'] = np.nan

    for pais in df_para_predecir["País"].unique():
        df_pais = df_para_predecir[df_para_predecir["País"] == pais].copy()

        if pais in modelos_svr_por_pais:
            modelo, columnas_utilizadas = modelos_svr_por_pais[pais]

            X_nuevo = df_pais[columnas_utilizadas]

            # Normalizar como en el entrenamiento
            scaler = StandardScaler()
            X_nuevo_scaled = scaler.fit_transform(X_nuevo)  # ⚠ Esto solo va bien si los datos están normalizados por país

            # Predicción
            predicciones = modelo.predict(X_nuevo_scaled)
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