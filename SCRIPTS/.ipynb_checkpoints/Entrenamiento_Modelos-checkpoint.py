# Imports necesarios

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelo GLM 

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def entrenar_modelo_glm(df, modelo_familia, variables_independientes, variable_dependiente, test_size=0.2, scaler=False):
    df = df.copy()

    # Construir fórmula en función de variables
    transformaciones = {
        'Contaminacion_aire': ['Contaminacion_aire', 'I(Contaminacion_aire**2)'],
        'Muertes_agua': ['Muertes_agua', 'I(Muertes_agua**2)','I(Muertes_agua**3)'],
        'Exp_plomo': ['Exp_plomo', 'I(Exp_plomo**2)', 'I(Exp_plomo**3)'],
        'Pesticidas': ['np.log1p(Pesticidas)'],
    }

    # Construir partes de fórmula
    partes_formula = []
    for var in variables_independientes:
        partes_formula.extend(transformaciones.get(var, [var]))

    formula = f"{variable_dependiente} ~ " + " + ".join(partes_formula)

    # Dividir en train/test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    # Crear matrices con patsy
    y_train, X_train = dmatrices(formula, data=df_train, return_type='dataframe')
    y_test, X_test = dmatrices(formula, data=df_test, return_type='dataframe')

        # Alinear índices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler_model = None
    if scaler:
        scaler_model = StandardScaler()
    
        # Separar columna intercepto
        intercept_train = X_train[['Intercept']]
        intercept_test = X_test[['Intercept']]
    
        # Escalar solo el resto
        X_train_scaled = scaler_model.fit_transform(X_train.drop(columns='Intercept'))
        X_test_scaled = scaler_model.transform(X_test.drop(columns='Intercept'))
    
        # Reconstruir DataFrames
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns.drop('Intercept'))
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns.drop('Intercept'))
    
        # Volver a unir la columna Intercept (sin escalar)
        X_train = pd.concat([intercept_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
        X_test = pd.concat([intercept_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

    

    # Entrenar modelo GLM
    modelo = sm.GLM(y_train, X_train, family=modelo_familia).fit()
    print(modelo.summary())

    # Evaluación
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return modelo, scaler_model, transformaciones,formula



# Modelo Random Forest
def entrenar_modelo_rf(df, variables_independientes, variable_dependiente, test_size=0.2):
    df = df.copy()

    X = df[variables_independientes]
    y = df[variable_dependiente]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    modelo = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return modelo, variables_independientes

# Modelo CGBoost
def entrenar_modelo_xgboost(df, variables_independientes, variable_dependiente, test_size=0.2):
    df = df.copy()

    # Separar las variables predictoras y la variable objetivo
    X = df[variables_independientes]
    y = df[variable_dependiente]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Crear el modelo XGBoost con los parámetros óptimos
    modelo = xgb.XGBRegressor(
        n_estimators=1000,         # Número de árboles
        learning_rate=0.05,        # Tasa de aprendizaje
        max_depth=7,               # Profundidad máxima de cada árbol
        min_child_weight=5,        # Mínimo peso de cada hoja
        subsample=0.8,             # Proporción de muestras para entrenar cada árbol
        colsample_bytree=1.0,      # Proporción de características para cada árbol
        random_state=42            # Semilla para asegurar reproducibilidad
    )
    
    # Ajustar el modelo
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Imprimir los resultados
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
    # Devolver el modelo entrenado y las variables utilizadas
    return modelo, variables_independientes

# Modelo SVR 
def entrenar_modelo_svr(df, test_size=0.2):
    df = df.copy()

    # Crear variables transformadas
    df['Pesticidas_log'] = np.log1p(df['Pesticidas'])

    # Variables originales y transformadas
    variables_originales = ['Contaminacion_aire', 'Muertes_agua', 'Exp_plomo']
    variables_transformadas = ['Pesticidas_log']
    variables_usar = variables_originales + variables_transformadas

    X = df[variables_usar]
    y = df['Parkinson']

    # División y escalado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear el modelo SVR con los mejores parámetros que ya obtuviste
    modelo = SVR(C=1000, epsilon=1, gamma=1, kernel='rbf')  # ¡Ojo! gamma=1 aquí según tus mejores resultados

    modelo.fit(X_train_scaled, y_train)

    # Evaluar el modelo
    y_pred = modelo.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Devuelve el modelo y resultados clave
    return modelo, variables_usar, X_test_scaled, y_test

# Modelo KNN
def entrenar_modelo_knn(df, test_size=0.2):
    df = df.copy()

    if 'Pesticidas' in df.columns:
        df['Pesticidas_log'] = np.log1p(df['Pesticidas'])

    # Variables originales y transformadas
    variables_originales = ['Contaminacion_aire', 'Muertes_agua', 'Exp_plomo']
    variables_transformadas = ['Pesticidas_log']

    # Crear variable que combine ambas listas
    variables_usar = variables_originales + variables_transformadas

    # Crear X e y
    X = df[variables_usar]
    y = df['Parkinson']
    
    # División y escalado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear el modelo KNN con los mejores parámetros encontrados
    knn_model = KNeighborsRegressor(n_neighbors=11, weights='distance', metric='manhattan')
    knn_model.fit(X_train_scaled, y_train)

    # Evaluar el modelo
    y_pred_knn = knn_model.predict(X_test_scaled)
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))

    print(f"MAE KNN con variables transformadas: {mae_knn:.2f}")
    print(f"RMSE KNN con variables transformadas: {rmse_knn:.2f}")

    # Devuelve también los datos de test ya escalados
    return knn_model, variables_usar, X_test_scaled, y_test

# Modelo MLP
def entrenar_modelo_mlp(df, test_size=0.2):
    df = df.copy()

    # Aplicar transformaciones si las columnas existen
    if 'Muertes_agua' in df.columns:
        df['Muertes_agua_2'] = df['Muertes_agua'] ** 2
    if 'Pesticidas' in df.columns:
        df['Pesticidas_log'] = np.log1p(df['Pesticidas'])

    # Variables originales y transformadas
    variables_originales = ['Contaminacion_aire', 'Exp_plomo', 'Pesticidas']
    variables_transformadas = [ 'Muertes_agua_2', 'Pesticidas_log']

    # ✅ Crear variable que combine ambas listas
    variables_usar = variables_originales + variables_transformadas

    # Crear X e y
    X = df[variables_usar]
    y = df['Parkinson']

    # División y escalado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear el modelo MLP
    mlp_model = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', max_iter=10000, alpha=0.01, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # Evaluar el modelo
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))

    print(f"MAE MLP con variables transformadas: {mae_mlp:.2f}")
    print(f"RMSE MLP con variables transformadas: {rmse_mlp:.2f}")

    return mlp_model, scaler, variables_usar, X_test_scaled, y_test