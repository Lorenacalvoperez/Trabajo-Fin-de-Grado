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

def entrenar_modelo_glm(df, modelo_familia, variables_independientes, variable_dependiente, test_size=0.2, scaler=False):
    df = df.copy()  # Evitar modificar el original
    
    # Aplicar transformaciones a las variables
    if 'Muertes_agua' in variables_independientes:
        df['Muertes_agua_2'] = df['Muertes_agua'] ** 2
    if 'Exp_plomo' in variables_independientes:
        df['Exp_plomo_2'] = df['Exp_plomo'] ** 2
    if 'Pesticidas' in variables_independientes:
        df['Pesticidas_log'] = np.log1p(df['Pesticidas'])
        
     # Seleccionar las variables transformadas
    nuevas_variables = [var for var in df.columns if var in variables_independientes or var.endswith('_2') or var.endswith('_log')]
   
    
    # Definir X (independientes) e y (dependiente)
    X = df[nuevas_variables]
    y = df[variable_dependiente]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Alinear índices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Escalar si es necesario
    scaler_model = None
    if scaler:
        scaler_model = StandardScaler()
        X_train_scaled = scaler_model.fit_transform(X_train)
        X_test_scaled = scaler_model.transform(X_test)

        # Reconstruir DataFrames conservando los nombres
        X_train = pd.DataFrame(X_train_scaled, columns=nuevas_variables)
        X_test = pd.DataFrame(X_test_scaled, columns=nuevas_variables)
    
    # Añadir constante
    X_train = sm.add_constant(X_train, has_constant='add')
    X_test = sm.add_constant(X_test, has_constant='add')
    
    # Entrenar el modelo GLM
    modelo = sm.GLM(y_train, X_train, family=modelo_familia).fit()
    
    # Mostrar resumen del modelo
    print(modelo.summary())
    
    # Verificación antes de predecir
    if X_test.shape[1] != len(modelo.params):
        print("⚠ ERROR: Desajuste entre columnas de X_test y los parámetros del modelo.")
        print("Columnas esperadas por el modelo:", modelo.params.index.tolist())
        print("Columnas reales en X_test:", X_test.columns.tolist())
        raise ValueError("¡Número de columnas de X_test no coincide con los parámetros del modelo!")

    # Evaluar el modelo
    y_pred = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return modelo, scaler_model, nuevas_variables



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