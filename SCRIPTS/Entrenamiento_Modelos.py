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
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


# Modelo GLM 

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

def entrenar_modelo_glm(df, modelo_familia, variables_independientes, variable_dependiente, test_size=0.2,ranking=False, scaler=False):
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
    if ranking:
        resumen = modelo.summary2().tables[1].reset_index().rename(columns={'index': 'Variable'})
        resumen = resumen[['Variable', 'P>|z|']]
        resumen = resumen[resumen['Variable'] != 'Intercept']
        resumen = resumen.sort_values(by='P>|z|', ascending=True).reset_index(drop=True)

        # Convertir p-valores a formato legible sin notación científica
        resumen['P>|z|'] = resumen['P>|z|'].apply(lambda x: f'{x:.5f}')  # Limitar a 5 decimales, o el formato que necesites

        # Mostrar el ranking por pantalla
        print("\nRanking de variables por p-valor (GLM):")
        print(resumen)
        resumen.to_csv("ranking_variables_glm.csv", index=False)

        # Gráfico de p-valores
        plt.figure(figsize=(8, max(4, len(resumen) * 0.4)))
        # Asignar un hue ficticio para evitar el mensaje de advertencia
        resumen['hue'] = 'p-valor'
        sns.barplot(data=resumen, y='Variable', x='P>|z|', hue='hue', palette='Blues_r')

        plt.title("Ranking de variables por p-valor (GLM)")
        plt.xlabel("P-valor")
        plt.ylabel("Variable")
        plt.legend()
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()

    return modelo, scaler_model, transformaciones, formula


# Modelo Random Forest
def entrenar_modelo_rf(df, variables_independientes, variable_dependiente, test_size=0.2,ranking=False):
    """
    Entrena un modelo Random Forest, escala las variables, y calcula la importancia de las variables 
    tanto por feature importance como por permutación.
    
    Parámetros:
    - df: DataFrame con los datos.
    - variables_independientes: lista de nombres de las columnas que se usarán como variables independientes.
    - variable_dependiente: nombre de la columna de la variable dependiente.
    - test_size: proporción del conjunto de test.
    
    Retorna:
    - modelo_global: el modelo entrenado de Random Forest.
    - importancia_df: DataFrame con la importancia de las variables (feature importance).
    - importancia_perm: DataFrame con la importancia por permutación.
    """
    
    # 1. Separar X e y
    X = df[variables_independientes].copy()
    y = df[variable_dependiente].copy()

    # 2. Escalar las variables independientes
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=variables_independientes)

    # 3. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # 4. Entrenar el modelo Random Forest
    modelo_global = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    modelo_global.fit(X_train, y_train)

    # 5. Evaluar el modelo
    y_pred = modelo_global.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # 6. Importancia estándar (feature importance)
    importancia_df = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia': modelo_global.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nImportancia de las variables (Feature Importance):")
    print(importancia_df)

    # 7. Importancia por permutación
    resultado = permutation_importance(modelo_global, X_test, y_test, n_repeats=30, random_state=42)

    importancia_perm = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia Media': resultado.importances_mean,
        'Desviación': resultado.importances_std
    }).sort_values(by='Importancia Media', ascending=False)

    print("\nImportancia de las variables (Permutación):")
    print(importancia_perm)
    importancia_perm.to_csv("ranking_variables_rf.csv", index=False)

    if ranking:
        # 8. Gráfico de importancia
        plt.figure(figsize=(8, 4))
        plt.barh(importancia_perm['Variable'], importancia_perm['Importancia Media'], 
                 xerr=importancia_perm['Desviación'])
        plt.xlabel("Importancia (disminución en score)")
        plt.title("Importancia de variables - Permutación (RF)")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()
    
    return modelo_global,variables_independientes

# Modelo CGBoost
def entrenar_modelo_xgboost(df, variables_independientes, variable_dependiente, test_size=0.2,ranking=False):
    df = df.copy()

    # 1. Separar X e y
    X = df[variables_independientes]
    y = df[variable_dependiente]

    # 2. Escalar variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=variables_independientes)

    # 3. Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # 4. Definir el modelo
    modelo = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42
    )

    # 5. Entrenar el modelo
    modelo.fit(X_train, y_train)

    # 6. Evaluación
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # 7. Importancia estándar de XGBoost
    importancia = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia': modelo.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nImportancia de las variables (Feature Importance - XGBoost):")
    print(importancia)

    # 8. Importancia por permutación
    resultado = permutation_importance(modelo, X_test, y_test, n_repeats=30, random_state=42)

    importancia_perm = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia Media': resultado.importances_mean,
        'Desviación': resultado.importances_std
    }).sort_values(by='Importancia Media', ascending=False)

    print("\nImportancia de las variables (Permutación):")
    print(importancia_perm)

    importancia_perm.to_csv("ranking_variables_xg.csv", index=False)

    if ranking:
        # 8. Gráfico de importancia
        plt.figure(figsize=(8, 4))
        plt.barh(importancia_perm['Variable'], importancia_perm['Importancia Media'], 
                 xerr=importancia_perm['Desviación'])
        plt.xlabel("Importancia (disminución en score)")
        plt.title("Importancia de variables - Permutación (XGBOOST)")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()
    
    return modelo, variables_independientes

# Modelo SVR 
def entrenar_modelo_svr(df, variables_independientes, variable_dependiente, test_size=0.2,ranking=False):
    df = df.copy()

    # 1. Separar X e y
    X = df[variables_independientes]
    y = df[variable_dependiente]

    # 2. Escalar variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=variables_independientes)

    # 3. Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # 4. Definir el modelo con tus hiperparámetros óptimos
    modelo = SVR(C=1000, epsilon=1, gamma=1, kernel='rbf')

    # 5. Entrenar el modelo
    modelo.fit(X_train, y_train)

    # 6. Evaluación
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # 7. Importancia por permutación
    resultado = permutation_importance(modelo, X_test, y_test, n_repeats=30, random_state=42)

    importancia_perm = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia Media': resultado.importances_mean,
        'Desviación': resultado.importances_std
    }).sort_values(by='Importancia Media', ascending=False)

    print("\nImportancia de las variables (Permutación - SVR):")
    print(importancia_perm)
    importancia_perm.to_csv("ranking_variables_svr.csv", index=False)

    if ranking:
        # 8. Gráfico de importancia
        plt.figure(figsize=(8, 4))
        plt.barh(importancia_perm['Variable'], importancia_perm['Importancia Media'], 
                 xerr=importancia_perm['Desviación'])
        plt.xlabel("Importancia (disminución en score)")
        plt.title("Importancia de variables - Permutación (SVR)")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()

    return modelo, variables_independientes

# Modelo KNN
def entrenar_modelo_knn(df, variables_independientes, variable_dependiente, test_size=0.2, ranking=False):
    df = df.copy()

    # 1. Separar X e y
    X = df[variables_independientes]
    y = df[variable_dependiente]

    # 2. Escalar variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=variables_independientes)

    # 3. Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # 4. Definir y entrenar el modelo KNN
    modelo = KNeighborsRegressor(n_neighbors=11, weights='distance', metric='manhattan')
    modelo.fit(X_train, y_train)

    # 5. Evaluación
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # 6. Importancia por permutación
    resultado = permutation_importance(modelo, X_test, y_test, n_repeats=30, random_state=42)

    importancia_perm = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia Media': resultado.importances_mean,
        'Desviación': resultado.importances_std
    }).sort_values(by='Importancia Media', ascending=False)

    print("\nImportancia de las variables (Permutación - KNN):")
    print(importancia_perm)
    importancia_perm.to_csv("ranking_variables_knn.csv", index=False)

    if ranking:
        # 7. Gráfico de importancia
        plt.figure(figsize=(8, 4))
        plt.barh(importancia_perm['Variable'], importancia_perm['Importancia Media'], 
                 xerr=importancia_perm['Desviación'])
        plt.xlabel("Importancia (disminución en score)")
        plt.title("Importancia de variables - Permutación (KNN)")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()

    return modelo, variables_independientes


# Modelo MLP
def entrenar_modelo_mlp(df, variables_independientes, variable_dependiente, test_size=0.2, ranking=False):
    df = df.copy()

    # 1. Separar X e y
    X = df[variables_independientes]
    y = df[variable_dependiente]

    # 2. Escalar variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=variables_independientes)

    # 3. Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # 4. Definir y entrenar el modelo MLP
    modelo = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', max_iter=10000, alpha=0.01, random_state=42)
    modelo.fit(X_train, y_train)

    # 5. Evaluación
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # 6. Importancia por permutación
    resultado = permutation_importance(modelo, X_test, y_test, n_repeats=30, random_state=42)

    importancia_perm = pd.DataFrame({
        'Variable': variables_independientes,
        'Importancia Media': resultado.importances_mean,
        'Desviación': resultado.importances_std
    }).sort_values(by='Importancia Media', ascending=False)

    print("\nImportancia de las variables (Permutación - MLP):")
    print(importancia_perm)
    importancia_perm.to_csv("ranking_variables_mlp.csv", index=False)

    if ranking:
        # 7. Gráfico de importancia
        plt.figure(figsize=(8, 4))
        plt.barh(importancia_perm['Variable'], importancia_perm['Importancia Media'],
                 xerr=importancia_perm['Desviación'])
        plt.xlabel("Importancia (disminución en score)")
        plt.title("Importancia de variables - Permutación (MLP)")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()

    return modelo, scaler,variables_independientes
