from shiny import App, reactive, render, ui
import plotly.express as px
import pandas as pd
from io import StringIO
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Cargar los archivos CSV en DataFrames
df_parkinson = pd.read_csv('Datos_Parkinson.csv').round(2)
df_contaminacion = pd.read_csv('Datos_contaminación_aire.csv').round(2)
df_plomo = pd.read_csv("Datos_exp_plomo.csv").round(2)
df_agua  = pd.read_csv("Datos_muertes_agua.csv").round(2)
df_pepticidas = pd.read_csv("Datos_uso_pepticidas.csv").round(2)
df_precipitaciones =  pd.read_csv("Datos_precipitaciones.csv").round(2)
df_pred_promedio = pd.read_csv("predicciones_modelos_promedio.csv").round(2)
df_pred_desviacion = pd.read_csv("predicciones_modelos_desviacion.csv").round(2)
df_pred_CV = pd.read_csv("predicciones_modelos_CV.csv").round(2)
df_realesVSpredichos = pd.read_csv("RealesVSPredichos.csv").round(2)
df_predicciones_GLM = pd.read_csv("pred_GLM.csv").round(2)
df_predicciones_RF = pd.read_csv("pred_RF.csv").round(2)
df_predicciones_XG = pd.read_csv("pred_XG.csv").round(2)
df_predicciones_SVR = pd.read_csv("pred_SVR.csv").round(2)
df_predicciones_KNN = pd.read_csv('pred_KNN.csv').round(2)
df_predicciones_MLP = pd.read_csv("pred_MLP.csv").round(2)
df_ranking = pd.read_csv("ranking_global_promedio.csv")

#Misma escala de distribicon para todos los mapas

min_parkinson = df_parkinson["Parkinson"].min()
max_parkinson = df_parkinson["Parkinson"].quantile(0.90)

min_contaminacion = df_contaminacion["Contaminacion_aire"].min()
max_contaminacion = df_contaminacion["Contaminacion_aire"].quantile(0.90)

min_plomo = df_plomo["Exp_plomo"].min()
max_plomo = df_plomo["Exp_plomo"].quantile(0.90)



min_agua = df_agua["Muertes_agua"].min()
max_agua = df_agua["Muertes_agua"].quantile(0.75)

min_pepticidas = df_pepticidas["Pesticidas"].min()
max_pepticidas = df_pepticidas["Pesticidas"].quantile(0.90)

min_precipitaciones = df_precipitaciones["Precipitaciones"].min()
max_precipitaciones = df_precipitaciones["Precipitaciones"].quantile(0.90)

min_val_glm = df_predicciones_GLM["Parkinson_Predicho"].min()
max_val_glm =df_predicciones_GLM["Parkinson_Predicho"].quantile(0.95)

min_val_rf = df_predicciones_RF["Parkinson_Predicho"].min()
max_val_rf =df_predicciones_RF["Parkinson_Predicho"].quantile(0.95)

min_val_xg = df_predicciones_XG["Parkinson_Predicho"].min()
max_val_xg = df_predicciones_XG["Parkinson_Predicho"].quantile(0.95)

min_val_svr = df_predicciones_SVR["Parkinson_Predicho"].min()
max_val_svr = df_predicciones_SVR["Parkinson_Predicho"].quantile(0.95)

min_val_knn = df_predicciones_KNN ["Parkinson_Predicho"].min()
max_val_knn = df_predicciones_KNN ["Parkinson_Predicho"].quantile(0.95)

min_val_mlp = df_predicciones_MLP  ["Parkinson_Predicho"].min()
max_val_mlp = df_predicciones_MLP  ["Parkinson_Predicho"].quantile(0.95)

min_val = df_pred_promedio["Parkinson_Predicho_Promedio"].min()
max_val = df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.95)

min_std = df_pred_desviacion["Desviacion"].min()
max_std = df_pred_desviacion["Desviacion"].quantile(0.95)

min_cv = df_pred_CV['CV'].min()
max_cv = df_pred_CV['CV'].quantile(0.95)

# 2. Calcular el error normalizado
df_realesVSpredichos["Error_Normalizado"] = (df_realesVSpredichos["Parkinson_Predicho_Promedio"] - df_realesVSpredichos["Parkinson_Real"]) / df_realesVSpredichos["Parkinson_Real"]
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
                ui.a("🧠 Enfermedad del Parkinson", class_="sidebar-link park-link", onclick="Shiny.setInputValue('page', 'section1')"),
                ui.a("🗺️ Mapa Mundial del Parkinson", class_="sidebar-link map-link", onclick="Shiny.setInputValue('page', 'section2')"),
                ui.a("🌿 Variables Ambientales", class_="sidebar-link env-link", onclick="Shiny.setInputValue('page', 'section3')"),
                ui.a("📈 Predicciones", class_="sidebar-link graph-link", onclick="Shiny.setInputValue('page', 'section4')"),
                ui.a("🔍 Análisis de datos", class_="sidebar-link analisis-link", onclick="Shiny.setInputValue('page', 'section5')"),
                ui.a("📞 Contacto", class_="sidebar-link contact-link", onclick="Shiny.setInputValue('page', 'section6')"),

                class_="sidebar"
            )
        ),
        ui.output_ui("content_display")
    )
    
)

# Definir la lógica del servidor
def server(input, output, session):

    @output
    @render.ui
    def content_display():
        if input.page() == "home":
            return ui.div(
                # Franja de color con el título
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
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
                        "Los datos utilizados provienen de Our World in Data (OWID), una plataforma global que recopila y presenta datos de salud pública, sociales y ambientales de todo el mundo. "
                    "La misión de OWID es hacer que los datos sean accesibles para cualquier persona, con el fin de fomentar una mayor comprensión y toma de decisiones informadas. En nuestro caso, hemos utilizado información sobre la Tasa de mortalidad por contaminación del aire, "
                    "la Tasa de carga de enfermedad por exposición al plomo, muertes atribuidas a fuentes de agua inseguras, el uso de pesticidas y precipitaciones anuales. ",
                        style="font-size: 18px; line-height: 1.6; color: #333333;"
                    ),
                    ui.a("Visita Our World in Data para más detalles", href="https://ourworldindata.org/", target="_blank", 
                     style="font-size: 18px; color: #3498db; text-decoration: none;"),
                    ui.p(
                        "Al combinar estos datos con análisis estadísticos y modelos predictivos, se puede obtener una visión más clara de cómo estos factores ambientales pueden afectar la prevalencia del Parkinson. "
                        "Además, este enfoque también ayuda a identificar posibles áreas geográficas donde el riesgo de Parkinson es más alto, lo que puede llevar a una mejor planificación de políticas públicas y estrategias de salud.",
                        style="font-size: 18px; line-height: 1.6; color: #333333;"
                    ),
                ),
            )
        

        page = input.page()
        if page == "section1":
            return ui.div(
                # Franja de título
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
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
                        "Aunque se desconoce la causa exacta del Parkinson, se sabe que resulta de la degeneración de las neuronas dopaminérgicas en "
                        "una región del cerebro llamada sustancia negra. Sin embargo, investigaciones recientes sugieren que factores ambientales pueden "
                        "jugar un papel relevante en el desarrollo de la enfermedad, especialmente en personas con cierta predisposición genética.",
                         style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
            
                    # Título de síntomas comunes
                    ui.h3("🚶‍♂️ Síntomas Comunes del Parkinson", style="color: black; text-align: center; margin-top: 20px;"),
            
                    # Descripción de los síntomas comunes
                    ui.p(
                        "Temblores: Los temblores son uno de los síntomas más reconocibles. Aparecen en reposo y afectan típicamente las manos, brazos y piernas.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Rigidez muscular: La rigidez en los músculos puede dificultar los movimientos y causar dolor.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Bradicinesia (lentitud de movimientos): La disminución de la velocidad al realizar movimientos, como caminar o escribir.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Inestabilidad postural: Los pacientes pueden tener problemas para mantener el equilibrio, lo que aumenta el riesgo de caídas.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
            
                    # Título de factores de riesgo
                    ui.h3("⚠️ Factores de Riesgo", style="color: black; text-align: center; margin-top: 20px;"),
            
                    # Descripción de los factores de riesgo
                    ui.p(
                        "Edad: La mayoría de las personas con Parkinson son mayores de 60 años.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Genética: Algunos casos tienen una predisposición genética, pero la mayoría de los casos son esporádicos (no hereditarios).",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Sexo: Los hombres tienen un mayor riesgo de desarrollar Parkinson que las mujeres.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    ui.p(
                        "Factores Ambientales: Exposición a sustancias químicas, como pesticidas, y la contaminación del aire pueden aumentar el riesgo.",
                        style="color: black; font-size: 16px; margin-bottom: 10px; text-align: left; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                )
            )


        
        
            



       
        elif page == "section2":
            return ui.div(
                # Franja de título
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
        
                # Contenido principal
                ui.div(
                    ui.output_ui("plot_parkinson"),
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
                    # Selector de años y países con múltiples selecciones
                    ui.div(
                        # Título antes de los selectores
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
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
        
                            # Columna derecha: botones
                            ui.div(
                                ui.download_button("downloadData", "Descargar CSV Filtrado"),
                                ui.download_button("downloadAll", "Descargar CSV Completo"),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
        
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )
        elif page == "europe_map":
            return ui.div(
                # Franja de título
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
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
                # Título de la aplicación
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
            
                # Instrucciones y resumen de variables
                ui.div(
                    ui.p(
                        "Explora cómo distintas variables ambientales están relacionadas con la prevalencia de la enfermedad de Parkinson en diferentes partes del mundo.",
                        style="font-size: 17px; margin: 10px 20px; color: #333;"
                    ),

                     #Factores Ambientales
                    ui.h3("🌍 Factores Ambientales que Pueden Influir", style="color: #2C3E50; text-align: left; margin-top: 30px; font-size: 24px;"),
        
                    ui.p(
                        "Diversos factores ambientales pueden influir en el riesgo de desarrollar la enfermedad de Parkinson, incluyendo:",
                        style="font-size: 16px; text-align: justify; margin: 10px 20px; color: #333;"
                    ),
                    ui.tags.ul(
                         ui.tags.li([
                        "🌫️ Contaminación del Aire: La exposición crónica a partículas finas (PM2.5) y dióxido de nitrógeno (NO₂) ha sido relacionada con un aumento en el riesgo de padecer Parkinson. ",
                        ui.tags.a("Accede aqui a los datos", href="https://ourworldindata.org/grapher/death-rates-from-air-pollution?tab=table", target="_blank", style="margin-left: 5px; color: #2980B9;")
                    ]),
                     ui.tags.li([
                        "🔩 Exposición al Plomo: La exposición prolongada a metales pesados, como el plomo, puede afectar el sistema nervioso central y se ha vinculado con el Parkinson. ",
                        ui.tags.a("Accede aqui a los datos", href="https://ourworldindata.org/grapher/rate-disease-burden-lead?tab=table", target="_blank", style="margin-left: 5px; color: #2980B9;")
                    ]),
                    ui.tags.li([
                        "🚰 Aguas Inseguras:  El consumo de agua contaminada por metales pesados o sustancias tóxicas también se ha relacionado con un posible mayor riesgo de Parkinson.  ",
                        ui.tags.a("Accede aqui a los datos", href="https://ourworldindata.org/grapher/deaths-due-to-unsafe-water-sources?tab=table", target="_blank", style="margin-left: 5px; color: #2980B9;")
                    ]),
                    ui.tags.li([
                        "🌿 Uso de Pesticidas: Sustancias como el paraquat y maneb, utilizados en la agricultura, han sido asociados con un mayor riesgo de Parkinson. ",
                        ui.tags.a("Accede aqui a los datos", href="https://ourworldindata.org/grapher/pesticide-use-tonnes?tab=table", target="_blank", style="margin-left: 5px; color: #2980B9;")
                    ]),
                    ui.tags.li([
                        "🌧️ Precipitaciones: Cambios en los patrones de lluvia pueden afectar la exposición a pesticidas o contaminantes ambientales. ",
                        ui.tags.a("Accede aqui a los datos", href="https://ourworldindata.org/grapher/average-precipitation-per-year?tab=table", target="_blank", style="margin-left: 5px; color: #2980B9;")
                    ]),
                    )
                ),
            
                # Botones de navegación
                ui.div(
                    ui.input_action_button("show_contaminacion", "🌫️Contaminación del Aire", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'contaminacion')"),
                    ui.input_action_button("show_plomo", "🔩 Exposición al Plomo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plomo')"),
                    ui.input_action_button("show_agua", "🚰 Muertes por aguas inseguras", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'agua')"),
                    ui.input_action_button("show_pesticidas", "🌿 Uso de pesticidas", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'pesticidas')"),
                    ui.input_action_button("show_precipitaciones", " 🌧️ Precipitaciones", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'precipitaciones')"),
                    style="display: flex; justify-content: space-around; margin: 30px 0 20px 0;"
                ),
        
                # Texto informativo (Tooltip o Texto destacando la acción)
                ui.div(
                    ui.p(
                        "💡 Para explorar cada variable en diferentes países, haz clic en los botones de arriba. Esto te llevará a ver los datos específicos de cada variable. ¡Explora y conoce más cada una de ellas!",
                        style="font-size: 16px; color: #555; text-align: center; margin-top: 20px; background-color: #ecf0f1; padding: 10px; border-radius: 8px;"
                    ),
                    style="margin-top: 20px;"
                )
            )


        elif page == "contaminacion":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.p(
    "Visualiza la tasa estimada de muertes atribuidas a la contaminación del aire, expresadas por cada 100.000 habitantes. "
    "Incluye contaminantes como el ozono en exteriores y puede reflejar múltiples factores de riesgo.",
    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
),
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
                    ui.div(
                        # Título antes de los selectores
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
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
        
                            # Columna derecha: botones
                            ui.div(
                                ui.download_button("downloadData_contaminacion", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_contaminacion_json", "Descargar JSON Filtrado"),
                                ui.download_button("downloadAll_contaminacion", "Descargar CSV Completo"),
                                ui.download_button("downloadAll_contaminacion_json", "Descargar JSON Completo"),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
        
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_contaminacion",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )
        elif page == "plot_europe_aire":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.p(
    "Visualiza la tasa de carga de enfermedad atribuida a la exposición al plomo. "
    "Esta métrica representa el número estimado de años de vida perdidos debido a muerte prematura "
    "o discapacidad causadas por dicha exposición, por cada 100.000 personas. "
    "Se expresa en AVAD (Años de Vida Ajustados por Discapacidad) y está ajustada por edad, "
    "lo que permite comparar países con diferentes estructuras demográficas.",
    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
),
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
                        ui.input_action_button("go_to_europe_plomo", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_plomo')"),
                    # Selector de años y países con múltiples selecciones
                    ui.div(
                        # Título antes de los selectores
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
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
        
                            # Columna derecha: botones
                            ui.div(
                                ui.download_button("downloadData_exposicion_plomo", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_exposicion_plomo_json", "Descargar JSON Filtrado"),
                                ui.download_button("downloadAll_exposicion_plomo", "Descargar CSV Completo"),
                                ui.download_button("downloadAll_exposicion_plomo_json", "Descargar JSON Completo"),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
        
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_plomo",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )
            )

        elif page == "plot_europe_plomo":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.p(
    "Visualiza el número estimado de muertes por cada 100.000 personas atribuibles a fuentes de agua insalubres. "
    "Esto incluye el consumo de agua contaminada o la falta de acceso a instalaciones seguras de saneamiento e higiene. "
    "Representa la carga de mortalidad que podría evitarse si toda la población tuviera acceso a agua potable y condiciones adecuadas de saneamiento.",
    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
),

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
                        ui.input_action_button("go_to_europe_agua", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_agua')"),
                    ui.div(
                        # Título antes de los selectores
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
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
        
                            # Columna derecha: botones
                            ui.div(
                                ui.download_button("downloadData_muertes_agua", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_muertes_agua_json", "Descargar JSON Filtrado"),
                                ui.download_button("downloadAll_muertes_agua", "Descargar CSV Completo"),
                                ui.download_button("downloadAll_muertes_agua_json", "Descargar JSON Completo"),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
        
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_agua",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )
            )

        elif page == "plot_europe_agua":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.p(
    "Visualiza el uso total de pesticidas en toneladas por país, entre 1990 y 2022. "
    "Este valor representa la cantidad total de pesticidas utilizados anualmente, incluyendo insecticidas, herbicidas y fungicidas, "
    "y refleja la intensidad del uso de productos químicos en la agricultura.",
    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
),

                ui.div(
                    ui.output_ui("plot_pepticidas"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_pepticidas["Año"].min(), 
                                        max=df_pepticidas["Año"].max(), 
                                        value=df_pepticidas["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_pepticidas", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_pepticidas')"),
                    ui.div(
                        # Título antes de los selectores
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
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
        
                            # Columna derecha: botones
                            ui.div(
                                ui.download_button("downloadData_uso_pesticidas", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_uso_pesticidas_json", "Descargar JSON Filtrado "),
                                ui.download_button("downloadAll_uso_pesticidas", "Descargar CSV Completo"),
                                ui.download_button("downloadAll_uso_pesticidas_json", "Descargar JSON "),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
        
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_pesticidas",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )
            )

        elif page == "plot_europe_pepticidas":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                ui.output_ui("plot_europe_pepticidas"),
                class_="content-box"
            )
            

        elif page == "precipitaciones":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.p(
    "Visualiza la cantidad total de precipitaciones anuales (lluvia y nieve) en cada país, medida como la profundidad del agua acumulada durante el año. "
    "Este indicador refleja el volumen total de agua que cae sobre la superficie terrestre, excluyendo fenómenos como la niebla o el rocío.",
    style="text-align: center; font-size: 16px; color: black; font-family: 'Arial', sans-serif; margin-top: 10px;"
),

                ui.div(
                    ui.output_ui("plot_precipitaciones"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_precipitaciones["Año"].min(), 
                                        max=df_precipitaciones["Año"].max(), 
                                        value=df_precipitaciones["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_precipitaciones", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_precipitaciones')"),
                    ui.div(
                        # Título antes de los selectores
                        ui.h3("📅 Filtra los datos por año y país", 
                              style="text-align: center; margin-bottom: 10px; font-family: 'Arial', sans-serif;"),
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
        
                            # Columna derecha: botones
                            ui.div(
                                ui.download_button("downloadData_precipitaciones", "Descargar CSV Filtrado"),
                                ui.download_button("downloadData_precipitaciones_json", "Descargar JSON Filtrado"),
                                ui.download_button("downloadAll_precipitaciones", "Descargar CSV Completo"),
                                ui.download_button("downloadAll_precipitaciones_json", "Descargar JSON Completo "),
                                style="flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: flex-start; margin-top: 25px;"
                            ),
        
                            style="display: flex; width: 100%;"
                        ),
        
                        style="width: 90%; margin: auto; margin-top: 20px;"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "volver_atras_precipitaciones",
                            "🔙 Volver Atrás",
                            class_="btn btn-secondary",
                            onclick="Shiny.setInputValue('page', 'section3')"
                        ),
                        style="text-align: center; margin-top: 30px;"
                    ),
        
                    class_="map-container"
                ),
        
                class_="content-box"
            )
            )

        elif page == "plot_europe_precipitaciones":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"
                    ),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
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
                    ui.h1("🌍 Parkinson Worldview", style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(
                    ui.p(
                        "En esta sección se presenta un análisis global de la importancia de las variables utilizadas en los modelos de predicción de Parkinson. "
                        "El gráfico que verás a continuación resume la influencia promedio de cada variable, calculada a partir de todos los modelos entrenados. "
                        "Esto proporciona una visión integral sobre qué factores tienen mayor peso en la predicción a nivel mundial. "
                        "Cuanto más bajo es el valor del ranking, mayor es la importancia de esa variable en los modelos. "
                        "Por ejemplo, la exposicion al plomo muestra el ranking promedio más bajo, lo que indica que es una de las variables más influyentes y consistentes en la predicción de la enfermedad. "
                        "Este tipo de visualización permite identificar patrones comunes en los modelos y orientar futuras investigaciones o estrategias de intervención.",
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
                    ui.input_action_button("show_learning_models", "Modelos de Aprendizaje Automático", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'modelos_learning')"),
                    style="display: flex; justify-content: space-around; margin: 30px 0 20px 0;"
                )
            )



        elif page == "modelo GLM":
            return ui.div(
                ui.div(
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Modelos basados en árboles", 
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                            onclick="Shiny.setInputValue('page', 'section5')"
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Modelos basados de aprendizaje automático", style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
                    style="background-color: #2C3E50; border-radius: 8px; width: 100%; margin-bottom: 20px;"
                ),
                ui.div(  # Descripción general
                    ui.p(
                        "En esta sección puedes explorar distintos modelos de aprendizaje automático aplicados a la predicción de Parkinson. "
                        "Al hacer clic en los botones que aparecen a continuación, podrás visualizar cómo se comporta cada modelo (SVR, KNN y MLP) y comparar sus resultados. "
                        "Esta exploración te permitirá identificar patrones de rendimiento y entender mejor cómo cada algoritmo procesa la información.",
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
                          style="margin: 0; padding: 10px; color: white; text-align: center; font-size: 40px; font-family: 'Arial', sans-serif;"),
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
                    ui.h1("🌍 Parkinson Worldview",
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
                            ui.p(ui.tags.a("github.com/Lorenacalvoperez", href="https://github.com/Lorenacalvoperez", target="_blank", style="color: #2980B9; font-size: 18px;")),
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
    def downloadAll_contaminacion():
        buffer = io.StringIO()
        df_parkinson.to_csv(buffer, index=False)
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
    @render.download(filename="Tasa_contaminacion_aire_completo.json")
    def downloadAll_contaminacion_json():
        json_str = df_parkinson.to_json(orient="records", indent=2)
    
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
    @render.download(filename="Exposicion_plomo_completo.csv")
    def downloadAll_exposicion_plomo():
        buffer = io.StringIO()
        df_parkinson.to_csv(buffer, index=False)
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
    @render.download(filename="Muertes_agua_completo.csv")
    def downloadAll_muertes_agua():
        buffer = io.StringIO()
        df_parkinson.to_csv(buffer, index=False)
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
    @render.download(filename="Uso_pesticidas_completo.csv")
    def downloadAll_uso_pesticidas():
        buffer = io.StringIO()
        df_parkinson.to_csv(buffer, index=False)
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
        df_parkinson.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @output
    @render.download(filename="Precipitaciones_filtrado.csv")
    def downloadData_precipitaciones():
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
        fig_parkinson_filtrado = px.choropleth(
            df_parkinson[df_parkinson["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Parkinson",
            hover_name="País",
            hover_data={"Parkinson": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_parkinson, max_parkinson),
            title=f"Prevalencia del Parkinson por País y Año - {año_seleccionado}"
        )
        fig_parkinson_filtrado.update_geos(
            projection_type="equirectangular",  # Mapa plano
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
        fig_parkinson_filtrado.update_layout(
            title={
                'text': f"<b>Prevalencia del Parkinson por País y Año - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Numero estimida de<br>casos de Parkinson"
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
    
        df_europa = df_parkinson[df_parkinson["País"].isin(paises_europa)]
        df_europa = df_europa[df_europa["Año"] == año_seleccionado]
    
        fig_europa = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson",
            hover_name="País",
            hover_data={"Parkinson": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_parkinson, max_parkinson),
            title=f"Prevalencia del Parkinson en Europa por País y Año - {año_seleccionado}"
        )
    
        fig_europa.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa.update_layout(
            title={
                'text': f"<b>Prevalencia del Parkinson en Europa por País y Año - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Numero estimida de<br>casos de Parkinson"
            )
        )
    
        return ui.HTML(fig_europa.to_html(full_html=False))


    @output
    @render.ui
    def plot_contaminacion():
        año_seleccionado = input.year()
        fig_contaminacion_filtrado = px.choropleth(
            df_contaminacion[df_contaminacion["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Contaminacion_aire",
            hover_name="País",
            hover_data={"Contaminacion_aire": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_contaminacion, max_contaminacion),
            labels={"Contaminacion_aire": "Tasa de mortalidad por<br>contaminación del aire"},
            title=f"Contaminación del Aire - {año_seleccionado}"
        )

        fig_contaminacion_filtrado.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )
        

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

        df_europa = df_contaminacion[df_contaminacion["País"].isin(paises_europa)]
        df_europa = df_europa[df_europa["Año"] == año_seleccionado]
    
    
        fig_europa_Aire = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Contaminacion_aire",
            hover_name="País",
            hover_data={"Contaminacion_aire": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_contaminacion, max_contaminacion),
            title=f"Contaminación del Aire en Europa - {año_seleccionado}"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
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
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Tasa de mortalidad por<br>contaminación del aire"
            )
        )


    
        return ui.HTML(fig_europa_Aire.to_html(full_html=False))


    @output
    @render.ui
    def plot_plomo():
        año_seleccionado = input.year()
        fig_plomo_filtrado = px.choropleth(
            df_plomo[df_plomo["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Exp_plomo",
            hover_name="País",
            hover_data={"Exp_plomo": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_plomo, max_plomo),
            labels={"Exp_plomo": "Impacto en la salud <br>por exposición al plomo"},
            title=f"Exposición al Plomo - {año_seleccionado}"
        )

        fig_plomo_filtrado.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_plomo_filtrado.update_layout(
        height=400,  # Hacerlo más grande
        margin={"r":0,"t":50,"l":0,"b":0}
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

        df_europa = df_plomo[df_plomo["País"].isin(paises_europa)]
        df_europa = df_europa[df_europa["Año"] == año_seleccionado]
    
    
        fig_europa_plomo = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Exp_plomo",
            hover_name="País",
            hover_data={"Exp_plomo": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_plomo, max_plomo),
            title=f"Exposición al Plomo - {año_seleccionado}"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
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
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Impacto en la salud <br>por exposición al plomo"
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
            hover_data={"Muertes_agua": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_agua, max_agua),
            labels={"Muertes_agua": "Muertes por fuentes<br>de agua inseguras"},
            title=f"Muertes de agua - {año_seleccionado}"
        )

        fig_agua_filtrado.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_agua_filtrado.update_layout(
        height=400,  # Hacerlo más grande
        margin={"r":0,"t":50,"l":0,"b":0}
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

        df_europa = df_agua[df_agua["País"].isin(paises_europa)]
        df_europa = df_europa[df_europa["Año"] == año_seleccionado]
    
    
        fig_europa_agua = px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Muertes_agua",
            hover_name="País",
            hover_data={"Muertes_agua": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_agua, max_agua),
            title=f"Muertes de agua - {año_seleccionado}"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
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
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Muertes por fuentes<br>de agua inseguras"
            )
        )


    
        return ui.HTML(fig_europa_agua.to_html(full_html=False))
    
    
    @output
    @render.ui
    def plot_pepticidas():
        año_seleccionado = input.year()
        fig_pepticidas_filtrado = px.choropleth(
            df_pepticidas[df_pepticidas["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Pesticidas",
            hover_name="País",
            hover_data={"Pesticidas": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_pepticidas, max_pepticidas),
            labels={"Pesticidas": "Uso de pesticidas<br>(Toneladas)"},
            title=f"Uso de pepticidas - {año_seleccionado}"
        )

        fig_pepticidas_filtrado.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_pepticidas_filtrado.update_layout(
        height=400,  # Hacerlo más grande
        margin={"r":0,"t":50,"l":0,"b":0}
    )
        return ui.HTML(fig_pepticidas_filtrado.to_html(full_html=False))

    @output
    @render.ui
    def plot_europe_pepticidas():
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

        df_europa = df_pepticidas[df_pepticidas["País"].isin(paises_europa)]
        df_europa = df_europa[df_europa["Año"] == año_seleccionado]
    
    
        fig_europa_pepticidas= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Pesticidas",
            hover_name="País",
            hover_data={"Pesticidas": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_pepticidas, max_pepticidas),
            title=f"Exposición al Plomo - {año_seleccionado}"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_pepticidas.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_pepticidas.update_layout(
            title={
                'text': f"<b>Uso de pepticidas - {año_seleccionado}</b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Uso de pesticidas<br>(Toneladas)"
            )
        )


    
        return ui.HTML(fig_europa_pepticidas.to_html(full_html=False))
    
    @output
    @render.ui
    def plot_precipitaciones():
        año_seleccionado = input.year()
        fig_precipitaciones_filtrado = px.choropleth(
            df_precipitaciones[df_precipitaciones["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Precipitaciones",
            hover_name="País",
            hover_data={"Precipitaciones": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_precipitaciones, max_precipitaciones),
            labels={"Precipitaciones": "Cantidad de<br>Precipitacion (mm)"},
            title=f"Precipitaciones - {año_seleccionado}"
        )
        fig_precipitaciones_filtrado.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )
        fig_precipitaciones_filtrado.update_layout(
        height=400,  # Hacerlo más grande
        margin={"r":0,"t":50,"l":0,"b":0}
    )
        return ui.HTML(fig_precipitaciones_filtrado.to_html(full_html=False))

    @output
    @render.ui
    def plot_europe_precipitaciones():
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

        df_europa = df_precipitaciones[df_precipitaciones["País"].isin(paises_europa)]
        df_europa = df_europa[df_europa["Año"] == año_seleccionado]
    
    
        fig_europa_precipitaciones= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Precipitaciones",
            hover_name="País",
            hover_data={"Precipitaciones": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_precipitaciones, max_precipitaciones),
            title=f"Precipitaciones - {año_seleccionado}"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
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
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Cantidad de<br>Precipitacion (mm)"
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
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_glm, max_val_glm),
            title=f"Predicción Prevalencia del Parkinson GLM"
        )
        fig_glm.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_glm.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson GLM </b>",
                'font': {'size': 20},
                'x': 0.78,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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

        df_europa = df_predicciones_GLM[df_predicciones_GLM["País"].isin(paises_europa)]
    
    
        fig_europa_glm= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_glm, max_val_glm),
            title=f"Predicción Prevalencia del Parkinson GLM"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_glm.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_glm.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson GLM </b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Prevalencia Parkinson"
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
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_rf, max_val_rf),
            title=f"Predicción Prevalencia del Parkinson RF"
        )
        fig_rf.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_rf.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson RF </b>",
                'font': {'size': 20},
                'x': 0.78,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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

        df_europa = df_predicciones_RF[df_predicciones_RF["País"].isin(paises_europa)]
    
    
        fig_europa_rf= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_rf, max_val_rf),
            title=f"Predicción Prevalencia del Parkinson RF"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_rf.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_rf.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson RF </b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Prevalencia Parkinson"
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
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_xg, max_val_xg),
            title=f"Predicción Prevalencia del Parkinson XGBoost Regressor"
        )
        fig_xg.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_xg.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson XGBoost Regressor </b>",
                'font': {'size': 20},
                'x': 0.78,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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

        df_europa = df_predicciones_XG[df_predicciones_XG["País"].isin(paises_europa)]
    
    
        fig_europa_xg= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_xg, max_val_xg),
            title=f"Predicción Prevalencia del Parkinson XGBoost Regressor"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_xg.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_xg.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson XGBoost Regressor </b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Prevalencia Parkinson"
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
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_svr, max_val_svr),
            title=f"Predicción Prevalencia del Parkinson SVR Regressor"
        )
        fig_svr.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_svr.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson SVR Regressor </b>",
                'font': {'size': 20},
                'x': 0.78,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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
    
    
        fig_europa_svr= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_svr, max_val_svr),
            title=f"Predicción Prevalencia del Parkinson SVR Regressor"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_svr.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_svr.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson SVR Regressor </b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Prevalencia Parkinson"
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
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_svr, max_val_svr),
            title=f"Predicción Prevalencia del Parkinson KNN Regressor"
        )
        fig_knn.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_knn.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson KNN Regressor </b>",
                'font': {'size': 20},
                'x': 0.78,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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
    
    
        fig_europa_knn= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_knn, max_val_knn),
            title=f"Predicción Prevalencia del Parkinson SVR Regressor"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_knn.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_knn.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson KNN Regressor </b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Prevalencia Parkinson"
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
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_mlp, max_val_mlp),
            title=f"Predicción Prevalencia del Parkinson MLP Regressor"
        )
        fig_mlp.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_mlp.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson MLP Regressor </b>",
                'font': {'size': 20},
                'x': 0.78,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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
    
    
        fig_europa_mlp= px.choropleth(
            df_europa,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho",
            hover_name="País",
            hover_data={"Parkinson_Predicho": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val_mlp, max_val_mlp),
            title=f"Predicción Prevalencia del Parkinson MLP Regressor"
        )
    
        # Usamos 'scope=europe' para centrar solo en Europa
        fig_europa_mlp.update_geos(
            projection_type="equirectangular",
            scope="europe",
            showland=True,
            landcolor="white",
            countrycolor="black"
        )

        fig_europa_mlp.update_layout(
            title={
                'text': f"<b>Predicción Prevalencia del Parkinson MLP Regressor </b>",
                'font': {'size': 20},
                'x': 0.7,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=15,
                y=0.5,
                title="Prevalencia Parkinson"
            )
        )


    
        return ui.HTML(fig_europa_mlp.to_html(full_html=False))

    @output
    @render.ui
    def plot_modelos_mapa():
        fig_modelos = px.choropleth(
            data_frame=df_pred_promedio,
            locations="País",
            locationmode="country names",
            color="Parkinson_Predicho_Promedio",
            hover_name="País",
            hover_data={"Parkinson_Predicho_Promedio": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_val, max_val),
            title=f"Predicción de prevalencia del Parkinson"
        )
        fig_modelos.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_modelos.update_layout(
            title={
                'text': f"<b>Predicción de prevalencia del Parkinson </b>",
                'font': {'size': 20},
                'x': 0.6,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Prevalencia Parkinson"
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
            hover_data={"Desviacion": True,"País":False},
            color_continuous_scale="Reds",
            range_color=(min_std, max_std),
            title=f"Incertidumbre del modelo de predicción"
        )
        fig_modelos_prueba.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )
        fig_modelos_prueba.update_layout(
            title={
                'text': f"<b>Incertidumbre del modelo de predicción",
                'font': {'size': 20},
                'x': 0.6,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Desviación estándar del modelo"
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
        title="Error Normalizado de Parkinson y Anomalías por País"
    )

        fig_modelos_prueba.update_geos(
            projection_type="equirectangular",  # <- Mapa plano
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
         )
        fig_modelos_prueba.update_layout(
            title={
                'text': f"<b> Error Normalizado de Parkinson: Sobre/Subestimación",
                'font': {'size': 20},
                'x': 0.82,
                'y' : 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,  # 🔽 Altura visual de la barra de colores (0.3 es más pequeña)
                thickness=20,
                y=0.5,
                title="Error_Normalizado"
            )
        )
        return ui.HTML(fig_modelos_prueba.to_html(full_html=False))

    @output
    @render.ui
    def plot_ranking_global():
        # Crear el gráfico
        plt.figure(figsize=(10, max(4, len(df_ranking) * 0.5)))
        sns.barplot(
            data=df_ranking,
            y='Variable',
            x='Ranking_Promedio',
            hue='Variable',
            palette='viridis',
            dodge=False,
            order=df_ranking.sort_values('Ranking_Promedio')['Variable']
        )
        plt.xlabel("Ranking Promedio (menor = más importante)")
        plt.ylabel("Variable")
        plt.title("Ranking Global Promedio de Variables")
        plt.tight_layout()
        plt.gca().invert_yaxis()
    
        # Guardar en un buffer en memoria como imagen PNG
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
    
        # Mostrar como imagen en la app
        return ui.img(src=f"data:image/png;base64,{img_base64}", style="width: 100%; height: auto; border: 1px solid #ccc; padding: 10px;")


# Crear y ejecutar la aplicación
app = App(app_ui, server)

