from shiny import App, reactive, render, ui
import plotly.express as px
import pandas as pd

# Cargar los archivos CSV en DataFrames
df_parkinson = pd.read_csv('Parkinson.csv').round(2)
df_contaminacion = pd.read_csv('Contaminacion_aire.csv').round(2)
df_plomo = pd.read_csv('Plomo.csv').round(2)
df_agua  = pd.read_csv("Datos_muertes_agua.csv").round(2)
df_pepticidas = pd.read_csv('Pepticidas.csv').round(2)
df_precipitaciones = pd.read_csv('Precipitaciones.csv').round(2)
df_pred_promedio = pd.read_csv("predicciones_modelos_promedio.csv").round(2)
df_pred_desviacion = pd.read_csv("predicciones_modelos_desviacion.csv").round(2)
df_pred_CV = pd.read_csv("predicciones_modelos_CV.csv").round(2)
df_realesVSpredichos = pd.read_csv("RealesVSPredichos.csv").round(2)

#Misma escala de distribicon para todos los mapas

min_parkinson = df_parkinson["Parkinson"].min()
max_parkinson = df_parkinson["Parkinson"].quantile(0.90)

min_contaminacion = df_contaminacion["Tasa_contaminacion_Aire"].min()
max_contaminacion = df_contaminacion["Tasa_contaminacion_Aire"].quantile(0.90)

min_plomo = df_plomo["Exp_Plomo"].min()
max_plomo = df_plomo["Exp_Plomo"].quantile(0.90)



min_agua = df_agua["Muertes_agua"].min()
max_agua = df_agua["Muertes_agua"].quantile(0.75)

min_pepticidas = df_pepticidas["Pesticidas"].min()
max_pepticidas = df_pepticidas["Pesticidas"].quantile(0.90)

min_precipitaciones = df_precipitaciones["Precipitación (mm)"].min()
max_precipitaciones = df_precipitaciones["Precipitación (mm)"].quantile(0.90)

min_val = df_pred_promedio["Parkinson_Predicho_Promedio"].min()
max_val = df_pred_promedio["Parkinson_Predicho_Promedio"].quantile(0.95)

min_std = df_pred_desviacion["Desviacion"].min()
max_std = df_pred_desviacion["Desviacion"].quantile(0.95)

min_cv = df_pred_CV['CV'].min()
max_cv = df_pred_CV['CV'].quantile(0.95)

# Mínimo y máximo reales (no truncados)
        
real_min = df_realesVSpredichos['Error_Absoluto'].min()
real_max = df_realesVSpredichos['Error_Absoluto'].max()


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
                ui.a("📞 Contacto", class_="sidebar-link contact-link", onclick="Shiny.setInputValue('page', 'section5')"),

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
                    ui.h3("¿Qué analizamos?", style="font-size: 24px; font-weight: bold; color: #2C3E50; margin-top: 20px;"),
                    ui.p(
                        "A través de esta aplicación, analizamos diferentes aspectos de la prevalencia de la enfermedad de Parkinson en función de los factores ambientales. "
                        "Entre los análisis realizados se incluyen la visualización geográfica de los países más afectados por la enfermedad, "
                        "y el uso de modelos predictivos entrenados con estos datos para predecir la prevalencia futura en distintas regiones del mundo.",
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
                    ui.div(
                        ui.input_action_button("go_to_europe", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'europe_map')"),
                        style="margin-top: 10px;"
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
                        onclick="Shiny.setInputValue('page', 'section1')"
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
                ui.div(
                    ui.output_ui("plot_contaminacion"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_parkinson["Año"].min(), 
                                        max=df_parkinson["Año"].max(), 
                                        value=df_parkinson["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_aire", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_aire')")
                    ),
                    ui.div(
                        ui.input_action_button("go_back", "Volver atrás", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')"),
                        style="margin-top: 10px;"
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
                        min=df_parkinson["Año"].min(),
                        max=df_parkinson["Año"].max(),
                        value=df_parkinson["Año"].min(),
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
                ui.div(
                    ui.output_ui("plot_plomo"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_parkinson["Año"].min(), 
                                        max=df_parkinson["Año"].max(), 
                                        value=df_parkinson["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_plomo", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_plomo')"),
                    ui.div(
                        ui.input_action_button("go_back", "Volver atrás", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')"),
                        style="margin-top: 10px;"
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
                        min=df_parkinson["Año"].min(),
                        max=df_parkinson["Año"].max(),
                        value=df_parkinson["Año"].min(),
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
                ui.div(
                    ui.output_ui("plot_agua"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_parkinson["Año"].min(), 
                                        max=df_parkinson["Año"].max(), 
                                        value=df_parkinson["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_agua", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_agua')"),
                    ui.div(
                        ui.input_action_button("go_back", "Volver atrás", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')"),
                        style="margin-top: 10px;"
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
                        min=df_parkinson["Año"].min(),
                        max=df_parkinson["Año"].max(),
                        value=df_parkinson["Año"].min(),
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
                ui.div(
                    ui.output_ui("plot_pepticidas"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_parkinson["Año"].min(), 
                                        max=df_parkinson["Año"].max(), 
                                        value=df_parkinson["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_pepticidas", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_pepticidas')"),
                    ui.div(
                        ui.input_action_button("go_back", "Volver atrás", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')"),
                        style="margin-top: 10px;"
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
                        min=df_parkinson["Año"].min(),
                        max=df_parkinson["Año"].max(),
                        value=df_parkinson["Año"].min(),
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
                ui.div(
                    ui.output_ui("plot_precipitaciones"),
                    ui.div(
                        ui.input_slider("year", "Selecciona el Año", 
                                        min=df_parkinson["Año"].min(), 
                                        max=df_parkinson["Año"].max(), 
                                        value=df_parkinson["Año"].min(), 
                                        step=1, 
                                        sep=""),
                        style="margin-top: 10px;"
                    ),
                    ui.div(
                        ui.input_action_button("go_to_europe_precipitaciones", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_precipitaciones')"),
                    ui.div(
                        ui.input_action_button("go_back", "Volver atrás", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')"),
                        style="margin-top: 10px;"
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
                        min=df_parkinson["Año"].min(),
                        max=df_parkinson["Año"].max(),
                        value=df_parkinson["Año"].min(),
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
                            ui.p(ui.tags.a("lorenainiciativa@gmail.com", href="mailto:tuemail@gmail.com", target="_blank", style="color: #2980B9; font-size: 18px;")),
                            style="background-color: #F2F3F4; padding: 20px; margin: 10px 0; border-radius: 8px;"
                        ),
                        
                        # GitHub
                        ui.div(
                            ui.h3("💻 GitHub", style="font-size: 20px; color: #8E44AD; text-align: center;"),
                            ui.p("Visita mi perfil de GitHub para ver otros proyectos y colaboraciones.", style="text-align: center; font-size: 16px;"),
                            ui.p(ui.tags.a("github.com/Lorenacalvoperez", href="https://github.com/Lorenacalvoperez", target="_blank", style="color: #2980B9; font-size: 18px;")),
                            style="background-color: #F2F3F4; padding: 20px; margin: 10px 0; border-radius: 8px;"
                        ),
                        
                        # Descomenta el siguiente bloque si deseas agregar teléfono
                        # ui.div(
                        #     ui.h3("📱 Teléfono", style="font-size: 20px; color: #8E44AD; text-align: center;"),
                        #     ui.p("Si prefieres contactar por teléfono, no dudes en llamarme.", style="text-align: center; font-size: 16px;"),
                        #     ui.p("+34 123 456 789", style="text-align: center; font-size: 18px; font-weight: bold; color: #2980B9;"),
                        #     style="background-color: #F2F3F4; padding: 20px; margin: 10px 0; border-radius: 8px;"
                        # ),
                        style="margin: 0 10px;"
                    ),
                    style="text-align: center; margin-top: 20px;"
                )
            )




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
                title="Parkinson"
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
                title="Parkinson"
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
            color="Tasa_contaminacion_Aire",
            hover_name="País",
            hover_data={"Tasa_contaminacion_Aire": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_contaminacion, max_contaminacion),
            labels={"Tasa_contaminacion_Aire": "Tasa de mortalidad por contaminación del aire"},
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
            color="Tasa_contaminacion_Aire",
            hover_name="País",
            hover_data={"Tasa_contaminacion_Aire": True,"País":False},
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
                title="Tasa de mortalidad por contaminación del aire"
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
            color="Exp_Plomo",
            hover_name="País",
            hover_data={"Exp_Plomo": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_plomo, max_plomo),
            labels={"Exp_Plomo": "Tasa de carga de enferemdad por exposición al plomo"},
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
            color="Exp_Plomo",
            hover_name="País",
            hover_data={"Exp_Plomo": True,"País":False},
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
                title="Tasa de carga de enferemdad por exposición al plomo"
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
            labels={"Muertes_agua": "Muertes por fuentes de agua inseguras"},
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
                title="Muertes por fuentes de agua inseguras"
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
            labels={"Pesticidas": "Uso de pesticidas (Toneladas)"},
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
                title="Uso de pesticidas (Toneladas)"
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
            color="Precipitación (mm)",
            hover_name="País",
            hover_data={"Precipitación (mm)": True,"País":False},
            color_continuous_scale="Viridis",
            range_color=(min_precipitaciones, max_precipitaciones),
            labels={"Precipitación (mm)": "Cantidad de Precipitacion (mm)"},
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
            color="Precipitación (mm)",
            hover_name="País",
            hover_data={"Precipitación (mm)": True,"País":False},
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
                title="Cantidad de Precicpitaciones (mm)"
            )
        )


    
        return ui.HTML(fig_europa_precipitaciones.to_html(full_html=False))

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
            title=f"Prevalencia del Parkinson Promedio predicho por País"
        )
        fig_modelos.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )

        fig_modelos.update_layout(
            title={
                'text': f"<b>Prevalencia de Parkinson Promedio predicho por País </b>",
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
            title=f"Prevalencia del Parkinson (Desviación Estándar) por País"
        )
        fig_modelos_prueba.update_geos(
        projection_type="equirectangular",  # <- Mapa plano
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
     )
        fig_modelos_prueba.update_layout(
            title={
                'text': f"<b>Prevalencia del Parkinson (Desviación Estándar) por País",
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
                title="Prevalencia Parkinson"
            )
        )
        return ui.HTML(fig_modelos_prueba.to_html(full_html=False))
    

    @output
    @render.ui
    def plot_vs():
        # Escala visual centrada en el 95% del valor absol
        midpoint_relative = (0 - real_min) / (real_max -real_min)
        # Escala de colores personalizada con 0 real en blanco
        colorscale = [
        [0.0, "red"],
        [midpoint_relative, "white"],
        [1.0, "blue"]
    ]
    
        fig_vs = px.choropleth(
            data_frame=df_realesVSpredichos,
            locations="País",
            locationmode="country names",
            color="Error_Absoluto",
            hover_name="País",
            hover_data={
                "Parkinson_Predicho_Promedio": True,
                "Parkinson_Real": True,
                "Error_Absoluto": True,
                "País": False
            },
            color_continuous_scale=colorscale,
            #color_continuous_midpoint=0,
            range_color=(real_min, real_max),  # Control visual
            title="Error Absoluto de Predicción de Parkinson por País"
        )
    
        fig_vs.update_geos(
            projection_type="equirectangular",
            showcoastlines=True,
            showland=True,
            fitbounds="locations"
        )
    
        fig_vs.update_layout(
            title={
                'text': "<b>Prevalencia de Parkinson: Real vs. Predicha por país",
                'font': {'size': 20},
                'x': 0.75,
                'y': 0.98,
                'xanchor': 'right'
            },
            height=400,
            margin={"r": 10, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                len=0.8,
                thickness=20,
                tickvals=[real_min, 0, real_max],  # Posiciones de los ticks
                ticktext=[f"{real_min:.2f}", "0", f"{real_max:.2f}"],  # Etiquetas visibles
                y=0.5,
                title="Prevalencia Parkinson",

            )
        )
    
        return ui.HTML(fig_vs.to_html(full_html=False))





# Crear y ejecutar la aplicación
app = App(app_ui, server)

