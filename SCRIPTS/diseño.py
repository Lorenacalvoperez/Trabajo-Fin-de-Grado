from shiny import App, reactive, render, ui
import plotly.express as px
import pandas as pd

# Cargar los archivos CSV en DataFrames
df_parkinson = pd.read_csv('Parkinson.csv')
df_contaminacion = pd.read_csv('Contaminacion_aire.csv')
df_plomo = pd.read_csv('Plomo.csv')
df_agua  = pd.read_csv("Datos_muertes_agua.csv")
df_pepticidas = pd.read_csv('Pepticidas.csv')
df_precipitaciones = pd.read_csv('Precipitaciones.csv')

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
        """),
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.a("🏠 Home", id="home_btn", onclick="Shiny.setInputValue('page', 'home')"),
                ui.a("Mapa Global del Parkinson", class_="nav-item", onclick="Shiny.setInputValue('page', 'section1')"),
                ui.a("Impacto de las Variables Ambientales", class_="nav-item", onclick="Shiny.setInputValue('page', 'section2')"),
                ui.a("Análisis Gráfico y Correlaciones", class_="nav-item", onclick="Shiny.setInputValue('page', 'section3')"),
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
                ui.navset_bar(
                    ui.nav_panel("Overview", "Información general sobre el proyecto"),
                    ui.nav_panel("Data", "Datos analizados sobre el Parkinson"),
                    title="Parkinson Worldview"
                ),
                ui.div(
                    ui.img(src="https://upload.wikimedia.org/wikipedia/commons/8/80/World_map_-_low_resolution.svg", height="300px"),
                    class_="home-container"
                ),
                ui.div(
                    ui.h3("Parkinson Worldview: Impacto Ambiental en el Parkinson", class_="home-title"),
                    ui.p("Esta aplicación visualiza cómo ciertas variables ambientales afectan la prevalencia y desarrollo de la enfermedad de Parkinson en diferentes países.",
                        class_="home-subtitle"),
                    class_="content-box"
                )
            )

        page = input.page()
        if page == "section1":
            return ui.div(
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

        elif page == "section2":
            return ui.div(
                # Barra lateral horizontal con botones
                ui.div(
                    ui.input_action_button("show_contaminacion", "Contaminación del Aire", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'contaminacion')"),
                    ui.input_action_button("show_plomo", "Exposición al Plomo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plomo')"),
                    ui.input_action_button("show_agua", "Muertes por aguas inseguras", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'agua')"),
                    ui.input_action_button("show_pesticidas", "Uso de pesticidas", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'pesticidas')"),
                    ui.input_action_button("show_precipitaciones", "Precipitaciones", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'precipitaciones')"),
                    style="display: flex; justify-content: space-around; margin-bottom: 20px;"  # Estilo para la barra horizontal
                )
            )

        elif page == "contaminacion":
            return ui.div(
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
                        ui.input_action_button("go_to_europe_aire", "🌍 Ver Mapa Europeo", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'plot_europe_aire')"),
                    ui.div(
                        ui.input_action_button("go_back", "Volver atrás", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')"),
                        style="margin-top: 10px;"
                    ),
                    class_="map-container"
                ),
                class_="content-box"
            )
            )

        elif page == "plot_europe_aire":
            return ui.div(
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
                ui.output_ui("plot_plomo"),
                ui.input_action_button("back_to_section2", "🔙 Volver a la Sección 2", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')")
            )

        elif page == "agua":
            return ui.div(
                ui.output_ui("plot_agua"),
                ui.input_action_button("back_to_section2", "🔙 Volver a la Sección 2", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')")
            )

        elif page == "pesticidas":
            return ui.div(
                ui.output_ui("plot_pepticidas"),
                ui.input_action_button("back_to_section2", "🔙 Volver a la Sección 2", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')")
            )

        elif page == "precipitaciones":
            return ui.div(
                ui.output_ui("plot_precipitaciones"),
                ui.input_action_button("back_to_section2", "🔙 Volver a la Sección 2", class_="btn btn-primary", onclick="Shiny.setInputValue('page', 'section2')")
            )

        elif page == "section3":
            return ui.div(
                "📌 Esta es la Sección 3, aún no tiene contenido.",
                class_="content-box"
            )

        else:
            return ui.div("👉 Click en una sección para navegar")
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
            hover_data={"Parkinson": True},
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
            hover_data={"Parkinson": True},
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
            hover_data={"Tasa_contaminacion_Aire": True},
            color_continuous_scale="Viridis",
            range_color=(min_contaminacion, max_contaminacion),
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
            hover_data={"Tasa_contaminacion_Aire": True},
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
                title="Contaminación_aire"
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
            hover_data={"Exp_Plomo": True},
            color_continuous_scale="Viridis",
            range_color=(min_plomo, max_plomo),
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
            hover_data={"Exp_Plomo": True},
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
                title="Exp_Plomo"
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
            hover_data={"Muertes_agua": True},
            color_continuous_scale="Viridis",
            range_color=(min_agua, max_agua),
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
    def plot_pepticidas():
        año_seleccionado = input.year()
        fig_pepticidas_filtrado = px.choropleth(
            df_pepticidas[df_pepticidas["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Pesticidas",
            hover_name="País",
            hover_data={"Pesticidas": True},
            color_continuous_scale="Viridis",
            range_color=(min_pepticidas, max_pepticidas),
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
    def plot_precipitaciones():
        año_seleccionado = input.year()
        fig_precipitaciones_filtrado = px.choropleth(
            df_precipitaciones[df_precipitaciones["Año"] == año_seleccionado],
            locations="País",
            locationmode="country names",
            color="Precipitación (mm)",
            hover_name="País",
            hover_data={"Precipitación (mm)": True},
            color_continuous_scale="Viridis",
            range_color=(min_precipitaciones, max_precipitaciones),
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


# Crear y ejecutar la aplicación
app = App(app_ui, server)

