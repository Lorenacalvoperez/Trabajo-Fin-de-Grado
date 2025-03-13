from shiny import App, reactive, render, ui
import plotly.express as px
import pandas as pd

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('Parkinson.csv')

# Crear el gráfico
fig = px.choropleth(
    df,
    locations="País",                
    locationmode="country names",    
    color="Parkinson",       
    hover_name="País",               
    hover_data={
        "Parkinson": True,
    },
    animation_frame="Año",         
    color_continuous_scale="Viridis",
    title="Indicadores por país y año"
)

# Generar el HTML del gráfico de Plotly
fig_html = fig.to_html(full_html=False)

# Definición de la interfaz de usuario con CSS global
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style(""" 
            .sidebar {
                background-color: #007BFF !important;
                color: white !important;
                padding: 15px !important;
                height: 100vh !important;
                width: 250px !important;
                position: fixed; /* Fija la barra lateral */
                top: 0; /* Asegura que esté alineada al principio */
                left: 0;
                z-index: 1000; /* Asegura que esté por encima del contenido */
            }
            .content-box {
                padding: 20px;
                border: none !important;
                background-color: transparent !important;
                margin-top: 10px;
                margin-left: 270px; /* Ajusta el contenido para no superponer la barra lateral */
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
                border-radius: 0px !important; /* Hace que la barra sea rectangular */
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
            /* Nueva clase para centrar el gráfico y hacerlo más grande */
            .map-container {
                width: 90%;  /* Hacemos el gráfico más grande */
                max-width: 1200px;  /* Limita el tamaño máximo */
                margin: 0 auto;  /* Centra el gráfico */
                height: 600px;  /* Puedes ajustar la altura también */
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

# Defino la lógica del servidor
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
                    ui.h3("Parkinso Worldview: Impacto Ambiental en el Parkinson", class_="home-title"),
                    ui.p("Esta aplicación visualiza cómo las variables ambientales, como la contaminación y la temperatura, afectan la prevalencia y desarrollo de la enfermedad de Parkinson en diferentes países.",
                        class_="home-subtitle"),
                    class_="content-box"
                )
            )
        
        page = input.page()
        if page == "section1":
            # Mostrar el gráfico interactivo como HTML y centrarlo
            return ui.div(
                 # Usar el HTML generado por Plotly
                ui.HTML(fig_html), 
                class_="map-container"  
            )
        elif page == "section2":
            return ui.div(
                ui.div(
                    ui.navset_pill(
                        ui.nav_panel("Parkinson", "Panel A content"),
                        ui.nav_panel("B", "Panel B content"),
                        ui.nav_panel("C", "Panel C content"),
                        ui.nav_menu(
                            "Other links",
                            ui.nav_panel("D", "Panel D content"),
                            "----",
                            "Description:",
                            ui.nav_control(
                                ui.a("Shiny", href="https://shiny.posit.co", target="_blank")
                            ),
                        ),
                        id="tab"
                    ),
                    class_="navset-pill"
                )
            )
        elif page == "section3":
            return ui.div(
                "📌 Esta es la Sección 3, aún no tiene contenido.",
                class_="content-box"
            )
        else:
            return ui.div("👉 Click on a section to navigate")

# Creo y ejecuto la aplicación
app = App(app_ui, server)
