from shiny import App, ui, render, reactive

# Crear la aplicación de Shiny
def server(input, output, session):
    # Reactiva para mostrar/ocultar el menú de mapas
    @reactive.Calc
    def map_dropdown_visible():
        return input.map_button() % 2 == 1  # Cambiar la visibilidad cada vez que se hace clic
    
    # Mostrar el contenido de la página seleccionada
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
        
        # Mostrar el menú de mapas solo si map_dropdown_visible es verdadero
        if map_dropdown_visible():
            return ui.div(
                ui.a(
                    "🌍 Ver Mapa Europeo",
                    class_="dropdown-item",
                    onclick="Shiny.setInputValue('page', 'europe_map')"
                ),
                ui.a(
                    "🧠 Ver Mapa Mundial de Parkinson",
                    class_="dropdown-item",
                    onclick="Shiny.setInputValue('page', 'global_parkinson_map')"
                ),
                class_="dropdown-menu show"
            )
        else:
            return ui.div()

# Interfaz de usuario
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
                margin: 30px; 
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
                margin-left: 0px;  
                width: 100%;  
            }
            .dropdown-menu {
                display: none;
                position: absolute;
                background-color: white;
                min-width: 160px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                z-index: 1;
                border-radius: 8px;
            }
            .dropdown-item {
                padding: 12px 16px;
                cursor: pointer;
            }
            .dropdown-item:hover {
                background-color: #ddd;
            }
            .show {
                display: block;
            }
        """),
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.a("🏠 Home", id="home_btn", onclick="Shiny.setInputValue('page', 'home')"),
                
                # Botón para mostrar el menú de Mapas
                ui.input_action_button("map_button", "🗺️ Mapas", class_="btn btn-secondary"),
                ui.div(
                    {"class": "dropdown-menu", "id": "map_dropdown"},
                    ui.a(
                        "🌍 Ver Mapa Europeo",
                        class_="dropdown-item",
                        onclick="Shiny.setInputValue('page', 'europe_map')"
                    ),
                    ui.a(
                        "🧠 Ver Mapa Mundial de Parkinson",
                        class_="dropdown-item",
                        onclick="Shiny.setInputValue('page', 'global_parkinson_map')"
                    ),
                ),
                ui.a("Impacto de las Variables Ambientales", class_="nav-item", onclick="Shiny.setInputValue('page', 'section2')"),
                ui.a("Análisis Gráfico y Correlaciones", class_="nav-item", onclick="Shiny.setInputValue('page', 'section3')"),
                class_="sidebar"
            )
        ),
        ui.output_ui("content_display")
    )
)

# Crear la aplicación de Shiny
app = App(app_ui, server)

