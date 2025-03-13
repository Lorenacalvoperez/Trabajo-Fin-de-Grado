from shiny import App, reactive, render, ui

# Define la interfaz de usuario con CSS global
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style("""
            .sidebar {
                background-color: #007BFF !important;
                color: white !important;
                padding: 15px !important;
                height: 100vh !important;
                width: 250px !important;
            }
            .content-box {
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                margin-top: 10px;
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
        """),
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.input_action_button("home_btn", "🏠 Home", class_="btn-primary"),
                ui.a("Section 1", class_="nav-item", onclick="Shiny.setInputValue('page', 'section1')"),
                ui.a("Section 2", class_="nav-item", onclick="Shiny.setInputValue('page', 'section2')"),
                ui.a("Section 3", class_="nav-item", onclick="Shiny.setInputValue('page', 'section3')"),
                class_="sidebar"
            )
        ),
        ui.output_ui("content_display")
    )
)

# Define la lógica del servidor
def server(input, output, session):
    @output
    @render.ui
    def content_display():
        if input.home_btn():  # Si se presiona el botón Home
            return ui.div(
                ui.h1("Bienvenido a Mi App", class_="text-center"),
                ui.img(src="https://www.python.org/static/community_logos/python-logo.png", height="100px"),
                ui.p("Esta es una aplicación creada con Shiny para Python.", class_="text-center"),
                class_="content-box text-center mt-4"
            )
        
        page = input.page()
        if page == "section1":
            return ui.div("📌 Welcome to Section 1", class_="content-box")
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
                ),
                class_="content-box"
            )
        elif page == "section3":
            return ui.div("📌 You are in Section 3", class_="content-box")
        else:
            return ui.div("👉 Click on a section to navigate", class_="content-box")

# Crea y ejecuta la aplicación
app = App(app_ui, server)
