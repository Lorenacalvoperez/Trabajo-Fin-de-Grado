import shiny
from shiny import ui, render, reactive

# Crear la interfaz de usuario (UI)
app_ui = ui.page_fluid(
    ui.h2("Mi primera aplicación Shiny en Python"),
    ui.input_text("nombre", "¿Cuál es tu nombre?", ""),
    ui.output_text("saludo")
)

# Lógica del servidor
def server(input, output, session):
    @render.text
    def saludo():
        nombre = input.nombre()  # Obtener el valor del nombre
        if nombre:  # Verificar que el nombre no esté vacío
            return f"¡Hola, {nombre}!"
        else:
            return "¡Hola, por favor ingresa tu nombre!"

# Crear la aplicación Shiny
app = shiny.App(app_ui, server)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run()

