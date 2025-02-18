import shiny
import requests

# Función para obtener los datos de la API
def obtener_datos_api():
    url = "https://ghoapi.azureedge.net/api/Indicator"  # URL para obtener la lista de indicadores
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()  # Devolvemos los datos en formato JSON
    else:
        return {"error": "No se pudo obtener los datos", "status_code": response.status_code}

# Definición de la interfaz de usuario de la app (sin elementos visuales para los datos)
app_ui = shiny.ui.page_fluid()

# Lógica del servidor
def server(input, output, session):
    # Al iniciar la app, se obtiene la información de la API, pero no se muestra en la interfaz
    obtener_datos_api()  # Realizamos la solicitud pero no hacemos nada con los datos obtenidos

# Ejecutar la app
app = shiny.App(app_ui, server)
app.run()