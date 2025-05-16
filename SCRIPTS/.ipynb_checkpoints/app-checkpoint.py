from shiny import App, ui, render
import pandas as pd
import requests

API_URL = "https://ghoapi.azureedge.net/api/SDGPM25"

def fetch_data():
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()["value"]
        return pd.DataFrame(data)
    return pd.DataFrame()


df = fetch_data()


app_ui = ui.page_fluid(
    ui.h2("Verificación de Datos desde la API"),
    ui.output_data_frame("data_table")  
)


def server(input, output, session):
    @output
    @render.data_frame
    def data_table():
        return df  

# Ejecutar la app
app = App(app_ui, server)