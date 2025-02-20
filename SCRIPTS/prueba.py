from shiny import App, ui, render
import pandas as pd
import requests

# URLs de la API
DATA_URL = "https://api.ourworldindata.org/v1/indicators/916408.data.json"
METADATA_URL = "https://api.ourworldindata.org/v1/indicators/916408.metadata.json"

# ---- FUNCIÓN PARA OBTENER DATOS ----
def fetch_data():
    # Obtener datos desde las APIs
    data_response = requests.get(DATA_URL).json()
    metadata_response = requests.get(METADATA_URL).json()

    # Extraer valores y años
    values = data_response["values"]
    years = [year["id"] for year in metadata_response["dimensions"]["years"]["values"]]

    # Extraer países
    countries = {c["id"]: c["name"] for c in metadata_response["dimensions"]["entities"]["values"]}

    # Verificar si los datos coinciden
    num_years = len(years)
    num_values = len(values)

    if num_values % num_years == 0:
        num_countries = num_values // num_years
        print(f"Se detectaron datos de {num_countries} países.")

        # Crear DataFrame con múltiples países
        data = []
        for i, (year, value) in enumerate(zip(years * num_countries, values)):
            country_id = list(countries.keys())[i // num_years]
            data.append({"Year": year, "Cases": value, "Country": countries[country_id]})

        df = pd.DataFrame(data)
    else:
        print("Los datos no coinciden exactamente. Se usará un subconjunto.")
        df = pd.DataFrame({"Year": years[:num_values], "Cases": values[:num_values]})
        df["Country"] = "World"

    return df

# Cargar datos al inicio de la app
df = fetch_data()

# ---- INTERFAZ ----
app_ui = ui.page_fluid(
    ui.h1("📊 Casos de Parkinson por Año y País"),
    ui.input_select("pais", "Selecciona un país:", sorted(df["Country"].unique())),
    ui.output_table("tabla")
)

# ---- SERVIDOR ----
def server(input, output, session):
    @output
    @render.table
    def tabla():
        return df[df["Country"] == input.pais()]

# ---- INICIAR APP ----
app = App(app_ui, server)
