# Importación de aplicaciones
import app0
import app1
import app2
import app3
import app4
import streamlit as st

emoticono_calendario = "https://cdn2.iconfinder.com/data/icons/calendar-rounded/512/xxx033-512.png"

html_image = """
<center><img src="https://cdn2.iconfinder.com/data/icons/resort-villa-hotel-tourist-worker-and-services/410/resort-holiday-hotel-001-512.png" style="width:250px;height:250px;"></center>
"""
html_image1 = """
<center><img src="https://cdn2.iconfinder.com/data/icons/calendar-rounded/512/xxx033-512.png" style="width:160px;height:160px;"></center>
"""

# Configuración de la página (icono, usar toda la página, nombre)
st.set_page_config(
    layout = "wide", 
    page_icon = emoticono_calendario, 
    page_title = "App - Cancelaciones de reservas"
    )

# Páginas de la app
PAGES = {
    "Introducción a la aplicación.": app0,
    "Predicción de cancelación de reserva individual.": app1,
    "Predicción de cancelación de reservas múltiples.": app2,
    "Análisis de sentimiento para una review.": app3,
    "Análisis de sentimiento para múltiples reviews.": app4
}

# Texto y páginas en la sidebar
st.sidebar.markdown(html_image, unsafe_allow_html=True)  
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.markdown("<h1 color: white;'> Navegador </h1>", unsafe_allow_html=True)
selection = st.sidebar.radio("Ir a:", list(PAGES.keys()))
page = PAGES[selection]
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.markdown(html_image1, unsafe_allow_html=True)  
page.app()
