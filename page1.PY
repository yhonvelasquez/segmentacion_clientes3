import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from st_aggrid import AgGrid
from PIL import Image

# Configuración básica de Streamlit (debe estar al principio)
st.set_page_config(page_title="Segmentación de Clientes", page_icon=":bar_chart:", layout="wide")

# Definir credenciales
USER_CREDENTIALS = {
    "usuario1": "password1",
    "usuario2": "password2"
}

# Estado de autenticación
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Función para verificar credenciales
def login(user, password):
    if user in USER_CREDENTIALS and USER_CREDENTIALS[user] == password:
        st.session_state.logged_in = True
        st.session_state.user = user
    else:
        st.error("Usuario o contraseña incorrectos")

# Formulario de login
if not st.session_state.logged_in:
    with st.form("login_form"):
        st.write("Por favor, inicia sesión")
        user = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Iniciar Sesión")

        if submit:
            login(user, password)

else:
    # Contenido principal de la aplicación (solo visible si el usuario ha iniciado sesión)
    st.success(f"Bienvenido {st.session_state.user}!")

    # Cargar el logo en la aplicación
    logo = Image.open("logo_tienda.png")  
    st.sidebar.image(logo, use_column_width=True)

    # Título en la página de inicio
    
   

    # Barra lateral con un menú
    st.sidebar.title("Menú")
    menu = st.sidebar.radio("Selecciona una opción", ["Inicio", "Análisis de Datos", "Segmentación", "Descargar Datos"])

    # Condicional para mostrar contenido basado en el menú seleccionado
    if menu == "Inicio":
        st.title("**¡Bienvenido a la aplicación de segmentación de clientes!**")
        st.write("""
        En esta aplicación podrás cargar tus datos de clientes, aplicar el algoritmo K-Means y visualizar los resultados de segmentación.  
        ¡Comienza a explorar los datos y a crear segmentos efectivos para tu negocio!
        """)

         # Opción para cerrar sesión
    if st.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    elif menu == "Análisis de Datos":
        st.header("Análisis Exploratorio de Datos")
        st.write("Aquí puedes analizar tus datos antes de aplicar la segmentación.")
    elif menu == "Segmentación":
        st.header("Segmentación de Clientes")
        st.write("Aquí puedes aplicar el algoritmo K-Means a tus datos.")
    elif menu == "Descargar Datos":
        st.header("Descargar Datos Segmentados")
        st.write("Aquí puedes descargar los resultados de la segmentación.")

    # Slider en la página principal
    st.sidebar.header("Ajustes de Segmentación")
    slider_value = st.sidebar.slider("Selecciona el número de clusters (k)", min_value=2, max_value=10, value=3)

    # Mostrar el valor del slider en la página principal
    st.write(f"Has seleccionado {slider_value} clusters.")

    # Agregar contenido adicional según la opción seleccionada
    if menu == "Segmentación":
        # Carga de datos de ejemplo (esto puede ser reemplazado por el código que tienes para cargar tus datos)
        data = pd.read_csv("data/Mall_Customers.csv")
        st.write(data.head())

        # Preprocesamiento de datos
        if st.button("Escalar datos"):
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))  # Solo datos numéricos
            data_scaled_df = pd.DataFrame(data_scaled, columns=data.select_dtypes(include=[np.number]).columns)
            st.write("Datos escalados:", data_scaled_df.head())

        if st.button("Aplicar K-Means"):
            kmeans = KMeans(n_clusters=slider_value, random_state=42)
            clusters = kmeans.fit_predict(data_scaled_df)
            data['Cluster'] = clusters
            st.write("Datos con Clusters Asignados:", data.head())
            
            # Visualización de los clusters
            st.subheader(f"Visualización de Clusters ({slider_value} clusters)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=data_scaled_df.iloc[:, 0], y=data_scaled_df.iloc[:, 1], hue=clusters, palette="viridis", s=100)
            plt.xlabel(data_scaled_df.columns[0])
            plt.ylabel(data_scaled_df.columns[1])
            st.pyplot(fig)
    
   