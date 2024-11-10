import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from st_aggrid import AgGrid

# Función de autenticación
def check_login(username, password):
    valid_users = {
        "usuario1": "password1", 
        "usuario2": "password2"
    }
    return valid_users.get(username) == password

# Función para cargar los datos
@st.cache_data
def load_data(file_path="data/Mall_Customers.csv"):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Archivo no encontrado. Asegúrate de que el archivo CSV esté en la ruta correcta.")
        return None

# Función para cargar un archivo CSV desde el cargador de archivos
def load_uploaded_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Hubo un error al cargar el archivo: {e}")
        return None

# Función de cerrar sesión
def logout():
    st.session_state["logged_in"] = False
    st.session_state["login_successful"] = False
    st.experimental_rerun()

# Página de login
def login_page():
    st.title("Iniciar sesión")
    st.write("Por favor, ingresa tus credenciales para acceder a la aplicación.")
    
    # Campos de usuario y contraseña
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    
    # Botón de login
    if st.button("Iniciar sesión"):
        if check_login(username, password):
            st.session_state["logged_in"] = True
            st.success("¡Acceso concedido!")
            st.session_state["login_successful"] = True  
        else:
            st.error("Credenciales incorrectas. Intenta nuevamente.")

# Página principal de la aplicación
def main_page():
    st.title("Segmentación de Clientes")
    st.write("Bienvenido a la aplicación de segmentación de clientes.")

# Página de Análisis de Clientes
def analysis_page(data):
    st.header("Análisis Exploratorio de Datos")
    
    if st.checkbox("Mostrar estadísticas descriptivas"):
        st.write(data.describe())
    
    # Gráfico de distribución de Edad
    st.subheader("Distribución de Edad")
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(data['Age'], kde=True, color='blue')
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    st.pyplot(fig)
    
    # Gráfico de distribución de Ingresos Anuales
    st.subheader("Distribución de Ingresos Anuales")
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(data['Annual Income (k$)'], kde=True, color='green')
    plt.xlabel("Ingreso Anual (k$)")
    plt.ylabel("Frecuencia")
    st.pyplot(fig)
    
    # Correlación entre variables numéricas
    if st.checkbox("Mostrar correlación entre variables"):
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            plt.title("Correlación entre Variables")
            st.pyplot(fig)
        else:
            st.write("No hay columnas numéricas para calcular la correlación.")

# Página de Segmentación y Reporte
def segmentation_page(data):
    st.header("Segmentación de Clientes y Reporte")
    
    # Escalar datos
    if "data_scaled_df" not in st.session_state:
        st.session_state["data_scaled_df"] = None

    if st.button("Escalar datos"):
        scaler = StandardScaler()
        numeric_data = data.select_dtypes(include=[np.number])
        data_scaled = scaler.fit_transform(numeric_data)
        st.session_state["data_scaled_df"] = pd.DataFrame(data_scaled, columns=numeric_data.columns)
        st.write("Datos escalados:", st.session_state["data_scaled_df"].head())

    # Selección de clusters
    num_clusters = st.slider("Seleccione el número de clusters (k)", min_value=2, max_value=10, value=5)
    
    # Aplicar K-Means y visualizar
    if st.button("Aplicar K-Means") and st.session_state["data_scaled_df"] is not None:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(st.session_state["data_scaled_df"])
        data['Cluster'] = clusters
        st.write("Datos con Clusters Asignados:", data.head())
        
        # Visualización interactiva de los clusters
        fig = px.scatter(st.session_state["data_scaled_df"], x=st.session_state["data_scaled_df"].columns[2],
                         y=st.session_state["data_scaled_df"].columns[3], color=data['Cluster'].astype(str),
                         title=f'Clusters ({num_clusters} clusters) | Ingreso Anual Vs. Puntaje de gasto', labels={'color': 'Cluster'})
        st.plotly_chart(fig)

        st.write("Eje X: Ingreso anual del cliente. Eje Y: Puntaje de gasto del cliente.")
        st.markdown("---")

        # Descargar datos segmentados
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Descargar CSV con Segmentación", data=csv,
                           file_name="clientes_segmentados.csv", mime="text/csv")
    else:
        st.write("Escala los datos y aplica K-Means antes de descargar.")

# Cargar los datos
data = load_data()

# Verifica si el usuario está logueado
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_page()  # Muestra la página de login si no está logueado
else:
    # Agregar logo en el sidebar y reducir su tamaño
    st.sidebar.image("logo_tienda.png")  # Tamaño reducido del logo
    st.sidebar.markdown("""
        <style>
            .sidebar .radio-label {
                background-color: #343a40;
                color: white;
                padding: 5px;
                border-radius: 5px;
                margin-bottom: 5px;
            }
            .sidebar .sidebar-header {
                text-align: center;
                background-color: #343a40;
                color: white;
                padding: 20px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Título del menú de navegación con fondo negro y centrado
    st.sidebar.markdown("<div style='background-color: black; color: white; padding: 2px; margin-bottom: 0px; text-align: center; border-radius: 0px;'>MENÚ DE NAVEGACION</div>", unsafe_allow_html=True)

    # Menú de navegación con estilo de botón
    option = st.sidebar.radio("Selecciona una opción", ["Inicio", "Análisis de Clientes", "Segmentación y Reporte", "Cargar Nuevo Dataset"], key="nav", label_visibility="collapsed")

    if option == "Inicio":
        st.title("Segmentación de Clientes")
        st.write("Bienvenido a la plataforma de segmentación de clientes. Esta aplicación te permite analizar y segmentar clientes utilizando el algoritmo de K-Means.")
        st.markdown("---")
        
        # Sección de métricas clave
        if data is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Clientes", len(data))
            col2.metric("Edad Promedio", int(data["Age"].mean()))
            col3.metric("Ingreso Promedio", f"${int(data['Annual Income (k$)'].mean())}k")

            # Mostrar gráficos de distribución
            st.subheader("Distribución de Edad")
            fig = px.histogram(data, x="Age", nbins=30, color_discrete_sequence=["#636EFA"], title="Distribución de Edad")
            fig.update_layout(xaxis_title="Edad", yaxis_title="Frecuencia")
            st.plotly_chart(fig)

            st.subheader("Distribución de Ingresos Anuales")
            fig = px.histogram(data, x="Annual Income (k$)", nbins=30, color_discrete_sequence=["#00CC96"], title="Distribución de Ingresos Anuales")
            fig.update_layout(xaxis_title="Ingreso Anual (k$)", yaxis_title="Frecuencia")
            st.plotly_chart(fig)

            # Si los datos ya están segmentados, mostrar el gráfico de clusters
            if "Cluster" in data.columns:
                st.subheader("Distribución de Clientes por Clusters")
                fig = px.scatter(data, x="Age", y="Annual Income (k$)", color="Cluster", title="Distribución por Clusters")
                st.plotly_chart(fig)
        else:
            st.write("Cargar el dataset para continuar.")
    
    elif option == "Análisis de Clientes":
        if data is not None:
            analysis_page(data)
        else:
            st.write("Cargar el dataset para continuar.")
    
    elif option == "Segmentación y Reporte":
        if data is not None:
            segmentation_page(data)
        else:
            st.write("Cargar el dataset para continuar.")
    
    elif option == "Cargar Nuevo Dataset":
        st.subheader("Sube tu nuevo archivo CSV para analizarlo")
        uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
        
        if uploaded_file is not None:
            new_data = load_uploaded_data(uploaded_file)
            if new_data is not None:
                st.session_state["data"] = new_data
                st.success("¡Archivo cargado exitosamente!")
                st.write(new_data.head())


# Título del menú de navegación con fondo negro y centrado
    st.sidebar.markdown("<div style='background-color: black; color: white; padding: 2px; margin-bottom: 5px; text-align: center; border-radius: 0px;'>INTEGRANTES</div>", unsafe_allow_html=True)
# Desarrolladores
    st.sidebar.markdown("""
    <div style="background-color:#2d2d2d; padding: 5px; border-radius: 0px; color: white; font-family: Arial, sans-serif; text-align:left;">
           <ul style="list-style-type: disc; padding-left: 10px;">
            <li>Yhon Velásquez B.</li>
            <li>José Freddy Sanga Ch.</li>
            <li>Jhenery Carbajal P.</li>
            <li>Alicia Quino L.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)