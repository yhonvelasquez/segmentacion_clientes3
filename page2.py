# Importa las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Configuración básica de Streamlit
st.title("Segmentación de Clientes")
st.write("Esta aplicación permite segmentar clientes usando K-Means.")

# Paso 1: Carga de datos
@st.cache_data
def load_data(file_path="data/Mall_Customers.csv"):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Archivo no encontrado. Asegúrate de que el archivo CSV esté en la ruta correcta.")
        return None

data = load_data()
if data is not None:
    st.write("Datos cargados:", data.head())

    # Paso 2: Análisis Exploratorio de Datos (opcional)
    st.header("Análisis Exploratorio de Datos")
    if st.checkbox("Mostrar estadísticas descriptivas"):
        st.write(data.describe())

    # Mostrar correlación entre variables numéricas
    if st.checkbox("Mostrar correlación entre variables"):
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("No hay columnas numéricas para calcular la correlación.")

    # Paso 3: Preprocesamiento de Datos
    st.header("Preprocesamiento de Datos")
    if "data_scaled_df" not in st.session_state:
        st.session_state["data_scaled_df"] = None

    if st.button("Escalar datos"):
        scaler = StandardScaler()
        numeric_data = data.select_dtypes(include=[np.number])
        data_scaled = scaler.fit_transform(numeric_data)
        st.session_state["data_scaled_df"] = pd.DataFrame(data_scaled, columns=numeric_data.columns)
        st.write("Datos escalados:", st.session_state["data_scaled_df"].head())

    # Paso 4: Segmentación con K-Means
    st.header("Segmentación de Clientes")
    num_clusters = st.slider("Seleccione el número de clusters (k)", min_value=2, max_value=10, value=3)

    # Asegúrate de que los datos estén escalados antes de aplicar K-Means
    if st.button("Aplicar K-Means") and st.session_state["data_scaled_df"] is not None:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(st.session_state["data_scaled_df"])
        data['Cluster'] = clusters
        st.write("Datos con Clusters Asignados:", data.head())

        # Visualización interactiva de los clusters
        st.subheader("Visualización de Clusters")
        fig = px.scatter(st.session_state["data_scaled_df"], x=st.session_state["data_scaled_df"].columns[0],
                         y=st.session_state["data_scaled_df"].columns[1], color=data['Cluster'].astype(str),
                         title=f'Clusters ({num_clusters} clusters)', labels={'color': 'Cluster'})
        st.plotly_chart(fig)

    # Paso 5: Descarga de resultados (opcional)
    if st.button("Descargar Datos Segmentados"):
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Descargar CSV con Segmentación", data=csv,
                           file_name="clientes_segmentados.csv", mime="text/csv")
else:
    st.write("No se pudo cargar el archivo de datos.")
