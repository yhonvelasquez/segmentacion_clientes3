# Importa las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from st_aggrid import AgGrid

# Configuración básica de Streamlit
st.title("Segmentación de Clientes")
st.write("Esta aplicación permite segmentar clientes usando K-Means.")

# Paso 1: Carga de datos
def load_data():
    # Reemplaza "clientes.csv" por el nombre de tu archivo CSV
    data = pd.read_csv("data/Mall_Customers.csv")
    return data

data = load_data()
st.write("Datos cargados:", data.head())

# Paso 2: Análisis Exploratorio de Datos (opcional)
st.header("Análisis Exploratorio de Datos")
if st.checkbox("Mostrar estadísticas descriptivas"):
    st.write(data.describe())

# Mostrar correlación entre variables (solo numéricas)
if st.checkbox("Mostrar correlación entre variables"):
    # Seleccionar solo las columnas numéricas
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Verificar si hay columnas numéricas para correlacionar
    if not numeric_data.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No hay columnas numéricas para calcular la correlación.")

# Paso 3: Preprocesamiento de Datos
st.header("Preprocesamiento de Datos")
data_scaled_df = None  # Definir aquí para mantener la referencia

# Escalar datos al hacer clic en el botón
if st.button("Escalar datos"):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))  # Solo datos numéricos
    data_scaled_df = pd.DataFrame(data_scaled, columns=data.select_dtypes(include=[np.number]).columns)
    st.write("Datos escalados:", data_scaled_df.head())

# Paso 4: Segmentación con K-Means
st.header("Segmentación de Clientes")
num_clusters = st.slider("Seleccione el número de clusters (k)", min_value=2, max_value=10, value=3)

# Asegúrate de que los datos estén escalados antes de aplicar K-Means
if st.button("Aplicar K-Means") and data_scaled_df is not None:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled_df)  # Usar el dataframe escalado (data_scaled_df)
    data['Cluster'] = clusters
    st.write("Datos con Clusters Asignados:", data.head())

    # Visualización de los clusters
    st.subheader("Visualización de Clusters")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data_scaled_df.iloc[:, 0], y=data_scaled_df.iloc[:, 1], hue=clusters, palette="viridis", s=100)
    plt.xlabel(data_scaled_df.columns[0])
    plt.ylabel(data_scaled_df.columns[1])
    plt.title(f'Visualización de Clusters ({num_clusters} clusters): Ingreso Anual vs. Puntaje de Gasto')
    st.pyplot(fig)

# Paso 5: Descarga de resultados (opcional)
if st.button("Descargar Datos Segmentados"):
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(label="Descargar CSV con Segmentación", data=csv, file_name="clientes_segmentados.csv", mime="text/csv")
