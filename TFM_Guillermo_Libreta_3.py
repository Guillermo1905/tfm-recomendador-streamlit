import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import requests
from PIL import Image
from io import BytesIO
import unicodedata
import os
import re

# Configuración de la página
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Función para normalizar texto (quitar tildes y convertir a minúsculas)
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    # Convertir a minúsculas y eliminar tildes
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return texto

RUTA_MODELOS = 'modelos'

# Funciones para cargar los datos y los modelos primero
@st.cache_resource
def cargar_modelo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer('distiluse-base-multilingual-cased-v2', device=device)

@st.cache_data
def cargar_datos():
    path_df = os.path.join(RUTA_MODELOS, 'df_recomendador_embeddings_separados.pkl')
    
    with open(path_df, 'rb') as file:
        df_recomendador = pickle.load(file)
    
    # Agregamos columnas normalizadas para búsquedas
    df_recomendador['titulo_normalizado'] = df_recomendador['titulo'].apply(normalizar_texto)
    df_recomendador['titulo_original_normalizado'] = df_recomendador['titulo_original'].apply(normalizar_texto)
    
    # Cargamos los embeddings que creamos por separado desde la carpeta modelos
    embeddings_por_campo = {}
    campos = ['keywords_texto', 'descripcion', 'generos_nombres', 'actores_texto']
    for campo in campos:
        path_emb = os.path.join(RUTA_MODELOS, f'embeddings_{campo}.npy')
        embeddings_por_campo[campo] = np.load(path_emb)
    
    return df_recomendador, embeddings_por_campo

# Cargamos los datos y los modelos usando las funciones previas
try:
    df_recomendador, embeddings_por_campo = cargar_datos()
    model = cargar_modelo()
    datos_cargados = True
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    datos_cargados = False

# Función para obtener las imágenes de los pósters que haya disponibles
def obtener_poster(poster_path):
    if not poster_path or pd.isna(poster_path):
        return None

    if not str(poster_path).startswith('/'):
        poster_path = '/' + str(poster_path)

    url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    try:
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return None

        img = Image.open(BytesIO(response.content))
        return img

    except:
        return None

# Función modificada para buscar películas por título o título original
def buscar_peliculas(busqueda):
    busqueda_normalizada = normalizar_texto(busqueda)
    # Buscamos coincidencias en ambos títulos
    coincidencias_titulo = df_recomendador[df_recomendador['titulo_normalizado'].str.contains(busqueda_normalizada)]
    coincidencias_titulo_original = df_recomendador[df_recomendador['titulo_original_normalizado'].str.contains(busqueda_normalizada)]
    
    # Combinamos los resultados y eliminamos duplicados
    coincidencias = pd.concat([coincidencias_titulo, coincidencias_titulo_original]).drop_duplicates(subset=['id'])
    return coincidencias

# Función para recomendar películas usando los pesos
def recomendar_peliculas(titulo_o_id, n=10, pesos=None):
    if pesos is None:
        pesos = {
            'keywords_texto': 0.4, 
            'descripcion': 0.3, 
            'generos_nombres': 0.2, 
            'actores_texto': 0.1
        }
    
    if isinstance(titulo_o_id, int) or (isinstance(titulo_o_id, str) and titulo_o_id.isdigit()):
        # Buscamos por ID
        id_pelicula = int(titulo_o_id)
        idx = df_recomendador[df_recomendador['id'] == id_pelicula].index
        if len(idx) == 0:
            st.error(f"No se encontró la película con ID {id_pelicula}")
            return None
        idx = idx[0]
    else:
        # No usamos esta parte ya que la selección se hará primero
        return None
    
    # Calculamos la similitud ponderada usando todos los campos
    similitud_ponderada = np.zeros(len(df_recomendador))
    
    for campo, peso in pesos.items():
        movie_embedding = embeddings_por_campo[campo][idx].reshape(1, -1)
        sim_scores = cosine_similarity(movie_embedding, embeddings_por_campo[campo]).flatten()
        similitud_ponderada += sim_scores * peso
    
    # Obtenemos los índices de las películas más similares y las películas recomendadas sin contar la introducida por el usuario
    similar_indices = similitud_ponderada.argsort()[::-1][1:n+21]
    recomendaciones = []
    titulos_vistos = set()
    count = 0
    
    for idx in similar_indices:
        titulo = df_recomendador.iloc[idx]['titulo']
        
        if titulo not in titulos_vistos and count < n:
            recomendaciones.append(df_recomendador.iloc[idx])
            titulos_vistos.add(titulo)
            count += 1
            
        if count >= n:
            break
    
    return pd.DataFrame(recomendaciones)

# Función para formatear los géneros
def formatear_generos(generos):
    if not generos:
        return ""
        
    if isinstance(generos, str):
        try:
            generos = ast.literal_eval(generos)
            return ", ".join(generos)
        except:
            return generos
    else:
        return ", ".join(generos) if generos else ""

# Función para formatear la fecha en año
def obtener_anio(fecha):
    if not fecha or pd.isna(fecha):
        return "Año desconocido"
    
    try:
        return fecha.split('-')[0]
    except:
        return "Año desconocido"

# Función para formatear los actores principales
def formatear_actores(actores):
    if not actores:
        return ""
        
    actores_list = []
    
    if isinstance(actores, str):
        try:
            actores_data = ast.literal_eval(actores)
            actores_list = [f"{actor['name']} ({actor.get('character', 'Sin personaje')})" 
                           for actor in actores_data if 'name' in actor]
        except:
            return actores
    else:
        actores_list = [f"{actor['name']} ({actor.get('character', 'Sin personaje')})" 
                       for actor in actores if 'name' in actor]
    
    return ", ".join(actores_list[:5]) if actores_list else ""

# Función para formatear directores
def formatear_directores(directores):
    if not directores:
        return ""
        
    if isinstance(directores, str):
        try:
            directores = ast.literal_eval(directores)
            return ", ".join(directores)
        except:
            return directores
    else:
        return ", ".join(directores) if directores else ""

# Función para mostrar los detalles de una película
def mostrar_detalles_pelicula(pelicula_id):
    pelicula = df_recomendador[df_recomendador['id'] == pelicula_id].iloc[0]
    
    st.title(f"Detalles de: {pelicula['titulo']}")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster = obtener_poster(pelicula.get('poster_path'))
        if poster:
            st.image(poster, width=300)
        else:
            st.write("🎬 Póster no disponible")
    
    with col2:
        if pelicula['titulo_original'] != pelicula['titulo']:
            st.subheader(f"Título original: {pelicula['titulo_original']}")
        
        # Información básica
        st.markdown(f"**Fecha de estreno:** {pelicula['fecha_estreno'] if not pd.isna(pelicula['fecha_estreno']) else 'Desconocida'}")
        st.markdown(f"**Géneros:** {formatear_generos(pelicula['generos_nombres'])}")
        st.markdown(f"**Valoración:** {pelicula['votos_promedio']:.1f}/10 ({pelicula['votos_cantidad']} votos)" if not pd.isna(pelicula['votos_promedio']) else "Valoración no disponible")
        st.markdown(f"**Popularidad:** {pelicula['popularidad']:.1f}" if not pd.isna(pelicula['popularidad']) else "Popularidad no disponible")
        st.markdown(f"**Duración:** {pelicula['duracion']} minutos" if not pd.isna(pelicula['duracion']) else "Duración desconocida")
        st.markdown(f"**Idioma original:** {pelicula['idioma_original']}" if not pd.isna(pelicula['idioma_original']) else "Idioma desconocido")
    
    # Más información
    st.subheader("Más información")
    
    col1, col2 = st.columns(2)
    
    with col1:
        directores = formatear_directores(pelicula['directores'])
        if directores:
            st.markdown(f"**Director(es):** {directores}")
        
        # Productoras
        if 'productoras' in pelicula and pelicula['productoras']:
            try:
                if isinstance(pelicula['productoras'], str):
                    productoras = ast.literal_eval(pelicula['productoras'])
                else:
                    productoras = pelicula['productoras']
                if productoras:
                    st.markdown(f"**Productoras:** {', '.join(productoras)}")
            except:
                pass
    
    with col2:
        # Actores principales
        actores = formatear_actores(pelicula['actores_principales'])
        if actores:
            st.markdown("**Actores principales:**")
            st.markdown(actores)
        
        # Países
        if 'paises' in pelicula and pelicula['paises']:
            try:
                if isinstance(pelicula['paises'], str):
                    paises = ast.literal_eval(pelicula['paises'])
                else:
                    paises = pelicula['paises']
                if paises:
                    st.markdown(f"**Países:** {', '.join(paises)}")
            except:
                pass
    
    # Descripción
    st.subheader("Descripción")
    if not pd.isna(pelicula['descripcion']):
        st.write(pelicula['descripcion'])
    else:
        st.write("No hay descripción disponible.")
    
    # Keywords (si están disponibles)
    if 'keywords' in pelicula and pelicula['keywords']:
        try:
            if isinstance(pelicula['keywords'], str):
                keywords = ast.literal_eval(pelicula['keywords'])
            else:
                keywords = pelicula['keywords']
            if keywords:
                st.subheader("Palabras clave")
                st.write(", ".join(keywords))
        except:
            pass
    
    # Botón para volver a las recomendaciones
    if st.button("← Volver a las recomendaciones", key="volver_btn"):
        # Eliminar el estado de detalle de película
        del st.session_state.pelicula_detalle
        st.rerun()

# Inicializamos el estado de la aplicación
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = 'inicio'

# Verificar si estamos en la página de detalles
if 'pelicula_detalle' in st.session_state and datos_cargados:
    mostrar_detalles_pelicula(st.session_state.pelicula_detalle)
else:
    # Título de la aplicación
    st.title("🎬 Recomendador de Películas")
    st.markdown("""
    Esta aplicación encuentra películas similares a una introducida por el usuario usando 
    un sistema de recomendación basado en embeddings.
    """)

    # Búsqueda por película
    st.subheader("Busca una película")

    col1, col2 = st.columns([3, 1])

    with col1:
        busqueda_input = st.text_input(
            "Introduce el título de una película que te haya gustado:",
            key="busqueda_input"
        )

    with col2:
        num_recomendaciones = st.slider(
            "Número de recomendaciones:",
            min_value=1,
            max_value=20,
            value=5
        )

    # Mostrar resultados de búsqueda
    if busqueda_input and datos_cargados:
        coincidencias = buscar_peliculas(busqueda_input)
        
        if len(coincidencias) > 0:
            st.write(f"Se encontraron {len(coincidencias)} películas:")
            
            # Crear opciones con el año entre paréntesis y título original si está disponible
            opciones = []
            for _, row in coincidencias.head(10).iterrows():
                titulo_display = row['titulo']
                if not pd.isna(row['titulo_original']) and row['titulo_original'] != row['titulo']:
                    titulo_display += f" / {row['titulo_original']}"
                opciones.append((titulo_display, row['fecha_estreno'], row['id']))
                
            opciones_display = [f"{titulo} ({obtener_anio(fecha)})" for titulo, fecha, _ in opciones]
            
            # Verificar si hay una selección previa
            if 'previous_selection' not in st.session_state:
                st.session_state.previous_selection = None
            
            # Dejar que el usuario seleccione la película
            seleccion = st.selectbox(
                "Selecciona la película para ver recomendaciones:",
                options=opciones_display,
                key="pelicula_selector_central"
            )
            
            # Encontrar el ID de la película seleccionada
            indice_seleccionado = opciones_display.index(seleccion)
            id_seleccionado = opciones[indice_seleccionado][2]
            
            # Si la selección ha cambiado, establecer la película base automáticamente
            if st.session_state.previous_selection != seleccion:
                st.session_state.previous_selection = seleccion
                st.session_state.pelicula_base = id_seleccionado
                st.rerun()  # Rerun para mostrar las recomendaciones
        else:
            st.error(f"No se encontraron películas con título similar a '{busqueda_input}'")

    # Mostramos las recomendaciones basadas en la película seleccionada
    if 'pelicula_base' in st.session_state and datos_cargados:
        # Un spinner que gire mientras se calculan las recomendaciones
        with st.spinner('Buscando películas similares...'):
            # Usar pesos personalizados si están definidos
            if 'pesos_personalizados' in st.session_state:
                recomendaciones = recomendar_peliculas(
                    str(st.session_state.pelicula_base), 
                    n=num_recomendaciones, 
                    pesos=st.session_state.pesos_personalizados
                )
            else:
                recomendaciones = recomendar_peliculas(
                    str(st.session_state.pelicula_base), 
                    n=num_recomendaciones
                )
        
        if recomendaciones is not None and not recomendaciones.empty:
            st.success(f"¡Encontradas {len(recomendaciones)} películas similares!")
            
            pelicula_ref = df_recomendador[df_recomendador['id'] == st.session_state.pelicula_base].iloc[0]
            
            st.subheader(f"Basado en: {pelicula_ref['titulo']}")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                poster = obtener_poster(pelicula_ref.get('poster_path'))
                if poster:
                    st.image(poster, width=200)
                else:
                    st.write("🎬 Póster no disponible")
            
            with col2:
                # Añadir título original si es diferente al título en español
                if not pd.isna(pelicula_ref['titulo_original']) and pelicula_ref['titulo_original'] != pelicula_ref['titulo']:
                    st.write(f"**Título original:** {pelicula_ref['titulo_original']}")
                st.write(f"**Géneros:** {formatear_generos(pelicula_ref['generos_nombres'])}")
                st.write(f"**Valoración:** {pelicula_ref['votos_promedio']:.1f}/10 ({pelicula_ref['votos_cantidad']} votos)")
                st.write(f"**Descripción:** {pelicula_ref['descripcion']}")
            
            # Mostrar recomendaciones
            st.subheader("Películas recomendadas")
            
            # Mostrar las recomendaciones en filas de 3 películas
            num_columnas = 3
            for i in range(0, len(recomendaciones), num_columnas):
                cols = st.columns(num_columnas)
                
                for j in range(num_columnas):
                    if i + j < len(recomendaciones):
                        peli = recomendaciones.iloc[i + j]
                        with cols[j]:
                            st.write(f"**{peli['titulo']}**")
                            
                            # Añadir título original si es diferente al título en español
                            if not pd.isna(peli['titulo_original']) and peli['titulo_original'] != peli['titulo']:
                                st.write(f"*({peli['titulo_original']})*")
                            
                            poster = obtener_poster(peli.get('poster_path'))
                            if poster:
                                st.image(poster, width=200)
                            else:
                                st.write("🎬 Póster no disponible")
                            
                            st.write(f"**Géneros:** {formatear_generos(peli['generos_nombres'])}")
                            st.write(f"**Valoración:** {peli['votos_promedio']:.1f}/10 ({peli['votos_cantidad']} votos)")
                            
                            # Botón para mostrar detalles
                            if st.button(f"Detalles de {peli['titulo']}", key=f"detalles_{peli['id']}"):
                                # Guardamos el ID de la película para mostrar sus detalles
                                st.session_state.pelicula_detalle = peli['id']
                                st.rerun()

        # Botón para volver a buscar otra película
        if st.button("Buscar otra película", key="buscar_otra"):
            # Limpiar los estados relacionados con películas
            if 'pelicula_base' in st.session_state:
                del st.session_state.pelicula_base
            if 'pelicula_detalle' in st.session_state:
                del st.session_state.pelicula_detalle
            st.rerun()

# Información adicional en el sidebar
st.sidebar.subheader("Sobre el recomendador")
st.sidebar.info("""
Este recomendador de películas usa embeddings que se han generado a partir de
las características más relevantes (palabras clave, descripción, géneros y actores) de un total de 30.000 películas.
""")

# Lo siguiente tiene que ver con dar la posibilidad al usuario de ajustar de manera manual la importancia que quiere darle a cada característica
st.sidebar.subheader("Ajustes avanzados")
with st.sidebar.expander("Pesos de las características"):
    peso_keywords = st.slider("Peso de las palabras clave", 0.0, 1.0, 0.4, 0.1)
    peso_descripcion = st.slider("Peso de las descripción", 0.0, 1.0, 0.3, 0.1)
    peso_generos = st.slider("Peso de los géneros", 0.0, 1.0, 0.2, 0.1)
    peso_actores = st.slider("Peso de los actores y actrices", 0.0, 1.0, 0.1, 0.1)
    
    # Normalización pesos
    suma_pesos = peso_keywords + peso_descripcion + peso_generos + peso_actores
    if suma_pesos > 0:
        pesos_personalizados = {
            'keywords_texto': peso_keywords / suma_pesos,
            'descripcion': peso_descripcion / suma_pesos,
            'generos_nombres': peso_generos / suma_pesos,
            'actores_texto': peso_actores / suma_pesos
        }
        st.session_state.pesos_personalizados = pesos_personalizados
    else:
        st.error("Al menos uno de los pesos debe ser mayor que cero.")
