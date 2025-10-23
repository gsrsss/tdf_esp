import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# --- Configuración de la Página ---
st.set_page_config(page_title="Demo TF-IDF Español", layout="wide", initial_sidebar_state="collapsed")

# --- Estilos CSS Personalizados ---
st.markdown("""
<style>
    /* Color de fondo de la app */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Contenedor principal para centrar y añadir padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Estilo de "Tarjeta" para las columnas de entrada */
    [data-testid="column"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px !important; /* !important para sobreescribir padding */
        box-shadow: 0 6px 20px rgba(0,0,0,0.04);
        border: 1px solid #e6e6e6;
        transition: all 0.3s ease;
    }
    
    [data-testid="column"]:hover {
         box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    }

    /* Espacio entre las tarjetas de columna */
    [data-testid="column"]:first-child {
        margin-right: 15px;
    }
    [data-testid="column"]:last-child {
        margin-left: 15px;
    }

    /* Botones de sugerencias (secundarios) */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #d0d0d0;
        background-color: #fafafa;
        color: #333;
    }
    .stButton > button:hover {
        background-color: #f0f0f0;
        border-color: #c0c0c0;
        color: #000;
    }
    
    /* Botón primario de "Analizar" */
    .stButton > button[kind="primary"] {
        background-color: #E91E63; /* Rosado */
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #C2185B; /* Rosado oscuro */
    }

    /* Estilo para la respuesta 'success' */
    .stSuccess {
        background-color: #e6f7ff;
        border: 1px solid #b3e0ff;
        border-radius: 8px;
        color: #0056b3;
        font-size: 1.05rem;
    }
    
    /* Títulos */
    h1 {
        color: #1a1a1a;
        font-weight: 600;
    }
    h3 {
        color: #333;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Título y Descripción ---
st.title("Demo de Búsqueda Semántica (TF-IDF) en Español")
st.write("""
Esta aplicación utiliza TF-IDF y la similitud del coseno para encontrar el documento más relevante para tu pregunta.
El análisis incluye *stemming* en español (ej: "jugar", "jugando", "juegan" se tratan como "jug").
""")

# --- Documentos de Ejemplo ---
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# --- Funciones de Procesamiento ---
# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# --- Lógica de Estado para Sugerencias ---

# Inicializar estado de sesión para la pregunta
if 'question' not in st.session_state:
    st.session_state.question = "¿Dónde juegan el perro y el gato?"

# Función callback para actualizar la pregunta
def set_question(q_text):
    st.session_state.question = q_text

# --- Layout de la Interfaz ---
st.header("1. Ingresa tus Datos")

# Usamos un contenedor para aplicar el fondo de tarjeta
with st.container():
    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area("Documentos (uno por línea):", default_docs, height=210)
        
        # El 'key' "question" vincula este input a st.session_state.question
        question = st.text_input("Escribe tu pregunta:", key="question")

    with col2:
        st.markdown("### 🫧 Preguntas sugeridas:")
        
        st.button(
            "¿Dónde juegan el perro y el gato?", 
            on_click=set_question, 
            args=["¿Dónde juegan el perro y el gato?"], 
            use_container_width=True
        )
        st.button(
            "¿Qué hacen los niños en el parque?", 
            on_click=set_question, 
            args=["¿Qué hacen los niños en el parque?"], 
            use_container_width=True
        )
        st.button(
            "¿Cuándo cantan los pájaros?", 
            on_click=set_question, 
            args=["¿Cuándo cantan los pájaros?"], 
            use_container_width=True
        )
        st.button(
            "¿Dónde suena la música alta?", 
            on_click=set_question, 
            args=["¿Dónde suena la música alta?"], 
            use_container_width=True
        )
        st.button(
            "¿Qué animal maúlla durante la noche?", 
            on_click=set_question, 
            args=["¿Qué animal maúlla durante la noche?"], 
            use_container_width=True
        )


st.write("---") # Separador

# --- Lógica de Análisis y Resultados ---
if st.button("🔍 Analizar y Encontrar Respuesta", type="primary", use_container_width=True):
    
    # Usar la pregunta de st.session_state
    current_question = st.session_state.question
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not current_question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        try:
            st.header("2. Resultados del Análisis")
            
            # Crear vectorizador TF-IDF
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                min_df=1  # Incluir todas las palabras
            )
            
            # Ajustar con documentos
            X = vectorizer.fit_transform(documents)
            
            # Calcular similitud con la pregunta
            question_vec = vectorizer.transform([current_question])
            similarities = cosine_similarity(question_vec, X).flatten()
            
            # Encontrar mejor respuesta
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]
            
            # --- Mostrar Respuesta Principal ---
            st.markdown("### 🎯 Respuesta Más Relevante")
            st.markdown(f"**Tu pregunta:** *{current_question}*")
            
            if best_score > 0.01:  # Umbral
                st.success(f"**Mejor coincidencia (Documento {best_idx+1}):**\n\n>{best_doc}")
                st.write(f"**Puntaje de similitud:** {best_score:.3f}")
                
                # Mostrar stems coincidentes
                vocab = vectorizer.get_feature_names_out()
                q_stems = tokenize_and_stem(current_question)
                matched = [s for s in q_stems if s in vocab and X[best_idx].toarray()[0][vectorizer.vocabulary_[s]] > 0]
                st.write("**Stems coincidentes:**", f"`{', '.join(matched) or 'Ninguno'}`")
                
            else:
                st.warning(f"No se encontró una buena coincidencia (puntaje más alto: {best_score:.3f}).")

            
            st.write("---") # Separador

            # --- Detalles Adicionales en Expanders ---
            st.markdown("### 🔬 Detalles del Análisis")

            with st.expander("Ver todos los puntajes de similitud"):
                sim_df = pd.DataFrame({
                    "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                    "Texto": documents,
                    "Similitud": similarities
                })
                st.dataframe(sim_df.sort_values("Similitud", ascending=False), use_container_width=True)

            with st.expander("Ver Matriz TF-IDF (stems)"):
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Doc {i+1}" for i in range(len(documents))]
                )
                st.dataframe(df_tfidf.round(3), use_container_width=True)

        except ValueError as e:
            if "empty vocabulary" in str(e):
                st.error("⚠️ Error: No se pudo construir un vocabulario. Asegúrate de que los documentos no estén vacíos o compuestos solo de 'stop words' (palabras comunes).")
            else:
                st.error(f"Ocurrió un error: {e}")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {e}")
