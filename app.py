import streamlit as st
from transformers import pipeline

# Cargar el modelo de análisis de sentimiento (solo una vez, al inicio)
@st.cache_resource  # Cachea el modelo para no recargarlo cada vez
def cargar_modelo():
    return pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english")

clasificador = cargar_modelo()

# Título de la app
st.title("Clasificador de Sentimiento con IA")

# Input del usuario
texto = st.text_input("Introduce una frase para analizar:", 
                      placeholder="Escribe aquí...")

# Botón para procesar
if st.button("Analizar Sentimiento"):
    if texto:
        # Procesar el texto
        resultado = clasificador(texto)[0]
        label = resultado['label']
        score = resultado['score']
        
        # Mostrar resultado
        st.success(f"Sentimiento: **{label}** (Confianza: {score:.2f})")
    else:
        st.warning("Por favor, introduce un texto.")
