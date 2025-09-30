import streamlit as st
from transformers import pipeline
import pandas as pd  # Nueva import para tablas

# Cargar el modelo de análisis de sentimiento (cacheado)
@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english")

clasificador = cargar_modelo()

# Título de la app
st.title("Clasificador de Sentimiento con IA - Versión Multi-Texto")

# Input del usuario: Ahora un textarea para múltiples líneas
texto = st.text_area("Introduce frases para analizar (una por línea):", 
                     placeholder="Frase 1\nFrase 2\n...")

# Botón para procesar
if st.button("Analizar Sentimiento"):
    if texto:
        # Dividir el input en líneas
        frases = texto.strip().split("\n")
        resultados = []
        
        for frase in frases:
            if frase.strip():  # Ignorar líneas vacías
                resultado = clasificador(frase)[0]
                resultados.append({
                    "Frase": frase,
                    "Sentimiento": resultado['label'],
                    "Confianza": f"{resultado['score']:.2f}"
                })
        
        # Mostrar resultados en una tabla
        if resultados:
            df = pd.DataFrame(resultados)
            st.table(df)
        else:
            st.warning("No hay frases válidas.")
    else:
        st.warning("Por favor, introduce al menos una frase.")