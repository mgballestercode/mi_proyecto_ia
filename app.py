import streamlit as st
from transformers import pipeline
import pandas as pd
import sqlite3

# -----------------------
# Autenticación (simple)
# -----------------------

def login_screen():
    st.header("Esta app es privada")
    st.subheader("Por favor, inicia sesión")
    st.button("Iniciar sesión con Google", on_click=st.login)

if not getattr(st, "user", None) or not getattr(st.user, "is_logged_in", False):
    login_screen()
    st.stop()

user_email = getattr(st.user, "email", None) or getattr(st.user, "name", None) or "usuario@desconocido"
st.sidebar.success(f"Bienvenido, {user_email}!")
if st.sidebar.button("Logout"):
    st.logout()
    st.experimental_rerun()

# -----------------------
# App principal
# -----------------------

# DB
DB_PATH = "historial.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS analisis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        frase TEXT,
        sentimiento TEXT,
        confianza REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Modelo cacheado
@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

clasificador = cargar_modelo()

# UI
st.title("Clasificador de Sentimiento con IA - Versión Segura")
texto = st.text_area("Introduce frases (una por línea):", placeholder="Frase 1\nFrase 2\n...")

if st.button("Analizar Sentimiento"):
    if texto:
        frases = [f.strip() for f in texto.split("\n") if f.strip()]
        resultados = []
        for frase in frases:
            resultado = clasificador(frase)[0]
            label = resultado['label']
            score = resultado['score']
            resultados.append({
                "Frase": frase,
                "Sentimiento": label,
                "Confianza": f"{score:.2f}"
            })
            cursor.execute('''
                INSERT INTO analisis (user_email, frase, sentimiento, confianza)
                VALUES (?, ?, ?, ?)
            ''', (user_email, frase, label, score))
            conn.commit()
        df = pd.DataFrame(resultados)
        st.table(df)
    else:
        st.warning("Introduce al menos una frase.")

st.subheader("Tu Historial de Análisis")
cursor.execute("SELECT frase, sentimiento, confianza, timestamp FROM analisis WHERE user_email = ? ORDER BY timestamp DESC", (user_email,))
historial = cursor.fetchall()
if historial:
    df_hist = pd.DataFrame(historial, columns=["Frase", "Sentimiento", "Confianza", "Fecha"])
    st.dataframe(df_hist)
else:
    st.info("No hay historial aún.")

conn.close()