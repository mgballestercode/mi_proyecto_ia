import streamlit as st
from transformers import pipeline
import pandas as pd
import sqlite3
import os
import toml
import logging
import sys

# Cargar secretos de st.secrets a os.environ (para compatibilidad local)
def cargar_secretos():
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            with open(".streamlit/secrets.toml", "r") as f:
                secrets = toml.load(f)
                # Cargar claves de la sección [auth]
                if "auth" in secrets:
                    # Exportar nivel superior de auth
                    for key, value in secrets["auth"].items():
                        if isinstance(value, dict):
                            continue
                        os.environ[key.upper()] = (
                            str(value) if not isinstance(value, list) else " ".join(value)
                        )

                    # Si tenemos redirect_uri a nivel [auth], propagar a OIDC_REDIRECT_URI si no está definido
                    if "redirect_uri" in secrets["auth"] and not os.environ.get("OIDC_REDIRECT_URI"):
                        os.environ["OIDC_REDIRECT_URI"] = str(secrets["auth"]["redirect_uri"])            

                    # Si existe [auth.oidc], mapear a variables OIDC_*
                    if isinstance(secrets["auth"].get("oidc"), dict):
                        oidc = secrets["auth"]["oidc"]
                        mapping = {
                            "client_id": "OIDC_CLIENT_ID",
                            "client_secret": "OIDC_CLIENT_SECRET",
                            "authorization_endpoint": "OIDC_AUTHORIZATION_ENDPOINT",
                            "token_endpoint": "OIDC_TOKEN_ENDPOINT",
                            "userinfo_endpoint": "OIDC_USERINFO_ENDPOINT",
                            "redirect_uri": "OIDC_REDIRECT_URI",
                            "scopes": "OIDC_SCOPES",
                        }
                        for key, env_key in mapping.items():
                            if key in oidc and oidc[key] is not None:
                                value = oidc[key]
                                os.environ[env_key] = (
                                    value if isinstance(value, str) else " ".join(value)
                                )

                        # Señalar proveedor elegido explícitamente
                        os.environ.setdefault("AUTH_PROVIDER", "oidc")
        # En Streamlit Cloud, los secretos ya están en os.environ
    except FileNotFoundError:
        st.warning("Archivo secrets.toml no encontrado localmente. Asegúrate de configurarlo.")
    except Exception as e:
        st.error(f"Error cargando secretos: {e}")

cargar_secretos()

# Configurar logging a consola
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Permitir HTTP en local para OAuth (solo desarrollo)
if os.environ.get("REDIRECT_URI", "http://localhost:8501").startswith("http://"):
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    log.debug("OAUTHLIB_INSECURE_TRANSPORT=1 habilitado por entorno local http://")

# Configuración de DB
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

# Cargar modelo (cacheado)
@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english")

clasificador = cargar_modelo()

def login_screen():
    st.header("Esta app es privada")
    st.subheader("Por favor, inicia sesión")
    st.button("Iniciar sesión con Google", on_click=st.login)

# Pantalla de login simple como en app-simple.py
if not getattr(st, "user", None) or not getattr(st.user, "is_logged_in", False):
    login_screen()
    st.stop()

# Si logueado, accede a info del usuario y muestra logout
user_email = getattr(st.user, "email", None) or getattr(st.user, "name", None) or "usuario@desconocido"
st.sidebar.success(f"Bienvenido, {user_email}!")
if st.sidebar.button("Logout"):
    st.logout()
    st.experimental_rerun()

# Título
st.title("Clasificador de Sentimiento con IA - Versión Segura")

# Input multi-texto
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

# Mostrar historial
st.subheader("Tu Historial de Análisis")
cursor.execute("SELECT frase, sentimiento, confianza, timestamp FROM analisis WHERE user_email = ? ORDER BY timestamp DESC", (user_email,))
historial = cursor.fetchall()
if historial:
    df_hist = pd.DataFrame(historial, columns=["Frase", "Sentimiento", "Confianza", "Fecha"])
    st.dataframe(df_hist)
else:
    st.info("No hay historial aún.")

# Cerrar DB
conn.close()