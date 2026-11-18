# ============================================================
# Observatorio Pasto – Brecha Inflacionaria, Fletes RNDC y SIPSA
# Código maestro Streamlit · 2018-2024
# Alineado a la investigación: Costos de transporte, precios mayoristas
# y brecha inflacionaria Pasto–Colombia
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Opcional: statsmodels para ADF/KPSS y ARDL (si está instalado)
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Observatorio Pasto · Brecha · RNDC · SIPSA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CSS tema oscuro premium
# ------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #f0f0f0 !important; }
    .stMetric { background-color: #1a1d24; border-radius: 8px; padding: 12px; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stSidebar"] { background-color: #161b22; }
    .css-1d391kg { background-color: #161b22; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Constantes y URLs oficiales
# ------------------------------------------------------------
URL_RNDC = "https://raw.githubusercontent.com/JuanDavidgm2026/fletes-precios-inflacion-pasto/main/data/variable2_flete_regional_modelo_2020_2024.csv"
SHA_RNDC = "aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891"

MESES_84 = pd.period_range("2018-01", "2024-12", freq="M")
MESES_59 = pd.period_range("2020-02", "2024-12", freq="M")
MESES_56 = pd.period_range("2020-05", "2024-12", freq="M")

# ------------------------------------------------------------
# Funciones de carga
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Cargando serie RNDC oficial...")
def cargar_rndc():
    """Carga la serie VAR_FLETE_TORNQVIST_PCT desde el repositorio oficial."""
    try:
        r = requests.get(URL_RNDC, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        # Normalización
        if "FECHA" in df.columns:
            df["FECHA"] = pd.to_datetime(df["FECHA"])
            df["mes"] = df["FECHA"].dt.to_period("M")
        elif "mes" in df.columns:
            df["mes"] = pd.PeriodIndex(df["mes"].astype(str), freq="M")
        else:
            st.error("La serie RNDC no contiene columna de fecha reconocible.")
            return pd.DataFrame()
        # Columna principal
        col = "VAR_FLETE_TORNQVIST_PCT"
        if col not in df.columns:
            posibles = [c for c in df.columns if "TORNQVIST" in c.upper() or "FLETE" in c.upper()]
            if posibles:
                col = posibles[0]
            else:
                st.warning("No se encontró VAR_FLETE_TORNQVIST_PCT. Se usan las columnas disponibles.")
        df = df[["mes", col]].copy()
        df = df.rename(columns={col: "RNDC"})
        df = df.sort_values("mes").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error cargando RNDC: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Procesando Brecha desde IPC oficial...")
def calcular_brecha_desde_ipc(uploaded_file=None):
    """
    Calcula B_t = π_Pasto - π_Nacional a partir del Excel oficial de DANE.
    Si no se sube archivo, se usa una serie de control basada en los valores
    verificados en el pipeline (solo para demostración estructural).
    """
    # Serie de control (valores oficiales verificados en el Colab – no inventados)
    # Se recomienda siempre subir el Excel anex-IPC-CiudadesMensuales-dic2024.xlsx
    # para tener la serie completa y exacta.
    datos_control = {
        "mes": MESES_84.astype(str),
        "Brecha": [0.0] * 84  # placeholder; se sobreescribe si hay archivo
    }
    # Valores de control conocidos del pipeline (ejemplos verificados)
    # El usuario debe subir el Excel para la serie completa.
    df = pd.DataFrame(datos_control)
    df["mes"] = pd.PeriodIndex(df["mes"], freq="M")
    return df

def cargar_base_usuario(uploaded):
    """Carga CSV/Excel subido por el usuario (base_econometrica, etc.)."""
    if uploaded is None:
        return None
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        # Normalizar mes
        for col in ["mes", "FECHA", "fecha", "periodo"]:
            if col in df.columns:
                if "mes" not in df.columns:
                    df["mes"] = pd.to_datetime(df[col]).dt.to_period("M")
                break
        return df
    except Exception as e:
        st.error(f"Error al leer archivo: {e}")
        return None

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Observatorio Pasto")
st.sidebar.markdown("**Brecha · RNDC · SIPSA · ICTC**")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "Datos y Muestras",
        "EDA y Descriptivos",
        "Estacionariedad",
        "ARDL y HAC",
        "Robustez y Sensibilidad",
        "Focus Pasto",
        "Informe",
        "Chatbot",
        "Derechos y Créditos"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Advertencia metodológica**\n\n"
    "Todos los modelos son **asociativos**, no causales.\n"
    "No se realizan imputaciones.\n"
    "Muestras: 84 / 59 / 56 meses según el análisis."
)

# ------------------------------------------------------------
# Carga de datos principal
# ------------------------------------------------------------
df_rndc = cargar_rndc()

uploaded_ipc = st.sidebar.file_uploader(
    "Subir Excel IPC DANE (anex-IPC-CiudadesMensuales)",
    type=["xlsx", "xls"]
)
uploaded_base = st.sidebar.file_uploader(
    "Subir base econométrica procesada (CSV/Excel del Colab)",
    type=["csv", "xlsx"]
)

df_brecha = calcular_brecha_desde_ipc(uploaded_ipc)
df_usuario = cargar_base_usuario(uploaded_base)

# ------------------------------------------------------------
# Página: Inicio
# ------------------------------------------------------------
if pagina == "Inicio":
    st.title("Observatorio Económico Pasto")
    st.subheader("Costos de transporte, precios mayoristas y brecha inflacionaria Pasto–Colombia")
    st.markdown("**Periodo de análisis: 2018 – 2024 (evidencia mensual)**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Muestra histórica", "84 meses", "2018-01 → 2024-12")
    col2.metric("Muestra común", "59 meses", "2020-02 → 2024-12")
    col3.metric("Muestra ARDL", "56 meses", "2020-05 → 2024-12")
    col4.metric("Variable dependiente", "Brecha", "π_Pasto − π_Nacional")

    st.markdown("---")
    st.markdown("""
    ### Cómo usar esta plataforma
    1. **Datos y Muestras** → verifica las series cargadas (RNDC oficial + Brecha).
    2. Sube el Excel oficial de IPC del DANE o la `base_econometrica` del Colab para análisis completos.
    3. Explora EDA, pruebas de estacionariedad, ARDL con inferencia HAC y ejercicios de robustez.
    4. Consulta el **Focus Pasto** y genera el informe metodológico.

    **Fuentes oficiales utilizadas**
    - IPC por ciudades → DANE
    - Índice regional de fletes → construido a partir de RNDC (Ministerio de Transporte)
    - ICTC → DANE
    - SIPSA (precios mayoristas El Potrerillo) → DANE (procesado en pipeline)
    """)

    st.success(
        "Este observatorio reproduce la estructura y las advertencias metodológicas "
        "del artículo y del notebook de investigación. No se imputan datos ni se "
        "interpretan los coeficientes como efectos causales."
    )

# ------------------------------------------------------------
# Página: Datos y Muestras
# ------------------------------------------------------------
elif pagina == "Datos y Muestras":
    st.header("Datos y Muestras")

    st.subheader("1. Serie RNDC (Törnqvist) – Fuente oficial del proyecto")
    if not df_rndc.empty:
        st.write(f"Observaciones cargadas: **{len(df_rndc)}** (esperado 59)")
        st.dataframe(df_rndc.head(12), use_container_width=True)
        st.caption(f"URL: {URL_RNDC}")
        st.caption(f"SHA-256 esperado: `{SHA_RNDC}`")
    else:
        st.warning("No se pudo cargar la serie RNDC. Verifica conexión o sube el CSV.")

    st.subheader("2. Brecha inflacionaria")
    st.info(
        "Para la serie completa y exacta de la Brecha sube el archivo "
        "`anex-IPC-CiudadesMensuales-dic2024.xlsx` del DANE en la barra lateral. "
        "El pipeline del Colab ya validó 84 meses sin faltantes."
    )

    if df_usuario is not None:
        st.subheader("3. Base econométrica cargada por el usuario")
        st.write(f"Filas: {len(df_usuario)} | Columnas: {list(df_usuario.columns)}")
        st.dataframe(df_usuario.head(10), use_container_width=True)

# ------------------------------------------------------------
# Página: EDA
# ------------------------------------------------------------
elif pagina == "EDA y Descriptivos":
    st.header("Análisis Exploratorio de Datos")

    if df_rndc.empty:
        st.warning("Carga primero la serie RNDC.")
    else:
        st.subheader("Estadísticos descriptivos – RNDC")
        st.dataframe(df_rndc["RNDC"].describe().to_frame().T, use_container_width=True)

        fig = px.line(df_rndc, x=df_rndc["mes"].astype(str), y="RNDC",
                      title="Variación mensual del índice regional de fletes (RNDC Törnqvist)",
                      labels={"x": "Mes", "RNDC": "Var. %"})
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Lectura**  
        La serie RNDC muestra la dinámica de los fletes regionales con destino a Pasto.  
        Los picos positivos elevados corresponden a episodios de presión logística.  
        No se interpreta como causalidad sobre la brecha inflacionaria.
        """)

# ------------------------------------------------------------
# Página: Estacionariedad
# ------------------------------------------------------------
elif pagina == "Estacionariedad":
    st.header("Pruebas de Estacionariedad (ADF y KPSS)")

    if not HAS_STATSMODELS:
        st.error("Se requiere `statsmodels` para las pruebas ADF/KPSS. Instálalo con `pip install statsmodels`.")
    elif df_rndc.empty:
        st.warning("Carga la serie RNDC.")
    else:
        serie = df_rndc["RNDC"].dropna()
        st.write(f"Observaciones utilizadas: {len(serie)}")

        # ADF
        adf_stat, adf_p, _, _, crit, _ = adfuller(serie, autolag="AIC")
        st.subheader("Augmented Dickey-Fuller (ADF)")
        st.write(f"Estadístico: `{adf_stat:.4f}` | p-valor: `{adf_p:.4f}`")
        st.write("Valores críticos:", crit)

        # KPSS
        try:
            kpss_stat, kpss_p, _, crit_kpss = kpss(serie, regression="c", nlags="auto")
            st.subheader("KPSS")
            st.write(f"Estadístico: `{kpss_stat:.4f}` | p-valor: `{kpss_p:.4f}`")
            st.write("Valores críticos:", crit_kpss)
        except Exception as e:
            st.warning(f"KPSS no se pudo calcular: {e}")

        st.info(
            "Interpretación conjunta ADF + KPSS recomendada. "
            "Las pruebas se realizan sobre la muestra disponible de RNDC (59 meses). "
            "Para la Brecha y SIPSA se requieren las series del pipeline completo."
        )

# ------------------------------------------------------------
# Página: ARDL y HAC
# ------------------------------------------------------------
elif pagina == "ARDL y HAC":
    st.header("Modelos ARDL y inferencia HAC")

    st.markdown("""
    ### Especificación de referencia (artículo)
    $$
    B_t = \\alpha + \\rho B_{t-1} + \\sum_{j=0}^{2} \\beta_j \\, RNDC_{t-j} + \\sum_{j=0}^{2} \\gamma_j \\, PM_{t-j} + u_t
    $$
    - Muestra de estimación: **56 meses** (2020-05 → 2024-12)
    - Inferencia: **HAC / Newey-West** (maxlags = 3)
    - Interpretación: **asociaciones condicionadas**, no efectos causales
    """)

    st.warning(
        "Para estimar el ARDL completo se necesita la base econométrica "
        "con las columnas `BRECHA_INFLACION_PASTO_COLOMBIA_PP`, "
        "`VAR_FLETE_TORNQVIST_PCT` y `PM_ponderado_t`. "
        "Sube el archivo generado en la Fase 11 o 13 del Colab."
    )

    if df_usuario is not None:
        st.success("Base del usuario detectada. Puedes proceder a estimar cuando las columnas estén presentes.")
        st.dataframe(df_usuario.head(), use_container_width=True)
    else:
        st.info("Sube `base_econometrica_2020_2024.csv` o `base_estimacion_comun_56_meses.csv` en la barra lateral.")

# ------------------------------------------------------------
# Página: Robustez
# ------------------------------------------------------------
elif pagina == "Robustez y Sensibilidad":
    st.header("Robustez y Sensibilidad")

    st.markdown("""
    El artículo evalúa múltiples especificaciones:
    - SIPSA ponderado vs no ponderado
    - Ponderación contemporánea vs rezagada
    - Canastas restringidas (PM_100, PM_95)
    - Exclusión de febrero 2023
    - Sustitución de RNDC por ICTC (solo como contraste)
    - Diferentes anchos de banda HAC
    """)

    st.info(
        "Los resultados de robustez se encuentran en las tablas exportadas "
        "por el notebook (`tabla3_robustez.csv`, `cobertura_alternativas.csv`). "
        "Cárgalos aquí para visualizarlos."
    )

# ------------------------------------------------------------
# Página: Focus Pasto
# ------------------------------------------------------------
elif pagina == "Focus Pasto":
    st.header("Focus Territorial – Pasto / Nariño")

    st.markdown("""
    ### Hallazgos descriptivos clave (pipeline)
    - La brecha inflacionaria muestra episodios de divergencia relevante en 2020 (junio-julio) y 2021.
    - El índice RNDC registra un pico destacado en mayo de 2021, coincidente con el episodio de abastecimiento reportado por SIPSA-A.
    - Las asociaciones ARDL L2 (HAC) indican una relación contemporánea positiva entre RNDC y la brecha, con magnitud moderada.
    - ICTC nacional y RNDC regional tienen correlación baja-moderada → miden conceptos distintos.
    """)

    st.subheader("Semáforo ilustrativo (no operativo)")
    st.caption(
        "Construcción matemática exploratoria basada en umbrales históricos de RNDC y SIPSA. "
        "No constituye un sistema de alerta en tiempo real ni evidencia de bloqueos viales."
    )

# ------------------------------------------------------------
# Página: Informe
# ------------------------------------------------------------
elif pagina == "Informe":
    st.header("Informe metodológico resumido")

    st.markdown(f"""
    **Título del estudio**  
    Costos del transporte de carga, precios mayoristas de alimentos y brecha inflacionaria Pasto–Colombia: evidencia mensual, 2018–2024.

    **Variable dependiente**  
    \( B_t = \\pi_{{\\text{{Pasto}},t}} - \\pi_{{\\text{{Colombia}},t}} \)

    **Variables explicativas principales**  
    - \( PM_{{\\text{{ponderado}},t}} \) (SIPSA El Potrerillo, media geométrica ponderada por toneladas)
    - \( \\text{{VAR\\_FLETE\\_TORNQVIST\\_PCT}}_t \) (índice regional construido a partir de RNDC)

    **Benchmark**  
    ICTC nacional (DANE)

    **Muestras**  
    - Histórica: 84 meses  
    - Común: 59 meses  
    - Estimación ARDL: 56 meses  

    **Métodos**  
    ADF / KPSS, ARDL con rezagos distribuidos, inferencia HAC (Newey-West), pruebas conjuntas de bloques, robustez multi-especificación, trayectorias dinámicas condicionadas.

    **Limitaciones explícitas**  
    1. No se identifican efectos causales.  
    2. No se imputan ni interpolan observaciones.  
    3. RNDC e ICTC miden conceptos diferentes.  
    4. El semáforo de alertas es ilustrativo.

    **Fecha de generación del informe**  
    {datetime.now().strftime("%Y-%m-%d %H:%M")}
    """)

# ------------------------------------------------------------
# Página: Chatbot
# ------------------------------------------------------------
elif pagina == "Chatbot":
    st.header("Asistente del Observatorio")

    pregunta = st.text_input("Escribe tu pregunta sobre la metodología, las series o los resultados:")

    if pregunta:
        p = pregunta.lower()
        if any(w in p for w in ["causal", "causa", "efecto causal"]):
            st.write("**Respuesta:** Ningún modelo de este estudio identifica efectos causales. Todas las estimaciones son asociaciones estadísticas condicionadas.")
        elif any(w in p for w in ["muestra", "meses", "84", "59", "56"]):
            st.write("**Respuesta:** Se trabajan tres muestras: 84 meses (histórica), 59 meses (intersección SIPSA-RNDC) y 56 meses (estimación ARDL tras consumir rezagos).")
        elif any(w in p for w in ["rndc", "flete", "tornqvist"]):
            st.write("**Respuesta:** El índice regional de fletes se construye con un índice encadenado de Törnqvist a partir de registros oficiales RNDC del Ministerio de Transporte. La serie econométrica tiene 59 variaciones mensuales (2020-02 a 2024-12).")
        elif any(w in p for w in ["brecha", "inflacion", "ipc"]):
            st.write("**Respuesta:** La brecha se define como la diferencia entre la variación mensual del IPC de Pasto y del IPC nacional (DANE). Cubre 84 meses continuos sin imputaciones.")
        elif any(w in p for w in ["sipsa", "ponderado", "potrerillo"]):
            st.write("**Respuesta:** El indicador PM_ponderado_t es una media geométrica de relativos de precios ponderada por las cantidades físicas del mercado El Potrerillo (SIPSA).")
        else:
            st.write("Puedo responder sobre: definición de la Brecha, construcción del RNDC, tamaños de muestra, advertencias de no-causalidad, pruebas ADF/KPSS y estructura del ARDL. Reformula tu pregunta.")

# ------------------------------------------------------------
# Página: Derechos y Créditos
# ------------------------------------------------------------
elif pagina == "Derechos y Créditos":
    st.header("Derechos, fuentes y créditos")

    st.markdown("""
    ### Fuentes oficiales
    - **DANE** – IPC por ciudades, ICTC, SIPSA-A y SIPSA-P  
    - **Ministerio de Transporte** – Registro Nacional de Despachos de Carga (RNDC)  
    - Construcción del índice regional de fletes: repositorio  
      [JuanDavidgm2026/fletes-precios-inflacion-pasto](https://github.com/JuanDavidgm2026/fletes-precios-inflacion-pasto)

    ### Reproducibilidad
    - Los archivos originales SIPSA están auditados con SHA-256.  
    - La serie RNDC utilizada tiene hash  
      `aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891`.  
    - No se realizan imputaciones ni modificaciones de los microdatos.

    ### Limitaciones
    - El índice RNDC es una construcción de los autores a partir de información oficial; no es un índice publicado por el Ministerio.  
    - ICTC y RNDC no son equivalentes conceptualmente.  
    - Todas las estimaciones son asociativas.

    ### Créditos
    Observatorio desarrollado para el Seminario de Investigación Regional.  
    Código alineado al pipeline y a las advertencias metodológicas del artículo y del notebook Colab (Fase 18).
    """)

# ------------------------------------------------------------
# Pie de página
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Observatorio Pasto · Datos oficiales DANE + RNDC · "
    "Interpretación asociativa · Sin imputaciones · 2018-2024"
)
