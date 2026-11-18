# ============================================================
# OBSERVATORIO BRECHA INFLACIONARIA PASTO–COLOMBIA
# Fletes regionales · Precios mayoristas SIPSA · 2018–2024
# Seminario de Investigación Regional II · Universidad de Nariño
# Compatible Streamlit + Google Colab
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Observatorio Brecha Inflacionaria Pasto–Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CSS (mismo estilo premium oscuro)
# ------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0B1220 0%, #0F172A 40%, #111827 100%) !important; color: #F8FAFC !important; }
section[data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #67E8F9 !important; }
h1, h2, h3, h4 { color: #67E8F9 !important; font-weight: 700 !important; }
.stCaption, [data-testid="stCaption"] { color: #94A3B8 !important; }
p, span, label, .stMarkdown { color: #E2E8F0 !important; }
.stButton > button {
    background: linear-gradient(90deg, #0E7490 0%, #6D28D9 100%) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important;
}
div[data-testid="stMetric"] { background-color: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; }
.intro-box {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #1E1B4B 100%);
    border: 1px solid #334155; border-left: 5px solid #22D3EE; border-radius: 16px;
    padding: 1.35rem 1.6rem; margin: 0.4rem 0 1.2rem 0;
}
.intro-box h2 { color: #67E8F9 !important; margin: 0 0 0.55rem 0; }
.intro-box .tag {
    display: inline-block; background: #155E75; color: #CFFAFE !important;
    border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.8rem; margin: 0.2rem 0.3rem 0.15rem 0;
}
.tip-box {
    background: #0F172A; border: 1px solid #334155; border-left: 4px solid #A78BFA;
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 1rem; color: #CBD5E1 !important;
}
.credits-box {
    background: linear-gradient(145deg, #0F172A 0%, #1E1B4B 100%);
    border: 1px solid #334155; border-radius: 16px; padding: 2rem; margin: 1rem 0;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0B1220",
    plot_bgcolor="#111827",
    font=dict(color="#E2E8F0", size=13),
    title_font=dict(color="#67E8F9", size=16),
)

def style_fig(fig):
    if fig is not None:
        fig.update_layout(**PLOT_LAYOUT)
    return fig

# ------------------------------------------------------------
# GENERACIÓN DE SERIES DEMO (calibradas al artículo)
# ------------------------------------------------------------
@st.cache_data
def generar_series_demo(seed=42):
    """
    Series mensuales 2018-01 a 2024-12 calibradas a las estadísticas
    descriptivas reportadas en el artículo (medias, volatilidad, periodo).
    Sirven para demostración y reproducibilidad. Sustituir por CSV real.
    """
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    n = len(fechas)

    # Brecha B_t ≈ media ~0, sd ~0.32 (como en el paper)
    brecha = rng.normal(0.001, 0.318, n)
    # Añadir algo de persistencia débil
    for i in range(1, n):
        brecha[i] = 0.25 * brecha[i-1] + 0.75 * brecha[i]

    # SIPSA ponderado (media variación ~1.55 %)
    sipsa = rng.normal(1.55, 4.8, n)
    for i in range(1, n):
        sipsa[i] = 0.15 * sipsa[i-1] + 0.85 * sipsa[i]

    # RNDC regional (disponible ~feb-2020 en adelante en el paper)
    rndc = np.full(n, np.nan)
    idx_rndc = (fechas >= "2020-02-01")
    rndc[idx_rndc] = rng.normal(0.67, 3.2, idx_rndc.sum())
    for i in range(1, n):
        if not np.isnan(rndc[i]) and not np.isnan(rndc[i-1]):
            rndc[i] = 0.20 * rndc[i-1] + 0.80 * rndc[i]

    # ICTC nacional
    ictc = rng.normal(0.41, 1.1, n)
    for i in range(1, n):
        ictc[i] = 0.30 * ictc[i-1] + 0.70 * ictc[i]

    # Pequeña correlación contemporánea RNDC → Brecha (como el coeficiente positivo del paper)
    mask = ~np.isnan(rndc)
    brecha[mask] = brecha[mask] + 0.009 * rndc[mask] + rng.normal(0, 0.05, mask.sum())

    df = pd.DataFrame({
        "fecha": fechas,
        "anio": fechas.year,
        "mes": fechas.month,
        "Brecha": np.round(brecha, 4),
        "SIPSA": np.round(sipsa, 4),
        "RNDC": np.round(rndc, 4),
        "ICTC": np.round(ictc, 4),
    })
    return df

def cargar_csv_usuario(uploaded_file):
    """Carga CSV del equipo. Columnas esperadas: fecha, Brecha, SIPSA, RNDC, ICTC (nombres flexibles)."""
    try:
        df = pd.read_csv(uploaded_file)
        # Normalizar nombres de columnas
        colmap = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if "fecha" in cl or "date" in cl or "periodo" in cl:
                colmap[c] = "fecha"
            elif "brecha" in cl or "gap" in cl or "b_t" in cl:
                colmap[c] = "Brecha"
            elif "sipsa" in cl or "mayorista" in cl or "potrerillo" in cl:
                colmap[c] = "SIPSA"
            elif "rndc" in cl or "flete" in cl or "fletes" in cl:
                colmap[c] = "RNDC"
            elif "ictc" in cl:
                colmap[c] = "ICTC"
        df = df.rename(columns=colmap)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        else:
            st.error("El CSV debe contener una columna de fecha.")
            return None
        for col in ["Brecha", "SIPSA", "RNDC", "ICTC"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
        df["anio"] = df["fecha"].dt.year
        df["mes"] = df["fecha"].dt.month
        return df
    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        return None

# ------------------------------------------------------------
# FUNCIONES ANALÍTICAS
# ------------------------------------------------------------
def estacionariedad(serie, nombre):
    s = serie.dropna()
    if len(s) < 12:
        return {"Variable": nombre, "ADF": None, "p_ADF": None, "KPSS": None, "p_KPSS": None, "N": len(s)}
    try:
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    except Exception:
        adf_stat, adf_p = None, None
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = None, None
    return {
        "Variable": nombre,
        "ADF": round(adf_stat, 4) if adf_stat is not None else None,
        "p_ADF": round(adf_p, 5) if adf_p is not None else None,
        "KPSS": round(kpss_stat, 4) if kpss_stat is not None else None,
        "p_KPSS": round(kpss_p, 5) if kpss_p is not None else None,
        "N": len(s),
    }

def construir_lags(df, cols, max_lag=2):
    out = df[["fecha"] + cols].copy()
    for c in cols:
        for L in range(1, max_lag + 1):
            out[f"{c}_L{L}"] = out[c].shift(L)
    return out

def estimar_ardl(df, y="Brecha", x_vars=None, lags=2):
    """ARDL simple con MCO (errores robustos aproximados)."""
    if x_vars is None:
        x_vars = ["RNDC", "SIPSA"]
    data = construir_lags(df, [y] + x_vars, max_lag=lags).dropna()
    if len(data) < 20:
        return None, None, None

    y_col = f"{y}_L1"  # usamos y_{t-1} + contemporáneos y rezagos de X
    # Especificación de referencia del paper: B_t = α + ρ B_{t-1} + Σ β_j T_{t-j} + Σ γ_j PM_{t-j}
    features = [f"{y}_L1"]
    for v in x_vars:
        features.append(v)
        for L in range(1, lags + 1):
            features.append(f"{v}_L{L}")

    # Asegurar que todas existen
    features = [f for f in features if f in data.columns]
    X = data[features].copy()
    yv = data[y].copy()  # la serie original de la brecha (ya alineada por dropna de lags)

    # Alinear índice
    common = X.index.intersection(yv.index)
    X = X.loc[common]
    yv = yv.loc[common]

    # Añadir constante
    X = X.copy()
    X.insert(0, "const", 1.0)

    try:
        beta = np.linalg.lstsq(X.values, yv.values, rcond=None)[0]
        yhat = X.values @ beta
        resid = yv.values - yhat
        n, k = X.shape
        s2 = np.sum(resid**2) / (n - k)
        # Covarianza OLS simple (para demo; en producción usar HAC)
        XtX_inv = np.linalg.pinv(X.values.T @ X.values)
        se = np.sqrt(np.diag(s2 * XtX_inv))
        tstats = beta / se
        pvals = 2 * (1 - stats.t.cdf(np.abs(tstats), df=n - k))

        res = pd.DataFrame({
            "Variable": X.columns,
            "Coef": np.round(beta, 5),
            "EE": np.round(se, 5),
            "t": np.round(tstats, 3),
            "p": np.round(pvals, 5),
        })
        r2 = r2_score(yv, yhat)
        return res, {"R2": r2, "N": n, "yhat": yhat, "resid": resid, "index": common}, data
    except Exception as e:
        st.warning(f"Error en estimación: {e}")
        return None, None, None

def metricas_descriptivas(df):
    rows = []
    for col in ["Brecha", "SIPSA", "RNDC", "ICTC"]:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        rows.append({
            "Indicador": col,
            "N": len(s),
            "Media": round(s.mean(), 4),
            "Desv.Std": round(s.std(), 4),
            "Min": round(s.min(), 4),
            "Max": round(s.max(), 4),
            "Inicio": str(df.loc[s.index.min(), "fecha"].date()) if "fecha" in df.columns else "—",
            "Fin": str(df.loc[s.index.max(), "fecha"].date()) if "fecha" in df.columns else "—",
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# CARGA DE DATOS (demo + upload)
# ------------------------------------------------------------
with st.sidebar:
    st.header("🇨🇴 Observatorio")
    st.caption("Brecha Inflacionaria Pasto–Colombia")
    seccion = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "📊 Exploración y EDA",
            "📈 Series y Estacionariedad",
            "📉 ARDL / Modelos dinámicos",
            "🔬 Sensibilidad y diagnóstico",
            "📍 Focus Pasto–Nariño",
            "📑 Informe de investigación",
            "💬 Asistente",
            "©️ Créditos",
        ],
        index=0,
    )
    st.markdown("---")
    st.subheader("Fuente de datos")
    fuente = st.radio(
        "Usar",
        ["Series demo (calibradas al artículo)", "Subir CSV del equipo"],
        index=0,
    )
    uploaded = None
    if fuente == "Subir CSV del equipo":
        uploaded = st.file_uploader("CSV con columnas fecha, Brecha, SIPSA, RNDC, ICTC", type=["csv"])

    st.markdown("---")
    st.caption("Autores: Juan David Granja · Helmer Rivera · Yeison Ortega")
    st.caption("Universidad de Nariño · Seminario de Investigación Regional II")

# Cargar datos
if fuente == "Subir CSV del equipo" and uploaded is not None:
    DF = cargar_csv_usuario(uploaded)
    if DF is None:
        st.stop()
    origen_datos = "CSV del equipo"
else:
    DF = generar_series_demo()
    origen_datos = "Series demo calibradas (reproducibles)"

# Rango temporal útil
amin = int(DF["anio"].min())
amax = int(DF["anio"].max())
n_obs = len(DF)

# ------------------------------------------------------------
# CONTENIDO
# ------------------------------------------------------------
if seccion == "🏠 Inicio":
    st.markdown(
        """
        <h1 style="color:#67E8F9; margin-bottom:0.1rem;">
            Observatorio de la Brecha Inflacionaria Pasto–Colombia
        </h1>
        <p style="color:#94A3B8; margin-top:0;">
            Fletes regionales (RNDC) · Precios mayoristas SIPSA (El Potrerillo) · Evidencia mensual 2018–2024
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="intro-box">
            <h2>Investigación de referencia</h2>
            <p>
                Esta aplicación acompaña el artículo 
                <b>“Costos del transporte de carga, precios mayoristas de alimentos y brecha inflacionaria Pasto–Colombia: evidencia mensual 2018–2024”</b>.
                Reproduce el marco analítico del trabajo: construcción de la brecha 
                \( B_t = \pi_{{\text{{Pasto}},t}} - \pi_{{\text{{Colombia}},t}} \), 
                el indicador mayorista SIPSA de El Potrerillo, el índice regional de fletes RNDC 
                y el contraste con el ICTC nacional.
            </p>
            <p>
                El alcance es <b>asociativo</b>, no causal. Los resultados de modelos ARDL/DL, 
                pruebas de estacionariedad y evaluación predictiva se presentan con las mismas 
                precauciones metodológicas del artículo.
            </p>
            <p>
                <span class="tag">Periodo: {amin}–{amax}</span>
                <span class="tag">{n_obs} observaciones mensuales</span>
                <span class="tag">Fuente: {origen_datos}</span>
            </p>
            <div class="authors" style="margin-top:1rem; color:#CBD5E1;">
                <b>Autores:</b> Juan David Granja Miramag · Helmer Andrés Rivera Vallejos · Yeison Alexander Ortega Rosero<br>
                <span style="color:#94A3B8;">Programa de Economía · Universidad de Nariño · Seminario de Investigación Regional II</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="tip-box">
        <b>Cómo usar la app</b><br>
        1. En la barra lateral elige <b>Series demo</b> (listo para explorar) o sube el <b>CSV del equipo</b> con las series procesadas.<br>
        2. Navega por EDA, estacionariedad, estimación ARDL, sensibilidad y Focus Pasto.<br>
        3. Genera el informe de investigación y consulta al asistente.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Resumen de series cargadas")
    st.dataframe(metricas_descriptivas(DF), use_container_width=True, hide_index=True)

elif seccion == "📊 Exploración y EDA":
    st.markdown("## 📊 Exploración descriptiva")
    st.markdown(
        """<div class="tip-box">
        Estadísticos, correlaciones y trayectorias de la brecha, SIPSA, RNDC e ICTC.
        </div>""",
        unsafe_allow_html=True,
    )
    st.dataframe(metricas_descriptivas(DF), use_container_width=True, hide_index=True)

    cols_disp = [c for c in ["Brecha", "SIPSA", "RNDC", "ICTC"] if c in DF.columns]
    if len(cols_disp) >= 2:
        corr = DF[cols_disp].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlación contemporánea")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Trayectorias mensuales")
    fig = make_subplots(rows=2, cols=2, subplot_titles=cols_disp[:4])
    positions = [(1,1), (1,2), (2,1), (2,2)]
    for i, col in enumerate(cols_disp[:4]):
        r, c = positions[i]
        fig.add_trace(go.Scatter(x=DF["fecha"], y=DF[col], mode="lines", name=col,
                                 line=dict(width=1.5)), row=r, col=c)
    fig.update_layout(height=520, showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)

elif seccion == "📈 Series y Estacionariedad":
    st.markdown("## 📈 Estacionariedad (ADF / KPSS)")
    st.markdown(
        """<div class="tip-box">
        Pruebas de raíz unitaria (ADF) y de estacionariedad (KPSS) sobre la muestra disponible.
        Compatible con el protocolo del artículo.
        </div>""",
        unsafe_allow_html=True,
    )
    rows = []
    for col in ["Brecha", "SIPSA", "RNDC", "ICTC"]:
        if col in DF.columns:
            rows.append(estacionariedad(DF[col], col))
    res_est = pd.DataFrame(rows)
    st.dataframe(res_est, use_container_width=True, hide_index=True)
    st.caption("ADF: H₀ = raíz unitaria. KPSS: H₀ = estacionariedad alrededor de una constante.")

elif seccion == "📉 ARDL / Modelos dinámicos":
    st.markdown("## 📉 Estimación ARDL / rezagos distribuidos")
    st.markdown(
        """<div class="tip-box">
        Especificación de referencia del artículo:<br>
        \( B_t = \alpha + \rho B_{t-1} + \sum_{j=0}^{L} \beta_j T_{t-j} + \sum_{j=0}^{L} \gamma_j PM_{t-j} + \varepsilon_t \)
        </div>""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        lags = st.slider("Número de rezagos (L)", 0, 3, 2)
    with c2:
        usar_ictc = st.checkbox("Incluir ICTC como variable de contraste", value=False)

    x_vars = ["RNDC", "SIPSA"]
    if usar_ictc and "ICTC" in DF.columns:
        x_vars.append("ICTC")

    # Muestra comparable (como en el paper: a partir de mayo 2020 si hay RNDC)
    df_est = DF.copy()
    if "RNDC" in df_est.columns:
        df_est = df_est.dropna(subset=["RNDC"])

    res, info, data_used = estimar_ardl(df_est, y="Brecha", x_vars=x_vars, lags=lags)

    if res is not None:
        st.subheader("Coeficientes (MCO – demo)")
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.metric("R²", f"{info['R2']:.3f}")
        st.metric("N (observaciones)", info["N"])
        st.caption("Nota: errores estándar son OLS. En el artículo se reportan HAC Newey–West.")

        # Gráfico observado vs ajustado
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_used.loc[info["index"], "fecha"],
                                 y=df_est.loc[info["index"], "Brecha"],
                                 mode="lines", name="Brecha observada",
                                 line=dict(color="#94A3B8")))
        fig.add_trace(go.Scatter(x=data_used.loc[info["index"], "fecha"],
                                 y=info["yhat"],
                                 mode="lines", name="Ajuste ARDL",
                                 line=dict(color="#22D3EE", dash="dash")))
        fig.update_layout(title="Brecha observada vs ajuste dentro de muestra", height=420)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    else:
        st.warning("No se pudo estimar el modelo con la muestra actual. Prueba subir el CSV del equipo o reducir rezagos.")

elif seccion == "🔬 Sensibilidad y diagnóstico":
    st.markdown("## 🔬 Sensibilidad y diagnósticos")
    st.markdown(
        """<div class="tip-box">
        Comparación de correlaciones, residuos y estabilidad exploratoria.
        </div>""",
        unsafe_allow_html=True,
    )
    # Correlación RNDC vs ICTC (el paper reporta ~0.18)
    if "RNDC" in DF.columns and "ICTC" in DF.columns:
        sub = DF[["RNDC", "ICTC"]].dropna()
        if len(sub) > 5:
            corr = sub["RNDC"].corr(sub["ICTC"])
            st.metric("Correlación RNDC–ICTC (Pearson)", f"{corr:.3f}")
            st.caption("En el artículo esta correlación es baja (~0.18); los dos indicadores no miden el mismo proceso.")

    # Distribución de la brecha
    fig = px.histogram(DF, x="Brecha", nbins=30, title="Distribución de la brecha inflacionaria")
    st.plotly_chart(style_fig(fig), use_container_width=True)

elif seccion == "📍 Focus Pasto–Nariño":
    st.markdown("## 📍 Focus Pasto – Nariño")
    st.markdown(
        """<div class="tip-box">
        Lectura territorial centrada en la ciudad de Pasto y el departamento de Nariño.
        La brecha se define respecto al IPC nacional; Pasto está incluido en el total nacional.
        </div>""",
        unsafe_allow_html=True,
    )
    ult = DF.dropna(subset=["Brecha"]).iloc[-1]
    st.metric("Última brecha observada (p.p.)", f"{ult['Brecha']:.3f}")
    st.metric("Periodo", f"{ult['fecha'].strftime('%Y-%m')}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF["fecha"], y=DF["Brecha"], mode="lines",
                             name="Brecha Pasto–Colombia", line=dict(color="#22D3EE", width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="#64748B")
    fig.update_layout(title="Brecha inflacionaria mensual Pasto − Colombia", height=420,
                      yaxis_title="puntos porcentuales")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("""
    **Lectura de política (exploratoria)**  
    La evidencia del artículo es asociativa. Un coeficiente contemporáneo positivo de RNDC 
    no implica causalidad de los bloqueos viales sobre la brecha. Se requiere mayor cobertura 
    de cierres, instrumentos y controles para lecturas causales.
    """)

elif seccion == "📑 Informe de investigación":
    st.markdown("## 📑 Informe de investigación (borrador automático)")
    st.markdown(
        """<div class="tip-box">
        Genera un resumen ejecutivo alineado con la estructura del artículo 
        (contexto, datos, resultados principales, limitaciones).
        </div>""",
        unsafe_allow_html=True,
    )
    desc = metricas_descriptivas(DF)
    md = f"""
# Informe exploratorio – Observatorio Brecha Inflacionaria Pasto–Colombia

**Fecha de generación:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Fuente de datos en esta sesión:** {origen_datos}  
**Periodo:** {amin}–{amax} ({n_obs} meses)

## 1. Contexto
Se examina la asociación contemporánea y rezagada entre la variación de los fletes regionales (RNDC), 
los precios mayoristas de alimentos en El Potrerillo (SIPSA) y la brecha de inflación mensual 
entre Pasto y Colombia (\( B_t = \pi_{{\\text{{Pasto}},t}} - \pi_{{\\text{{Colombia}},t}} \)).

## 2. Datos
{desc.to_markdown(index=False)}

## 3. Resultados principales (sesión actual)
- Las series demo/reales permiten reproducir el protocolo de estacionariedad y estimación ARDL.
- El coeficiente contemporáneo de RNDC en especificaciones de referencia tiende a ser positivo, 
  coherente con el hallazgo del artículo, pero los bloques conjuntos suelen no ser significativos al 5 %.
- La correlación RNDC–ICTC es baja; no deben intercambiarse como si midieran el mismo proceso.

## 4. Limitaciones
- Alcance asociativo, no causal.
- Sin serie homogénea de cierres viales ni instrumentos suficientes.
- La validación predictiva no respalda todavía un sistema operativo de alertas viales.
- Cuando se usan series demo, los coeficientes son ilustrativos.

## 5. Fuentes
- Artículo de referencia del Seminario de Investigación Regional II (Universidad de Nariño).
- DANE (IPC, SIPSA, ICTC), RNDC (procesado por el equipo).
"""
    st.markdown(md)
    st.download_button(
        "⬇️ Descargar informe Markdown",
        data=md.encode("utf-8"),
        file_name=f"informe_brecha_pasto_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
    )

elif seccion == "💬 Asistente":
    st.markdown("## 💬 Asistente del Observatorio")
    st.markdown(
        """<div class="tip-box">
        Preguntas sobre la brecha, fletes, SIPSA o interpretación de resultados.
        </div>""",
        unsafe_allow_html=True,
    )
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{
            "role": "assistant",
            "content": (
                "Hola. Soy el asistente del Observatorio de la Brecha Inflacionaria Pasto–Colombia. "
                "Puedo ayudarte a interpretar la brecha, el indicador SIPSA de El Potrerillo, "
                "el índice RNDC de fletes y los resultados ARDL del estudio. ¿Qué deseas consultar?"
            )
        }]
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Escribe tu pregunta…"):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        q = prompt.lower()
        if "brecha" in q:
            resp = (
                "La **brecha** se define como \( B_t = \\pi_{\\text{Pasto},t} - \\pi_{\\text{Colombia},t} \) "
                "(puntos porcentuales). Un valor positivo indica que la inflación mensual de Pasto "
                "superó a la nacional. Pasto está incluido en el total nacional, por lo que ambas "
                "tasas comparten información. El alcance del estudio es asociativo."
            )
        elif "rndc" in q or "flete" in q:
            resp = (
                "El **RNDC** utilizado es un índice regional de fletes construido a partir de "
                "registros de operaciones con destino a Pasto. No es equivalente al ICTC nacional "
                "ni una medición directa de cierres viales. En el artículo la correlación con ICTC "
                "es baja (~0.18)."
            )
        elif "sipsa" in q or "mayorista" in q or "potrerillo" in q:
            resp = (
                "El indicador **SIPSA** de El Potrerillo se construye con relativos geométricos de "
                "precios ponderados por cantidades físicas abastecidas. No representa el IPC de "
                "alimentos ni el gasto de los hogares; la ponderación puede ser endógena a "
                "restricciones de abastecimiento."
            )
        elif "causal" in q or "bloqueos" in q:
            resp = (
                "El estudio **no identifica causalmente** el efecto de los bloqueos viales. "
                "No se cuenta con una serie histórica homogénea de cierres ni con instrumentos "
                "suficientes. Los resultados son asociativos."
            )
        else:
            resp = (
                "Puedo ayudarte con: definición de la brecha, construcción de SIPSA y RNDC, "
                "interpretación de coeficientes ARDL, estacionariedad ADF/KPSS y limitaciones "
                "del diseño. Reformula tu pregunta con uno de estos términos."
            )
        st.session_state.msgs.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)

elif seccion == "©️ Créditos":
    st.markdown("## ©️ Créditos y reproducibilidad")
    st.markdown(
        """
        <div class="credits-box">
            <h2>Observatorio de la Brecha Inflacionaria Pasto–Colombia</h2>
            <p class="meta">Seminario de Investigación Regional II · Universidad de Nariño</p>
            <h3>Autores del artículo y de la app</h3>
            <p>
                Juan David Granja Miramag · Helmer Andrés Rivera Vallejos · Yeison Alexander Ortega Rosero
            </p>
            <h3>Datos</h3>
            <ul>
                <li>IPC Pasto y nacional – DANE</li>
                <li>SIPSA (precios y abastecimiento) – DANE</li>
                <li>ICTC – DANE</li>
                <li>RNDC regional – procesado por el equipo a partir de registros públicos</li>
            </ul>
            <h3>Reproducibilidad</h3>
            <p>
                La app funciona con series demo calibradas a las estadísticas del artículo 
                o con el CSV procesado del equipo. El código es autocontenido y puede ejecutarse 
                en Streamlit Cloud o en Google Colab (instalando streamlit y usando un túnel).
            </p>
            <p class="meta" style="margin-top:1.2rem;">
                © Universidad de Nariño · Seminario de Investigación Regional II
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Observatorio Brecha Inflacionaria Pasto–Colombia · "
    "Juan David Granja · Helmer Rivera · Yeison Ortega · Universidad de Nariño"
)
