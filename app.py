from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_breuschpagan,
    linear_reset,
)
from statsmodels.stats.outliers_influence import (
    OLSInfluence,
    variance_inflation_factor,
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


st.set_page_config(
    page_title="Observatorio Económico Pasto–Nariño",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(150deg,#0b1425,#112039 65%,#141c32);
    color: #edf4fa;
}
section[data-testid="stSidebar"] {
    background: #101b2f;
    border-right: 1px solid #2b405d;
}
h1,h2,h3 {
    color: #95e9f0!important;
    letter-spacing: -.02em;
}
[data-testid="stMetric"] {
    background: #1b2b43;
    border: 1px solid #334b66;
    border-radius: 14px;
    padding: 14px;
}
.stButton button[kind="primary"] {
    background: linear-gradient(95deg,#087f9b,#7053c9);
    border: 0;
}
.hero {
    border: 1px solid #355c75;
    border-radius: 17px;
    padding: 23px 27px;
    background: linear-gradient(110deg,#123d54,#2c244e);
    margin-bottom: 17px;
}
.hero h2 {
    color: #b8f6f2!important;
    margin: 0 0 8px;
}
.hero p {
    font-size: 1.07rem;
    line-height: 1.65;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


REPO = "JuanDavidgm2026/fletes-precios-inflacion-pasto"
RESPALDO = "JuanDavidgm2026/SIPSA-PASTO"
TAG = "sipsa-dane-originales-v1"

Y = "BRECHA_INFLACION_PASTO_COLOMBIA_PP"
T = "VAR_FLETE_TORNQVIST_PCT"
P = "PM_ponderado_t"
I = "ICTC_MENSUAL_PCT"

M84 = pd.period_range("2018-01", "2024-12", freq="M")
M59 = pd.period_range("2020-02", "2024-12", freq="M")
M56 = pd.period_range("2020-05", "2024-12", freq="M")

RAW = f"https://raw.githubusercontent.com/{REPO}/main/data"

SIPSA_ARCHIVO = (
    "SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"
)

URLS = {
    "SIPSA": [
        f"{RAW}/{SIPSA_ARCHIVO}",
    ],
    "RNDC": [
        f"{RAW}/variable2_flete_regional_modelo_2020_2024.csv",
    ],
    "IPC": [
        "https://www.dane.gov.co/files/operaciones/IPC/"
        "dic2024/anex-IPC-CiudadesMensuales-dic2024.xlsx",
        f"https://github.com/{RESPALDO}/releases/download/"
        f"{TAG}/anex-IPC-CiudadesMensuales-dic2024.xlsx",
    ],
    "ICTC": [
        "https://www.dane.gov.co/files/operaciones/ICTC/"
        "anex-ICTC-HistVariaContrib-dic2024.xlsx",
        f"https://github.com/{RESPALDO}/releases/download/"
        f"{TAG}/anex-ICTC-HistVariaContrib-dic2024.xlsx",
    ],
}

HASHES = {
    "SIPSA": (
        "227448e82618c3f9dee2f7e72d922cfea5811878ebc02ddb90998c5d10913676"
    ),
    "RNDC": (
        "aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891"
    ),
    "IPC": (
        "3e584414f3f89c78f99ce92576a0715d3d51fac5d6dcd75addf84065e4cd4076"
    ),
    "ICTC": (
        "6fe232548e47dbf9801b75c620209f579ebd0faa2a46fc30a9a305453878205a"
    ),
}

NAMES = {
    Y: "Brecha Pasto–Colombia (p. p.)",
    T: "Fletes RNDC Törnqvist (%)",
    P: "SIPSA ponderado El Potrerillo (%)",
    I: "ICTC nacional (%)",
    "IPC_MENSUAL_PASTO_PCT": "IPC mensual Pasto (%)",
    "IPC_MENSUAL_NACIONAL_PCT": "IPC mensual Colombia (%)",
    "PM_no_ponderado_t": "SIPSA no ponderado (%)",
    "PM_ponderado_lagQ_t": "SIPSA con cantidades rezagadas (%)",
    "PM_100": "SIPSA canasta continua 100 % (%)",
    "PM_95": "SIPSA canasta continua 95 % (%)",
    "kg_total": "Abastecimiento total (kg)",
    "kg_usados_indice": "Abastecimiento cubierto (kg)",
    "cobertura_indice_pct": "Cobertura física SIPSA (%)",
    "n_productos_indice": "Productos comparables SIPSA",
    "n_productos_lagQ": "Productos con pesos rezagados",
    "VAR_FLETE_FISHER_PCT": "RNDC Fisher (%)",
    "VAR_FLETE_CRUDO_PCT": "RNDC sin depuración (%)",
    "VAR_FLETE_SIN_2000_PCT": "RNDC sin registro $2.000 (%)",
    "INDICE_FLETE_TORNQVIST": "Índice Törnqvist",
    "INDICE_FLETE_FISHER": "Índice Fisher",
    "CELDAS_COMPARABLES": "Celdas RNDC comparables",
    "COB_VALOR_T_PCT": "Cobertura RNDC por valor (%)",
    "COB_VIAJES_T_PCT": "Cobertura RNDC por viajes (%)",
    "COB_KG_T_PCT": "Cobertura RNDC por peso (%)",
    "Ajustado": "Brecha ajustada ARDL (p. p.)",
}

PRINCIPALES = [Y, T, P, I]


def nombre(variable):
    return NAMES.get(variable, variable)


def sha256(datos):
    return hashlib.sha256(datos).hexdigest()


def normalizar_texto(valor):
    if pd.isna(valor):
        return ""

    texto = unicodedata.normalize(
        "NFKD",
        str(valor).strip().lower(),
    )

    return " ".join(
        "".join(
            caracter
            for caracter in texto
            if not unicodedata.combining(caracter)
        ).split()
    )


def convertir_numero(valor):
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (int, float, np.number)):
        return float(valor)

    return float(
        pd.to_numeric(
            str(valor).strip().replace("%", "").replace(",", "."),
            errors="coerce",
        )
    )


def validar_calendario(datos, meses, variables):
    if datos["mes"].tolist() != list(meses):
        raise ValueError(
            f"Calendario inválido: {meses[0]} a {meses[-1]}"
        )

    if datos["mes"].duplicated().any():
        raise ValueError("Se encontraron meses duplicados")

    valores = datos[variables].to_numpy(dtype=float)

    if not np.isfinite(valores).all():
        raise ValueError(
            "Existen valores faltantes o no finitos"
        )

    return datos


def procesar_sipsa(contenido):
    datos = pd.read_csv(
        io.BytesIO(contenido),
        encoding="utf-8-sig",
    )

    obligatorias = [
        "mes",
        P,
        "PM_no_ponderado_t",
        "PM_ponderado_lagQ_t",
        "cobertura_indice_pct",
        "n_productos_indice",
    ]

    faltantes = [
        columna
        for columna in obligatorias
        if columna not in datos
    ]

    if faltantes:
        raise ValueError(
            f"Columnas SIPSA ausentes: {faltantes}"
        )

    datos["mes"] = pd.PeriodIndex(
        datos["mes"].astype(str),
        freq="M",
    )

    for columna in NAMES:
        if columna in datos and columna != "mes":
            datos[columna] = pd.to_numeric(
                datos[columna],
                errors="coerce",
            )

    datos = datos.sort_values(
        "mes"
    ).reset_index(drop=True)

    validar_calendario(
        datos,
        M84,
        [
            P,
            "PM_no_ponderado_t",
            "cobertura_indice_pct",
            "n_productos_indice",
        ],
    )

    if not datos["cobertura_indice_pct"].between(
        0, 100
    ).all():
        raise ValueError(
            "Cobertura SIPSA fuera de rango"
        )

    return datos[
        ["mes"] + [
            c for c in NAMES if c in datos
        ]
    ]


def procesar_rndc(contenido):
    datos = pd.read_csv(
        io.BytesIO(contenido),
        encoding="utf-8-sig",
        dtype={"PERIODO": "string"},
    )

    obligatorias = [
        "FECHA",
        "PERIODO",
        T,
        "VAR_FLETE_FISHER_PCT",
        "CELDAS_COMPARABLES",
    ]

    if any(c not in datos for c in obligatorias):
        raise ValueError(
            "El CSV RNDC no contiene su estructura auditada"
        )

    datos["mes"] = pd.to_datetime(
        datos["FECHA"],
        errors="raise",
    ).dt.to_period("M")

    if not (
        datos["mes"].dt.strftime("%Y%m")
        == datos["PERIODO"].str.strip()
    ).all():
        raise ValueError(
            "FECHA y PERIODO del RNDC no coinciden"
        )

    for c in NAMES:
        if c in datos:
            datos[c] = pd.to_numeric(
                datos[c],
                errors="coerce",
            )

    datos = datos.sort_values(
        "mes"
    ).reset_index(drop=True)

    validar_calendario(
        datos,
        M59,
        [
            T,
            "VAR_FLETE_FISHER_PCT",
            "CELDAS_COMPARABLES",
        ],
    )

    return datos[
        ["mes"] + [
            c for c in NAMES if c in datos
        ]
    ]


def procesar_ipc(contenido):
    hoja = pd.read_excel(
        io.BytesIO(contenido),
        sheet_name="VarCiudades",
        header=None,
    )

    bloques = []

    for fila, valor in enumerate(hoja.iloc[:, 0]):
        encontrado = re.search(
            r"variaciones mensuales\s+(20\d{2})",
            normalizar_texto(valor),
        )

        if encontrado:
            bloques.append(
                (int(encontrado.group(1)), fila)
            )

    if [x[0] for x in bloques] != list(range(2017, 2025)):
        raise ValueError(
            "Estructura histórica IPC diferente del Colab"
        )

    meses = (
        "enero febrero marzo abril mayo junio julio agosto "
        "septiembre octubre noviembre diciembre"
    ).split()

    registros = []

    for indice, (anio, inicio) in enumerate(bloques):
        encabezados = [
            normalizar_texto(hoja.iat[inicio + 1, j])
            for j in range(1, 13)
        ]

        if encabezados != meses:
            raise ValueError(
                f"Encabezados IPC incorrectos en {anio}"
            )

        fin = (
            bloques[indice + 1][1]
            if indice + 1 < len(bloques)
            else len(hoja)
        )

        def localizar(nombre_fila):
            return [
                j for j in range(inicio + 2, fin)
                if normalizar_texto(
                    hoja.iat[j, 0]
                ) == nombre_fila
            ]

        pasto = localizar("pasto")
        nacional = localizar("total ipc")

        if len(pasto) != 1 or len(nacional) != 1:
            raise ValueError(
                f"IPC Pasto/Nacional no identificados en {anio}"
            )

        for mes in range(1, 13):
            if anio < 2018:
                continue

            vp = convertir_numero(
                hoja.iat[pasto[0], mes]
            )

            vn = convertir_numero(
                hoja.iat[nacional[0], mes]
            )

            registros.append({
                "mes": pd.Period(
                    f"{anio}-{mes:02d}",
                    freq="M",
                ),
                "IPC_MENSUAL_PASTO_PCT": vp,
                "IPC_MENSUAL_NACIONAL_PCT": vn,
                Y: vp - vn,
            })

    datos = pd.DataFrame(registros)

    validar_calendario(
        datos,
        M84,
        [
            Y,
            "IPC_MENSUAL_PASTO_PCT",
            "IPC_MENSUAL_NACIONAL_PCT",
        ],
    )

    ultimo = datos.iloc[-1]

    if not (
        np.isclose(
            ultimo["IPC_MENSUAL_PASTO_PCT"], 0.47
        )
        and np.isclose(
            ultimo["IPC_MENSUAL_NACIONAL_PCT"], 0.46
        )
    ):
        raise ValueError(
            "Control IPC diciembre 2024 no coincide"
        )

    return datos


def procesar_ictc(contenido):
    hoja = pd.read_excel(
        io.BytesIO(contenido),
        sheet_name="Anexo1",
        header=None,
    ).iloc[:, :3].copy()

    hoja.columns = [
        "anio_original",
        "mes_original",
        "valor_original",
    ]

    def convertir_anio(valor):
        try:
            numero = int(float(valor))

            if 2000 <= numero <= 2100:
                return numero

        except (TypeError, ValueError):
            pass

        return np.nan

    hoja["anio"] = hoja[
        "anio_original"
    ].map(convertir_anio).ffill()

    meses = (
        "enero febrero marzo abril mayo junio julio agosto "
        "septiembre octubre noviembre diciembre"
    ).split()

    mapa = {
        mes: j + 1
        for j, mes in enumerate(meses)
    }

    hoja["numero_mes"] = (
        hoja["mes_original"]
        .map(normalizar_texto)
        .map(mapa)
    )

    hoja[I] = hoja[
        "valor_original"
    ].map(convertir_numero)

    hoja = hoja[
        hoja["anio"].between(2018, 2024)
        & hoja["numero_mes"].notna()
        & hoja[I].notna()
    ].copy()

    hoja["mes"] = pd.PeriodIndex.from_fields(
        year=hoja["anio"].astype(int),
        month=hoja["numero_mes"].astype(int),
        freq="M",
    )

    datos = hoja[
        ["mes", I]
    ].sort_values(
        "mes"
    ).reset_index(drop=True)

    validar_calendario(
        datos,
        M84,
        [I],
    )

    if not (
        np.isclose(datos.iloc[0][I], 0.70)
        and np.isclose(datos.iloc[1][I], 0.83)
    ):
        raise ValueError(
            "Controles ICTC enero/febrero 2018 no coinciden"
        )

    return datos


def descargar_fuente(nombre_fuente):
    errores = []

    for url in URLS[nombre_fuente]:
        try:
            respuesta = requests.get(
                url,
                timeout=(12, 75),
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )

            respuesta.raise_for_status()

            contenido = respuesta.content
            hash_obtenido = sha256(contenido)

            if hash_obtenido != HASHES[nombre_fuente]:
                raise ValueError(
                    f"SHA-256 incompatible: {hash_obtenido}"
                )

            parsers = {
                "SIPSA": procesar_sipsa,
                "RNDC": procesar_rndc,
                "IPC": procesar_ipc,
                "ICTC": procesar_ictc,
            }

            datos = parsers[nombre_fuente](
                contenido
            )

            auditoria = {
                "Conjunto": nombre_fuente,
                "Observaciones": len(datos),
                "SHA-256": hash_obtenido,
                "Fuente": url,
                "Estado": "Validado",
            }

            return datos, auditoria

        except Exception as exc:
            errores.append(
                f"{type(exc).__name__}: {str(exc)[:170]}"
            )

    raise RuntimeError(" | ".join(errores))


@st.cache_data(
    ttl=1800,
    show_spinner="Recuperando y validando fuentes públicas...",
)
def cargar_datos():
    resultados = {}
    errores = {}

    with ThreadPoolExecutor(max_workers=4) as ejecutor:
        futuros = {
            nombre_fuente: ejecutor.submit(
                descargar_fuente,
                nombre_fuente,
            )
            for nombre_fuente in URLS
        }

        for nombre_fuente, futuro in futuros.items():
            try:
                resultados[nombre_fuente] = futuro.result()
            except Exception as exc:
                errores[nombre_fuente] = str(exc)

    datos = pd.DataFrame({
        "mes": M84
    })

    for nombre_fuente in [
        "IPC", "SIPSA", "RNDC", "ICTC"
    ]:
        if nombre_fuente not in resultados:
            continue

        tabla = resultados[nombre_fuente][0]

        datos = datos.merge(
            tabla,
            on="mes",
            how="left",
            validate="one_to_one",
        )

    auditoria = pd.DataFrame([
        resultados[k][1]
        for k in [
            "IPC", "SIPSA", "RNDC", "ICTC"
        ]
        if k in resultados
    ])

    return datos, auditoria, errores


def muestra_comun(datos, variables, calendario=M59):
    ausentes = [
        c for c in variables
        if c not in datos
    ]

    if ausentes:
        raise ValueError(
            "Variables ausentes: "
            + ", ".join(map(nombre, ausentes))
        )

    variables = list(dict.fromkeys(variables))

    muestra = datos.loc[
        datos["mes"].isin(calendario),
        ["mes"] + variables,
    ].copy().reset_index(drop=True)

    return validar_calendario(
        muestra,
        calendario,
        variables,
    )


def base_rezagada(datos):
    muestra = muestra_comun(
        datos, [Y, T, P]
    )

    for lag in range(4):
        muestra[f"T_L{lag}"] = muestra[T].shift(lag)
        muestra[f"P_L{lag}"] = muestra[P].shift(lag)

    muestra["B_L1"] = muestra[Y].shift(1)

    muestra = muestra.iloc[3:].reset_index(drop=True)

    columnas = (
        [Y, "B_L1"]
        + [f"T_L{j}" for j in range(4)]
        + [f"P_L{j}" for j in range(4)]
    )

    return validar_calendario(
        muestra,
        M56,
        columnas,
    )


def estimar_modelo(
    datos,
    familia="ARDL",
    L=2,
    hac=3,
    transporte=T,
    precios=P,
    excluir=None,
    trimestres=False,
):
    muestra = muestra_comun(
        datos,
        [Y, transporte, precios],
    )

    for j in range(4):
        muestra[f"T_L{j}"] = muestra[transporte].shift(j)
        muestra[f"P_L{j}"] = muestra[precios].shift(j)

    muestra["B_L1"] = muestra[Y].shift(1)

    muestra = muestra.iloc[3:].reset_index(drop=True)

    validar_calendario(
        muestra,
        M56,
        [Y, "B_L1", "T_L3", "P_L3"],
    )

    regresores = (
        (["B_L1"] if familia == "ARDL" else [])
        + [f"T_L{j}" for j in range(L + 1)]
        + [f"P_L{j}" for j in range(L + 1)]
    )

    if excluir:
        muestra = muestra[
            muestra["mes"] != pd.Period(excluir)
        ].copy()

    if trimestres:
        trimestre = (
            (muestra["mes"].dt.month - 1) // 3 + 1
        )

        for j in [2, 3, 4]:
            columna = f"Q{j}"
            muestra[columna] = (
                trimestre == j
            ).astype(int)
            regresores.append(columna)

    X = sm.add_constant(
        muestra[regresores].astype(float),
        has_constant="add",
    )

    if (
        len(muestra) <= X.shape[1] + 12
        or np.linalg.matrix_rank(X) != X.shape[1]
    ):
        raise ValueError(
            "Matriz singular o muestra insuficiente"
        )

    mco = sm.OLS(
        muestra[Y],
        X,
    ).fit()

    robusto = sm.OLS(
        muestra[Y],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac},
        use_t=True,
    )

    if not np.isfinite(
        robusto.bse
    ).all():
        raise ValueError(
            "La covarianza HAC no es finita"
        )

    return robusto, mco, muestra, X


def contrastes_conjuntos(modelo, L=2):
    filas = []

    for prefijo, etiqueta_bloque in [
        ("T", "RNDC / transporte"),
        ("P", "SIPSA / precios"),
    ]:
        terminos = [
            f"{prefijo}_L{j}"
            for j in range(L + 1)
        ]

        R = np.zeros(
            (len(terminos), len(modelo.params))
        )

        for j, termino in enumerate(terminos):
            posicion = modelo.params.index.get_loc(
                termino
            )
            R[j, posicion] = 1

        conjunta = modelo.wald_test(
            R,
            use_f=True,
            scalar=True,
        )

        suma = modelo.t_test(
            R.sum(axis=0)
        )

        intervalo = suma.conf_int()[0]

        filas.append({
            "Bloque": etiqueta_bloque,
            "p conjunta HAC": float(conjunta.pvalue),
            "Suma": float(
                modelo.params[terminos].sum()
            ),
            "p suma": float(suma.pvalue),
            "IC suma inferior": float(intervalo[0]),
            "IC suma superior": float(intervalo[1]),
        })

    return pd.DataFrame(filas)


def tabla_coeficientes(modelo):
    intervalos = modelo.conf_int()

    return pd.DataFrame({
        "Regresor": modelo.params.index,
        "Coeficiente": modelo.params.values,
        "Error HAC": modelo.bse.values,
        "p HAC": modelo.pvalues.values,
        "IC inferior": intervalos.iloc[:, 0].values,
        "IC superior": intervalos.iloc[:, 1].values,
    })


def estacionariedad(datos, variables):
    resultados = []

    for variable in variables:
        try:
            muestra = muestra_comun(
                datos,
                [variable],
            )

            adf = adfuller(
                muestra[variable],
                maxlag=6,
                regression="c",
                autolag="AIC",
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                prueba_kpss = kpss(
                    muestra[variable],
                    regression="c",
                    nlags="auto",
                )

            resultados.append({
                "Serie": nombre(variable),
                "N": len(muestra),
                "ADF": adf[0],
                "p ADF": adf[1],
                "KPSS": prueba_kpss[0],
                "p KPSS tabulado": prueba_kpss[1],
                "Compatible I(0)": bool(
                    adf[1] < 0.05
                    and prueba_kpss[1] >= 0.05
                ),
            })

        except Exception as exc:
            resultados.append({
                "Serie": nombre(variable),
                "Incidencia": str(exc),
            })

    return pd.DataFrame(resultados)


def diagnosticos(modelo_mco, muestra, X):
    residuos = modelo_mco.resid

    lb = acorr_ljungbox(
        residuos,
        lags=[6, 12],
        return_df=True,
    )

    estadisticos = {
        "Jarque–Bera p": jarque_bera(residuos)[1],
        "Ljung–Box (6) p": lb.loc[6, "lb_pvalue"],
        "Ljung–Box (12) p": lb.loc[12, "lb_pvalue"],
        "Breusch–Godfrey (3) p": (
            acorr_breusch_godfrey(
                modelo_mco,
                nlags=3,
            )[1]
        ),
        "Breusch–Pagan p": (
            het_breuschpagan(residuos, X)[1]
        ),
        "CUSUM p": (
            breaks_cusumolsresid(
                residuos,
                ddof=X.shape[1],
            )[1]
        ),
        "Durbin–Watson": (
            sm.stats.stattools.durbin_watson(
                residuos
            )
        ),
        "Número de condición": (
            np.linalg.cond(X)
        ),
    }

    try:
        estadisticos["RESET p"] = float(
            linear_reset(
                modelo_mco,
                power=2,
                use_f=True,
            ).pvalue
        )
    except Exception:
        estadisticos["RESET p"] = np.nan

    tabla_residuos = muestra[
        ["mes", Y]
    ].copy()

    tabla_residuos["Ajustado"] = (
        modelo_mco.fittedvalues.to_numpy()
    )

    tabla_residuos["Residuo"] = (
        residuos.to_numpy()
    )

    tabla_residuos["Cook"] = np.asarray(
        OLSInfluence(
            modelo_mco
        ).cooks_distance[0]
    )

    vif = pd.DataFrame([
        {
            "Regresor": columna,
            "VIF": variance_inflation_factor(
                X.to_numpy(float),
                indice,
            ),
        }
        for indice, columna in enumerate(X.columns)
        if columna != "const"
    ])

    return (
        pd.DataFrame([estadisticos]),
        tabla_residuos,
        vif,
    )


def robustez(datos):
    especificaciones = [
        (
            f"{familia} L{L}",
            {"familia": familia, "L": L},
        )
        for familia in ["DL", "ARDL"]
        for L in range(4)
    ]

    especificaciones += [
        ("HAC(1)", {"hac": 1}),
        ("HAC(6)", {"hac": 6}),
        (
            "Excluir febrero 2023",
            {"excluir": "2023-02"},
        ),
        (
            "Estacionalidad trimestral",
            {"trimestres": True},
        ),
    ]

    for variable in [
        "PM_no_ponderado_t",
        "PM_ponderado_lagQ_t",
        "PM_100",
        "PM_95",
    ]:
        if (
            variable in datos
            and datos.loc[
                datos["mes"].isin(M59),
                variable,
            ].notna().all()
        ):
            especificaciones.append((
                f"SIPSA: {nombre(variable)}",
                {"precios": variable},
            ))

    for variable in [
        I,
        "VAR_FLETE_FISHER_PCT",
        "VAR_FLETE_CRUDO_PCT",
        "VAR_FLETE_SIN_2000_PCT",
    ]:
        if (
            variable in datos
            and datos.loc[
                datos["mes"].isin(M59),
                variable,
            ].notna().all()
        ):
            especificaciones.append((
                f"Transporte: {nombre(variable)}",
                {"transporte": variable},
            ))

    resultados = []

    for etiqueta_modelo, parametros in especificaciones:
        try:
            m, mco, muestra, _ = estimar_modelo(
                datos,
                **parametros,
            )

            pruebas = contrastes_conjuntos(
                m,
                parametros.get("L", 2),
            )

            resultados.append({
                "Especificación": etiqueta_modelo,
                "N": len(muestra),
                "R² ajustado": mco.rsquared_adj,
                "AIC": mco.aic,
                "BIC": mco.bic,
                "p conjunta transporte": (
                    pruebas.iloc[0]["p conjunta HAC"]
                ),
                "p conjunta SIPSA": (
                    pruebas.iloc[1]["p conjunta HAC"]
                ),
                "Suma transporte": (
                    pruebas.iloc[0]["Suma"]
                ),
                "Estado": "Estimado",
            })

        except Exception as exc:
            resultados.append({
                "Especificación": etiqueta_modelo,
                "Estado": str(exc),
            })

    return pd.DataFrame(resultados)


def ventanas_moviles(datos, ancho=36):
    muestra = base_rezagada(datos)

    regresores = (
        ["B_L1"]
        + [f"T_L{j}" for j in range(3)]
        + [f"P_L{j}" for j in range(3)]
    )

    resultados = []

    for inicio in range(len(muestra) - ancho + 1):
        ventana = muestra.iloc[
            inicio:inicio + ancho
        ]

        X = sm.add_constant(
            ventana[regresores],
            has_constant="add",
        )

        modelo = sm.OLS(
            ventana[Y],
            X,
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3},
            use_t=True,
        )

        pruebas = contrastes_conjuntos(
            modelo
        )

        resultados.append({
            "Inicio": str(ventana["mes"].iloc[0]),
            "Fin": str(ventana["mes"].iloc[-1]),
            "Suma RNDC": pruebas.iloc[0]["Suma"],
            "RNDC IC inferior": (
                pruebas.iloc[0]["IC suma inferior"]
            ),
            "RNDC IC superior": (
                pruebas.iloc[0]["IC suma superior"]
            ),
            "Suma SIPSA": pruebas.iloc[1]["Suma"],
        })

    return pd.DataFrame(resultados)


def comparar_huber(datos):
    muestra = base_rezagada(datos)

    regresores = (
        ["B_L1"]
        + [f"T_L{j}" for j in range(3)]
        + [f"P_L{j}" for j in range(3)]
    )

    X = sm.add_constant(
        muestra[regresores],
        has_constant="add",
    )

    mco = sm.OLS(
        muestra[Y],
        X,
    ).fit()

    huber = sm.RLM(
        muestra[Y],
        X,
        M=sm.robust.norms.HuberT(),
    ).fit()

    resultados = []

    for prefijo, etiqueta_bloque in [
        ("T", "RNDC"),
        ("P", "SIPSA"),
    ]:
        terminos = [
            f"{prefijo}_L{j}"
            for j in range(3)
        ]

        resultados.append({
            "Bloque": etiqueta_bloque,
            "Suma MCO": float(
                mco.params[terminos].sum()
            ),
            "Suma Huber": float(
                huber.params[terminos].sum()
            ),
        })

    return pd.DataFrame(resultados)


def multiplicadores(
    modelo,
    prefijo,
    cambio=1.0,
    horizonte=9,
):
    parametros = list(modelo.params.index)
    covarianza = modelo.cov_params().to_numpy(float)

    rho = float(
        modelo.params.get("B_L1", 0)
    )

    posicion_rho = (
        parametros.index("B_L1")
        if "B_L1" in parametros
        else None
    )

    respuesta = 0.0
    gradiente = np.zeros(len(parametros))
    resultados = []

    for h in range(horizonte + 1):
        respuesta_anterior = respuesta
        gradiente_anterior = gradiente.copy()

        termino = f"{prefijo}_L{h}"

        respuesta = (
            rho * respuesta_anterior
            + float(
                modelo.params.get(termino, 0)
            )
        )

        gradiente = rho * gradiente_anterior

        if posicion_rho is not None:
            gradiente[posicion_rho] += (
                respuesta_anterior
            )

        if termino in parametros:
            gradiente[
                parametros.index(termino)
            ] += 1

        error = abs(cambio) * np.sqrt(
            max(
                float(
                    gradiente
                    @ covarianza
                    @ gradiente
                ),
                0,
            )
        )

        resultados.append({
            "Horizonte": h,
            "Respuesta (p. p.)": cambio * respuesta,
            "IC inferior": (
                cambio * respuesta - 1.96 * error
            ),
            "IC superior": (
                cambio * respuesta + 1.96 * error
            ),
        })

    return pd.DataFrame(resultados)


def analizar_pca(datos, variables, k=3):
    muestra = muestra_comun(
        datos,
        variables,
    )

    X = StandardScaler().fit_transform(
        muestra[variables]
    )

    pca = PCA(
        n_components=2
    ).fit(X)

    coordenadas = pca.transform(X)

    agrupamiento = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    grupos = agrupamiento.fit_predict(X)

    puntos = pd.DataFrame({
        "Mes": muestra["mes"].astype(str),
        "PC1": coordenadas[:, 0],
        "PC2": coordenadas[:, 1],
        "Grupo": grupos.astype(str),
    })

    cargas = pd.DataFrame(
        pca.components_.T,
        index=[
            nombre(variable)
            for variable in variables
        ],
        columns=["PC1", "PC2"],
    )

    silueta = silhouette_score(
        X,
        grupos,
    )

    return (
        puntos,
        cargas,
        pca.explained_variance_ratio_,
        silueta,
    )


def modelos_multivariantes(
    datos,
    variables,
    rezagos=1,
    horizonte=8,
):
    muestra = muestra_comun(
        datos,
        variables,
    )

    registros = []

    for variable in variables:
        registros.append({
            "Serie": nombre(variable),
            "p ADF nivel": adfuller(
                muestra[variable],
                autolag="AIC",
            )[1],
            "p KPSS nivel": kpss(
                muestra[variable],
                regression="c",
                nlags="auto",
            )[1],
            "p ADF diferencia": adfuller(
                muestra[variable].diff().dropna(),
                autolag="AIC",
            )[1],
        })

    pruebas = pd.DataFrame(registros)

    I0 = (
        (pruebas["p ADF nivel"] < 0.05)
        & (pruebas["p KPSS nivel"] >= 0.05)
    )

    I1 = (
        (pruebas["p ADF nivel"] >= 0.05)
        & (pruebas["p ADF diferencia"] < 0.05)
    )

    if I0.all():
        modelo = VAR(
            muestra[variables]
        ).fit(rezagos)

        if not modelo.is_stable():
            raise ValueError(
                "VAR inestable: no se calculan IRF"
            )

        return (
            pruebas,
            "VAR",
            modelo,
            modelo.irf(horizonte).irfs,
        )

    if I1.all():
        johansen = coint_johansen(
            muestra[variables],
            det_order=0,
            k_ar_diff=max(rezagos - 1, 0),
        )

        rechazos = (
            johansen.lr1 > johansen.cvt[:, 1]
        )

        rango = next(
            (
                i
                for i, rechaza in enumerate(rechazos)
                if not rechaza
            ),
            len(variables),
        )

        tabla = pd.DataFrame({
            "Rango nulo": range(len(variables)),
            "Traza": johansen.lr1,
            "Crítico 95 %": johansen.cvt[:, 1],
        })

        if 0 < rango < len(variables):
            modelo = VECM(
                muestra[variables],
                k_ar_diff=max(rezagos - 1, 0),
                coint_rank=rango,
                deterministic="ci",
            ).fit()

            return (
                pruebas,
                "VECM",
                modelo,
                tabla,
            )

        return (
            pruebas,
            "Johansen",
            rango,
            tabla,
        )

    return (
        pruebas,
        "No admisible",
        None,
        None,
    )


def evaluar_ml(datos, algoritmos):
    muestra = muestra_comun(
        datos,
        [Y, T, P],
    )

    tabla = pd.DataFrame({
        "mes": muestra["mes"],
        "objetivo": muestra[Y],
    })

    for variable in [Y, T, P]:
        for lag in [1, 2]:
            tabla[f"{variable}_L{lag}"] = (
                muestra[variable].shift(lag)
            )

    tabla = tabla.dropna().reset_index(drop=True)

    predictores = [
        c for c in tabla
        if c not in ["mes", "objetivo"]
    ]

    inicio_prueba = len(tabla) - 26

    if inicio_prueba < 24:
        raise ValueError(
            "Muestra de entrenamiento insuficiente"
        )

    constructores = {
        "Ridge": lambda: make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Lasso": lambda: make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=0.01,
                max_iter=15000,
            ),
        ),
        "SVR": lambda: make_pipeline(
            StandardScaler(),
            SVR(C=1.0, epsilon=0.1),
        ),
        "MLP": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(12,),
                max_iter=700,
                random_state=42,
            ),
        ),
    }

    resultados = []

    for indice in range(
        inicio_prueba,
        len(tabla),
    ):
        entrenamiento = tabla.iloc[:indice]
        prueba = tabla.iloc[[indice]]

        X = entrenamiento[
            predictores
        ].to_numpy(float)

        X_test = prueba[
            predictores
        ].to_numpy(float)

        y = entrenamiento[
            "objetivo"
        ].to_numpy(float)

        pronosticos = {
            "Media histórica": float(y.mean())
        }

        X_ar = sm.add_constant(
            entrenamiento[
                [f"{Y}_L1"]
            ].to_numpy(float),
            has_constant="add",
        )

        Xt_ar = sm.add_constant(
            prueba[
                [f"{Y}_L1"]
            ].to_numpy(float),
            has_constant="add",
        )

        ajuste_ar = sm.OLS(
            y, X_ar
        ).fit()

        pronosticos["AR(1)"] = float(
            ajuste_ar.predict(Xt_ar)[0]
        )

        ajuste_arx = sm.OLS(
            y,
            sm.add_constant(
                X,
                has_constant="add",
            ),
        ).fit()

        pronosticos["ARX"] = float(
            ajuste_arx.predict(
                sm.add_constant(
                    X_test,
                    has_constant="add",
                )
            )[0]
        )

        for etiqueta_modelo, constructor in constructores.items():
            if etiqueta_modelo not in algoritmos:
                continue

            modelo = constructor().fit(
                X, y
            )

            pronosticos[etiqueta_modelo] = float(
                modelo.predict(X_test)[0]
            )

        for etiqueta_modelo, pronostico in pronosticos.items():
            if etiqueta_modelo not in algoritmos:
                continue

            resultados.append({
                "Mes": str(
                    prueba["mes"].iloc[0]
                ),
                "Modelo": etiqueta_modelo,
                "Observado": float(
                    prueba["objetivo"].iloc[0]
                ),
                "Pronóstico": pronostico,
            })

    predicciones = pd.DataFrame(resultados)

    if predicciones.empty:
        raise ValueError(
            "Selecciona al menos un modelo"
        )

    metricas = []

    for etiqueta_modelo, grupo in predicciones.groupby("Modelo"):
        metricas.append({
            "Modelo": etiqueta_modelo,
            "N": len(grupo),
            "RMSE (p. p.)": np.sqrt(
                mean_squared_error(
                    grupo["Observado"],
                    grupo["Pronóstico"],
                )
            ),
            "MAE (p. p.)": mean_absolute_error(
                grupo["Observado"],
                grupo["Pronóstico"],
            ),
        })

    metricas = pd.DataFrame(
        metricas
    ).sort_values("RMSE (p. p.)")

    return (
        metricas,
        predicciones,
        tabla,
        predictores,
        inicio_prueba,
    )


def explicabilidad_ml(
    tabla,
    predictores,
    inicio,
    algoritmo,
):
    algoritmos = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(
            alpha=0.01,
            max_iter=15000,
        ),
        "SVR": SVR(
            C=1.0,
            epsilon=0.1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(12,),
            max_iter=700,
            random_state=42,
        ),
    }

    modelo = make_pipeline(
        StandardScaler(),
        algoritmos[algoritmo],
    )

    modelo.fit(
        tabla.iloc[:inicio][predictores],
        tabla.iloc[:inicio]["objetivo"],
    )

    importancia = permutation_importance(
        modelo,
        tabla.iloc[inicio:][predictores],
        tabla.iloc[inicio:]["objetivo"],
        n_repeats=12,
        random_state=42,
        scoring="neg_mean_squared_error",
    )

    return pd.DataFrame({
        "Variable": predictores,
        "Importancia": importancia.importances_mean,
        "Desviación": importancia.importances_std,
    }).sort_values("Importancia")


def semaforo_historico(datos):
    muestra = muestra_comun(
        datos,
        [T, P],
    )

    registros = []

    for indice in range(12, len(muestra)):
        historico = muestra.iloc[:indice]
        actual = muestra.iloc[indice]

        umbral_t = historico[T].quantile(0.90)
        umbral_p = historico[P].quantile(0.90)

        señales = (
            int(actual[T] > umbral_t)
            + int(actual[P] > umbral_p)
        )

        registros.append({
            "Mes": str(actual["mes"]),
            "RNDC": actual[T],
            "SIPSA": actual[P],
            "Umbral RNDC": umbral_t,
            "Umbral SIPSA": umbral_p,
            "Clasificación": [
                "Verde",
                "Amarillo",
                "Rojo",
            ][señales],
        })

    return pd.DataFrame(registros)


def mostrar_grafico(figura, altura=440):
    figura.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101b30",
        plot_bgcolor="#11213b",
        font=dict(color="#e8f0f7"),
        height=altura,
        margin=dict(
            t=60,
            l=25,
            r=20,
            b=35,
        ),
    )

    st.plotly_chart(
        figura,
        use_container_width=True,
    )

    return figura


def grafico_series(
    datos,
    variables,
    titulo,
    estandarizar=False,
):
    tabla = datos[
        ["mes"] + variables
    ].copy()

    tabla["mes"] = tabla["mes"].astype(str)

    if estandarizar:
        for variable in variables:
            desviacion = tabla[variable].std()

            if pd.notna(desviacion) and desviacion > 0:
                tabla[variable] = (
                    tabla[variable] - tabla[variable].mean()
                ) / desviacion

    largo = tabla.melt(
        id_vars="mes",
        var_name="Variable",
        value_name="Valor",
    ).dropna()

    largo["Variable"] = largo[
        "Variable"
    ].map(nombre)

    figura = px.line(
        largo,
        x="mes",
        y="Valor",
        color="Variable",
        title=titulo,
        facet_row=(
            "Variable"
            if len(variables) > 1 and not estandarizar
            else None
        ),
    )

    if len(variables) > 1 and not estandarizar:
        figura.update_yaxes(
            matches=None
        )

    altura = (
        max(440, 215 * len(variables))
        if not estandarizar
        else 460
    )

    return mostrar_grafico(
        figura,
        altura,
    )


def descargar_csv(datos, archivo):
    tabla = datos.copy()

    if "mes" in tabla:
        tabla["mes"] = tabla["mes"].astype(str)

    st.download_button(
        "⬇️ Descargar CSV",
        tabla.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=archivo,
        mime="text/csv",
    )


def descargar_grafico(figura, archivo):
    st.download_button(
        "⬇️ Exportar gráfico interactivo",
        figura.to_html(
            include_plotlyjs="cdn"
        ).encode("utf-8"),
        file_name=archivo,
        mime="text/html",
    )


def comprobar_muestra(datos, variables):
    try:
        muestra_comun(
            datos,
            variables,
        )
        return True

    except Exception:
        st.info(
            "Esta herramienta requiere una muestra "
            "mensual completa y validada de las variables seleccionadas."
        )
        return False


PAGINAS = [
    "🏠 Inicio",
    "📊 Explorador",
    "🗄️ Datos y calidad",
    "📈 Econometría",
    "🧩 PCA y sistemas",
    "🤖 Predicción y XAI",
    "🎯 Simulador y Focus Pasto",
    "📑 Informes",
    "💬 Asistente",
    "© Fuentes y créditos",
]

with st.sidebar:
    st.markdown("### 📊 OBSERVATORIO PASTO")

    st.caption(
        "Fletes · El Potrerillo · Brecha inflacionaria"
    )

    pagina = st.radio(
        "Navegación",
        PAGINAS,
    )

    st.divider()

    if st.button("🔄 Actualizar fuentes públicas"):
        cargar_datos.clear()
        st.session_state.pop(
            "resultado_ml",
            None,
        )
        st.rerun()


DATA, AUDITORIA, ERRORES = cargar_datos()

DISPONIBLES = [
    c for c in PRINCIPALES
    if c in DATA and DATA[c].notna().any()
]

COMPLEMENTARIAS = [
    c for c in NAMES
    if c not in PRINCIPALES
    and c in DATA
    and DATA[c].notna().any()
]


if pagina == "🏠 Inicio":
    st.title(
        "Observatorio Económico de Pasto y Nariño"
    )

    st.markdown("""
<div class="hero">
<h2>¡Bienvenidos al observatorio! 👋</h2>
<p>
Explora datos reales de 2018 a 2024 sobre transporte
de carga, precios mayoristas de El Potrerillo y
la inflación de Pasto frente a Colombia.
Compara tendencias, interpreta modelos, descubre
patrones y descarga gráficos y resultados
desde un solo lugar.
</p>
</div>
""", unsafe_allow_html=True)

    for tarjeta, variable in zip(
        st.columns(4),
        [Y, T, P, I],
    ):
        n = (
            int(DATA[variable].count())
            if variable in DATA
            else 0
        )

        tarjeta.metric(
            nombre(variable),
            f"{n} meses" if n else "Sin datos",
        )

    if DISPONIBLES:
        izquierda, derecha = st.columns(
            [1.7, 1]
        )

        with izquierda:
            elegido = st.selectbox(
                "Indicador principal",
                DISPONIBLES,
                format_func=nombre,
            )

            observados = DATA.dropna(
                subset=[elegido]
            )

            ultimo = observados.iloc[-1]

            st.metric(
                f"Última observación · {ultimo['mes']}",
                f"{ultimo[elegido]:+.3f}",
            )

            grafico_series(
                DATA,
                [elegido],
                "Evolución mensual",
            )

        with derecha:
            cobertura = pd.DataFrame({
                "Serie": [
                    nombre(c)
                    for c in DISPONIBLES
                ],
                "Meses": [
                    int(DATA[c].count())
                    for c in DISPONIBLES
                ],
            })

            mostrar_grafico(
                px.bar(
                    cobertura,
                    x="Meses",
                    y="Serie",
                    orientation="h",
                    title="Cobertura histórica",
                ),
                440,
            )

    if len(DISPONIBLES) > 1:
        st.subheader(
            "Comparador de indicadores"
        )

        elegidas = st.multiselect(
            "Series",
            DISPONIBLES,
            default=DISPONIBLES,
            format_func=nombre,
        )

        if elegidas:
            zscore = st.toggle(
                "Comparar patrones estandarizados",
                value=True,
            )

            grafico_series(
                DATA,
                elegidas,
                "Tendencias comparadas",
                estandarizar=zscore,
            )

    a, b, c = st.columns(3)

    a.markdown(
        "**📊 Explorar**\n\n"
        "Datos, variabilidad y correlaciones."
    )

    b.markdown(
        "**📈 Estimar**\n\n"
        "ARDL/HAC, robustez y estabilidad."
    )

    c.markdown(
        "**🤖 Experimentar**\n\n"
        "PCA, predicción, simulación e informes."
    )


elif pagina == "📊 Explorador":
    st.title(
        "Explorador interactivo de indicadores"
    )

    opciones = (
        DISPONIBLES + COMPLEMENTARIAS
    )

    if opciones:
        seleccionadas = st.multiselect(
            "Variables",
            opciones,
            default=DISPONIBLES,
            format_func=nombre,
        )

        periodos = [
            str(mes) for mes in M84
        ]

        desde, hasta = st.select_slider(
            "Periodo",
            options=periodos,
            value=(
                periodos[0],
                periodos[-1],
            ),
        )

        vista = DATA[
            DATA["mes"].between(
                pd.Period(desde),
                pd.Period(hasta),
            )
        ].copy()

        if seleccionadas:
            tabs = st.tabs([
                "Tendencias",
                "Descriptivos",
                "Asociaciones",
                "Distribuciones",
                "Datos exportables",
            ])

            with tabs[0]:
                figura = grafico_series(
                    vista,
                    seleccionadas,
                    "Series en unidades originales",
                )

                descargar_grafico(
                    figura,
                    "series_interactivas.html",
                )

                if len(seleccionadas) > 1:
                    grafico_series(
                        vista,
                        seleccionadas,
                        "Patrones estandarizados",
                        estandarizar=True,
                    )

            with tabs[1]:
                descriptivos = (
                    vista[seleccionadas]
                    .describe()
                    .T
                    .reset_index(names="Serie")
                )

                descriptivos["Serie"] = (
                    descriptivos["Serie"].map(nombre)
                )

                st.dataframe(
                    descriptivos,
                    hide_index=True,
                    use_container_width=True,
                )

                descargar_csv(
                    descriptivos,
                    "estadisticos.csv",
                )

                variable = st.selectbox(
                    "Autocorrelación",
                    seleccionadas,
                    format_func=nombre,
                )

                serie = vista[
                    variable
                ].dropna()

                if len(serie) > 20:
                    from statsmodels.tsa.stattools import acf

                    rezagos = min(
                        12,
                        len(serie) // 3,
                    )

                    valores_acf = acf(
                        serie,
                        nlags=rezagos,
                        fft=False,
                    )

                    mostrar_grafico(
                        px.bar(
                            x=np.arange(rezagos + 1),
                            y=valores_acf,
                            labels={
                                "x": "Rezago",
                                "y": "ACF",
                            },
                            title="Autocorrelación mensual",
                        )
                    )

            with tabs[2]:
                if len(seleccionadas) > 1:
                    metodo = st.radio(
                        "Método",
                        ["pearson", "spearman"],
                        horizontal=True,
                    )

                    correlacion = vista[
                        seleccionadas
                    ].corr(
                        method=metodo,
                        min_periods=12,
                    )

                    mostrar_grafico(
                        px.imshow(
                            correlacion,
                            zmin=-1,
                            zmax=1,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            title="Correlaciones pareadas",
                        )
                    )

                    xcol, ycol = st.columns(2)

                    x = xcol.selectbox(
                        "Eje X",
                        seleccionadas,
                        format_func=nombre,
                    )

                    y = ycol.selectbox(
                        "Eje Y",
                        seleccionadas,
                        index=1,
                        format_func=nombre,
                    )

                    mostrar_grafico(
                        px.scatter(
                            vista.dropna(
                                subset=[x, y]
                            ),
                            x=x,
                            y=y,
                            title="Dispersión observada",
                        )
                    )

            with tabs[3]:
                distribucion = vista[
                    seleccionadas
                ].melt(
                    var_name="Serie",
                    value_name="Valor",
                ).dropna()

                distribucion["Serie"] = (
                    distribucion["Serie"].map(nombre)
                )

                mostrar_grafico(
                    px.box(
                        distribucion,
                        x="Serie",
                        y="Valor",
                        color="Serie",
                        points="outliers",
                        title="Rangos y valores extremos",
                    ),
                    520,
                )

                variable_hist = st.selectbox(
                    "Histograma",
                    seleccionadas,
                    format_func=nombre,
                )

                mostrar_grafico(
                    px.histogram(
                        vista,
                        x=variable_hist,
                        nbins=18,
                        title=f"Distribución: {nombre(variable_hist)}",
                    )
                )

            with tabs[4]:
                tabla = vista[
                    ["mes"] + seleccionadas
                ].copy()

                tabla["mes"] = (
                    tabla["mes"].astype(str)
                )

                st.dataframe(
                    tabla,
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )

                descargar_csv(
                    tabla,
                    "datos_filtrados.csv",
                )


elif pagina == "🗄️ Datos y calidad":
    st.title(
        "Datos y auditoría de integridad"
    )

    cobertura = []

    for variable in [Y, T, P, I]:
        fechas = (
            DATA.loc[
                DATA[variable].notna(),
                "mes",
            ]
            if variable in DATA
            else pd.Series(dtype="object")
        )

        cobertura.append({
            "Indicador": nombre(variable),
            "Meses": len(fechas),
            "Primer mes": (
                str(fechas.min())
                if len(fechas)
                else "—"
            ),
            "Último mes": (
                str(fechas.max())
                if len(fechas)
                else "—"
            ),
        })

    st.dataframe(
        pd.DataFrame(cobertura),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader(
        "Trazabilidad de las descargas"
    )

    st.dataframe(
        AUDITORIA,
        use_container_width=True,
        hide_index=True,
    )

    if ERRORES:
        with st.expander(
            f"Incidencias de fuentes ({len(ERRORES)})"
        ):
            for fuente, error in ERRORES.items():
                st.write(
                    f"**{fuente}:** {error}"
                )

    if P in DATA:
        st.subheader(
            "Cobertura física SIPSA"
        )

        columnas = [
            c for c in [
                "cobertura_indice_pct",
                "n_productos_indice",
                "kg_total",
                "kg_usados_indice",
            ]
            if c in DATA
        ]

        if columnas:
            grafico_series(
                DATA,
                columnas[:2],
                "Cobertura y comparabilidad SIPSA",
            )

        alternativas = [
            c for c in [
                P,
                "PM_no_ponderado_t",
                "PM_ponderado_lagQ_t",
                "PM_100",
                "PM_95",
            ]
            if c in DATA
        ]

        if len(alternativas) > 1:
            grafico_series(
                DATA,
                alternativas,
                "Sensibilidad del indicador SIPSA",
            )

    if T in DATA:
        st.subheader(
            "Calidad de registros RNDC"
        )

        columnas = [
            c for c in [
                "COB_VALOR_T_PCT",
                "COB_VIAJES_T_PCT",
                "COB_KG_T_PCT",
                "CELDAS_COMPARABLES",
            ]
            if c in DATA
        ]

        if columnas:
            grafico_series(
                DATA,
                columnas,
                "Cobertura y comparabilidad RNDC",
            )

        alternativas = [
            c for c in [
                T,
                "VAR_FLETE_FISHER_PCT",
                "VAR_FLETE_CRUDO_PCT",
            ]
            if c in DATA
        ]

        if len(alternativas) > 1:
            grafico_series(
                DATA,
                alternativas,
                "Variantes del índice regional de fletes",
            )

    descargar_csv(
        DATA,
        "base_integrada_observatorio.csv",
    )

    descargar_csv(
        AUDITORIA,
        "auditoria_sha256.csv",
    )


elif pagina == "📈 Econometría":
    st.title(
        "Laboratorio econométrico"
    )

    tabs = st.tabs([
        "Estacionariedad",
        "DL/ARDL + HAC",
        "Diagnósticos",
        "Robustez",
        "Ventanas móviles",
    ])

    with tabs[0]:
        seleccionadas = st.multiselect(
            "Series para ADF y KPSS",
            DISPONIBLES,
            default=DISPONIBLES,
            format_func=nombre,
        )

        if seleccionadas:
            resultado = estacionariedad(
                DATA,
                seleccionadas,
            )

            st.dataframe(
                resultado,
                hide_index=True,
                use_container_width=True,
            )

            descargar_csv(
                resultado,
                "estacionariedad.csv",
            )

    with tabs[1]:
        st.latex(
            r"B_t=\alpha+\rho B_{t-1}"
            r"+\sum_{j=0}^{L}\beta_jT_{t-j}"
            r"+\sum_{j=0}^{L}\gamma_jPM_{t-j}+u_t"
        )

        if comprobar_muestra(DATA, [Y, T, P]):
            c1, c2, c3 = st.columns(3)

            familia = c1.selectbox(
                "Familia",
                ["ARDL", "DL"],
            )

            L = c2.slider(
                "Rezagos L",
                0,
                3,
                2,
            )

            ancho_hac = c3.slider(
                "Ancho HAC",
                0,
                8,
                3,
            )

            try:
                modelo, mco, muestra, X = estimar_modelo(
                    DATA,
                    familia=familia,
                    L=L,
                    hac=ancho_hac,
                )

                pruebas = contrastes_conjuntos(
                    modelo,
                    L,
                )

                indicadores = [
                    ("Observaciones", len(muestra)),
                    (
                        "R² ajustado",
                        f"{mco.rsquared_adj:.4f}",
                    ),
                    ("AIC", f"{mco.aic:.2f}"),
                    ("BIC", f"{mco.bic:.2f}"),
                ]

                for tarjeta, (etiqueta_kpi, valor) in zip(
                    st.columns(4),
                    indicadores,
                ):
                    tarjeta.metric(
                        etiqueta_kpi,
                        valor,
                    )

                st.subheader(
                    "Estimación e inferencia HAC"
                )

                coeficientes = tabla_coeficientes(
                    modelo
                )

                st.dataframe(
                    coeficientes,
                    hide_index=True,
                    use_container_width=True,
                )

                st.subheader(
                    "Restricciones conjuntas y sumas"
                )

                st.dataframe(
                    pruebas,
                    hide_index=True,
                    use_container_width=True,
                )

                ajuste = muestra[
                    ["mes", Y]
                ].copy()

                ajuste["Ajustado"] = (
                    mco.fittedvalues.to_numpy()
                )

                grafico_series(
                    ajuste,
                    [Y, "Ajustado"],
                    "Brecha observada y ajuste dentro de muestra",
                )

                coef_graf = coeficientes[
                    coeficientes["Regresor"] != "const"
                ].copy()

                mostrar_grafico(
                    px.scatter(
                        coef_graf,
                        x="Coeficiente",
                        y="Regresor",
                        error_x=(
                            coef_graf["IC superior"]
                            - coef_graf["Coeficiente"]
                        ),
                        error_x_minus=(
                            coef_graf["Coeficiente"]
                            - coef_graf["IC inferior"]
                        ),
                        title="Coeficientes e intervalos HAC",
                    )
                )

                st.write(
                    "RNDC contemporáneo: "
                    f"{modelo.params.get('T_L0', np.nan):.5f}; "
                    "p HAC individual: "
                    f"{modelo.pvalues.get('T_L0', np.nan):.4f}. "
                    "p conjunta RNDC: "
                    f"{pruebas.iloc[0]['p conjunta HAC']:.4f}; "
                    "p conjunta SIPSA: "
                    f"{pruebas.iloc[1]['p conjunta HAC']:.4f}."
                )

                if (
                    familia == "ARDL"
                    and L == 2
                    and ancho_hac == 3
                ):
                    referencias = {
                        "Coef. RNDC contemporáneo": (
                            0.00913,
                            float(modelo.params["T_L0"]),
                        ),
                        "p RNDC contemporáneo": (
                            0.03750,
                            float(modelo.pvalues["T_L0"]),
                        ),
                        "p conjunta RNDC": (
                            0.14273,
                            float(
                                pruebas.iloc[0]["p conjunta HAC"]
                            ),
                        ),
                        "p conjunta SIPSA": (
                            0.23749,
                            float(
                                pruebas.iloc[1]["p conjunta HAC"]
                            ),
                        ),
                    }

                    contraste = pd.DataFrame([
                        {
                            "Indicador": clave,
                            "Referencia del artículo": valores[0],
                            "Cálculo de la app": valores[1],
                            "Diferencia absoluta": abs(
                                valores[1] - valores[0]
                            ),
                        }
                        for clave, valores in referencias.items()
                    ])

                    with st.expander(
                        "Verificación de reproducción frente al artículo"
                    ):
                        st.dataframe(
                            contraste,
                            hide_index=True,
                            use_container_width=True,
                        )

                        if (
                            (
                                contraste["Diferencia absoluta"]
                                < 0.001
                            ).all()
                            and len(muestra) == 56
                        ):
                            st.success(
                                "Los cuatro controles redondeados "
                                "del artículo coinciden."
                            )
                        else:
                            st.error(
                                "Se detectaron diferencias con "
                                "las referencias del artículo."
                            )

                descargar_csv(
                    coeficientes,
                    "coeficientes_hac.csv",
                )

                descargar_csv(
                    pruebas,
                    "contrastes_conjuntos.csv",
                )

            except Exception as exc:
                st.error(
                    f"Estimación no disponible: {exc}"
                )

    with tabs[2]:
        if comprobar_muestra(DATA, [Y, T, P]):
            try:
                _, mco, muestra, X = estimar_modelo(
                    DATA
                )

                tabla, residuos, vif = diagnosticos(
                    mco,
                    muestra,
                    X,
                )

                st.dataframe(
                    tabla.T,
                    use_container_width=True,
                )

                st.dataframe(
                    vif,
                    hide_index=True,
                    use_container_width=True,
                )

                izquierda, derecha = st.columns(2)

                with izquierda:
                    mostrar_grafico(
                        px.scatter(
                            residuos,
                            x="Ajustado",
                            y="Residuo",
                            title="Residuos frente al ajuste",
                        )
                    )

                with derecha:
                    mostrar_grafico(
                        px.histogram(
                            residuos,
                            x="Residuo",
                            nbins=15,
                            title="Distribución de residuos",
                        )
                    )

                residuos["Mes"] = (
                    residuos["mes"].astype(str)
                )

                mostrar_grafico(
                    px.bar(
                        residuos,
                        x="Mes",
                        y="Cook",
                        title="Influencia de observaciones",
                    )
                )

                descargar_csv(
                    residuos,
                    "diagnostico_residuos.csv",
                )

            except Exception as exc:
                st.info(str(exc))

    with tabs[3]:
        if comprobar_muestra(DATA, [Y, T, P]):
            if st.button(
                "Calcular comparación de especificaciones",
                type="primary",
            ):
                resultado = robustez(DATA)

                st.dataframe(
                    resultado,
                    hide_index=True,
                    use_container_width=True,
                )

                descargar_csv(
                    resultado,
                    "robustez.csv",
                )

                calculados = resultado[
                    resultado["Estado"] == "Estimado"
                ]

                if not calculados.empty:
                    mostrar_grafico(
                        px.bar(
                            calculados,
                            x="Especificación",
                            y="p conjunta transporte",
                            title="Sensibilidad de restricciones",
                        )
                    )

            if st.button("MCO frente a Huber"):
                try:
                    st.dataframe(
                        comparar_huber(DATA),
                        hide_index=True,
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(str(exc))

    with tabs[4]:
        if (
            comprobar_muestra(DATA, [Y, T, P])
            and st.button("Estimar ventanas de 36 meses")
        ):
            try:
                resultado = ventanas_moviles(
                    DATA
                )

                st.metric(
                    "Ventanas",
                    len(resultado),
                )

                st.dataframe(
                    resultado,
                    hide_index=True,
                    use_container_width=True,
                )

                mostrar_grafico(
                    px.line(
                        resultado,
                        x="Fin",
                        y=[
                            "Suma RNDC",
                            "Suma SIPSA",
                        ],
                        title="Estabilidad de sumas",
                    )
                )

                descargar_csv(
                    resultado,
                    "ventanas_moviles.csv",
                )

            except Exception as exc:
                st.error(str(exc))


elif pagina == "🧩 PCA y sistemas":
    st.title(
        "Patrones y dinámica multivariante"
    )

    tabs = st.tabs([
        "PCA y biplot",
        "K-Means",
        "VAR / VECM",
    ])

    with tabs[0]:
        seleccionadas = st.multiselect(
            "Series",
            DISPONIBLES,
            default=DISPONIBLES,
            format_func=nombre,
            key="pca_vars",
        )

        if (
            len(seleccionadas) >= 2
            and comprobar_muestra(DATA, seleccionadas)
        ):
            try:
                puntos, cargas, varianza, _ = analizar_pca(
                    DATA,
                    seleccionadas,
                    3,
                )

                st.metric(
                    "Varianza PC1 + PC2",
                    f"{sum(varianza):.1%}",
                )

                fig = px.scatter(
                    puntos,
                    x="PC1",
                    y="PC2",
                    hover_data=["Mes"],
                    title="Biplot mensual",
                )

                amplitud = max(
                    np.abs(
                        puntos[["PC1", "PC2"]]
                        .to_numpy()
                    ).max(),
                    1.0,
                ) * 0.5

                for _, fila in cargas.iterrows():
                    x = float(
                        fila["PC1"] * amplitud
                    )

                    y = float(
                        fila["PC2"] * amplitud
                    )

                    fig.add_annotation(
                        x=x,
                        y=y,
                        ax=0,
                        ay=0,
                        text=str(fila.name),
                        showarrow=True,
                        arrowhead=2,
                    )

                mostrar_grafico(
                    fig,
                    560,
                )

                st.dataframe(
                    cargas,
                    use_container_width=True,
                )

                descargar_csv(
                    puntos,
                    "pca_coordenadas.csv",
                )

            except Exception as exc:
                st.info(str(exc))

    with tabs[1]:
        seleccionadas = st.multiselect(
            "Series",
            DISPONIBLES,
            default=DISPONIBLES,
            format_func=nombre,
            key="kmeans_vars",
        )

        k = st.slider(
            "Número de grupos",
            2,
            6,
            3,
        )

        if (
            len(seleccionadas) >= 2
            and comprobar_muestra(DATA, seleccionadas)
        ):
            try:
                puntos, _, _, silueta = analizar_pca(
                    DATA,
                    seleccionadas,
                    k,
                )

                st.metric(
                    "Coeficiente silhouette",
                    f"{silueta:.3f}",
                )

                mostrar_grafico(
                    px.scatter(
                        puntos,
                        x="PC1",
                        y="PC2",
                        color="Grupo",
                        hover_data=["Mes"],
                        title="K-Means: agrupación de meses",
                    ),
                    520,
                )

                st.dataframe(
                    puntos,
                    hide_index=True,
                    use_container_width=True,
                )

                descargar_csv(
                    puntos,
                    "kmeans_meses.csv",
                )

            except Exception as exc:
                st.info(str(exc))

    with tabs[2]:
        seleccionadas = st.multiselect(
            "Variables endógenas",
            DISPONIBLES,
            default=DISPONIBLES[:3],
            max_selections=3,
            format_func=nombre,
        )

        rezagos = st.slider(
            "Rezagos del sistema",
            1,
            3,
            1,
        )

        if (
            len(seleccionadas) >= 2
            and st.button("Evaluar VAR / cointegración")
        ):
            try:
                pruebas, familia, modelo, resultado = (
                    modelos_multivariantes(
                        DATA,
                        seleccionadas,
                        rezagos,
                    )
                )

                st.dataframe(
                    pruebas,
                    hide_index=True,
                    use_container_width=True,
                )

                if familia == "VAR":
                    st.metric(
                        "AIC",
                        f"{modelo.aic:.3f}",
                    )

                    izquierda, derecha = st.columns(2)

                    impulsora = izquierda.selectbox(
                        "Innovación en",
                        seleccionadas,
                        format_func=nombre,
                    )

                    respuesta = derecha.selectbox(
                        "Respuesta de",
                        seleccionadas,
                        format_func=nombre,
                    )

                    valores = resultado[
                        :,
                        seleccionadas.index(respuesta),
                        seleccionadas.index(impulsora),
                    ]

                    mostrar_grafico(
                        px.line(
                            x=range(len(valores)),
                            y=valores,
                            markers=True,
                            labels={
                                "x": "Meses",
                                "y": "Respuesta",
                            },
                            title="IRF reducida exploratoria",
                        )
                    )

                elif familia == "VECM":
                    st.subheader(
                        "Velocidades de ajuste α"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            modelo.alpha,
                            index=seleccionadas,
                        ),
                        use_container_width=True,
                    )

                    st.dataframe(
                        resultado,
                        hide_index=True,
                    )

                elif familia == "Johansen":
                    st.write(
                        f"Rango obtenido por traza: {modelo}"
                    )

                    st.dataframe(
                        resultado,
                        hide_index=True,
                    )

                else:
                    st.info(
                        "No se estimó un sistema con los "
                        "supuestos de integración definidos."
                    )

            except Exception as exc:
                st.error(str(exc))


elif pagina == "🤖 Predicción y XAI":
    st.title(
        "Pronóstico retrospectivo y explicabilidad"
    )

    if comprobar_muestra(DATA, [Y, T, P]):
        algoritmos = st.multiselect(
            "Modelos para comparación",
            [
                "Media histórica",
                "AR(1)",
                "ARX",
                "Ridge",
                "Lasso",
                "SVR",
                "MLP",
            ],
            default=[
                "Media histórica",
                "AR(1)",
                "ARX",
                "Ridge",
                "Lasso",
                "SVR",
            ],
        )

        if (
            st.button(
                "Evaluar con origen expansivo",
                type="primary",
            )
            and algoritmos
        ):
            try:
                st.session_state["resultado_ml"] = evaluar_ml(
                    DATA,
                    algoritmos,
                )

            except Exception as exc:
                st.error(str(exc))

        if "resultado_ml" in st.session_state:
            (
                metricas,
                predicciones,
                datos_ml,
                predictores,
                inicio,
            ) = st.session_state["resultado_ml"]

            st.dataframe(
                metricas,
                hide_index=True,
                use_container_width=True,
            )

            pronosticados = predicciones[
                ["Mes", "Modelo", "Pronóstico"]
            ].rename(
                columns={"Pronóstico": "Valor"}
            )

            observados = (
                predicciones
                .drop_duplicates("Mes")
                [["Mes", "Observado"]]
                .rename(
                    columns={"Observado": "Valor"}
                )
            )

            observados["Modelo"] = "Observado"

            mostrar_grafico(
                px.line(
                    pd.concat([
                        pronosticados,
                        observados,
                    ]),
                    x="Mes",
                    y="Valor",
                    color="Modelo",
                    title=(
                        "Pronósticos de un mes frente "
                        "a la brecha observada"
                    ),
                )
            )

            descargar_csv(
                predicciones,
                "predicciones.csv",
            )

            modelo_xai = st.selectbox(
                "Modelo para importancia por permutación",
                [
                    "Ridge",
                    "Lasso",
                    "SVR",
                    "MLP",
                ],
            )

            if st.button("Calcular XAI"):
                try:
                    importancia = explicabilidad_ml(
                        datos_ml,
                        predictores,
                        inicio,
                        modelo_xai,
                    )

                    mostrar_grafico(
                        px.bar(
                            importancia,
                            x="Importancia",
                            y="Variable",
                            orientation="h",
                            title="Importancia de predictores",
                        )
                    )

                    st.dataframe(
                        importancia,
                        hide_index=True,
                    )

                    descargar_csv(
                        importancia,
                        "explicabilidad.csv",
                    )

                except Exception as exc:
                    st.error(str(exc))


elif pagina == "🎯 Simulador y Focus Pasto":
    st.title(
        "Escenarios condicionados y Focus Pasto"
    )

    tabs = st.tabs([
        "Simulador ARDL",
        "Episodio 2021",
        "Semáforo retrospectivo",
    ])

    with tabs[0]:
        if comprobar_muestra(DATA, [Y, T, P]):
            variable = st.selectbox(
                "Indicador del escenario",
                [T, P],
                format_func=nombre,
            )

            izquierda, derecha = st.columns(2)

            cambio = izquierda.number_input(
                "Cambio hipotético (p. p.)",
                value=1.0,
                step=0.5,
            )

            horizonte = derecha.slider(
                "Horizonte en meses",
                1,
                15,
                9,
            )

            try:
                modelo, _, _, _ = estimar_modelo(
                    DATA
                )

                respuesta = multiplicadores(
                    modelo,
                    "T" if variable == T else "P",
                    cambio,
                    horizonte,
                )

                figura = go.Figure()

                figura.add_trace(
                    go.Scatter(
                        x=respuesta["Horizonte"],
                        y=respuesta["IC superior"],
                        line=dict(width=0),
                        showlegend=False,
                    )
                )

                figura.add_trace(
                    go.Scatter(
                        x=respuesta["Horizonte"],
                        y=respuesta["IC inferior"],
                        fill="tonexty",
                        line=dict(width=0),
                        name="IC 95 %",
                    )
                )

                figura.add_trace(
                    go.Scatter(
                        x=respuesta["Horizonte"],
                        y=respuesta["Respuesta (p. p.)"],
                        mode="lines+markers",
                        name="Respuesta condicional",
                    )
                )

                figura.add_hline(
                    y=0,
                    line_dash="dot",
                )

                mostrar_grafico(figura)

                st.dataframe(
                    respuesta,
                    hide_index=True,
                    use_container_width=True,
                )

                descargar_csv(
                    respuesta,
                    "escenario_ardl.csv",
                )

            except Exception as exc:
                st.error(str(exc))

    with tabs[1]:
        st.subheader(
            "Abastecimiento de El Potrerillo: 2021"
        )

        st.metric(
            "Variación entre primeras quincenas, DANE",
            "−56,98 %",
        )

        st.write(
            "4.618 toneladas del 1 al 15 de abril "
            "y 1.987 toneladas del 1 al 15 de mayo de 2021."
        )

        episodio = DATA[
            DATA["mes"].between(
                pd.Period("2021-04"),
                pd.Period("2021-08"),
            )
        ].copy()

        if DISPONIBLES:
            grafico_series(
                episodio,
                DISPONIBLES,
                "Contexto mensual del episodio",
            )

            tabla = episodio[
                ["mes"] + DISPONIBLES
            ].copy()

            tabla["mes"] = tabla["mes"].astype(str)

            st.dataframe(
                tabla,
                hide_index=True,
                use_container_width=True,
            )

    with tabs[2]:
        if comprobar_muestra(DATA, [T, P]):
            resultado = semaforo_historico(
                DATA
            )

            mostrar_grafico(
                px.scatter(
                    resultado,
                    x="RNDC",
                    y="SIPSA",
                    color="Clasificación",
                    hover_data=["Mes"],
                    title="Presiones de transporte y precios",
                )
            )

            st.dataframe(
                resultado,
                hide_index=True,
                use_container_width=True,
            )

            descargar_csv(
                resultado,
                "semaforo_retrospectivo.csv",
            )


elif pagina == "📑 Informes":
    st.title(
        "Informe calculado desde el observatorio"
    )

    informe = [
        "# Informe económico Pasto–Colombia",
        f"Fecha: {datetime.now():%Y-%m-%d %H:%M}",
        "## Cobertura de variables",
    ]

    for variable in [Y, T, P, I]:
        n = (
            int(DATA[variable].count())
            if variable in DATA
            else 0
        )

        informe.append(
            f"- {nombre(variable)}: {n} meses"
        )

    informe.append(
        "## Estadísticos de series recuperadas"
    )

    for variable in DISPONIBLES:
        serie = DATA[
            variable
        ].dropna()

        informe.append(
            f"- {nombre(variable)}: "
            f"media={serie.mean():.5f}; "
            f"desviación={serie.std():.5f}; "
            f"mínimo={serie.min():.5f}; "
            f"máximo={serie.max():.5f}; "
            f"N={len(serie)}."
        )

    try:
        modelo, mco, muestra, _ = estimar_modelo(
            DATA
        )

        pruebas = contrastes_conjuntos(
            modelo
        )

        informe.extend([
            "## ARDL L2 con HAC(3)",
            (
                f"N={len(muestra)}, "
                f"R² ajustado={mco.rsquared_adj:.5f}, "
                f"AIC={mco.aic:.5f}"
            ),
            (
                f"RNDC contemporáneo="
                f"{modelo.params['T_L0']:.6f}; "
                f"p HAC={modelo.pvalues['T_L0']:.6f}"
            ),
            (
                "p conjunto RNDC="
                f"{pruebas.iloc[0]['p conjunta HAC']:.6f}; "
                "p conjunto SIPSA="
                f"{pruebas.iloc[1]['p conjunta HAC']:.6f}"
            ),
            (
                "Modelo observacional: asociación "
                "condicionada, no causalidad."
            ),
        ])

        descargar_csv(
            tabla_coeficientes(modelo),
            "tabla_ardl.csv",
        )

    except Exception as exc:
        informe.extend([
            "## Modelo no reproducido",
            str(exc),
        ])

    informe.extend([
        "## Fuentes",
        (
            "DANE SIPSA/IPC/ICTC y RNDC procesado: "
            f"https://github.com/{REPO}"
        ),
        "## Restricciones",
        (
            "No se imputaron observaciones ni "
            "se generaron valores ficticios."
        ),
    ])

    reporte = "\n\n".join(informe)

    st.markdown(reporte)

    st.download_button(
        "⬇️ Descargar informe Markdown",
        reporte.encode("utf-8-sig"),
        file_name="informe_pasto.md",
        mime="text/markdown",
    )

    descargar_csv(
        DATA,
        "base_integrada.csv",
    )

    descargar_csv(
        AUDITORIA,
        "fuentes_sha256.csv",
    )


elif pagina == "💬 Asistente":
    st.title(
        "Asistente de consultas del observatorio"
    )

    if "chat_obs" not in st.session_state:
        st.session_state["chat_obs"] = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! Consulta valores mensuales, "
                    "promedios y coeficientes del observatorio."
                ),
            }
        ]

    for mensaje in st.session_state["chat_obs"]:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input(
        "Ej.: ¿Cuál fue el RNDC en mayo de 2021?"
    )

    if pregunta:
        consulta = normalizar_texto(
            pregunta
        )

        if any(
            termino in consulta
            for termino in [
                "sipsa", "mayorista", "potrerillo"
            ]
        ):
            variable = P

        elif any(
            termino in consulta
            for termino in [
                "rndc", "fletes", "flete"
            ]
        ):
            variable = T

        elif "ictc" in consulta:
            variable = I

        else:
            variable = Y

        encontrado = re.search(
            r"(20(?:18|19|2[0-4]))[-/ ](0?[1-9]|1[0-2])",
            consulta,
        )

        periodo = (
            pd.Period(
                f"{encontrado.group(1)}-"
                f"{int(encontrado.group(2)):02d}"
            )
            if encontrado
            else None
        )

        if periodo is None:
            meses = (
                "enero febrero marzo abril mayo junio "
                "julio agosto septiembre octubre "
                "noviembre diciembre"
            ).split()

            for j, mes in enumerate(meses, 1):
                match = re.search(
                    rf"\b{mes}\s+(?:de\s+)?(20\d{{2}})\b",
                    consulta,
                )

                if match:
                    periodo = pd.Period(
                        f"{match.group(1)}-{j:02d}"
                    )
                    break

        if any(
            termino in consulta
            for termino in [
                "causa", "causal", "bloqueo"
            ]
        ):
            respuesta = (
                "El observatorio estima asociaciones, "
                "no efectos causales ni presencia "
                "de cierres viales."
            )

        elif (
            variable not in DATA
            or DATA[variable].notna().sum() == 0
        ):
            respuesta = (
                "No se dispone de observaciones "
                f"verificadas para {nombre(variable)}."
            )

        elif periodo is not None:
            valores = DATA.loc[
                DATA["mes"] == periodo,
                variable,
            ]

            if (
                len(valores)
                and pd.notna(valores.iloc[0])
            ):
                respuesta = (
                    f"{nombre(variable)} en {periodo}: "
                    f"**{float(valores.iloc[0]):.5f}**."
                )
            else:
                respuesta = (
                    "No existe observación válida de "
                    f"{nombre(variable)} para {periodo}."
                )

        elif any(
            termino in consulta
            for termino in ["media", "promedio"]
        ):
            respuesta = (
                f"Promedio de {nombre(variable)}: "
                f"**{DATA[variable].mean():.5f}** "
                f"({DATA[variable].count()} meses)."
            )

        elif any(
            termino in consulta
            for termino in [
                "maximo", "minimo", "pico"
            ]
        ):
            tabla = DATA[
                ["mes", variable]
            ].dropna()

            indice = (
                tabla[variable].idxmin()
                if "minimo" in consulta
                else tabla[variable].idxmax()
            )

            respuesta = (
                "Extremo observado: "
                f"**{tabla.loc[indice, variable]:.5f}**, "
                f"mes {tabla.loc[indice, 'mes']}."
            )

        elif any(
            termino in consulta
            for termino in [
                "coeficiente", "ardl", "significancia"
            ]
        ):
            try:
                modelo, _, _, _ = estimar_modelo(
                    DATA
                )

                respuesta = (
                    "ARDL L2: RNDC contemporáneo "
                    f"{modelo.params['T_L0']:.6f}; "
                    f"p HAC {modelo.pvalues['T_L0']:.5f}."
                )

            except Exception as exc:
                respuesta = (
                    f"El ARDL no se pudo calcular: {exc}"
                )

        else:
            respuesta = (
                "Indica RNDC, SIPSA, ICTC o brecha "
                "y un mes AAAA-MM, o consulta "
                "promedios y valores extremos."
            )

        st.session_state["chat_obs"].extend([
            {
                "role": "user",
                "content": pregunta,
            },
            {
                "role": "assistant",
                "content": respuesta,
            },
        ])

        st.rerun()


else:
    st.title(
        "Fuentes, metodología y créditos"
    )

    st.markdown(f"""
**Universidad de Nariño · Economía · Investigación regional**

**Objeto:** Fletes del transporte de carga hacia Pasto,
precios mayoristas de El Potrerillo y brecha inflacionaria
Pasto–Colombia (2018–2024).

**Autores del artículo:** Juan David Granja Miramag,
Helmer Andrés Rivera Vallejos y Yeison Alexander Ortega Rosero.

**Fuentes primarias:** DANE (IPC, SIPSA-P, SIPSA-A e ICTC);
Ministerio de Transporte (registros RNDC).
El índice regional RNDC es una construcción de los investigadores.

**Repositorios y documentación:**

- [Indicadores procesados](https://github.com/{REPO})
- [Respaldos originales SIPSA](https://github.com/{RESPALDO}/releases/tag/{TAG})
- [Colab de investigación](https://colab.research.google.com/drive/1Tu3ZzP1L8XlUejqbS_0eXgZ8QszqiMr7)

**Reproducibilidad:** El observatorio verifica que los archivos
publicados coincidan byte a byte con los SHA-256 auditados.
No carga documentos del usuario, no interpola observaciones ni
simula datos. El archivo SIPSA corresponde al producto mensual
procesado por el Colab y no debe confundirse con los microdatos
originales.

**Metodología:** 84 meses descriptivos, 59 meses comunes y
56 observaciones para los ocho modelos DL/ARDL.
ARDL L2 con errores HAC(3) es la especificación principal.
La inferencia es asociativa, no causal.

**Alcance territorial:** La variable de inflación corresponde
a Pasto y no representa automáticamente la inflación del
departamento de Nariño. El ICTC nacional no es equivalente al
índice de fletes regional RNDC.

**Análisis complementarios:** VAR/VECM requieren supuestos de
integración. Los clústeres agrupan meses, no municipios.
Machine Learning se evalúa retrospectivamente sin comprobar
calendarios históricos de publicación.
La importancia por permutación no supone explicaciones causales.

**Simulaciones:** Las trayectorias ARDL son condicionadas.
El semáforo es retrospectivo e ilustrativo.
No constituye un sistema validado para predecir cierres viales.

**Econometría espacial:** No se calculan Moran/LISA porque las
series disponibles no constituyen un panel territorial
georreferenciado ni proporcionan una matriz de vecindad válida.

**Próxima etapa:** El sistema RAG documental y el artículo final
se desarrollarán por separado.
""")

    st.subheader(
        "Fuentes auditadas"
    )

    st.dataframe(
        AUDITORIA,
        hide_index=True,
        use_container_width=True,
    )

    if ERRORES:
        with st.expander(
            "Incidencias de integridad o conectividad"
        ):
            for fuente, error in ERRORES.items():
                st.write(
                    f"**{fuente}:** {error}"
                )


st.divider()

st.caption(
    "Universidad de Nariño · Observatorio Pasto–Colombia · "
    "Datos DANE y RNDC · Serie histórica 2018–2024"
)
