
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
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
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


st.set_page_config(
    page_title="Observatorio Pasto–Nariño",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg,#0b1424,#131d34);
    color: #e4eef8;
}
section[data-testid="stSidebar"] {
    background: #101b2e;
    border-right: 1px solid #34465e;
}
h1,h2,h3 {
    color: #92e7f0!important;
}
[data-testid="stMetric"] {
    background: #1c2a42;
    border: 1px solid #354861;
    border-radius: 13px;
    padding: 12px;
}
.stButton button[kind="primary"] {
    background: linear-gradient(90deg,#0d829e,#7654c5);
    border: 0;
}
.hero {
    background: linear-gradient(110deg,#123c53,#2a2449);
    border: 1px solid #3b6174;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.hero h2 {
    margin-top: 0;
    color: #b1f4f4!important;
}
.hero p {
    font-size: 1.05rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


REPO = "JuanDavidgm2026/fletes-precios-inflacion-pasto"
BACKUP = "JuanDavidgm2026/SIPSA-PASTO"
TAG = "sipsa-dane-originales-v1"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/data"

Y = "BRECHA_INFLACION_PASTO_COLOMBIA_PP"
T = "VAR_FLETE_TORNQVIST_PCT"
P = "PM_ponderado_t"
I = "ICTC_MENSUAL_PCT"

M84 = pd.period_range("2018-01", "2024-12", freq="M")
M59 = pd.period_range("2020-02", "2024-12", freq="M")
M56 = pd.period_range("2020-05", "2024-12", freq="M")

FUENTES = {
    "SIPSA": (
        [f"{RAW}/SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"],
        "227448e82618c3f9dee2f7e72d922cfea5811878ebc02ddb90998c5d10913676",
    ),
    "RNDC": (
        [f"{RAW}/variable2_flete_regional_modelo_2020_2024.csv"],
        "aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891",
    ),
    "IPC": (
        [
            "https://www.dane.gov.co/files/operaciones/IPC/dic2024/anex-IPC-CiudadesMensuales-dic2024.xlsx",
            f"https://github.com/{BACKUP}/releases/download/{TAG}/anex-IPC-CiudadesMensuales-dic2024.xlsx",
        ],
        "3e584414f3f89c78f99ce92576a0715d3d51fac5d6dcd75addf84065e4cd4076",
    ),
    "ICTC": (
        [
            "https://www.dane.gov.co/files/operaciones/ICTC/anex-ICTC-HistVariaContrib-dic2024.xlsx",
            f"https://github.com/{BACKUP}/releases/download/{TAG}/anex-ICTC-HistVariaContrib-dic2024.xlsx",
        ],
        "6fe232548e47dbf9801b75c620209f579ebd0faa2a46fc30a9a305453878205a",
    ),
}

NOMBRES = {
    Y: "Brecha Pasto–Colombia (p. p.)",
    T: "Fletes RNDC regional (%)",
    P: "Precios SIPSA ponderados (%)",
    I: "ICTC nacional (%)",
    "IPC_MENSUAL_PASTO_PCT": "Inflación mensual Pasto (%)",
    "IPC_MENSUAL_NACIONAL_PCT": "Inflación mensual nacional (%)",
    "PM_no_ponderado_t": "SIPSA no ponderado (%)",
    "PM_ponderado_lagQ_t": "SIPSA pesos rezagados (%)",
    "PM_100": "SIPSA canasta 100 % (%)",
    "PM_95": "SIPSA canasta 95 % (%)",
    "cobertura_indice_pct": "Cobertura física SIPSA (%)",
    "n_productos_indice": "Productos SIPSA comparables",
    "kg_total": "Abastecimiento total (kg)",
    "kg_usados_indice": "Abastecimiento cubierto (kg)",
    "VAR_FLETE_FISHER_PCT": "RNDC Fisher (%)",
    "VAR_FLETE_CRUDO_PCT": "RNDC sin depuración (%)",
    "VAR_FLETE_SIN_2000_PCT": "RNDC sin registro 2.000 (%)",
    "COB_VALOR_T_PCT": "Cobertura RNDC valor (%)",
    "COB_VIAJES_T_PCT": "Cobertura RNDC viajes (%)",
    "CELDAS_COMPARABLES": "Celdas RNDC comparables",
    "Ajustado": "Brecha ajustada (p. p.)",
}


def etiqueta(c):
    return NOMBRES.get(c, c)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normal(x):
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKD", str(x).strip().lower())
    return " ".join(
        "".join(c for c in s if not unicodedata.combining(c)).split()
    )


def num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    return float(
        pd.to_numeric(
            str(x).strip().replace("%", "").replace(",", "."),
            errors="coerce",
        )
    )


def validar(d, calendario, columnas):
    if d["mes"].tolist() != list(calendario) or d["mes"].duplicated().any():
        raise ValueError(
            f"Meses faltantes, extras o duplicados en {calendario[0]}–{calendario[-1]}"
        )
    if not np.isfinite(d[columnas].to_numpy(float)).all():
        raise ValueError(
            "Valores faltantes/no finitos en columnas obligatorias"
        )
    return d


def periodo(serie):
    if isinstance(serie.dtype, pd.PeriodDtype):
        return serie.dt.asfreq("M")

    s = serie.astype("string").str.strip()

    if s.dropna().str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(
            s, format="%Y%m", errors="coerce"
        ).dt.to_period("M")

    if s.dropna().str.fullmatch(r"\d{4}-\d{2}").all():
        return pd.to_datetime(
            s + "-01", errors="coerce"
        ).dt.to_period("M")

    return pd.to_datetime(
        serie, errors="coerce"
    ).dt.to_period("M")


def parse_sipsa(raw):
    d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")

    campos = [
        "mes",
        P,
        "PM_no_ponderado_t",
        "PM_ponderado_lagQ_t",
        "cobertura_indice_pct",
        "n_productos_indice",
    ]

    if any(c not in d for c in campos):
        raise ValueError(
            "Columnas SIPSA no corresponden al Colab"
        )

    d["mes"] = periodo(d["mes"])

    for c in NOMBRES:
        if c in d and c != "Ajustado":
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.sort_values("mes").reset_index(drop=True)

    validar(
        d,
        M84,
        [
            P,
            "PM_no_ponderado_t",
            "cobertura_indice_pct",
            "n_productos_indice",
        ],
    )

    if not d["cobertura_indice_pct"].between(0, 100).all():
        raise ValueError(
            "Cobertura SIPSA fuera de 0–100 %"
        )

    return d[
        ["mes"] + [
            c for c in NOMBRES
            if c in d and c != "Ajustado"
        ]
    ]


def parse_rndc(raw):
    d = pd.read_csv(
        io.BytesIO(raw),
        encoding="utf-8-sig",
        dtype={"PERIODO": "string"},
    )

    campos = [
        "FECHA",
        "PERIODO",
        T,
        "VAR_FLETE_FISHER_PCT",
        "CELDAS_COMPARABLES",
    ]

    if any(c not in d for c in campos):
        raise ValueError(
            "Columnas RNDC no corresponden al Colab"
        )

    d["mes"] = pd.to_datetime(
        d["FECHA"],
        errors="raise",
    ).dt.to_period("M")

    if not (
        d["PERIODO"].str.strip()
        == d["mes"].dt.strftime("%Y%m")
    ).all():
        raise ValueError(
            "FECHA y PERIODO del RNDC no coinciden"
        )

    for c in NOMBRES:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.sort_values("mes").reset_index(drop=True)

    validar(
        d,
        M59,
        [
            T,
            "VAR_FLETE_FISHER_PCT",
            "CELDAS_COMPARABLES",
        ],
    )

    return d[
        ["mes"] + [
            c for c in NOMBRES
            if c in d and c != "Ajustado"
        ]
    ]


def parse_ipc(raw):
    a = pd.read_excel(
        io.BytesIO(raw),
        sheet_name="VarCiudades",
        header=None,
    )

    bloques = []

    for j, valor in enumerate(a.iloc[:, 0]):
        m = re.search(
            r"variaciones mensuales\s+(20\d{2})",
            normal(valor),
        )
        if m:
            bloques.append((int(m.group(1)), j))

    if [x[0] for x in bloques] != list(range(2017, 2025)):
        raise ValueError(
            "El Excel IPC no contiene bloques 2017–2024 auditados"
        )

    nombres_meses = (
        "enero febrero marzo abril mayo junio julio agosto "
        "septiembre octubre noviembre diciembre"
    ).split()

    salida = []

    for k, (anio, inicio) in enumerate(bloques):
        encabezados = [
            normal(a.iat[inicio + 1, j])
            for j in range(1, 13)
        ]

        if encabezados != nombres_meses:
            raise ValueError(
                f"Encabezado IPC incorrecto en {anio}"
            )

        fin = (
            bloques[k + 1][1]
            if k + 1 < len(bloques)
            else len(a)
        )

        def ubicar(nombre_fila):
            return [
                j for j in range(inicio + 2, fin)
                if normal(a.iat[j, 0]) == nombre_fila
            ]

        p = ubicar("pasto")
        n = ubicar("total ipc")

        if len(p) != 1 or len(n) != 1:
            raise ValueError(
                f"Pasto/Total IPC ambiguos en {anio}"
            )

        for mes in range(1, 13):
            if anio >= 2018:
                vp = num(a.iat[p[0], mes])
                vn = num(a.iat[n[0], mes])

                salida.append({
                    "mes": pd.Period(f"{anio}-{mes:02d}"),
                    "IPC_MENSUAL_PASTO_PCT": vp,
                    "IPC_MENSUAL_NACIONAL_PCT": vn,
                    Y: vp - vn,
                })

    d = pd.DataFrame(salida)

    validar(
        d,
        M84,
        [
            Y,
            "IPC_MENSUAL_PASTO_PCT",
            "IPC_MENSUAL_NACIONAL_PCT",
        ],
    )

    if not (
        np.isclose(
            d.iloc[-1]["IPC_MENSUAL_PASTO_PCT"], .47
        )
        and np.isclose(
            d.iloc[-1]["IPC_MENSUAL_NACIONAL_PCT"], .46
        )
    ):
        raise ValueError(
            "Control IPC diciembre 2024 incompatible"
        )

    return d


def parse_ictc(raw):
    a = pd.read_excel(
        io.BytesIO(raw),
        sheet_name="Anexo1",
        header=None,
    ).iloc[:, :3].copy()

    a.columns = [
        "anio_raw",
        "mes_raw",
        "valor_raw",
    ]

    def anio(v):
        try:
            n = int(float(v))
            return float(n) if 2000 <= n <= 2100 else np.nan
        except (ValueError, TypeError):
            return np.nan

    a["anio"] = a.anio_raw.map(anio).ffill()

    meses = (
        "enero febrero marzo abril mayo junio julio agosto "
        "septiembre octubre noviembre diciembre"
    ).split()

    a["nmes"] = a.mes_raw.map(normal).map({
        m: j + 1
        for j, m in enumerate(meses)
    })

    a[I] = a.valor_raw.map(num)

    a = a[
        a.anio.between(2018, 2024)
        & a.nmes.notna()
        & a[I].notna()
    ].copy()

    a["mes"] = pd.to_datetime({
        "year": a.anio.astype(int),
        "month": a.nmes.astype(int),
        "day": 1,
    }).dt.to_period("M")

    d = a[
        ["mes", I]
    ].sort_values("mes").reset_index(drop=True)

    validar(d, M84, [I])

    if not (
        np.isclose(d.iloc[0][I], .70)
        and np.isclose(d.iloc[1][I], .83)
    ):
        raise ValueError(
            "Controles ICTC enero/febrero 2018 incompatibles"
        )

    return d


def parse_combinado(raw):
    d = pd.read_csv(
        io.BytesIO(raw),
        encoding="utf-8-sig",
    )

    campos = [
        "mes",
        Y,
        "IPC_MENSUAL_PASTO_PCT",
        "IPC_MENSUAL_NACIONAL_PCT",
        I,
    ]

    if any(c not in d for c in campos):
        raise ValueError(
            "CSV conjunto sin columnas IPC/ICTC obligatorias"
        )

    d["mes"] = periodo(d["mes"])

    for c in campos[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[campos].sort_values("mes").reset_index(drop=True)

    validar(d, M84, campos[1:])

    if not np.allclose(
        d[Y],
        d["IPC_MENSUAL_PASTO_PCT"]
        - d["IPC_MENSUAL_NACIONAL_PCT"],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            "Brecha no coincide con IPC Pasto menos IPC nacional"
        )

    if not (
        np.isclose(d.iloc[0][I], .70)
        and np.isclose(d.iloc[1][I], .83)
        and np.isclose(
            d.iloc[-1]["IPC_MENSUAL_PASTO_PCT"], .47
        )
        and np.isclose(
            d.iloc[-1]["IPC_MENSUAL_NACIONAL_PCT"], .46
        )
    ):
        raise ValueError(
            "CSV conjunto no supera controles del Colab"
        )

    return d


def descargar(url):
    r = requests.get(
        url,
        timeout=(12, 65),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    return r.content


def fuente_original(clave):
    urls, esperado = FUENTES[clave]
    errores = []

    for url in urls:
        try:
            raw = descargar(url)

            if sha(raw) != esperado:
                raise ValueError(
                    f"SHA distinto al esperado ({sha(raw)})"
                )

            parser = {
                "SIPSA": parse_sipsa,
                "RNDC": parse_rndc,
                "IPC": parse_ipc,
                "ICTC": parse_ictc,
            }[clave]

            d = parser(raw)

            return d, {
                "Fuente": clave,
                "Meses": len(d),
                "Origen": url,
                "SHA-256": sha(raw),
                "Control": "Coincide con SHA auditado",
            }

        except Exception as e:
            errores.append(
                f"{url}: {type(e).__name__}: {str(e)[:180]}"
            )

    raise RuntimeError(" | ".join(errores))


def fuente_combinada():
    url = f"{RAW}/ipc_ictc_pasto_2018_2024.csv"
    manifest = f"{RAW}/manifest_ipc_ictc_sha256.json"

    r = requests.get(
        url,
        timeout=(12, 65),
        headers={"User-Agent": "Mozilla/5.0"},
    )

    if r.status_code == 404:
        return None

    r.raise_for_status()
    raw = r.content

    j = requests.get(
        manifest,
        timeout=(12, 30),
        headers={"User-Agent": "Mozilla/5.0"},
    )

    esperado = None

    if j.ok:
        esperado = j.json().get("sha256")

        if not esperado:
            raise ValueError(
                "Manifiesto IPC/ICTC sin SHA-256"
            )

        if esperado != sha(raw):
            raise ValueError(
                "CSV IPC/ICTC no coincide con su manifiesto"
            )

    d = parse_combinado(raw)

    return d, {
        "Fuente": "IPC + ICTC",
        "Meses": len(d),
        "Origen": url,
        "SHA-256": sha(raw),
        "Control": (
            "SHA del manifiesto + datos"
            if esperado
            else "Datos validados; sin manifiesto"
        ),
    }


@st.cache_data(
    ttl=3600,
    show_spinner="Recuperando y verificando datos DANE / GitHub…",
)
def cargar():
    resultados = {}
    incidencias = {}

    try:
        conjunto = fuente_combinada()

        if conjunto is not None:
            resultados["IPC+ICTC"] = conjunto

    except Exception as e:
        incidencias["IPC+ICTC procesado"] = str(e)

    claves = ["RNDC", "SIPSA"]

    if "IPC+ICTC" not in resultados:
        claves.extend(["IPC", "ICTC"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        trabajos = {
            c: pool.submit(fuente_original, c)
            for c in claves
        }

        for c, trabajo in trabajos.items():
            try:
                resultados[c] = trabajo.result()
            except Exception as e:
                incidencias[c] = str(e)

    base = pd.DataFrame({"mes": M84})

    for c in ["IPC+ICTC", "IPC", "ICTC", "SIPSA", "RNDC"]:
        if c in resultados:
            base = base.merge(
                resultados[c][0],
                on="mes",
                how="left",
                validate="one_to_one",
            )

    audit = pd.DataFrame([
        resultados[c][1]
        for c in resultados
    ])

    return base, audit, incidencias


def muestra(base, columnas, calendario=M59):
    faltantes = [
        c for c in columnas
        if c not in base
    ]

    if faltantes:
        raise ValueError(
            "No se recuperaron: "
            + ", ".join(map(etiqueta, faltantes))
        )

    columnas = list(dict.fromkeys(columnas))

    d = base.loc[
        base.mes.isin(calendario),
        ["mes"] + columnas,
    ].reset_index(drop=True)

    return validar(d, calendario, columnas)


def ajustar(
    base,
    familia="ARDL",
    L=2,
    hac=3,
    transporte=T,
    precio=P,
    excluir=None,
    estacionalidad=False,
):
    d = muestra(base, [Y, transporte, precio])

    for j in range(4):
        d[f"T{j}"] = d[transporte].shift(j)
        d[f"P{j}"] = d[precio].shift(j)

    d["B1"] = d[Y].shift(1)
    d = d.iloc[3:].reset_index(drop=True)

    columnas = (
        (["B1"] if familia == "ARDL" else [])
        + [f"T{j}" for j in range(L + 1)]
        + [f"P{j}" for j in range(L + 1)]
    )

    validar(
        d,
        M56,
        [Y, "B1", "T3", "P3"],
    )

    if excluir:
        d = d[
            d.mes != pd.Period(excluir)
        ].copy()

    if estacionalidad:
        q = ((d.mes.dt.month - 1) // 3 + 1)

        for j in (2, 3, 4):
            d[f"Q{j}"] = (q == j).astype(int)
            columnas.append(f"Q{j}")

    X = sm.add_constant(
        d[columnas].astype(float),
        has_constant="add",
    )

    if (
        len(d) <= X.shape[1] + 10
        or np.linalg.matrix_rank(X) != X.shape[1]
    ):
        raise ValueError(
            "Matriz singular o grados de libertad insuficientes"
        )

    o = sm.OLS(d[Y], X).fit()

    m = sm.OLS(d[Y], X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac},
        use_t=True,
    )

    if not np.isfinite(m.bse).all():
        raise ValueError(
            "Covarianza HAC inválida"
        )

    return m, o, d, X


def conjuntas(m, L=2):
    filas = []

    for pref, nombre_bloque in [
        ("T", "Transporte"),
        ("P", "SIPSA"),
    ]:
        terms = [
            f"{pref}{j}"
            for j in range(L + 1)
        ]

        R = np.zeros(
            (len(terms), len(m.params))
        )

        for j, term in enumerate(terms):
            R[j, m.params.index.get_loc(term)] = 1

        prueba = m.wald_test(
            R,
            use_f=True,
            scalar=True,
        )

        suma = m.t_test(
            R.sum(axis=0)
        )

        ci = suma.conf_int()[0]

        filas.append({
            "Bloque": nombre_bloque,
            "p conjunta HAC": float(prueba.pvalue),
            "Suma": float(m.params[terms].sum()),
            "p suma": float(suma.pvalue),
            "IC suma inferior": float(ci[0]),
            "IC suma superior": float(ci[1]),
        })

    return pd.DataFrame(filas)


def tabla_coef(m):
    ci = m.conf_int()

    return pd.DataFrame({
        "Regresor": m.params.index,
        "Coeficiente": m.params.values,
        "EE HAC": m.bse.values,
        "p HAC": m.pvalues.values,
        "IC inferior": ci.iloc[:, 0].values,
        "IC superior": ci.iloc[:, 1].values,
    })


def estacionariedad(base, cols):
    filas = []

    for c in cols:
        try:
            z = muestra(base, [c])
            adf = adfuller(z[c], autolag="AIC")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp = kpss(
                    z[c],
                    regression="c",
                    nlags="auto",
                )

            filas.append({
                "Serie": etiqueta(c),
                "ADF": adf[0],
                "p ADF": adf[1],
                "KPSS": kp[0],
                "p KPSS tabulada": kp[1],
                "Compatible I(0)": bool(
                    adf[1] < .05
                    and kp[1] >= .05
                ),
            })

        except Exception as e:
            filas.append({
                "Serie": etiqueta(c),
                "Incidencia": str(e),
            })

    return pd.DataFrame(filas)


def diagnosticos(o, z, X):
    e = o.resid

    lb = acorr_ljungbox(
        e,
        lags=[6, 12],
        return_df=True,
    )

    data = {
        "Jarque–Bera p": float(jarque_bera(e)[1]),
        "Ljung–Box 6 p": float(
            lb.loc[6, "lb_pvalue"]
        ),
        "Ljung–Box 12 p": float(
            lb.loc[12, "lb_pvalue"]
        ),
        "Breusch–Godfrey p": float(
            acorr_breusch_godfrey(
                o, nlags=3
            )[1]
        ),
        "Breusch–Pagan p": float(
            het_breuschpagan(e, X)[1]
        ),
        "CUSUM p": float(
            breaks_cusumolsresid(
                e,
                ddof=X.shape[1],
            )[1]
        ),
        "Durbin–Watson": float(
            sm.stats.stattools.durbin_watson(e)
        ),
    }

    try:
        data["RESET p"] = float(
            linear_reset(
                o,
                power=2,
                use_f=True,
            ).pvalue
        )
    except Exception:
        data["RESET p"] = np.nan

    r = z[
        ["mes", Y]
    ].copy()

    r["Ajustado"] = o.fittedvalues.to_numpy()
    r["Residuo"] = e.to_numpy()
    r["Cook"] = np.asarray(
        OLSInfluence(o).cooks_distance[0]
    )

    vif = pd.DataFrame([
        {
            "Regresor": c,
            "VIF": variance_inflation_factor(
                X.to_numpy(float), j
            ),
        }
        for j, c in enumerate(X.columns)
        if c != "const"
    ])

    return pd.DataFrame([data]), r, vif


def robustez(base):
    specs = [
        (
            f"{f} L{L}",
            dict(familia=f, L=L),
        )
        for f in ("DL", "ARDL")
        for L in range(4)
    ]

    specs += [
        ("HAC(1)", {"hac": 1}),
        ("HAC(6)", {"hac": 6}),
        (
            "Excluir febrero 2023",
            {"excluir": "2023-02"},
        ),
        (
            "Estacionalidad trimestral",
            {"estacionalidad": True},
        ),
    ]

    for c in [
        "PM_no_ponderado_t",
        "PM_ponderado_lagQ_t",
        "PM_100",
        "PM_95",
    ]:
        if (
            c in base
            and base.loc[
                base.mes.isin(M59), c
            ].notna().all()
        ):
            specs.append((
                f"SIPSA: {etiqueta(c)}",
                {"precio": c},
            ))

    for c in [
        I,
        "VAR_FLETE_FISHER_PCT",
        "VAR_FLETE_CRUDO_PCT",
        "VAR_FLETE_SIN_2000_PCT",
    ]:
        if (
            c in base
            and base.loc[
                base.mes.isin(M59), c
            ].notna().all()
        ):
            specs.append((
                f"Transporte: {etiqueta(c)}",
                {"transporte": c},
            ))

    filas = []

    for nombre_spec, params in specs:
        try:
            m, o, z, _ = ajustar(
                base,
                **params,
            )

            j = conjuntas(
                m,
                params.get("L", 2),
            )

            filas.append({
                "Especificación": nombre_spec,
                "N": len(z),
                "R² ajustado": o.rsquared_adj,
                "AIC": o.aic,
                "BIC": o.bic,
                "p transporte": j.iloc[0]["p conjunta HAC"],
                "p SIPSA": j.iloc[1]["p conjunta HAC"],
                "Estado": "Estimado",
            })

        except Exception as e:
            filas.append({
                "Especificación": nombre_spec,
                "Estado": str(e),
            })

    return pd.DataFrame(filas)


def ventanas(base, ancho=36):
    d = muestra(base, [Y, T, P])

    for j in range(4):
        d[f"T{j}"] = d[T].shift(j)
        d[f"P{j}"] = d[P].shift(j)

    d["B1"] = d[Y].shift(1)
    d = d.iloc[3:].reset_index(drop=True)

    regs = [
        "B1", "T0", "T1", "T2",
        "P0", "P1", "P2",
    ]

    filas = []

    for ini in range(len(d) - ancho + 1):
        sub = d.iloc[ini:ini + ancho]

        X = sm.add_constant(
            sub[regs],
            has_constant="add",
        )

        m = sm.OLS(sub[Y], X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3},
            use_t=True,
        )

        j = conjuntas(m)

        filas.append({
            "Inicio": str(sub.mes.iloc[0]),
            "Fin": str(sub.mes.iloc[-1]),
            "Suma RNDC": j.iloc[0]["Suma"],
            "Suma SIPSA": j.iloc[1]["Suma"],
            "RNDC IC inferior": j.iloc[0]["IC suma inferior"],
            "RNDC IC superior": j.iloc[0]["IC suma superior"],
        })

    return pd.DataFrame(filas)


def huber(base):
    _, o, z, X = ajustar(base)

    m = sm.RLM(
        z[Y],
        X,
        M=sm.robust.norms.HuberT(),
    ).fit()

    return pd.DataFrame([
        {
            "Bloque": lab,
            "Suma MCO": o.params[cols].sum(),
            "Suma Huber": m.params[cols].sum(),
        }
        for lab, cols in [
            ("RNDC", ["T0", "T1", "T2"]),
            ("SIPSA", ["P0", "P1", "P2"]),
        ]
    ])


def respuesta_dinamica(m, pref, cambio, horizonte):
    cols = list(m.params.index)
    cov = m.cov_params().to_numpy(float)

    rho = float(m.params.get("B1", 0))
    i_rho = cols.index("B1") if "B1" in cols else None

    r = 0.0
    g = np.zeros(len(cols))
    filas = []

    for h in range(horizonte + 1):
        old_r = r
        old_g = g.copy()

        termino = f"{pref}{h}"

        r = (
            rho * old_r
            + float(m.params.get(termino, 0))
        )

        g = rho * old_g

        if i_rho is not None:
            g[i_rho] += old_r

        if termino in cols:
            g[cols.index(termino)] += 1

        ee = abs(cambio) * np.sqrt(
            max(float(g @ cov @ g), 0)
        )

        filas.append({
            "Horizonte": h,
            "Respuesta": cambio * r,
            "IC inferior": cambio * r - 1.96 * ee,
            "IC superior": cambio * r + 1.96 * ee,
        })

    return pd.DataFrame(filas)


def pca_cluster(base, cols, k):
    d = muestra(base, cols)

    X = StandardScaler().fit_transform(
        d[cols]
    )

    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)

    grupos = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42,
    ).fit_predict(X)

    pts = pd.DataFrame({
        "Mes": d.mes.astype(str),
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Grupo": grupos.astype(str),
    })

    cargas = pd.DataFrame(
        pca.components_.T,
        index=[
            etiqueta(c) for c in cols
        ],
        columns=["PC1", "PC2"],
    )

    return (
        pts,
        cargas,
        pca.explained_variance_ratio_,
        silhouette_score(X, grupos),
    )


def var_condicionado(base, cols, lags, horizonte):
    d = muestra(base, cols)
    filas = []

    for c in cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pk = kpss(
                d[c],
                regression="c",
                nlags="auto",
            )[1]

        filas.append({
            "Serie": etiqueta(c),
            "p ADF nivel": adfuller(
                d[c],
                autolag="AIC",
            )[1],
            "p KPSS nivel": pk,
            "p ADF diferencia": adfuller(
                d[c].diff().dropna(),
                autolag="AIC",
            )[1],
        })

    tests = pd.DataFrame(filas)

    I0 = (
        tests["p ADF nivel"].lt(.05)
        & tests["p KPSS nivel"].ge(.05)
    ).all()

    I1 = (
        tests["p ADF nivel"].ge(.05)
        & tests["p ADF diferencia"].lt(.05)
    ).all()

    if I0:
        mod = VAR(d[cols]).fit(lags)

        if not mod.is_stable():
            raise ValueError("VAR inestable")

        return (
            tests,
            "VAR",
            mod,
            mod.irf(horizonte).irfs,
        )

    if I1:
        j = coint_johansen(
            d[cols],
            det_order=0,
            k_ar_diff=lags,
        )

        traza = pd.DataFrame({
            "Traza": j.lr1,
            "Crítico 95 %": j.cvt[:, 1],
        })

        return (
            tests,
            "Johansen (evaluación previa a VECM)",
            None,
            traza,
        )

    return (
        tests,
        "Integración no concluyente: sistema no estimado",
        None,
        None,
    )


def evaluar_ml(base, elegidos):
    d = muestra(base, [Y, T, P])

    tabla = pd.DataFrame({
        "mes": d.mes,
        "objetivo": d[Y],
    })

    for c in (Y, T, P):
        for j in (1, 2):
            tabla[f"{c}_L{j}"] = d[c].shift(j)

    tabla = tabla.dropna().reset_index(drop=True)

    features = [
        c for c in tabla
        if c not in ("mes", "objetivo")
    ]

    inicio = len(tabla) - 26

    factories = {
        "Ridge": lambda: make_pipeline(
            StandardScaler(),
            Ridge(alpha=1),
        ),
        "Lasso": lambda: make_pipeline(
            StandardScaler(),
            Lasso(alpha=.01, max_iter=15000),
        ),
        "SVR": lambda: make_pipeline(
            StandardScaler(),
            SVR(C=1, epsilon=.1),
        ),
        "MLP": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(12,),
                max_iter=600,
                random_state=42,
            ),
        ),
    }

    filas = []

    for fin in range(inicio, len(tabla)):
        ent = tabla.iloc[:fin]
        prueba = tabla.iloc[[fin]]

        X = ent[features].to_numpy(float)
        Xt = prueba[features].to_numpy(float)
        y = ent.objetivo.to_numpy(float)

        pred = {
            "Media histórica": float(y.mean())
        }

        arX = sm.add_constant(
            ent[[f"{Y}_L1"]].to_numpy(float),
            has_constant="add",
        )

        art = sm.add_constant(
            prueba[[f"{Y}_L1"]].to_numpy(float),
            has_constant="add",
        )

        pred["AR(1)"] = float(
            sm.OLS(y, arX).fit().predict(art)[0]
        )

        pred["ARX"] = float(
            sm.OLS(
                y,
                sm.add_constant(
                    X,
                    has_constant="add",
                ),
            ).fit().predict(
                sm.add_constant(
                    Xt,
                    has_constant="add",
                )
            )[0]
        )

        for nombre_m, maker in factories.items():
            if nombre_m in elegidos:
                pred[nombre_m] = float(
                    maker().fit(X, y).predict(Xt)[0]
                )

        for nombre_m, valor in pred.items():
            if nombre_m in elegidos:
                filas.append({
                    "Mes": str(prueba.mes.iloc[0]),
                    "Modelo": nombre_m,
                    "Real": float(prueba.objetivo.iloc[0]),
                    "Pronóstico": valor,
                })

    pred = pd.DataFrame(filas)

    if pred.empty:
        raise ValueError("Selecciona modelos")

    metricas = pd.DataFrame([
        {
            "Modelo": m,
            "N": len(z),
            "RMSE": np.sqrt(
                mean_squared_error(
                    z.Real,
                    z.Pronóstico,
                )
            ),
            "MAE": mean_absolute_error(
                z.Real,
                z.Pronóstico,
            ),
        }
        for m, z in pred.groupby("Modelo")
    ]).sort_values("RMSE")

    return metricas, pred, tabla, features, inicio


def xai(tabla, features, inicio, alg):
    models = {
        "Ridge": Ridge(alpha=1),
        "Lasso": Lasso(
            alpha=.01,
            max_iter=15000,
        ),
        "SVR": SVR(
            C=1,
            epsilon=.1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(12,),
            max_iter=600,
            random_state=42,
        ),
    }

    model = make_pipeline(
        StandardScaler(),
        models[alg],
    )

    model.fit(
        tabla.iloc[:inicio][features],
        tabla.iloc[:inicio].objetivo,
    )

    r = permutation_importance(
        model,
        tabla.iloc[inicio:][features],
        tabla.iloc[inicio:].objetivo,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_squared_error",
    )

    return pd.DataFrame({
        "Predictor": features,
        "Importancia": r.importances_mean,
        "Desviación": r.importances_std,
    }).sort_values("Importancia")


def semaforo(base):
    d = muestra(base, [T, P])
    filas = []

    for i in range(12, len(d)):
        old = d.iloc[:i]
        row = d.iloc[i]

        ut = old[T].quantile(.9)
        up = old[P].quantile(.9)

        nivel = (
            int(row[T] > ut)
            + int(row[P] > up)
        )

        filas.append({
            "Mes": str(row.mes),
            "RNDC": row[T],
            "SIPSA": row[P],
            "Umbral RNDC": ut,
            "Umbral SIPSA": up,
            "Clasificación": (
                "Verde", "Amarillo", "Rojo"
            )[nivel],
        })

    return pd.DataFrame(filas)


def grafico(fig, alto=440):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101b30",
        plot_bgcolor="#11213b",
        font=dict(color="#e8f0f7"),
        height=alto,
        margin=dict(
            t=58,
            l=24,
            r=20,
            b=35,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    return fig


def lineas(base, cols, titulo, estandar=False):
    z = base[
        ["mes"] + cols
    ].copy()

    z["mes"] = z.mes.astype(str)

    if estandar:
        for c in cols:
            desv = z[c].std()

            if pd.notna(desv) and desv > 0:
                z[c] = (
                    z[c] - z[c].mean()
                ) / desv

    long = z.melt(
        "mes",
        var_name="Serie",
        value_name="Valor",
    ).dropna()

    long.Serie = long.Serie.map(etiqueta)

    fig = px.line(
        long,
        x="mes",
        y="Valor",
        color="Serie",
        facet_row=(
            "Serie"
            if len(cols) > 1 and not estandar
            else None
        ),
        title=titulo,
    )

    if len(cols) > 1 and not estandar:
        fig.update_yaxes(matches=None)

    altura = (
        max(440, 205 * len(cols))
        if not estandar
        else 450
    )

    return grafico(fig, altura)


def csv(d, nombre_archivo):
    z = d.copy()

    if "mes" in z:
        z["mes"] = z.mes.astype(str)

    st.download_button(
        "Descargar CSV",
        z.to_csv(index=False).encode("utf-8-sig"),
        file_name=nombre_archivo,
        mime="text/csv",
    )


def html(fig, nombre_archivo):
    st.download_button(
        "Exportar gráfico",
        fig.to_html(
            include_plotlyjs="cdn"
        ).encode("utf-8"),
        file_name=nombre_archivo,
        mime="text/html",
    )


def lista_faltantes(base, cols):
    try:
        muestra(base, cols)
        return []
    except Exception as e:
        return [str(e)]


PAGINAS = [
    "🏠 Inicio",
    "📊 Exploración",
    "🗃️ Datos y calidad",
    "📈 Econometría",
    "🧩 PCA y K-Means",
    "🔁 VAR y cointegración",
    "🤖 Predicción y XAI",
    "🎛️ Simulaciones y Pasto",
    "📑 Informe",
    "💬 Consultas",
    "© Créditos y metodología",
]

with st.sidebar:
    st.markdown("### 📊 OBSERVATORIO")
    st.caption("Pasto–Colombia · 2018–2024")

    pagina = st.radio(
        "Navegación",
        PAGINAS,
    )

    st.divider()

    if st.button("Actualizar fuentes"):
        cargar.clear()
        st.session_state.pop("ml_obs", None)
        st.rerun()


BASE, AUDITORIA, INCIDENCIAS = cargar()

DISP = [
    c for c in (Y, T, P, I)
    if c in BASE and BASE[c].notna().any()
]

OTRAS = [
    c for c in NOMBRES
    if c not in DISP
    and c not in (Y, T, P, I, "Ajustado")
    and c in BASE
    and BASE[c].notna().any()
]


if pagina == "🏠 Inicio":
    st.title("Observatorio Económico de Pasto y Nariño")

    st.markdown("""
<div class="hero">
<h2>¡Bienvenidos! 👋</h2>
<p>
Explora la inflación de Pasto frente a Colombia, los fletes
regionales y los precios mayoristas de El Potrerillo.
Contrasta indicadores, conoce los patrones mensuales,
experimenta con modelos econométricos y descarga
resultados verificables.
</p>
</div>
""", unsafe_allow_html=True)

    for card, c in zip(st.columns(4), (Y, T, P, I)):
        n = int(BASE[c].count()) if c in BASE else 0

        card.metric(
            etiqueta(c),
            f"{n} meses" if n else "Sin serie",
        )

    if DISP:
        a, b = st.columns([1.7, 1])

        with a:
            c = st.selectbox(
                "Indicador",
                DISP,
                format_func=etiqueta,
            )

            sub = BASE.dropna(subset=[c])

            st.metric(
                f"Último mes: {sub.mes.iloc[-1]}",
                f"{sub[c].iloc[-1]:+.3f}",
            )

            lineas(
                BASE,
                [c],
                "Dinámica mensual",
            )

        with b:
            cob = pd.DataFrame({
                "Indicador": [
                    etiqueta(c) for c in DISP
                ],
                "Meses": [
                    int(BASE[c].count()) for c in DISP
                ],
            })

            grafico(
                px.bar(
                    cob,
                    x="Meses",
                    y="Indicador",
                    orientation="h",
                    title="Cobertura de datos",
                )
            )

    if len(DISP) > 1:
        sel = st.multiselect(
            "Comparador",
            DISP,
            default=DISP,
            format_func=etiqueta,
        )

        if sel:
            lineas(
                BASE,
                sel,
                "Patrones estandarizados",
                True,
            )

    a, b, c = st.columns(3)

    a.markdown(
        "**📊 Explorar:** evolución, correlaciones, distribución y estacionalidad."
    )
    b.markdown(
        "**📈 Modelar:** ARDL/HAC, diagnósticos y estabilidad."
    )
    c.markdown(
        "**🧠 Analizar:** PCA, clustering, ML, XAI y escenarios."
    )


elif pagina == "📊 Exploración":
    st.title("Explorador de indicadores")

    elegir = st.multiselect(
        "Variables",
        DISP + OTRAS,
        default=DISP,
        format_func=etiqueta,
    )

    meses = [str(m) for m in M84]

    inicio, fin = st.select_slider(
        "Periodo de visualización",
        options=meses,
        value=(meses[0], meses[-1]),
    )

    vista = BASE.loc[
        BASE.mes.between(
            pd.Period(inicio),
            pd.Period(fin),
        )
    ]

    if elegir:
        a, b, c, d = st.tabs([
            "Series",
            "Descriptivos",
            "Correlación",
            "Distribuciones",
        ])

        with a:
            fig = lineas(
                vista,
                elegir,
                "Indicadores en unidades originales",
            )

            html(fig, "series_pasto.html")

            if len(elegir) > 1:
                lineas(
                    vista,
                    elegir,
                    "Comparación normalizada",
                    True,
                )

        with b:
            desc = (
                vista[elegir]
                .describe()
                .T
                .reset_index(names="Indicador")
            )

            desc.Indicador = desc.Indicador.map(etiqueta)

            st.dataframe(
                desc,
                hide_index=True,
                use_container_width=True,
            )

            csv(desc, "descriptivos.csv")

            columna = st.selectbox(
                "Serie año × mes",
                elegir,
                format_func=etiqueta,
            )

            z = vista[
                ["mes", columna]
            ].dropna().copy()

            z["Año"] = z.mes.dt.year
            z["Mes"] = z.mes.dt.month

            matriz = z.pivot(
                index="Año",
                columns="Mes",
                values=columna,
            )

            grafico(
                px.imshow(
                    matriz,
                    aspect="auto",
                    text_auto=".2f",
                    title="Calendario térmico mensual",
                ),
                460,
            )

            s = vista[columna].dropna()

            if len(s) > 18:
                grafico(
                    px.bar(
                        x=list(range(13)),
                        y=acf(
                            s,
                            nlags=12,
                            fft=False,
                        ),
                        title="Autocorrelación (ACF)",
                    )
                )

        with c:
            if len(elegir) > 1:
                metodo = st.radio(
                    "Método",
                    ["pearson", "spearman"],
                    horizontal=True,
                )

                corr = vista[elegir].corr(
                    method=metodo,
                    min_periods=12,
                )

                grafico(
                    px.imshow(
                        corr,
                        zmin=-1,
                        zmax=1,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Correlaciones",
                    )
                )

                x, y = st.columns(2)

                vx = x.selectbox(
                    "Eje X",
                    elegir,
                    format_func=etiqueta,
                )

                vy = y.selectbox(
                    "Eje Y",
                    elegir,
                    index=1,
                    format_func=etiqueta,
                )

                grafico(
                    px.scatter(
                        vista.dropna(
                            subset=[vx, vy]
                        ),
                        x=vx,
                        y=vy,
                        title="Relación bivariada",
                    )
                )

        with d:
            long = vista[elegir].melt(
                var_name="Serie",
                value_name="Valor",
            ).dropna()

            long.Serie = long.Serie.map(etiqueta)

            grafico(
                px.box(
                    long,
                    x="Serie",
                    y="Valor",
                    color="Serie",
                    points="outliers",
                    title="Variación y extremos",
                ),
                520,
            )

            hist = st.selectbox(
                "Histograma",
                elegir,
                format_func=etiqueta,
            )

            grafico(
                px.histogram(
                    vista,
                    x=hist,
                    nbins=18,
                    title=f"Histograma: {etiqueta(hist)}",
                )
            )

        csv(
            vista[["mes"] + elegir],
            "series_filtradas.csv",
        )


elif pagina == "🗃️ Datos y calidad":
    st.title("Cobertura y trazabilidad")

    cob = []

    for c in (Y, T, P, I):
        s = (
            BASE.loc[BASE[c].notna(), "mes"]
            if c in BASE
            else pd.Series(dtype="object")
        )

        cob.append({
            "Indicador": etiqueta(c),
            "Meses": len(s),
            "Desde": str(s.min()) if len(s) else "—",
            "Hasta": str(s.max()) if len(s) else "—",
        })

    st.dataframe(
        pd.DataFrame(cob),
        hide_index=True,
        use_container_width=True,
    )

    st.dataframe(
        AUDITORIA,
        hide_index=True,
        use_container_width=True,
    )

    if INCIDENCIAS:
        with st.expander(
            "Diagnóstico exacto de fuentes"
        ):
            for fuente, error in INCIDENCIAS.items():
                st.write(
                    f"**{fuente}:** {error}"
                )

    grupos = [
        (
            "Comparabilidad de SIPSA",
            [
                "cobertura_indice_pct",
                "n_productos_indice",
            ],
        ),
        (
            "Alternativas SIPSA",
            [
                P,
                "PM_no_ponderado_t",
                "PM_ponderado_lagQ_t",
                "PM_100",
                "PM_95",
            ],
        ),
        (
            "Alternativas RNDC",
            [
                T,
                "VAR_FLETE_FISHER_PCT",
                "VAR_FLETE_CRUDO_PCT",
            ],
        ),
        (
            "Coberturas RNDC",
            [
                "COB_VALOR_T_PCT",
                "COB_VIAJES_T_PCT",
                "CELDAS_COMPARABLES",
            ],
        ),
    ]

    for cabecera, cols in grupos:
        cols = [
            c for c in cols
            if c in BASE
            and BASE[c].notna().any()
        ]

        if cols:
            st.subheader(cabecera)
            lineas(BASE, cols, cabecera)

    csv(
        BASE,
        "base_integrada_2018_2024.csv",
    )

    csv(
        AUDITORIA,
        "auditoria_fuentes.csv",
    )


elif pagina == "📈 Econometría":
    st.title("Estimación econométrica")

    a, b, c, d = st.tabs([
        "ADF / KPSS",
        "DL y ARDL · HAC",
        "Diagnósticos",
        "Robustez y estabilidad",
    ])

    with a:
        if DISP:
            tabla = estacionariedad(BASE, DISP)

            st.dataframe(
                tabla,
                hide_index=True,
                use_container_width=True,
            )

            csv(tabla, "adf_kpss.csv")

    with b:
        st.latex(
            r"B_t=\alpha+\rho B_{t-1}"
            r"+\sum_{j=0}^{L}\beta_j T_{t-j}"
            r"+\sum_{j=0}^{L}\gamma_j PM_{t-j}+u_t"
        )

        problemas = lista_faltantes(
            BASE,
            [Y, T, P],
        )

        if problemas:
            st.error(
                "Modelo no disponible: "
                + " | ".join(problemas)
            )

        else:
            co1, co2, co3 = st.columns(3)

            familia = co1.selectbox(
                "Modelo",
                ["ARDL", "DL"],
            )

            L = co2.slider(
                "Rezagos",
                0,
                3,
                2,
            )

            ancho = co3.slider(
                "HAC maxlags",
                0,
                8,
                3,
            )

            try:
                m, o, z, X = ajustar(
                    BASE,
                    familia,
                    L,
                    ancho,
                )

                j = conjuntas(m, L)

                indicadores = [
                    ("N", len(z)),
                    ("R² ajustado", f"{o.rsquared_adj:.4f}"),
                    ("AIC", f"{o.aic:.2f}"),
                    ("BIC", f"{o.bic:.2f}"),
                ]

                for card, (texto, valor) in zip(
                    st.columns(4),
                    indicadores,
                ):
                    card.metric(texto, valor)

                coef = tabla_coef(m)

                st.dataframe(
                    coef,
                    hide_index=True,
                    use_container_width=True,
                )

                st.dataframe(
                    j,
                    hide_index=True,
                    use_container_width=True,
                )

                ajustadas = z[
                    ["mes", Y]
                ].copy()

                ajustadas["Ajustado"] = (
                    o.fittedvalues.to_numpy()
                )

                lineas(
                    ajustadas,
                    [Y, "Ajustado"],
                    "Brecha observada vs. ajuste dentro de muestra",
                )

                coef_g = coef.loc[
                    coef.Regresor.ne("const")
                ].copy()

                grafico(
                    px.scatter(
                        coef_g,
                        x="Coeficiente",
                        y="Regresor",
                        error_x=(
                            coef_g["IC superior"].to_numpy()
                            - coef_g["Coeficiente"].to_numpy()
                        ),
                        error_x_minus=(
                            coef_g["Coeficiente"].to_numpy()
                            - coef_g["IC inferior"].to_numpy()
                        ),
                        title="Coeficientes HAC e IC 95 %",
                    )
                )

                if (
                    familia == "ARDL"
                    and L == 2
                    and ancho == 3
                ):
                    referencia = [
                        (
                            "Coef RNDC(t)",
                            .00913,
                            float(m.params["T0"]),
                        ),
                        (
                            "p RNDC(t)",
                            .03750,
                            float(m.pvalues["T0"]),
                        ),
                        (
                            "p bloque RNDC",
                            .14273,
                            float(j.iloc[0]["p conjunta HAC"]),
                        ),
                        (
                            "p bloque SIPSA",
                            .23749,
                            float(j.iloc[1]["p conjunta HAC"]),
                        ),
                    ]

                    ref = pd.DataFrame([
                        {
                            "Control": k,
                            "Artículo": r,
                            "App": v,
                            "Diferencia": abs(r - v),
                        }
                        for k, r, v in referencia
                    ])

                    with st.expander(
                        "Contraste de reproducción frente al artículo"
                    ):
                        st.dataframe(
                            ref,
                            hide_index=True,
                        )

                csv(
                    coef,
                    "coeficientes_hac.csv",
                )

                csv(
                    j,
                    "contrastes_conjuntos.csv",
                )

            except Exception as e:
                st.error(
                    f"Estimación: {e}"
                )

    with c:
        try:
            _, o, z, X = ajustar(BASE)
            diag, residuos, vif = diagnosticos(o, z, X)

            st.dataframe(
                diag.T,
                use_container_width=True,
            )

            st.dataframe(
                vif,
                hide_index=True,
            )

            iz, de = st.columns(2)

            with iz:
                grafico(
                    px.scatter(
                        residuos,
                        x="Ajustado",
                        y="Residuo",
                        title="Residuos y ajustes",
                    )
                )

            with de:
                grafico(
                    px.histogram(
                        residuos,
                        x="Residuo",
                        nbins=15,
                        title="Histograma residual",
                    )
                )

            grafico(
                px.bar(
                    residuos.assign(
                        Mes=residuos.mes.astype(str)
                    ),
                    x="Mes",
                    y="Cook",
                    title="Distancia de Cook",
                )
            )

            csv(
                residuos,
                "residuos_ardl.csv",
            )

        except Exception as e:
            st.info(
                f"Diagnóstico no calculable: {e}"
            )

    with d:
        if st.button(
            "Calcular ocho especificaciones y sensibilidad",
            type="primary",
        ):
            r = robustez(BASE)

            st.dataframe(
                r,
                hide_index=True,
                use_container_width=True,
            )

            csv(r, "robustez.csv")

        x, y = st.columns(2)

        with x:
            if st.button("MCO vs. Huber"):
                try:
                    st.dataframe(
                        huber(BASE),
                        hide_index=True,
                    )
                except Exception as e:
                    st.error(str(e))

        with y:
            if st.button("Ventanas móviles (36 meses)"):
                try:
                    r = ventanas(BASE)

                    st.metric(
                        "Ventanas",
                        len(r),
                    )

                    grafico(
                        px.line(
                            r,
                            x="Fin",
                            y=[
                                "Suma RNDC",
                                "Suma SIPSA",
                            ],
                            title="Estabilidad de sumas",
                        )
                    )

                    st.dataframe(
                        r,
                        hide_index=True,
                    )

                    csv(
                        r,
                        "ventanas.csv",
                    )

                except Exception as e:
                    st.error(str(e))


elif pagina == "🧩 PCA y K-Means":
    st.title("Análisis de patrones mensuales")

    cols = st.multiselect(
        "Indicadores",
        DISP,
        default=DISP,
        format_func=etiqueta,
    )

    k = st.slider(
        "Grupos K-Means",
        2,
        6,
        3,
    )

    if len(cols) >= 2:
        try:
            pts, cargas, var, sil = pca_cluster(
                BASE,
                cols,
                k,
            )

            a, b = st.columns(2)

            a.metric(
                "Varianza PC1+PC2",
                f"{sum(var):.1%}",
            )

            b.metric(
                "Silhouette",
                f"{sil:.3f}",
            )

            fig = px.scatter(
                pts,
                x="PC1",
                y="PC2",
                color="Grupo",
                hover_data=["Mes"],
                title="PCA y clusters de meses",
            )

            amplitud = max(
                np.abs(
                    pts[["PC1", "PC2"]]
                    .to_numpy()
                ).max(),
                1,
            ) * .5

            for nombre_c, fila in cargas.iterrows():
                fig.add_annotation(
                    x=float(fila.PC1 * amplitud),
                    y=float(fila.PC2 * amplitud),
                    ax=0,
                    ay=0,
                    showarrow=True,
                    arrowhead=2,
                    text=str(nombre_c),
                )

            grafico(fig, 540)

            st.dataframe(
                cargas,
                use_container_width=True,
            )

            st.dataframe(
                pts,
                hide_index=True,
            )

            csv(
                pts,
                "clusters_pca.csv",
            )

        except Exception as e:
            st.info(
                f"PCA / clustering no ejecutable: {e}"
            )


elif pagina == "🔁 VAR y cointegración":
    st.title("Sistemas de series temporales")

    cols = st.multiselect(
        "Variables endógenas",
        DISP,
        default=DISP[:3],
        max_selections=3,
        format_func=etiqueta,
    )

    lag = st.slider(
        "Rezagos del VAR/Johansen",
        1,
        3,
        1,
    )

    if len(cols) >= 2 and st.button(
        "Evaluar el sistema"
    ):
        try:
            pruebas, familia, modelo, resultado = (
                var_condicionado(
                    BASE,
                    cols,
                    lag,
                    8,
                )
            )

            st.dataframe(
                pruebas,
                hide_index=True,
                use_container_width=True,
            )

            st.write(familia)

            if familia == "VAR":
                st.metric(
                    "AIC",
                    f"{modelo.aic:.4f}",
                )

                x, y = st.columns(2)

                impulso = x.selectbox(
                    "Impulso en",
                    cols,
                    format_func=etiqueta,
                )

                respuesta = y.selectbox(
                    "Respuesta de",
                    cols,
                    format_func=etiqueta,
                )

                r = resultado[
                    :,
                    cols.index(respuesta),
                    cols.index(impulso),
                ]

                grafico(
                    px.line(
                        x=list(range(len(r))),
                        y=r,
                        markers=True,
                        title="IRF reducida exploratoria",
                    )
                )

            elif resultado is not None:
                st.dataframe(
                    resultado,
                    hide_index=True,
                )

        except Exception as e:
            st.info(str(e))


elif pagina == "🤖 Predicción y XAI":
    st.title("Machine Learning y explicabilidad")

    modelos = st.multiselect(
        "Modelos",
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
            "Evaluar con ventanas expansivas",
            type="primary",
        )
        and modelos
    ):
        try:
            st.session_state["ml_obs"] = evaluar_ml(
                BASE,
                modelos,
            )
        except Exception as e:
            st.error(str(e))

    if "ml_obs" in st.session_state:
        metricas, pred, tabla, features, inicio = (
            st.session_state["ml_obs"]
        )

        st.dataframe(
            metricas,
            hide_index=True,
            use_container_width=True,
        )

        observado = (
            pred.drop_duplicates("Mes")
            [["Mes", "Real"]]
            .rename(columns={"Real": "Valor"})
        )

        observado["Modelo"] = "Observado"

        estimado = (
            pred[["Mes", "Modelo", "Pronóstico"]]
            .rename(columns={"Pronóstico": "Valor"})
        )

        grafico(
            px.line(
                pd.concat([estimado, observado]),
                x="Mes",
                y="Valor",
                color="Modelo",
                title="Predicciones vs observado",
            )
        )

        csv(
            pred,
            "predicciones_ml.csv",
        )

        alg = st.selectbox(
            "Modelo para XAI",
            [
                "Ridge",
                "Lasso",
                "SVR",
                "MLP",
            ],
        )

        if st.button(
            "Calcular XAI por permutaciones"
        ):
            imp = xai(
                tabla,
                features,
                inicio,
                alg,
            )

            grafico(
                px.bar(
                    imp,
                    x="Importancia",
                    y="Predictor",
                    orientation="h",
                    title="Importancia por permutación",
                )
            )

            st.dataframe(
                imp,
                hide_index=True,
            )

            csv(
                imp,
                "xai.csv",
            )


elif pagina == "🎛️ Simulaciones y Pasto":
    st.title("Escenarios y Focus Pasto")

    a, b, c = st.tabs([
        "Simulador ARDL",
        "Episodio 2021",
        "Semáforo histórico",
    ])

    with a:
        try:
            mod, _, _, _ = ajustar(BASE)

            var = st.selectbox(
                "Indicador del escenario",
                [T, P],
                format_func=etiqueta,
            )

            cambio = st.number_input(
                "Cambio hipotético en p. p.",
                value=1.0,
                step=.5,
            )

            horizonte = st.slider(
                "Meses de propagación",
                1,
                15,
                9,
            )

            r = respuesta_dinamica(
                mod,
                "T" if var == T else "P",
                cambio,
                horizonte,
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=r.Horizonte,
                    y=r["IC superior"],
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=r.Horizonte,
                    y=r["IC inferior"],
                    fill="tonexty",
                    line=dict(width=0),
                    name="IC 95 %",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=r.Horizonte,
                    y=r.Respuesta,
                    mode="lines+markers",
                    name="Respuesta condicional",
                )
            )

            fig.add_hline(
                y=0,
                line_dash="dot",
            )

            grafico(fig)

            st.dataframe(
                r,
                hide_index=True,
            )

            csv(
                r,
                "simulador_ardl.csv",
            )

        except Exception as e:
            st.info(str(e))

    with b:
        st.metric(
            "Cambio de abastecimiento primeras quincenas de 2021, DANE",
            "−56,98 %",
        )

        st.write(
            "El Potrerillo: 4.618 toneladas (1–15 de abril) "
            "y 1.987 toneladas (1–15 de mayo)."
        )

        ep = BASE.loc[
            BASE.mes.between(
                pd.Period("2021-04"),
                pd.Period("2021-08"),
            )
        ]

        if DISP:
            lineas(
                ep,
                DISP,
                "Series durante el episodio de 2021",
            )

            st.dataframe(
                ep[["mes"] + DISP].assign(
                    mes=lambda q: q.mes.astype(str)
                ),
                hide_index=True,
            )

    with c:
        try:
            r = semaforo(BASE)

            grafico(
                px.scatter(
                    r,
                    x="RNDC",
                    y="SIPSA",
                    color="Clasificación",
                    hover_data=["Mes"],
                    title="Percentiles históricos",
                )
            )

            st.dataframe(
                r,
                hide_index=True,
            )

            csv(
                r,
                "semaforo_historico.csv",
            )

        except Exception as e:
            st.info(str(e))


elif pagina == "📑 Informe":
    st.title("Informe automático")

    partes = [
        "# Informe del Observatorio Pasto–Nariño",
        f"Generado: {datetime.now():%Y-%m-%d %H:%M}",
        "## Cobertura",
    ]

    for c in (Y, T, P, I):
        partes.append(
            f"- {etiqueta(c)}: "
            f"{int(BASE[c].count()) if c in BASE else 0} meses"
        )

    partes.append("## Estadísticos descriptivos")

    for c in DISP:
        s = BASE[c].dropna()

        partes.append(
            f"- {etiqueta(c)}: N={len(s)}, "
            f"media={s.mean():.5f}, "
            f"sd={s.std():.5f}, "
            f"min={s.min():.5f}, "
            f"max={s.max():.5f}"
        )

    try:
        mod, o, z, _ = ajustar(BASE)
        b = conjuntas(mod)

        partes += [
            "## ARDL L2 · HAC(3)",
            (
                f"N={len(z)}; "
                f"R² ajustado={o.rsquared_adj:.5f}; "
                f"AIC={o.aic:.5f}"
            ),
            (
                f"Coef RNDC contemporáneo="
                f"{mod.params['T0']:.6f}; "
                f"p HAC={mod.pvalues['T0']:.5f}"
            ),
            (
                f"p conjunta RNDC="
                f"{b.iloc[0]['p conjunta HAC']:.5f}; "
                f"p conjunta SIPSA="
                f"{b.iloc[1]['p conjunta HAC']:.5f}"
            ),
        ]

    except Exception as e:
        partes += [
            "## ARDL no estimado",
            str(e),
        ]

    partes += [
        "## Fuentes",
        f"DANE y RNDC procesado: https://github.com/{REPO}",
        "## Alcance",
        (
            "No hay imputaciones. Las estimaciones son "
            "asociativas, no causales."
        ),
    ]

    texto = "\n\n".join(partes)

    st.markdown(texto)

    st.download_button(
        "Descargar informe Markdown",
        texto.encode("utf-8"),
        file_name="informe_pasto.md",
        mime="text/markdown",
    )

    csv(
        BASE,
        "base_integrada.csv",
    )

    csv(
        AUDITORIA,
        "auditoria_sha.csv",
    )


elif pagina == "💬 Consultas":
    st.title("Asistente de consultas numéricas")

    if "chat_obs" not in st.session_state:
        st.session_state.chat_obs = []

    for mensaje in st.session_state.chat_obs:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input(
        "Ej.: RNDC en 2021-05, media SIPSA, coeficiente ARDL"
    )

    if pregunta:
        q = normal(pregunta)

        if any(
            x in q
            for x in (
                "sipsa", "mayorista", "potrerillo"
            )
        ):
            var = P
        elif any(
            x in q
            for x in ("rndc", "flete")
        ):
            var = T
        elif "ictc" in q:
            var = I
        else:
            var = Y

        match = re.search(
            r"(20(?:18|19|2[0-4]))[-/ ](0?[1-9]|1[0-2])",
            q,
        )

        mes = (
            pd.Period(
                f"{match.group(1)}-{int(match.group(2)):02d}"
            )
            if match else None
        )

        if mes is None:
            meses = (
                "enero febrero marzo abril mayo junio "
                "julio agosto septiembre octubre "
                "noviembre diciembre"
            ).split()

            for j, n in enumerate(meses, 1):
                m = re.search(
                    rf"\b{n}\s+(?:de\s+)?(20\d{{2}})\b",
                    q,
                )

                if m:
                    mes = pd.Period(
                        f"{m.group(1)}-{j:02d}"
                    )
                    break

        if any(
            x in q
            for x in (
                "causal", "causa", "bloqueo"
            )
        ):
            respuesta = (
                "El observatorio no identifica efectos "
                "causales ni cierres viales."
            )

        elif "coeficiente" in q or "ardl" in q:
            try:
                m, _, _, _ = ajustar(BASE)

                respuesta = (
                    f"ARDL L2: RNDC(t)="
                    f"{m.params['T0']:.6f}, "
                    f"p HAC={m.pvalues['T0']:.5f}."
                )

            except Exception as e:
                respuesta = (
                    f"Modelo no disponible: {e}"
                )

        elif var not in BASE or BASE[var].notna().sum() == 0:
            respuesta = (
                f"No hay valores verificables de "
                f"{etiqueta(var)}."
            )

        elif mes is not None:
            z = BASE.loc[
                BASE.mes.eq(mes),
                var,
            ]

            respuesta = (
                f"{etiqueta(var)} en {mes}: "
                f"**{z.iloc[0]:.5f}**"
                if len(z) and pd.notna(z.iloc[0])
                else f"Sin observación para {mes}."
            )

        elif any(
            x in q
            for x in ("promedio", "media")
        ):
            respuesta = (
                f"{etiqueta(var)}: "
                f"media={BASE[var].mean():.5f}, "
                f"n={BASE[var].count()}."
            )

        elif any(
            x in q
            for x in ("maximo", "minimo", "pico")
        ):
            z = BASE[
                ["mes", var]
            ].dropna()

            i = (
                z[var].idxmin()
                if "minimo" in q
                else z[var].idxmax()
            )

            respuesta = (
                f"{etiqueta(var)}: "
                f"{z.loc[i, var]:.5f} "
                f"en {z.loc[i, 'mes']}."
            )

        else:
            respuesta = (
                "Consulta por variable y mes (AAAA-MM), "
                "promedio, máximo, mínimo o ARDL."
            )

        st.session_state.chat_obs += [
            {
                "role": "user",
                "content": pregunta,
            },
            {
                "role": "assistant",
                "content": respuesta,
            },
        ]

        st.rerun()


else:
    st.title("Fuentes, metodología y créditos")

    st.markdown(f"""
**Universidad de Nariño · Economía · Investigación regional**

**Investigación:** Costos del transporte de carga, precios mayoristas
de alimentos y brecha inflacionaria Pasto–Colombia, 2018–2024.

**Fuentes:** IPC, SIPSA-P, SIPSA-A e ICTC del DANE; registros RNDC del
Ministerio de Transporte. El índice regional de fletes es construido
por los investigadores. GitHub es repositorio de respaldo, no fuente primaria.

**Cobertura:** 84 meses para SIPSA, IPC e ICTC; 59 para RNDC y
56 para la comparación DL/ARDL. Se respetan las unidades:
la brecha en puntos porcentuales; RNDC, SIPSA e ICTC en variaciones
porcentuales.

**Reproducción:** las series RNDC y SIPSA se contrastan con SHA-256
fijo. Para IPC/ICTC se utiliza primero un CSV procesado y su
manifiesto si están publicados; de lo contrario se reconstruyen
desde Excel originales con hashes fijados en el Colab. El CSV
procesado sin manifiesto tiene validación de estructura y controles,
no un hash histórico independiente. El estado de cada archivo se
presenta en «Datos y calidad».

**Interpretación:** ARDL/HAC y sus sumas son asociaciones, no causalidad.
ICTC es un contraste nacional y no equivale al índice regional RNDC.
Pasto es la ciudad estudiada; no se atribuye su inflación a todos
los municipios de Nariño. No se imputan observaciones. PCA/K-Means
agrupan meses, el VAR se estima solo bajo supuestos adecuados,
y ML no se presenta como pronóstico en tiempo real.

**Fuera de alcance:** mapas coropléticos sin indicadores geográficos
municipales compatibles, Moran/LISA sin panel espacial, RAG documental
y detección operativa de bloqueos. El RAG se abordará después.

**Repositorios:** [Indicadores](https://github.com/{REPO}) ·
[Respaldo SIPSA](https://github.com/{BACKUP}/releases/tag/{TAG}) ·
[Colab de investigación](https://colab.research.google.com/drive/1Tu3ZzP1L8XlUejqbS_0eXgZ8QszqiMr7).
""")

    st.dataframe(
        AUDITORIA,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Incidencias de fuentes"):
        for k, v in INCIDENCIAS.items():
            st.write(
                f"**{k}**: {v}"
            )


st.divider()

st.caption(
    "Observatorio Pasto–Colombia · Datos verificables "
    "del DANE y registros RNDC · 2018–2024"
)
