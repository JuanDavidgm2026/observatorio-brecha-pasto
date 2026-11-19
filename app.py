
from __future__ import annotations

import hashlib
import html
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
from scipy import stats
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
    acorr_breusch_godfrey, acorr_ljungbox, breaks_cusumolsresid,
    het_breuschpagan, linear_reset,
)
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="Observatorio Pasto–Nariño", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp{background:linear-gradient(145deg,#0b1526,#121e35);color:#e9f3fb}
section[data-testid="stSidebar"]{background:#0e1a2f;border-right:1px solid #304660}
h1,h2,h3{color:#9ceaf0!important;letter-spacing:-.025em}
[data-testid="stMetric"]{background:#1b2e46;border:1px solid #38546b;border-radius:15px;padding:15px}
[data-testid="stTabs"] button{font-weight:600}
.stButton button[kind="primary"]{background:linear-gradient(95deg,#087e9b,#7354bc);border:0}
[data-testid="stDownloadButton"] button{border-radius:10px}
.hero{background:linear-gradient(110deg,#133e53,#2c2751);border:1px solid #46617a;
 border-radius:18px;padding:28px 30px;margin-bottom:20px}
.hero h2{margin:0 0 8px;color:#b2f5f4!important}.hero p{font-size:1.08rem;line-height:1.7;margin:0}
.guide{background:#152841;border:1px solid #38516c;border-left:4px solid #52c6db;
 border-radius:12px;padding:15px 20px;margin:10px 0 22px}
.guide strong{color:#b9f1f6}.guide p{color:#d9e7f3;line-height:1.6;margin:5px 0}
.devline{color:#8eaac5;font-size:.84rem;margin-top:12px}
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
DESARROLLADORES = "Juan David Granja · Helmer Rivera · Yeison Ortega"

FUENTES = {
    "SIPSA": ([f"{RAW}/SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"],
              "227448e82618c3f9dee2f7e72d922cfea5811878ebc02ddb90998c5d10913676"),
    "RNDC": ([f"{RAW}/variable2_flete_regional_modelo_2020_2024.csv"],
             "aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891"),
    "IPC": (["https://www.dane.gov.co/files/operaciones/IPC/dic2024/anex-IPC-CiudadesMensuales-dic2024.xlsx",
             f"https://github.com/{BACKUP}/releases/download/{TAG}/anex-IPC-CiudadesMensuales-dic2024.xlsx"],
            "3e584414f3f89c78f99ce92576a0715d3d51fac5d6dcd75addf84065e4cd4076"),
    "ICTC": (["https://www.dane.gov.co/files/operaciones/ICTC/anex-ICTC-HistVariaContrib-dic2024.xlsx",
              f"https://github.com/{BACKUP}/releases/download/{TAG}/anex-ICTC-HistVariaContrib-dic2024.xlsx"],
             "6fe232548e47dbf9801b75c620209f579ebd0faa2a46fc30a9a305453878205a"),
}

NOMBRES = {
    Y:"Brecha Pasto–Colombia (p. p.)", T:"Fletes RNDC regional (%)",
    P:"Precios SIPSA ponderados (%)", I:"ICTC nacional (%)",
    "IPC_MENSUAL_PASTO_PCT":"Inflación mensual Pasto (%)",
    "IPC_MENSUAL_NACIONAL_PCT":"Inflación mensual nacional (%)",
    "PM_no_ponderado_t":"SIPSA no ponderado (%)",
    "PM_ponderado_lagQ_t":"SIPSA pesos rezagados (%)",
    "PM_100":"SIPSA canasta 100 % (%)", "PM_95":"SIPSA canasta 95 % (%)",
    "cobertura_indice_pct":"Cobertura física SIPSA (%)",
    "n_productos_indice":"Productos SIPSA comparables",
    "kg_total":"Abastecimiento total (kg)", "kg_usados_indice":"Abastecimiento cubierto (kg)",
    "VAR_FLETE_FISHER_PCT":"RNDC Fisher (%)",
    "VAR_FLETE_CRUDO_PCT":"RNDC sin depuración (%)",
    "VAR_FLETE_SIN_2000_PCT":"RNDC sin registro 2.000 (%)",
    "COB_VALOR_T_PCT":"Cobertura RNDC valor (%)",
    "COB_VIAJES_T_PCT":"Cobertura RNDC viajes (%)",
    "CELDAS_COMPARABLES":"Celdas RNDC comparables",
    "Ajustado":"Brecha ajustada (p. p.)",
}


def etiqueta(c): return NOMBRES.get(c, c)
def sha(raw): return hashlib.sha256(raw).hexdigest()


def normal(x):
    if pd.isna(x): return ""
    s = unicodedata.normalize("NFKD", str(x).strip().lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    return float(pd.to_numeric(str(x).strip().replace("%", "").replace(",", "."), errors="coerce"))


def validar(d, calendario, columnas):
    if d["mes"].tolist() != list(calendario) or d["mes"].duplicated().any():
        raise ValueError(f"Meses faltantes, extras o duplicados en {calendario[0]}–{calendario[-1]}")
    if not np.isfinite(d[columnas].to_numpy(float)).all():
        raise ValueError("Valores faltantes o no finitos en columnas obligatorias")
    return d


def periodo(serie):
    if isinstance(serie.dtype, pd.PeriodDtype): return serie.dt.asfreq("M")
    s = serie.astype("string").str.strip()
    if s.dropna().str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(s, format="%Y%m", errors="coerce").dt.to_period("M")
    if s.dropna().str.fullmatch(r"\d{4}-\d{2}").all():
        return pd.to_datetime(s + "-01", errors="coerce").dt.to_period("M")
    return pd.to_datetime(serie, errors="coerce").dt.to_period("M")


def parse_sipsa(raw):
    d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    campos = ["mes", P, "PM_no_ponderado_t", "PM_ponderado_lagQ_t", "cobertura_indice_pct", "n_productos_indice"]
    if any(c not in d for c in campos): raise ValueError("Columnas SIPSA no corresponden al Colab")
    d["mes"] = periodo(d["mes"])
    for c in NOMBRES:
        if c in d and c != "Ajustado": d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("mes").reset_index(drop=True)
    validar(d, M84, [P, "PM_no_ponderado_t", "cobertura_indice_pct", "n_productos_indice"])
    if not d["cobertura_indice_pct"].between(0, 100).all():
        raise ValueError("Cobertura SIPSA fuera de 0–100 %")
    return d[["mes"] + [c for c in NOMBRES if c in d and c != "Ajustado"]]


def parse_rndc(raw):
    d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype={"PERIODO": "string"})
    campos = ["FECHA", "PERIODO", T, "VAR_FLETE_FISHER_PCT", "CELDAS_COMPARABLES"]
    if any(c not in d for c in campos): raise ValueError("Columnas RNDC no corresponden al Colab")
    d["mes"] = pd.to_datetime(d["FECHA"], errors="raise").dt.to_period("M")
    if not (d["PERIODO"].str.strip() == d["mes"].dt.strftime("%Y%m")).all():
        raise ValueError("FECHA y PERIODO del RNDC no coinciden")
    for c in NOMBRES:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("mes").reset_index(drop=True)
    validar(d, M59, [T, "VAR_FLETE_FISHER_PCT", "CELDAS_COMPARABLES"])
    return d[["mes"] + [c for c in NOMBRES if c in d and c != "Ajustado"]]


def parse_ipc(raw):
    a = pd.read_excel(io.BytesIO(raw), sheet_name="VarCiudades", header=None)
    bloques = []
    for j, valor in enumerate(a.iloc[:, 0]):
        m = re.search(r"variaciones mensuales\s+(20\d{2})", normal(valor))
        if m: bloques.append((int(m.group(1)), j))
    if [x[0] for x in bloques] != list(range(2017, 2025)):
        raise ValueError("El Excel IPC no contiene bloques 2017–2024 auditados")
    nombres_meses = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
    salida = []
    for k, (anio, inicio) in enumerate(bloques):
        encabezados = [normal(a.iat[inicio + 1, j]) for j in range(1, 13)]
        if encabezados != nombres_meses: raise ValueError(f"Encabezado IPC incorrecto en {anio}")
        fin = bloques[k + 1][1] if k + 1 < len(bloques) else len(a)
        def ubicar(nombre_fila):
            return [j for j in range(inicio + 2, fin) if normal(a.iat[j, 0]) == nombre_fila]
        p, n = ubicar("pasto"), ubicar("total ipc")
        if len(p) != 1 or len(n) != 1: raise ValueError(f"Pasto/Total IPC ambiguos en {anio}")
        for mes in range(1, 13):
            if anio >= 2018:
                vp, vn = num(a.iat[p[0], mes]), num(a.iat[n[0], mes])
                salida.append({"mes":pd.Period(f"{anio}-{mes:02d}"),
                    "IPC_MENSUAL_PASTO_PCT":vp, "IPC_MENSUAL_NACIONAL_PCT":vn, Y:vp-vn})
    d = pd.DataFrame(salida)
    validar(d, M84, [Y, "IPC_MENSUAL_PASTO_PCT", "IPC_MENSUAL_NACIONAL_PCT"])
    if not (np.isclose(d.iloc[-1]["IPC_MENSUAL_PASTO_PCT"], .47)
            and np.isclose(d.iloc[-1]["IPC_MENSUAL_NACIONAL_PCT"], .46)):
        raise ValueError("Control IPC diciembre 2024 incompatible")
    return d


def parse_ictc(raw):
    a = pd.read_excel(io.BytesIO(raw), sheet_name="Anexo1", header=None).iloc[:, :3].copy()
    a.columns = ["anio_raw", "mes_raw", "valor_raw"]
    def anio(v):
        try:
            n = int(float(v))
            return float(n) if 2000 <= n <= 2100 else np.nan
        except (ValueError, TypeError): return np.nan
    a["anio"] = a.anio_raw.map(anio).ffill()
    meses = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
    a["nmes"] = a.mes_raw.map(normal).map({m:j+1 for j,m in enumerate(meses)})
    a[I] = a.valor_raw.map(num)
    a = a[a.anio.between(2018, 2024) & a.nmes.notna() & a[I].notna()].copy()
    a["mes"] = pd.to_datetime({"year":a.anio.astype(int),"month":a.nmes.astype(int),"day":1}).dt.to_period("M")
    d = a[["mes", I]].sort_values("mes").reset_index(drop=True)
    validar(d, M84, [I])
    if not (np.isclose(d.iloc[0][I], .70) and np.isclose(d.iloc[1][I], .83)):
        raise ValueError("Controles ICTC enero/febrero 2018 incompatibles")
    return d


def parse_combinado(raw):
    d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    campos = ["mes", Y, "IPC_MENSUAL_PASTO_PCT", "IPC_MENSUAL_NACIONAL_PCT", I]
    if any(c not in d for c in campos): raise ValueError("CSV conjunto sin columnas IPC/ICTC")
    d["mes"] = periodo(d["mes"])
    for c in campos[1:]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[campos].sort_values("mes").reset_index(drop=True)
    validar(d, M84, campos[1:])
    if not np.allclose(d[Y], d["IPC_MENSUAL_PASTO_PCT"]-d["IPC_MENSUAL_NACIONAL_PCT"], rtol=0, atol=1e-9):
        raise ValueError("Brecha no coincide con IPC Pasto menos IPC nacional")
    if not (np.isclose(d.iloc[0][I], .70) and np.isclose(d.iloc[1][I], .83)
            and np.isclose(d.iloc[-1]["IPC_MENSUAL_PASTO_PCT"], .47)
            and np.isclose(d.iloc[-1]["IPC_MENSUAL_NACIONAL_PCT"], .46)):
        raise ValueError("CSV conjunto no supera controles del Colab")
    return d


def descargar(url):
    r = requests.get(url, timeout=(12, 65), headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.content


def fuente_original(clave):
    urls, esperado = FUENTES[clave]
    errores = []
    for url in urls:
        try:
            raw = descargar(url)
            if sha(raw) != esperado: raise ValueError(f"SHA distinto al esperado ({sha(raw)})")
            d = {"SIPSA":parse_sipsa,"RNDC":parse_rndc,"IPC":parse_ipc,"ICTC":parse_ictc}[clave](raw)
            return d, {"Fuente":clave,"Meses":len(d),"Origen":url,"SHA-256":sha(raw),"Control":"Coincide con SHA auditado"}
        except Exception as e:
            errores.append(f"{url}: {type(e).__name__}: {str(e)[:180]}")
    raise RuntimeError(" | ".join(errores))


def fuente_combinada():
    url = f"{RAW}/ipc_ictc_pasto_2018_2024.csv"
    manifest = f"{RAW}/manifest_ipc_ictc_sha256.json"
    r = requests.get(url, timeout=(12, 65), headers={"User-Agent":"Mozilla/5.0"})
    if r.status_code == 404: return None
    r.raise_for_status()
    raw = r.content
    j = requests.get(manifest, timeout=(12, 30), headers={"User-Agent":"Mozilla/5.0"})
    esperado = None
    if j.ok:
        esperado = j.json().get("sha256")
        if not esperado: raise ValueError("Manifiesto IPC/ICTC sin SHA-256")
        if esperado != sha(raw): raise ValueError("CSV IPC/ICTC no coincide con su manifiesto")
    d = parse_combinado(raw)
    return d, {"Fuente":"IPC + ICTC","Meses":len(d),"Origen":url,"SHA-256":sha(raw),
               "Control":"SHA del manifiesto + datos" if esperado else "Datos validados; sin manifiesto"}


@st.cache_data(ttl=3600, show_spinner="Recuperando y verificando datos DANE / GitHub…")
def cargar():
    resultados, incidencias = {}, {}
    try:
        conjunto = fuente_combinada()
        if conjunto is not None: resultados["IPC+ICTC"] = conjunto
    except Exception as e: incidencias["IPC+ICTC procesado"] = str(e)
    claves = ["RNDC", "SIPSA"]
    if "IPC+ICTC" not in resultados: claves.extend(["IPC", "ICTC"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        trabajos = {c:pool.submit(fuente_original, c) for c in claves}
        for c, trabajo in trabajos.items():
            try: resultados[c] = trabajo.result()
            except Exception as e: incidencias[c] = str(e)
    base = pd.DataFrame({"mes":M84})
    for c in ["IPC+ICTC", "IPC", "ICTC", "SIPSA", "RNDC"]:
        if c in resultados:
            base = base.merge(resultados[c][0], on="mes", how="left", validate="one_to_one")
    audit = pd.DataFrame([resultados[c][1] for c in resultados])
    return base, audit, incidencias


def muestra(base, columnas, calendario=M59):
    faltantes = [c for c in columnas if c not in base]
    if faltantes: raise ValueError("No se recuperaron: " + ", ".join(map(etiqueta, faltantes)))
    columnas = list(dict.fromkeys(columnas))
    d = base.loc[base.mes.isin(calendario), ["mes"] + columnas].reset_index(drop=True)
    return validar(d, calendario, columnas)


def ajustar(base, familia="ARDL", L=2, hac=3, transporte=T, precio=P, excluir=None, estacionalidad=False):
    d = muestra(base, [Y, transporte, precio])
    for j in range(4):
        d[f"T{j}"] = d[transporte].shift(j)
        d[f"P{j}"] = d[precio].shift(j)
    d["B1"] = d[Y].shift(1)
    d = d.iloc[3:].reset_index(drop=True)
    columnas = (["B1"] if familia == "ARDL" else []) + [f"T{j}" for j in range(L+1)] + [f"P{j}" for j in range(L+1)]
    validar(d, M56, [Y, "B1", "T3", "P3"])
    if excluir: d = d[d.mes != pd.Period(excluir)].copy()
    if estacionalidad:
        q = ((d.mes.dt.month-1)//3+1)
        for j in (2,3,4):
            d[f"Q{j}"] = (q==j).astype(int)
            columnas.append(f"Q{j}")
    X = sm.add_constant(d[columnas].astype(float), has_constant="add")
    if len(d) <= X.shape[1]+10 or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Matriz singular o grados de libertad insuficientes")
    o = sm.OLS(d[Y], X).fit()
    m = sm.OLS(d[Y], X).fit(cov_type="HAC",cov_kwds={"maxlags":hac},use_t=True)
    if not np.isfinite(m.bse).all(): raise ValueError("Covarianza HAC inválida")
    return m,o,d,X


def conjuntas(m, L=2):
    filas=[]
    for pref,nombre_bloque in (("T","Transporte"),("P","SIPSA")):
        terms=[f"{pref}{j}" for j in range(L+1)]
        R=np.zeros((len(terms),len(m.params)))
        for j,term in enumerate(terms): R[j,m.params.index.get_loc(term)]=1
        prueba=m.wald_test(R,use_f=True,scalar=True)
        suma=m.t_test(R.sum(axis=0))
        ci=suma.conf_int()[0]
        filas.append({"Bloque":nombre_bloque,"p conjunta HAC":float(prueba.pvalue),
                      "Suma":float(m.params[terms].sum()),"p suma":float(suma.pvalue),
                      "IC suma inferior":float(ci[0]),"IC suma superior":float(ci[1])})
    return pd.DataFrame(filas)


def tabla_coef(m):
    ci=m.conf_int()
    return pd.DataFrame({"Regresor":m.params.index,"Coeficiente":m.params.values,
        "EE HAC":m.bse.values,"p HAC":m.pvalues.values,
        "IC inferior":ci.iloc[:,0].values,"IC superior":ci.iloc[:,1].values})


def estacionariedad(base,cols):
    filas=[]
    for c in cols:
        try:
            z=muestra(base,[c])
            adf=adfuller(z[c],autolag="AIC")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp=kpss(z[c],regression="c",nlags="auto")
            filas.append({"Serie":etiqueta(c),"ADF":adf[0],"p ADF":adf[1],
                "KPSS":kp[0],"p KPSS tabulada":kp[1],
                "Compatible I(0)":bool(adf[1]<.05 and kp[1]>=.05)})
        except Exception as e: filas.append({"Serie":etiqueta(c),"Incidencia":str(e)})
    return pd.DataFrame(filas)


def diagnosticos(o,z,X):
    e=o.resid
    lb=acorr_ljungbox(e,lags=[6,12],return_df=True)
    data={"Jarque–Bera p":float(jarque_bera(e)[1]),
        "Ljung–Box 6 p":float(lb.loc[6,"lb_pvalue"]),
        "Ljung–Box 12 p":float(lb.loc[12,"lb_pvalue"]),
        "Breusch–Godfrey p":float(acorr_breusch_godfrey(o,nlags=3)[1]),
        "Breusch–Pagan p":float(het_breuschpagan(e,X)[1]),
        "CUSUM p":float(breaks_cusumolsresid(e,ddof=X.shape[1])[1]),
        "Durbin–Watson":float(sm.stats.stattools.durbin_watson(e))}
    try: data["RESET p"]=float(linear_reset(o,power=2,use_f=True).pvalue)
    except Exception: data["RESET p"]=np.nan
    r=z[["mes",Y]].copy()
    r["Ajustado"]=o.fittedvalues.to_numpy()
    r["Residuo"]=e.to_numpy()
    r["Cook"]=np.asarray(OLSInfluence(o).cooks_distance[0])
    vif=pd.DataFrame([{"Regresor":c,"VIF":variance_inflation_factor(X.to_numpy(float),j)}
                      for j,c in enumerate(X.columns) if c!="const"])
    return pd.DataFrame([data]),r,vif


def robustez(base):
    specs=[(f"{f} L{L}",dict(familia=f,L=L)) for f in ("DL","ARDL") for L in range(4)]
    specs += [("HAC(1)",{"hac":1}),("HAC(6)",{"hac":6}),
              ("Excluir febrero 2023",{"excluir":"2023-02"}),
              ("Estacionalidad trimestral",{"estacionalidad":True})]
    for c in ["PM_no_ponderado_t","PM_ponderado_lagQ_t","PM_100","PM_95"]:
        if c in base and base.loc[base.mes.isin(M59),c].notna().all():
            specs.append((f"SIPSA: {etiqueta(c)}",{"precio":c}))
    for c in [I,"VAR_FLETE_FISHER_PCT","VAR_FLETE_CRUDO_PCT","VAR_FLETE_SIN_2000_PCT"]:
        if c in base and base.loc[base.mes.isin(M59),c].notna().all():
            specs.append((f"Transporte: {etiqueta(c)}",{"transporte":c}))
    filas=[]
    for nombre_spec,params in specs:
        try:
            m,o,z,_=ajustar(base,**params)
            j=conjuntas(m,params.get("L",2))
            filas.append({"Especificación":nombre_spec,"N":len(z),"R² ajustado":o.rsquared_adj,
                          "AIC":o.aic,"BIC":o.bic,"p transporte":j.iloc[0]["p conjunta HAC"],
                          "p SIPSA":j.iloc[1]["p conjunta HAC"],"Estado":"Estimado"})
        except Exception as e: filas.append({"Especificación":nombre_spec,"Estado":str(e)})
    return pd.DataFrame(filas)


def ventanas(base,ancho=36):
    d=muestra(base,[Y,T,P])
    for j in range(4):
        d[f"T{j}"]=d[T].shift(j)
        d[f"P{j}"]=d[P].shift(j)
    d["B1"]=d[Y].shift(1)
    d=d.iloc[3:].reset_index(drop=True)
    regs=["B1","T0","T1","T2","P0","P1","P2"]
    filas=[]
    for ini in range(len(d)-ancho+1):
        sub=d.iloc[ini:ini+ancho]
        X=sm.add_constant(sub[regs],has_constant="add")
        m=sm.OLS(sub[Y],X).fit(cov_type="HAC",cov_kwds={"maxlags":3},use_t=True)
        j=conjuntas(m)
        filas.append({"Inicio":str(sub.mes.iloc[0]),"Fin":str(sub.mes.iloc[-1]),
            "Suma RNDC":j.iloc[0]["Suma"],"Suma SIPSA":j.iloc[1]["Suma"],
            "RNDC IC inferior":j.iloc[0]["IC suma inferior"],
            "RNDC IC superior":j.iloc[0]["IC suma superior"]})
    return pd.DataFrame(filas)


def huber(base):
    _,o,z,X=ajustar(base)
    m=sm.RLM(z[Y],X,M=sm.robust.norms.HuberT()).fit()
    return pd.DataFrame([{"Bloque":lab,"Suma MCO":o.params[cols].sum(),
        "Suma Huber":m.params[cols].sum()} for lab,cols in
        [("RNDC",["T0","T1","T2"]),("SIPSA",["P0","P1","P2"])]] )


def respuesta_dinamica(m,pref,cambio,horizonte):
    cols=list(m.params.index)
    cov=m.cov_params().to_numpy(float)
    rho=float(m.params.get("B1",0))
    i_rho=cols.index("B1") if "B1" in cols else None
    r=0.;g=np.zeros(len(cols));filas=[]
    for h in range(horizonte+1):
        old_r,old_g=r,g.copy()
        termino=f"{pref}{h}"
        r=rho*old_r+float(m.params.get(termino,0))
        g=rho*old_g
        if i_rho is not None: g[i_rho]+=old_r
        if termino in cols: g[cols.index(termino)]+=1
        ee=abs(cambio)*np.sqrt(max(float(g@cov@g),0))
        filas.append({"Horizonte":h,"Respuesta":cambio*r,
            "IC inferior":cambio*r-1.96*ee,"IC superior":cambio*r+1.96*ee})
    return pd.DataFrame(filas)


def pca_cluster(base,cols,k):
    d=muestra(base,cols)
    X=StandardScaler().fit_transform(d[cols])
    pca=PCA(n_components=2).fit(X)
    coords=pca.transform(X)
    grupos=KMeans(n_clusters=k,n_init=20,random_state=42).fit_predict(X)
    pts=pd.DataFrame({"Mes":d.mes.astype(str),"PC1":coords[:,0],"PC2":coords[:,1],"Grupo":grupos.astype(str)})
    cargas=pd.DataFrame(pca.components_.T,index=[etiqueta(c) for c in cols],columns=["PC1","PC2"])
    return pts,cargas,pca.explained_variance_ratio_,silhouette_score(X,grupos)


def var_condicionado(base,cols,lags,horizonte):
    d=muestra(base,cols);filas=[]
    for c in cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pk=kpss(d[c],regression="c",nlags="auto")[1]
        filas.append({"Serie":etiqueta(c),"p ADF nivel":adfuller(d[c],autolag="AIC")[1],
                      "p KPSS nivel":pk,"p ADF diferencia":adfuller(d[c].diff().dropna(),autolag="AIC")[1]})
    tests=pd.DataFrame(filas)
    I0=(tests["p ADF nivel"].lt(.05)&tests["p KPSS nivel"].ge(.05)).all()
    I1=(tests["p ADF nivel"].ge(.05)&tests["p ADF diferencia"].lt(.05)).all()
    if I0:
        mod=VAR(d[cols]).fit(lags)
        if not mod.is_stable(): raise ValueError("VAR inestable")
        return tests,"VAR",mod,mod.irf(horizonte).irfs
    if I1:
        j=coint_johansen(d[cols],det_order=0,k_ar_diff=lags)
        return tests,"Johansen (evaluación previa a VECM)",None,pd.DataFrame({"Traza":j.lr1,"Crítico 95 %":j.cvt[:,1]})
    return tests,"Integración no concluyente: sistema no estimado",None,None


def evaluar_ml(base,elegidos):
    d=muestra(base,[Y,T,P]);tabla=pd.DataFrame({"mes":d.mes,"objetivo":d[Y]})
    for c in (Y,T,P):
        for j in (1,2): tabla[f"{c}_L{j}"]=d[c].shift(j)
    tabla=tabla.dropna().reset_index(drop=True)
    features=[c for c in tabla if c not in ("mes","objetivo")]
    inicio=len(tabla)-26
    factories={"Ridge":lambda:make_pipeline(StandardScaler(),Ridge(alpha=1)),
      "Lasso":lambda:make_pipeline(StandardScaler(),Lasso(alpha=.01,max_iter=15000)),
      "SVR":lambda:make_pipeline(StandardScaler(),SVR(C=1,epsilon=.1)),
      "MLP":lambda:make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(12,),max_iter=600,random_state=42))}
    filas=[]
    for fin in range(inicio,len(tabla)):
        ent=tabla.iloc[:fin];prueba=tabla.iloc[[fin]]
        X=ent[features].to_numpy(float);Xt=prueba[features].to_numpy(float);y=ent.objetivo.to_numpy(float)
        pred={"Media histórica":float(y.mean())}
        arX=sm.add_constant(ent[[f"{Y}_L1"]].to_numpy(float),has_constant="add")
        art=sm.add_constant(prueba[[f"{Y}_L1"]].to_numpy(float),has_constant="add")
        pred["AR(1)"]=float(sm.OLS(y,arX).fit().predict(art)[0])
        pred["ARX"]=float(sm.OLS(y,sm.add_constant(X,has_constant="add")).fit().predict(
            sm.add_constant(Xt,has_constant="add"))[0])
        for nom,maker in factories.items():
            if nom in elegidos: pred[nom]=float(maker().fit(X,y).predict(Xt)[0])
        for nom,v in pred.items():
            if nom in elegidos:
                filas.append({"Mes":str(prueba.mes.iloc[0]),"Modelo":nom,
                              "Real":float(prueba.objetivo.iloc[0]),"Pronóstico":v})
    pred=pd.DataFrame(filas)
    if pred.empty: raise ValueError("Selecciona modelos")
    metricas=pd.DataFrame([{"Modelo":m,"N":len(z),
        "RMSE":np.sqrt(mean_squared_error(z.Real,z.Pronóstico)),
        "MAE":mean_absolute_error(z.Real,z.Pronóstico)} for m,z in pred.groupby("Modelo")]).sort_values("RMSE")
    return metricas,pred,tabla,features,inicio


def xai(tabla,features,inicio,alg):
    models={"Ridge":Ridge(alpha=1),"Lasso":Lasso(alpha=.01,max_iter=15000),
            "SVR":SVR(C=1,epsilon=.1),
            "MLP":MLPRegressor(hidden_layer_sizes=(12,),max_iter=600,random_state=42)}
    model=make_pipeline(StandardScaler(),models[alg])
    model.fit(tabla.iloc[:inicio][features],tabla.iloc[:inicio].objetivo)
    r=permutation_importance(model,tabla.iloc[inicio:][features],tabla.iloc[inicio:].objetivo,
                             n_repeats=10,random_state=42,scoring="neg_mean_squared_error")
    return pd.DataFrame({"Predictor":features,"Importancia":r.importances_mean,
                         "Desviación":r.importances_std}).sort_values("Importancia")


def semaforo(base):
    d=muestra(base,[T,P]);filas=[]
    for i in range(12,len(d)):
        old=d.iloc[:i];row=d.iloc[i]
        ut,up=old[T].quantile(.9),old[P].quantile(.9)
        nivel=int(row[T]>ut)+int(row[P]>up)
        filas.append({"Mes":str(row.mes),"RNDC":row[T],"SIPSA":row[P],
            "Umbral RNDC":ut,"Umbral SIPSA":up,
            "Clasificación":("Verde","Amarillo","Rojo")[nivel]})
    return pd.DataFrame(filas)


def grafico(fig,alto=440):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#101b30",plot_bgcolor="#11213b",
                      font=dict(color="#e8f0f7"),height=alto,
                      margin=dict(t=58,l=24,r=20,b=35))
    st.plotly_chart(fig,use_container_width=True)
    return fig


def figura_series(base,cols,titulo,estandar=False):
    z=base[["mes"]+cols].copy();z["mes"]=z.mes.astype(str)
    if estandar:
        for c in cols:
            desv=z[c].std()
            if pd.notna(desv) and desv>0: z[c]=(z[c]-z[c].mean())/desv
    long=z.melt("mes",var_name="Serie",value_name="Valor").dropna()
    long.Serie=long.Serie.map(etiqueta)
    fig=px.line(long,x="mes",y="Valor",color="Serie",
                facet_row="Serie" if len(cols)>1 and not estandar else None,title=titulo)
    if len(cols)>1 and not estandar: fig.update_yaxes(matches=None)
    altura=max(440,205*len(cols)) if not estandar else 450
    fig.update_layout(height=altura)
    return fig


def lineas(base,cols,titulo,estandar=False):
    fig=figura_series(base,cols,titulo,estandar)
    return grafico(fig,max(440,205*len(cols)) if not estandar else 450)


def csv(d,nombre_archivo):
    z=d.copy()
    if "mes" in z: z["mes"]=z.mes.astype(str)
    st.download_button("Descargar CSV",z.to_csv(index=False).encode("utf-8-sig"),
                       file_name=nombre_archivo,mime="text/csv")


def export_html_fig(fig,nombre_archivo):
    st.download_button("Exportar gráfico",fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                       file_name=nombre_archivo,mime="text/html")


def lista_faltantes(base,cols):
    try: muestra(base,cols);return []
    except Exception as e:return [str(e)]


def guia(titulo,descripcion,uso):
    st.markdown(
        "<div class='guide'><strong>🧭 "+html.escape(titulo)+"</strong>"
        "<p>"+html.escape(descripcion)+"</p>"
        "<p><b>Cómo utilizarlo:</b> "+html.escape(uso)+"</p></div>",
        unsafe_allow_html=True,
    )


# El informe se calcula sobre las series seleccionadas; no se copia del artículo.
def reporte_completo(base, audit, seleccion, desde, hasta, opciones):
    sub = base.loc[base.mes.between(pd.Period(desde), pd.Period(hasta))].copy()
    secciones = []
    resumen = []
    def agregar(titulo, parrafos=None, tablas=None, figuras=None):
        secciones.append({"titulo":titulo,"parrafos":parrafos or [],
                          "tablas":tablas or [],"figuras":figuras or []})

    cobertura = []
    descriptivos = []
    tendencias = []
    for c in seleccion:
        d = sub.loc[sub[c].notna(), ["mes", c]].copy()
        n = len(d)
        cobertura.append({"Serie":etiqueta(c),"Observaciones":n,
            "Meses del periodo":len(sub),"Ausentes":len(sub)-n,
            "Cobertura %":round(n/max(len(sub),1)*100,2),
            "Primera observación":str(d.mes.iloc[0]) if n else "—",
            "Última observación":str(d.mes.iloc[-1]) if n else "—"})
        if n==0:
            resumen.append(f"{etiqueta(c)} no tiene observaciones en el intervalo seleccionado.")
            continue
        s=d[c].astype(float)
        indice_max=s.idxmax();indice_min=s.idxmin()
        descriptivos.append({"Serie":etiqueta(c),"N":n,"Media":s.mean(),
            "Mediana":s.median(),"Desv. estándar":s.std(),
            "P5":s.quantile(.05),"P95":s.quantile(.95),
            "Mínimo":s.min(),"Mes mínimo":str(d.loc[indice_min,"mes"]),
            "Máximo":s.max(),"Mes máximo":str(d.loc[indice_max,"mes"])})
        resumen.append(
            f"{etiqueta(c)}: promedio {s.mean():.4f}; máximo {s.max():.4f} "
            f"({d.loc[indice_max,'mes']}) y mínimo {s.min():.4f} ({d.loc[indice_min,'mes']})."
        )
        if n>=18 and s.std()>0:
            # El eje temporal no se comprime cuando faltan meses.
            x = np.array([int(m.ordinal - sub.mes.iloc[0].ordinal) for m in d.mes],dtype=float)
            X=sm.add_constant(x)
            model=sm.OLS(s.to_numpy(float),X).fit(cov_type="HAC",cov_kwds={"maxlags":3},use_t=True)
            pendiente=float(model.params[1]);pval=float(model.pvalues[1]);ic=model.conf_int()[1]
            tendencias.append({"Serie":etiqueta(c),"N":n,"Cambio lineal por mes":pendiente,
                "p HAC":pval,"IC inferior":float(ic[0]),"IC superior":float(ic[1]),
                "Lectura":"Tendencia distinta de cero al 5 %" if pval<.05 else "Sin evidencia al 5 %"})

    agregar("1. Síntesis ejecutiva",[
        f"Análisis generado el {datetime.now():%Y-%m-%d %H:%M} con los datos recuperados por la aplicación.",
        f"Periodo descriptivo: {desde} a {hasta} ({len(sub)} meses calendario). "
        f"Series elegidas: {', '.join(map(etiqueta,seleccion))}.",
        *resumen,
    ])
    agregar("2. Calidad, cobertura y trazabilidad",[
        "El porcentaje de cobertura se calcula respecto a los meses calendario elegidos. "
        "Los faltantes se conservan como ausentes, sin interpolación ni imputación."
    ],[("Cobertura por indicador",pd.DataFrame(cobertura)),
       ("Fuentes verificadas",audit)])

    figs=[]
    if len(seleccion)>0:
        figs.append(figura_series(sub,seleccion,"Evolución histórica por indicador"))
        if len(seleccion)>1:
            figs.append(figura_series(sub,seleccion,"Patrones expresados en desviaciones estándar",True))
    agregar("3. Comportamiento temporal",[
        "Cada serie se muestra en su unidad original y en su propio eje vertical. "
        "La comparación estandarizada sirve para patrones, no para comparar magnitudes económicas."
    ],figuras=figs)

    if descriptivos:
        graf = px.bar(pd.DataFrame(descriptivos),x="Serie",y="Desv. estándar",
                      title="Dispersión por indicador (unidades originales)")
        agregar("4. Estadística descriptiva y valores extremos",[
            "Se presentan media, mediana, dispersión, percentiles y meses extremos calculados sobre las observaciones disponibles. "
            "Las desviaciones estándar no son directamente comparables cuando las unidades difieren."
        ],[("Estadísticos calculados",pd.DataFrame(descriptivos))],[graf])
    if tendencias:
        tdf=pd.DataFrame(tendencias)
        agregar("5. Tendencias temporales",[
            "Para cada serie se ajustó una tendencia lineal sobre meses calendario, con errores HAC(3). "
            "La pendiente describe una asociación temporal; no identifica un mecanismo causal.",
            *[f"{fila['Serie']}: cambio lineal estimado {fila['Cambio lineal por mes']:.5f} por mes "
              f"(p HAC={fila['p HAC']:.4f}); {fila['Lectura'].lower()}." for _,fila in tdf.iterrows()],
        ],[("Regresiones de tendencia",tdf)])

    seleccion_cal = next((c for c in seleccion if sub[c].notna().sum()>=12),None)
    if seleccion_cal:
        tmp=sub[["mes",seleccion_cal]].dropna().copy()
        tmp["Año"]=tmp.mes.dt.year;tmp["Mes"]=tmp.mes.dt.month
        matriz=tmp.pivot(index="Año",columns="Mes",values=seleccion_cal)
        fig=px.imshow(matriz,aspect="auto",text_auto=".2f",title="Mapa térmico año-mes: "+etiqueta(seleccion_cal),
                      labels={"x":"Mes","y":"Año","color":"Valor"})
        agregar("6. Calendario mensual y variación estacional",[
            "El mapa térmico permite localizar meses con valores altos o bajos, "
            "sin atribuir automáticamente los cambios a estacionalidad ni a sucesos externos."
        ],figuras=[fig])

    pares=[]
    for i,c1 in enumerate(seleccion):
        for c2 in seleccion[i+1:]:
            pair=sub[[c1,c2]].dropna()
            if len(pair)>=12 and pair[c1].std()>0 and pair[c2].std()>0:
                r,pv=stats.pearsonr(pair[c1],pair[c2])
                pares.append({"Serie 1":etiqueta(c1),"Serie 2":etiqueta(c2),
                              "N común":len(pair),"r Pearson":float(r),"p (sin ajuste múltiple)":float(pv)})
    if pares:
        rel=pd.DataFrame(pares)
        matriz=sub[seleccion].corr(min_periods=12)
        fig=px.imshow(matriz,zmin=-1,zmax=1,text_auto=".2f",color_continuous_scale="RdBu_r",
                      title="Matriz de correlaciones por parejas observadas")
        mas_fuerte=max(pares,key=lambda a:abs(a["r Pearson"]))
        agregar("7. Asociaciones bivariadas",[
            "Se calcularon correlaciones Pearson por pares con meses coincidentes; "
            "los valores p son exploratorios y no se ajustaron por pruebas múltiples.",
            f"La asociación de mayor magnitud absoluta entre los pares disponibles fue "
            f"{mas_fuerte['Serie 1']} frente a {mas_fuerte['Serie 2']}: "
            f"r={mas_fuerte['r Pearson']:.4f}, N={mas_fuerte['N común']}. "
            "Esto no prueba transmisión causal.",
        ],[("Coeficientes y tamaño muestral",rel)],[fig])
    elif len(seleccion)>1:
        agregar("7. Asociaciones bivariadas",[
            "No hay pares con al menos 12 meses comunes y variación suficiente para calcular Pearson."
        ])

    test_rows=[]
    if opciones["estacionariedad"]:
        for c in seleccion:
            serie=sub[c].dropna()
            if len(serie)<24 or serie.nunique()<5:
                test_rows.append({"Serie":etiqueta(c),"Estado":"Muestra/variabilidad insuficiente"})
                continue
            try:
                ad=adfuller(serie,autolag="AIC")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kp=kpss(serie,regression="c",nlags="auto")
                test_rows.append({"Serie":etiqueta(c),"N":len(serie),"p ADF":float(ad[1]),
                    "p KPSS (tabulada)":float(kp[1]),
                    "Lectura":"Compatible con I(0) según ambas pruebas" if ad[1]<.05 and kp[1]>=.05
                              else "Pruebas no concluyentes para I(0)"})
            except Exception as e:
                test_rows.append({"Serie":etiqueta(c),"Estado":"Prueba no calculable: "+str(e)})
        agregar("8. Estacionariedad ADF/KPSS",[
            "ADF contrasta raíz unitaria y KPSS contrasta estacionariedad. "
            "Los valores p KPSS son aproximaciones tabuladas; los resultados son diagnósticos, no demostraciones."
        ],[("Pruebas por serie seleccionada",pd.DataFrame(test_rows))])

    modelo_incluido=all(c in seleccion for c in (Y,T,P))
    permite_fechas=(pd.Period(desde)<=M59[0] and pd.Period(hasta)>=M59[-1])
    model_ok=False
    if opciones["econometria"]:
        if not modelo_incluido:
            agregar("9. Modelo del artículo",[
                "No se estimó ARDL porque la selección del informe no incluye conjuntamente brecha, RNDC y SIPSA. "
                "Las otras variables pueden analizarse descriptivamente sin agregarlas al modelo principal."
            ])
        elif not permite_fechas:
            agregar("9. Modelo del artículo",[
                "No se estimó el ARDL: el rango descriptivo elegido no abarca los 59 meses comunes "
                "(2020-02 a 2024-12). No se alteró la muestra fija de 56 meses del artículo."
            ])
        else:
            try:
                m,o,z,X=ajustar(base)
                con=conjuntas(m);co=tabla_coef(m)
                fitted=z[["mes",Y]].copy();fitted["Ajustado"]=o.fittedvalues.to_numpy()
                f1=figura_series(fitted,[Y,"Ajustado"],"Brecha observada y ajuste dentro de muestra")
                coeff=co.loc[co.Regresor.ne("const")].copy()
                f2=px.scatter(coeff,x="Coeficiente",y="Regresor",
                    error_x=(coeff["IC superior"]-coeff["Coeficiente"]).to_numpy(),
                    error_x_minus=(coeff["Coeficiente"]-coeff["IC inferior"]).to_numpy(),
                    title="Coeficientes ARDL con intervalos HAC al 95 %")
                interpretation=[
                    f"ARDL L2 con HAC(3), N={len(z)}, R² ajustado={o.rsquared_adj:.4f}, "
                    f"AIC={o.aic:.3f} y BIC={o.bic:.3f}.",
                    f"RNDC contemporáneo: coeficiente {m.params['T0']:.5f} "
                    f"(p HAC={m.pvalues['T0']:.4f}). "
                    f"El bloque conjunto de transporte tiene p={con.iloc[0]['p conjunta HAC']:.4f} "
                    f"y el de SIPSA p={con.iloc[1]['p conjunta HAC']:.4f}.",
                    "Los contrastes conjuntos y los coeficientes aislados responden a hipótesis distintas. "
                    "La asociación estimada no debe interpretarse como efecto causal."
                ]
                agregar("9. ARDL L2 e inferencia HAC",interpretation,
                        [("Coeficientes",co),("Contrastes conjuntos",con)], [f1,f2])
                model_ok=True
                if opciones["diagnosticos"]:
                    dg,res,vif=diagnosticos(o,z,X)
                    f3=px.scatter(res,x="Ajustado",y="Residuo",title="Residuos frente al ajuste")
                    f4=px.histogram(res,x="Residuo",nbins=15,title="Distribución de residuos")
                    hallazgos_diag=[]
                    for indicador,hipotesis in [
                        ("Jarque–Bera p","normalidad residual"),
                        ("Ljung–Box 12 p","ausencia de autocorrelación hasta 12 rezagos"),
                        ("Breusch–Pagan p","homocedasticidad"),
                        ("CUSUM p","estabilidad de parámetros")]:
                        pv=float(dg.iloc[0][indicador])
                        hallazgos_diag.append(
                            f"{indicador}={pv:.4f}: " +
                            (f"evidencia contraria a {hipotesis} al 5 %." if pv<.05
                             else f"no se rechaza {hipotesis} al 5 %."))
                    agregar("10. Diagnósticos y multicolinealidad",[
                        "Jarque–Bera examina normalidad, Ljung–Box/Breusch–Godfrey autocorrelación, "
                        "Breusch–Pagan heterocedasticidad y CUSUM estabilidad. VIF describe colinealidad. "
                        "Son diagnósticos con potencia limitada por el tamaño muestral.",
                        *hallazgos_diag,
                    ],[("Resultados de pruebas",dg.T.reset_index().rename(columns={"index":"Diagnóstico",0:"Valor"})),
                       ("Factores de inflación de varianza",vif)], [f3,f4])
                if opciones["robustez"]:
                    rob=robustez(base)
                    good=rob.loc[rob.Estado.eq("Estimado")].copy()
                    figs=[]
                    if len(good): figs.append(px.bar(good,x="Especificación",y="p transporte",
                                              title="Sensibilidad del contraste conjunto de transporte"))
                    win=ventanas(base)
                    figs.append(px.line(win,x="Fin",y=["Suma RNDC","Suma SIPSA"],
                                         title="Evolución de sumas estimadas en 21 ventanas de 36 meses"))
                    num_sig=int(good["p transporte"].lt(.05).sum()) if len(good) else 0
                    agregar("11. Robustez y estabilidad temporal",[
                        f"Se evaluaron {len(rob)} especificaciones (incluidas alternativas disponibles) "
                        f"y {len(win)} ventanas móviles de 36 meses.",
                        f"Entre las {len(good)} especificaciones estimadas, {num_sig} presentan p conjunta "
                        "del bloque de transporte inferior a 0,05. Estas especificaciones no son pruebas independientes.",
                        f"La suma RNDC de las ventanas osciló entre {win['Suma RNDC'].min():.5f} "
                        f"y {win['Suma RNDC'].max():.5f}; "
                        "los coeficientes deben leerse junto con sus intervalos y los cambios de muestra."
                    ],[("Especificaciones",rob),("Ventanas móviles",win)],figs)
            except Exception as e:
                agregar("9. Modelo del artículo",["No pudo estimarse con los datos válidos de la aplicación: "+str(e)])

    if opciones["pca"] and len(seleccion)>=2:
        try:
            pts,cargas,var,sil=pca_cluster(base,seleccion,3)
            fig=px.scatter(pts,x="PC1",y="PC2",color="Grupo",hover_data=["Mes"],
                           title="PCA y K-Means de meses (k=3)")
            agregar("12. PCA y patrones mensuales",[
                f"PC1 y PC2 explican conjuntamente {sum(var):.1%} de la variabilidad estandarizada. "
                f"Silhouette para tres agrupaciones: {sil:.3f}.",
                "Los grupos representan meses similares, nunca departamentos ni causalidad."
            ],[("Cargas de componentes",cargas.reset_index(names="Indicador")),
               ("Asignación temporal",pts)], [fig])
        except Exception as e:
            agregar("12. PCA y patrones mensuales",["No aplicable a la selección: "+str(e)])

    if opciones["ml"]:
        if not modelo_incluido or not permite_fechas:
            agregar("13. Predicción retrospectiva",[
                "No se calculó: requiere brecha, RNDC, SIPSA y todo el periodo histórico común."
            ])
        else:
            try:
                met,pred,dat_ml,predictores,inicio_ml=evaluar_ml(base,["Media histórica","AR(1)","ARX","Ridge","Lasso","SVR"])
                obs=pred.drop_duplicates("Mes")[["Mes","Real"]].rename(columns={"Real":"Valor"})
                obs["Modelo"]="Observado"
                pr=pred[["Mes","Modelo","Pronóstico"]].rename(columns={"Pronóstico":"Valor"})
                fig=px.line(pd.concat([obs,pr],ignore_index=True),x="Mes",y="Valor",color="Modelo",
                            title="Evaluación retrospectiva: valores y pronósticos")
                best=met.iloc[0]
                imp=xai(dat_ml,predictores,inicio_ml,"Ridge")
                fig_imp=px.bar(imp,x="Importancia",y="Predictor",orientation="h",
                               title="XAI: importancia por permutación (Ridge)")
                agregar("13. Comparación predictiva y XAI",[
                    f"Evaluación con origen expansivo, {int(best['N'])} meses de prueba. "
                    f"El menor RMSE corresponde a {best['Modelo']} ({best['RMSE']:.4f} p. p.).",
                    "Las importancias se calcularon por permutación para Ridge sobre los meses de prueba. "
                    "No implican causalidad ni identifican necesariamente el mecanismo económico.",
                    "No se han validado calendarios históricos de publicación; no se presenta como pronóstico operativo."
                ],[("Métricas por algoritmo",met),("Predicciones",pred),
                   ("Importancia por permutación Ridge",imp)], [fig,fig_imp])
            except Exception as e:
                agregar("13. Comparación predictiva",["No calculable: "+str(e)])

    agregar("14. Fuentes y alcance",[
        f"Datos procesados: https://github.com/{REPO}",
        f"Respaldos DANE: https://github.com/{BACKUP}/releases/tag/{TAG}",
        "La brecha corresponde a Pasto menos Colombia, y no a la inflación departamental de Nariño. "
        "El RNDC regional es un índice construido por los autores a partir de registros oficiales.",
        "El informe es reproducible con los CSV auditados y las operaciones de la app. "
        "No se imputan observaciones, no se generan resultados ficticios y no se afirma causalidad.",
        f"Desarrolladores: {DESARROLLADORES}.",
    ])
    return {"secciones":secciones,"seleccion":seleccion,"desde":desde,"hasta":hasta,"modelo":model_ok}


def tabla_html(d):
    t=d.copy()
    for c in t.select_dtypes(include=["number"]).columns:
        t[c]=t[c].map(lambda x: f"{x:,.5f}" if pd.notna(x) and np.isfinite(x) else "—")
    return t.to_html(index=False,escape=True,border=0,classes="report-table")


def informe_html(reporte):
    css="""<style>body{font:15px/1.65 'Segoe UI',Arial,sans-serif;background:#f5f8fc;color:#17283a;
     margin:0 auto;max-width:1120px;padding:36px}h1{color:#113953}h2{color:#15516a;margin-top:42px}
     .header{background:linear-gradient(110deg,#113f59,#292c60);color:white;padding:30px;border-radius:15px}
     .header h1{color:white;margin:0}.header p{margin:6px 0}section{background:white;padding:22px 28px;
     margin:22px 0;border:1px solid #dbe5ee;border-radius:12px;page-break-inside:avoid}
     .report-table{border-collapse:collapse;width:100%;font-size:12px;margin:15px 0}
     .report-table th,.report-table td{border-bottom:1px solid #dce5ef;padding:8px;text-align:left}
     .report-table th{background:#e8f4f7}figure{margin:18px 0}.fig{page-break-inside:avoid}
     footer{color:#657d91;text-align:center;margin-top:30px}@media print{body{padding:4px;background:white}
     section{border:0;padding:6px 0;box-shadow:none}.modebar{display:none!important}}</style>"""
    head=f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport'
     content='width=device-width,initial-scale=1'><title>Informe Pasto–Nariño</title>{css}</head><body>
     <div class='header'><h1>Informe estadístico Pasto–Nariño</h1>
     <p>Periodo descriptivo: {html.escape(reporte['desde'])} a {html.escape(reporte['hasta'])}</p>
     <p>Generado: {datetime.now():%Y-%m-%d %H:%M} · Desarrolladores: {html.escape(DESARROLLADORES)}</p></div>"""
    partes=[head]
    primera=True
    for sec in reporte["secciones"]:
        partes.append("<section><h2>"+html.escape(sec["titulo"])+"</h2>")
        for p in sec["parrafos"]: partes.append("<p>"+html.escape(str(p))+"</p>")
        for nombre_tab,tabla in sec["tablas"]:
            partes.append("<h3>"+html.escape(nombre_tab)+"</h3>"+tabla_html(tabla))
        for f in sec["figuras"]:
            # Una sola copia embebida de plotly.js: HTML sin dependencia de internet.
            partes.append("<div class='fig'>"+f.to_html(full_html=False,
                include_plotlyjs="inline" if primera else False,
                config={"responsive":True,"displaylogo":False})+"</div>")
            primera=False
        partes.append("</section>")
    partes.append("<footer>Observatorio Pasto–Nariño · Fuente DANE / RNDC · Análisis asociativo</footer></body></html>")
    return "\n".join(partes)


PAGINAS = ["🏠 Inicio","📊 Exploración","🗃️ Datos y calidad","📈 Econometría",
    "🧩 PCA y K-Means","🔁 VAR y cointegración","🤖 Predicción y XAI",
    "🎛️ Simulaciones y Pasto","📑 Informe","💬 Consultas","© Créditos y metodología"]

with st.sidebar:
    st.markdown("### 📊 OBSERVATORIO")
    st.caption("Pasto–Colombia · 2018–2024")
    pagina=st.radio("Navegación",PAGINAS)
    st.divider()
    if st.button("Actualizar fuentes"):
        cargar.clear()
        st.session_state.pop("ml_obs",None)
        st.session_state.pop("reporte_obs",None)
        st.rerun()
    st.markdown(f"<div class='devline'>Desarrolladores:<br>{DESARROLLADORES}</div>",unsafe_allow_html=True)

BASE,AUDITORIA,INCIDENCIAS=cargar()
DISP=[c for c in (Y,T,P,I) if c in BASE and BASE[c].notna().any()]
OTRAS=[c for c in NOMBRES if c not in DISP and c not in (Y,T,P,I,"Ajustado")
       and c in BASE and BASE[c].notna().any()]

if pagina=="🏠 Inicio":
    st.title("Observatorio Económico de Pasto y Nariño")
    st.markdown("""<div class='hero'><h2>¡Bienvenidos! 👋</h2>
    <p>Explora cómo evolucionaron los fletes, los precios mayoristas de El Potrerillo y la
    inflación mensual de Pasto frente a Colombia. Contrasta series, descubre patrones,
    interpreta modelos y prepara informes con cifras y gráficos calculados desde las fuentes.</p>
    </div>""",unsafe_allow_html=True)
    guia("¿Qué encontrarás aquí?",
        "Este observatorio combina consulta de datos, análisis estadístico, econometría temporal y métodos predictivos sobre la investigación 2018–2024.",
        "Comienza con los indicadores y tendencias de esta portada. Luego utiliza el menú lateral para explorar datos, estimar modelos, analizar escenarios o descargar un informe a medida.")
    for card,c in zip(st.columns(4),(Y,T,P,I)):
        n=int(BASE[c].count()) if c in BASE else 0
        card.metric(etiqueta(c),f"{n} meses" if n else "Sin serie")
    if DISP:
        a,b=st.columns([1.7,1])
        with a:
            c=st.selectbox("Indicador destacado",DISP,format_func=etiqueta)
            sub=BASE.dropna(subset=[c])
            st.metric(f"Último mes: {sub.mes.iloc[-1]}",f"{sub[c].iloc[-1]:+.3f}")
            lineas(BASE,[c],"Evolución mensual")
        with b:
            cob=pd.DataFrame({"Indicador":[etiqueta(c) for c in DISP],
                              "Meses":[int(BASE[c].count()) for c in DISP]})
            grafico(px.bar(cob,x="Meses",y="Indicador",orientation="h",title="Cobertura de datos"))
    if len(DISP)>1:
        sel=st.multiselect("Comparador",DISP,default=DISP,format_func=etiqueta)
        if sel: lineas(BASE,sel,"Patrones estandarizados",True)
    a,b,c=st.columns(3)
    a.markdown("**📊 Explorar**\n\nDistribuciones, correlaciones, estacionalidad y trazabilidad.")
    b.markdown("**📈 Modelar**\n\nARDL/HAC, diagnósticos, sensibilidad y sistemas temporales.")
    c.markdown("**📑 Comunicar**\n\nEscenarios, predicciones e informes con gráficos exportables.")
    st.caption("Desarrolladores: "+DESARROLLADORES)

elif pagina=="📊 Exploración":
    st.title("Explorador de indicadores")
    guia("¿Para qué sirve la exploración?",
         "Permite entender trayectoria, valores extremos, dispersión, estacionalidad visual y asociación bivariada de las series auténticas.",
         "Elige variables y rango de fechas. Cambia entre Series, Descriptivos, Correlación y Distribuciones; descarga los CSV o gráficos de interés.")
    elegir=st.multiselect("Variables",DISP+OTRAS,default=DISP,format_func=etiqueta)
    meses=[str(m) for m in M84]
    inicio,fin=st.select_slider("Periodo de visualización",options=meses,value=(meses[0],meses[-1]))
    vista=BASE.loc[BASE.mes.between(pd.Period(inicio),pd.Period(fin))]
    if elegir:
        a,b,c,d=st.tabs(["Series","Descriptivos","Correlación","Distribuciones"])
        with a:
            st.caption("Cada indicador usa un eje independiente. La estandarización solo cambia la escala del gráfico.")
            fig=lineas(vista,elegir,"Indicadores en unidades originales")
            export_html_fig(fig,"series_pasto.html")
            if len(elegir)>1: lineas(vista,elegir,"Comparación normalizada",True)
        with b:
            desc=vista[elegir].describe().T.reset_index(names="Indicador")
            desc.Indicador=desc.Indicador.map(etiqueta)
            st.dataframe(desc,hide_index=True,use_container_width=True)
            csv(desc,"descriptivos.csv")
            columna=st.selectbox("Serie año × mes",elegir,format_func=etiqueta)
            z=vista[["mes",columna]].dropna().copy()
            z["Año"]=z.mes.dt.year;z["Mes"]=z.mes.dt.month
            matriz=z.pivot(index="Año",columns="Mes",values=columna)
            if not matriz.empty:
                grafico(px.imshow(matriz,aspect="auto",text_auto=".2f",title="Calendario térmico mensual"))
            s=vista[columna].dropna()
            if len(s)>18: grafico(px.bar(x=list(range(13)),y=acf(s,nlags=12,fft=False),title="Autocorrelación (ACF)"))
        with c:
            st.caption("Las correlaciones son descriptivas: no implican causalidad.")
            if len(elegir)>1:
                metodo=st.radio("Método",["pearson","spearman"],horizontal=True)
                corr=vista[elegir].corr(method=metodo,min_periods=12)
                grafico(px.imshow(corr,zmin=-1,zmax=1,text_auto=".2f",color_continuous_scale="RdBu_r",title="Correlaciones"))
                x,y=st.columns(2)
                vx=x.selectbox("Eje X",elegir,format_func=etiqueta)
                vy=y.selectbox("Eje Y",elegir,index=1,format_func=etiqueta)
                grafico(px.scatter(vista.dropna(subset=[vx,vy]),x=vx,y=vy,title="Relación bivariada"))
        with d:
            long=vista[elegir].melt(var_name="Serie",value_name="Valor").dropna()
            long.Serie=long.Serie.map(etiqueta)
            grafico(px.box(long,x="Serie",y="Valor",color="Serie",points="outliers",title="Variación y extremos"),520)
            hist=st.selectbox("Histograma",elegir,format_func=etiqueta)
            grafico(px.histogram(vista,x=hist,nbins=18,title=f"Histograma: {etiqueta(hist)}"))
        csv(vista[["mes"]+elegir],"series_filtradas.csv")

elif pagina=="🗃️ Datos y calidad":
    st.title("Cobertura y trazabilidad")
    guia("¿Qué verifica esta sección?",
         "La integridad de las descargas, los meses disponibles y los indicadores de cobertura de SIPSA y RNDC.",
         "Revisa los 84/59 meses esperados, los hashes y las incidencias de fuentes. Descarga la base integrada o compara variantes metodológicas.")
    cob=[]
    for c in (Y,T,P,I):
        s=BASE.loc[BASE[c].notna(),"mes"] if c in BASE else pd.Series(dtype="object")
        cob.append({"Indicador":etiqueta(c),"Meses":len(s),"Desde":str(s.min()) if len(s) else "—",
                    "Hasta":str(s.max()) if len(s) else "—"})
    st.dataframe(pd.DataFrame(cob),hide_index=True,use_container_width=True)
    st.dataframe(AUDITORIA,hide_index=True,use_container_width=True)
    if INCIDENCIAS:
        with st.expander("Diagnóstico exacto de fuentes"):
            for fuente,error in INCIDENCIAS.items(): st.write(f"**{fuente}:** {error}")
    grupos=[("Comparabilidad de SIPSA",["cobertura_indice_pct","n_productos_indice"]),
            ("Alternativas SIPSA",[P,"PM_no_ponderado_t","PM_ponderado_lagQ_t","PM_100","PM_95"]),
            ("Alternativas RNDC",[T,"VAR_FLETE_FISHER_PCT","VAR_FLETE_CRUDO_PCT"]),
            ("Coberturas RNDC",["COB_VALOR_T_PCT","COB_VIAJES_T_PCT","CELDAS_COMPARABLES"])]
    for cabecera,cols in grupos:
        cols=[c for c in cols if c in BASE and BASE[c].notna().any()]
        if cols:
            st.subheader(cabecera)
            lineas(BASE,cols,cabecera)
    csv(BASE,"base_integrada_2018_2024.csv")
    csv(AUDITORIA,"auditoria_fuentes.csv")

elif pagina=="📈 Econometría":
    st.title("Estimación econométrica")
    guia("¿Qué hacen los modelos econométricos?",
         "ADF/KPSS evalúa propiedades de las series. DL/ARDL estima asociaciones contemporáneas y rezagadas con inferencia HAC. "
         "Los diagnósticos y la robustez permiten evaluar límites y sensibilidad de esas asociaciones.",
         "Empieza por las pruebas de estacionariedad, examina el ARDL L2 con HAC(3), luego interpreta los contrastes conjuntos, residuos y especificaciones alternativas.")
    a,b,c,d=st.tabs(["ADF / KPSS","DL y ARDL · HAC","Diagnósticos","Robustez y estabilidad"])
    with a:
        st.caption("ADF: hipótesis nula de raíz unitaria. KPSS: hipótesis nula de estacionariedad.")
        if DISP:
            tabla=estacionariedad(BASE,DISP)
            st.dataframe(tabla,hide_index=True,use_container_width=True)
            csv(tabla,"adf_kpss.csv")
    with b:
        st.latex(r"B_t=\alpha+\rho B_{t-1}+\sum_{j=0}^{L}\beta_j T_{t-j}+\sum_{j=0}^{L}\gamma_j PM_{t-j}+u_t")
        problemas=lista_faltantes(BASE,[Y,T,P])
        if problemas: st.error("Modelo no disponible: "+" | ".join(problemas))
        else:
            co1,co2,co3=st.columns(3)
            familia=co1.selectbox("Modelo",["ARDL","DL"])
            L=co2.slider("Rezagos",0,3,2)
            ancho=co3.slider("HAC maxlags",0,8,3)
            try:
                m,o,z,X=ajustar(BASE,familia,L,ancho)
                j=conjuntas(m,L)
                indicadores=[("N",len(z)),("R² ajustado",f"{o.rsquared_adj:.4f}"),
                             ("AIC",f"{o.aic:.2f}"),("BIC",f"{o.bic:.2f}")]
                for card,(texto,valor) in zip(st.columns(4),indicadores): card.metric(texto,valor)
                coef=tabla_coef(m)
                st.dataframe(coef,hide_index=True,use_container_width=True)
                st.dataframe(j,hide_index=True,use_container_width=True)
                ajustadas=z[["mes",Y]].copy();ajustadas["Ajustado"]=o.fittedvalues.to_numpy()
                lineas(ajustadas,[Y,"Ajustado"],"Brecha observada vs. ajuste dentro de muestra")
                coef_g=coef.loc[coef.Regresor.ne("const")].copy()
                grafico(px.scatter(coef_g,x="Coeficiente",y="Regresor",
                    error_x=(coef_g["IC superior"]-coef_g["Coeficiente"]).to_numpy(),
                    error_x_minus=(coef_g["Coeficiente"]-coef_g["IC inferior"]).to_numpy(),
                    title="Coeficientes HAC e IC 95 %"))
                if familia=="ARDL" and L==2 and ancho==3:
                    referencia=[("Coef RNDC(t)",.00913,float(m.params["T0"])),
                        ("p RNDC(t)",.03750,float(m.pvalues["T0"])),
                        ("p bloque RNDC",.14273,float(j.iloc[0]["p conjunta HAC"])),
                        ("p bloque SIPSA",.23749,float(j.iloc[1]["p conjunta HAC"]))]
                    ref=pd.DataFrame([{"Control":k,"Artículo":r,"App":v,"Diferencia":abs(r-v)} for k,r,v in referencia])
                    with st.expander("Contraste de reproducción frente al artículo"):
                        st.dataframe(ref,hide_index=True)
                csv(coef,"coeficientes_hac.csv")
                csv(j,"contrastes_conjuntos.csv")
            except Exception as e: st.error(f"Estimación: {e}")
    with c:
        st.caption("Úsalos para identificar autocorrelación, heterocedasticidad, normalidad e influencia potencial de observaciones.")
        try:
            _,o,z,X=ajustar(BASE)
            diag,residuos,vif=diagnosticos(o,z,X)
            st.dataframe(diag.T,use_container_width=True)
            st.dataframe(vif,hide_index=True)
            iz,de=st.columns(2)
            with iz:grafico(px.scatter(residuos,x="Ajustado",y="Residuo",title="Residuos y ajustes"))
            with de:grafico(px.histogram(residuos,x="Residuo",nbins=15,title="Histograma residual"))
            grafico(px.bar(residuos.assign(Mes=residuos.mes.astype(str)),x="Mes",y="Cook",title="Distancia de Cook"))
            csv(residuos,"residuos_ardl.csv")
        except Exception as e:st.info(f"Diagnóstico no calculable: {e}")
    with d:
        st.caption("Contrasta rezagos, covarianzas HAC y alternativas de índices, y observa la estabilidad en ventanas móviles.")
        if st.button("Calcular ocho especificaciones y sensibilidad",type="primary"):
            r=robustez(BASE);st.dataframe(r,hide_index=True,use_container_width=True);csv(r,"robustez.csv")
        x,y=st.columns(2)
        with x:
            if st.button("MCO vs. Huber"):
                try:st.dataframe(huber(BASE),hide_index=True)
                except Exception as e:st.error(str(e))
        with y:
            if st.button("Ventanas móviles (36 meses)"):
                try:
                    r=ventanas(BASE)
                    st.metric("Ventanas",len(r))
                    grafico(px.line(r,x="Fin",y=["Suma RNDC","Suma SIPSA"],title="Estabilidad de sumas"))
                    st.dataframe(r,hide_index=True);csv(r,"ventanas.csv")
                except Exception as e:st.error(str(e))

elif pagina=="🧩 PCA y K-Means":
    st.title("Análisis de patrones mensuales")
    guia("¿Qué aportan PCA y K-Means?",
         "PCA resume la variación conjunta en componentes; K-Means agrupa meses con perfiles similares de indicadores estandarizados.",
         "Selecciona dos o más series, elige el número de grupos y examina la varianza explicada, las cargas y la nube de meses coloreados.")
    cols=st.multiselect("Indicadores",DISP,default=DISP,format_func=etiqueta)
    k=st.slider("Grupos K-Means",2,6,3)
    if len(cols)>=2:
        try:
            pts,cargas,var,sil=pca_cluster(BASE,cols,k)
            a,b=st.columns(2)
            a.metric("Varianza PC1+PC2",f"{sum(var):.1%}")
            b.metric("Silhouette",f"{sil:.3f}")
            fig=px.scatter(pts,x="PC1",y="PC2",color="Grupo",hover_data=["Mes"],title="PCA y clusters de meses")
            amplitud=max(np.abs(pts[["PC1","PC2"]].to_numpy()).max(),1)*.5
            for nombre_c,fila in cargas.iterrows():
                fig.add_annotation(x=float(fila.PC1*amplitud),y=float(fila.PC2*amplitud),
                                   ax=0,ay=0,showarrow=True,arrowhead=2,text=str(nombre_c))
            grafico(fig,540)
            st.dataframe(cargas,use_container_width=True)
            st.dataframe(pts,hide_index=True)
            csv(pts,"clusters_pca.csv")
        except Exception as e:st.info(f"PCA / clustering no ejecutable: {e}")

elif pagina=="🔁 VAR y cointegración":
    st.title("Sistemas de series temporales")
    guia("¿Para qué sirven VAR y cointegración?",
         "Exploran la dinámica conjunta de series temporales. Antes de calcular respuestas a impulsos se evalúan estacionariedad y estabilidad.",
         "Elige entre dos y tres variables, fija los rezagos y consulta las pruebas. La app solo representa IRF cuando el VAR en niveles resulta admisible.")
    cols=st.multiselect("Variables endógenas",DISP,default=DISP[:3],max_selections=3,format_func=etiqueta)
    lag=st.slider("Rezagos del VAR/Johansen",1,3,1)
    if len(cols)>=2 and st.button("Evaluar el sistema"):
        try:
            pruebas,familia,modelo,resultado=var_condicionado(BASE,cols,lag,8)
            st.dataframe(pruebas,hide_index=True,use_container_width=True)
            st.write(familia)
            if familia=="VAR":
                st.metric("AIC",f"{modelo.aic:.4f}")
                x,y=st.columns(2)
                impulso=x.selectbox("Impulso en",cols,format_func=etiqueta)
                respuesta=y.selectbox("Respuesta de",cols,format_func=etiqueta)
                r=resultado[:,cols.index(respuesta),cols.index(impulso)]
                grafico(px.line(x=list(range(len(r))),y=r,markers=True,title="IRF reducida exploratoria"))
            elif resultado is not None:st.dataframe(resultado,hide_index=True)
        except Exception as e:st.info(str(e))

elif pagina=="🤖 Predicción y XAI":
    st.title("Machine Learning y explicabilidad")
    guia("¿Qué permite esta sección?",
         "Compara pronósticos retrospectivos de la brecha mediante entrenamiento cronológico expansivo. XAI muestra sensibilidad predictiva por permutación.",
         "Elige modelos, ejecuta la evaluación y compara RMSE/MAE; después selecciona un algoritmo para visualizar la importancia de sus predictores.")
    modelos=st.multiselect("Modelos",["Media histórica","AR(1)","ARX","Ridge","Lasso","SVR","MLP"],
        default=["Media histórica","AR(1)","ARX","Ridge","Lasso","SVR"])
    if st.button("Evaluar con ventanas expansivas",type="primary") and modelos:
        try:st.session_state["ml_obs"]=evaluar_ml(BASE,modelos)
        except Exception as e:st.error(str(e))
    if "ml_obs" in st.session_state:
        metricas,pred,tabla,features,inicio=st.session_state["ml_obs"]
        st.dataframe(metricas,hide_index=True,use_container_width=True)
        observado=pred.drop_duplicates("Mes")[["Mes","Real"]].rename(columns={"Real":"Valor"})
        observado["Modelo"]="Observado"
        estimado=pred[["Mes","Modelo","Pronóstico"]].rename(columns={"Pronóstico":"Valor"})
        grafico(px.line(pd.concat([estimado,observado]),x="Mes",y="Valor",color="Modelo",title="Predicciones vs. observado"))
        csv(pred,"predicciones_ml.csv")
        alg=st.selectbox("Modelo para XAI",["Ridge","Lasso","SVR","MLP"])
        if st.button("Calcular XAI por permutaciones"):
            imp=xai(tabla,features,inicio,alg)
            grafico(px.bar(imp,x="Importancia",y="Predictor",orientation="h",title="Importancia por permutación"))
            st.dataframe(imp,hide_index=True)
            csv(imp,"xai.csv")

elif pagina=="🎛️ Simulaciones y Pasto":
    st.title("Escenarios y Focus Pasto")
    guia("¿Qué muestran los escenarios?",
         "Los multiplicadores ARDL ilustran respuestas condicionadas ante cambios hipotéticos de RNDC o SIPSA. "
         "El episodio 2021 y el semáforo permiten describir periodos históricos.",
         "Elige un indicador, la magnitud y el horizonte del cambio; compara con el episodio de 2021 y explora la clasificación retrospectiva de meses.")
    a,b,c=st.tabs(["Simulador ARDL","Episodio 2021","Semáforo histórico"])
    with a:
        try:
            mod,_,_,_=ajustar(BASE)
            var=st.selectbox("Indicador del escenario",[T,P],format_func=etiqueta)
            cambio=st.number_input("Cambio hipotético en p. p.",value=1.0,step=.5)
            horizonte=st.slider("Meses de propagación",1,15,9)
            r=respuesta_dinamica(mod,"T" if var==T else "P",cambio,horizonte)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=r.Horizonte,y=r["IC superior"],line=dict(width=0),showlegend=False))
            fig.add_trace(go.Scatter(x=r.Horizonte,y=r["IC inferior"],fill="tonexty",line=dict(width=0),name="IC 95 %"))
            fig.add_trace(go.Scatter(x=r.Horizonte,y=r.Respuesta,mode="lines+markers",name="Respuesta condicional"))
            fig.add_hline(y=0,line_dash="dot")
            grafico(fig);st.dataframe(r,hide_index=True);csv(r,"simulador_ardl.csv")
        except Exception as e:st.info(str(e))
    with b:
        st.metric("Cambio de abastecimiento primeras quincenas de 2021, DANE","−56,98 %")
        st.write("El Potrerillo: 4.618 toneladas (1–15 de abril) y 1.987 toneladas (1–15 de mayo).")
        ep=BASE.loc[BASE.mes.between(pd.Period("2021-04"),pd.Period("2021-08"))]
        if DISP:
            lineas(ep,DISP,"Series durante el episodio de 2021")
            st.dataframe(ep[["mes"]+DISP].assign(mes=lambda q:q.mes.astype(str)),hide_index=True)
    with c:
        try:
            r=semaforo(BASE)
            grafico(px.scatter(r,x="RNDC",y="SIPSA",color="Clasificación",hover_data=["Mes"],title="Percentiles históricos"))
            st.dataframe(r,hide_index=True);csv(r,"semaforo_historico.csv")
        except Exception as e:st.info(str(e))

elif pagina=="📑 Informe":
    st.title("Generador de informes de investigación")
    guia("Informe completo y configurable",
         "El generador recalcula resultados para las variables elegidas: cobertura, descriptivos, extremos, tendencias, correlaciones, pruebas de estacionariedad y figuras. "
         "Si se incluyen las tres variables del artículo, permite incorporar ARDL/HAC, diagnósticos, robustez y pronósticos.",
         "Selecciona variables, rango y módulos; pulsa «Generar informe» y revisa las secciones antes de descargar el HTML interactivo y autocontenido.")
    seleccion=st.multiselect("Variables que incluirá el informe",DISP+OTRAS,default=DISP,format_func=etiqueta,key="informe_vars")
    periodos=[str(x) for x in M84]
    desde,hasta=st.select_slider("Periodo descriptivo del informe",options=periodos,
                                value=(periodos[0],periodos[-1]),key="informe_periodo")
    with st.expander("Módulos del informe",expanded=True):
        c1,c2,c3=st.columns(3)
        opc_est=c1.checkbox("ADF / KPSS",value=True)
        opc_econ=c1.checkbox("ARDL del artículo",value=True)
        opc_diag=c2.checkbox("Diagnósticos de residuos",value=True)
        opc_rob=c2.checkbox("Robustez y ventanas",value=True)
        opc_pca=c3.checkbox("PCA / Clustering",value=False)
        opc_ml=c3.checkbox("Comparación predictiva",value=False)
    opciones={"estacionariedad":opc_est,"econometria":opc_econ,"diagnosticos":opc_diag,
              "robustez":opc_rob,"pca":opc_pca,"ml":opc_ml}
    firma=(tuple(seleccion),desde,hasta,tuple(opciones.items()))
    if st.button("Generar informe completo",type="primary",disabled=(not seleccion)):
        with st.spinner("Calculando tablas, interpretaciones y gráficos del informe…"):
            try:
                rep=reporte_completo(BASE,AUDITORIA,seleccion,desde,hasta,opciones)
                contenido=informe_html(rep)
                st.session_state["reporte_obs"]=(firma,rep,contenido)
            except Exception as e:
                st.session_state.pop("reporte_obs",None)
                st.error("No pudo generarse el informe: "+str(e))
    activo=st.session_state.get("reporte_obs")
    if activo is not None:
        firma_ant,rep,contenido=activo
        if firma_ant!=firma:
            st.info("Las opciones cambiaron. Pulsa «Generar informe completo» para actualizarlas.")
        else:
            st.success(f"Informe calculado: {len(rep['secciones'])} secciones y gráficos interactivos.")
            st.download_button("⬇️ Descargar informe HTML completo (funciona sin conexión)",
                data=contenido.encode("utf-8"),file_name="informe_observatorio_pasto.html",
                mime="text/html",type="primary")
            for sec in rep["secciones"]:
                with st.expander(sec["titulo"],expanded=sec["titulo"].startswith("1.")):
                    for p in sec["parrafos"]:st.write(p)
                    for tname,tab in sec["tablas"]:
                        st.markdown("**"+tname+"**")
                        st.dataframe(tab,use_container_width=True,hide_index=True)
                    for fig in sec["figuras"]:grafico(fig)
            csv(BASE,"base_integrada.csv")
            csv(AUDITORIA,"auditoria_sha.csv")

elif pagina=="💬 Consultas":
    st.title("Asistente de consultas numéricas")
    guia("¿Para qué sirve el asistente?",
         "Responde consultas simples sobre valores observados, promedios, extremos y resultados del ARDL que la aplicación pueda estimar.",
         "Escribe una variable y un mes (por ejemplo, «RNDC 2021-05»), pide su media, máximo, mínimo o el coeficiente ARDL.")
    if "chat_obs" not in st.session_state:st.session_state.chat_obs=[]
    for mensaje in st.session_state.chat_obs:
        with st.chat_message(mensaje["role"]):st.markdown(mensaje["content"])
    pregunta=st.chat_input("Ej.: RNDC en 2021-05, media SIPSA, coeficiente ARDL")
    if pregunta:
        q=normal(pregunta)
        if any(x in q for x in ("sipsa","mayorista","potrerillo")): var=P
        elif any(x in q for x in ("rndc","flete")):var=T
        elif "ictc" in q:var=I
        else:var=Y
        match=re.search(r"(20(?:18|19|2[0-4]))[-/ ](0?[1-9]|1[0-2])",q)
        mes=pd.Period(f"{match.group(1)}-{int(match.group(2)):02d}") if match else None
        if mes is None:
            meses="enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
            for j,n in enumerate(meses,1):
                m=re.search(rf"\b{n}\s+(?:de\s+)?(20\d{{2}})\b",q)
                if m:mes=pd.Period(f"{m.group(1)}-{j:02d}");break
        if any(x in q for x in ("causal","causa","bloqueo")):
            respuesta="El observatorio no identifica efectos causales ni cierres viales."
        elif "coeficiente" in q or "ardl" in q:
            try:
                m,_,_,_=ajustar(BASE)
                respuesta=f"ARDL L2: RNDC(t)={m.params['T0']:.6f}, p HAC={m.pvalues['T0']:.5f}."
            except Exception as e:respuesta=f"Modelo no disponible: {e}"
        elif var not in BASE or BASE[var].notna().sum()==0:
            respuesta=f"No hay valores verificables de {etiqueta(var)}."
        elif mes is not None:
            z=BASE.loc[BASE.mes.eq(mes),var]
            respuesta=(f"{etiqueta(var)} en {mes}: **{z.iloc[0]:.5f}**" if len(z) and pd.notna(z.iloc[0])
                       else f"Sin observación para {mes}.")
        elif any(x in q for x in ("promedio","media")):
            respuesta=f"{etiqueta(var)}: media={BASE[var].mean():.5f}, n={BASE[var].count()}."
        elif any(x in q for x in ("maximo","minimo","pico")):
            z=BASE[["mes",var]].dropna()
            i=z[var].idxmin() if "minimo" in q else z[var].idxmax()
            respuesta=f"{etiqueta(var)}: {z.loc[i,var]:.5f} en {z.loc[i,'mes']}."
        else:respuesta="Consulta por variable y mes (AAAA-MM), promedio, máximo, mínimo o ARDL."
        st.session_state.chat_obs += [{"role":"user","content":pregunta},
                                      {"role":"assistant","content":respuesta}]
        st.rerun()

else:
    st.title("Fuentes, metodología y créditos")
    guia("Acerca del proyecto",
         "Documenta autoría, procedencia de las series y alcance de las herramientas estadísticas del observatorio.",
         "Consulta los repositorios y los hashes verificados, o inspecciona las incidencias técnicas de descarga cuando una fuente no esté disponible.")
    st.subheader("Equipo de desarrollo")
    for c,n in zip(st.columns(3),("Juan David Granja","Helmer Rivera","Yeison Ortega")):
        c.markdown(f"**👨‍💻 {n}**")
    st.markdown(f"""
**Universidad de Nariño · Economía · Investigación regional**

**Investigación:** Costos del transporte de carga, precios mayoristas de alimentos y brecha inflacionaria Pasto–Colombia, 2018–2024.

**Desarrolladores:** {DESARROLLADORES}.

**Fuentes:** IPC, SIPSA-P, SIPSA-A e ICTC del DANE; registros RNDC del Ministerio de Transporte. El índice regional de fletes es construido por los investigadores. GitHub conserva productos y respaldos; no sustituye las fuentes oficiales.

**Cobertura:** 84 meses para SIPSA, IPC e ICTC; 59 para RNDC y 56 para la comparación DL/ARDL. La brecha se expresa en puntos porcentuales; RNDC, SIPSA e ICTC en variaciones porcentuales.

**Integridad:** RNDC y SIPSA tienen hashes fijados. IPC/ICTC se lee desde un CSV procesado y su manifiesto cuando está disponible, o desde anexos Excel con SHA fijado. El estado específico de cada fuente se muestra en «Datos y calidad».

**Interpretación:** ARDL/HAC y sus sumas son asociaciones, no causalidad. El ICTC nacional no equivale a RNDC regional. Pasto es una ciudad, no el departamento completo de Nariño. No se imputan observaciones. PCA/K-Means agrupan meses y ML se evalúa retrospectivamente.

**Informe:** el HTML incluye resultados recalculados y gráficos interactivos en un solo archivo sin depender de conexión. La comparación ARDL exige las tres variables centrales y el calendario completo del modelo. Las tendencias y correlaciones no implican causalidad.

**Fuera de alcance:** mapas coropléticos sin datos municipales compatibles, Moran/LISA sin panel espacial, RAG documental y detección operativa de bloqueos.

**Repositorios:** [Indicadores](https://github.com/{REPO}) · [Respaldo SIPSA](https://github.com/{BACKUP}/releases/tag/{TAG}) · [Colab](https://colab.research.google.com/drive/1Tu3ZzP1L8XlUejqbS_0eXgZ8QszqiMr7).
""")
    st.dataframe(AUDITORIA,hide_index=True,use_container_width=True)
    with st.expander("Incidencias de fuentes"):
        for k,v in INCIDENCIAS.items():st.write(f"**{k}**: {v}")

st.divider()
st.caption("Observatorio Pasto–Colombia · DANE y RNDC · 2018–2024 · Desarrolladores: "+DESARROLLADORES)
