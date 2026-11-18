# 🇨🇴 Observatorio de la Brecha Inflacionaria Pasto–Colombia (2018–2024)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://observatorio-brecha-inflacionaria.streamlit.app)

Aplicación interactiva en **Streamlit** que acompaña el artículo *"Costos del transporte de carga, precios mayoristas de alimentos y brecha inflacionaria Pasto–Colombia: evidencia mensual 2018–2024"*, desarrollada en el marco del **Seminario de Investigación Regional II** del Programa de Economía de la **Universidad de Nariño**.

---

## 📋 Descripción general

El observatorio permite reproducir y explorar el marco empírico del estudio sobre la dinámica de precios regionales en San Juan de Pasto frente al agregado nacional:

- **Brecha inflacionaria ($B_t$):** Diferencial mensual entre la inflación de Pasto y la de Colombia ($B_t = \pi_{\text{Pasto},t} - \pi_{\text{Colombia},t}$).
- **Precios mayoristas (SIPSA):** Indicador construido a partir de precios y cantidades físicas abastecidas en el mercado mayorista **El Potrerillo** (DANE).
- **Fletes regionales (RNDC):** Índice regional de costos de transporte de carga con destino a Pasto, contrastado frente al índice nacional **ICTC** (DANE).

> **Alcance metodológico:** Los modelos y resultados presentados tienen un alcance estrictamente **asociativo**, no causal.

---

## 🚀 Módulos de la aplicación

1. **🏠 Inicio:** Resumen ejecutivo del estudio, metadatos de las series cargadas y estadísticos descriptivos.
2. **📊 Exploración y EDA:** Matriz de correlaciones contemporáneas y visualización de trayectorias mensuales (2018–2024).
3. **📈 Series y Estacionariedad:** Pruebas formales de raíz unitaria **ADF** (Augmented Dickey-Fuller) y estacionariedad **KPSS**.
4. **📉 ARDL / Modelos dinámicos:** Estimación interactiva de modelos autorregresivos de rezagos distribuidos con selección de rezagos ($L \in [0,3]$), inclusión opcional del ICTC y gráfico de ajuste dentro de muestra.
5. **🔬 Sensibilidad y diagnóstico:** Contraste de correlación RNDC–ICTC y análisis distribucional de la brecha.
6. **📍 Focus Pasto–Nariño:** Lectura territorial de la trayectoria de la brecha y consideraciones de política pública.
7. **📑 Informe de investigación:** Generación y descarga automática de un borrador de reporte en formato Markdown (`.md`).
8. **💬 Asistente:** Módulo de consulta rápida sobre definiciones metodológicas, variables y alcance del estudio.

---

## 📂 Fuentes de datos soportadas

La aplicación ofrece dos modos de operación desde la barra lateral:
- **Series demo calibradas:** Generadas de forma reproducible (`seed=42`) respetando las propiedades estadísticas descriptivas reportadas en el artículo.
- **Carga de CSV propio:** Permite subir un archivo `.csv` con las series procesadas del equipo. El cargador normaliza automáticamente columnas equivalentes a `fecha`, `Brecha`, `SIPSA`, `RNDC` e `ICTC`.

---

## 🛠️ Instalación y ejecución local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/JuanDavidgm2026/observatorio-brecha-inflacionaria.git](https://github.com/JuanDavidgm2026/observatorio-brecha-inflacionaria.git)
   cd observatorio-brecha-inflacionaria
