import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


# =============================================================================
# Definicion IDENTICA de la clase del interpolador (necesaria para que
# joblib.load pueda reconstruir el objeto guardado en best_model.pkl)
# =============================================================================
class COMSOLVesselInterpolator:
    def __init__(self, tiempos, diametros, tabla_temperaturas):
        self.times = np.asarray(tiempos, dtype=float)
        self.diameters = np.asarray(diametros, dtype=float)
        self.temps = np.asarray(tabla_temperaturas, dtype=float)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        t_min, t_max = self.times.min(), self.times.max()
        d_min, d_max = self.diameters.min(), self.diameters.max()
        preds = []
        for t_val, d_val in X:
            t_val = np.clip(t_val, t_min, t_max)
            d_val = np.clip(d_val, d_min, d_max)
            curvas_por_diametro = [np.interp(t_val, self.times, self.temps[:, j])
                                    for j in range(len(self.diameters))]
            v_final = np.interp(d_val, self.diameters, curvas_por_diametro)
            preds.append(v_final)
        return np.array(preds)


# =============================================================================
# CONFIGURACION DE LA PAGINA
# =============================================================================
st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Ablación Tumoral mediante IA")
st.markdown("""
Esta aplicación web interactiva funciona como un **Modelo Subrogado de Inteligencia Artificial**.
Utiliza regresores e interpoladores calibrados con las soluciones numéricas del modelo COMSOL
Multiphysics (ecuación de biocalor de Pennes + cinética de daño de Arrhenius) para predecir, en
tiempo real, lo que antes solo se podía obtener corriendo la simulación de elementos finitos.
""")

# =============================================================================
# CARGA DE LOS MODELOS PRE-ENTRENADOS (.PKL)
# =============================================================================
if os.path.exists("best_model.pkl") and os.path.getsize("best_model.pkl") > 0:
    try:
        modelos = joblib.load("best_model.pkl")
        modelo_dano = modelos["dano"]
        modelo_vaso = modelos["vaso"]
    except Exception as e:
        st.error(f"Error al decodificar 'best_model.pkl': {e}. Vuelve a correr training.py.")
        st.stop()
else:
    st.error("⚠️ No se encontró 'best_model.pkl'. Ejecuta primero `python training.py` "
              "(debe estar en la misma carpeta que fracionamiento_de_daño.txt).")
    st.stop()

# =============================================================================
# DATOS HISTORICOS (leidos de los CSV generados por training.py — ningun
# valor transcrito a mano aqui, todo viene de los archivos guardados)
# =============================================================================
df_dano = pd.read_csv("tumor_data.csv")
df_vaso = pd.read_csv("vessel_data.csv")

radios_comsol = sorted(df_dano["Distancia_mm"].unique())
diametros_comsol = sorted(df_vaso["Diametro_mm"].unique())

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Daño Tisular (Arrhenius)",
    "🩸 Vaso Sanguíneo (Heat-Sink)",
    "⚡ Sensibilidad a la Potencia (teórico)",
    "📐 Comparación lado a lado",
])

# =============================================================================
# SIDEBAR — parametros compartidos por todas las pestañas
# =============================================================================
st.sidebar.header("🕹️ Parámetros de Predicción en Vivo")
tiempo_input = st.sidebar.slider("Tiempo de tratamiento (min):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Distancia al electrodo (mm):", 4.0, 20.0, 8.0, step=0.5)
diametro_input = st.sidebar.slider("Diámetro del vaso sanguíneo (mm):", 1.0, 5.0, 3.0, step=0.1)

pred_dano_viva = float(np.clip(modelo_dano.predict([[tiempo_input, distancia_input]])[0], 0.0, 1.0))
pred_vaso_viva = float(modelo_vaso.predict([[tiempo_input, diametro_input]])[0])

# =============================================================================
# PESTAÑA 1 — DAÑO TISULAR
# =============================================================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📍 Fracción de Daño Tisular Predicho",
                   f"{pred_dano_viva:.4f} ({pred_dano_viva*100:.2f}%)")
    with col2:
        status = "🔴 Necrosis Crítica (>70%)" if pred_dano_viva >= 0.7 else "🟡 Lesión Parcial / Tejido Viable"
        st.metric("⚠️ Estado Celular Estimado", status)

    st.subheader("📊 Curvas de Daño (COMSOL vs. predicción IA)")
    t_cont = np.linspace(0, 10.52, 200)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colores = {radios_comsol[0]: "blue", radios_comsol[1]: "orange", radios_comsol[2]: "green"}
    for r in radios_comsol:
        sub = df_dano[df_dano["Distancia_mm"] == r]
        ax.scatter(sub["Tiempo_min"], sub["Fraccion_Dano"], color=colores[r], alpha=0.6,
                    label=f"COMSOL histórico ({r:.0f} mm)")
    Xc = np.column_stack((t_cont, np.full_like(t_cont, distancia_input)))
    pred_c = np.clip(modelo_dano.predict(Xc), 0.0, 1.0)
    ax.plot(t_cont, pred_c, "r--", linewidth=2.5, label=f"Predicción IA continua ({distancia_input} mm)")
    ax.plot(tiempo_input, pred_dano_viva, "kX", markersize=12, label="Punto en vivo seleccionado")
    ax.set_title("Evolución del Daño Tisular: Simulación Física vs. Regresión de IA")
    ax.set_xlabel("Tiempo de exposición (min)"); ax.set_ylabel("Fracción de daño celular (0 a 1)")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(loc="upper left")
    st.pyplot(fig)

    with st.expander("⚠️ Limitación importante del modelo (léase antes de interpretar clínicamente)"):
        st.markdown("""
        El modelo se entrenó con **solo 3 radios simulados en COMSOL** (4, 12 y 20 mm). Entre
        r=4 mm y r=12 mm, el daño real de COMSOL es casi plano (0.810 → 0.799, apenas 1% de
        diferencia), pero entre r=12 mm y r=20 mm cae fuerte (0.799 → 0.522). Cualquier modelo
        suave que intente conectar "casi plano, luego caída brusca" genera un pequeño sobre-pico
        artificial alrededor de r≈8 mm. **Esto es una limitación de tener solo 3 sondas
        radiales, no un error del algoritmo.**
        """)

    st.subheader("🗺️ Mapa de daño Ω(r, t) y frente de necrosis")
    r_grid = np.linspace(4, 20, 100)
    t_grid = np.linspace(0, 10.52, 100)
    RR, TTd = np.meshgrid(r_grid, t_grid)
    DANO_map = np.clip(modelo_dano.predict(np.column_stack((RR.ravel(), TTd.ravel()))), 0.0, 1.0).reshape(RR.shape)

    fig1b, ax1b = plt.subplots(figsize=(10, 4.5))
    cf1b = ax1b.contourf(RR, TTd, DANO_map, levels=25, cmap="hot")
    cs1b = ax1b.contour(RR, TTd, DANO_map, levels=[0.63], colors="cyan", linewidths=2.5)
    ax1b.clabel(cs1b, fmt={0.63: "Frente de necrosis (Ω=1)"}, fontsize=8)
    ax1b.scatter(df_dano["Distancia_mm"], df_dano["Tiempo_min"], c="lime", edgecolor="k", s=12, zorder=5)
    ax1b.axvline(distancia_input, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1b.axhline(tiempo_input, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax1b.set_xlabel("Distancia radial al electrodo, r (mm)"); ax1b.set_ylabel("Tiempo (min)")
    ax1b.set_title("Fracción de daño Ω(r,t) — puntos verdes = datos reales de COMSOL")
    fig1b.colorbar(cf1b, ax=ax1b, label="Fracción de daño")
    st.pyplot(fig1b)

    st.subheader("🔍 Detección de patrones: velocidad de daño (dΩ/dt)")
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4))
    marcadores = {radios_comsol[0]: "o-", radios_comsol[1]: "s-", radios_comsol[2]: "^-"}
    for r in radios_comsol:
        sub = df_dano[df_dano["Distancia_mm"] == r].sort_values("Tiempo_min")
        ax2a.plot(sub["Tiempo_min"], sub["Fraccion_Dano"] * 100, marcadores[r], color=colores[r],
                   label=f"{r:.0f} mm", markersize=4)
        rate = np.gradient(sub["Fraccion_Dano"].values, sub["Tiempo_min"].values)
        ax2b.plot(sub["Tiempo_min"], rate, marcadores[r], color=colores[r], label=f"{r:.0f} mm", markersize=4)
    ax2a.set_xlabel("Tiempo (min)"); ax2a.set_ylabel("Fracción de daño (%)")
    ax2a.legend(); ax2a.grid(True, linestyle=":", alpha=0.5); ax2a.set_title("Daño acumulado")
    ax2b.set_xlabel("Tiempo (min)"); ax2b.set_ylabel("dΩ/dt (velocidad de daño)")
    ax2b.legend(); ax2b.grid(True, linestyle=":", alpha=0.5); ax2b.set_title("Velocidad de daño")
    fig2.tight_layout()
    st.pyplot(fig2)
    st.caption("La velocidad de daño se dispara igual para los 3 radios entre t≈6-9 min "
                "(pico de dΩ/dt), consistente con la cinética de Arrhenius: la destrucción "
                "proteica se acelera exponencialmente una vez cruzado el umbral de 50-55°C.")

# =============================================================================
# PESTAÑA 2 — VASO SANGUINEO / HEAT-SINK
# =============================================================================
with tab2:
    st.markdown("""
    Predice la **temperatura en la pared del vaso sanguíneo** según su diámetro y el tiempo de
    ablación. A diferencia del modelo de daño, aquí se usa un **interpolador lineal exacto**
    (no un polinomio ni un GPR): con solo 5 tiempos × 3 diámetros, cualquier ajuste "suave"
    puede oscilar entre puntos (se probó y falló — ver nota abajo); la interpolación lineal
    conecta los puntos reales de COMSOL sin inventar curvatura entre ellos.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"🌡️ Temperatura en pared del vaso (D={diametro_input:.1f} mm, t={tiempo_input:.1f} min)",
            value=f"{pred_vaso_viva:.2f} °C",
        )
    with col2:
        if pred_vaso_viva >= 100:
            alerta = "🔴 Riesgo de vaporización (≥100°C)"
        elif pred_vaso_viva >= 50:
            alerta = "🟢 Rango de coagulación/necrosis (50-100°C)"
        else:
            alerta = "🟡 Aún no alcanza umbral de necrosis (<50°C)"
        st.metric("Estado térmico estimado en esa posición", alerta)

    st.subheader("📊 Respuesta térmica del vaso sanguíneo (COMSOL vs. interpolación IA)")
    t_cont2 = np.linspace(0, 10.0, 200)
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    marcadores_d = {diametros_comsol[0]: "^-", diametros_comsol[1]: "s-", diametros_comsol[2]: "o-"}
    colores_d = {diametros_comsol[0]: "green", diametros_comsol[1]: "orange", diametros_comsol[2]: "blue"}
    for d in diametros_comsol:
        sub = df_vaso[df_vaso["Diametro_mm"] == d].sort_values("Tiempo_min")
        ax3.plot(sub["Tiempo_min"], sub["Temperatura_C"], marcadores_d[d], color=colores_d[d],
                  alpha=0.8, label=f"COMSOL vaso {d:.0f} mm")
    Xc2 = np.column_stack((t_cont2, np.full_like(t_cont2, diametro_input)))
    pred_c2 = modelo_vaso.predict(Xc2)
    ax3.plot(t_cont2, pred_c2, color="purple", linestyle="--", linewidth=2.5,
              label=f"Interpolación IA (D={diametro_input:.2f} mm)")
    ax3.plot(tiempo_input, pred_vaso_viva, "kX", markersize=12, label="Punto en vivo seleccionado")
    ax3.set_xlabel("Tiempo (min)"); ax3.set_ylabel("Temperatura (°C)")
    ax3.set_title("Mapeo de curvas térmicas del vaso sin oscilaciones matemáticas")
    ax3.legend(); ax3.grid(True, linestyle=":", alpha=0.5)
    st.pyplot(fig3)

    with st.expander("🔎 Por qué se descartó el polinomio y el GPR para este modelo"):
        st.markdown("""
        Se probaron polinomios de grado 2 (subajusta la meseta térmica en varios °C), grado 3
        (oscila de forma irreal entre puntos: sube, baja y vuelve a subir) y grado 4 —ajuste
        exacto, tantos parámetros como datos— (sobreajuste severo tipo Runge: llega a predecir
        que la temperatura *cae* con el tiempo, físicamente imposible). Un Gaussian Process
        también es una opción razonable, pero la interpolación lineal por tramos es la más
        simple y transparente que garantiza *cero* oscilaciones dentro del rango entrenado.
        """)

    with st.expander("🔎 Observación: la temperatura sube (no baja) con el diámetro"):
        st.markdown(f"""
        En la meseta (t=10 min): D=1mm → {df_vaso[(df_vaso.Diametro_mm==diametros_comsol[0])].Temperatura_C.max():.2f}°C,
        D=3mm → {df_vaso[(df_vaso.Diametro_mm==diametros_comsol[1])].Temperatura_C.max():.2f}°C,
        D=5mm → {df_vaso[(df_vaso.Diametro_mm==diametros_comsol[2])].Temperatura_C.max():.2f}°C.
        A primera vista podría parecer contra-intuitivo (uno esperaría que un vaso más grande
        enfríe más y por lo tanto la pared quede más fría). Puede explicarse por la geometría
        del modelo: en `parametros.txt`, la posición del vaso se define como
        `y_vaso = R_tumor + D_vaso + 1mm`, por lo que al aumentar el diámetro también cambia la
        posición relativa del vaso respecto al electrodo. Vale la pena que el grupo lo revise
        y lo mencione como parte de la discusión del proyecto.
        """)

    st.subheader("🗺️ Mapa de temperatura T(D, t) del vaso")
    D_grid = np.linspace(1, 5, 60)
    t_grid2 = np.linspace(0, 10, 60)
    DD, TT2 = np.meshgrid(D_grid, t_grid2)
    TEMP_map = modelo_vaso.predict(np.column_stack((TT2.ravel(), DD.ravel()))).reshape(DD.shape)
    fig3b, ax3b = plt.subplots(figsize=(10, 4.5))
    cf3b = ax3b.contourf(DD, TT2, TEMP_map, levels=25, cmap="inferno")
    ax3b.scatter(df_vaso["Diametro_mm"], df_vaso["Tiempo_min"], c="cyan", edgecolor="k", s=25, zorder=5)
    ax3b.axvline(diametro_input, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax3b.axhline(tiempo_input, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax3b.set_xlabel("Diámetro del vaso (mm)"); ax3b.set_ylabel("Tiempo (min)")
    ax3b.set_title("Temperatura T(D,t) — la interpolación lineal se ve 'facetada', es esperado")
    fig3b.colorbar(cf3b, ax=ax3b, label="Temperatura (°C)")
    st.pyplot(fig3b)
    st.caption("A diferencia del mapa de daño (suave), este mapa muestra quiebres visibles: "
                "es la firma característica de una interpolación lineal exacta, no un error.")

# =============================================================================
# PESTAÑA 3 — SENSIBILIDAD A LA POTENCIA (extensión analítica, NO ENTRENADA)
# =============================================================================
with tab3:
    st.warning("""
    **Esta pestaña NO es un modelo de IA entrenado.** Los datos disponibles corresponden
    únicamente a V0 = 22 V — no existe en el proyecto un barrido paramétrico de voltaje. Lo que
    sigue es una **extensión analítica** basada en las ecuaciones del propio proyecto, no un
    resultado predictivo validado con datos.
    """)

    st.markdown(r"""
    ### Justificación física
    $$Q_{RF} = \sigma |\nabla V|^2 \qquad \Rightarrow \qquad
    \Delta T_{meseta}(V_0) \approx \Delta T_{meseta}(22\text{V}) \cdot \left(\frac{V_0}{22}\right)^2$$

    La ecuación de Pennes que resuelve COMSOL es lineal en el término fuente $Q_{RF}$, así que
    el incremento de temperatura en estado estacionario escala aproximadamente con $V_0^2$.
    """)

    V0_ref = 22.0
    T_base = 37.0
    dT_ref = float(df_vaso[df_vaso["Tiempo_min"] == df_vaso["Tiempo_min"].max()]["Temperatura_C"].mean()) - T_base

    V_grid = np.linspace(10, 35, 100)
    T_est = T_base + dT_ref * (V_grid / V0_ref) ** 2

    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    ax4.plot(V_grid, T_est, color="darkred", linewidth=2.5, label="Estimación teórica ΔT ∝ V² (no simulada)")
    ax4.scatter([V0_ref], [T_base + dT_ref], color="black", zorder=5, s=80,
                 label=f"Único dato real disponible (V₀={V0_ref:.0f}V, COMSOL)")
    ax4.axhline(100, color="gray", linestyle=":", label="Umbral de vaporización (100°C)")
    ax4.set_xlabel("Voltaje aplicado V₀ (V)"); ax4.set_ylabel("Temperatura de meseta estimada (°C)")
    ax4.set_title("Sensibilidad teórica de la meseta térmica al voltaje")
    ax4.legend(); ax4.grid(alpha=0.3)
    st.pyplot(fig4)

    V_input = st.slider("Voltaje a evaluar (V):", 10.0, 35.0, 22.0, step=0.5)
    T_pred_teorica = T_base + dT_ref * (V_input / V0_ref) ** 2
    st.metric("Meseta térmica estimada (extrapolación teórica)", f"{T_pred_teorica:.1f} °C")
    st.caption("No reemplaza una simulación COMSOL real a otros voltajes.")

# =============================================================================
# PESTAÑA 4 — COMPARACION LADO A LADO
# =============================================================================
with tab4:
    st.markdown(f"""
    Vista general de los tres análisis para el instante **t = {tiempo_input:.1f} min**
    seleccionado en el sidebar.
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🔥 Daño vs. distancia**")
        r_fino = np.linspace(4, 20, 100)
        p_r = np.clip(modelo_dano.predict(np.column_stack((np.full_like(r_fino, tiempo_input), r_fino))), 0, 1)
        figc1, axc1 = plt.subplots(figsize=(4, 3.3))
        axc1.plot(r_fino, p_r, color="firebrick")
        axc1.axvline(distancia_input, color="gray", linestyle="--", linewidth=1)
        axc1.scatter(df_dano[np.isclose(df_dano.Tiempo_min, tiempo_input, atol=0.6)]["Distancia_mm"],
                      df_dano[np.isclose(df_dano.Tiempo_min, tiempo_input, atol=0.6)]["Fraccion_Dano"],
                      color="k", s=15, zorder=5)
        axc1.set_xlabel("r (mm)"); axc1.set_ylabel("Daño"); axc1.grid(alpha=0.3)
        st.pyplot(figc1)

    with c2:
        st.markdown("**🩸 Temperatura vs. diámetro**")
        d_fino = np.linspace(1, 5, 100)
        p_d = modelo_vaso.predict(np.column_stack((np.full_like(d_fino, tiempo_input), d_fino)))
        figc2, axc2 = plt.subplots(figsize=(4, 3.3))
        axc2.plot(d_fino, p_d, color="teal")
        axc2.axvline(diametro_input, color="gray", linestyle="--", linewidth=1)
        axc2.set_xlabel("D (mm)"); axc2.set_ylabel("Temp (°C)"); axc2.grid(alpha=0.3)
        st.pyplot(figc2)

    with c3:
        st.markdown("**⚡ Meseta vs. voltaje (teórico)**")
        figc3, axc3 = plt.subplots(figsize=(4, 3.3))
        axc3.plot(V_grid, T_est, color="darkred")
        axc3.axvline(22, color="gray", linestyle="--", linewidth=1)
        axc3.set_xlabel("V₀ (V)"); axc3.set_ylabel("T meseta (°C)"); axc3.grid(alpha=0.3)
        st.pyplot(figc3)

    st.info("Las líneas grises punteadas marcan los valores actualmente seleccionados en el "
             "sidebar (distancia, diámetro) o el valor de referencia (22V).")

st.markdown("""
---
### 🛠️ Diagrama de Flujo del Proceso
`COMSOL Multiphysics` ➡️ `Lectura directa de los .txt / guardado en .csv` ➡️
`Entrenamiento (polinomial para daño, interpolación exacta para el vaso)` ➡️
`Despliegue en Streamlit Web App`
""")
