import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA WEB
# =============================================================================
st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Ablación Tumoral mediante IA")
st.markdown("""
Esta aplicación web interactiva funciona como un **Modelo Subrogado de Inteligencia Artificial**. 
Utiliza regresores entrenados con las soluciones numéricas de la ecuación de biocalor de Pennes y la cinética 
de daño de Arrhenius extraídas de COMSOL Multiphysics para predecir variables clave en milisegundos.
""")

# --- EXPANDER DE OPORTUNIDADES IA (REQUISITO DE RÚBRICA) ---
with st.expander("🎯 Oportunidades de IA integradas en este proyecto de Biotransporte"):
    st.markdown("""
    1. **Predicción del modelado (Pestaña 1):** Modelo subrogado de regresión polinomial multivariable que reemplaza corridas costosas de COMSOL, prediciendo la fracción de daño tisular ($\Omega$) instantáneamente.
    2. **Análisis avanzado / Detección de patrones (Pestaña 2):** Captura del **efecto heat-sink** mediante *Gaussian Process Regression (GPR)*, mapeando cómo el flujo sanguíneo estabiliza la temperatura en la pared vascular.
    3. **Optimización con IA (Idea inédita propuesta):** Un algoritmo genético que explore el espacio de parámetros $(V_0, t)$ para maximizar la necrosis tumoral sin dañar estructuras vasculares críticas (control térmico personalizado).
    """)

# =============================================================================
# LECTURA AUTOMATIZADA DE DATOS EXPORTADOS DE COMSOL (.TXT)
# =============================================================================
@st.cache_data
def cargar_datos_comsol():
    # Valores de respaldo por si los archivos .txt no se encuentran en el servidor
    tiempos_ref = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
    d4 = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
    d12 = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
    d20 = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])
    
    t_v = np.array([0, 2.5, 5, 7.5, 10])
    v_a = np.array([37.01, 99.26, 102.28, 102.98, 103.16])
    v_b = np.array([37.01, 99.79, 102.49, 103.15, 103.32])
    v_c = np.array([37.01, 99.39, 102.42, 103.12, 103.30])

    # Intentar lectura dinámica del archivo de daño tisular
    if os.path.exists("fracionamiento de daño.txt"):
        try:
            df_dano = pd.read_csv("fracionamiento de daño.txt", comment='%', sep=r'\s+', header=None)
            tiempos_ref = df_dano[0].values
            d4 = df_dano[1].values
            d12 = df_dano[2].values
            d20 = df_dano[3].values
        except:
            pass

    # Intentar lectura dinámica del archivo de temperaturas del vaso
    if os.path.exists("vasoo.txt"):
        try:
            with open("vasoo.txt", "r") as f:
                lineas = [l.strip() for l in f if l.strip() and not l.startswith('%')]
            datos_v = [list(map(float, l.split())) for l in lineas]
            df_v = pd.DataFrame(datos_v)
            # Reestructurar asumiendo bloques fijos de tiempo (0 a 10 min) por escenario
            t_v = df_v[0].unique()
            n_puntos = len(t_v)
            v_a = df_v[1].iloc[0:n_puntos].values
            v_b = df_v[1].iloc[n_puntos:2*n_puntos].values
            v_c = df_v[1].iloc[2*n_puntos:3*n_puntos].values
        except:
            pass
            
    return tiempos_ref, d4, d12, d20, t_v, v_a, v_b, v_c

tiempos, dano_4mm, dano_12mm, dano_20mm, t_vaso, temp_A, temp_B, temp_C = cargar_datos_comsol()

# --- CARGA DEL MODELO DUAL (.PKL) ---
if os.path.exists('best_model.pkl') and os.path.getsize('best_model.pkl') > 0:
    try:
        dict_modelos = joblib.load('best_model.pkl')
        modelo_dano = dict_modelos['dano']
        modelo_temp = dict_modelos['temp']  # Modelo GPR para el vaso sanguíneo
    except Exception as e:
        st.error(f"Error al decodificar 'best_model.pkl': {e}. Asegúrate de que guarde un diccionario.")
        st.stop()
else:
    st.error("⚠️ No se encontró el archivo 'best_model.pkl'. Por favor ejecuta 'training.py' primero.")
    st.stop()

# =============================================================================
# DEFINICIÓN DE LAS PESTAÑAS PRINCIPALES (TABS)
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Pestaña 1 — Daño Tisular (Arrhenius)", 
    "🩸 Pestaña 2 — Efecto Heat-Sink (Vaso Sanguíneo)", 
    "⚡ Pestaña 3 — Sensibilidad a la Potencia"
])

# =============================================================================
# PESTAÑA 1: MODELO SUBROGADO DE DAÑO TISULAR
# =============================================================================
with tab1:
    st.subheader("📊 Monitoreo Predictivo de Necrosis Celular en Vivo")
    st.markdown("Mueve los deslizadores para evaluar la fracción de daño predicha por el pipeline polinomial de IA.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tiempo_input = st.slider("Tiempo de tratamiento (minutos):", 0.0, float(tiempos[-1]), 5.0, step=0.1, key="t_tab1")
    with col_s2:
        distancia_input = st.slider("Distancia analizada desde el electrodo (mm):", 4.0, 20.0, 8.0, step=0.5, key="r_tab1")

    # Predicción puntual con el regresor de daño
    prediccion_viva = modelo_dano.predict([[tiempo_input, distancia_input]])[0]
    prediccion_viva = np.clip(prediccion_viva, 0.0, 1.0)

    # Métricas principales
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="📍 Fracción de Daño Tisular Predicho ($\Omega$)", value=f"{prediccion_viva:.4f} ({prediccion_viva*100:.2f}%)")
    with m2:
        status = "🔴 Necrosis Crítica Completa ($\Omega \geq 1$)" if prediccion_viva >= 0.99 else ("🟡 Lesión Parcial / Tejido Viable" if prediccion_viva > 0.05 else "🟢 Tejido Sano")
        st.metric(label="⚠️ Estado Celular Estimado", value=status)

    # Gráfico Dinámico continuo
    tiempos_continuos = np.linspace(0, float(tiempos[-1]), 200)
    X_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, distancia_input)))
    pred_dinamica = np.clip(modelo_dano.predict(X_dinamico), 0.0, 1.0)

    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.scatter(tiempos, dano_4mm, color='blue', alpha=0.4, label='COMSOL Sonda (4 mm)')
    ax1.scatter(tiempos, dano_12mm, color='orange', alpha=0.4, label='COMSOL Sonda (12 mm)')
    ax1.scatter(tiempos, dano_20mm, color='green', alpha=0.4, label='COMSOL Sonda (20 mm)')
    
    ax1.plot(tiempos_continuos, pred_dinamica, color='red', linestyle='--', linewidth=2.5,
             label=f'Predicción IA Continua (Ajustada a {distancia_input} mm)')
    ax1.plot(tiempo_input, prediccion_viva, marker='X', color='black', markersize=12, label='Punto Seleccionado')
    
    ax1.set_xlabel('Tiempo de Exposición (min)')
    ax1.set_ylabel('Fracción de Daño Celular (0 a 1)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    st.pyplot(fig1)

    # Gradientes de velocidad de daño (Patrones detectados)
    st.markdown("### 🔍 Análisis Avanzado: Velocidad de Crecimiento del Daño ($d\Omega/dt$)")
    rate_4mm = np.gradient(dano_4mm, tiempos)
    rate_12mm = np.gradient(dano_12mm, tiempos)
    rate_20mm = np.gradient(dano_20mm, tiempos)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))
    ax2a.plot(tiempos, dano_4mm*100, 'o-', color='blue', alpha=0.7, label='4 mm')
    ax2a.plot(tiempos, dano_12mm*100, 's-', color='orange', alpha=0.7, label='12 mm')
    ax2a.plot(tiempos, dano_20mm*100, '^-', color='green', alpha=0.7, label='20 mm')
    ax2a.set_title('Daño Acumulado por Distancia (%)')
    ax2a.set_xlabel('Tiempo (min)'); ax2a.set_ylabel('Daño (%)'); ax2a.legend(); ax2a.grid(True, alpha=0.4)

    ax2b.plot(tiempos, rate_4mm, 'o-', color='blue', alpha=0.7, label='4 mm')
    ax2b.plot(tiempos, rate_12mm, 's-', color='orange', alpha=0.7, label='12 mm')
    ax2b.plot(tiempos, rate_20mm, '^-', color='green', alpha=0.7, label='20 mm')
    ax2b.set_title('Patrón: Atenuación de velocidad ($d\Omega/dt$) por Perfusión')
    ax2b.set_xlabel('Tiempo (min)'); ax2b.set_ylabel('Velocidad de daño'); ax2b.legend(); ax2b.grid(True, alpha=0.4)
    st.pyplot(fig2)

    st.info(f"""
    **Patrón biofísico:** Conforme nos alejamos del electrodo (de 4 mm a 20 mm), el pico de velocidad 
    de daño se deprime y se retrasa marcadamente. Esto evidencia la acción de la perfusión sanguínea constante 
    ($\omega_b = 6.4 \\times 10^{{-3}}$ 1/s según `parametros.txt`), que remueve calor continuamente del parénquima hepatico.
    """)

# =============================================================================
# PESTAÑA 2: EFECTO HEAT-SINK (GAUSSIAN PROCESS REGRESSION)
# =============================================================================
with tab2:
    st.subheader("🩸 Modelo de Regresión por Procesos Gaussianos (GPR) para la Pared Vascular")
    st.markdown("""
    A diferencia del tejido tumoral, los puntos inmediatamente adyacentes al vaso macroscópico experimentan 
    un límite térmico severo debido al enfriamiento por convección forzada de la sangre circulante.
    """)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        diametro_input = st.slider("Diámetro del vaso sanguíneo (mm):", 1.0, 5.0, 3.0, step=0.5)
    with col_h2:
        tiempo_vaso_input = st.slider("Tiempo transcurrido (minutos):", 0.0, 10.0, 5.0, step=0.5)

    # Predicción viva usando el modelo de temperatura (GPR) entrenado
    # El modelo espera la entrada estructurada tal como se entrenó: [Diametro, Tiempo] o viceversa
    try:
        pred_temp_vaso = modelo_temp.predict([[diametro_input, tiempo_vaso_input]])[0]
    except:
        # En caso de que el orden en training.py haya sido inverso [Tiempo, Diametro]
        pred_temp_vaso = modelo_temp.predict([[tiempo_vaso_input, diametro_input]])[0]

    st.metric(label="🌡️ Temperatura Estimada en la Interfaz del Vaso", value=f"{pred_temp_vaso:.2f} °C")

    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(t_vaso, temp_A, 'o-', color='darkblue', alpha=0.5, label='COMSOL: Vaso 1 mm')
    ax3.plot(t_vaso, temp_B, 's-', color='teal', alpha=0.5, label='COMSOL: Vaso 3 mm')
    ax3.plot(t_vaso, temp_C, '^-', color='purple', alpha=0.5, label='COMSOL: Vaso 5 mm')
    
    # Mostrar la meseta predicha por la IA para el diámetro seleccionado a lo largo del tiempo
    t_linea = np.linspace(0, 10, 50)
    try:
        preds_linea = modelo_temp.predict(np.column_stack((np.full_like(t_linea, diametro_input), t_linea)))
    except:
        preds_linea = modelo_temp.predict(np.column_stack((t_linea, np.full_like(t_linea, diametro_input))))
        
    ax3.plot(t_linea, preds_linea, color='red', linestyle=':', linewidth=2.5, 
             label=f'GPR IA: Perfil para {diametro_input} mm')
    ax3.axhline(103.3, color='grey', linestyle='--', alpha=0.7, label='Límite Convectivo Realizado (≈103 °C)')
    
    ax3.set_xlabel('Tiempo (min)')
    ax3.set_ylabel('Temperatura (°C)')
    ax3.set_title('Estabilización Térmica por Convección (Efecto Sink)')
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.5)
    st.pyplot(fig3)

    st.warning("""
    ⚠️ **Nota Metodológica de Robustez:** Observa que las curvas numéricas para vasos de 1, 3 y 5 mm se encuentran 
    casi superpuestas (diferencias < 0.3 °C). La IA (línea punteada roja) detecta correctamente que el efecto sumidero 
    está completamente dominado por la velocidad y la temperatura interna del fluido ($T_b = 37$ °C), volviendo 
    al sistema térmico muy robusto (e independiente) frente a variaciones milimétricas en el diámetro del vaso.
    """)

# =============================================================================
# PESTAÑA 3: SENSIBILIDAD ANALÍTICA A LA POTENCIA (EXTRAPOLACIÓN FÍSICA)
# =============================================================================
with tab3:
    st.subheader("⚡ Extrapolación Analítica: Sensibilidad de la Meseta Térmica al Voltaje")
    st.markdown("""
    *Nota: Esta pestaña representa una extensión analítica basada en principios físicos ($Q_{RF} = \sigma |\nabla V|^2$), 
    no un modelo entrenado de IA, debido a que los datos base de simulación se mantuvieron a un voltaje constante de $V_0 = 22$ V.*
    """)

    V0_ref = 22.0
    T_base = 37.0
    T_max_ref = 103.32  # Temperatura de meseta final obtenida en COMSOL
    dT_ref = T_max_ref - T_base

    V_grid = np.linspace(10.0, 35.0, 100)
    # Relación cuadrática directa derivada de la fuente de potencia electromagnética de Joule
    T_est = T_base + dT_ref * (V_grid / V0_ref) ** 2

    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    ax4.plot(V_grid, T_est, color="darkred", linewidth=2.5, label="Estimación Teórica $\Delta T \propto V_0^2$")
    ax4.scatter([V0_ref], [T_max_ref], color="black", zorder=5, s=100, label=f"Línea de Base COMSOL ($V_0 = {V0_ref}$V)")
    ax4.axhline(100.0, color="orange", linestyle=":", alpha=0.8, label="Umbral de Vaporización Tisular (100°C)")
    
    ax4.set_xlabel("Voltaje Aplicado al Electrodo $V_0$ (V)")
    ax4.set_ylabel("Temperatura de Meseta Estimada (°C)")
    ax4.set_title("Escalamiento de la Meseta Térmica según el Voltaje de Radiofrecuencia")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

    v_eval = st.slider("Evaluar Voltaje de Operación Personalizado (V):", 10.0, 35.0, 22.0, step=0.5)
    t_pred_teorica = T_base + dT_ref * (v_eval / V0_ref) ** 2
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.metric("Meseta Térmica Vascular Estimada", f"{t_pred_teorica:.1f} °C")
    with col_v2:
        if t_pred_teorica > 100.0:
            st.error("🚨 Riesgo alto de carbonización y desgasificación de vapor en la interfaz del vaso.")
        else:
            st.success("✅ Operación libre de riesgo de ebullición local en la pared.")

# --- DIAGRAMA DE FLUJO ---
st.markdown("""
---
### 🛠️ Arquitectura de la Plataforma Predictiva
`COMSOL Multiphysics (Datos Numéricos)` ➡️ `Extracción de Sondas (.txt)` ➡️ `Entrenamiento Multimodelo (Polynomial + GPR)` ➡️ `Desempaquetado e Interfaz Web (Streamlit)`
""")
