
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Daño Tisular mediante IA")
st.markdown("""
Esta aplicación web interactiva funciona como un **Modelo Subrogado de Inteligencia Artificial Colectivo**. 
Utiliza regresores entrenados con soluciones numéricas multifísicas provenientes de COMSOL Multiphysics.
""")

# --- CARGA DEL MODELO PRE-ENTRENADO CON ADAPTACIÓN DE SEGURIDAD ---
if os.path.exists('best_model.pkl') and os.path.getsize('best_model.pkl') > 0:
    try:
        modelo_cargado = joblib.load('best_model.pkl')
        if isinstance(modelo_cargado, dict) and 'model_dano' in modelo_cargado:
            modelo_dano = modelo_cargado['model_dano']
            modelo_vaso = modelo_cargado['model_vaso']
        else:
            st.error("⚠️ El archivo 'best_model.pkl' detectado es el antiguo. Por favor ejecuta el nuevo 'training.py' localmente y sube el archivo generado a tu GitHub.")
            st.stop()
    except Exception as e:
        st.error(f"Error al decodificar el archivo del modelo: {e}.")
        st.stop()
else:
    st.error("⚠️ No se encontró el archivo 'best_model.pkl'. Asegúrate de subirlo a tu repositorio.")
    st.stop()

# --- DATOS HISTÓRICOS REALES COMSOL ---
tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

t_vaso = np.array([0, 2.5, 5, 7.5, 10])
temp_5mm = np.array([37.01357406218045, 91.20003415821034, 93.83253548911045, 94.44218612048912, 94.62108754938538])
temp_3mm = np.array([37.01343745229815, 90.39493541249876, 92.68835103219796, 93.24920048749480, 93.42238392224195])
temp_1mm = np.array([37.01332988963907, 89.28820214853024, 91.82831014749000, 92.35752500000000, 92.51853700000000])

# --- CONTROLES INTERACTIVOS (SIDEBAR) ---
st.sidebar.header("🕹️ Parámetros de Predicción en Vivo")
tiempo_input = st.sidebar.slider("Tiempo de tratamiento (minutos):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Distancia analizada desde el electrodo (mm):", 4.0, 20.0, 8.0, step=0.5)

# NUEVO SLIDER: Control directo por milímetros de diámetro real
diametro_input = st.sidebar.slider("Diámetro real del Vaso Sanguíneo (mm):", 1.0, 5.0, 3.0, step=0.1, 
                                    help="Ajusta el diámetro anatómico del vaso. La IA interpolará la transferencia de calor de forma continua.")

# Predicciones ejecutadas por la IA
prediccion_viva = modelo_dano.predict([[tiempo_input, distancia_input]])[0]
prediccion_viva = np.clip(prediccion_viva, 0.0, 1.0) 

prediccion_vaso_viva = modelo_vaso.predict([[tiempo_input, diametro_input]])[0]

# Panel de métricas superiores
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📍 Daño Tisular Predicho", value=f"{prediccion_viva:.4f} ({prediccion_viva*100:.2f}%)")
with col2:
    status = "🔴 Necrosis Crítica (>70%)" if prediccion_viva >= 0.7 else "🟡 Tejido Viable / Parcial"
    st.metric(label="⚠️ Estado Celular Estimado", value=status)
with col3:
    st.metric(label="🩸 Temperatura del Vaso Predicha", value=f"{prediccion_vaso_viva:.2f} °C")

# --- GRÁFICA 1: DAÑO TISULAR ---
st.subheader("📊 Análisis Gráfico de Curvas de Daño Continuas")
tiempos_continuos = np.linspace(0, 10.52, 200)

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(tiempos, dano_4mm, color='blue', alpha=0.6, label='COMSOL Histórico (4 mm)')
ax.scatter(tiempos, dano_12mm, color='orange', alpha=0.6, label='COMSOL Histórico (12 mm)')
ax.scatter(tiempos, dano_20mm, color='green', alpha=0.6, label='COMSOL Histórico (20 mm)')

X_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, distancia_input)))
pred_dinamica = np.clip(modelo_dano.predict(X_dinamico), 0.0, 1.0)

ax.plot(tiempos_continuos, pred_dinamica, color='red', linestyle='--', linewidth=2.5, label=f'Predicción IA Continua ({distancia_input} mm)')
ax.plot(tiempo_input, prediccion_viva, marker='X', color='black', markersize=12, label='Punto Temporal en Vivo')
ax.set_xlabel('Tiempo de Exposición (min)')
ax.set_ylabel('Fracción de Daño Celular')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')
st.pyplot(fig)

# --- GRÁFICA 2: COMPORTAMIENTO CONTINUO DEL VASO (CORREGIDA) ---
st.markdown("---")
st.subheader("🩸 Efecto Heat-Sink: Curva de Enfriamiento Dinámico según el Diámetro del Vaso")
st.markdown(f"La línea morada representa la predicción matemática continua de la IA para un vaso con un diámetro exacto de **{diametro_input:.2f} mm**.")

X_vaso_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, diametro_input)))
pred_vaso_dinamica = modelo_vaso.predict(X_vaso_dinamico)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.scatter(t_vaso, temp_5mm, color='blue', alpha=0.5, marker='o', label='COMSOL Histórico (Vaso 5 mm)')
ax3.scatter(t_vaso, temp_3mm, color='orange', alpha=0.5, marker='s', label='COMSOL Histórico (Vaso 3 mm)')
ax3.scatter(t_vaso, temp_1mm, color='green', alpha=0.5, marker='^', label='COMSOL Histórico (Vaso 1 mm)')

ax3.plot(tiempos_continuos, pred_vaso_dinamica, color='purple', linestyle='--', linewidth=2.5, 
         label=f'Curva IA Interpolada ({diametro_input:.2f} mm)')
ax3.plot(tiempo_input, prediccion_vaso_viva, marker='X', color='black', markersize=12, label='Punto en Vivo Vaso')

ax3.axhline(94.62, color='red', linestyle=':', alpha=0.4, label='Límite Térmico Máximo (~94.6 °C)')
ax3.set_xlabel('Tiempo (min)')
ax3.set_ylabel('Temperatura (°C)')
ax3.set_title('Respuesta Térmica del Vaso Sanguíneo Calibrada en Milímetros Reales')
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.5)
st.pyplot(fig3)

# --- DETECCIÓN DE PATRONES ---
st.markdown("---")
st.subheader("🔍 Detección de Patrones: Efecto de la Distancia sobre la Velocidad de Daño")
rate_4mm = np.gradient(dano_4mm, tiempos)
rate_12mm = np.gradient(dano_12mm, tiempos)
rate_20mm = np.gradient(dano_20mm, tiempos)

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))
ax2a.plot(tiempos, dano_4mm*100, 'o-', color='blue', label='4 mm')
ax2a.plot(tiempos, dano_12mm*100, 's-', color='orange', label='12 mm')
ax2a.plot(tiempos, dano_20mm*100, '^-', color='green', label='20 mm')
ax2a.set_xlabel('Tiempo (min)')
ax2a.set_ylabel('Fracción de daño (%)')
ax2a.legend()
ax2a.grid(True, linestyle=':', alpha=0.5)

ax2b.plot(tiempos, rate_4mm, 'o-', color='blue', label='4 mm')
ax2b.plot(tiempos, rate_12mm, 's-', color='orange', label='12 mm')
ax2b.plot(tiempos, rate_20mm, '^-', color='green', label='20 mm')
ax2b.set_xlabel('Tiempo (min)')
ax2b.set_ylabel('dD/dt (velocidad de daño)')
ax2b.legend()
ax2b.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig2)
