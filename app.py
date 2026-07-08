import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Declaración idéntica para la lectura correcta del binario de Joblib
class COMSOLVesselInterpolator:
    def __init__(self, csv_path="vessel_data.csv"):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.times = df["Tiempo (min)"].values
            self.diameters = np.array([1.0, 3.0, 5.0])
            self.temps = np.zeros((len(self.times), 3))
            self.temps[:, 0] = df["Temperatura_Vaso_1mm (C)"].values
            self.temps[:, 1] = df["Temperatura_Vaso_3mm (C)"].values
            self.temps[:, 2] = df["Temperatura_Vaso_5mm (C)"].values
        else:
            self.times = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
            self.diameters = np.array([1.0, 3.0, 5.0])
            self.temps = np.zeros((5, 3))

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for i in range(len(X)):
            t_val = np.clip(X[i, 0], 0.0, 10.52)
            d_val = np.clip(X[i, 1], 1.0, 5.0)
            v_1mm = np.interp(t_val, self.times, self.temps[:, 0])
            v_3mm = np.interp(t_val, self.times, self.temps[:, 1])
            v_5mm = np.interp(t_val, self.times, self.temps[:, 2])
            v_final = np.interp(d_val, self.diameters, [v_1mm, v_3mm, v_5mm])
            predictions.append(v_final)
        return np.array(predictions)

st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Daño Tisular mediante IA")
st.markdown("""
Esta aplicación funciona como un **Modelo Subrogado de Inteligencia Artificial**. 
Carga de forma dinámica los archivos de datos guardados desde simulaciones matemáticas.
""")

# --- SEGURIDAD EN LA CARGA DEL MODELO ---
if os.path.exists('best_model.pkl') and os.path.getsize('best_model.pkl') > 0:
    try:
        modelo_cargado = joblib.load('best_model.pkl')
        modelo_dano = modelo_cargado['model_dano']
        modelo_vaso = modelo_cargado['model_vaso']
    except Exception as e:
        st.error(f"Error al decodificar 'best_model.pkl': {e}.")
        st.stop()
else:
    st.error("⚠️ No se detectó 'best_model.pkl'.")
    st.stop()

# --- CARGA DINÁMICA DE LOS CONTROLES DESDE LOS CSV ---
vessel_csv = "vessel_data.csv"
if os.path.exists(vessel_csv):
    df_vessel_raw = pd.read_csv(vessel_csv)
    t_vaso = df_vessel_raw["Tiempo (min)"].values
    temp_5mm = df_vessel_raw["Temperatura_Vaso_5mm (C)"].values
    temp_3mm = df_vessel_raw["Temperatura_Vaso_3mm (C)"].values
    temp_1mm = df_vessel_raw["Temperatura_Vaso_1mm (C)"].values
else:
    st.error("⚠️ Falta el archivo 'vessel_data.csv'. Por favor ejecuta 'training.py' primero.")
    st.stop()

# Carga estática de respaldo para gráficas fijas de daño tisular
tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

# --- MENÚ INTERACTIVO (SIDEBAR) ---
st.sidebar.header("🕹️ Parámetros de Predicción en Vivo")
tiempo_input = st.sidebar.slider("Tiempo de tratamiento (minutos):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Distancia analizada desde el electrodo (mm):", 4.0, 20.0, 8.0, step=0.5)
diametro_input = st.sidebar.slider("Diámetro real del Vaso Sanguíneo (mm):", 1.0, 5.0, 3.0, step=0.1)

# Inferencia rápida del modelo subrogado
prediccion_viva = np.clip(modelo_dano.predict([[tiempo_input, distancia_input]])[0], 0.0, 1.0)
prediccion_vaso_viva = modelo_vaso.predict([[tiempo_input, diametro_input]])[0]

# Despliegue de métricas superiores
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📍 Daño Tisular Predicho", value=f"{prediccion_viva:.4f} ({prediccion_viva*100:.2f}%)")
with col2:
    if prediccion_viva >= 0.7:
        status = "🔴 Necrosis Crítica (>70%)"
    elif prediccion_viva >= 0.1:
        status = "🟡 Tejido Viable / Parcial"
    else:
        status = "🟢 Tejido Sano / Sin Afectación"
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

# --- GRÁFICA 2: COMPORTAMIENTO CONTINUO DEL VASO DESDE EL CSV ---
st.markdown("---")
st.subheader("🩸 Respuesta Térmica Exacta del Vaso Sanguíneo (Efecto Heat-Sink)")
st.markdown(f"La línea morada representa la predicción exacta de la IA para un vaso de **{diametro_input:.2f} mm** basada en los datos dinámicos.")

X_vaso_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, diametro_input)))
pred_vaso_dinamica = modelo_vaso.predict(X_vaso_dinamico)

fig3, ax3 = plt.subplots(figsize=(10, 4))
# Graficamos las curvas cargadas de forma automatizada desde el CSV de vasos
ax3.plot(t_vaso, temp_5mm, 'o-', color='blue', alpha=0.7, label='COMSOL Vaso 5 mm')
ax3.plot(t_vaso, temp_3mm, 's-', color='orange', alpha=0.7, label='COMSOL Vaso 3 mm')
ax3.plot(t_vaso, temp_1mm, '^-', color='green', alpha=0.7, label='COMSOL Vaso 1 mm')

ax3.plot(tiempos_continuos, pred_vaso_dinamica, color='purple', linestyle='--', linewidth=2.5, 
         label=f'Modelo Subrogado ({diametro_input:.2f} mm)')
ax3.plot(tiempo_input, prediccion_vaso_viva, marker='X', color='black', markersize=12, label='Punto en Vivo')

ax3.set_xlabel('Tiempo (min)')
ax3.set_ylabel('Temperatura (°C)')
ax3.set_title('Mapeo de Curvas Térmicas Extraídas de vessel_data.csv')
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.5)
st.pyplot(fig3)
