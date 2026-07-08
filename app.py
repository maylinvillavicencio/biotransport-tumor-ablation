import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =====================================================================
# CLASE MODELO: DEFINICIÓN REQUERIDA PARA LA DECODIFICACIÓN DE JOBLIB
# =====================================================================
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

# Configuración de la interfaz web
st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Daño Tisular mediante IA")
st.markdown("""
Esta aplicación funciona como un **Modelo Subrogado de Inteligencia Artificial** en tiempo real. 
Predice el comportamiento térmico y el daño celular de forma instantánea.
""")

# --- CARGA DEL MODELO BINARIO GENERADO POR TRAINING.PY ---
if os.path.exists('best_model.pkl') and os.path.getsize('best_model.pkl') > 0:
    try:
        modelo_cargado = joblib.load('best_model.pkl')
        modelo_dano = modelo_cargado['model_dano']
        modelo_vaso = modelo_cargado['model_vaso']
    except Exception as e:
        st.error(f"Error al decodificar 'best_model.pkl': {e}.")
        st.stop()
else:
    st.error("⚠️ No se detectó 'best_model.pkl'. Por favor ejecuta primero tu script 'training.py'.")
    st.stop()

# --- CARGA DE LOS DATOS HISTÓRICOS DESDE EL CSV DE VASOS ---
vessel_csv = "vessel_data.csv"
if os.path.exists(vessel_csv):
    df_vv = pd.read_csv(vessel_csv)
    t_vaso = df_vv["Tiempo (min)"].values
    temp_5mm = df_vv["Temperatura_Vaso_5mm (C)"].values
    temp_3mm = df_vv["Temperatura_Vaso_3mm (C)"].values
    temp_1mm = df_vv["Temperatura_Vaso_1mm (C)"].values
else:
    st.error(f"⚠️ Falta el archivo '{vessel_csv}' requerido para graficar las bases.")
    st.stop()

# --- MENÚ INTERACTIVO (CONTROLES LATERALES) ---
st.sidebar.header("🕹️ Parámetros de Predicción en Vivo")
tiempo_input = st.sidebar.slider("Tiempo de tratamiento (minutos):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Distancia analizada desde el electrodo (mm):", 4.0, 20.0, 8.0, step=0.5)
diametro_input = st.sidebar.slider("Diámetro real del Vaso Sanguíneo (mm):", 1.0, 5.0, 3.0, step=0.1)

# =====================================================================
# CÁLCULOS MULTIVARIABLES EN VIVO USANDO LA IA SUBROGADA
# =====================================================================
# 1. Predicción del Daño Tisular (Modelo Polinomial)
prediccion_dano_viva = np.clip(modelo_dano.predict([[tiempo_input, distancia_input]])[0], 0.0, 1.0)

# 2. Predicción de la Temperatura del Vaso (Modelo Interpolador)
prediccion_vaso_viva = modelo_vaso.predict([[tiempo_input, diametro_input]])[0]

# 3. Clasificación Automática del Estado Celular
if prediccion_dano_viva >= 0.7:
    estado_celular = "🔴 Necrosis Crítica (Tejido Ablacionado)"
elif prediccion_dano_viva >= 0.1:
    estado_celular = "🟡 Tejido Viable / Parcial"
else:
    estado_celular = "🟢 Tejido Sano / Sin Afectación"


# =====================================================================
# DESPLIEGUE VISUAL DE LOS INDICADORES SOLICITADOS (TARJETAS EN VIVO)
# =====================================================================
st.subheader("📊 Métricas de Predicción Instantánea")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📍 Daño Tisular Predicho", 
        value=f"{prediccion_dano_viva:.4f} ({prediccion_dano_viva * 100:.2f}%)"
    )

with col2:
    st.metric(
        label="⚠️ Estado Celular Estimado", 
        value=estado_celular
    )

with col3:
    st.metric(
        label="🩸 Temperatura del Vaso Predicha", 
        value=f"{prediccion_vaso_viva:.2f} °C"
    )

st.markdown("---")

# =====================================================================
# GRÁFICAS COMPLEMENTARIAS DINÁMICAS
# =====================================================================
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("📈 Curva de Daño Continuo")
    tiempos_continuos = np.linspace(0, 10.52, 200)
    X_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, distancia_input)))
    pred_dinamica = np.clip(modelo_dano.predict(X_dinamico), 0.0, 1.0)
    
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(tiempos_continuos, pred_dinamica, color='red', linestyle='--', linewidth=2, label=f'Predicción IA ({distancia_input} mm)')
    ax1.plot(tiempo_input, prediccion_dano_viva, marker='X', color='black', markersize=10, label='Punto actual')
    ax1.set_xlabel('Tiempo (min)')
    ax1.set_ylabel('Fracción de Daño')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    st.pyplot(fig1)

with col_graf2:
    st.subheader("🩸 Respuesta Térmica del Vaso")
    X_vaso_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, diametro_input)))
    pred_vaso_dinamica = modelo_vaso.predict(X_vaso_dinamico)
    
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.plot(t_vaso, temp_5mm, 'o:', color='blue', alpha=0.4, label='COMSOL 5mm')
    ax2.plot(t_vaso, temp_3mm, 's:', color='orange', alpha=0.4, label='COMSOL 3mm')
    ax2.plot(t_vaso, temp_1mm, '^:', color='green', alpha=0.4, label='COMSOL 1mm')
    
    ax2.plot(tiempos_continuos, pred_vaso_dinamica, color='purple', linestyle='-', linewidth=2.5, label=f'IA Vaso ({diametro_input} mm)')
    ax2.plot(tiempo_input, prediccion_vaso_viva, marker='X', color='black', markersize=10, label='Punto actual')
    ax2.set_xlabel('Tiempo (min)')
    ax2.set_ylabel('Temperatura (°C)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)
