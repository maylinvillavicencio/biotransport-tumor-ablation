import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =====================================================================
# CLASE REQUERIDA PARA DECODIFICAR EL ARCHIVO .PKL
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

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(page_title="BioAI - Ablación Tumoral 2D", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte y Mapeo 2D")
st.markdown("""
Plataforma impulsada por IA que simula y renderiza el comportamiento térmico y el daño tisular basado en datos multifísicos.
""")

# --- CARGA DEL MODELO Y DATOS ---
if os.path.exists('best_model.pkl') and os.path.getsize('best_model.pkl') > 0:
    try:
        modelo_cargado = joblib.load('best_model.pkl')
        modelo_dano = modelo_cargado['model_dano']
        modelo_vaso = modelo_cargado['model_vaso']
    except Exception as e:
        st.error(f"Error al decodificar 'best_model.pkl': {e}.")
        st.stop()
else:
    st.error("⚠️ No se detectó 'best_model.pkl'. Ejecuta training.py primero.")
    st.stop()

vessel_csv = "vessel_data.csv"
if os.path.exists(vessel_csv):
    df_vessel_raw = pd.read_csv(vessel_csv)
    t_vaso = df_vessel_raw["Tiempo (min)"].values
    temp_5mm = df_vessel_raw["Temperatura_Vaso_5mm (C)"].values
    temp_3mm = df_vessel_raw["Temperatura_Vaso_3mm (C)"].values
    temp_1mm = df_vessel_raw["Temperatura_Vaso_1mm (C)"].values
else:
    st.error("⚠️ Falta el archivo 'vessel_data.csv'.")
    st.stop()

tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

# =====================================================================
# MENÚ LATERAL INTERACTIVO
# =====================================================================
st.sidebar.header("🕹️ Parámetros de Simulación")
tiempo_input = st.sidebar.slider("Tiempo de exposición (min):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Ubicación del Vaso (Distancia en mm):", 4.0, 20.0, 8.0, step=0.5)
diametro_input = st.sidebar.slider("Diámetro del Vaso (mm):", 1.0, 5.0, 3.0, step=0.1)

# Inferencia
prediccion_viva = np.clip(modelo_dano.predict([[tiempo_input, distancia_input]])[0], 0.0, 1.0)
prediccion_vaso_viva = modelo_vaso.predict([[tiempo_input, diametro_input]])[0]

# =====================================================================
# TARJETAS DE MÉTRICAS DINÁMICAS (SEPARADAS POR CONTEXTO)
# =====================================================================
st.subheader("📊 Monitoreo Celular y Térmico en Vivo")

# Fila 1: Contexto del Tejido/Tumor
st.markdown("##### 🔬 En el Tejido Tumoral (a la distancia seleccionada)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📍 Fracción de Daño Predicha", value=f"{prediccion_viva:.4f} ({prediccion_viva*100:.2f}%)")
with col2:
    status_tejido = "🔴 Necrosis Crítica (>70%)" if prediccion_viva >= 0.7 else ("🟡 Lesión Parcial" if prediccion_viva >= 0.1 else "🟢 Tejido Sano")
    st.metric(label="⚠️ Viabilidad Celular", value=status_tejido)
with col3:
    st.metric(label="📏 Distancia Evaluada", value=f"{distancia_input} mm")

# Fila 2: Contexto del Vaso Sanguíneo
st.markdown("##### 🩸 En el Vaso Sanguíneo Periférico")
col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    st.metric(label="🌡️ Temperatura Predicha", value=f"{prediccion_vaso_viva:.2f} °C")
with col_v2:
    if prediccion_vaso_viva >= 100.0:
        estado_v = "🔴 Ebullición / Riesgo Vascular"
    elif prediccion_vaso_viva >= 45.0:
        estado_v = "🟡 Heat-Sink Convectivo Activo"
    else:
        estado_v = "🟢 Rango Térmico Fisiológico"
    st.metric(label="🛡️ Estado Estructural", value=estado_v)
with col3: # Reutilizamos col3 visualmente para alinear
    pass 

st.markdown("---")

# =====================================================================
# MAPEO 2D: RECONSTRUCCIÓN ESPACIAL MEDIANTE IA
# =====================================================================
st.subheader("🗺️ Renderizado 2D Espacial del Daño Tisular")
st.markdown("Visualización transversal calculada milímetro a milímetro por la red neuronal a partir de la geometría del electrodo.")

# Creación de la malla 2D matemática
radio_maximo = 25
x_grid = np.linspace(-radio_maximo, radio_maximo, 150)
y_grid = np.linspace(-radio_maximo, radio_maximo, 150)
X, Y = np.meshgrid(x_grid, y_grid)
R = np.sqrt(X**2 + Y**2)

# Evaluamos la malla entera en la IA (limitando el radio exterior para evitar extrapolaciones locas)
R_flat = R.flatten()
R_clipped = np.clip(R_flat, 4.0, 20.0)
X_pred = np.column_stack((np.full_like(R_clipped, tiempo_input), R_clipped))

# Calculamos daño radial
damage_flat = np.clip(modelo_dano.predict(X_pred), 0.0, 1.0)
# Ajuste físico: Todo lo que esté a menos de 4mm del electrodo se considera necrosis total (100%)
damage_flat[R_flat < 4.0] = 1.0 
Damage_2D = damage_flat.reshape(X.shape)

fig2d, ax2d = plt.subplots(figsize=(8, 6))
# Mapa de calor de daño
c = ax2d.contourf(X, Y, Damage_2D, levels=np.linspace(0, 1, 30), cmap='turbo')
cbar = plt.colorbar(c, ax=ax2d)
cbar.set_label('Fracción de Daño Celular (0 a 1)', rotation=270, labelpad=20)

# Dibujamos el electrodo central
ax2d.plot(0, 0, marker='*', color='white', markersize=15, markeredgecolor='black', label='Electrodo RF')

# Dibujamos el Vaso Sanguíneo respetando su distancia y diámetro
ax2d.axvspan(distancia_input - diametro_input/2, distancia_input + diametro_input/2, 
             color='cyan', alpha=0.5, hatch='//', label=f'Vaso ({diametro_input} mm)')

ax2d.set_xlabel('Eje X (mm)')
ax2d.set_ylabel('Eje Y (mm)')
ax2d.set_title(f'Distribución Radial de Necrosis en el Minuto {tiempo_input:.1f}')
ax2d.legend(loc='upper left', framealpha=0.9)
ax2d.grid(True, linestyle=':', alpha=0.3)
ax2d.set_aspect('equal') # Mantiene la proporción circular real

st.pyplot(fig2d)

# =====================================================================
# GRÁFICAS DE CURVAS TRADICIONALES EN UNA SOLA FILA
# =====================================================================
st.markdown("---")
st.subheader("📈 Análisis de Tendencias Temporales (1D)")

col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    tiempos_continuos = np.linspace(0, 10.52, 200)
    X_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, distancia_input)))
    pred_dinamica = np.clip(modelo_dano.predict(X_dinamico), 0.0, 1.0)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(tiempos, dano_4mm, 'o:', color='blue', alpha=0.3, label='COMSOL 4 mm')
    ax1.plot(tiempos, dano_12mm, 's:', color='orange', alpha=0.3, label='COMSOL 12 mm')
    ax1.plot(tiempos, dano_20mm, '^:', color='green', alpha=0.3, label='COMSOL 20 mm')
    
    ax1.plot(tiempos_continuos, pred_dinamica, color='red', linestyle='-', linewidth=2.5, label=f'IA Continua ({distancia_input} mm)')
    ax1.plot(tiempo_input, prediccion_viva, marker='X', color='black', markersize=10, label='Minuto Evaluado')
    
    ax1.set_xlabel('Tiempo (min)')
    ax1.set_ylabel('Fracción de Daño')
    ax1.set_title('Evolución del Daño Tisular')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize='small')
    st.pyplot(fig1)

with col_graf2:
    X_vaso_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, diametro_input)))
    pred_vaso_dinamica = modelo_vaso.predict(X_vaso_dinamico)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(t_vaso, temp_5mm, 'o:', color='blue', alpha=0.4, label='COMSOL 5 mm')
    ax3.plot(t_vaso, temp_3mm, 's:', color='orange', alpha=0.4, label='COMSOL 3 mm')
    ax3.plot(t_vaso, temp_1mm, '^:', color='green', alpha=0.4, label='COMSOL 1 mm')

    ax3.plot(tiempos_continuos, pred_vaso_dinamica, color='purple', linestyle='-', linewidth=2.5, label=f'Modelo Vaso ({diametro_input} mm)')
    ax3.plot(tiempo_input, prediccion_vaso_viva, marker='X', color='black', markersize=10, label='Minuto Evaluado')

    ax3.set_xlabel('Tiempo (min)')
    ax3.set_ylabel('Temperatura (°C)')
    ax3.set_title('Efecto Heat-Sink del Vaso')
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='lower right', fontsize='small')
    st.pyplot(fig3)
