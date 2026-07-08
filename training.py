# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# =====================================================================
# GENERACIÓN AUTOMÁTICA DE DATOS HISTÓRICOS DE LOS VASOS (vessel_data.csv)
# =====================================================================
# Creamos el archivo CSV estructurado con las lecturas exactas de tu archivo vasoo.txt
vessel_file = "vessel_data.csv"

datos_vaso_comsol = {
    "Tiempo (min)": [0.0, 2.5, 5.0, 7.5, 10.0],
    "Temperatura_Vaso_5mm (C)": [37.01357406218045, 91.20003415821034, 93.83253548911045, 94.44218612048912, 94.62108754938538],
    "Temperatura_Vaso_3mm (C)":  [37.01332988963907, 89.28820214853024, 91.82831014749000, 92.35752500000000, 92.51853700000000],
    "Temperatura_Vaso_1mm (C)": [37.01332988963907, 99.38768378853024, 102.41853714749703, 103.12056757724878, 103.30220118154881]
}

df_vessels_export = pd.DataFrame(datos_vaso_comsol)
df_vessels_export.to_csv(vessel_file, index=False)
print(f"✅ Archivo '{vessel_file}' verificado y guardado con éxito.")


# =====================================================================
# CLASE MODELO: INTERPOLADOR BILINEAL BASADO EN EL NUEVO CSV
# =====================================================================
class COMSOLVesselInterpolator:
    def __init__(self, csv_path="vessel_data.csv"):
        # Cargamos los datos directamente desde el CSV guardado
        df = pd.read_csv(csv_path)
        self.times = df["Tiempo (min)"].values
        self.diameters = np.array([1.0, 3.0, 5.0]) # Ordenados para interpolación continua
        
        # Mapeamos las columnas del CSV a nuestra matriz de interpolación (Filas: Tiempos, Columnas: Diámetros)
        self.temps = np.zeros((len(self.times), 3))
        self.temps[:, 0] = df["Temperatura_Vaso_1mm (C)"].values
        self.temps[:, 1] = df["Temperatura_Vaso_3mm (C)"].values
        self.temps[:, 2] = df["Temperatura_Vaso_5mm (C)"].values

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for i in range(len(X)):
            t_val = np.clip(X[i, 0], 0.0, 10.52)
            d_val = np.clip(X[i, 1], 1.0, 5.0)
            
            # Interpolación en el tiempo exacta por cada columna del CSV
            v_1mm = np.interp(t_val, self.times, self.temps[:, 0])
            v_3mm = np.interp(t_val, self.times, self.temps[:, 1])
            v_5mm = np.interp(t_val, self.times, self.temps[:, 2])
            
            # Interpolación en el espacio para el diámetro del slider
            v_final = np.interp(d_val, self.diameters, [v_1mm, v_3mm, v_5mm])
            predictions.append(v_final)
        return np.array(predictions)


# =====================================================================
# 1. ENTRENAMIENTO DEL MODELO DE DAÑO TISULAR DESDE TU TUMOR_DATA.CSV
# =====================================================================
print("Cargando datos de tumor_data.csv...")
try:
    # Leemos omitiendo las líneas de metadatos de COMSOL que empiezan con '%'
    df_tumor = pd.read_csv('tumor_data.csv', comment='%', sep=r'\s+', header=None)
    tiempos = df_tumor[0].values
    dano_4mm = df_tumor[1].values
    dano_12mm = df_tumor[2].values
    dano_20mm = df_tumor[3].values
except Exception as e:
    print(f"Aviso: No se pudo leer tumor_data.csv de forma automatizada ({e}). Usando respaldo estático.")
    tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
    dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
    dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
    dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

T_all = np.concatenate([tiempos, tiempos, tiempos])
Dist_all = np.concatenate([np.full_like(tiempos, 4), np.full_like(tiempos, 12), np.full_like(tiempos, 20)])
Dano_all = np.concatenate([dano_4mm, dano_12mm, dano_20mm])

X_dano = np.column_stack((T_all, Dist_all))
y_dano = Dano_all

model_dano = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model_dano.fit(X_dano, y_dano)

# =====================================================================
# 2. GENERACIÓN DEL MODELO SUBROGADO DEL VASO SANGUÍNEO
# =====================================================================
model_vaso = COMSOLVesselInterpolator(csv_path=vessel_file)

# =====================================================================
# 3. GUARDADO DEL PAQUETE COMPLETO (.PKL)
# =====================================================================
paquete_modelos = {
    'model_dano': model_dano,
    'model_vaso': model_vaso
}

joblib.dump(paquete_modelos, 'best_model.pkl')
print("¡Éxito rotundo! El modelo predictivo y tus datos de vasos han sido sincronizados en 'best_model.pkl'.")
