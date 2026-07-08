# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. MODELO DE DAÑO TISULAR (TUMOR)
# ==========================================
tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

T_all = np.concatenate([tiempos, tiempos, tiempos])
Dist_all = np.concatenate([np.full_like(tiempos, 4), np.full_like(tiempos, 12), np.full_like(tiempos, 20)])
Dano_all = np.concatenate([dano_4mm, dano_12mm, dano_20mm])

X_dano = np.column_stack((T_all, Dist_all))
y_dano = Dano_all

# Guardar los datos en el CSV por orden del repositorio
df = pd.DataFrame({'Tiempo_min': T_all, 'Distancia_mm': Dist_all, 'Fraccion_Dano': Dano_all})
df.to_csv('tumor_data.csv', index=False)

# Entrenar pipeline polinomial para daño
model_dano = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model_dano.fit(X_dano, y_dano)


# ==========================================
# 2. MODELO DE TEMPERATURA DEL VASO SANGUÍNEO (NUEVO)
# ==========================================
# Datos extraídos de vasoo.txt
t_vaso = np.array([0, 2.5, 5, 7.5, 10])
temp_A = np.array([37.01357406218045, 99.25661184628018, 102.28051861732422, 102.98079064649181, 103.16208754938538])
temp_B = np.array([37.01343745229815, 99.78851398109765, 102.48551032319796, 103.1450506874948, 103.32238392224195])
temp_C = np.array([37.01332988963907, 99.38768378853024, 102.41853714749703, 103.12056757724878, 103.30220118154881])

T_vaso_all = np.concatenate([t_vaso, t_vaso, t_vaso])
# Mapeamos Escenarios a valores numéricos continuos: A=1.0, B=2.0, C=3.0
Esc_all = np.concatenate([np.full_like(t_vaso, 1.0), np.full_like(t_vaso, 2.0), np.full_like(t_vaso, 3.0)])
Temp_vaso_all = np.concatenate([temp_A, temp_B, temp_C])

X_vaso = np.column_stack((T_vaso_all, Esc_all))
y_vaso = Temp_vaso_all

# Entrenar pipeline polinomial para temperatura del vaso
model_vaso = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model_vaso.fit(X_vaso, y_vaso)


# ==========================================
# 3. GUARDADO CONSOLIDADO
# ==========================================
# Guardamos ambos modelos en un diccionario dentro del mismo archivo .pkl
paquete_modelos = {
    'model_dano': model_dano,
    'model_vaso': model_vaso
}

joblib.dump(paquete_modelos, 'best_model.pkl')
print("¡Éxito! Ambos modelos subrogados han sido consolidados en 'best_model.pkl'.")
