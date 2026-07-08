# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# =====================================================================
# 1. ENTRENAMIENTO DEL MODELO DE DAÑO TISULAR (TUMOR)
# =====================================================================
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
# 2. ENTRENAMIENTO DEL MODELO DEL VASO CON DIÁMETROS REALES (5mm, 3mm, 1mm)
# =====================================================================
t_vaso = np.array([0, 2.5, 5, 7.5, 10])
temp_5mm = np.array([37.01357406218045, 91.20003415821034, 93.83253548911045, 94.44218612048912, 94.62108754938538])
temp_3mm = np.array([37.01343745229815, 90.39493541249876, 92.68835103219796, 93.24920048749480, 93.42238392224195])
temp_1mm = np.array([37.01332988963907, 89.28820214853024, 91.82831014749000, 92.35752500000000, 92.51853700000000])

T_vaso_all = np.concatenate([t_vaso, t_vaso, t_vaso])
# Mapeamos usando las dimensiones geométricas reales de COMSOL
Diam_all = np.concatenate([np.full_like(t_vaso, 5.0), np.full_like(t_vaso, 3.0), np.full_like(t_vaso, 1.0)])
Temp_vaso_all = np.concatenate([temp_5mm, temp_3mm, temp_1mm])

X_vaso = np.column_stack((T_vaso_all, Diam_all))
y_vaso = Temp_vaso_all

# Usamos grado 2 para mantener estabilidad matemática perfecta y evitar ondulaciones raras
model_vaso = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model_vaso.fit(X_vaso, y_vaso)


# =====================================================================
# 3. GUARDADO DEL PAQUETE
# =====================================================================
paquete_modelos = {
    'model_dano': model_dano,
    'model_vaso': model_vaso
}

joblib.dump(paquete_modelos, 'best_model.pkl')
print("¡Éxito! El modelo geométrico calibrado ha sido guardado en 'best_model.pkl'.")
