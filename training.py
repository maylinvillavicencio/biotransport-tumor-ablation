# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Clase personalizada para garantizar una interpolación exacta sin oscilaciones matemáticas
class COMSOLVesselInterpolator:
    def __init__(self):
        self.times = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        self.diameters = np.array([1.0, 3.0, 5.0]) # Ordenados de menor a mayor
        
        # Matriz de almacenamiento térmico (Filas: Tiempos, Columnas: Diámetros)
        self.temps = np.zeros((5, 3))
        
        # Asignación exacta de tus nuevos datos de COMSOL
        self.temps[:, 0] = [37.01332988963907, 89.28820214853024, 91.82831014749000, 92.35752500000000, 92.51853700000000] # Vaso 1mm (Datos C)
        self.temps[:, 1] = [37.01343745229815, 90.39493541249876, 92.68835103219796, 93.24920048749480, 93.42238392224195] # Vaso 3mm (Datos B)
        self.temps[:, 2] = [37.01357406218045, 91.20003415821034, 93.83253548911045, 94.44218612048912, 94.62108754938538] # Vaso 5mm (Datos A)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for i in range(len(X)):
            t_val = np.clip(X[i, 0], 0.0, 10.52)
            d_val = np.clip(X[i, 1], 1.0, 5.0)
            
            # Interpolación exacta en el tiempo para cada curva base
            v_1mm = np.interp(t_val, self.times, self.temps[:, 0])
            v_3mm = np.interp(t_val, self.times, self.temps[:, 1])
            v_5mm = np.interp(t_val, self.times, self.temps[:, 2])
            
            # Interpolación exacta en el espacio para el diámetro seleccionado
            v_final = np.interp(d_val, self.diameters, [v_1mm, v_3mm, v_5mm])
            predictions.append(v_final)
        return np.array(predictions)

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
# 2. INSTANCIACIÓN DEL NUEVO MODELO PARA EL VASO
# =====================================================================
model_vaso = COMSOLVesselInterpolator()

# =====================================================================
# 3. CONSOLIDACIÓN SEGURA EN EL ARCHIVO (.PKL)
# =====================================================================
paquete_modelos = {
    'model_dano': model_dano,
    'model_vaso': model_vaso
}

joblib.dump(paquete_modelos, 'best_model.pkl')
print("¡Éxito! El modelo subrogado con interpolación exacta ha sido consolidado en 'best_model.pkl'.")
