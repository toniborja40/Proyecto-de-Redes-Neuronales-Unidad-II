import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Oculta mensajes informativos de TensorFlow

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Creación del conjunto de datos 

data = {
    "temperatura": [65, 70, 75, 80, 68, 90, 95, 85, 60, 72, 88, 93, 77, 66, 82, 91, 69, 74, 87, 96, 64, 78, 89, 92],
    "vibracion": [0.20, 0.25, 0.30, 0.45, 0.22, 0.60, 0.75, 0.50, 0.18, 0.28, 0.55, 0.70, 0.35, 0.21, 0.48, 0.68, 0.24, 0.32, 0.58, 0.80, 0.19, 0.40, 0.62, 0.73],
    "presion": [30, 32, 35, 38, 31, 45, 48, 40, 29, 34, 43, 47, 36, 30, 39, 46, 33, 35, 44, 49, 28, 37, 42, 46],
    "consumo_energia": [200, 210, 220, 250, 205, 300, 320, 270, 190, 215, 280, 310, 230, 198, 255, 305, 208, 225, 285, 330, 195, 240, 290, 315],
    "horas_funcionamiento": [1000, 1200, 1500, 1800, 1100, 2500, 2700, 2000, 900, 1300, 2200, 2600, 1600, 1050, 1900, 2550, 1150, 1450, 2250, 2800, 950, 1700, 2350, 2650],
    "velocidad_rotacion": [1500, 1520, 1480, 1450, 1510, 1380, 1360, 1420, 1530, 1490, 1400, 1370, 1470, 1525, 1440, 1385, 1505, 1485, 1395, 1350, 1535, 1460, 1410, 1375],
    "nivel_ruido": [60, 62, 65, 70, 61, 80, 85, 75, 58, 64, 78, 83, 68, 60, 72, 82, 63, 66, 79, 86, 59, 69, 77, 84],
    "historial_mantenimiento": [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    "falla": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop("falla", axis=1)
y = df["falla"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Etapa de normalizacion z-score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construccion del Modelo

model = Sequential([
    Input(shape=(8,)),
    Dense(10, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Etapa de compilación del modelo

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del Modelo

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=4,
    validation_split=0.2,
    verbose=0
)

# Resumen del entrenamiento 

print("\nResumen final del entrenamiento:")
print(f"Accuracy entrenamiento: {history.history['accuracy'][-1]:.4f}")
print(f"Loss entrenamiento: {history.history['loss'][-1]:.4f}")
print(f"Accuracy validación: {history.history['val_accuracy'][-1]:.4f}")
print(f"Loss validación: {history.history['val_loss'][-1]:.4f}")

# Evaluación del modelo

y_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_prob > 0.5).astype(int).flatten()

print("\nProbabilidades predichas:")
print(np.round(y_prob.flatten(), 4))

print("\nPredicciones finales:")
print(y_pred)

print(f"\nAccuracy final: {accuracy_score(y_test, y_pred):.4f}")

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Tabla final comparativa de los resultados

resultados = X_test.copy()
resultados["Real"] = y_test.values
resultados["Probabilidad_predicha"] = np.round(y_prob.flatten(), 4)
resultados["Prediccion"] = y_pred

print("\nTabla comparativa final:")
print(resultados)