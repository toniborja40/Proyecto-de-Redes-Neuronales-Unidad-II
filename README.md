# Proyecto de Redes Neuronales - Unidad II

Este proyecto presenta la implementación de una **red neuronal artificial** para la **predicción de fallas en maquinaria industrial**, desarrollada en Python con TensorFlow/Keras.

El propósito del modelo es apoyar la **toma de decisiones automatizada** en entornos industriales, permitiendo clasificar si una máquina se encuentra en **funcionamiento normal** o en **riesgo de falla** a partir de variables operativas simuladas.

---

## Caso de estudio

El caso abordado corresponde a la **predicción de fallas en maquinaria industrial**, utilizando un conjunto de datos académicos que simula condiciones de operación de equipos monitoreados mediante sensores.

La red neuronal fue entrenada para reconocer patrones asociados a dos posibles salidas:

- **0:** funcionamiento normal  
- **1:** riesgo de falla  

---

## Arquitectura del modelo

El modelo implementado corresponde a un **Perceptrón Multicapa (MLP)** con la siguiente estructura:

- **Capa de entrada:** 8 variables
- **Primera capa oculta:** 10 neuronas con activación ReLU
- **Segunda capa oculta:** 6 neuronas con activación ReLU
- **Capa de salida:** 1 neurona con activación sigmoide

Esta arquitectura permite resolver un problema de **clasificación binaria**.

---

## Datos de entrada

El prototipo utiliza las siguientes variables:

- temperatura
- vibración
- presión
- consumo de energía
- horas de funcionamiento
- velocidad de rotación
- nivel de ruido
- historial de mantenimiento

La variable de salida es:

- **falla**, donde:
  - `0` = funcionamiento normal
  - `1` = riesgo de falla

---

## Base de datos utilizada

Se trabajó con un conjunto de **24 registros simulados con fines académicos**:

- **18 registros** para entrenamiento
- **6 registros** para prueba

Antes del entrenamiento, los datos son somrtifos a un proceso de **normalización Z-Score** mediante `StandardScaler`.

---

## Librerías utilizadas

Este proyecto utiliza las siguientes librerías:

- `numpy`
- `pandas`
- `tensorflow`
- `scikit-learn`

---

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.9 o superior
- pip

---
