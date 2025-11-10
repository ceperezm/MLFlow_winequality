# Wine Quality Prediction - MLOps Project
Sistema completo de Machine Learning Operations (MLOps) para predecir la calidad de vinos utilizando modelos de ML, MLflow para tracking y registro de modelos, y GenAI para generar explicaciones automáticas.

# Descripción del Proyecto
Este proyecto implementa un pipeline completo de MLOps que incluye:

Entrenamiento de modelos de Machine Learning para predecir calidad de vinos (0-10)

Seguimiento de experimentos con MLflow

Registro de modelos en MLflow Model Registry

Interfaz web interactiva con Gradio para realizar predicciones

Explicaciones GenAI automáticas usando Gemini u OpenRouter

Gestión de versiones de modelos (Staging vs Production) desde mlflow ui

Modelos de Machine Learning
Random Forest Regressor para predicción de calidad

Preprocesamiento con StandardScaler

Evaluación con métricas: R², RMSE, MAE

Feature importance analysis
#  Interfaz de Usuario (Gradio)
Predicción individual: Sliders para características del vino

Predicción por lote: Carga de archivos CSV

Comparación de modelos: Staging vs Production

Explicaciones GenAI: Análisis automático en lenguaje natural

# Crear entorno conda
conda env create -f conda.yaml
conda activate wine-quality-env


# Para Gemini (Google AI)
GEMINI_API_KEY=tu_api_key_de_google

# Para OpenRouter (alternativa)
OPENROUTER_API_KEY=tu_api_key_de_openrouter

# Iniciar MLfloe
mlflow ui

# Ejecutar experimentos predefinidos
mlflow run . -e experiment --experiment-name wine-quality-prediction --run-name "rf_prediccion"

# Iniciar app en gradio
python app.py



Usar la Interfaz
📊 Pestaña "Predicción Individual"
Ajusta los sliders para las características del vino

Selecciona el modelo y versión (Staging/Production)

Elige el modelo GenAI para explicaciones

Haz clic en "Predecir Calidad"

📁 Pestaña "Predicción desde CSV"
Sube un archivo CSV con las columnas requeridas

Obtén predicciones por lote con estadísticas

Explicación GenAI basada en el promedio del lote

🔄 Pestaña "Comparar Modelos"
Compara métricas entre versiones Staging y Production

Visualiza diferencias en rendimiento

# Modelos de IA generativa
Google Gemini: gemini-2.0-flash

OpenRouter z-ai/glm-4.6










