"""
Interfaz Gradio para predicción de calidad de vinos
Integra MLflow Model Registry, predicciones y explicaciones GenAI
"""

import gradio as gr
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
import joblib

# Cargar .env lo antes posible
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    env_path = find_dotenv()
    if env_path:
        if load_dotenv(env_path):
            print(f"[App] .env cargado en app: {env_path}")
except Exception:
    pass

# Agregar el directorio project al path para importar genai_explainer
project_dir = os.path.join(os.path.dirname(__file__), "..", "project")
project_dir = os.path.abspath(project_dir)
sys.path.insert(0, project_dir)
from genai_explainer import WineExplainer  # type: ignore

# Configurar MLflow
mlruns_path = os.path.join(os.path.dirname(__file__), "..", "project", "mlruns")
mlruns_path = os.path.abspath(mlruns_path)
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

# Nombres de modelos registrados
MODEL_NAMES = {
    "random_forest": "wine-quality-random_forest"
}

def load_model_from_registry(model_name, stage="Production"):
    """Carga un modelo desde MLflow Model Registry"""
    try:
        if stage != "None":
            try:
                model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
                print(f"Modelo cargado desde {stage}")
                return model
            except Exception as e1:
                print(f"Error cargando desde {stage}: {e1}")
        
        client = mlflow.tracking.MlflowClient()
        try:
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if all_versions:
                latest = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
                model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}")
                print(f"Modelo cargado desde versión {latest.version}")
                return model
        except Exception as e2:
            print(f"Error buscando versiones: {e2}")
        
        return None
    except Exception as e:
        print(f"Error general cargando modelo {model_name}: {e}")
        return None

def load_scaler_from_run(model_name, stage="Production"):
    """Carga el scaler desde los artifacts de un run"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        if stage != "None":
            try:
                latest_version = client.get_latest_versions(model_name, stages=[stage])
                if latest_version:
                    version = latest_version[0]
                    run_id = version.run_id
                    artifact_path = client.download_artifacts(run_id, "scaler.pkl")
                    scaler = joblib.load(artifact_path)
                    return scaler
            except Exception as e1:
                print(f"Error cargando scaler desde {stage}: {e1}")
        
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if all_versions:
            latest = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
            run_id = latest.run_id
            artifact_path = client.download_artifacts(run_id, "scaler.pkl")
            scaler = joblib.load(artifact_path)
            return scaler
    except Exception as e:
        print(f"Error cargando scaler: {e}")
    return None

def get_model_metrics(model_name, stage="Production"):
    """Obtiene las métricas del modelo desde MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        if stage != "None":
            try:
                latest_version = client.get_latest_versions(model_name, stages=[stage])
                if latest_version:
                    version = latest_version[0]
                    run = client.get_run(version.run_id)
                    metrics = {
                        "test_r2": run.data.metrics.get("test_r2", "N/A"),
                        "test_rmse": run.data.metrics.get("test_rmse", "N/A"),
                        "test_mae": run.data.metrics.get("test_mae", "N/A"),
                        "cv_rmse_mean": run.data.metrics.get("cv_rmse_mean", "N/A")
                    }
                    return metrics, version.version
            except Exception as e1:
                print(f"Error obteniendo métricas desde {stage}: {e1}")
        
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if all_versions:
            latest = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
            run = client.get_run(latest.run_id)
            metrics = {
                "test_r2": run.data.metrics.get("test_r2", "N/A"),
                "test_rmse": run.data.metrics.get("test_rmse", "N/A"),
                "test_mae": run.data.metrics.get("test_mae", "N/A"),
                "cv_rmse_mean": run.data.metrics.get("cv_rmse_mean", "N/A")
            }
            return metrics, latest.version
    except Exception as e:
        print(f"Error obteniendo métricas: {e}")
    return None, None

def log_prediction_explanation(features_dict, prediction, explanation, model_name, stage, genai_model):
    """Registra la explicación GenAI en MLflow como artifact"""
    try:
        experiment_name = "wine-quality-predictions_genai"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception:
            pass
        
        active_run = mlflow.active_run()
        run_created = False
        
        if active_run is None:
            run_name = f"prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name, nested=False)
            run_created = True
        
        try:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_stage", stage)
            mlflow.log_param("genai_model", genai_model)
            mlflow.log_metric("predicted_quality", float(prediction))
            
            for key, value in features_dict.items():
                mlflow.log_param(f"feature_{key.replace(' ', '_')}", float(value))
            
            try:
                mlflow.log_text(explanation, "genai_explanation.txt")
            except Exception as e:
                print(f"  No se pudo registrar explicación con log_text: {e}")
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                        f.write(f"Predicción: {prediction:.2f}/10\n\n")
                        f.write(f"Características:\n")
                        for key, value in features_dict.items():
                            f.write(f"  {key}: {value:.4f}\n")
                        f.write(f"\nExplicación GenAI:\n{explanation}\n")
                        temp_path = f.name
                    
                    mlflow.log_artifact(temp_path, "genai_explanations")
                    os.unlink(temp_path)
                except Exception as e2:
                    print(f"  No se pudo registrar explicación como artifact: {e2}")
            
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            if run_id:
                print(f"  Explicación registrada en MLflow run: {run_id}")
        finally:
            if run_created and mlflow.active_run():
                mlflow.end_run()
            
    except Exception as e:
        print(f"  Error registrando explicación en MLflow: {e}")

def format_prediction_result(prediction, metrics, version, stage, sample_count=1):
    """Formatea el resultado de la predicción con una estructura visual consistente"""
    prediction_rounded = round(prediction, 2)
    
    # Formatear métricas
    metrics_text = ""
    if metrics:
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else x
        
        metrics_text = f"""

** MÉTRICAS DEL MODELO (v{version}, {stage})**

- **R² Score:** {fmt(metrics.get('test_r2', 'N/A'))}
- **RMSE:** {fmt(metrics.get('test_rmse', 'N/A'))}
- **MAE:** {fmt(metrics.get('test_mae', 'N/A'))}
- **CV RMSE:** {fmt(metrics.get('cv_rmse_mean', 'N/A'))}

"""
    
    # Encabezado principal de predicción
    sample_info = f"\n**Muestras procesadas:** {sample_count}" if sample_count > 1 else ""
    
    result_text = f"""
###  PREDICCIÓN DE CALIDAD

   # {prediction_rounded}/10

   ** {sample_info}**
{metrics_text}
"""
    
    return result_text

def predict_wine_quality(
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
    pH, sulphates, alcohol, model_type="random_forest", stage="Production",
    genai_model="gemini-2.0-flash"
):
    """Realiza predicción de calidad de vino y registra explicación GenAI en MLflow"""
    
    features = {
        'fixed acidity': [float(fixed_acidity)],
        'volatile acidity': [float(volatile_acidity)],
        'citric acid': [float(citric_acid)],
        'residual sugar': [float(residual_sugar)],
        'chlorides': [float(chlorides)],
        'free sulfur dioxide': [float(free_sulfur_dioxide)],
        'total sulfur dioxide': [float(total_sulfur_dioxide)],
        'density': [float(density)],
        'pH': [float(pH)],
        'sulphates': [float(sulphates)],
        'alcohol': [float(alcohol)]
    }
    
    df = pd.DataFrame(features)
    
    model_name = MODEL_NAMES.get(model_type, MODEL_NAMES["random_forest"])
    model = load_model_from_registry(model_name, stage)
    
    if model is None:
        error_msg = " **Error:** No se pudo cargar el modelo. Asegúrate de haber entrenado y registrado el modelo primero."
        return error_msg, error_msg
    
    scaler = load_scaler_from_run(model_name, stage)
    
    if scaler is not None:
        X_scaled = scaler.transform(df)
    else:
        X_scaled = df.values
    
    prediction = model.predict(X_scaled)[0]
    
    explanation = ""
    try:
        sample_features = {k: float(v[0]) for k, v in features.items()}
        explainer = WineExplainer(model=genai_model)
        explanation = explainer.explain(sample_features, float(prediction))
        
        if not explanation or len(explanation) < 8:
            explanation = f"Predicción {round(prediction, 2)}/10. Alcohol, acidez volátil, sulfatos y pH influyen en la calidad."
    except Exception as e:
        explanation = f" **Error generando explicación:** {str(e)}"
    
    sample_features = {k: float(v[0]) for k, v in features.items()}
    log_prediction_explanation(sample_features, prediction, explanation, model_name, stage, genai_model)
    
    metrics, version = get_model_metrics(model_name, stage)
    
    result_text = format_prediction_result(prediction, metrics, version, stage, 1)
    
    explanation_text = f"""
##  EXPLICACIÓN GENAI

{explanation}
"""
    
    return result_text, explanation_text

def predict_from_csv(csv_file, model_type="random_forest", stage="Production", genai_model="gemini-2.0-flash"):
    """Predice desde un archivo CSV y genera explicación GenAI promedio."""
    try:
        df = pd.read_csv(csv_file.name)
        
        required_cols = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f" **Error:** Faltan columnas en el CSV: {', '.join(missing_cols)}"
        
        model_name = MODEL_NAMES.get(model_type, MODEL_NAMES["random_forest"])
        model = load_model_from_registry(model_name, stage)
        scaler = load_scaler_from_run(model_name, stage)
        
        if model is None:
            return " **Error:** No se pudo cargar el modelo."
        
        X = df[required_cols]
        X_scaled = scaler.transform(X) if scaler is not None else X.values
        
        predictions = model.predict(X_scaled)
        df['predicted_quality'] = predictions.round(2)
        
        explanation = ""
        try:
            sample = df[required_cols].mean().to_dict()
            sample_features = {k: float(v) for k, v in sample.items()}
            explainer = WineExplainer(model=genai_model)
            explanation = explainer.explain(sample_features, float(predictions.mean()))
            
            log_prediction_explanation(sample_features, float(predictions.mean()), explanation,
                                       model_name, stage, genai_model)
        except Exception as e:
            explanation = f" **No se pudo generar explicación GenAI:** {e}"
        
        metrics, version = get_model_metrics(model_name, stage)
        
        result_text = format_prediction_result(predictions.mean(), metrics, version, stage, len(df))
        
        stats_text = f"""
##  ESTADÍSTICAS DEL LOTE

- **Mínimo:** {predictions.min():.2f}/10
- **Máximo:** {predictions.max():.2f}/10  
- **Desviación estándar:** {predictions.std():.2f}
- **Muestras con calidad ≥7:** {(predictions >= 7).sum()}
"""
        
        full_result = result_text + stats_text
        
        if explanation:
            full_result += f"""

##  EXPLICACIÓN GENAI

{explanation}
"""
        
        return full_result
        
    except Exception as e:
        return f" **Error procesando CSV:** {str(e)}"

def compare_models(model_type="random_forest"):
    """Compara versiones Staging y Production del modelo"""
    model_name = MODEL_NAMES.get(model_type, MODEL_NAMES["random_forest"])
    
    staging_metrics, staging_version = get_model_metrics(model_name, "Staging")
    prod_metrics, prod_version = get_model_metrics(model_name, "Production")
    
    comparison = """
##  COMPARACIÓN DE MODELOS

"""
    
    def create_model_section(metrics, version, stage):
        if not metrics:
            return f"""
### {stage}

**No disponible**
"""
        
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else 'N/A'
        
        return f"""
### {stage} (v{version})

- **R² Score:** {fmt(metrics.get('test_r2'))}
- **RMSE:** {fmt(metrics.get('test_rmse'))}
- **MAE:** {fmt(metrics.get('test_mae'))}
- **CV RMSE:** {fmt(metrics.get('cv_rmse_mean'))}

"""
    
    with gr.Row():
        comparison += create_model_section(staging_metrics, staging_version, " Staging")
        comparison += create_model_section(prod_metrics, prod_version, " Production")
    
    if not staging_metrics and not prod_metrics:
        comparison = """
##  NO HAY MODELOS REGISTRADOS

Asegúrate de haber entrenado y registrado modelos en Staging o Production.
"""
    
    return comparison

# Crear interfaz Gradio
with gr.Blocks(title="Predicción de Calidad de Vinos", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    #  Predicción de Calidad de Vinos con MLflow y GenAI
    
    Esta interfaz permite predecir la calidad de vinos usando modelos entrenados con MLflow,
    y generar explicaciones automáticas mediante GenAI (Gemini u OpenRouter).
    """)
    
    with gr.Tabs():
        with gr.Tab(" Predicción Individual"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Características del Vino")
                    fixed_acidity = gr.Slider(4.0, 16.0, value=7.4, label="Acidez Fija")
                    volatile_acidity = gr.Slider(0.1, 1.6, value=0.7, label="Volatile Acidity")
                    citric_acid = gr.Slider(0.0, 1.0, value=0.0, label="Ácido Cítrico")
                    residual_sugar = gr.Slider(0.9, 15.0, value=1.9, label="Azúcar Residual")
                    chlorides = gr.Slider(0.01, 0.6, value=0.076, label="Cloruros")
                    free_sulfur_dioxide = gr.Slider(1.0, 72.0, value=11.0, label="Free Sulfur Dioxide")
                    total_sulfur_dioxide = gr.Slider(6.0, 289.0, value=34.0, label="Total Sulfur Dioxide")
                    density = gr.Slider(0.99, 1.01, value=0.9978, label="Density")
                    pH = gr.Slider(2.7, 4.0, value=3.51, label="pH")
                    sulphates = gr.Slider(0.3, 2.0, value=0.56, label="Sulphates")
                    alcohol = gr.Slider(8.0, 15.0, value=9.4, label="Alcohol (%)")
                
                with gr.Column():
                    gr.Markdown("### Configuración del Modelo")
                    model_type = gr.Radio(
                        choices=["random_forest"],
                        value="random_forest",
                        label="Tipo de Modelo"
                    )
                    stage = gr.Radio(
                        choices=["Production", "Staging", "None"],
                        value="Production",
                        label="Versión del Modelo"
                    )
                    
                    gr.Markdown("### Configuración GenAI")
                    genai_model = gr.Radio(
                        choices=["gemini-2.0-flash", "z-ai/glm-4.6"],
                        value="gemini-2.0-flash",
                        label="Modelo GenAI"
                    )
                    
                    predict_btn = gr.Button(" Predecir Calidad", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    prediction_output = gr.Markdown(label="Predicción")
                with gr.Column():
                    explanation_output = gr.Markdown(label="Explicación")
        
        with gr.Tab(" Predicción desde CSV"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Carga un archivo CSV con las características de los vinos")
                    csv_input = gr.File(label="Archivo CSV", file_types=[".csv"])
                    
                    gr.Markdown("### Configuración")
                    csv_model_type = gr.Radio(
                        choices=["random_forest"],
                        value="random_forest",
                        label="Tipo de Modelo"
                    )
                    csv_stage = gr.Radio(
                        choices=["Production", "Staging"],
                        value="Production",
                        label="Versión del Modelo"
                    )
                    csv_genai_model = gr.Radio(
                        choices=["gemini-2.0-flash", "z-ai/glm-4.6"],
                        value="gemini-2.0-flash",
                        label="Modelo GenAI"
                    )
                    
                    csv_predict_btn = gr.Button(" Predecir desde CSV", variant="primary")
                
                with gr.Column():
                    csv_output = gr.Markdown(label="Resultados")
        
        with gr.Tab(" Comparar Modelos"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Compara versiones Staging vs Production")
                    compare_model_type = gr.Radio(
                        choices=["random_forest"],
                        value="random_forest",
                        label="Tipo de Modelo"
                    )
                    compare_btn = gr.Button("🔄 Comparar", variant="primary")
                
                with gr.Column():
                    comparison_output = gr.Markdown(label="Comparación")
    
    # Conectar eventos
    predict_btn.click(
        fn=predict_wine_quality,
        inputs=[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol, model_type, stage, genai_model
        ],
        outputs=[prediction_output, explanation_output]
    )
    
    csv_predict_btn.click(
        fn=predict_from_csv,
        inputs=[csv_input, csv_model_type, csv_stage, csv_genai_model],
        outputs=[csv_output]
    )
    
    compare_btn.click(
        fn=compare_models,
        inputs=[compare_model_type],
        outputs=[comparison_output]
    )
    
    gr.Markdown("""
    ---
    Asegúrate de tener:
    1. MLflow corriendo y modelos registrados
    2. GEMINI_API_KEY (para Gemini) o OPENROUTER_API_KEY (para OpenRouter) configuradas  
    3. Los modelos entrenados y registrados en el Model Registry
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="localhost", server_port=7860)