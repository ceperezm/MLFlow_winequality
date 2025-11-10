"""
Script de entrenamiento del modelo de predicción de calidad de vinos
Integra MLflow tracking, logging y model registry
"""

import sys
import os
import argparse
import warnings
import json
import joblib
from scipy.io import arff

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Importar el explainer GenAI
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genai_explainer import WineExplainer


def load_wine_data(data_path='../data/winequality-red.csv.arff'):
    """Carga el dataset de vinos desde archivo ARFF local"""
    data, meta = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    return df   

def prepare_data(data, test_size=0.2, random_state=42):
    """Prepara los datos para entrenamiento"""
    
    # Eliminar duplicados
    data = data.drop_duplicates()
    
    # Separamos los datos en X y y
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Escalar los valores numéricos con StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_model(model_type, X_train, y_train, **params):
    """Entrena el modelo según el tipo especificado"""
    if model_type == "random_forest":
        model = RandomForestRegressor(**params)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evalúa el modelo y retorna métricas"""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_rmse": mean_squared_error(y_train, y_pred_train, squared=False),
        "test_rmse": mean_squared_error(y_test, y_pred_test, squared=False),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
    }
    
    
    return metrics

def get_feature_importance(model, feature_names, top_n=10):
    """Obtiene la importancia de features (solo para RF)"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        return importance_df
    return None

def get_artifact_base_path():
    """Obtiene la ruta base de artifacts para copias locales"""
    try:
        tracking_uri = mlflow.get_tracking_uri()
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        
        if not run_id:
            return None
            
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        run_info = client.get_run(run_id)
        artifact_uri = run_info.info.artifact_uri
        experiment_id = run_info.info.experiment_id
        
        # Si el artifact URI tiene el esquema problemático, construir ruta local
        if artifact_uri.startswith("mlflow-artifacts://"):
            if tracking_uri.startswith("file:///"):
                base_path = tracking_uri[7:]  # Remover "file:///"
            elif tracking_uri.startswith("file://"):
                base_path = tracking_uri[7:]  # Remover "file://"
            else:
                # Si no es file://, intentar usar el directorio actual de mlruns
                base_path = os.path.join(os.getcwd(), "mlruns")
            
            artifact_base = os.path.join(base_path, str(experiment_id), run_id, "artifacts")
            return artifact_base
        
        return None
    except Exception:
        return None

def safe_log_artifact(local_path, artifact_path=None):
    """Loggea un artifact de manera segura, manejando errores de URI"""
    try:
        mlflow.log_artifact(local_path, artifact_path)
        return True
    except Exception as e:
        error_str = str(e)
        # Si el error es por esquema de URI incompatible, copiar localmente
        if "mlflow-artifact scheme" in error_str or "invalid for use with the proxy" in error_str:
            artifact_base = get_artifact_base_path()
            if artifact_base:
                try:
                    if artifact_path:
                        artifact_full_path = os.path.join(artifact_base, artifact_path)
                        os.makedirs(os.path.dirname(artifact_full_path), exist_ok=True)
                    else:
                        artifact_full_path = os.path.join(artifact_base, os.path.basename(local_path))
                        os.makedirs(artifact_base, exist_ok=True)
                    
                    import shutil
                    shutil.copy2(local_path, artifact_full_path)
                    print(f"  ✓ Artifact {os.path.basename(local_path)} guardado localmente")
                    return True
                except Exception as e2:
                    print(f"   No se pudo guardar artifact {local_path} localmente: {e2}")
                    return False
            else:
                print(f"   No se pudo determinar ruta de artifacts para {local_path}")
                return False
        else:
            print(f"   Error loggeando artifact {local_path}: {e}")
            return False

def safe_log_model(model, artifact_path, signature=None, registered_model_name=None):
    """Loggea un modelo de manera segura, manejando errores de URI"""
    try:
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            signature=signature,
            registered_model_name=registered_model_name
        )
        return True
    except Exception as e:
        error_str = str(e)
        # Si el error es por esquema de URI incompatible, guardar localmente
        if "mlflow-artifact scheme" in error_str or "invalid for use with the proxy" in error_str:
            artifact_base = get_artifact_base_path()
            if artifact_base:
                try:
                    # Guardar el modelo localmente primero
                    temp_model_path = "temp_model_dir"
                    mlflow.sklearn.save_model(model, temp_model_path)
                    
                    # Copiar el directorio completo al artifact base
                    artifact_full_path = os.path.join(artifact_base, artifact_path)
                    import shutil
                    if os.path.exists(artifact_full_path):
                        shutil.rmtree(artifact_full_path)
                    shutil.copytree(temp_model_path, artifact_full_path)
                    
                    # Limpiar el directorio temporal
                    shutil.rmtree(temp_model_path)
                    
                    # Registrar el modelo en el registry si se especificó
                    # Nota: El registro puede fallar si hay problemas con el artifact URI,
                    # pero el modelo ya está guardado localmente
                    if registered_model_name:
                        try:
                            from mlflow.tracking import MlflowClient
                            client = MlflowClient()
                            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
                            if run_id:
                                # Intentar registrar usando el URI local
                                # Usar el path absoluto del artifact como source
                                artifact_abs_path = os.path.abspath(artifact_full_path)
                                # Intentar con file:// URI
                                model_uri = f"file:///{artifact_abs_path.replace(os.sep, '/')}"
                                try:
                                    client.create_model_version(
                                        name=registered_model_name,
                                        source=model_uri,
                                        run_id=run_id
                                    )
                                    print(f"  ✓ Modelo registrado: {registered_model_name}")
                                except Exception:
                                    # Si falla, al menos intentar sin el URI absoluto
                                    try:
                                        model_uri = f"runs:/{run_id}/{artifact_path}"
                                        client.create_model_version(
                                            name=registered_model_name,
                                            source=model_uri,
                                            run_id=run_id
                                        )
                                        print(f"  ✓ Modelo registrado: {registered_model_name}")
                                    except Exception as e3:
                                        print(f"   Modelo guardado pero no registrado (puedes registrarlo manualmente): {e3}")
                        except Exception as e3:
                            print(f"   Modelo guardado pero no registrado (puedes registrarlo manualmente): {e3}")
                    
                    print(f"  ✓ Modelo guardado localmente en {artifact_path}")
                    return True
                except Exception as e2:
                    print(f"   No se pudo guardar modelo localmente: {e2}")
                    return False
            else:
                print(f"   No se pudo determinar ruta de artifacts para el modelo")
                return False
        else:
            raise  # Re-raise si es otro tipo de error

def main(model_type, n_estimators=100, max_depth=10, alpha=1.0, random_state=42, run_name: str | None = None):
    """Función principal de entrenamiento - Versión ultra-simplificada"""
    
    # Configurar tracking URI PRIMERO, antes de cualquier operación de MLflow
    try:
        current_uri = mlflow.get_tracking_uri()
        # Si no está configurado o es un URI problemático, usar sistema de archivos local
        if not current_uri or current_uri.startswith("sqlite://") or "mlflow-artifacts://" in str(current_uri):
            # Usar el directorio mlruns local en el directorio del proyecto
            mlruns_path = os.path.join(os.path.dirname(__file__), "mlruns")
            mlruns_path = os.path.abspath(mlruns_path)
            # Asegurar que el directorio existe
            os.makedirs(mlruns_path, exist_ok=True)
            # Configurar tracking URI como file://
            tracking_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"
            mlflow.set_tracking_uri(tracking_uri)
            print(f" Tracking URI configurado: {tracking_uri}")
    except Exception as e:
        print(f" No se pudo configurar tracking URI: {e}")
    
    # Configurar nombre del experimento ANTES de verificar si hay run activo
    experiment_name = "wine-quality-prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f" Experiment '{experiment_name}' creado")
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f" No se pudo configurar experimento: {e}")
        experiment_id = None
    
    # Crear nombre descriptivo para el run (si no se proporcionó)
    if not run_name:
        if model_type == "random_forest":
            run_name = f"{model_type}_n{n_estimators}_d{max_depth}"
        else:
            run_name = f"{model_type}_alpha{alpha}"
    
    # Verificar si ya hay un run activo (cuando se ejecuta con mlflow run)
    run = mlflow.active_run()
    
    if run is None:
        # No hay run activo - crear uno nuevo con nombre
        mlflow.start_run(run_name=run_name)
        print(f" Run iniciado: {run_name}")
    else:
        # Ya hay un run activo (cuando se ejecuta con mlflow run)
        # Verificar si el run está en el experimento correcto
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            run_info = client.get_run(run.info.run_id)
            current_experiment_id = run_info.info.experiment_id
            
            # Si el run está en un experimento diferente, intentar moverlo
            if experiment_id and str(current_experiment_id) != str(experiment_id):
                # Obtener el experimento actual
                current_experiment = client.get_experiment(current_experiment_id)
                if current_experiment.name == "Default":
                    print(f" Run está en experimento 'Default'. Usando experimento '{experiment_name}' para nuevos logs.")
                    # Aunque no podemos mover el run, podemos asegurarnos de que los logs futuros vayan al experimento correcto
                    # El run seguirá en Default, pero crearemos un nuevo run en el experimento correcto
                    # Cerrar el run actual y crear uno nuevo en el experimento correcto
                    mlflow.end_run()
                    mlflow.set_experiment(experiment_name)
                    mlflow.start_run(run_name=run_name)
                    print(f" Nuevo run creado en experimento '{experiment_name}': {run_name}")
                else:
                    # El run está en otro experimento, solo renombrarlo
                    mlflow.set_tag("mlflow.runName", run_name)
                    print(f" Run activo renombrado: {run_name} (experimento: {current_experiment.name})")
            else:
                # El run está en el experimento correcto, solo renombrarlo
                mlflow.set_tag("mlflow.runName", run_name)
                print(f" Run activo renombrado: {run_name}")
        except Exception as e:
            # Si hay algún error, solo intentar renombrar el run
            try:
                mlflow.set_tag("mlflow.runName", run_name)
                print(f" Run activo renombrado: {run_name}")
            except Exception as e2:
                print(f" No se pudo renombrar el run: {e2}")
    
    try:
        # Log de parámetros generales
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)
        
        # Cargar y preparar datos
        print("Cargando datos...")
        data = load_wine_data()
        mlflow.log_param("dataset_size", len(data))
        
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
            data, random_state=random_state
        )
        
        # Configurar parámetros del modelo, solo random forest
        if model_type == "random_forest":
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_state': random_state,
                'n_jobs': -1
            }        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Log de hiperparámetros
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Entrenar modelo
        print(f"   Entrenando modelo {model_type}...")
        model = train_model(model_type, X_train, y_train, **model_params)
        
        # Evaluar modelo
        print(" Evaluando modelo...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Log de métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Feature importance (solo para RF)
        if model_type == "random_forest":
            importance_df = get_feature_importance(model, feature_names)
            if importance_df is not None:
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                safe_log_artifact(importance_path)
                print(f"\n Top features más importantes:")
                print(importance_df.to_string(index=False))
        
        # Guardar información del dataset
        data_info = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(feature_names),
            "features": feature_names,
            "target_mean": float(y_train.mean()),
            "target_std": float(y_train.std())
        }
        
        with open("data_info.json", "w") as f:
            json.dump(data_info, f, indent=2)
        safe_log_artifact("data_info.json")
        
        # Log del modelo con signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        safe_log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=f"wine-quality-{model_type}"
        )
        
        # Guardar scaler como artifact
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        safe_log_artifact(scaler_path)
        
        # Generar explicación GenAI para una muestra de ejemplo
        print("\n Generando explicación con GenAI...")
        try:
            sample_idx = 0
            sample_features = X_test[sample_idx:sample_idx+1]
            sample_prediction = model.predict(sample_features)[0]
            
            sample_dict = {feature_names[i]: float(sample_features[0][i]) 
                          for i in range(len(feature_names))}
            
            explainer = WineExplainer(provider="ollama", model="llama2")
            explanation = explainer.explain(
                features_dict=sample_dict,
                prediction=float(sample_prediction),
                probability=1.0
            )
            
            if "Error" in explanation or len(explanation) < 20:
                explanation = f"""Predicción de calidad: {sample_prediction:.2f}/10.
                Características principales: alcohol {sample_dict.get('alcohol', 0):.2f}%,
                acidez volátil {sample_dict.get('volatile acidity', 0):.2f},
                sulfatos {sample_dict.get('sulphates', 0):.2f}."""
            
            explanation_path = "model_explanation.txt"
            with open(explanation_path, "w", encoding="utf-8") as f:
                f.write(f"Explicación del modelo para muestra de ejemplo:\n")
                f.write(f"Predicción: {sample_prediction:.2f}/10\n\n")
                f.write(f"Características:\n")
                for key, value in sample_dict.items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write(f"\nExplicación GenAI:\n{explanation}\n")
            
            safe_log_artifact(explanation_path)
            try:
                mlflow.log_text(explanation, "genai_explanation.txt")
            except Exception as e:
                print(f"   No se pudo loggear texto: {e}")
            print(f"  Explicación generada y registrada")
            
        except Exception as e:
            print(f"   Error generando explicación: {e}")
            explanation = f"Modelo {model_type} entrenado. Predicción de calidad basada en características químicas del vino."
            try:
                mlflow.log_text(explanation, "genai_explanation.txt")
            except Exception as e2:
                print(f"   No se pudo loggear texto de fallback: {e2}")
        
        print(f"\n Modelo entrenado y registrado exitosamente!")
        print(f" Métricas registradas: {list(metrics.keys())}")
        
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        
        # Cerrar el run si lo iniciamos nosotros
        if run is None:
            mlflow.end_run()
        
        return run_id
            
    except Exception as e:
        print(f" Error durante el entrenamiento: {e}")
        # Asegurar que el run se cierra incluso si hay error
        if run is None and mlflow.active_run():
            mlflow.end_run()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de vinos")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["random_forest", "ridge"],
                       help="Tipo de modelo a entrenar")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Número de árboles (solo RF)")
    parser.add_argument("--max-depth", type=int, default=10,
                       help="Profundidad máxima (solo RF)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Parámetro de regularización (solo Ridge)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Semilla aleatoria")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Nombre del run en MLflow (opcional)")
    
    args = parser.parse_args()
    
    main(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        alpha=args.alpha,
        random_state=args.random_state,
        run_name=args.run_name
    )