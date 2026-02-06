# main.py
# PIPELINE PRINCIPAL PARA TABLA H7: ESTADOS DE MOMENTO ENTRELAZADOS

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Importar clases personalizadas
from src.classes.data_manager import DataManager
from src.classes.model_trainer import ModelTrainer

# Importar configuración
from config import (
    TRAIN_FILE, TEST_FILE, VAL_FILE,
    MODEL_OUTPUT_DIR, LOGS_DIR, RESULTS_DIR,
    TARGET_COLUMN, H7_EXPECTED_COLUMNS,
    SAVE_LOGS, LOG_FILE, VERBOSE,
    SAVE_EVALUATION_REPORT, EVALUATION_REPORT_FILE,
    SAVE_CONFUSION_MATRIX, CONFUSION_MATRIX_FILE,
    SAVE_FEATURE_IMPORTANCE, FEATURE_IMPORTANCE_FILE
)

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logging():
    """
    Configura el sistema de logging para guardar en archivo y consola.
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if SAVE_LOGS:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    return logging.getLogger(__name__)

# ============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ============================================================================

def run_pipeline():
    """
    Ejecuta el pipeline completo de entrenamiento y evaluación para Tabla H7.
    
    Pipeline:
    1. Configurar logging
    2. Cargar datos (train + test/val)
    3. Validar estructura de datos
    4. Dividir datos (si es necesario)
    5. Entrenar modelo
    6. Evaluar en validación
    7. Generar predicciones en test
    8. Guardar resultados
    """
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE PARA TABLA H7 (ESTADOS DE MOMENTO ENTRELAZADOS)")
    logger.info("=" * 80)
    
    try:
        # ====================================================================
        # FASE 0: PREPARACIÓN
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FASE 0: PREPARACIÓN")
        logger.info("=" * 80)
        
        # Crear directorios si no existen
        for dir_path in [MODEL_OUTPUT_DIR, LOGS_DIR, RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"✓ Directorio verificado: {dir_path}")
        
        logger.info(f"✓ Target column: {TARGET_COLUMN}")
        logger.info(f"✓ Columnas esperadas: {H7_EXPECTED_COLUMNS}")
        
        # ====================================================================
        # FASE 1: CARGA DE DATOS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FASE 1: CARGA DE DATOS")
        logger.info("=" * 80)
        
        data_manager = DataManager()
        
        # Intentar cargar train.csv
        if not os.path.exists(TRAIN_FILE):
            logger.error(f"✗ Archivo no encontrado: {TRAIN_FILE}")
            logger.info(f"  Archivos disponibles en '{os.path.dirname(TRAIN_FILE)}':")
            if os.path.exists(os.path.dirname(TRAIN_FILE)):
                for file in os.listdir(os.path.dirname(TRAIN_FILE)):
                    logger.info(f"    - {file}")
            return False
        
        logger.info(f"Cargando archivo de entrenamiento: {TRAIN_FILE}")
        train_data = data_manager.load_data(TRAIN_FILE)
        
        if train_data is None:
            logger.error("✗ Error al cargar datos de entrenamiento")
            return False
        
        logger.info(f"✓ Datos de entrenamiento cargados: {train_data.shape[0]} filas, {train_data.shape[1]} columnas")
        
        # Intentar cargar test.csv o val.csv
        test_data = None
        val_data = None
        
        # Opción 1: Archivos separados (train + val + test)
        if os.path.exists(VAL_FILE):
            logger.info(f"Cargando archivo de validación: {VAL_FILE}")
            val_data = data_manager.load_data(VAL_FILE)
            if val_data is not None:
                logger.info(f"✓ Datos de validación cargados: {val_data.shape[0]} filas")
        
        if os.path.exists(TEST_FILE):
            logger.info(f"Cargando archivo de prueba: {TEST_FILE}")
            test_data = data_manager.load_data(TEST_FILE)
            if test_data is not None:
                logger.info(f"✓ Datos de prueba cargados: {test_data.shape[0]} filas")
        
        # Opción 2: Si no existen test/val, dividir train_data
        if val_data is None and test_data is None:
            logger.info("No se encontraron archivos separados de validación/test. Dividiendo datos de entrenamiento...")
            try:
                train_set, val_set = data_manager.split_data(
                    train_data, 
                    target_column=TARGET_COLUMN
                )
                train_data = train_set
                val_data = val_set
                logger.info(f"✓ Datos divididos: {len(train_data)} entrenamiento, {len(val_data)} validación")
            except ValueError as e:
                logger.error(f"✗ Error al dividir datos: {e}")
                return False
        
        # ====================================================================
        # FASE 2: VALIDACIÓN DE ESTRUCTURA
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FASE 2: VALIDACIÓN DE ESTRUCTURA")
        logger.info("=" * 80)
        
        # Validar columnas en train
        logger.info("Validando columnas en datos de entrenamiento...")
        missing_cols = H7_EXPECTED_COLUMNS - set(train_data.columns)
        extra_cols = set(train_data.columns) - H7_EXPECTED_COLUMNS
        
        if missing_cols:
            logger.warning(f"⚠ Columnas faltantes en train: {missing_cols}")
        
        if extra_cols:
            logger.info(f"ℹ Columnas extra en train: {extra_cols}")
        
        logger.info(f"✓ Estructura de train validada")
        
        # Validar columna objetivo
        if TARGET_COLUMN not in train_data.columns:
            logger.error(f"✗ Columna objetivo '{TARGET_COLUMN}' no encontrada en train")
            logger.error(f"  Columnas disponibles: {list(train_data.columns)}")
            return False
        
        logger.info(f"✓ Columna objetivo '{TARGET_COLUMN}' presente")
        
        # Validar clases
        classes = train_data[TARGET_COLUMN].unique()
        logger.info(f"✓ Clases encontradas: {sorted(classes)}")
        logger.info(f"  Distribución:")
        for cls, count in train_data[TARGET_COLUMN].value_counts().items():
            pct = (count / len(train_data)) * 100
            logger.info(f"    - {cls}: {count} muestras ({pct:.1f}%)")
        
        # ====================================================================
        # FASE 3: ENTRENAR MODELO
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FASE 3: ENTRENAMIENTO DEL MODELO")
        logger.info("=" * 80)
        
        model_trainer = ModelTrainer(target_column=TARGET_COLUMN)
        
        logger.info("Entrenando modelo con datos de entrenamiento...")
        model_trainer.train(train_data)
        
        logger.info("✓ Modelo entrenado exitosamente")
        
        # ====================================================================
        # FASE 4: EVALUAR MODELO
        # ====================================================================
        if val_data is not None:
            logger.info("\n" + "=" * 80)
            logger.info("FASE 4: EVALUACIÓN DEL MODELO")
            logger.info("=" * 80)
            
            logger.info("Evaluando modelo en conjunto de validación...")
            metrics = model_trainer.evaluate(val_data)
            
            # Guardar reporte de evaluación
            if SAVE_EVALUATION_REPORT:
                logger.info(f"Guardando reporte de evaluación en: {EVALUATION_REPORT_FILE}")
                
                metrics_report = {
                    'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
                    'value': [
                        metrics['accuracy'],
                        metrics['precision'],
                        metrics['recall'],
                        metrics['f1_score']
                    ]
                }
                
                report_df = pd.DataFrame(metrics_report)
                report_df.to_csv(EVALUATION_REPORT_FILE, index=False)
                logger.info(f"✓ Reporte guardado")
            
            # Guardar matriz de confusión (visualización)
            if SAVE_CONFUSION_MATRIX:
                try:
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import ConfusionMatrixDisplay
                    
                    logger.info(f"Guardando matriz de confusión en: {CONFUSION_MATRIX_FILE}")
                    
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=metrics['confusion_matrix'],
                        display_labels=sorted(train_data[TARGET_COLUMN].unique())
                    )
                    disp.plot(cmap='Blues', values_format='d')
                    plt.title('Matriz de Confusión - Tabla H7')
                    plt.tight_layout()
                    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"✓ Matriz de confusión guardada")
                except ImportError:
                    logger.warning("⚠ matplotlib no disponible, omitiendo visualización de matriz")
                except Exception as e:
                    logger.warning(f"⚠ Error al guardar matriz de confusión: {e}")
            
            # Guardar feature importance
            if SAVE_FEATURE_IMPORTANCE:
                try:
                    import matplotlib.pyplot as plt
                    
                    logger.info(f"Guardando importancia de características en: {FEATURE_IMPORTANCE_FILE}")
                    
                    model_info = model_trainer.get_model_info()
                    features = model_info['feature_names']
                    importances = model_info['feature_importances'].values()
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(features, importances)
                    plt.xlabel('Importancia')
                    plt.title('Importancia de Características - Tabla H7')
                    plt.tight_layout()
                    plt.savefig(FEATURE_IMPORTANCE_FILE, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"✓ Importancia guardada")
                except ImportError:
                    logger.warning("⚠ matplotlib no disponible, omitiendo visualización de importancia")
                except Exception as e:
                    logger.warning(f"⚠ Error al guardar importancia: {e}")
        else:
            logger.info("\n⚠ No hay datos de validación disponibles, omitiendo evaluación")
        
        # ====================================================================
        # FASE 5: PREDICCIONES EN CONJUNTO DE PRUEBA
        # ====================================================================
        if test_data is not None:
            logger.info("\n" + "=" * 80)
            logger.info("FASE 5: PREDICCIONES EN CONJUNTO DE PRUEBA")
            logger.info("=" * 80)
            
            logger.info("Generando predicciones en conjunto de prueba...")
            
            # Identificar columna de IDs
            if 'id' in test_data.columns:
                test_ids = test_data['id'].values
            elif 'n' in test_data.columns:
                test_ids = test_data['n'].values
            else:
                test_ids = np.arange(len(test_data))
                logger.warning("⚠ No se encontró columna 'id', usando índices como identificadores")
            
            # Obtener predicciones
            result = model_trainer.predict(test_data)
            predictions = result['predictions']
            probabilities = result['probabilities']
            
            # Crear dataframe de sumisión
            submission_data = {
                'id': test_ids,
                'prediction': predictions
            }
            
            # Agregar probabilidades si están disponibles
            if probabilities is not None:
                unique_classes = sorted(train_data[TARGET_COLUMN].unique())
                for i, cls in enumerate(unique_classes):
                    submission_data[f'prob_{cls}'] = probabilities[:, i]
            
            submission_df = pd.DataFrame(submission_data)
            
            # Guardar sumisión
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                MODEL_OUTPUT_DIR, 
                f'submission_h7_{timestamp}.csv'
            )
            
            submission_df.to_csv(output_filename, index=False)
            logger.info(f"✓ Predicciones guardadas en: {output_filename}")
            logger.info(f"  - Muestras predichas: {len(submission_df)}")
            logger.info(f"  - Distribución de predicciones:")
            for cls, count in pd.Series(predictions).value_counts().items():
                pct = (count / len(predictions)) * 100
                logger.info(f"    - {cls}: {count} ({pct:.1f}%)")
        else:
            logger.info("\n⚠ No hay datos de prueba disponibles, omitiendo predicciones")
        
        # ====================================================================
        # FASE 6: GUARDAR MODELO (Opcional)
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FASE 6: INFORMACIÓN FINAL DEL MODELO")
        logger.info("=" * 80)
        
        model_info = model_trainer.get_model_info()
        logger.info(f"Tipo de modelo: {model_info['model_type']}")
        logger.info(f"Número de características: {model_info['n_features']}")
        logger.info(f"Características utilizadas: {model_info['feature_names']}")
        logger.info(f"Parámetros del modelo: {model_info['model_params']}")
        
        # ====================================================================
        # FINALIZACIÓN
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("✓ PIPELINE FINALIZADO EXITOSAMENTE")
        logger.info("=" * 80)
        
        logger.info(f"Tiempo final: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Archivos generados en: {MODEL_OUTPUT_DIR}/")
        
        return True
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrumpido por usuario")
        return False
    
    except Exception as e:
        logger.error(f"\n✗ Error inesperado: {e}", exc_info=True)
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
