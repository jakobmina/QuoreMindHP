# config.py
# CONFIGURACIÓN PARA TABLA H7: ESTADOS DE MOMENTO ENTRELAZADOS

import os
from pathlib import Path

# ============================================================================
# RUTAS DE DATOS
# ============================================================================
# Directorio raíz de datos
DATA_DIR = 'data'

# Archivos de entrada para Tabla H7
TRAIN_FILE = os.path.join(DATA_DIR, 'tabla_h7_train.csv')
VAL_FILE = os.path.join(DATA_DIR, 'tabla_h7_val.csv')
TEST_FILE = os.path.join(DATA_DIR, 'tabla_h7_test.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

# Alternativa: Si usas un único archivo y lo divides dentro del pipeline
# TABLA_H7_FILE = os.path.join(DATA_DIR, 'tabla_h7_complete.csv')

# ============================================================================
# DIRECTORIO DE SALIDA
# ============================================================================
MODEL_OUTPUT_DIR = 'models'
LOGS_DIR = 'logs'
RESULTS_DIR = 'results'

# Crear directorios si no existen
for dir_path in [MODEL_OUTPUT_DIR, LOGS_DIR, RESULTS_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ============================================================================
# PARÁMETROS DEL MODELO: GradientBoostingClassifier para Tabla H7
# ============================================================================
# Tabla H7 tiene 6 clases (001, 010, 011, 100, 101, 110)
# Estos parámetros están ajustados para clasificación multiclase

MODEL_PARAMS = {
    'n_estimators': 100,           # Número de árboles de decisión
    'max_depth': 5,                # Profundidad máxima (evita overfitting)
    'learning_rate': 0.1,          # Tasa de aprendizaje (0.01-0.2)
    'subsample': 0.8,              # Fracción de muestras para cada árbol
    'min_samples_split': 5,        # Mínimo de muestras para dividir nodo
    'min_samples_leaf': 2,         # Mínimo de muestras en hoja
    'loss': 'log_loss',            # Para clasificación multiclase
    'random_state': 42             # Para reproducibilidad
}

# ============================================================================
# PARÁMETROS DE DIVISIÓN DE DATOS
# ============================================================================
# Tabla H7 tiene solo 6 muestras en este ejemplo
# En producción con datos más grandes, ajustar estos valores

TEST_SIZE = 0.2                    # 20% para validación (o test si usas 2 splits)
RANDOM_STATE_SPLIT = 42            # Seed para reproducibilidad de splits

# Alternativa para división 3-way (train/val/test):
# TRAIN_SIZE = 0.7
# VAL_SIZE = 0.15
# TEST_SIZE = 0.15

# ============================================================================
# PARÁMETROS ESPECÍFICOS DE TABLA H7
# ============================================================================

# Columna objetivo (Target quántico de 3 bits)
TARGET_COLUMN = 'Target |x⟩'

# Columnas numéricas esperadas en Tabla H7
H7_NUMERIC_FEATURES = [
    'Momento',
    'Complemento H7',
    'E_Metriplética',
    'Fase Berry (rad)'
]

# Columnas categóricas/índices (no usar como features)
H7_NON_FEATURES = {
    'n',                           # Índice
    'Estado 2-1',                  # Estado entrelazado (categórico)
}

# Columnas derivadas (se generan en preprocesamiento)
H7_DERIVED_FEATURES = {
    'mahalanobis_distance'         # Distancia de Mahalanobis
}

# Todas las columnas esperadas en Tabla H7
H7_EXPECTED_COLUMNS = {
    'n',
    'Momento',
    'Complemento H7',
    'Estado 2-1',
    'E_Metriplética',
    'Fase Berry (rad)',
    TARGET_COLUMN
}

# Clases esperadas (códigos binarios de 3 qubits)
H7_EXPECTED_CLASSES = ['001', '010', '011', '100', '101', '110']

# Valores válidos para Estado 2-1 (simetría de entrelazamiento)
H7_VALID_STATES = ['(0, 1)', '(1, 0)']

# ============================================================================
# PARÁMETROS DE VALIDACIÓN Y PREPROCESAMIENTO
# ============================================================================

# Tolerancia para valores nulos (máximo % de nulos permitido por columna)
MAX_NULL_PERCENT = 0.5             # 50%

# Método para llenar valores nulos en características numéricas
NULL_FILL_METHOD = 'mean'          # 'mean', 'median', 'forward_fill', etc.

# ============================================================================
# PARÁMETROS DE LOGGING Y REPORTES
# ============================================================================

# Nivel de verbosidad en consola
VERBOSE = True

# Guardar logs en archivo
SAVE_LOGS = True
LOG_FILE = os.path.join(LOGS_DIR, 'pipeline.log')

# Generar reporte de evaluación (CSV con métricas)
SAVE_EVALUATION_REPORT = True
EVALUATION_REPORT_FILE = os.path.join(RESULTS_DIR, 'evaluation_metrics.csv')

# Guardar matriz de confusión como imagen
SAVE_CONFUSION_MATRIX = True
CONFUSION_MATRIX_FILE = os.path.join(RESULTS_DIR, 'confusion_matrix.png')

# Guardar feature importance como imagen
SAVE_FEATURE_IMPORTANCE = True
FEATURE_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, 'feature_importance.png')

# ============================================================================
# PARÁMETROS DE MODELO ALTERNATIVO (Comentado - para futuro)
# ============================================================================

# Si en el futuro quieres usar un modelo diferente:
"""
from sklearn.ensemble import RandomForestClassifier, XGBClassifier

# RandomForest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

# XGBoost (si está disponible)
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'objective': 'multi:softmax',
    'num_class': 6,  # 6 clases en H7
    'random_state': 42
}
"""

# ============================================================================
# VALIDACIONES DE CONFIGURACIÓN
# ============================================================================

def validate_config():
    """
    Valida que la configuración sea consistente para Tabla H7.
    
    Raises:
        ValueError: Si hay inconsistencias en la configuración.
    """
    # Validar que target_column esté en expected_columns
    if TARGET_COLUMN not in H7_EXPECTED_COLUMNS:
        raise ValueError(
            f"TARGET_COLUMN '{TARGET_COLUMN}' no está en H7_EXPECTED_COLUMNS"
        )
    
    # Validar que todas las features numéricas estén en expected_columns
    for feat in H7_NUMERIC_FEATURES:
        if feat not in H7_EXPECTED_COLUMNS:
            raise ValueError(
                f"Feature '{feat}' en H7_NUMERIC_FEATURES no está en H7_EXPECTED_COLUMNS"
            )
    
    # Validar que non_features no se superpongan con numeric_features
    overlap = set(H7_NUMERIC_FEATURES) & H7_NON_FEATURES
    if overlap:
        raise ValueError(
            f"Solapamiento entre H7_NUMERIC_FEATURES y H7_NON_FEATURES: {overlap}"
        )
    
    # Validar parámetros del modelo
    if not isinstance(MODEL_PARAMS, dict):
        raise ValueError("MODEL_PARAMS debe ser un diccionario")
    
    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        raise ValueError("TEST_SIZE debe estar entre 0 y 1")
    
    print("✓ Configuración validada exitosamente para Tabla H7")

# Validar configuración al importar
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"⚠ Advertencia de configuración: {e}")

# ============================================================================
# INFORMACIÓN ÚTIL
# ============================================================================
"""
ESTRUCTURA ESPERADA DE TABLA H7:

n | Momento | Complemento H7 | Estado 2-1 | E_Metriplética | Fase Berry (rad) | Target |x⟩
--|---------|----------------|------------|----------------|------------------|----------
1 |    1    |        6       |  (1, 0)   |     0.4783    |      1.2650      |   001
2 |    2    |        5       |  (1, 0)   |     0.4609    |      2.1626      |   010
3 |    3    |        4       |  (1, 0)   |     0.4513    |      3.0602      |   011
4 |    4    |        3       |  (0, 1)   |     0.4513    |      3.2230      |   100
5 |    5    |        2       |  (0, 1)   |     0.4609    |      4.1206      |   101
6 |    6    |        1       |  (0, 1)   |     0.4783    |      5.0182      |   110

CARACTERÍSTICAS NUMÉRICAS (4):
- Momento: índice de momento cuántico (1-6)
- Complemento H7: complemento a 7 del momento (1-6)
- E_Metriplética: energía métrica (0.45-0.49)
- Fase Berry (rad): fase de Berry en radianes (1.26-5.02)

VARIABLES CATEGÓRICAS:
- Estado 2-1: tupla que indica estado entrelazado (0,1) o (1,0)
  * Muestra simetría: exactamente 3 de cada tipo

OBJETIVO:
- Target |x⟩: Estado quántico de 3 qubits (6 clases: 001-110)
  * Binario de 3 bits
  * Distribución equilibrada (1 muestra por clase en ejemplo)

ÍNDICE:
- n: numeración de filas (1-6)
  * NO es característica del modelo
"""
