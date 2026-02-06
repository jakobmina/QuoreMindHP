import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .data_manager import DataManager
from config import MODEL_PARAMS
from quoremindhp_integration import BayesianAnalysisH7

class ModelTrainer:
    """
    Clase para entrenar, evaluar y hacer predicciones con un modelo de ML.
    Adaptada específicamente para Tabla H7 (ESTADOS DE MOMENTO ENTRELAZADOS).
    """

    def __init__(self, target_column: str = 'Target |x⟩'):
        """
        Inicializador de la clase ModelTrainer.
        
        Args:
            target_column (str): Nombre de la columna objetivo (default: 'Target |x⟩' para Tabla H7).
        """
        self.model = GradientBoostingClassifier(**MODEL_PARAMS)
        self.target_column = target_column
        self.data_manager = DataManager()
        self.feature_columns = None  # Almacenar columnas de entrenamiento
        self.feature_names = None    # Nombres de características para reporting

    def _get_numeric_features(self, data: pd.DataFrame, exclude_target: bool = True) -> pd.DataFrame:
        """
        Obtiene solo las características numéricas relevantes para H7.
        
        Args:
            data (pd.DataFrame): DataFrame procesado.
            exclude_target (bool): Si excluir la columna objetivo.
            
        Returns:
            pd.DataFrame: DataFrame con solo características numéricas.
        """
        # Columnas a excluir (no son características de modelo)
        exclude_cols = {'n', 'Estado 2-1', 'mahalanobis_distance'}
        
        if exclude_target and self.target_column in data.columns:
            exclude_cols.add(self.target_column)
        
        # Obtener columnas numéricas
        numeric_data = data.select_dtypes(include=['number'])
        
        # Excluir columnas no deseadas
        feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')
        
        return feature_data

    def train(self, train_data: pd.DataFrame):
        """
        Entrena el modelo con los datos proporcionados.
        
        Pasos:
        1. Validar estructura de Tabla H7
        2. Preprocesar datos
        3. Extraer características numéricas
        4. Entrenar modelo GradientBoosting
        5. Reportar información de entrenamiento

        Args:
            train_data (pd.DataFrame): DataFrame de entrenamiento que incluye 'Target |x⟩'.
            
        Raises:
            ValueError: Si falta la columna objetivo o no hay características válidas.
        """
        if self.target_column not in train_data.columns:
            raise ValueError(
                f"La columna objetivo '{self.target_column}' no se encuentra en los datos de entrenamiento. "
                f"Columnas disponibles: {list(train_data.columns)}"
            )
        
        print("=" * 70)
        print("FASE 1: PREPROCESAMIENTO DE DATOS DE ENTRENAMIENTO")
        print("=" * 70)
        
        # Preprocesar los datos
        processed_train_data = self.data_manager.preprocess_data(train_data, is_train=True)
        
        print("\n" + "=" * 70)
        print("FASE 2: EXTRACCIÓN DE CARACTERÍSTICAS")
        print("=" * 70)
        
        # Extraer características numéricas
        X_train = self._get_numeric_features(processed_train_data, exclude_target=True)
        y_train = processed_train_data[self.target_column]
        
        # Validar que haya características válidas
        if X_train.empty:
            raise ValueError(
                f"No hay características numéricas válidas después del preprocesamiento. "
                f"Columnas disponibles: {list(processed_train_data.columns)}"
            )
        
        # Almacenar información de características para consistencia
        self.feature_columns = X_train.columns.tolist()
        self.feature_names = {i: col for i, col in enumerate(self.feature_columns)}
        
        print(f"✓ Características extraídas: {len(X_train.columns)}")
        print(f"  Nombres: {self.feature_columns}")
        print(f"✓ Muestras de entrenamiento: {X_train.shape[0]}")
        print(f"✓ Distribución de targets:")
        for target_val, count in y_train.value_counts().items():
            print(f"    - {target_val}: {count} muestras ({count/len(y_train)*100:.1f}%)")
        
        print("\n" + "=" * 70)
        print("FASE 3: ENTRENAMIENTO DEL MODELO")
        print("=" * 70)
        
        # Entrenar el modelo
        model_name = type(self.model).__name__
        print(f"Inicializando {model_name}...")
        print(f"  Parámetros: {MODEL_PARAMS}")
        
        self.model.fit(X_train, y_train)
        
        # Reportar información de entrenamiento
        print(f"\n✓ Modelo {model_name} entrenado exitosamente.")
        print(f"  Profundidad máxima: {self.model.max_depth}")
        print(f"  Número de estimadores: {self.model.n_estimators}")
        print(f"  Learning rate: {self.model.learning_rate}")
        print(f"  Importancia de características:")
        for feat_name, importance in zip(self.feature_columns, self.model.feature_importances_):
            print(f"    - {feat_name}: {importance:.4f}")
        
        print("\n" + "=" * 70)

    def evaluate(self, val_data: pd.DataFrame) -> dict:
      # En model_trainer.py, en evaluate()


# Análisis Bayesiano de predicciones
bayes_h7 = BayesianAnalysisH7(precision_dps=100)

# Para cada clase predicha
for target_class in np.unique(y_val):
    # Si tienes Estado 2-1 en val_data:
    if 'Estado 2-1' in val_data.columns:
        estado = val_data['Estado 2-1'].iloc[0]
        coherence = bayes_h7.calculate_entanglement_coherence(estado)
        print(f"  Coherencia ({target_class}): {mpmath.nstr(coherence, n=15)}")
        print("=" * 70)
        print("FASE 4: EVALUACIÓN EN CONJUNTO DE VALIDACIÓN")
        print("=" * 70)
        
        # Preprocesar los datos de validación
        processed_val_data = self.data_manager.preprocess_data(val_data, is_train=False)

        # Extraer características con las mismas columnas del entrenamiento
        X_val = self._get_numeric_features(processed_val_data, exclude_target=True)
        y_val = processed_val_data[self.target_column]
        
        # Validar que las características coincidan
        if set(X_val.columns) != set(self.feature_columns):
            missing = set(self.feature_columns) - set(X_val.columns)
            extra = set(X_val.columns) - set(self.feature_columns)
            if missing:
                print(f"⚠ Advertencia: Características faltantes: {missing}")
            if extra:
                print(f"⚠ Advertencia: Características extra: {extra}")
            X_val = X_val[self.feature_columns]  # Reordenar y seleccionar solo las del entrenamiento

        print(f"✓ Muestras de validación: {X_val.shape[0]}")
        print(f"✓ Características usadas: {len(X_val.columns)}")
        
        # Realizar predicciones
        print("\n  Realizando predicciones...")
        y_pred = self.model.predict(X_val)
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'y_true': y_val.values,
            'y_pred': y_pred
        }
        
        # Reportar métricas
        print("\n" + "-" * 70)
        print("RESULTADOS DE EVALUACIÓN")
        print("-" * 70)
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\n  Matriz de Confusión:")
        print(f"  {conf_matrix}")
        
        # Análisis por clase (si es multi-clase)
        if len(np.unique(y_val)) > 2:
            print(f"\n  Métricas por clase:")
            precision_per_class = precision_score(y_val, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_val, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
            
            for i, class_label in enumerate(np.unique(y_val)):
                print(f"    Clase '{class_label}':")
                print(f"      - Precision: {precision_per_class[i]:.4f}")
                print(f"      - Recall:    {recall_per_class[i]:.4f}")
                print(f"      - F1-Score:  {f1_per_class[i]:.4f}")
        
        print("=" * 70)
        
        return metrics

    def predict(self, test_data: pd.DataFrame) -> dict:
        """
        Realiza predicciones en un nuevo conjunto de datos.
        
        Args:
            test_data (pd.DataFrame): DataFrame con los datos de prueba (sin Target |x⟩).

        Returns:
            dict: Diccionario con:
                - 'predictions': array de predicciones
                - 'probabilities': matriz de probabilidades (si está disponible)
                - 'n_samples': número de muestras procesadas
                
        Raises:
            ValueError: Si el modelo no ha sido entrenado o faltan características.
        """
        if self.model is None or self.feature_columns is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecute train() primero.")
        
        print("=" * 70)
        print("FASE 5: PREDICCIONES EN NUEVOS DATOS")
        print("=" * 70)
        
        # Preprocesar los datos de prueba
        processed_test_data = self.data_manager.preprocess_data(test_data, is_train=False)
        
        # Extraer características
        X_test = self._get_numeric_features(processed_test_data, exclude_target=False)
        
        # Validar y alinear características
        if set(X_test.columns) != set(self.feature_columns):
            missing = set(self.feature_columns) - set(X_test.columns)
            if missing:
                raise ValueError(f"Características faltantes en datos de prueba: {missing}")
            X_test = X_test[self.feature_columns]
        
        print(f"✓ Muestras de prueba: {X_test.shape[0]}")
        print(f"✓ Características usadas: {len(X_test.columns)}")
        
        # Realizar predicciones
        print("\n  Realizando predicciones...")
        predictions = self.model.predict(X_test)
        
        # Obtener probabilidades si está disponible
        try:
            probabilities = self.model.predict_proba(X_test)
            print(f"  ✓ Probabilidades calculadas")
        except:
            probabilities = None
            print(f"  ⚠ Probabilidades no disponibles para este modelo")
        
        # Reportar resultados
        print(f"\n✓ Predicciones completadas.")
        print(f"  Distribución de predicciones:")
        unique, counts = np.unique(predictions, return_counts=True)
        for pred_val, count in zip(unique, counts):
            print(f"    - {pred_val}: {count} muestras ({count/len(predictions)*100:.1f}%)")
        
        print("=" * 70)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'n_samples': len(predictions),
            'feature_columns': self.feature_columns
        }

    def get_model_info(self) -> dict:
        """
        Retorna información del modelo entrenado.
        
        Returns:
            dict: Información del modelo (parámetros, características, importancias).
        """
        if self.model is None:
            return {'status': 'No entrenado'}
        
        return {
            'model_type': type(self.model).__name__,
            'target_column': self.target_column,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'feature_names': self.feature_columns,
            'feature_importances': dict(zip(self.feature_columns, self.model.feature_importances_)) if self.feature_columns else {},
            'model_params': self.model.get_params()
        }
