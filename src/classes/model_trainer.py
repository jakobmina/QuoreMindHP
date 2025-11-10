# src/classes/model_trainer.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .data_manager import DataManager
from config import MODEL_PARAMS

class ModelTrainer:
    """
    Clase para entrenar, evaluar y hacer predicciones con un modelo de ML.
    """

    def __init__(self, target_column: str = 'target'):
        """
        Inicializador de la clase ModelTrainer.
        """
        self.model = GradientBoostingClassifier(**MODEL_PARAMS)
        self.target_column = target_column
        self.data_manager = DataManager() # Instancia para usar el preprocesamiento

    def train(self, train_data: pd.DataFrame):
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            train_data (pd.DataFrame): DataFrame de entrenamiento que incluye la columna objetivo.
        """
        if self.target_column not in train_data.columns:
            raise ValueError(f"La columna objetivo '{self.target_column}' no se encuentra en los datos de entrenamiento.")
        
        # 1. Preprocesar los datos
        processed_train_data = self.data_manager.preprocess_data(train_data, is_train=True)
        
        # 2. Separar características (X) y objetivo (y)
        X_train = processed_train_data.drop(columns=[self.target_column])
        y_train = processed_train_data[self.target_column]
        
        # Asegurarse de que no haya columnas no numéricas que no se hayan procesado
        X_train = X_train.select_dtypes(include=['number'])
        
        # 3. Entrenar el modelo
        print(f"Entrenando un modelo {type(self.model).__name__} con {X_train.shape[1]} características...")
        self.model.fit(X_train, y_train)
        print("Modelo entrenado exitosamente.")

    def evaluate(self, val_data: pd.DataFrame) -> dict:
        """
        Evalúa el modelo en el conjunto de validación.

        Args:
            val_data (pd.DataFrame): DataFrame de validación que incluye la columna objetivo.

        Returns:
            dict: Un diccionario con las métricas de evaluación.
        """
        if self.target_column not in val_data.columns:
            raise ValueError(f"La columna objetivo '{self.target_column}' no se encuentra en los datos de validación.")
            
        # 1. Preprocesar los datos de validación
        processed_val_data = self.data_manager.preprocess_data(val_data, is_train=False)

        # 2. Separar características (X) y objetivo (y)
        X_val = processed_val_data.drop(columns=[self.target_column])
        y_val = processed_val_data[self.target_column]
        
        # Asegurarse de usar las mismas columnas que en el entrenamiento
        X_val = X_val.select_dtypes(include=['number'])

        # 3. Realizar predicciones
        y_pred = self.model.predict(X_val)
        
        # 4. Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted')
        }
        
        return metrics

    def predict(self, test_data: pd.DataFrame):
        """
        Realiza predicciones en un nuevo conjunto de datos.

        Args:
            test_data (pd.DataFrame): DataFrame con los datos de prueba.

        Returns:
            np.ndarray: Un array con las predicciones.
        """
        # 1. Preprocesar los datos de prueba
        processed_test_data = self.data_manager.preprocess_data(test_data, is_train=False)
        
        # Asegurarse de usar las mismas columnas que en el entrenamiento
        X_test = processed_test_data.select_dtypes(include=['number'])
        
        # 2. Realizar predicciones
        print(f"Realizando predicciones sobre {len(X_test)} muestras...")
        predictions = self.model.predict(X_test)
        
        return predictions
