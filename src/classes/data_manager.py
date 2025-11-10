# src/classes/data_manager.py

import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE_SPLIT

class DataManager:
    """
    Clase para manejar la carga, división y preprocesamiento de datos.
    """

    def __init__(self):
        """
        Inicializador de la clase DataManager.
        """
        pass  # En futuras versiones, podrías inicializar aquí parámetros de preprocesamiento

    def load_data(self, file_path: str):
        """
        Carga datos desde un archivo CSV.

        Args:
            file_path (str): La ruta al archivo CSV.

        Returns:
            pandas.DataFrame: Un DataFrame con los datos cargados, o None si hay un error.
        """
        try:
            print(f"Cargando datos desde: {file_path}")
            df = pd.read_csv(file_path)
            print("Datos cargados exitosamente.")
            return df
        except FileNotFoundError:
            print(f"Error: El archivo no fue encontrado en la ruta '{file_path}'")
            return None
        except Exception as e:
            print(f"Ocurrió un error inesperado al cargar el archivo: {e}")
            return None

    def split_data(self, data: pd.DataFrame, target_column: str = 'target'):
        """
        Divide los datos en conjuntos de entrenamiento y validación.

        Args:
            data (pd.DataFrame): El DataFrame que se va a dividir.
            target_column (str): El nombre de la columna objetivo (la que se quiere predecir).

        Returns:
            tuple: Una tupla conteniendo (X_train, X_val, y_train, y_val).
        """
        print(f"Dividiendo los datos. Tamaño del conjunto de prueba: {TEST_SIZE}, estado aleatorio: {RANDOM_STATE_SPLIT}")

        if target_column not in data.columns:
            raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")
            
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE_SPLIT,
            stratify=y if y.nunique() > 1 else None  # Usar estratificación si es un problema de clasificación
        )
        
        print(f"División completada. Tamaño del conjunto de entrenamiento: {len(X_train)}, Tamaño del conjunto de validación: {len(X_val)}")
        
        # Para simplificar, devolvemos los DataFrames de entrenamiento y validación completos
        # El ModelTrainer se encargará de separar X e y internamente
        train_set = pd.concat([X_train, y_train], axis=1)
        val_set = pd.concat([X_val, y_val], axis=1)
        
        return train_set, val_set
        
    def preprocess_data(self, data: pd.DataFrame, is_train: bool = True):
        """
        Realiza el preprocesamiento de los datos.
        
        Esta es una función de ejemplo. Aquí es donde implementarías la limpieza,
        manejo de valores nulos, escalado de características, etc.

        Args:
            data (pd.DataFrame): El DataFrame a preprocesar.
            is_train (bool): Indica si se está procesando el conjunto de entrenamiento.

        Returns:
            pd.DataFrame: El DataFrame preprocesado.
        """
        print("Realizando preprocesamiento de datos...")
        
        # Copiamos para no modificar el dataframe original
        processed_data = data.copy()
        
        # Ejemplo de preprocesamiento simple: Rellenar valores nulos con la media
        for col in processed_data.select_dtypes(include=['float64', 'int64']).columns:
            if processed_data[col].isnull().any():
                mean_value = processed_data[col].mean()
                processed_data[col].fillna(mean_value, inplace=True)
                print(f"  - Columna '{col}': Rellenados valores nulos con la media ({mean_value:.2f})")

        # Aquí podrías añadir más pasos:
        # - Codificación de variables categóricas (One-Hot Encoding, Label Encoding)
        # - Escalado de características (StandardScaler, MinMaxScaler)
        # - Creación de nuevas características (Feature Engineering)
        
        print("Preprocesamiento completado.")
        return processed_data
