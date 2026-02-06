import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE_SPLIT
from quoremindhp import StatisticalAnalysisHP
from quoremindhp_integration import MahalanobisHP

class DataManager:
    """
    Clase para manejar la carga, división y preprocesamiento de datos.
    Adaptada para procesar ESTADOS DE MOMENTO ENTRELAZADOS (Tabla H7).
    """

    def __init__(self):
        """
        Inicializador de la clase DataManager.
        """
        self.mahalanobis_mean_vector = None
        self.mahalanobis_inv_cov_matrix = None
        self.tabla_h7_columns = {
            'n', 'Momento', 'Complemento H7', 'Estado 2-1', 
            'E_Metriplética', 'Fase Berry (rad)', 'Target |x⟩'
        }

    def load_data(self, file_path: str):
        """
        Carga datos desde un archivo CSV.
        Específicamente diseñado para la Tabla H7.

        Args:
            file_path (str): La ruta al archivo CSV.

        Returns:
            pandas.DataFrame: Un DataFrame con los datos cargados, o None si hay un error.
        """
        try:
            print(f"Cargando datos desde: {file_path}")
            df = pd.read_csv(file_path)
            
            # Validar estructura de Tabla H7
            if not self.tabla_h7_columns.issubset(set(df.columns)):
                missing = self.tabla_h7_columns - set(df.columns)
                print(f"⚠ Advertencia: Columnas faltantes para Tabla H7: {missing}")
            
            print("Datos cargados exitosamente.")
            print(f"Forma del dataset: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: El archivo no fue encontrado en la ruta '{file_path}'")
            return None
        except Exception as e:
            print(f"Ocurrió un error inesperado al cargar el archivo: {e}")
            return None

    def split_data(self, data: pd.DataFrame, target_column: str = 'Target |x⟩'):
        """
        Divide los datos en conjuntos de entrenamiento y validación.

        Args:
            data (pd.DataFrame): El DataFrame que se va a dividir.
            target_column (str): El nombre de la columna objetivo (default: 'Target |x⟩' para Tabla H7).

        Returns:
            tuple: Una tupla conteniendo (train_set, val_set) como DataFrames completos.
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
            stratify=y if y.nunique() > 1 else None
        )
        
        print(f"División completada.")
        print(f"  - Entrenamiento: {len(X_train)} muestras")
        print(f"  - Validación: {len(X_val)} muestras")
        
        train_set = pd.concat([X_train, y_train], axis=1)
        val_set = pd.concat([X_val, y_val], axis=1)
        
        return train_set, val_set
        
    def preprocess_data(self, data: pd.DataFrame, is_train: bool = True):
        """
        Realiza el preprocesamiento de los datos para TABLA H7.
        
        Procesa específicamente:
        - Columnas cuantitativas: Momento, Complemento H7, E_Metriplética, Fase Berry (rad)
        - Columnas categóricas: Estado 2-1, Target |x⟩
        - Valida la estructura de datos cuánticos entrelazados

        Args:
            data (pd.DataFrame): El DataFrame a preprocesar.
            is_train (bool): Indica si se está procesando el conjunto de entrenamiento.

        Returns:
            pd.DataFrame: El DataFrame preprocesado.
        """
        print("Realizando preprocesamiento de datos (TABLA H7 - Estados Entrelazados)...")
        
        processed_data = data.copy()
        
        # Validar estructura esperada
        expected_cols = {
            'n', 'Momento', 'Complemento H7', 'Estado 2-1', 
            'E_Metriplética', 'Fase Berry (rad)', 'Target |x⟩'
        }
        missing_cols = expected_cols - set(processed_data.columns)
        if missing_cols:
            print(f"  ⚠ Advertencia: Faltan columnas esperadas: {missing_cols}")
        
        # FASE 1: Procesar columnas numéricas (sin incluir índice 'n')
        numeric_cols = ['Momento', 'Complemento H7', 'E_Metriplética', 'Fase Berry (rad)']
        for col in numeric_cols:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                if processed_data[col].isnull().any():
                    mean_value = processed_data[col].mean()
                    processed_data[col].fillna(mean_value, inplace=True)
                    print(f"  - Columna '{col}': Rellenados valores nulos con media ({mean_value:.4f})")
                else:
                    print(f"  - Columna '{col}': Validada (sin valores nulos)")
        
        # FASE 2: Validar estructura de Estado 2-1 (tuplas)
        if 'Estado 2-1' in processed_data.columns:
            # Convertir a string si no lo es, para validar formato
            processed_data['Estado 2-1'] = processed_data['Estado 2-1'].astype(str)
            valid_estados = processed_data['Estado 2-1'].isin(['(0, 1)', '(1, 0)'])
            if not valid_estados.all():
                print(f"  ⚠ Advertencia: {(~valid_estados).sum()} estados con formato inválido detectados")
            else:
                print(f"  - Columna 'Estado 2-1': Validada (todos los estados en formato correcto)")
        
        # FASE 3: Validar Target |x⟩ (valores binarios: 000-110)
        if 'Target |x⟩' in processed_data.columns:
            processed_data['Target |x⟩'] = processed_data['Target |x⟩'].astype(str)
            valid_targets = processed_data['Target |x⟩'].str.match(r'^[01]{3}$')
            if not valid_targets.all():
                print(f"  ⚠ Advertencia: {(~valid_targets).sum()} targets con formato inválido detectados")
            else:
                print(f"  - Columna 'Target |x⟩': Validada (todos los estados son qubits válidos)")
        
        # FASE 4: Calcular simetría de Mahalanobis solo si es apropiado
       # En data_manager.py, en preprocess_data()

      if is_train:
          # Usar ALTA PRECISIÓN para entrenamiento
          mahal_hp = MahalanobisHP(precision_dps=100)
          mean_vec, inv_cov = mahal_hp.precompute_components(
            train_data_for_mahalanobis.values.tolist()
          )
    
          # Calcular distancias
          distances = []
         for _, row in train_data_for_mahalanobis.iterrows():
            result = mahal_hp.calculate_for_point(
            row.values.tolist(),
            mean_vec, 
            inv_cov
        )
        distances.append(float(result.distance))
    
    processed_data['mahalanobis_distance'] = distances
    print("✓ Mahalanobis HP: Precisión 100 dígitos")
            if not numeric_features.empty:
                self.mahalanobis_mean_vector, self.mahalanobis_inv_cov_matrix = \
                    StatisticalAnalysisHP.precompute_mahalanobis_components(numeric_features.values.tolist())

                mahalanobis_distances = []
                for index, row in numeric_features.iterrows():
                    point = row.values.tolist()
                    distance = StatisticalAnalysisHP.calculate_mahalanobis_for_point(
                        point, self.mahalanobis_mean_vector, self.mahalanobis_inv_cov_matrix
                    )
                    mahalanobis_distances.append(float(distance))
                
                processed_data['mahalanobis_distance'] = mahalanobis_distances
                print("  - Característica 'mahalanobis_distance': Creada para conjunto de entrenamiento")
                print(f"    Rango: [{min(mahalanobis_distances):.4f}, {max(mahalanobis_distances):.4f}]")

        else:
            # Aplicar Mahalanobis a validación
            if self.mahalanobis_mean_vector is not None and self.mahalanobis_inv_cov_matrix is not None:
                numeric_features = processed_data.select_dtypes(include=['number']).drop(
                    columns=[col for col in ['n', 'target', 'id'] if col in processed_data.columns], 
                    errors='ignore'
                )
                
                if not numeric_features.empty:
                    mahalanobis_distances = []
                    for index, row in numeric_features.iterrows():
                        point = row.values.tolist()
                        distance = StatisticalAnalysisHP.calculate_mahalanobis_for_point(
                            point, self.mahalanobis_mean_vector, self.mahalanobis_inv_cov_matrix
                        )
                        mahalanobis_distances.append(float(distance))
                    
                    processed_data['mahalanobis_distance'] = mahalanobis_distances
                    print("  - Característica 'mahalanobis_distance': Creada para conjunto de validación")

        print("✓ Preprocesamiento completado exitosamente.")
        return processed_data
