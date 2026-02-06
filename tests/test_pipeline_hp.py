import pytest
import pandas as pd
import numpy as np
import os
from src.classes.data_manager import DataManager
from src.classes.model_trainer import ModelTrainer
from config import H7_EXPECTED_COLUMNS, TARGET_COLUMN

def test_full_pipeline_hp():
    """Verifica que el pipeline completo funcione con QuoreMindHP habilitado."""
    # 1. Crear datos de prueba (Mini Tabla H7)
    data = {
        'n': [1, 2, 3, 4, 5, 6],
        'Momento': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Complemento H7': [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        'Estado 2-1': ['(1, 0)', '(1, 0)', '(1, 0)', '(0, 1)', '(0, 1)', '(0, 1)'],
        'E_Metriplética': [0.4783, 0.4609, 0.4513, 0.4513, 0.4609, 0.4783],
        'Fase Berry (rad)': [1.2650, 2.1626, 3.0602, 3.2230, 4.1206, 5.0182],
        TARGET_COLUMN: ['001', '010', '011', '100', '101', '110']
    }
    df = pd.DataFrame(data)
    
    # 2. Inicializar componentes
    dm = DataManager()
    trainer = ModelTrainer(target_column=TARGET_COLUMN)
    
    # 3. Preprocesamiento (Fase 4 activará MahalanobisHP)
    processed_df = dm.preprocess_data(df, is_train=True)
    
    assert 'mahalanobis_distance' in processed_df.columns
    assert not processed_df['mahalanobis_distance'].isnull().any()
    
    # 4. Entrenamiento
    # Nota: Con tan pocos datos, el estratificado podría fallar si TEST_SIZE es muy grande
    # Pero aquí usamos el DataFrame completo para entrenar
    trainer.train(df)
    
    # 5. Predicción (debería incluir bayesian_confidence)
    test_df = df.drop(columns=[TARGET_COLUMN])
    results = trainer.predict(test_df)
    
    assert 'predictions' in results
    assert 'bayesian_confidence' in results
    assert len(results['bayesian_confidence']) == 6
    assert all(0 <= p <= 1 for p in results['bayesian_confidence'])
    
    print("✓ Pipeline HP Test: PASSED")

if __name__ == "__main__":
    test_full_pipeline_hp()
