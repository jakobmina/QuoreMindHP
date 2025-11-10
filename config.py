# config.py

import os

# Rutas a los archivos de datos
DATA_DIR = 'data'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

# Directorio para guardar los modelos entrenados
MODEL_OUTPUT_DIR = 'models'

# Parámetros del modelo (ejemplo)
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

# Parámetros para la división de datos
TEST_SIZE = 0.2
RANDOM_STATE_SPLIT = 42
