# main.py (debería verse así)

import os
import pandas as pd
from datetime import datetime
from src.classes.data_manager import DataManager  # Importa la clase
from src.classes.model_trainer import ModelTrainer # Importa la clase
from config import TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE, MODEL_OUTPUT_DIR

def run_pipeline():
    """
    Función principal para ejecutar el pipeline de entrenamiento del modelo.
    """
    print("Iniciando el pipeline de entrenamiento...")

    # Crear una instancia del gestor de datos
    data_manager = DataManager()
    
    # Cargar los datos
    train_data = data_manager.load_data(TRAIN_FILE)
    test_data = data_manager.load_data(TEST_FILE)

    if train_data is None or test_data is None:
        print("Error: No se pudieron cargar los archivos de datos. Asegúrate de que los archivos 'train.csv' y 'test.csv' estén en la carpeta 'data'.")
        return

    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    # Asegúrate de que tu train.csv tenga una columna llamada 'target'
    try:
        train_set, val_set = data_manager.split_data(train_data, target_column='target')
    except ValueError as e:
        print(f"Error al dividir los datos: {e}")
        print("Asegúrate de que el archivo 'train.csv' tenga una columna llamada 'target'.")
        return
    
    # Crear una instancia del entrenador de modelos
    model_trainer = ModelTrainer(target_column='target')
    
    # Entrenar el modelo
    print("\nEntrenando el modelo...")
    model_trainer.train(train_set)
    print("Entrenamiento completado.")
    
    # Evaluar el modelo
    print("\nEvaluando el modelo en el conjunto de validación...")
    evaluation_metrics = model_trainer.evaluate(val_set)
    print(f"Métricas de evaluación: {evaluation_metrics}")
    
    # Generar predicciones en el conjunto de prueba
    print("\nGenerando predicciones en el conjunto de prueba...")
    # Asegúrate de que el archivo 'test.csv' tenga una columna 'id' para la sumisión
    if 'id' not in test_data.columns:
        print("Advertencia: El archivo de prueba no tiene una columna 'id'. Se usará el índice como identificador.")
        test_ids = test_data.index
    else:
        test_ids = test_data['id']
        
    predictions = model_trainer.predict(test_data)
    
    # Guardar las predicciones en un archivo CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(MODEL_OUTPUT_DIR, f'submission_{timestamp}.csv')
    
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    submission_df = pd.DataFrame({'id': test_ids, 'prediction': predictions})
    submission_df.to_csv(output_filename, index=False)
    
    print(f"\nPredicciones guardadas en: {output_filename}")
    print("Pipeline finalizado exitosamente.")

if __name__ == "__main__":
    run_pipeline()
