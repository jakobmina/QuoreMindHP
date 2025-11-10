
import unittest
import pandas as pd
from src.classes.data_manager import DataManager

class TestDataManager(unittest.TestCase):

    def setUp(self):
        self.data_manager = DataManager()
        self.train_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [6, 7, 8, 9, 10],
            'target': [0, 1, 0, 1, 0]
        })

    def test_preprocess_data(self):
        processed_data = self.data_manager.preprocess_data(self.train_data.copy(), is_train=True)
        self.assertIn('mahalanobis_distance', processed_data.columns)
        self.assertEqual(len(processed_data), 5)

if __name__ == '__main__':
    unittest.main()
