
import unittest
import mpmath
from quoremindhp import StatisticalAnalysisHP

class TestStatisticalAnalysisHP(unittest.TestCase):

    def setUp(self):
        self.data = [
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1],
            [7.0, 8.0],
            [7.1, 8.1],
            [9.0, 10.0],
            [9.1, 10.1]
        ]
        self.point = [2.5, 3.5]

    def test_compute_mahalanobis_distance_hp(self):
        distance = StatisticalAnalysisHP.compute_mahalanobis_distance_hp(self.data, self.point)
        self.assertIsInstance(distance, type(mpmath.mpf(1.0)))

if __name__ == '__main__':
    unittest.main()
