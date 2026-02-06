import pytest
import mpmath
from quoremindhp_integration import MahalanobisHP, BayesianAnalysisH7, ShannonEntropyHP

def test_mahalanobis_hp_precision():
    """Verifica que MahalanobisHP mantenga la precisión de 100 dígitos."""
    mahal = MahalanobisHP(precision_dps=100)
    data = [
        [1.0, 2.0],
        [1.1, 2.1],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    point = [2.5, 3.5]
    
    mean_vec, inv_cov = mahal.precompute_components(data)
    result = mahal.calculate_for_point(point, mean_vec, inv_cov)
    
    # Verificar que el resultado sea mpf y tenga alta precisión
    assert isinstance(result.distance, type(mpmath.mpf(0)))
    assert result.precision_digits == 100
    assert result.distance > 0

def test_bayesian_analysis_h7():
    """Verifica los cálculos del análisis Bayesiano."""
    bayes = BayesianAnalysisH7(precision_dps=100)
    
    # Coherencia
    coh_high = bayes.calculate_entanglement_coherence('(0, 1)')
    coh_low = bayes.calculate_entanglement_coherence('(0, 0)') # Not strictly in H7 but for test
    
    assert float(coh_high) == 0.9
    assert float(coh_low) == 0.3
    
    # Incertidumbre
    # Fase Berry de pi debería dar incertidumbre 0
    unc_0 = bayes.calculate_measurement_uncertainty(3.141592653589793)
    assert float(unc_0) < 1e-10
    
    # Probabilidad de clase
    prob = bayes.calculate_probability_class('001', coh_high, unc_0)
    assert 0 < float(prob) < 1

def test_shannon_entropy_hp():
    """Verifica el cálculo de entropía de Shannon."""
    data = [1, 2, 2, 3, 3, 3] # Distribución desigual
    entropy = ShannonEntropyHP.calculate(data, precision_dps=100)
    
    assert isinstance(entropy, type(mpmath.mpf(0)))
    assert float(entropy) > 1.0 # Entropía de 3 valores con distribución desigual es > 1 bit
