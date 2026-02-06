"""
quoremindhp_integration.py

Integración especializada de QuoreMindHP v1.0.0 con el pipeline ML de Tabla H7.

Este módulo proporciona:
  1. Wrapper para usar StatisticalAnalysisHP en data_manager.py
  2. Cálculo de Mahalanobis de ALTA PRECISIÓN para datos cuánticos
  3. Análisis Bayesiano HP para predicciones confiables
  4. Métricas de entropía de Shannon con precisión arbitraria

Autor: Claude (Anthropic)
Fecha: 6 febrero 2025
Versión: 1.0.0
"""

import mpmath
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import numpy as np

# Importar desde QuoreMindHP (asumiendo que está en el mismo proyecto)
# Si QuoreMindHP está instalado como paquete: from quoremindhp import StatisticalAnalysisHP, BayesLogicHP
# Para este ejemplo, asumimos importación directa del módulo

# ============================================================================
# CONFIGURACIÓN GLOBAL DE PRECISIÓN
# ============================================================================

# Precisión para cálculos cuánticos de Tabla H7
# Los datos cuánticos requieren alta precisión para mantener coherencia
QUANTUM_PRECISION_DPS = 100  # 100 dígitos decimales para datos cuánticos
mpmath.mp.dps = QUANTUM_PRECISION_DPS

print(f"✓ QuoreMindHP Integration: Precisión cuántica = {QUANTUM_PRECISION_DPS} dígitos")

# ============================================================================
# WRAPPER: MAHALANOBIS DE ALTA PRECISIÓN PARA TABLA H7
# ============================================================================

@dataclass
class MahalanobisResultHP:
    """Resultado detallado del cálculo de Mahalanobis de alta precisión."""
    distance: mpmath.mpf
    distance_float: float
    precision_digits: int
    mean_vector: Any  # mpmath.matrix
    inv_cov_matrix: Any  # mpmath.matrix
    
    def __str__(self):
        distance_str = mpmath.nstr(self.distance, n=self.precision_digits)
        return f"MahalanobisHP(distance={distance_str}, float≈{self.distance_float:.15f})"


class MahalanobisHP:
    """
    Cálculo de distancia de Mahalanobis usando QuoreMindHP StatisticalAnalysisHP.
    
    Especializado para datos cuánticos de Tabla H7 donde la precisión es crítica.
    """
    
    def __init__(self, precision_dps: int = QUANTUM_PRECISION_DPS):
        """
        Inicializa con precisión configurada.
        
        Args:
            precision_dps: Dígitos decimales de precisión (default: 100 para datos cuánticos)
        """
        self.precision_dps = precision_dps
        self.original_dps = mpmath.mp.dps
        mpmath.mp.dps = precision_dps
        
    def precompute_components(self, data: List[List[Union[float, str]]]) -> Tuple[Any, Any]:
        """
        Precomputa componentes para cálculos rápidos posteriores.
        
        Usa StatisticalAnalysisHP.precompute_mahalanobis_components()
        
        Args:
            data: Lista de puntos de datos [[x1, y1], [x2, y2], ...]
            
        Returns:
            (mean_vector, inv_cov_matrix) para usar en calculate_for_point()
        """
        try:
            # Importar aquí para flexibilidad
            from quoremindhp import StatisticalAnalysisHP
            mean_vector, inv_cov_matrix = StatisticalAnalysisHP.precompute_mahalanobis_components(data)
            return mean_vector, inv_cov_matrix
        except ImportError:
            # Fallback: implementación manual si StatisticalAnalysisHP no está disponible
            return self._manual_precompute(data)
    
    def _manual_precompute(self, data: List[List[Union[float, str]]]) -> Tuple[Any, Any]:
        """
        Implementación manual de precomputación usando mpmath directamente.
        """
        try:
            data_mp_list = [[mpmath.mpf(str(elem)) for elem in row] for row in data]
            data_mp = mpmath.matrix(data_mp_list)
        except Exception as e:
            raise ValueError(f"Error al convertir datos a mpmath.matrix: {e}")
        
        n_rows, n_cols = data_mp.rows, data_mp.cols
        
        # Calcular media
        mean_vector = mpmath.matrix(1, n_cols)
        for j in range(n_cols):
            col_values = [data_mp[i, j] for i in range(n_rows)]
            mean_vector[0, j] = mpmath.fsum(col_values) / n_rows
        
        # Calcular covarianza
        cov_matrix = mpmath.zeros(n_cols)
        for i in range(n_cols):
            for j in range(i, n_cols):
                products = []
                for k in range(n_rows):
                    products.append((data_mp[k, i] - mean_vector[0, i]) * 
                                   (data_mp[k, j] - mean_vector[0, j]))
                denom = mpmath.mpf(n_rows - 1) if n_rows > 1 else mpmath.mpf(1)
                cov_ij = mpmath.fsum(products) / denom
                cov_matrix[i, j] = cov_ij
                if i != j:
                    cov_matrix[j, i] = cov_ij
        
        # Invertir covarianza con regularización si es necesario
        try:
            inv_cov_matrix = mpmath.inverse(cov_matrix)
        except (ZeroDivisionError, ValueError):
            # Regularizar: agregar pequeño valor a diagonal
            identity = mpmath.eye(n_cols)
            cov_matrix += identity * mpmath.mpf('1e-30')
            try:
                inv_cov_matrix = mpmath.inverse(cov_matrix)
            except:
                raise ValueError("Matriz de covarianza singular incluso después de regularización")
        
        return mean_vector, inv_cov_matrix
    
    def calculate_for_point(self, point: List[Union[float, str]], 
                           mean_vector: Any, 
                           inv_cov_matrix: Any) -> MahalanobisResultHP:
        """
        Calcula distancia de Mahalanobis para un punto usando componentes precomputados.
        
        Args:
            point: Punto de datos [x1, x2, ...]
            mean_vector: Vector de medias precomputado
            inv_cov_matrix: Inversa de covarianza precomputada
            
        Returns:
            MahalanobisResultHP con resultado de alta precisión
        """
        try:
            point_mp = mpmath.matrix([[mpmath.mpf(str(p)) for p in point]])
        except Exception as e:
            raise ValueError(f"Error al convertir punto a mpmath.matrix: {e}")
        
        # Diferencia
        diff = point_mp - mean_vector
        
        # Distancia de Mahalanobis: sqrt((x-μ)' * Σ⁻¹ * (x-μ))
        mahal_sq = diff * inv_cov_matrix * diff.T
        mahal_sq_val = mahal_sq[0, 0]
        
        # Asegurar no-negativo
        if mahal_sq_val < 0:
            mahal_sq_val = mpmath.mpf("0")

        
        distance_hp = mpmath.sqrt(mahal_sq_val)
        distance_float = float(distance_hp)
        
        return MahalanobisResultHP(
            distance=distance_hp,
            distance_float=distance_float,
            precision_digits=self.precision_dps,
            mean_vector=mean_vector,
            inv_cov_matrix=inv_cov_matrix
        )
    
    def __del__(self):
        """Restaurar precisión original al limpiar."""
        mpmath.mp.dps = self.original_dps


# ============================================================================
# ANÁLISIS BAYESIANO HP PARA TABLA H7
# ============================================================================

class BayesianAnalysisH7:
    """
    Análisis Bayesiano especializado para Tabla H7 usando precisión arbitraria.
    
    Integra:
    - Coherencia cuántica (entrelazamiento)
    - Simetría de estados (Estado 2-1)
    - Incertidumbre de medición
    """
    
    def __init__(self, precision_dps: int = QUANTUM_PRECISION_DPS):
        """
        Args:
            precision_dps: Precisión para cálculos Bayesianos
        """
        self.precision_dps = precision_dps
        self.original_dps = mpmath.mp.dps
        mpmath.mp.dps = precision_dps
    
    def calculate_entanglement_coherence(self, estado_2_1: str) -> mpmath.mpf:
        """
        Calcula coherencia de entrelazamiento basada en Estado 2-1.
        
        States (0,1) y (1,0) representan máximo entrelazamiento.
        
        Args:
            estado_2_1: String '(0, 1)' o '(1, 0)'
            
        Returns:
            Coherencia como mpmath.mpf en [0, 1]
        """
        # Estados entrelazados tienen coherencia máxima
        if estado_2_1 in ['(0, 1)', '(1, 0)']:
            return mpmath.mpf("0.9")  # Entrelazamiento fuerte
        else:
            return mpmath.mpf("0.3")  # No entrelazado
    
    def calculate_measurement_uncertainty(self, fase_berry: float) -> mpmath.mpf:
        """
        Calcula incertidumbre de medición desde Fase Berry.
        
        Fase Berry refleja acumulación geométrica en mecánica cuántica.
        
        Args:
            fase_berry: Valor de Fase Berry (rad)
            
        Returns:
            Incertidumbre normalizada como mpmath.mpf
        """
        fase_mp = mpmath.mpf(str(fase_berry))
        
        # Normalizar fase a [0, 2π]
        dos_pi = mpmath.mpf(2) * mpmath.pi
        fase_norm = mpmath.fmod(fase_mp, dos_pi)
        
        pi = mpmath.pi
        uncertainty = mpmath.fabs(fase_norm - pi) / pi
        
        # Usar min/max nativos
        return mpmath.mpf(max(0.0, min(1.0, float(uncertainty))))

    
    def calculate_probability_class(self, target: str, 
                                    entanglement: mpmath.mpf,
                                    uncertainty: mpmath.mpf) -> mpmath.mpf:
        """
        Calcula probabilidad Bayesiana de una clase objetivo.
        
        P(Class|Evidence) ∝ P(Evidence|Class) * P(Class)
        
        Args:
            target: Clase '001'-'110'
            entanglement: Coherencia de entrelazamiento [0,1]
            uncertainty: Incertidumbre de medición [0,1]
            
        Returns:
            Probabilidad posterior como mpmath.mpf
        """
        one = mpmath.mpf(1)
        
        # Prior: cada clase tiene probabilidad inicial
        prior = mpmath.mpf(1) / mpmath.mpf(6)  # 6 clases en H7
        
        # Likelihood: clases con más bits 1 tienen mayor probabilidad con coherencia alta
        num_ones = target.count('1')
        likelihood_base = mpmath.mpf(num_ones) / mpmath.mpf(3)
        
        # Modular por entrelazamiento y incertidumbre
        likelihood = likelihood_base * (mpmath.mpf(0.5) + mpmath.mpf(0.5) * entanglement) * \
                     (one - mpmath.mpf(0.3) * uncertainty)
        
        # Probabilidad posterior (sin normalización completa, suficiente para ranking)
        posterior = prior * likelihood
        
        return mpmath.mpf(min(1.0, float(posterior)))

    
    def __del__(self):
        """Restaurar precisión original."""
        mpmath.mp.dps = self.original_dps


# ============================================================================
# SHANNON ENTROPY DE ALTA PRECISIÓN
# ============================================================================

class ShannonEntropyHP:
    """Cálculo de entropía de Shannon con precisión arbitraria."""
    
    @staticmethod
    def calculate(data: List[Any], precision_dps: int = QUANTUM_PRECISION_DPS) -> mpmath.mpf:
        """
        Calcula entropía de Shannon con alta precisión.
        
        Args:
            data: Lista de valores
            precision_dps: Dígitos decimales de precisión
            
        Returns:
            Entropía como mpmath.mpf (en bits)
        """
        original_dps = mpmath.mp.dps
        mpmath.mp.dps = precision_dps
        
        if not data:
            mpmath.mp.dps = original_dps
            return mpmath.mpf(0)
        
        # Contar ocurrencias
        values, counts = np.unique(data, return_counts=True)
        n = len(data)
        
        # Calcular entropía
        entropy = mpmath.mpf(0)
        for count in counts:
            if count > 0:
                p = mpmath.mpf(str(count)) / mpmath.mpf(str(n))
                entropy -= p * mpmath.log(p, 2)

        
        mpmath.mp.dps = original_dps
        return entropy


# ============================================================================
# FUNCTION: COMPARACIÓN HP vs ESTÁNDAR
# ============================================================================

def compare_mahalanobis_precision(data: List[List[float]], 
                                  point: List[float],
                                  precision_dps: int = 100) -> Dict[str, Any]:
    """
    Compara cálculo de Mahalanobis de alta precisión vs estándar.
    
    Útil para validar que QuoreMindHP mejora la precisión en datos cuánticos.
    
    Args:
        data: Datos de entrenamiento
        point: Punto a evaluar
        precision_dps: Precisión para cálculo HP
        
    Returns:
        Diccionario con resultados de ambos métodos
    """
    # Método 1: Alta precisión
    mahal_hp = MahalanobisHP(precision_dps=precision_dps)
    mean_vec, inv_cov = mahal_hp.precompute_components(
        [[str(x) for x in row] for row in data]
    )
    result_hp = mahal_hp.calculate_for_point(
        [str(x) for x in point],
        mean_vec,
        inv_cov
    )
    
    # Método 2: Estándar NumPy (para comparación)
    try:
        from sklearn.covariance import EmpiricalCovariance
        from scipy.spatial.distance import mahalanobis
        
        data_np = np.array(data, dtype=np.float64)
        point_np = np.array(point, dtype=np.float64)
        
        cov = EmpiricalCovariance().fit(data_np).covariance_
        try:
            inv_cov_np = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov_np = np.linalg.pinv(cov)
        
        mean_np = np.mean(data_np, axis=0)
        distance_np = mahalanobis(point_np, mean_np, inv_cov_np)
        
        return {
            'hp_distance': float(result_hp.distance),
            'np_distance': float(distance_np),
            'difference': float(mpmath.fabs(result_hp.distance - mpmath.mpf(str(distance_np)))),
            'hp_precision': precision_dps,
            'hp_result': result_hp
        }
    except ImportError:
        return {
            'hp_distance': float(result_hp.distance),
            'hp_precision': precision_dps,
            'hp_result': result_hp,
            'numpy_comparison': 'Not available'
        }


# ============================================================================
# EJEMPLO DE USO INTEGRADO
# ============================================================================

def example_tabla_h7_analysis():
    """
    Ejemplo: Análisis de Tabla H7 usando QuoreMindHP.
    """
    print("\n" + "="*80)
    print("EJEMPLO: ANÁLISIS DE TABLA H7 CON QUOREMINDHP")
    print("="*80)
    
    # Datos de ejemplo de Tabla H7
    data_h7_train = [
        [1.0, 6.0, 0.4783, 1.2650],    # 001
        [2.0, 5.0, 0.4609, 2.1626],    # 010
        [3.0, 4.0, 0.4513, 3.0602],    # 011
        [4.0, 3.0, 0.4513, 3.2230],    # 100
        [5.0, 2.0, 0.4609, 4.1206],    # 101
        [6.0, 1.0, 0.4783, 5.0182],    # 110
    ]
    
    point_test = [2.5, 4.5, 0.4650, 2.7128]
    
    # 1. MAHALANOBIS DE ALTA PRECISIÓN
    print("\n1. DISTANCIA DE MAHALANOBIS (ALTA PRECISIÓN)")
    print("-" * 80)
    
    comparison = compare_mahalanobis_precision(data_h7_train, point_test, precision_dps=100)
    print(f"   HP Distance (100 dígitos):  {comparison['hp_distance']:.15f}")
    if 'np_distance' in comparison:
        print(f"   NumPy Distance (float64):   {comparison['np_distance']:.15f}")
        print(f"   Diferencia:                 {comparison['difference']:.2e}")
    
    # 2. ANÁLISIS BAYESIANO PARA TABLA H7
    print("\n2. ANÁLISIS BAYESIANO H7")
    print("-" * 80)
    
    bayes = BayesianAnalysisH7(precision_dps=100)
    
    # Caso 1: Estado entrelazado (0, 1)
    coherence = bayes.calculate_entanglement_coherence('(0, 1)')
    uncertainty = bayes.calculate_measurement_uncertainty(2.7128)
    
    print(f"   Estado 2-1: (0, 1)")
    print(f"   Coherencia de entrelazamiento: {mpmath.nstr(coherence, n=10)}")
    print(f"   Incertidumbre de medición:     {mpmath.nstr(uncertainty, n=10)}")
    
    # Probabilidades para cada clase
    print(f"\n   Probabilidades Bayesianas por clase:")
    for target_class in ['001', '010', '011', '100', '101', '110']:
        prob = bayes.calculate_probability_class(target_class, coherence, uncertainty)
        print(f"      P({target_class}|Evidence) = {mpmath.nstr(prob, n=10)}")
    
    # 3. ENTROPÍA DE SHANNON
    print("\n3. ENTROPÍA DE SHANNON (ALTA PRECISIÓN)")
    print("-" * 80)
    
    data_entropy = [1, 2, 2, 3, 3, 3, 1, 2, 1, 3]
    entropy = ShannonEntropyHP.calculate(data_entropy, precision_dps=100)
    print(f"   Datos: {data_entropy}")
    print(f"   Entropía: {mpmath.nstr(entropy, n=20)} bits")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    example_tabla_h7_analysis()
