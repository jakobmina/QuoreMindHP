# 🧬 QuoreMindHP v1.0.0 + Tabla H7: Referencia Técnica

Este proyecto integra **QuoreMindHP** para el análisis de alta precisión de los datos cuánticos entrelazados de la **Tabla H7**. La integración ya está implementada en el núcleo del pipeline (`config.py`, `data_manager.py`, `model_trainer.py`).

## 📊 Arquitectura de Precisión

Tabla H7 requiere precisión arbitraria debido a la naturaleza de sus datos:

- ✅ **Precisión arbitraria**: 100+ dígitos para evitar pérdida de coherencia.
- ✅ **Cálculos de Mahalanobis estables**: Manejo de matrices de covarianza altamente correlacionadas.
- ✅ **Confianza Bayesiana**: Métricas basadas en entrelazamiento y Fase Berry.

---

## ⚙️ Configuración Actual (`config.py`)

La integración se gestiona mediante las siguientes constantes:

```python
QUOREMINDHP_ENABLED = True         # Habilitar integración
QUANTUM_PRECISION_DPS = 100        # Precisión recomendada (dps)
MAHALANOBIS_METHOD = "quoremindhp"  # Motor de cálculo
```

---

## 🛠️ Componentes de Integración (`quoremindhp_integration.py`)

### 1. MahalanobisHP

Maneja el cálculo de distancias estadísticas con precisión arbitraria.

- **Uso en Pipeline**: Se ejecuta automáticamente en `DataManager.preprocess_data()`.
- **Beneficio**: Determinantes exactos e inversiones de matriz estables incluso con singularidad en float64.

### 2. BayesianAnalysisH7

Calcula la probabilidad posterior de las predicciones.

- **Coherencia**: Basada en el *Estado 2-1* (0.9 para entrelazado, 0.3 para estándar).
- **Incertidumbre**: Derivada de la *Fase Berry (rad)*.

### 3. ShannonEntropyHP

Cálculo exacto de entropía de información en bits.

---

## 📈 Comparativa de Desempeño

```text
┌────────────────────────┬─────────────┬───────────────┐
│ MÉTRICA                │ FLOAT64     │ QUOREMINDHP   │
├────────────────────────┼─────────────┼───────────────┤
│ Precisión Decimal      │ 15-17       │ 100+          │
│ Error Mahalanobis      │ ~1e-14      │ ~1e-100       │
│ Coherencia Cuántica    │ Perdida     │ Preservada    │
│ Estabilidad Matricial  │ Baja        │ Crítica/Alta  │
└────────────────────────┴─────────────┴───────────────┘
```

---

## 🚀 Guía de Uso Rápido

### Verificación de Precisión

Para comparar la precisión HP vs NumPy en un punto específico:

```python
from quoremindhp_integration import compare_mahalanobis_precision

res = compare_mahalanobis_precision(data, point, precision_dps=100)
print(f"Diferencia detectada: {res['difference']}")
```

### Acceso a Confianza Bayesiana

Las predicciones del `ModelTrainer` ahora incluyen este campo:

```python
trainer = ModelTrainer()
results = trainer.predict(test_data)
# results['bayesian_confidence'] contiene los valores mpf
```

---

## 📖 Referencias

- **mpmath**: Motor de aritmética de precisión arbitraria.
- **El Mandato Metriplético**: Los cálculos respetan la dualidad conservativa/disipativa mediante el ajuste de incertidumbre geométrica.

---
**Documentación actualizada v1.0.0** 🧬
