---
title: "Métricas de regresión"
knowledge_area: "metrics"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Métricas de regresión

## 1. Propósito y alcance

Este documento ayuda a elegir e interpretar métricas para problemas de regresión según el objetivo de evaluación y el tipo de error que interesa controlar. Una respuesta cuantitativa activa una tarea de regresión; antes de medirla, debe confirmarse que una salida numérica sea realmente lo que necesita el consumidor de la predicción. **Explícito de las fuentes:** la formulación y la métrica dependen del objetivo, y la evaluación supervisada compara predicciones con valores observados.

La métrica no reemplaza la formulación del problema ni determina por sí sola cómo obtener una estimación válida de generalización. La estrategia de partición, validación y reserva del conjunto final se trata en `knowledge/validation/validation_strategy.md`. Aquí solo se explican las cantidades que se calculan una vez definidos los datos de evaluación.

## 2. Qué significa evaluar un modelo de regresión

Para cada observación (i), el modelo produce una predicción \(\hat y_i\) que se compara con el valor observado \(y_i\). La diferencia por observación conserva información local; una métrica agrega las diferencias de varias observaciones en una cifra. MAE, MSE y RMSE agregan magnitudes de error de formas distintas; \(R^2\) compara la variabilidad no explicada con la variabilidad total.

**Explícito de las fuentes:** el objetivo predictivo es obtener \(\hat Y=\hat f(X)\) con poco error, pero siempre puede quedar un componente irreducible asociado a variación no explicable por los predictores observados. El desempeño relevante para generalización se estima sobre observaciones no utilizadas para ajustar el modelo, no mediante el error de entrenamiento.

**Síntesis integrada:** evaluar regresión requiere mantener separadas tres capas:

1. la predicción individual \(\hat y_i\);
2. el error o residuo de esa observación;
3. el resumen agregado elegido según el objetivo.

Una cifra agregada necesita contexto. Un MAE de 1 200, por ejemplo, no dice por sí solo si el modelo aporta valor; debe compararse con la escala del objetivo, un baseline comprensible y, cuando sea posible, la utilidad de la mejora.

## 3. Errores y residuos

En los resúmenes consultados, un **residuo** de un modelo ajustado se define como

\[
e_i = y_i-\hat y_i,
\]

donde \(y_i\) es el valor observado, \(\hat y_i\) la predicción ajustada y \(e_i\) el residuo de la observación \(i\). Con esta convención, un residuo positivo significa que la predicción quedó por debajo del valor observado; uno negativo, que quedó por encima. La magnitud \(|e_i|\) expresa la distancia entre ambos en las unidades del objetivo. **Síntesis integrada:** la interpretación del signo y de la magnitud se deduce directamente de la fórmula respaldada.

Los residuos de entrenamiento sirven para diagnosticar el ajuste: deberían quedar aproximadamente centrados en cero y sin patrón sistemático; curvatura puede sugerir no linealidad, forma de embudo heteroscedasticidad y residuos consecutivos parecidos dependencia. Sin embargo, esta inspección no estima por sí sola el error en datos nuevos.

**Explícito de las fuentes:** el MSE de entrenamiento usa observaciones empleadas en el ajuste, mientras que el MSE de prueba se refiere a observaciones no usadas para ajustar. Elegir un modelo por el menor error de entrenamiento favorece métodos demasiado flexibles. Por tanto, “residuo de ajuste” y “error de generalización estimado” no son intercambiables.

## 4. Métricas principales

Todas las métricas de error incluidas aquí siguen la convención “menor es mejor”. Las fórmulas usan \(n\) observaciones evaluadas, \(y_i\) como valor observado y \(\hat y_i\) como predicción.

### 4.1 MAE

| Métrica | Qué resume | Fórmula respaldada | Interpretación | Limitación |
|---|---|---|---|---|
| Error absoluto medio (MAE) | Magnitud absoluta media de los errores | \(\displaystyle \operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|\) | Queda en las unidades del objetivo; representa una distancia absoluta media y un valor menor es mejor. | Enfatiza menos los fallos grandes que RMSE; puede no representar bien un objetivo donde esos fallos sean especialmente costosos. |

Símbolos: \(n\) es el número de observaciones evaluadas; \(i\) identifica una observación; \(y_i\) es el valor observado; \(\hat y_i\) es la predicción; \(|y_i-\hat y_i|\) es el error absoluto. **Explícito de las fuentes:** MAE es menos sensible a outliers que RMSE y conserva las unidades del objetivo. Esto no significa que sea universalmente preferible: reduce el énfasis relativo sobre errores grandes.

### 4.2 MSE

| Métrica | Qué resume | Fórmula respaldada | Interpretación | Limitación |
|---|---|---|---|---|
| Error cuadrático medio (MSE) | Promedio de los errores elevados al cuadrado | \(\displaystyle \operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2\) | Un valor menor indica predicciones más cercanas bajo penalización cuadrática. **Síntesis integrada:** su resultado queda en unidades del objetivo al cuadrado. | La elevación al cuadrado da mayor peso a errores grandes y dificulta una lectura directa en las unidades originales. |

Los símbolos tienen el significado definido al inicio de la sección. La fórmula está respaldada explícitamente para MSE de entrenamiento; el resumen de ISLP distingue además el MSE de prueba sobre observaciones no usadas para ajustar. **Síntesis integrada:** la unidad al cuadrado y el mayor peso relativo de errores grandes se derivan de la operación cuadrática y de la relación documentada entre normas y errores grandes.

### 4.3 RMSE

| Métrica | Qué resume | Fórmula respaldada | Interpretación | Limitación |
|---|---|---|---|---|
| Raíz del error cuadrático medio (RMSE) | Raíz del promedio de errores cuadrados | \(\displaystyle \operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}\) | Conserva las unidades del objetivo, penaliza especialmente errores grandes y un valor menor es mejor. | Es sensible a outliers; no conviene elegirla automáticamente cuando unos pocos extremos dominan el resultado. |

Los símbolos tienen el significado definido al inicio de la sección. **Explícito de las fuentes:** RMSE corresponde a una medida de tipo \(\ell_2\), queda en unidades del objetivo y es apropiada cuando interesa penalizar errores grandes y los outliers son raros.

### 4.4 R²

| Métrica | Qué resume | Fórmula respaldada | Interpretación | Limitación |
|---|---|---|---|---|
| Coeficiente de determinación \(R^2\) | Variabilidad no explicada (RSS) respecto de la variabilidad total (TSS) | \(\displaystyle R^2=1-\frac{RSS}{TSS}\), con \(\displaystyle RSS=\sum_i(y_i-\hat y_i)^2\) y \(\displaystyle TSS=\sum_i(y_i-\bar y)^2\) | En los modelos con intercepto descritos por ISLP, mide la fracción de variabilidad explicada en la muestra. Es adimensional. | No mide la magnitud del error en las unidades del objetivo; un valor alto no garantiza especificación correcta ni generalización. En entrenamiento no disminuye al añadir predictores. |

Símbolos: \(RSS\) es la suma de cuadrados de residuos; \(TSS\) es la variabilidad total; \(\bar y\) es la media observada; los demás símbolos mantienen el significado anterior. **Explícito de las fuentes:** para los modelos con intercepto presentados por ISLP, \(R^2\in[0,1]\) en la muestra ajustada. Este alcance no se generaliza aquí a configuraciones que los resúmenes no explican.

\(R^2\) no es un “porcentaje de predicciones correctas”. Puede ser intuitivo como medida relativa de ajuste, pero debe acompañarse de una métrica de magnitud cuando el problema exige interpretar el tamaño de los errores.

### 4.5 Otras métricas respaldadas por las fuentes

| Métrica | Qué resume | Fórmula respaldada | Interpretación | Limitación |
|---|---|---|---|---|
| Error absoluto mediano | La mediana de las magnitudes de error | \(\displaystyle \operatorname{MedAE}=\operatorname{mediana}_i\bigl(|y_i-\hat y_i|\bigr)\) | Conserva las unidades del objetivo y describe mejor el error típico cuando pocos outliers dominan la media. | Puede ocultar errores extremos que sí sean relevantes para el objetivo. |

La fórmula expresa directamente la definición textual del resumen de Molin; los símbolos mantienen el significado anterior. **Explícito de la fuente:** esta medida se propone cuando hay outliers en los residuos y se busca el error típico del grueso de las observaciones.

| Métrica | Qué resume | Fórmula respaldada | Interpretación | Limitación |
|---|---|---|---|---|
| Error estándar residual (RSE) | Escala residual del ajuste, descontando parámetros estimados | \(\displaystyle RSE=\sqrt{\frac{RSS}{n-p-1}}\) | Estima la desviación típica del error y conserva las unidades de la respuesta. | Es una medida de ajuste del modelo lineal descrito, dependiente de \(p\) y de sus supuestos; no sustituye una evaluación fuera de muestra. |

Símbolos: \(p\) es el número de predictores y \(n-p-1\) descuenta los parámetros estimados; \(RSS\), \(y_i\) y \(\hat y_i\) se definieron antes. Se incluye para distinguir una medida de ajuste lineal de una métrica predictiva de evaluación.

## 5. Sensibilidad a escala y valores extremos

MAE, RMSE, error absoluto mediano y RSE conservan las unidades del objetivo; MSE las eleva al cuadrado; \(R^2\) es adimensional por ser una razón de sumas de cuadrados. Por ello, una cifra de MAE o RMSE solo se interpreta conociendo la unidad y escala del objetivo. **Síntesis integrada:** comparar directamente MAE o RMSE de objetivos expresados en escalas distintas no produce una comparación homogénea sin contexto adicional.

RMSE y MSE dan mayor peso relativo a errores grandes por la operación cuadrática. MAE usa penalización lineal y las fuentes la describen como menos sensible a outliers que RMSE. El error absoluto mediano reduce todavía más la influencia de unos pocos errores extremos sobre el “error típico”, pero puede esconder precisamente los casos que más importan.

La distribución del objetivo también importa: los resúmenes piden revisar su distribución, outliers y posibles topes. Un objetivo truncado puede enseñar un límite artificial al modelo. **Conclusión práctica derivada:** antes de elegir una métrica, revisar si los extremos son errores de datos, casos válidos poco frecuentes o eventos cuyo coste exige enfatizarlos.

## 6. Baselines para regresión

Un baseline aporta el “respecto a qué” que una métrica aislada no ofrece. Si existe un proceso operativo, su rendimiento es la referencia prioritaria. Si no existe, las fuentes respaldan una heurística simple y explicable:

- una predicción constante basada en la media o mediana del objetivo;
- mediana cuando hay muchos outliers;
- una regla lineal basada en una variable dominante y aceptada por el negocio;
- una tabla de promedio del objetivo por una o dos variables clave;
- para forecasting, persistencia, tendencia lineal o el valor de la misma temporada anterior, cuando la temporalidad forma parte de la tarea.

Los parámetros del baseline se calculan solo con entrenamiento y se evalúan con la misma métrica y las mismas observaciones que el modelo. Una referencia débil fija un suelo, pero superarla no demuestra por sí sola utilidad. Ningún baseline de la lista es apropiado para todos los problemas.

## 7. Cómo elegir una métrica

**Síntesis integrada a partir de las reglas explícitas de las fuentes.**

| Objetivo o característica del problema | Métrica que puede considerarse | Motivo | Precaución |
|---|---|---|---|
| Interpretar la magnitud media en unidades del objetivo | MAE | Promedia distancias absolutas y conserva unidades | Enfatiza menos los errores grandes |
| Penalizar especialmente errores grandes | RMSE o MSE | La penalización cuadrática aumenta su peso | Son sensibles a outliers; MSE queda en unidades al cuadrado |
| Mantener penalización cuadrática con lectura en unidades originales | RMSE | Aplica raíz al promedio cuadrático | No asumir que es adecuada si extremos dominan sin corresponder al objetivo |
| Describir el error típico cuando pocos outliers dominan la media | Error absoluto mediano | Usa la mediana de los errores absolutos | Puede ocultar extremos relevantes |
| Comparar variabilidad explicada frente a una referencia centrada en la media | \(R^2\) | Resume RSS respecto de TSS | No informa magnitud del error ni es porcentaje de aciertos |
| Diagnosticar escala residual de un modelo lineal ajustado | RSE | Conserva unidades y descuenta parámetros | No sustituye una evaluación fuera de muestra |
| Competición con métrica oficial | La métrica oficial, si representa el objetivo planteado | Alinea selección local con el criterio de evaluación | Documentar si la métrica oficial no refleja todo el coste del problema |
| Varios objetivos o segmentos relevantes | Métrica principal más resultados complementarios o estratificados | Un agregado puede ocultar comportamiento desigual | No cambiar de métrica o split entre modelos sin registrarlo |

La elección de métrica y la estrategia de validación son decisiones distintas: la primera define qué error resumir; la segunda define sobre qué observaciones y particiones estimarlo. La validación debe preservar la unidad real de generalización y mantener comparables las evaluaciones.

## 8. Implementación mínima en Python

El siguiente ejemplo calcula únicamente métricas respaldadas y distingue valores observados de predicciones. No presupone un dataset ni incluye entrenamiento.

```python
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

y_observado = np.asarray([...], dtype=float)
y_predicho = np.asarray([...], dtype=float)

resultados = {
    "mae": mean_absolute_error(y_observado, y_predicho),
    "mse": mean_squared_error(y_observado, y_predicho),
    "rmse": root_mean_squared_error(y_observado, y_predicho),
    "r2": r2_score(y_observado, y_predicho),
    "error_absoluto_mediano": median_absolute_error(
        y_observado, y_predicho
    ),
}
```

Nota práctica: adaptación moderna de la idea descrita en la fuente. El resumen de Géron identifica `root_mean_squared_error` como API moderna; la forma respaldada y compatible conceptualmente es `mean_squared_error(y_observado, y_predicho) ** 0.5`.

Baseline constante respaldado para regresión con outliers:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

valor_baseline = np.median(y_entrenamiento)
prediccion_baseline = np.full(len(y_validacion), valor_baseline)

mae_baseline = mean_absolute_error(y_validacion, prediccion_baseline)
mae_modelo = mean_absolute_error(y_validacion, y_predicho_validacion)
```

La mediana se estima con entrenamiento; baseline y modelo se miden sobre la misma validación.

## 9. Errores frecuentes y riesgos de interpretación

- Interpretar MAE o RMSE sin conocer las unidades, escala y distribución del objetivo.
- Comparar valores calculados sobre escalas distintas como si fueran directamente equivalentes. **Síntesis práctica.**
- Ignorar que MSE/RMSE aumentan el peso de errores grandes o, en el extremo contrario, usar una mediana que oculta fallos costosos.
- Reportar una única cifra agregada cuando existen varios objetivos o segmentos relevantes; un promedio puede ocultar grupos débiles. **Síntesis integrada.**
- Interpretar \(R^2\) como porcentaje de predicciones correctas, magnitud del error o garantía de buena especificación.
- Añadir predictores para elevar \(R^2\) de entrenamiento; en los modelos descritos no disminuye al agregar columnas.
- Evaluar con las observaciones usadas para ajustar el modelo y presentar ese resultado como generalización.
- Elegir el modelo por el menor error de entrenamiento.
- Ajustar estadísticas del baseline, imputaciones o escalas con validación o test.
- Comparar modelos en particiones diferentes: el ranking puede depender más del split que del modelo.
- Confundir una inspección de residuos de entrenamiento con evidencia fuera de muestra.
- Ignorar patrones de residuos: una métrica agregada no descarta no linealidad, heteroscedasticidad o dependencia.
- Usar `neg_mean_squared_error` sin invertir el signo antes de aplicar la raíz.
- Reoptimizar después de mirar el test final; destruye su independencia.

## 10. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Confirmar que la salida sea cuantitativa y leer la métrica oficial antes de modelar.
2. Revisar unidad, escala, distribución, outliers y posibles topes del objetivo.
3. Decidir qué error importa: magnitud media, énfasis en fallos grandes, error típico o variabilidad explicada.
4. Construir un baseline simple con parámetros aprendidos solo de entrenamiento.
5. Evaluar baseline y candidatos con la misma métrica, las mismas observaciones y una partición estable.
6. Interpretar la métrica junto con la distribución o patrones de los errores; no depender de una única cifra si el objetivo tiene varias dimensiones.
7. Mantener intacto el conjunto final y usar la estrategia apropiada descrita en `knowledge/validation/validation_strategy.md`.
8. Registrar métrica, fórmula o función, dirección de mejora, unidades, split, baseline y justificación.

En una competición, la métrica oficial gobierna el ranking; aun así, una métrica complementaria respaldada puede ayudar a entender por qué un cambio mejora o empeora. Bajo presión, el baseline permite detectar rápidamente pipelines rotos, validaciones dudosas o modelos que no superan una estadística elemental.

## 11. Checklist operativo

- [ ] Confirmé que el problema es de regresión y que el objetivo es cuantitativo.
- [ ] Revisé unidades, escala, distribución, outliers y topes del objetivo.
- [ ] Identifiqué qué tipo de error debe representar la evaluación.
- [ ] Confirmé la métrica exigida por el problema o la competición.
- [ ] Documenté fórmula, símbolos, unidades, dirección de mejora y limitaciones.
- [ ] Construí un baseline respaldado y estimé sus parámetros solo con entrenamiento.
- [ ] Calculé baseline y modelos sobre las mismas observaciones y con la misma métrica.
- [ ] Revisé residuos o errores además de la cifra agregada.
- [ ] No interpreté \(R^2\) como porcentaje de aciertos ni como magnitud del error.
- [ ] Usé una estrategia de validación apropiada sin tocar el test final.
- [ ] Conservé una partición comparable entre alternativas.
- [ ] Registré cualquier objetivo, segmento o coste no capturado por la métrica principal.

## 12. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con las métricas de regresión |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Define el objetivo y la métrica inicial coherente con el problema |
| `knowledge/eda/dataset_audit.md` | Aporta información sobre escala, distribución y valores del objetivo |
| `knowledge/validation/validation_strategy.md` | Define cómo obtener una evaluación válida |
| `knowledge/preprocessing/data_leakage.md` | Advierte sobre evaluaciones optimistas por leakage |
| `knowledge/models/tabular_models.md` | Describe los modelos que se comparan |
| `knowledge/tuning/hyperparameter_tuning.md` | Utiliza una métrica para comparar configuraciones |

## 13. Vacíos y limitaciones de las fuentes

- Los resúmenes consultados no ofrecen cobertura suficiente y verificable de MAPE, sMAPE, RMSLE ni otras métricas porcentuales o relativas; no se añaden definiciones ni fórmulas externas.
- Molin advierte que el error cuadrático logarítmico no admite valores o predicciones negativas, pero el resumen no aporta una fórmula ni guía suficiente para incorporarlo como métrica operativa.
- No se encontró una guía completa para elegir métricas ante ceros, objetivos negativos, colas muy pesadas o costes asimétricos definidos por negocio.
- Las fuentes respaldan MAE como menos sensible a outliers que RMSE y el error absoluto mediano para describir el grueso, pero no fijan umbrales universales de robustez.
- La interpretación documentada de \(R^2\) está acotada a los modelos y condiciones descritos por los resúmenes; no se extrapola a casos no cubiertos.
- RSE se respalda como medida de ajuste lineal, no como sustituto general de validación predictiva.
- La elección final depende del objetivo, del coste real de los errores y, en competiciones, de la métrica oficial.
- La utilidad de una métrica y de un baseline debe comprobarse experimentalmente con una validación adecuada; este documento no prescribe el splitter concreto.

## 14. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning* | [Resumen](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Fórmulas de MAE y RMSE, unidades, sensibilidad a errores grandes, elección según outliers, evaluación y código moderno |
| Müller y Guido, *Introduction to Machine Learning with Python* | [Resumen](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | Separación entre métrica y validación, generalización, comparabilidad y criterio de métrica alineado con uso futuro |
| James et al., *Introduction to Statistical Learning with Python* | [Resumen](../../sources/islp_python/ch02_statistical_learning.md) | Predicción, error reducible/irreducible, MSE de entrenamiento y prueba, riesgo de seleccionar por entrenamiento |
| James et al., *Introduction to Statistical Learning with Python* | [Resumen](../../sources/islp_python/ch03_linear_regression.md) | Residuo, RSS, RSE, \(R^2\), diagnósticos y límites de interpretación del ajuste |
| Molin, *Hands-On Data Analysis with Pandas* | [Resumen](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md) | MAE, RMSE, \(R^2\), error absoluto mediano, residuos, distribución del objetivo e implementación mínima |
| Lakshmanan et al., *Machine Learning Design Patterns* | [Resumen](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) | Particiones repetibles y necesidad de comparar modelos sobre una evaluación estable |
| Lakshmanan et al., *Machine Learning Design Patterns* | [Resumen](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md) | Heuristic Benchmark, baselines de media/mediana, evaluación comparable y utilidad de la mejora |

## 15. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Géron ch. 2; Molin ch. 9 | Síntesis integrada |
| 2. Predicción, error y agregación | ISLP ch. 2; Molin ch. 9; Patterns ch. 7 | Síntesis integrada |
| 2. Error reducible e irreducible | ISLP ch. 2 | Explícito de la fuente |
| 3. Definición de residuo | ISLP ch. 3; Molin ch. 9 | Explícito de la fuente |
| 3. Signo y magnitud del residuo | Fórmula de ISLP ch. 3 | Síntesis integrada |
| 3. Residuo de ajuste frente a error en datos nuevos | ISLP ch. 2–3 | Síntesis integrada |
| 4.1. Fórmula, unidades e interpretación de MAE | Géron ch. 2; Molin ch. 9 | Explícito de la fuente |
| 4.2. Fórmula y criterio de MSE | ISLP ch. 2 | Explícito de la fuente |
| 4.2. Unidades cuadradas y énfasis en errores grandes | Fórmula de ISLP ch. 2; Géron ch. 2 | Síntesis integrada |
| 4.3. Fórmula, unidades y sensibilidad de RMSE | Géron ch. 2; Molin ch. 9 | Explícito de la fuente |
| 4.4. Fórmula e interpretación acotada de \(R^2\) | ISLP ch. 3; Molin ch. 9 | Explícito de la fuente |
| 4.4. \(R^2\) no es porcentaje de aciertos | Definición y límites de ISLP ch. 3 y Molin ch. 9 | Conclusión práctica derivada |
| 4.5. Error absoluto mediano | Molin ch. 9 | Explícito de la fuente |
| 4.5. RSE | ISLP ch. 3 | Explícito de la fuente |
| 5. Escala, unidades y extremos | Géron ch. 2; Molin ch. 9; ISLP ch. 3 | Síntesis integrada |
| 6. Baselines | Patterns ch. 7 | Explícito de la fuente |
| 7. Tabla de decisión | Géron ch. 2; Molin ch. 9; ISLP ch. 3; Müller ch. 5 | Síntesis integrada |
| 7. Métrica oficial de competición | Müller ch. 5, aplicación competitiva | Recomendación para coding test |
| 8. Implementación de métricas | Géron ch. 2; Molin ch. 9 | Explícito de la fuente |
| 8. `root_mean_squared_error` | Géron ch. 2, nota de API actualizada | Conclusión práctica derivada |
| 8. Baseline de mediana | Patterns ch. 7 | Explícito de la fuente |
| 9. Riesgos de interpretación | Todas las fuentes utilizadas | Síntesis integrada |
| 10. Procedimiento para coding test o datathon | Géron ch. 2; Müller ch. 5; Molin ch. 9; Patterns ch. 6–7 | Recomendación para coding test |
| 11. Checklist | Todas las fuentes utilizadas | Recomendación para coding test |
| 12. Relación documental | Encargo de integración | Síntesis integrada |
| 13. Vacíos y limitaciones | Cobertura comprobada de los resúmenes consultados | Conclusión práctica derivada |
