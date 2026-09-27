---
title: "Ajuste de hiperparámetros"
knowledge_area: "tuning"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Ajuste de hiperparámetros

## 1. Propósito y alcance

El ajuste de hiperparámetros (*tuning*) compara configuraciones de un modelo para encontrar opciones que mejoren un objetivo de evaluación bajo una estrategia de validación definida. La búsqueda forma un bucle exterior alrededor del entrenamiento: recibe un espacio de configuraciones, entrena candidatos y los compara mediante una métrica calculada fuera de los datos usados para ajustar cada candidato ([Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

**Síntesis integrada.** El tuning no garantiza una mejora. El resultado depende de los datos, la métrica, el modelo, el espacio explorado y el diseño experimental. La mejor configuración observada es la mejor dentro de lo probado y bajo esa evaluación; no garantiza un óptimo global ni generalización ([Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

Este documento se limita a decidir cuándo, qué y cómo ajustar. La formulación, la elección detallada de métricas, la estrategia general de validación y la selección inicial de modelos son decisiones distintas y se remiten a la sección 14.

## 2. Parámetros e hiperparámetros

Un **parámetro** se aprende a partir de los datos durante el ajuste, como los pesos mencionados por el patrón *Hyperparameter Tuning*. Un **hiperparámetro** es una configuración elegida antes del ajuste y no aprendida directamente de los datos ([Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md); [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

Ejemplos explícitos son `C` y `gamma` para SVC; `n_estimators`, `max_features` y `bootstrap` para Random Forest; y *learning rate*, *batch size*, profundidad, neuronas, *kernel* o épocas en redes. Decisiones inciertas de preparación también pueden tratarse como hiperparámetros si se ajustan dentro de la validación ([Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md); [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md)). No son rangos universales.

## 3. Cuándo iniciar el tuning

Las fuentes convergen en condiciones previas, no en una secuencia universal rígida:

- formulación y métrica alineadas con el uso;
- test reservado, ajeno a selección y ajuste;
- baseline registrado;
- validación representativa de los datos futuros;
- transformaciones aprendidas encapsuladas dentro de cada partición;
- candidatos razonables preseleccionados e hiperparámetros con impacto plausible ([Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md); [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

No conviene usar tuning para compensar formulación incorrecta, datos defectuosos, *leakage* o una métrica muy ruidosa. Si un entrenamiento consume todo el presupuesto, tampoco procede una búsqueda de varios trials ([ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

**Conclusión práctica derivada.** *Heuristic Benchmark* propone comprobar primero si el modelo supera una referencia simple o el proceso operativo vigente con la misma métrica y datos comparables. Superar una heurística débil no demuestra utilidad, pero no superarla cuestiona añadir complejidad ([ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md)).

## 4. Diseño del espacio de búsqueda

El espacio especifica qué hiperparámetros se varían y, según el método, valores, rangos, distribuciones, tipos, escalas y restricciones. Conviene seleccionar pocas decisiones con impacto plausible, comenzar con valores espaciados y refinar regiones prometedoras. Un mejor resultado en el borde puede justificar ampliar el rango; una superficie plana puede señalar un rango inadecuado o un parámetro irrelevante ([Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md)).

Las opciones condicionales deben expresarse como rejillas separadas. En pipelines, los parámetros internos usan `<paso>__<parametro>` ([Müller, caps. 5–6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)).

| Decisión de búsqueda | Pregunta que debe responderse | Precaución |
|---|---|---|
| Parámetros incluidos | ¿Qué decisiones tienen impacto plausible? | Cada dimensión eleva coste y adaptación a validación. |
| Dominio | ¿Lista, rango, distribución o elección condicional? | No se recuperan valores fuera del espacio. |
| Escala | ¿Lineal o logarítmica según la fuente y el parámetro? | Una escala inadecuada desperdicia trials. |
| Coherencia | ¿Hay opciones válidas solo para cierto modelo? | Separar combinaciones condicionales. |
| Granularidad | ¿Es amplio sin ser innecesariamente denso? | Una rejilla granular puede ser prohibitiva. |
| Presupuesto | ¿Cuántos candidatos, folds y ajustes caben? | El coste crece con candidatos × folds. |
| Evidencia | ¿El óptimo cae en un borde o los scores son planos? | Refinar con validación, nunca con test. |

**Tipo:** síntesis integrada de Müller 5–6, Molin 10 y ML Design Patterns 4.

## 5. Métodos de búsqueda

### 5.1 Ajuste manual

- **Funcionamiento:** una persona propone configuraciones usando conocimiento del modelo.
- **Entradas:** pocos hiperparámetros, modelo rápido, métrica y validación estables.
- **Selección:** mejor evaluación observada bajo condiciones consistentes.
- **Ventajas:** simple y guiada por conocimiento.
- **Limitaciones:** consume tiempo, puede no aislar efectos, es poco sistemática y no demuestra optimalidad.
- **Costo/escalabilidad:** aceptable con muy pocas decisiones; empeora al crecer el espacio.
- **Cuándo considerarlo:** exploración acotada con pocos parámetros y entrenamientos baratos ([ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

### 5.2 Búsqueda en rejilla

- **Funcionamiento:** evalúa el producto cartesiano de los valores mediante CV.
- **Entradas:** estimador/pipeline, `param_grid`, `cv` y `scoring`.
- **Selección:** mejor score medio; con `refit`, se reajusta sobre todo train.
- **Ventajas:** exhaustiva sobre la rejilla finita, reproducible y detallada.
- **Limitaciones:** no aprende de pruebas anteriores ni extrapola fuera del espacio.
- **Costo/escalabilidad:** combinaciones × folds; paralelizar puede elevar memoria.
- **Cuándo considerarla:** espacio discreto y pequeño; no para rejillas enormes ([Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)).

### 5.3 Búsqueda aleatoria

- **Funcionamiento:** muestrea `n_iter` configuraciones en vez de enumerarlas.
- **Entradas:** listas/distribuciones, `n_iter`, métrica, CV y estado aleatorio.
- **Selección:** mejor validación entre las muestras ejecutadas.
- **Ventajas:** controla el presupuesto y explora espacios más amplios.
- **Limitaciones:** puede omitir la mejor región; depende del espacio planteado.
- **Costo/escalabilidad:** `n_iter` × folds × coste de ajuste.
- **Cuándo considerarla:** espacio amplio o presupuesto fijo; no es universalmente superior a grid ([Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md)).

### 5.4 Otros métodos respaldados por las fuentes

**Optimización bayesiana.** Aprende una función sustituta con trials previos y propone configuraciones prometedoras hasta agotar el presupuesto. Puede ahorrar entrenamientos completos cuando cada trial es caro, pero añade coordinación, todos los trials siguen costando y no garantiza un óptimo global. Más paralelismo reduce tiempo de pared pero también el feedback secuencial ([ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md); [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md)).

**Algoritmos genéticos.** Solo se mencionan como alternativa que usa resultados previos, requiere muchos entrenamientos y crece en complejidad. No hay respaldo suficiente para un procedimiento seguro; se registra la limitación ([ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

## 6. Evaluación de configuraciones

Las configuraciones deben compararse con la misma métrica y una validación compatible con la unidad de generalización. K-fold produce un score por fold; media y dispersión muestran nivel y sensibilidad. Clasificación puede requerir estratificación; entidades repetidas, grupos; dependencia temporal, orden preservado ([Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md)).

- **Training:** mide ajuste sobre datos vistos. Error alto allí es compatible con subajuste; training mucho mejor que validación, con sobreajuste.
- **Validación:** guía comparación y tuning sobre datos no usados para ajustar cada candidato.
- **Evaluación final:** ocurre una vez, tras fijar decisiones, sobre el test reservado. Retocar después destruye su independencia ([Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md)).

La selección de la métrica no es tuning. Véanse `knowledge/metrics/classification_metrics.md` y `knowledge/metrics/regression_metrics.md`.

**Conclusión práctica derivada.** Una mejora pequeña debe interpretarse junto con dispersión, coste y estabilidad. No existe en estas fuentes un umbral universal de mejora significativa.

## 7. Relación entre tuning, validación y data leakage

La búsqueda evalúa el flujo completo. Toda operación que aprenda estadísticas, categorías, escalas, variables o componentes debe ajustarse solo con el fold de entrenamiento. `Pipeline` conserva este orden y permite buscar con `<paso>__<parametro>` ([Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)).

El test queda fuera de exploración, selección, tuning y umbral. *Repeatable Splitting* pide agrupar observaciones correlacionadas con una clave estable y usar separación secuencial si se evalúa futuro. Particiones inconsistentes pueden cambiar el ranking por el split, no por el modelo ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

Véanse `knowledge/validation/validation_strategy.md` y `knowledge/preprocessing/data_leakage.md`.

## 8. Presupuesto computacional y criterios para detenerse

El coste depende de configuraciones × folds × coste por ajuste. Más opciones, folds y validación anidada multiplican entrenamientos; paralelizar puede elevar memoria. Randomized Search usa `n_iter`; la optimización bayesiana, máximo de trials y paralelismo ([Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

Criterios respaldados: agotar el presupuesto fijado; evitar productos de combinaciones inviables; no iniciar búsqueda informada si un entrenamiento excede todo el presupuesto; considerar tiempo de ajuste/inferencia; y, bajo presión, preferir búsqueda gruesa y acotada ([Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md); [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md)).

**Limitación:** no se prescriben umbrales universales de mejora, tiempo o memoria. *Early stopping* de trials aparece con el riesgo de descartar configuraciones de convergencia lenta, sin una regla general.

## 9. Implementación mínima en Python

Ejemplo basado en `Pipeline` + `GridSearchCV`. Los valores de `C` y `gamma` aparecen en el resumen de Müller y no son recomendaciones universales. `X_train` excluye el test final.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scale", MinMaxScaler()),
    ("model", SVC()),
])
search_space = {
    "model__C": [0.1, 1, 10],
    "model__gamma": [0.01, 0.1, 1],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid=search_space,
    scoring="roc_auc",  # solo si representa el objetivo
    cv=cv,
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)

final_test_score = search.score(X_test, y_test)  # solo al cerrar decisiones
```

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** Se declaran splitter, semilla, métrica, `refit` y pipeline. Estratificación y ROC-AUC deben sustituirse si grupos, tiempo u objetivo exigen otra elección ([Müller, caps. 5–6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)).

## 10. Reglas de decisión

| Situación | Estrategia de ajuste que puede considerarse | Motivo | Precaución |
|---|---|---|---|
| Muy pocos parámetros y modelos rápidos | Manual acotado | Simple y guiado | Poco sistemático; registrar cambios. |
| Espacio discreto pequeño | Grid search | Recorre la rejilla definida | Coste combinaciones × folds. |
| Espacio amplio y presupuesto fijo | Random search | Controla `n_iter` | Puede omitir la mejor región. |
| Trials caros y repetibles | Optimización bayesiana | Usa feedback previo | Compleja; no garantiza óptimo. |
| Transformaciones ajustables | Búsqueda sobre pipeline | Ajusta dentro de cada fold | No preprocesar globalmente. |
| Parámetros condicionales | Lista de rejillas | Evita combinaciones incoherentes | Cada diccionario es otra rejilla. |
| Entidades repetidas | Splitter por grupos | Evita cruce de entidades | `groups` no es el target. |
| Dependencia temporal | Split temporal | Respeta orden futuro | No barajar pasado y futuro. |
| Ranking inestable | Split repetible/CV adecuada | Comparación consistente | Una semilla no corrige mala agrupación. |
| Mejora marginal y cara | Baseline + benchmark | Contextualiza la ganancia | Una referencia débil no prueba utilidad. |

**Tipo:** síntesis integrada de las fuentes citadas en las secciones 3–8.

## 11. Errores frecuentes y riesgos

**Explícitos en las fuentes:** iniciar sin baseline o validación confiable; optimizar la métrica equivocada; mirar repetidamente el test; ajustar transformaciones fuera de folds; comparar con splits incompatibles; explorar demasiadas combinaciones; seleccionar por training; excluir regiones útiles; sobreajustar decisiones a la misma validación; e ignorar dispersión y coste ([Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Müller, caps. 5–6](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md); [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md)).

**Síntesis práctica derivada:** una mejora menor que la variación entre folds requiere cautela; el mejor resultado observado no garantiza generalización; corregir leakage puede bajar la cifra pero mejorar la validez; guardar solo `best_params_` impide auditar los trials.

## 12. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Construir baseline y, si procede, heurística simple.
2. Confirmar métrica y congelar validación coherente con clases, grupos o tiempo.
3. Mantener fuera el test final o holdout local.
4. Elegir pocos hiperparámetros plausibles y usar pipeline.
5. Limitar el espacio: manual/grid pequeño para pocos valores; aleatoria con `n_iter` para más amplitud.
6. Registrar configuración, media, dispersión, tiempo y estado.
7. Comparar contra baseline bajo idénticas condiciones.
8. Detenerse al agotar tiempo/presupuesto o no hallar evidencia consistente; no hay umbral numérico universal.

*Workflow Pipeline* sugiere separar `prepare → features → train → evaluate → predict` y conservar trazabilidad. En un coding test basta un flujo determinista; containerización y orquestación completas suelen exceder el alcance ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

## 13. Checklist operativo

- [ ] Definí un baseline antes de ajustar.
- [ ] Confirmé la métrica que guiará la búsqueda.
- [ ] Elegí validación adecuada para clases, grupos o tiempo.
- [ ] Reservé test para una única evaluación final.
- [ ] Seleccioné hiperparámetros respaldados por modelo y fuentes.
- [ ] Definí un espacio compatible con el presupuesto.
- [ ] Incluí transformaciones aprendidas dentro del pipeline.
- [ ] Comparé configuraciones bajo condiciones consistentes.
- [ ] Registré parámetros, scores, variación, tiempo y decisiones.
- [ ] Revisé que no exista *data leakage*.
- [ ] Comparé con baseline y documenté la parada.

## 14. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con el ajuste de hiperparámetros |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Define la tarea y el objetivo del modelo |
| `knowledge/validation/validation_strategy.md` | Determina cómo evaluar las configuraciones |
| `knowledge/metrics/classification_metrics.md` | Define métricas para tuning en clasificación |
| `knowledge/metrics/regression_metrics.md` | Define métricas para tuning en regresión |
| `knowledge/models/tabular_models.md` | Describe los modelos y sus hiperparámetros |
| `knowledge/ensembles/boosting_models.md` | Describe parámetros de ensambles y boosting |
| `knowledge/preprocessing/data_leakage.md` | Ayuda a prevenir leakage durante búsqueda y evaluación |
| `knowledge/workflows/competition_workflow.md` | Integra el tuning en el flujo de competición |

## 15. Vacíos y limitaciones de las fuentes

- Algoritmos genéticos carecen de detalle suficiente para implementación.
- No se cubren otros métodos fuera de manual, grid, random, bayesiana y esa mención limitada.
- No hay umbrales universales de presupuesto, memoria, mejora o parada.
- No existen rangos transferibles exhaustivos; requieren validación experimental.
- Los ejemplos dependen de versiones de API y no garantizan compatibilidad universal.
- *Early stopping* de trials no incluye una política general.
- No se cuantifica cuántas iteraciones sobreajustan una validación.
- *Repeatable Splitting* y *Workflow Pipeline* mejoran reproducibilidad, no garantizan corrección estadística.
- Métrica, splitter y modelo concretos requieren sus documentos especializados y evidencia del caso.

## 16. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, cap. 2 | [Resumen](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Flujo previo, grid/random, CV, test, coste y pipelines. |
| Géron, apéndice B | [Resumen](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md) | Baseline, shortlist, búsqueda aleatoria/bayesiana y test reservado. |
| Müller, cap. 5 | [Resumen](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | GridSearchCV, rejillas, splitters, nested CV y evaluación. |
| Müller, cap. 6 | [Resumen](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Pipeline, nombres de parámetros y leakage. |
| Molin, cap. 10 | [Resumen](../../sources/molin_hands_on_pandas/ch10_model_optimization.md) | Definición, Grid/Randomized Search, presupuesto y tiempo. |
| ML Design Patterns, cap. 4 | [Resumen](../../sources/ml_design_patterns/ch04_model_training_patterns.md) | Hyperparameter Tuning, espacio, trials y métodos. |
| ML Design Patterns, cap. 6 | [Resumen](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) | Repeatable Splitting y Workflow Pipeline. |
| ML Design Patterns, cap. 7 | [Resumen](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md) | Heuristic Benchmark. |

## 17. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1 | Müller 5; MLDP 4; Géron 2 | Explícito de la fuente / Síntesis integrada |
| 2 | Molin 10; MLDP 4; Müller 6; Géron 2 y ap. B | Explícito de la fuente |
| 3 | Géron ap. B; Müller 5; MLDP 4 y 7 | Síntesis integrada / Conclusión práctica derivada |
| 4 | Müller 5–6; Molin 10; MLDP 4 | Síntesis integrada |
| 5.1 | MLDP 4 | Explícito de la fuente |
| 5.2 | Müller 5; Géron 2; Molin 10; MLDP 4 | Explícito de la fuente / Síntesis integrada |
| 5.3 | Géron 2 y ap. B; Molin 10; MLDP 4 | Explícito de la fuente / Síntesis integrada |
| 5.4 | Géron ap. B; MLDP 4 | Explícito de la fuente |
| 6 | Géron 2 y ap. B; Müller 5; Molin 10 | Síntesis integrada / Conclusión práctica derivada |
| 7 | Müller 6; MLDP 6 | Explícito de la fuente / Síntesis integrada |
| 8 | Müller 5; Molin 10; Géron ap. B; MLDP 4 | Síntesis integrada |
| 9 | Müller 5–6 | Explícito de la fuente / Conclusión práctica derivada |
| 10 | Fuentes de secciones 3–8 | Síntesis integrada |
| 11 | Géron 2; Müller 5–6; Molin 10; MLDP 4 | Explícito de la fuente / Conclusión práctica derivada |
| 12 | Géron 2 y ap. B; Molin 10; MLDP 4, 6 y 7 | Recomendación para coding test |
| 13 | Integración de fuentes | Conclusión práctica derivada |
| 14 | Estructura requerida | Síntesis integrada |
| 15 | Límites declarados | Síntesis integrada |

**MLDP** = *Machine Learning Design Patterns*.
