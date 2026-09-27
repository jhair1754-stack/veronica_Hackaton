---
title: "Detección y prevención de data leakage"
knowledge_area: "preprocessing"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Detección y prevención de data leakage

## 1. Propósito y alcance

Este documento ayuda a impedir que el entrenamiento, la selección o la evaluación utilicen información que no estaría disponible bajo las condiciones reales de predicción. Cubre el orden entre partición y preparación, el ajuste de transformaciones, la construcción temporal de variables, las relaciones entre observaciones y la preservación del conjunto de evaluación.

**Síntesis integrada.** El control de leakage no es una operación aislada: depende de qué se predice, cuándo se predice, qué filas están relacionadas y qué decisiones aprenden de los datos. Aquí se documentan controles de preparación y diagnóstico; la elección completa del esquema de validación corresponde a `knowledge/validation/validation_strategy.md`. [Müller, cap. 5–6; ML Design Patterns, cap. 6]

## 2. Qué significa data leakage

**Explícito de las fuentes.** Hay fuga cuando información perteneciente a validación o prueba influye en un paso que se presenta como aprendido solo con entrenamiento. Esto ocurre, por ejemplo, si un escalador o un selector de variables se ajusta antes de la validación cruzada, si filas de una misma entidad quedan a ambos lados del split, si una partición aleatoria permite usar futuro para evaluar pasado o si el test guía la selección. La consecuencia es una evaluación que deja de representar datos realmente nuevos y puede resultar optimista. [Müller, cap. 5–6; ML Design Patterns, cap. 6]

Conviene separar tres estados:

- **Variable sospechosa:** su origen, instante de disponibilidad o relación con otras filas aún no está aclarado. Una asociación fuerte con el objetivo no confirma por sí sola una fuga.
- **Transformación incorrecta:** existe evidencia de que sus estadísticas, categorías, variables seleccionadas o artefactos se aprendieron fuera del entrenamiento permitido.
- **Fuga confirmada:** se reconstruye el flujo y se demuestra que el modelo o una decisión de modelado recibió información de evaluación, de observaciones relacionadas que debían estar aisladas o de un instante posterior al de predicción. [Síntesis integrada de Müller, cap. 5–6; Molin, cap. 4; ML Design Patterns, cap. 6]

Una puntuación extraordinaria es solo una señal para investigar. Confirmar leakage exige evidencia sobre procedencia, tiempo, particiones y operaciones de ajuste; la métrica aislada no identifica la causa.

## 3. Tipos o vías de leakage respaldadas por las fuentes

### 3.1 Leakage por preparación previa a la partición

**Explícito de las fuentes.** Toda operación que aprende de los datos —escalamiento, imputación, discretización por cuantiles, selección o extracción de variables, estadísticas grupales o embeddings— debe ajustarse con el entrenamiento autorizado. En validación cruzada, el ajuste debe repetirse dentro de cada fold. La selección global de variables puede distorsionar mucho más la evaluación que un escalamiento global. [Müller, cap. 6; McKinney, cap. 10; Molin, cap. 4; ML Design Patterns, cap. 6]

| Vía de leakage | Cómo se produce | Señal observable | Cómo verificarla | Prevención respaldada |
|---|---|---|---|---|
| Estadísticas globales de preparación | Se calculan medias, medianas, desviaciones, cuantiles, bins o estadísticas por grupo antes de separar | El artefacto se creó con el dataset completo | Rastrear filas usadas por cada `fit`, agregado o cálculo | Separar primero; ajustar con train y aplicar el artefacto aprendido a validación/test |
| Selección o extracción global | Se eligen variables usando todos los datos o folds de validación | La selección ocurre antes de `cross_val_score`/`GridSearchCV` | Revisar si el selector está dentro del estimador evaluado | Incluir selector y modelo en un `Pipeline` |
| Embeddings o artefactos aprendidos antes del split | La representación incorpora estructura de evaluación | El embedding aparece como entrada ya calculada para todas las filas | Identificar dataset y momento con que fue entrenado | Calcularlo después del split y solo con train, salvo pretraining realmente independiente |

### 3.2 Leakage por información futura o no disponible

**Explícito de las fuentes.** Una partición aleatoria es inadecuada cuando el objetivo es extrapolar al futuro. También hay fuga si una ventana incluye el evento actual o posterior, si un join temporal toma el vecino futuro o si se usan timestamps que no respetan la disponibilidad real. [Molin, cap. 4; ML Design Patterns, cap. 6]

| Vía de leakage | Cómo se produce | Señal observable | Cómo verificarla | Prevención respaldada |
|---|---|---|---|---|
| Split aleatorio en datos temporales | Pasado y futuro se mezclan | Fechas posteriores aparecen en train respecto de evaluación | Comparar rangos de tiempo por partición y objetivo de despliegue | Usar un split secuencial cuando se evalúa inferencia futura |
| Ventanas con presente/futuro | La fila actual o eventos posteriores contribuyen a su feature | `rolling`, agregados o rellenos no tienen desplazamiento/límite temporal defendible | Recalcular una fila usando solo datos disponibles en su timestamp | Ordenar, particionar por entidad y excluir presente/futuro cuando corresponda |
| Join temporal no causal | Se alinea con el evento más cercano aunque sea posterior | `direction="nearest"`, tolerancias amplias o timestamps ambiguos | Inspeccionar pares de timestamps y regla de disponibilidad | Usar una regla temporal coherente y verificar corrección *point-in-time* |

### 3.3 Leakage por construcción del objetivo o de las variables

**Cobertura parcial.** Las fuentes consultadas respaldan el riesgo de variables agregadas, ventanas, embeddings y joins calculados con datos de evaluación o con información futura. También respaldan que un selector supervisado que usa `y` debe ajustarse dentro de cada fold. [Müller, cap. 6; Molin, cap. 4; ML Design Patterns, cap. 6]

No ofrecen una taxonomía suficientemente explícita de variables construidas directamente a partir del resultado ni un procedimiento general para auditar la creación del target. Por ello, una columna muy asociada con `y` se conserva como **sospecha**, no como fuga confirmada. La verificación mínima derivada consiste en reconstruir su definición, fuentes y timestamp, y demostrar si utiliza el resultado o información posterior al momento de predicción.

| Vía de leakage | Cómo se produce | Señal observable | Cómo verificarla | Prevención respaldada |
|---|---|---|---|---|
| Selector supervisado fuera del fold | La elección de variables consulta `y` de evaluación | Selector ejecutado antes de CV | Revisar el orden de llamadas y filas usadas | Colocar selección y modelo dentro del pipeline |
| Variable de procedencia dudosa | Su construcción podría usar el resultado o futuro | Asociación excepcional o disponibilidad incierta | Revisar definición, tablas fuente y timestamps | No usarla hasta demostrar disponibilidad; la eliminación automática no está respaldada |

### 3.4 Leakage entre observaciones relacionadas

**Explícito de las fuentes.** Si varias filas pertenecen a la misma persona, paciente, hablante, dispositivo, fecha o evento, repartirlas entre train y validación puede medir reconocimiento de grupos conocidos en vez de generalización a grupos nuevos. Una semilla reproducible no corrige una clave de partición incorrecta. [Müller, cap. 5; ML Design Patterns, cap. 6]

| Vía de leakage | Cómo se produce | Señal observable | Cómo verificarla | Prevención respaldada |
|---|---|---|---|---|
| Entidades repetidas cruzan splits | Filas correlacionadas se dividen de forma independiente | El mismo ID aparece en train y evaluación | Intersectar claves de grupo entre particiones | Mantener grupos completos; usar un splitter por grupos si la generalización buscada es a grupos nuevos |
| Duplicados o vecinos del mismo evento cruzan splits | La clave elegida no representa la dependencia real | Registros casi idénticos en ambos lados | Auditar duplicados, fecha, sesión, fuente o evento | Definir una clave estable que confine la correlación a un split |

### 3.5 Leakage durante selección, ajuste o evaluación

**Explícito de las fuentes.** El test no debe intervenir en exploración, selección de variables, elección de modelo, tuning ni ajuste de umbral. `GridSearchCV` debe operar dentro del conjunto de desarrollo; usar repetidamente el test para elegir produce una estimación optimista. [Müller, cap. 5–6]

| Vía de leakage | Cómo se produce | Señal observable | Cómo verificarla | Prevención respaldada |
|---|---|---|---|---|
| Tuning contra test | Se conservan parámetros/modelos según su score de test | Muchas evaluaciones del mismo test antes del resultado final | Revisar historial de experimentos y decisiones | Reservar test y seleccionar con validación/CV dentro de train |
| Ajuste global dentro de CV | El preprocesamiento se ejecuta una vez antes de crear folds | Matriz ya transformada entregada a CV | Revisar si cada fold ejecuta `fit` de los transformadores | Entregar el pipeline completo a CV o búsqueda |
| Decisiones repetidas sobre la misma evaluación | Configuraciones y umbrales se adaptan a sus resultados | La evaluación deja de ser un uso final | Trazar qué decisión siguió a cada consulta | Separar selección de evaluación final; considerar evaluación anidada cuando el objetivo lo justifique |

## 4. Cómo detectar señales de leakage

**Síntesis integrada basada en las fuentes.** Investigar en este orden:

1. Definir la unidad y el instante de predicción. Registrar qué datos existirían entonces.
2. Para cada variable sospechosa, documentar fuente, timestamp, regla de cálculo y disponibilidad en inferencia.
3. Reconstruir la creación del target. Las fuentes no ofrecen una auditoría general del target, pero esta revisión permite decidir si una variable depende de él o de un resultado posterior.
4. Dibujar el orden real: split → ajuste de artefactos → transformación → selección/tuning → evaluación.
5. Para cada operación con `fit`, agregado, vocabulario, embedding o selector, enumerar las filas usadas para aprenderla.
6. Comprobar intersecciones de IDs, sesiones, fechas, eventos o duplicados entre particiones.
7. En datos temporales, inspeccionar rangos, orden, ventanas y joins; reproducir algunas features usando únicamente el historial disponible.
8. Auditar cuántas veces el test o evaluación final fue consultado y qué decisiones se tomaron después.
9. Si el rendimiento parece incompatible con el proceso de datos, repetir la evaluación después de aislar la vía sospechosa. Un descenso apoya el diagnóstico, pero la confirmación proviene del flujo demostrado, no del descenso por sí solo.

## 5. Procedimiento para prevenirlo

1. **Definir el contrato de predicción.** Especificar unidad, momento y entradas disponibles. [Síntesis integrada de Molin, cap. 4; ML Design Patterns, cap. 6]
2. **Elegir y registrar la lógica de partición.** Aislar primero el test; después decidir si las dependencias exigen grupos o secuencia temporal. La decisión completa se desarrolla en `knowledge/validation/validation_strategy.md`. [Müller, cap. 5; ML Design Patterns, cap. 6]
3. **Separar antes de aprender transformaciones.** Ninguna estadística, selección, embedding o artefacto dependiente de datos debe conocer validación/test. [Müller, cap. 6; ML Design Patterns, cap. 6]
4. **Ajustar solo con entrenamiento.** Dentro de CV, “entrenamiento” significa el fold de entrenamiento actual. [Müller, cap. 6]
5. **Aplicar, sin reajustar, a validación o test.** Reutilizar estadísticas, categorías y orden de pasos aprendidos; definir tratamiento para grupos desconocidos cuando corresponda. [McKinney, cap. 10; Müller, cap. 6]
6. **Encapsular el flujo.** Usar un pipeline para que transformación, selección y modelo se ajusten en el orden correcto. Versionar la lógica y sus artefactos. [Müller, cap. 6; ML Design Patterns, cap. 6]
7. **Preservar causalidad temporal.** Ordenar por entidad/tiempo, excluir presente o futuro cuando el caso lo requiera y verificar joins *point-in-time*. [Molin, cap. 4; ML Design Patterns, cap. 6]
8. **Seleccionar dentro de desarrollo.** Ajustar modelo, hiperparámetros y umbral mediante validación/CV; no usar test para decidir. [Müller, cap. 5]
9. **Evaluar una vez con decisiones cerradas.** Registrar resultado, versión del split, pipeline y artefactos. [Müller, cap. 5; ML Design Patterns, cap. 6]

## 6. Reglas de decisión

| Situación observada | Verificación necesaria | Acción recomendada | Motivo | Precaución |
|---|---|---|---|---|
| Columna muy correlacionada con el target | Origen, fórmula y disponibilidad temporal | Mantenerla en cuarentena hasta aclararla | Correlación no demuestra fuga | No eliminarla automáticamente |
| Transformación creada antes del split | Filas usadas en su ajuste | Reajustar con train y repetir evaluación | Validación/test influyeron en el artefacto | Conservar idéntico tratamiento al aplicar |
| Filas repetidas por entidad | Unidad real de generalización e intersección de IDs | Usar partición grupal si se busca generalizar a entidades nuevas | Evita compartir información correlacionada | Un grupo incorrecto puede sesgar o impedir el uso de una variable |
| Fechas con orden predictivo | Si el despliegue extrapola al futuro | Usar partición temporal y features causales | Un split aleatorio puede filtrar futuro | Estacionalidad y cambio de distribución requieren análisis propio |
| Rolling o agregado temporal | Límite superior de la ventana por fila | Desplazar/excluir presente o futuro cuando corresponda | Evita que la observación contribuya a su propia feature | La regla exacta depende del momento de predicción |
| Score sorprendentemente alto | Pipeline, solapes, timestamps y uso de test | Auditar antes de concluir | Es señal, no prueba | Un buen score también puede ser legítimo |
| Test consultado durante tuning | Decisiones tomadas después de cada consulta | Reconstituir una evaluación no usada, si existe | El test dejó de ser final | Las fuentes no fijan un número universal de consultas aceptables |
| Estadísticas por grupo | Si se calcularon solo con train y existen grupos nuevos | Guardar mapeo de train y respaldo explícito | Evita fuga y fallos de inferencia | Grupos pequeños pueden ser inestables |

## 7. Implementación mínima en Python

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** Este ejemplo es apropiado solo cuando una partición aleatoria representa el problema. Si existen grupos o tiempo, se debe sustituir el splitter conforme a `knowledge/validation/validation_strategy.md`.

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reservar evaluación final antes de decidir modelos o transformaciones.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42
)

# El pipeline aprende imputación y escala únicamente con X_train.
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    estimator,
)
pipeline.fit(X_train, y_train)
validation_score = pipeline.score(X_valid, y_valid)

# Usar validation_score para decisiones. No consultar X_test/y_test todavía.
# Tras cerrar todas las decisiones, reajustar con X_dev/y_dev y evaluar test una vez.
```

Para validación cruzada, el mismo principio exige pasar el `pipeline` completo a `cross_val_score` o `GridSearchCV`; no se debe entregar una matriz transformada previamente. [Müller, cap. 6]

## 8. Errores frecuentes y riesgos

- Ajustar imputación, escala, bins, estadísticas grupales, embeddings o selección antes de separar.
- Incluir variables o ventanas que aún no existen al momento de predecir.
- Ignorar procedencia, timestamp o fórmula de una columna.
- Mezclar entidades, eventos o duplicados relacionados entre train y evaluación.
- Barajar datos cuando el orden temporal define el escenario de generalización.
- Reutilizar test para seleccionar modelos, parámetros o umbrales.
- Suponer que una puntuación alta confirma generalización o confirma leakage.
- Aplicar a evaluación una ruta u orden de transformaciones diferente de entrenamiento.
- Creer que una semilla arregla una partición por grupos o tiempo mal formulada.
- Automatizar un pipeline incorrecto: la repetibilidad conserva el error si la lógica estadística es defectuosa. [ML Design Patterns, cap. 6]

## 9. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

Antes del baseline, definir cuándo se predice y revisar IDs, fechas, sesiones, fuentes y columnas sospechosas. Preguntar cómo se generaron el target y cada variable dudosa; registrar la respuesta o la falta de evidencia. Después, comprobar que la partición ocurre antes del ajuste de transformaciones y que selección/modelo viven en un pipeline evaluado como unidad.

Durante la iteración, conservar una evaluación final separada, comparar modelos sobre la misma lógica de split y anotar cada sospecha, verificación y decisión. Si una mejora es extraordinaria tras seleccionar variables, crear agregados o añadir ventanas, auditar el orden de ajuste y la disponibilidad temporal antes de aceptarla. No construir infraestructura desproporcionada: basta un flujo corto, reproducible y trazable.

## 10. Checklist operativo

- [ ] Definí el momento y la unidad en que se realizará la predicción.
- [ ] Revisé cuándo estaría disponible cada variable relevante.
- [ ] Documenté el origen del target y de las columnas sospechosas.
- [ ] Separé los datos antes de ajustar transformaciones o representaciones.
- [ ] Ajusté los tratamientos solo con entrenamiento o con el fold de entrenamiento.
- [ ] Apliqué, sin reajustar, los tratamientos aprendidos a validación o prueba.
- [ ] Revisé si existen grupos, duplicados, sesiones o dependencia temporal.
- [ ] Verifiqué que ventanas y joins no usen presente/futuro indebido.
- [ ] Evité usar evaluación final para decisiones repetidas.
- [ ] Investigué señales sospechosas antes de eliminar columnas.
- [ ] Registré split, artefactos, verificaciones y decisiones.

## 11. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con data leakage |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Ayuda a precisar el objetivo y el momento de predicción |
| `knowledge/eda/dataset_audit.md` | Registra señales iniciales de columnas o estructuras sospechosas |
| `knowledge/preprocessing/data_cleaning.md` | Coordina el ajuste seguro de tratamientos de datos |
| `knowledge/validation/validation_strategy.md` | Define una partición coherente con grupos o tiempo |
| `knowledge/feature_engineering/feature_engineering.md` | Revisa riesgos en la construcción de variables |
| `knowledge/workflows/competition_workflow.md` | Integra controles de leakage en el flujo completo |

## 12. Vacíos y limitaciones de las fuentes

- No se encontró una taxonomía completa y explícita de *target leakage* ni un procedimiento general para auditar la construcción del objetivo.
- La cobertura de variables derivadas directamente del resultado es parcial; la revisión de procedencia y tiempo se presenta como síntesis práctica, no como tipología completa de una fuente.
- La corrección de una variable depende del proceso real de generación y disponibilidad; ningún resumen permite inferirlo solo por nombre o correlación.
- Las fuentes explican grupos y tiempo, pero la selección completa del splitter, sus tamaños y su diseño experimental pertenece a `knowledge/validation/validation_strategy.md`.
- No hay umbrales universales para declarar sospechoso un score, una correlación, una tolerancia temporal o un número de consultas al test.
- Ventanas, joins y agregados requieren validación experimental en el dominio; una operación causal puede seguir siendo poco representativa.
- La auditoría profunda de construcción de variables debe ampliarse en `knowledge/feature_engineering/feature_engineering.md`.

## 13. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Müller, *Introduction to Machine Learning with Python*, cap. 6 | [Resumen](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Fuga por preprocesamiento, pipelines, selección dentro de CV y evaluación final |
| Müller, *Introduction to Machine Learning with Python*, cap. 5 | [Resumen](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | Separación train/validation/test, grupos, tuning y riesgo de decidir con test |
| McKinney, *Python for Data Analysis*, cap. 10 | [Resumen](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md) | Estadísticas e imputación por grupo y riesgos derivados entre entidades relacionadas |
| Molin, *Hands-On Data Analysis with Pandas*, cap. 4 | [Resumen](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md) | Ventanas, lags, joins temporales, cuantiles e información futura |
| Lakshmanan, Robinson y Munn, *Machine Learning Design Patterns*, cap. 6 | [Resumen](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) | Transform, Repeatable Splitting, Windowed Inference, Workflow Pipeline y corrección temporal |

## 14. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Müller cap. 5–6; ML Design Patterns cap. 6 | Síntesis integrada |
| 2. Definición y consecuencias | Müller cap. 5–6; ML Design Patterns cap. 6 | Síntesis integrada |
| 2. Estados de sospecha/confirmación | Conjunto de fuentes utilizadas | Conclusión práctica derivada |
| 3.1 Preparación previa | Müller cap. 6; McKinney cap. 10; Molin cap. 4; ML Design Patterns cap. 6 | Explícito de la fuente y síntesis integrada |
| 3.2 Futuro/no disponible | Molin cap. 4; ML Design Patterns cap. 6 | Explícito de la fuente |
| 3.3 Construcción del objetivo/variables | Müller cap. 6; limitaciones de cobertura | Síntesis integrada |
| 3.4 Observaciones relacionadas | Müller cap. 5; ML Design Patterns cap. 6 | Explícito de la fuente |
| 3.5 Selección, ajuste y evaluación | Müller cap. 5–6 | Explícito de la fuente |
| 4. Investigación de señales | Todas las fuentes utilizadas | Conclusión práctica derivada |
| 5. Prevención | Todas las fuentes utilizadas | Síntesis integrada |
| 6. Reglas de decisión | Todas las fuentes utilizadas | Conclusión práctica derivada |
| 7. Python | Müller cap. 5–6 | Conclusión práctica derivada |
| 8. Errores | Todas las fuentes utilizadas | Síntesis integrada |
| 9. Coding test/datathon | Todas las fuentes utilizadas | Recomendación para coding test |
| 10. Checklist | Todas las fuentes utilizadas | Conclusión práctica derivada |
| 11. Documentos relacionados | Encargo y organización del repositorio | Síntesis integrada |
| 12. Vacíos | Revisión comparada de fuentes utilizadas | Conclusión práctica derivada |
