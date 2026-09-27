---
title: "Flujo de trabajo para competiciones de Machine Learning"
knowledge_area: "workflows"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "competiciones de Machine Learning"
---

# Flujo de trabajo para competiciones de Machine Learning

## 1. Propósito y alcance

Este documento integra en un procedimiento práctico las decisiones de formulación, auditoría, preparación, validación, modelado y evaluación de problemas tabulares. Busca que cada comparación responda al objetivo declarado y que las decisiones puedan repetirse y revisarse.

Es una guía adaptable: no todos los retos requieren todos los pasos ni las mismas técnicas. La ruta depende del objetivo, la estructura y calidad de los datos, la métrica oficial, las reglas de la competición y los recursos disponibles. Las etapas pueden repetirse o cambiar de orden cuando una comprobación revele un problema de formulación o de datos. Este resumen no sustituye las guías temáticas enlazadas en la sección 11.

La evidencia técnica se limita a los resúmenes Markdown permitidos de `sources/`. Las recomendaciones para coding tests y datathons se identifican como síntesis o recomendaciones derivadas cuando no son procedimientos textuales de una fuente.

## 2. Principios de trabajo

- **Formular antes de elegir modelos.** Identificar la tarea, el resultado que se necesita y cómo se reconocerá un resultado útil. **Explícito de la fuente:** Géron relaciona tarea, experiencia y medida de rendimiento; su proyecto end-to-end empieza por el objetivo y el uso previsto. [Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).
- **Entender los datos antes de transformarlos.** Revisar esquema, tipos, distribuciones, faltantes y posibles problemas de calidad antes de decidir una limpieza. **Síntesis integrada** de los flujos de inspección y preparación de Géron y McKinney. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md).
- **Alinear la métrica con el objetivo.** La métrica oficial determina qué comparar; en clasificación, desbalance y costes distintos de falsos positivos y falsos negativos pueden volver engañosa la exactitud. **Explícito de la fuente.** [Géron, cap. 3](../../sources/geron_hands_on_ml/ch03_classification.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md).
- **Establecer una referencia antes de atribuir valor a una mejora.** Comparar un modelo con un predictor trivial, una heurística o el proceso vigente cuando exista. La referencia debe usar la misma métrica y datos comparables. **Explícito para el patrón Heuristic Benchmark; su uso temprano en competición es síntesis práctica.** [ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md).
- **Mantener una evaluación consistente.** Reutilizar los mismos folds o partición al comparar candidatos; elegir el método de partición según la unidad de generalización, las clases, los grupos o el tiempo. **Explícito de las fuentes.** [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md).
- **Evitar data leakage.** Las transformaciones que aprenden estadísticas, vocabularios, selección de variables u otros parámetros deben ajustarse únicamente dentro de los datos de entrenamiento de cada fold. **Explícito y reforzado por síntesis integrada.** [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).
- **Registrar decisiones y preferir experimentos reproducibles.** Guardar la configuración de partición, métrica, transformaciones y resultados; encapsular el proceso para repetirlo. **Síntesis integrada** de Repeatable Splitting, Workflow Pipeline y las listas operativas de Géron. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md), [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md).

## 3. Vista general del flujo

```text
Formular la tarea, el objetivo y el criterio de éxito
                         ↓
Auditar estructura, calidad y señales de grupos/tiempo/leakage
                         ↓
Definir métrica, partición y función del test final
                         ↓
Preparar features dentro de un flujo ajustado solo con entrenamiento
                         ↓
Medir una referencia simple y un baseline
                         ↓
Comparar familias de modelos bajo las mismas condiciones
                         ↓
Analizar errores y justificar cambios de representación o modelo
                         ↓
Afinar una shortlist dentro del presupuesto disponible
                         ↓
Cerrar selección, evaluar el test según su función y preparar entrega
                         ↓
Registrar configuración, resultados y limitaciones
```

La secuencia es una **síntesis integrada**, no una receta universal. Se puede volver a formular el problema, auditar una variable o revisar la estrategia de validación si los errores, el score o la estructura descubierta no concuerdan con el objetivo. En una competición, la validación local aproxima la evaluación oficial; el leaderboard no sustituye una evaluación local consistente ni justifica reutilizar el conjunto de prueba para iterar. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md).

## 4. Etapas del flujo de trabajo

### 4.1 Comprender y formular el problema

**Objetivo.** Traducir el enunciado a una tarea evaluable antes de escoger un algoritmo.

**Preguntas.** ¿Qué decisión o resultado se quiere apoyar? ¿Cuál es la unidad de análisis? ¿Qué columna es el target y qué información estará disponible al predecir? ¿Hay etiquetas? ¿El target es una clase, un valor numérico u otra estructura? ¿Qué métrica y restricciones define el reto?

**Acciones respaldadas.** Distinguir predicción supervisada de tareas no supervisadas y clasificación de regresión; describir tarea, datos/experiencia y medida de rendimiento. Confirmar que el target representa el objetivo y no solo un proxy conveniente. El patrón `Reframing` puede considerarse si la formulación de clasificación o regresión no representa la incertidumbre o el uso final; cambiar la representación tiene costes y puede introducir cortes arbitrarios o sesgo de etiqueta, por lo que sus alternativas deben contrastarse con la métrica pertinente. No se aplica automáticamente. [Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [ML Design Patterns, cap. 3](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

**Resultado esperado — síntesis práctica.** Una definición breve de unidad, target, tipo de tarea, uso previsto, métrica y restricciones conocidas.

**Avanzar o volver.** Avanzar cuando la tarea y el criterio de evaluación estén identificados. Volver a esta etapa si se descubre que el target no mide el objetivo, que sus valores requieren otra formulación o que la métrica del reto cambia qué salida debe producirse.

**Fuentes utilizadas.** [Géron, caps. 1–2](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [ML Design Patterns, cap. 3](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

**Documento relacionado.** [`knowledge/problem_definition/problem_formulation.md`](../problem_definition/problem_formulation.md).

### 4.2 Auditar el dataset

**Objetivo.** Entender qué representan las filas y columnas y detectar problemas que puedan invalidar la preparación o evaluación.

**Preguntas.** ¿Cuántas filas y variables hay? ¿Qué tipos y cardinalidades tienen? ¿Hay faltantes, duplicados, valores centinela, extremos, categorías inesperadas o target inconsistente? ¿Existen varias filas por entidad, orden temporal o relaciones entre filas? ¿Hay columnas que podrían revelar directa o indirectamente el target?

**Acciones respaldadas.** Empezar con una inspección estructural que permita reconocer filas, columnas, target, grupos y tiempo. Si existe un test etiquetado reservado, apartarlo antes de examinar en profundidad relaciones y distribuciones para orientar selección. Después, inspeccionar tipos, resúmenes y distribuciones; comparar faltantes y conteos por grupos; revisar duplicados y valores sospechosos antes de eliminarlos; observar el target y relaciones entre variables. La rareza por sí sola no demuestra que una fila sea un error. Cuando las filas dependen de grupos o tiempo, anotar esta estructura para diseñar luego la partición. La auditoría no debe convertirse en una exploración ilimitada: su propósito es informar decisiones posteriores. [Géron, caps. 2–3](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md), [McKinney, cap. 10](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md).

**Resultado esperado — síntesis práctica.** Un inventario del esquema, calidad, target, grupos/tiempo y riesgos de leakage que pueden afectar el split o el preprocesamiento.

**Avanzar o volver.** Avanzar si las columnas y observaciones pueden interpretarse lo suficiente para separar entradas y target. Investigar más o reformular si aparecen etiquetas dudosas, duplicados que alteren la unidad de análisis, señales de target derivado o estructura temporal/grupal no reflejada por un split aleatorio.

**Fuentes utilizadas.** [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [McKinney, caps. 7 y 10](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md), [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md).

**Documento relacionado.** [`knowledge/eda/dataset_audit.md`](../eda/dataset_audit.md).

### 4.3 Definir la evaluación

**Objetivo.** Estimar desempeño de forma alineada con la evaluación oficial y con la unidad a la que se espera generalizar.

**Preguntas.** ¿Qué métrica oficial se calcula y sobre qué predicción? ¿Importan probabilidades, ranking, clases o valores numéricos? ¿La evaluación debe generalizar a filas, entidades nuevas o periodos futuros? ¿Hay clases desbalanceadas? ¿El conjunto de prueba provisto es una evaluación final o parte del proceso de desarrollo?

**Acciones respaldadas.** Elegir la métrica con base en el objetivo y los costes; exactitud no basta automáticamente con clases desbalanceadas. Para clasificación considerar matriz de confusión y métricas apropiadas; para regresión, una métrica coherente con el coste de errores. Antes de la exploración detallada o de seleccionar transformaciones y modelos, apartar el test etiquetado para su función prevista. Elegir holdout o validación cruzada según tamaño, coste y estabilidad; estratificar cuando preserve proporciones importantes, separar por grupos si se generaliza a entidades no vistas y respetar el orden si el problema es temporal. Fijar la semilla cuando la estrategia contenga aleatoriedad y reutilizar particiones al comparar modelos. No seleccionar repetidamente mirando la puntuación de test. [Géron, caps. 1 y 3](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md).

**Resultado esperado — síntesis práctica.** Una métrica primaria, procedimiento de validación y política clara para cada conjunto. Si la competición no da test etiquetado, el holdout local reservado sigue siendo distinto de los folds usados para desarrollo.

**Avanzar o volver.** Avanzar cuando la partición simule razonablemente la situación de evaluación y la métrica esté definida. Revisar la estrategia ante solapamiento de entidades, dependencia temporal, clases muy raras, alta variación entre folds o desacuerdo entre validación y reglas oficiales.

**Fuentes utilizadas.** [Géron, caps. 1–3](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

**Documento relacionado.** [`knowledge/validation/validation_strategy.md`](../validation/validation_strategy.md).

### 4.4 Preparar los datos

**Objetivo.** Producir entradas utilizables conservando la separación entre lo aprendido durante ajuste y lo aplicado a datos nuevos.

**Preguntas.** ¿Qué tratamiento requieren faltantes y categorías? ¿Qué transformaciones dependen de estadísticas de los datos? ¿Qué features existirán también al predecir? ¿Alguna operación usa información del target o de filas que no deberían estar disponibles?

**Acciones respaldadas.** Tomar decisiones explícitas sobre imputación, codificación, escalado, selección y transformaciones; no convertir automáticamente todo valor extremo o faltante en error. Ajustar estadísticas de preparación solo en el training fold y aplicar esos parámetros a validación/test. Encapsular pasos y estimador en un pipeline cuando sea posible. Verificar que operaciones por grupo, joins y agregaciones no filtren información entre particiones ni creen features indisponibles al predecir. El patrón `Transform` destaca que la transformación y sus artefactos deben mantenerse coherentes entre ajuste e inferencia. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [McKinney, cap. 10](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

**Resultado esperado — síntesis práctica.** Una transformación repetible que recibe features crudas y puede aplicarse de la misma manera dentro de cada fold y sobre el conjunto de evaluación.

**Avanzar o volver.** Avanzar cuando cada paso sea ajustable solo con training y sus entradas estén disponibles en evaluación. Volver a auditoría o evaluación si la limpieza cambia qué constituye una fila, el target, el orden temporal o la composición de grupos.

**Fuentes utilizadas.** [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [McKinney, caps. 7 y 10](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

**Documento relacionado.** [`knowledge/preprocessing/data_cleaning.md`](../preprocessing/data_cleaning.md), [`knowledge/preprocessing/data_leakage.md`](../preprocessing/data_leakage.md).

### 4.5 Crear un baseline

**Objetivo.** Obtener una referencia medible antes de invertir en complejidad.

**Preguntas.** ¿Qué predictor sencillo o proceso vigente sirve como punto de comparación? ¿Un modelo básico supera una regla trivial bajo la misma métrica y evaluación? ¿El resultado se reproduce?

**Acciones respaldadas.** Establecer un benchmark heurístico comprensible: usar una práctica operativa existente si la hay; de lo contrario, elegir una regla trivial con sentido contextual. Calcular cualquier estadística de la regla solo con entrenamiento y evaluarla con la misma métrica y partición que los modelos. Después, crear un baseline de ML sencillo, elegido según la tarea y datos. Las fuentes presentan familias y ejemplos distintos; ninguna fuente justifica un ganador universal. **La secuencia de benchmark y baseline de modelo es una síntesis para competencias.** [ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md), [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [Molin, cap. 9](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md).

**Resultado esperado — síntesis práctica.** Score de referencia y un procedimiento base reproducible, junto con la métrica, split y configuración utilizados.

**Avanzar o volver.** Avanzar con el baseline funcional y comparable. Si un predictor trivial parece extraordinariamente fuerte, revisar target, split y leakage; si el baseline es deficiente, confirmar primero que la formulación, el preprocesamiento y la métrica sean correctos.

**Fuentes utilizadas.** [ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md), [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [Molin, cap. 9](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md).

**Documento relacionado.** [`knowledge/models/tabular_models.md`](../models/tabular_models.md).

### 4.6 Entrenar y comparar modelos

**Objetivo.** Determinar si alternativas justificadas mejoran el baseline en condiciones comparables.

**Preguntas.** ¿Qué supuestos de modelo concuerdan con la estructura observada? ¿El algoritmo necesita escalado o una codificación particular? ¿La diferencia entre entrenamiento y validación sugiere infraajuste o sobreajuste? ¿Cuánto cuesta cada comparación?

**Acciones respaldadas.** Probar una shortlist pequeña de familias razonables, no un catálogo indiscriminado. Géron compara alternativas lineales y basadas en árboles; ISLP y Molin resumen árboles y ensembles, pero sus resultados de ejemplos no establecen un vencedor general. Comparar con la misma métrica, partición y preprocesamiento; observar medidas fuera de muestra y su variación, además de errores de entrenamiento como diagnóstico. En clasificación desbalanceada, contrastar con baseline y revisar errores por clase en lugar de depender solo de accuracy. [Géron, caps. 1–3](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md), [Molin, caps. 9–10](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md).

**Resultado esperado — síntesis práctica.** Comparación legible de candidatos con score de validación, variación, coste aproximado y observaciones relevantes.

**Avanzar o volver.** Avanzar si existe una alternativa consistente que merezca análisis o tuning. Si todos los candidatos fallan de modo parecido, revisar target, métrica, datos y transformaciones; si la brecha train-validación crece, investigar complejidad y leakage.

**Fuentes utilizadas.** [Géron, caps. 1–3](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md), [Molin, caps. 9–10](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md).

**Documento relacionado.** [`knowledge/models/tabular_models.md`](../models/tabular_models.md), [`knowledge/ensembles/boosting_models.md`](../ensembles/boosting_models.md).

### 4.7 Mejorar variables o modelos

**Objetivo.** Convertir un diagnóstico concreto en un cambio que pueda evaluarse.

**Preguntas.** ¿Qué error o patrón motivó el cambio? ¿Tiene sentido construir, transformar, seleccionar o descartar variables? ¿El cambio conserva la misma información disponible al predecir? ¿Es posible aislar su efecto?

**Acciones respaldadas.** Revisar errores, importancia o estructura de variables y probar una modificación coherente: transformar distribuciones, derivar variables disponibles en el momento de predicción, ajustar complejidad/regularización, o ensayar un ensemble si hay candidatos sólidos con errores complementarios. `Reframing` es una opción condicionada por el uso real y la métrica. Cada variante debe compararse mediante el mismo procedimiento; una mejora observada no demuestra por sí sola generalización. [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md), [ML Design Patterns, caps. 3 y 7](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

**Resultado esperado — síntesis práctica.** Cambio y razón anotados, comparación contra la referencia y decisión de conservarlo o revertirlo.

**Avanzar o volver.** Conservar un cambio cuando mejore el criterio relevante sin romper validez o reproducibilidad. Si empeora o la evidencia es inestable, revertir y revisar si la hipótesis era adecuada; no acumular transformaciones sin evaluación independiente.

**Fuentes utilizadas.** [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md), [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [ML Design Patterns, caps. 3 y 7](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

**Documento relacionado.** [`knowledge/feature_engineering/feature_engineering.md`](../feature_engineering/feature_engineering.md).

### 4.8 Ajustar hiperparámetros

**Objetivo.** Afinar una alternativa ya funcional dentro de una evaluación estable y un presupuesto disponible.

**Preguntas.** ¿El parámetro puede afectar materialmente el resultado? ¿La validación es confiable? ¿Qué rango y método de búsqueda caben en el presupuesto? ¿El preprocesamiento también forma parte de las decisiones?

**Acciones respaldadas.** Ajustar solo después de contar con baseline y validación definida. Usar una búsqueda acotada: grid si el espacio discreto es pequeño, aleatoria si es más amplio; métodos informados pueden ser opción cuando cada trial es costoso y existe presupuesto. Incluir parámetros de transformadores en la evaluación conjunta cuando corresponda. Revisar resultados de CV y registrar todos los trials; no expandir búsquedas sobre el test final. [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [Müller, caps. 5–6](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md).

**Resultado esperado — síntesis práctica.** Configuración elegida con método de búsqueda, métrica, partición, espacio probado y coste registrados.

**Avanzar o volver.** Detener o reducir la búsqueda cuando las mejoras no compensen coste, variación o complejidad. Volver a etapas anteriores si el mejor valor queda en el borde del espacio probado, si la validación parece contaminada o si la métrica no refleja el objetivo. No hay un número universal de pruebas ni un umbral de parada en las fuentes.

**Fuentes utilizadas.** [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [Müller, caps. 5–6](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md).

**Documento relacionado.** [`knowledge/tuning/hyperparameter_tuning.md`](../tuning/hyperparameter_tuning.md).

### 4.9 Seleccionar y evaluar la solución final

**Objetivo.** Cerrar selección con datos de desarrollo y usar el test solo según su función prevista.

**Preguntas.** ¿La selección se basó en validación y en la métrica declarada? ¿Hay variabilidad entre folds o señales de sobreajuste? ¿Qué evalúa exactamente el test disponible? ¿Se conoce cómo generar la predicción de entrega?

**Acciones respaldadas.** Seleccionar la configuración mediante validación/cross-validation. Si existe un test reservado con etiquetas y está destinado a la evaluación final, usarlo una vez al cerrar decisiones; no reajustar mirando ese resultado. Géron describe reentrenar la opción elegida con los datos de entrenamiento disponibles antes de test. En una competición donde test no tiene etiquetas, generar predicciones con el procedimiento acordado y cumplir formato y reglas; no se puede usar su score oculto como validación local. Una sola puntuación no garantiza generalización. [Géron, caps. 1–3](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md).

**Resultado esperado — síntesis práctica.** Modelo y transformaciones seleccionados, estimación de validación, evaluación final si está disponible y artefacto de predicción conforme a las instrucciones del reto.

**Avanzar o volver.** Entregar si la evaluación sigue la política declarada y la salida cumple el contrato de la competición. Si el test final contradice fuertemente la validación, reportarlo e investigar sin convertir el test en una nueva fuente iterativa de selección.

**Fuentes utilizadas.** [Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md).

**Documento relacionado.** [`knowledge/validation/validation_strategy.md`](../validation/validation_strategy.md).

### 4.10 Registrar y comunicar resultados

**Objetivo.** Hacer inteligible qué se probó, por qué se eligió el resultado y bajo qué límites puede interpretarse.

**Preguntas.** ¿Se puede repetir la partición y el pipeline? ¿Qué métrica y score corresponden a cada experimento? ¿Qué decisiones, fallos y supuestos importan para la entrega?

**Acciones respaldadas.** Registrar semillas/particiones, métrica, transformaciones, configuración de modelo, resultados de validación y coste observado cuando se comparen opciones. Comunicar score junto con benchmark, método de evaluación, supuestos y limitaciones; si es pertinente, comparar segmentos para que un promedio no oculte fallos. **La tabla mínima es síntesis práctica** de Repeatable Splitting, Workflow Pipeline, Heuristic Benchmark y las listas de proyecto. [ML Design Patterns, caps. 6–7](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md), [ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md), [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md).

**Resultado esperado — síntesis práctica.** Registro breve que conecta la definición de tarea con los experimentos y el resultado entregado.

**Avanzar o volver.** Cerrar cuando otra persona pueda identificar la configuración ganadora, la evaluación usada y las limitaciones. Volver a ejecutar si falta información necesaria para reproducir el resultado.

**Fuentes utilizadas.** [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md), [ML Design Patterns, caps. 6–7](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

**Documento relacionado.** [`knowledge/validation/validation_strategy.md`](../validation/validation_strategy.md).

## 5. Puntos de decisión y rutas alternativas

| Hallazgo o condición | Decisión posible | Etapa afectada | Fuente o justificación |
|---|---|---|---|
| Target categórico o numérico | Formular clasificación o regresión y seleccionar una métrica correspondiente | 4.1, 4.3 | Explícito de Géron, cap. 1; Müller, cap. 5 |
| Target o uso no bien representado por una clasificación/regresión simple | Evaluar `Reframing` o una alternativa manteniendo el uso final y la métrica como criterio | 4.1, 4.7 | Patrón explícito y trade-offs en ML Design Patterns, cap. 3 |
| Clases desbalanceadas | Contrastar con predictor simple; inspeccionar matriz de confusión y métricas sensibles a clase; considerar estratificación | 4.3, 4.6 | Géron, cap. 3; Müller, cap. 5; Molin, cap. 9 |
| Filas de una misma entidad | Usar partición por grupos si la evaluación pretende generalizar a entidades nuevas | 4.3 | Müller, cap. 5 |
| Orden temporal o predicción hacia el futuro | Respetar el orden; no mezclar aleatoriamente sin justificarlo | 4.3 | Géron, cap. 3; Müller, cap. 5 |
| Faltantes o códigos centinela | Verificar su significado y comparar tratamientos de preparación; ajustar imputación con entrenamiento | 4.2, 4.4 | McKinney, cap. 7; Géron, cap. 2 |
| Agregaciones por grupo | Comprobar unidad, claves y disponibilidad; calcular estadísticas con training y aplicarlas después | 4.2, 4.4 | McKinney, cap. 10; síntesis de prevención de leakage |
| Selección o métrica sospechosamente alta | Auditar si se ajustaron transformaciones o selección fuera de cada fold | 4.4, 4.6 | Müller, cap. 6 |
| Validation muy variable o pocos datos | Considerar cross-validation con folds comunes; valorar el coste | 4.3, 4.6 | ISLP, cap. 5; Müller, cap. 5 |
| Modelo lento o recursos limitados | Reducir candidatos, usar una búsqueda acotada o muestreo exploratorio si es razonable; anotar el posible sesgo de una muestra | 4.6, 4.8 | Géron, apéndice B; ML Design Patterns, cap. 4 |
| No hay baseline comprensible | Definir una heurística contextual evaluada en datos comparables | 4.5 | Patrón Heuristic Benchmark, ML Design Patterns, cap. 7 |
| Score de test accesible repetidamente | Preservar el test para la evaluación prevista y no convertirlo en señal iterativa de tuning | 4.3, 4.8, 4.9 | Géron, caps. 1–2; Müller, cap. 5 |
| El catálogo o los resúmenes no cubren el problema con suficiente detalle | Registrar el límite y evitar completar con procedimientos externos | Cualquier etapa | Restricción de este documento y cobertura descrita en `source_catalog.md` |

## 6. Gestión de tiempo y presupuesto computacional

Las fuentes relacionan el coste con cantidad de datos, familia del modelo, folds y número de combinaciones. Cross-validation multiplica ajustes; búsqueda en rejilla puede crecer con el producto de candidatos y folds; búsqueda aleatoria permite limitar la exploración; ensembles o boosting pueden requerir más entrenamiento; optimización bayesiana se presenta como opción para trials costosos, no como garantía. [Géron, caps. 1–2 y apéndice B](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ML Design Patterns, cap. 4](../../sources/ml_design_patterns/ch04_model_training_patterns.md).

**Recomendación para coding tests — síntesis:** asegurar primero formulación, evaluación limpia, baseline y pipeline repetible; explorar después una shortlist pequeña y afinar solo una alternativa prometedora. Medir coste observado junto con score. Simplificar o detener cuando el coste adicional no esté justificado por evidencia útil. No se fijan minutos, cantidades de experimentos, umbrales de mejora ni reglas universales de parada porque las fuentes no los establecen para competiciones.

## 7. Reproducibilidad y registro de experimentos

Repeatable Splitting se ocupa de generar particiones estables que respeten correlaciones y eviten leakage; Workflow Pipeline integra pasos reproducibles; Transform procura que la representación de features sea coherente durante entrenamiento y evaluación. Para una competición tabular, se sintetizan estas prácticas en el siguiente registro mínimo. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md), [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md).

| Elemento que se registra | Por qué ayuda | Respaldo o tipo de contenido |
|---|---|---|
| Definición de tarea, unidad, target y métrica | Permite interpretar qué se estaba optimizando | Síntesis integrada; Géron, caps. 1–2; Müller, cap. 5 |
| Identificadores/regla de partición, grupos u orden y semilla si aplica | Permite repetir la evaluación y detectar un split incompatible | Explícito en Repeatable Splitting; síntesis adaptada a la competición |
| Pasos de preparación y dónde se ajustaron | Hace visible el flujo y permite auditar leakage | Explícito en Müller, cap. 6, y patrón Transform |
| Familia de modelo y parámetros | Identifica el candidato evaluado | Síntesis práctica de listas de comparación y tuning |
| Score de validación y variación entre folds, cuando corresponda | Evita presentar solo una medición como certeza | Explícito en ISLP, cap. 5, y Müller, cap. 5 |
| Tiempo/coste observado y notas de error | Ayuda a juzgar si la complejidad adicional fue útil | Recomendación para coding test derivada de los límites de coste descritos en las fuentes |
| Decisión de selección, score final y limitaciones | Hace auditable la conclusión y comunica su alcance | Síntesis de Géron, apéndice B, y Heuristic Benchmark |

Esto es una recomendación de registro concisa, no una especificación de tracking o plataforma. Las fuentes no determinan un formato de archivo ni una herramienta de seguimiento obligatoria.

## 8. Implementación mínima del flujo

El siguiente pseudocódigo conecta las decisiones sin presuponer dataset, métrica, librería o tipo de split. Es una **síntesis práctica** de los procedimientos de partición, evaluación y pipeline descritos en las fuentes; no introduce resultados ni código específico de plataforma.

```text
leer reglas, datos disponibles, target y métrica oficial
formular unidad de análisis y tarea
auditar columnas, tipos, calidad, grupos, tiempo y posibles fugas

definir evaluación de desarrollo que represente el uso previsto
reservar cualquier test etiquetado para su función final

separar X = predictores e y = target
construir benchmark simple usando solo los datos de entrenamiento

para cada candidato justificado:
    para cada split/fold definido:
        ajustar transformaciones solo con la porción de entrenamiento
        transformar la porción de validación con esos parámetros
        entrenar y calcular la métrica elegida
    registrar resultados, variación y coste

seleccionar según validación, métrica y restricciones
ajustar la solución elegida según el protocolo definido
evaluar test una sola vez si corresponde
guardar configuración, resultados y limitaciones
generar entrega conforme a las reglas de la competición
```

El orden exacto de creación del holdout y la auditoría depende del protocolo: algunos resúmenes recomiendan apartar test antes de la exploración detallada. La restricción esencial aquí es que las decisiones de transformación y selección no usen información de evaluación fuera de su papel. Si se implementa en Python, cualquier API concreta debe tratarse como adaptación moderna y seguir la semántica de la partición y el pipeline descritos por las fuentes; no se incluye código porque los resúmenes cubren patrones distintos y no hace falta imponer una implementación única.

## 9. Errores frecuentes y riesgos

| Error o riesgo | Tipo de respaldo | Implicación operativa |
|---|---|---|
| Elegir modelo sin aclarar tarea, target y medida | Explícito de Géron; síntesis | Volver a formular antes de interpretar scores |
| Transformar antes de entender valores, tipos o faltantes | Síntesis integrada de Géron y McKinney | Inspeccionar antes de alterar o eliminar observaciones |
| Usar métrica desconectada del objetivo o solo accuracy con desbalance | Explícito de Géron y Müller | Elegir métrica y errores relevantes antes de comparar |
| Evaluar en los mismos datos usados para ajustar | Explícito de Géron e ISLP | Usar datos fuera de muestra para comparar y evaluar |
| Ajustar transformaciones o selección antes de dividir/fuera del fold | Explícito de Müller; patrón Transform | Incorporar operaciones aprendidas en el pipeline de cada fold |
| Cambiar particiones entre candidatos | Explícito de ISLP y Repeatable Splitting | Reutilizar splits comunes para que la comparación sea interpretable |
| Tunear antes de un baseline o con validación inestable | Explícito de ML Design Patterns, cap. 4; síntesis | Asegurar primero referencia y evaluación repetible |
| Usar repetidamente el test para escoger configuración | Explícito de Géron y Müller | Mantener test para la evaluación final prevista |
| Presentar una puntuación como garantía de generalización | Conclusión práctica derivada de evaluación fuera de muestra | Comunicar método y límites junto con la puntuación |
| Perseguir pequeñas ganancias sin considerar coste y repetibilidad | Recomendación para coding tests derivada | Comparar valor de la mejora y coste observado; registrar experimentos |
| Tratar cualquier outlier como error o toda rareza como anomalía | Explícito de Géron y McKinney | Investigar contexto antes de eliminar/corregir |
| Elegir una familia, ensemble o baseline como ganador universal | Explícito de No Free Lunch y ejemplos comparativos | Probar candidatos razonables bajo la evaluación del problema |

## 10. Checklist operativo para una competición

- [ ] Comprendí la tarea, el objetivo y el uso de la predicción.
- [ ] Identifiqué la unidad de análisis y el target.
- [ ] Confirmé la métrica y el formato de evaluación oficial.
- [ ] Revisé estructura, tipos, calidad, distribución y faltantes.
- [ ] Busqué grupos, orden temporal y posibles señales de leakage.
- [ ] Definí una estrategia de evaluación coherente y repetible.
- [ ] Separé predictores y target cuando aplica.
- [ ] Ajusté transformaciones solo en entrenamiento y dentro de cada fold.
- [ ] Construí un benchmark o baseline acorde con la tarea.
- [ ] Comparé candidatos con la misma métrica y particiones.
- [ ] Apliqué mejoras solo con una hipótesis evaluable.
- [ ] Afiné hiperparámetros después del baseline y dentro del presupuesto disponible.
- [ ] Preservé el conjunto de prueba para la función que le corresponde.
- [ ] Registré decisiones, configuración, métricas, variación y limitaciones.
- [ ] Verifiqué que la solución produzca la salida que exige la competición.

Los puntos anteriores son un checklist adaptable: omitir los que no apliquen y anotar la razón cuando una condición (por ejemplo, grupos o tiempo) requiera otra ruta.

## 11. Relación con otros documentos de `knowledge/`

| Documento relacionado | Papel dentro del flujo |
|---|---|
| [`knowledge/problem_definition/problem_formulation.md`](../problem_definition/problem_formulation.md) | Define la tarea, el target y el objetivo |
| [`knowledge/eda/dataset_audit.md`](../eda/dataset_audit.md) | Organiza la inspección inicial del dataset |
| [`knowledge/preprocessing/data_cleaning.md`](../preprocessing/data_cleaning.md) | Trata problemas de calidad de datos |
| [`knowledge/preprocessing/data_leakage.md`](../preprocessing/data_leakage.md) | Previene fugas de información |
| [`knowledge/feature_engineering/feature_engineering.md`](../feature_engineering/feature_engineering.md) | Orienta la representación y construcción de variables |
| [`knowledge/validation/validation_strategy.md`](../validation/validation_strategy.md) | Define la estrategia de evaluación |
| [`knowledge/metrics/classification_metrics.md`](../metrics/classification_metrics.md) | Ayuda a elegir e interpretar métricas de clasificación |
| [`knowledge/metrics/regression_metrics.md`](../metrics/regression_metrics.md) | Ayuda a elegir e interpretar métricas de regresión |
| [`knowledge/models/tabular_models.md`](../models/tabular_models.md) | Compara familias de modelos tabulares |
| [`knowledge/ensembles/boosting_models.md`](../ensembles/boosting_models.md) | Desarrolla ensambles y boosting |
| [`knowledge/tuning/hyperparameter_tuning.md`](../tuning/hyperparameter_tuning.md) | Organiza la búsqueda de hiperparámetros |

Los once documentos enlazados existen en el workspace al momento de redactar este flujo. Se enlazan como guías relacionadas; la evidencia de este documento procede de `sources/`, no de dichos documentos.

## 12. Vacíos y limitaciones de las fuentes

- Los resúmenes cubren clasificación y regresión tabulares ampliamente, pero no proporcionan un procedimiento completo para todos los targets estructurados o tareas no supervisadas. En esas situaciones, formular alcance y evaluación requiere material específico.
- La cobertura de series temporales es parcial y no basta para describir de forma completa validación predictiva temporal, ventanas y restricciones propias de cada competición. La recomendación general de respetar el tiempo no reemplaza una estrategia temporal específica.
- Los resúmenes no conocen las reglas particulares de una plataforma: definición exacta de leaderboard, datos ocultos, formato de entrega, límites de ejecución o política de uso de test deben verificarse en el enunciado de cada reto.
- No hay presupuestos universales de tiempo, número de experimentos, mejora mínima ni parada. Se deben decidir según cómputo, estabilidad de validación y reglas disponibles, y registrar la decisión.
- `Reframing`, ensembles, boosting y búsquedas avanzadas tienen cobertura temática, pero no son pasos obligatorios ni ganadores universales; su utilidad debe demostrarse con una evaluación pertinente.
- El patrón Heuristic Benchmark ofrece opciones de referencia, pero la heurística correcta depende del dominio y del proceso existente. La traducción de score a utilidad puede requerir datos ausentes en una competición.
- Los resúmenes muestran ejemplos de API y adaptaciones modernas, pero no establecen una plantilla única de software para todo coding test. El pseudocódigo evita fijar una biblioteca o estrategia inadecuada.
- La reproducibilidad aquí se limita a particiones, transformaciones, configuración y registro de comparaciones. No se especifica una plataforma de seguimiento ni un proceso completo de producción.
- La selección de features, los ensembles y el tuning repetido pueden adaptarse a la validación; una evaluación final separada ayuda a controlar ese riesgo, pero no elimina incertidumbre ni garantiza desempeño fuera de muestra.

## 13. Fuentes utilizadas

Se incluyen únicamente resúmenes consultados y utilizados para elaborar este flujo.

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning*, cap. 1 | [ch01_ml_landscape.md](../../sources/geron_hands_on_ml/ch01_ml_landscape.md) | Tipos de tarea, generalización y separación entre selección y evaluación final |
| Géron, cap. 2 | [ch02_end_to_end_ml_project.md](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Flujo end-to-end, EDA, preparación, pipelines, comparación y evaluación |
| Géron, cap. 3 | [ch03_classification.md](../../sources/geron_hands_on_ml/ch03_classification.md) | Evaluación de clasificación, desbalance, métricas y consideración del orden temporal |
| Géron, apéndice B | [appendix_b_machine_learning_project_checklist.md](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md) | Checklist de proyecto, candidatos, tuning, test final y coste |
| Müller, cap. 4 | [ch04_feature_engineering.md](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md) | Representación y construcción de features |
| Müller, cap. 5 | [ch05_model_evaluation.md](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | Métricas, splits por clase/grupo, CV, selección y tuning |
| Müller, cap. 6 | [ch06_pipelines.md](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Pipelines, ajuste dentro de folds y prevención de leakage |
| McKinney, cap. 7 | [ch07_data_cleaning.md](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) | Faltantes, duplicados, categorías y decisiones de limpieza |
| McKinney, cap. 10 | [ch10_groupby_aggregation.md](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md) | Inspección y transformaciones por grupos; riesgos de fuga en estadísticas grupales |
| ISLP, cap. 5 | [ch05_resampling_methods.md](../../sources/islp_python/ch05_resampling_methods.md) | Resampling, selección comparable y coste/variación de evaluación |
| ISLP, cap. 8 | [ch08_tree_based_methods.md](../../sources/islp_python/ch08_tree_based_methods.md) | Árboles, ensembles y comparación sin ganador universal |
| Molin, cap. 3 | [ch03_data_wrangling.md](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) | Auditoría y preparación tabular |
| Molin, cap. 9 | [ch09_ml_with_sklearn.md](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md) | Baseline tabular, pipeline, separación de variables y métricas |
| Molin, cap. 10 | [ch10_model_optimization.md](../../sources/molin_hands_on_pandas/ch10_model_optimization.md) | Mejora controlada, tuning acotado, desbalance y coste |
| ML Design Patterns, cap. 3 | [ch03_problem_representation_patterns.md](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md) | Patrón Reframing y límites de cambiar la representación |
| ML Design Patterns, cap. 4 | [ch04_model_training_patterns.md](../../sources/ml_design_patterns/ch04_model_training_patterns.md) | Patrón Hyperparameter Tuning, métodos y trade-offs de coste |
| ML Design Patterns, cap. 6 | [ch06_reproducibility_patterns.md](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) | Transform, Repeatable Splitting y Workflow Pipeline |
| ML Design Patterns, cap. 7 | [ch07_responsible_ai_patterns.md](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md) | Heuristic Benchmark y comparación interpretable |

## 14. Trazabilidad

| Etapa o sección del flujo | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Géron, caps. 1–2; resumen integrado | Síntesis integrada |
| 2. Principios de trabajo | Géron, caps. 1–3 y apéndice B; Müller, caps. 5–6; Design Patterns, caps. 6–7 | Síntesis integrada |
| 3. Vista general | Géron, cap. 2 y apéndice B; ISLP, cap. 5; Müller, cap. 6 | Síntesis integrada |
| 4.1 Formular tarea y target | Géron, cap. 1; Design Patterns, cap. 3 | Explícito de la fuente; Reframing condicionado |
| 4.2 Auditar dataset | Géron, cap. 2; McKinney, caps. 7 y 10; Molin, cap. 3 | Explícito de la fuente |
| 4.3 Definir métrica y validación | Géron, caps. 1 y 3; Müller, cap. 5; ISLP, cap. 5 | Explícito de la fuente; síntesis integrada |
| 4.4 Preparar datos sin leakage | Géron, cap. 2; Müller, cap. 6; McKinney, caps. 7 y 10; Design Patterns, cap. 6 | Explícito de la fuente; síntesis integrada |
| 4.5 Benchmark y baseline | Design Patterns, cap. 7; Géron, apéndice B; Molin, cap. 9 | Explícito de la fuente; recomendación para coding test |
| 4.6 Comparar modelos | Géron, caps. 1–3; ISLP, cap. 8; Molin, caps. 9–10 | Explícito de la fuente; síntesis integrada |
| 4.7 Mejorar features/modelos | Müller, cap. 4; Géron, apéndice B; Design Patterns, cap. 3 | Explícito de la fuente; conclusión práctica derivada |
| 4.8 Ajustar hiperparámetros | Géron, apéndice B; Müller, caps. 5–6; Design Patterns, cap. 4 | Explícito de la fuente; recomendación para coding test |
| 4.9 Selección y evaluación final | Géron, caps. 1–3; Müller, cap. 5; ISLP, cap. 5 | Explícito de la fuente; síntesis integrada |
| 4.10 Registrar y comunicar | Géron, apéndice B; Design Patterns, caps. 6–7 | Síntesis integrada; recomendación para coding test |
| 5. Rutas alternativas | Géron, caps. 1–3; Müller, cap. 5; McKinney, cap. 10; Design Patterns, caps. 3, 6 y 7 | Explícito de la fuente; síntesis integrada |
| 6. Tiempo y presupuesto | Géron, caps. 1–2 y apéndice B; Müller, cap. 5; Design Patterns, cap. 4 | Explícito de la fuente; recomendación para coding test |
| 7. Reproducibilidad | Müller, cap. 6; Design Patterns, cap. 6; Géron, apéndice B | Explícito de la fuente; síntesis integrada |
| 8. Pseudocódigo mínimo | Géron, cap. 2; Müller, cap. 6; Design Patterns, cap. 6 | Conclusión práctica derivada |
| 9. Errores y riesgos | Géron, caps. 1–3; Müller, caps. 5–6; McKinney, cap. 7; ISLP, cap. 5; Design Patterns, cap. 4 | Explícito de la fuente; síntesis integrada |
| 10. Checklist | Géron, apéndice B; Müller, caps. 5–6; Molin, cap. 9 | Recomendación para coding test |
| 11. Documentos relacionados | Estructura existente de `knowledge/` | Relación documental; no es evidencia técnica |
| 12. Vacíos y limitaciones | `sources/source_catalog.md` y resúmenes citados | Limitación de cobertura |
| 13. Fuentes utilizadas | Resúmenes listados y consultados | Explícito de la fuente |
