---
title: "Estrategias de validación para Machine Learning"
knowledge_area: "validation"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Estrategias de validación para Machine Learning

## 1. Propósito y alcance

La validación estima cómo se comportará un procedimiento de aprendizaje ante observaciones que no se usaron para ajustarlo y permite comparar modelos, hiperparámetros, transformaciones y umbrales antes de la evaluación final. El objetivo es medir generalización, no premiar el ajuste sobre los datos de entrenamiento. **Explícito de las fuentes:** esta separación entre ajuste, selección y evaluación final aparece de forma consistente en los resúmenes de Géron, Müller y Guido, ISLP y Molin ([Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md), [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md)).

La estrategia no se elige de forma universal: debe representar la estructura de los datos y el uso futuro. Clases desbalanceadas, entidades repetidas, correlación entre filas u orden temporal cambian qué particiones son válidas. La estimación observada tampoco garantiza un rendimiento futuro idéntico: depende de que la muestra y las particiones sean representativas y puede variar por muestreo, distribución, drift o por el propio split ([Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

Este documento se concentra en decisiones de partición y comparación experimental. No es una guía exhaustiva de métricas ni de ajuste de hiperparámetros. Esos temas se remiten a los documentos relacionados de la sección 13.

## 2. Qué estima la validación

**Explícito de las fuentes.** Evaluar fuera de muestra significa ajustar el procedimiento con una parte de los datos y calcular su métrica en observaciones retenidas. En K-Fold, cada fold actúa una vez como validación y los demás como entrenamiento; el resultado es un score por fold. La media resume el rendimiento estimado y la dispersión muestra sensibilidad a la partición o escasez de datos. La validación cruzada evalúa un algoritmo o procedimiento sobre un dataset; por sí sola no entrega el modelo final ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).

El error de entrenamiento describe el ajuste a observaciones ya utilizadas y no estima por sí mismo la generalización. Una gran diferencia entre entrenamiento y validación puede ser señal de overfitting; un error alto incluso en entrenamiento puede ser compatible con underfitting. El error de prueba se calcula al final sobre un conjunto independiente que no intervino en selección ([ISLP, cap. 2](../../sources/islp_python/ch02_statistical_learning.md), [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)).

**Síntesis integrada.** La estimación solo responde a la pregunta representada por el split. Un split aleatorio estima generalización entre filas bajo supuestos de independencia; uno por grupos estima generalización a grupos retenidos; uno temporal aproxima el paso de pasado a futuro. Si esa pregunta no coincide con el uso esperado, una cifra técnicamente correcta puede ser operativamente engañosa.

## 3. Particiones de datos y su función

| Partición | Para qué se usa | Qué decisiones permite | Precaución |
|---|---|---|---|
| Entrenamiento | Ajustar parámetros del modelo y estadísticas de transformaciones | Construir cada candidato dentro del protocolo fijado | No incorporar información de validación o test al ajuste |
| Validación | Comparar modelos, hiperparámetros, features, métricas operativas o umbrales | Elegir entre alternativas antes del cierre del experimento | Reutilizarla muchas veces puede adaptar decisiones a esa partición; mantener un test final independiente |
| Prueba final | Estimar una vez el rendimiento del procedimiento ya elegido | Aceptar, rechazar o comunicar el resultado final | No usarla para tuning, selección de variables, umbrales ni retoques repetidos |

La tabla integra la separación train/validation/test documentada por [Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md) y [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

### 3.1 Entrenamiento

Contiene las observaciones con las que se aprenden los parámetros del modelo. También es el único origen válido para estadísticas aprendidas por imputación, escalado, codificación, selección de variables, embeddings o benchmarks basados en datos. En validación cruzada, “entrenamiento” cambia por fold: cada transformación debe reajustarse solo con los folds de entrenamiento ([Müller y Guido, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

### 3.2 Validación

Es el conjunto retenido para tomar decisiones de desarrollo. Puede ser un único holdout o una sucesión de folds. Permite comparar candidatos y, cuando corresponde, escoger hiperparámetros o umbrales sin tocar el test. Para que la comparación sea justa, los candidatos deben usar la misma estrategia, métrica y, de ser posible, los mismos folds ([ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md)).

### 3.3 Prueba final

Se mantiene aislada hasta cerrar modelo, preprocesamiento, hiperparámetros, features, umbral y protocolo. Después de la selección, el procedimiento elegido puede reajustarse con todos los datos de desarrollo disponibles y evaluarse una sola vez en test. Cambiar decisiones tras ver ese resultado destruye la independencia que justificaba la estimación ([Géron, cap. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).

## 4. Estrategias de validación

### 4.1 Holdout

**Cómo divide.** Realiza una única separación entre entrenamiento y validación. Si existe test final, este se reserva antes y el holdout se crea dentro de los datos de desarrollo.

**Qué estima.** El rendimiento del procedimiento ajustado con el subconjunto de entrenamiento sobre el bloque retenido. ISLP advierte que puede variar mucho según el split y que suele ser una estimación pesimista para el modelo que después se reajustará con todos los datos de desarrollo ([ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md)).

**Cuándo considerarlo.** Comprobación rápida, presupuesto computacional ajustado o ajuste muy costoso. En datasets grandes, una única división controlada puede ser suficiente para una primera comparación ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).

**Ventajas.** Es simple, barato y fácil de reproducir si se fija la aleatoriedad o se materializa una regla estable.

**Limitaciones y supuestos.** Depende de una sola asignación; reduce el número de observaciones disponibles para ajustar; supone que la separación representa el uso futuro. No protege por sí mismo contra grupos, tiempo, duplicados ni desbalance.

**Riesgo de leakage.** Ajustar transformaciones antes del split o tomar decisiones repetidas sobre el mismo holdout transfiere información de evaluación al desarrollo.

### 4.2 Validación cruzada K-Fold

**Cómo divide.** Separa los datos en `k` folds no solapados; cada fold se usa una vez como validación y los restantes como entrenamiento.

**Qué estima.** Produce `k` scores fuera de muestra; su media resume el rendimiento y su dispersión ayuda a observar sensibilidad al split. Al comparar candidatos deben reutilizarse los mismos folds ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md)).

**Cuándo considerarlo.** Evaluación y selección rutinarias cuando las observaciones pueden dividirse aproximadamente como unidades independientes y el coste de `k` ajustes es asumible.

**Ventajas.** Usa cada observación para validación una vez, reduce la dependencia de un único holdout y expone variación entre folds.

**Limitaciones y supuestos.** Multiplica el coste de entrenamiento y no modela grupos ni tiempo por defecto. El resumen de ISLP documenta 5 o 10 folds como compromiso habitual, pero este documento no los adopta como regla universal: el número depende de tamaño, coste y estructura.

**Riesgo de leakage.** Toda transformación aprendida debe vivir dentro del procedimiento ajustado en cada fold. K-Fold aleatorio no es adecuado cuando mezcla entidades relacionadas o información futura.

### 4.3 Stratified K-Fold

**Cómo divide.** Es una variante de K-Fold que conserva aproximadamente la proporción de clases en cada fold.

**Qué estima.** El rendimiento de clasificación a través de folds con composición de clases más comparable.

**Cuándo considerarlo.** Clasificación, en especial cuando las clases están ordenadas en los datos o tienen proporciones desiguales ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [Géron, cap. 3](../../sources/geron_hands_on_ml/ch03_classification.md)).

**Ventajas.** Reduce folds poco informativos y facilita que la clase minoritaria esté representada.

**Limitaciones y supuestos.** Estratificar no corrige una métrica mal elegida, no crea ejemplos nuevos y no resuelve dependencia entre filas. Si hay grupos o tiempo, preservar clases no autoriza a romper esas unidades.

**Riesgo de leakage.** El mismo que en K-Fold para transformaciones; además, usar la etiqueta para estratificar es parte de la partición, no una licencia para emplear información del fold en el ajuste.

### 4.4 Particiones repetibles

**Explícito de las fuentes — patrón `Repeatable Splitting`.** Una semilla estabiliza una ejecución aleatoria, pero puede no conservar asignaciones cuando el dataset crece ni garantizar equivalencia entre implementaciones. El patrón propone identificar la unidad de correlación, aplicar un hash determinista a una clave estable y asignar grupos completos a train, validation o test. Esto mantiene juntas filas relacionadas y estabiliza la comparación sin guardar copias completas de cada split ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

**Cuándo considerarlo.** Datasets crecientes, múltiples ejecuciones, comparación de modelos sobre una evaluación constante o presencia de personas, sesiones, fechas, dispositivos, ubicaciones o eventos repetidos.

**Ventajas.** Independencia del orden de filas, estabilidad al añadir datos y menor riesgo de contaminación entre observaciones correlacionadas.

**Limitaciones y condiciones.** Requiere una clave estable con cardinalidad suficiente y que represente la dependencia real. Deben verificarse las distribuciones de label y features; una clave equivocada puede mantener la reproducibilidad y aun así producir una evaluación inválida.

**Síntesis integrada.** Fijar `random_state`, persistir índices o usar hash estable son mecanismos distintos. La elección depende de si el dataset está congelado, crece o debe reproducirse entre sistemas.

### 4.5 Datos temporales

**Cómo divide.** Conserva el orden: se entrena con observaciones anteriores y se valida con observaciones posteriores. `TimeSeriesSplit` aparece en el resumen de Molin como herramienta para usar muestras sucesivas sin barajar el futuro con el pasado ([Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md)). `Repeatable Splitting` recomienda split secuencial cuando la inferencia futura, la autocorrelación o el comportamiento adaptativo forman parte del problema ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

**Qué estima.** Una aproximación al rendimiento al pasar de historia disponible a observaciones posteriores.

**Cuándo considerarlo.** Cuando existe un orden temporal que determina qué información estaría disponible al predecir o cuando el objetivo es extrapolar al futuro.

**Ventajas.** Evita que observaciones futuras entrenen un modelo evaluado sobre el pasado y permite revelar cambios temporales que un split aleatorio ocultaría.

**Limitaciones y supuestos.** Las fuentes permitidas no desarrollan con suficiente detalle tamaños de ventana, gaps, horizontes múltiples, ventanas expansivas frente a deslizantes ni backtesting completo. Tampoco ofrecen una regla universal para estacionalidad. Esas decisiones deben definirse según el momento real de predicción y validarse experimentalmente.

**Riesgo de leakage.** Barajar, calcular ventanas con información futura, rellenar hacia atrás o construir joins/features temporales sin respetar el tiempo del evento puede filtrar futuro ([McKinney, cap. 11](../../sources/mckinney_python_data_analysis/ch11_time_series.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

### 4.6 Datos agrupados o relacionados

**Cómo divide.** `GroupKFold` asigna grupos completos, no filas, a los folds. Todas las observaciones de una entidad quedan en entrenamiento o validación, pero no en ambos.

**Qué estima.** Generalización hacia grupos retenidos cuando la unidad real de generalización es una entidad nueva.

**Cuándo considerarlo.** Varias filas por paciente, persona, hablante u otra entidad relacionada ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)). `Repeatable Splitting` extiende el principio a fechas, dispositivos, ubicaciones y eventos correlacionados ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

**Ventajas.** Reduce la fuga causada por ejemplos altamente relacionados a ambos lados del split.

**Limitaciones y supuestos.** Exige grupos fiables y suficientes para repartir. La variable `groups` identifica entidades, no clases. Si la unidad elegida no coincide con la generalización deseada, el score responde a otra pregunta.

**Riesgo de leakage.** Dividir por fila, por hash de fila completa o por una clave que no captura la relación semántica puede dejar duplicados o vecinos correlacionados en ambos lados.

## 5. Cómo elegir una estrategia

| Estructura o condición de los datos | Estrategia que puede considerarse | Motivo | Precaución |
|---|---|---|---|
| Filas aproximadamente independientes; coste bajo o medio | K-Fold | Reduce dependencia de un solo split y muestra variación | Confirmar que no hay grupos ni tiempo ocultos |
| Ajuste caro o dataset grande | Holdout controlado o `ShuffleSplit` con presupuesto acotado | Reduce número o tamaño de los ajustes | Un único resultado puede ser inestable; fijar la aleatoriedad si se baraja |
| Dataset pequeño y modelo barato | Más folds o Leave-One-Out | Usa más observaciones por ajuste | Coste alto; LOOCV puede tener mayor varianza que K-Fold con `k<n` |
| Clasificación con clases desiguales | Stratified K-Fold o holdout estratificado | Conserva aproximadamente proporciones de clase | No sustituye una métrica adecuada ni resuelve grupos |
| Entidades repetidas y objetivo de generalizar a entidades nuevas | GroupKFold o asignación estable por grupo | Mantiene la entidad en una sola partición | Verificar número, tamaño y significado de los grupos |
| Orden temporal relevante | Split secuencial o `TimeSeriesSplit` | Respeta disponibilidad de información y evalúa hacia adelante | No barajar; auditar ventanas, rellenos y joins |
| Dataset creciente con clave estable | `Repeatable Splitting` mediante hash de grupo | Mantiene asignaciones estables al incorporar datos | Validar cardinalidad y distribuciones resultantes |
| Experimentos repetidos | Mismos índices/folds, semilla o regla versionada | Separa el efecto del modelo del ruido de partición | Una seed no corrige un split conceptualmente incorrecto |
| Baseline sin contexto | `Heuristic Benchmark` en el mismo split | Establece un suelo interpretable | Estimar la heurística solo con training y no confundir superar una regla débil con utilidad |

**Conclusión práctica derivada.** Elegir primero la unidad que no debe cruzar particiones —fila, entidad o tiempo— y después decidir entre holdout y múltiples folds. La necesidad de reproducibilidad no debe imponerse sobre la validez estructural: un split incorrecto pero determinista sigue siendo incorrecto.

## 6. Reproducibilidad y comparación de experimentos

Una comparación justa mantiene constantes el conjunto de evaluación, la métrica, el preprocesamiento permitido, la definición del target y la unidad de split. ISLP recomienda usar folds comunes al comparar niveles de flexibilidad; Müller y Guido piden la misma partición y criterio entre pipelines ([ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md), [Müller y Guido, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)).

El patrón **`Repeatable Splitting`** amplía esta idea: identificar correlaciones, usar una clave estable, versionar la regla y comprobar distribuciones. Esto es más fuerte que fijar una semilla cuando los datos crecen o la misma partición debe reproducirse entre lenguajes ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

El patrón **`Workflow Pipeline`** hace explícitas las etapas, entradas, salidas, dependencias, artefactos y lineage. Para un coding test no exige infraestructura: una secuencia determinista `prepare → split → transform → train → evaluate` y un registro de configuración materializan su aporte esencial. En flujos repetidos o multiusuario, el patrón documenta que el DAG y los artefactos versionados permiten repetir y comparar ejecuciones ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).

El patrón **`Heuristic Benchmark`** exige calcular la referencia con training y evaluarla con la misma métrica y sobre el mismo conjunto que el modelo. Así se detectan pipelines rotos o complejidad que no supera una regla comprensible ([ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md)).

Registro mínimo recomendado como **conclusión práctica derivada**: versión o regla del split, columnas de grupo/tiempo, semilla cuando aplique, métrica, pipeline, parámetros y scores por fold. Las fuentes respaldan registrar splits, configuración, artefactos y lineage; el formato concreto depende del proyecto.

## 7. Prevención de data leakage durante la validación

**Reglas explícitas respaldadas:**

- Ajustar imputación, escalado, codificación, selección o extracción de variables solo con el entrenamiento de cada fold.
- Entregar el pipeline completo a la validación cruzada para que transforme validación sin reajustar con ella.
- Mantener entidades correlacionadas dentro de una única partición.
- En tiempo, evitar barajar futuro con pasado y evitar features, rellenos o joins que incorporen información no disponible en el momento de predicción.
- Estimar estadísticas de un `Heuristic Benchmark` solo con training.
- Mantener test fuera de selección de modelos, tuning y elección de umbral.

Estas reglas están documentadas en [Müller y Guido, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) y [ML Design Patterns, cap. 7](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md).

**Síntesis práctica.** Auditar leakage significa revisar no solo el estimador, sino el origen temporal y grupal de cada feature y cada estadística. Un pipeline evita fuga entre folds para transformaciones encapsuladas, pero no corrige una columna construida previamente con futuro ni una clave de grupo mal definida.

El tratamiento general y detallado debe mantenerse en `knowledge/preprocessing/data_leakage.md`.

## 8. Implementación mínima en Python

Los ejemplos usan nombres genéricos y no prescriben un número universal de folds.

### K-Fold reproducible con transformación dentro de cada fold

```python
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

validation_scheme = KFold(
    n_splits=number_of_folds,
    shuffle=True,
    random_state=random_seed,
)
model_pipeline = make_pipeline(StandardScaler(), estimator)

results = cross_validate(
    model_pipeline,
    features,
    target,
    cv=validation_scheme,
    scoring=evaluation_metric,
)
```

Nota práctica: adaptación moderna de la idea descrita en la fuente.

La estructura se apoya en [ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md) y [Müller y Guido, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md). `number_of_folds`, `random_seed`, `estimator` y `evaluation_metric` deben definirse según el problema.

### Clasificación estratificada

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

validation_scheme = StratifiedKFold(
    n_splits=number_of_folds,
    shuffle=True,
    random_state=random_seed,
)
scores = cross_val_score(
    model_pipeline,
    features,
    target,
    cv=validation_scheme,
    scoring=evaluation_metric,
)
```

Nota práctica: adaptación moderna de la idea descrita en la fuente.

El barajado solo es apropiado si el orden no contiene dependencia que deba conservarse ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [Géron, cap. 3](../../sources/geron_hands_on_ml/ch03_classification.md)).

### Validación por grupos

```python
from sklearn.model_selection import GroupKFold, cross_val_score

validation_scheme = GroupKFold(n_splits=number_of_folds)
scores = cross_val_score(
    model_pipeline,
    features,
    target,
    groups=group_identifiers,
    cv=validation_scheme,
    scoring=evaluation_metric,
)
```

Nota práctica: adaptación moderna de la idea descrita en la fuente.

`group_identifiers` representa entidades relacionadas, no clases ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).

### Validación temporal mínima

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

validation_scheme = TimeSeriesSplit(n_splits=number_of_splits)
scores = cross_val_score(
    model_pipeline,
    time_ordered_features,
    time_ordered_target,
    cv=validation_scheme,
    scoring=evaluation_metric,
)
```

Nota práctica: adaptación moderna de la idea descrita en la fuente.

Este ejemplo refleja únicamente el orden sucesivo respaldado por [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md). No resuelve automáticamente gaps, horizontes, estacionalidad ni features construidas con futuro.

## 9. Reglas de decisión

| Situación | Estrategia que puede considerarse | Por qué | Qué verificar antes de usarla |
|---|---|---|---|
| Se necesita una comprobación rápida | Holdout reproducible | Un ajuste por candidato y bajo coste | Representatividad, estabilidad y ausencia de dependencias |
| Un holdout cambia la decisión al variar la semilla | K-Fold | Reduce dependencia de una sola separación | Coste y validez del supuesto de independencia |
| La clasificación tiene proporciones desiguales | Stratified K-Fold | Mantiene proporciones aproximadas por fold | Métrica, soporte de la minoría y grupos ocultos |
| Varias filas pertenecen a la misma entidad | GroupKFold o split estable por grupo | Evita compartir entidad entre ajuste y evaluación | Que el grupo sea la unidad real de generalización |
| La predicción ocurre sobre el futuro | Split secuencial o `TimeSeriesSplit` | Mantiene el sentido temporal | Orden, momento de disponibilidad, ventanas, joins y drift |
| El dataset se actualiza y deben conservarse asignaciones | `Repeatable Splitting` | Hash estable por clave o grupo | Clave inmutable, cardinalidad y distribución |
| Se ajustan transformaciones o features | Pipeline dentro del splitter | Aprende cada transformación solo con train | Que ninguna feature externa ya contenga información futura o global |
| Se comparan modelos o hiperparámetros | Mismos folds y misma métrica | Reduce ruido de partición en la comparación | Que test final permanezca fuera del proceso |
| La métrica carece de contexto | `Heuristic Benchmark` en la misma evaluación | Aporta un suelo interpretable | Calcular la referencia solo con training y mantenerla simple |

Todas las reglas son condicionales. Ninguna convierte holdout, K-Fold o estratificación en solución universal.

## 10. Errores frecuentes y riesgos

- **Explícito:** evaluar sobre los mismos datos usados para ajustar mide ajuste, no generalización ([ISLP, cap. 2](../../sources/islp_python/ch02_statistical_learning.md)).
- **Explícito:** seleccionar modelos, parámetros o umbrales con test produce una estimación optimista ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).
- **Explícito:** interpretar un único holdout como estable ignora la variación entre splits ([ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md)).
- **Explícito:** comparar candidatos con particiones distintas mezcla el efecto del modelo con ruido de partición ([ISLP, cap. 5](../../sources/islp_python/ch05_resampling_methods.md)).
- **Explícito:** ajustar transformaciones fuera de cada fold causa leakage ([Müller y Guido, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)).
- **Explícito:** usar K-Fold aleatorio cuando existen grupos o tiempo puede mezclar entidades relacionadas o información futura ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).
- **Explícito:** estratificar conserva proporciones, pero no corrige una métrica inadecuada ni la dependencia entre observaciones ([Müller y Guido, cap. 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).
- **Explícito:** aplicar remuestreo para desbalance fuera de los folds o antes de separar test contamina la evaluación ([Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md)).
- **Explícito:** una seed no basta para datos crecientes o grupos correlacionados; la lógica del split también debe ser estable y adecuada ([ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md)).
- **Síntesis integrada:** elegir una estrategia que no refleja el uso esperado responde con precisión a la pregunta equivocada.
- **Conclusión práctica derivada:** una mejora grande e inesperada después de selección de variables, agregaciones o features temporales debe activar una auditoría de leakage antes de celebrarse.

## 11. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Identificar el target, la métrica y la unidad de generalización: fila, entidad o tiempo.
2. Revisar IDs repetidos, sesiones, usuarios, fechas, duplicados, desbalance y orden de llegada.
3. Reservar el test final o equivalente local antes de explorar alternativas.
4. Elegir holdout, K-Fold, estratificación, grupos o tiempo según la estructura observada; registrar el motivo.
5. Fijar la partición mediante índices, semilla o regla estable, según el tipo de dataset.
6. Construir un `Heuristic Benchmark` y un baseline de modelo; evaluarlos con la misma métrica y el mismo split.
7. Encapsular transformaciones aprendidas en un pipeline y comparar cada cambio sobre folds comunes.
8. Registrar configuración, scores por fold, media, dispersión y coste suficiente para reproducir la decisión.
9. Cerrar features, modelo, hiperparámetros y umbral antes de consultar el test final.
10. Comunicar tanto el resultado como las limitaciones del split y los casos que no cubre.

En una competición, un leaderboard no sustituye una validación local representativa. Esta es una recomendación para datathons derivada de mantener test fuera de las decisiones repetidas; no se atribuye como afirmación literal de los libros.

## 12. Checklist operativo

- [ ] Definí qué representa una observación nueva en producción.
- [ ] Revisé si los datos tienen grupos, orden temporal, duplicados o dependencia.
- [ ] Revisé balance de clases y soporte por partición cuando corresponde.
- [ ] Elegí una estrategia compatible con la estructura observada.
- [ ] Separé claramente entrenamiento, validación y prueba final.
- [ ] Hice reproducible la partición con un mecanismo apropiado.
- [ ] Mantuve la misma estrategia, folds y métrica al comparar experimentos.
- [ ] Ajusté transformaciones y remuestreo solo dentro de la partición de entrenamiento.
- [ ] Comparé el modelo con un benchmark calculado únicamente con training.
- [ ] Evité usar el conjunto de prueba para decisiones repetidas.
- [ ] Registré la estrategia, su motivo, sus parámetros y sus limitaciones.

## 13. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con validación |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Aporta el tipo de problema y el momento de predicción |
| `knowledge/eda/dataset_audit.md` | Identifica grupos, tiempo, desbalance y estructura del dataset |
| `knowledge/preprocessing/data_cleaning.md` | Coordina los tratamientos dentro de las particiones |
| `knowledge/preprocessing/data_leakage.md` | Desarrolla la prevención de fuga durante preparación y evaluación |
| `knowledge/metrics/classification_metrics.md` | Define métricas para clasificación |
| `knowledge/metrics/regression_metrics.md` | Define métricas para regresión |
| `knowledge/tuning/hyperparameter_tuning.md` | Explica cómo ajustar hiperparámetros usando una estrategia de evaluación |
| `knowledge/workflows/competition_workflow.md` | Integra validación y evaluación en el flujo de competición |

La tabla expresa relaciones conceptuales solicitadas. Este documento no presupone que todos esos archivos ya existan ni desarrolla aquí su contenido.

## 14. Vacíos y limitaciones de las fuentes

- La validación temporal está cubierta a nivel de orden sucesivo, riesgo de futuro y uso de `TimeSeriesSplit`, pero no se desarrollan suficientemente gaps, embargo, horizontes múltiples, ventanas expansivas/deslizantes ni backtesting detallado.
- No hay una guía integrada para combinar simultáneamente grupos y estratificación; las fuentes advierten que estratificar no debe romper grupos, pero no especifican una API o algoritmo universal.
- No se define un número universal de folds ni una proporción universal de train/validation/test. Los valores mostrados en ejemplos pertenecen a contextos concretos.
- `Repeatable Splitting` describe hashing por grupo y condiciones de cardinalidad, pero la clave correcta solo puede determinarse con conocimiento del dataset y del uso esperado.
- La estabilidad estadística con muy pocos grupos, clases extremadamente raras o fuerte drift requiere validación experimental adicional.
- Las APIs de Python pueden cambiar; los ejemplos son adaptaciones modernas de ideas respaldadas, no transcripciones normativas de una versión concreta.
- La selección de métrica, el tuning exhaustivo, el tratamiento completo del desbalance y el análisis detallado de leakage quedan fuera del alcance y deben resolverse en sus documentos específicos.
- La validación cruzada anidada y Leave-One-Out aparecen en las fuentes, pero no se desarrollan como estrategias principales aquí porque la estructura obligatoria se centra en holdout, K-Fold, estratificación, repetibilidad, tiempo y grupos.
- No se cubren de forma suficiente otros datos estructurados como redes, jerarquías complejas, datos espaciales o múltiples dependencias cruzadas; no se añaden reglas externas.

## 15. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning*, cap. 1 | [Resumen](../../sources/geron_hands_on_ml/ch01_ml_landscape.md) | Generalización, holdout, validación repetida y separación del test |
| Géron, cap. 2 | [Resumen](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Split reproducible/estratificado, K-Fold, pipeline y evaluación final |
| Géron, cap. 3 | [Resumen](../../sources/geron_hands_on_ml/ch03_classification.md) | Stratified K-Fold, desbalance, scores fuera de muestra y límites con grupos/tiempo |
| Müller y Guido, cap. 5 | [Resumen](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | K-Fold, Stratified K-Fold, GroupKFold, ShuffleSplit, train/validation/test y métricas |
| Müller y Guido, cap. 6 | [Resumen](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Pipelines y prevención de leakage dentro de folds |
| ISLP, cap. 2 | [Resumen](../../sources/islp_python/ch02_statistical_learning.md) | Diferencia entre error de entrenamiento y error fuera de muestra |
| ISLP, cap. 5 | [Resumen](../../sources/islp_python/ch05_resampling_methods.md) | Holdout, K-Fold, LOOCV, variabilidad, folds comunes y coste |
| McKinney, cap. 11 | [Resumen](../../sources/mckinney_python_data_analysis/ch11_time_series.md) | Orden temporal y riesgos de leakage en rellenos, ventanas y features |
| Molin, cap. 9 | [Resumen](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md) | Split antes de transformar, estratificación y reproducibilidad |
| Molin, cap. 10 | [Resumen](../../sources/molin_hands_on_pandas/ch10_model_optimization.md) | TimeSeriesSplit, tuning sin test, coste, desbalance y remuestreo dentro de folds |
| *Machine Learning Design Patterns*, cap. 6 | [Resumen](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) | `Repeatable Splitting`, `Workflow Pipeline`, grupos, tiempo, hashing, lineage y leakage |
| *Machine Learning Design Patterns*, cap. 7 | [Resumen](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md) | `Heuristic Benchmark` comparable y calculado solo con training |

## 16. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Géron 1; Müller 5; ISLP 5; Molin 10 | Síntesis integrada |
| 2. Qué estima la validación | Müller 5; ISLP 2; Géron 2 | Explícito de la fuente / Síntesis integrada |
| 3. Particiones de datos y su función | Géron 1–2; Müller 5–6; ISLP 5 | Síntesis integrada |
| 4.1 Holdout | ISLP 5; Müller 5 | Explícito de la fuente |
| 4.2 K-Fold | Müller 5; ISLP 5 | Explícito de la fuente |
| 4.3 Stratified K-Fold | Müller 5; Géron 3 | Explícito de la fuente |
| 4.4 Particiones repetibles | ML Design Patterns 6; Géron 2 | Explícito de la fuente / Síntesis integrada |
| 4.5 Datos temporales | Molin 10; ML Design Patterns 6; McKinney 11 | Síntesis integrada |
| 4.6 Datos agrupados | Müller 5; ML Design Patterns 6 | Síntesis integrada |
| 5. Cómo elegir una estrategia | Müller 5; ISLP 5; ML Design Patterns 6–7 | Conclusión práctica derivada |
| 6. Reproducibilidad y comparación | ISLP 5; Müller 6; ML Design Patterns 6–7 | Síntesis integrada |
| 7. Prevención de leakage | Müller 5–6; ML Design Patterns 6–7 | Síntesis integrada |
| 8. Implementación mínima | ISLP 5; Müller 5–6; Molin 10 | Conclusión práctica derivada |
| 9. Reglas de decisión | Fuentes técnicas de las secciones 4–8 | Conclusión práctica derivada |
| 10. Errores y riesgos | ISLP 2 y 5; Müller 5–6; Molin 10; ML Design Patterns 6 | Síntesis integrada |
| 11. Coding test o datathon | Conjunto de fuentes utilizadas | Recomendación para coding test |
| 12. Checklist operativo | Conjunto de fuentes utilizadas | Conclusión práctica derivada |
| 13. Relación con otros documentos | Estructura solicitada | Síntesis integrada |
| 14. Vacíos y limitaciones | Alcance observado en los resúmenes consultados | Síntesis integrada |
| 15. Fuentes utilizadas | Resúmenes leídos | Explícito de la fuente |
