---
title: "Formulación del problema de Machine Learning"
knowledge_area: "problem_definition"
status: "draft"
version: "1.0"
created_from:
  - "sources/geron_hands_on_ml/"
  - "sources/muller_introduction_ml_python/"
  - "sources/mckinney_python_data_analysis/"
  - "sources/islp_python/"
  - "sources/molin_hands_on_pandas/"
  - "sources/ml_design_patterns/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Formulación del problema de Machine Learning

## 1. Propósito

Este documento convierte una necesidad de negocio o de análisis en una tarea de Machine Learning evaluable. La formulación debe conectar la tarea, los datos disponibles y una medida de rendimiento: definir qué debe aprenderse, sobre qué observaciones y cómo se sabrá si funciona. Elegir un algoritmo antes de resolver estas preguntas puede optimizar una tarea distinta de la necesaria. [Fuente: panorama de ML](../../sources/geron_hands_on_ml/ch01_ml_landscape.md) y [proyecto end-to-end](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

**Síntesis integrada.** Una formulación queda operativa cuando especifica objetivo, decisión apoyada, unidad de análisis, target, predictores disponibles en inferencia, momento y horizonte de predicción, salida, restricciones, métrica, baseline y protocolo de evaluación.

## 2. Preguntas que debe responder la formulación

- ¿Qué resultado real se quiere mejorar y qué decisión utilizará la predicción?
- ¿Qué representa una fila y cuál es la entidad o evento sobre el que se predice?
- ¿Existe una respuesta observable (`y`) o solo se busca estructura en `X`?
- ¿El target es una clase, varias clases simultáneas, una cantidad continua, un evento raro o un valor futuro?
- ¿Qué datos existirán exactamente cuando se produzca la predicción?
- ¿Qué horizonte separa la información de entrada del resultado que se quiere anticipar?
- ¿Se necesita una clase, probabilidad, valor puntual, intervalo, distribución, grupo o alerta?
- ¿Qué errores son más costosos y qué métrica representa ese coste?
- ¿Qué restricciones físicas, temporales, legales, de latencia, cómputo o cobertura condicionan la solución?
- ¿Qué referencia simple debe superar el modelo y con qué split se compararán ambos?

Estas preguntas integran el encuadre negocio–modelo, la distinción entre predicción e inferencia y la definición de tipos de aprendizaje descritos en [Géron ch. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Géron ch. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) e [ISLP ch. 2](../../sources/islp_python/ch02_statistical_learning.md).

## 3. Diferencia entre problema de negocio y problema de Machine Learning

El problema de negocio expresa un resultado o una decisión: reducir defectos, priorizar casos, estimar demanda o detectar actividad anómala. El problema de ML expresa una tarea aprendible: entradas, target, salida y criterio de evaluación. Las fuentes advierten que equipos distintos pueden optimizar objetivos incompatibles y que la métrica técnica no es intercambiable con la utilidad del sistema. Si los falsos negativos cuestan más que los falsos positivos, esa asimetría debe influir en el criterio del modelo. [Fuente: desafíos y objetivos múltiples](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

**Conclusión práctica derivada.** La traducción es correcta solo si una mejora medible del modelo puede vincularse con una mejor decisión. Un proxy fácil de etiquetar —por ejemplo, clic— puede no representar el resultado real —por ejemplo, utilidad o tiempo visto—. [Fuente: `Reframing`](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

## 4. Componentes del problema

### 4.1 Objetivo del proyecto

Describe el resultado deseado, quién consume la salida y para qué. Debe anteceder a la elección del algoritmo y de la métrica. El caso end-to-end muestra que el uso posterior de la predicción puede cambiar incluso la elección entre regresión y clasificación. [Fuente](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

### 4.2 Unidad de análisis

Es la entidad, evento o intervalo representado por una observación: cliente, transacción, distrito, dispositivo-hora. Debe ser compatible con el target y con la decisión. La inspección debe aclarar qué representa cada fila, columna, unidad y rango válido. [Fuente: data wrangling](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md).

### 4.3 Variable objetivo

La variable objetivo (`y`, response o label) es lo que se quiere estimar. En supervisado debe existir para entrenamiento y evaluación; en no supervisado no hay una respuesta correcta conocida. Un target truncado, inconsistente, subjetivo o basado en un proxy cambia lo que el modelo puede aprender. [Fuentes: proyecto end-to-end](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [aprendizaje estadístico](../../sources/islp_python/ch02_statistical_learning.md) y [calidad de datos](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

### 4.4 Variables predictoras

Las variables predictoras (`X`, features) describen la observación antes de producir la salida. Deben ser relevantes, disponibles en inferencia y procesables de la misma forma en entrenamiento y uso real. Una asociación puede ser útil para predicción sin demostrar causalidad; correlación lineal tampoco captura toda relación posible. [Fuentes](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) y [ISLP](../../sources/islp_python/ch02_statistical_learning.md).

### 4.5 Momento de predicción

Fija el instante de corte: todo predictor debe conocerse entonces. La calidad temporal incluye la latencia entre evento e incorporación al dataset; conservar ambos timestamps ayuda a evitar usar información que solo apareció después. [Fuente](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

### 4.6 Horizonte de predicción

Es la distancia entre el corte de información y el resultado futuro. **Síntesis integrada:** debe declararse porque modifica el target, la disponibilidad de features y el split. Las fuentes cubren operaciones temporales, drift y prevención de fuga, pero no presentan una metodología completa de forecasting. [Fuentes](../../sources/mckinney_python_data_analysis/ch11_time_series.md), [drift](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

### 4.7 Restricciones operativas

Incluyen latencia, memoria, coste de entrenamiento, actualización batch u online, rango físico de salida, datos sensibles, cobertura de grupos y capacidad operativa. La escala afecta ingesta, training y serving; añadir complejidad sin capacidad para operarla es un riesgo. [Fuentes](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [desafíos de diseño](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

### 4.8 Tipo de salida esperada

Debe distinguirse entre etiqueta, probabilidades, valor puntual, distribución, grupo o alerta. `predict()` y `predict_proba()` responden preguntas diferentes; `Reframing` muestra que representar un continuo mediante probabilidades por intervalos puede preservar incertidumbre que una media oculta. [Fuentes](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md), [`Reframing`](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

## 5. Selección del tipo de problema

| Señal del problema | Tipo de problema | Variable objetivo | Ejemplo de salida | Precaución |
|---|---|---|---|---|
| Dos clases excluyentes | Clasificación binaria | Label con dos valores | Clase y/o probabilidad positiva | Umbral y coste de FP/FN deben representar la decisión |
| Más de dos clases excluyentes | Clasificación multiclase | Un label por fila | Una clase o probabilidades por clase | Accuracy puede ocultar clases minoritarias |
| Varias etiquetas pueden coexistir | Clasificación multilabel | Vector de indicadores | Conjunto de etiquetas | No tratar las etiquetas como mutuamente excluyentes |
| Respuesta cuantitativa | Regresión | Valor continuo | Estimación numérica | Revisar rango, outliers y coste de errores grandes |
| No hay labels y se buscan grupos | Clustering | No existe `y` supervisada | Identificador de grupo | Los grupos requieren interpretación posterior |
| Se buscan desviaciones del patrón normal | Detección de anomalías | Puede no existir label completo | Score o alerta | Un caso raro puede ser válido, no necesariamente un error |
| Se predice un valor futuro con orden temporal | Forecasting | Valor/evento posterior al corte | Valor futuro | Las fuentes no desarrollan un flujo completo; evitar splits aleatorios incompatibles con el tiempo |

Clasificación, regresión, clustering y anomalías están explícitamente descritos en [Géron ch. 1](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [Molin ch. 9](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md) y [Géron ch. 9](../../sources/geron_hands_on_ml/ch09_unsupervised_learning.md). Multilabel se desarrolla en [Géron ch. 3](../../sources/geron_hands_on_ml/ch03_classification.md). **Limitación:** ranking aparece como posible consumo por cercanía dentro de `Reframing`, pero no está suficientemente desarrollado como tarea autónoma.

## 6. Flujo para formular un problema de Machine Learning

1. Comprender el objetivo de negocio y quién consumirá el resultado.
2. Identificar la decisión que la salida debe apoyar.
3. Definir la unidad de análisis.
4. Declarar qué representa cada observación, campo, unidad y timestamp.
5. Definir el target y separar target real de proxies.
6. Verificar que el target pueda observarse, medirse y obtenerse con calidad suficiente.
7. Identificar únicamente variables disponibles antes del momento de predicción.
8. Seleccionar supervisado/no supervisado y el tipo de tarea según la salida.
9. Definir si se necesita clase, probabilidad, cantidad, distribución, grupo o alerta.
10. Elegir una métrica coherente con el coste de errores y el uso final.
11. Revisar calidad, cobertura, temporalidad, escala y restricciones operativas.
12. Establecer una heurística, constante o proceso vigente como baseline.
13. Confirmar que baseline y modelo pueden compararse con el mismo split y métrica.

Los pasos 1, 7, 9 y 10 se apoyan en [el proyecto end-to-end](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); 3 y 4 en [data wrangling](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md); 5, 8 y 9 en [tipos de aprendizaje](../../sources/geron_hands_on_ml/ch01_ml_landscape.md) y [`Reframing`](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md); 12 y 13 en [Heuristic Benchmark](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md).

## 7. Formulación del objetivo y de la variable objetivo

Convertir una necesidad general en tarea predictiva exige definir una respuesta observable y una salida consumible. `y` no es cualquier columna correlacionada: es la representación operacional del resultado que se pretende anticipar. `X` contiene información disponible antes de producir esa respuesta. Predicción busca bajo error fuera de muestra; inferencia busca comprender asociaciones, dirección y magnitud. Clasificar asigna etiquetas; estimar produce una magnitud; decidir usa esas salidas junto con costes y restricciones. [Fuente](../../sources/islp_python/ch02_statistical_learning.md).

| Elemento | Pregunta de control | Ejemplo correcto | Riesgo |
|---|---|---|---|
| Objetivo | ¿Qué resultado y decisión se quieren mejorar? | Estimar una cantidad para planificar capacidad | Optimizar una métrica sin utilidad operativa |
| Unidad | ¿Sobre qué entidad/evento se predice? | Una fila por transacción al autorizarla | Mezclar granularidades o duplicar entidades |
| Target | ¿Qué resultado observable representa el objetivo? | Clase de fraude confirmada después | Proxy ambiguo, label incorrecto o truncado |
| Predictores | ¿Existirán al predecir? | Datos capturados antes de autorizar | Información futura o posterior al outcome |
| Corte temporal | ¿Hasta qué instante se permite información? | Campos disponibles al inicio del evento | Target leakage por timestamps tardíos |
| Salida | ¿Qué consume el sistema? | Probabilidad para aplicar un umbral | Entregar una clase cuando se requiere graduar riesgo |

**Conclusión práctica derivada.** Antes de modelar, auditar cobertura y distribución del target, duplicados por unidad, timestamps y campos derivados del resultado. La separación train/test debe preceder a transformaciones aprendidas. [Fuentes](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md), [pipeline sin leakage](../../sources/muller_introduction_ml_python/ch06_pipelines.md).

## 8. Objetivo de negocio frente a objetivo del modelo

```text
Objetivo del negocio
        ↓
Decisión que se desea apoyar
        ↓
Variable objetivo
        ↓
Predicción del modelo
        ↓
Métrica de evaluación
```

**Contenido explícito de las fuentes.** El objetivo del negocio determina formulación, algoritmos, métrica y esfuerzo; múltiples stakeholders pueden definir éxito de forma diferente, y los costes relativos de errores deben traducirse a criterios del modelo. [Fuentes](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [Multiple Objectives](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

**Conclusión práctica derivada.** Recorrer también la cadena en sentido inverso: comprobar que mejorar la métrica cambia la predicción de un modo útil para la decisión y que esa decisión contribuye al objetivo. Si el vínculo falla, debe revisarse target, salida, umbral o tarea antes de optimizar el modelo.

## 9. Patrón `Reframing`

### Problema

Una formulación intuitiva puede ser demasiado rígida: una regresión puntual oculta multimodalidad o incertidumbre; una clasificación puede optimizar un proxy que no coincide con la decisión. Más capacidad, datos o features no corrigen un objetivo mal formulado. [Fuente](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

### Contexto de aplicación

Targets continuos con distribuciones condicionales anchas o multimodales; decisiones tomadas por rangos; outputs que deben acotarse; labels desalineados con utilidad; o casos donde una representación continua sirve mejor que clases discretas.

### Reformulación

Revisar la decisión final y cambiar, cuando tenga significado operativo, regresión por clasificación en buckets, clasificación por una representación continua, el label proxy por otro más alineado, o una salida libre por una salida acotada. Deben cambiar conjuntamente target, output, loss, métricas e inferencia.

### Por qué puede funcionar

Los buckets pierden resolución pero pueden expresar probabilidades por intervalo y conservar varios modos. Una representación continua puede reflejar afinidad o cercanía. Un target alineado evita optimizar un proxy indeseado.

### Riesgos

Bins arbitrarios crean discontinuidades; demasiadas clases exigen más datos; las probabilidades pueden estar mal calibradas; se pierde precisión dentro del intervalo; cambiar de tarea impide comparar directamente métricas heterogéneas; y el nuevo label puede introducir sesgo.

### Alternativas

Mantener regresión cuando la distribución sea estrecha y se requiera precisión; considerar cuantiles, regresión probabilística o multitarea solo como alternativas registradas por la fuente, cuya implementación no se desarrolla aquí.

### Aplicación en una competición

**Recomendación específica para coding tests.** Considerar `Reframing` si el target continuo tiene picos, muchos ceros, dispersión heterogénea o si la evaluación usa categorías derivadas. Comparar la formulación original y la nueva con validación coherente y la métrica oficial; no asumir que bucketizar mejora un objetivo continuo. [Fuente](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

## 10. Reglas de decisión

| Situación | Formulación recomendada | Motivo | Precaución |
|---|---|---|---|
| Hay labels y la salida es discreta | Clasificación supervisada | Aprende clases conocidas | Definir umbral y coste de errores |
| Hay labels y la salida es cuantitativa | Regresión supervisada | Estima magnitud | Revisar rango, outliers y uso final |
| No hay labels y se buscan grupos | Clustering | Descubre estructura | Los clusters no son clases verdaderas |
| Se necesita probabilidad y no solo clase | Clasificación con scores/probabilidades | Permite umbrales según decisión | Revisar calibración |
| Target continuo se consume por rangos | Comparar regresión con clasificación por buckets | Puede alinear salida y decisión | Justificar cortes y pérdida de resolución |
| Label es un proxy desalineado | Reformular target | Evita optimizar comportamiento indeseado | El target alternativo debe ser observable |
| Clases desbalanceadas | Mantener clasificación y elegir métrica/umbral adecuados; considerar rebalancing después | Accuracy puede ocultar errores relevantes | No confundir desbalance con fairness |
| Datos tienen dependencia temporal | Formulación temporal con corte y horizonte explícitos | Impide usar futuro y representa el uso real | Las fuentes no dan un método completo de forecasting |
| Se busca comprender asociaciones | Modelo interpretable y objetivo de inferencia | La forma de la relación importa | Asociación no demuestra causalidad |
| Se busca apoyar una decisión | Predecir una cantidad/probabilidad y definir regla de decisión aparte | Separa estimación de costes y política | No evaluar solo la predicción si la utilidad depende de la decisión |

Respaldado por [tipos de ML](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [predicción e inferencia](../../sources/islp_python/ch02_statistical_learning.md), [evaluación](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) y [`Reframing`](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

## 11. Selección inicial de la métrica

La métrica debe corresponder al tipo de salida y al coste operativo. Esta sección solo fija una primera elección; el tratamiento exhaustivo corresponde a los documentos de métricas.

| Objetivo del problema | Tipo de salida | Métrica inicial posible | Precaución |
|---|---|---|---|
| Acertar clases con importancia similar | Clase | Accuracy | Puede ser engañosa con desbalance |
| Priorizar no perder positivos | Score/clase | Recall | Puede aumentar falsos positivos |
| Evitar falsas alarmas | Score/clase | Precision | Puede perder positivos |
| Equilibrar precision y recall | Clase | F1 | El balance implícito puede no igualar el coste real |
| Comparar ranking de scores binarios | Score continuo | ROC-AUC o PR-AUC | Elegir según desbalance y objetivo; no fija umbral |
| Penalizar especialmente errores grandes | Valor continuo | RMSE | Sensible a outliers |
| Reducir influencia de outliers | Valor continuo | MAE | Penaliza linealmente errores grandes |
| Descubrir grupos | Cluster | Métrica interna más interpretación | Una métrica geométrica no garantiza utilidad |

Fuentes: [clasificación](../../sources/geron_hands_on_ml/ch03_classification.md), [evaluación de modelos](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [proyecto de regresión](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

## 12. Baseline y verificabilidad del planteamiento

Un baseline da contexto al score y permite detectar si el modelo aporta señal sobre una referencia comprensible. Puede ser una constante, frecuencia, tabla, persistencia, regla trivial o proceso operativo vigente. Baseline y modelo deben medirse con la misma métrica, split y población. Superar una referencia débil no demuestra utilidad; cuando existe un proceso actual, compararse con él es más informativo. [Fuente: Heuristic Benchmark](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md).

**Síntesis práctica derivada de las fuentes.** Si un pipeline básico no supera la constante adecuada, revisar target, unidad, features, disponibilidad temporal, split y métrica antes de hacer tuning. Si train mejora pero validación no, puede haber overfitting; si ni train mejora, puede faltar señal, capacidad o una formulación adecuada. Mantener fijos split, preprocesamiento, métrica y baseline hace atribuible la comparación. [Fuentes](../../sources/geron_hands_on_ml/ch01_ml_landscape.md), [proyecto end-to-end](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

## 13. Implementación mínima para un coding test

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

target = "target"
X = data.drop(columns=[target])
y = data[target]

# Clasificación: y contiene etiquetas discretas.
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

baseline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    LogisticRegression(max_iter=1000),
)
baseline.fit(X_train, y_train)
valid_score = baseline.score(X_valid, y_valid)
```

El ejemplo separa `X` e `y`, divide antes de aprender la imputación y encapsula la transformación para evitar leakage. `stratify=y` corresponde a clasificación cuando se necesita conservar proporciones. [Fuentes](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md), [pipelines](../../sources/muller_introduction_ml_python/ch06_pipelines.md).

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** `max_iter=1000` hace explícito un límite suficiente para reducir fallos de convergencia; no constituye una regla universal. Para regresión debe cambiarse estimador, split estratificado y métrica; para tiempo o grupos, `train_test_split` aleatorio puede ser inadecuado.

## 14. Errores frecuentes y riesgos

**Riesgos explícitos de las fuentes:**

- Target erróneo, truncado, inconsistente o con labels de baja calidad.
- Optimizar un proxy que no representa la decisión real.
- Usar una métrica incompatible con el coste de errores.
- Ajustar imputación, escalado o selección antes de dividir los datos.
- Usar test para decisiones iterativas o comparar modelos.
- Ignorar desbalance, grupos, drift o diferencias entre training y producción.
- Convertir un continuo a categorías con bins arbitrarios o datos insuficientes.
- Confiar en una métrica global que oculta fallos por clase o slice.
- Reformular y comparar directamente métricas que ya no miden la misma tarea.

Fuentes: [`Reframing`](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md), [pipelines](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [evaluación](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md), [calidad y objetivos](../../sources/ml_design_patterns/ch01_ml_design_challenges.md).

**Síntesis práctica derivada:**

- Una unidad de análisis incorrecta puede duplicar entidades entre splits y producir una estimación optimista.
- Confundir correlación con objetivo predictivo no define quién actúa ni cuándo.
- Una feature disponible en el dataset histórico puede no existir al momento real de predicción.
- Un problema sin ground truth observable, split representativo o criterio de éxito no es evaluable todavía.
- Optimizar modelos antes de definir la decisión convierte el tuning en búsqueda sobre una pregunta inestable.

## 15. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.** Consultar este documento al recibir el enunciado y antes del EDA profundo. Durante los primeros minutos: localizar identificadores, granularidad, target, timestamps, tipos, faltantes, balance, métrica oficial y columnas sospechosas de contener el resultado. Definir `X` e `y` explícitamente y reservar evaluación antes de aprender transformaciones.

Procedimiento rápido:

1. Identificar la fila y la unidad de análisis.
2. Identificar el target.
3. Determinar el tipo de salida.
4. Revisar disponibilidad temporal de las variables.
5. Elegir una métrica inicial.
6. Crear un baseline.
7. Verificar que la formulación sea coherente.

Considerar `Reframing` solo ante evidencia de desalineación, multimodalidad, necesidad de rangos o proxy inadecuado. Construir primero un baseline simple con el preprocesamiento indispensable; no iniciar tuning masivo, ingeniería extensa ni ensembles antes de validar el planteamiento. Señales para revisar más: score anormalmente alto, target derivable de una feature, duplicados, IDs repetidos entre splits, timestamps posteriores, clases casi ausentes, cambio temporal, target truncado o mejora técnica sin utilidad aparente. [Fuentes](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [pipeline](../../sources/muller_introduction_ml_python/ch06_pipelines.md), [benchmark](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md).

## 16. Checklist operativo

- [ ] Definí el objetivo del problema.
- [ ] Identifiqué la decisión que el modelo debe apoyar.
- [ ] Definí qué representa cada fila.
- [ ] Definí la unidad de análisis.
- [ ] Separé claramente `X` y `y`.
- [ ] Confirmé que el target sea observable.
- [ ] Verifiqué que ninguna variable use información futura.
- [ ] Identifiqué el tipo de problema.
- [ ] Definí la salida esperada.
- [ ] Seleccioné una métrica inicial coherente.
- [ ] Consideré desbalance, grupos o temporalidad.
- [ ] Establecí un baseline.
- [ ] Evalué si necesito aplicar `Reframing`.
- [ ] Registré las decisiones y sus justificaciones.

## 17. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con este documento |
|---|---|
| `knowledge/eda/dataset_audit.md` | Verifica si los datos permiten sostener la formulación |
| `knowledge/preprocessing/data_cleaning.md` | Prepara los datos después de definir el problema |
| `knowledge/preprocessing/data_leakage.md` | Profundiza en riesgos de información indebida |
| `knowledge/validation/validation_strategy.md` | Define cómo evaluar la formulación y el modelo |
| `knowledge/metrics/classification_metrics.md` | Desarrolla métricas para clasificación |
| `knowledge/metrics/regression_metrics.md` | Desarrolla métricas para regresión |
| `knowledge/workflows/competition_workflow.md` | Convierte la formulación en un flujo completo |

## 18. Vacíos y limitaciones de las fuentes

- Ranking se menciona como posible uso de representaciones continuas, pero no se desarrolla como tarea autónoma.
- Las fuentes cubren operaciones temporales, drift, lags y prevención de futuro, pero no ofrecen aquí una metodología completa de forecasting ni selección de horizonte.
- La definición de unidad de análisis se deriva de las reglas de representación de filas y del flujo de datos; no aparece como taxonomía formal independiente.
- La observabilidad del target, el coste de etiquetado y la demora del ground truth requieren validación con el dominio y experimentación.
- `Reframing` enumera alternativas probabilísticas y de cuantiles, pero estos resúmenes no bastan para documentar su implementación completa.
- No faltó ninguna de las fuentes Markdown seleccionadas y no se detectaron capítulos ilegibles; los límites anteriores son de cobertura conceptual.

## 19. Fuentes utilizadas

| Fuente | Archivo o carpeta | Aporte utilizado |
|---|---|---|
| Géron, panorama de ML | [ch01_ml_landscape.md](../../sources/geron_hands_on_ml/ch01_ml_landscape.md) | Tarea–datos–medida, tipos de aprendizaje, generalización y riesgos |
| Géron, proyecto end-to-end | [ch02_end_to_end_ml_project.md](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Objetivo de negocio, salida, flujo, métricas, baseline y split |
| Géron, clasificación | [ch03_classification.md](../../sources/geron_hands_on_ml/ch03_classification.md) | Binaria, multiclase, multilabel, probabilidades y umbrales |
| Géron, no supervisado | [ch09_unsupervised_learning.md](../../sources/geron_hands_on_ml/ch09_unsupervised_learning.md) | Clustering y anomalías |
| Müller, evaluación | [ch05_model_evaluation.md](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | Métricas, desbalance y evaluación coherente |
| Müller, pipelines | [ch06_pipelines.md](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Prevención de leakage y comparación de flujos completos |
| ISLP, aprendizaje estadístico | [ch02_statistical_learning.md](../../sources/islp_python/ch02_statistical_learning.md) | Predicción, inferencia, respuesta, predictores y error fuera de muestra |
| Molin, data wrangling | [ch03_data_wrangling.md](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) | Significado de filas, columnas, unidades, tiempo y calidad |
| Molin, ML con scikit-learn | [ch09_ml_with_sklearn.md](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md) | Separación `X`/`y`, tipos de tarea y código mínimo |
| McKinney, series temporales | [ch11_time_series.md](../../sources/mckinney_python_data_analysis/ch11_time_series.md) | Operaciones temporales y límite de cobertura de forecasting |
| ML Design Patterns, desafíos | [ch01_ml_design_challenges.md](../../sources/ml_design_patterns/ch01_ml_design_challenges.md) | Calidad, reproducibilidad, drift, escala y objetivos múltiples |
| ML Design Patterns, representación | [ch03_problem_representation_patterns.md](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md) | Patrón `Reframing`, proxies, salidas y trade-offs |
| ML Design Patterns, Responsible AI | [ch07_responsible_ai_patterns.md](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md) | Heuristic Benchmark, utilidad y evaluación por slices |

## 20. Trazabilidad de afirmaciones

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1–3. Propósito, preguntas y negocio frente a ML | Géron ch. 1–2; ISLP ch. 2; Design Patterns ch. 1 | Explícito de la fuente + Síntesis integrada |
| 4. Componentes | Géron ch. 2; Molin ch. 3; ISLP ch. 2; Design Patterns ch. 1 y 3 | Síntesis integrada |
| 5. Tipos de problema | Géron ch. 1, 3 y 9; Molin ch. 9 | Explícito de la fuente |
| 6. Flujo de formulación | Géron ch. 1–2; Molin ch. 3; Design Patterns ch. 7 | Síntesis integrada |
| 7–8. Target y cadena de objetivos | ISLP ch. 2; Géron ch. 2; Design Patterns ch. 1 | Explícito de la fuente + Conclusión práctica derivada |
| 9. `Reframing` | Design Patterns ch. 3, Pattern 5 | Explícito de la fuente + Recomendación para coding test |
| 10. Reglas de decisión | Géron ch. 1; ISLP ch. 2; Müller ch. 5; Design Patterns ch. 3 | Síntesis integrada |
| 11. Métrica inicial | Géron ch. 2–3; Müller ch. 5 | Explícito de la fuente |
| 12. Baseline | Design Patterns ch. 7; Géron ch. 1–2 | Explícito de la fuente + Síntesis integrada |
| 13. Implementación mínima | Molin ch. 9; Müller ch. 6 | Recomendación para coding test + Adaptación moderna |
| 14. Riesgos | Géron ch. 1–2; Müller ch. 5–6; Design Patterns ch. 1 y 3 | Explícito de la fuente + Síntesis integrada |
| 15–16. Coding test y checklist | Fuentes anteriores | Recomendación para coding test |
| 17–18. Relaciones y límites | Catálogo de destinos de las fuentes utilizadas | Síntesis integrada |
