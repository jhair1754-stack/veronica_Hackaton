---
title: "Tabular Classification"
document_type: "playbook"
status: "draft"
scope: "Clasificación supervisada con datos tabulares"
---

# Playbook de clasificación tabular

## 1. Propósito y alcance

Guía operativa para resolver problemas de clasificación supervisada tabular en coding tests, datathons y proyectos. Lleva desde la definición del objetivo hasta la entrega, con decisiones condicionadas por el enunciado y la estructura observada. Cubre clasificación binaria y multiclase en la medida descrita por las fuentes; no extiende estas reglas a multilabel.

La secuencia es iterativa: un hallazgo sobre el target, la calidad o el split puede exigir volver a una etapa anterior. No hay un modelo, una métrica, una proporción de partición ni un método para corregir desbalance que sea universal. Las recomendaciones basadas en las fuentes se enlazan en cada etapa; las adaptaciones prácticas se marcan como derivadas.

## 2. Información que debes obtener al iniciar

Antes de modelar, anota las respuestas o marca lo que aún no sabes:

- **Objetivo y decisión:** ¿qué resultado se quiere predecir y quién usará la predicción?
- **Target:** nombre, significado, momento en que se conoce, valores posibles, clases presentes y si las clases son mutuamente excluyentes.
- **Unidad de análisis:** qué representa una fila; si hay varias filas por entidad, evento, sesión o periodo.
- **Predictores disponibles:** qué campos existirán al momento real de predecir; origen y significado de identificadores, fechas y columnas sospechosas.
- **Datos:** formato, diccionario, train/test disponibles, faltantes, duplicados, tipos, categorías y distribución del target.
- **Evaluación del reto:** métrica oficial, salida solicitada (clase o puntuación/probabilidad), esquema de validación indicado y reglas/formato de entrega.
- **Restricciones:** tiempo, memoria, cómputo y límites del entorno que consten en el reto.

Si no se confirma target, clases o unidad de fila, no interpretes aún una puntuación. Si hay tiempo, grupos o filas relacionadas, define qué tipo de observación nueva debe representar la evaluación antes de escoger el splitter. [Formulación del problema](../knowledge/problem_definition/problem_formulation.md), [auditoría del dataset](../knowledge/eda/dataset_audit.md), [validación](../knowledge/validation/validation_strategy.md)

## 3. Flujo de decisión

| Señal o situación | Acción recomendada | Motivo | Precaución | Fuente |
|---|---|---|---|---|
| El target tiene dos valores excluyentes | Tratarlo como clasificación binaria; confirmar cuál es la clase positiva si la métrica o decisión la distingue | Las métricas y salidas dependen de las clases y de la regla de decisión | No asumir que la codificación numérica define por sí sola la clase positiva | [Formulación](../knowledge/problem_definition/problem_formulation.md), [métricas](../knowledge/metrics/classification_metrics.md) |
| El target tiene más de dos valores excluyentes | Tratarlo como multiclase; revisar soporte por clase y cómo se agregará la métrica | Una cifra agregada puede ocultar el desempeño de clases minoritarias | No tratar clases nominales como una escala ordenada | [Modelos](../knowledge/models/tabular_models.md), [métricas](../knowledge/metrics/classification_metrics.md) |
| La proporción de clases es desigual | Cuantificar y reportar distribución; elegir la métrica oficial o la que refleje el coste de errores; valorar estratificación si es compatible con el split | Accuracy puede ocultar el desempeño de clases raras y la evaluación depende de la distribución | No hay ratio límite ni corrección automática respaldada aquí | [Auditoría](../knowledge/eda/dataset_audit.md), [métricas](../knowledge/metrics/classification_metrics.md), [validación](../knowledge/validation/validation_strategy.md) |
| Filas de una misma entidad, grupo o evento están relacionadas | Definir si se busca generalizar a grupos nuevos y mantener los grupos completos dentro de una partición; considerar splitter por grupos | Separar filas relacionadas puede medir reconocimiento de entidades vistas y causar fuga | Identificar una clave de grupo válida requiere conocimiento del dataset | [Validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md) |
| Se predice sobre periodos futuros o hay orden temporal | Respetar el orden temporal y evaluar pasado frente a futuro con la estrategia documentada | Un split aleatorio puede permitir información futura y no responder al uso esperado | La estrategia temporal detallada puede ser un vacío; no apliques shuffle por defecto | [Validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md) |
| Hay faltantes | Medirlos, entender su significado y seleccionar tratamiento según variable y estimador; ajustar el tratamiento dentro de cada partición | Faltantes pueden tener significados distintos y las estadísticas de imputación se aprenden | No existe una imputación universal | [Limpieza](../knowledge/preprocessing/data_cleaning.md), [leakage](../knowledge/preprocessing/data_leakage.md) |
| Hay variables categóricas | Distinguir nominales, ordinales y códigos; codificar de manera compatible con el modelo | La representación depende de semántica y familia de modelos | No imponer orden a categorías nominales; contemplar categorías nuevas según el esquema | [Feature engineering](../knowledge/feature_engineering/feature_engineering.md), [modelos](../knowledge/models/tabular_models.md) |
| El reto pide una métrica distinta de la disponible en la interfaz o de la usada en el baseline | Implementar y comparar con la métrica oficial sobre la salida requerida; mantener claro si se evalúan etiquetas o puntuaciones | Accuracy o el score por defecto no sustituyen el criterio de la competición | No cambiar métrica tras ver resultados para favorecer un candidato | [Métricas](../knowledge/metrics/classification_metrics.md), [flujo de competición](../knowledge/workflows/competition_workflow.md) |
| Validación y leaderboard/test oficial difieren | Diagnosticar split, métrica, formato y distribución; elegir localmente con validación representativa | El test/leaderboard no debe dirigir ajustes repetidos | No asumir que se conoce el motivo de la diferencia sin evidencia | [Validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md) |

## 4. Procedimiento paso a paso

### Paso 1. Entender el objetivo de clasificación

- **Objetivo.** Traducir el enunciado a una tarea evaluable.
- **Acciones concretas.** Identifica la decisión y la unidad de análisis; localiza el target; comprueba que cada fila tenga una etiqueta; lista clases y confirma si son mutuamente excluyentes; identifica el instante de predicción, los datos disponibles entonces y la salida exigida. Lee la métrica y las reglas del reto.
- **Decisión que se debe tomar.** Confirmar binaria o multiclase, clase de interés cuando aplique, salida requerida y criterios de éxito. Si faltan definiciones esenciales, registra la duda y evita suponerlas.
- **Resultado esperado.** Definición de una línea: qué representa una fila, qué clase se predice, con qué información, para qué decisión y con qué métrica/salida.
- **Errores que se deben evitar.** Modelar un proxy distinto del objetivo; usar campos que solo se conocen después de predecir; confundir clasificación multiclase con varias etiquetas simultáneas.
- **Fuente.** [Formulación del problema](../knowledge/problem_definition/problem_formulation.md), [métricas de clasificación](../knowledge/metrics/classification_metrics.md).

### Paso 2. Auditar el dataset

- **Objetivo.** Determinar si el dataset puede interpretarse y qué hallazgos afectan preparación o validación.
- **Acciones concretas.** Inspecciona dimensiones, nombres, tipos físicos y semánticos, ejemplos, cardinalidad y faltantes; cuantifica clases; revisa duplicados y claves repetidas; identifica IDs, grupos, fechas y columnas de procedencia; compara train/test si el reto los da; registra columnas potencialmente posteriores o derivadas del target.
- **Decisión que se debe tomar.** Resolver el significado de filas y campos relevantes; distinguir observación confirmada de sospecha; elegir qué requiere limpieza y si hay dependencia entre observaciones.
- **Resultado esperado.** Inventario breve de target/clases, esquema, calidad, unidad, grupos/tiempo y riesgos, más preguntas pendientes.
- **Errores que se deben evitar.** Borrar automáticamente IDs, duplicados, extremos o filas incompletas; inferir fuga solo por alta asociación; modificar datos durante la auditoría.
- **Fuente.** [Auditoría del dataset](../knowledge/eda/dataset_audit.md), [limpieza de datos](../knowledge/preprocessing/data_cleaning.md).

### Paso 3. Definir validación y métrica

- **Objetivo.** Hacer que la evaluación represente las observaciones a las que se aplicará el modelo.
- **Acciones concretas.** Lee primero la métrica y el formato del reto. Decide la unidad de generalización (filas independientes, grupos nuevos o futuro). Considera holdout cuando se necesita una comprobación simple/de bajo coste; K-Fold para scores por fold y sensibilidad al split; Stratified K-Fold si preservar proporciones de clase es pertinente y no rompe grupos; splitter por grupos o temporal cuando lo exige la estructura. Usa la misma métrica y particiones al comparar candidatos. Separa un test final si los datos y el presupuesto lo permiten.
- **Decisión que se debe tomar.** Fijar splitter, semilla o regla estable, métrica, función de cada partición y salida que se puntuará. Si ninguna estrategia descrita responde al escenario, anota el vacío y no declares representativa una partición aleatoria.
- **Resultado esperado.** Protocolo reproducible con justificación y limitaciones; ninguna observación de evaluación intervino en el ajuste de transformaciones.
- **Errores que se deben evitar.** Elegir la métrica por costumbre; mezclar entidades relacionadas; barajar tiempo; estratificar de modo que se separen grupos; usar test para escoger modelo, variables, hiperparámetros o umbral.
- **Fuente.** [Estrategias de validación](../knowledge/validation/validation_strategy.md), [métricas de clasificación](../knowledge/metrics/classification_metrics.md), [prevención de leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 4. Preparar un baseline

- **Objetivo.** Obtener una referencia reproducible antes de invertir en cambios complejos.
- **Acciones concretas.** Separa explícitamente `X` e `y`; aplica la validación definida; usa la métrica del reto; prepara solo lo necesario para el estimador; ajusta el preprocesamiento con el train de cada split; usa un modelo simple respaldado por `knowledge/`, por ejemplo regresión logística cuando sea adecuada, o un árbol de decisión. Fija aleatoriedad si el procedimiento es aleatorio. Guarda predicciones/scores necesarios, configuración y resultado.
- **Decisión que se debe tomar.** Determinar si el baseline es válido y supera una referencia trivial/heurística disponible bajo el mismo criterio; identificar problemas de convergencia, datos o evaluación antes de avanzar.
- **Resultado esperado.** Una métrica base comparable, su estrategia y variación observada (si hay folds), y un flujo que puede repetirse.
- **Errores que se deben evitar.** Ajustar imputación/codificación/escala antes de separar; usar `.score()` sin confirmar qué métrica representa; comparar con un candidato evaluado bajo otro split.
- **Fuente.** [Modelos tabulares](../knowledge/models/tabular_models.md), [estrategia de validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md), [flujo de competición](../knowledge/workflows/competition_workflow.md).

### Paso 5. Preprocesar los datos sin leakage

- **Objetivo.** Convertir las columnas en entradas válidas para el estimador sin que evaluación influya en los tratamientos aprendidos.
- **Acciones concretas.** Corrige formatos con base en su significado; decide de forma explícita tratamiento de faltantes, categorías y escala según la familia candidata. Coloca imputadores, codificadores, escaladores, selectores y otras transformaciones aprendidas dentro del pipeline/flujo evaluado. En validación cruzada, cada paso se reajusta en el fold de entrenamiento y se aplica al fold retenido.
- **Decisión que se debe tomar.** Conservar un tratamiento solo si tiene justificación semántica y es compatible con predicción/validación; registrar cómo se manejarán niveles no vistos si aplica.
- **Resultado esperado.** Un único proceso ajustable que transforma datos de entrenamiento y de predicción de forma coherente.
- **Errores que se deben evitar.** Calcular estadísticas, vocabularios, bins, selecciones o agregados con el dataset completo; tratar entero como magnitud sin verificar; crear features temporales con información futura.
- **Fuente.** [Limpieza](../knowledge/preprocessing/data_cleaning.md), [feature engineering](../knowledge/feature_engineering/feature_engineering.md), [leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 6. Entrenar y evaluar modelos

- **Objetivo.** Comparar candidatos razonables con evidencia equivalente.
- **Acciones concretas.** Elige familias compatibles con target, representación, tamaño, interpretabilidad y coste. La regresión logística ofrece una referencia lineal; árboles, k-NN, SVM u otros modelos tabulares pueden considerarse según sus requisitos de escala y supuestos. Añade random forest o boosting si el baseline justifica explorar esa familia y el presupuesto lo permite. Reutiliza particiones/folds, preprocesamiento y métrica.
- **Decisión que se debe tomar.** Elegir la alternativa mejor respaldada por validación y requisitos operativos, considerando variación entre folds y coste, no solo un score aislado.
- **Resultado esperado.** Comparación registrada de baseline y candidatos, con scores comparables y razones para continuar/descartar.
- **Errores que se deben evitar.** Asumir ganador universal; favorecer más complejidad sin mejora; comparar etiquetas y probabilidades como si fueran la misma salida; tomar una partición como garantía de generalización.
- **Fuente.** [Modelos tabulares](../knowledge/models/tabular_models.md), [ensembles y boosting](../knowledge/ensembles/boosting_models.md), [validación](../knowledge/validation/validation_strategy.md).

### Paso 7. Revisar errores y mejorar el modelo

- **Objetivo.** Identificar qué errores comete el candidato y qué hipótesis merece probarse.
- **Acciones concretas.** Revisa matriz de confusión y métricas por clase cuando correspondan; distingue precisión/recall y errores FP/FN según el objetivo; inspecciona ejemplos erróneos y slices pertinentes; compara entrenamiento y validación para señales de underfitting/overfitting; audita procedencia y momento de disponibilidad ante scores inesperados. Formula una hipótesis por cambio de representación/modelo y mídela bajo el mismo protocolo.
- **Decisión que se debe tomar.** Probar únicamente cambios apoyados por un hallazgo (p. ej., codificación o transformación compatible, una familia diferente, o ajuste del umbral si la métrica/decisión lo permite). Retener un cambio solo con evidencia fuera de muestra y coste aceptable.
- **Resultado esperado.** Errores caracterizados y una lista corta de experimentos justificados o conclusión de que no hay cambio con evidencia suficiente.
- **Errores que se deben evitar.** Leer asociación/importancia como causalidad; perseguir cada error con ingeniería extensa; aceptar mejora en una única partición como prueba de generalización; elegir umbral con test.
- **Fuente.** [Métricas](../knowledge/metrics/classification_metrics.md), [feature engineering](../knowledge/feature_engineering/feature_engineering.md), [leakage](../knowledge/preprocessing/data_leakage.md), [modelos](../knowledge/models/tabular_models.md).

### Paso 8. Ajustar hiperparámetros

- **Objetivo.** Afinar candidatos prometedores después de que baseline, métrica y validación sean confiables.
- **Acciones concretas.** Selecciona pocos parámetros con impacto plausible; delimita el espacio de búsqueda al tiempo disponible; usa búsqueda manual o rejilla pequeña para espacios discretos acotados, o aleatoria para explorar más configuraciones. Incluye el pipeline completo en CV y registra scores y coste.
- **Decisión que se debe tomar.** Conservar una configuración solo si mejora la métrica objetivo de forma consistente y la ganancia compensa variación y coste. Si el presupuesto es insuficiente o el baseline no es confiable, no inicies tuning.
- **Resultado esperado.** Shortlist pequeña, configuración seleccionada dentro del espacio probado y evidencia que no usa el test para decidir.
- **Errores que se deben evitar.** Búsqueda ilimitada; tuning para compensar leakage/datos mal formulados; afirmar óptimo global; interpretar mejora marginal frente a variación como evidencia firme.
- **Fuente.** [Ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md), [validación](../knowledge/validation/validation_strategy.md).

### Paso 9. Seleccionar y guardar el modelo final

- **Objetivo.** Cerrar una solución respaldada por validación y capaz de generar la salida requerida.
- **Acciones concretas.** Elige configuración según métrica, estrategia y restricciones del reto; confirma que el test final no participó en decisiones. Reajusta el procedimiento elegido con los datos de desarrollo permitidos y evalúa test una vez si existe. Conserva conjuntamente estimador y transformaciones necesarias, semilla/regla de partición, configuración y resultados.
- **Decisión que se debe tomar.** Seleccionar modelo y datos de reajuste según las reglas concretas del reto. Si leaderboard, validación y test no concuerdan, investiga la diferencia sin iterar contra el test como si fuese validación.
- **Resultado esperado.** Artefacto/proceso reproducible y evaluación final con limitaciones registradas.
- **Errores que se deben evitar.** Seleccionar por entrenamiento; alterar el modelo tras consultar test; guardar solo estimador si depende de transformaciones; afirmar generalización garantizada.
- **Fuente.** [Validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md), [flujo de competición](../knowledge/workflows/competition_workflow.md).

### Paso 10. Verificar la entrega

- **Objetivo.** Confirmar que la solución corre y cumple literalmente los requisitos de entrega.
- **Acciones concretas.** Vuelve al enunciado: verifica filas previstas, orden/identificadores si se requieren, tipo de salida (clase o score), cantidad y orden de columnas, representación/clases permitidas, restricciones de ejecución y reproducibilidad. Genera la salida con el proceso final y comprueba dimensiones, tipos y valores permitidos frente a la especificación.
- **Decisión que se debe tomar.** Entregar solo cuando el archivo/proceso satisface las condiciones expresas del reto; ante un formato no documentado, consultar instrucciones disponibles y marcarlo pendiente, no inventarlo.
- **Resultado esperado.** Salida trazable, compatible con el formato solicitado y producida por las mismas transformaciones/modelo evaluados.
- **Errores que se deben evitar.** Inventar nombres de archivos o columnas; predecir con un proceso de preprocesamiento distinto; confundir la métrica de validación con el formato de submission.
- **Fuente.** [Formulación](../knowledge/problem_definition/problem_formulation.md), [flujo de competición](../knowledge/workflows/competition_workflow.md), [feature engineering](../knowledge/feature_engineering/feature_engineering.md).

## 5. Reglas de decisión

- Si la métrica oficial está especificada, úsala como criterio principal; interpreta además métricas complementarias solo cuando aclaren errores relevantes. Si el coste de FP y FN es asimétrico, refleja esa prioridad en la evaluación según la definición del problema.
- En binaria, accuracy cuenta aciertos globales; precision y recall responden a errores distintos; F1 combina precision y recall; ROC-AUC y PR-AUC/average precision evalúan puntuaciones/ranking sobre umbrales. No son intercambiables: conserva la salida y definición de métrica que exige el reto.
- En multiclase, reporta resultados por clase y conoce la agregación (macro, micro o weighted) cuando aplique; un promedio puede ocultar clases con peor desempeño.
- Usa holdout si importa sencillez/coste y representa el futuro; CV si necesitas scores por folds y observar sensibilidad; estratificación cuando ayuda a conservar proporciones de clases; grupos/tiempo si la dependencia lo requiere. No hay splitter que resuelva automáticamente todas las estructuras.
- Si la limpieza o feature engineering no mejora una comparación válida o aumenta complejidad/coste sin beneficio suficiente, no la conserves por intuición.
- Una discrepancia entre validación local y métrica de entrega puede reflejar una diferencia de split, distribución, métrica o formato; localiza la causa antes de cambiar el modelo.

Fuentes: [métricas de clasificación](../knowledge/metrics/classification_metrics.md), [validación](../knowledge/validation/validation_strategy.md), [feature engineering](../knowledge/feature_engineering/feature_engineering.md), [modelos tabulares](../knowledge/models/tabular_models.md).

## 6. Baseline mínimo

El baseline debe contestar la pregunta del reto con el menor flujo justificable y bajo la validación ya definida. No se impone un clasificador único: regresión logística es un candidato lineal y un árbol de decisión otra referencia tabular documentada; la elección depende de datos y requisitos. Una referencia trivial/heurística también puede ser útil si el dominio o el reto la define.

Procedimiento:

1. Separar `X` y `y`; excluir target y columnas que no estén disponibles en predicción solo cuando la auditoría justifique esa decisión.
2. Fijar splitter, semilla/regla estable y métrica oficial. Si hay grupos/tiempo, usar la estrategia correspondiente, no el ejemplo de holdout aleatorio.
3. Definir tratamiento mínimo de faltantes/categorías/escala según variables y modelo. Si el tratamiento aprende parámetros, integrarlo con el estimador en un pipeline que se ajuste dentro de cada fold.
4. Entrenar y calcular el score fuera de muestra. Registrar configuración, validación, métrica, resultado y observaciones.
5. Comparar cualquier candidato posterior bajo las mismas condiciones.

Ejemplo esquemático de un caso **solo cuando las filas sean intercambiables y el holdout aleatorio sea representativo**. `preprocesador` debe contener las transformaciones necesarias, ajustables, y `clasificador` debe ser una familia escogida según el problema; el ejemplo no fija métrica ni estimador:

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X = datos.drop(columns=[columna_objetivo])
y = datos[columna_objetivo]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, stratify=y, random_state=semilla
)

baseline = make_pipeline(preprocesador, clasificador)
baseline.fit(X_train, y_train)
prediccion = baseline.predict(X_valid)  # o score/probabilidad si lo requiere la métrica
resultado = metrica_del_reto(y_valid, prediccion)
```

La estratificación del ejemplo es opcional y solo procede si es compatible con la estructura; sustituye el splitter en casos temporales o agrupados. `preprocesador`, clasificador, métrica y semilla deben definirse explícitamente y registrarse. Esta plantilla es una recomendación práctica derivada de las fuentes, no una configuración universal. [Modelos](../knowledge/models/tabular_models.md), [validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md)

## 7. Tratamiento del desbalance de clases

**Observación del dataset.** Cuenta ejemplos por clase y por partición; conserva conteos y proporciones. La distribución por sí sola describe los datos, no determina una intervención.

**Recomendación respaldada.** Accuracy puede resultar engañosa cuando una clase domina. Selecciona la métrica según objetivo, clase de interés y coste de errores; inspecciona métricas por clase/matriz de confusión. Stratified K-Fold puede conservar proporciones aproximadas por fold si resulta compatible con la estructura de datos. En binaria, las curvas PR y ROC y las métricas por umbral responden preguntas diferentes; usa la que corresponda al criterio del reto.

**Alternativas mencionadas por las fuentes.** Los documentos de validación/modelado mencionan estratificación y, en algunos contextos, ponderación de clases o remuestreo dentro de los folds de entrenamiento. Su aplicabilidad depende de la fuente/modelo y del objetivo concreto; este playbook no elige una de forma automática.

**Recomendación práctica derivada.** Para cada experimento de tratamiento, mantén el mismo splitter y criterio, aplica cualquier remuestreo solo al entrenamiento de cada fold y compara contra el baseline. No uses el resultado de una única partición como garantía.

**Vacío.** Las fuentes consultadas no respaldan una proporción límite para declarar desbalance severo ni un protocolo universal para escoger entre ponderación, sobremuestreo o submuestreo. Si el reto y las fuentes específicas del candidato no justifican una técnica, limita el trabajo a medir la distribución, evaluar con criterio apropiado y registrar la limitación.

Fuentes: [auditoría](../knowledge/eda/dataset_audit.md), [métricas](../knowledge/metrics/classification_metrics.md), [validación](../knowledge/validation/validation_strategy.md), [tuning](../knowledge/tuning/hyperparameter_tuning.md).

## 8. Comparación de experimentos y registro

Registra cada experimento con al menos:

| Campo | Qué guardar |
|---|---|
| Identificador/fecha | Una referencia estable para encontrar la ejecución |
| Hipótesis y configuración | Modelo, parámetros, variables y transformaciones modificadas |
| Validación | Splitter, semilla/regla, folds y limitaciones |
| Métrica | Nombre, definición/averaging si aplica y tipo de salida evaluada |
| Resultado | Score fuera de muestra; por fold y resumen si hay CV |
| Observaciones | Errores relevantes, coste, warnings, fugas investigadas y decisión |

Compara candidatos con el mismo protocolo. Una mejora en una partición no garantiza mejor generalización; considera dispersión de folds cuando exista. Mantén el test final fuera de la selección. Registra también cambios que no ayudaron para no repetirlos como si fueran evidencia nueva.

Fuente: [validación](../knowledge/validation/validation_strategy.md), [tuning](../knowledge/tuning/hyperparameter_tuning.md), [flujo de competición](../knowledge/workflows/competition_workflow.md).

## 9. Criterios para detenerse

Detén la exploración y prepara la entrega cuando se haya cerrado la métrica y el protocolo, exista un candidato reproducible que supere el baseline bajo condiciones comparables, las transformaciones no presenten fugas conocidas y los requisitos de salida estén verificados. También es razonable detener el tuning cuando se agota el presupuesto o no aparece evidencia consistente que compense el coste. No hay umbral universal de mejora mínima, número de experimentos o tiempo: registra el criterio usado y sus límites.

Si siguen sin resolverse target, unidad, métrica, estructura grupal/temporal, disponibilidad de variables o formato de entrega, la condición de parada es documentar el impedimento; no presentes el score como validación concluyente.

Fuente: [tuning](../knowledge/tuning/hyperparameter_tuning.md), [validación](../knowledge/validation/validation_strategy.md), [formulación](../knowledge/problem_definition/problem_formulation.md).

## 10. Riesgos y controles

| Riesgo | Control |
|---|---|
| Target ambiguo, proxy o mal etiquetado | Confirmar significado, momento y soporte por clase; registrar dudas |
| Métrica no alineada | Leer la definición oficial y los errores que penaliza antes de modelar |
| Leakage por preprocesamiento | Separar primero; ajustar transformaciones solo dentro de entrenamiento/fold |
| Leakage por tiempo o entidades | Auditar timestamps, claves, grupos, duplicados y disponibilidad de campos; elegir split coherente |
| Test usado repetidamente | Reservarlo para evaluación final cuando exista y cerrar decisiones antes de consultarlo |
| Score extraordinario o inestable | Revisar procedencia, target, split y transformaciones; la cifra sola no confirma ni descarta fuga |
| Comparaciones desiguales | Mantener mismos folds, métrica, salida y preprocesamiento evaluado como flujo completo |
| Transformación no reproducible en inferencia | Guardar y reutilizar el proceso completo de preparación más estimador |
| Desbalance tratado con regla inventada | Reportar distribución/métricas por clase; no introducir ratios o técnicas sin respaldo |

Fuentes: [leakage](../knowledge/preprocessing/data_leakage.md), [validación](../knowledge/validation/validation_strategy.md), [auditoría](../knowledge/eda/dataset_audit.md), [métricas](../knowledge/metrics/classification_metrics.md).

## 11. Checklist de ejecución

- [ ] Confirmé target, clases, unidad de fila, momento de predicción y salida requerida.
- [ ] Leí la métrica y las reglas exactas del reto.
- [ ] Cuantifiqué el soporte por clase y revisé faltantes, duplicados, tipos y categorías.
- [ ] Revisé grupos, filas repetidas, timestamps y disponibilidad de variables.
- [ ] Elegí validación que corresponda a la unidad de generalización.
- [ ] Separé `X`/`y` y mantuve test fuera de las decisiones.
- [ ] Integré transformaciones aprendidas con el modelo dentro del flujo evaluado.
- [ ] Entrené y registré un baseline reproducible.
- [ ] Comparé cambios bajo la misma métrica y las mismas particiones.
- [ ] Revisé errores por clase y variabilidad de resultados cuando procede.
- [ ] Ajusté hiperparámetros solo después de validar el baseline y con un presupuesto definido.
- [ ] Guardé modelo y transformaciones, semilla/regla, configuración y resultado.
- [ ] Verifiqué que la salida coincide con el formato que el reto sí especifica.

## 12. Fuentes de `knowledge/`

Documentos consultados y usados:

- [Formulación del problema](../knowledge/problem_definition/problem_formulation.md)
- [Auditoría del dataset](../knowledge/eda/dataset_audit.md)
- [Limpieza de datos](../knowledge/preprocessing/data_cleaning.md)
- [Detección y prevención de data leakage](../knowledge/preprocessing/data_leakage.md)
- [Feature engineering](../knowledge/feature_engineering/feature_engineering.md)
- [Estrategias de validación](../knowledge/validation/validation_strategy.md)
- [Métricas de clasificación](../knowledge/metrics/classification_metrics.md)
- [Modelos tabulares](../knowledge/models/tabular_models.md)
- [Ensembles y boosting](../knowledge/ensembles/boosting_models.md)
- [Ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md)
- [Flujo de trabajo para competiciones](../knowledge/workflows/competition_workflow.md)

## 13. Vacíos o límites del playbook

- No se encontró una métrica, modelo, proporción de split, semilla ni hiperparámetro que sea adecuado universalmente; deben decidirse desde el reto y la estructura real.
- La cobertura de validación temporal es parcial: no basta para documentar gaps, embargo, ventanas múltiples o backtesting detallado. Si el problema requiere esos elementos, la estrategia específica queda como vacío.
- No hay una receta universal para combinar simultáneamente grupos, tiempo y estratificación. Hace falta identificar la unidad de generalización y documentar el límite.
- Las fuentes no ofrecen umbrales para declarar una clase suficientemente rara ni un procedimiento universal de corrección del desbalance. El texto limita esas decisiones a evaluación respaldada y técnicas justificadas por el modelo/fuentes.
- La cobertura de leakage por columnas derivadas directamente del objetivo es parcial; la procedencia y disponibilidad deben verificarse con el proceso de datos.
- No se cubre calibración de probabilidades de forma suficiente. Una probabilidad producida por un clasificador no debe suponerse calibrada.
- La construcción final del archivo de entrega depende del enunciado concreto; aquí no se inventan formatos ni nombres.
- No se ha validado este procedimiento con casos prácticos; el estado permanece `draft`.
