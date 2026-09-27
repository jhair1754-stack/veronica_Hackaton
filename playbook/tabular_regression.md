---
title: "Tabular Regression"
document_type: "playbook"
status: "draft"
scope: "Regresión supervisada con datos tabulares"
---

# Playbook de regresión tabular

## 1. Propósito y alcance

Guía operativa para resolver tareas supervisadas de regresión con tablas en coding tests, datathons y proyectos. Sigue el ciclo desde aclarar el objetivo hasta comprobar la entrega; las decisiones se apoyan en validación fuera de muestra y en la estructura real de los datos. Las filas pueden depender de entidades o del tiempo: no se presupone que una partición aleatoria sea válida. [Formulación](../knowledge/problem_definition/problem_formulation.md) · [Validación](../knowledge/validation/validation_strategy.md)

El playbook no sustituye las fuentes temáticas. Cuando se requiera detalle, sigue los enlaces en cada paso. Las recomendaciones derivadas de combinar esas fuentes se identifican como **síntesis práctica**. No se fijan umbrales universales de mejora, valores atípicos o hiperparámetros.

## 2. Información que debes obtener al iniciar

Antes de modelar, anota:

- **Objetivo y uso:** qué decisión apoyará la predicción y qué entrega pide el reto.
- **Unidad de análisis:** qué representa cada fila, si una entidad aparece varias veces y cuál es la unidad a la que se desea generalizar.
- **Objetivo (`y`):** columna, significado, unidad de medida, momento en que se observa, rango plausible y métrica oficial.
- **Predictores (`X`):** columnas disponibles en el momento real de predicción; identifica IDs, metadatos y variables cuya procedencia o disponibilidad no está clara.
- **Estructura:** fechas, orden temporal, grupos, claves repetidas, duplicados y particiones provistas por el reto.
- **Calidad:** tipos semánticos, faltantes, centinelas, categorías, unidades, extremos y posibles topes del objetivo.
- **Restricciones:** presupuesto de cómputo, interpretabilidad, tiempo de inferencia y formato de salida.

Si falta el diccionario o una definición, registra la pregunta y no conviertas una suposición en una corrección automática. [Formulación](../knowledge/problem_definition/problem_formulation.md) · [Auditoría](../knowledge/eda/dataset_audit.md)

## 3. Flujo de decisión

| Señal o situación | Acción recomendada | Motivo | Precaución | Fuente |
|---|---|---|---|---|
| Filas aproximadamente independientes y sin grupos o tiempo relevantes | Considerar holdout o K-Fold según coste y estabilidad requerida | K-Fold ofrece resultados por fold; holdout cuesta menos | Confirmar primero que la independencia sea razonable | [Validación](../knowledge/validation/validation_strategy.md) |
| Varias filas por entidad y se quiere generalizar a entidades nuevas | Mantener grupos completos en las particiones; considerar GroupKFold | Evita compartir información relacionada entre train y validación | El grupo debe representar la unidad real de generalización | [Validación](../knowledge/validation/validation_strategy.md) |
| El orden temporal determina qué información estará disponible | Entrenar con pasado y validar con periodos posteriores | Evalúa el paso de historia a futuro y evita mezclar futuro con pasado | Definir horizonte, ventanas y disponibilidad; la fuente no prescribe un backtest completo | [Validación](../knowledge/validation/validation_strategy.md) · [Leakage](../knowledge/preprocessing/data_leakage.md) |
| La métrica oficial está alineada con el objetivo del reto | Usarla para seleccionar candidatos y conservar unidades/contexto para interpretarla | La selección debe reflejar el criterio de evaluación | MAE, RMSE, MSE y R² resumen errores de manera distinta; no son intercambiables | [Métricas](../knowledge/metrics/regression_metrics.md) |
| MAE interesa por distancia absoluta media en unidades del target | Considerar MAE | Resume error absoluto medio y es menos sensible a extremos que RMSE | Da menos peso relativo a errores grandes | [Métricas](../knowledge/metrics/regression_metrics.md) |
| Penalizar más los errores grandes es importante para el objetivo | Considerar RMSE (o MSE si el criterio lo requiere) | La penalización cuadrática aumenta el peso de errores grandes | RMSE es sensible a extremos; MSE queda en unidades al cuadrado | [Métricas](../knowledge/metrics/regression_metrics.md) |
| Pocos errores extremos dominan el error medio y se quiere resumir el caso típico | Revisar el error absoluto mediano además de la métrica principal | La mediana reduce la influencia de unos pocos errores grandes | Puede ocultar extremos relevantes para el uso | [Métricas](../knowledge/metrics/regression_metrics.md) |
| Hay faltantes | Medir patrón y significado; comparar tratamientos justificables | La ausencia puede ser informativa y eliminar/imputar cambia los datos | No hay tratamiento universal; aprender imputación solo con train | [Auditoría](../knowledge/eda/dataset_audit.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) |
| Hay categorías nominales | Codificar según el significado, por ejemplo one-hot cuando la cardinalidad lo permita | Evita imponer orden numérico artificial | Definir el tratamiento de categorías no vistas y ajustar vocabulario en train | [Limpieza](../knowledge/preprocessing/data_cleaning.md) · [Features](../knowledge/feature_engineering/feature_engineering.md) |
| Un valor extremo parece anómalo | Contrastar con dominio, unidad, captura y distribución antes de actuar | Un extremo puede ser válido o un error | No hay umbral universal; no eliminar o recortar solo por rareza | [Auditoría](../knowledge/eda/dataset_audit.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) |
| Un modelo lineal deja patrón o subajusta, o se considera un árbol/boosting | Formular un cambio de representación o familia y compararlo con el baseline | Lineales requieren representar la forma; árboles aprenden cortes/interacciones; boosting corrige señal secuencialmente | Un patrón o mejora no demuestra por sí mismo la causa ni la generalización | [Modelos](../knowledge/models/tabular_models.md) · [Ensembles](../knowledge/ensembles/boosting_models.md) |
| Train resulta mucho mejor que validación, los folds varían o una mejora es pequeña | Revisar complejidad, split, estabilidad y coste antes de añadir tuning | La brecha y dispersión informan sobreajuste o sensibilidad | No existe un umbral universal de brecha o mejora suficiente | [Validación](../knowledge/validation/validation_strategy.md) · [Tuning](../knowledge/tuning/hyperparameter_tuning.md) |

## 4. Procedimiento paso a paso

### Paso 1. Entender el objetivo de regresión

- **Objetivo.** Asegurar que una salida numérica responde a la tarea real.
- **Acciones concretas.** Define la decisión que apoyará el modelo; precisa qué representa una fila; identifica `y`, su unidad, rango y cuándo se conoce. Confirma el instante y horizonte de predicción, qué entradas estarán disponibles entonces y qué salida solicita el reto.
- **Decisión que se debe tomar.** Determina si el target es una cantidad continua y si la predicción puntual pedida corresponde a la necesidad. Documenta ambigüedades o proxies.
- **Resultado esperado.** Una definición verificable de unidad, `y`, unidad de medida, instante y salida.
- **Errores que se deben evitar.** Elegir modelo antes de formular la pregunta; confundir predictor disponible después del evento con predictor válido; interpretar asociación como causalidad.
- **Fuente.** [Formulación del problema](../knowledge/problem_definition/problem_formulation.md).

### Paso 2. Auditar el dataset

- **Objetivo.** Detectar problemas de estructura, significado y calidad antes de limpiar o entrenar.
- **Acciones concretas.** Registra forma, claves, unidad de cada fila, tipos físicos y semánticos, valores únicos, categorías, faltantes por columna/grupo/periodo, filas idénticas y claves repetidas. Resume distribuciones y rangos de `y` y predictores; busca unidades mezcladas, topes y posibles anomalías. Identifica columnas posteriores al resultado o relacionadas con captura/etiquetado.
- **Decisión que se debe tomar.** Clasifica cada hallazgo como error confirmado, caso válido, duda de dominio o riesgo de partición/leakage. Conserva los casos no resueltos como preguntas.
- **Resultado esperado.** Inventario de columnas, definición de unidad, calidad y dependencias relevantes para la validación.
- **Errores que se deben evitar.** Borrar duplicados por defecto, convertir todo entero en magnitud continua, o descartar extremos porque son raros. Una correlación o un score inusual son señales para investigar, no prueba de fuga.
- **Fuente.** [Auditoría](../knowledge/eda/dataset_audit.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) · [Leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 3. Definir validación y métrica

- **Objetivo.** Medir el desempeño que corresponde al uso futuro y al criterio del reto.
- **Acciones concretas.** Fija primero qué unidad no puede cruzar particiones: fila independiente, entidad o tiempo. Usa partición por grupos si se busca generalizar a entidades nuevas; si importa el futuro, ordena y evalúa sobre periodos posteriores. Para filas independientes, elige holdout o K-Fold según coste y necesidad de observar variación. Alinea la métrica con el objetivo y verifica si coincide con la oficial; reserva un test final si el diseño del reto lo permite.
- **Decisión que se debe tomar.** Registra splitter, unidad/grupo/tiempo, folds o regla de holdout, métrica principal y métricas complementarias justificadas.
- **Resultado esperado.** Protocolo fijo para baseline y todos los experimentos, que estima el tipo de generalización esperado.
- **Errores que se deben evitar.** Suponer que split aleatorio sirve siempre; usar test para seleccionar o ajustar repetidamente; escoger MAE/RMSE/R² sin entender qué penalizan o representan.
- **Fuente.** [Validación](../knowledge/validation/validation_strategy.md) · [Métricas de regresión](../knowledge/metrics/regression_metrics.md).

### Paso 4. Preparar un baseline

- **Objetivo.** Obtener una referencia sencilla antes de añadir complejidad.
- **Acciones concretas.** Separa `X` e `y`; implementa una predicción constante basada en media o mediana, o mide el proceso vigente si existe. Calcula la estadística solo con el entrenamiento y evalúa con el splitter y métrica definidos. Si los datos y la representación lo permiten, añade una regresión lineal como primer modelo simple. Registra configuración y resultado.
- **Decisión que se debe tomar.** Decide si el baseline está correctamente implementado y si tiene sentido comparativo para la tarea.
- **Resultado esperado.** Score de referencia reproducible bajo el mismo protocolo que usarán candidatos posteriores.
- **Errores que se deben evitar.** Calcular la constante con validación/test, comparar modelos en particiones distintas o asumir que superar una heurística débil demuestra utilidad.
- **Fuente.** [Métricas](../knowledge/metrics/regression_metrics.md) · [Modelos tabulares](../knowledge/models/tabular_models.md) · [Validación](../knowledge/validation/validation_strategy.md).

### Paso 5. Preprocesar los datos sin leakage

- **Objetivo.** Preparar una representación coherente con las columnas y el modelo, preservando una evaluación limpia.
- **Acciones concretas.** Corrige únicamente formatos/unidades con evidencia. Para faltantes, compara conservar, imputar u otra acción justificada; para nominales, codifica respetando semántica. Escala cuando la familia dependa de magnitudes (por ejemplo, métodos de distancia o modelos penalizados), no por rutina. Encapsula imputación, codificación, escalado y estimador en un pipeline cuando esas transformaciones aprendan de los datos. Ajusta cada `fit` solo en training y transforma validación/test sin reajustar.
- **Decisión que se debe tomar.** Elige el tratamiento compatible con el significado, modelo y categorías esperadas; deja las dudas de dominio sin resolver explícitas.
- **Resultado esperado.** Flujo único, reproducible para entrenamiento y predicción, con transformaciones aprendidas dentro de cada fold.
- **Errores que se deben evitar.** Ajustar imputación, escaladores, vocabularios, cuantiles, selección de variables o agregados antes de separar; escalar árboles por costumbre; transformar validación por separado.
- **Fuente.** [Limpieza](../knowledge/preprocessing/data_cleaning.md) · [Leakage](../knowledge/preprocessing/data_leakage.md) · [Features](../knowledge/feature_engineering/feature_engineering.md) · [Modelos](../knowledge/models/tabular_models.md).

### Paso 6. Entrenar y evaluar modelos

- **Objetivo.** Comprobar si candidatos razonables mejoran la referencia fuera de muestra.
- **Acciones concretas.** Empieza por un modelo simple compatible con tarea y datos. Considera regresión lineal cuando la forma lineal sea razonable; árboles si hay umbrales/interacciones y un regresor de bosque o boosting si un candidato arbóreo inestable o con señal residual justifica compararlos. Mantén fijos métrica, folds/particiones y datos de evaluación. Guarda score por fold y resumen, junto a coste y configuración.
- **Decisión que se debe tomar.** Compara desempeño de validación, dispersión, restricciones e interpretabilidad; continúa solo con candidatos que aporten una hipótesis útil bajo el protocolo común.
- **Resultado esperado.** Comparación auditable del baseline y candidatos sin usar test para elegir.
- **Errores que se deben evitar.** Elegir por score de entrenamiento, asumir que un algoritmo gana siempre o que un ensemble mejora automáticamente, comparar resultados con splits incompatibles.
- **Fuente.** [Modelos](../knowledge/models/tabular_models.md) · [Ensembles y boosting](../knowledge/ensembles/boosting_models.md) · [Validación](../knowledge/validation/validation_strategy.md).

### Paso 7. Revisar errores y mejorar el modelo

- **Objetivo.** Encontrar observaciones o grupos de errores que orienten una revisión, sin atribuirles causalidad automática.
- **Acciones concretas.** Conserva predicción, valor observado y residuo `y - ŷ` por fila. Resume errores absolutos y revisa residuos frente a predicciones, variables relevantes y orden temporal/grupos cuando proceda. Inspecciona casos con errores grandes y compáralos con unidad, datos originales, faltantes y procedencia de features. Si aparece curvatura, embudo, dependencia consecutiva, brecha train-validación o fallos concentrados en grupos, formula una hipótesis sobre representación, calidad, estructura de split o complejidad y prueba un cambio controlado.
- **Decisión que se debe tomar.** Elige una revisión de datos, features, validación o modelo respaldada por evidencia observable; considera métricas complementarias si el objetivo necesita distinguir errores típicos de extremos.
- **Resultado esperado.** Lista priorizada de hipótesis comprobables y un experimento por cambio relevante.
- **Errores que se deben evitar.** Tratar una gráfica o patrón como prueba de causa; diagnosticar generalización solo con residuos de entrenamiento; eliminar extremos o cambiar target sin confirmar dominio.
- **Fuente.** [Métricas y residuos](../knowledge/metrics/regression_metrics.md) · [Auditoría](../knowledge/eda/dataset_audit.md) · [Features](../knowledge/feature_engineering/feature_engineering.md).

### Paso 8. Ajustar hiperparámetros

- **Objetivo.** Explorar configuraciones después de tener baseline y validación representativa.
- **Acciones concretas.** Elige pocos hiperparámetros con impacto plausible y un presupuesto que quepa. Usa ajuste manual acotado para pocos parámetros, grid para un espacio discreto pequeño o búsqueda aleatoria para controlar pruebas en un espacio más amplio. Evalúa el pipeline completo dentro del splitter; compara bajo los mismos folds. Registra parámetros, scores, dispersión y coste.
- **Decisión que se debe tomar.** Conserva configuraciones cuya diferencia y estabilidad justifiquen la complejidad; no hay mejora mínima universal establecida por las fuentes.
- **Resultado esperado.** Configuración seleccionada sin contaminación del test y con el espacio de búsqueda documentado.
- **Errores que se deben evitar.** Usar tuning para compensar fuga o formulación defectuosa; construir búsquedas inviables; escoger hiperparámetros con test.
- **Fuente.** [Ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md) · [Validación](../knowledge/validation/validation_strategy.md) · [Leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 9. Seleccionar y guardar el modelo final

- **Objetivo.** Cerrar el candidato que mejor satisface la evaluación y las restricciones del reto.
- **Acciones concretas.** Revisa los resultados registrados bajo la métrica principal, estabilidad, coste e interpretabilidad. Una vez cerradas decisiones, reajusta el procedimiento elegido con los datos de desarrollo disponibles; si existe un test final separado, evalúalo una vez. Guarda estimador y transformaciones necesarias como una unidad, además de versión/regla de split, semilla cuando aplique, columnas y configuración.
- **Decisión que se debe tomar.** Selecciona un candidato que cumpla criterio y operación del reto; documenta si los resultados son inestables o la evidencia insuficiente.
- **Resultado esperado.** Artefacto reproducible que transforma entradas igual que durante validación y produce predicciones para las filas requeridas.
- **Errores que se deben evitar.** Retocar el modelo después de mirar test; guardar solo el estimador sin los pasos aprendidos; declarar que un score garantiza generalización.
- **Fuente.** [Validación](../knowledge/validation/validation_strategy.md) · [Leakage](../knowledge/preprocessing/data_leakage.md) · [Features](../knowledge/feature_engineering/feature_engineering.md).

### Paso 10. Verificar la entrega

- **Objetivo.** Confirmar que el resultado técnico se puede consumir conforme a las instrucciones.
- **Acciones concretas.** Comprueba que se predijeron las filas requeridas, que el número y orden corresponden al identificador esperado, que las predicciones son numéricas y que el esquema coincide con el ejemplo/especificación del reto. Verifica el archivo de entrega, nombre, columnas y formato contra las instrucciones originales; vuelve a ejecutar la generación desde el flujo guardado.
- **Decisión que se debe tomar.** Entrega solo cuando esquema y formato coincidan exactamente con los requisitos publicados.
- **Resultado esperado.** Salida reproducible y aceptable por el mecanismo del reto.
- **Errores que se deben evitar.** Inventar nombres de archivo o columnas, omitir IDs requeridos, cambiar orden de filas o guardar predicciones de validación como entrega.
- **Fuente.** [Formulación](../knowledge/problem_definition/problem_formulation.md) para requisitos de salida. **Vacío de conocimiento:** los documentos consultados no especifican formatos universales de entrega; confirmar el formato en las instrucciones del reto.

## 5. Reglas de decisión

- Si no está clara la unidad de fila o cuándo están disponibles las columnas, detén la selección de modelos y aclara esos contratos primero. [Formulación](../knowledge/problem_definition/problem_formulation.md)
- Si hay entidades repetidas y la pregunta es rendimiento en entidades nuevas, particiona por entidad; si la pregunta es predecir futuro, preserva el orden temporal. [Validación](../knowledge/validation/validation_strategy.md)
- Si la métrica oficial corresponde al objetivo, úsala para comparar; acompáñala de métricas con interpretación distinta solo si aportan información útil. MAE expresa distancia absoluta media, RMSE da peso mayor a errores grandes y R² no expresa magnitud de error en unidades. [Métricas](../knowledge/metrics/regression_metrics.md)
- Si faltan datos, categoriza o codifica según semántica; no decidas borrar, imputar o asignar orden solo por el `dtype`. Aprende parámetros y vocabularios con training. [Limpieza](../knowledge/preprocessing/data_cleaning.md)
- Si un extremo parece sospechoso, verifica dominio y procedencia; conservar, corregir, filtrar o limitar depende de evidencia. Las fuentes no fijan umbrales generales. [Auditoría](../knowledge/eda/dataset_audit.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md)
- Si un candidato mejora train pero no validación, o cambia mucho entre folds, revisa complejidad, partición y costo antes de seguir. No existe regla numérica universal para brecha o mejora suficiente. [Validación](../knowledge/validation/validation_strategy.md) · [Tuning](../knowledge/tuning/hyperparameter_tuning.md)
- Si se añade una transformación, feature, familia, ensemble o hiperparámetro, cambia una hipótesis a la vez cuando sea posible y compárala con el baseline bajo condiciones coherentes. Esta es una **síntesis práctica** de las recomendaciones de comparación y trazabilidad. [Features](../knowledge/feature_engineering/feature_engineering.md) · [Modelos](../knowledge/models/tabular_models.md) · [Ensembles](../knowledge/ensembles/boosting_models.md) · [Tuning](../knowledge/tuning/hyperparameter_tuning.md)

## 6. Baseline mínimo

1. Separa explícitamente `X` (predictoras) e `y` (objetivo); excluye `y` e información no disponible al predecir de `X`.
2. Define la validación desde la unidad de generalización: holdout/K-Fold para filas independientes, grupos para entidades relacionadas o secuencia para tiempo. Fija semilla si la partición es aleatoria y repetible.
3. Usa la métrica del reto si refleja el objetivo. Conserva el contexto de escala y unidad; evalúa todos los baselines y modelos sobre las mismas observaciones/folds.
4. Calcula con training una constante de media o mediana y evalúala primero. Si el esquema y la semántica son compatibles, añade una regresión lineal como modelo simple.
5. Para un modelo que necesite tratamiento, incorpora imputación/codificación (y escalado solo si la familia lo requiere) en el flujo de ajuste. En CV, todo lo que aprende parámetros debe ajustarse dentro de cada fold.
6. Anota la configuración, splitter, métrica y resultado para que cada cambio posterior tenga una referencia reproducible.

La media/mediana constante y la regresión lineal están documentadas como opciones de baseline, no como elección universal. Una heurística o proceso operativo existente puede ser más apropiado. [Métricas](../knowledge/metrics/regression_metrics.md) · [Modelos](../knowledge/models/tabular_models.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) · [Validación](../knowledge/validation/validation_strategy.md)

## 7. Análisis de errores y predicciones

Guarda al menos identificador/unidad, valor observado, predicción y residuo `y - ŷ`. La magnitud del residuo está en unidades del objetivo; positivo significa subpredicción con esta convención. Revisa errores por grupos o periodos solo si son parte de la estructura relevante y tienen suficientes observaciones para interpretación.

Los residuos de ajuste aproximadamente centrados sin patrón sistemático son una guía diagnóstica; curvatura puede sugerir forma no lineal no representada, embudo variación no constante y residuos consecutivos parecidos dependencia. Estos patrones motivan investigar datos, features, modelo o validación, pero no demuestran una causa ni estiman por sí solos el error futuro. Contrasta con resultados fuera de muestra. [Métricas](../knowledge/metrics/regression_metrics.md) · [Validación](../knowledge/validation/validation_strategy.md)

Al inspeccionar errores grandes, comprueba si corresponden a errores de datos, unidades mezcladas, target truncado, grupos/periodos distintos o casos válidos extremos. No cambies el target ni descartes filas sin evidencia. Si unos pocos extremos dominan la media, reporta una métrica complementaria pertinente (por ejemplo, MedAE) sin sustituir silenciosamente la métrica oficial. [Auditoría](../knowledge/eda/dataset_audit.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) · [Métricas](../knowledge/metrics/regression_metrics.md)

## 8. Comparación de experimentos y registro

Para cada prueba, registra como mínimo:

| Campo | Qué anotar |
|---|---|
| Configuración | Modelo, transformaciones/features, hiperparámetros y versión del código/datos si está disponible |
| Validación | Holdout/folds, clave de grupo o rango temporal, semilla/regla y observaciones evaluadas |
| Métrica | Nombre, dirección de mejora y unidades/contexto cuando aplique |
| Resultado | Score por fold, resumen y baseline bajo el mismo protocolo |
| Observaciones | Brecha train-validación, dispersión, errores relevantes, costo y limitaciones |

Compara en condiciones coherentes: misma definición de `y`, población, partición y métrica. Un cambio de limpieza, representación, variables, modelo, ensemble o hiperparámetros constituye una hipótesis experimental; incorpora solo alternativas respaldadas por las fuentes y pertinentes a los datos. Si varias dimensiones cambian a la vez, limita lo que puede atribuirse al cambio. Una mejora en una sola partición no garantiza generalización y reutilizar repetidamente la misma validación puede adaptar decisiones a ella. Mantén un test final sin usar para selección, si existe. [Validación](../knowledge/validation/validation_strategy.md) · [Features](../knowledge/feature_engineering/feature_engineering.md) · [Ensembles](../knowledge/ensembles/boosting_models.md) · [Tuning](../knowledge/tuning/hyperparameter_tuning.md)

## 9. Criterios para detenerse

- Detén el tuning al agotar el presupuesto fijado o cuando los ajustes sean demasiado costosos para el tiempo disponible; no hay umbral universal de tiempo, mejora o memoria.
- Si un entrenamiento ya consume el presupuesto, no inicies una búsqueda de múltiples pruebas que no podrás completar.
- Si un candidato no supera una referencia válida, revisa formulación, target, split, métrica y preprocesamiento antes de añadir complejidad. Superar una referencia débil tampoco prueba utilidad.
- Si las diferencias observadas son pequeñas frente a la dispersión o inestabilidad entre particiones, registra la incertidumbre y prioriza estabilidad/costo según el reto, sin declarar una mejora concluyente.
- Cierra decisiones antes de la evaluación final; una consulta al test seguida de ajustes lo convierte en parte del desarrollo.

Los primeros dos criterios y el carácter no universal de los umbrales están expresos en la fuente de tuning. La comparación con referencia y el cierre del test se respaldan en [Validación](../knowledge/validation/validation_strategy.md), [Tuning](../knowledge/tuning/hyperparameter_tuning.md) y [Métricas](../knowledge/metrics/regression_metrics.md).

## 10. Riesgos y controles

| Riesgo | Control operativo | Fuente |
|---|---|---|
| Target ambiguo, mal medido o truncado | Revisar definición, unidad, faltantes, rango y proceso de captura antes de entrenar | [Formulación](../knowledge/problem_definition/problem_formulation.md) · [Auditoría](../knowledge/eda/dataset_audit.md) |
| Predictor posterior al evento o derivado del resultado | Reconstruir procedencia, fórmula y timestamp; mantener en duda hasta aclarar disponibilidad | [Leakage](../knowledge/preprocessing/data_leakage.md) |
| Fuga por preparación previa a partición | Ajustar imputación, escalado, categorías, selección y agregados solo en train/fold mediante pipeline | [Leakage](../knowledge/preprocessing/data_leakage.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) |
| Filas correlacionadas cruzan particiones | Mantener juntas las entidades relacionadas si esa es la unidad de generalización | [Validación](../knowledge/validation/validation_strategy.md) |
| Futuro en train, features o joins | Split secuencial; revisar ventanas, rellenos y joins contra el momento real de predicción | [Leakage](../knowledge/preprocessing/data_leakage.md) |
| Métrica oculta los errores que más importan | Elegir por objetivo; revisar métrica complementaria con interpretación explícita | [Métricas](../knowledge/metrics/regression_metrics.md) |
| Eliminar duplicados o extremos válidos | Investigar unidad, dominio y causa antes de limpiar | [Auditoría](../knowledge/eda/dataset_audit.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) |
| Sobreajuste o validación reutilizada | Comparar train/validación, dispersión entre folds y conservar test aislado | [Validación](../knowledge/validation/validation_strategy.md) · [Tuning](../knowledge/tuning/hyperparameter_tuning.md) |
| Modelo final no reproduce la predicción | Guardar pipeline y parámetros; reaplicar el mismo flujo a la entrada de entrega | [Features](../knowledge/feature_engineering/feature_engineering.md) · [Limpieza](../knowledge/preprocessing/data_cleaning.md) |

## 11. Checklist de ejecución

- [ ] Definí qué representa una fila y la unidad de generalización.
- [ ] Confirmé el target, su unidad, disponibilidad y salida solicitada.
- [ ] Revisé fechas, grupos, duplicados, faltantes, tipos, rangos y posibles topes.
- [ ] Investigué columnas que puedan contener información posterior o relacionada con `y`.
- [ ] Elegí una validación compatible con filas, entidades o tiempo.
- [ ] Elegí métrica por objetivo y comprobé el criterio oficial.
- [ ] Registré un baseline evaluado con el mismo protocolo.
- [ ] Aprendí transformaciones solo con train/fold y guardé el pipeline completo.
- [ ] Comparé hipótesis y candidatos con particiones y métricas coherentes.
- [ ] Revisé errores, variación entre particiones, costo y restricciones.
- [ ] Cerré selección antes de consultar test final.
- [ ] Verifiqué reproducibilidad y formato exacto de entrega contra las instrucciones.

## 12. Fuentes de `knowledge/`

Documentos consultados (todos existen en el repositorio):

| Documento | Uso en este playbook |
|---|---|
| [Formulación del problema](../knowledge/problem_definition/problem_formulation.md) | Unidad, target, predictores disponibles, objetivo y entrega |
| [Auditoría del dataset](../knowledge/eda/dataset_audit.md) | Estructura, tipos, faltantes, duplicados, rangos, grupos y señales de fuga |
| [Limpieza de datos](../knowledge/preprocessing/data_cleaning.md) | Limpieza justificada, faltantes, categorías, extremos y pipeline |
| [Data leakage](../knowledge/preprocessing/data_leakage.md) | Disponibilidad temporal, ajuste de transformaciones y particiones |
| [Feature engineering](../knowledge/feature_engineering/feature_engineering.md) | Transformaciones, representación y comparación de variables |
| [Validación](../knowledge/validation/validation_strategy.md) | Holdout, K-Fold, grupos, temporalidad, reproducibilidad y test |
| [Métricas de regresión](../knowledge/metrics/regression_metrics.md) | MAE, MSE, RMSE, R², MedAE y baselines |
| [Modelos tabulares](../knowledge/models/tabular_models.md) | Regresión lineal, árboles, complejidad y comparación de familias |
| [Ensembles y boosting](../knowledge/ensembles/boosting_models.md) | Bosques, boosting, stacking, control de complejidad y costos |
| [Ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md) | Métodos de búsqueda, condiciones previas y presupuesto |

## 13. Vacíos o límites del playbook

- **Forecasting temporal:** validación respalda splits secuenciales y cautela con ventanas, pero las fuentes consultadas no definen tamaños de ventana, gaps, horizonte múltiple, backtesting completo ni manejo general de estacionalidad. Se debe declarar como límite y definir con el contexto del reto.
- **Transformaciones del target:** las fuentes consultadas no proporcionan un procedimiento general para decidir/validar transformaciones de `y`; no se prescriben aquí.
- **Outliers y datos faltantes:** no existe una política universal de umbral, recorte o imputación ni diagnóstico completo de mecanismos de ausencia. Investigar dominio y comparar tratamientos dentro de validación.
- **Selección estadística de métrica/modelo:** no hay umbral universal para mejora suficiente, brecha de sobreajuste o estabilidad; no se atribuye significancia estadística a diferencias sin procedimiento documentado.
- **Entrega:** los nombres, columnas y formatos dependen de las instrucciones de cada reto; no se define un formato genérico.
- **Cobertura de familias:** las fuentes desarrollan de forma acotada algunos regresores especializados; cualquier técnica no cubierta debe tratarse como vacío de conocimiento, no como recomendación validada.
- **Estado:** el documento permanece en `draft`; no se afirma validación práctica mediante casos de uso.
