---
title: "Competition Workflow"
document_type: "playbook"
status: "draft"
scope: "Flujo general para coding tests y competencias de Machine Learning"
---

# Flujo de trabajo para competencias de Machine Learning

## 1. Propósito y alcance

Este playbook guía el trabajo desde la lectura del reto hasta la entrega de una solución reproducible. Su ruta principal cubre problemas tabulares supervisados de clasificación y regresión. Cada etapa produce una decisión o artefacto verificable; si una inspección invalida una decisión anterior, vuelve al paso correspondiente.

La ruta no presupone que todas las filas sean independientes ni que todos los retos sean tabulares. Si hay orden temporal, entidades repetidas, imágenes, texto, clustering u otra estructura especial, primero identifica la unidad de generalización y busca conocimiento específico en `knowledge/`. La validación temporal disponible cubre principios de orden y prevención de futuro, pero no un procedimiento completo de forecasting. No apliques automáticamente un split aleatorio ni un modelo tabular. [Formulación](../knowledge/problem_definition/problem_formulation.md), [validación](../knowledge/validation/validation_strategy.md)

Las recomendaciones se marcan como **Respaldada** cuando los documentos citados las sustentan, **Síntesis** cuando integran más de una fuente, y **Recomendación operativa derivada** cuando organizan el trabajo sin ser una regla prescrita en `knowledge/`. Los enlaces llevan a los documentos temáticos para el detalle; este playbook no los reemplaza.

## 2. Información que debes obtener al iniciar

Antes de modelar, reúne y anota:

- **Objetivo y uso:** qué decisión o resultado apoya la predicción y qué salida se espera.
- **Unidad de análisis:** qué representa una fila; si hay entidades repetidas, grupos o secuencia temporal.
- **Target:** nombre, significado, tipo y momento en que se genera. Comprueba que corresponda al objetivo y no sea solo un proxy conveniente.
- **Predictores disponibles:** qué información existe realmente en el momento de predecir; identifica IDs, timestamps y campos que podrían revelar el target o información futura.
- **Datos y archivos:** formatos, esquema, diccionario, archivos de entrenamiento/prueba y cualquier archivo de ejemplo o plantilla.
- **Evaluación oficial:** métrica, dirección de mejora, formato de predicción y reglas indicadas por la competencia. Confirma estos detalles en el enunciado; no los infieras del nombre de una columna.
- **Restricciones:** recursos, dependencias, latencia, memoria, límite de envíos y archivos que se deben entregar, cuando el enunciado los especifique.

**Punto de control:** no avances a modelado si no puedes distinguir con confianza `X` (predictores) de `y` (target), o si no sabes qué representa el score oficial. Registra lo que siga sin confirmarse como pregunta pendiente. [Formulación](../knowledge/problem_definition/problem_formulation.md), [auditoría del dataset](../knowledge/eda/dataset_audit.md)

## 3. Flujo de decisión general

```text
Leer reglas, objetivo, archivos y formato esperado
                         ↓
Definir unidad, target, predictores y tipo de tarea
                         ↓
Auditar calidad, grupos, tiempo, desbalance y leakage posible
                         ↓
Elegir métrica y validación que representen el reto
                         ↓
Crear baseline y guardar sus predicciones/configuración
                         ↓
Comparar pipelines y modelos con evaluación consistente
                         ↓
Mejorar variables y ajustar una shortlist si hay presupuesto
                         ↓
Elegir y ajustar el flujo final; verificar entrega y reproducibilidad
```

**Regla de retorno:** si el score es sorprendentemente bueno, el target no representa el uso, una columna parece posterior al resultado, los grupos se cruzan entre particiones o los resultados cambian mucho entre splits, detén la comparación afectada y revisa formulación, auditoría, leakage o validación antes de seguir. Estas señales están documentadas como riesgos; la decisión de volver a la etapa es una síntesis operativa. [Auditoría](../knowledge/eda/dataset_audit.md), [leakage](../knowledge/preprocessing/data_leakage.md), [validación](../knowledge/validation/validation_strategy.md)

## 4. Procedimiento paso a paso

### Paso 1. Interpretar el reto

**Objetivo.** Convertir el enunciado en una tarea evaluable y ejecutable.

**Acciones.** Lee reglas y criterios de entrega. Describe en pocas líneas el objetivo, quién o qué representa cada observación, la salida requerida, la columna objetivo, la métrica oficial y las restricciones explícitas. Clasifica la tarea (por ejemplo, clasificación o regresión) y confirma qué datos estarán disponibles al predecir. Si no hay target, investiga si el problema es realmente no supervisado en lugar de fabricar uno.

**Decisión.** ¿La tarea, el target y la salida esperada están definidos y son coherentes? Si no, formula preguntas o registra la ambigüedad; no elijas un modelo para compensarla.

**Resultado esperado.** Una ficha del reto con objetivo, unidad, `y`, `X` candidato, tipo de tarea, métrica y requisitos por confirmar.

**Errores que se deben evitar.** Elegir algoritmo antes de formular; confundir ID con predictor útil; usar una etiqueta proxy sin comprobar su relación con la decisión; asumir que la métrica técnica equivale automáticamente a utilidad.

**Referencia.** [Formulación del problema](../knowledge/problem_definition/problem_formulation.md).

### Paso 2. Inspeccionar los archivos y el dataset

**Objetivo.** Confirmar que los archivos y datos corresponden a la tarea y localizar riesgos antes de transformarlos.

**Acciones.** Lista los archivos y formatos; revisa dimensiones, nombres y tipos de columnas, significado semántico, filas de muestra, valores faltantes, duplicados, rangos y distribuciones. Comprueba el target y su distribución. Busca IDs repetidos, unidades agrupadas, orden temporal, valores centinela y campos derivados del resultado. Conserva evidencia del estado original antes de limpiar.

**Decisión.** ¿Las observaciones son aproximadamente independientes, o hay grupos/tiempo que cambien el split? ¿Los valores raros son errores, eventos válidos o incógnitas de dominio? ¿Hay cobertura suficiente para la evaluación? Trata las señales como hipótesis para investigar, no como prueba automática de un defecto.

**Resultado esperado.** Un inventario de archivos y columnas, confirmación provisional de `X`/`y`, hallazgos de calidad y riesgos/preguntas de grupos, tiempo y leakage.

**Errores que se deben evitar.** Limpiar antes de entender valores y granularidad; borrar duplicados sin determinar si son registros repetidos válidos; inferir semántica solo por `dtype`; asumir que una correlación demuestra causalidad.

**Referencia.** [Auditoría del dataset](../knowledge/eda/dataset_audit.md), [limpieza de datos](../knowledge/preprocessing/data_cleaning.md), [leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 3. Definir validación y métrica

**Objetivo.** Estimar el rendimiento en observaciones que representen el uso previsto y medir el objetivo correcto.

**Acciones.** Define primero qué unidad no debe cruzar particiones: fila, entidad o tiempo. Para filas independientes, considera holdout o K-Fold según coste y estabilidad deseada. En clasificación con clases desiguales, considera estratificar, comprobando que esto no rompa grupos o tiempo. Si se generaliza a entidades nuevas, conserva cada grupo en una sola partición. Si la predicción es hacia el futuro, conserva el orden y usa split secuencial o validación temporal documentada. Elige la métrica oficial; usa métricas complementarias solo para interpretar aspectos pertinentes. Mantén idénticos split/folds y métrica al comparar candidatos.

**Decisión.** ¿El esquema local aproxima la evaluación de la competencia? ¿La métrica refleja la salida y coste del error? Si la estructura conjunta de grupos, tiempo y clases no queda cubierta por las guías, marca el caso como vacío y no lo resuelvas con una división aleatoria por conveniencia.

**Resultado esperado.** Protocolo escrito con métrica, dirección de mejora, unidad de generalización, partición/folds, aleatoriedad fijada cuando corresponda y función de cualquier conjunto final reservado.

**Errores que se deben evitar.** Compartir entidad relacionada entre train y validación cuando se quiere generalizar a entidades nuevas; mezclar futuro con pasado; tratar estratificación como solución a dependencia; cambiar métrica o split entre experimentos; usar el conjunto final repetidamente para seleccionar.

**Referencia.** [Estrategia de validación](../knowledge/validation/validation_strategy.md), [métricas de clasificación](../knowledge/metrics/classification_metrics.md), [métricas de regresión](../knowledge/metrics/regression_metrics.md).

### Paso 4. Construir un baseline

**Objetivo.** Obtener una referencia sencilla que permita detectar errores de flujo y medir si un cambio aporta.

**Acciones.** Separa `X` e `y` con columnas justificadas. Elige un predictor trivial/heurística cuando proceda y un modelo inicial respaldado por la tarea y los datos. Incluye el preprocesamiento indispensable en un pipeline. Evalúa el baseline con la misma métrica y partición que se usarán para los candidatos; guarda score y predicciones de validación.

**Decisión.** ¿El baseline se ejecuta y produce predicciones con forma compatible? ¿Supera una referencia trivial cuando esta tiene sentido? Si el resultado parece imposible o falla, depura datos, target, métrica y pipeline antes de añadir complejidad.

**Resultado esperado.** Una ejecución reproducible con configuración, split, métrica, score, predicciones y artefacto del flujo de preprocesamiento/modelo.

**Errores que se deben evitar.** Presentar un baseline como universal; calcular estadísticas del baseline o transformaciones con validación; cambiar el split para mejorar el score; comparar resultados obtenidos sobre observaciones distintas.

**Referencia.** [Modelos tabulares](../knowledge/models/tabular_models.md), [métricas de clasificación](../knowledge/metrics/classification_metrics.md), [métricas de regresión](../knowledge/metrics/regression_metrics.md), [validación](../knowledge/validation/validation_strategy.md).

### Paso 5. Preparar datos y prevenir leakage

**Objetivo.** Crear transformaciones justificadas que puedan repetirse en entrenamiento y predicción sin contaminar la evaluación.

**Acciones.** Decide cómo tratar faltantes, tipos inconsistentes, categorías, duplicados y valores extremos según la evidencia y el dominio. Separa predictores no disponibles al momento real de inferencia. Pon imputación, codificación, escalado, selección de variables y cualquier transformación que aprenda parámetros dentro del pipeline ajustado en cada fold. Revisa también cómo se construyeron previamente las columnas: un pipeline no corrige una feature que ya contiene información futura o global.

**Decisión.** ¿Cada transformación tiene motivo observable y se aprende solo de entrenamiento? Si una opción depende de conocimiento del dominio que no está disponible, conserva la incertidumbre en vez de convertirla en regla.

**Resultado esperado.** Pipeline que transforma train y validación de forma coherente y que puede aplicarse en inferencia, más decisiones de limpieza registradas.

**Errores que se deben evitar.** Ajustar imputadores, escaladores, codificadores o selección sobre todo el dataset antes del split; incorporar datos posteriores al resultado; transformar train y validación con reglas distintas; eliminar filas sin evaluar el efecto sobre grupos y distribución.

**Referencia.** [Limpieza de datos](../knowledge/preprocessing/data_cleaning.md), [prevención de data leakage](../knowledge/preprocessing/data_leakage.md), [validación](../knowledge/validation/validation_strategy.md).

### Paso 6. Entrenar y comparar modelos

**Objetivo.** Comprobar si otras familias de modelos mejoran el baseline bajo la misma pregunta de evaluación.

**Acciones.** Selecciona candidatos compatibles con el tipo de tarea, tamaño/estructura de los datos y restricciones. Entrena cada uno mediante el mismo protocolo de validación y pipeline. Registra score por fold cuando aplique y su variación, junto con coste o limitaciones observadas. Inspecciona errores para detectar subgrupos o patrones que el promedio oculte.

**Decisión.** Conserva candidatos con mejora útil respecto del baseline y resultados suficientemente estables para el objetivo del reto. Si los cambios de orden entre candidatos son pequeños frente a la variación observada, evita afirmar que hay un ganador claro.

**Resultado esperado.** Comparación trazable de modelos y una shortlist justificada.

**Errores que se deben evitar.** Comparar entrenamientos con particiones diferentes; escoger por score de entrenamiento; probar muchas familias antes de comprobar fuga y validez; interpretar una cifra agregada sin unidades, clases o segmentos relevantes.

**Referencia.** [Modelos tabulares](../knowledge/models/tabular_models.md), [validación](../knowledge/validation/validation_strategy.md), [métricas de clasificación](../knowledge/metrics/classification_metrics.md), [métricas de regresión](../knowledge/metrics/regression_metrics.md).

### Paso 7. Mejorar variables y modelos

**Objetivo.** Probar cambios guiados por el problema y por evidencia de errores.

**Acciones.** Inspecciona errores/residuos y distribuciones; formula una hipótesis sobre qué representación o señal puede ayudar. Prueba nuevas variables, transformaciones o alternativas de modelo de una en una o en un conjunto controlado. Si las transformaciones se aprenden de los datos, intégralas al pipeline. Incluye boosting o ensembles solo si son compatibles y hay presupuesto para evaluarlos.

**Decisión.** Mantén el cambio si mejora la métrica relevante en la validación definida y no introduce leakage, inestabilidad o costes incompatibles. Si no mejora, revierte o registra la prueba como negativa; una mejora dudosa puede requerir repetir evaluación acorde a la variación observada.

**Resultado esperado.** Cambio reproducible con hipótesis, comparación con baseline y explicación de por qué se conserva o descarta.

**Errores que se deben evitar.** Añadir variables por volumen sin hipótesis; inspeccionar el conjunto final para diseñar features; aplicar técnicas no documentadas como si tuvieran respaldo; declarar que una familia siempre gana.

**Referencia.** [Ingeniería de variables](../knowledge/feature_engineering/feature_engineering.md), [boosting y ensembles](../knowledge/ensembles/boosting_models.md), [leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 8. Ajustar hiperparámetros

**Objetivo.** Afinar una shortlist prometedora después de que formulación, baseline y evaluación sean confiables.

**Acciones.** Delimita parámetros relevantes y un presupuesto viable. Usa búsqueda manual, en rejilla o aleatoria; considera opciones más costosas solo si el presupuesto y la infraestructura lo justifican. Evalúa el flujo completo con la misma métrica y validación; el preprocesamiento debe ajustarse dentro de cada fold. Guarda configuración, resultados por fold y coste.

**Decisión.** Inicia tuning cuando el baseline está registrado, el modelo candidato ya es razonable y existe presupuesto. Detente cuando se agote el presupuesto definido, cuando los ensayos no aporten mejora útil, cuando los resultados sean inestables o cuando el tiempo deba reservarse para verificaciones y entrega. **No hay umbrales universales** de mejora o duración en `knowledge/`; fija el criterio para el reto y etiquétalo como decisión operativa.

**Resultado esperado.** Mejor configuración conocida dentro del espacio y presupuesto probados, con comparación reproducible; no una afirmación de óptimo global.

**Errores que se deben evitar.** Tunear para compensar un split inválido, leakage, target equivocado o métrica ruidosa; combinar espacios de búsqueda inviables; ajustar con test final; olvidar el coste de entrenamiento/inferencia.

**Referencia.** [Ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md), [validación](../knowledge/validation/validation_strategy.md), [leakage](../knowledge/preprocessing/data_leakage.md).

### Paso 9. Seleccionar y guardar la solución final

**Objetivo.** Convertir la decisión de modelado en un artefacto ejecutable y repetible.

**Acciones.** Elige entre los candidatos con el protocolo declarado y considera score, variación, restricciones y errores relevantes. Conserva preprocesamiento y estimador en un flujo completo. Registra columnas, configuración, versión/regla del split y semilla cuando corresponda. Si existe una evaluación final separada, úsala según su función documentada y no para reiniciar un ciclo ilimitado de selección. Guarda las predicciones locales que sustentan la comparación.

**Decisión.** ¿La mejora frente al baseline es útil para el reto y coherente con la variación? ¿El artefacto reproduce el mismo flujo desde datos crudos hasta predicción? Si no, conserva la mejor alternativa verificable y registra la limitación.

**Resultado esperado.** Modelo/pipeline seleccionado, configuración y métricas de selección guardadas, con procedimiento de predicción identificable.

**Errores que se deben evitar.** Guardar solo el estimador si el preprocesamiento es necesario; elegir por la evaluación final después de muchas comparaciones; confundir una semilla con reproducibilidad completa; omitir el esquema de entrada.

**Referencia.** [Validación](../knowledge/validation/validation_strategy.md), [modelos tabulares](../knowledge/models/tabular_models.md), [ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md).

### Paso 10. Revisar y entregar

**Objetivo.** Confirmar que la solución seleccionada se puede ejecutar y cumple las instrucciones de la competencia.

**Acciones.** Repite la ruta documentada desde los archivos originales o deja instrucciones claras para hacerlo. Comprueba que el flujo de predicción incluye todas las transformaciones; que ninguna se ajustó con validación/test; que el target no aparece entre las entradas; que columnas, número de predicciones y valores son compatibles con el archivo requerido; y que los artefactos y dependencias usados estén identificados. Revisa el nombre, formato, ubicación y contenido solicitados en el enunciado.

**Decisión.** Entrega cuando el artefacto se genera de forma reproducible y pasa las comprobaciones que especifica el reto. Si falta un requisito en el enunciado o no se puede verificar, documéntalo como pendiente; no inventes un formato.

**Resultado esperado.** Archivos de entrega presentes y compatibles con las instrucciones, además de código/configuración suficiente para reproducir la solución.

**Errores que se deben evitar.** Asumir nombres o formatos de archivo; cambiar el preprocesamiento después de seleccionar sin reevaluar; olvidar que el orden/identificador de predicciones puede ser parte del contrato de entrega.

**Referencia.** [Formulación](../knowledge/problem_definition/problem_formulation.md), [validación](../knowledge/validation/validation_strategy.md), [prevención de leakage](../knowledge/preprocessing/data_leakage.md).

## 5. Reglas de decisión

Las reglas de la tabla son condicionales: su aplicación depende de la estructura y del objetivo observados. “Fuente” enlaza el respaldo; las decisiones de organización que no constituyen regla técnica se identifican como derivadas.

| Señal o situación | Acción recomendada | Motivo | Precaución | Fuente |
|---|---|---|---|---|
| Target o unidad de fila no está claro | Pausar modelado y completar la formulación | El algoritmo no resuelve una tarea mal definida | Confirmar la semántica con enunciado/dominio | [Formulación](../knowledge/problem_definition/problem_formulation.md) |
| Filas aproximadamente independientes | Holdout o K-Fold según coste y variación que se necesite observar | Estima generalización bajo la hipótesis de independencia | Verificar primero que no haya grupos o tiempo | [Validación](../knowledge/validation/validation_strategy.md) |
| Clases desiguales | Considerar holdout estratificado o Stratified K-Fold y una métrica adecuada | Mantiene proporciones aproximadas de clase | No resuelve grupos, dependencia ni una métrica mal elegida | [Validación](../knowledge/validation/validation_strategy.md), [clasificación](../knowledge/metrics/classification_metrics.md) |
| Se espera generalizar a entidades no vistas | Mantener cada grupo completo en una partición (por ejemplo, GroupKFold) | Evita compartir una entidad entre ajuste y evaluación | La definición de grupo debe reflejar la generalización real | [Validación](../knowledge/validation/validation_strategy.md) |
| Orden temporal determina qué se conoce al predecir | Usar división secuencial/temporal; no barajar futuro y pasado | Evalúa con información disponible en el momento previsto | `knowledge/` no desarrolla gaps, horizontes múltiples ni backtesting completo | [Validación](../knowledge/validation/validation_strategy.md), [formulación](../knowledge/problem_definition/problem_formulation.md) |
| Hay simultáneamente grupos, temporalidad y desbalance | Buscar procedimiento específico o registrar vacío; no romper unidades para estratificar | Las guías advierten que estratificar no debe vulnerar grupos/tiempo | No existe una receta general conjunta en `knowledge/` | [Validación](../knowledge/validation/validation_strategy.md) |
| Métrica de clasificación oficial | Optimizar esa métrica; revisar desbalance, umbral y errores según la salida requerida | Accuracy, precision, recall, F-score y AUC describen aspectos distintos | No escoger una métrica por costumbre; comprobar la regla oficial y la clase positiva | [Métricas de clasificación](../knowledge/metrics/classification_metrics.md) |
| Target continuo | Escoger entre MAE, RMSE/MSE, R² u otra métrica permitida según el coste de error | MAE y RMSE reaccionan distinto a errores grandes; la escala importa | No hay una métrica de regresión universal; seguir score oficial | [Métricas de regresión](../knowledge/metrics/regression_metrics.md) |
| Score inesperadamente alto o feature posterior/derivada del target | Auditar construcción de variables y protocolo por leakage antes de comparar | La fuga puede producir evaluación optimista | Un pipeline no corrige features contaminadas antes de entrar al pipeline | [Leakage](../knowledge/preprocessing/data_leakage.md), [auditoría](../knowledge/eda/dataset_audit.md) |
| Transformación aprende parámetros de los datos | Ajustarla dentro del pipeline y de cada fold de entrenamiento | Impide que validación participe en el ajuste | Revisar también transformaciones globales previas | [Leakage](../knowledge/preprocessing/data_leakage.md), [validación](../knowledge/validation/validation_strategy.md) |
| Nueva feature, modelo, ensemble o cambio de preprocesamiento | Comparar contra baseline manteniendo protocolo; cambiar cantidad controlada de elementos | Aísla mejor la causa del cambio y conserva comparabilidad | La recomendación de controlar cambios por experimento es operativa derivada | [Ingeniería de variables](../knowledge/feature_engineering/feature_engineering.md), [ensembles](../knowledge/ensembles/boosting_models.md), [validación](../knowledge/validation/validation_strategy.md) |
| Se plantea tuning | Comenzar con shortlist, validación definida y presupuesto posible | El tuning consume evaluación y cómputo | No compensa un problema de formulación, datos o validación | [Tuning](../knowledge/tuning/hyperparameter_tuning.md) |

## 6. Baseline mínimo

El baseline debe ser deliberadamente sencillo, pero válido para el problema. No hay una combinación universal de split, modelo o métrica.

1. **Separar `X` e `y`.** Excluir el target, identificadores sin papel predictivo y columnas no disponibles al momento de predicción. Documentar la unidad de análisis y columnas retenidas. [Formulación](../knowledge/problem_definition/problem_formulation.md), [auditoría](../knowledge/eda/dataset_audit.md)
2. **Definir la validación.** Elegir holdout, folds, estratificación, grupos o tiempo según la generalización esperada. Fijar índices/regla y semilla si la partición usa aleatoriedad. La semilla no vuelve correcto un split conceptual incorrecto. [Validación](../knowledge/validation/validation_strategy.md)
3. **Preparar lo imprescindible.** Resolver tipos/faltantes/categorías en un flujo ajustado solo con entrenamiento. Evitar limpieza o selección guiada por validación. [Limpieza](../knowledge/preprocessing/data_cleaning.md), [leakage](../knowledge/preprocessing/data_leakage.md)
4. **Elegir referencia y modelo inicial.** Usa una heurística o predictor trivial si corresponde, y después un modelo tabular sencillo compatible con tarea/datos. En regresión, media/mediana son referencias posibles si encajan con distribución y métrica; para clasificación, el documento de métricas respalda referencias simples según el objetivo. Ninguna es universal. [Modelos tabulares](../knowledge/models/tabular_models.md), [métricas de regresión](../knowledge/metrics/regression_metrics.md), [métricas de clasificación](../knowledge/metrics/classification_metrics.md)
5. **Usar la métrica oficial.** Confirma dirección de mejora, clases positivas/salida requerida o unidades de regresión. Si el score oficial no basta para diagnosticar, conserva una métrica complementaria justificada sin sustituirlo. [Clasificación](../knowledge/metrics/classification_metrics.md), [regresión](../knowledge/metrics/regression_metrics.md)
6. **Guardar comparación.** Registra configuración, protocolo, versión/índices del split, semilla cuando corresponda, score, predicciones de validación y pipeline. Candidatos posteriores deben evaluarse en las mismas condiciones. [Validación](../knowledge/validation/validation_strategy.md)

**Punto de control:** el baseline debe correr desde datos disponibles para el reto hasta predicciones de validación sin ajustar transformaciones con esas predicciones. No se prescribe aquí código único porque el split y la métrica dependen de la tarea.

## 7. Registro de experimentos

Mantén una tabla o archivo de registro simple, legible y versionado junto al código. Como mínimo, anota:

| Campo | Qué registrar |
|---|---|
| Identificador y fecha | Nombre único del experimento y cuándo se ejecutó |
| Hipótesis/cambio | Qué se cambió y qué se esperaba observar |
| Datos y columnas | Archivos/versión, `X`, `y` y features añadidas o excluidas |
| Validación | Tipo de split, índices/regla, grupos/tiempo y semilla cuando aplique |
| Pipeline/modelo | Transformaciones, estimador, hiperparámetros y versiones necesarias |
| Métrica | Nombre oficial, dirección, valor y scores por fold si aplica |
| Resultado | Predicciones/artefacto, coste observado y decisión: conservar/revertir |

Compara cada prueba con el baseline y cambia una cantidad controlada de elementos por experimento. Esto es una **recomendación operativa derivada** para facilitar atribución; `knowledge/` respalda mantener folds, métrica y configuración comparables y registrar el flujo. Incluye pruebas negativas para evitar repetirlas. [Validación](../knowledge/validation/validation_strategy.md), [tuning](../knowledge/tuning/hyperparameter_tuning.md)

## 8. Gestión del tiempo y criterios para detenerse

No hay porcentajes ni duraciones obligatorias en `knowledge/`. Como **recomendación operativa derivada**, organiza el tiempo para que formulación, validación, baseline y entrega queden resueltos antes de dedicar el presupuesto restante a mejorar score. Confirma al inicio cualquier deadline, límite de cómputo o envíos que sí establezca la competencia.

Pasa de etapa cuando exista su resultado esperado y los controles críticos estén satisfechos: target y métrica interpretables; validación compatible con unidades/tiempo; baseline reproducible; pipeline sin ajuste sobre validación; comparación trazable. Si algo falla, corrige la causa antes de añadir modelos.

Detén una línea de experimentación cuando ocurra alguno de estos criterios observables:

- se alcanzó el presupuesto de tiempo/cómputo que se decidió para el reto;
- las pruebas recientes no muestran mejora útil en la validación declarada;
- la variación entre folds/splits hace que la supuesta mejora no sea concluyente;
- un entrenamiento ocupa el presupuesto disponible y no deja espacio viable para búsqueda;
- seguir probando compromete reproducibilidad, inspección de errores o verificaciones de entrega.

Los criterios de presupuesto y coste están respaldados por el documento de tuning; “mejora útil” y la prioridad explícita de reservar tiempo para entrega son decisiones del reto, no umbrales universales. Si la evaluación es inestable, revisa protocolo y datos antes de ampliar la búsqueda. [Tuning](../knowledge/tuning/hyperparameter_tuning.md), [validación](../knowledge/validation/validation_strategy.md)

## 9. Riesgos y controles

| Riesgo | Control operativo |
|---|---|
| Target equivocado, proxy o unidad mal definida | Confirmar significado, salida y uso con el enunciado; volver a formular si no encajan. [Formulación](../knowledge/problem_definition/problem_formulation.md) |
| Split que estima una pregunta distinta | Revisar independencia, entidades y tiempo antes de escoger partición. [Validación](../knowledge/validation/validation_strategy.md) |
| Desbalance escondido por una métrica agregada | Examinar distribución de clases y métrica pertinente; estratificar solo si no rompe grupos/tiempo. [Auditoría](../knowledge/eda/dataset_audit.md), [clasificación](../knowledge/metrics/classification_metrics.md) |
| Leakage por preparación o variables | Auditar origen y disponibilidad temporal; ajustar transformaciones dentro de los folds. [Leakage](../knowledge/preprocessing/data_leakage.md) |
| Comparación no reproducible | Mantener particiones/métrica y registrar pipeline, configuración y resultados. [Validación](../knowledge/validation/validation_strategy.md) |
| Optimización de una métrica sin utilidad clara | Relacionar el score con objetivo/costes conocidos; anotar cuando la utilidad no pueda estimarse con los datos disponibles. [Formulación](../knowledge/problem_definition/problem_formulation.md), [métricas de regresión](../knowledge/metrics/regression_metrics.md) |
| Entrega no compatible | Confirmar formato, columnas, IDs, ubicación y artefactos en instrucciones oficiales; generar y revisar la salida. **Control operativo derivado**, pues depende del reto. |

## 10. Checklist de ejecución

- [ ] Leí las reglas, métrica, restricciones y formato oficial de entrega.
- [ ] Definí objetivo, unidad de análisis, target, tipo de salida y predictores disponibles.
- [ ] Inspeccioné archivos, esquema, distribuciones, faltantes, duplicados, grupos y tiempo.
- [ ] Revisé posibles features posteriores, derivadas del target o no disponibles en inferencia.
- [ ] Elegí métrica y validación compatibles con la tarea, la unidad de generalización y el uso futuro.
- [ ] Guardé la regla/índices de partición y semilla cuando correspondía.
- [ ] Construí un baseline reproducible y conservé score y predicciones.
- [ ] Incluí las transformaciones aprendidas dentro del pipeline de validación.
- [ ] Comparé preprocesamiento, features, modelos, boosting/ensembles y tuning bajo condiciones consistentes.
- [ ] Registré hipótesis, configuración, validación, scores, coste observado y decisión por experimento.
- [ ] Seleccioné el modelo con la validación declarada y revisé estabilidad/errores relevantes.
- [ ] Guardé el flujo completo de predicción y comprobé que se reproduce desde las entradas.
- [ ] Verifiqué formato y archivos requeridos contra las instrucciones, sin asumir requisitos no escritos.
- [ ] Documenté limitaciones y vacíos que afectaron decisiones.

## 11. Fuentes de `knowledge/`

Solo se consultaron documentos Markdown de `knowledge/`; no se usaron fuentes externas. Todos los enlaces de esta lista y las referencias inline apuntan a rutas existentes desde `playbook/`.

- [Formulación del problema](../knowledge/problem_definition/problem_formulation.md): objetivo, unidad, target, salida y restricciones.
- [Auditoría de datasets](../knowledge/eda/dataset_audit.md): esquema, calidad, distribuciones, clases, grupos, tiempo y riesgos por investigar.
- [Limpieza de datos](../knowledge/preprocessing/data_cleaning.md): decisiones para faltantes, tipos, duplicados y transformaciones.
- [Prevención de data leakage](../knowledge/preprocessing/data_leakage.md): fuga por preparación, features y evaluación.
- [Ingeniería de variables](../knowledge/feature_engineering/feature_engineering.md): creación y representación de features.
- [Estrategias de validación](../knowledge/validation/validation_strategy.md): holdout, folds, estratificación, grupos, tiempo y reproducibilidad.
- [Métricas de clasificación](../knowledge/metrics/classification_metrics.md): métricas y consideraciones para clasificación/desbalance.
- [Métricas de regresión](../knowledge/metrics/regression_metrics.md): MAE, RMSE, R², baselines y sensibilidad a errores.
- [Modelos tabulares](../knowledge/models/tabular_models.md): familias y elección contextual de modelos.
- [Boosting y ensembles](../knowledge/ensembles/boosting_models.md): alternativas de ensamble y precauciones.
- [Ajuste de hiperparámetros](../knowledge/tuning/hyperparameter_tuning.md): cuándo ajustar, métodos, coste y criterios de parada.
- [Flujo de trabajo existente](../knowledge/workflows/competition_workflow.md): síntesis previa de formulación, evaluación, experimentación, reproducibilidad y entrega que se adaptó a la estructura solicitada aquí.

## 12. Vacíos o límites del playbook

- `knowledge/` no define una metodología completa para series temporales/forecasting: faltan, entre otros aspectos, reglas detalladas para gaps, embargo, horizontes múltiples y backtesting. La validación temporal solo justifica preservar orden y evitar fuga de futuro.
- No hay una receta general para combinar simultáneamente tiempo, grupos y estratificación. En ese caso se necesita procedimiento específico; no se debe fingir que un split aleatorio resuelve el conflicto.
- Imágenes, texto y clustering requieren rutas específicas si existen; no se desarrolla aquí su preprocesamiento ni su evaluación particular. Si `knowledge/` no contiene esa guía, se registra como ausencia de cobertura.
- Los documentos no fijan umbrales universales de mejora, duración, memoria o número de ensayos. Presupuesto y criterio de mejora útil deben adaptarse al reto.
- La utilidad de una métrica, la calidad semántica del target, la definición del grupo y el tratamiento de valores atípicos pueden requerir contexto que no esté en los archivos. No deben darse por resueltos solo con el score.
- El formato y los archivos de entrega dependen del enunciado de cada competencia; este documento no prescribe nombres ni estructura.
- No faltó ninguno de los documentos temáticos consultados para el flujo general. Los límites anteriores son de cobertura, no archivos ausentes.
