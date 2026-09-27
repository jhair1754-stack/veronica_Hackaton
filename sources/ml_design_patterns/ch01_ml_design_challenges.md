---
title: "The Need for Machine Learning Design Patterns"
book: "Machine Learning Design Patterns"
chapter: "1"
topics:
  - "Machine Learning design patterns"
  - "Machine Learning terminology"
  - "Data quality"
  - "Reproducibility"
  - "Data drift"
  - "Scale"
  - "Multiple objectives"
patterns: []
target_knowledge_files:
  - "knowledge/problem_definition/ml_design_challenges.md"
  - "knowledge/eda/data_quality.md"
  - "knowledge/reproducibility/reproducibility_requirements.md"
  - "knowledge/mlops/data_drift.md"
  - "knowledge/mlops/scaling_constraints.md"
  - "knowledge/metrics/multiple_objectives.md"
---

# 1. The Need for Machine Learning Design Patterns

## 1. Propósito del capítulo

El capítulo establece el marco conceptual del catálogo. Explica qué es un design pattern, fija el vocabulario que el libro usa para datos, modelos y ciclo de vida de ML, y presenta cinco familias de desafíos recurrentes: **Data Quality**, **Reproducibility**, **Data Drift**, **Scale** y **Multiple Objectives**.

Estos problemas aparecen desde la obtención y preparación de datos hasta training, evaluation, deployment y serving. Justifican el uso de patrones porque una solución de ML debe responder no solo a la calidad predictiva, sino también a cambios en los datos, repetibilidad experimental, restricciones de infraestructura y objetivos organizacionales potencialmente incompatibles.

**Resultado de extracción:** el capítulo 1 no define ningún Machine Learning Design Pattern numerado ni desarrolla la plantilla problema-solución-discusión de un patrón. Los patterns mencionados pertenecen expresamente a capítulos posteriores y no se documentan aquí como unidades del capítulo.

# 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Ninguno definido en el capítulo | El capítulo introduce el catálogo y sus desafíos, pero no presenta un pattern completo. | Marco general | No aplica |

## 2.1 Referencias a patterns externos al capítulo

Estas referencias se registran únicamente para evitar falsos positivos durante una extracción posterior. El capítulo 1 no contiene suficiente material para documentarlas de manera autónoma.

| Pattern citado | Ubicación del desarrollo según el capítulo | Motivo de la mención en el capítulo 1 |
| -------------- | ----------------------------------------- | -------------------------------------- |
| Transform | Chapter 6 | Ejemplo de patrón que separa inputs, features y transforms, y persiste las transformaciones para facilitar producción. |
| Keyed Predictions | Chapter 5 | Ejemplo de patrón para distribuir batch predictions a gran escala. |
| Fairness Lens | Chapter 7 | Remisión desde la discusión sobre sesgo e inconsistencias entre etiquetadores. |
| Bridged Schema | Chapter 6 | Remisión desde Data Drift para datasets cuya disponibilidad de features mejora con el tiempo. |

# 3. Patrones

No hay patterns individuales que puedan documentarse con la plantilla obligatoria sin atribuir al capítulo contenido que no presenta. En particular:

- **Transform** aparece solo como una descripción introductoria de una oración.
- **Keyed Predictions** aparece solo como otro ejemplo breve.
- **Fairness Lens** y **Bridged Schema** son referencias cruzadas a secciones posteriores.
- Las cinco subsecciones de desafíos no están nombradas ni numeradas como design patterns.

Por esta razón no se crean secciones `Pattern [N]` ni se completan artificialmente campos como solución, trade-offs, alternativas o implementación.

## 3.1 Conocimiento fundacional: Data Quality

### Contenido del libro

Un modelo refleja la calidad de sus datos. El capítulo organiza la calidad en cuatro dimensiones:

| Dimensión | Problema observable | Consecuencia descrita |
| --------- | ------------------- | --------------------- |
| Accuracy | Features con errores de captura, typos, duplicados, mediciones inconsistentes, valores ausentes o labels incorrectos. | El modelo aprende una representación distorsionada; los duplicados pueden dar peso excesivo a ciertos ejemplos y los labels erróneos alteran el aprendizaje. |
| Completeness | Faltan clases, condiciones o variaciones que aparecerán en producción. | El modelo solo puede elegir entre las categorías que conoce y puede responder con alta confianza ante entradas fuera de su mundo de entrenamiento. |
| Consistency | Personas, sensores o fuentes usan criterios, formatos o unidades diferentes. | Features y labels dejan de ser comparables; el resultado se vuelve menos fiable. |
| Timeliness | Existe latencia entre el evento y su incorporación a la base de datos. | Las features pueden representar información que no habría estado disponible en el momento real de la decisión. |

El texto recomienda analizar el origen y la recolección, buscar errores y duplicados, asegurar variedad dentro de cada label, estandarizar la recolección y el etiquetado, y conservar tanto el timestamp del evento como el de incorporación al dataset cuando exista latencia. Para etiquetas subjetivas, propone múltiples etiquetadores y el label más aplicado como una forma de mejorar consistencia.

### Inferencia práctica

- Comparar training, validation y test exige que las particiones tengan propiedades estadísticas similares.
- Una categoría de rechazo como “not a cat” solo puede aprenderse si está representada mediante datos y label adecuados.
- Registrar tiempo de evento y tiempo de disponibilidad permite construir features compatibles con la información disponible en inferencia y reduce el riesgo de leakage temporal.

### Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Antes de modelar, conviene revisar duplicados, valores faltantes, rangos y unidades, cobertura de labels y diferencias de distribución entre particiones. En una prueba con tiempo limitado, estos controles pueden evitar que una métrica aparente provenga de datos inconsistentes o de una división no representativa.

## 3.2 Conocimiento fundacional: Reproducibility

### Contenido del libro

Los modelos incorporan aleatoriedad, por ejemplo en la inicialización de pesos y el barajado de datos. Dos ejecuciones del mismo código y dataset pueden producir resultados distintos, lo que dificulta comparar experimentos.

Para repetir resultados deben fijarse, como mínimo:

1. Los datos utilizados.
2. El mecanismo de división en training y validation.
3. La preparación de datos.
4. Los hiperparámetros del modelo.
5. Variables como batch size y learning-rate schedule.
6. Las seeds relevantes.
7. Las versiones de frameworks y librerías.
8. El entorno de ejecución, considerando que el entrenamiento distribuido introduce retos adicionales.

El capítulo muestra `tf.random.set_seed(value)` y el parámetro `random_state` de `sklearn.utils.shuffle` como ejemplos mínimos. También recomienda contenedores y versiones estandarizadas de librerías.

### Inferencia práctica

Una seed aislada no garantiza reproducibilidad si cambian los datos, el split, las dependencias o la estrategia distribuida. La unidad reproducible es la configuración completa del experimento.

### Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Fijar seeds y conservar exactamente el split permite atribuir los cambios de métrica a una modificación del experimento y no a una muestra diferente. Esto es especialmente valioso al comparar rápidamente features o arquitecturas.

## 3.3 Conocimiento fundacional: Data Drift

### Contenido del libro

Un modelo suele representar una relación estática entre inputs y outputs, mientras que los datos cambian con el tiempo. Por ello, un modelo entrenado con datos históricos puede dejar de representar el entorno actual.

El capítulo propone tres respuestas generales:

1. Actualizar continuamente el dataset de training.
2. Reentrenar el modelo.
3. Modificar el peso asignado a grupos concretos de datos de entrada.

El ejemplo de titulares muestra la aparición de vocabulario nuevo. El ejemplo de tormentas de NOAA muestra que una tendencia observada puede combinar cambios reales con cambios del sistema de medición: mejor tecnología, más eventos registrados y nuevas features disponibles. El análisis exploratorio ayuda a detectar estos cambios y a seleccionar la ventana de training.

### Inferencia práctica

No todo cambio en una serie refleja el fenómeno objetivo; también puede reflejar cómo se observa o registra. Antes de escoger una ventana temporal, hay que distinguir cambio del mundo, cambio de instrumentación y cambio de esquema.

### Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Cuando existe tiempo, ordenar métricas y disponibilidad de features por fecha puede revelar que una partición aleatoria oculta drift. El capítulo no prescribe un esquema de validación temporal concreto, por lo que no se añade uno aquí.

## 3.4 Conocimiento fundacional: Scale

### Contenido del libro

La escala afecta varias etapas:

| Etapa | Restricción destacada |
| ----- | --------------------- |
| Ingesta y preprocessing | El tamaño del dataset determina las herramientas y pipelines necesarios. |
| Training | Volumen y tipo de datos condicionan duración, coste e infraestructura; los modelos de imágenes suelen requerir más recursos que los de datos tabulares. |
| Serving | La infraestructura de un prototipo usado por pocos científicos de datos difiere de la necesaria para millones de requests por hora. |

El capítulo añade que algunos patrones de resiliencia y reproducibilidad deben elegirse según el caso de uso y la capacidad de la organización para absorber complejidad.

### Inferencia práctica

Una solución debe evaluarse con respecto a la etapa que realmente limita el sistema. Un problema de serving no se resuelve necesariamente aumentando la infraestructura de training.

### Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Bajo límites de tiempo o cómputo, el tamaño y modalidad del dataset deben influir en la elección de herramientas y en el presupuesto experimental. El capítulo no ofrece umbrales ni configuraciones concretas.

## 3.5 Conocimiento fundacional: Multiple Objectives

### Contenido del libro

Los equipos que construyen y consumen un modelo pueden definir éxito de forma distinta. En el ejemplo de inspección de productos, el científico de datos busca minimizar cross-entropy loss, producto quiere reducir defectos enviados a clientes y dirección busca aumentar ingresos.

La respuesta propuesta es traducir las necesidades organizacionales al comportamiento y a las métricas del modelo antes de construir la solución. Si los false negatives cuestan cinco veces más que los false positives, esa asimetría debe influir en la prioridad entre recall y precision. El objetivo es encontrar compromisos entre los distintos criterios.

### Inferencia práctica

La función de pérdida, las métricas de evaluación y el objetivo de negocio no son intercambiables. Deben vincularse explícitamente para que una mejora técnica represente una mejora relevante.

### Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Conviene leer la métrica y el coste de errores antes de optimizar. Una solución con buena loss pero orientada al tipo de error equivocado puede no responder al objetivo evaluado.

# 4. Reglas de decisión del capítulo

No existen reglas de selección entre patterns del capítulo porque no contiene patterns propios. Las siguientes reglas se derivan de sus desafíos fundacionales y no deben interpretarse como nombres oficiales de patrones.

| Situación | Acción respaldada por el capítulo | Motivo | Precaución |
| --------- | --------------------------------- | ------ | ---------- |
| Datos con errores, duplicados o labels dudosos | Analizar y corregir la calidad antes de confiar en el modelo. | El modelo reproduce los defectos del dataset. | Revisar features y ground truth, no solo una de las dos. |
| Clases o condiciones de producción ausentes | Ampliar la cobertura del dataset y sus labels. | El modelo no puede aprender una salida que no existe en training. | Una predicción confiada no demuestra que la entrada esté cubierta. |
| Resultados distintos entre ejecuciones | Fijar datos, split, preparación, hiperparámetros, seeds y dependencias. | Todos esos artefactos afectan la repetibilidad. | Una seed por sí sola es insuficiente. |
| Datos o mediciones cambian con el tiempo | Actualizar datos, reentrenar y revisar la ventana de training. | La relación histórica puede dejar de representar producción. | Separar cambios reales de cambios en observación o esquema. |
| Volumen o tráfico crece | Elegir herramientas e infraestructura según la etapa afectada. | Ingesta, training y serving tienen necesidades distintas. | No añadir complejidad que la organización no pueda operar. |
| Equipos optimizan resultados distintos | Traducir costes y objetivos a criterios del modelo. | Permite equilibrar métricas técnicas y resultados de negocio. | Explicitar el coste relativo de diferentes errores. |

# 5. Comparación entre patrones del capítulo

No aplica: no hay patterns definidos en el capítulo. Comparar **Data Quality**, **Reproducibility**, **Data Drift**, **Scale** y **Multiple Objectives** como si fueran patterns alteraría la taxonomía del libro.

# 6. Riesgos globales del capítulo

| Riesgo | Manifestación en el capítulo |
| ------ | ---------------------------- |
| Datos o labels inexactos | El modelo aprende errores, inconsistencias o ponderaciones artificiales causadas por duplicados. |
| Cobertura incompleta | El modelo fuerza entradas desconocidas hacia las clases disponibles. |
| Distribuciones no comparables | Training, validation y test pueden medir poblaciones distintas. |
| Leakage temporal | **Inferencia práctica:** una feature puede contener información que todavía no estaba disponible si no se distingue el tiempo del evento del tiempo de ingestión. |
| Falta de reproducibilidad | Aleatoriedad, splits, preparación, hiperparámetros, dependencias o ejecución distribuida impiden comparar experimentos. |
| Drift | Datos, vocabulario, instrumentación o esquema cambian después del training. |
| Complejidad operacional | La infraestructura requerida cambia con dataset, modalidad, training y volumen de serving. |
| Optimización mal alineada | Loss, precision, recall e impacto de negocio pueden empujar decisiones diferentes. |
| Sesgo de etiquetado | Distintos etiquetadores pueden aplicar criterios incompatibles. |

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

- **Alta prioridad:** confirmar target, tipo de problema y métrica; revisar duplicados, missing values, labels, unidades, rangos y cobertura.
- **Alta prioridad:** identificar si existe una dimensión temporal o si la disponibilidad de una feature cambia con el tiempo.
- **Prioridad media:** estimar si volumen, modalidad o recursos condicionarán preprocessing y training.

### Durante el baseline

- Conservar un split estable, seeds y configuración para que el baseline sea comparable.
- Comprobar que training, validation y test tengan propiedades compatibles con el objetivo de evaluación.
- Relacionar la métrica con el coste de falsos positivos y falsos negativos cuando el problema lo requiera.

### Durante la mejora del modelo

- Cambiar un factor a la vez sobre una configuración reproducible.
- Investigar si una mejora proviene de datos más representativos o de una partición accidentalmente favorable.
- Ajustar el presupuesto experimental a las restricciones reales de cómputo.

### Durante validación

- Mantener el test fuera de training y de la selección de hiperparámetros.
- Revisar diferencias estadísticas entre particiones.
- Si existe tiempo, examinar si los resultados se mantienen frente a periodos o esquemas distintos; esta es una inferencia práctica y no un protocolo prescrito por el capítulo.

### Bajo presión de tiempo

| Prioridad | Acciones derivadas del capítulo |
| --------- | ------------------------------- |
| Alta | Calidad básica de datos, definición del objetivo, split estable y seeds. |
| Media | Análisis de drift, consistencia de etiquetado y coste diferencial de errores. |
| Baja | Infraestructura sofisticada que no sea necesaria para el volumen disponible. |

# 8. Checklist global del capítulo

- [ ] Confirmar si el problema es supervised o unsupervised y, si aplica, clasificación o regresión.
- [ ] Separar inputs, features, labels, predictions y training examples con la terminología del capítulo.
- [ ] Examinar accuracy, completeness, consistency y timeliness de los datos.
- [ ] Confirmar que training, validation y test sean independientes y estadísticamente comparables.
- [ ] Registrar datos, split, preparación, hiperparámetros, seeds y versiones.
- [ ] Buscar cambios temporales en datos, instrumentación y disponibilidad de features.
- [ ] Identificar la restricción de escala en ingesta, training o serving.
- [ ] Traducir objetivos de stakeholders a métricas y costes de error.
- [ ] No promover las referencias a otros capítulos a patterns del capítulo 1.

# 9. Mapeo hacia `knowledge/`

Este mapeo corresponde a conocimiento fundacional, no a patterns individuales.

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Marco de ML design patterns | `knowledge/problem_definition/ml_design_challenges.md` | Media |
| Data Quality | `knowledge/eda/data_quality.md` | Alta |
| Reproducibility | `knowledge/reproducibility/reproducibility_requirements.md` | Alta |
| Data Drift | `knowledge/mlops/data_drift.md` | Alta |
| Scale | `knowledge/mlops/scaling_constraints.md` | Media |
| Multiple Objectives | `knowledge/metrics/multiple_objectives.md` | Alta |

# 10. Índice de patrones reutilizables

No hay patterns reutilizables definidos en el capítulo. Para facilitar la localización del conocimiento fundacional se ofrece este índice diferenciado:

| Conocimiento | Activador principal | Acción principal | Knowledge destino |
| ------------ | ------------------- | ---------------- | ----------------- |
| Data Quality | Errores, duplicados, labels dudosos, cobertura o formatos inconsistentes. | Auditar accuracy, completeness, consistency y timeliness. | `knowledge/eda/data_quality.md` |
| Reproducibility | Las métricas cambian entre ejecuciones comparables. | Fijar todos los artefactos y dependencias del experimento. | `knowledge/reproducibility/reproducibility_requirements.md` |
| Data Drift | El entorno, vocabulario, medición o esquema cambia con el tiempo. | Actualizar datos, reentrenar y revisar la ventana temporal. | `knowledge/mlops/data_drift.md` |
| Scale | Datos, entrenamiento o tráfico superan la capacidad actual. | Dimensionar herramientas por etapa del ciclo de vida. | `knowledge/mlops/scaling_constraints.md` |
| Multiple Objectives | Equipos o métricas definen éxito de forma diferente. | Traducir costes y objetivos a criterios del modelo. | `knowledge/metrics/multiple_objectives.md` |

# 11. Trazabilidad

La numeración corresponde a las páginas impresas del capítulo; la página 18 del PDF está en blanco.

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Definición y propósito de design patterns | “What Are Design Patterns?”, pp. 1–3 |
| Forma prevista de cada pattern | “How to Use This Book”, p. 3 |
| Terminología de modelos y frameworks | “Models and Frameworks”, pp. 4–6; Figure 1-1, p. 5 |
| Data and Feature Engineering | “Data and Feature Engineering”, pp. 6–7 |
| Ciclo de vida: training, evaluation, serving y prediction | “The Machine Learning Process”, pp. 7–8 |
| Data and Model Tooling | “Data and Model Tooling”, pp. 8–9 |
| Roles | “Roles”, pp. 9–11; Figure 1-2, p. 11 |
| Data Quality | “Data Quality”, pp. 11–13 |
| Reproducibility | “Reproducibility”, pp. 13–14 |
| Data Drift | “Data Drift”, pp. 14–15; Figure 1-3, p. 15 |
| Scale | “Scale”, p. 16 |
| Multiple Objectives | “Multiple Objectives”, pp. 16–17 |
| Síntesis del capítulo | “Summary”, p. 17 |
| Transform y Keyed Predictions, solo referencias | p. 2 |
| Fairness Lens, solo referencia | p. 13 |
| Bridged Schema, solo referencia | p. 15 |

