---
title: "Responsible AI"
book: "Machine Learning Design Patterns"
chapter: "7"
topics:
  - "Responsible AI"
  - "Model evaluation"
  - "Explainability"
  - "Fairness"
patterns:
  - "Heuristic Benchmark"
  - "Explainable Predictions"
  - "Fairness Lens"
target_knowledge_files:
  - "knowledge/models/heuristic_benchmark.md"
  - "knowledge/metrics/model_utility.md"
  - "knowledge/interpretability/explainable_predictions.md"
  - "knowledge/responsible_ai/fairness_lens.md"
---

# 7. Responsible AI

## 1. Propósito del capítulo

El capítulo aborda problemas que aparecen cuando un modelo deja de ser solo un artefacto técnico y afecta a personas, procesos de negocio, auditorías y decisiones reguladas. Sitúa estos problemas a lo largo de todo el ciclo de vida: definición de un punto de comparación, evaluación previa al despliegue, explicación de predicciones, análisis de datos antes del entrenamiento, evaluación por grupos después del entrenamiento y supervisión continua.

Los patrones se justifican porque una métrica aislada no comunica si un modelo es útil, una predicción correcta no explica por qué ocurrió y una buena métrica global puede ocultar daños o disparidades entre grupos. El capítulo propone tres unidades independientes para cubrir esas brechas: **Heuristic Benchmark**, **Explainable Predictions** y **Fairness Lens**.

# 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Heuristic Benchmark | Las métricas del modelo carecen de un contexto comprensible para decidir si el resultado es bueno o merece desplegarse. | Metrics / Model evaluation | Alta |
| Explainable Predictions | Las métricas no revelan qué señales producen una predicción ni permiten verificar el comportamiento del modelo. | Interpretability | Alta |
| Fairness Lens | Los datos, labels, objetivos o umbrales pueden causar resultados injustos entre grupos o escenarios. | Responsible AI | Alta |

# 3. Patrones

## Pattern 28: Heuristic Benchmark

### 3.1 Problema

Una métrica como MAE, MAP o accuracy describe el rendimiento, pero no ofrece por sí sola un punto de referencia comprensible para usuarios o responsables de negocio. Explicar la fórmula de la métrica tampoco responde si el valor es bueno en el contexto concreto.

La solución ingenua de comparar con un modelo simple puede fallar como comunicación: incluso una regresión lineal puede dejar de ser intuitiva al añadir variables categóricas, múltiples entradas o features construidas. Si ya existe un proceso operativo, inventar un benchmark nuevo también elimina el punto de comparación más relevante.

### 3.2 Contexto de aplicación

Se aplica al presentar un modelo nuevo o una nueva versión a responsables de negocio, usuarios o equipos que deciden el despliegue. Cubre regresión, clasificación binaria y multiclase, series temporales, detección de anomalías, recomendación, mantenimiento preventivo y tareas realizadas por expertos humanos.

Es especialmente útil en tareas *green-field* sin método de producción anterior. Si existe una práctica operativa, aunque no use ML, el capítulo indica compararla directamente con el modelo.

### 3.3 Señales que indican que el patrón puede ser necesario

- Los stakeholders preguntan si una cifra como MAE 1,200 o MAP 95% es buena o mala.
- No existe una versión productiva ni un proceso anterior con métricas comparables.
- La explicación de la métrica no permite interpretar el beneficio del modelo.
- Un modelo complejo no supera de forma clara una regla elemental.
- El rendimiento agregado oculta un desempeño muy diferente entre segmentos.
- **Inferencia práctica:** el coste de operar el modelo podría superar la utilidad de una mejora pequeña.

### 3.4 Solución propuesta

1. Si existe una práctica operativa, usar su rendimiento como estándar.
2. Si no existe, definir una heurística trivial de calcular y fácil de explicar.
3. Elegir una regla con sentido para la tarea: constante, estadístico agregado, regla de negocio, persistencia temporal, desempeño humano u otra referencia simple.
4. Evaluar el modelo y la heurística con el mismo criterio y sobre datos comparables.
5. Durante el desarrollo, comparar también por estratos relevantes para localizar fallos.
6. Traducir, cuando sea posible, la mejora de la métrica a utilidad: dinero, menos desperdicio, detección más temprana, mejores resultados o menos abandonos.

### 3.5 Cómo funciona

El patrón agrega un punto de referencia interpretable al flujo de evaluación:

`datos de evaluación → heurística comprensible → métrica base → comparación del modelo → utilidad de la mejora`.

Ejemplos propuestos por el capítulo:

- Regresión sin relaciones de features bien comprendidas: media o mediana del label; preferir mediana con muchos outliers.
- Clasificación binaria: fracción global de positivos como probabilidad constante.
- Multiclase: distribución de labels.
- Una variable numérica dominante y aceptada por el negocio: regla lineal basada en ella.
- Una o dos variables clave: tabla de consulta con promedio o distribución del label por celda.
- Forecasting: persistencia, tendencia lineal o valor de la misma temporada anterior.
- Trabajo actualmente humano: rendimiento de un panel de expertos.
- Mantenimiento preventivo: calendario fijo derivado, por ejemplo, del tiempo mediano hasta fallo.
- Anomalías: percentil alto estimado con training.
- Recomendación: elemento más popular de la categoría asociada al último interés.

### 3.6 Por qué funciona

La heurística convierte una cifra abstracta en una comparación que los stakeholders ya pueden razonar. También establece el mínimo que justifica la complejidad de ML. Cuando la comparación se desglosa por segmentos, sirve como prueba de desarrollo: un benchmark por estación y hora punta, por ejemplo, puede revelar que el modelo favorece estaciones frecuentes e ignora las raras.

Traducir la mejora a utilidad conecta el rendimiento estadístico con la decisión de despliegue, que también debe considerar fiabilidad, presupuesto de errores y coste operacional.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Training para estimar la regla; evaluación comparable; conocimiento de una regla aceptada por el negocio o rendimiento del proceso vigente. |
| Transformación | Construcción de una predicción trivial y cálculo de la misma métrica usada por el modelo. |
| Salida | Métrica base interpretable, diferencia respecto al modelo y, si es posible, valor de utilidad. |

### 3.8 Procedimiento operativo

1. Definir la tarea y la métrica que representa el objetivo.
2. Confirmar si ya existe un proceso operativo medido.
3. Si no existe, seleccionar la heurística más simple que conserve sentido contextual.
4. Calcular sus parámetros solo con training cuando provengan de datos.
5. Ejecutar la heurística sobre el mismo conjunto de evaluación que el modelo.
6. Comparar resultados globales y por estratos relevantes.
7. Investigar cualquier segmento donde el modelo no supere el benchmark.
8. Estimar el impacto práctico de la diferencia.
9. Decidir si el beneficio compensa desplegar y mantener el modelo.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Media frente a mediana | Resume el label con distinta sensibilidad a extremos. | La media puede quedar dominada por outliers. |
| Feature(s) para una regla o tabla | Aporta contexto del negocio. | Usar importancia aprendida por el mismo modelo reduce la intuición y circulariza la explicación. |
| Estratos de evaluación | Localizan segmentos débiles. | Estratos con pocos ejemplos producen estimaciones inestables. |
| Persistencia, tendencia o estacionalidad | Cambia la referencia en forecasting. | Combinar referencias con pesos poco intuitivos destruye la claridad del benchmark. |
| Panel humano | Contextualiza tareas expertas. | Requiere recogida adicional y los expertos también cometen errores. |
| Umbral de utilidad | Vincula métrica y despliegue. | Una mejora estadística puede no cubrir el coste de producción. |

### 3.10 Ventajas

- Comunica el rendimiento en términos intuitivos.
- Establece un mínimo que el modelo debe superar.
- Ayuda a diagnosticar enfoques deficientes durante el desarrollo.
- Permite analizar rendimiento por estratos.
- Facilita decidir si una mejora justifica el despliegue.
- Puede aprovechar reglas, estadísticas o procesos existentes.

### 3.11 Limitaciones

- Puede requerir una recogida especial de datos, sobre todo para medir expertos humanos.
- Una comparación de métricas puede seguir sin explicar el valor práctico.
- Una heurística demasiado sofisticada deja de ser comprensible.
- El benchmark no explica por sí mismo cómo toma decisiones el modelo.

### 3.12 Riesgos y trade-offs

- **Leakage:** estadísticas, percentiles o tablas calculadas con test contaminarían la evaluación; deben estimarse con training.
- **Distribución:** un único valor agregado puede ocultar grupos raros o estaciones poco frecuentes.
- **Complejidad:** una referencia más potente puede ser menos explicable y perder su propósito.
- **Coste:** usar expertos mejora el contexto, pero eleva tiempo y coste.
- **Operación:** una ganancia pequeña puede no compensar fiabilidad, mantenimiento y presupuesto de errores.
- **Dependencia de heurísticas:** superar una regla débil no demuestra que el modelo sea útil.

### 3.13 Cuándo utilizarlo

- Al construir el primer modelo para una tarea sin proceso previo.
- Al presentar rendimiento a stakeholders no técnicos.
- Como control rápido durante el desarrollo.
- Cuando se necesita justificar que ML supera una solución simple.
- Cuando la mejora puede traducirse a utilidad de negocio o social.

### 3.14 Cuándo no utilizarlo

- No crear una heurística artificial si ya existe una práctica operativa: comparar contra esa práctica.
- No usar un modelo ML como benchmark si deja de ser intuitivo.
- No confiar solo en el benchmark cuando se requiere explicar predicciones o evaluar fairness.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Proceso operativo actual | Es el estándar real que debe superarse. | Puede no usar ML ni tener evaluación histórica suficiente. | Siempre que exista y sus métricas puedan medirse. |
| Constante o estadístico global | Máxima simplicidad. | Ignora estructura de features. | Primera referencia en problemas poco comprendidos. |
| Tabla por una o dos features | Incorpora reglas de negocio conocidas. | Crece con cardinalidad y puede tener celdas escasas. | Cuando pocas variables aceptadas dominan la tarea. |
| Expertos humanos | Contextualiza el nivel de desempeño humano. | Costoso; existe desacuerdo. | Tareas actualmente resueltas por especialistas. |
| Utility value | Responde si la mejora merece desplegarse. | Requiere cuantificar efectos. | Decisiones de producción, además de la comparación técnica. |

### 3.16 Implementación práctica

Idea mínima para regresión con outliers:

```python
benchmark_value = y_train.median()
y_benchmark = [benchmark_value] * len(y_valid)
benchmark_mae = mean_absolute_error(y_valid, y_benchmark)
model_mae = mean_absolute_error(y_valid, model.predict(X_valid))
```

La lógica importante es que el valor se estima con training y se evalúa en validación/test exactamente como el modelo. Para tablas de consulta, calcular el promedio del label por combinación de features clave y definir una regla explícita para combinaciones no vistas.

### 3.17 Ejemplo mínimo conceptual

Una agencia predice duración de alquileres. La cifra de MAE del modelo no tiene contexto. Se crea una tabla por estación y condición de hora punta con la duración promedio del training. Si el modelo mejora esa tabla en el conjunto de evaluación y especialmente en estaciones poco frecuentes, existe una evidencia más comprensible de valor. La mejora todavía debe traducirse, si es posible, a disponibilidad o beneficio.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En un coding test o datathon, el patrón tiene prioridad alta porque una constante, frecuencia de clase, persistencia temporal o tabla pequeña se construye rápido. Permite detectar pipelines rotos, validaciones dudosas y modelos que no aprenden más que una estadística elemental. También ayuda a decidir si merece invertir tiempo en más complejidad.

No garantiza una mejora de leaderboard; su valor es fijar un suelo verificable. Bajo presión, conviene mantenerlo separado del modelo, calcularlo con training y evaluarlo en el mismo split. En series temporales, una referencia estacional o de persistencia es más informativa que una media global cuando el capítulo indica que la temporalidad forma parte de la tarea.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar si existe un proceso operativo comparable.
- [ ] Elegir una heurística trivial y explicable.
- [ ] Estimar estadísticas solo con training.
- [ ] Usar la misma métrica y datos de evaluación que el modelo.
- [ ] Comparar globalmente y por estratos relevantes.
- [ ] Investigar segmentos donde el modelo pierde.
- [ ] Traducir la mejora a utilidad cuando sea posible.
- [ ] Rechazar complejidad que no supere claramente el benchmark.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Explainable Predictions | Complementario | El benchmark compara el rendimiento global; la explicación muestra qué señales producen predicciones. Una heurística puede inspirar el baseline de atribución. |
| Fairness Lens | Puede combinarse | La evaluación estratificada puede revelar grupos donde el modelo no supera una referencia o rinde de forma desigual. |

## Pattern 29: Explainable Predictions

### 3.1 Problema

Accuracy, precision, recall o MSE indican cuánto acierta un modelo frente a ground truth, pero no por qué llega a una salida. Esto debilita la confianza del usuario, dificulta auditorías y limita la capacidad del equipo para detectar señales espurias, sesgo o features innecesarias.

En imágenes médicas, por ejemplo, una confianza alta no confirma que el modelo miró la región enferma y no una anotación. En crédito, un score no revela si la decisión depende de pagos, historial o datos demográficos. En desarrollo, tampoco se sabe si el modelo usa veinte features o solo dos.

### 3.2 Contexto de aplicación

Se aplica a modelos tabulares, de imagen y texto; a regresión y clasificación; durante el desarrollo y en serving. Los modelos simples pueden ser interpretables por diseño, mientras que redes profundas suelen requerir métodos *post hoc*. El capítulo se concentra en atribuciones de features, tanto locales como globales.

Stakeholders: usuarios afectados, profesionales que usan predicciones, auditores y compliance, data scientists y ML engineers.

### 3.3 Señales que indican que el patrón puede ser necesario

- Los usuarios no pueden justificar o confiar en una predicción.
- Una predicción de alta confianza podría depender de una señal irrelevante.
- Auditoría o regulación exige transparencia.
- El equipo no sabe qué features dominan el comportamiento.
- Hay sospecha de bias, shortcuts o anotaciones espurias.
- El sistema debe entregar explicaciones junto con predicciones desplegadas.
- **Inferencia práctica:** retirar features sin saber su contribución podría degradar o simplificar innecesariamente el modelo.

### 3.4 Solución propuesta

1. Determinar si el modelo ya es interpretable por diseño o necesita explicación *post hoc*.
2. Definir el nivel: explicación de una instancia o comportamiento global.
3. Elegir un baseline informativo o no informativo apropiado.
4. Seleccionar un método de atribución compatible con modelo y datos.
5. Calcular contribuciones de features respecto al baseline.
6. Visualizar y comprobar coherencia, incluido que las atribuciones aproximen la diferencia entre predicción y baseline.
7. Usar la explicación para depurar datos/modelo y comunicar el comportamiento.
8. Tratarla como una señal adicional, no como prueba absoluta.

### 3.5 Cómo funciona

Las atribuciones asignan a cada feature un valor que estima cuánto contribuyó a mover la salida desde una predicción base hasta la predicción observada.

- **Instance-level:** explica una predicción individual, como por qué se denegó una solicitud o qué píxeles activaron una clase.
- **Global:** agrega atribuciones de múltiples ejemplos, normalmente del test, para resumir el comportamiento del modelo.
- **Sampled Shapley / SHAP:** estima contribuciones marginales al añadir o retirar features en múltiples combinaciones.
- **Integrated Gradients (IG):** integra gradientes a lo largo del camino desde un baseline hasta el input.
- **XRAI:** parte de IG y suaviza para devolver regiones en imágenes.

Un baseline no informativo representa un caso promedio o ausencia de señal: medianas numéricas, una imagen negra/blanca o embeddings nulos. Uno informativo compara con una alternativa específica, como la mediana de transacciones no fraudulentas.

### 3.6 Por qué funciona

En vez de inspeccionar pesos internos difíciles de interpretar, los métodos *post hoc* aproximan la relación entre inputs y output. El baseline proporciona el “respecto a qué” necesario para dar sentido a una contribución. Agregar explicaciones locales permite reconocer tendencias globales sin asumir que una sola instancia representa todo el modelo.

El patrón mejora la depuración porque puede revelar que el modelo usa paddles en vez de la forma del kayak, anotaciones en una radiografía o términos de identidad en vez de palabras de sentimiento.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Modelo, ejemplos, features, predicción a explicar y baseline elegido. |
| Transformación | Método de atribución, análisis contrafactual o comparación con ejemplos similares. |
| Salida | Contribuciones por feature/píxel/token a nivel local o resumen global. |

### 3.8 Procedimiento operativo

1. Formular la pregunta concreta: “¿por qué X en vez de Y?”.
2. Seleccionar local o global según el stakeholder.
3. Definir un baseline coherente con la pregunta y el dominio.
4. Elegir el método: SHAP/Sampled Shapley, IG, XRAI u otra alternativa discutida.
5. Calcular atribuciones sobre instancias relevantes.
6. Para una vista global, agregar un conjunto suficientemente representativo.
7. Ordenar importancia por valor absoluto sin perder el signo de cada contribución.
8. Verificar que las atribuciones sumen aproximadamente la diferencia respecto al baseline cuando el método lo permita.
9. Contrastar la explicación con conocimiento del problema y la distribución de datos.
10. Corregir datos o modelo si aparecen señales espurias; repetir evaluación.
11. En producción, generar explicaciones junto con predicciones cuando el caso lo exija.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Baseline informativo/no informativo | Define la comparación y cambia las atribuciones. | Un baseline inadecuado genera explicaciones engañosas. |
| Explicación local/global | Responde a casos individuales o conducta agregada. | Generalizar desde una instancia o esconder casos al promediar. |
| Sampled Shapley, IG o XRAI | Determina compatibilidad y granularidad. | No todos sirven para todo modelo o tipo de dato. |
| Número de pasos/rutas o combinaciones | Puede mejorar exactitud de la explicación. | Aumenta coste computacional. |
| Muestra para explicación global | Define qué conducta se resume. | Una muestra no representativa replica sesgo de selección. |
| Granularidad en imagen | IG ofrece píxel fino; XRAI regiones. | La granularidad incorrecta dificulta interpretar el caso. |

### 3.10 Ventajas

- Aumenta transparencia y puede mejorar confianza.
- Permite auditoría local y resumen global.
- Revela features espurias, bias y problemas de datos.
- Ayuda a simplificar o mejorar modelos.
- Puede integrarse con predicciones desplegadas.
- Es aplicable a diferentes arquitecturas mediante métodos agnósticos o específicos.

### 3.11 Limitaciones

- Las explicaciones son aproximaciones, no acceso perfecto al razonamiento del modelo.
- Su calidad depende de training data, modelo y baseline.
- Las relaciones halladas pertenecen al contexto del dataset y modelo, no prueban causalidad universal.
- Diferentes métodos o baselines pueden producir conclusiones distintas.
- Las explicaciones no sustituyen métricas ni evaluación de datos.

### 3.12 Riesgos y trade-offs

- **Interpretación incorrecta:** magnitudes de coeficientes no son comparables si las features usan unidades distintas; tampoco capturan necesariamente correlaciones.
- **Confianza excesiva:** una explicación plausible puede llevar a aceptar un modelo o datos deficientes.
- **Bias:** el método puede exponer sesgo, pero también reflejarlo si la muestra no representa a la población.
- **Baseline:** una referencia inválida puede dominar el resultado.
- **Coste:** más pasos o combinaciones mejoran precisión de atribución a cambio de cómputo y latencia.
- **Globalización:** promediar explicaciones locales puede esconder subgrupos.
- **Serving:** devolver explicaciones añade componentes y coste operativo.

### 3.13 Cuándo utilizarlo

- Cuando una decisión necesita justificación local.
- Cuando compliance requiere una visión del comportamiento.
- Al depurar shortcuts, leakage semántico o features espurias.
- Para resumir señales importantes en un conjunto de test.
- Cuando especialistas deben validar regiones o features relevantes.
- En modelos complejos que no son interpretables por diseño.

### 3.14 Cuándo no utilizarlo

- No usar explicaciones como único criterio de calidad.
- No extrapolar una relación más allá del dataset, modelo y baseline.
- No interpretar directamente magnitudes de coeficientes con escalas heterogéneas como importancia absoluta.
- No elegir un método incompatible con la arquitectura o modalidad.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Modelo interpretable por diseño | Relación más directa entre estructura y salida. | Los pesos aún pueden malinterpretarse; puede no cubrir tareas complejas. | Árboles o modelos lineales adecuados al problema. |
| Feature attributions | Cuantifican contribuciones locales/globales. | Dependen del baseline y son aproximadas. | Cuando interesa qué features movieron la salida. |
| Counterfactual analysis | Compara instancias parecidas con predicciones distintas. | Depende de la cobertura y similitud del dataset. | Para explorar qué cambios acompañan otra decisión. |
| Example-based explanations | Vinculan la predicción a ejemplos de training y suelen ser intuitivas. | Funcionan mejor en imagen o texto. | Cuando ejemplos similares comunican mejor que valores de atribución. |

### 3.16 Implementación práctica

Ejemplo mínimo de la lógica SHAP descrita:

```python
explainer = shap.DeepExplainer(model, X_train_sample)
local_values = explainer.shap_values(X_test_sample)

# Vista local: baseline + contribuciones -> predicción aproximada
shap.force_plot(explainer.expected_value[0], local_values[0][0], X_test_sample.iloc[0])

# Vista global: agregación de varias explicaciones locales
shap.summary_plot(local_values, X_test_sample)
```

**Nota práctica de actualización:** las firmas y objetos devueltos por SHAP y los servicios cloud pueden haber cambiado desde 2020. Cualquier adaptación de API debe conservar la lógica del capítulo y documentarse como actualización, no como contenido original.

### 3.17 Ejemplo mínimo conceptual

Un modelo predice una duración de bicicleta de 10.7 minutos frente a un baseline de 13.6, calculado con medianas. Las atribuciones muestran cuánto reduce o aumenta cada feature la predicción; la distancia aporta la mayor reducción. La suma aproximada de contribuciones debe explicar la diferencia. El resultado esperado no es causalidad, sino una explicación contextual del comportamiento del modelo.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En datathons, una explicación global sobre una muestra de validación puede detectar rápido variables espurias o features que el modelo ignora. Una explicación local es útil al revisar errores extremos. Tiene prioridad media después de fijar baseline y validación, salvo que interpretabilidad sea parte explícita de la evaluación.

En coding tests con tiempo limitado, explicar un árbol o modelo lineal puede ser suficiente; SHAP/IG puede consumir tiempo y cómputo. El patrón puede mejorar indirectamente el leaderboard al revelar un problema de datos o una señal espuria, pero el capítulo no justifica una ganancia automática. Debe evitarse ajustar el pipeline solo para producir una explicación más atractiva.

### 3.19 Checklist operativo del patrón

- [ ] Definir la pregunta “por qué X frente a Y”.
- [ ] Elegir explicación local o global.
- [ ] Seleccionar y justificar el baseline.
- [ ] Verificar compatibilidad del método con modelo y datos.
- [ ] Usar ejemplos representativos.
- [ ] Comprobar coherencia aproximada de las atribuciones.
- [ ] Buscar señales espurias o bias.
- [ ] No confundir asociación con causalidad.
- [ ] Combinar explicaciones con métricas y evaluación de datos.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Heuristic Benchmark | Complementario | El benchmark da contexto global; sus técnicas pueden ayudar a elegir una referencia para explicaciones. |
| Fairness Lens | Complementario | Explainability puede diagnosticar bias, pero fairness exige evaluación y mitigación específicas por grupos. |
| Reframing | Puede combinarse | Cambiar regresión por clasificación cambia el baseline de uno a un valor por clase. Se menciona solo como ejemplo. |

## Pattern 30: Fairness Lens

### 3.1 Problema

Los modelos aprenden de datasets creados por personas y pueden reflejar o amplificar bias. El capítulo distingue bias natural, como la rareza real del fraude, de **problematic bias**, que perjudica de manera diferente a grupos de personas.

Las fuentes incluyen distribución que no representa a la población, representación desigual de grupos dentro de labels, reporting bias, experimenter bias en el etiquetado, proxy bias en variables correlacionadas y objetivos de entrenamiento que optimizan la media mientras fallan en minorías. Quitar una feature sensible puede empeorar el diagnóstico si otras variables, como código postal o ingreso, actúan como proxies.

### 3.2 Contexto de aplicación

Aplica a todo el flujo, desde recolección y labeling hasta splits, entrenamiento, evaluación, despliegue y mantenimiento. Es especialmente importante en datasets centrados en personas y decisiones de crédito, salud, contenido o acceso a servicios, pero también cubre imagen y texto.

No existe una definición ni métrica universal de fairness. La evaluación depende de quién recibe el impacto, de qué manera y del objetivo concreto. El patrón usa el What-If Tool como ejemplo model-agnostic de análisis pre y postentrenamiento.

### 3.3 Señales que indican que el patrón puede ser necesario

- Un grupo demográfico o escenario está subrepresentado.
- Labels positivos/negativos se distribuyen de forma diferente por grupo.
- El etiquetado es subjetivo y los labelers no son diversos.
- La accuracy global es alta, pero cambia notablemente entre slices.
- Matrices de confusión o tasas de aprobación difieren entre grupos.
- Features aparentemente neutrales pueden funcionar como proxies.
- Términos de identidad alteran predicciones que deberían mantenerse.
- Train, validation y test no conservan el balance relevante.
- **Inferencia práctica:** una métrica única puede ocultar daño sobre una minoría.

### 3.4 Solución propuesta

1. Identificar quién será afectado y cómo.
2. Distinguir skew natural de potential problematic bias.
3. Antes de entrenar, analizar distribución y representación por slices relevantes.
4. Revisar diversidad y subjetividad del proceso de labeling.
5. Ajustar datos, cobertura o incluso la tarea de predicción si faltan casos.
6. Mantener el balance relevante en train, validation y test.
7. Entrenar con objetivos coherentes con el daño que se quiere evitar.
8. Después de entrenar, evaluar métricas, matrices de confusión y resultados por grupo.
9. Elegir una noción de fairness apropiada, por ejemplo demographic parity o equality of opportunity.
10. Mitigar mediante datos, métrica de optimización, threshold u otras técnicas.
11. Repetir el análisis de forma continua y automatizarlo al operacionalizar.

### 3.5 Cómo funciona

El patrón coloca una lente por grupos sobre cada etapa:

`población afectada → datos por slices → splits balanceados → objetivo de entrenamiento → métricas por slices → mitigación → monitorización`.

Antes del entrenamiento, examina:

- **Data distribution bias:** faltan grupos o variaciones que aparecerán en producción.
- **Data representation/reporting bias:** el dataset puede estar balanceado en cantidad, pero representar los grupos de forma desigual dentro de labels o contextos.
- **Experimenter bias:** el etiquetado subjetivo refleja a quienes etiquetan.
- **Implicit/proxy bias:** eliminar la feature sensible no elimina su señal indirecta.

Después del entrenamiento, compara accuracy y errores por slices y permite probar thresholds. **Demographic parity** busca igualar la proporción de resultados positivos entre grupos; **equality of opportunity** busca igual oportunidad de predicción positiva entre quienes realmente pertenecen a la clase positiva.

### 3.6 Por qué funciona

Las métricas agregadas promedian diferencias. El slicing hace visibles los efectos sobre subpoblaciones y permite relacionarlos con cobertura de datos, labels, objetivo o threshold. Actuar antes de entrenar evita que el modelo aprenda una distribución defectuosa; actuar después detecta bias introducido o amplificado por arquitectura y optimización.

La repetición a lo largo del ciclo reconoce que fairness no es una comprobación única. Las condiciones, datos y usos cambian, por lo que la evaluación debe acompañar al modelo desplegado.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Dataset, labels, slices relevantes, modelo, predicciones, ground truth y contexto de impacto. |
| Transformación | Análisis de distribución, métricas por slice, comparación de errores y optimización/mitigación. |
| Salida | Disparidades documentadas, datos o tarea refinados, thresholds/métricas elegidos y límites de uso. |

### 3.8 Procedimiento operativo

1. Enumerar usuarios, grupos y consecuencias de errores.
2. Definir slices relevantes sin asumir que quitar atributos sensibles resuelve el problema.
3. Medir representación total y distribución de labels por slice.
4. Auditar proceso y diversidad de labeling.
5. Buscar proxies y relaciones de representación.
6. Añadir, retirar o corregir ejemplos; si no hay cobertura suficiente, restringir o reformular la tarea.
7. Dividir conservando la representación relevante en train/validation/test.
8. Elegir métricas que reflejen el daño, no solo accuracy total.
9. Entrenar y obtener predicciones sobre test suficientemente balanceado.
10. Comparar métricas y matrices de confusión por slice.
11. Elegir la noción de fairness apropiada al caso.
12. Mitigar con recolección/rebalanceo, cambio de objetivo, threshold, augmentación o reglas de salida.
13. Documentar escenarios de uso y no uso.
14. Automatizar evaluación y repetir durante mantenimiento.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Slices protegidos o relevantes | Define dónde se mide disparidad. | Omitir intersecciones puede esconder daño. |
| Métrica global frente a métricas por slice | Cambia qué fallos se ven. | Accuracy global oculta clases/grupos minoritarios. |
| Classification threshold | Cambia positivos, falsos positivos y falsos negativos. | Mejorar una noción de fairness puede degradar utilidad o accuracy. |
| Demographic parity | Iguala tasa de resultados positivos. | Puede no ser la noción adecuada si las condiciones base difieren. |
| Equality of opportunity | Iguala oportunidad entre positivos reales. | Requiere ground truth fiable y no resuelve todas las disparidades. |
| Representación en splits | Hace comparable entrenamiento y evaluación. | Un test escaso no permite medir grupos problemáticos. |
| Inclusión de features sensibles | Facilita detectar y corregir bias. | Uso indebido puede causar daño; eliminarlas puede esconder proxies. |
| Ablation o augmentación | Reduce dependencia de términos/sesgos. | Puede eliminar señal válida o no transferir bien entre modalidades. |

### 3.10 Ventajas

- Detecta bias antes y después del entrenamiento.
- Evita confiar en métricas globales.
- Integra datos, labeling, objetivo, thresholds y despliegue.
- Permite comparar múltiples nociones de fairness.
- Favorece documentación y accountability.
- Puede automatizarse en pipelines de evaluación continua.

### 3.11 Limitaciones

- No hay una definición ni métrica universal de fairness.
- Las herramientas y técnicas no constituyen una lista exhaustiva.
- Algunas disparidades provienen de bias sistémico y proxies difíciles de eliminar.
- Datos escasos limitan la evaluación de ciertos slices.
- Corregir una métrica no garantiza equidad general.
- Las decisiones requieren contexto humano sobre impacto.

### 3.12 Riesgos y trade-offs

- **Accuracy/fairness:** cambiar thresholds u objetivos puede redistribuir errores.
- **Representación:** rebalancear sin distinguir bias natural y problemático puede distorsionar el problema.
- **Proxy bias:** quitar raza o género puede volver invisible el bias que permanece en código postal o ingreso.
- **Labeling:** ground truth subjetivo puede contener experimenter bias.
- **Validación:** un test no balanceado produce conclusiones engañosas.
- **Leakage conceptual:** diseñar slices o mitigaciones usando repetidamente el test puede sobreajustar decisiones de evaluación; **inferencia práctica** derivada de la separación de splits exigida por el capítulo.
- **Operación:** reglas, listas y evaluaciones añaden mantenimiento.
- **Fairness washing:** una sola cifra puede presentarse como prueba de justicia cuando solo cubre una definición.

### 3.13 Cuándo utilizarlo

- Cuando personas o grupos pueden recibir impactos distintos.
- En datasets human-centered o labels subjetivos.
- Cuando existen atributos sensibles o proxies plausibles.
- Cuando el rendimiento por clases/slices difiere.
- Antes de producción y de forma continua después del despliegue.

### 3.14 Cuándo no utilizarlo

- No aplicar técnicas de fairness automáticamente a todo imbalance natural sin analizar daño; para skew no problemático, el capítulo remite a Rebalancing.
- No asumir que eliminar features sensibles soluciona el bias.
- No declarar un modelo “fair” a partir de una sola métrica o revisión puntual.
- No usar un grupo con muy pocos ejemplos como evidencia concluyente sin corregir cobertura.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| What-If Tool | Análisis interactivo de datos y modelos; model-agnostic. | Trabajo manual y exploratorio. | Etapas iniciales y diagnóstico por slices. |
| Fairness Indicators (TFDV/TFMA) | Integra validación de datos y evaluación por subsets, incluso en TFX. | Mayor infraestructura. | Pipelines repetibles y automatización. |
| Rebalancing | Trata imbalance cuando no implica daño diferencial. | Cambia la distribución y no define fairness. | Skew natural o falta de ejemplos tras análisis. |
| Ajuste de threshold | Intervención rápida postentrenamiento. | Solo modifica la decisión; no corrige datos o representación. | Cuando una métrica de fairness y el contexto justifican otro umbral. |
| Allow/disallow lists | Bloquean labels o palabras dañinas en datos o salida. | Hardcoding y mantenimiento; no corrige el modelo. | Cuando el bias inherente no puede corregirse directamente. |
| Data augmentation / ablation | Reduce dependencia de términos o genera variantes. | Puede borrar señal o introducir artefactos. | Texto u otros datos donde las variaciones sean controlables. |
| Model Cards | Documentan capacidades, métricas, límites y usos. | No mitigan por sí mismas. | Transparencia, accountability y control de contexto de uso. |

### 3.16 Implementación práctica

Idea mínima de evaluación por slices:

```python
for group_name, group in test_data.groupby("agency_group"):
    y_true = group["label"]
    y_score = model.predict_proba(group[features])[:, 1]
    y_pred = y_score >= threshold
    report[group_name] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "positive_rate": y_pred.mean(),
    }
```

El código mínimo preserva la lógica: no quedarse con el promedio global y comparar tasas y errores por slices. La elección de threshold o métrica debe derivarse del contexto, no de un valor universal.

**Nota práctica de actualización:** What-If Tool, TFX, TensorBoard y servicios cloud corresponden al ecosistema descrito en 2020; cambios de API deben marcarse como actualización técnica.

### 3.17 Ejemplo mínimo conceptual

Un modelo de aprobación de hipotecas alcanza buena accuracy global. Al segmentar por agencia, la accuracy y la tasa de aprobación difieren de forma marcada. El análisis preentrenamiento muestra representación desigual. Se mejora la cobertura, se conservan balances en los splits y se reevalúan thresholds según una noción explícita de fairness. El resultado esperado es hacer visibles y reducir disparidades, no certificar fairness absoluta.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En un datathon con datos de personas, revisar balance por label y slices durante EDA tiene prioridad alta. Una matriz de confusión por grupo puede revelar por qué una métrica global parece fuerte. Si el reto incluye fairness, thresholds y métricas por slice deben formar parte de validación desde el principio.

Bajo presión, lo mínimo viable es identificar grupos afectados, contar ejemplos por label/slice, verificar los tres splits y comparar errores. La augmentación, automatización o reglas de producción tienen prioridad menor si el reto no las exige. El patrón puede evitar un leaderboard engañoso cuando el score agregado premia resultados desiguales; no promete mejorar el score principal.

### 3.19 Checklist operativo del patrón

- [ ] Identificar quién recibe el impacto y qué errores importan.
- [ ] Distinguir imbalance natural de problematic bias.
- [ ] Medir cobertura y labels por slice.
- [ ] Revisar labeling subjetivo y diversidad de labelers.
- [ ] Buscar proxy bias antes de eliminar features sensibles.
- [ ] Mantener representación relevante en todos los splits.
- [ ] Evaluar métricas y matrices de confusión por grupo.
- [ ] Elegir una noción explícita de fairness.
- [ ] Mitigar en datos, training o postprocesamiento según la causa.
- [ ] Documentar límites y repetir evaluación tras el despliegue.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Explainable Predictions | Complementario | Las explicaciones pueden diagnosticar bias, pero no constituyen por sí solas una evaluación de fairness. |
| Heuristic Benchmark | Puede combinarse | Los resultados por slices pueden compararse contra una referencia comprensible. |
| Rebalancing | Complementario | Se usa cuando faltan ejemplos o hay imbalance, después de distinguir skew natural de problematic bias. |
| Reframing | Puede combinarse | Restringir o cambiar la tarea puede ser preferible cuando no hay datos suficientes. |
| Workflow Pipeline | Dependencia operativa | Permite automatizar evaluación de datos y fairness en un flujo repetible. |
| Continued Model Evaluation | Complementario | Lleva la revisión de fairness al mantenimiento continuo. |

# 4. Reglas de decisión del capítulo

| Situación | Pattern recomendado | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| No existe proceso anterior y la métrica no tiene contexto | Heuristic Benchmark | Crea un suelo intuitivo. | Mantener la regla trivial y estimarla con training. |
| Existe proceso operativo | Heuristic Benchmark contra el proceso actual | El estándar real es más relevante que una heurística inventada. | Medir ambos de forma comparable. |
| Una mejora técnica debe justificar despliegue | Heuristic Benchmark + utility value | Convierte la diferencia en impacto. | Considerar coste, fiabilidad y error budget. |
| Se necesita justificar una predicción individual | Explainable Predictions, instance-level | Relaciona features con una salida concreta. | La explicación depende del baseline. |
| Se necesita resumir comportamiento general | Explainable Predictions, global | Agrega atribuciones sobre ejemplos. | Usar una muestra representativa. |
| Hay sospecha de señal espuria | Explainable Predictions | Muestra qué inputs influyen. | No confundir explicación con causalidad. |
| La métrica global oculta diferencias por grupos | Fairness Lens | Evalúa slices y matrices de confusión. | Definir la noción de fairness adecuada. |
| Faltan ejemplos de grupos relevantes | Fairness Lens antes de training | Corrige cobertura o restringe la tarea. | Mantener representación en todos los splits. |
| El skew es natural y no causa daño diferencial | Rebalancing, referido por Fairness Lens | Trata imbalance sin afirmar que es problematic bias. | No equiparar balance estadístico con fairness. |
| No puede corregirse directamente una salida dañina | Fairness Lens con allow/disallow lists | Añade una barrera operativa. | Es hardcoding, no una corrección del modelo. |

# 5. Comparación entre patrones del capítulo

| Pattern | Problema que resuelve | Coste | Beneficio | Complejidad |
| ------- | --------------------- | ----- | --------- | ----------- |
| Heuristic Benchmark | Falta de contexto y suelo de rendimiento. | Bajo, salvo benchmarks humanos o recolección especial. | Comunicación, diagnóstico y decisión de despliegue. | Baja. |
| Explainable Predictions | Falta de transparencia sobre señales de una predicción. | Medio/alto según método, muestra y serving. | Confianza, auditoría y depuración. | Media/alta. |
| Fairness Lens | Disparidades entre grupos a lo largo del flujo. | Medio/alto por análisis, datos, mitigación y monitorización. | Reducción de bias problemático y mejor accountability. | Alta. |

Los patrones son complementarios. El benchmark responde “¿el modelo mejora una referencia?”, Explainable Predictions responde “¿qué señales movieron la predicción?” y Fairness Lens responde “¿cómo se distribuyen los efectos y errores entre grupos?”. Ninguna respuesta sustituye a las otras.

# 6. Riesgos globales del capítulo

- **Distribución incorrecta:** datos que no cubren población, clases o variaciones de producción.
- **Validación incorrecta:** métricas agregadas o test sin casos suficientes por slice.
- **Data leakage:** inferencia práctica relevante al calcular benchmarks o decisiones sobre test; las estadísticas deben provenir de training y el test reservarse para evaluación.
- **Dependencia excesiva de heurísticas:** superar un benchmark débil no demuestra utilidad.
- **Interpretabilidad aparente:** explicaciones plausibles pueden generar confianza excesiva.
- **Baseline inadecuado:** cambia la lectura de atribuciones.
- **Fairness incompleta:** una única métrica o threshold no certifica equidad.
- **Proxy bias:** eliminar atributos sensibles puede esconder la señal en otras features.
- **Complejidad operacional:** explicaciones en serving, reglas y evaluación continua requieren mantenimiento.
- **Cambios tras despliegue:** fairness y comportamiento deben revisarse de forma iterativa.

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

1. Construir un benchmark trivial adecuado a la tarea.
2. Revisar distribución de labels y slices que puedan afectar a grupos.
3. Detectar si hay variables sensibles, proxies, labeling subjetivo o cobertura desigual.
4. Confirmar que el esquema de splits permitirá evaluar esos casos.

### Durante el baseline

- Comparar el primer modelo contra constante, frecuencia, tabla o persistencia.
- Medir no solo el score global, sino también estratos relevantes.
- Conservar un modelo interpretable si resuelve suficientemente la tarea.

### Durante la mejora del modelo

- Usar explicaciones globales y locales para detectar señales espurias y errores extremos.
- Añadir o corregir datos si las explicaciones revelan shortcuts.
- Comparar métricas por slice y evaluar mitigaciones solo cuando el contexto las justifique.

### Durante validación

- Usar la misma métrica y split para benchmark y modelo.
- Mantener cobertura relevante en train, validation y test.
- No extrapolar explicaciones fuera del dataset/modelo.
- Documentar la definición de fairness elegida y sus límites.

### Bajo presión de tiempo

- **Alta prioridad:** benchmark simple, métricas por slices, revisión del balance de splits.
- **Prioridad media:** explicaciones globales sobre una muestra, análisis local de errores, ajuste contextual de threshold.
- **Baja prioridad:** explicaciones servidas en producción, automatización completa, Model Cards y reglas operativas, salvo que sean entregables explícitos.

# 8. Checklist global del capítulo

- [ ] Identificar stakeholders, usuarios y consecuencias.
- [ ] Establecer un benchmark o proceso operativo comparable.
- [ ] Evaluar el modelo globalmente y por segmentos.
- [ ] Elegir un baseline de explicación coherente.
- [ ] Buscar features espurias y proxy bias.
- [ ] Verificar cobertura y balance relevante en todos los splits.
- [ ] Separar accuracy, explainability y fairness como preguntas distintas.
- [ ] Traducir mejoras a utilidad antes del despliegue.
- [ ] Documentar límites, usos previstos y contraindicaciones.
- [ ] Planificar evaluación continua después del despliegue.

# 9. Mapeo hacia `knowledge/`

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Heuristic Benchmark | `knowledge/models/heuristic_benchmark.md` | Alta |
| Utility value y comparación operativa | `knowledge/metrics/model_utility.md` | Alta |
| Explainable Predictions | `knowledge/interpretability/explainable_predictions.md` | Alta |
| Baselines de explicación y feature attributions | `knowledge/interpretability/feature_attributions.md` | Alta |
| Fairness Lens | `knowledge/responsible_ai/fairness_lens.md` | Alta |
| Bias de distribución, representación, proxy y labeling | `knowledge/responsible_ai/data_bias.md` | Alta |
| Evaluación por slices y thresholds de fairness | `knowledge/metrics/fairness_metrics.md` | Alta |
| Automatización de evaluación responsable | `knowledge/mlops/responsible_ai_monitoring.md` | Media |
| Balance de splits por grupos | `knowledge/validation/fairness_aware_splitting.md` | Media |

# 10. Índice de patrones reutilizables

| Pattern | Activador principal | Acción principal | Knowledge destino |
| ------- | ------------------- | ---------------- | ----------------- |
| Heuristic Benchmark | La métrica no dice si el modelo es bueno o útil. | Comparar con proceso vigente o regla simple y traducir la mejora a utilidad. | `knowledge/models/heuristic_benchmark.md` |
| Explainable Predictions | Se desconoce por qué el modelo predice X. | Calcular explicación local/global respecto a un baseline. | `knowledge/interpretability/explainable_predictions.md` |
| Fairness Lens | Resultados, datos o errores pueden diferir entre grupos. | Evaluar y mitigar por slices durante todo el flujo. | `knowledge/responsible_ai/fairness_lens.md` |

# 11. Trazabilidad

Las páginas indicadas son las páginas impresas del capítulo; el PDF adjunto contiene exactamente las páginas 319–358.

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Propósito y stakeholders de Responsible AI | pp. 319–320 |
| Pattern 28: Heuristic Benchmark - Problem y Solution | pp. 320–321 |
| Tipos de benchmarks | Tabla 7-1, pp. 322–323 |
| Development check | p. 324 |
| Human experts y Utility value | pp. 324–325 |
| Pattern 29: Explainable Predictions - Problem y Solution | pp. 326–329 |
| Instance-level y global feature attributions | pp. 329–330 |
| Sampled Shapley e Integrated Gradients | Tabla 7-2, p. 330 |
| Model baseline; informative y uninformative | pp. 330–332; Figura 7-2, p. 332 |
| Relación entre Heuristic Benchmark y model baseline | p. 333 |
| SHAP local y global | pp. 333–335; Figuras 7-3 y 7-4, pp. 334–335 |
| Explanations from deployed models; IG, Sampled Shapley y XRAI | pp. 335–339; Figuras 7-5 y 7-6, p. 338 |
| Data selection bias y alternativas de explicación | pp. 339–342; Figuras 7-7 y 7-8, pp. 341–342 |
| Limitations of explanations | pp. 342–343 |
| Pattern 30: Fairness Lens - Problem | pp. 343–345 |
| Fairness Lens - Solution y análisis antes de training | pp. 345–350 |
| Tipos de data bias | Tabla 7-3, p. 347 |
| What-If Tool y distribución por agencia | Figuras 7-9 a 7-12, pp. 346–350 |
| Counterfactual fairness y análisis después de training | pp. 351–353 |
| Thresholds, demographic parity y equality of opportunity | p. 353 |
| Evaluación continua y Fairness Indicators | pp. 354–355; Figura 7-15, p. 355 |
| Allow/disallow lists y data augmentation | pp. 356–357 |
| Model Cards y fairness versus explainability | pp. 357–358; Figura 7-16, p. 357 |
| Resumen del capítulo | p. 358 |

## Control de alcance y procesamiento

- Fuente exclusiva: capítulo 7 del PDF indicado.
- No se usaron fuentes externas ni otros capítulos para completar contenido.
- Las referencias a otros patrones se conservaron solo cuando el propio capítulo las utiliza para explicar relaciones.
- Se revisaron las 40 páginas del archivo, correspondientes a las páginas impresas 319–358.
- Texto, tablas y figuras relevantes pudieron procesarse. No se detectaron páginas ilegibles ni secciones omitidas.
