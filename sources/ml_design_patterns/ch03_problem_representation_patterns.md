---
title: "Problem Representation Design Patterns"
book: "Machine Learning Design Patterns"
chapter: "3"
topics:
  - "Problem representation"
  - "Model output design"
  - "Imbalanced learning"
  - "Composite models"
patterns:
  - "Reframing"
  - "Multilabel"
  - "Ensembles"
  - "Cascade"
  - "Neutral Class"
  - "Rebalancing"
target_knowledge_files:
  - "knowledge/problem_definition/reframing.md"
  - "knowledge/models/multilabel.md"
  - "knowledge/ensembles/ensembles.md"
  - "knowledge/mlops/cascade.md"
  - "knowledge/problem_definition/neutral_class.md"
  - "knowledge/preprocessing/rebalancing.md"
  - "knowledge/metrics/imbalanced_classification.md"
---

# 3. Problem Representation Design Patterns

## 1. Propósito del capítulo

El capítulo aborda decisiones de representación del problema que aparecen antes y durante el diseño del modelo: cómo formular el objetivo, representar salidas múltiples o inciertas, combinar o encadenar modelos, tratar casos ambiguos y entrenar con distribuciones de etiquetas desbalanceadas. Estos problemas justifican patrones porque una arquitectura o métrica aparentemente válida puede imponer un objetivo equivocado, ocultar errores de la clase minoritaria o producir diferencias entre entrenamiento e inferencia.

# 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Reframing | La formulación regresión/clasificación no representa bien la incertidumbre o el objetivo real | Problem representation | Alta |
| Multilabel | Una instancia puede pertenecer a más de una clase | Problem representation | Alta |
| Ensembles | Un modelo individual tiene bias o variance reducible | Ensembling | Alta |
| Cascade | La salida de un modelo condiciona otro modelo | Pipeline | Media |
| Neutral Class | Etiquetas binarias fuerzan decisiones arbitrarias o marginales | Problem representation | Media |
| Rebalancing | Una clase o región del target está muy infrarrepresentada | Training / Metrics | Alta |

# 3. Patrones

## Pattern 5: Reframing

### 3.1 Problema

Una formulación intuitiva puede ser demasiado rígida. Una regresión produce un único valor aun cuando el target sea multimodal, tenga alta varianza condicional o requiera expresar incertidumbre. Una clasificación puede optimizar una acción proxy -por ejemplo, el clic- que no coincide con el objetivo real -por ejemplo, tiempo visto o utilidad-. Añadir capacidad, datos o features no corrige por sí mismo un objetivo mal formulado.

### 3.2 Contexto de aplicación

Regresión con distribuciones condicionales anchas, no normales o multimodales; clasificación cuyo label no está alineado con la decisión de negocio; targets continuos que operativamente se consumen por rangos; salidas que deben restringirse; recomendaciones o localización donde una representación numérica puede ser más reutilizable.

### 3.3 Señales que indican que el patrón puede ser necesario

- Para inputs similares aparecen varios valores plausibles del target.
- La media oculta modos o una incertidumbre que cambia con los inputs.
- El consumidor toma decisiones por rangos, no por el valor exacto.
- La métrica mejora mientras el comportamiento del producto empeora por label bias.
- La salida debe permanecer en un intervalo físico válido.
- **Inferencia práctica:** el error irreducible domina y el caso de uso necesita la distribución, no solo una estimación puntual.

### 3.4 Solución propuesta

1. Revisar qué decisión debe soportar la predicción.
2. Cambiar regresión por clasificación mediante buckets cuando interese una distribución discreta o categorías operativas.
3. Cambiar clasificación por regresión cuando una representación continua refleje mejor utilidad, afinidad o ubicación.
4. Adaptar label, capa de salida, loss, métricas y lógica de inferencia.
5. Validar que el nuevo label no introduzca sesgo y que la pérdida de precisión por discretización sea aceptable.

### 3.5 Cómo funciona

Al discretizar un target continuo, el modelo devuelve probabilidades por intervalo y puede conservar multimodalidad e incertidumbre. En sentido inverso, una representación continua permite buscar elementos cercanos o predecir una magnitud más alineada con el objetivo. Para limitar una regresión a `[MIN_Y, MAX_Y]`, el capítulo propone una salida sigmoide en `[0,1]` seguida de un escalado lineal.

### 3.6 Por qué funciona

La nueva representación cambia qué estructura puede aprender el modelo. Los buckets sacrifican resolución, pero aportan una densidad discreta más expresiva que una media. Un label alineado con la decisión evita optimizar proxies indeseados. Una salida acotada incorpora la restricción en la arquitectura.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Features originales y definición del uso final |
| Transformación | Bucketización, cambio de label, representación continua o escalado acotado |
| Salida | Probabilidades por bucket, valor continuo reinterpretado o valor restringido |

### 3.8 Procedimiento operativo

1. Inspeccionar la distribución del target condicionada por combinaciones de inputs.
2. Determinar si se necesita valor puntual, intervalo, distribución o ranking por cercanía.
3. Definir buckets o representación continua con significado operativo.
4. Entrenar ambas formulaciones cuando sean comparables.
5. Evaluar con métricas propias de cada formulación y con la decisión final.
6. Revisar calibración, data sufficiency y label bias.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Número y ancho de bins | Controla precisión y granularidad | Pocos bins pierden detalle; muchos exigen más datos |
| Umbrales de buckets | Alinean el modelo con decisiones | Cortes arbitrarios crean discontinuidades |
| Label objetivo | Define lo que optimiza el modelo | Un proxy inadecuado genera label bias |
| Distribución paramétrica alternativa | Modela incertidumbre directamente | Requiere especificar una familia correcta |

### 3.10 Ventajas

- Captura incertidumbre y multimodalidad.
- Puede alinear mejor entrenamiento y objetivo del producto.
- Permite restringir el rango de salida.
- Puede reducir la dificultad operativa si las decisiones ya usan categorías.

### 3.11 Limitaciones

- La bucketización pierde precisión dentro de cada intervalo.
- Las probabilidades de clasificación pueden estar mal calibradas.
- Más categorías requieren más ejemplos.
- Comparar RMSE con accuracy directamente no es válido.

### 3.12 Riesgos y trade-offs

El trade-off central es precisión puntual frente a expresividad distributiva. Existe riesgo de label bias, bins mal elegidos y probabilidades sobreconfiadas. Una sigmoide acotada se aproxima a los extremos sin alcanzarlos. Cambiar la formulación altera métricas y dificulta comparaciones directas.

### 3.13 Cuándo utilizarlo

Cuando el output tiene incertidumbre relevante, la distribución es compleja, el uso final opera con rangos o el objetivo actual incentiva resultados indeseados.

### 3.14 Cuándo no utilizarlo

Cuando la distribución condicional es estrecha y se necesita máxima precisión numérica; cuando no hay datos suficientes por categoría; o cuando la reformulación no mejora la alineación con el uso final.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Quantile regression | Estima cuantiles sin bins | No representa necesariamente toda la forma | Cuando interesan intervalos condicionales |
| Regresión probabilística | Modela una distribución explícita | Exige asumir/modelar la familia | Cuando existe una distribución adecuada |
| Multitask learning | Conserva clasificación y regresión | Más arquitectura y losses | Cuando ambas tareas aportan señal |
| Mantener regresión | Máxima resolución | Predicción puntual rígida | Distribución estrecha y target continuo requerido |

### 3.16 Implementación práctica

Idea mínima para salida acotada: `sigmoid(x) * (MAX_Y - MIN_Y) + MIN_Y`. Para reframing a clasificación: convertir el target continuo en bins, entrenar softmax y devolver las probabilidades de todos los bins, no solo `argmax`.

### 3.17 Ejemplo mínimo conceptual

Precipitación con varios valores plausibles para las mismas condiciones → bucketizar cantidades y predecir su distribución → conservar modos e incertidumbre que una media ocultaría.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es de alta prioridad cuando el target continuo tiene picos, muchos ceros o dispersión heterogénea, o cuando el enunciado evalúa categorías derivadas. Bajo presión, probar bins es rápido; la comparación debe hacerse con la métrica oficial y una validación coherente. Puede mejorar un baseline si la métrica recompensa la nueva representación, pero no garantiza ventaja si exige precisión continua.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar el objetivo real y el label proxy.
- [ ] Inspeccionar incertidumbre y forma condicional del target.
- [ ] Definir bins o representación alternativa con significado.
- [ ] Ajustar output, loss, métricas e inferencia.
- [ ] Revisar calibración, data sufficiency y label bias.
- [ ] Comparar contra la formulación original.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Neutral Class | Puede combinarse | Añade una categoría para casos marginales |
| Rebalancing | Puede combinarse | Los buckets pueden quedar desbalanceados |
| Cascade | Alternativo | Una reformulación puede evitar encadenar modelos |

## Pattern 6: Multilabel

### 3.1 Problema

Una instancia puede tener varias etiquetas correctas. Softmax y `argmax` fuerzan exclusividad y pierden etiquetas solapadas, jerárquicas o aportadas por distintos anotadores.

### 3.2 Contexto de aplicación

Imágenes con varios objetos, texto con múltiples tags o tipos de toxicidad, pacientes con varias condiciones, jerarquías de etiquetas y descripciones humanas simultáneamente válidas.

### 3.3 Señales que indican que el patrón puede ser necesario

- Varias clases son verdaderas a la vez.
- Los anotadores difieren, pero sus descripciones son compatibles.
- Existen jerarquías o combinaciones recurrentes de labels.
- **Inferencia práctica:** `argmax` elimina información necesaria para la aplicación.

### 3.4 Solución propuesta

1. Codificar labels como vector multi-hot.
2. Usar una salida sigmoid por clase.
3. Entrenar con binary cross-entropy.
4. Elegir umbrales por clase según el uso.
5. Asegurar cobertura de combinaciones de etiquetas.

### 3.5 Cómo funciona

Cada nodo de salida resuelve una clasificación binaria independiente. Las probabilidades no tienen que sumar 1; varias pueden superar su umbral simultáneamente.

### 3.6 Por qué funciona

Elimina la competencia artificial impuesta por softmax y permite estimar pertenencia a cada clase por separado.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Ejemplo y vector multi-hot de longitud N |
| Transformación | N salidas sigmoid con binary cross-entropy |
| Salida | N probabilidades independientes y labels umbralizados |

### 3.8 Procedimiento operativo

1. Verificar que las clases no sean mutuamente excluyentes.
2. Construir el vocabulario y multi-hot labels.
3. Auditar frecuencias individuales y combinaciones.
4. Entrenar salida sigmoid con BCE.
5. Calibrar y seleccionar umbrales por clase.
6. Evaluar la salida completa y revisar clases raras.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Umbral por clase | Controla etiquetas emitidas | Un umbral global puede ser inadecuado |
| Enfoque plano o jerárquico | Simplicidad frente a especialización | El plano puede diluir hojas raras |
| Máximo de etiquetas humanas | Controla ruido | Puede descartar descripciones válidas |

### 3.10 Ventajas

- Conserva múltiples etiquetas verdaderas.
- Flexibiliza el parsing según el producto.
- Integra desacuerdos compatibles de anotadores.

### 3.11 Limitaciones

- Requiere umbrales y posible calibración por clase.
- El balance se complica por combinaciones y dependencias.
- Jerarquías grandes pueden perder detalle en un modelo plano.

### 3.12 Riesgos y trade-offs

Umbrales bajos elevan falsos positivos; altos elevan falsos negativos. Correlaciones espurias entre labels pueden impedir aprender etiquetas de forma autónoma. Miles de clases aumentan coste y necesidad de calibración.

### 3.13 Cuándo utilizarlo

Cuando una instancia admite varias etiquetas no excluyentes, jerárquicas o solapadas.

### 3.14 Cuándo no utilizarlo

Cuando exactamente una clase es correcta; use softmax. En clasificación binaria exclusiva, una sola sigmoid es suficiente, pero eso no constituye un caso multilabel.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| One-vs-rest | Compatible con clasificadores binarios; ayuda con clases raras | Entrena y sirve muchos modelos | Arquitectura sin soporte multilabel |
| Cascade | Conserva jerarquía | Mayor complejidad y propagación de errores | Etiquetas jerárquicas que justifican modelos especializados |
| Softmax | Simple | Fuerza exclusividad | Una sola clase por ejemplo |

### 3.16 Implementación práctica

`Dense(num_labels, activation="sigmoid")` con `binary_crossentropy`; en inferencia, iterar todas las salidas y aplicar el umbral correspondiente a cada label.

### 3.17 Ejemplo mínimo conceptual

Pregunta sobre gráficos con pandas → sigmoid independiente para `Python`, `pandas` y `visualization` → emitir todas las etiquetas cuya probabilidad supere su umbral.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Detectarlo correctamente es crítico: una salida softmax invalida el problema. Un baseline sigmoid+BCE es rápido. Las mejoras más prácticas son umbrales por clase, calibración y revisión de combinaciones raras; one-vs-rest es útil si el modelo base solo admite clasificación binaria.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar no exclusividad.
- [ ] Crear labels multi-hot.
- [ ] Usar sigmoid y BCE.
- [ ] Revisar cobertura de combinaciones.
- [ ] Elegir umbrales por clase.
- [ ] Calibrar probabilidades si se selecciona top-K.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Cascade | Alternativo | Puede modelar jerarquías con etapas |
| Rebalancing | Complementario | Ayuda a clases raras en one-vs-rest |

## Pattern 7: Ensembles

### 3.1 Problema

Un modelo individual puede tener alto bias, alta variance o errores inestables. Aumentar o reducir complejidad desplaza el trade-off sin resolver necesariamente ambos componentes.

### 3.2 Contexto de aplicación

Problemas pequeños o medianos, modelos de alta variance, weak learners con alto bias y conjuntos de modelos diversos cuyos errores no estén perfectamente correlacionados.

### 3.3 Señales que indican que el patrón puede ser necesario

- Error de entrenamiento bajo y error de validación alto: variance.
- Entrenamiento y validación pobres: bias.
- Seeds o arquitecturas distintas cometen errores diferentes.
- **Inferencia práctica:** varios modelos competitivos ya entrenados ofrecen diversidad aprovechable.

### 3.4 Solución propuesta

Combinar predicciones mediante: bagging, que entrena el mismo algoritmo sobre muestras bootstrap y promedia/vota; boosting, que agrega weak learners que corrigen residuos; o stacking, que entrena un meta-modelo sobre las salidas de modelos completos.

### 3.5 Cómo funciona

Bagging paraleliza submodelos y agrega por media o voto. Boosting construye secuencialmente modelos sobre errores residuales. Stacking congela o reutiliza modelos base y aprende una función de combinación con un modelo secundario.

### 3.6 Por qué funciona

Errores parcialmente independientes se cancelan al promediar; con errores no correlacionados, el error cuadrático esperado disminuye con el número de modelos. Boosting aumenta capacidad al concentrarse en ejemplos difíciles. Stacking aprende pesos e interacciones más flexibles que un promedio.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Datos para miembros base; en stacking, predicciones base |
| Transformación | Bootstrap, corrección secuencial o meta-modelo |
| Salida | Predicción agregada |

### 3.8 Procedimiento operativo

1. Diagnosticar bias o variance.
2. Elegir bagging para variance, boosting para bias o stacking para modelos diversos.
3. Crear diversidad real entre miembros.
4. Entrenar y definir agregación.
5. Evaluar rendimiento y recursos frente a un modelo simple.
6. Medir latencia, mantenimiento e interpretabilidad.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Número de miembros | Más oportunidad de reducir error | Más coste y retornos decrecientes |
| Diversidad/correlación | Determina ganancia del promedio | Errores correlacionados no ayudan |
| Learning rate y learners en boosting | Controlan corrección secuencial | Exceso de capacidad/tiempo |
| Agregador/meta-modelo | Aprende combinación | Puede sobreajustar |

### 3.10 Ventajas

- Reduce variance con bagging.
- Reduce bias con boosting.
- Suele superar modelos individuales cuando existe diversidad.
- Stacking combina arquitecturas distintas.

### 3.11 Limitaciones

- Mayor tiempo de diseño y entrenamiento.
- Serving, recursos y mantenimiento más complejos.
- Menor interpretabilidad local.

### 3.12 Riesgos y trade-offs

Sin diversidad, el coste no compra mejora. Stacking puede sobreajustar la combinación. Boosting aumenta capacidad. Producción debe coordinar varios artefactos y predicciones. Un random forest es menos interpretable que un árbol individual.

### 3.13 Cuándo utilizarlo

Bagging ante alta variance; boosting ante alto bias; stacking cuando modelos distintos capturan señales complementarias y el presupuesto lo permite.

### 3.14 Cuándo no utilizarlo

Cuando los miembros cometen errores muy correlacionados, el modelo simple ya satisface el objetivo o la latencia, explicabilidad y mantenimiento dominan el beneficio marginal.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Promedio/weighted average | Barato y robusto | Menos flexible | Pocos modelos ya entrenados |
| Dropout | Aproxima bagging en una red | Miembros comparten parámetros y no convergen por separado | Regularización eficiente |
| Modelo único | Menor coste e interpretación más clara | Puede rendir menos | Restricciones operativas fuertes |

### 3.16 Implementación práctica

Bagging: entrenar sobre muestras con reemplazo y promediar regresiones o votar clases. Stacking: generar outputs de modelos base, mantenerlos fijos y entrenar un meta-modelo sobre esas salidas.

### 3.17 Ejemplo mínimo conceptual

Varias redes con errores parcialmente independientes → promedio de predicciones → cancelación parcial de errores aleatorios y menor variance.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Alta prioridad al final de un datathon si ya existen modelos diversos. Promediar es rápido; stacking exige validación más cuidadosa y tiempo adicional. Bagging ayuda a modelos inestables y boosting constituye un baseline fuerte para datos tabulares. La ganancia razonable depende de diversidad, no del mero número de modelos.

### 3.19 Checklist operativo del patrón

- [ ] Diagnosticar bias/variance.
- [ ] Medir diversidad de errores.
- [ ] Seleccionar método de agregación.
- [ ] Comparar con modelo simple.
- [ ] Medir coste, latencia e interpretabilidad.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Rebalancing | Puede combinarse | Entrena miembros con distintos subconjuntos de la mayoría |
| Cascade | Alternativo | Cascade encadena dependencias; ensemble agrega respuestas |

## Pattern 8: Cascade

### 3.1 Problema

Un problema contiene regímenes raros y comunes igualmente importantes, o una predicción upstream debe ser input/selector de modelos downstream. Entrenar downstream con labels perfectos crea training-serving skew porque en inferencia recibirá predicciones con errores.

### 3.2 Contexto de aplicación

Actividad usual frente a inusual, OCR u otro modelo preentrenado como feature, detección seguida de clasificación fina y regresión solo en casos detectados como positivos.

### 3.3 Señales que indican que el patrón puede ser necesario

- La salida de un modelo alimenta o selecciona otro.
- Un segmento raro se comporta de forma cualitativamente distinta.
- Se requiere consistencia entre varias predicciones relacionadas.
- El componente upstream cambiará de versión.

### 3.4 Solución propuesta

1. Entrenar el modelo upstream.
2. Ejecutarlo sobre datos para producir predicciones reales.
3. Construir datasets downstream a partir de esas predicciones, no de labels perfectos.
4. Entrenar modelos especializados y, si corresponde, un combinador probabilístico.
5. Automatizar todo como un pipeline versionado y reentrenable.

### 3.5 Cómo funciona

El primer modelo clasifica el régimen o genera una representación. Sus outputs, incluidos errores, determinan los datos que ven los modelos posteriores. En serving se replica la misma cadena.

### 3.6 Por qué funciona

Los modelos downstream aprenden la distribución efectiva que recibirán en producción. La especialización permite modelar regímenes distintos sin sacrificar el caso común, y el pipeline mantiene dependencias coherentes.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Features originales y/o output del modelo upstream |
| Transformación | Routing o encadenamiento de predicciones |
| Salida | Predicción especializada o combinación final |

### 3.8 Procedimiento operativo

1. Justificar que un modelo único no basta.
2. Definir contratos entre etapas.
3. Separar datos para entrenar y evaluar upstream.
4. Generar predicciones upstream para formar datasets downstream.
5. Entrenar especialistas sobre esas predicciones.
6. Evaluar el sistema end-to-end.
7. Reentrenar downstream ante cambios upstream.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Umbral/routing upstream | Distribuye ejemplos entre especialistas | Errores se propagan |
| Número de etapas | Aumenta especialización | Mayor fragilidad |
| Hard routing o combinación probabilística | Simplicidad frente a suavidad | Hard routing crea discontinuidades |

### 3.10 Ventajas

- Especializa modelos en regímenes diferentes.
- Reutiliza modelos preentrenados.
- Mantiene consistencia si el pipeline se entrena end-to-end operacionalmente.

### 3.11 Limitaciones

- Alta complejidad experimental y operativa.
- Puede rendir peor que un modelo único.
- Necesita historial, automatización y reentrenamiento coordinado.

### 3.12 Riesgos y trade-offs

Propagación de errores, skew por entrenar con ground truth upstream, vocabularios no vistos, latencia acumulada y mantenimiento de varios modelos. La evaluación aislada de etapas puede ocultar fallos end-to-end.

### 3.13 Cuándo utilizarlo

Cuando una predicción debe alimentar otra, se reutiliza un modelo externo o existen regímenes raros cualitativamente diferentes y relevantes.

### 3.14 Cuándo no utilizarlo

Si la condición es determinista -añádala como feature-, depende de un extremo de una sola variable -considere Mixed Input Representation-, o es suficientemente común para que un único modelo la aprenda.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Modelo único | Menor complejidad | Menos especialización | Escenario común o learnable conjuntamente |
| Feature determinista | Sin propagación de error | Requiere condición conocida | Segmento observable directamente |
| Reframing | Evita routing | Cambia resolución/objetivo | Buckets representan bien regímenes |
| Rebalancing | Atiende clase rara | No modela dependencias downstream | Solo interesa clasificar el caso raro |

### 3.16 Implementación práctica

Entrenar `classifier`; usar `classifier.predict()` para crear los datasets `predicted_typical` y `predicted_rare`; entrenar cada regresor con el conjunto predicho correspondiente; encapsular las etapas en un pipeline reproducible.

### 3.17 Ejemplo mínimo conceptual

OCR de matrícula → texto predicho, con errores reales → modelo de autorización entrenado con ese texto → comportamiento coherente cuando cambia la calidad del OCR.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Prioridad media o baja bajo presión: solo compensa cuando los regímenes son claramente distintos y un baseline único falla en ambos. Hay que generar features/predicciones upstream sin contaminación y validar el flujo completo; de otro modo la mejora local puede ser engañosa.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar necesidad de múltiples modelos.
- [ ] Entrenar downstream con predicciones upstream.
- [ ] Evitar usar labels perfectos para routing de training.
- [ ] Evaluar end-to-end.
- [ ] Versionar dependencias y reentrenar en orden.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Rebalancing | Complementario | Puede entrenar el clasificador raro upstream |
| Reframing | Alternativo | Puede evitar la cascada mediante buckets |
| Ensembles | Diferente | Ensemble agrega modelos; Cascade conecta outputs e inputs |
| Workflow Pipelines | Dependencia | Automatiza y registra la cadena |

## Pattern 9: Neutral Class

### 3.1 Problema

Una clasificación binaria obliga a etiquetar como Yes/No casos arbitrarios, marginales o donde expertos discrepan, introduciendo ruido y falsa confianza.

### 3.2 Contexto de aplicación

Decisiones humanas subjetivas, ratings centrales, casos clínicos marginales, acciones donde una zona de no decisión es útil y registros que pueden capturar aceptabilidad de alternativas.

### 3.3 Señales que indican que el patrón puede ser necesario

- Expertos competentes discrepan sistemáticamente.
- Muchas etiquetas cercanas al umbral son arbitrarias.
- Una probabilidad cercana a 0.5 mezcla ambigüedad con falta de datos.
- Downstream es sensible a cambios en casos marginales.

### 3.4 Solución propuesta

Diseñar la recolección o relabeling para crear una tercera clase disjunta, Neutral/Maybe, y entrenar una clasificación de tres clases.

### 3.5 Cómo funciona

Los casos sin preferencia clara dejan de contaminar las dos clases polarizadas. El modelo aprende regiones separadas para decisión positiva, negativa y neutral.

### 3.6 Por qué funciona

Retira de la frontera binaria etiquetas esencialmente aleatorias. Esto reduce el error irreducible inducido por la anotación y diferencia desacuerdo humano de incertidumbre por escasez de datos.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Ejemplos con decisión o validación de alternativas |
| Transformación | Mapeo a Yes / Neutral / No |
| Salida | Probabilidad para tres clases disjuntas |

### 3.8 Procedimiento operativo

1. Detectar casos arbitrarios o marginales.
2. Rediseñar la captura o usar varios labelers.
3. Definir criterios explícitos y disjuntos.
4. Entrenar tres clases.
5. Evaluar aceptación, estabilidad y efecto downstream.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Límites de Neutral | Controlan zona de abstención | Zona excesiva reduce decisiones útiles |
| Regla de acuerdo | Traduce votos a clases | Puede ocultar minorías expertas |
| Ponderar acuerdo | Mantiene binario | No distingue desacuerdo de falta de datos |

### 3.10 Ventajas

- Reduce ruido por decisiones arbitrarias.
- Evita falsa confianza.
- Mejora estabilidad de cascadas.
- Puede conservar información útil para embeddings.

### 3.11 Limitaciones

- Requiere diseñar la recolección; no debe fabricarse sin evidencia.
- Puede necesitar relabeling humano.
- Añade una clase y decisiones de límites.

### 3.12 Riesgos y trade-offs

Una clase neutral mal definida puede convertirse en cajón de sastre. Descartar casos ambiguos pierde información y puede desestabilizar downstream. Ponderar ejemplos conserva binario, pero confunde las causas de una probabilidad intermedia.

### 3.13 Cuándo utilizarlo

Cuando existe una región real de indiferencia, desacuerdo o acción no justificable y puede documentarse desde la captura de datos.

### 3.14 Cuándo no utilizarlo

Cuando todas las instancias tienen una clase binaria objetiva o no hay evidencia para asignar neutralidad.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Descartar desacuerdos | Simplifica labels | Pierde información y estabilidad | Solo si esos casos no importan |
| Peso por acuerdo | Conserva binario | Ambigüedad y falta de datos se mezclan | Se necesita salida binaria |
| Reframing | Alinea acción final | Requiere redefinir target | La neutralidad corresponde a un intervalo operativo |

### 3.16 Implementación práctica

Convertir una escala de satisfacción en `bad`, `neutral`, `good` usando límites definidos por el caso de uso, y entrenar un clasificador multiclase.

### 3.17 Ejemplo mínimo conceptual

Ratings 1-10 → 1-4 negativos, 5-7 neutrales, 8-10 positivos → el modelo no gasta capacidad separando respuestas centrales casi equivalentes.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Úselo solo si el target y la métrica permiten redefinir clases. Puede limpiar labels cercanos a un umbral, pero en una competición con labels fijos no debe alterarse sin justificar su correspondencia con el objetivo evaluado.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar ambigüedad real.
- [ ] Definir clase neutral desde datos o relabeling.
- [ ] Mantener clases disjuntas.
- [ ] Evaluar falsa confianza y estabilidad.
- [ ] Revisar impacto downstream.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Reframing | Puede combinarse | Convierte una magnitud en tres acciones |
| Cascade | Complementario | Neutral estabiliza decisiones downstream |

## Pattern 10: Rebalancing

### 3.1 Problema

Una clase domina el dataset y la accuracy oculta que el modelo ignora la minoría. En regresión, regiones extremas pueden estar infrarrepresentadas. El patrón no corrige falta de representación de una población o entorno: eso exige recolectar datos.

### 3.2 Contexto de aplicación

Fraude, default, defectos, enfermedad, spam, logs raros, clasificación binaria/multiclase/multilabel y regresión con extremos escasos.

### 3.3 Señales que indican que el patrón puede ser necesario

- Accuracy cercana a la proporción mayoritaria.
- Confusion matrix con bajo reconocimiento minoritario.
- Precision/recall minoritarias pobres.
- Muy pocos ejemplos en regiones críticas del target.
- **Inferencia práctica:** cambiar el threshold altera fuertemente falsos positivos y negativos.

### 3.4 Solución propuesta

1. Evaluar con precision, recall, F-measure o average precision-recall.
2. Mantener validación/test sin re-muestrear y con distribución representativa.
3. Aplicar downsampling, class weighting, output bias o upsampling/SMOTE solo al training.
4. Combinar técnicas si es necesario.
5. Considerar Reframing, Cascade o anomaly detection para casos adecuados.

### 3.5 Cómo funciona

Downsampling reduce la mayoría; weighting aumenta el coste de errores minoritarios; output bias inicializa la salida con el log-ratio de clases y acelera convergencia; upsampling replica o sintetiza minoría. Un ensemble puede reutilizar distintos subconjuntos de mayoría sin descartarlos globalmente.

### 3.6 Por qué funciona

Modifica la señal efectiva que llega al optimizador para que la mayoría no domine loss y gradientes. Métricas adecuadas hacen visibles los errores relevantes. Mantener test original conserva una estimación realista de producción.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Training desbalanceado; validation/test representativos |
| Transformación | Muestreo, pesos, bias, síntesis o reformulación |
| Salida | Modelo sensible a minoría, evaluado en distribución real |

### 3.8 Procedimiento operativo

1. Medir prevalencia y confusion matrix.
2. Elegir métrica alineada con el coste de errores.
3. Separar validation/test antes de cualquier sampling.
4. Probar weighting como baseline y down/up-sampling según volumen.
5. Ajustar threshold para el trade-off operativo.
6. Comparar contra baseline en datos no muestreados.
7. Revisar explicaciones de predicciones raras.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Ratio de downsampling | Expone la minoría | Pierde información mayoritaria |
| Class weights | Cambia coste de error | Pesos extremos desestabilizan |
| Output bias `log(n_min/n_maj)` | Acelera convergencia | No sustituye rebalancing |
| Vecinos/ratio SMOTE | Controla síntesis | Ejemplos irreales |
| Threshold | Cambia precision/recall | Optimización equivocada del coste |

### 3.10 Ventajas

- Evita modelos triviales de mayoría.
- Permite alinear errores con costes de negocio.
- Puede reducir entrenamiento mediante downsampling.
- Técnicas combinables con ensembles.

### 3.11 Limitaciones

- Downsampling pierde información y falla si la minoría solo tiene cientos de casos.
- Upsampling puede duplicar ruido o crear ejemplos sintéticos débiles.
- Texto es difícil de sintetizar.
- No soluciona poblaciones ausentes.

### 3.12 Riesgos y trade-offs

Re-muestrear validación/test produce métricas engañosas. Optimizar una clase puede reducir accuracy global. Downsampling aumenta bias sobre la mayoría; oversampling puede overfit. Reframing y cascadas añaden complejidad y propagación de error. La explicación es especialmente importante para verificar señales minoritarias.

### 3.13 Cuándo utilizarlo

Cuando clases o regiones relevantes son raras pero existe suficiente señal y ejemplos para aprenderlas.

### 3.14 Cuándo no utilizarlo

Cuando falta una población/entorno entero, cuando la minoría es demasiado pequeña para aprender o cuando el aparente desbalance no es relevante para la decisión.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Class weighting | Conserva todos los datos | Requiere ajustar pesos | Dataset grande |
| Downsampling | Simple y reduce coste | Descarta mayoría | Muchísima mayoría disponible |
| SMOTE/augmentación | Aumenta minoría | Puede crear artefactos | Tabular o imagen con síntesis válida |
| Anomaly detection | No exige balance binario convencional | Umbral/distancia difíciles | Labels escasos o streaming |
| Reframing/Cascade | Trata extremos como régimen | Más complejidad | Regresión con colas críticas |

### 3.16 Implementación práctica

Separar por clase; muestrear aleatoriamente la mayoría solo en train; concatenar y barajar. Para weighting, usar pesos inversos a frecuencia ajustados al coste. Evaluar siempre sobre el test original. Con ensemble, entrenar cada miembro con la misma minoría y diferentes particiones de la mayoría, y agregar por mediana.

### 3.17 Ejemplo mínimo conceptual

0.1% de fraude → accuracy trivial de 99.9% → downsampling/weights en train y evaluación con precision-recall sobre prevalencia original → medir detección real de fraude.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Alta prioridad al inicio: inspeccionar prevalencia, métrica y confusion matrix. Class weights y threshold tuning son rápidos; downsampling acelera iteraciones; SMOTE requiere validar artefactos. Mantener folds sin re-muestrear evita leaderboard local engañoso. El patrón puede mejorar una métrica sensible a minoría, no promete mejorar accuracy.

### 3.19 Checklist operativo del patrón

- [ ] Medir distribución por clase/región.
- [ ] Elegir precision, recall, F-score o PR según objetivo.
- [ ] Separar test antes de re-muestrear.
- [ ] Aplicar cambios solo a train.
- [ ] Comparar weighting, sampling y threshold.
- [ ] Revisar explicaciones y señales espurias.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Ensembles | Complementario | Reutiliza particiones de la mayoría |
| Reframing | Alternativo/Complementario | Convierte extremos en clases |
| Cascade | Puede combinarse | Clasifica régimen y aplica regresor especializado |
| Multilabel | Complementario | Atiende etiquetas individuales raras |

# 4. Reglas de decisión del capítulo

| Situación | Pattern recomendado | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Target continuo multimodal o incierto | Reframing | Devuelve distribución discreta | Pérdida de precisión y calibración |
| Varias etiquetas válidas | Multilabel | Probabilidades independientes | Umbrales por clase |
| Alta variance | Ensembles - bagging | Promedia errores diversos | Correlación entre miembros |
| Alto bias | Ensembles - boosting | Corrige residuos iterativamente | Mayor capacidad y coste |
| Output upstream alimenta downstream | Cascade | Alinea training con serving | Entrenar con predicciones, no labels perfectos |
| Expertos discrepan en casos marginales | Neutral Class | Representa ambigüedad explícita | Diseñar/recolectar la etiqueta neutral |
| Clase rara pero aprendible | Rebalancing | Refuerza su señal | Test debe conservar prevalencia |

# 5. Comparación entre patrones del capítulo

| Pattern | Problema que resuelve | Coste | Beneficio | Complejidad |
| ------- | --------------------- | ----- | --------- | ----------- |
| Reframing | Objetivo/representación inadecuados | Bajo-Medio | Mejor alineación y expresividad | Media |
| Multilabel | Múltiples clases verdaderas | Medio | No pierde labels | Media |
| Ensembles | Bias/variance reducible | Alto | Robustez/rendimiento | Media-Alta |
| Cascade | Dependencias entre modelos | Alto | Especialización coherente | Alta |
| Neutral Class | Ambigüedad forzada | Medio | Menos ruido/falsa confianza | Media |
| Rebalancing | Minoría ignorada | Bajo-Medio | Sensibilidad a casos raros | Media |

# 6. Riesgos globales del capítulo

- **Distribución incorrecta:** re-muestrear test invalida la evaluación; bins y combinaciones de labels pueden quedar mal representados.
- **Validación incorrecta:** accuracy oculta fallos minoritarios y métricas distintas no son comparables directamente.
- **Data leakage:** en cascadas, las predicciones que forman inputs downstream deben generarse con un diseño experimental que represente inferencia; usar labels upstream perfectos crea skew.
- **Overfitting/variance:** ensembles sin diversidad y upsampling repetitivo pueden no generalizar.
- **Training-serving skew:** cascadas y transformaciones de output deben reproducirse en serving.
- **Complejidad operacional:** stacking y cascadas multiplican modelos, latencia y mantenimiento.
- **Interpretabilidad:** ensembles y detección rara exigen revisar señales y atribuciones.
- **Sesgo:** un label proxy mal elegido introduce label bias; Rebalancing no reemplaza datos de poblaciones ausentes.

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

Determinar si el target es exclusivo o multilabel, medir balance, observar distribución y extremos, identificar ambigüedad y confirmar que la métrica corresponde al objetivo.

### Durante el baseline

Usar formulación simple alineada, sigmoid+BCE para multilabel, class weights para imbalance y un modelo único antes de Cascade. Registrar precision/recall además de accuracy cuando corresponda.

### Durante la mejora del modelo

Probar Reframing si la distribución lo justifica; bagging/boosting según variance/bias; threshold tuning; sampling; y promedio de modelos diversos. Reservar stacking y Cascade para evidencia clara.

### Durante validación

Conservar distribución original en folds, evaluar el pipeline completo, evitar entrenar downstream con señales perfectas inexistentes en inferencia y comparar siempre con baseline.

### Bajo presión de tiempo

- **Alta prioridad:** correcta formulación, Multilabel, métricas de imbalance, class weights, threshold tuning y promedios simples.
- **Prioridad media:** bucketización justificada, bagging/boosting y downsampling.
- **Baja prioridad:** stacking complejo, Cascade de varias etapas y relabeling Neutral Class sin datos ya disponibles.

# 8. Checklist global del capítulo

- [ ] Confirmar que el output representa la decisión real.
- [ ] Identificar exclusividad, ambigüedad y jerarquía de labels.
- [ ] Medir balance y seleccionar métricas apropiadas.
- [ ] Separar validation/test antes de sampling.
- [ ] Diagnosticar bias frente a variance.
- [ ] Evitar skew en cadenas de modelos.
- [ ] Comparar cada patrón contra un baseline simple.
- [ ] Medir coste, interpretabilidad y mantenibilidad.

# 9. Mapeo hacia `knowledge/`

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Reframing | `knowledge/problem_definition/reframing.md` | Alta |
| Multilabel | `knowledge/models/multilabel.md` | Alta |
| Ensembles | `knowledge/ensembles/ensembles.md` | Alta |
| Cascade | `knowledge/mlops/cascade.md` | Media |
| Neutral Class | `knowledge/problem_definition/neutral_class.md` | Media |
| Rebalancing | `knowledge/preprocessing/rebalancing.md` | Alta |
| Métricas para imbalance | `knowledge/metrics/imbalanced_classification.md` | Alta |
| Thresholding multilabel/imbalanced | `knowledge/metrics/threshold_selection.md` | Alta |

# 10. Índice de patrones reutilizables

| Pattern | Activador principal | Acción principal | Knowledge destino |
| ------- | ------------------- | ---------------- | ----------------- |
| Reframing | Output no refleja incertidumbre/objetivo | Cambiar representación del target | `knowledge/problem_definition/` |
| Multilabel | Varias etiquetas correctas | Sigmoid+BCE y umbrales | `knowledge/models/` |
| Ensembles | Bias o variance reducible | Agregar modelos | `knowledge/ensembles/` |
| Cascade | Predicción alimenta otra etapa | Entrenar y servir cadena coherente | `knowledge/mlops/` |
| Neutral Class | Casos arbitrarios/marginales | Añadir clase neutral basada en datos | `knowledge/problem_definition/` |
| Rebalancing | Minoría ignorada | Sampling, weights, bias y métricas | `knowledge/preprocessing/`, `knowledge/metrics/` |

# 11. Trazabilidad

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Propósito y mapa del capítulo | pp. 79-80; resumen pp. 137-138 |
| Reframing | pp. 80-90; Figuras 3-1 a 3-6; Tabla 3-1 |
| Multilabel | pp. 90-99; Figuras 3-7 a 3-10 |
| Ensembles | pp. 99-108; Figuras 3-11 a 3-15; Tabla 3-2 |
| Cascade | pp. 108-117; Figuras 3-16 y 3-17 |
| Neutral Class | pp. 117-122 |
| Rebalancing | pp. 122-137; Figuras 3-18 a 3-21; Tablas 3-3 a 3-7 |
| Reglas, comparación y riesgos globales | Síntesis derivada exclusivamente de pp. 79-138 |
