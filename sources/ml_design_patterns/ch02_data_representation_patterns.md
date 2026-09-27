---
title: "Data Representation Design Patterns"
book: "Machine Learning Design Patterns"
chapter: "2"
topics:
  - "Data representation"
  - "Categorical encoding"
  - "Dimensionality reduction"
  - "Feature engineering"
  - "Multimodal learning"
patterns:
  - "Hashed Feature"
  - "Embeddings"
  - "Feature Cross"
  - "Multimodal Input"
target_knowledge_files:
  - "knowledge/preprocessing/hashed_features.md"
  - "knowledge/feature_engineering/embeddings.md"
  - "knowledge/feature_engineering/feature_crosses.md"
  - "knowledge/models/multimodal_input.md"
---

# 2. Data Representation Design Patterns

## 1. Propósito del capítulo

El capítulo aborda cómo convertir inputs del mundo real en representaciones numéricas compatibles con la función matemática de un modelo. El problema aparece durante preprocessing y feature engineering, pero sus decisiones afectan training, tamaño del modelo, serving, generalización e interpretabilidad.

Los autores distinguen entre *input* (dato real) y *feature* (representación transformada que consume el modelo). Primero presentan idioms para números, arrays y categorías; después desarrollan cuatro patterns numerados para vocabularios incompletos, alta cardinalidad, interacciones entre variables y combinación de modalidades.

## 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Hashed Feature | Vocabulario incompleto, alta cardinalidad y cold start en variables categóricas. | Data representation | Alta |
| Embeddings | Representar datos de alta dimensión preservando relaciones relevantes para la tarea. | Feature engineering | Alta |
| Feature Cross | Hacer explícitas interacciones que un modelo simple no aprende con facilidad. | Feature engineering | Alta |
| Multimodal Input | Combinar tipos de datos o varias representaciones de un mismo dato. | Model architecture | Media |

### 2.1 Idioms previos que no constituyen patterns numerados

El capítulo presenta escalado lineal, clipping, Z-score, winsorizing, transformaciones no lineales, histogram equalization, Box-Cox, resúmenes de arrays, one-hot encoding y multi-hot encoding como soluciones comunes. Son contexto técnico para los cuatro patterns, no unidades independientes del catálogo.

# 3. Patrones

## Pattern 1: Hashed Feature

### 3.1 Problema

El one-hot encoding exige conocer de antemano todo el vocabulario. En categorías como `hospital_id`, `physician_id` o aeropuertos, el training sample puede omitir valores válidos, el vector puede tener miles o millones de posiciones y producción puede recibir categorías nuevas. Mantener vocabulario y pesos también puede volver el modelo demasiado grande para serving.

### 3.2 Contexto de aplicación

Variables categóricas de alta cardinalidad, vocabulario abierto o incompleto, modelos con límites de memoria y sistemas con cold start. El ejemplo del capítulo es el aeropuerto de salida para predecir retrasos.

### 3.3 Señales que indican que el patrón puede ser necesario

- Aparecen categorías válidas fuera del vocabulario de training.
- El one-hot produce una matriz muy ancha y dispersa.
- Guardar vocabulario y pesos dificulta el despliegue.
- Se esperan entidades nuevas después del lanzamiento.
- **Inferencia práctica:** el identificador es útil, pero su representación exacta consume más capacidad que la disponible.

### 3.4 Solución propuesta

1. Convertir cada categoría en una cadena única.
2. Aplicar una función fingerprint determinista, portable y sin seeds ni salt.
3. Aplicar módulo por un número prefijado de buckets.
4. Tomar el valor absoluto después del módulo.
5. Entrenar con el bucket como representación categórica.

### 3.5 Cómo funciona

Todas las categorías, incluidas las no vistas, se proyectan al rango fijo `[0, num_buckets-1]`. Varias categorías pueden compartir bucket. El modelo aprende un peso por bucket y una categoría nueva hereda inicialmente el comportamiento agregado de las categorías que colisionan con ella.

### 3.6 Por qué funciona

La transformación no depende de un vocabulario almacenado y produce el mismo resultado en training y serving. Limitar los buckets acota tamaño y memoria. Las colisiones permiten una predicción razonable para valores no vistos, aunque sacrifican especificidad y accuracy.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Categoría expresable como cadena. |
| Transformación | Fingerprint determinista, módulo y valor absoluto. |
| Salida | Índice de bucket dentro de un rango fijo. |

### 3.8 Procedimiento operativo

1. Confirmar que vocabulario, cardinalidad o cold start justifican pérdida de información.
2. Elegir una función fingerprint disponible en training y serving.
3. Definir un rango sensato de `num_buckets` respecto de la cardinalidad.
4. Calcular `ABS(MOD(fingerprint(value), num_buckets))` en ese orden.
5. Medir colisiones, skew y calidad por categorías frecuentes y raras.
6. Ajustar `num_buckets` como hiperparámetro.
7. Considerar L2 si pudiera haber buckets vacíos.
8. Reentrenar periódicamente para incorporar el comportamiento de entidades nuevas.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| `num_buckets` pequeño | Modelo compacto y mayor intercambio estadístico. | Más colisiones y pérdida de accuracy. |
| `num_buckets` grande | Menos categorías por bucket. | No elimina la probabilidad de colisión y aumenta tamaño. |
| Fingerprint | Garantiza consistencia training-serving. | Una función no portable rompe la representación. |
| Regularización L2 | Lleva pesos de buckets vacíos cerca de cero. | Añade una decisión de tuning. |

El libro propone como regla orientativa unos cinco valores por bucket y recomienda tunear el número dentro de un rango sensato.

### 3.10 Ventajas

- Maneja valores out-of-vocabulary sin errores de serving.
- Acota dimensionalidad, memoria y tamaño del modelo.
- No requiere almacenar el vocabulario.
- Ofrece un comportamiento inicial para categorías nuevas.
- Es compatible con reentrenamiento periódico.

### 3.11 Limitaciones

- Es una representación con pérdida.
- No preserva identidad individual cuando hay colisiones.
- No garantiza ausencia de colisiones ni con muchos buckets.
- El rendimiento puede ser malo para categorías raras que colisionan con una categoría dominante.

### 3.12 Riesgos y trade-offs

El trade-off central es robustez y tamaño frente a precisión. Con distribución sesgada, una categoría frecuente puede imponer su señal a otras del mismo bucket. Un aggregate feature calculado con todo el dataset introduciría leakage; debe derivarse solo de training. Aplicar una función distinta en serving causa training-serving skew. El orden incorrecto `ABS(fingerprint)` antes de `MOD` puede provocar overflow en el mínimo `INT64`.

### 3.13 Cuándo utilizarlo

Cuando el vocabulario no es cerrado, la cardinalidad hace inviable one-hot o se esperan nuevas categorías y se aceptan colisiones.

### 3.14 Cuándo no utilizarlo

Cuando el vocabulario es conocido y relativamente pequeño, cold start no importa y existe suficiente data para aprender una representación exacta. El libro señala que miles de categorías pueden ser aceptables con millones de ejemplos.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| One-hot con vocabulario fijo | Conserva identidad y evita colisiones. | No resuelve vocabulario abierto; aumenta dimensión. | Vocabulario pequeño, conocido y estable. |
| Aggregate feature por categoría | Conserva señal específica ante skew. | Requiere cálculo consistente y cuidado con leakage. | Colisiones frecuentes o distribución sesgada. |
| Binary encoding | Reduce dimensión. | No resuelve OOV ni cold start y puede crear correlaciones espurias. | El capítulo no la recomienda como sustituto general. |
| Descartar la variable | Elimina coste. | Pierde toda su señal. | Solo si la variable no aporta valor. |

### 3.16 Implementación práctica

```sql
ABS(MOD(FARM_FINGERPRINT(category), num_buckets))
```

Debe usarse una función fingerprint, no un hash criptográfico con salt o coste variable.

**Nota práctica de actualización:** las APIs de `tf.feature_column` mostradas por el libro pueden ser legacy en stacks actuales. La lógica que debe conservarse es una capa de hashing determinista idéntica en training y serving.

### 3.17 Ejemplo mínimo conceptual

Aeropuertos nuevos o raros → hash a un conjunto fijo de buckets → el modelo puede inferir sin ampliar vocabulario, aceptando que comparta pesos con otros aeropuertos.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En un coding test o datathon, es útil para IDs y categorías de alta cardinalidad cuando train y test no comparten exactamente vocabulario. Es rápida de implementar y puede hacer viable un baseline. Conviene comparar varios tamaños de bucket y revisar si categorías muy frecuentes dominan los buckets; no debe asumirse una mejora automática frente a one-hot.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar alta cardinalidad, OOV o cold start.
- [ ] Usar fingerprint determinista y portable.
- [ ] Aplicar módulo antes de valor absoluto.
- [ ] Tunear `num_buckets` en un rango sensato.
- [ ] Medir colisiones y skew.
- [ ] Calcular aggregates solo desde training.
- [ ] Reutilizar exactamente la transformación en serving.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Embeddings | Puede combinarse | Un bucket o categoría puede proyectarse a un espacio denso. |
| Feature Cross | Dependencia técnica posible | Implementaciones de crosses pueden hashear combinaciones. |

## Pattern 2: Embeddings

### 3.1 Problema

One-hot genera representaciones dispersas y de alta dimensión y trata categorías distintas como independientes. Esto resulta inadecuado para millones de usuarios, miles de ítems, grandes vocabularios, imágenes o audio, donde existen relaciones de similitud relevantes.

### 3.2 Contexto de aplicación

Categorías de alta cardinalidad, recomendación, texto, imágenes, audio y cualquier input cuya estructura relevante pueda aprenderse en un espacio denso menor. Puede entrenarse de forma supervisada, mediante tareas auxiliares o reutilizarse preentrenado.

### 3.3 Señales que indican que el patrón puede ser necesario

- El one-hot es extremadamente ancho y disperso.
- Categorías distintas tienen relaciones que el one-hot oculta.
- El modelo debe recuperar ítems o usuarios similares.
- Hay pocos labels, pero abundan datos sin etiquetar o embeddings preentrenados.
- **Inferencia práctica:** el feature engineering manual no captura bien la estructura semántica.

### 3.4 Solución propuesta

1. Mapear cada valor a un índice.
2. Insertar una embedding layer de dimensión elegida.
3. Aprender sus pesos junto con la tarea principal, o mediante una tarea auxiliar.
4. Usar el vector denso como input del resto del modelo.
5. Cuando proceda, reutilizar embeddings preentrenados mediante transfer learning.

### 3.5 Cómo funciona

La embedding layer es una capa oculta con pesos entrenables. Proyecta el input de alta cardinalidad a un vector real de menor dimensión. Gradient descent ajusta esa proyección según la loss de la tarea. Para texto se tokeniza y se convierten secuencias a índices; para imágenes puede retirarse el clasificador final de una red entrenada y usar su vector interno.

### 3.6 Por qué funciona

La optimización aproxima valores con efectos similares sobre el objetivo. Así, el espacio denso conserva información útil para la tarea y permite cuantificar similitud, a diferencia de vectores one-hot ortogonales. Menor dimensión reduce pesos y puede facilitar generalización.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Índice categórico, secuencia tokenizada o señal de alta dimensión. |
| Transformación | Proyección aprendida por una embedding layer o encoder. |
| Salida | Vector denso de dimensión seleccionada. |

### 3.8 Procedimiento operativo

1. Definir el vocabulario o mecanismo de indexación.
2. Tratar padding y longitudes cuando haya secuencias.
3. Elegir dimensión inicial y rango de búsqueda.
4. Integrar la embedding layer con la loss relevante.
5. Entrenar y validar la tarea principal.
6. Inspeccionar desempeño y, si aporta valor, similitudes aprendidas.
7. Con pocos labels, evaluar autoencoder, language model o embedding preentrenado.
8. Persistir la indexación y la proyección para inferencia.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Dimensión pequeña | Compresión fuerte y menos parámetros. | Pierde contexto e información. |
| Dimensión grande | Mayor capacidad. | Se aproxima al problema de one-hot y debilita la compresión. |
| Entrenamiento supervisado | Alinea representación con la tarea. | Requiere muchos labels. |
| Pretrained/auxiliary | Aprovecha datos externos a la tarea o sin labels dentro del dataset disponible. | Puede preservar contexto no alineado con el objetivo. |

El capítulo recomienda experimentar o tunear la dimensión y ofrece dos heurísticas discrepantes como puntos de partida, no como valores garantizados.

### 3.10 Ventajas

- Reduce dimensionalidad y dispersión.
- Aprende relaciones de cercanía relevantes para el objetivo.
- Reduce feature engineering manual.
- Permite búsquedas por similitud y transfer learning.
- Autoencoders permiten usar datos sin labels.

### 3.11 Limitaciones

- La compresión pierde información.
- Necesita elegir dimensionalidad.
- Entrenarlo de forma supervisada puede exigir gran volumen etiquetado.
- El embedding refleja la tarea con que se entrenó, no una semántica universal.

### 3.12 Riesgos y trade-offs

Dimensión demasiado pequeña causa underfitting representacional; demasiado grande aumenta parámetros y puede erosionar generalización. Tokenización, padding y truncation deben ser idénticos en inferencia. Promediar word vectors elimina orden. Precomputar embeddings reduce coste posterior, pero exige mantener su versión y consistencia.

### 3.13 Cuándo utilizarlo

Con alta cardinalidad, texto, imagen o audio, y cuando la similitud entre valores sea útil para la tarea.

### 3.14 Cuándo no utilizarlo

Cuando una representación simple y pequeña basta, no hay data suficiente para aprender pesos útiles o la pérdida de información no es aceptable.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| One-hot | Exacto e interpretable por categoría. | Disperso y sin noción de similitud. | Cardinalidad baja. |
| PCA | Reducción lineal. | No aprende reducción no lineal ni se alinea necesariamente con la tarea. | Estructura aproximadamente lineal. |
| Clustering | Segmentos explícitos. | Requiere etapa separada. | Cuando se necesitan grupos discretos. |
| Autoencoder | No necesita labels adicionales. | Optimiza reconstrucción, no directamente la tarea final. | Muchos datos sin etiquetar. |
| Pretrained embedding | Ahorra entrenamiento. | Contexto de pretraining puede no ajustarse. | Pocos labels y dominio compatible. |

### 3.16 Implementación práctica

```python
embedding = Embedding(input_dim=vocab_size + 1, output_dim=embed_dim)
```

Para secuencias, el capítulo tokeniza, aplica padding y después agrega los vectores antes del clasificador.

**Nota práctica de actualización:** las utilidades concretas de tokenización y `tf.feature_column` del libro pueden ser legacy. Debe preservarse la secuencia conceptual: vocabulario/indexación → padding consistente → embedding entrenable.

### 3.17 Ejemplo mínimo conceptual

Millones de clientes one-hot → embedding de baja dimensión aprendido con historial de visualización → clientes y videos con comportamiento parecido quedan próximos y el modelo generaliza con menos pesos.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es valioso para texto, IDs masivos y categorías con estructura latente. Bajo presión, un embedding preentrenado o pequeño puede superar una matriz one-hot inviable; entrenar uno complejo desde cero puede exceder el presupuesto. Debe compararse contra BOW u otro baseline simple y tunearse la dimensión sin sobreajustar validation.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar alta cardinalidad o relaciones entre categorías.
- [ ] Congelar vocabulario, indexación y padding.
- [ ] Definir dimensión y rango de tuning.
- [ ] Comparar contra one-hot o BOW.
- [ ] Evaluar data etiquetada disponible.
- [ ] Considerar autoencoder o pretrained embedding.
- [ ] Persistir versión y preprocessing para serving.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Feature Cross | Complementario | Reduce la dispersión de crosses de alta cardinalidad. |
| Multimodal Input | Dependencia frecuente | Convierte texto o imagen antes de concatenarlo con datos tabulares. |
| Hashed Feature | Alternativo / combinable | Ambos controlan cardinalidad, con mecanismos y pérdidas diferentes. |

## Pattern 3: Feature Cross

### 3.1 Problema

Las features individuales pueden no hacer linealmente separable el objetivo. Aumentar capas o complejidad permite aprender interacciones, pero encarece training y requiere más datos.

### 3.2 Contexto de aplicación

Interacciones plausibles entre dos o más variables categóricas; modelos lineales sobre datasets grandes; variables numéricas previamente bucketizadas; ejemplos del capítulo: día×hora, sexo×pluralidad y coordenadas discretizadas.

### 3.3 Señales que indican que el patrón puede ser necesario

- El efecto conjunto de dos variables difiere de sus efectos aislados.
- Un modelo lineal falla en una frontera que puede expresarse por regiones.
- Un DNN aprende la relación, pero su coste es excesivo.
- **Inferencia práctica:** el dominio sugiere combinaciones con significado, como rush hour por día.

### 3.4 Solución propuesta

1. Elegir features categóricas cuya interacción sea relevante.
2. Bucketizar primero cualquier feature continua.
3. Formar el producto cartesiano o concatenación ordenada.
4. Codificar cada combinación como feature separada.
5. Usar indicador, hashing o embedding según cardinalidad.
6. Regularizar si la expansión genera buckets con pocos ejemplos.

### 3.5 Cómo funciona

Para features con `m` y `n` valores, el cross crea hasta `m*n` combinaciones. Cada combinación obtiene un peso propio. Esto introduce no linealidad explícita y permite que un modelo lineal trate cada región o combinación de manera diferente.

### 3.6 Por qué funciona

Traslada complejidad desde la arquitectura hacia la representación. La interacción ya está disponible como señal directa, por lo que el optimizador no tiene que descubrirla a través de capas. En el ejemplo del capítulo, un modelo lineal con cross alcanzó error comparable y training mucho más rápido que un DNN sin cross.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Dos o más features categóricas o numéricas bucketizadas. |
| Transformación | Producto cartesiano/concatenación, opcionalmente hashing o embedding. |
| Salida | Vector categórico disperso o representación densa del cross. |

### 3.8 Procedimiento operativo

1. Formular una hipótesis de interacción.
2. Verificar que las variables no sean redundantes o altamente correlacionadas.
3. Bucketizar continuas con resolución suficiente, pero no memorística.
4. Estimar cardinalidad del producto.
5. Crear cross y escoger indicator o embedding.
6. Entrenar contra baseline sin cross.
7. Aplicar L1 o L2 cuando la cardinalidad sea alta.
8. Validar tiempo, error y estabilidad fuera de training.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Features a cruzar | Define la interacción disponible. | Cross irrelevante añade ruido. |
| Límites de buckets | Controlan resolución de numéricas. | Demasiado finos memorizan; gruesos pierden señal. |
| Hash bucket size | Acota representación. | Colisiones. |
| Indicator vs embedding | Exactitud dispersa vs compresión densa. | Tamaño o pérdida de información. |
| L1/L2 | Reduce ruido u overfitting. | Exceso de regularización elimina señal. |

### 3.10 Ventajas

- Añade expresividad a modelos simples.
- Hace explícitas interacciones de dominio.
- Puede reducir tiempo y coste frente a una red más profunda.
- Escala a grandes datasets.

### 3.11 Limitaciones

- La cardinalidad crece multiplicativamente.
- No debe aplicarse directamente a continuas.
- Exige hipótesis útil; features correlacionadas aportan poco.
- Puede crear muchos buckets con pocos ejemplos.

### 3.12 Riesgos y trade-offs

La expansión produce sparsity, ruido y overfitting. Buckets espaciales demasiado finos permiten memorizar ubicaciones. Hashing introduce colisiones. Una granularidad elegida mirando el conjunto de evaluación puede sobreajustar validation. El cross debe transformarse de igual modo durante serving.

### 3.13 Cuándo utilizarlo

Cuando existe una interacción plausible y se dispone de suficiente data por combinación, especialmente para mantener un modelo lineal rápido.

### 3.14 Cuándo no utilizarlo

No cruzar continuas sin bucketizar, features altamente correlacionadas, combinaciones sin soporte o crosses masivos que conviertan cada ejemplo en casi único.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| DNN o árbol | Puede aprender interacciones automáticamente. | Más coste, datos o complejidad. | Muchas interacciones desconocidas. |
| Cross + embedding | Reduce sparsity y aprende cercanía. | Añade pérdida y tuning. | Cross de alta cardinalidad. |
| Cross + hashing | Acota tamaño. | Colisiones. | Cardinalidad demasiado alta para indicador. |

### 3.16 Implementación práctica

```python
cross = crossed_column([feature_a, feature_b], hash_bucket_size=n)
crossed_feature = embedding_column(cross, dimension=d)
```

**Nota práctica de actualización:** `tf.feature_column` puede ser legacy. La implementación moderna debe conservar bucketización, producto cartesiano, representación y regularización dentro del pipeline reproducible.

### 3.17 Ejemplo mínimo conceptual

Día y hora por separado → cross de 7×24 combinaciones → el modelo asigna un peso propio a “viernes 17:00” y aprende rush hour con una arquitectura lineal.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es una mejora rápida para tabular data cuando EDA revela interacciones obvias. Puede elevar un baseline lineal sin entrenar un modelo profundo. Bajo tiempo limitado conviene probar pocos crosses justificables, medir cardinalidad y regularizar; generar combinaciones indiscriminadas suele consumir tiempo y sobreajustar.

### 3.19 Checklist operativo del patrón

- [ ] Formular interacción con sentido.
- [ ] Bucketizar continuas antes de cruzar.
- [ ] Calcular cardinalidad y soporte por combinación.
- [ ] Evitar features altamente correlacionadas.
- [ ] Elegir indicator, hashing o embedding.
- [ ] Aplicar regularización si corresponde.
- [ ] Comparar contra baseline sin cross.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Embeddings | Complementario | Comprime crosses dispersos y permite generalizar entre combinaciones. |
| Hashed Feature | Puede combinarse | Acota el espacio de combinaciones mediante buckets. |
| Multimodal Input | Complementario | Los crosses pueden formar parte de la rama tabular. |

## Pattern 4: Multimodal Input

### 3.1 Problema

Los modelos estándar suelen aceptar un tipo de input, pero una predicción puede necesitar imagen, texto y metadata a la vez. Además, una sola representación puede descartar señal: embeddings pierden longitud o puntuación; pooling puede perder detalle; un número puede tener tanto significado continuo como categórico.

### 3.2 Contexto de aplicación

Texto más datos tabulares, imagen más metadata, o una misma variable representada de varias maneras. Requiere una arquitectura con ramas especializadas y una operación de combinación.

### 3.3 Señales que indican que el patrón puede ser necesario

- El dataset combina texto/imágenes con metadata útil.
- Una representación elimina información relevante.
- BOW y embeddings capturan señales distintas.
- Pixels globales y ventanas locales aportan perspectivas complementarias.
- **Inferencia práctica:** errores del modelo se explican por contexto disponible que todavía no entra al modelo.

### 3.4 Solución propuesta

1. Identificar modalidades o representaciones complementarias.
2. Preprocesar cada una con una rama adecuada.
3. Reducir cada rama a una representación compatible.
4. Concatenar las ramas.
5. Alimentar la representación conjunta a capas compartidas y al output.
6. Comparar contra modelos unimodales.

### 3.5 Cómo funciona

Cada input conserva su estructura: embeddings para texto, convoluciones para imágenes y Dense/encodings para tabular data. La concatenación crea un vector conjunto sobre el que capas posteriores aprenden relaciones. También puede concatenarse más de una representación del mismo input.

### 3.6 Por qué funciona

Las representaciones preservan señales complementarias. BOW destaca palabras significativas, embeddings capturan contexto; pixels reflejan focos globales y tiles detectan bordes; un valor numérico conserva granularidad mientras su bucket aporta una separación no lineal.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Dos o más modalidades o representaciones. |
| Transformación | Ramas especializadas y concatenación. |
| Salida | Representación conjunta para una predicción única. |

### 3.8 Procedimiento operativo

1. Inventariar modalidades y metadata disponibles en inferencia.
2. Definir preprocessing por rama.
3. Establecer shapes y mecanismos de reducción.
4. Concatenar outputs intermedios.
5. Entrenar capas compartidas.
6. Hacer ablation: cada modalidad sola y combinación.
7. Revisar coste, latencia e interpretabilidad.
8. Garantizar que todas las ramas estén disponibles en serving.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Modalidades incluidas | Amplía señal. | Una rama espuria puede dominar. |
| Dimensión por rama | Equilibra aportes. | Una representación muy grande domina optimización. |
| Vocabulario BOW | Controla cobertura y sparsity. | Muy pequeño omite señal; muy grande añade ceros. |
| Kernel/stride/pooling | Controla detalle local en imagen. | Pooling o stride excesivos pierden información. |
| Buckets tabulares | Añaden vista categórica. | Umbrales arbitrarios crean discontinuidades. |

### 3.10 Ventajas

- Usa contexto que un modelo unimodal ignora.
- Recupera información perdida por una representación.
- Permite combinar señales globales y locales.
- Admite texto, imagen y metadata en un modelo único.

### 3.11 Limitaciones

- Aumenta complejidad de arquitectura y preprocessing.
- Exige disponibilidad sincronizada de todas las ramas.
- Puede aumentar coste, latencia y necesidad de data.
- Las features se vuelven dependientes y más difíciles de explicar.

### 3.12 Riesgos y trade-offs

Más señal implica más capacidad y riesgo de overfitting. Una modalidad puede contener atajos espurios, como anotaciones en imágenes de laboratorio. La combinación dificulta atribuir una predicción. Diferencias de preprocessing o ausencia de metadata en producción generan training-serving skew.

### 3.13 Cuándo utilizarlo

Cuando las modalidades están disponibles en training y serving, aportan señales complementarias y la mejora justifica la complejidad.

### 3.14 Cuándo no utilizarlo

Cuando una representación simple es suficiente, una modalidad no estará disponible en inferencia, el dataset es pequeño para la capacidad añadida o la explicación clara es una restricción dominante.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Modelo unimodal | Simple, rápido y más fácil de operar. | Descarta contexto. | La modalidad principal basta. |
| BOW con modelo simple | Baseline rápido. | Ignora orden y significado. | Prototipado de texto. |
| Embedding solo | Captura relaciones semánticas. | Puede perder longitud, puntuación o señales explícitas. | Contexto semántico dominante. |
| Pixels o CNN solos | Menos ramas. | Pierde perspectiva local o global complementaria. | Imágenes simples o presupuesto limitado. |

### 3.16 Implementación práctica

```python
merged = keras.layers.concatenate([text_branch, tabular_branch])
hidden = Dense(16, activation="relu")(merged)
output = Dense(1)(hidden)
model = Model(inputs=[text_input, tabular_input], outputs=output)
```

La idea mínima es mantener preprocessing especializado por rama y concatenar sus representaciones, no copiar la arquitectura completa del ejemplo.

### 3.17 Ejemplo mínimo conceptual

Review text más precio y tipo de comida → embedding para texto y encoding tabular para metadata → concatenación → una predicción que utiliza contenido y contexto.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es útil cuando el dataset ofrece texto o imagen junto a metadata. Bajo presión conviene comenzar con cada rama por separado y combinar solo si ambas validan señal. BOW+tabular puede ser un baseline rápido; embeddings/CNN+metadata requieren más tiempo. Ablations evitan atribuir una mejora a una rama que solo explota un atajo.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar disponibilidad de cada modalidad en inferencia.
- [ ] Definir preprocessing y shape por rama.
- [ ] Construir baselines unimodales.
- [ ] Concatenar representaciones compatibles.
- [ ] Ejecutar ablations.
- [ ] Revisar atajos espurios e interpretabilidad.
- [ ] Medir coste y latencia de serving.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Embeddings | Dependencia frecuente | Representa texto o imagen antes de combinarlo. |
| Feature Cross | Puede combinarse | La rama tabular puede contener interacciones explícitas. |

# 4. Reglas de decisión del capítulo

| Situación | Pattern recomendado | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Vocabulario abierto o IDs nuevos | Hashed Feature | Produce rango fijo sin vocabulario exhaustivo. | Medir colisiones y skew. |
| Categorías masivas con relaciones | Embeddings | Aprende representación densa y similitud. | Tunear dimensión y asegurar data suficiente. |
| Interacción categórica conocida | Feature Cross | Hace explícita la relación para un modelo simple. | Controlar cardinalidad y regularizar. |
| Texto o imagen más metadata | Multimodal Input | Integra señales complementarias. | Validar disponibilidad en serving. |
| Variable continua muy sesgada | Idiom de transformación, no pattern numerado | Log, Box-Cox o histogram equalization pueden mejorar distribución antes de scaling. | Estimar parámetros solo con training. |
| Categoría pequeña y estable | One-hot idiom | Conserva identidad sin colisiones. | Reservar OOV si puede aparecer. |

# 5. Comparación entre patrones del capítulo

| Pattern | Problema que resuelve | Coste | Beneficio | Complejidad |
| ------- | --------------------- | ----- | --------- | ----------- |
| Hashed Feature | Vocabulario/cardinalidad/cold start. | Bajo | Representación acotada y robusta a OOV. | Baja |
| Embeddings | Alta dimensión y falta de similitud. | Medio-alto | Compresión aprendida y relaciones densas. | Media-alta |
| Feature Cross | Interacciones difíciles para modelos simples. | Bajo-medio | Más expresividad y training rápido. | Media |
| Multimodal Input | Señales heterogéneas o complementarias. | Alto | Mayor cobertura informativa. | Alta |

# 6. Riesgos globales del capítulo

- **Data leakage:** estadísticas, límites, vocabularios, aggregates o transform parameters deben aprenderse únicamente con training.
- **Distribución incorrecta:** scaling o transform inadecuados pueden comprimir la región útil; hashing bajo skew mezcla categorías incompatibles.
- **Overfitting:** crosses demasiado finos y arquitecturas multimodales con mucha capacidad pueden memorizar.
- **Training-serving skew:** hashing, scaling, tokenización, padding, bucketización y concatenación deben ser idénticos en inferencia.
- **Complejidad operacional:** embeddings y múltiples ramas requieren versionar más artefactos.
- **Interpretabilidad:** embeddings y dependencias entre modalidades dificultan explicar predicciones.
- **Coste computacional:** mayor dimensión, redes profundas y múltiples representaciones aumentan training y serving.

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

Inspeccionar tipos, cardinalidad, distribución, OOV entre splits, skew numérico y presencia de texto, imagen o metadata. Identificar interacciones plausibles y qué inputs estarán disponibles en inferencia.

### Durante el baseline

Usar scaling coherente para numéricas, one-hot para categorías pequeñas, hashing para cardinalidad alta y BOW para texto cuando se necesite rapidez. Estos idioms permiten decidir si la tarea tiene señal antes de una arquitectura compleja.

### Durante la mejora del modelo

Probar embeddings cuando one-hot sea disperso, crosses con hipótesis claras y Multimodal Input solo después de demostrar valor en cada rama. Comparar siempre con el baseline.

### Durante validación

Ajustar transformaciones solo con training; verificar OOV, colisiones, soporte de crosses y ablations multimodales. Evitar seleccionar buckets o dimensiones por repetidas mejoras accidentales en validation.

### Bajo presión de tiempo

- **Alta prioridad:** representación correcta, scaling, one-hot/hash y crosses pequeños justificables.
- **Prioridad media:** embeddings compactos o preentrenados y BOW+metadata.
- **Baja prioridad:** multimodal profundo desde cero o búsquedas amplias de arquitecturas sin baseline estable.

# 8. Checklist global del capítulo

- [ ] Separar input real de feature consumida por el modelo.
- [ ] Ajustar parámetros de representación solo con training.
- [ ] Comprobar cardinalidad, OOV y skew.
- [ ] Elegir la representación más simple que conserve señal.
- [ ] Mantener preprocessing idéntico en training y serving.
- [ ] Comparar cada pattern contra un baseline.
- [ ] Medir coste, tamaño, latencia e interpretabilidad.

# 9. Mapeo hacia `knowledge/`

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Hashed Feature | `knowledge/preprocessing/hashed_features.md` | Alta |
| Embeddings | `knowledge/feature_engineering/embeddings.md` | Alta |
| Feature Cross | `knowledge/feature_engineering/feature_crosses.md` | Alta |
| Multimodal Input | `knowledge/models/multimodal_input.md` | Media |
| Scaling y nonlinear transformations | `knowledge/preprocessing/numerical_scaling.md` | Alta |
| Categorical and array idioms | `knowledge/preprocessing/categorical_and_array_representations.md` | Media |

# 10. Índice de patrones reutilizables

| Pattern | Activador principal | Acción principal | Knowledge destino |
| ------- | ------------------- | ---------------- | ----------------- |
| Hashed Feature | Alta cardinalidad, OOV o cold start. | Hashear a buckets fijos. | `knowledge/preprocessing/hashed_features.md` |
| Embeddings | Representación dispersa y relaciones latentes. | Aprender proyección densa. | `knowledge/feature_engineering/embeddings.md` |
| Feature Cross | Interacción explícita entre variables. | Crear producto categórico. | `knowledge/feature_engineering/feature_crosses.md` |
| Multimodal Input | Modalidades o vistas complementarias. | Procesar por ramas y concatenar. | `knowledge/models/multimodal_input.md` |

# 11. Trazabilidad

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Propósito y conceptos de representación | pp. 19-21; Figures 2-1 y 2-2 |
| Idioms: numerical inputs y scaling | pp. 22-25; Figure 2-3 |
| Idioms: nonlinear transformations | pp. 26-27; Figure 2-4 |
| Idioms: arrays y categorical inputs | pp. 27-31 |
| Pattern 1: Hashed Feature | pp. 32-39; Tables 2-1 y 2-2 |
| Pattern 2: Embeddings | pp. 39-52; Tables 2-3 a 2-7; Figures 2-5 a 2-13 |
| Pattern 3: Feature Cross | pp. 52-62; Tables 2-8 a 2-10; Figures 2-14 a 2-18 |
| Pattern 4: Multimodal Input | pp. 62-77; Figures 2-19 a 2-25 |
| Resumen del capítulo | p. 77 |

**Cobertura del archivo fuente:** se revisaron las 60 páginas del PDF suministrado. Contienen las páginas impresas 19-77 del capítulo; la última página del archivo está en blanco. No se detectaron páginas, figuras, tablas o fragmentos imposibles de procesar para esta extracción.
