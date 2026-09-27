---
title: "Working with Text Data"
topics:
  - "Representación de texto para Machine Learning"
  - "Bag of Words, tf-idf, n-grams y topic modeling"
target_knowledge_files:
  - "knowledge/preprocessing/text_vectorization.md"
  - "knowledge/models/text_classification.md"
  - "knowledge/feature_engineering/n_grams_and_normalization.md"
  - "knowledge/validation/text_model_validation.md"
  - "knowledge/models/topic_modeling.md"
---

# 7. Working with Text Data

## 1. Propósito del capítulo

El capítulo explica cómo transformar texto, almacenado normalmente como cadenas de longitud variable, en características numéricas utilizables por modelos de Machine Learning. Presenta un flujo de clasificación supervisada de reseñas de películas y, al final, topic modeling no supervisado. El foco práctico está en representar texto con bag-of-words, ponderar con tf-idf, incorporar contexto mediante n-grams, normalizar tokens y validar modelos lineales sobre matrices dispersas.

El ejemplo principal es IMDb: 25.000 reseñas de entrenamiento y 25.000 de prueba, etiquetadas como positivas o negativas. El capítulo ilustra tanto la construcción de un baseline como la inspección de vocabulario y coeficientes para entender qué aprende el modelo.

## 2. Conceptos esenciales

### 2.1. No toda cadena es texto libre

Antes de vectorizar hay que decidir qué representa una columna de tipo string:

* **Categórica:** valores de una lista fija; se codifica como categoría, no como lenguaje.
* **Cadena libre asignable a categorías:** respuestas manuales con errores, sinónimos o valores raros. Conviene consolidarlas en categorías definidas por el dominio (incluyendo, si hace falta, `other`). El capítulo advierte que esta limpieza suele requerir trabajo manual.
* **Cadena estructurada:** direcciones, nombres, fechas, teléfonos o identificadores. Su tratamiento depende del contexto y no se resuelve automáticamente con el flujo de texto del capítulo.
* **Texto libre:** frases o documentos como tweets, chats o reseñas. Un conjunto de documentos es un *corpus*.

Conclusión práctica derivada: inspeccionar valores únicos, frecuencias y significado antes de aplicar `CountVectorizer`; una columna string no implica por sí sola que deba tokenizarse.

### 2.2. Clasificación de documentos

Cada documento se transforma en un vector de características y se asocia a una etiqueta. En IMDb se trata de clasificación binaria. La representación debe aprenderse con entrenamiento y reutilizarse sin volver a ajustar el vocabulario sobre validación o test.

### 2.3. Bag-of-words

La representación *bag-of-words* descarta orden, párrafos y formato, y conserva conteos de tokens. El procedimiento tiene tres etapas:

1. **Tokenización:** separar cada documento en tokens según una regla de caracteres y límites de palabra.
2. **Construcción del vocabulario:** reunir las palabras observadas y asignarles índices.
3. **Codificación:** contar cuántas veces aparece cada token en cada documento.

El resultado es una matriz documento-característica. En textos reales es muy grande y casi toda entrada es cero; por ello se almacena como matriz dispersa CSR. No se debe convertir a densa salvo en ejemplos pequeños o para una inspección controlada.

### 2.4. Tokenización por defecto

`CountVectorizer` usa por defecto la expresión regular `\\b\\w\\w+\\b`: recoge secuencias de al menos dos letras o números, separadas por límites de palabra, y pasa los tokens a minúsculas. No conserva palabras de una sola letra y divide contracciones como `doesn't`; también puede tratar cadenas alfanuméricas como un único token.

### 2.5. Regularización del vocabulario

`min_df` elimina tokens que aparecen en menos documentos que el umbral indicado. En IMDb, pasar de vocabulario completo a `min_df=5` reduce unas 74.849 características a 27.271 sin cambiar la mejor exactitud de validación (89% en el experimento). El capítulo señala que esto puede acelerar y hacer más interpretable el modelo, además de reducir ruido.

Las palabras no vistas al transformar un nuevo documento se ignoran porque no están en el diccionario de entrenamiento. El capítulo sugiere que, en aplicaciones como spam, puede ser útil una característica adicional que cuente palabras fuera de vocabulario, pero solo si el vocabulario se restringe durante entrenamiento (por ejemplo, con `min_df`).

### 2.6. Stopwords

Se pueden quitar palabras con una lista fija (`stop_words="english"`) o con un umbral de frecuencia (`max_df`). En el experimento, la lista inglesa incorporada elimina 305 características adicionales y reduce ligeramente el rendimiento (de 0,89 a 0,88); por tanto, no es automáticamente beneficiosa. Una lista fija puede ser más útil con conjuntos pequeños, pero siempre debe medirse en validación.

### 2.7. tf-idf

tf-idf asigna más peso a un término frecuente en un documento pero poco frecuente en el corpus. `TfidfTransformer` transforma una matriz de conteos; `TfidfVectorizer` combina extracción y ponderación. El capítulo usa un `Pipeline` porque tf-idf aprende estadísticas del conjunto de entrenamiento. La normalización L2 posterior hace que la longitud del documento no domine la representación.

### 2.8. n-grams

Los unigramas pierden por completo el orden. Los bigramas y trigramas permiten capturar contexto local, por ejemplo la diferencia entre `not worth` y `well worth`. Se controlan con `ngram_range=(mínimo, máximo)`. El beneficio suele venir sobre todo de añadir bigramas; los n-grams largos multiplican el espacio de características, elevan coste y pueden sobreajustar.

### 2.9. Interpretabilidad de modelos lineales

Para clasificación lineal, cada coeficiente se asocia con un token o n-gram. Coeficientes positivos y negativos indican asociación con las clases según la codificación usada. Inspeccionar los extremos ayuda a comprobar que el modelo aprende señales plausibles y a detectar tokens espurios o artefactos del corpus. La importancia tf-idf, en cambio, es no supervisada: un término con tf-idf alto distingue documentos, pero no necesariamente la etiqueta.

### 2.10. Stemming y lemmatization

Ambos son formas de normalización que confluyen variantes relacionadas:

* **Stemming:** heurísticas, normalmente recortando sufijos; puede producir raíces no lingüísticas (`was` → `wa`).
* **Lemmatization:** diccionario y contexto gramatical; puede obtener formas base más interpretables (`was` → `be`, `worse` → `bad`).

El capítulo muestra que la lematización reduce el vocabulario de 27.271 a 21.596 características y mejora modestamente (0,721 a 0,731) cuando solo se usa 1% de los datos para entrenamiento. El efecto depende del corpus; debe probarse, no asumirse.

### 2.11. Topic modeling y LDA

El topic modeling asigna uno o varios temas a cada documento sin supervisión. En `LatentDirichletAllocation` un “tema” es un patrón de palabras que coaparecen y un documento es una mezcla de temas; no tiene por qué coincidir con un tema humano. `components_` contiene la importancia de cada palabra por tema y `transform` devuelve el peso de los temas por documento. Los resultados son aleatorios y deben validarse inspeccionando documentos representativos, no solo las palabras principales.

## 3. Flujo o procedimiento presentado

1. Determinar si cada columna string es categórica, estructurada o texto libre.
2. Cargar los documentos y etiquetas manteniendo una separación explícita entre `train` y `test`.
3. Limpiar formato claramente ajeno al contenido (el ejemplo elimina `<br />` de IMDb).
4. Inspeccionar balance de clases y, cuando proceda, frecuencia de valores.
5. Ajustar `CountVectorizer` únicamente con entrenamiento y transformar entrenamiento/test con el mismo vocabulario.
6. Entrenar un clasificador lineal adecuado para alta dimensionalidad y matrices dispersas.
7. Medir un baseline con validación cruzada; ajustar `C` mediante `GridSearchCV`.
8. Reducir ruido con `min_df`, evaluar stopwords y comparar resultados, tamaño y coste.
9. Probar `TfidfVectorizer` dentro de un `Pipeline` para evitar leakage de estadísticas.
10. Comparar `ngram_range=(1,1)`, `(1,2)` y, si el coste lo permite, `(1,3)`; vigilar el crecimiento del vocabulario.
11. Inspeccionar coeficientes y tokens importantes para interpretar y detectar artefactos.
12. Solo si aporta valor, probar stemming o lemmatization mediante un tokenizer personalizado.
13. Para exploración sin etiquetas, limitar vocabulario/frecuencia, ajustar LDA y verificar cada tema con documentos reales.
14. Seleccionar la configuración por validación y capacidad de generalización; evaluar una sola vez en test.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| String con valores de lista fija | Codificación categórica | La cadena representa una categoría, no lenguaje | Consolidar errores de escritura con el mismo significado |
| Texto libre para clasificación | `CountVectorizer` + modelo lineal | Baseline simple, potente y disperso | Ajustar el vocabulario solo con entrenamiento |
| Muchos tokens raros o errores | `min_df` | Reduce dimensiones y ruido | Puede eliminar señales útiles en corpus pequeño |
| Palabras muy frecuentes sin poder predictivo | `max_df` o lista de stopwords | Disminuye términos dominantes | En IMDb la lista fija empeoró ligeramente; medir siempre |
| La longitud del documento no debe dominar | tf-idf con normalización L2 | Repondera términos distintivos y normaliza longitud | Ajustar dentro de un `Pipeline` |
| El orden local cambia el significado | Bigrams; considerar trigrams | Captura frases como `not worth` | El espacio crece y puede sobreajustar |
| Variantes morfológicas fragmentan datos | Lemmatization o stemming experimental | Confluye formas relacionadas | Requiere tokenizer externo y coste adicional |
| Alta dimensionalidad y matriz dispersa | Regresión logística lineal | Escala bien en el ejemplo | Ajustar regularización `C` y convergencia |
| No hay etiquetas y se buscan grupos | LDA/topic modeling | Aprende coaparición de palabras | Los temas pueden no ser semánticos; inspeccionar documentos |

## 5. Métodos y técnicas

### `CountVectorizer`

* **Objetivo:** convertir documentos en conteos de tokens.
* **Entrada:** iterable de textos; opcionalmente `min_df`, `max_df`, `stop_words`, `ngram_range` y tokenizer.
* **Salida:** matriz CSR de tamaño `(n_documentos, n_características)` y un vocabulario aprendido.
* **Funcionamiento:** tokeniza, construye el diccionario en `fit` y cuenta en `transform`.
* **Hiperparámetros relevantes:** `min_df`, `max_df`, `stop_words`, `ngram_range`, tokenizer.
* **Ventajas:** sencillo, rápido como baseline, interpretable.
* **Limitaciones:** pierde orden con unigramas, genera muchas columnas y no entiende semántica.
* **Adecuado:** clasificación de texto, exploración rápida y modelos lineales.
* **No conviene:** cuando la cadena es una categoría o cuando se necesita comprender estructura lingüística larga.

### Regresión logística sobre texto disperso

* **Objetivo:** clasificación supervisada lineal.
* **Entrada:** matriz dispersa de conteos/tf-idf y etiquetas.
* **Salida:** clase o probabilidades y un coeficiente por característica.
* **Funcionamiento:** aprende una combinación lineal regularizada; el signo y magnitud de cada coeficiente permiten inspección.
* **Hiperparámetro:** `C`, inverso de la fuerza de regularización; el ejemplo selecciona `C=0,1` para conteos y `C=100` para tf-idf con n-grams.
* **Ventajas:** adecuado para muchas características dispersas; baseline fuerte.
* **Limitaciones:** lineal; depende de la representación y puede aprender artefactos.
* **Adecuado:** clasificación de spam, sentimiento u otras tareas con bolsa de palabras.
* **No conviene:** si se requiere dependencia de largo alcance o representación semántica profunda.

### `TfidfVectorizer`

* **Objetivo:** combinar conteos y ponderación por distintividad.
* **Entrada/salida:** textos a matriz dispersa reponderada.
* **Funcionamiento:** multiplica frecuencia en el documento por información inversa de documento y normaliza L2.
* **Hiperparámetros:** los de vectorización (`min_df`, `ngram_range`) y opciones de normalización.
* **Ventajas:** reduce el peso de palabras omnipresentes; mejoró el baseline del ejemplo.
* **Limitaciones:** la importancia es no supervisada y puede resaltar nombres de películas/franquicias en vez de sentimiento.
* **Adecuado:** cuando la frecuencia relativa aporta más que el conteo bruto.
* **No conviene:** asumir que los términos de mayor tf-idf son los más predictivos de la etiqueta.

### n-grams

* **Objetivo:** conservar contexto contiguo.
* **Entrada:** texto tokenizado; **salida:** características para secuencias de longitud indicada.
* **Hiperparámetro:** `ngram_range`.
* **Ventajas:** bigramas mejoran la captura de negación y frases.
* **Limitaciones:** cardinalidad aproximadamente combinatoria, memoria/coste y riesgo de sobreajuste.
* **Adecuado:** probar `(1,2)` como extensión del baseline; `(1,3)` solo con recursos y validación suficientes.

### Stemming y lemmatization

* **Objetivo:** normalizar variantes de una palabra.
* **Entrada:** tokens/documentos; **salida:** tokens raíz o lemas.
* **Ventajas:** menos dimensiones y posible mejora cuando hay pocos datos.
* **Limitaciones:** stemming puede generar raíces artificiales; lemmatization es más costosa y dependiente del idioma/modelo.
* **Adecuado:** experimento de mejora después del baseline.
* **No conviene:** añadirlo bajo presión sin medir coste y validación.

### Latent Dirichlet Allocation

* **Objetivo:** descubrir grupos de palabras y mezcla de temas sin supervisión.
* **Entrada:** matriz de conteos, normalmente con términos muy frecuentes filtrados.
* **Salida:** `components_` (palabra por tema) y representación documento-temas.
* **Hiperparámetros:** número de temas, `learning_method`, `max_iter`, `random_state`; el capítulo usa 10/100 temas, `batch`, 25 iteraciones y semilla 0.
* **Ventajas:** exploración de corpus y representación compacta.
* **Limitaciones:** temas inestables, dependientes del número elegido y no necesariamente interpretables.
* **Adecuado:** análisis exploratorio o usar pesos de temas como representación auxiliar.
* **No conviene:** tratar el nombre de un tema como verdad semántica sin inspeccionar documentos.

## 6. Fórmulas esenciales

El capítulo define el tf-idf usado por `TfidfTransformer` y `TfidfVectorizer`:

\[
\operatorname{tfidf}(w,d)=tf(w,d)\,\log\left(\frac{N+1}{N_w+1}\right)+1
\]

* `w`: término; `d`: documento.
* `tf(w,d)`: número de apariciones de `w` en `d`.
* `N`: número de documentos del conjunto de entrenamiento.
* `N_w`: número de documentos de entrenamiento que contienen `w`.
* Mide una combinación de frecuencia local y rareza en el corpus: sube cuando el término se repite en un documento y baja cuando aparece en muchos documentos.
* Después se aplica normalización L2 por documento, de modo que cada vector tiene norma euclídea 1 y la longitud textual no cambia por sí sola la representación.
* Supuesto operativo: las estadísticas `N` y `N_w` proceden solo del entrenamiento; por eso el vectorizador debe estar dentro del pipeline de validación.

La representación bag-of-words no necesita una fórmula adicional: cada entrada es el conteo del token correspondiente. LDA se describe conceptualmente; el capítulo no exige una fórmula esencial para aplicarlo.

## 7. Implementación práctica

### Baseline con conteos y validación

```python
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

train = load_files("data/aclImdb/train/")
test = load_files("data/aclImdb/test/")
text_train = [doc.replace(b"<br />", b" ") for doc in train.data]
text_test = [doc.replace(b"<br />", b" ") for doc in test.data]

pipe = make_pipeline(
    CountVectorizer(min_df=5, ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000, random_state=0),
)
grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=5)
grid.fit(text_train, train.target)
test_score = grid.score(text_test, test.target)
```

**Nota práctica (adaptación moderna):** el capítulo usa `get_feature_names()` y no siempre fija convergencia; en APIs actuales se recomienda `get_feature_names_out()` para inspección y ajustar `max_iter` si el solver lo requiere. Se conserva la idea original, no se presenta como código literal del libro. El uso de `Pipeline` también evita ajustar transformaciones fuera de cada partición de validación.

### tf-idf con búsqueda de `C`

```python
from sklearn.feature_extraction.text import TfidfVectorizer

pipe = make_pipeline(
    TfidfVectorizer(min_df=5, ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000, random_state=0),
)
grid = GridSearchCV(
    pipe,
    {
        "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
)
grid.fit(text_train, train.target)
```

### LDA exploratorio

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=10_000, max_df=0.15)
X = vectorizer.fit_transform(text_train)
lda = LatentDirichletAllocation(
    n_components=10, learning_method="batch", max_iter=25, random_state=0
)
document_topics = lda.fit_transform(X)
feature_names = vectorizer.get_feature_names_out()  # adaptación moderna
```

La lista de palabras principales de cada tema debe complementarse con la inspección de documentos con mayor peso en ese tema. No convertir la matriz dispersa a densa para un corpus real.

## 8. Supuestos y condiciones de aplicación

* Los documentos son comparables en idioma y convención de tokenización; el ejemplo asume inglés y lenguas con límites de palabra.
* El texto libre contiene señales suficientes para la tarea; un identificador o dirección no se vuelve útil por vectorizarlo como prosa.
* El vocabulario y toda estadística de frecuencia se aprenden solo en entrenamiento.
* Las matrices son potencialmente muy dispersas; memoria y tiempo dependen del vocabulario y de `ngram_range`.
* La etiqueta IMDb es una simplificación binaria de ratings: 6 o más positivo, el resto negativo.
* La interpretación de coeficientes depende de la clase positiva y de la codificación de etiquetas.
* Stopwords, stemming y lemmatization son dependientes del idioma y del corpus.
* LDA supone documentos como mezclas de temas y coaparición informativa; sus temas no garantizan semántica humana.

## 9. Errores y riesgos

* Vectorizar antes de dividir o fuera del pipeline puede filtrar estadísticas del conjunto de validación.
* Ajustar el vectorizador en test cambia el vocabulario y hace inválida la evaluación.
* Convertir matrices grandes a densas puede agotar memoria.
* Usar solo unigramas puede perder negaciones y contexto (`not good`).
* Añadir trigrams sin controlar dimensiones puede aumentar coste y sobreajuste.
* Eliminar stopwords por costumbre puede quitar señal útil; en el experimento bajó ligeramente la exactitud.
* Confundir tf-idf alto con importancia para la etiqueta: tf-idf no usa las clases.
* Tomar como verdad un tema de LDA solo por sus diez palabras principales; verificar documentos y estabilidad ante `random_state`.
* Tratar strings categóricos o estructurados como texto libre produce features sin significado.
* La limpieza incompleta (HTML, errores, codificaciones) introduce tokens espurios.
* La lemmatization puede reducir dimensiones pero también añadir dependencia, latencia y errores del modelo lingüístico.
* Con pocos datos, un vocabulario muy grande permite memorizar errores ortográficos o nombres propios.
* El ejemplo separa train/test, pero la selección real debe considerar estratificación y la métrica apropiada para el balance de clases.

## 10. Aplicación en un coding test

### Síntesis práctica derivada del capítulo

Consultar este capítulo cuando el dataset contiene una columna de texto libre y el objetivo es clasificar, detectar spam, analizar sentimiento o explorar documentos sin etiquetas. Para un baseline rápido: limpiar artefactos evidentes, usar `TfidfVectorizer(min_df=...)` con unigramas/bigramas y una regresión logística lineal dentro de un pipeline; validar `C` con una búsqueda pequeña.

Si el baseline es competitivo, inspeccionar vocabulario y coeficientes antes de complicar el modelo. Probar bigramas cuando hay negaciones o frases; probar `max_df`/stopwords solo si la validación y el tamaño justifican el cambio. Lematización es una mejora secundaria, especialmente defendible con pocos ejemplos; LDA sirve para exploración, no como primera opción para maximizar una métrica supervisada.

No implementar bajo presión de tiempo un tokenizer lingüístico complejo, trigrams extensos, 100 temas de LDA ni conversiones densas sin evidencia de beneficio. Señales del dataset que activan estas técnicas: muchas cadenas libres, vocabulario muy grande, documentos con frases cuyo orden importa, alta proporción de tokens raros, o ausencia de etiquetas con necesidad de descubrir grupos.

## 11. Checklist operativo

* [ ] Confirmar que la columna es texto libre y no categoría o identificador.
* [ ] Separar entrenamiento, validación y test antes de ajustar transformaciones.
* [ ] Limpiar HTML/formatos ajenos al texto y revisar codificación.
* [ ] Medir número de documentos, balance de clases y tamaño del vocabulario.
* [ ] Construir baseline con `CountVectorizer` o `TfidfVectorizer` y matriz dispersa.
* [ ] Usar `Pipeline` para cualquier transformación que aprenda estadísticas.
* [ ] Ajustar `min_df` y `C` con validación cruzada.
* [ ] Comparar unigramas frente a bigramas; vigilar memoria con trigrams.
* [ ] Evaluar stopwords, stemming o lemmatization, sin asumir mejora.
* [ ] Revisar coeficientes y tokens extremos para detectar artefactos.
* [ ] Si no hay etiquetas, limitar vocabulario, ajustar LDA e inspeccionar documentos representativos.
* [ ] Reservar el conjunto de test para la evaluación final.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Tipos de strings y decisión texto/categoría | `knowledge/problem_definition/string_feature_triage.md` | Alta |
| Limpieza de HTML y codificación | `knowledge/preprocessing/text_cleaning.md` | Alta |
| Bag-of-words, vocabulario y matrices dispersas | `knowledge/preprocessing/text_vectorization.md` | Alta |
| tf-idf y normalización | `knowledge/feature_engineering/tf_idf.md` | Alta |
| n-grams y control de dimensión | `knowledge/feature_engineering/n_grams_and_normalization.md` | Alta |
| Validación con pipeline y búsqueda de `C` | `knowledge/validation/text_model_validation.md` | Alta |
| Regresión logística para texto | `knowledge/models/text_classification.md` | Alta |
| Inspección de coeficientes | `knowledge/interpretability/linear_text_models.md` | Media |
| Stemming y lemmatization | `knowledge/preprocessing/text_normalization.md` | Media |
| LDA y document-topic representations | `knowledge/models/topic_modeling.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------ | ------------------------------ |
| Propósito y tipos de strings | pp. 323–325 (PDF pp. 1–3) |
| Carga, limpieza y balance de IMDb | pp. 325–327 (PDF pp. 3–5) |
| Bag-of-words y matrices dispersas | pp. 327–330 (PDF pp. 5–8) |
| Vocabulario, `min_df` y palabras fuera de vocabulario | pp. 330–334 (PDF pp. 8–12) |
| Stopwords | pp. 334–336 (PDF pp. 12–14) |
| tf-idf, fórmula y pipeline | pp. 336–338 (PDF pp. 14–16) |
| Coeficientes e interpretación | pp. 338–339 (PDF pp. 16–17) |
| Bigrams, trigrams y búsqueda de parámetros | pp. 339–343 (PDF pp. 17–21) |
| Stemming y lemmatization | pp. 344–347 (PDF pp. 22–25) |
| Topic modeling y LDA | pp. 347–354 (PDF pp. 25–32) |
| Límites, resumen y perspectivas | pp. 354–356 (PDF pp. 32–34) |

No se identificaron páginas ilegibles ni secciones imposibles de procesar. Las adaptaciones de API están marcadas explícitamente como notas prácticas y no se presentan como contenido original del capítulo.
