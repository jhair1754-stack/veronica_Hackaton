---
title: "Feature engineering para Machine Learning"
knowledge_area: "feature_engineering"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Feature engineering para Machine Learning

## 1. Propósito y alcance

La ingeniería de variables modifica o construye la representación que recibe un modelo para que los datos sean utilizables o expongan mejor una señal relevante. La representación adecuada depende del significado de la variable, del momento de predicción y de la familia de modelos: una codificación que ayuda a un modelo lineal puede ser innecesaria para un árbol, y una transformación técnicamente válida puede ser inútil si no mejora una evaluación fuera de muestra. Esta definición integra el énfasis en representación de [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md), el flujo de proyecto de [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) y la distinción entre *input* y *feature* de [ML Design Patterns, cap. 2](../../sources/ml_design_patterns/ch02_data_representation_patterns.md). **Tipo de contenido: síntesis integrada.**

Conviene separar cuatro tareas:

- **Limpiar o corregir datos:** resolver errores, formatos, valores ausentes o categorías inconsistentes. Es una condición previa de calidad, pero no equivale por sí sola a diseñar una representación. Se remite a `knowledge/preprocessing/data_cleaning.md`.
- **Transformar o codificar:** cambiar la escala o forma de una variable sin necesariamente añadir una idea nueva, por ejemplo estandarizar, aplicar `log1p`, discretizar o hacer one-hot encoding.
- **Crear variables:** derivar ratios, componentes de calendario, agregados por grupo, lags o interacciones que no estaban explícitos en la tabla original.
- **Seleccionar o reducir:** conservar un subconjunto de variables originales o proyectarlas a menos dimensiones. Selección y extracción no son sinónimos: PCA combina variables, mientras un selector descarta columnas. [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md). **Tipo de contenido: explícito de la fuente y síntesis integrada.**

Más columnas no implican necesariamente mejor modelo. Las expansiones pueden aumentar dispersión, memoria, varianza y riesgo de ajustar ruido; deben conservarse solo si aportan bajo una validación comparable y un coste aceptable. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md). Este documento no desarrolla limpieza general, elección completa de modelos, estrategia de validación ni *tuning*; solo explica cómo esas decisiones limitan una ingeniería de variables segura.

## 2. Qué es una variable útil para el modelo

Una variable es potencialmente útil cuando cumple, al menos, estas condiciones:

1. **Tiene significado compatible con su representación.** Un entero que identifica un código postal o una clase de producto no se vuelve continuo por su tipo de almacenamiento. Una categoría nominal no debe recibir orden o distancia artificial sin justificación. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md).
2. **Está disponible en el instante real de predicción.** Una derivación del objetivo, un agregado que incorpora el futuro o una señal conocida solo después del evento produce fuga, aunque correlacione fuertemente con `y`. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [McKinney, cap. 11](../../sources/mckinney_python_data_analysis/ch11_time_series.md).
3. **Su geometría es utilizable por el algoritmo.** Los métodos basados en distancias, kernels o densidad pueden quedar dominados por variables de mayor magnitud; los modelos lineales pueden necesitar bins o interacciones para representar no linealidad; los árboles suelen aprender cortes e interacciones sin expansión explícita. [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md); [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md).
4. **Puede reproducirse fuera de la muestra.** Vocabulario, escalas, límites, índices, ventanas y tablas de consulta deben poder reaplicarse con la misma semántica en validación, test e inferencia. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).
5. **Tiene soporte suficiente.** Bins, cruces y estadísticas de grupos demasiado finos pueden memorizar observaciones o quedar sustentados por muy pocos ejemplos. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [McKinney, cap. 10](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md); [ML Design Patterns, cap. 2](../../sources/ml_design_patterns/ch02_data_representation_patterns.md).
6. **Aporta bajo una comparación válida.** La máxima varianza de PCA, una correlación, una importancia de árbol o una frecuencia alta no garantizan utilidad predictiva. La variable se juzga dentro del flujo y con la misma partición y métrica del baseline. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md).

**Límite de interpretación.** Una asociación observada —incluidos coeficientes, correlaciones, importancias o mejoras en una partición— no demuestra causalidad ni garantiza rendimiento cuando cambien población, tiempo o proceso de generación. Las fuentes respaldan explícitamente que correlación e importancia no son causalidad y que la mejora debe verificarse fuera de muestra. **Tipo de contenido: explícito de la fuente.**

## 3. Flujo de trabajo para diseñar variables

1. **Definir unidad, objetivo y momento de predicción.** Precisar qué representa una fila, qué se predice y qué información existe entonces. Esto descarta variables imposibles de servir y agregados futuros. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).
2. **Entender el significado y tipo semántico de cada entrada.** Separar continuas, conteos, ordinales, nominales, fechas, texto libre, strings estructurados e identificadores. Inspeccionar categorías, frecuencias, rangos y distribuciones. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [Müller, cap. 7](../../sources/muller_introduction_ml_python/ch07_text_data.md).
3. **Separar evaluación antes de aprender representaciones.** Elegir una partición compatible con la tarea —por ejemplo, cronológica cuando se predice el futuro— y mantener test fuera del diseño. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md).
4. **Construir una representación mínima y un baseline.** Usar una codificación simple y coherente con el modelo; registrar métrica, número de variables, tiempo y memoria. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md). **Tipo de contenido: síntesis práctica derivada.**
5. **Formular una hipótesis para cada cambio.** Ejemplos: comprimir una cola de conteos, exponer periodicidad, representar una categoría nominal, añadir una interacción plausible o condensar variables correlacionadas.
6. **Ajustar solo con entrenamiento.** Medias, desviaciones, mínimos, máximos, cuantiles, vocabularios, bins, componentes, selectores y agregados deben aprenderse en train —y dentro de cada fold cuando hay validación cruzada—; validación y test reciben solo `transform`. [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md); [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md).
7. **Comparar contra la misma referencia.** Mantener partición y métrica; añadir transformaciones de una en una o en grupos interpretables para atribuir el efecto. [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md). **Tipo de contenido: conclusión práctica derivada.**
8. **Auditar fallos y coste.** Revisar categorías o grupos no vistos, valores fuera de rango, combinaciones sin soporte, dimensión, memoria, latencia e interpretabilidad.
9. **Registrar y empaquetar.** Guardar lógica, parámetros aprendidos, columnas de entrada/salida y justificación junto al modelo o en una dependencia versionada. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

Los pasos 4, 5, 7 y 8 son una **síntesis práctica integrada** de los procedimientos de las fuentes; no constituyen una receta literal de un único resumen.

## 4. Familias de transformaciones

| Tipo de variable | Transformación respaldada | Cuándo considerarla | Precaución |
|---|---|---|---|
| Numérica continua | Estandarización, min-max o robusta | Algoritmo sensible a escala o unidades heterogéneas | Ajustar estadísticas solo con train; no escalar por costumbre |
| Conteo no negativo sesgado | `log1p` | Muchos valores pequeños y pocos muy grandes | Dominio no negativo; validar por columna |
| Numérica con relación por regiones | Binning | Modelo lineal incapaz de capturar una relación no lineal simple | Pierde detalle; límites y número de bins controlan complejidad |
| Nominal de cardinalidad manejable | One-hot encoding | No existe orden cuantitativo | Política para categorías desconocidas y esquema fijo |
| Nominal de cardinalidad alta o vocabulario abierto | Hashing; embeddings como opción más compleja | One-hot es inviable o se esperan valores nuevos | Colisiones en hashing; datos, coste y menor interpretabilidad en embeddings |
| Fecha/hora | Componentes de calendario, lags, ventanas y agregación temporal | Existen ciclos o dependencia del pasado | Respetar el orden y excluir el futuro |
| Texto libre | Conteos, TF-IDF, n-grams | Clasificación o representación documental | Vocabulario y estadísticas solo de train; dimensión y memoria |
| Variables con interacción plausible | Productos, polinomios o *feature crosses* | Modelo simple no aprende el efecto conjunto | Crecimiento dimensional y escasez por combinación |
| Alta dimensión correlacionada | PCA | Compresión, coste o representación lineal | Escala, pérdida de interpretación y varianza no igual a utilidad predictiva |
| Muchas variables posiblemente irrelevantes | Selección univariante, basada en modelo, RFE o baja varianza | Reducir coste o complejidad | Toda selección supervisada se ajusta solo con train |

### 4.1 Variables numéricas

**Escalamiento.** `StandardScaler` centra por variable y lleva a varianza unitaria; `RobustScaler` usa mediana y cuartiles para reducir la influencia de outliers; `MinMaxScaler` reutiliza mínimos y máximos de train; `Normalizer` opera por fila y conserva dirección, no magnitud. Son pertinentes para SVM, redes, distancias, densidad, PCA o regularización, pero no agregan información y una geometría mal elegida puede perjudicar. Los valores de test fuera del rango de train con min-max son posibles y no justifican reajustar. [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md). **Tipo: explícito de la fuente.**

**Transformaciones no lineales.** `log1p(x)` está respaldado para conteos no negativos con cola derecha; comprime valores grandes y admite cero. Las fuentes también mencionan funciones periódicas y otras transformaciones, pero insisten en que no son universales y dependen del dominio de entrada. Para árboles, una transformación monotónica suele aportar menos porque preserva el orden de los valores. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md). **Tipo: explícito de la fuente.**

**Discretización.** Convierte una continua en pertenencia a intervalos. Tras one-hot, un modelo lineal aprende un nivel por intervalo; combinar bins con la variable original o interactuar ambos permite pendiente global o por región. Puede ayudar a modelos lineales, pero pierde variación dentro del bin, y pocos/muchos intervalos pueden causar infraajuste/sobreajuste. `cut` usa límites del rango y `qcut` cuantiles observados, por lo que estos últimos son parámetros aprendidos de la muestra. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [McKinney, cap. 10](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md).

**Ratios y agregados.** Un ratio normaliza una cantidad por una base cuando el denominador es válido y distinto de cero. Los agregados por entidad o segmento —conteos, medias, desviaciones— pueden representar contexto, pero grupos pequeños o casi únicos producen estimaciones inestables. Deben calcularse con train y mapearse a otras particiones, con respaldo para grupos desconocidos. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [McKinney, cap. 10](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md). La política para grupos nuevos es **conclusión práctica derivada** señalada por el propio resumen.

### 4.2 Variables categóricas

**One-hot.** Sustituye una variable de `k` categorías por indicadores. Es la primera opción respaldada para nominales pequeñas o estables porque evita imponer orden. Antes conviene auditar espacios, mayúsculas, sinónimos y errores que crearían categorías artificiales. El encoder se ajusta con train y debe conservar significado y orden de columnas; las categorías desconocidas requieren política explícita. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

**Ordinal.** Solo es defendible cuando el orden forma parte del significado. Para casos ambiguos —por ejemplo, una valoración— las fuentes recomiendan comparar representación ordinal y categórica en vez de asumir continuidad. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md).

**Hashed Feature.** Proyecta categorías a un número fijo de buckets sin mantener vocabulario exhaustivo. Tolera valores nuevos y acota memoria, pero varias categorías pueden colisionar; una frecuente puede dominar a otras raras. Se considera cuando el vocabulario es abierto o one-hot es inviable, y se compara contra one-hot cuando el vocabulario es pequeño y estable. [ML Design Patterns, cap. 2](../../sources/ml_design_patterns/ch02_data_representation_patterns.md).

**Embeddings.** Proyectan categorías de alta dimensión a vectores densos aprendidos y pueden capturar relaciones relevantes para la tarea. Requieren indexación estable, datos suficientes y más ajuste; reducen interpretabilidad y no son la primera opción bajo presupuesto limitado. [ML Design Patterns, cap. 2](../../sources/ml_design_patterns/ch02_data_representation_patterns.md).

### 4.3 Variables temporales

Un timestamp absoluto no expresa automáticamente periodicidad. Hora, día de semana u otros componentes de calendario pueden reutilizar patrones recurrentes; un cruce día×hora permite a un modelo lineal aprender un efecto por combinación. Los árboles no extrapolan más allá del rango observado, por lo que un entero POSIX puede memorizar train y fallar en fechas futuras. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md).

`shift` permite lags; `resample` agrega a otra frecuencia; `rolling`, `expanding` y `ewm` crean estadísticas del pasado con ventanas fijas, acumulativas o ponderadas. Deben definirse frecuencia, alineación, `closed`/`label`, `min_periods` y, sobre todo, qué observaciones estaban disponibles en cada instante. `bfill` puede introducir futuro; un `shift` de valores no equivale a desplazar el índice con `freq`. [McKinney, cap. 11](../../sources/mckinney_python_data_analysis/ch11_time_series.md). **Tipo: explícito de la fuente.**

La fuente temporal cubre las operaciones de pandas y advierte sobre leakage, pero no ofrece una arquitectura general para computar todas las ventanas por entidad dentro de un pipeline de scikit-learn. Esa implementación queda como limitación; no se improvisa código aquí.

### 4.4 Texto y otras representaciones

Primero se decide si un string es categoría, cadena estructurada o texto libre. Para texto libre, bag-of-words tokeniza, construye vocabulario y cuenta tokens; TF-IDF repondera usando frecuencia del documento y rareza en el corpus; n-grams conservan contexto local, como negaciones. Unigramas son un baseline; bigramas pueden aportar contexto; n-grams largos elevan dimensión, coste y sobreajuste. [Müller, cap. 7](../../sources/muller_introduction_ml_python/ch07_text_data.md).

`min_df` reduce tokens raros; stopwords, stemming y lematización son dependientes del corpus e idioma y deben validarse, no asumirse beneficiosos. El vocabulario, frecuencia documental y normalización TF-IDF se aprenden solo con train y conviene mantener matrices dispersas. Topic modeling mediante LDA puede producir una representación documento-temas, pero los temas no garantizan significado humano y requieren inspección. [Müller, cap. 7](../../sources/muller_introduction_ml_python/ch07_text_data.md).

Las fuentes permitidas también cubren embeddings y entradas multimodales a nivel conceptual. Integrar texto, imagen y metadatos solo es pertinente si todas las ramas estarán disponibles en inferencia y cada una demuestra señal por separado; aumenta coste y riesgo de atajos espurios. [ML Design Patterns, cap. 2](../../sources/ml_design_patterns/ch02_data_representation_patterns.md). No hay cobertura suficiente en los resúmenes revisados para detallar extracción moderna de imágenes o audio.

### 4.5 Interacciones y combinaciones

`PolynomialFeatures` crea potencias y productos. Puede hacer visible curvatura o efectos conjuntos a un modelo lineal, pero crece combinatoriamente, puede producir escalas extremas y sobreajustar en regiones con pocos datos. Grado bajo y pocas variables con hipótesis plausible son el caso respaldado; árboles y bosques suelen aprender interacciones sin expansión explícita. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md).

Un *feature cross* crea combinaciones categóricas; si parte de continuas, las fuentes indican bucketizarlas antes. Día×hora es el ejemplo conceptual: el modelo aprende un peso para cada combinación. Cardinalidad, soporte y regularización condicionan su uso; hashing o embeddings pueden acotar/comprimir la expansión con sus propios costes. [ML Design Patterns, cap. 2](../../sources/ml_design_patterns/ch02_data_representation_patterns.md).

### 4.6 Selección o reducción de variables

- **Selección univariante:** rápida y útil en alta dimensión; evalúa cada variable aislada y puede perder señal que solo aparece en combinación.
- **Selección basada en modelo:** usa coeficientes o importancias conjuntas y hereda sesgos y límites del estimador.
- **RFE:** reentrena iterativamente y puede ser demasiado costosa bajo presión.
- **Baja varianza:** elimina variables casi constantes, pero el umbral depende de la escala y baja varianza no equivale siempre a irrelevancia.
- **PCA:** combina variables en componentes ortogonales de máxima varianza. Puede comprimir, reducir coste o ruido; requiere atención a escala, pierde interpretación y no garantiza mayor rendimiento predictivo.
- **NMF:** extrae componentes aditivos de entradas no negativas; puede ser más interpretable, pero exige no negatividad y depende de inicialización.
- **t-SNE:** se reserva para visualización exploratoria; la interfaz descrita no transforma casos nuevos y no se presenta como paso de producción.

Fuentes: [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md), [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md) y [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md). Toda selección supervisada usa `y` y debe ajustarse solo con entrenamiento.

## 5. Criterios para elegir una transformación

| Característica de los datos o modelo | Transformación que podría considerarse | Motivo | Precaución |
|---|---|---|---|
| Algoritmo sensible a escala y unidades heterogéneas | Standard, robust o min-max según geometría/outliers | Evita que la magnitud domine | Ajustar con train; los árboles no siempre la necesitan |
| Conteo no negativo con cola derecha | `log1p` | Comprime extremos relativos | No aplicar a negativos ni indiscriminadamente |
| Nominal pequeña y estable | One-hot | No impone orden | OOV y cardinalidad |
| Categoría ordinal real | Representación ordinal y comparación | Conserva el orden semántico | No confundir códigos nominales con orden |
| Vocabulario abierto o cardinalidad extrema | Hashing | Dimensión fija y soporte de valores nuevos | Colisiones y pérdida de identidad |
| Categorías masivas con relaciones latentes y datos suficientes | Embedding | Representación densa aprendida | Coste, dimensión, estabilidad e interpretación |
| Modelo lineal con relación por tramos | Binning | Añade no linealidad controlada | Límites aprendidos en train; pérdida intrabin |
| Interacción de dominio plausible | Producto/polinomio/cross selectivo | Expone el efecto conjunto | Cardinalidad, soporte y regularización |
| Fecha usada para predicción futura | Componentes calendario y, si procede, lags/ventanas pasadas | Expone ciclos/contexto reutilizable | Split cronológico y cero información futura |
| Texto libre | Conteos o TF-IDF; bigramas si el contexto local importa | Representación dispersa interpretable | Ajustar vocabulario/IDF en train y vigilar memoria |
| Muchas numéricas correlacionadas | PCA después de escalado cuando proceda | Compresión lineal | Componentes difíciles de interpretar; validar utilidad |
| Muchas variables y coste alto | Selección simple dentro del pipeline | Reduce dimensión/tiempo | No seleccionar con validación o test |
| Entidades repetidas con contexto estable | Agregados por grupo | Resume historial/segmento | Train-only, grupos nuevos y partición por entidad/tiempo |

**Tipo de contenido:** síntesis integrada de las reglas explícitas de las fuentes. Ninguna fila prescribe una transformación universal; todas requieren validación con el modelo y la partición pertinentes.

## 6. Ajuste seguro y prevención de data leakage

Aprenden parámetros de los datos, entre otras, estas operaciones: escalado, imputación, límites cuantílicos, vocabularios, TF-IDF, bins aprendidos, agregados por grupo, PCA/NMF, selección supervisada y cualquier estadística temporal o global. Aunque una transformación no use `y`, puede filtrar información de validación si ve su distribución. [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md); [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md).

El contrato seguro es:

1. dividir primero;
2. ejecutar `fit` o `fit_transform` solo en entrenamiento;
3. conservar el objeto o artefacto ajustado;
4. ejecutar únicamente `transform` sobre validación, test e inferencia;
5. en validación cruzada, repetir ese ajuste dentro de cada fold mediante un pipeline;
6. para tiempo o entidades, asegurar además que el conjunto de entrenamiento de la transformación solo contiene historia o grupos permitidos por la estrategia de validación.

Ejemplos de parámetros que deben persistirse: `mean_`/`scale_`, mínimos y máximos, categorías y orden de columnas, vocabulario e IDF, fronteras de bins, componentes, máscara de selección, mapeos por grupo y reglas de calendario. El patrón `Transform` añade que lógica y artefactos deben versionarse juntos para evitar *training-serving skew*. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

El análisis general de fuga corresponde a `knowledge/preprocessing/data_leakage.md`. La limpieza de faltantes, duplicados y formatos corresponde a `knowledge/preprocessing/data_cleaning.md`.

## 7. Uso de pipelines y reproducibilidad

Un `Pipeline` encadena transformadores y estimador bajo una interfaz única. Durante validación, cada transformador se ajusta con el fold de entrenamiento y solo transforma el fold de validación; al predecir, reutiliza los pasos sin reajustar. `ColumnTransformer` permite rutas distintas para columnas numéricas, categóricas y textuales. [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md); [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md).

El patrón **Transform** amplía esta idea al contrato de inferencia: `input crudo → transformación versionada → feature → modelo`. Distingue operaciones por instancia de las que analizan el dataset, y exige adjuntar escalas, vocabularios, lookup tables u otros artefactos a la versión del modelo o a una dependencia versionada. La consistencia no garantiza corrección estadística, pero evita que training e inferencia implementen lógicas diferentes. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md). **Tipo: explícito de la fuente.**

El patrón **Workflow Pipeline** organiza ingestión, validación, transformación, entrenamiento, evaluación y despliegue como componentes con entradas, salidas, dependencias, artefactos y *lineage*. Es pertinente cuando el proceso se repite o intervienen varias personas; para un experimento breve puede bastar un flujo determinista sin infraestructura de orquestación. Un pipeline tampoco garantiza por sí mismo que el split o la estadística sean correctos. [ML Design Patterns, cap. 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md). **Tipo: explícito de la fuente y conclusión práctica derivada para competiciones.**

## 8. Reglas de decisión

| Situación | Transformación o estrategia que puede considerarse | Motivo | Riesgo o precaución |
|---|---|---|---|
| Entero sin significado métrico | Tratar como categórico y comparar one-hot/ordinal según semántica | El dtype no define el tipo conceptual | Orden artificial |
| SVM, distancia, densidad, PCA o regularización con escalas distintas | Escalado dentro del pipeline | Geometría y penalización dependen de magnitudes | Elegir escalador según outliers/rango |
| Árbol con variables numéricas originales | Mantener originales como baseline | Aprende cortes e interacciones | No asumir que binning/polinomios ayudarán |
| Modelo lineal con curvatura localizada | Bins o polinomios de grado bajo | Aumenta expresividad | Dimensión y bordes inestables |
| Categorías nuevas esperables | One-hot con política OOV o hashing | Evita fallo y conserva contrato | Ignore puede producir fila sin indicador; hashing colisiona |
| Texto libre con clasificación | Count/TF-IDF + modelo lineal | Baseline disperso e interpretable | Pipeline y memoria |
| Negaciones o frases cortas relevantes | Añadir bigramas | Conserva orden local | Vocabulario mayor; validar |
| Fecha futura | Split cronológico + calendario/lag pasado | Representación reutilizable | Timestamp absoluto y futuro inducen errores/leakage |
| ID o entidad repetida | Agregados de train y respaldo global | Añade contexto | Entidad casi única, grupos nuevos y split incorrecto |
| Alta dimensión correlacionada | PCA escalado | Reduce coste y combina señal | Varianza no equivale a predicción |
| Muchas variables de ruido | Selector dentro del pipeline | Reduce coste y posible varianza | Selector fuera de CV produce leakage |
| Interacción conocida con soporte | Cross selectivo | Hace visible el efecto a un modelo simple | Producto cartesiano y memorizar combinaciones |
| Varias transformaciones encadenadas | Pipeline | Orden y ajuste reproducibles | No corrige una lógica conceptualmente errónea |

**Tipo de contenido:** síntesis integrada; las condiciones y precauciones derivan de las fuentes citadas en las secciones 4–7.

## 9. Implementación mínima en Python

El ejemplo siguiente muestra únicamente la mecánica segura: las columnas se suministran como parámetros; el preprocesamiento se ajusta con train y se reutiliza sobre validación. No presupone dataset, target ni modelo final concretos.

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** Usa `OneHotEncoder(handle_unknown="ignore")` y un `ColumnTransformer` dentro del pipeline, siguiendo las adaptaciones modernas ya marcadas en los resúmenes de Géron y Müller.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_pipeline(numeric_columns, categorical_columns):
    features = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("model", Ridge()),
        ]
    )


pipeline = build_feature_pipeline(
    numeric_columns=numeric_columns,
    categorical_columns=categorical_columns,
)
pipeline.fit(X_train, y_train)        # aprende solo con entrenamiento
validation_score = pipeline.score(X_valid, y_valid)  # no reajusta
validation_predictions = pipeline.predict(X_valid)
```

Para reducción dimensional, PCA debe quedar en el mismo objeto evaluado:

**Nota práctica: adaptación moderna de la idea descrita en la fuente.**

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_features = make_pipeline(
    StandardScaler(),
    PCA(n_components=number_of_components, random_state=42),
)
X_train_reduced = pca_features.fit_transform(X_train_numeric)
X_valid_reduced = pca_features.transform(X_valid_numeric)
```

Para agregados por grupo, se aprende el mapeo con train y se define respaldo para grupos nuevos:

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** La operación `groupby` está respaldada por McKinney; la separación train/validación y el respaldo global son una conclusión práctica contra leakage explícitamente marcada en el resumen.

```python
group_means = X_train.groupby(group_column)[value_column].mean()
global_mean = X_train[value_column].mean()

train_group_feature = X_train[group_column].map(group_means).fillna(global_mean)
valid_group_feature = X_valid[group_column].map(group_means).fillna(global_mean)
```

No se incluye código de ventanas temporales por entidad: los resúmenes respaldan `shift`, `rolling` y `resample`, pero no especifican un transformador general que garantice orden, corte y estado correctos para todos los esquemas tabulares.

## 10. Errores frecuentes y riesgos

**Riesgos explícitos en las fuentes:**

- tratar cualquier entero como continuo o cualquier string como texto libre;
- codificar train y test por separado, cambiando columnas o significado;
- incluir el objetivo o una derivación suya en `X`;
- ajustar escalas, bins, vocabularios, TF-IDF, PCA o selección antes del split o fuera de CV;
- aplicar escalado sin que el método o la geometría lo requieran;
- generar demasiados bins, polinomios, n-grams o crosses y ajustar ruido;
- no definir categorías, grupos o valores fuera de vocabulario;
- confundir TF-IDF alto, correlación, varianza de PCA o importancia de árbol con utilidad predictiva o causalidad;
- esperar extrapolación temporal de árboles a partir de un timestamp creciente;
- usar `bfill`, ventanas o agregados que incorporan el futuro;
- confundir `shift` de valores con desplazamiento del índice;
- transformar texto o fecha perdiendo señales relevantes sin comparar contra baseline;
- convertir matrices de texto dispersas a densas;
- usar t-SNE como transformación para casos nuevos;
- comparar representaciones sobre particiones distintas;
- reproducir una transformación incorrecta: los pipelines aseguran consistencia, no validez.

Fuentes: [Müller, caps. 3, 4, 6 y 7](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md), [McKinney, caps. 10 y 11](../../sources/mckinney_python_data_analysis/ch11_time_series.md), [ML Design Patterns, caps. 2 y 6](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md).

**Conclusiones prácticas derivadas:**

- una mejora extraordinaria tras selección o agregados debe activar una auditoría de leakage;
- una variable compleja sin hipótesis ni ablación dificulta atribuir la mejora;
- una feature disponible en el archivo de competición puede seguir siendo inválida si no existiría en producción al predecir;
- repetir muchas decisiones mirando la misma validación puede sobreajustar esa validación, aunque cada pipeline esté bien implementado.

## 11. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Identificar rápido unidad de observación, target, tipos semánticos, IDs, fechas, texto y columnas sospechosas de revelar el resultado.
2. Reservar una validación que imite la evaluación; usar orden temporal o grupos cuando la estructura lo exija.
3. Construir una representación simple: numéricas sin expansión innecesaria; one-hot para nominales pequeñas; hashing solo si la cardinalidad/OOV lo justifica; TF-IDF para texto libre; calendario básico para fechas.
4. Encapsular todo parámetro aprendido en un pipeline y fijar un baseline reproducible.
5. Añadir transformaciones de una en una o en bloques interpretables: `log1p` para conteos compatibles, pocos cruces con hipótesis, agregados de train o PCA si el coste/dimensión lo exige.
6. Comparar siempre con la misma partición y métrica; registrar también columnas, memoria y tiempo.
7. Auditar errores, categorías desconocidas, grupos nuevos y disponibilidad al predecir.
8. Evitar, bajo tiempo limitado, polinomios altos, RFE costosa, tokenización lingüística compleja, multimodal profundo, una Feature Store real o un orquestador distribuido sin necesidad.

La prioridad es una representación correcta y reproducible, no la cantidad de ideas probadas. Una transformación compleja se justifica solo por hipótesis, evidencia de validación y viabilidad de inferencia.

## 12. Checklist operativo

- [ ] Entendí la unidad de observación, el tipo y el significado de cada variable transformada.
- [ ] Separé limpieza, transformación, creación y selección de variables.
- [ ] Justifiqué cada transformación con el problema, los datos o las necesidades del modelo.
- [ ] Confirmé que inputs y variables derivadas estarán disponibles al predecir.
- [ ] Definí una partición compatible con tiempo, grupos o clases antes de ajustar transformaciones.
- [ ] Ajusté con entrenamiento las transformaciones que aprenden parámetros.
- [ ] Apliqué sin reajuste la misma transformación a validación o prueba.
- [ ] Encapsulé en el pipeline los pasos que deben repetirse dentro de cada fold.
- [ ] Comparé con un baseline usando la misma partición y métrica.
- [ ] Revisé categorías, grupos y valores no vistos.
- [ ] Controlé dimensión, soporte por bin/cruce/grupo, memoria y latencia.
- [ ] Registré la lógica, los parámetros aprendidos y el motivo de conservarla.
- [ ] Evité interpretar asociación o importancia como causalidad.

## 13. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con ingeniería de variables |
|---|---|
| `knowledge/eda/dataset_audit.md` | Aporta información sobre tipos, distribuciones y problemas de las variables |
| `knowledge/preprocessing/data_cleaning.md` | Trata la corrección y preparación de datos antes de transformarlos |
| `knowledge/preprocessing/data_leakage.md` | Profundiza en la prevención de fuga durante transformaciones |
| `knowledge/validation/validation_strategy.md` | Define cómo evaluar cambios en las variables |
| `knowledge/models/tabular_models.md` | Describe necesidades de representación según la familia de modelos |
| `knowledge/workflows/competition_workflow.md` | Integra la ingeniería de variables en el proceso completo |

Esta tabla expresa relaciones documentales solicitadas; no afirma que esos archivos existan ya ni desarrolla su contenido.

## 14. Vacíos y limitaciones de las fuentes

- Los resúmenes cubren bien tabular, texto clásico, tiempo con pandas, PCA/NMF, selección y pipelines, pero no ofrecen una guía suficiente para extracción moderna de imágenes, audio, grafos o geodatos.
- Embeddings y multimodalidad aparecen a nivel de patrones y arquitectura; no hay criterios cuantitativos universales para dimensión, volumen mínimo de datos o superioridad frente a baselines.
- No hay umbrales universales respaldados para número de bins, cardinalidad “alta”, `min_df`, grado polinómico, cantidad de componentes o tamaño de hash. Todos dependen de validación.
- La fuente temporal explica operaciones y riesgos, pero no proporciona un transformador general y seguro para ventanas por entidad, horizontes múltiples y estado online.
- Las categorías desconocidas están cubiertas como problema, pero la elección entre ignorar, reservar OOV, hashing o reentrenar depende del sistema y no se resuelve con una regla única.
- PCA conserva varianza, no necesariamente señal del objetivo; las fuentes exigen evaluar el modelo posterior y no dan garantía de mejora.
- La selección de variables puede reducir coste e interpretación, pero las fuentes advierten que no suele producir grandes mejoras universales.
- Varias APIs de pandas, scikit-learn y TensorFlow de las fuentes son históricas. Los ejemplos de este documento son adaptaciones modernas explícitas; la compatibilidad exacta depende de la versión instalada.
- Las transformaciones de target, técnicas avanzadas de *target encoding*, codificación cíclica concreta con seno/coseno, selección causal y monitoreo de drift de features no tienen cobertura suficiente en el conjunto revisado para desarrollarlas aquí.
- Toda utilidad predictiva depende de una estrategia de validación válida; este documento no sustituye `knowledge/validation/validation_strategy.md`.

## 15. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning*, cap. 2 | [Resumen](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Flujo end-to-end, tipos, ratios, codificación, escalado, `ColumnTransformer`, pipelines y evaluación protegida |
| Müller y Guido, cap. 3 | [Resumen](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md) | Escaladores, PCA, NMF, t-SNE y ajuste train-only |
| Müller y Guido, cap. 4 | [Resumen](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md) | Semántica de variables, one-hot, binning, `log1p`, polinomios, interacciones, selección y calendario |
| Müller y Guido, cap. 6 | [Resumen](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Pipelines, ajuste dentro de CV, consistencia y prevención de leakage |
| Müller y Guido, cap. 7 | [Resumen](../../sources/muller_introduction_ml_python/ch07_text_data.md) | Tipos de strings, bag-of-words, TF-IDF, n-grams, normalización textual y LDA |
| McKinney, cap. 10 | [Resumen](../../sources/mckinney_python_data_analysis/ch10_groupby_aggregation.md) | Agregados y transformaciones por grupo, `cut`/`qcut`, grupos nuevos y riesgos de leakage |
| McKinney, cap. 11 | [Resumen](../../sources/mckinney_python_data_analysis/ch11_time_series.md) | Fechas, lags, resampling, ventanas, alineación y riesgos temporales |
| Molin, cap. 10 | [Resumen](../../sources/molin_hands_on_pandas/ch10_model_optimization.md) | Construcción/selección, PCA, complejidad, comparación con baseline y coste |
| *Machine Learning Design Patterns*, cap. 2 | [Resumen](../../sources/ml_design_patterns/ch02_data_representation_patterns.md) | Hashed Feature, Embeddings, Feature Cross, Multimodal Input y consistencia de representación |
| *Machine Learning Design Patterns*, cap. 6 | [Resumen](../../sources/ml_design_patterns/ch06_reproducibility_patterns.md) | Patrones Transform y Workflow Pipeline, artefactos, lineage y training-serving skew |

`sources/source_catalog.md` se consultó únicamente como índice de localización; no se utilizó como respaldo técnico.

## 16. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Géron 2; Müller 4; Molin 10; Patterns 2 | Síntesis integrada |
| 2. Variable útil | Müller 3 y 4; Géron 2; Patterns 2 y 6 | Síntesis integrada |
| 3. Flujo de trabajo | Géron 2; Müller 4 y 6; Molin 10; Patterns 6 | Síntesis integrada |
| 4.1 Escalado, transformaciones y binning | Müller 3 y 4; McKinney 10 | Explícito de la fuente |
| 4.1 Ratios y agregados grupales seguros | Géron 2; McKinney 10 | Síntesis integrada |
| 4.2 Categóricas | Müller 4; Géron 2; Patterns 2 | Síntesis integrada |
| 4.3 Temporales | Müller 4; McKinney 11 | Síntesis integrada |
| 4.4 Texto | Müller 7; Patterns 2 | Síntesis integrada |
| 4.5 Interacciones | Müller 4; Molin 10; Patterns 2 | Síntesis integrada |
| 4.6 Selección y reducción | Müller 3 y 4; Molin 10 | Síntesis integrada |
| 5. Criterios | Fuentes de las secciones 4.1–4.6 | Conclusión práctica derivada |
| 6. Leakage | Müller 3 y 6; McKinney 10 y 11; Patterns 6 | Síntesis integrada |
| 7. Pipelines y reproducibilidad | Géron 2; Müller 6; Patterns 6 | Síntesis integrada |
| 8. Reglas de decisión | Fuentes de las secciones 4–7 | Conclusión práctica derivada |
| 9. Código de pipeline y PCA | Géron 2; Müller 3, 4 y 6 | Conclusión práctica derivada |
| 9. Código de agregados por grupo | McKinney 10; Müller 6 | Conclusión práctica derivada |
| 10. Riesgos explícitos | Todas las fuentes técnicas utilizadas | Explícito de la fuente |
| 10. Auditorías prácticas | Müller 6; Molin 10; Patterns 6 | Conclusión práctica derivada |
| 11. Coding test o datathon | Todas las fuentes técnicas utilizadas | Recomendación para coding test |
| 12. Checklist | Todas las fuentes técnicas utilizadas | Conclusión práctica derivada |
| 13. Documentos relacionados | Estructura requerida por la solicitud | Síntesis integrada |
| 14. Vacíos | Cobertura observada en los resúmenes consultados | Conclusión práctica derivada |
