---
title: "Modelos de Machine Learning para datos tabulares"
knowledge_area: "models"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Modelos de Machine Learning para datos tabulares

## 1. Propósito y alcance

Este documento ayuda a seleccionar, entrenar inicialmente y comparar familias comunes de modelos supervisados para datos tabulares. La comparación considera la naturaleza del objetivo, la estructura de los datos, los supuestos del método, la preparación requerida, la interpretabilidad, el coste computacional y la evidencia obtenida fuera del conjunto de entrenamiento.

**Explícito de las fuentes.** Clasificación predice etiquetas discretas y regresión predice valores continuos. La generalización es el rendimiento sobre datos no vistos; un ajuste excelente en entrenamiento no basta para establecerla. La elección del modelo depende de la tarea, la representación y el régimen de datos, y ninguna familia domina en todas las condiciones [M02, I04].

**Síntesis integrada.** Aquí se cubren regresión lineal y logística, regularización, k-nearest neighbors (k-NN), árboles de decisión, máquinas de soporte vectorial (SVM), Naive Bayes, LDA, QDA y, de forma acotada, regresión de Poisson. Ensembles y boosting solo se mencionan como siguiente familia y se remiten a `knowledge/ensembles/boosting_models.md`.

El documento no establece un ganador universal ni reemplaza la validación experimental. Tampoco desarrolla en profundidad limpieza, ingeniería de variables, validación, métricas o tuning: esas decisiones se coordinan con los documentos relacionados de la sección 15.

## 2. Cómo comparar familias de modelos

**Explícito de las fuentes.** La flexibilidad puede reducir el error de entrenamiento y, al mismo tiempo, elevar la varianza y empeorar el rendimiento futuro. La interpretabilidad suele disminuir cuando aumenta la flexibilidad. Los requisitos de escala, la forma de la frontera, el tamaño y dimensionalidad de los datos, y la necesidad de extrapolar cambian qué modelos son candidatos razonables [M02, I03, I09].

| Criterio | Pregunta de comparación | Por qué importa |
|---|---|---|
| Tipo de tarea | ¿El objetivo es continuo, binario, multiclase nominal o un conteo? | Delimita regresores, clasificadores y modelos con soporte específico; usar regresión lineal para clases nominales puede imponer orden o producir valores que no son probabilidades [I04]. |
| Forma y supuestos | ¿Es defendible una relación lineal, log-odds lineales, normalidad por clase, independencia condicional o una estructura local por distancias? | Cada supuesto habilita una familia y señala diagnósticos; una violación relevante puede introducir sesgo o invalidar inferencia [I03, I04]. |
| Escala y representación | ¿Las variables están en unidades comparables? ¿Las categorías tienen orden real? | k-NN y SVM RBF dependen de distancias; Ridge/Lasso dependen de la escala de la penalización. Las categorías nominales no deben recibir un orden artificial [M02, G04, G02]. |
| No linealidad e interacciones | ¿La señal parece lineal, local, basada en umbrales o curva? | Lineales trabajan sobre la representación disponible; términos transformados amplían esa representación, mientras árboles y kernels capturan otras formas [G04, G06, I09]. |
| Interpretabilidad | ¿Se necesitan coeficientes condicionados, reglas visibles o solo predicciones? | Un modelo lineal o un árbol pequeño puede inspeccionarse; k-NN y SVM con kernel ofrecen menos explicación directa [I03, G06, M02]. |
| Coste computacional | ¿Cuántas filas y columnas hay, y cuánto cuesta predecir? | k-NN desplaza el coste a predicción; SVM con kernels y búsquedas de hiperparámetros pueden ser costosos; best subset crece como `2^p` [M02, I06, I09]. |
| Riesgo de sobreajuste | ¿Existe brecha entre entrenamiento y validación? ¿Qué control de complejidad ofrece la familia? | `K`, `alpha`/`lambda`, `C`, profundidad, tamaño de hoja o poda regulan flexibilidad de maneras distintas [M02, G04, G06, I08]. |
| Faltantes y categorías | ¿Hay nulos o variables nominales/ordinales? | Las fuentes describen imputación y codificación como pasos separados; los estimadores aquí no deben asumirse capaces de recibir cualquier valor faltante o texto crudo [G02, P09]. |
| Alta dimensionalidad | ¿`p` se acerca o supera a `n`? ¿La matriz es dispersa? | OLS puede no tener solución única; regularización y modelos lineales son candidatos, mientras k-NN se deteriora por la maldición de la dimensionalidad [I06, M02]. |
| Extrapolación | ¿Se predecirá fuera del rango observado? | Los árboles de regresión producen valores derivados de las hojas y no extrapolan tendencias; la regresión lineal también exige cautela fuera del soporte [M02, P09]. |
| Utilidad operativa | ¿La mejora supera una heurística o el proceso vigente con el mismo criterio? | Una métrica aislada no indica si la complejidad añade valor; el benchmark debe calcularse con training y compararse sobre datos equivalentes [D07]. |

**Conclusión práctica derivada.** La comparación válida es entre flujos completos compatibles —preparación más modelo— evaluados con las mismas particiones y el mismo criterio. No es válido favorecer una familia dándole más información, una preparación aprendida sobre validación o una métrica distinta [G02, M06, I06].

## 3. Modelos lineales

### 3.1 Regresión lineal

**Objetivo y funcionamiento.** La regresión lineal predice un objetivo continuo mediante una combinación lineal de las columnas de entrada. Mínimos cuadrados ordinarios (OLS) elige coeficientes que minimizan la suma de residuos cuadrados. En el caso múltiple, cada coeficiente describe una asociación parcial manteniendo fijos los demás predictores incluidos [I03].

**Supuestos o condiciones.** Para predicción, la media del objetivo debe ser aproximable por una función lineal de las columnas usadas. Para los errores estándar, tests e intervalos clásicos se requieren condiciones adicionales: errores no correlacionados, varianza común y una matriz de diseño identificable; la normalidad sustenta inferencia exacta en muestras pequeñas. Las interacciones o potencias permiten relajar aditividad o linealidad en las variables originales, pero aumentan complejidad [I03].

**Hiperparámetros y complejidad.** OLS básico no tiene un hiperparámetro de regularización. La especificación —columnas, interacciones y grado polinómico— controla su capacidad. La solución mediante pseudoinversa/SVD tolera singularidad mejor que invertir directamente `XᵀX`, pero el ajuste sigue sujeto a problemas de alta dimensión [G04].

**Ventajas.** Es rápida, sirve como baseline, permite examinar coeficientes y puede ser estable cuando la forma lineal es razonable y hay suficientes observaciones por predictor [I03, G04].

**Limitaciones y riesgos.** Puede subajustar relaciones no representadas, sufrir por colinealidad, outliers o puntos de alto leverage y producir inferencia engañosa con errores dependientes o heteroscedásticos. `R²` de entrenamiento no demuestra generalización y siempre puede aumentar al añadir columnas. Un coeficiente o p-value indica asociación dentro de la especificación, no causalidad [I03].

**Preparación.** Las categorías nominales requieren indicadores; con intercepto, una variable de `L` niveles se representa con `L-1` indicadores en el tratamiento descrito por ISLP. Imputación y codificación deben aprenderse solo con entrenamiento. El escalado no se presenta como requisito universal de OLS cerrado, pero sí es necesario para descenso de gradiente rápido y para comparar penalizaciones regularizadas [I03, G02, G04].

**Fuentes:** [G04, I03, P09].

### 3.2 Regresión logística

**Objetivo y funcionamiento.** La regresión logística binaria estima `P(Y=1|X)` aplicando la función logística a una combinación lineal; los coeficientes son lineales en log-odds y se estiman por máxima verosimilitud. La versión multinomial produce probabilidades para clases nominales mutuamente excluyentes [G04, I04].

**Supuestos o condiciones.** Impone linealidad de los log-odds en las columnas de diseño, no linealidad de la probabilidad. La interpretación de un coeficiente es condicional al resto de predictores. El umbral `0.5` solo representa una regla adecuada bajo costes simétricos; otro objetivo puede justificar otro umbral [I04].

**Hiperparámetros importantes.** En `LogisticRegression` de scikit-learn, `C` es el inverso de la fuerza de regularización: un `C` mayor regulariza menos. También importan la penalización compatible, el solver y `max_iter`; el umbral se decide después del ajuste y no debe confundirse con un parámetro del modelo [G04, P09].

**Ventajas.** Produce probabilidades en `[0,1]`, tiene entrenamiento y predicción rápidos, ofrece una frontera lineal y funciona como baseline de clasificación binaria o multiclase [I04, M02].

**Limitaciones y riesgos.** Puede subajustar una frontera no lineal, ser inestable ante separación fuerte y confundir asociaciones marginales con parciales. La codificación numérica arbitraria de clases nominales y la interpretación causal de coeficientes son errores [I04].

**Preparación.** Requiere una matriz numérica/codificada. Las transformaciones aprendidas se ajustan dentro del training o de cada fold. El escalado es coherente cuando se aplica regularización y evita que las unidades condicionen la penalización; las fuentes implementan esta combinación mediante pipeline [G04, M02, M06].

**Fuentes:** [G04, M02, I04, P09].

### 3.3 Regularización

**Explícito de las fuentes.** La regularización introduce sesgo para reducir varianza. Ridge usa una penalización L2 y contrae todos los coeficientes; normalmente no los lleva a cero. Lasso usa L1 y puede producir coeficientes exactamente cero. Elastic Net mezcla L1 y L2 mediante `l1_ratio`; esta variante está documentada por Géron [G04, I06].

| Método | Efecto documentado | Control principal | Ventaja | Limitación |
|---|---|---|---|---|
| Ridge | Contrae coeficientes sin selección dura habitual | `alpha`/`lambda`; mayor valor implica mayor regularización | Estabiliza con multicolinealidad, `p` grande o señales pequeñas repartidas | Conserva todas las variables y depende de la escala |
| Lasso | Puede fijar coeficientes en cero | `alpha`/`lambda` | Modelo sparse y selección simultánea | Puede ser inestable con predictores correlacionados; no siempre predice mejor que Ridge |
| Elastic Net | Combina L1 y L2 | `alpha` y `l1_ratio` | Combina contracción y sparsity; Géron lo presenta como alternativa cuando Lasso es inestable | Añade una decisión de mezcla que debe validarse |

Los predictores deben estar en escalas comparables para que la penalización no dependa arbitrariamente de las unidades. La estadística de escalado, la selección de `alpha`/`lambda` y cualquier selección de variables se aprenden dentro del entrenamiento. La penalización pertenece al objetivo de ajuste, no a la métrica final [G04, I06].

**Riesgo de sobreajuste o subajuste.** Regularización insuficiente puede dejar alta varianza; regularización excesiva puede borrar señal y causar subajuste. En `p≈n` o `p>n`, OLS puede ser muy variable o no único; Ridge y Lasso son candidatos, pero su rendimiento debe compararse con validación común [I06].

**Conclusión práctica derivada.** Considere Ridge cuando la señal parezca distribuida entre muchas variables y Lasso cuando se espere señal dispersa o se necesite parsimonia. Esta es una regla condicional, no una garantía. Coeficientes cero no prueban irrelevancia causal ni selección estable [I06].

**Fuentes:** [G04, M02, I06].

## 4. Modelos basados en vecinos

**Explícito de las fuentes.** k-NN memoriza el conjunto de entrenamiento. En clasificación asigna la clase por la pluralidad de los `K` vecinos; en regresión promedia sus respuestas. La distancia euclídea es la opción predeterminada descrita por Müller. Un `K` pequeño produce mayor flexibilidad, menor sesgo y mayor varianza; un `K` grande suaviza la frontera o curva y puede elevar el sesgo [M02, I03, I04].

- **Objetivo:** clasificación o regresión local sin imponer una forma paramétrica global.
- **Hiperparámetros:** `n_neighbors`/`K` y métrica de distancia.
- **Ventajas:** entrenamiento barato, lógica sencilla y capacidad de capturar no linealidad local en baja dimensión.
- **Limitaciones:** predicción costosa con muchas filas; sensibilidad a escala, variables irrelevantes y dimensionalidad; poca interpretación global y ausencia de coeficientes.
- **Preparación requerida:** variables en escalas comparables y una noción de distancia significativa. El scaler debe ajustarse solo con entrenamiento y vivir dentro del pipeline durante validación.
- **Riesgo de ajuste:** `K=1` puede interpolar entrenamiento y presentar alta varianza; aumentar `K` reduce flexibilidad y puede subajustar.

**Costo computacional documentado.** El `fit` principalmente almacena datos y el trabajo costoso ocurre al predecir. Por ello, k-NN deja de ser atractivo con muchas observaciones o variables; también funciona mal con matrices dispersas de alta dimensión [M02].

**Conclusión práctica derivada.** Úselo como candidato pequeño y controlado cuando el dataset sea denso, de dimensión moderada y con suficientes ejemplos locales. No lo elija solo porque logra error de entrenamiento bajo; compare `K` con el mismo esquema de validación [M02, I03].

**Fuentes:** [M02, I03, I04].

## 5. Árboles de decisión

**Objetivo y funcionamiento.** Un árbol crea una jerarquía de preguntas binarias `variable <= umbral`. CART elige de forma voraz la característica y el punto de corte que mejor reducen el criterio local. En clasificación, la hoja devuelve la clase mayoritaria y proporciones de clase; en regresión, la media de los objetivos de la hoja [G06, I08].

**Criterios de división.** Para clasificación, Gini y entropía miden impureza y son más sensibles que la tasa de error para construir el árbol. Géron indica que suelen producir árboles parecidos. En regresión, las divisiones minimizan el error cuadrático ponderado de las regiones hijas [G06, I08].

**Controles de complejidad.** `max_depth`, `max_leaf_nodes`, `max_features`, `min_samples_split`, `min_samples_leaf` y `min_weight_fraction_leaf` limitan el crecimiento. ISLP también cubre poda por complejidad de coste: `ccp_alpha` penaliza el número de hojas y debe elegirse mediante validación dentro del entrenamiento [G06, I08].

**Ventajas.** Un árbol pequeño expone reglas, admite relaciones no lineales y umbrales, no necesita escalado o centrado y predice con rapidez. Es útil cuando una regla visible forma parte del entregable [G06, M02].

**Limitaciones y riesgos.** Un árbol sin restricciones memoriza con facilidad y tiene alta varianza. Las fronteras son paralelas a los ejes; rotaciones o cambios pequeños en los datos pueden modificar mucho la estructura. Un regresor arbóreo predice valores derivados de respuestas observadas en hojas y no extrapola tendencias. Las proporciones de una hoja no deben asumirse calibradas [G06, M02].

**Preparación.** El escalado no es obligatorio. Aun así, faltantes y categorías deben tratarse de acuerdo con la implementación. Aunque el tratamiento conceptual permite divisiones categóricas, el laboratorio de ISLP con scikit-learn usa one-hot; no se generaliza aquí que todos los árboles acepten texto crudo o faltantes [I08].

**Interpretabilidad.** La legibilidad depende del tamaño. `feature_importances_ == 0` no demuestra que una variable carezca de señal, y una importancia no negativa no indica dirección del efecto ni causalidad [M02].

**Alcance de ensembles.** Si un árbol resulta inestable, las fuentes proponen bagging o random forests para reducir varianza y boosting para aprendizaje secuencial. Su desarrollo, hiperparámetros y comparación se remiten a `knowledge/ensembles/boosting_models.md` [I08].

**Fuentes:** [G06, M02, I08].

## 6. Otras familias respaldadas por las fuentes

### Máquinas de soporte vectorial

El clasificador de margen máximo busca el hiperplano con mayor margen, pero exige separabilidad y puede ser sensible a una observación. El margen suave permite violaciones; solo los puntos sobre el margen o que lo violan determinan directamente la frontera. Los kernels lineal, polinomial y RBF permiten fronteras de distinta flexibilidad sin materializar siempre la expansión [I09].

- **Tarea cubierta:** clasificación binaria; multiclase mediante one-versus-one u one-versus-rest. SVR aparece solo de forma conceptual y no se desarrolla aquí.
- **Hiperparámetros:** en `sklearn.svm.SVC`, `C` alto penaliza más las violaciones y ajusta más el entrenamiento; con RBF se ajustan conjuntamente `C` y `gamma`; el kernel polinomial añade `degree`.
- **Preparación:** representación numérica y escalas coherentes, especialmente para RBF por su dependencia de distancia euclídea; ajuste dentro del pipeline.
- **Ventajas:** frontera lineal robusta en alta dimensión o frontera no lineal flexible con kernels; depende de vectores de soporte.
- **Limitaciones:** el score de `decision_function` no es una probabilidad; los kernels exigen búsqueda cuidadosa y el coste crece por similitudes entre pares. Müller advierte costes importantes alrededor de 100 000 muestras.
- **Riesgo:** `C` o `gamma` altos pueden lograr entrenamiento casi perfecto y peor validación. Una SVM no lineal no se justifica si la lineal valida igual.

La fuente advierte dos convenciones de `C`: la formulación teórica del presupuesto de holgura y el parámetro de coste de `SVC` cambian en sentidos opuestos. Este documento usa la convención de scikit-learn indicada arriba [I09].

### Naive Bayes

Naive Bayes estima probabilidades posteriores con estadísticas por variable y clase, suponiendo independencia condicional entre predictores. `GaussianNB` corresponde a variables continuas resumidas por media y desviación; `BernoulliNB`, a variables binarias; `MultinomialNB`, a conteos no negativos [M02, I04].

- **Ventajas:** entrenamiento y predicción muy rápidos, pocos parámetros y utilidad potencial con alta dimensionalidad dispersa o `p` grande respecto de `n`.
- **Hiperparámetro documentado:** `alpha` suaviza conteos en Bernoulli/Multinomial; valores mayores simplifican el modelo.
- **Limitaciones:** no modela interacciones; dependencias condicionales fuertes pueden degradarlo. Müller señala que suele ser algo menos preciso que modelos lineales.
- **Decisión de uso:** la variante debe corresponder a la representación. No se presenta como método de regresión.

### LDA y QDA

LDA y QDA son clasificadores generativos que estiman la distribución de `X` dentro de cada clase y aplican Bayes. LDA supone normalidad multivariante y una covarianza común, por lo que produce fronteras lineales. QDA mantiene normalidad pero estima una covarianza por clase y produce fronteras cuadráticas [I04].

- **LDA:** menor varianza relativa; candidato con muestra modesta si normalidad y covarianza común son razonables.
- **QDA:** más flexible cuando las covarianzas difieren, pero estima muchos más parámetros y requiere suficientes observaciones por clase.
- **Precaución:** ninguna es universalmente superior a logística. La elección depende del ajuste de los supuestos y de validación comparable.

### Regresión de Poisson

ISLP cubre Poisson para objetivos de conteo entero no negativo. Usa enlace log y produce medias no negativas. Supone igualdad condicional entre media y varianza; la sobredispersión vuelve demasiado optimistas los estadísticos `z`. La fuente no desarrolla soluciones a la sobredispersión, por lo que se registra como limitación y no se completa aquí [I04].

### Familias deliberadamente no desarrolladas

MLP aparece en Müller como modelo flexible de clasificación o regresión, sensible a escala, inicialización, tuning y coste, pero no se desarrolla porque el objetivo de este documento es una comparación inicial y las fuentes lo desaconsejan como baseline tabular rápido. Random forests, bagging y boosting tienen cobertura suficiente, pero se remiten al documento de ensembles por restricción de alcance [M02, I08].

## 7. Requisitos de preparación según el modelo

| Familia de modelo | Preparación respaldada por las fuentes | Motivo | Precaución |
|---|---|---|---|
| OLS lineal | Codificar nominales; revisar colinealidad, residuos, leverage y soporte; imputar antes de ajustar | La matriz debe ser numérica e identificable; los diagnósticos sostienen la especificación | No ajustar imputación/codificación con validación; no asumir causalidad [I03, G02] |
| Ridge, Lasso, Elastic Net | Estandarizar o llevar predictores a escala comparable | La penalización depende de la magnitud de los coeficientes y, por tanto, de las unidades | Escalar y seleccionar `alpha` dentro de folds [G04, I06] |
| Logística | Matriz numérica/codificada; escalado cuando la penalización lo requiera | Modela log-odds lineales sobre las columnas disponibles | Declarar clase positiva, orden de `classes_` y umbral [G04, I04] |
| k-NN | Escalar; retirar o controlar variables irrelevantes; resolver faltantes | Distancias y vecindarios dependen de unidades y dimensión | Alta dimensión o matriz sparse degradan el significado de cercanía [M02, I03] |
| Árbol individual | No requiere escalado; codificar categorías según implementación; resolver faltantes si el estimador no los admite | Los cortes por umbral son invariantes al reescalado monótono | No inferir soporte nativo universal para texto/nulos; one-hot se usa en el laboratorio ISLP [G06, I08] |
| SVM lineal/RBF | Escalar dentro de pipeline; codificar categorías | RBF usa distancia euclídea y el margen depende de la representación | Ajustar `C` y, para RBF, `gamma` solo con entrenamiento [I09, M02] |
| Gaussian/Bernoulli/Multinomial NB | Elegir representación continua, binaria o conteos no negativos según variante | Cada variante aprende estadísticas diferentes | No intercambiar variantes sin verificar el soporte de entrada [M02] |
| LDA/QDA | Matriz numérica/codificada; comprobar plausibilidad de distribución y suficiente muestra por clase | Estiman medias y covarianzas bajo normalidad por clase | QDA es especialmente costosa en parámetros cuando crece `p` [I04] |
| Poisson | Objetivo entero no negativo; revisar relación media-varianza | El soporte y enlace están definidos para conteos | La sobredispersión invalida inferencia ingenua [I04] |

**Regla transversal explícita.** Imputación, escalado, codificación aprendida, selección y expansión de variables deben ajustarse solo con la parte de entrenamiento de cada partición. `Pipeline` y `ColumnTransformer` encapsulan esa secuencia y evitan fuga entre folds [G02, M06, P09].

## 8. Sobreajuste, subajuste y complejidad

**Explícito de las fuentes.** Sobreajuste significa aprender particularidades del training y fallar al generalizar; subajuste significa que el modelo es demasiado simple incluso para capturar la variación del training. Una brecha grande con entrenamiento mejor que validación sugiere alta varianza; rendimientos pobres y próximos sugieren alto sesgo [M02, G04].

| Familia | Más flexibilidad según las fuentes | Menos flexibilidad según las fuentes |
|---|---|---|
| k-NN | `K` pequeño | `K` grande |
| Ridge/Lasso | `alpha`/`lambda` pequeño | `alpha`/`lambda` grande |
| Logística/SVM en scikit-learn | `C` grande | `C` pequeño |
| Árbol | Mayor profundidad, más hojas, hojas pequeñas | Menor profundidad/hojas, `min_samples_leaf` mayor o poda mayor |
| SVM RBF | `gamma` alto produce influencia más local e irregular | `gamma` bajo suaviza la frontera |
| Polinomio | Grado alto | Grado bajo |

Estas direcciones no sustituyen la validación: un nivel intermedio suele equilibrar sesgo y varianza, pero el punto depende de los datos. Más datos diversos pueden sostener mayor complejidad; duplicar observaciones no aporta la misma variedad [M02, I06].

La comparación de complejidad debe hacerse con rendimiento de entrenamiento y validación calculado de forma consistente. Learning curves son un diagnóstico documentado: errores altos y cercanos sugieren subajuste; una brecha persistente sugiere sobreajuste [G04]. El diseño de particiones, validación cruzada, incertidumbre de la estimación y selección final corresponde a `knowledge/validation/validation_strategy.md`; la búsqueda corresponde a `knowledge/tuning/hyperparameter_tuning.md`.

## 9. Reglas de decisión

| Situación o necesidad | Familia que puede considerarse | Motivo | Precaución |
|---|---|---|---|
| Objetivo continuo y relación aproximadamente lineal | OLS o Ridge | Baseline rápido; Ridge reduce varianza con muchas señales o colinealidad | Revisar residuos, escala para Ridge y extrapolación [I03, I06] |
| Clasificación binaria con probabilidades y frontera aproximadamente lineal | Regresión logística | Probabilidades válidas y coeficientes sobre log-odds | Umbral no universal; asociación no es causalidad [I04] |
| `p` cercano o superior a `n` | Ridge o Lasso; SVM lineal como candidato de clasificación | Reducen flexibilidad o aprovechan alta dimensión | Estandarizar, validar y no interpretar selección como causal [I06, I09] |
| Se espera señal repartida entre muchas variables | Ridge | Contrae todos los coeficientes | No produce subconjunto sparse [I06] |
| Se espera señal dispersa y se desea parsimonia | Lasso o Elastic Net | L1 puede anular coeficientes | Selección inestable con correlación; no garantiza mejor predicción [G04, I06] |
| Dataset pequeño, denso y de baja dimensión con patrón local | k-NN | Flexible y barato de entrenar | Escalar; predicción cara y maldición de dimensionalidad [M02, I03] |
| Reglas explicables, umbrales e interacción no lineal | Árbol pequeño/podado | Reglas visibles y sin escalado | Alta varianza, fronteras axiales y falta de extrapolación [G06, I08] |
| Frontera lineal con margen útil o `p≫n` | SVM lineal | Controla margen y depende de vectores de soporte | Ajustar `C`; score no es probabilidad [I09] |
| Frontera local no lineal y tamaño moderado | SVM RBF | Kernel flexible sin expansión explícita | Escalar; ajustar `C`/`gamma`; coste y sobreajuste [I09, M02] |
| Variables binarias o conteos sparse y baseline muy rápido | BernoulliNB o MultinomialNB | Estadísticas simples, alta velocidad | La variante debe coincidir con la representación [M02] |
| `p/n` alto con independencia condicional aproximadamente razonable | Naive Bayes | Reduce parámetros y varianza | No captura interacciones; dependencia fuerte lo perjudica [I04] |
| Normalidad por clase y covarianza común plausible | LDA | Frontera lineal generativa de menor varianza relativa | Verificar supuestos y comparar con logística [I04] |
| Covarianzas por clase diferentes y muestra suficiente | QDA | Frontera cuadrática más flexible | Muchos parámetros; alta varianza con pocos casos [I04] |
| Objetivo de conteo no negativo | Regresión de Poisson | Respeta soporte y usa enlace log | Revisar sobredispersión [I04] |
| Se requiere extrapolar una tendencia | Modelo cuya forma permita esa extrapolación, evaluando una especificación lineal si es razonable | Los árboles no extrapolan fuera de valores de hojas | Toda extrapolación fuera del soporte es riesgosa [M02, P09] |

## 10. Baseline y comparación inicial

**Explícito de las fuentes.** Un benchmark debe ser comprensible y medirse con el mismo criterio y sobre los mismos datos que el modelo. Si ya existe un proceso operativo, ese proceso es la referencia preferente. Si no existe, puede usarse una heurística simple estimada con training: media o mediana del objetivo para regresión, frecuencia/distribución de clases para clasificación, o una regla de negocio defendible [D07].

Una segunda referencia puede ser un modelo simple compatible con la tarea: `LinearRegression` para regresión o `LogisticRegression` para clasificación aparecen como baselines en las fuentes [G02, P09]. Esta elección no es obligatoria ni universal; debe ser compatible con la representación y el objetivo.

**Secuencia de comparación inicial, síntesis integrada:**

1. Fijar la tarea y el protocolo de evaluación antes de comparar.
2. Construir la heurística o medir el proceso vigente usando solo training para cualquier estadística.
3. Ajustar un modelo simple y auditable.
4. Añadir un conjunto pequeño de familias con sesgos distintos y requisitos compatibles con los datos; Géron propone conservar de dos a cinco candidatos antes de tuning intensivo [G02].
5. Mantener constantes filas de entrenamiento/evaluación, particiones, definición del objetivo y criterio de evaluación.
6. Comparar también entrenamiento frente a validación para diagnosticar complejidad.
7. Afinar solo después de confirmar que el pipeline supera una referencia válida.

La métrica concreta se define en los documentos de métricas. La partición y la validación cruzada se definen en el documento de validación. Superar una heurística débil no demuestra utilidad: la magnitud de mejora debe contrastarse con coste, fiabilidad y contexto operativo [D07].

## 11. Implementación mínima en Python

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** El ejemplo integra imputación, escalado y one-hot en un pipeline; separa preparación, modelo y evaluación, y aprende todas las estadísticas solo con training. La estrategia de validación es ilustrativa; su diseño completo pertenece a `knowledge/validation/validation_strategy.md` [G02, M06, P09].

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1) Separación: ocurre antes de aprender imputación, categorías o escalas.
X_train, X_valid, y_train, y_valid = train_test_split(
    feature_table,
    target,
    test_size=0.20,
    stratify=target,
    random_state=42,
)

# 2) Preparación: cada estadística se aprende únicamente con X_train.
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preparation = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]
)

# 3) Modelo: baseline lineal regularizado para clasificación.
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

classification_pipeline = Pipeline(
    steps=[
        ("preparation", preparation),
        ("model", model),
    ]
)

classification_pipeline.fit(X_train, y_train)
validation_predictions = classification_pipeline.predict(X_valid)
validation_probabilities = classification_pipeline.predict_proba(X_valid)

# 4) Evaluación: aplicar aquí la métrica definida para la tarea,
# siempre sobre y_valid y sin reajustar el pipeline con X_valid.
```

Para regresión, las fuentes respaldan sustituir el estimador por `LinearRegression`, `Ridge` o `Lasso` según el diagnóstico y mantener el preprocesamiento dentro del pipeline [G04, I06]. Para un árbol, el escalado numérico puede omitirse porque no es un requisito del algoritmo, pero imputación y codificación siguen dependiendo de la implementación [G06, I08]. No se incluye tuning extenso.

## 12. Errores frecuentes y riesgos

**Riesgos explícitos de las fuentes:**

- Elegir por rendimiento de entrenamiento, `R²` de training o una frontera visualmente compleja.
- Ajustar imputación, escalado, selección, PCA o codificación con todo el dataset antes de separar o fuera de los folds [G02, M06, I06].
- Usar k-NN o RBF SVM sin escalado, permitiendo que las unidades dominen la distancia [M02, I09].
- Confundir `alpha` con `C`: mayor `alpha` aumenta regularización en Ridge/Lasso; mayor `C` reduce regularización en logística y penaliza más violaciones en `SVC` [G04, M02, I09].
- Dejar un árbol crecer sin límites y confundir hojas puras con generalización [G06].
- Interpretar proporciones de hojas o scores de SVM como probabilidades calibradas [G06, I09].
- Elegir variables por `R²` creciente, hacer selección antes de CV o interpretar ceros de Lasso como verdad causal [I03, I06].
- Usar categorías nominales como enteros ordinales; crear distancias inexistentes [G02, P09].
- Comparar modelos con particiones, datos o métricas diferentes [I06, I08].
- Ignorar coste: predicción de k-NN, kernels por pares, best subset exponencial o CV multiplicada por candidatos [M02, I06, I09].

**Síntesis práctica:**

- Aplicar una familia sin revisar qué supuesto o representación la hace razonable.
- Ajustar complejidad mirando repetidamente el test final.
- Presentar coeficientes, importancias o asociaciones como causalidad.
- Declarar una familia superior sin una comparación reproducible fuera de training.
- Incluir más algoritmos cuando el baseline, la partición o el pipeline todavía no son confiables.

## 13. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Identificar si el objetivo es continuo, discreto, multiclase o conteo; confirmar que esa salida coincide con la necesidad del problema.
2. Separar una evaluación protegida antes de aprender estadísticas. Para clasificación, conservar proporciones cuando corresponda; para estructuras temporales o de grupos, no asumir que un split aleatorio es válido.
3. Construir una referencia: proceso vigente si existe; de lo contrario, una heurística simple calculada con training. Evaluarla con exactamente el mismo criterio que los modelos.
4. Crear un pipeline mínimo para faltantes, categorías y escala. Solo escalar las familias que lo requieren, pero mantener todo paso aprendido dentro del pipeline.
5. Entrenar un modelo lineal como referencia cuando sea compatible: OLS/Ridge para regresión o logística para clasificación.
6. Elegir pocas familias adicionales según señales del dataset: k-NN para baja dimensión densa; árbol pequeño para reglas y umbrales; SVM lineal/RBF para márgenes o no linealidad en tamaño manejable; Naive Bayes para una representación compatible; LDA/QDA solo si sus supuestos y tamaño de muestra son defendibles.
7. Comparar con las mismas particiones y evaluación. Registrar training y validación para distinguir subajuste de sobreajuste.
8. Posponer tuning hasta tener una línea base válida. Afinar solo hiperparámetros que controlan una hipótesis clara de complejidad.
9. Mantener el test interno o conjunto final intacto; una mejora sobre el leaderboard no sustituye una validación local coherente.
10. Documentar por qué se eligió la familia, qué preparación necesita, qué limitación permanece y qué evidencia supera al baseline.

**Recomendación para coding test.** Bajo presión, priorice una solución completa, reproducible y sin leakage frente a una búsqueda amplia. Dos o tres familias bien justificadas comunican mejor criterio que una lista larga comparada de forma inconsistente [G02, M06, D07].

## 14. Checklist operativo

- [ ] Confirmé si el objetivo es clasificación, regresión o conteo.
- [ ] Confirmé que la salida modelada coincide con la necesidad del problema.
- [ ] Separé evaluación antes de aprender imputación, categorías, escalas o variables.
- [ ] Revisé supuestos y requisitos de preparación de cada familia candidata.
- [ ] Establecí un proceso vigente, heurística o modelo simple como baseline.
- [ ] Comparé alternativas con los mismos datos, particiones y evaluación.
- [ ] Revisé rendimiento de entrenamiento y validación para detectar sobreajuste o subajuste.
- [ ] Ajusté solo los controles de complejidad relevantes y sin mirar el test final.
- [ ] Consideré interpretabilidad, coste de entrenamiento y coste de predicción.
- [ ] Evité interpretar coeficientes, selección o importancias como causalidad.
- [ ] Registré las razones, limitaciones y evidencia para elegir el modelo.
- [ ] Remití métricas, validación, feature engineering y tuning a sus documentos específicos.

## 15. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con modelos tabulares |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Define la tarea y el objetivo que debe resolver el modelo |
| `knowledge/eda/dataset_audit.md` | Describe la estructura y calidad de los datos |
| `knowledge/preprocessing/data_cleaning.md` | Prepara los datos antes del ajuste |
| `knowledge/preprocessing/data_leakage.md` | Evita fugas durante preparación y evaluación |
| `knowledge/feature_engineering/feature_engineering.md` | Describe transformaciones y representaciones de variables |
| `knowledge/validation/validation_strategy.md` | Define cómo comparar el rendimiento |
| `knowledge/metrics/classification_metrics.md` | Define métricas para modelos de clasificación |
| `knowledge/metrics/regression_metrics.md` | Define métricas para modelos de regresión |
| `knowledge/ensembles/boosting_models.md` | Desarrolla ensembles y boosting |
| `knowledge/tuning/hyperparameter_tuning.md` | Desarrolla la búsqueda de hiperparámetros |

## 16. Vacíos y limitaciones de las fuentes

- La cobertura es suficiente para una selección inicial, pero no ofrece un experimento común que compare todas las familias sobre los mismos datasets; cualquier ranking sería inventado.
- SVR solo aparece conceptualmente en el resumen de SVM; no hay respaldo suficiente para un procedimiento completo de preparación, tuning y comparación de regresión con SVM.
- Naive Bayes está cubierto con supuestos y variantes, pero no con una guía tabular unificada para variables mixtas continuas, binarias y de conteo en un mismo pipeline.
- LDA y QDA tienen criterios estadísticos de uso, pero no un ejemplo moderno seguro con preparación heterogénea comparable al pipeline de la sección 11.
- Poisson está cubierto, pero las soluciones a sobredispersión no se desarrollan; no se completan con conocimiento externo.
- Los resúmenes no respaldan que una sola estrategia de imputación sea adecuada para todos los modelos o dominios. La mediana/moda del código es un ejemplo mínimo, no una regla universal.
- El soporte de valores faltantes y categorías cambia entre implementaciones; las fuentes no permiten atribuir soporte nativo general a familias enteras.
- La calibración se menciona, pero no se desarrolla un procedimiento de calibración o comparación de probabilidades.
- MLP aparece con requisitos y limitaciones, pero se omite del desarrollo inicial por coste, sensibilidad y menor ajuste al objetivo de baseline rápido.
- Ensembles y boosting están ampliamente cubiertos, pero se excluyen deliberadamente del detalle y se remiten a `knowledge/ensembles/boosting_models.md`.
- Las recomendaciones para coding tests y datathons son derivaciones operativas; no representan resultados universales ni garantizan rendimiento de competición.
- La decisión final, el umbral, la métrica, la partición y los hiperparámetros requieren experimentación sobre el problema concreto.

## 17. Fuentes utilizadas

Las claves entre corchetes usadas en el texto remiten únicamente a los resúmenes consultados de las carpetas permitidas.

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| G02 — Géron, cap. 2 | [Resumen](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Encuadre, preparación, baseline lineal, comparación inicial, pipeline, CV, underfitting/overfitting |
| G04 — Géron, cap. 4 | [Resumen](../../sources/geron_hands_on_ml/ch04_training_models.md) | Regresión lineal/logística, Ridge, Lasso, Elastic Net, complejidad y learning curves |
| G06 — Géron, cap. 6 | [Resumen](../../sources/geron_hands_on_ml/ch06_decision_trees.md) | Árboles de clasificación/regresión, Gini, entropía, CART, regularización, coste e inestabilidad |
| M02 — Müller, cap. 2 | [Resumen](../../sources/muller_introduction_ml_python/ch02_supervised_learning.md) | k-NN, lineales, Naive Bayes, árboles, SVM, escala, generalización y coste |
| M06 — Müller, cap. 6 | [Resumen](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Pipelines, validación sin fuga y comparación de flujos completos |
| I03 — ISLP, cap. 3 | [Resumen](../../sources/islp_python/ch03_linear_regression.md) | OLS, supuestos, diagnósticos, categorías, interacciones, KNN de regresión y causalidad |
| I04 — ISLP, cap. 4 | [Resumen](../../sources/islp_python/ch04_classification.md) | Logística, LDA, QDA, Naive Bayes, KNN, Poisson, umbrales y supuestos |
| I06 — ISLP, cap. 6 | [Resumen](../../sources/islp_python/ch06_linear_model_selection_and_regularization.md) | Ridge/Lasso, alta dimensión, sesgo-varianza, selección y coste computacional |
| I08 — ISLP, cap. 8 | [Resumen](../../sources/islp_python/ch08_tree_based_methods.md) | Árboles, poda por complejidad, estabilidad y delimitación respecto de ensembles |
| I09 — ISLP, cap. 9 | [Resumen](../../sources/islp_python/ch09_support_vector_machines.md) | Margen, kernels, `C`, `gamma`, multiclase, coste y límite de SVR |
| P09 — Molin, cap. 9 | [Resumen](../../sources/molin_hands_on_pandas/ch09_ml_with_sklearn.md) | Preparación tabular, API de scikit-learn, baseline lineal/logístico y riesgos de leakage |
| D07 — ML Design Patterns, cap. 7 | [Resumen](../../sources/ml_design_patterns/ch07_responsible_ai_patterns.md) | Heuristic Benchmark, comparación equivalente y utilidad operativa |

## 18. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Tarea, generalización y ausencia de ganador universal | M02, I04 | Explícito de la fuente / Síntesis integrada |
| 2. Criterios de comparación | G02, G04, G06, M02, M06, I03, I04, I06, I09, D07 | Síntesis integrada |
| 3.1 Regresión lineal | G04, I03, P09 | Explícito de la fuente |
| 3.2 Regresión logística | G04, M02, I04, P09 | Explícito de la fuente |
| 3.3 Regularización | G04, M02, I06 | Explícito de la fuente / Conclusión práctica derivada |
| 4. k-NN | M02, I03, I04 | Explícito de la fuente / Conclusión práctica derivada |
| 5. Árboles de decisión | G06, M02, I08 | Explícito de la fuente / Síntesis integrada |
| 6. SVM | M02, I09 | Explícito de la fuente |
| 6. Naive Bayes | M02, I04 | Explícito de la fuente |
| 6. LDA, QDA y Poisson | I04 | Explícito de la fuente |
| 6. Delimitación de MLP y ensembles | M02, I08 | Síntesis integrada |
| 7. Preparación por familia | G02, G04, G06, M02, M06, I03, I04, I06, I08, I09, P09 | Síntesis integrada |
| 8. Complejidad, sobreajuste y subajuste | G04, G06, M02, I06, I08, I09 | Síntesis integrada |
| 9. Reglas de decisión | G04, G06, M02, I03, I04, I06, I08, I09, P09 | Conclusión práctica derivada |
| 10. Baseline heurístico y modelos de referencia | D07, G02, P09 | Explícito de la fuente / Síntesis integrada |
| 11. Código con pipeline | G02, G04, M06, I06, P09 | Conclusión práctica derivada |
| 12. Errores y riesgos | Todas las fuentes de la sección 17 | Síntesis integrada |
| 13. Flujo para coding test o datathon | G02, M02, M06, I03, I04, I06, I08, I09, P09, D07 | Recomendación para coding test |
| 14. Checklist | Todas las fuentes de la sección 17 | Recomendación para coding test |
| 15. Relación con otros documentos | Estructura solicitada y límites temáticos de los resúmenes | Síntesis integrada |
| 16. Vacíos y limitaciones | Cobertura observada en los resúmenes consultados | Síntesis integrada |
