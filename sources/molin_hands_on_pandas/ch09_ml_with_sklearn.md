---
title: "Getting Started with Machine Learning in Python"
topics:
  - "preprocesamiento y pipelines con scikit-learn"
  - "clustering con k-means"
  - "regresión lineal y evaluación"
  - "clasificación logística y métricas"
target_knowledge_files:
  - "knowledge/preprocessing/scikit_learn_pipelines.md"
  - "knowledge/models/kmeans_clustering.md"
  - "knowledge/models/linear_and_logistic_regression.md"
  - "knowledge/metrics/model_evaluation.md"
---

# 9. Getting Started with Machine Learning in Python

## 1. Propósito del capítulo

El capítulo introduce el vocabulario y el flujo mínimo de trabajo de Machine Learning con `scikit-learn`: reconocer el tipo de aprendizaje y de tarea, explorar los datos, separar entrenamiento y prueba, preprocesar sin fuga de información, ajustar un modelo y evaluarlo con métricas adecuadas.

El recorrido práctico cubre tres familias de problemas: clustering con k-means sobre datos sin etiqueta, regresión lineal para predecir una variable numérica y regresión logística para clasificación. Los ejemplos usan propiedades químicas de vinos y características orbitales de planetas. El objetivo es aprender un patrón de implementación reutilizable, no profundizar en la derivación matemática de los algoritmos.

## 2. Conceptos esenciales

### Tipos de aprendizaje y tareas

- **Aprendizaje supervisado:** dispone de etiquetas verdaderas y permite medir el rendimiento comparando predicciones con esas etiquetas.
- **Aprendizaje no supervisado:** no dispone de una respuesta correcta; busca estructura útil en los datos y es más difícil de optimizar objetivamente.
- **Aprendizaje por refuerzo:** aprende a responder a la retroalimentación del entorno; se menciona, pero queda fuera del alcance práctico del capítulo.
- **Clustering:** agrupa observaciones de modo que los miembros de un grupo sean similares entre sí y los grupos estén separados. En el caso no supervisado, los grupos deben interpretarse después.
- **Clasificación:** predice una etiqueta discreta. A diferencia del clustering, importa acertar la clase, no la proximidad espacial entre miembros.
- **Regresión:** predice un valor numérico y modela la fuerza y magnitud de las relaciones entre variables.

### API común de scikit-learn

- `fit(X, y)` aprende parámetros a partir de los datos; en aprendizaje no supervisado puede no requerir `y`.
- `transform(X)` lleva los datos a un nuevo espacio o aplica un preprocesamiento aprendido.
- `fit_transform(X)` combina ajuste y transformación; debe reservarse para entrenamiento o análisis no supervisado apropiado.
- `predict(X)` produce etiquetas o valores predichos.
- `predict_proba(X)` devuelve probabilidades por clase cuando el estimador lo soporta.
- `score(X, y)` aplica la métrica predeterminada del estimador; su significado cambia según el modelo.

### Generalización, ajuste y fuga

- **Overfitting:** el modelo aprende ruido o peculiaridades de la muestra y no generaliza a datos nuevos.
- **Underfitting:** el modelo o los datos disponibles no capturan la estructura relevante.
- **Data leakage:** información externa al conjunto de entrenamiento interviene en el ajuste. El ejemplo central es calcular la media o la escala con todo el dataset antes de dividirlo.
- La separación entrenamiento/prueba debe ocurrir antes de aprender imputaciones, escalas o cualquier otra transformación dependiente de los datos.
- El capítulo propone como regla orientativa reservar entre 10% y 30% para prueba, ajustando según el volumen de datos y el coste computacional. Para modelos que requieren tuning, anticipa una división adicional de validación, desarrollada en el capítulo siguiente.

### EDA orientado al modelado

Antes de modelar se deben revisar tipos, valores faltantes, escalas, distribución de la variable objetivo, balance de clases, correlaciones y diferencias de distribución entre clases. Las correlaciones pueden sugerir variables útiles, pero no establecen causalidad. La exploración debe incorporar conocimiento de dominio.

**Conclusión práctica derivada:** en un coding test, una inspección compacta con `shape`, `info()`, `describe()`, frecuencias del objetivo, correlaciones y unos pocos gráficos dirigidos suele detectar los riesgos que cambian el pipeline.

## 3. Flujo o procedimiento presentado

1. **Definir la tarea:** determinar si hay etiquetas y si la salida esperada es un grupo, una clase discreta o un valor numérico.
2. **Separar entradas y objetivo:** construir `X` con los predictores y `y` con la variable a predecir.
3. **Explorar los datos:** revisar tipos, faltantes, escalas, distribuciones, correlaciones, desbalance y posibles variables informativas.
4. **Dividir antes de ajustar transformaciones:** crear conjuntos de entrenamiento y prueba con `train_test_split()`; usar `stratify=y` cuando sea necesario preservar la proporción de clases.
5. **Diseñar el preprocesamiento:** decidir por columna si corresponde escalar, codificar, imputar, añadir un indicador de ausencia o aplicar una transformación matemática.
6. **Encapsular transformaciones:** usar `ColumnTransformer` cuando distintos grupos de columnas requieren tratamientos diferentes.
7. **Construir un pipeline:** encadenar preprocesamiento y estimador para que el mismo proceso se aplique durante entrenamiento y predicción.
8. **Ajustar solo con entrenamiento:** ejecutar `fit()` sobre `X_train` y `y_train`.
9. **Predecir sobre prueba:** usar `predict()` o `predict_proba()` según la evaluación requerida.
10. **Evaluar según el problema:** métricas internas para clustering; residuos y errores para regresión; matriz de confusión, precision, recall, F-score y curvas de umbral para clasificación.
11. **Interpretar en contexto:** comprobar utilidad de los clusters, plausibilidad de coeficientes, costes de falsos positivos y falsos negativos, y validez de los supuestos.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| No hay etiquetas y se buscan grupos similares | Clustering; k-means como método introductorio | Descubre estructura sin una respuesta conocida | Los clusters requieren interpretación y `k` cambia el resultado |
| El objetivo es numérico | Regresión | Modela magnitud y relaciones entre variables | Verificar supuestos, residuos y rango de aplicación |
| El objetivo es una etiqueta discreta | Clasificación; regresión logística como baseline | Produce clases y probabilidades | Elegir umbral y métrica según el coste de errores |
| Un modelo usa distancias y las variables tienen escalas distintas | `StandardScaler`, `MinMaxScaler` o `RobustScaler` | Evita que la escala domine la distancia | Ajustar el scaler solo con entrenamiento |
| Hay outliers y se requiere escalado | `RobustScaler` | Usa mediana e IQR, menos sensibles a outliers | El capítulo solo lo presenta brevemente; validar su efecto |
| Categoría binaria | Codificación 0/1 | Una columna representa ambas opciones | Documentar cuál clase es 1 |
| Categorías ordenadas | Codificación ordinal | Conserva el orden | Los enteros pueden imponer distancias artificiales |
| Categorías nominales sin orden | One-hot encoding | No inventa jerarquía | Evitar redundancia si el modelo o análisis lo exige |
| Faltantes con una estrategia de sustitución defendible | `SimpleImputer` | Permite completar por media, mediana, moda o constante | Una imputación inadecuada puede empeorar el modelo |
| La ausencia puede aportar señal o no hay imputación razonable | `MissingIndicator` | Conserva el patrón de faltantes como variable | No convierte por sí solo el dato ausente en un valor utilizable |
| Clases desbalanceadas | Split estratificado, pesos balanceados, precision/recall y curva PR | Accuracy y AUROC pueden ser optimistas | Elegir la clase positiva y el coste de cada error |
| Se desconoce `k` en k-means | Método del codo y conocimiento de dominio | Equilibra inercia y cantidad de clusters | El codo no garantiza grupos útiles |
| Se desea penalizar errores grandes en regresión | RMSE | Eleva al cuadrado los errores antes de promediar | Es sensible a outliers |
| Hay outliers en los residuos y se busca el error típico | Error absoluto mediano | Resume mejor el grueso de los errores | Puede ocultar errores extremos relevantes |
| Se comparan clasificadores a través de umbrales con clases razonablemente balanceadas | ROC/AUROC | Resume TPR frente a FPR | Bajo desbalance puede resultar optimista |
| La clase positiva es rara | Curva precision-recall, AUPR o AP | Centra la evaluación en recuperación y pureza de positivos | AUPR no se compara directamente entre datasets |

## 5. Métodos y técnicas

### Separación de entrenamiento y prueba

- **Objetivo:** estimar la capacidad de generalización sobre datos no usados para ajustar el modelo.
- **Entradas:** `X`, `y`, proporción de prueba, semilla y, opcionalmente, etiquetas para estratificar.
- **Salida:** `X_train`, `X_test`, `y_train`, `y_test` con filas alineadas.
- **Hiperparámetros relevantes:** `test_size`, `random_state`, `stratify`.
- **Ventajas:** evaluación más realista, reproducibilidad y preservación de clases con estratificación.
- **Limitaciones:** un único split puede ser variable; el capítulo no desarrolla validación cruzada.
- **Uso adecuado:** todo problema supervisado del capítulo.
- **No conviene:** no debe aplicarse mecánicamente al clustering puramente exploratorio presentado, donde no se pretende predecir etiquetas futuras.

### Escalado

- **Objetivo:** poner variables numéricas en escalas comparables.
- **Entradas:** matriz numérica.
- **Salida:** matriz transformada.
- **Métodos:** `StandardScaler` centra y escala con media y desviación estándar; `MinMaxScaler` lleva al intervalo `[0, 1]`; `RobustScaler` usa mediana e IQR.
- **Hiperparámetros:** `with_mean`, `with_std` para estandarización; rango objetivo en min-max.
- **Ventajas:** imprescindible cuando el algoritmo calcula distancias, como k-means.
- **Limitaciones:** la estandarización y min-max son sensibles, de distintas maneras, a distribuciones extremas; todo parámetro debe aprenderse con entrenamiento.
- **Casos adecuados:** variables con órdenes de magnitud distintos y modelos sensibles a escala.
- **Casos donde no conviene:** no escalar por rutina sin revisar requisitos del modelo.

### Codificación categórica

- **Objetivo:** convertir categorías en representación numérica.
- **Entradas:** columnas binarias, ordinales o nominales.
- **Salida:** una o varias columnas numéricas.
- **Funcionamiento:** binaria 0/1; ordinal mediante enteros con orden; one-hot mediante indicadores por categoría.
- **Ventajas:** hace utilizables variables no numéricas; one-hot evita asignar orden inexistente.
- **Limitaciones:** ordinal puede inducir relaciones aritméticas falsas; one-hot crea columnas adicionales y una puede ser redundante.
- **Casos adecuados:** codificación ordinal solo cuando el orden es real; one-hot para categorías nominales.
- **Casos donde no conviene:** no usar enteros ordinales en categorías sin jerarquía.

### Imputación e indicador de ausencia

- **Objetivo:** manejar datos faltantes.
- **Entradas:** columnas con valores ausentes.
- **Salida:** valores sustituidos o indicadores booleanos de ausencia.
- **Hiperparámetros de `SimpleImputer`:** `strategy` (`mean`, `median`, `most_frequent`, `constant`) y `fill_value`.
- **Ventajas:** integra el tratamiento de faltantes en un transformador consistente.
- **Limitaciones:** una estadística global puede ser una hipótesis pobre; en el ejemplo de planetas, la autora advierte que imputar puede ser peor que descartar filas para el objetivo concreto.
- **Casos adecuados:** cuando la estrategia de imputación está justificada y se aprende solo con entrenamiento.
- **Casos donde no conviene:** cuando el patrón físico o de selección hace inverosímil la sustitución disponible.

### `ColumnTransformer` y `Pipeline`

- **Objetivo:** aplicar tratamientos distintos por columnas y encadenarlos con el modelo.
- **Entradas:** transformadores, selectores de columnas y estimador final.
- **Salida:** un objeto con interfaz uniforme para ajustar, transformar y predecir.
- **Ventajas:** reproduce el mismo procesamiento en entrenamiento y prueba; reduce errores manuales y data leakage; permite acceder a pasos nombrados.
- **Limitaciones:** exige controlar el orden, los nombres y la compatibilidad de salidas.
- **Casos adecuados:** casi cualquier flujo con preprocesamiento dependiente de datos.
- **Casos donde no conviene:** no hay una contraindicación sustantiva en el capítulo; un pipeline trivial puede ser innecesario para una transformación puramente exploratoria.

### k-means

- **Objetivo:** dividir observaciones en `k` grupos según proximidad a centroides.
- **Entradas:** variables numéricas; en el ejemplo, semieje mayor y periodo orbital sin faltantes.
- **Salida:** etiqueta de cluster por fila, centroides, inercia y distancias a centroides.
- **Funcionamiento:** inicializa centroides y alterna asignación al centro más cercano con actualización de centros.
- **Hiperparámetros:** `n_clusters`, `init`, `n_init`, `max_iter`, `tol`, `random_state`.
- **Ventajas:** baseline simple, rápido e interpretable mediante centroides.
- **Limitaciones:** sensible a escala, inicialización y elección de `k`; un buen valor de inercia o separación no garantiza utilidad de negocio.
- **Casos adecuados:** segmentación o exploración cuando la distancia representa similitud.
- **Casos donde no conviene:** datos sin métrica de distancia significativa, faltantes sin resolver o cuando se esperan formas de cluster incompatibles con centroides.

### Regresión lineal por OLS

- **Objetivo:** predecir un valor continuo mediante una combinación lineal de predictores.
- **Entradas:** matriz `X` y objetivo continuo `y`.
- **Salida:** intercepto, coeficientes y predicciones.
- **Funcionamiento:** mínimos cuadrados ordinarios elige coeficientes que minimizan la suma de errores cuadrados.
- **Hiperparámetros:** el capítulo usa `LinearRegression` con valores predeterminados; el foco está en selección de variables y evaluación.
- **Ventajas:** baseline rápido y coeficientes interpretables cuando se cumplen supuestos.
- **Limitaciones:** no captura relaciones no lineales; extrapolar fuera del rango observado puede producir resultados físicamente imposibles; `R²` por sí solo no valida la especificación.
- **Casos adecuados:** relación aproximadamente lineal, residuos adecuados y predictores sin multicolinealidad grave.
- **Casos donde no conviene:** patrones claros en residuos, heteroscedasticidad severa, fuerte multicolinealidad o necesidad de extrapolar más allá del soporte de entrenamiento.

### Regresión logística

- **Objetivo:** estimar probabilidades de clases y convertirlas en etiquetas mediante un umbral.
- **Entradas:** `X` y etiqueta discreta `y`.
- **Salida:** probabilidades y clases predichas.
- **Funcionamiento:** aplica la función sigmoide a una combinación lineal; el capítulo la usa en clasificación binaria de vinos.
- **Hiperparámetros relevantes:** `C` (regularización), `class_weight`, `solver`, `max_iter`, `random_state` y el umbral aplicado a la probabilidad.
- **Ventajas:** baseline simple, probabilístico y compatible con evaluación por umbral.
- **Limitaciones:** el umbral predeterminado de 0.5 puede no representar los costes reales; la calidad depende de la señal presente en las variables.
- **Casos adecuados:** clasificación binaria o multiclase con una relación razonable entre predictores y log-odds.
- **Casos donde no conviene:** cuando la señal disponible no separa clases o el patrón requiere una frontera mucho más compleja sin ingeniería adicional.

## 6. Fórmulas esenciales

### Estandarización

\[
z_i = \frac{x_i - \mu}{\sigma}
\]

`x_i` es el valor original, `μ` la media y `σ` la desviación estándar aprendidas en entrenamiento. Mide cuántas desviaciones estándar se aleja un valor de la media. Valores cercanos a 0 están cerca del centro; el supuesto práctico es que media y escala del entrenamiento son representativas.

### Coeficiente de silueta

\[
s(i) = \frac{b(i)-a(i)}{\max(a(i), b(i))}
\]

`a(i)` es la distancia media de la observación `i` a los puntos de su propio cluster; `b(i)` es la menor distancia media a otro cluster. El resultado está en `[-1, 1]`: 1 indica buena separación, 0 solapamiento y -1 asignación probablemente incorrecta. Requiere que la distancia escogida sea significativa.

### Regresión lineal múltiple

\[
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_{i1} + \cdots + \hat{\beta}_p x_{ip}
\]

`ŷ_i` es la predicción; `β̂₀` el intercepto; `β̂_j` el cambio estimado en `y` por una unidad de `x_j`, manteniendo los demás predictores constantes; `p` es el número de regresores. Esa interpretación pierde validez con multicolinealidad fuerte o fuera del rango observado.

### Coeficiente de determinación y R² ajustado

\[
R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}
\]

`y_i` es el valor real, `ŷ_i` la predicción y `ȳ` la media del objetivo. Mide la proporción de varianza explicada; un valor alto no garantiza que el modelo sea correcto.

\[
R^2_{aj} = 1-(1-R^2)\frac{n-1}{n-p-1}
\]

`n` es el número de observaciones y `p` el número de regresores. Penaliza predictores adicionales que no aportan mejora suficiente.

### Errores de regresión

\[
MAE = \frac{1}{n}\sum_i |y_i-\hat{y}_i|
\]

MAE es el error absoluto medio en las unidades del objetivo; menor es mejor.

\[
RMSE = \sqrt{\frac{1}{n}\sum_i (y_i-\hat{y}_i)^2}
\]

RMSE también está en las unidades del objetivo, pero penaliza más los errores grandes. El error absoluto mediano usa la mediana de `|y_i-ŷ_i|` y describe mejor el error típico cuando unos pocos outliers dominan la media.

### Función sigmoide

\[
\sigma(z) = \frac{1}{1+e^{-z}}
\]

`z` es la combinación lineal de predictores. La salida pertenece a `[0, 1]` y se interpreta como probabilidad de la clase positiva; un umbral la transforma en etiqueta.

### Métricas de clasificación

Con `TP`, `TN`, `FP` y `FN` como verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos:

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}, \qquad Error = 1-Accuracy
\]

\[
Precision = \frac{TP}{TP+FP}, \qquad Recall = \frac{TP}{TP+FN}
\]

\[
F_1 = 2\frac{Precision \cdot Recall}{Precision+Recall}
\]

\[
F_\beta = (1+\beta^2)\frac{Precision \cdot Recall}{\beta^2 Precision+Recall}
\]

`β < 1` da más peso a precision; `β > 1`, a recall. La especificidad es `TN/(TN+FP)`. Accuracy es adecuada cuando las clases son comparables y los costes también; precision importa al limitar falsos positivos; recall al limitar falsos negativos.

## 7. Implementación práctica

### Pipeline de preprocesamiento y clasificación

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

preprocess = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_columns,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ]
)

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)
class_predictions = model.predict(X_test)
positive_probabilities = model.predict_proba(X_test)[:, 1]
```

**Nota práctica - adaptación moderna:** el capítulo ilustra los componentes por separado y usa una versión antigua de `scikit-learn`. El ejemplo anterior conserva su recomendación central - dividir primero y encapsular transformaciones - y añade `handle_unknown="ignore"` para que categorías no vistas en prueba no rompan el pipeline. También aumenta `max_iter` para reducir fallos de convergencia; ambas son adaptaciones, no afirmaciones originales del libro.

### k-means con escalado

```python
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cluster_model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, n_init="auto", random_state=42),
)
cluster_labels = cluster_model.fit_predict(cluster_features)
```

**Nota práctica - adaptación moderna:** el libro muestra `n_init=10` como valor de su versión. `n_init="auto"` corresponde a la API moderna; fijar `random_state` sigue siendo necesario para reproducibilidad.

### Regresión y evaluación de residuos

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

regressor = LinearRegression().fit(X_train, y_train)
predictions = regressor.predict(X_test)
residuals = y_test - predictions

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
```

Además de las métricas, el capítulo exige inspeccionar la distribución de residuos y su dispersión: deberían estar aproximadamente centrados en cero, sin patrón sistemático y con varianza similar.

### Métricas para clases desbalanceadas

```python
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

print(classification_report(y_test, class_predictions))
matrix = confusion_matrix(y_test, class_predictions)
precision, recall, thresholds = precision_recall_curve(
    y_test, positive_probabilities
)
average_precision = average_precision_score(y_test, positive_probabilities)
```

La elección del umbral debe seguir el coste de `FP` y `FN`, no conservar 0.5 por costumbre.

### Cambios de nombre relevantes

- **Nota práctica - API actual:** `calinski_harabaz_score`, usado en el capítulo, fue renombrado a `calinski_harabasz_score`.
- **Nota práctica - API actual:** en versiones modernas, `OneHotEncoder(sparse_output=False)` reemplaza el argumento antiguo `sparse=False` cuando se necesita una salida densa. Para modelado suele ser preferible conservar la matriz dispersa predeterminada.

## 8. Supuestos y condiciones de aplicación

1. La muestra de entrenamiento debe representar suficientemente los casos futuros; de lo contrario, el rendimiento de prueba no garantiza comportamiento en producción.
2. Las etiquetas supervisadas deben ser confiables. En el ejemplo de calidad del vino, la valoración humana puede ser subjetiva y limitar la señal aprendible.
3. Las variables empleadas por algoritmos de distancia deben estar en escalas comparables cuando sus unidades no deben determinar la importancia.
4. El significado de distancia debe ser válido para k-means y la elección de `k` debe responder a utilidad o conocimiento de dominio, no solo a una gráfica.
5. La regresión lineal supone residuos aproximadamente normales y homoscedásticos y ausencia de multicolinealidad elevada entre regresores.
6. La interpretación de un coeficiente exige poder mantener los demás regresores constantes.
7. Las predicciones lineales son más defendibles por interpolación, dentro de los rangos observados; la extrapolación puede violar restricciones físicas o del negocio.
8. Las métricas de clasificación exigen declarar la clase positiva y comprender el efecto del umbral.
9. Accuracy resulta informativa solo si el balance de clases y los costes de error la hacen pertinente.
10. Las transformaciones aprendidas - imputación, centro, escala y codificación - deben ajustarse exclusivamente con entrenamiento.

## 9. Errores y riesgos

- Explorar superficialmente los datos y modelar sin comprender faltantes, escalas, sesgo de etiquetas o desbalance.
- Usar correlación como prueba de causalidad.
- Escalar o imputar antes del split, introduciendo data leakage.
- Evaluar sobre el conjunto usado para entrenar y sobreestimar la generalización.
- Aplicar imputación por media, mediana o constante sin una justificación del dominio.
- Usar codificación ordinal en categorías nominales o aceptar sin análisis las distancias numéricas que introduce.
- Ejecutar k-means sin escalado cuando las magnitudes de las variables difieren.
- Elegir `k` únicamente por el método del codo; el ejemplo muestra que dos clusters pueden separar outliers y ser menos útiles que ocho grupos interpretables.
- Confundir números de cluster con etiquetas semánticas: los identificadores son arbitrarios.
- Concluir que una regresión es buena solo por correlación alta o `R²` alto; ninguno mide por sí mismo magnitud del error ni especificación correcta.
- Ignorar patrones, colas o heteroscedasticidad en residuos.
- Interpretar el intercepto o coeficientes fuera del dominio válido.
- Extrapolar fuera del rango de entrenamiento; el ejemplo planetario produce predicciones físicas imposibles.
- Añadir regresores solo para elevar `R²`; usar `R²` ajustado y validación conceptual.
- Usar error cuadrático logarítmico cuando existen valores o predicciones negativas.
- Confiar en accuracy con clases desbalanceadas; un clasificador trivial puede aparentar buen rendimiento.
- Confiar solo en AUROC bajo desbalance; puede resultar optimista por incluir verdaderos negativos.
- Ignorar que reducir falsos positivos normalmente aumenta falsos negativos, y viceversa.
- Comparar AUPR entre datasets con prevalencias distintas como si fueran directamente equivalentes.
- Incrementar datos de entrenamiento sin considerar retornos decrecientes y costes computacionales.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

Consultar este capítulo cuando el reto requiera construir rápidamente un baseline tabular con `scikit-learn`, especialmente si hay columnas mixtas, faltantes, desbalance o dudas sobre la métrica.

- **Decisiones que permite tomar:** distinguir clustering, regresión y clasificación; decidir si escalar; elegir codificación; justificar imputación o descarte; estratificar; seleccionar métricas y umbral.
- **Baseline viable:** split reproducible, EDA breve, `ColumnTransformer`, `Pipeline`, regresión lineal o logística, y un conjunto pequeño de métricas alineadas con el objetivo.
- **Mejoras con alto retorno:** analizar residuos; introducir pesos de clase; ajustar el umbral con la curva precision-recall; comparar `k` mediante inercia, silueta y utilidad; revisar variables cuya distribución difiere entre clases.
- **Qué evitar bajo presión:** desarrollar visualizaciones decorativas, implementar utilidades extensas ya disponibles, interpretar en exceso coeficientes con multicolinealidad o probar muchos algoritmos antes de asegurar un pipeline sin leakage.
- **Señales que activan técnicas:** escalas muy distintas activan escalado; categoría nominal activa one-hot; orden real activa codificación ordinal; faltantes activan una decisión explícita entre descarte, imputación e indicador; minoría pequeña activa split estratificado y métricas PR; residuos con patrón activan transformación o modelo no lineal; caída brusca de inercia activa revisión del codo.

**Recomendación para competencias:** mantener todo preprocesamiento dentro de un pipeline ofrece una defensa directa contra leakage y permite intercambiar el estimador sin alterar el tratamiento de datos. Esta recomendación deriva del patrón que el capítulo enseña, no de una comparación experimental de múltiples pipelines.

## 11. Checklist operativo

- [ ] Identificar si la tarea es clustering, regresión o clasificación.
- [ ] Separar `X` y `y`; definir claramente la clase positiva cuando corresponda.
- [ ] Revisar tipos, faltantes, escalas, outliers y distribución del objetivo.
- [ ] Hacer el split antes de ajustar cualquier transformación.
- [ ] Usar `stratify=y` si hay desbalance de clases.
- [ ] Justificar descarte, imputación o indicador de faltantes.
- [ ] Codificar nominales sin imponer orden y ordinales solo con orden real.
- [ ] Escalar si el algoritmo depende de distancias.
- [ ] Encapsular preprocesamiento y modelo en un pipeline.
- [ ] Fijar `random_state` en split y estimadores estocásticos.
- [ ] Evaluar clustering por separación y utilidad, no solo por inercia.
- [ ] Revisar residuos y magnitud de errores en regresión.
- [ ] Evitar accuracy como única métrica con clases desbalanceadas.
- [ ] Elegir el umbral según el coste de falsos positivos y falsos negativos.
- [ ] Confirmar que no se extrapola sin justificación.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Split reproducible, estratificación y prevención de leakage | `knowledge/validation/train_test_splitting.md` | Alta |
| EDA previa a modelos tabulares | `knowledge/eda/modeling_readiness_checks.md` | Alta |
| Escalado, codificación, imputación e indicadores | `knowledge/preprocessing/tabular_preprocessing.md` | Alta |
| `ColumnTransformer` y pipelines | `knowledge/preprocessing/scikit_learn_pipelines.md` | Alta |
| Selección de variables mediante distribuciones y correlación | `knowledge/feature_engineering/feature_screening.md` | Media |
| k-means, selección de `k` e interpretación de centroides | `knowledge/models/kmeans_clustering.md` | Alta |
| Silueta, Davies-Bouldin y Calinski-Harabasz | `knowledge/metrics/clustering_metrics.md` | Alta |
| Regresión lineal, supuestos e interpretación | `knowledge/models/linear_regression.md` | Alta |
| Residuos, R², R² ajustado, MAE y RMSE | `knowledge/metrics/regression_metrics.md` | Alta |
| Regresión logística, probabilidades y umbrales | `knowledge/models/logistic_regression.md` | Alta |
| Matriz de confusión, precision, recall y F-score | `knowledge/metrics/classification_metrics.md` | Alta |
| ROC/AUROC frente a curvas precision-recall | `knowledge/metrics/threshold_curves.md` | Alta |
| Coste de FP/FN y selección de umbral | `knowledge/problem_definition/error_costs.md` | Media |
| Restricciones de interpretabilidad y dominio | `knowledge/interpretability/model_constraints.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura y temas del capítulo, pp. 502–503 |
| 2. Tipos de aprendizaje y tareas | “Learning the lingo”, pp. 504–507 |
| 2. API común de scikit-learn | “Learning the lingo”, p. 507; tabla de métodos, p. 542 |
| 2. EDA orientado al modelado | “Exploratory data analysis”, pp. 507–519 |
| 2–3. Generalización, split y fuga | “Training and testing sets”, pp. 520–523; “Pipelines”, pp. 531–532 |
| 4–5. Escalado | “Scaling and centering data”, pp. 523–524 |
| 4–5. Codificación | “Encoding data”, pp. 525–528 |
| 4–5. Imputación y transformadores | “Imputing” y “Additional transformers”, pp. 528–530 |
| 4–5. `ColumnTransformer` y pipelines | pp. 530–532 |
| 4–5. k-means y selección de `k` | “Clustering”, “k-means” y método del codo, pp. 533–539 |
| 5. Centroides y espacio de distancias | pp. 539–541 |
| 5–6. Evaluación de clustering | pp. 542–544 |
| 5–8. Regresión lineal, supuestos e interpretación | pp. 544–548 |
| 6–9. Residuos y métricas de regresión | pp. 549–555 |
| 5. Regresión logística e implementación | pp. 555–559 |
| 6–9. Matriz de confusión y métricas de clasificación | pp. 559–567 |
| 6–9. ROC/AUROC | pp. 567–570 |
| 6–10. Precision-recall, AP y elección de umbral | pp. 571–574 |
| 10–11. Síntesis y checklist | Derivados del flujo completo; respaldo principal en el resumen, pp. 574–575 |
| Material reducido o excluido | Ejercicios y lecturas adicionales, pp. 575–578 |

