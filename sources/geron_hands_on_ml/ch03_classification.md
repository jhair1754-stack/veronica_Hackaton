---
title: "Classification"
topics:
  - "Binary, multiclass, multilabel and multioutput classification"
  - "Classification metrics and threshold selection"
  - "Cross-validation and error analysis"
target_knowledge_files:
  - "knowledge/metrics/classification_metrics.md"
  - "knowledge/validation/stratified_cross_validation.md"
  - "knowledge/models/classification_strategies.md"
  - "knowledge/eda/classification_error_analysis.md"
---

# 3. Classification

## 1. Propósito del capítulo

El capítulo presenta un flujo práctico para construir y evaluar sistemas de clasificación usando MNIST. Parte de un clasificador binario, desarrolla métricas y selección de umbral, extiende el problema a clasificación multiclase, y termina con clasificación multilabel y multioutput.

Su aporte principal es mostrar que entrenar un clasificador es solo una parte del trabajo: hay que generar predicciones fuera de muestra, elegir métricas coherentes con el coste de los errores, ajustar el umbral y analizar patrones de fallo antes de mejorar el modelo.

## 2. Conceptos esenciales

### 2.1 Tipos de clasificación

- **Binaria:** cada instancia pertenece a una de dos clases. El ejemplo es detectar `5` frente a `no 5`.
- **Multiclase (multiclass):** cada instancia recibe exactamente una clase entre más de dos opciones. MNIST usa las clases 0 a 9.
- **Multietiqueta (multilabel):** una instancia puede recibir varias etiquetas binarias simultáneas. El capítulo etiqueta cada dígito según si es grande y si es impar.
- **Multioutput-multiclase (multioutput):** se predicen varias salidas y cada una puede tomar más de dos valores. El ejemplo reconstruye los 784 valores de píxel de una imagen limpia.

### 2.2 Separación y validación

**Contenido del autor.** El conjunto de prueba debe apartarse antes de inspeccionar los datos en profundidad y reservarse para la evaluación final. MNIST ya viene dividido en 60 000 observaciones de entrenamiento y 10 000 de prueba. El conjunto de entrenamiento está mezclado, lo que ayuda a que los folds contengan todas las clases y evita secuencias largas de ejemplos similares. Mezclar no siempre es correcto; las series temporales son una excepción explícita.

La validación cruzada estratificada conserva aproximadamente la proporción de cada clase en cada fold. `cross_val_predict()` produce una predicción fuera de muestra para cada observación de entrenamiento: cada predicción procede de un modelo que no vio esa observación durante el ajuste.

### 2.3 Matriz de confusión

Las filas representan clases reales y las columnas clases predichas. En clasificación binaria:

- **TP (true positives):** positivos detectados correctamente.
- **TN (true negatives):** negativos detectados correctamente.
- **FP (false positives):** negativos predichos como positivos.
- **FN (false negatives):** positivos omitidos.

La diagonal contiene aciertos. Fuera de la diagonal aparecen confusiones específicas entre clases. Normalizar cada fila por el número de ejemplos de su clase permite comparar tasas de error sin que las clases abundantes dominen el análisis.

### 2.4 Métricas y desbalance

La exactitud (`accuracy`) puede ser engañosa cuando una clase es rara. En el ejemplo binario, predecir siempre `no 5` supera el 90 % de accuracy porque aproximadamente nueve de cada diez observaciones pertenecen a la clase negativa.

- **Precision:** de las predicciones positivas, qué fracción es correcta.
- **Recall, sensitivity o TPR:** de los positivos reales, qué fracción se detecta.
- **F1:** media armónica de precision y recall; solo es alta cuando ambas lo son.
- **FPR:** fracción de negativos reales clasificados como positivos.
- **Specificity o TNR:** fracción de negativos reales correctamente descartados.
- **ROC AUC:** resume el rendimiento del ranking a través de todos los umbrales; 1 representa separación perfecta y 0.5 el comportamiento de un clasificador aleatorio.

**Conclusión práctica derivada.** Una métrica agregada nunca sustituye la definición del coste de FP y FN. Un F1 alto favorece equilibrio, pero no sirve si el problema exige deliberadamente mucha más precision que recall, o viceversa.

### 2.5 Umbral y curvas

Un clasificador basado en score asigna la clase positiva cuando el score supera un umbral. Elevarlo tiende a aumentar precision y reducir recall; bajarlo tiende a aumentar recall y producir más falsos positivos. El capítulo obtiene scores fuera de muestra mediante `decision_function` y elige el menor umbral que alcanza una precision objetivo.

La curva precision-recall (PR) muestra el intercambio entre ambas métricas. La curva ROC muestra TPR frente a FPR para todos los umbrales. El capítulo recomienda preferir PR cuando la clase positiva es rara o importan especialmente los falsos positivos; ROC es adecuada en los demás casos.

### 2.6 Estrategias multiclase

- **One-versus-the-rest (OvR):** entrena un clasificador binario por clase y elige el score mayor. Para `N` clases requiere `N` modelos.
- **One-versus-one (OvO):** entrena un clasificador por cada par de clases y decide por duelos. Requiere `N(N-1)/2` modelos, pero cada uno usa solo datos de dos clases.
- Algunos algoritmos admiten multiclase de forma nativa, como SGD, Random Forest y naive Bayes en la exposición del capítulo.
- Otros algoritmos binarios se combinan mediante OvR u OvO. Para métodos que escalan mal con el número de observaciones, como SVC en el ejemplo, OvO puede ser preferible; para la mayoría de clasificadores binarios, el capítulo prefiere OvR.

## 3. Flujo o procedimiento presentado

1. Cargar `X` e `y`; comprobar formas, tipos y significado de las features y etiquetas.
2. Apartar el test set antes de explorar en profundidad.
3. Definir el tipo de salida: binaria, multiclase, multilabel o multioutput.
4. Construir un primer clasificador reproducible con `random_state` cuando el estimador sea estocástico.
5. Obtener métricas o predicciones fuera de muestra con validación cruzada estratificada.
6. Comparar el modelo con un baseline trivial; no interpretar accuracy sin conocer la distribución de clases.
7. Examinar la matriz de confusión y calcular precision, recall y F1.
8. Si el modelo expone scores o probabilidades, trazar PR y/o ROC y elegir el umbral según el coste de FP y FN.
9. Comparar modelos con scores obtenidos bajo el mismo esquema de validación.
10. En multiclase, revisar scores por clase y la matriz de confusión normalizada por clase real.
11. Inspeccionar los pares de clases y ejemplos con más errores; formular mejoras de datos, features o preprocesamiento.
12. Mantener el test set intacto hasta seleccionar el sistema final.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Clases desbalanceadas | Matriz de confusión, precision, recall y F1 | Accuracy puede parecer alta con un predictor trivial | Comparar contra la prevalencia y un baseline simple |
| Positivo raro o alto interés en falsos positivos | Curva PR | Hace visible la degradación entre precision y recall | Reportar ambas métricas en el umbral elegido |
| Positivos no raros y comparación global de rankings | Curva ROC y ROC AUC | Resume TPR frente a FPR en todos los umbrales | ROC AUC puede parecer optimista con positivos raros |
| Se exige precision mínima | Seleccionar umbral sobre scores fuera de muestra | Permite imponer una restricción operativa | Confirmar el recall resultante; precision alta con recall ínfimo no es útil |
| Se exige capturar casi todos los positivos | Reducir el umbral y optimizar recall | Disminuye falsos negativos | Aceptar y cuantificar el aumento de falsos positivos |
| Evaluación antes del modelo final | `cross_val_predict()` | Genera una predicción no entrenada sobre cada fila | No sustituye la evaluación final en test |
| Muchas clases y clasificador binario común | OvR | Solo requiere un modelo por clase | Comparar scores coherentes entre clasificadores |
| Algoritmo que escala mal con muchas filas, como SVC | OvO | Cada modelo se entrena con solo dos clases | El número de modelos crece cuadráticamente |
| Métricas multilabel con etiquetas igualmente importantes | F1 macro | Da el mismo peso a cada etiqueta | No refleja diferencias de prevalencia o importancia |
| Etiquetas multilabel con soportes distintos | F1 ponderado por soporte | Da más peso a etiquetas frecuentes | Una etiqueta rara pero crítica puede quedar ocultada |
| Matriz de confusión dominada por clases abundantes | Normalización por fila | Compara tasas de error por clase | Conservar también los conteos para conocer el volumen real |
| Datos temporales | División que respete el tiempo | Mezclar puede incorporar información futura | No aplicar shuffling o K-fold aleatorio sin justificarlo |

## 5. Métodos y técnicas

### 5.1 `SGDClassifier`

- **Objetivo:** baseline eficiente para clasificación binaria o multiclase.
- **Entradas:** matriz `X` de observaciones por features y vector de clases `y`.
- **Salida:** clase predicha y scores mediante `decision_function()`.
- **Funcionamiento:** optimiza un modelo de forma estocástica, procesando observaciones individualmente; esto favorece datasets grandes y aprendizaje online.
- **Hiperparámetros destacados por el capítulo:** `random_state` para reproducibilidad. El capítulo no desarrolla aquí la optimización del resto.
- **Ventajas:** rápido, escalable, scores accesibles y soporte multiclase directo.
- **Limitaciones:** es sensible a la escala; en MNIST el escalado mejora claramente el accuracy. Como modelo lineal sobre píxeles, es sensible a desplazamientos y rotaciones y confunde clases que difieren en pocos píxeles.
- **Casos adecuados:** baseline rápido, datasets grandes y evaluación de umbrales.
- **No conviene usarlo sin más:** cuando la frontera requiere invariancia espacial o relaciones no lineales que las features no representan.

### 5.2 `RandomForestClassifier`

- **Objetivo:** clasificación binaria o multiclase mediante un ensamble de árboles.
- **Entradas:** `X` e `y`.
- **Salida:** clases y probabilidades por clase con `predict_proba()`.
- **Funcionamiento relevante aquí:** usa la probabilidad de la clase positiva como score para PR/ROC.
- **Hiperparámetro mostrado:** `random_state`.
- **Ventajas:** soporta multiclase directamente y, en el ejemplo binario, supera ampliamente la ROC del SGD.
- **Limitaciones:** el capítulo no desarrolla su coste, ajuste ni calibración; una probabilidad de `predict_proba()` se usa como score, no se demuestra que esté calibrada.
- **Casos adecuados:** comparación de un modelo no lineal contra un baseline lineal.
- **No conviene usarlo como elección automática:** cuando solo se dispone de la AUC del ejemplo; la selección debe validarse en los datos del problema.

### 5.3 Validación cruzada estratificada

- **Objetivo:** estimar rendimiento manteniendo proporciones de clase.
- **Entradas:** estimador, `X`, `y`, número de folds y métrica o método de predicción.
- **Salida:** scores por fold (`cross_val_score`) o predicciones/scores fuera de muestra (`cross_val_predict`).
- **Funcionamiento:** cada fold se predice con un clon entrenado en los folds restantes.
- **Hiperparámetro:** `n_splits` o `cv`.
- **Ventajas:** usa eficientemente el conjunto de entrenamiento y evita evaluar cada fila con el mismo modelo que la ajustó.
- **Limitaciones:** cuesta varios entrenamientos; no respeta por sí sola dependencias temporales o agrupadas.
- **Caso adecuado:** selección inicial de modelos y análisis de errores.
- **No conviene:** si el esquema de partición contradice la estructura real de los datos.

### 5.4 Selección de umbral

- **Objetivo:** convertir scores continuos en decisiones acordes al coste de error.
- **Entradas:** etiquetas reales y scores fuera de muestra.
- **Salida:** umbral y predicciones binarias.
- **Funcionamiento:** calcula precision y recall para umbrales posibles y elige un punto operativo.
- **Hiperparámetro operativo:** el propio umbral o la restricción mínima de precision/recall.
- **Ventajas:** adapta un modelo entrenado a la necesidad del proyecto sin reentrenarlo.
- **Limitaciones:** optimizar y reportar sobre las mismas predicciones produce una estimación optimista si no se valida después; el capítulo conserva test para la comprobación final.
- **Caso adecuado:** alertas, filtros y detección donde FP y FN tienen costes distintos.
- **No conviene:** fijarlo solo por una cifra atractiva sin revisar la métrica complementaria.

### 5.5 OvR y OvO

- **Objetivo:** extender clasificadores binarios a multiclase.
- **Entradas:** clasificador base, `X` y etiquetas multiclase.
- **Salida:** una clase por instancia.
- **Funcionamiento:** OvR elige entre clasificadores clase-contra-resto; OvO agrega duelos entre pares.
- **Hiperparámetro conceptual:** estrategia de descomposición.
- **Ventajas:** reutiliza estimadores binarios. OvR usa pocos modelos; OvO reduce los datos por modelo.
- **Limitaciones:** OvO requiere un número cuadrático de clasificadores; OvR entrena cada clasificador con todo el dataset.
- **Casos adecuados:** OvO para algoritmos caros en número de filas; OvR como opción general.
- **No conviene:** forzar una estrategia sin considerar si el algoritmo ya maneja multiclase de forma nativa.

### 5.6 Clasificación multilabel con KNN

- **Objetivo:** predecir varios indicadores binarios por instancia.
- **Entradas:** `X` y una matriz objetivo de forma `(n_instancias, n_etiquetas)`.
- **Salida:** vector de etiquetas por instancia.
- **Funcionamiento:** `KNeighborsClassifier` admite varias columnas objetivo.
- **Hiperparámetros:** el capítulo no los ajusta en este ejemplo; en ejercicios menciona `n_neighbors` y `weights` para MNIST.
- **Ventajas:** demuestra una implementación directa de multilabel.
- **Limitaciones:** no todos los clasificadores soportan multilabel; la validación cruzada sobre MNIST con KNN es computacionalmente lenta.
- **Casos adecuados:** varias propiedades simultáneas y dataset compatible con vecinos.
- **No conviene:** cuando el coste de predicción por distancia o el volumen del dataset es prohibitivo.

### 5.7 Clasificación multioutput

- **Objetivo:** predecir múltiples valores discretos por instancia.
- **Entradas:** observación y matriz objetivo con varias columnas multicategoría.
- **Salida:** un valor por salida; en el ejemplo, 784 intensidades de píxel.
- **Funcionamiento:** KNN aprende a mapear imágenes con ruido a imágenes originales.
- **Ventajas:** representa salidas estructuradas de forma sencilla.
- **Limitaciones:** la frontera entre clasificación y regresión puede ser ambigua cuando la salida es numérica; el ejemplo llega a mirar el test antes de la evaluación y el autor lo señala como una práctica incorrecta.
- **Casos adecuados:** problemas con varias salidas discretizadas.
- **No conviene:** asumir que toda salida numérica entera debe tratarse como clase.

## 6. Fórmulas esenciales

### Accuracy

\[
\mathrm{accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

Mide la proporción total de aciertos. Puede ocultar un modelo inútil si una clase domina.

### Precision

\[
\mathrm{precision}=\frac{TP}{TP+FP}
\]

Mide la fiabilidad de las predicciones positivas. Aumentarla suele reducir recall.

### Recall o TPR

\[
\mathrm{recall}=\mathrm{TPR}=\frac{TP}{TP+FN}
\]

Mide la cobertura de la clase positiva. Es prioritaria cuando omitir positivos es costoso.

### F1

\[
F_1=\frac{2\,\mathrm{precision}\,\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}
=\frac{2TP}{2TP+FP+FN}
\]

Es la media armónica de precision y recall. Penaliza que una de las dos sea baja y supone que ambas importan de forma comparable.

### FPR y specificity

\[
\mathrm{FPR}=\frac{FP}{FP+TN}=1-\mathrm{TNR}
\]

\[
\mathrm{specificity}=\mathrm{TNR}=\frac{TN}{TN+FP}
\]


FPR mide qué fracción de negativos se marca incorrectamente como positiva. Specificity mide qué fracción se descarta correctamente. La curva ROC grafica TPR contra FPR.

### Número de clasificadores OvO

\[
n_{OvO}=\frac{N(N-1)}{2}
\]

`N` es el número de clases. Expresa el crecimiento cuadrático del número de duelos; con 10 clases se entrenan 45 clasificadores.

## 7. Implementación práctica

### 7.1 Evaluación binaria sin tocar test

```python
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
classifier = SGDClassifier(random_state=42)

validation_predictions = cross_val_predict(
    classifier,
    X_train,
    y_train_is_five,
    cv=cv,
)

print(confusion_matrix(y_train_is_five, validation_predictions))
print(precision_score(y_train_is_five, validation_predictions))
print(recall_score(y_train_is_five, validation_predictions))
print(f1_score(y_train_is_five, validation_predictions))
```

**Nota práctica - adaptación moderna.** El código del capítulo pasa `random_state` a `StratifiedKFold` sin activar `shuffle`. En APIs actuales, `random_state` solo tiene efecto cuando `shuffle=True`; aquí se activa explícitamente. No debe hacerse en series temporales.

### 7.2 Selección de un umbral de precision

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

validation_scores = cross_val_predict(
    classifier,
    X_train,
    y_train_is_five,
    cv=cv,
    method="decision_function",
)

precisions, recalls, thresholds = precision_recall_curve(
    y_train_is_five,
    validation_scores,
)

valid_positions = np.flatnonzero(precisions[:-1] >= 0.90)
if valid_positions.size == 0:
    raise ValueError("No threshold reaches the required precision")

selected_threshold = thresholds[valid_positions[0]]
validation_predictions = validation_scores >= selected_threshold
```

**Nota práctica - adaptación moderna.** Se excluye el último elemento de `precisions` porque `precision_recall_curve()` devuelve una precision y un recall más que el número de thresholds. También se controla el caso en que ningún umbral cumple el objetivo.

### 7.3 Escalado dentro de cada fold

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

scaled_sgd = make_pipeline(
    StandardScaler(),
    SGDClassifier(random_state=42),
)

cv_accuracy = cross_val_score(
    scaled_sgd,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
)
```

**Nota práctica - adaptación moderna.** El capítulo escala todo `X_train` antes de la validación cruzada. Encapsular el escalador en un `Pipeline` hace que se ajuste solo con el subconjunto de entrenamiento de cada fold y evita leakage entre folds.

### 7.4 Matriz de errores multiclase normalizada

```python
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

validation_predictions = cross_val_predict(scaled_sgd, X_train, y_train, cv=cv)
error_matrix = confusion_matrix(y_train, validation_predictions, normalize="true")
np.fill_diagonal(error_matrix, 0.0)
```

**Nota práctica - adaptación moderna.** `normalize="true"` reemplaza el cálculo manual de sumas por fila y expresa directamente la tasa condicional por clase real.

### 7.5 Carga reproducible de MNIST

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data
y = mnist.target.astype("uint8")
```

**Nota práctica - adaptación moderna.** `as_frame=False` conserva arrays NumPy y permite aplicar directamente operaciones como `reshape`, tal como hace el capítulo. `parser="auto"` corresponde a la API actual y no cambia la idea original.

## 8. Supuestos y condiciones de aplicación

1. Las etiquetas verdaderas son fiables; métricas y análisis de errores pierden sentido si el target contiene ruido no identificado.
2. Los folds deben representar el escenario de generalización. La estratificación conserva proporciones de clase, pero no resuelve dependencia temporal, grupos o duplicados.
3. El test set permanece sin usar durante exploración, ajuste de umbral y selección de modelo.
4. Precision, recall, PR y ROC requieren definir cuál es la clase positiva.
5. Los scores usados para seleccionar umbral deben ser comparables y generados fuera de muestra.
6. La interpretación de F1 supone que precision y recall merecen un equilibrio; si sus costes difieren, hay que inspeccionarlas por separado.
7. La comparación ROC AUC presupone el mismo conjunto de evaluación y la misma definición de positivo.
8. El escalado puede ser necesario para modelos sensibles a magnitudes, como el SGD presentado.
9. OvO y OvR presuponen que el estimador base y sus scores permiten resolver cada subproblema binario.
10. En multilabel, la forma de promediar la métrica debe reflejar la importancia y soporte de las etiquetas.
11. En imágenes, un modelo lineal sobre píxeles no incorpora por sí mismo invariancia a traslación o rotación.

## 9. Errores y riesgos

- Evaluar con el test set antes de terminar la selección del modelo.
- Concluir que un clasificador es bueno por accuracy alta en datos desbalanceados.
- Ajustar transformaciones con todos los datos antes de separar folds; el ejemplo de escalado del capítulo debe modernizarse con `Pipeline` para evitar leakage de validación.
- Elegir un umbral sobre predicciones de entrenamiento en muestra en vez de scores fuera de muestra.
- Presumir que 99 % de precision es útil sin preguntar cuál es el recall.
- Usar ROC AUC como única evidencia cuando la clase positiva es rara.
- Leer solo el valor total de la matriz de confusión y no los pares de clases confundidos.
- Comparar conteos absolutos por clase sin normalizar cuando los soportes difieren.
- Suponer que la matriz de confusión es simétrica; confundir A con B no implica la misma tasa que confundir B con A.
- Interpretar índices de `decision_function()` como etiquetas sin consultar `classes_`.
- Mezclar series temporales o datos agrupados como si fueran observaciones independientes.
- Aplicar una métrica macro multilabel cuando las etiquetas no tienen igual importancia, o una ponderada cuando una etiqueta rara es crítica.
- Ignorar el coste computacional: KNN multilabel con validación cruzada sobre 60 000 imágenes puede tardar mucho; OvO crece como `N(N-1)/2`.
- Confundir score o probabilidad usada para ranking con una probabilidad necesariamente calibrada.
- Analizar demasiados errores individuales bajo presión de tiempo antes de revisar patrones agregados.
- Introducir data augmentation en el test set; los desplazamientos sugeridos en los ejercicios pertenecen solo al entrenamiento.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Consulta este capítulo cuando el target sea categórico, exista desbalance, haya que justificar una métrica, seleccionar un umbral o diagnosticar confusiones entre clases.

- **Decisiones que permite tomar:** métrica principal; definición de clase positiva; esquema estratificado; necesidad de umbral; PR frente a ROC; OvR frente a OvO; promedio macro o ponderado en multilabel.
- **Baseline viable:** partición con test reservado, `SGDClassifier(random_state=42)` dentro de un pipeline de escalado, validación cruzada estratificada, matriz de confusión y precision/recall/F1. Un predictor de clase mayoritaria sirve como control mínimo.
- **Mejoras con mayor retorno:** ajustar el umbral con scores fuera de muestra, probar un modelo no lineal, normalizar la matriz por clase, inspeccionar los principales pares de confusión y diseñar features o preprocesamiento dirigidos a esos errores.
- **Qué evitar bajo presión:** implementar manualmente K-fold si las utilidades estándar bastan; entrenar todos los esquemas OvO/OvR sin necesidad; revisar cientos de errores individuales; ejecutar KNN con CV completa en datasets grandes sin estimar coste; hacer augmentation antes de disponer de un baseline medido.
- **Señales activadoras:** clase positiva rara -> PR, precision, recall y F1; costes asimétricos -> ajuste de umbral; muchas clases -> matriz normalizada y estrategia multiclase; varias etiquetas por fila -> multilabel; varias salidas multicategoría -> multioutput; errores entre clases visualmente similares -> preprocesamiento, features o augmentation del entrenamiento.

**Recomendación para competencias derivada del capítulo.** Antes de optimizar el leaderboard, reproduce localmente la métrica oficial con predicciones fuera de fold. Si la métrica admite probabilidades o ranking, conserva los scores; si exige clases, trata el umbral como una decisión validable. Usa la matriz de confusión para dirigir el siguiente experimento, no una búsqueda indiscriminada.

## 11. Checklist operativo

- [ ] Identificar si el problema es binario, multiclase, multilabel o multioutput.
- [ ] Separar y congelar el test set antes de explorar y ajustar.
- [ ] Comprobar prevalencias y construir un baseline trivial.
- [ ] Elegir un splitter compatible con clases, tiempo y grupos.
- [ ] Encapsular preprocesamiento y modelo en un pipeline.
- [ ] Generar predicciones o scores fuera de muestra.
- [ ] Revisar matriz de confusión, precision, recall y F1.
- [ ] Elegir PR o ROC según prevalencia y coste de errores.
- [ ] Seleccionar el umbral y verificar la métrica complementaria.
- [ ] En multiclase, normalizar la matriz por clase real.
- [ ] Inspeccionar los pares de confusión con mayor tasa.
- [ ] Estimar el coste computacional antes de KNN u OvO extensivos.
- [ ] Evaluar una sola vez en test tras cerrar las decisiones.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Matriz de confusión, precision, recall, F1, FPR y specificity | `knowledge/metrics/classification_metrics.md` | Alta |
| Curvas PR, ROC, AUC y selección de umbral | `knowledge/metrics/classification_thresholds.md` | Alta |
| Predicciones fuera de muestra y validación estratificada | `knowledge/validation/stratified_cross_validation.md` | Alta |
| Prevención de leakage mediante pipelines | `knowledge/validation/leakage_prevention.md` | Alta |
| SGD y efecto del escalado | `knowledge/models/sgd_classifier.md` | Media |
| Estrategias OvR y OvO | `knowledge/models/multiclass_strategies.md` | Alta |
| Clasificación multilabel y multioutput | `knowledge/models/multilabel_multioutput.md` | Media |
| Matriz normalizada y análisis dirigido de errores | `knowledge/eda/classification_error_analysis.md` | Alta |
| Preprocesamiento y augmentation guiados por errores de imagen | `knowledge/feature_engineering/image_augmentation.md` | Media |
| Ajuste de `n_neighbors` y `weights` sugerido en ejercicios | `knowledge/tuning/knn_tuning.md` | Baja |

## 13. Trazabilidad


| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | “Classification”, “Performance Measures” y cierre del capítulo, pp. 85, 88 y 108 |
| 2.1 Tipos de clasificación | “Training a Binary Classifier”, “Multiclass Classification”, “Multilabel Classification” y “Multioutput Classification”, pp. 88, 100-107 |
| 2.2 Separación y validación | “MNIST” y “Measuring Accuracy Using Cross-Validation”, pp. 85-90 |
| 2.3 Matriz de confusión | “Confusion Matrix” y “Error Analysis”, pp. 90-105 |
| 2.4 Métricas y desbalance | “Measuring Accuracy Using Cross-Validation” y “Precision and Recall”, pp. 89-93 |
| 2.5 Umbral y curvas | “Precision/Recall Trade-off” y “The ROC Curve”, pp. 93-99 |
| 2.6 Estrategias multiclase | “Multiclass Classification”, pp. 100-102 |