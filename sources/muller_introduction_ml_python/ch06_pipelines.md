---
title: "Algorithm Chains and Pipelines"
topics:
  - "Pipelines de scikit-learn"
  - "Prevención de data leakage"
  - "Búsqueda conjunta de preprocesamiento y modelos"
target_knowledge_files:
  - "knowledge/preprocessing/pipelines.md"
  - "knowledge/validation/data_leakage.md"
  - "knowledge/tuning/pipeline_grid_search.md"
  - "knowledge/models/model_selection.md"
---

# 6. Algorithm Chains and Pipelines

## 1. Propósito del capítulo

El capítulo explica cómo encapsular transformaciones y modelos en un único estimador mediante `Pipeline`. Su objetivo principal es conservar el orden correcto de ajuste y transformación, evitar fuga de información (*data leakage*) durante validación cruzada y permitir que `GridSearchCV` optimice conjuntamente el preprocesamiento y el modelo.

## 2. Conceptos esenciales

- **Cadena de algoritmos:** una aplicación real suele combinar extracción o selección de características, escalado y un estimador final, no un algoritmo aislado.
- **Pipeline:** objeto que presenta la interfaz habitual de scikit-learn (`fit`, `predict`, `score` y, cuando corresponde, `transform`) mientras ejecuta internamente una secuencia de pasos.
- **Ajuste sin fuga:** cualquier operación que aprenda información de los datos debe ajustarse solo con la parte de entrenamiento de cada partición. La validación cruzada debe ser el bucle exterior del proceso.
- **Transformadores intermedios:** todos los pasos salvo el último deben implementar `transform` y, normalmente, `fit` o `fit_transform`. El último paso solo necesita `fit`; sus otros métodos determinan qué métodos expone el pipeline.
- **Espacio de parámetros:** un parámetro interno se referencia como `<paso>__<parametro>`. Por ejemplo, `svm__C` identifica `C` en el paso llamado `svm`.
- **Inspección:** `named_steps` permite acceder a un paso ajustado; tras una búsqueda, el pipeline ganador está en `best_estimator_`.

**Contenido explícito del autor:** el escalado aprendido antes de la validación cruzada utiliza información de los folds de validación y hace que la evaluación deje de representar datos realmente nuevos. La selección o extracción de características fuera del bucle puede producir distorsiones mucho mayores que el escalado.

**Conclusión práctica derivada:** toda transformación dependiente de los datos debe incluirse en el objeto entregado a `cross_val_score` o `GridSearchCV`, aunque no use `y` explícitamente.

## 3. Flujo o procedimiento presentado

1. Separar una vez los datos de desarrollo y el conjunto de prueba final.
2. Definir una secuencia ordenada de transformadores y un estimador final.
3. Construir el `Pipeline` con nombres explícitos o con `make_pipeline`.
4. Definir los hiperparámetros mediante la sintaxis `<paso>__<parametro>`.
5. Entregar el pipeline completo a `GridSearchCV` o `cross_val_score`.
6. En cada fold, dejar que el pipeline ajuste los transformadores solo con el fold de entrenamiento, transforme entrenamiento y validación, y ajuste/evalúe el estimador final.
7. Seleccionar la mejor configuración con la puntuación de validación cruzada.
8. Evaluar una sola vez el pipeline seleccionado sobre el conjunto de prueba reservado.
9. Si hace falta interpretar el modelo, acceder al estimador ajustado mediante `best_estimator_.named_steps[...]`.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Hay escalado, selección o extracción de características antes del modelo | Incluir todos los pasos en `Pipeline` | Cada transformación se ajusta únicamente con el fold de entrenamiento | No transformar todo el conjunto antes de la validación cruzada |
| Se deben ajustar parámetros del transformador y del modelo | `GridSearchCV` sobre el pipeline | Evalúa cada combinación como un flujo completo | El número de modelos crece con todas las combinaciones |
| Se comparan familias de modelos con necesidades distintas | Lista de diccionarios en `param_grid` y reemplazo de pasos | Permite asociar a cada modelo su preprocesamiento y sus parámetros | Evitar una búsqueda exhaustiva sin criterio |
| Un modelo no necesita un paso de preprocesamiento | Establecer ese paso en `None` dentro de la búsqueda | Permite omitirlo sin cambiar la estructura general | La omisión debe corresponder a la familia de modelo evaluada |
| Se necesitan nombres semánticos o hay pasos repetidos | Usar `Pipeline` con nombres explícitos | Facilita configurar e inspeccionar los pasos | Los nombres no pueden contener `__` |
| La cadena es simple y no requiere nombres personalizados | Usar `make_pipeline` | Reduce código y genera nombres automáticamente | Verificar los nombres generados antes de construir el grid |

## 5. Métodos y técnicas

### `Pipeline`

- **Objetivo:** encapsular una cadena reproducible de transformaciones y un estimador final.
- **Entradas:** lista ordenada de tuplas `(nombre, estimador)`; durante `fit`, recibe `X` y, cuando corresponde, `y`.
- **Salida:** un estimador compuesto; `predict`, `score` o `transform` delegan en el último paso después de transformar secuencialmente `X`.
- **Funcionamiento:** para cada paso intermedio ejecuta ajuste y transformación; sobre el último ejecuta `fit`. Al predecir, aplica `transform` sin reajustar y llama al método final.
- **Hiperparámetros:** todos los parámetros expuestos por sus pasos, direccionados con `nombre__parametro`.
- **Ventajas:** reduce errores de orden, evita olvidar transformaciones en test, integra correctamente validación y búsqueda, y mantiene una interfaz uniforme.
- **Limitaciones:** los pasos intermedios deben transformar los datos; una cadena o grid demasiado amplio aumenta el coste y la complejidad.
- **Casos adecuados:** clasificación, regresión, clustering o cadenas solo de transformación, siempre que los estimadores cumplan la interfaz.
- **Casos donde no conviene usarlo:** el capítulo no identifica una contraindicación general; sí desaconseja sobrecomplicar el flujo con componentes cuya utilidad no se evalúa.

### `make_pipeline`

- **Objetivo:** crear rápidamente un pipeline sin asignar nombres manuales.
- **Entradas:** estimadores en el orden de ejecución.
- **Salida:** `Pipeline` con nombres derivados de las clases en minúsculas; añade sufijos numéricos si una clase aparece varias veces.
- **Ventajas:** sintaxis breve.
- **Limitaciones:** los nombres automáticos pueden ser menos claros, especialmente con pasos repetidos.
- **Caso adecuado:** cadenas cortas y sin ambigüedad semántica.
- **Caso donde no conviene usarlo:** cuando se necesita controlar nombres para grids complejos o inspección frecuente.

### `GridSearchCV` aplicado a pipelines

- **Objetivo:** seleccionar hiperparámetros de transformadores y modelos, e incluso reemplazar pasos completos.
- **Entradas:** pipeline, `param_grid`, estrategia o número de folds y métrica implícita o configurada.
- **Salida:** puntuaciones de validación, `best_params_`, `best_score_` y `best_estimator_` reajustado sobre todos los datos de entrenamiento.
- **Hiperparámetros relevantes en los ejemplos:** `svm__C`, `svm__gamma`, `polynomialfeatures__degree`, `ridge__alpha`, `classifier__max_features` y el propio estimador asignado a un paso.
- **Ventajas:** compara flujos completos sin contaminar los folds.
- **Limitaciones:** prueba el producto de combinaciones del grid; agregar dimensiones incrementa rápidamente el número de ajustes.
- **Casos adecuados:** elegir grado polinómico junto con regularización, decidir si escalar o comparar SVC con Random Forest.
- **Casos donde no conviene usarlo:** búsquedas exhaustivas de todas las técnicas imaginables sin restringir el espacio.

### Selección de características dentro del pipeline

- **Objetivo:** aprender qué variables conservar usando únicamente el fold de entrenamiento.
- **Entradas:** `X`, `y`, función de puntuación y proporción seleccionada.
- **Salida:** matriz con el subconjunto de características elegido.
- **Ventaja:** evita correlaciones espurias inducidas por mirar los folds de validación.
- **Riesgo:** ajustarla globalmente antes de CV puede convertir datos aleatorios en una puntuación aparentemente excelente.

## 6. Fórmulas esenciales

El ejemplo de regresión interpreta el coeficiente de determinación usado por `Ridge`:

\[
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
\]

- \(y_i\): valor real de la observación \(i\).
- \(\hat{y}_i\): predicción para la observación \(i\).
- \(\bar{y}\): media de los valores reales evaluados.
- **Qué mide:** mejora relativa frente a predecir siempre la media.
- **Interpretación:** valores cercanos a 1 indican buen ajuste; un valor negativo significa que el modelo es peor que el predictor constante de la media.
- **Supuesto importante:** la cifra solo es informativa si todo aprendizaje, incluida la selección de variables, se realiza dentro de cada fold.

Nota de trazabilidad: el capítulo usa e interpreta \(R^2\), pero no imprime su fórmula; se incluye aquí únicamente para hacer operativa la métrica citada.

## 7. Implementación práctica

Ejemplo mínimo de clasificación y ajuste sin fuga:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svm", SVC()),
])

search = GridSearchCV(
    pipeline,
    param_grid={
        "svm__C": [0.1, 1, 10],
        "svm__gamma": [0.01, 0.1, 1],
    },
    cv=5,
)
search.fit(X_train, y_train)
test_score = search.score(X_test, y_test)
```

**Adaptación moderna:** se usa `return_X_y=True`, `test_size` explícito y `stratify=y`; estas elecciones mejoran claridad y estabilidad en clasificación, pero no forman parte del ejemplo original. El principio original conservado es que el escalador vive dentro del pipeline.

Ejemplo mínimo de selección de características sin fuga:

```python
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=5),
    Ridge(),
)
scores = cross_val_score(pipeline, X, y, cv=5)
```

## 8. Supuestos y condiciones de aplicación

1. Cada paso intermedio implementa una transformación compatible con la salida del paso anterior.
2. El último paso implementa `fit`; debe implementar `predict`, `score` o `transform` si se desea invocar ese método sobre el pipeline.
3. Los transformadores que aprenden estadísticas se ajustan solo con datos de entrenamiento del fold actual.
4. El conjunto de prueba final permanece fuera tanto del ajuste como de la selección de hiperparámetros.
5. Los parámetros del grid usan exactamente el nombre real del paso seguido de `__`.
6. La validación elegida representa razonablemente el modo en que llegarán los datos nuevos.
7. La comparación entre pipelines usa la misma partición y criterio de evaluación.

## 9. Errores y riesgos

- Ajustar `MinMaxScaler`, `StandardScaler`, selección o extracción de características antes de la validación cruzada.
- Confundir el test final con los folds internos de validación: ninguno debe influir en el ajuste de las transformaciones que lo evalúan.
- Interpretar como señal real una puntuación inflada por selección de variables global. En el ejemplo con datos aleatorios, el procedimiento incorrecto obtiene \(R^2=0.91\), mientras el pipeline correcto obtiene \(R^2=-0.25\).
- Olvidar transformar el conjunto de test o aplicar pasos en un orden diferente al entrenamiento.
- Usar una clave de grid sin el prefijo del paso o con un nombre generado distinto del esperado.
- Construir un producto cartesiano excesivo: el coste de `GridSearchCV` aumenta con cada opción añadida.
- Elegir un pipeline más complejo sin comprobar si cada componente aporta mejora.
- **Sobreajuste de selección:** probar muchas configuraciones puede adaptar las decisiones al esquema de validación; el capítulo enfatiza mantener la búsqueda acotada, aunque no desarrolla validación anidada.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** cuando el baseline requiere imputación, escalado, selección, expansión polinómica u otra transformación aprendida; también cuando la CV parece sorprendentemente alta.
- **Qué decisiones permite tomar:** dónde colocar el preprocesamiento, cómo nombrar hiperparámetros, cómo comparar variantes de transformación y cómo recuperar el modelo ganador.
- **Baseline:** pipeline corto con el preprocesamiento indispensable y un modelo; validación cruzada sobre el objeto completo.
- **Mejora del modelo:** ajustar conjuntamente parámetros del transformador y del estimador, o comparar unos pocos flujos coherentes mediante una lista de grids.
- **Qué evitar bajo presión:** grids grandes que combinan múltiples transformadores, familias de modelos y rangos densos; componentes que no tengan una hipótesis clara.
- **Señales activadoras:** escalas muy distintas para SVC, transformaciones calculadas con estadísticas de muestra, selección supervisada de muchas variables, creación de interacciones o necesidad de comparar modelos con requisitos de escalado diferentes.
- **Recomendación para competencias:** tratar una mejora extraordinaria tras selección de características como una señal de posible leakage y auditar si la selección fue ajustada dentro de cada fold.

## 11. Checklist operativo

- [ ] Reservar el conjunto de prueba antes de seleccionar parámetros.
- [ ] Identificar todos los pasos que aprenden de `X` o `y`.
- [ ] Incluir esos pasos dentro de un `Pipeline`.
- [ ] Confirmar que cada transformador se ajusta dentro de cada fold.
- [ ] Verificar los nombres de pasos y las claves `<paso>__<parametro>`.
- [ ] Mantener acotado el producto de combinaciones del grid.
- [ ] Comparar pipelines con la misma validación y métrica.
- [ ] Evaluar el mejor pipeline una sola vez en el test reservado.
- [ ] Inspeccionar el estimador final mediante `best_estimator_.named_steps` si se requieren atributos.
- [ ] Justificar cada componente añadido al flujo.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Construcción e interfaz de pipelines | `knowledge/preprocessing/pipelines.md` | Alta |
| Ajuste de transformadores dentro de CV | `knowledge/validation/data_leakage.md` | Alta |
| Sintaxis de parámetros y búsqueda conjunta | `knowledge/tuning/pipeline_grid_search.md` | Alta |
| Comparación de modelos y pasos opcionales | `knowledge/models/model_selection.md` | Media |
| Selección de características sin fuga | `knowledge/feature_engineering/feature_selection.md` | Alta |
| Interpretación de resultados espurios de CV | `knowledge/validation/cross_validation.md` | Alta |
| Acceso a componentes y coeficientes del pipeline ganador | `knowledge/interpretability/pipeline_introspection.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura del capítulo, p. 305; Summary and Outlook, pp. 320-321 |
| 2. Conceptos esenciales | Parameter Selection with Preprocessing, pp. 306-307; The General Pipeline Interface, pp. 312-313 |
| 3. Flujo o procedimiento presentado | Building Pipelines, p. 308; Using Pipelines in Grid Searches, pp. 309-310 |
| 4. Reglas de decisión | pp. 307-310, 313-314 y 319-320 |
| 5. Métodos y técnicas | Building Pipelines, p. 308; The General Pipeline Interface, pp. 312-316; Grid-Searching Which Model To Use, pp. 319-320 |
| 6. Fórmulas esenciales | Illustrating Information Leakage, pp. 310-311 (uso e interpretación de \(R^2\); la fórmula no aparece impresa) |
| 7. Implementación práctica | Ejemplos In[5]-In[14], pp. 308-311; adaptaciones modernas identificadas en el resumen |
| 8. Supuestos y condiciones | The General Pipeline Interface, pp. 312-313 |
| 9. Errores y riesgos | pp. 306-311, 319 y 321 |
| 10. Aplicación en un coding test | Síntesis derivada de pp. 305-321 |
| 11. Checklist operativo | Síntesis derivada de pp. 307-321 |
| 12. Mapeo hacia `knowledge/` | Síntesis documental del capítulo completo, pp. 305-321 |

