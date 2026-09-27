---
title: "Model Evaluation and Improvement"
topics:
  - "cross-validation"
  - "model selection and grid search"
  - "classification and regression metrics"
  - "decision thresholds and imbalanced data"
target_knowledge_files:
  - "knowledge/validation/cross_validation.md"
  - "knowledge/tuning/grid_search.md"
  - "knowledge/metrics/classification_metrics.md"
  - "knowledge/metrics/regression_metrics.md"
---

# 5. Model Evaluation and Improvement

## 1. Propósito del capítulo

El capítulo amplía la evaluación de modelos supervisados más allá de una única división entrenamiento/prueba. Presenta validación cruzada para estimar generalización con mayor estabilidad, búsqueda en rejilla para seleccionar hiperparámetros sin contaminar el conjunto de prueba y métricas adecuadas para clasificación binaria, multiclase y regresión. Su idea central es que tanto el esquema de partición como la métrica deben representar el uso futuro del modelo. **[Contenido explícito del autor; PDF pp. 1-2 y 52-53, capítulo pp. 251-252 y 302-303]**

## 2. Conceptos esenciales

- **Generalización.** El interés no es el ajuste sobre entrenamiento, sino el rendimiento sobre observaciones no vistas. Un test independiente aproxima ese comportamiento futuro. **[Contenido explícito; PDF pp. 1-2]**
- **Validación cruzada (cross-validation, CV).** Divide repetidamente los datos, entrena varios modelos y devuelve un score por partición. En *k-fold CV*, cada fold actúa una vez como validación y los restantes como entrenamiento. La media resume el rendimiento esperado y la dispersión revela sensibilidad a la partición o escasez de datos. CV evalúa un algoritmo sobre un dataset; no entrega por sí sola un modelo final. **[Contenido explícito; PDF pp. 2-4]**
- **CV estratificada.** En clasificación conserva aproximadamente la proporción de clases en cada fold y evita folds poco informativos cuando las etiquetas están ordenadas o desbalanceadas. Para regresión, el capítulo usa *k-fold* estándar. **[Contenido explícito; PDF pp. 5-6]**
- **Grupos.** Si varias filas pertenecen a una misma entidad (persona, paciente, hablante), todas deben permanecer juntas en entrenamiento o validación cuando se quiere generalizar a entidades nuevas. La variable `groups` no es la etiqueta objetivo. **[Contenido explícito; PDF pp. 9-10]**
- **Búsqueda en rejilla (grid search).** Evalúa todas las combinaciones de valores indicadas. La rejilla debe cubrir escalas suficientemente amplias; un óptimo en el borde sugiere ampliar el rango. Una superficie constante puede indicar rango inadecuado o hiperparámetro irrelevante. **[Contenido explícito; PDF pp. 10-11 y 17-20]**
- **Separación train/validation/test.** Entrenamiento ajusta el modelo; validación selecciona modelo, hiperparámetros y umbral; test se reserva para una evaluación final. Elegir a partir del test filtra información y produce una estimación optimista. **[Contenido explícito; PDF pp. 11-16 y 52]**
- **Validación cruzada anidada (nested CV).** La CV interna selecciona hiperparámetros y la externa estima generalización. El resultado es una lista de scores, no un modelo ni una configuración única. **[Contenido explícito; PDF pp. 22-24]**
- **Métrica de negocio y métrica sustituta.** La métrica de evaluación debe aproximar el efecto real de las predicciones. Exactitud alta no implica utilidad, especialmente con clases desbalanceadas o costes asimétricos. **[Contenido explícito; PDF pp. 25-27 y 52-53]**
- **Matriz de confusión.** Filas: clase real; columnas: clase predicha. La diagonal contiene aciertos; fuera de ella aparecen falsos positivos y falsos negativos. **[Contenido explícito; PDF pp. 29-32]**
- **Umbral de decisión.** `predict` aplica normalmente 0 a `decision_function` o 0.5 a `predict_proba` en clasificación binaria. Mover el umbral cambia el compromiso precisión-recall; debe elegirse en validación, no en test. **[Contenido explícito; PDF pp. 36-39]**
- **Curvas precision-recall (PR) y ROC.** Describen todos los umbrales. PR es especialmente informativa cuando importa la clase positiva; ROC representa TPR frente a FPR. Un único F1 refleja solo el umbral por defecto y puede ocultar diferencias entre modelos. **[Contenido explícito; PDF pp. 39-46]**
- **AUC.** Resume el ranking de positivos frente a negativos. Un AUC de 0.5 corresponde a ranking aleatorio y 1 a ranking perfecto; no garantiza que el umbral por defecto sea útil. **[Contenido explícito; PDF pp. 44-46]**
- **Promedios multiclase.** `macro` pondera cada clase por igual; `weighted`, por soporte; `micro` agrega TP, FP y FN antes de calcular la métrica. **[Contenido explícito; PDF pp. 47-49]**

## 3. Flujo o procedimiento presentado

1. Definir el objetivo de uso y traducirlo a una métrica evaluable que refleje costes de FP y FN. **[Contenido explícito; PDF pp. 25-27]**
2. Separar una vez un conjunto de test que no participará en exploración, selección de variables, elección de modelo, tuning ni ajuste de umbral. **[Contenido explícito; PDF pp. 12-13 y 52]**
3. Elegir el splitter según la unidad real de generalización: estratificado para clasificación, `KFold` para regresión, `GroupKFold` cuando existen entidades repetidas; barajar solo si el orden no contiene una dependencia que deba preservarse. **[Conclusión práctica derivada de PDF pp. 5-10]**
4. Construir un baseline, incluido un `DummyClassifier` cuando haya desbalance, y evaluarlo con la métrica elegida. **[Contenido explícito; PDF pp. 27-32]**
5. Ejecutar CV en el conjunto de entrenamiento y revisar tanto media como variación entre folds. **[Contenido explícito; PDF pp. 2-4]**
6. Diseñar una rejilla inicial pequeña y amplia, preferentemente en escala logarítmica para parámetros como `C` y `gamma`; usar rejillas condicionales cuando unos parámetros solo sean válidos con determinadas opciones. **[Conclusión práctica derivada de PDF pp. 17-22]**
7. Ajustar `GridSearchCV` solo sobre entrenamiento. Inspeccionar `cv_results_`, no solo `best_params_`, para detectar rangos planos, parámetros irrelevantes u óptimos en fronteras. **[Contenido explícito; PDF pp. 15-22]**
8. Si se requiere una estimación menos dependiente de una única partición, evaluar el proceso completo con CV anidada. **[Contenido explícito; PDF pp. 22-24]**
9. En clasificación, revisar matriz de confusión, informe por clase y curvas PR/ROC. Elegir el punto operativo en validación si el umbral por defecto no satisface el coste del problema. **[Contenido explícito; PDF pp. 29-46]**
10. Una vez fijadas todas las decisiones, reajustar con todos los datos de entrenamiento disponibles y evaluar una sola vez sobre test. **[Contenido explícito; PDF pp. 12-17 y 52]**

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --- | --- | --- | --- |
| Clasificación con proporciones desiguales | `StratifiedKFold` | Conserva aproximadamente la proporción de clases por fold | Estratificar no corrige por sí solo una métrica inadecuada **[Explícito; PDF pp. 5-6]** |
| Filas repetidas por paciente, persona o hablante y objetivo de generalizar a entidades nuevas | `GroupKFold` | Impide que una entidad aparezca a ambos lados del split | `groups` identifica entidades, no clases **[Explícito; PDF pp. 9-10]** |
| Dataset pequeño | Más folds o `LeaveOneOut` | Usa más observaciones para ajustar cada modelo | Coste elevado y potencial variabilidad; LOO entrena un modelo por muestra **[Explícito; PDF pp. 4 y 7]** |
| Dataset grande o modelo lento | `ShuffleSplit` con fracciones/submuestreo o una sola división controlada | Separa número de iteraciones de tamaños train/test y reduce coste | Fijar semilla para reproducibilidad cuando se baraja **[Explícito; PDF pp. 7-8 y 22]** |
| Selección de hiperparámetros | `GridSearchCV` dentro de train | Usa CV como validación y reajusta el mejor estimador sobre todo train | Mantener test completamente fuera de la búsqueda **[Explícito; PDF pp. 13-17]** |
| Hiperparámetros condicionales | Lista de diccionarios en `param_grid` | Evita combinaciones sin sentido o desperdiciadas | Cada diccionario define una rejilla independiente **[Explícito; PDF pp. 21-22]** |
| Óptimo en el borde del mapa de scores | Ampliar el rango hacia ese borde | Puede haber mejores valores fuera de la rejilla | Ajustar rangos usando CV, nunca test **[Explícito; PDF pp. 19-21]** |
| Clases desbalanceadas | Matriz de confusión, F1, PR/AP o ROC-AUC | Accuracy puede igualar a un predictor constante | Comparar con baseline `DummyClassifier` **[Explícito; PDF pp. 27-35 y 42-46]** |
| FP más costosos | Priorizar precisión | Precisión penaliza positivos predichos incorrectamente | La mejora suele reducir recall **[Explícito; PDF p. 33]** |
| FN más costosos | Priorizar recall | Recall mide cuántos positivos reales se recuperan | Bajar el umbral aumenta normalmente FP **[Explícito; PDF pp. 33 y 37-39]** |
| Cada clase importa por igual | F1 macro | Promedia clases sin ponderar por frecuencia | Puede reflejar clases raras con alta influencia **[Explícito; PDF p. 49]** |
| Cada muestra importa por igual | F1 micro | Agrega conteos de todas las clases | Puede quedar dominado por clases frecuentes **[Explícito; PDF p. 49]** |
| Regresión sin coste específico | `R²` | El capítulo lo considera generalmente intuitivo | Si negocio penaliza magnitud absoluta/cuadrática, usar MAE/MSE **[Explícito; PDF p. 49]** |

## 5. Métodos y técnicas

### 5.1 K-fold y Stratified K-fold

- **Objetivo:** estimar rendimiento de generalización mediante `k` entrenamientos.
- **Entradas:** estimador, `X`, `y`, número de folds o splitter.
- **Salida:** un score por fold; no devuelve un modelo final.
- **Funcionamiento:** cada fold se usa una vez para validar y los demás para entrenar; la variante estratificada conserva proporciones de clase.
- **Hiperparámetros:** `n_splits`, `shuffle`, `random_state` si se baraja.
- **Ventajas:** usa mejor los datos, reduce dependencia de un único split y muestra variabilidad.
- **Limitaciones:** coste aproximadamente `k` veces mayor y no modela dependencias de grupos o tiempo por defecto.
- **Casos adecuados:** supervisado i.i.d.; clasificación con estratificación y regresión con K-fold estándar.
- **No conviene:** si filas relacionadas pueden cruzar folds; debe usarse un splitter de grupos. **[Contenido explícito; PDF pp. 2-7]**

### 5.2 Leave-One-Out y ShuffleSplit

- **Objetivo:** LOO maximiza el tamaño de entrenamiento; `ShuffleSplit` controla de forma independiente iteraciones y tamaños.
- **Entradas:** datos, estimador y configuración de splits.
- **Salida:** scores repetidos.
- **Funcionamiento:** LOO valida una observación cada vez; `ShuffleSplit` muestrea subconjuntos disjuntos train/test repetidamente.
- **Hiperparámetros:** LOO no requiere número de folds; `ShuffleSplit`: `n_splits`, `train_size`, `test_size`, `random_state`.
- **Ventajas:** LOO puede ser útil con muestras pequeñas; `ShuffleSplit` permite experimentar con una fracción de datasets grandes.
- **Limitaciones:** LOO es muy costoso; `ShuffleSplit` puede omitir observaciones en una iteración y requiere variante estratificada para clasificación desbalanceada.
- **Casos adecuados:** LOO en datasets pequeños; `ShuffleSplit` cuando se desean tamaños o repeticiones específicos.
- **No conviene:** LOO en datasets grandes. **[Contenido explícito; PDF pp. 7-8]**

### 5.3 GroupKFold

- **Objetivo:** medir generalización a grupos no vistos.
- **Entradas:** `X`, `y`, identificador de grupo por fila, estimador.
- **Salida:** scores con grupos enteros en train o validación.
- **Funcionamiento:** asigna grupos, no muestras individuales, a folds.
- **Hiperparámetros:** `n_splits`.
- **Ventajas:** evita fuga entre ejemplos altamente relacionados.
- **Limitaciones:** necesita grupos fiables y suficientes grupos para repartir.
- **Casos adecuados:** múltiples muestras por paciente, persona o hablante.
- **No conviene:** si la unidad de generalización no es el grupo definido. **[Contenido explícito; PDF pp. 9-10]**

### 5.4 GridSearchCV

- **Objetivo:** elegir la combinación de hiperparámetros con mejor score medio de CV.
- **Entradas:** estimador, `param_grid`, splitter `cv`, métrica `scoring`.
- **Salida:** `best_params_`, `best_score_`, `best_estimator_`, `cv_results_`; con `refit` predeterminado, un estimador reajustado sobre todo train.
- **Funcionamiento:** expande combinaciones, evalúa cada una en CV, selecciona la mejor y la reajusta.
- **Hiperparámetros:** `param_grid`, `cv`, `scoring`, `n_jobs`; los parámetros del estimador forman la rejilla.
- **Ventajas:** interfaz reproducible y resultados detallados por combinación y fold.
- **Limitaciones:** coste multiplicativo: combinaciones × folds; alto uso de memoria al paralelizar.
- **Casos adecuados:** espacios discretos manejables y parámetros importantes conocidos.
- **No conviene:** rejillas enormes o combinaciones condicionales no separadas; en estas últimas usar lista de rejillas. **[Contenido explícito; PDF pp. 10-25]**

### 5.5 Nested cross-validation

- **Objetivo:** estimar el rendimiento del procedimiento completo de selección de hiperparámetros.
- **Entradas:** búsqueda interna, splitter externo, `X`, `y`.
- **Salida:** scores externos.
- **Funcionamiento:** cada fold externo ejecuta una búsqueda completa en su subconjunto de entrenamiento y evalúa el mejor resultado en su fold externo.
- **Hiperparámetros:** folds internos/externos y rejilla.
- **Ventajas:** reduce dependencia de una única partición train/test al comparar procedimientos.
- **Limitaciones:** coste muy alto; no produce directamente un modelo desplegable ni un conjunto único de parámetros.
- **Casos adecuados:** evaluación rigurosa de algoritmos en un dataset.
- **No conviene:** cuando el objetivo inmediato es obtener un único modelo bajo fuerte límite computacional. **[Contenido explícito; PDF pp. 22-24]**

### 5.6 Matriz de confusión y métricas derivadas

- **Objetivo:** separar tipos de acierto y error.
- **Entradas:** etiquetas reales y predichas.
- **Salida:** conteos TN, FP, FN, TP; precisión, recall, F1 y soporte por clase.
- **Funcionamiento:** resume predicciones a un umbral específico.
- **Hiperparámetros:** clase positiva y, para multiclase, estrategia de promedio.
- **Ventajas:** expone fallos que accuracy oculta.
- **Limitaciones:** depende del umbral y de qué clase se declare positiva; un promedio ponderado puede disimular una clase minoritaria deficiente.
- **Casos adecuados:** clasificación binaria o multiclase, especialmente con desbalance.
- **No conviene:** como único análisis si aún debe estudiarse el comportamiento en múltiples umbrales. **[Contenido explícito; PDF pp. 29-36 y 47-49]**

### 5.7 Curvas PR, ROC, average precision y ROC-AUC

- **Objetivo:** comparar clasificadores a través de todos los umbrales y seleccionar un punto operativo.
- **Entradas:** `y_true` y scores continuos de `decision_function` o probabilidad de la clase positiva.
- **Salida:** PR: precisión/recall/umbrales; ROC: FPR/TPR/umbrales; AP y AUC: resúmenes escalares.
- **Funcionamiento:** ordena ejemplos por score y evalúa cada umbral posible.
- **Hiperparámetros:** no hay hiperparámetros del cálculo; el punto operativo es una decisión de aplicación.
- **Ventajas:** revela regiones en las que un modelo domina y distingue rankings que accuracy trata como iguales.
- **Limitaciones:** AP/AUC resumen y pueden ocultar la región operativa; AUC no selecciona el umbral; probabilidades no necesariamente están calibradas.
- **Casos adecuados:** clasificación binaria desbalanceada o costes dependientes del umbral.
- **No conviene:** usar `predict` como entrada; se requieren scores continuos. **[Contenido explícito; PDF pp. 36-46]**

## 6. Fórmulas esenciales

Sea `TP` verdaderos positivos, `TN` verdaderos negativos, `FP` falsos positivos y `FN` falsos negativos.

### Accuracy

\[
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

Mide la fracción total de aciertos. Es interpretable cuando las clases y costes son comparables; puede ser engañosa con fuerte desbalance. **[Contenido explícito; PDF p. 32]**

### Precision

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
\]

Mide qué fracción de los positivos predichos es realmente positiva. Valores altos implican pocos FP. **[Contenido explícito; PDF p. 33]**

### Recall / sensibilidad / TPR

\[
\mathrm{Recall}=\mathrm{TPR}=\frac{TP}{TP+FN}
\]

Mide qué fracción de los positivos reales fue recuperada. Valores altos implican pocos FN. **[Contenido explícito; PDF p. 33]**

### F1

\[
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
\]

Es la media armónica de precisión y recall. Exige que ambas sean altas, pero representa un solo umbral y es menos intuitiva que accuracy. **[Contenido explícito; PDF pp. 33-34]**

### False positive rate

\[
\mathrm{FPR}=\frac{FP}{FP+TN}
\]

Mide la fracción de negativos reales clasificados como positivos. En ROC se interpreta junto con TPR; la zona ideal está cerca de TPR = 1 y FPR = 0. **[Contenido explícito; PDF p. 43]**

El capítulo menciona `R²`, MSE y MAE para regresión, pero no desarrolla sus fórmulas en estas páginas; por fidelidad documental no se añaden fórmulas externas. **[Contenido explícito; PDF p. 49]**

## 7. Implementación práctica

### Evaluación con el splitter correcto

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
print(scores.mean(), scores.std())
```

**[Adaptación moderna claramente marcada]** El libro muestra que `cross_val_score` usaba tres folds por defecto; se declara `n_splits=5` para no depender de un valor predeterminado que puede variar y se fija `random_state` porque hay barajado. La elección de F1 macro debe responder a que todas las clases importen por igual. Base conceptual: PDF pp. 3, 6-7 y 49.

### CV por grupos

```python
from sklearn.model_selection import GroupKFold, cross_val_score

group_cv = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=group_ids, cv=group_cv)
```

**[Adaptación moderna claramente marcada]** Se pasa `groups` por nombre para hacer explícita la unidad de separación. Cada entidad permanece íntegramente en train o validación. Base conceptual: PDF pp. 9-10.

### Búsqueda sin tocar el test

```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

param_grid = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]},
    {"kernel": ["linear"], "C": [0.1, 1, 10]},
]
search = GridSearchCV(
    SVC(), param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)
search.fit(X_train, y_train)
final_test_score = search.score(X_test, y_test)
```

**[Adaptación moderna claramente marcada]** Se añade `stratify=y` en el holdout y se especifica `scoring`. `search.score` utiliza el scorer configurado. `n_jobs=-1` sigue la recomendación del capítulo, pero debe vigilarse memoria y paralelismo anidado. Base conceptual: PDF pp. 15-25 y 50-51.

### Matriz, reporte y ajuste de umbral en validación

```python
from sklearn.metrics import classification_report, confusion_matrix

validation_scores = model.decision_function(X_valid)
threshold = -0.8  # elegido con validación, nunca con test
validation_predictions = validation_scores > threshold

print(confusion_matrix(y_valid, validation_predictions))
print(classification_report(y_valid, validation_predictions))
```

El valor `-0.8` reproduce únicamente la idea demostrativa del capítulo; no es un umbral universal. **[Contenido explícito y adaptación mínima; PDF pp. 36-39]**

### PR-AUC y ROC-AUC con scores continuos

```python
from sklearn.metrics import average_precision_score, roc_auc_score

test_scores = model.decision_function(X_test)
average_precision = average_precision_score(y_test, test_scores)
roc_auc = roc_auc_score(y_test, test_scores)
```

No se debe pasar `model.predict(X_test)` porque perdería la información de ranking y umbrales. **[Contenido explícito; PDF pp. 42-45]**

**[Adaptación moderna claramente marcada]** El ejemplo final del libro importa `SCORERS` desde una ruta interna y usa nombres positivos para pérdidas. En código actual conviene consultar los scorers públicos con `sklearn.metrics.get_scorer_names()` y usar nombres de pérdidas negadas como `neg_mean_squared_error` o `neg_mean_absolute_error` cuando corresponda. Esta nota actualiza la API; no es contenido original del capítulo.

## 8. Supuestos y condiciones de aplicación

1. Los folds deben representar el escenario de generalización. Si hay entidades correlacionadas, se separan por grupo; si el orden tiene significado, no se baraja sin justificarlo. **[Conclusión práctica derivada de PDF pp. 5-10]**
2. Las observaciones del test no intervienen en ninguna decisión de modelado, incluidas visualizaciones exploratorias, rejillas y umbrales. **[Contenido explícito; PDF pp. 12-13 y 52]**
3. La estratificación presupone que conservar la distribución de clases es apropiado para el escenario futuro. **[Conclusión práctica derivada de PDF pp. 5-6]**
4. Los scores de CV dependen del dataset, algoritmo, métrica y splitter concretos; no son propiedades universales del estimador. **[Contenido explícito; PDF pp. 3-4 y 23]**
5. Precision, recall y F1 requieren definir la clase positiva. Cambiarla altera sustancialmente la interpretación. **[Contenido explícito; PDF pp. 33-36]**
6. `decision_function` puede tener escala arbitraria; `predict_proba` está entre 0 y 1, pero no toda probabilidad está calibrada. **[Contenido explícito; PDF pp. 36-38]**
7. AP y AUC evalúan ranking a través de umbrales; para producir decisiones útiles aún hace falta elegir un punto operativo. **[Contenido explícito; PDF pp. 39-46]**
8. Paralelizar requiere memoria suficiente y evitar paralelismo anidado entre estimador y búsqueda. **[Contenido explícito; PDF pp. 24-25]**

## 9. Errores y riesgos

- **Optimizar sobre test:** probar rejillas, modelos o umbrales en test y luego reportar el mejor resultado. Produce estimaciones optimistas. **[Contenido explícito; PDF pp. 11-13, 20-21 y 38]**
- **Confundir `best_score_` con test:** es la media de CV dentro de train, no el rendimiento final de generalización. **[Contenido explícito; PDF p. 17]**
- **Interpretar CV como modelo:** `cross_val_score` devuelve scores; los modelos internos sirven para evaluación. **[Contenido explícito; PDF p. 4]**
- **Usar folds ordinarios con etiquetas ordenadas:** puede dejar clases ausentes de train o validación. **[Contenido explícito; PDF pp. 5-7]**
- **Separar filas de la misma entidad:** produce fuga y mide reconocimiento de entidades conocidas, no generalización a nuevas. **[Contenido explícito; PDF pp. 9-10]**
- **Confiar en accuracy con desbalance:** un predictor constante puede lograr 90% o 99% sin aprender la clase de interés. **[Contenido explícito; PDF pp. 27-29]**
- **Mirar solo promedios:** un promedio ponderado puede ocultar desempeño nulo en la minoría; revisar scores por clase y soporte. **[Conclusión práctica derivada de PDF pp. 34-36 y 47-49]**
- **Ajustar solo precisión o recall:** cualquiera puede maximizarse trivialmente sacrificando la otra. **[Contenido explícito; PDF p. 33]**
- **Pasar etiquetas a métricas de ranking:** AP/AUC requieren scores continuos, no predicciones ya umbralizadas. **[Contenido explícito; PDF pp. 42-45]**
- **Suponer que AUC alto basta:** AUC ignora el umbral operativo; puede requerirse ajuste posterior en validación. **[Contenido explícito; PDF pp. 45-46]**
- **Rejilla mal especificada:** rangos estrechos, lineales o con óptimo en el borde pueden perder buenas regiones; combinaciones condicionales desperdician cómputo. **[Contenido explícito; PDF pp. 17-22]**
- **Explosión computacional:** coste aproximado = combinaciones × folds; nested CV multiplica además por folds externos. El paralelismo puede agotar memoria. **[Contenido explícito; PDF pp. 14, 23-25]**
- **Variabilidad por tamaño pequeño:** gran dispersión entre folds puede reflejar sensibilidad del modelo o simplemente pocas muestras. **[Contenido explícito; PDF p. 3]**

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- Consultar este capítulo cuando haya que justificar un esquema de validación, seleccionar hiperparámetros, tratar desbalance o elegir una métrica alineada con costes.
- Para un baseline rápido: holdout estratificado, un modelo sencillo, `DummyClassifier`, matriz de confusión y métrica apropiada. Si hay grupos, sustituir inmediatamente el split aleatorio por uno grupal.
- Para mejorar el modelo: ejecutar una rejilla pequeña con CV, observar `cv_results_`, ampliar solo rangos prometedores y cambiar el umbral si el objetivo exige una precisión o recall concretos.
- Señales de activación: proporción de clases muy desigual; múltiples filas por usuario/paciente; etiquetas ordenadas; accuracy cercana al baseline mayoritario; costes distintos de FP/FN; gran variación entre folds; mejor hiperparámetro en el borde de la rejilla.
- Bajo presión de tiempo, no conviene implementar CV anidada, rejillas masivas ni LOO en datasets grandes. Tampoco conviene explorar muchos umbrales si el enunciado no define el coste; es preferible reportar PR/ROC y declarar la decisión pendiente.
- En competiciones, mantener un holdout local verdaderamente reservado ayuda a evitar sobreajuste al propio proceso de CV; la métrica local debe coincidir con la métrica oficial cuando esta represente el objetivo. **[Recomendación para competencias derivada de los principios de PDF pp. 12-25 y 50-53; la referencia a la métrica oficial es una aplicación al contexto competitivo, no una frase del autor]**

## 11. Checklist operativo

- [ ] Definir unidad de generalización y coste relativo de FP/FN.
- [ ] Reservar test antes de explorar o ajustar.
- [ ] Comprobar balance, orden y grupos repetidos.
- [ ] Elegir `StratifiedKFold`, `KFold` o `GroupKFold` según el caso.
- [ ] Fijar `random_state` cuando haya aleatoriedad.
- [ ] Comparar contra un baseline trivial si existe desbalance.
- [ ] Elegir `scoring` antes de iniciar tuning.
- [ ] Ejecutar una rejilla pequeña y revisar media, desviación y bordes.
- [ ] Examinar matriz de confusión y métricas por clase.
- [ ] Elegir umbral únicamente con validación.
- [ ] Reajustar el modelo con todo train usando decisiones ya cerradas.
- [ ] Evaluar test una sola vez y reportar la métrica pertinente.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| --- | --- | --- |
| K-fold, estratificación, LOO y ShuffleSplit | `knowledge/validation/cross_validation.md` | Alta |
| Separación por entidades con GroupKFold | `knowledge/validation/group_aware_validation.md` | Alta |
| Train/validation/test y prevención de leakage | `knowledge/validation/train_validation_test_split.md` | Alta |
| GridSearchCV, inspección de `cv_results_` y rejillas condicionales | `knowledge/tuning/grid_search.md` | Alta |
| Validación cruzada anidada | `knowledge/validation/nested_cross_validation.md` | Media |
| Matriz de confusión, precision, recall y F1 | `knowledge/metrics/classification_metrics.md` | Alta |
| Curvas PR, average precision, ROC y AUC | `knowledge/metrics/ranking_metrics.md` | Alta |
| Ajuste de umbral y punto operativo | `knowledge/metrics/decision_thresholds.md` | Alta |
| Promedios micro, macro y weighted | `knowledge/metrics/multiclass_metrics.md` | Alta |
| R², MSE y MAE como criterios de regresión | `knowledge/metrics/regression_metrics.md` | Media |
| Coste computacional y paralelización del tuning | `knowledge/tuning/search_compute_tradeoffs.md` | Media |

No se crean ni modifican archivos de `knowledge/` en esta tarea.

## 13. Trazabilidad

La numeración “PDF” empieza en la primera página del archivo adjunto; entre paréntesis se muestra la página impresa del capítulo.

| Sección del resumen | Sección o páginas del capítulo |
| --- | --- |
| 1. Propósito | PDF pp. 1-2, 52-53 (cap. pp. 251-252, 302-303) |
| 2. Cross-validation y splitters | PDF pp. 2-10 (cap. pp. 252-260) |
| 2. Grid search y separación train/validation/test | PDF pp. 10-25 (cap. pp. 260-275) |
| 2. Métricas y umbrales | PDF pp. 25-51 (cap. pp. 275-301) |
| 3. Flujo operativo | PDF pp. 2-25, 29-53 (cap. pp. 252-275, 279-303) |
| 4. Reglas de decisión | PDF pp. 5-10, 13-25, 27-51 (cap. pp. 255-260, 263-275, 277-301) |
| 5.1 K-fold / Stratified K-fold | PDF pp. 2-7 (cap. pp. 252-257) |
| 5.2 LOO / ShuffleSplit | PDF pp. 7-8 (cap. pp. 257-258) |
| 5.3 GroupKFold | PDF pp. 9-10 (cap. pp. 259-260) |
| 5.4 GridSearchCV | PDF pp. 10-25 (cap. pp. 260-275) |
| 5.5 Nested CV | PDF pp. 22-24 (cap. pp. 272-274) |
| 5.6 Matriz y métricas derivadas | PDF pp. 29-36, 47-49 (cap. pp. 279-286, 297-299) |
| 5.7 PR/ROC/AP/AUC | PDF pp. 36-46 (cap. pp. 286-296) |
| 6. Fórmulas | PDF pp. 32-33, 43 (cap. pp. 282-283, 293) |
| 7. Implementación | PDF pp. 3, 6-10, 15-25, 36-45, 49-52 (cap. pp. 253, 256-260, 265-275, 286-295, 299-302) |
| 8. Supuestos | PDF pp. 5-13, 23-25, 33-46 (cap. pp. 255-263, 273-275, 283-296) |
| 9. Errores y riesgos | PDF pp. 4-25, 27-46, 52-53 (cap. pp. 254-275, 277-296, 302-303) |
| 10. Coding test | Síntesis derivada de PDF pp. 5-25, 27-53 (cap. pp. 255-275, 277-303) |
| 11. Checklist | Síntesis derivada de PDF pp. 2-53 (cap. pp. 252-303) |
| 12. Mapeo | Destinos propuestos a partir de PDF pp. 2-53 (cap. pp. 252-303) |

### Cobertura y legibilidad

- Páginas físicas revisadas: 54.
- Páginas con contenido del capítulo: 53, correspondientes a las páginas impresas 251-303.
- Página física 54: en blanco; no contiene material procesable.
- Secciones no procesables: ninguna. Texto, código, tablas y Figuras 5-1 a 5-18 fueron legibles. Las figuras se usaron para verificar particiones, mapas de calor, matrices de confusión y curvas PR/ROC; no se reprodujeron íntegramente.
- Fuente utilizada: exclusivamente el PDF adjunto del capítulo 5; no se usaron fuentes externas.
