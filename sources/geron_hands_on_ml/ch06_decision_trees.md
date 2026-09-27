---
title: "Decision Trees"
topics:
  - "Árboles de decisión para clasificación y regresión"
  - "CART, impurity, regularización e inestabilidad"
target_knowledge_files:
  - "knowledge/models/decision_trees.md"
  - "knowledge/validation/regularization_and_model_stability.md"
---

# 6. Decision Trees

## 1. Propósito del capítulo

El capítulo explica cómo entrenar, interpretar y regularizar árboles de decisión para clasificación y regresión. Presenta el algoritmo CART usado por Scikit-Learn, sus criterios de división, complejidad, limitaciones e inestabilidad.

## 2. Conceptos esenciales

- Un árbol divide el espacio de características mediante preguntas binarias hasta llegar a una hoja. Cada hoja produce una clase, probabilidades de clase o un valor numérico.
- Es un modelo de caja blanca: las reglas de decisión son inspeccionables y pueden aplicarse manualmente.
- Requiere poca preparación: no necesita escalado ni centrado de características.
- En clasificación, `samples` cuenta las observaciones que llegan a un nodo, `value` cuenta las observaciones por clase y `gini` mide su impureza.
- Una hoja pura contiene ejemplos de una sola clase (`Gini = 0`).
- CART construye árboles binarios mediante divisiones recursivas y codiciosas. El óptimo global es intratable en general; el algoritmo busca una solución razonablemente buena.
- Las predicciones de clasificación son constantes por región. `predict_proba` devuelve la proporción de cada clase en la hoja alcanzada.
- En regresión, cada hoja predice la media de los objetivos de sus observaciones.
- Los árboles no restringidos son propensos al sobreajuste. La profundidad, el tamaño mínimo de nodos y el número de hojas controlan su libertad.
- Las fronteras son ortogonales a los ejes; el modelo es sensible a rotaciones, pequeños cambios en los datos y, en Scikit-Learn, a la selección aleatoria de características por nodo.

## 3. Flujo o procedimiento presentado

1. Seleccionar las características y el objetivo.
2. Crear `DecisionTreeClassifier` o `DecisionTreeRegressor`.
3. Ajustar el modelo con el conjunto de entrenamiento.
4. Inspeccionar reglas, nodos, hojas, impureza y probabilidades; opcionalmente exportar el árbol con `export_graphviz`.
5. Regularizar mediante hiperparámetros si el árbol sobreajusta.
6. Evaluar con validación separada o validación cruzada; ajustar la complejidad usando solo entrenamiento.
7. Para regresión, comprobar que la regularización evita regiones demasiado específicas.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se necesita un modelo interpretable con reglas explícitas | Árbol de decisión | Sus decisiones se pueden inspeccionar y explicar | Un árbol grande pierde legibilidad y puede sobreajustar |
| Variables en escalas muy distintas | Árbol sin escalado obligatorio | Las divisiones dependen de umbrales, no de distancias | El modelo sigue siendo sensible a la orientación de los datos |
| Clasificación con clases mezcladas en un nodo | CART con `criterion="gini"` o `"entropy"` | Elige la división que reduce la impureza ponderada | Gini y entropía suelen producir árboles parecidos |
| Sobreajuste | Reducir `max_depth`, `max_leaf_nodes` o `max_features`; aumentar parámetros `min_*` | Restringe la estructura del árbol | Elegir valores sin validación puede causar underfitting |
| Regresión con predicciones muy dentadas | Aumentar `min_samples_leaf` o limitar profundidad/hojas | Obliga a promediar más observaciones por región | Puede perder patrones locales reales |
| Cambios pequeños producen árboles muy distintos | Usar más robustez mediante ensambles, como Random Forests | Promediar muchos árboles limita la inestabilidad | El ensamble es menos interpretable que un árbol único |
| Datos rotados o frontera diagonal | Considerar una transformación como PCA | Puede mejorar la orientación respecto de los ejes | PCA pertenece a otro capítulo; aquí se menciona como mitigación |

## 5. Métodos y técnicas

### `DecisionTreeClassifier`

- **Objetivo:** clasificar instancias y estimar probabilidades de clase.
- **Entradas:** matriz de características `X` y etiquetas `y`.
- **Salida:** clase mediante `predict`; proporciones por clase mediante `predict_proba`.
- **Funcionamiento:** divide recursivamente con una característica y umbral que minimizan la impureza ponderada.
- **Hiperparámetros:** `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, `max_leaf_nodes`, `max_features`, `criterion` y `random_state`.
- **Ventajas:** interpretable, rápido en predicción, versátil y sin necesidad de escalado.
- **Limitaciones:** sobreajuste, sensibilidad a cambios y fronteras ortogonales.
- **Adecuado:** baselines interpretables y reglas no lineales.
- **No conviene:** como árbol ilimitado cuando se requiere estabilidad o generalización robusta.

### Impureza Gini

Es el criterio predeterminado del capítulo. Mide la mezcla de clases en un nodo; vale cero cuando el nodo es puro. Suele ser ligeramente más rápido que la entropía.

### Entropía

Se activa con `criterion="entropy"`. También mide impureza y suele producir resultados similares. Cuando difieren, Gini tiende a aislar la clase más frecuente y entropía puede producir árboles algo más balanceados.

### CART para clasificación

Para cada nodo prueba características y umbrales, pondera la impureza de los dos subconjuntos por su tamaño y escoge la menor función de coste. Repite hasta cumplir una condición de parada.

### `DecisionTreeRegressor`

- **Objetivo:** predecir un valor continuo.
- **Entradas:** `X` y objetivos numéricos `y`.
- **Salida:** media del objetivo de la hoja alcanzada.
- **Funcionamiento:** divide para minimizar el MSE ponderado de las regiones hijas.
- **Hiperparámetros:** especialmente `max_depth` y `min_samples_leaf`, además de los límites de estructura comunes.
- **Ventajas:** captura relaciones no lineales sin especificar una forma funcional.
- **Limitaciones:** predicciones constantes por región y fuerte sobreajuste sin regularización.
- **Adecuado:** relaciones con umbrales o regiones diferenciadas.
- **No conviene:** si se requiere suavidad continua o estabilidad frente a pequeñas perturbaciones sin usar un ensamble.

### Regularización y poda

Reducir `max_depth`, `max_leaf_nodes` o `max_features`, o aumentar `min_samples_split`, `min_samples_leaf` y `min_weight_fraction_leaf`, reduce la libertad del árbol. El capítulo también describe la poda posterior basada en si la mejora de pureza es estadísticamente significativa, aunque no desarrolla una implementación de Scikit-Learn para ella.

## 6. Fórmulas esenciales

### Impureza Gini

\[
G_i = 1 - \sum_{k=1}^{n} p_{i,k}^{2}
\]

`p_{i,k}` es la proporción de instancias de la clase `k` en el nodo `i`; `n` es el número de clases. Mide la mezcla de clases: `0` significa pureza total. Supone que las proporciones del nodo representan la distribución relevante para la decisión.

### Coste CART de clasificación

\[
J(k,t_k)=\frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}
\]

`k` es la característica, `t_k` el umbral, `m` el tamaño del nodo padre y `m_left`, `m_right` los tamaños hijos. El algoritmo busca minimizar la impureza ponderada.

### Entropía

\[
H_i=-\sum_{k=1}^{n} p_{i,k}\log_2(p_{i,k}),\quad p_{i,k}\ne 0
\]

Mide el contenido medio de información o impureza. Es cero si solo existe una clase; el logaritmo se define para probabilidades no nulas.

### Predicción de regresión y MSE

\[
\hat{y}_{node}=\frac{1}{m_{node}}\sum_{i\in node}y_i
\]

La hoja predice la media de sus objetivos. El error del nodo es:

\[
MSE_{node}=\frac{1}{m_{node}}\sum_{i\in node}(y_i-\hat{y}_{node})^2
\]

El CART de regresión escoge la característica y el umbral que minimizan el MSE ponderado de las hojas hijas. El supuesto operativo es que la media representa adecuadamente cada región.

### Coste CART de regresión

\[
J(k,t_k)=\frac{m_{left}}{m}MSE_{left}+\frac{m_{right}}{m}MSE_{right}
\]

## 7. Implementación práctica

Clasificación mínima, adaptada a una API moderna y con una restricción explícita para evitar un árbol ilimitado:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X, y)

probabilities = tree.predict_proba([[5.0, 1.5]])
prediction = tree.predict([[5.0, 1.5]])
```

Regresión:

```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=10,
    random_state=42,
)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_valid)
```

El uso de `random_state` es una adaptación práctica para reproducibilidad. El capítulo muestra `export_graphviz` y Graphviz para visualizar, pero la visualización no es necesaria para entrenar o evaluar el modelo.

## 8. Supuestos y condiciones de aplicación

- No se exige linealidad ni escalado/centrado de características.
- Los splits son binarios en CART y generan fronteras ortogonales.
- Las probabilidades de clasificación son proporciones de entrenamiento dentro de la hoja, no necesariamente probabilidades calibradas.
- La media de una hoja es la predicción de regresión y el MSE guía las divisiones.
- La complejidad debe restringirse cuando el conjunto es pequeño, ruidoso o el árbol empieza a memorizar.
- La estabilidad no está garantizada: eliminar observaciones, rotar variables o cambiar la selección aleatoria puede modificar mucho el árbol.
- Para evaluar generalización, la elección de hiperparámetros debe hacerse con datos de entrenamiento y validación separada.

## 9. Errores y riesgos

- Entrenar con parámetros ilimitados y tomar el ajuste perfecto como evidencia de calidad.
- Confundir `predict_proba` con una probabilidad calibrada: es la frecuencia de clases en la hoja.
- Escalar características por costumbre; no es necesario para el árbol, aunque las transformaciones pueden afectar orientación o interpretabilidad.
- Ignorar el sobreajuste al usar `max_depth=None` o hojas con muy pocas observaciones.
- Elegir Gini frente a entropía esperando una mejora sistemática; el capítulo indica que normalmente la diferencia es pequeña.
- Evaluar o seleccionar la estructura usando el conjunto de prueba.
- Interpretar una frontera ortogonal como una relación estable cuando los datos pueden rotarse o cambiar ligeramente.
- Usar un árbol único cuando la inestabilidad exige promediar modelos.
- Coste de entrenamiento elevado al crecer el número de muestras y características; la búsqueda examina características y muestras en cada nodo. Las predicciones son aproximadamente `O(log2(m))` en árboles balanceados y casi independientes del número de características; el entrenamiento se presenta como aproximadamente `O(n × m log2(m))`, con `n` características y `m` muestras.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Consultar este capítulo cuando el dataset tenga relaciones no lineales, umbrales claros, variables mixtas o se necesite un baseline interpretable. Un baseline razonable es un `DecisionTreeClassifier` o `DecisionTreeRegressor` con profundidad y tamaño mínimo de hoja restringidos, `random_state` fijado y evaluación en validación separada.

Las señales para activar estas técnicas son: patrones por regiones, reglas simples, ausencia de necesidad de escalado y un árbol lineal que no captura interacciones. Para mejorar el modelo, ajustar `max_depth`, `min_samples_leaf` y `max_leaf_nodes` mediante validación; comparar Gini y entropía solo si la métrica lo justifica.

Bajo presión de tiempo no conviene implementar poda estadística ni intentar encontrar el árbol global óptimo. Tampoco conviene entregar un árbol ilimitado solo porque logra alta precisión de entrenamiento. Si el árbol cambia mucho entre particiones, considerar un Random Forest como siguiente paso, aunque su explicación sea menos directa.

## 11. Checklist operativo

- [ ] Identificar si el problema es clasificación o regresión.
- [ ] Separar entrenamiento y validación/prueba antes de ajustar.
- [ ] Entrenar un árbol pequeño y reproducible con `random_state`.
- [ ] Verificar que no se está escalando sin necesidad.
- [ ] Revisar profundidad, número de hojas y tamaño de las hojas.
- [ ] Comparar rendimiento de entrenamiento frente a validación.
- [ ] Regularizar si existe una brecha grande o predicciones demasiado dentadas.
- [ ] Interpretar `predict_proba` como proporciones de la hoja.
- [ ] Comprobar sensibilidad a cambios de datos si la estabilidad importa.
- [ ] Pasar a un ensamble si un árbol único resulta inestable.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Clasificación y regresión con árboles | `knowledge/models/decision_trees.md` | Alta |
| Impureza Gini, entropía y CART | `knowledge/models/decision_trees.md` | Alta |
| Regularización por estructura | `knowledge/tuning/tree_regularization.md` | Alta |
| Separación entrenamiento/validación para elegir complejidad | `knowledge/validation/model_selection.md` | Alta |
| Interpretación de reglas y probabilidades de hojas | `knowledge/interpretability/tree_interpretability.md` | Media |
| Complejidad e inestabilidad | `knowledge/models/model_stability.md` | Media |
| Relación con Random Forests | `knowledge/ensembles/random_forests.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------ | ------------------------------ |
| Propósito y conceptos esenciales | pp. 175-178, introducción, entrenamiento, predicción e interpretación |
| Flujo e implementación de clasificación | pp. 175-177 |
| Gini, entropía y CART | pp. 177-181 |
| Regularización y poda | pp. 181-182 |
| Regresión y MSE | pp. 183-184 |
| Complejidad computacional | p. 180 |
| Limitaciones e inestabilidad | pp. 185-186 |
| Aplicación en coding test y checklist | Síntesis práctica derivada de pp. 175-186 |

Nota de procesamiento: el PDF recibido contiene las páginas impresas 175-187; la página 187 contiene únicamente el cierre de ejercicios y la última página del archivo no contiene texto. No se identificaron páginas ilegibles. Las recomendaciones marcadas como síntesis práctica o adaptación moderna no se presentan como afirmaciones textuales del autor.
