---
title: "Ensemble Learning and Random Forests"
topics:
  - "Ensemble learning"
  - "Voting, bagging, Random Forests, boosting y stacking"
  - "OOB evaluation y early stopping"
target_knowledge_files:
  - "knowledge/ensembles/ensemble_learning.md"
  - "knowledge/models/random_forests.md"
  - "knowledge/tuning/boosting_and_early_stopping.md"
---

# 7. Ensemble Learning and Random Forests

## 1. Propósito del capítulo

Explica cómo combinar predictores para obtener un modelo más fuerte que sus componentes. Presenta voting, bagging/pasting, Random Forests, Extra-Trees, AdaBoost, Gradient Boosting y stacking. Es especialmente útil después de construir varios modelos razonables o en competiciones.

## 2. Conceptos esenciales

- Un **ensemble** combina varios predictores. La diversidad es clave: errores poco correlacionados producen mayor beneficio.
- Un conjunto de weak learners puede formar un strong learner si hay suficientes modelos diversos.
- En clasificación, **hard voting** elige la clase más votada; **soft voting** promedia probabilidades y suele ponderar mejor la confianza.
- **Bagging** usa muestras con reemplazo; **pasting**, sin reemplazo. La agregación suele reducir varianza.
- **Out-of-bag (OOB)** evalúa cada predictor con las instancias que no entraron en su muestra bootstrap.
- Random Forest combina árboles, bootstrap y selección aleatoria de características en cada división.
- Extra-Trees añade umbrales aleatorios, aumentando diversidad y reduciendo el coste de buscar el mejor umbral.
- **Boosting** entrena secuencialmente: cada predictor corrige al anterior. AdaBoost repondera ejemplos difíciles; Gradient Boosting ajusta residuos.
- **Stacking** entrena un blender/meta-learner con las predicciones de una primera capa.

## 3. Flujo o procedimiento presentado

1. Entrenar varios modelos individuales, preferiblemente diversos.
2. Elegir agregación por voting, muestreo paralelo, boosting o stacking.
3. Mantener validación separada; en stacking, generar predicciones sobre hold-out u out-of-fold no visto por los modelos base.
4. Controlar número de estimadores, profundidad, hojas, learning rate y fracciones de muestras/características.
5. Evaluar con validación, OOB cuando corresponda o early stopping.
6. Comparar el ensemble con cada modelo individual.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Clasificadores diversos con probabilidades | Soft voting | Promedia confianza | Todos deben ofrecer `predict_proba`; SVC con `probability=True` es más costoso |
| Clasificadores diversos sin probabilidades comparables | Hard voting | Voto mayoritario simple | Ignora la confianza |
| Árbol de alta varianza | Bagging o Random Forest | Reduce varianza mediante agregación | No corrige leakage |
| Se necesita evaluación interna con bootstrap | OOB | Usa ejemplos no vistos por cada árbol | No sustituye necesariamente la prueba final |
| Muchas características, por ejemplo imágenes | Random Patches/Subspaces | Aumenta diversidad | Puede aumentar sesgo |
| Se busca entrenar árboles más rápido | Extra-Trees | Usa umbrales aleatorios | Comparar con Random Forest mediante validación |
| Learners débiles y casos difíciles | AdaBoost | Repondera errores sucesivos | Secuencial y sensible al sobreajuste |
| Regresión no lineal con residuos corregibles | Gradient Boosting | Cada árbol ajusta residuos | Regularizar profundidad, tasa y número de árboles |
| El error de validación empeora al seguir entrenando | Early stopping | Elige o detiene el tamaño óptimo | La validación debe estar separada |
| Se desea aprender la combinación | Stacking/blending | El meta-modelo aprende la agregación | Evitar predicciones in-sample |

## 5. Métodos y técnicas

### Voting Classifier

Combina clasificadores como regresión logística, SVM y Random Forest. `voting="hard"` usa la moda; `voting="soft"` promedia probabilidades y puede superar al voto duro. Requiere modelos suficientemente diversos. En el ejemplo del capítulo, el ensemble supera ligeramente a sus componentes.

### Bagging y pasting

`BaggingClassifier`/`BaggingRegressor` entrenan el mismo algoritmo en subconjuntos aleatorios. `bootstrap=True` significa bagging (con reemplazo); `False`, pasting. En clasificación se agregan votos/probabilidades y en regresión promedios. `n_estimators`, `max_samples`, `bootstrap`, `n_jobs`, `max_features` y `bootstrap_features` son los controles principales. Los modelos pueden entrenarse y predecirse en paralelo. Bagging suele preferirse porque su mayor diversidad reduce más la varianza, aunque puede tener algo más de sesgo que pasting.

### OOB evaluation

Con bootstrap y muestras del tamaño del entrenamiento, cada predictor usa aproximadamente 63% de las instancias; el resto, aproximadamente 37%, son OOB. Como el predictor no vio esas filas, se pueden agregar sus evaluaciones para estimar el ensemble. En Scikit-Learn se activa con `oob_score=True`; el resultado está en `oob_score_` y, cuando hay probabilidades, en `oob_decision_function_`.

### Random Patches y Random Subspaces

`max_features` y `bootstrap_features` muestrean columnas. Muestrear instancias y características es Random Patches; conservar todas las instancias y muestrear características es Random Subspaces. Ambos aumentan diversidad a cambio de sesgo adicional.

### Random Forests

Es un ensemble de árboles entrenado normalmente con bagging y selección aleatoria de características en cada nodo. `RandomForestClassifier` y `RandomForestRegressor` combinan parámetros del árbol con los del ensemble. `n_estimators`, `max_leaf_nodes`, profundidad, restricciones de hojas y `n_jobs` son relevantes. `feature_importances_` mide la reducción media ponderada de impureza y se normaliza para sumar 1; no implica causalidad.

### Extra-Trees

Selecciona subconjuntos aleatorios de características y umbrales aleatorios. Puede ser más rápido que Random Forest, porque no busca el mejor umbral. No se puede saber de antemano cuál rinde mejor: se comparan ambos con validación cruzada.

### AdaBoost

Entrena un predictor, aumenta el peso de los ejemplos mal clasificados, normaliza pesos y repite. Los predictores tienen pesos según su error. El ejemplo usa stumps (`max_depth=1`), `n_estimators=200` y `learning_rate=0.5`. Reducir estimadores o regularizar el estimador base puede ayudar contra overfitting. El entrenamiento es secuencial, por lo que escala peor que bagging.

### Gradient Boosting

Añade árboles secuencialmente, cada uno ajustado a los residuos del ensemble anterior. `GradientBoostingRegressor` controla el árbol (`max_depth`, `min_samples_leaf`) y el ensemble (`n_estimators`, `learning_rate`, `subsample`, `loss`). Un `learning_rate` bajo suele generalizar mejor, pero exige más árboles (shrinkage). `subsample<1` produce Stochastic Gradient Boosting: más sesgo, menor varianza y menor coste. `staged_predict()` permite escoger el número de árboles con menor error de validación; `warm_start=True` permite detener el entrenamiento cuando deja de mejorar.

El capítulo menciona XGBoost como implementación rápida, escalable y frecuente en competiciones, con API similar y early stopping. No se desarrolla aquí como contenido principal.

### Stacking/blending

Se divide el entrenamiento: una parte entrena modelos base y otra genera predicciones limpias para entrenar el blender. Para varias capas se necesitan más particiones o predicciones out-of-fold. La idea es aprender una agregación, no aplicar una función fija. El texto indica que la API de Scikit-Learn descrita no ofrece stacking directo.

## 6. Fórmulas esenciales

### Error ponderado de AdaBoost

\[
r_j=\frac{\sum_{i=1}^{m}w_i\mathbf{1}(\hat y_i^{(j)}\ne y_i)}{\sum_{i=1}^{m}w_i}
\]

`w_i` es el peso de la instancia, `y_i` su clase y `\hat y_i^{(j)}` la predicción del predictor `j`. Mide el error dando más importancia a los casos difíciles.

### Peso del predictor

\[
\alpha_j=\eta\log\left(\frac{1-r_j}{r_j}\right)
\]

`\eta` es `learning_rate`; menor `r_j` implica mayor peso. Si el predictor apenas supera el azar, el peso se aproxima a cero; si es peor que el azar, puede ser negativo.

### Actualización y predicción AdaBoost

\[
w_i\leftarrow\begin{cases}w_i&\hat y_i^{(j)}=y_i\\w_i e^{\alpha_j}&\hat y_i^{(j)}\ne y_i\end{cases}
\qquad
\hat y(x)=\arg\max_k\sum_{j=1}^{N}\mathbf{1}(\hat y_j(x)=k)\alpha_j
\]

Después de actualizar se normalizan los pesos. La segunda expresión escoge la clase con mayor voto ponderado.

## 7. Implementación práctica

Ejemplo mínimo de soft voting. `random_state` y `max_iter` son adaptaciones prácticas modernas para reproducibilidad/convergencia, no contenido original:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ensemble = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ],
    voting="soft",
)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_valid)
```

Bagging con OOB; `estimator` es el nombre actual de lo que el texto antiguo llamaba `base_estimator`:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, oob_score=True, n_jobs=-1, random_state=42
)
bagging.fit(X_train, y_train)
oob_estimate = bagging.oob_score_
```

Selección de árboles en Gradient Boosting:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)
errors = [mean_squared_error(y_valid, p)
          for p in gbrt.staged_predict(X_valid)]
best_n = min(range(1, len(errors) + 1), key=lambda n: errors[n - 1])
```

## 8. Supuestos y condiciones de aplicación

- Los modelos deben aportar diversidad; errores perfectamente correlacionados limitan la ganancia.
- OOB requiere bootstrap y suficientes predictores para obtener evaluaciones útiles.
- Soft voting requiere probabilidades disponibles y comparables.
- Boosting es secuencial y necesita controlar complejidad, número de estimadores y learning rate.
- Stacking requiere predicciones no contaminadas por entrenamiento sobre las mismas filas.
- La importancia de árboles es reducción de impureza ponderada, no evidencia causal.

## 9. Errores y riesgos

- Combinar modelos similares puede no mejorar.
- Evaluar solo en entrenamiento oculta overfitting.
- Usar predicciones in-sample para el blender produce leakage.
- OOB es una estimación interna, no necesariamente el reemplazo de una prueba final.
- Demasiados árboles/profundidad sobreajustan Gradient Boosting; muy pocos underfit.
- Bajar `learning_rate` sin aumentar árboles puede underfit.
- AdaBoost puede enfocarse demasiado en ruido y outliers.
- Boosting no se paraleliza como bagging.
- `probability=True` en SVC aumenta el coste.
- `feature_importances_` no debe interpretarse como causalidad.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.** Consulta este capítulo cuando un modelo individual ya funciona y aparecen alta varianza, fronteras inestables o errores complementarios. Para un baseline tabular, probaría Random Forest o Extra-Trees con validación; con varios clasificadores distintos, compararía hard/soft voting. Para regresión no lineal, Gradient Boosting con profundidad pequeña, learning rate controlado y validación del número de árboles.

Bajo presión no implementaría stacking multilayer desde cero ni una búsqueda exhaustiva. Sí comprobaría diversidad, separación de validación, leakage, coste y curva de error frente a árboles. Un árbol muy sobreajustado activa bagging/Random Forest; mejora secuencial y casos difíciles activan boosting; predicciones distintas activan voting o stacking.

## 11. Checklist operativo

- [ ] Comparar modelos individuales y diversidad de errores.
- [ ] Elegir voting, bagging, Random Forest, boosting o stacking.
- [ ] Separar entrenamiento, validación y prueba final.
- [ ] Usar OOB solo con bootstrap apropiado.
- [ ] Controlar estimadores, profundidad, hojas y learning rate.
- [ ] Usar `n_jobs=-1` donde sea aplicable.
- [ ] Confirmar probabilidades para soft voting.
- [ ] Generar predicciones hold-out/out-of-fold para stacking.
- [ ] Seleccionar árboles con validación/early stopping.
- [ ] Comparar Extra-Trees y Random Forest empíricamente.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Diversidad, voting y agregación | `knowledge/ensembles/ensemble_learning.md` | Alta |
| Bagging, pasting, OOB y paralelización | `knowledge/ensembles/bagging_and_oob.md` | Alta |
| Random Forest, Extra-Trees e importancias | `knowledge/models/random_forests.md` | Alta |
| AdaBoost | `knowledge/models/adaboost.md` | Media |
| Gradient Boosting y stochastic boosting | `knowledge/models/gradient_boosting.md` | Alta |
| Early stopping y selección de árboles | `knowledge/tuning/early_stopping.md` | Alta |
| Stacking/blending | `knowledge/ensembles/stacking.md` | Media |
| Leakage en meta-modelos | `knowledge/validation/data_leakage.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------ | ------------------------------ |
| Propósito y conceptos | Introducción, pp. 189–191 |
| Voting Classifiers | pp. 189–192 |
| Bagging, pasting y OOB | pp. 192–196 |
| Random Patches/Subspaces y Random Forests | pp. 196–198 |
| Extra-Trees e importancia | pp. 197–199 |
| AdaBoost y fórmulas | pp. 199–203 |
| Gradient Boosting | pp. 203–205 |
| Shrinkage, early stopping y stochastic boosting | pp. 205–207 |
| XGBoost mencionado | p. 208 |
| Stacking/blending | pp. 208–211 |
| Ejercicios no resumidos operativamente | pp. 211–212 |

Todas las páginas 189–212 fueron legibles y procesadas; no se usaron fuentes externas ni se modificaron archivos de `knowledge/`.
