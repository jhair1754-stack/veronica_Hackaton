---
title: "Making Better Predictions - Optimizing Models"
topics:
* "Hyperparameter tuning and cross-validation"
* "Feature engineering and dimensionality reduction"
* "Ensemble methods"
* "Class imbalance and regularization"
target_knowledge_files:
* "knowledge/tuning/hyperparameter_search.md"
* "knowledge/feature_engineering/feature_construction_and_selection.md"
* "knowledge/ensembles/tree_ensembles_and_voting.md"
* "knowledge/preprocessing/class_imbalance.md"
* "knowledge/models/regularized_linear_models.md"
---

# 10. Making Better Predictions - Optimizing Models

## 1. Propósito del capítulo

El capítulo presenta formas de mejorar modelos ya construidos: ajustar hiperparámetros sin contaminar el test, crear o reducir variables, combinar estimadores, inspeccionar la confianza de clasificación, tratar desbalance de clases y regularizar regresiones. Su valor práctico está en convertir un baseline en un proceso de mejora medible, manteniendo una evaluación final sobre datos no usados durante el ajuste.

## 2. Conceptos esenciales

- **Hiperparámetro:** configuración elegida antes del ajuste y no aprendida directamente de los datos. El rendimiento se optimiza sobre una métrica explícita.
- **Grid search:** evaluación exhaustiva de las combinaciones de un espacio de hiperparámetros. `GridSearchCV` integra búsqueda y validación cruzada.
- **Validación cruzada k-fold:** divide entrenamiento en `k` subconjuntos, entrena `k` veces y promedia el resultado. En clasificación, la estratificación conserva aproximadamente la proporción de clases.
- **Separación de datos:** validación sirve para selección; test queda reservado para la estimación final. Usar test durante el tuning sobreestima la generalización.
- **Feature engineering:** incluye transformación, construcción, selección y extracción de variables. Más variables pueden aumentar complejidad, dispersión y riesgo de ajustar ruido (curse of dimensionality).
- **Interacciones y polinomios:** productos y potencias permiten a modelos lineales representar relaciones no lineales, a costa de un fuerte crecimiento dimensional.
- **Selección frente a extracción:** la selección descarta variables; la extracción las combina en una representación de menor dimensión.
- **PCA:** proyección lineal sensible a escala que genera componentes ortogonales ordenados por varianza explicada.
- **Ensembles:** bagging reduce varianza mediante modelos sobre muestras bootstrap; boosting encadena aprendices para reducir sesgo; voting combina decisiones de modelos distintos.
- **Confianza predictiva:** `predict_proba()` permite separar aciertos y errores por probabilidad, revisar umbrales y estudiar casos difíciles.
- **Desbalance:** under-sampling elimina ejemplos mayoritarios; over-sampling aumenta la minoría, incluso con ejemplos sintéticos como SMOTE. Ninguno debe aplicarse antes de medir un baseline.
- **Regularización:** añade una penalización para contraer coeficientes y reducir sobreajuste. Ridge usa L2, LASSO usa L1 y elastic net combina ambas.

## 3. Flujo o procedimiento presentado

1. Separar entrenamiento y test; estratificar cuando sea clasificación con clases desbalanceadas.
2. Construir un baseline y elegir una métrica coherente con el objetivo.
3. Encapsular preprocesamiento y estimador en un `Pipeline` para que cada fold ajuste transformaciones solo con sus datos de entrenamiento.
4. Definir un espacio pequeño y espaciado de hiperparámetros; referirse a parámetros internos como `paso__parametro`.
5. Ejecutar `GridSearchCV` o, si el espacio es grande, `RandomizedSearchCV`; seleccionar el esquema de CV apropiado, por ejemplo estratificado o temporal.
6. Revisar `best_params_`, `best_score_`, `best_estimator_` y `cv_results_['mean_fit_time']`.
7. Evaluar una sola vez el estimador elegido sobre test.
8. Probar mejoras justificadas: interacciones, selección por varianza, PCA, importancia de variables, ensembles, remuestreo o regularización.
9. Comparar cada variante con el baseline usando la misma partición y métrica, incluyendo costo de entrenamiento e inferencia.
10. Inspeccionar probabilidades y errores para decidir si conviene ajustar umbral, investigar outliers o combinar modelos.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Pocos hiperparámetros y espacio acotado | `GridSearchCV` | Prueba todas las combinaciones y promedia folds | El costo crece con combinaciones, folds y tiempo de ajuste |
| Espacio de búsqueda grande | `RandomizedSearchCV` | Evalúa una muestra de combinaciones | Puede omitir la mejor región; controlar `n_iter` |
| Clasificación con clases desiguales | CV estratificada | Conserva proporciones por fold | No usar test para escoger parámetros |
| Serie temporal | `TimeSeriesSplit` | Respeta el orden y usa muestras sucesivas | No barajar observaciones futuras con pasadas |
| Relaciones o fronteras no lineales en un modelo lineal | Interacciones o polinomios | Linealizan términos construidos | Grados altos disparan dimensionalidad y overfitting |
| Variables casi constantes | `VarianceThreshold` | Reduce variables poco informativas | El umbral depende de la escala |
| Muchas variables con información distribuida | PCA tras escalado | Condensa varianza en menos componentes | Pierde interpretabilidad y solo captura estructura lineal |
| Árbol individual sobreajusta | Random forest | Bagging y selección aleatoria de variables reducen varianza | Menos interpretable y más costoso que un árbol |
| Se busca corregir errores secuencialmente | Gradient boosting | Cada árbol contribuye a reducir la pérdida | Sensible al ruido y entrenamiento serial más lento |
| Modelos distintos cometen errores diferentes | `VotingClassifier` | Agrega decisiones complementarias | Soft voting requiere probabilidades; validar pesos y umbral |
| Dataset pequeño y desbalanceado | Probar primero baseline; luego over-sampling | Under-sampling eliminaría demasiada información | SMOTE exige que la minoría observada represente su población |
| Muchos datos mayoritarios y costo computacional relevante | Under-sampling | Reduce volumen y balancea clases | Puede degradar mucho el modelo al descartar señal |
| Regresión con sobreajuste o coeficientes inestables | Ridge, LASSO o elastic net con escalado | Contrae coeficientes; LASSO puede seleccionar variables | Elegir fuerza y mezcla por CV |

## 5. Métodos y técnicas

### GridSearchCV y RandomizedSearchCV

- **Objetivo:** escoger hiperparámetros maximizando una métrica de validación.
- **Entradas:** estimador/pipeline, diccionario de búsqueda, estrategia `cv`, `scoring` y, si hay varias métricas, `refit`.
- **Salida:** mejor configuración, score CV, estimador reajustado y resultados por combinación.
- **Funcionamiento:** Grid Search prueba el producto cartesiano; Randomized Search prueba `n_iter` combinaciones aleatorias.
- **Hiperparámetros clave:** `cv`, `scoring`, `refit`, `n_jobs`, `n_iter`.
- **Ventajas:** proceso reproducible, compatible con pipelines y múltiples métricas.
- **Limitaciones:** multiplica entrenamientos; una malla grande y granular puede ser prohibitiva.
- **Uso adecuado:** empezar con valores separados y refinar alrededor de regiones prometedoras.
- **No conviene:** búsqueda exhaustiva sobre muchos parámetros o modelos muy lentos sin acotar el espacio.

### PolynomialFeatures y FeatureUnion

- **Objetivo:** construir potencias/interacciones y combinar transformaciones paralelas.
- **Entradas:** matriz de variables; `degree`, `interaction_only`, `include_bias`; lista de transformadores para `FeatureUnion`.
- **Salida:** matriz ampliada; en una unión, bloques transformados concatenados por columnas.
- **Funcionamiento:** con dos variables y grado 2 produce sesgo, originales, cuadrados e interacción; `interaction_only=True` omite potencias.
- **Ventajas:** introduce no linealidad en modelos lineales y combina distintas representaciones.
- **Limitaciones:** explosión de columnas, redundancia y riesgo de overfitting.
- **Uso adecuado:** pocas variables con interacciones plausibles.
- **No conviene:** grados altos o muchas variables sin control de complejidad.

### VarianceThreshold, PCA e importancia de variables

- **Objetivo:** reducir dimensionalidad o identificar variables relevantes.
- **Entradas:** variables; umbral para `VarianceThreshold`; `n_components` para PCA; datos etiquetados para un árbol.
- **Salida:** subconjunto de columnas, componentes o vector `feature_importances_` cuya suma es 1.
- **Funcionamiento:** `VarianceThreshold` elimina varianza baja; PCA proyecta sobre ejes ortogonales de máxima varianza; árboles asignan importancia según sus particiones.
- **Hiperparámetros:** `threshold`, `n_components`, y límites del árbol como `max_depth`/`min_samples_leaf`.
- **Ventajas:** menor costo; PCA conserva información distribuida; árboles ayudan a simplificar modelos.
- **Limitaciones:** PCA depende de escala y reduce interpretabilidad; importancia de árbol refleja un aprendiz codicioso y no garantiza selección óptima.
- **Uso adecuado:** usar varianza acumulada y scree plot para escoger componentes; usar importancia como señal para probar modelos más simples.
- **No conviene:** descartar variables o componentes sin verificar el rendimiento posterior.

### Random forest, gradient boosting y voting

- **Objetivo:** mejorar generalización combinando estimadores.
- **Entradas:** datos de entrenamiento y estimadores base.
- **Salida:** predicción agregada y, en bosques, importancias.
- **Funcionamiento:** random forest entrena árboles en bootstrap y subconjuntos aleatorios de variables; boosting entrena árboles en serie siguiendo la reducción de pérdida; hard voting usa mayoría y soft voting suma probabilidades.
- **Hiperparámetros:** `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `oob_score`; en boosting, `learning_rate`; en voting, `voting` y `weights`.
- **Ventajas:** bosque robusto a outliers y fronteras no lineales; boosting puede reducir sesgo; voting explota diversidad.
- **Limitaciones:** menor interpretabilidad; boosting es más sensible a ruido y más lento por su secuencia; ensembles cuestan más que un modelo único.
- **Uso adecuado:** árboles con sobreajuste, fronteras complejas o modelos con errores complementarios.
- **No conviene:** si la mejora marginal no justifica latencia, memoria o entrenamiento.

### Confianza, umbrales y Cohen's kappa

- **Objetivo:** entender desacuerdos, errores y seguridad de clasificación.
- **Entradas:** predicciones, probabilidades y etiquetas reales.
- **Salida:** kappa en `[-1, 1]`, distribuciones de probabilidad, tabla de aciertos/errores y umbral operativo.
- **Funcionamiento:** kappa mide acuerdo entre clasificadores; probabilidades se segmentan por resultado correcto/incorrecto. ROC o precision-recall ayudan a asociar un umbral con el compromiso deseado.
- **Ventajas:** revela casos límite y apoya decisiones de threshold o ensemble.
- **Limitaciones:** una KDE puede extender visualmente densidad fuera de `[0,1]`; confirmar con histograma y observaciones.

### Under-sampling y SMOTE

- **Objetivo:** alterar la distribución del entrenamiento para aprender la clase minoritaria.
- **Entradas:** solo entrenamiento y etiquetas; en SMOTE, número de vecinos.
- **Salida:** conjunto remuestreado equilibrado o menos desigual.
- **Funcionamiento:** under-sampling descarta mayoría; SMOTE crea ejemplos minoritarios sintéticos basados en vecinos.
- **Hiperparámetros:** `random_state`, `k_neighbors` y estrategia de muestreo.
- **Ventajas:** puede aumentar recall minoritario; under-sampling reduce costo; SMOTE conserva los ejemplos mayoritarios.
- **Limitaciones:** pérdida de datos o supuestos fuertes sobre representatividad y estabilidad de la minoría.
- **Uso adecuado:** después del baseline, comparando métricas por clase.
- **No conviene:** under-sampling con pocos datos; SMOTE si la muestra minoritaria no cubre la población real.

### Ridge, LASSO y elastic net

- **Objetivo:** reducir overfitting penalizando coeficientes grandes.
- **Entradas:** regresores escalados, objetivo y fuerza de penalización.
- **Salida:** modelo con coeficientes contraídos; LASSO puede llevar algunos exactamente a cero.
- **Funcionamiento:** Ridge aplica L2, LASSO L1 y elastic net una combinación ponderada.
- **Hiperparámetros:** en las clases de scikit-learn, `alpha`; en `ElasticNet`, además `l1_ratio`.
- **Ventajas:** controla complejidad; LASSO también realiza selección; elastic net permite mezclar comportamientos.
- **Limitaciones:** requiere escalado para que la magnitud de coeficientes sea comparable; la penalización debe validarse.
- **Uso adecuado:** regresión con muchos predictores o riesgo de sobreajuste.
- **No conviene:** comparar directamente penalizaciones sobre variables en escalas incompatibles.

## 6. Fórmulas esenciales

### Penalización L2 (ridge)

$$
P_{L2}=\lambda\sum_j \hat{\beta}_j^2
$$

- $\hat{\beta}_j$: coeficiente estimado de la variable $j$.
- $\lambda$: peso global de la penalización; si es cero se recupera mínimos cuadrados ordinarios.
- Mide el costo añadido por coeficientes grandes; al minimizar la función total, los contrae sin forzarlos normalmente a cero.
- Supuesto práctico: escalar las variables antes, porque la penalización depende de magnitudes.

### Penalización L1 (LASSO)

$$
P_{L1}=\lambda\sum_j |\hat{\beta}_j|
$$

- Las variables y $\lambda$ tienen el mismo significado anterior.
- Penaliza valores absolutos y puede llevar coeficientes a cero, actuando como selección de variables.
- El capítulo la presenta como menos sensible a valores extremos que L2.

### Penalización elastic net

$$
P_{EN}=\lambda\left(\frac{1-\alpha}{2}\sum_j \hat{\beta}_j^2+\alpha\sum_j |\hat{\beta}_j|\right)
$$

- $\alpha$: proporción L1 de la mezcla; el resto corresponde a L2.
- $\lambda$: intensidad total de regularización.
- Interpreta los extremos como predominio de ridge o LASSO y los valores intermedios como combinación.
- Nota de nomenclatura del capítulo: en scikit-learn, $\lambda$ corresponde a `alpha` y el $\alpha$ de la fórmula a `l1_ratio`.

## 7. Implementación práctica

Ejemplo mínimo de tuning sin leakage:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=0
)

pipeline = Pipeline([
    ("scale", MinMaxScaler()),
    ("model", LogisticRegression(class_weight="balanced", random_state=0)),
])
search = GridSearchCV(
    pipeline,
    {"model__C": np.logspace(-1, 1, 10), "model__fit_intercept": [True, False]},
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
).fit(X_train, y_train)

test_predictions = search.predict(X_test)
```

Ejemplo mínimo de regularización con escalado y CV incorporada:

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scale", StandardScaler()),
    ("elastic", ElasticNetCV(
        alphas=[0.001, 0.01, 0.1, 1.0],
        l1_ratio=[0.1, 0.5, 0.9],
        cv=5,
        random_state=0,
    )),
]).fit(X_train, y_train)
```

**Notas prácticas de actualización de API (no son formulación original del capítulo):**

- El ejemplo antiguo ajusta `LinearRegression(normalize=...)`; en API actual se elimina esa opción y el escalado se controla con un paso explícito del pipeline.
- El argumento `iid=False` de `GridSearchCV` ya no se usa.
- Para visualizar probabilidades, sustituir `seaborn.distplot()` por `histplot(..., kde=True)` o por `histplot()` y `kdeplot()`.
- Mantener remuestreo como paso dentro de un pipeline compatible con `imblearn` cuando se combine con CV, para ajustarlo únicamente en cada fold de entrenamiento. Esta es una salvaguarda moderna contra leakage derivada del principio de separación expuesto por el capítulo.

## 8. Supuestos y condiciones de aplicación

1. Entrenamiento, validación y test son disjuntos; test no participa en tuning ni transformaciones.
2. La métrica de búsqueda representa el objetivo real; con desbalance, accuracy por sí sola puede ocultar el desempeño minoritario.
3. La estratificación es apropiada para preservar clases; los datos temporales requieren particiones sucesivas sin shuffle.
4. PCA y regularización requieren escalado previo; PCA describe estructura lineal y sus componentes son linealmente no correlacionados.
5. Interacciones/polinomios presuponen que esas combinaciones pueden contener señal y que hay datos suficientes para la complejidad añadida.
6. La selección por varianza considera poco informativas las variables casi constantes, pero el umbral solo es interpretable con atención a la escala.
7. SMOTE presupone que la minoría observada representa el espectro poblacional y que esa relación no cambia de forma invalidante con el tiempo.
8. Under-sampling presupone suficiente volumen para descartar mayoría sin perder señal esencial.
9. Soft voting exige estimadores capaces de producir probabilidades; weighting exige evidencia de que ciertos modelos merecen más influencia.
10. Los tiempos de ajuste e inferencia forman parte de la viabilidad del método.

## 9. Errores y riesgos

- Ajustar hiperparámetros sobre test: produce leakage y una estimación optimista.
- Preprocesar antes de CV en vez de usar pipeline: comparte información de validación con entrenamiento.
- Crear una malla excesivamente granular desde el inicio: desperdicia cómputo; conviene búsqueda gruesa y refinamiento.
- Optimizar la métrica equivocada o mezclar la métrica de CV con otra al comparar test.
- Añadir grados polinómicos altos: crecimiento combinatorio y overfitting.
- Interpretar varianza explicada de PCA como rendimiento predictivo: debe verificarse el modelo posterior.
- Tomar `feature_importances_` como verdad causal o selección óptima: describe cómo dividió ese árbol.
- Permitir árboles demasiado profundos o hojas con una sola observación.
- Usar boosting con datos muy ruidosos sin validar: es más sensible que random forest.
- Preferir un ensemble por una mejora mínima ignorando tiempo, memoria e inferencia.
- Concluir demasiado de una KDE de probabilidades fuera de `[0,1]`; contrastarla con histograma/rug.
- Remuestrear antes de separar test o fuera de los folds: filtra información y distorsiona la evaluación.
- Aplicar under-sampling a datasets pequeños: puede eliminar la mayoría del entrenamiento.
- Aplicar SMOTE a una minoría no representativa: sintetiza sesgos y no garantiza generalización.
- Regularizar variables sin escalar: la penalización depende artificialmente de sus unidades.
- Ignorar costos de predicción: k-NN aprende de forma perezosa y puede volverse inviable con alta dimensión.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- Consultar este capítulo cuando existe un baseline funcional pero se necesita una mejora controlada, hay muchas variables, desbalance, overfitting o modelos con errores complementarios.
- Para un baseline: split estratificado, pipeline de escalado + modelo, métrica alineada al problema y una CV pequeña.
- Para mejorar: búsqueda gruesa, interacciones selectivas, PCA/selección, regularización o ensembles de árboles; comparar siempre contra el baseline.
- Señales activadoras: gran brecha train/validation (regularización, límites de árbol, bagging); variables casi constantes (selección); alta dimensionalidad (PCA); frontera no lineal (polinomios o árboles); minoría con recall bajo (ponderación/remuestreo); modelos que discrepan de forma útil (voting).
- Bajo presión de tiempo, no conviene una malla enorme, polinomios de grado alto, visualizaciones 3D, ensembles complejos ni SMOTE sin CV segura.
- Priorizar un pipeline reproducible y una métrica correcta sobre pequeñas ganancias obtenidas con procedimientos costosos.

## 11. Checklist operativo

- [ ] Reservar test antes de cualquier ajuste.
- [ ] Estratificar clasificación o usar split temporal cuando corresponda.
- [ ] Definir la métrica objetivo y construir un baseline.
- [ ] Encapsular todo el preprocesamiento en un pipeline.
- [ ] Comenzar con una búsqueda pequeña y espaciada.
- [ ] Registrar score CV, tiempo de ajuste y mejores parámetros.
- [ ] Evaluar el mejor estimador una sola vez en test.
- [ ] Revisar complejidad, overfitting y costo de inferencia.
- [ ] Inspeccionar probabilidades y ejemplos mal clasificados.
- [ ] Remuestrear solo entrenamiento/dentro de CV y comparar con baseline.
- [ ] Escalar antes de PCA o regularización.
- [ ] Documentar cualquier actualización de API respecto del capítulo.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Grid/Randomized Search, métricas, CV y costo | `knowledge/tuning/hyperparameter_search.md` | Alta |
| Separación train/validation/test y pipelines sin leakage | `knowledge/validation/model_selection_without_leakage.md` | Alta |
| Polinomios, interacciones y unión de variables | `knowledge/feature_engineering/feature_construction_and_selection.md` | Alta |
| Selección por varianza e importancias | `knowledge/feature_engineering/feature_selection.md` | Media |
| PCA y selección del número de componentes | `knowledge/preprocessing/dimensionality_reduction.md` | Alta |
| Random forest y gradient boosting | `knowledge/ensembles/tree_ensembles_and_voting.md` | Alta |
| Hard/soft voting y kappa | `knowledge/ensembles/voting_and_model_agreement.md` | Media |
| Probabilidades, umbrales y análisis de errores | `knowledge/metrics/classification_thresholds.md` | Alta |
| Under-sampling y SMOTE | `knowledge/preprocessing/class_imbalance.md` | Alta |
| Ridge, LASSO y elastic net | `knowledge/models/regularized_linear_models.md` | Alta |
| TimeSeriesSplit para tuning temporal | `knowledge/time_series/time_aware_validation.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito | Apertura y lista de temas, pp. 579-580 |
| 2. Conceptos esenciales | Hyperparameter tuning, Feature engineering, Ensemble methods, confidence, imbalance y Regularization, pp. 582-623 |
| 3. Flujo o procedimiento | Preparación y GridSearchCV, pp. 581-589; evaluación de alternativas, pp. 590-623 |
| 4. Reglas de decisión | Grid search, pp. 582-589; feature engineering, pp. 590-605; ensembles, pp. 606-611; imbalance y regularización, pp. 615-623 |
| 5. Métodos y técnicas | PolynomialFeatures/PCA/FeatureUnion/importances, pp. 591-605; ensembles, pp. 606-611; confidence, pp. 612-615; sampling, pp. 615-620; regularization, pp. 621-623 |
| 6. Fórmulas esenciales | Regularization, pp. 621-622 |
| 7. Implementación práctica | Código de GridSearchCV, pp. 582-589; regularización, pp. 622-623; adaptaciones modernas identificadas como notas |
| 8. Supuestos y condiciones | Advertencias sobre test/CV, escala, complejidad, representatividad y costo, pp. 584-589, 591, 595-600, 607-620, 622 |
| 9. Errores y riesgos | Riesgos y limitaciones distribuidos en pp. 584, 589-590, 591, 604, 607-609, 612-620 y 622 |
| 10. Aplicación en coding test | Síntesis práctica derivada de pp. 582-623 |
| 11. Checklist operativo | Síntesis operativa derivada de pp. 581-623 |
| 12. Mapeo hacia `knowledge/` | Taxonomía derivada de las secciones del capítulo, pp. 582-623 |

