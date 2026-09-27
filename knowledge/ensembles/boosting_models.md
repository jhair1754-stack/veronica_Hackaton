---
title: "Ensembles y modelos de boosting para datos tabulares"
knowledge_area: "ensembles"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Ensembles y modelos de boosting para datos tabulares

## 1. Propósito y alcance

Este documento organiza los métodos de ensamble cubiertos por los resúmenes permitidos y pone el foco en *boosting* para clasificación y regresión tabular. Incluye *voting*, *bagging*, *random forests*, AdaBoost, *gradient boosting*, *stacking* y otras variantes solo hasta el nivel respaldado por esas fuentes.

No establece un ganador universal ni reemplaza la validación con los datos, la métrica y las restricciones del problema. Un ensamble solo merece adoptarse si supera un *baseline* comparable y su mejora compensa complejidad, tiempo, memoria, latencia e interpretabilidad.

Convención de trazabilidad usada en el texto:

- **Explícito de la fuente:** afirmación presentada directamente por uno o más resúmenes.
- **Síntesis integrada:** relación construida al combinar afirmaciones compatibles de varios resúmenes.
- **Conclusión práctica derivada:** consecuencia operativa razonada desde las fuentes, no cita directa.
- **Recomendación para coding test:** priorización práctica bajo tiempo limitado.

## 2. Qué es un ensemble

**Explícito de las fuentes.** Un *ensemble* combina las predicciones de varios modelos. La combinación puede ser fija —voto, promedio o promedio de probabilidades—, resultar del promedio de modelos entrenados con muestras o variables distintas, construirse secuencialmente para corregir errores anteriores, o aprenderse mediante un meta-modelo. La diversidad importa: cuando los miembros cometen errores poco correlacionados, la agregación puede cancelar parte del error; miembros muy similares limitan la ganancia. [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md) y [ML Design Patterns, cap. 3](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

**Síntesis integrada.** Las fuentes asocian *bagging* con reducción de varianza, *boosting* con corrección secuencial y posible reducción de sesgo, y *stacking* con una combinación aprendida. Esto no implica mejora automática: debe existir diversidad o señal residual aprovechable, y el diseño de evaluación debe impedir contaminación entre entrenamiento y validación.

## 3. Familias de métodos de ensamble

| Familia o método | Cómo combina modelos | Uso descrito en las fuentes | Ventaja | Limitación |
|---|---|---|---|---|
| *Voting* | Voto mayoritario (*hard*) o promedio/suma de probabilidades (*soft*); puede usar pesos | Clasificadores distintos con errores complementarios | Combinación simple de modelos ya entrenados | *Soft voting* exige probabilidades disponibles y comparables; modelos correlacionados aportan poco |
| *Bagging* | Promedia regresiones o vota clases de modelos ajustados sobre muestras *bootstrap* | Reducir la varianza de un aprendiz inestable, especialmente árboles | Paralelizable; admite estimación OOB | Más coste y menos interpretación; árboles correlacionados limitan la reducción de varianza |
| *Random forest* | Agrega árboles *bootstrap* y restringe aleatoriamente las variables candidatas en cada división | Clasificación o regresión tabular, en especial ante árboles de alta varianza o predictores dominantes/correlacionados | Descorrelaciona árboles y suele dar predicción estable | No siempre supera a *bagging*; usa más memoria y no conserva la explicación de un árbol único |
| AdaBoost | Secuencia de predictores ponderados; aumenta el peso de ejemplos mal clasificados | Clasificación con aprendices débiles y casos difíciles | Concentra pasos posteriores en errores previos | Secuencial; puede enfocarse demasiado en ruido y *outliers* |
| *Gradient boosting* | Suma árboles pequeños; cada nuevo árbol reduce la pérdida o ajusta residuos del ensamble actual | Clasificación y regresión tabular competitiva | Modelo flexible y compacto; corrige señal residual gradualmente | Requiere coordinar tasa, número y tamaño de árboles; puede sobreajustar y es serial |
| *Stacking* | Un meta-modelo aprende a combinar predicciones de modelos base | Modelos diversos con señales complementarias | Aprende una agregación más flexible que un promedio fijo | Predicciones *in-sample* producen *leakage*; añade coste y riesgo de sobreajuste |
| Extra-Trees | Agrega árboles con variables y umbrales de división aleatorios | Alternativa a *random forest* cuando se desea más aleatoriedad o menor coste de buscar cortes | Puede entrenar más rápido y aumentar diversidad | No puede saberse de antemano si superará a *random forest* |
| BART | Promedia, tras *burn-in*, predicciones de sumas de árboles pequeños perturbados secuencialmente | Principalmente regresión e incertidumbre predictiva en el resumen | Ofrece colección posterior de predicciones y suele requerir poca afinación | Mayor coste; clasificación y detalles bayesianos no están desarrollados suficientemente |

Fuentes de respaldo de la tabla: [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md), [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md), [Müller, cap. 2](../../sources/muller_introduction_ml_python/ch02_supervised_learning.md), [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md) y [ML Design Patterns, cap. 3](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

### 3.1 Voting

**Explícito de las fuentes.** *Hard voting* selecciona la clase con más votos. *Soft voting* promedia o suma probabilidades y puede ponderar la confianza; requiere estimadores con `predict_proba` y probabilidades comparables. `weights` controla la influencia relativa, pero debe validarse. Su objetivo es aprovechar errores complementarios de modelos distintos, no multiplicar copias equivalentes. [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md) y [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md).

### 3.2 Bagging

**Explícito de las fuentes.** *Bagging* entrena el mismo tipo de modelo sobre muestras con reemplazo y agrega las predicciones. En árboles, el promedio reduce varianza si los miembros no están perfectamente correlacionados. `n_estimators`, `max_samples`, `bootstrap`, `max_features`, `bootstrap_features` y los controles del estimador base aparecen como decisiones relevantes. Los miembros pueden entrenarse en paralelo. [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md) e [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md).

Con *bootstrap*, las observaciones no usadas por cada árbol permiten una estimación *out-of-bag* (OOB). OOB es una evaluación interna útil cuando hay suficientes árboles, pero no equivale necesariamente a una prueba final reservada. *Pasting* —muestreo sin reemplazo—, *Random Patches* y *Random Subspaces* están cubiertos por Géron, aunque con menos detalle operativo.

### 3.3 Random forests

**Explícito de las fuentes.** Un *random forest* añade a *bagging* un subconjunto aleatorio nuevo de predictores candidatos en cada división. Esta restricción reduce la correlación entre árboles y puede reducir más la varianza cuando hay predictores dominantes o correlacionados. En regresión agrega por promedio; en clasificación las fuentes describen voto mayoritario o promedio de probabilidades según la implementación. [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md) y [Müller, cap. 2](../../sources/muller_introduction_ml_python/ch02_supervised_learning.md).

Parámetros relevantes: `n_estimators`, `max_features`, profundidad, límites de hojas y `n_jobs`; `oob_score` habilita la estimación OOB cuando corresponde. Las importancias por impureza describen el uso de variables en las divisiones, no causalidad. Más árboles elevan coste y memoria; las fuentes indican que aumentar su número no se trata como el mismo mecanismo de sobreajuste que en *gradient boosting*, aunque sí conviene comprobar que el error se estabilice.

### 3.4 Boosting

**Explícito de las fuentes.** *Boosting* construye aprendices de forma secuencial: cada uno depende del ensamble anterior. Las variantes no deben confundirse:

- **AdaBoost:** repondera ejemplos mal clasificados y pondera los predictores según su error. Puede concentrarse excesivamente en ruido y *outliers*.
- **Gradient boosting:** añade árboles que corrigen residuos o reducen la pérdida del ensamble. En la exposición detallada de ISLP se inicializa la predicción, se ajusta un árbol pequeño a los residuos, se suma con tasa `learning_rate` y se actualizan los residuos.
- **Stochastic gradient boosting:** en Géron, `subsample < 1` entrena cada árbol con una fracción de instancias; aumenta sesgo, reduce varianza y coste.

`n_estimators`, `learning_rate`, profundidad o número de hojas, `min_samples_leaf`, `subsample` y `loss` aparecen en las fuentes, con distinto grado de cobertura. Una tasa menor suele requerir más árboles. A diferencia de *bagging*, demasiadas iteraciones o árboles demasiado complejos pueden sobreajustar. [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md), [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md) y [Müller, cap. 2](../../sources/muller_introduction_ml_python/ch02_supervised_learning.md).

### 3.5 Stacking y otras familias respaldadas

**Explícito de las fuentes.** *Stacking* entrena un *blender* o meta-modelo sobre predicciones de modelos base. Esas predicciones deben proceder de filas no vistas por el modelo base —partición *hold-out* o predicciones *out-of-fold*— para evitar *leakage*. Su ventaja es aprender la combinación; sus costes son una validación más delicada, más artefactos, latencia y menor interpretabilidad. [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md) y [ML Design Patterns, cap. 3](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

Extra-Trees está respaldado como alternativa más aleatoria a *random forest* y debe compararse experimentalmente. BART está respaldado por ISLP principalmente para regresión y cuantificación de incertidumbre, pero queda fuera del foco operativo por coste y cobertura limitada. XGBoost solo es mencionado por Géron como implementación rápida y frecuente en competiciones; el resumen no permite documentar aquí su algoritmo, API ni parámetros en detalle.

## 4. Cómo funciona el boosting

**Síntesis integrada.** El patrón común es construir una suma de contribuciones pequeñas y dependientes, pero la señal que recibe cada nuevo miembro cambia por variante:

1. Se parte de un predictor inicial o del primer aprendiz.
2. Se mide qué observaciones o residuos no están bien explicados.
3. Se ajusta un nuevo aprendiz a esa señal.
4. Se incorpora su contribución con un peso o tasa de aprendizaje.
5. Se repite hasta el número de iteraciones seleccionado con validación.

En AdaBoost, la señal son pesos mayores sobre ejemplos mal clasificados; los predictores también reciben un peso ligado a su error. En *gradient boosting* de regresión, cada árbol se ajusta a los residuos actuales y se añade multiplicado por la tasa de aprendizaje. Estas descripciones no son intercambiables: AdaBoost no debe describirse simplemente como ajuste de residuos, ni toda variante de *gradient boosting* como reponderación de observaciones.

**Explícito de las fuentes.** En el esquema de regresión de ISLP:

\[
\hat f(x) \leftarrow \hat f(x) + \lambda \hat f^b(x),
\qquad
r_i \leftarrow r_i - \lambda \hat f^b(x_i)
\]

Tras \(B\) pasos, la predicción es la suma de los árboles escalados. \(B\) controla el número de pasos, \(\lambda\) la velocidad de aprendizaje y la profundidad o número de divisiones \(d\) la complejidad de cada árbol y de las interacciones. Con árboles de una división, el resumen de ISLP describe un modelo aditivo sin interacciones. [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md).

`staged_predict()` permite observar el error por número de árboles y seleccionar el tamaño con validación; Géron también respalda *early stopping* y `warm_start`. La partición usada para escoger la iteración debe permanecer separada del ajuste, y el conjunto de prueba no debe dirigir esa selección.

## 5. Diferencias entre familias

| Método | Forma de combinación | Dependencia entre modelos | Aspecto que intenta mejorar | Precaución |
|---|---|---|---|---|
| *Voting* | Voto o promedio de probabilidades | Miembros entrenables de forma independiente | Aprovechar errores complementarios | Verificar diversidad y comparabilidad de probabilidades |
| *Bagging* | Promedio o voto sobre muestras *bootstrap* | Independientes y paralelizables | Varianza | Correlación entre miembros y validez de OOB |
| *Random forest* | *Bagging* más variables candidatas aleatorias por nodo | Independientes y paralelizables | Varianza y correlación entre árboles | `max_features` debe validarse; no siempre supera a *bagging* |
| AdaBoost | Voto ponderado de aprendices sucesivos | Cada paso depende de errores anteriores | Casos difíciles y capacidad del conjunto | Ruido, *outliers* y coste serial |
| *Gradient boosting* | Suma escalada de árboles pequeños | Secuencial | Residuos/pérdida y sesgo predictivo | Coordinar árboles, tasa y profundidad; vigilar sobreajuste |
| *Stacking* | Meta-modelo sobre predicciones base | Bases pueden ser independientes; meta-modelo depende de sus salidas | Aprender una combinación flexible | Usar predicciones *hold-out* u OOF, nunca *in-sample* |

**Síntesis integrada.** La diferencia principal no es “cuántos modelos” hay, sino cómo se induce diversidad, qué dependencia existe entre miembros y dónde puede entrar contaminación en la evaluación.

## 6. Selección de un método de ensamble

| Necesidad o condición | Método que puede considerarse | Motivo | Qué verificar |
|---|---|---|---|
| Árbol individual con gran brecha entre entrenamiento y validación | *Bagging* o *random forest* | La agregación puede reducir varianza | Misma validación, estabilidad del error, coste y ausencia de *leakage* |
| Predictor dominante o árboles muy correlacionados | *Random forest* con `max_features < p` | Da oportunidad a otras variables y descorrelaciona árboles | Comparar `max_features` mediante OOB o validación |
| Modelos distintos ya competitivos y con errores complementarios | *Voting* | Agrega diversidad sin meta-modelo | *Hard* frente a *soft*, probabilidades y pesos |
| Correcciones pequeñas siguen reduciendo el error de validación | *Gradient boosting* | Aprende secuencialmente señal residual | Curva por iteración, tasa, profundidad y tiempo |
| Casos mal clasificados merecen atención y el ruido está controlado | AdaBoost | Repondera errores sucesivos | Sensibilidad a ruido y *outliers* |
| La combinación fija parece insuficiente y hay presupuesto | *Stacking* | El meta-modelo aprende la agregación | Predicciones OOF/*hold-out*, sobreajuste y complejidad operativa |
| Se desea más aleatoriedad o menor coste de búsqueda de cortes | Extra-Trees | Usa umbrales aleatorios | Comparación empírica con *random forest* |
| Se necesita regresión con una distribución de predicciones y se acepta mayor coste | BART | Promedia predicciones posteriores al *burn-in* | Presupuesto, dependencia y cobertura limitada de clasificación |

**Conclusión práctica derivada.** Si el modelo simple ya satisface el objetivo o las restricciones de explicación, latencia o mantenimiento dominan, no hay justificación automática para añadir un ensamble.

## 7. Hiperparámetros y control de complejidad

| Hiperparámetro o decisión | Qué controla | Efecto atribuido por las fuentes | Riesgo de un valor inadecuado |
|---|---|---|---|
| `n_estimators` / \(B\) | Número de miembros o pasos | En bosques estabiliza el promedio; en *boosting* eleva capacidad | Pocos miembros dejan error inestable o *underfitting*; demasiados pasos de *boosting* pueden sobreajustar y cuestan más |
| `learning_rate` / \(\lambda\) | Tamaño de cada corrección en *boosting* | Tasas menores hacen el aprendizaje más gradual y suelen exigir más árboles | Tasa baja con pocos árboles infraajusta; combinación agresiva puede elevar complejidad |
| `max_depth`, `max_leaf_nodes`, número de divisiones | Complejidad del árbol base | Árboles pequeños limitan cada corrección; profundidad permite más interacciones | Árboles profundos elevan capacidad y riesgo de sobreajuste |
| `min_samples_leaf` y límites de nodo | Tamaño mínimo de regiones | Restringen hojas y regularizan árboles | Hojas demasiado pequeñas pueden memorizar; límites excesivos infraajustan |
| `max_features` | Variables candidatas por división o miembro | Menos variables puede aumentar diversidad y reducir correlación | Puede añadir sesgo; `max_features=p` equivale a *bagging* en el esquema de ISLP |
| `bootstrap`, `max_samples` | Muestreo de filas | Define *bagging* y cantidad de datos por miembro | Menos datos elevan diversidad pero pueden elevar sesgo; OOB requiere *bootstrap* apropiado |
| `subsample` | Fracción de filas por paso en *gradient boosting* | Valores menores que 1 producen *stochastic gradient boosting*, con menor varianza y coste pero más sesgo | Fracción inadecuada puede perder señal |
| `loss` | Función optimizada por *gradient boosting* | Cambia la señal que corrigen los árboles | Las fuentes no dan un criterio universal de elección |
| `weights` y `voting` | Regla de combinación en *voting* | Ajustan influencia y uso de clases o probabilidades | Pesos sin evidencia o probabilidades no comparables pueden degradar el conjunto |
| Agregador/meta-modelo | Flexibilidad de *stacking* | Aprende pesos e interacciones entre salidas | Puede sobreajustar y filtrar información si usa predicciones *in-sample* |

No se proponen rangos universales. La selección debe hacerse con validación, registrar métrica y coste, y reservar el test para la estimación final. La búsqueda sistemática corresponde a `knowledge/tuning/hyperparameter_tuning.md`, no a este documento. [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md).

## 8. Sobreajuste, limitaciones y costo computacional

**Explícito de las fuentes.** Los principales riesgos documentados son:

- *Boosting* es secuencial y no se paraleliza como *bagging*; el entrenamiento puede ser más lento.
- Demasiados pasos o árboles complejos pueden sobreajustar en *gradient boosting*; AdaBoost puede enfatizar ruido y *outliers*.
- *Bagging* y bosques permiten paralelismo, pero cientos o miles de árboles consumen tiempo y memoria.
- *Stacking* añade entrenamiento, inferencia, mantenimiento y una ruta específica de *leakage* si el meta-modelo ve predicciones generadas sobre filas usadas para ajustar los modelos base.
- Los ensambles pierden la representación de un único árbol y sus importancias por impureza no son explicaciones causales.
- *Soft voting* puede encarecer un estimador; Géron señala específicamente el coste adicional de habilitar probabilidades en SVC.
- Los modelos arbóreos conservan limitaciones de extrapolación y no son la opción destacada por Müller para datos *sparse* de altísima dimensión.

**Conclusión práctica derivada.** Rendimiento predictivo, coste e interpretabilidad deben medirse por separado. Una mejora marginal no justifica por sí sola un ensamble más caro, y ninguna estimación OOB reemplaza automáticamente una evaluación final coherente con el problema.

## 9. Implementación mínima en Python

Los fragmentos asumen que `X_train`, `X_valid`, `y_train` y `y_valid` ya fueron separados. Cualquier transformación aprendida debe ajustarse solo con entrenamiento o dentro de un `Pipeline`/pliegue de validación.

Clasificación con árboles potenciados:

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
```

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** La API y la configuración aparecen en el resumen de [Müller, cap. 2](../../sources/muller_introduction_ml_python/ch02_supervised_learning.md); los valores son los de su fragmento adaptado y no constituyen rangos recomendados.

Regresión con selección del número de árboles sobre validación:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=120,
    random_state=42,
)
model.fit(X_train, y_train)

validation_errors = [
    mean_squared_error(y_valid, prediction)
    for prediction in model.staged_predict(X_valid)
]
best_n_estimators = min(
    range(1, len(validation_errors) + 1),
    key=lambda n: validation_errors[n - 1],
)
```

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** El fragmento procede del resumen de [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md). `y_valid` guía la selección; no debe reemplazarse por el test final.

## 10. Reglas de decisión

| Situación | Método que puede considerarse | Motivo | Riesgo o precaución |
|---|---|---|---|
| Alta varianza observada fuera de entrenamiento | *Bagging* | Promedia aprendices inestables | Sin diversidad, la reducción es limitada |
| Árboles correlacionados por variables dominantes | *Random forest* | Aleatoriza candidatos por nodo | Validar `max_features`; no garantiza vencer a *bagging* |
| Alto sesgo o señal residual que aún mejora validación | *Boosting* | Añade capacidad de manera secuencial | Más capacidad también puede sobreajustar |
| Modelos heterogéneos con desacuerdos útiles | *Voting* | Combina predicciones con una regla simple | *Soft voting* exige probabilidades comparables |
| Promedio o voto no captura bien la combinación | *Stacking* | Aprende una función de agregación | Generar predicciones OOF/*hold-out* y controlar coste |
| Presupuesto estricto o necesidad de explicación exacta | Modelo único o árbol podado | Menor complejidad operativa e interpretativa | Confirmar que satisface la métrica objetivo |
| Se dispone de poco tiempo para competir | *Random forest* o *gradient boosting* como candidatos, no ganadores | Las fuentes los describen como opciones tabulares fuertes | Comparar con baseline usando el mismo protocolo |

Las seis primeras filas son **síntesis integradas** de las fuentes; la última es una **recomendación para coding test** apoyada por las secciones prácticas de Géron, Müller, Molin y ML Design Patterns.

## 11. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Definir la tarea y una métrica; separar validación y test antes de ajustar decisiones.
2. Entrenar un *baseline* simple y registrar su configuración, puntuación y tiempo.
3. Diagnosticar la razón para probar un ensamble: varianza, sesgo, errores complementarios o relación tabular no lineal.
4. Elegir una sola familia coherente con ese diagnóstico y comenzar con una configuración acotada.
5. Compararla con el *baseline* usando exactamente la misma partición o estrategia de validación y la misma métrica.
6. Registrar hiperparámetros, semilla, puntuación, tiempo de ajuste y, si importa, coste de inferencia.
7. Revisar la curva de validación por número de árboles en *boosting*; no usar el test para elegir la iteración.
8. Exigir una mejora repetible antes de ampliar el *tuning*. Bajo presión, evitar una malla enorme o *stacking* multicapa sin evidencia.
9. Considerar memoria, latencia, interpretabilidad y tiempo restante antes de conservar la solución.
10. Evaluar una sola vez en el test reservado después de cerrar las decisiones.

Estas recomendaciones no son citas directas. Integran el flujo de comparación y coste de [Molin, cap. 10](../../sources/molin_hands_on_pandas/ch10_model_optimization.md), las precauciones de validación de [Géron, cap. 7](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md) e [ISLP, cap. 8](../../sources/islp_python/ch08_tree_based_methods.md), y la priorización competitiva de [ML Design Patterns, cap. 3](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md).

## 12. Checklist operativo

- [ ] Establecí un *baseline* comparable.
- [ ] Confirmé que la familia de ensamble responde a un diagnóstico de varianza, sesgo o diversidad.
- [ ] Separé entrenamiento, validación y test antes de ajustar decisiones.
- [ ] Revisé los parámetros que controlan número y complejidad de miembros.
- [ ] Comparé con la misma estrategia de validación y métrica.
- [ ] Evité predicciones *in-sample* al entrenar un meta-modelo.
- [ ] Revisé rendimiento, estabilidad, tiempo, memoria e inferencia.
- [ ] Registré configuración, semilla y resultado.
- [ ] Verifiqué que la mejora se mantiene fuera de entrenamiento.
- [ ] Reservé el test final para una evaluación posterior a la selección.

## 13. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con ensembles y boosting |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Define la tarea que debe resolver el ensamble |
| `knowledge/eda/dataset_audit.md` | Aporta información sobre estructura y calidad de los datos |
| `knowledge/preprocessing/data_cleaning.md` | Describe la preparación necesaria de datos |
| `knowledge/preprocessing/data_leakage.md` | Ayuda a prevenir fugas durante el entrenamiento y evaluación |
| `knowledge/feature_engineering/feature_engineering.md` | Trata la representación de variables |
| `knowledge/validation/validation_strategy.md` | Define cómo evaluar los métodos |
| `knowledge/metrics/classification_metrics.md` | Define métricas para clasificación |
| `knowledge/metrics/regression_metrics.md` | Define métricas para regresión |
| `knowledge/models/tabular_models.md` | Compara modelos individuales con los ensambles |
| `knowledge/tuning/hyperparameter_tuning.md` | Desarrolla la búsqueda de parámetros |

## 14. Vacíos y limitaciones de las fuentes

- XGBoost se menciona, pero no se desarrolla con detalle suficiente para documentar su algoritmo, parámetros, API o criterios de selección.
- BART está desarrollado principalmente para regresión; la clasificación queda fuera del desarrollo principal del resumen.
- *Pasting*, *Random Patches* y *Random Subspaces* aparecen en Géron, pero con menor cobertura comparativa que *bagging* y *random forest*.
- Las fuentes no ofrecen una regla universal para escoger `loss`, tasa de aprendizaje, número de árboles, profundidad, `subsample`, `max_features` o pesos de *voting*.
- No hay rangos universales transferibles; los valores de los ejemplos no deben tratarse como recomendaciones generales.
- AdaBoost tiene respaldo conceptual y fórmulas, pero el documento no incluye código para evitar convertir una configuración ilustrativa en receta general.
- La calibración necesaria para comparar probabilidades en *soft voting* se menciona como condición, pero no se desarrolla un procedimiento de calibración.
- La cobertura de *stacking* explica la separación *hold-out*/OOF y el riesgo de *leakage*, pero no proporciona un protocolo completo para múltiples capas.
- No se documentan resultados comparativos generales entre familias; las decisiones requieren validación experimental con cada dataset.
- Versiones modernas de APIs pueden diferir de las descritas originalmente; las adaptaciones incluidas están marcadas.

## 15. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning*, cap. 7 | [Resumen](../../sources/geron_hands_on_ml/ch07_ensemble_learning_and_random_forests.md) | Definición de ensemble; *voting*, *bagging*, OOB, *random forest*, Extra-Trees, AdaBoost, *gradient boosting*, *stochastic boosting*, *early stopping* y *stacking* |
| ISLP, cap. 8 | [Resumen](../../sources/islp_python/ch08_tree_based_methods.md) | Sesgo-varianza de árboles; *bagging*, OOB, *random forest*, algoritmo e hiperparámetros de *boosting* de regresión, BART, riesgos y coste |
| Müller y Guido, cap. 2 | [Resumen](../../sources/muller_introduction_ml_python/ch02_supervised_learning.md) | *Random forest* y *gradient boosted decision trees* para tabular; clasificación/regresión; complejidad, extrapolación e implementación moderna |
| Molin, cap. 10 | [Resumen](../../sources/molin_hands_on_pandas/ch10_model_optimization.md) | Comparación de *bagging*, *boosting* y *voting*; validación, coste, sensibilidad al ruido y control de tuning sin contaminar test |
| *Machine Learning Design Patterns*, cap. 3 | [Resumen](../../sources/ml_design_patterns/ch03_problem_representation_patterns.md) | Patrón Ensembles: diagnóstico de sesgo/varianza, diversidad, combinación, coste y priorización en competiciones |

## 16. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Catálogo y cinco resúmenes utilizados | Síntesis integrada |
| 2. Qué es un ensemble | Géron 7; ML Design Patterns 3 | Explícito de la fuente / Síntesis integrada |
| 3.1 Voting | Géron 7; Molin 10 | Explícito de la fuente |
| 3.2 Bagging | Géron 7; ISLP 8 | Explícito de la fuente |
| 3.3 Random forests | Géron 7; ISLP 8; Müller 2 | Explícito de la fuente |
| 3.4 Boosting y variantes | Géron 7; ISLP 8; Müller 2; Molin 10 | Explícito de la fuente / Síntesis integrada |
| 3.5 Stacking, Extra-Trees, BART y límite sobre XGBoost | Géron 7; ISLP 8; ML Design Patterns 3 | Explícito de la fuente / Síntesis integrada |
| 4. Funcionamiento de boosting | Géron 7; ISLP 8 | Explícito de la fuente / Síntesis integrada |
| 5. Diferencias entre familias | Géron 7; ISLP 8; Molin 10; ML Design Patterns 3 | Síntesis integrada |
| 6. Selección de método | Cinco resúmenes utilizados | Conclusión práctica derivada |
| 7. Hiperparámetros | Géron 7; ISLP 8; Müller 2; Molin 10 | Explícito de la fuente / Síntesis integrada |
| 8. Sobreajuste, limitaciones y coste | Cinco resúmenes utilizados | Explícito de la fuente / Conclusión práctica derivada |
| 9. Código de clasificación | Müller 2 | Explícito de la fuente / adaptación moderna declarada |
| 9. Código de regresión | Géron 7 | Explícito de la fuente / adaptación moderna declarada |
| 10. Reglas de decisión | Cinco resúmenes utilizados | Síntesis integrada / Recomendación para coding test |
| 11. Coding test o datathon | Géron 7; ISLP 8; Molin 10; ML Design Patterns 3 | Recomendación para coding test |
| 12. Checklist | Cinco resúmenes utilizados | Conclusión práctica derivada |
| 13. Documentos relacionados | Estructura requerida por el encargo | Síntesis integrada |
| 14. Vacíos y limitaciones | Cinco resúmenes utilizados | Conclusión práctica derivada |

Las categorías combinadas indican que una sección contiene tanto hechos explícitos como su integración o consecuencia práctica; las etiquetas locales del texto permiten distinguirlos.
