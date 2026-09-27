---
title: "Tree-Based Methods"
topics:
  - "Decision trees for regression and classification"
  - "Cost-complexity pruning"
  - "Bagging and out-of-bag error"
  - "Random forests"
  - "Boosting"
  - "Bayesian additive regression trees (BART)"
target_knowledge_files:
  - "knowledge/models/decision_trees.md"
  - "knowledge/models/tree_ensembles.md"
  - "knowledge/validation/oob_and_cross_validation.md"
  - "knowledge/tuning/tree_based_models.md"
  - "knowledge/interpretability/tree_feature_importance.md"
---

# 8. Tree-Based Methods

## 1. Propósito del capítulo

El capítulo presenta métodos basados en árboles para regresión y clasificación. Un árbol divide el espacio de predictores en regiones rectangulares y asigna una predicción constante dentro de cada región. Su principal ventaja es la interpretabilidad; sus desventajas son la inestabilidad y, con frecuencia, una precisión predictiva inferior a la de otros métodos supervisados.

Para mejorar la predicción, el capítulo desarrolla cuatro ensambles de árboles: *bagging*, *random forests*, *boosting* y *Bayesian additive regression trees* (BART). Todos combinan múltiples árboles, pero difieren en cómo generan y relacionan esos árboles.

## 2. Conceptos esenciales

### Árboles de decisión

- Un nodo interno contiene una regla de partición, por ejemplo, `Years < 4.5`.
- Una rama representa uno de los resultados de la regla.
- Un nodo terminal u hoja (*terminal node*, *leaf*) corresponde a una región del espacio de predictores.
- En regresión, la predicción de una hoja es la media de las respuestas de entrenamiento que caen en ella.
- En clasificación, la predicción es la clase mayoritaria; las proporciones de clase de la hoja expresan la confianza o pureza local.
- La partición se construye mediante división binaria recursiva (*recursive binary splitting*): un procedimiento descendente y voraz que elige en cada paso el predictor y punto de corte con la mejor mejora inmediata.
- Las fronteras producidas por este procedimiento son paralelas a los ejes. Esto facilita la interpretación, pero restringe las formas que el árbol puede representar eficientemente.

### Complejidad y poda

Un árbol grande suele ajustar muy bien el entrenamiento, pero puede tener alta varianza y sobreajustar. La poda por complejidad de coste (*cost-complexity pruning* o *weakest-link pruning*) genera una secuencia anidada de subárboles. El parámetro no negativo \(\alpha\) penaliza el número de hojas: al aumentar \(\alpha\), se favorecen árboles más pequeños. El valor debe elegirse con validación o validación cruzada, no con el error de entrenamiento.

### Criterios de clasificación

- La tasa de error de clasificación mide la fracción que no pertenece a la clase mayoritaria de una región.
- El índice Gini y la entropía miden pureza y son más sensibles que la tasa de error para decidir divisiones.
- Para construir el árbol se recomienda Gini o entropía. Para podar cuando el objetivo final es exactitud de clasificación, el capítulo prefiere la tasa de error.
- Una división puede dejar la misma clase predicha en ambos hijos y aun ser útil: puede aumentar la pureza y diferenciar la confianza de las predicciones.

### Ensambles

- **Bagging:** entrena árboles profundos y sin podar sobre muestras *bootstrap* independientes y promedia sus predicciones o aplica voto mayoritario. Reduce la varianza de árboles individuales.
- **Random forest:** añade a *bagging* una selección aleatoria de \(m\) predictores candidatos en cada división. Esto descorrelaciona los árboles y puede reducir más la varianza.
- **Boosting:** construye árboles pequeños de forma secuencial; cada árbol intenta explicar los residuos que deja el ensamble actual. Aprende gradualmente mediante una tasa \(\lambda\).
- **BART:** mantiene una suma de árboles pequeños y los perturba secuencialmente para ajustar residuos parciales. Promedia predicciones posteriores a un período de calentamiento (*burn-in*) y permite resumir incertidumbre mediante percentiles de esas predicciones.

### Interpretabilidad y estabilidad

Un árbol pequeño es fácil de explicar y visualizar, acepta relaciones no lineales y puede manejar predictores cualitativos conceptualmente sin variables indicadoras. Sin embargo, un cambio pequeño en los datos puede producir un árbol muy diferente. Los ensambles mejoran la precisión y estabilidad, pero pierden la representación mediante un único árbol.

## 3. Flujo o procedimiento presentado

### Árbol de regresión con poda

1. Dividir los datos en entrenamiento y evaluación; cualquier selección de complejidad ocurre únicamente dentro del entrenamiento.
2. Hacer crecer un árbol grande con división binaria recursiva, deteniéndose cuando las hojas contienen menos que un mínimo de observaciones u otro criterio de parada.
3. Aplicar poda por complejidad de coste al árbol grande para obtener una secuencia de mejores subárboles indexados por \(\alpha\).
4. Para cada pliegue de validación cruzada, repetir el crecimiento y la poda sin usar el pliegue retenido.
5. Medir el MSE de cada \(\alpha\) en los pliegues retenidos y promediarlo.
6. Elegir el \(\alpha\) con menor error medio.
7. Recuperar del árbol grande el subárbol asociado a ese \(\alpha\).
8. Evaluar una sola vez sobre el conjunto de prueba.

### Árbol de clasificación

1. Elegir Gini o entropía como criterio de división.
2. En cada nodo, evaluar predictores y cortes candidatos y seleccionar la mayor mejora de pureza.
3. Continuar recursivamente hasta el criterio de parada.
4. Estimar por hoja tanto la clase mayoritaria como las proporciones de clase.
5. Si se poda, elegir \(\alpha\) mediante validación cruzada sobre entrenamiento.
6. Informar una métrica de prueba apropiada y, cuando sea útil, una matriz de confusión.

### Bagging y random forest

1. Generar \(B\) muestras *bootstrap* a partir del entrenamiento.
2. Ajustar un árbol profundo y sin podar por muestra.
3. En *bagging*, permitir los \(p\) predictores en cada división; en *random forest*, sortear \(m<p\) candidatos nuevos en cada nodo.
4. Para regresión, promediar las \(B\) predicciones; para clasificación, usar voto mayoritario.
5. Calcular el error *out-of-bag* (OOB) usando para cada observación solo los árboles que no la incluyeron.
6. Aumentar \(B\) hasta que el error se estabilice y comparar valores de \(m\).

### Boosting de regresión

1. Inicializar \(\hat f(x)=0\) y los residuos \(r_i=y_i\).
2. Ajustar un árbol pequeño de \(d\) divisiones a \((X,r)\).
3. Sumar al modelo el nuevo árbol multiplicado por \(\lambda\).
4. Restar esa contribución de los residuos.
5. Repetir hasta \(B\) árboles.
6. Elegir \(B\), \(\lambda\) y \(d\) mediante validación, porque *boosting* sí puede sobreajustar al crecer demasiado.

### BART

1. Inicializar \(K\) árboles de una sola hoja para que su suma reproduzca la media de \(Y\).
2. En cada iteración, actualizar cada árbol por turno.
3. Para el árbol \(k\), calcular residuos parciales restando de \(Y\) la contribución de los otros \(K-1\) árboles.
4. Perturbar aleatoriamente el árbol anterior: agregar o podar ramas y modificar predicciones de hojas, favoreciendo cambios que mejoren el ajuste al residuo parcial.
5. Sumar los \(K\) árboles para obtener la predicción de la iteración.
6. Descartar las primeras \(L\) iteraciones de *burn-in*.
7. Promediar las predicciones restantes; usar sus percentiles si se necesita una medida de incertidumbre.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se necesita una regla pequeña, visual y explicable | Árbol podado | La estructura y las regiones se interpretan directamente | Puede perder precisión y ser inestable ante cambios pequeños de datos |
| La relación parece aproximadamente lineal | Modelo lineal antes que árbol único | Un modelo lineal explota esa estructura y puede superar al árbol | Confirmar la comparación con error de validación o prueba |
| La frontera o relación es compleja y no lineal | Árbol o ensamble de árboles | Las particiones capturan no linealidades e interacciones sin especificarlas de antemano | Las fronteras de un árbol son paralelas a los ejes |
| Un árbol presenta alta varianza | Bagging | El promedio de árboles *bootstrap* reduce varianza | Los árboles correlacionados limitan la reducción de varianza |
| Existe un predictor dominante o muchos predictores correlacionados | Random forest con \(m<p\) | Restringir candidatos descorrelaciona los árboles y da oportunidad a otros predictores | Comparar \(m\) con OOB o validación; \(m=p\) equivale a *bagging* |
| El coste de validación cruzada de un bosque es alto | Error OOB | Predice cada fila solo con árboles que no la usaron; con \(B\) grande se aproxima a LOOCV | Esperar a que haya suficientes árboles y que el error se estabilice |
| Se busca mejorar gradualmente el residuo con árboles pequeños | Boosting | Cada árbol ataca señal no capturada por los anteriores | Ajustar conjuntamente \(B\), \(\lambda\) y profundidad; puede sobreajustar |
| Se desea un modelo aditivo sin interacciones | Boosting con tocones, \(d=1\) | Cada árbol usa una sola variable | Puede infraajustar si las interacciones son importantes |
| Se desean buen rendimiento inicial y una distribución de predicciones | BART | Promedia muestras posteriores al *burn-in* y suele requerir poca afinación | Tiene mayor coste y el capítulo desarrolla principalmente regresión |

## 5. Métodos y técnicas

### Árbol de regresión

- **Objetivo:** predecir una respuesta cuantitativa mediante regiones disjuntas.
- **Entradas:** matriz de predictores \(X\) y respuesta numérica \(y\).
- **Salida:** reglas binarias, regiones terminales y media predicha por región.
- **Funcionamiento:** selecciona repetidamente el predictor y corte que minimizan el RSS resultante; el procedimiento es voraz y no revisa decisiones anteriores.
- **Hiperparámetros:** profundidad, mínimo de observaciones para dividir o por hoja y \(\alpha\) de poda.
- **Ventajas:** interpretación directa, visualización y representación de no linealidades e interacciones.
- **Limitaciones:** alta varianza, inestabilidad, superficie escalonada y posible desventaja si la verdad es lineal.
- **Casos adecuados:** reglas explicables y relaciones complejas donde una partición rectangular sea razonable.
- **No conviene:** cuando prima la máxima exactitud y un ensamble es viable, o cuando la relación es esencialmente lineal.

### Árbol de clasificación

- **Objetivo:** predecir una respuesta cualitativa.
- **Entradas:** \(X\) y etiquetas de clase.
- **Salida:** clase mayoritaria y proporciones de clase por hoja.
- **Funcionamiento:** división binaria recursiva guiada normalmente por Gini o entropía.
- **Hiperparámetros:** criterio (`gini` o `entropy`), profundidad, mínimos por nodo/hoja y \(\alpha\) de poda.
- **Ventajas:** reglas legibles y probabilidades locales derivadas de proporciones de clase.
- **Limitaciones:** las mismas de los árboles de regresión; la tasa de error es poco sensible para crecer el árbol.
- **Casos adecuados:** clasificación con estructura no lineal y necesidad de explicación.
- **No conviene:** cuando pequeñas variaciones de muestra no deberían cambiar sustancialmente la decisión o cuando un ensamble ofrece una mejora necesaria.

### Poda por complejidad de coste

- **Objetivo:** reducir la varianza de un árbol grande con una penalización explícita por número de hojas.
- **Entradas:** árbol grande y candidatos \(\alpha\).
- **Salida:** secuencia anidada de subárboles y un subárbol elegido por validación.
- **Funcionamiento:** minimiza error de entrenamiento más \(\alpha|T|\).
- **Hiperparámetro principal:** \(\alpha\).
- **Ventajas:** evalúa una secuencia manejable de subárboles en lugar de todos los subárboles posibles.
- **Limitaciones:** la selección por validación tiene varianza y podar no garantiza mejorar la prueba en una partición concreta.
- **Casos adecuados:** árboles únicos que crecieron deliberadamente grandes.
- **No conviene:** seleccionar \(\alpha\) observando el conjunto de prueba.

### Bagging

- **Objetivo:** reducir la varianza promediando árboles de alta varianza y bajo sesgo.
- **Entradas:** entrenamiento, número de árboles \(B\) y todos los predictores disponibles en cada división.
- **Salida:** promedio de regresión o voto mayoritario de clasificación.
- **Funcionamiento:** cada árbol profundo y sin podar se ajusta a una muestra *bootstrap* independiente.
- **Hiperparámetros:** principalmente \(B\); en una implementación de bosque, `max_features=p`.
- **Ventajas:** mejora frecuente de precisión; ofrece estimación OOB sin validación cruzada adicional.
- **Limitaciones:** pierde la interpretación de un árbol único y puede producir árboles muy correlacionados.
- **Casos adecuados:** árbol base inestable y coste aceptable de entrenar muchos árboles.
- **No conviene:** cuando se exige una única regla plenamente explicable.

### Random forest

- **Objetivo:** reducir varianza más que *bagging* al disminuir la correlación entre árboles.
- **Entradas:** las de *bagging* y tamaño \(m\) del subconjunto de predictores por división.
- **Salida:** predicción agregada, error OOB e importancia basada en disminución de impureza.
- **Funcionamiento:** en cada nodo se sortea un nuevo subconjunto de \(m\) predictores y solo ellos compiten por la división.
- **Hiperparámetros:** \(B\), \(m\), profundidad y mínimos de nodo/hoja.
- **Ventajas:** robustez predictiva, manejo flexible de no linealidades y descorrelación de árboles.
- **Limitaciones:** menor interpretabilidad; la importancia por impureza resume asociación con divisiones, no causalidad.
- **Casos adecuados:** muchos predictores correlacionados o un predictor dominante.
- **No conviene:** asumir que siempre vence a *bagging*; el laboratorio muestra un caso donde no ocurre.

### Boosting

- **Objetivo:** construir una suma fuerte de árboles pequeños que corrijan secuencialmente errores previos.
- **Entradas:** datos originales; en regresión, residuos actualizados en cada paso.
- **Salida:** suma ponderada de \(B\) árboles.
- **Funcionamiento:** cada árbol depende de los anteriores; no usa muestras *bootstrap* en el esquema presentado.
- **Hiperparámetros:** número de árboles \(B\), *learning rate* o *shrinkage* \(\lambda\), y número de divisiones/profundidad \(d\).
- **Ventajas:** aprendizaje lento y flexible; árboles pequeños suelen bastar.
- **Limitaciones:** entrenamiento secuencial, afinación conjunta y riesgo de sobreajuste con demasiadas iteraciones.
- **Casos adecuados:** cuando una secuencia de correcciones pequeñas mejora el error de validación.
- **No conviene:** fijar un \(B\) grande sin seguir el error fuera de muestra.

### BART

- **Objetivo:** ajustar una suma bayesiana de árboles y producir una colección de predicciones posteriores.
- **Entradas:** \(X\), respuesta cuantitativa, \(K\) árboles, \(B\) iteraciones y \(L\) iteraciones de *burn-in*.
- **Salida:** promedio posterior de predicción y percentiles que cuantifican incertidumbre.
- **Funcionamiento:** actualiza cada árbol usando residuos parciales y pequeñas perturbaciones aleatorias de su estructura y valores terminales.
- **Hiperparámetros:** \(K\), \(B\) y \(L\); el capítulo propone como valores razonables \(K=200\), \(B=1000\), \(L=100\).
- **Ventajas:** rendimiento competitivo con poca afinación y control del ajuste mediante árboles pequeños y perturbaciones.
- **Limitaciones:** mayor coste; los detalles bayesianos y la clasificación quedan fuera del desarrollo principal.
- **Casos adecuados:** regresión donde importan rendimiento inicial e incertidumbre predictiva.
- **No conviene:** cuando la dependencia o el coste de muestreo no caben en el entorno de ejecución.

## 6. Fórmulas esenciales

### RSS de una partición

\[
\sum_{j=1}^{J}\sum_{i:x_i\in R_j}(y_i-\hat y_{R_j})^2
\]

- \(R_j\): región terminal \(j\); \(J\): número de regiones.
- \(y_i\): respuesta observada; \(\hat y_{R_j}\): media de las respuestas de entrenamiento en \(R_j\).
- Mide el error cuadrático dentro de las hojas. La división busca la mayor reducción de este valor.
- Supone una predicción constante e igual a la media dentro de cada región.

### Candidato de división binaria

\[
R_1(j,s)=\{X\mid X_j<s\},\qquad R_2(j,s)=\{X\mid X_j\ge s\}
\]

Se eligen el predictor \(j\) y corte \(s\) que minimizan la suma de RSS de ambas regiones. El método es voraz: optimiza el paso actual, no el árbol futuro completo.

### Poda por complejidad de coste

\[
\sum_{m=1}^{|T|}\sum_{i:x_i\in R_m}(y_i-\hat y_{R_m})^2+\alpha|T|
\]

- \(T\): subárbol; \(|T|\): número de hojas.
- \(\alpha\ge0\): penalización por complejidad.
- Con \(\alpha=0\) se favorece el árbol grande por ajuste de entrenamiento; al crecer \(\alpha\), se prefieren subárboles menores.
- La elección de \(\alpha\) requiere validación fuera de las observaciones usadas para ajustar cada árbol.

### Error de clasificación de un nodo

\[
E=1-\max_k(\hat p_{mk})
\]

- \(\hat p_{mk}\): proporción de la clase \(k\) entre las observaciones de entrenamiento de la región \(m\).
- Mide la fracción que quedaría mal clasificada al predecir la clase mayoritaria.
- Es útil para evaluar predicción final, pero poco sensible para elegir divisiones.

### Índice Gini

\[
G=\sum_{k=1}^{K}\hat p_{mk}(1-\hat p_{mk})
\]

- \(K\): número de clases.
- Mide la variabilidad total entre clases de un nodo.
- Valores cercanos a cero indican alta pureza: las proporciones están cerca de cero o uno.

### Entropía

\[
D=-\sum_{k=1}^{K}\hat p_{mk}\log(\hat p_{mk})
\]

- Usa las mismas proporciones de clase.
- También se aproxima a cero en nodos puros y suele comportarse de modo parecido a Gini.
- Se adopta la convención límite correspondiente cuando una proporción es cero.

### Predicción de bagging

\[
\hat f_{bag}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat f^{*b}(x)
\]

- \(B\): número de muestras *bootstrap* y árboles.
- \(\hat f^{*b}\): modelo ajustado sobre la muestra *bootstrap* \(b\).
- En regresión se promedia; en clasificación, el capítulo presenta voto mayoritario.
- El beneficio proviene de reducir varianza; disminuye si las predicciones están muy correlacionadas.

### Actualización de boosting de regresión

\[
\hat f(x)\leftarrow\hat f(x)+\lambda\hat f^b(x),\qquad
r_i\leftarrow r_i-\lambda\hat f^b(x_i)
\]

La salida tras \(B\) pasos es:

\[
\hat f(x)=\sum_{b=1}^{B}\lambda\hat f^b(x)
\]

- \(\hat f^b\): árbol pequeño ajustado a los residuos actuales.
- \(\lambda>0\): tasa de aprendizaje; valores pequeños ralentizan el aprendizaje y pueden requerir más árboles.
- \(d\): número de divisiones por árbol; controla la profundidad de interacción. Con \(d=1\), el modelo es aditivo.

### Promedio posterior de BART

\[
\hat f(x)=\frac{1}{B-L}\sum_{b=L+1}^{B}\hat f^b(x)
\]

- \(B\): iteraciones totales; \(L\): iteraciones descartadas como *burn-in*.
- \(\hat f^b(x)\): suma de los \(K\) árboles en la iteración \(b\).
- El promedio produce la predicción final; los percentiles de las predicciones retenidas expresan incertidumbre.

## 7. Implementación práctica

### Árbol de clasificación con poda seleccionada dentro del entrenamiento

```python
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0, stratify=y
)

base_tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
base_tree.fit(X_train, y_train)
alpha_path = base_tree.cost_complexity_pruning_path(X_train, y_train)

cv = KFold(n_splits=10, shuffle=True, random_state=1)
search = GridSearchCV(
    DecisionTreeClassifier(criterion="entropy", random_state=0),
    {"ccp_alpha": alpha_path.ccp_alphas},
    scoring="accuracy",
    cv=cv,
    refit=True,
)
search.fit(X_train, y_train)
test_accuracy = accuracy_score(y_test, search.predict(X_test))
```

**Contenido explícito del capítulo:** `cost_complexity_pruning_path()`, `GridSearchCV`, `KFold`, `ccp_alpha`, separación entrenamiento/prueba y evaluación final sobre prueba.

**Nota práctica derivada:** `stratify=y` conserva aproximadamente la proporción de clases en la partición; no aparece en el laboratorio. Si hay preprocesamiento aprendido, debe ajustarse dentro de cada pliegue mediante un `Pipeline` para evitar *leakage*.

### Bagging y random forest para regresión

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

bagging = RandomForestRegressor(
    n_estimators=500,
    max_features=X_train.shape[1],
    oob_score=True,
    random_state=0,
    n_jobs=-1,
)
bagging.fit(X_train, y_train)
bagging_mse = mean_squared_error(y_test, bagging.predict(X_test))

forest = RandomForestRegressor(
    n_estimators=500,
    max_features=6,
    oob_score=True,
    random_state=0,
    n_jobs=-1,
)
forest.fit(X_train, y_train)
forest_mse = mean_squared_error(y_test, forest.predict(X_test))
```

**Contenido explícito del capítulo:** usar `max_features=p` convierte el bosque en *bagging*; un valor menor produce *random forest*. El laboratorio compara MSE de prueba e inspecciona `feature_importances_`.

**Adaptación moderna:** `mean_squared_error()` hace explícita la métrica y `n_jobs=-1` aprovecha paralelismo; no alteran el método descrito. `oob_score=True` materializa la estimación OOB explicada en la teoría.

### Boosting de regresión

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

boosted_tree = GradientBoostingRegressor(
    n_estimators=5000,
    learning_rate=0.001,
    max_depth=3,
    random_state=0,
)
boosted_tree.fit(X_train, y_train)
boosting_mse = mean_squared_error(y_test, boosted_tree.predict(X_test))
```

Para estudiar sobreajuste, el laboratorio recorre `staged_predict()` y compara error de entrenamiento y prueba a lo largo del número de árboles. En una competencia, la prueba no debe usarse para elegir la iteración: esa curva debe construirse con validación.

### BART

```python
from ISLP.bart import BART

bart = BART(random_state=0, burnin=100, ndraw=1000)
bart.fit(X_train, y_train)
y_pred = bart.predict(X_test.astype("float32"))
```

El laboratorio usa una implementación de `ISLP` con pocas extracciones solo para demostrar la API. Los valores mayores de *burn-in* e iteraciones de la sección teórica son más representativos, sujetos al presupuesto computacional.

## 8. Supuestos y condiciones de aplicación

1. Las observaciones usadas para evaluar deben quedar fuera del ajuste y de la selección de hiperparámetros.
2. La predicción de un árbol es constante dentro de cada hoja; en regresión esa constante es la media de entrenamiento.
3. La división binaria recursiva restringe las regiones a particiones rectangulares alineadas con los ejes.
4. La búsqueda de divisiones es voraz; no garantiza la partición global óptima.
5. La poda parte de un árbol suficientemente grande y elige su complejidad con validación.
6. La reducción de varianza por promedio depende de que los árboles no estén perfectamente correlacionados.
7. En *bagging*, cada árbol usa una muestra *bootstrap*; en promedio, alrededor de dos tercios de las observaciones aparecen en un árbol y el tercio restante es OOB.
8. La estimación OOB es válida porque cada fila se predice solo con árboles que no fueron ajustados con ella; necesita \(B\) suficientemente grande.
9. En random forest se toma un subconjunto nuevo de predictores en cada división, no uno único por árbol.
10. *Boosting* es secuencial y cada árbol depende de los anteriores; la descripción detallada del capítulo se centra en regresión.
11. BART requiere descartar el *burn-in* y limitar el tamaño de los árboles para reducir sobreajuste.
12. Aunque el método conceptual puede dividir directamente niveles de variables cualitativas, la implementación de `sklearn` mostrada usa codificación *one-hot*.

## 9. Errores y riesgos

- **Evaluar con entrenamiento:** produce una visión optimista; el laboratorio insiste en separar entrenamiento y prueba.
- **Elegir poda con prueba:** filtra información de evaluación. \(\alpha\) se selecciona por validación cruzada dentro del entrenamiento.
- **Preprocesar antes de dividir:** cualquier codificación o transformación aprendida con todos los datos puede introducir *leakage*. Esta precaución es una síntesis práctica coherente con el esquema de validación del capítulo.
- **Detener demasiado pronto el crecimiento:** una división débil puede habilitar otra posterior muy útil. El capítulo prefiere crecer un árbol grande y podarlo.
- **Dejar un árbol grande sin regularización:** reduce error de entrenamiento, pero puede aumentar varianza y error de prueba.
- **Usar error de clasificación para crecer:** puede ignorar mejoras de pureza; Gini o entropía son más sensibles.
- **Interpretar dos hojas con la misma clase como división inútil:** pueden tener proporciones de clase muy distintas y, por tanto, diferente certeza.
- **Suponer estabilidad del árbol:** pequeños cambios en los datos pueden cambiar sustancialmente reglas y estructura.
- **Suponer que más árboles sobreajustan bagging o random forest:** el capítulo indica que aumentar \(B\) no produce sobreajuste, aunque pocos árboles sí pueden infraajustar o dejar una estimación inestable.
- **Suponer que random forest siempre supera a bagging:** depende del problema y de \(m\); el ejemplo de Boston obtiene peor MSE con el valor ensayado de random forest.
- **Usar pocos árboles:** el error puede no haberse estabilizado; seguir la curva OOB o de validación.
- **Usar demasiadas iteraciones de boosting sin control:** a diferencia de *bagging* y random forest, *boosting* puede sobreajustar.
- **Combinar \(\lambda\) muy pequeño con pocos árboles:** el aprendizaje puede quedar incompleto.
- **Tratar importancia por impureza como efecto causal:** la medida solo acumula disminuciones de RSS o Gini debidas a divisiones; esta limitación interpretativa se deriva de su definición.
- **Comparar modelos con particiones distintas:** la varianza de la selección puede ocultar diferencias; el propio laboratorio muestra que podar no siempre mejora una prueba concreta.
- **Coste computacional:** validación cruzada de poda repite crecimiento y poda; bosques entrenan cientos o miles de árboles; *boosting* es secuencial; BART requiere muchas iteraciones y *burn-in*.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo:** problemas tabulares de clasificación o regresión con no linealidades, interacciones, variables mixtas o cuando un modelo lineal deja estructura sin capturar.
- **Decisiones que permite tomar:** árbol interpretable frente a ensamble; Gini frente a entropía; poda mediante `ccp_alpha`; *bagging* frente a random forest; número de árboles; `max_features`; `learning_rate` y profundidad en *boosting*.
- **Baseline útil:** un árbol poco profundo ofrece una regla visible; un random forest con semilla fija suele ser un baseline no lineal más robusto. Esta priorización es derivada: el capítulo no prescribe un único baseline universal.
- **Mejoras de modelo:** validar poda, aumentar árboles hasta estabilización, usar OOB, comparar \(m=p\) con \(m<p\), y afinar conjuntamente \(B\), \(\lambda\) y profundidad de *boosting*.
- **Bajo presión de tiempo:** no implementar manualmente división recursiva, poda ni BART; tampoco explorar una cuadrícula enorme. Usar implementaciones probadas, pocos candidatos informados y una validación limpia.
- **Señales que activan estas técnicas:** degradación marcada de un árbol entre entrenamiento y validación; variables correlacionadas o una variable dominante; frontera no lineal; interacciones plausibles; necesidad de reglas explicables; curva OOB aún descendente; error de *boosting* que empieza a subir mientras el de entrenamiento continúa bajando.
- **Métricas:** MSE para regresión y exactitud/error o matriz de confusión para clasificación, tal como ejemplifica el capítulo. Si el reto impone otra métrica, la selección debe alinearse con ella; esto último es una recomendación de competencia, no una afirmación del capítulo.

## 11. Checklist operativo

- [ ] Identificar si la respuesta es cuantitativa o cualitativa.
- [ ] Reservar prueba antes de seleccionar complejidad o hiperparámetros.
- [ ] Ajustar transformaciones únicamente con entrenamiento y dentro de cada pliegue.
- [ ] Entrenar un árbol base y comparar error de entrenamiento y validación.
- [ ] Para clasificación, usar Gini o entropía al crecer el árbol.
- [ ] Obtener la ruta de poda y elegir `ccp_alpha` con validación cruzada.
- [ ] Comparar el árbol podado con *bagging* (`max_features=p`).
- [ ] Probar random forest con \(m<p\), especialmente con predictores correlacionados.
- [ ] Aumentar el número de árboles hasta que OOB o validación se estabilice.
- [ ] En *boosting*, validar conjuntamente árboles, tasa de aprendizaje y profundidad.
- [ ] Revisar importancia de variables como resumen del modelo, no como causalidad.
- [ ] Evaluar una sola vez en prueba y conservar la semilla usada.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| División binaria recursiva, predicción por hojas y criterios de parada | `knowledge/models/decision_trees.md` | Alta |
| Gini, entropía y error de clasificación en árboles | `knowledge/models/decision_trees.md` | Alta |
| Poda por complejidad de coste y selección de `ccp_alpha` | `knowledge/tuning/tree_based_models.md` | Alta |
| Flujo de validación anidado entrenamiento/CV/prueba | `knowledge/validation/oob_and_cross_validation.md` | Alta |
| Bagging y voto/promedio de árboles | `knowledge/ensembles/bagging.md` | Alta |
| Random forests y descorrelación mediante `max_features` | `knowledge/ensembles/random_forests.md` | Alta |
| Error OOB y su relación con LOOCV | `knowledge/validation/oob_and_cross_validation.md` | Alta |
| Boosting secuencial, *shrinkage* y profundidad de interacción | `knowledge/ensembles/gradient_boosting.md` | Alta |
| BART, residuos parciales, *burn-in* e incertidumbre | `knowledge/models/bart.md` | Media |
| Importancia por disminución de RSS o Gini | `knowledge/interpretability/tree_feature_importance.md` | Media |
| Comparación árbol frente a estructura lineal | `knowledge/models/model_selection_by_structure.md` | Media |
| MSE, exactitud y matriz de confusión en evaluación | `knowledge/metrics/regression_and_classification_metrics.md` | Media |

## 13. Trazabilidad

La página PDF 1 corresponde a la página impresa 331. Se revisaron las páginas PDF 1–36, equivalentes a las páginas impresas 331–366. La página PDF 37, impresa 367, inicia el capítulo 9 y fue excluida.

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura del capítulo, p. 331 |
| 2. Árboles de decisión, nodos y regiones | §8.1.1, pp. 331–335; Figuras 8.1–8.3 |
| 2. Complejidad y poda | §8.1.1, pp. 335–337; Algoritmo 8.1; Figuras 8.4–8.5 |
| 2. Criterios de clasificación | §8.1.2, pp. 337–341; ecuaciones 8.5–8.7; Figura 8.6 |
| 2. Árboles frente a modelos lineales | §8.1.3, pp. 341–342; ecuaciones 8.8–8.9; Figura 8.7 |
| 2. Ventajas, desventajas y estabilidad | §8.1.4, p. 342 |
| 2–5. Bagging y error OOB | §8.2.1, pp. 343–346; Figuras 8.8–8.9 |
| 2–5. Random forests | §8.2.2, pp. 346–348; Figura 8.10 |
| 2–6. Boosting | §8.2.3, pp. 347–350; Algoritmo 8.2; ecuaciones 8.10–8.12; Figura 8.11 |
| 2–6. BART | §8.2.4, pp. 350–353; Algoritmo 8.3; Figuras 8.12–8.13 |
| 4. Comparación de ensambles | §8.2.5, pp. 353–354 |
| 7. Árbol de clasificación y poda en Python | §8.3.1, pp. 354–358 |
| 7. Árbol de regresión en Python | §8.3.2, pp. 358–360 |
| 7. Bagging y random forest en Python | §8.3.3, pp. 360–361 |
| 7. Boosting en Python | §8.3.4, pp. 361–362 |
| 7. BART en Python | §8.3.5, pp. 362–363 |
| 8–11. Supuestos, riesgos y aplicación operativa | Síntesis derivada de §8.1–§8.3, pp. 331–363 |
| Ejercicios consultados solo para confirmar alcance | §8.4, pp. 363–366 |

No se detectaron páginas ilegibles ni secciones del capítulo que no pudieran procesarse.
