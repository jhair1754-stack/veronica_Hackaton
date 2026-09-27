---
title: "Supervised Learning"
topics:
  - "clasificación y regresión supervisadas"
  - "generalización, overfitting y underfitting"
  - "modelos supervisados y control de complejidad"
  - "incertidumbre de clasificadores"
target_knowledge_files:
  - "knowledge/problem_definition/supervised_learning.md"
  - "knowledge/models/supervised_models.md"
  - "knowledge/tuning/model_complexity.md"
  - "knowledge/preprocessing/feature_scaling.md"
  - "knowledge/interpretability/prediction_uncertainty.md"
---

# 2. Supervised Learning

## 1. Propósito del capítulo

El capítulo explica cómo plantear problemas supervisados de clasificación y regresión, cómo evaluar si un modelo generaliza y cómo controlar el equilibrio entre underfitting y overfitting. Presenta los principales algoritmos supervisados de scikit-learn, sus hiperparámetros, requisitos de preprocesamiento, ventajas y limitaciones, y cierra con la interpretación de `decision_function` y `predict_proba`.

## 2. Conceptos esenciales

### Contenido explicado explícitamente por los autores

- En aprendizaje supervisado se aprende de pares entrada/salida del conjunto de entrenamiento para predecir resultados de datos nuevos con características comparables.
- **Clasificación** predice una etiqueta de una lista discreta. Puede ser binaria o multiclase. En clasificación binaria, “positiva” designa la clase de interés, no una valoración favorable.
- **Regresión** predice un valor continuo. La continuidad entre resultados posibles distingue regresión de clasificación.
- **Generalización** es el rendimiento sobre datos no vistos. La evaluación en test es la medida disponible para estimarla.
- **Overfitting**: el modelo se ajusta a particularidades del entrenamiento, obtiene rendimiento alto allí y falla al generalizar. **Underfitting**: el modelo es demasiado simple incluso para capturar la variación del entrenamiento.
- La complejidad apropiada depende de la variedad informativa de los datos. Más observaciones diversas permiten modelos más complejos; duplicar puntos no aporta esa variedad.
- En scikit-learn, `fit` aprende el modelo, `predict` devuelve predicciones y `score` aplica la métrica por defecto del estimador. Los atributos aprendidos terminan en `_`, por ejemplo `coef_`, `intercept_` y `classes_`.
- Volver a llamar a `fit` reinicia lo aprendido por el estimador.

### Conclusiones prácticas derivadas del contenido

- Comparar rendimiento de entrenamiento y validación permite diagnosticar el régimen: ambos bajos sugieren underfitting; entrenamiento muy superior a validación sugiere overfitting.
- La representación de variables es parte de la elección del modelo: k-NN, SVM y MLP son sensibles a escala; los árboles no lo son.
- La complejidad no se compara por un único hiperparámetro universal: cada familia posee controles distintos (`n_neighbors`, `alpha`, `C`, profundidad, número de árboles, arquitectura).

## 3. Flujo o procedimiento presentado

1. Determinar si la salida es discreta (clasificación) o continua (regresión).
2. Separar datos de entrenamiento y test; estratificar en clasificación cuando sea pertinente.
3. Elegir un baseline acorde con tamaño y representación: vecinos, modelo lineal o naive Bayes.
4. Ajustar el estimador solo con entrenamiento mediante `fit`.
5. Evaluar entrenamiento y test con la misma métrica para detectar underfitting u overfitting.
6. Ajustar el hiperparámetro de complejidad específico del modelo.
7. Si el algoritmo es sensible a escala, estimar la transformación solo en entrenamiento y reutilizarla en test.
8. Inspeccionar coeficientes, importancias, estructura del árbol o vectores soporte cuando sean interpretables.
9. En clasificación, revisar `decision_function` o `predict_proba` cuando importe la confianza, respetando el orden de `classes_`.
10. Escalar a modelos más complejos solo si el baseline y el análisis del error lo justifican.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Dataset pequeño y baseline explicable | k-nearest neighbors | Simple, comprensible y suele rendir razonablemente con poco ajuste | Escalar variables; la predicción se vuelve lenta con muchas filas o variables y funciona mal con alta dimensionalidad dispersa |
| Muchas variables, datos dispersos o dataset muy grande | Modelo lineal | Entrenamiento y predicción rápidos; escala bien y funciona con datos sparse | Regularizar y no interpretar coeficientes de forma causal, especialmente con variables correlacionadas |
| Conteos dispersos, especialmente texto | `MultinomialNB` | Muy rápido y adecuado para conteos | Puede generalizar algo peor que clasificadores lineales |
| Variables binarias dispersas | `BernoulliNB` | Modela presencia/ausencia por clase | El supuesto binario debe cumplirse |
| Variables continuas y baseline extremadamente rápido | `GaussianNB` | Mantiene media y desviación por clase; sirve en alta dimensión | Sus supuestos simples pueden limitar precisión |
| Se necesita explicación mediante reglas y escalas mezcladas | Árbol de decisión podado | No necesita escalado y puede visualizarse | Un árbol sin restricciones tiende a memorizar; no extrapola en regresión |
| Tabular general con poco preprocesamiento | Random forest | Robusto, reduce overfitting del árbol individual y suele rendir mejor que este | No es ideal para datos dispersos de muy alta dimensión; puede consumir memoria y perder interpretabilidad |
| Tabular donde se busca algo más de precisión | Gradient boosting | Conjunto de árboles débiles que suele superar ligeramente al bosque | Requiere ajuste cuidadoso; demasiados árboles pueden sobreajustar |
| Dataset mediano, frontera no lineal y variables comparables | SVM con kernel RBF | Produce fronteras complejas mediante el kernel trick | Escalar; ajustar `C` y `gamma` juntos; tiempo y memoria empeoran con muchas muestras |
| Mucha información, cómputo disponible y relación compleja | MLP | Aprende funciones muy complejas | Estandarizar; ajustar arquitectura, regularización e iteraciones; entrenamiento e interpretación difíciles |
| Forecasting que exige extrapolar fuera del rango temporal observado | Modelo capaz de extrapolar, como regresión lineal si la relación transformada es apropiada | Los árboles solo repiten valores/respuestas observados en hojas | Validar temporalmente; el ejemplo no afirma que todos los árboles sean inútiles en series temporales |

## 5. Métodos y técnicas

### k-nearest neighbors (k-NN)

- **Objetivo:** clasificación por voto mayoritario y regresión por promedio de los vecinos más cercanos.
- **Entradas:** matriz de variables, objetivo y una métrica de distancia; por defecto se usa distancia euclídea.
- **Salida:** clase más frecuente o media de objetivos vecinos.
- **Funcionamiento:** `fit` almacena el conjunto; el cálculo costoso ocurre al predecir.
- **Hiperparámetros:** `n_neighbors` y métrica de distancia. Menor `k` implica mayor complejidad; mayor `k`, frontera o curva más suave.
- **Ventajas:** baseline simple, explicable y rápido de entrenar.
- **Limitaciones:** predicción lenta en datasets grandes; sensible a escala; débil con cientos de variables o datos sparse.
- **Adecuado:** datasets pequeños, densos y de dimensionalidad moderada.
- **No conviene:** muchas muestras, muchas variables o mayoría de entradas cero.

### Regresión lineal, Ridge y Lasso

- **Objetivo:** predecir una combinación lineal de variables.
- **Entradas:** variables numéricas y objetivo continuo.
- **Salida:** valor continuo; los pesos se exponen en `coef_` y el intercepto en `intercept_`.
- **Funcionamiento:** OLS minimiza el error cuadrático de entrenamiento. Ridge añade penalización L2; Lasso añade penalización L1 y puede llevar coeficientes exactamente a cero.
- **Hiperparámetros:** OLS no controla complejidad; en Ridge/Lasso, `alpha` grande refuerza regularización y simplifica el modelo.
- **Ventajas:** rápidos, escalables y fuertes cuando hay muchas variables; Ridge es estable y Lasso puede seleccionar variables.
- **Limitaciones:** la relación lineal puede underfit; OLS puede overfit en alta dimensión; Lasso puede eliminar demasiadas variables y requerir más iteraciones.
- **Adecuado:** Ridge como opción general con muchas variables; Lasso si se espera que pocas sean relevantes o se busca parsimonia.
- **No conviene:** relaciones fuertemente no lineales sin una representación apropiada.

### Clasificación lineal: LogisticRegression y LinearSVC

- **Objetivo:** separar clases mediante una función lineal y asignar según el signo o mayor puntuación.
- **Entradas:** variables y etiquetas binarias o multiclase.
- **Salida:** etiqueta; coeficientes e intercepto por clase.
- **Funcionamiento:** en multiclase puede emplearse one-vs.-rest: un clasificador por clase y gana la puntuación mayor.
- **Hiperparámetros:** `C` es inverso a la fuerza de regularización: `C` pequeño produce modelo más simple; `C` grande ajusta más el entrenamiento. L1 selecciona variables; L2 usa el conjunto de variables de forma más distribuida.
- **Ventajas:** entrenamiento y predicción rápidos; funcionan con datos sparse y alta dimensión.
- **Limitaciones:** fronteras lineales en la representación usada; coeficientes inestables o difíciles de explicar con variables correlacionadas.
- **Adecuado:** primer modelo para datos grandes o de alta dimensionalidad.
- **No conviene:** frontera no lineal que no pueda capturarse mediante ingeniería de variables.

### Naive Bayes

- **Objetivo:** clasificación eficiente mediante estadísticas simples por variable y clase.
- **Entradas:** continuas (`GaussianNB`), binarias (`BernoulliNB`) o conteos no negativos (`MultinomialNB`).
- **Salida:** clase con mejor concordancia con las estadísticas aprendidas.
- **Funcionamiento:** Bernoulli cuenta no ceros; Multinomial considera valores medios por clase; Gaussian almacena media y desviación por variable y clase.
- **Hiperparámetros:** `alpha` en Bernoulli/Multinomial suaviza conteos; valores mayores generan modelos más simples.
- **Ventajas:** entrenamiento/predicción extremadamente rápidos, robustos al ajuste y fuertes con alta dimensionalidad sparse.
- **Limitaciones:** a menudo algo menos precisos que modelos lineales.
- **Adecuado:** baseline de texto o datasets enormes.
- **No conviene:** regresión; representaciones incompatibles con la variante elegida.

### Árboles de decisión

- **Objetivo:** clasificación o regresión mediante una jerarquía de pruebas `variable <= umbral`.
- **Entradas:** variables binarias o continuas, incluso con escalas heterogéneas.
- **Salida:** clase mayoritaria o media del objetivo en la hoja alcanzada.
- **Funcionamiento:** particiona recursivamente con la prueba más informativa hasta detenerse o producir hojas puras.
- **Hiperparámetros:** `max_depth`, `max_leaf_nodes`, `min_samples_leaf` y otros controles de pre-pruning.
- **Ventajas:** no requiere normalización, admite tipos de variables mezclados y es interpretable si es pequeño.
- **Limitaciones:** árboles profundos overfit; fronteras paralelas a ejes; la importancia cero no implica que la variable carezca de información; regresión no extrapola fuera del rango visto.
- **Adecuado:** reglas explicables y datos tabulares con escalas dispares.
- **No conviene:** extrapolación o cuando un árbol individual no generaliza.

### Random forests

- **Objetivo:** reducir la varianza de un árbol promediando árboles diversos.
- **Entradas:** datos tabulares para clasificación o regresión.
- **Salida:** promedio en regresión; en clasificación, promedio de probabilidades y clase de mayor probabilidad.
- **Funcionamiento:** cada árbol usa una muestra bootstrap y, en cada división, un subconjunto aleatorio de variables (`max_features`).
- **Hiperparámetros:** `n_estimators`, `max_features`, parámetros de poda y `n_jobs`.
- **Ventajas:** robusto, potente, poco sensible a escalado y paralelizable; importancias suelen ser más estables que las de un árbol.
- **Limitaciones:** más lento y menos interpretable que un árbol; muchos árboles requieren memoria; débil en datos sparse de altísima dimensión; conserva la incapacidad de extrapolar de los árboles.
- **Adecuado:** excelente opción general para tabular.
- **No conviene:** cuando importan modelo compacto, explicación exacta o extrapolación.

### Gradient boosted decision trees

- **Objetivo:** combinar secuencialmente árboles poco profundos que corrigen errores previos.
- **Entradas:** datos tabulares de clasificación o regresión.
- **Salida:** predicción agregada del conjunto.
- **Funcionamiento:** agrega weak learners de forma iterativa.
- **Hiperparámetros:** `n_estimators`, `learning_rate`, `max_depth` o `max_leaf_nodes`. Menor `learning_rate` exige más árboles; aumentar `n_estimators` eleva complejidad y puede overfit.
- **Ventajas:** entre los modelos supervisados más potentes; suele ser algo más preciso, más compacto y más rápido al predecir que random forest.
- **Limitaciones:** tuning cuidadoso y entrenamiento potencialmente lento; no destaca en alta dimensión sparse.
- **Adecuado:** tabular competitivo cuando hay presupuesto para validar hiperparámetros.
- **No conviene:** baseline bajo presión extrema o datos dispersos masivos.

### SVM con kernels

- **Objetivo:** construir fronteras no lineales sin materializar explícitamente una expansión enorme de variables.
- **Entradas:** variables numéricas comparables y etiquetas; el capítulo se concentra en `SVC`.
- **Salida:** clase y puntuación; los puntos relevantes son vectores soporte.
- **Funcionamiento:** el kernel calcula productos escalares en una representación ampliada. RBF usa similitud gaussiana; también existe kernel polinómico.
- **Hiperparámetros:** `C`, `gamma` y kernel. `gamma` grande estrecha el radio de influencia y aumenta complejidad; `C` grande permite que puntos individuales influyan más.
- **Ventajas:** fronteras suaves y complejas en baja o alta dimensión.
- **Limitaciones:** sensible a escalado y tuning; difícil de explicar; alrededor de 100 000 muestras puede ser costoso en tiempo/memoria.
- **Adecuado:** datasets medianos con variables homogéneas.
- **No conviene:** datasets muy grandes o variables heterogéneas sin preprocesamiento.

### Multilayer perceptrons (MLP)

- **Objetivo:** clasificación o regresión mediante capas de combinaciones ponderadas y activaciones no lineales.
- **Entradas:** variables numéricas, idealmente con media 0 y varianza 1.
- **Salida:** clase o valor continuo.
- **Funcionamiento:** pesos entre entrada, capas ocultas y salida; `relu` trunca negativos y `tanh` satura entre -1 y 1.
- **Hiperparámetros:** `hidden_layer_sizes`, `activation`, `alpha`, solver, `max_iter` y semilla. Más capas/unidades aumentan número de pesos y complejidad.
- **Ventajas:** captura relaciones muy complejas y aprovecha grandes cantidades de datos.
- **Limitaciones:** lento, sensible a escalado/inicialización/tuning y difícil de interpretar; una advertencia de convergencia exige revisar iteraciones u optimización.
- **Adecuado:** datos abundantes y relativamente homogéneos con presupuesto de cómputo.
- **No conviene:** baseline tabular rápido o necesidad fuerte de explicación.

### Estimaciones de incertidumbre

- **`decision_function`:** en binario devuelve `(n_samples,)`; valores positivos favorecen `classes_[1]` y negativos `classes_[0]`. Su escala es arbitraria. En multiclase devuelve `(n_samples, n_classes)` y se elige el máximo.
- **`predict_proba`:** devuelve una columna por clase, valores entre 0 y 1 y suma por fila igual a 1; la clase con probabilidad máxima es la predicción.
- **Calibración:** un modelo está calibrado si, por ejemplo, predicciones con confianza 70 % son correctas aproximadamente 70 % de las veces. Un modelo overfit puede mostrar confianza excesiva.

## 6. Fórmulas esenciales

### Predicción lineal

\[
\hat{y} = \sum_{j=0}^{p} w_j x_j + b
\]

- \(x_j\): valor de la variable \(j\); \(w_j\): coeficiente aprendido; \(b\): intercepto; \(\hat y\): predicción.
- En regresión produce un valor continuo. En clasificación lineal binaria, el signo de la puntuación separa las clases; en multiclase se escoge la puntuación máxima.
- Supuesto práctico: la relación es lineal en la representación de variables disponible, no necesariamente en las variables originales si se expandieron.

### Error cuadrático medio usado por OLS

\[
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

- \(n\): observaciones; \(y_i\): valor real; \(\hat y_i\): predicción.
- Mide el promedio de error al cuadrado y penaliza con mayor fuerza errores grandes. OLS aprende \(w,b\) minimizándolo en entrenamiento.

### Coeficiente de determinación

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]

- \(\bar y\): media del objetivo de referencia; el resto de símbolos conserva la definición anterior.
- \(R^2=1\) indica predicción perfecta y \(R^2=0\) equivale al modelo constante que predice la media. El capítulo usa `score` para obtenerlo en regresores.

### Kernel RBF

\[
k_{\mathrm{RBF}}(x_1,x_2)=\exp\left(-\gamma\lVert x_1-x_2\rVert^2\right)
\]

- \(x_1,x_2\): observaciones; \(\lVert x_1-x_2\rVert\): distancia euclídea; \(\gamma\): inverso del ancho de influencia.
- Mide similitud: puntos cercanos producen valores mayores. Un \(\gamma\) alto hace el modelo más local y complejo; uno bajo suaviza la frontera.
- Nota de trazabilidad: la extracción del PDF omite visualmente el signo menos, pero la forma gaussiana descrita y la figura corresponden a la expresión anterior.

### Activaciones de MLP

\[
\operatorname{ReLU}(z)=\max(0,z), \qquad \tanh(z)\in[-1,1]
\]

- \(z\): combinación ponderada que entra a una unidad oculta.
- Introducen no linealidad; sin activación, varias sumas ponderadas equivaldrían a una sola transformación lineal.

## 7. Implementación práctica

Los siguientes fragmentos conservan la idea del capítulo y usan API actual. **Nota práctica - adaptación moderna:** se usan `Pipeline`, `StandardScaler`/`MinMaxScaler`, `solver` en MLP y opciones actuales de `LogisticRegression`, evitando el código manual de escalado y el parámetro antiguo `algorithm`.

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,  # retirar en regresión
    random_state=42,
)
```

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, penalty="l2", max_iter=2000, random_state=42),
)
model.fit(X_train, y_train)
valid_accuracy = model.score(X_valid, y_valid)
```

El pipeline aprende media y desviación solo con `X_train`, lo que evita leakage al transformar validación.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)

boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42,
).fit(X_train, y_train)
```

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

svc = make_pipeline(
    MinMaxScaler(),
    SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
)
svc.fit(X_train, y_train)
class_probabilities = svc.predict_proba(X_valid)
```

`probability=True` permite `predict_proba`, pero añade coste. Si solo se requieren rankings o márgenes, usar `decision_function`.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=1.0,
        solver="adam",
        max_iter=1000,
        random_state=42,
    ),
)
mlp.fit(X_train, y_train)
```

## 8. Supuestos y condiciones de aplicación

1. Entrenamiento y evaluación deben representar el mismo proceso generador; de lo contrario, la generalización estimada no es fiable.
2. El objetivo debe estar correctamente tipado: clases discretas para clasificación y continuidad real para regresión.
3. k-NN presupone que la distancia entre observaciones es significativa; exige escalas comparables.
4. Modelos lineales suponen linealidad en la representación de entrada; expansiones no lineales pueden cambiar esa capacidad.
5. `BernoulliNB` requiere variables binarias; `MultinomialNB`, conteos; `GaussianNB`, variables continuas resumibles por media y desviación por clase.
6. Árboles no necesitan escalado, pero sus regresores solo predicen combinaciones derivadas de respuestas observadas y no extrapolan.
7. Random forest y gradient boosting admiten escalas y tipos numéricos mezclados, pero no son la opción preferida para alta dimensión sparse.
8. SVM RBF necesita escalado y tuning conjunto de `C` y `gamma`; es más viable en tamaño mediano.
9. MLP necesita variables de magnitud comparable, preferentemente estandarizadas, y suficiente dato/cómputo para el número de pesos.
10. Las probabilidades solo deben interpretarse literalmente como confianza si el modelo está razonablemente calibrado.

## 9. Errores y riesgos

- Medir éxito por ajuste del entrenamiento, incluso 100 %, en vez de por datos no vistos.
- Ajustar escaladores con todo el dataset. El capítulo recalca calcular mínimo/rango o media/desviación en entrenamiento y reutilizarlos en test.
- Usar k-NN o SVM sin escalar variables, haciendo que las de mayor magnitud dominen distancias.
- Confundir `alpha` y `C`: en Ridge/Lasso/MLP, mayor `alpha` regulariza más; en clasificadores lineales y SVM, mayor `C` regulariza menos.
- Dejar un árbol crecer hasta hojas puras y confundir memorización con generalización.
- Interpretar `feature_importances_ == 0` como ausencia de señal; otra variable correlacionada pudo ser elegida.
- Inferir dirección del efecto desde la importancia de un árbol: es no negativa y no indica qué clase favorecen valores altos.
- Sobreinterpretar el signo de un coeficiente lineal; puede cambiar con regularización o correlación entre variables.
- Usar regresores de árboles para extrapolar tendencias más allá del rango de entrenamiento.
- Aumentar `n_estimators` en gradient boosting como si siempre ayudara; a diferencia del bosque, eleva complejidad y puede overfit.
- Tratar las puntuaciones de `decision_function` como probabilidades; su rango es arbitrario.
- Mapear `argmax` directamente a etiquetas sin pasar por `classes_`, sobre todo con strings o enteros no consecutivos.
- Ignorar avisos de convergencia del MLP o aumentar iteraciones suponiendo que mejorará validación; puede mejorar solo entrenamiento.
- Omitir `random_state` en modelos y particiones estocásticas, dificultando reproducibilidad.
- Evaluar forecasting con una partición aleatoria; conclusión práctica derivada: el ejemplo temporal separa pasado y futuro, por lo que debe conservarse el orden.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo al decidir entre un baseline lineal, vecinos, naive Bayes, árboles/ensembles, SVM o MLP, y al diagnosticar underfitting/overfitting.
- Para un baseline: modelo lineal en alta dimensión; naive Bayes para conteos sparse; k-NN para datos pequeños; árbol podado si la explicación importa; random forest para tabular general.
- Para mejorar: ajustar regularización en escala logarítmica, escalar dentro de pipeline, controlar profundidad y combinar árboles; en boosting, coordinar `learning_rate` y `n_estimators`.
- Bajo presión de tiempo, evitar MLP grande, búsqueda extensa de SVM RBF o visualización completa de árboles profundos. Empezar simple y conservar un conjunto de validación fiable.
- Señales activadoras: muchas columnas frente a filas -> regularización lineal; conteos sparse -> naive Bayes; escalas heterogéneas -> árbol o pipeline con scaler; frontera no lineal y dataset mediano -> RBF SVM; tabular con interacciones -> ensembles; necesidad de extrapolar -> evitar regresores arbóreos.
- Si la métrica depende de ranking o umbral, obtener `decision_function`/`predict_proba` y verificar `classes_`; no confundir confianza extrema con calibración.

## 11. Checklist operativo

- [ ] Determinar clasificación o regresión a partir de la naturaleza del objetivo.
- [ ] Separar entrenamiento y validación con `random_state`; estratificar clasificación.
- [ ] Elegir un baseline simple compatible con tamaño y representación.
- [ ] Ajustar transformaciones exclusivamente con entrenamiento.
- [ ] Comparar puntuación de entrenamiento y validación.
- [ ] Identificar el hiperparámetro que controla complejidad en la familia elegida.
- [ ] Buscar `alpha`, `C` o `gamma` en escala logarítmica cuando corresponda.
- [ ] Limitar profundidad/hojas de árboles individuales.
- [ ] Coordinar `learning_rate` y `n_estimators` en boosting.
- [ ] Revisar convergencia y sensibilidad a semilla en MLP.
- [ ] Usar `classes_` al interpretar scores o probabilidades.
- [ ] Evitar árboles de regresión cuando sea necesaria extrapolación.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Clasificación frente a regresión y definición del objetivo | `knowledge/problem_definition/supervised_learning.md` | Alta |
| Generalización, underfitting, overfitting y complejidad | `knowledge/validation/generalization_diagnostics.md` | Alta |
| k-NN para clasificación y regresión | `knowledge/models/knn.md` | Media |
| OLS, Ridge, Lasso y clasificadores lineales | `knowledge/models/linear_models.md` | Alta |
| Naive Bayes por tipo de representación | `knowledge/models/naive_bayes.md` | Media |
| Árboles, random forest y gradient boosting | `knowledge/models/tree_ensembles.md` | Alta |
| Relación entre `learning_rate`, árboles y profundidad | `knowledge/tuning/gradient_boosting.md` | Alta |
| Escalado requerido por k-NN, SVM y MLP | `knowledge/preprocessing/feature_scaling.md` | Alta |
| SVM RBF, `C`, `gamma` y kernels | `knowledge/models/kernel_svm.md` | Media |
| Arquitectura, regularización y convergencia de MLP | `knowledge/models/neural_networks.md` | Media |
| `decision_function`, `predict_proba`, `classes_` y calibración | `knowledge/interpretability/prediction_uncertainty.md` | Alta |
| Límite de extrapolación de modelos arbóreos | `knowledge/time_series/extrapolation_limits.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito; clasificación y regresión | “Classification and Regression”, pp. 25–26 |
| Generalización, overfitting, underfitting y tamaño de datos | “Generalization, Overfitting, and Underfitting” y “Relation of Model Complexity to Dataset Size”, pp. 26–29 |
| Flujo general y datasets de ejemplo | “Supervised Machine Learning Algorithms” y “Some Sample Datasets”, pp. 29–34 |
| k-nearest neighbors | “k-Nearest Neighbors”, pp. 35–44 |
| Regresión lineal, Ridge y Lasso | “Linear Models” / “Linear models for regression”, pp. 45–55 |
| Clasificación lineal y multiclase | “Linear models for classification” y “Linear models for multiclass classification”, pp. 56–68 |
| Method chaining y reinicio por `fit` | “Method Chaining”, p. 68; recuadro “fit Resets a Model”, p. 119 |
| Naive Bayes | “Naive Bayes Classifiers”, pp. 68–70 |
| Árboles de decisión e importancias | “Decision Trees”, pp. 70–83 |
| Random forests | “Ensembles of Decision Trees” / “Random forests”, pp. 83–88 |
| Gradient boosting | “Gradient boosted regression trees”, pp. 88–92 |
| SVM con kernels y escalado | “Kernelized Support Vector Machines”, pp. 92–104 |
| MLP y redes neuronales | “Neural Networks (Deep Learning)”, pp. 104–119 |
| Incertidumbre binaria y multiclase | “Uncertainty Estimates from Classifiers”, pp. 119–127 |
| Criterios finales de elección | “Summary and Outlook”, pp. 127–129 |

### Estado de lectura

- Páginas impresas revisadas: 25–129 (páginas PDF 1–105).
- Página PDF 106: en blanco, sin contenido del capítulo.
- Secciones ilegibles o no procesables: ninguna.
