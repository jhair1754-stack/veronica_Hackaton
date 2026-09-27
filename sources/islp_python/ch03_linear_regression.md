---
title: "Linear Regression"
topics:
  - "simple and multiple linear regression"
  - "least squares, inference, and prediction intervals"
  - "categorical predictors, interactions, and polynomial terms"
  - "regression diagnostics and collinearity"
  - "K-nearest neighbors regression"
  - "Python implementation with statsmodels"
target_knowledge_files:
  - "knowledge/models/linear_regression.md"
  - "knowledge/models/knn_regression.md"
  - "knowledge/metrics/regression_fit_metrics.md"
  - "knowledge/feature_engineering/categorical_interactions_polynomials.md"
  - "knowledge/validation/regression_diagnostics.md"
  - "knowledge/interpretability/linear_model_inference.md"
---

# 3. Linear Regression

## 1. Propósito del capítulo

El capítulo desarrolla la regresión lineal como método supervisado para predecir una respuesta cuantitativa y para estudiar la asociación entre esa respuesta y uno o varios predictores. Explica el ajuste por mínimos cuadrados, la incertidumbre de los coeficientes y de las predicciones, las extensiones mediante variables cualitativas, interacciones y polinomios, y los principales diagnósticos que pueden invalidar una interpretación. También compara el enfoque paramétrico lineal con K-nearest neighbors (KNN) para regresión y muestra una implementación en Python con `statsmodels`.

**Convención de procedencia:** salvo donde se indique **“Conclusión práctica derivada”**, **“Síntesis práctica derivada del capítulo”**, **“Recomendación para competencias”** o **“Nota práctica - adaptación moderna”**, el contenido siguiente parafrasea afirmaciones explícitas del capítulo.

## 2. Conceptos esenciales

### Regresión lineal simple y múltiple

La regresión lineal simple aproxima la respuesta (Y) con una función lineal de un único predictor (X). El intercepto representa el valor esperado de (Y) cuando (X=0), y la pendiente representa el cambio promedio en (Y) asociado a una unidad adicional de (X). La regresión múltiple asigna un coeficiente a cada predictor; cada pendiente se interpreta manteniendo fijos los demás predictores.

Esta interpretación condicional es crucial: una regresión simple puede atribuir a una variable el efecto de otra variable correlacionada. En el ejemplo de publicidad, `newspaper` parece asociado a ventas cuando se analiza solo, pero deja de ser significativo al controlar por `TV` y `radio`.

### Ajuste por mínimos cuadrados

Los coeficientes se eligen para minimizar la suma de cuadrados de los residuos (residual sum of squares, RSS). Un residuo es la diferencia entre la respuesta observada y la predicción ajustada. En regresión múltiple, la solución se calcula convenientemente mediante álgebra matricial.

La recta estimada cambia entre muestras aunque la relación poblacional sea la misma. Bajo los supuestos del modelo, los estimadores de mínimos cuadrados son insesgados; sus errores estándar cuantifican la variabilidad muestral y permiten construir intervalos de confianza y contrastes de hipótesis.

### Inferencia sobre coeficientes y modelo

- Un estadístico (t) contrasta si un coeficiente individual es cero, condicionado a los demás predictores incluidos.
- El estadístico (F) contrasta si todos los coeficientes de pendiente, o un subconjunto de ellos, son simultáneamente cero.
- El test (F) global es preferible a buscar algún p-value individual pequeño cuando hay muchos predictores, porque múltiples contrastes producen falsos positivos por azar.
- Cuando (p>n), mínimos cuadrados no puede estimar de forma única todos los coeficientes y los contrastes presentados dejan de ser aplicables.

Un p-value pequeño aporta evidencia de asociación dentro del modelo; no mide tamaño del efecto, calidad predictiva ni causalidad. La magnitud, unidades, intervalo de confianza y contexto deben acompañarlo.

### Calidad del ajuste y predicción

- **RSE (residual standard error):** estima la desviación típica del error y conserva las unidades de la respuesta. Mide el tamaño típico del desajuste.
- **(R^2):** proporción de variabilidad de (Y) explicada por el modelo. Siempre aumenta o permanece igual al agregar predictores en entrenamiento, por lo que no prueba que mejorará la generalización.
- **Intervalo de confianza:** cuantifica la incertidumbre sobre la respuesta media (f(X)).
- **Intervalo de predicción:** cuantifica la incertidumbre sobre una observación futura (Y=f(X)+\epsilon); es más ancho porque incorpora además el error irreducible.

### Predictores cualitativos

Una variable binaria se representa con una variable indicadora (dummy). Para una variable con (L) niveles se usan (L-1) indicadores si el modelo contiene intercepto; el nivel omitido es la categoría de referencia. Cambiar la referencia modifica la interpretación de los coeficientes, pero no las predicciones. Para evaluar globalmente una variable cualitativa con varios niveles, el capítulo recomienda un test (F) conjunto, que no depende de la codificación elegida.

### Interacciones y no linealidad

El modelo aditivo supone que el efecto de un predictor no depende de los valores de los demás. Un término producto (X_1X_2) permite que la pendiente de (X_1) cambie con (X_2), y viceversa. El principio jerárquico exige conservar ambos efectos principales cuando se incluye su interacción, aunque sus p-values individuales sean grandes.

La linealidad se refiere a los coeficientes, no obliga a usar predictores sin transformar. Agregar (X^2), (X^3) u otras transformaciones permite representar relaciones curvas mediante regresión lineal. Grados innecesariamente altos pueden producir ajustes ondulados y sobreajuste.

### Diagnóstico del modelo

El capítulo destaca seis problemas:

1. no linealidad de la relación respuesta-predictores;
2. correlación entre errores;
3. varianza no constante de los errores (heteroscedasticidad);
4. outliers en la respuesta;
5. observaciones de alto leverage en los predictores;
6. colinealidad o multicolinealidad.

Los residuos frente a valores ajustados deberían carecer de patrones. Curvatura sugiere no linealidad; una forma de embudo sugiere heteroscedasticidad. Residuos consecutivos parecidos en datos ordenados temporalmente sugieren dependencia. Los residuos studentizados ayudan a detectar outliers; el leverage detecta configuraciones de predictores inusuales. El variance inflation factor (VIF) mide cuánto se infla la varianza de un coeficiente por colinealidad.

### Regresión lineal frente a KNN

La regresión lineal es paramétrica: suele ser estable, interpretable y eficaz cuando la forma lineal es razonable o hay pocas observaciones por predictor. KNN es no paramétrico: predice promediando las respuestas de los (K) vecinos más próximos. Un (K) pequeño reduce sesgo pero aumenta varianza; un (K) grande suaviza el ajuste y puede aumentar sesgo.

KNN puede superar a la regresión lineal ante relaciones no lineales en baja dimensión, pero se deteriora rápidamente al crecer el número de predictores: las observaciones quedan dispersas y los “vecinos” dejan de ser cercanos, fenómeno llamado maldición de la dimensionalidad.

## 3. Flujo o procedimiento presentado

1. **Definir respuesta y predictores.** Confirmar que la respuesta sea cuantitativa y que las unidades permitan interpretar coeficientes y error.
2. **Explorar relaciones y tipos.** Visualizar respuesta frente a predictores; identificar variables cualitativas, posibles curvaturas, interacciones y correlaciones entre predictores.
3. **Construir la matriz de diseño.** Incluir intercepto, variables numéricas, (L-1) dummies por variable categórica, y solo transformaciones o interacciones justificadas.
4. **Ajustar por mínimos cuadrados.** Estimar los coeficientes minimizando RSS.
5. **Comprobar evidencia global.** Usar el test (F) para saber si al menos un predictor aporta asociación; no depender de una colección de tests individuales.
6. **Interpretar efectos condicionales.** Revisar estimaciones, errores estándar, intervalos de confianza y p-values, manteniendo fijos los demás predictores.
7. **Evaluar ajuste.** Examinar RSE y (R^2), recordando que son medidas calculadas sobre los datos ajustados y que (R^2) no penaliza añadir variables.
8. **Diagnosticar supuestos.** Revisar residuos frente a ajustados, residuos en orden temporal, residuos studentizados, leverage y VIF.
9. **Corregir problemas detectados.** Considerar transformaciones, términos polinómicos o interacciones; investigar outliers; usar weighted least squares si se conoce la estructura de varianzas; retirar o combinar predictores redundantes ante colinealidad.
10. **Predecir con la incertidumbre adecuada.** Usar intervalo de confianza para la media e intervalo de predicción para una observación individual.
11. **Comparar alternativas.** Si la relación parece no lineal, comparar una extensión interpretable del modelo lineal con KNN, eligiendo (K) según error de prueba estimado.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Respuesta cuantitativa y relación aproximadamente lineal | Regresión lineal por mínimos cuadrados | Ajuste simple, interpretable y con inferencia disponible | Verificar residuos y evitar interpretación causal automática |
| Varios predictores potencialmente correlacionados | Regresión múltiple en un único modelo | Estima el efecto parcial de cada predictor controlando por los demás | Los coeficientes pueden ser inestables bajo colinealidad |
| Se pregunta si existe alguna asociación | Test (F) global | Contrasta conjuntamente todos los coeficientes de pendiente | Requiere que (p) sea suficientemente menor que (n) |
| Se evalúa un predictor individual dentro de un modelo | Test (t), intervalo de confianza y tamaño del coeficiente | Cuantifica evidencia, dirección y precisión del efecto parcial | Un p-value no mide capacidad predictiva ni causalidad |
| Variable cualitativa con (L) niveles | (L-1) dummies y test (F) conjunto | Evita colinealidad con el intercepto y evalúa la variable completa | La referencia cambia coeficientes, no predicciones |
| El efecto de un predictor parece depender de otro | Añadir interacción y mantener ambos efectos principales | Relaja el supuesto aditivo | Aplicar el principio jerárquico |
| Residuos con curvatura | Transformar predictores o añadir términos polinómicos | Permite representar una relación no lineal | Grados altos pueden sobreajustar |
| Residuos con forma de embudo | Transformar (Y) con una función cóncava o usar weighted least squares si se conocen varianzas | Puede estabilizar la varianza | La transformación cambia la escala de interpretación |
| Errores correlacionados en el tiempo o por grupos | Revisar el diseño y usar métodos que modelen dependencia | Los errores estándar clásicos quedan subestimados | No confiar en p-values ni intervalos del ajuste iid |
| VIF alto | Retirar una variable redundante o combinar predictores correlacionados | Reduce la varianza de los coeficientes | Comprobar que no se pierda información sustantiva |
| Relación no lineal y pocos predictores | Comparar KNN con regresión lineal | KNN puede reducir el sesgo de forma flexible | Escalar predictores y validar (K); sufre en alta dimensión |
| Se predice una media poblacional | Intervalo de confianza | Mide incertidumbre sobre (f(X)) | No representa la dispersión de una observación individual |
| Se predice un caso futuro | Intervalo de predicción | Incluye incertidumbre del ajuste y error irreducible | Es necesariamente más ancho |

## 5. Métodos y técnicas

### Regresión lineal simple

- **Objetivo:** predecir una respuesta cuantitativa desde un predictor y cuantificar su asociación lineal.
- **Entradas:** pares ((x_i,y_i)).
- **Salida:** intercepto, pendiente, valores ajustados, residuos e inferencia.
- **Funcionamiento:** minimiza RSS para obtener la recta de mínimos cuadrados.
- **Hiperparámetros:** ninguno en su forma básica.
- **Ventajas:** cálculo directo, interpretación clara y diagnóstico visual sencillo.
- **Limitaciones:** ignora otros predictores y puede confundir asociaciones por variables omitidas.
- **Casos adecuados:** baseline e inferencia con una relación aproximadamente lineal.
- **No conviene:** relación muy curva, errores dependientes o efecto confundido por otras variables.

### Regresión lineal múltiple

- **Objetivo:** predecir (Y) y estimar asociaciones parciales de múltiples predictores.
- **Entradas:** matriz de diseño (X) y respuesta (y).
- **Salida:** coeficientes, predicciones, residuos, RSE, (R^2), tests (t/F) e intervalos.
- **Funcionamiento:** minimiza RSS simultáneamente respecto de todos los coeficientes.
- **Hiperparámetros:** ninguno en mínimos cuadrados ordinarios; la elección de variables y términos define la especificación.
- **Ventajas:** controla por los predictores incluidos y mantiene interpretabilidad.
- **Limitaciones:** supone forma lineal/aditiva salvo extensiones; es sensible a colinealidad, leverage y outliers; no funciona de forma única cuando (p>n).
- **Casos adecuados:** datos tabulares con respuesta continua, inferencia y baseline predictivo.
- **No conviene:** alta dimensión sin selección o regularización, o estructura no lineal no representada.

### Selección clásica de variables

- **Objetivo:** encontrar un subconjunto manejable de predictores.
- **Entradas:** conjunto candidato de (p) variables.
- **Salida:** secuencia de modelos y subconjunto elegido mediante un criterio de parada.
- **Funcionamiento:** forward selection agrega la variable que más reduce RSS; backward selection elimina la de mayor p-value; mixed selection alterna ambos movimientos.
- **Parámetros relevantes:** regla de parada y criterio de comparación, como (C_p), AIC, BIC o (R^2) ajustado.
- **Ventajas:** evita revisar los (2^p) subconjuntos.
- **Limitaciones:** forward es greedy y puede retener temprano variables después redundantes; backward no es posible si (p>n); la selección basada en p-values puede ser inestable.
- **Casos adecuados:** número moderado de predictores y necesidad de simplificación.
- **No conviene:** usar (R^2) de entrenamiento como único criterio o hacer búsqueda exhaustiva con (p) moderado/grande.

### Interacciones

- **Objetivo:** modelar efectos no aditivos.
- **Entradas:** dos predictores y su producto.
- **Salida:** una pendiente condicional que cambia con el otro predictor.
- **Funcionamiento:** incorpora (X_1X_2) junto con (X_1) y (X_2).
- **Hiperparámetros:** selección de pares a interactuar.
- **Ventajas:** conserva un modelo lineal en coeficientes y captura sinergias.
- **Limitaciones:** aumenta complejidad y exige interpretación conjunta.
- **Casos adecuados:** evidencia de que el efecto de una variable depende de otra.
- **No conviene:** incluir productos masivamente sin justificación o sin validar.

### Regresión polinómica

- **Objetivo:** representar curvatura mediante potencias de un predictor.
- **Entradas:** (X,X^2,\ldots,X^d) y (Y).
- **Salida:** función polinómica ajustada.
- **Funcionamiento:** trata cada potencia como una columna de la matriz de diseño.
- **Hiperparámetro:** grado (d).
- **Ventajas:** extensión directa del ajuste lineal y útil para curvatura suave.
- **Limitaciones:** grados altos pueden ser inestables, difíciles de interpretar y propensos a overfitting.
- **Casos adecuados:** patrón curvo visible o presente en residuos.
- **No conviene:** extrapolación lejana o elegir un grado alto solo por mayor (R^2) de entrenamiento.

### KNN para regresión

- **Objetivo:** predecir localmente sin imponer forma paramétrica.
- **Entradas:** predictores, respuesta, punto (x_0), distancia y (K).
- **Salida:** promedio de las respuestas de los (K) vecinos más cercanos.
- **Funcionamiento:** identifica el vecindario (N_0) y promedia sus (y_i).
- **Hiperparámetro principal:** (K); también importan escala y métrica de distancia.
- **Ventajas:** flexible y eficaz para no linealidad en baja dimensión.
- **Limitaciones:** poca interpretabilidad, alta varianza con (K) pequeño, alto sesgo con (K) grande y fuerte deterioro por dimensionalidad.
- **Casos adecuados:** muchos datos respecto del número de predictores y estructura local.
- **No conviene:** pocos casos por predictor, muchos predictores irrelevantes o necesidad prioritaria de inferencia.

## 6. Fórmulas esenciales

### Modelo lineal y predicción

\[
Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\epsilon,
\qquad
\hat y=\hat\beta_0+\sum_{j=1}^{p}\hat\beta_jx_j
\]

- (Y): respuesta cuantitativa; (X_j): predictor (j); (\beta_0): intercepto; (\beta_j): asociación parcial; (\epsilon): error aleatorio; (\hat y): predicción.
- Cada (\beta_j) es el cambio promedio en (Y) por una unidad de (X_j), manteniendo los demás predictores constantes.
- Supone que la media condicional puede aproximarse por una combinación lineal de las columnas de diseño.

### Mínimos cuadrados, RSS y RSE

\[
RSS=\sum_{i=1}^{n}(y_i-\hat y_i)^2,
\qquad
RSE=\sqrt{\frac{RSS}{n-p-1}}
\]

- (n): observaciones; (p): predictores; (y_i-\hat y_i=e_i): residuo.
- RSS es el criterio minimizado; RSE estima la desviación típica del error en unidades de (Y).
- El denominador descuenta los (p+1) parámetros estimados.

En regresión simple:

\[
\hat\beta_1=
\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},
\qquad
\hat\beta_0=\bar y-\hat\beta_1\bar x
\]

### Coeficientes: error estándar, intervalo y test

Para regresión simple:

\[
SE(\hat\beta_1)^2=
\frac{\sigma^2}{\sum_i(x_i-\bar x)^2},
\qquad
t=\frac{\hat\beta_j-0}{SE(\hat\beta_j)}
\]

Un intervalo aproximado del 95 % es:

\[
\hat\beta_j \pm 2\,SE(\hat\beta_j)
\]

- El error estándar disminuye al aumentar la información y, para la pendiente simple, la dispersión de (X).
- El estadístico (t) mide cuántos errores estándar separan la estimación de cero.
- La aproximación precisa usa el cuantil apropiado de una distribución (t); la inferencia clásica asume errores no correlacionados con varianza común y, para resultados exactos en muestras pequeñas, normalidad.

### Coeficiente de determinación

\[
R^2=1-\frac{RSS}{TSS},
\qquad
TSS=\sum_{i=1}^{n}(y_i-\bar y)^2
\]

- (TSS): variabilidad total; (RSS): variabilidad no explicada.
- (R^2\in[0,1]) en estos modelos con intercepto y mide la fracción de variabilidad explicada en la muestra.
- En regresión simple, (R^2=\operatorname{Cor}(X,Y)^2); en regresión múltiple, (R^2=\operatorname{Cor}(Y,\hat Y)^2).

### Test (F) global y comparación de modelos anidados

\[
F=\frac{(TSS-RSS)/p}{RSS/(n-p-1)}
\]

contrasta (H_0:\beta_1=\cdots=\beta_p=0). Para eliminar conjuntamente (q) variables:

\[
F=\frac{(RSS_0-RSS)/q}{RSS/(n-p-1)}
\]

- (RSS_0): RSS del modelo reducido; (RSS): RSS del modelo completo.
- Un valor mucho mayor que 1 aporta evidencia contra el modelo reducido, pero el p-value depende de (n) y (p).

### Interacción

\[
Y=\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_1X_2+\epsilon
\]

La pendiente de (X_1) es (\beta_1+\beta_3X_2); por tanto, (\beta_3) cuantifica cómo cambia el efecto de (X_1) al variar (X_2).

### Leverage y VIF

En regresión simple:

\[
h_i=\frac{1}{n}+\frac{(x_i-\bar x)^2}{\sum_{i'=1}^{n}(x_{i'}-\bar x)^2}
\]

La media de leverage en regresión múltiple es ((p+1)/n); valores muy superiores sugieren observaciones de alto leverage.

\[
VIF(\hat\beta_j)=\frac{1}{1-R^2_{X_j\mid X_{-j}}}
\]

- (R^2_{X_j\mid X_{-j}}): (R^2) al predecir (X_j) con los demás predictores.
- El mínimo es 1. El capítulo usa como regla práctica que valores mayores que 5 o 10 indican colinealidad problemática.

### Predicción KNN

\[
\hat f(x_0)=\frac{1}{K}\sum_{x_i\in N_0}y_i
\]

- (N_0): conjunto de los (K) puntos de entrenamiento más cercanos a (x_0).
- (K) pequeño produce un ajuste más flexible; (K) grande suaviza al promediar más observaciones.

## 7. Implementación práctica

El laboratorio del capítulo usa `statsmodels.api.OLS`, una matriz de diseño construida con `ISLP.models.ModelSpec`, `get_prediction()` para intervalos, `get_influence()` para leverage, `variance_inflation_factor()` para VIF y `anova_lm()` para comparar modelos anidados.

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Separar antes de cualquier transformación aprendida.
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")

features = ["x1", "x2"]
X_train = sm.add_constant(train[features], has_constant="add")
y_train = train["target"]

model = sm.OLS(y_train, X_train).fit()

X_valid = sm.add_constant(valid[features], has_constant="add")
prediction = model.get_prediction(X_valid).summary_frame(alpha=0.05)

diagnostics = pd.DataFrame({
    "fitted": model.fittedvalues,
    "residual": model.resid,
    "studentized_residual": model.get_influence().resid_studentized_external,
    "leverage": model.get_influence().hat_matrix_diag,
})

vif = pd.Series(
    [variance_inflation_factor(X_train.to_numpy(), i)
     for i in range(1, X_train.shape[1])],
    index=features,
    name="VIF",
)
```

`summary_frame()` separa `mean_ci_lower`/`mean_ci_upper` para la media de la respuesta y `obs_ci_lower`/`obs_ci_upper` para una observación individual.

**Nota práctica - adaptación moderna:** para un baseline predictivo con categóricas, polinomios o KNN, una `Pipeline` de `scikit-learn` permite ajustar transformaciones únicamente con entrenamiento y aplicar exactamente las mismas operaciones a validación. Esta API no es la implementación original del laboratorio, pero preserva la lógica `fit`/`transform` que el capítulo explica.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = ["x1", "x2"]
categorical = ["category"]

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", drop="first"),
    ), categorical),
])

baseline = make_pipeline(preprocess, LinearRegression())
baseline.fit(train[numeric + categorical], train["target"])
valid_predictions = baseline.predict(valid[numeric + categorical])
```

La semilla no interviene en mínimos cuadrados deterministas. Debe fijarse `random_state` cuando la división de datos o el procedimiento de validación sea aleatorio; el capítulo no desarrolla esa partición en este laboratorio.

## 8. Supuestos y condiciones de aplicación

1. La media de (Y) puede aproximarse linealmente mediante las columnas incluidas en la matriz de diseño.
2. El efecto es aditivo salvo que se incluyan interacciones.
3. Los errores tienen media cero y son independientes de los predictores.
4. Para los errores estándar clásicos, los errores deben ser no correlacionados y tener varianza común.
5. La normalidad de errores sustenta tests e intervalos exactos en muestras pequeñas; con muestras grandes, el test (F) es aproximadamente válido bajo condiciones más amplias.
6. Las observaciones deben aportar información independiente; series temporales, familias o grupos pueden violar esta condición.
7. La matriz de diseño debe tener rango suficiente; colinealidad exacta impide identificar coeficientes y colinealidad fuerte los vuelve inestables.
8. Mínimos cuadrados ordinarios presentado aquí requiere (p<n) para el ajuste e inferencia estándar.
9. Las variables cualitativas con intercepto deben codificarse con una categoría de referencia para evitar redundancia exacta.
10. Las predicciones son más fiables dentro del rango y estructura de los predictores observados; el capítulo no respalda extrapolaciones lejanas.
11. KNN necesita una noción de distancia útil y suficientes observaciones cercanas; la escala y la dimensionalidad determinan qué significa “cercano”.

## 9. Errores y riesgos

- Interpretar una pendiente simple como efecto parcial cuando existen predictores correlacionados omitidos.
- Convertir asociación en causalidad; los modelos del capítulo no identifican efectos causales.
- Concluir que un predictor no importa solo por un p-value grande sin revisar colinealidad, potencia y especificación.
- Buscar p-values pequeños entre muchos predictores sin un test global o control de multiplicidad.
- Elegir variables por el mayor (R^2) de entrenamiento: (R^2) nunca disminuye al añadir columnas.
- Evaluar transformaciones o modelos repetidamente sobre el conjunto final de prueba. **Recomendación para competencias:** reservar ese conjunto para la evaluación final; esta es una derivación del énfasis del capítulo en test MSE, no un protocolo descrito en el laboratorio.
- Ajustar imputación, escalado, codificación o selección con todos los datos antes de separar validación: produce data leakage. **Conclusión práctica derivada** de la lógica `fit`/`transform` presentada en el laboratorio.
- Ignorar patrones de residuos: invalida inferencia y puede degradar predicción.
- Confundir un outlier de respuesta con un punto de alto leverage. Un caso puede ser ambos, combinación especialmente influyente.
- Eliminar automáticamente un outlier: puede ser un error de captura, pero también evidencia de un predictor omitido o modelo deficiente.
- Conservar errores correlacionados: subestima errores estándar, estrecha intervalos y reduce artificialmente p-values.
- Ignorar heteroscedasticidad: vuelve poco fiables tests e intervalos clásicos.
- Usar un polinomio de grado alto por su ajuste visual o (R^2) de entrenamiento: riesgo de sobreajuste e inestabilidad.
- Eliminar efectos principales al conservar una interacción: cambia el significado del término producto y viola el principio jerárquico.
- Interpretar coeficientes de dummies sin declarar la categoría de referencia.
- Usar backward selection cuando (p>n): el modelo completo no puede ajustarse como se requiere.
- Usar KNN sin escalar variables o con muchos predictores irrelevantes. **Nota práctica:** el escalado no se desarrolla en este capítulo, pero se deriva de que KNN depende de distancias.
- Seleccionar (K) por error de entrenamiento: (K=1) interpola y puede tener varianza alta.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo:** ante una respuesta continua, especialmente en datos tabulares donde se necesita un baseline rápido, interpretable y fácil de diagnosticar.
- **Decisiones que permite tomar:** distinguir efecto marginal de efecto parcial; elegir entre términos aditivos, interacciones y polinomios; decidir si una categórica necesita dummies; detectar colinealidad, heteroscedasticidad, outliers y leverage; escoger intervalo de confianza o de predicción.
- **Baseline:** una regresión lineal con preprocesamiento ajustado solo en entrenamiento, métricas de validación y gráfico de residuos. Mantener una versión simple facilita localizar problemas.
- **Mejoras del modelo:** agregar una interacción respaldada por residuos o contexto; introducir un término cuadrático ante curvatura; retirar o combinar variables redundantes con VIF alto; comparar KNN si hay baja dimensión y no linealidad.
- **Qué no implementar bajo presión:** búsqueda exhaustiva de (2^p) subconjuntos, polinomios de grado alto, todas las interacciones por pares, eliminación automática de outliers o inferencia extensa si el objetivo es solo predictivo.
- **Señales del dataset:** respuesta continua activa regresión; categóricas activan one-hot encoding; curvatura residual activa transformaciones; embudo activa revisión de varianza; orden temporal o grupos activan sospecha de errores correlacionados; VIF alto activa revisión de redundancia; muchos predictores con pocos casos desaconsejan KNN y mínimos cuadrados ordinarios sin regularización.
- **Recomendación para competencias:** comparar el baseline lineal y KNN con la misma validación. Si KNN mejora solo en entrenamiento, aumentar (K), reducir variables irrelevantes o descartarlo; si el lineal falla con patrón residual, probar primero una extensión interpretable antes de saltar a un modelo mucho más complejo.

## 11. Checklist operativo

- [ ] Confirmar que la respuesta sea cuantitativa y revisar sus unidades.
- [ ] Separar entrenamiento y validación antes de ajustar transformaciones.
- [ ] Identificar numéricas, categóricas y categoría de referencia.
- [ ] Visualizar relaciones y correlaciones entre predictores.
- [ ] Ajustar un baseline lineal con intercepto.
- [ ] Revisar test (F) global, coeficientes, intervalos y tamaños de efecto.
- [ ] Evaluar RSE/(R^2) de ajuste y una métrica fuera de muestra.
- [ ] Graficar residuos frente a valores ajustados.
- [ ] Revisar residuos en orden temporal o por grupos cuando corresponda.
- [ ] Inspeccionar residuos studentizados y leverage sin borrar casos automáticamente.
- [ ] Calcular VIF y revisar variables redundantes.
- [ ] Añadir interacción solo con efectos principales.
- [ ] Probar términos polinómicos solo ante evidencia de curvatura y validarlos.
- [ ] Usar intervalo de predicción para casos individuales.
- [ ] Si se prueba KNN, escalar dentro del pipeline y validar (K).
- [ ] Documentar supuestos incumplidos y limitar la interpretación.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Regresión lineal simple y múltiple por mínimos cuadrados | `knowledge/models/linear_regression.md` | Alta |
| Interpretación condicional de coeficientes | `knowledge/interpretability/linear_coefficients.md` | Alta |
| Tests (t/F), errores estándar e intervalos | `knowledge/interpretability/linear_model_inference.md` | Alta |
| RSS, RSE y (R^2) | `knowledge/metrics/regression_fit_metrics.md` | Alta |
| Intervalos de confianza frente a predicción | `knowledge/metrics/regression_prediction_intervals.md` | Alta |
| Dummies y categoría de referencia | `knowledge/feature_engineering/categorical_encoding_linear_models.md` | Alta |
| Interacciones y principio jerárquico | `knowledge/feature_engineering/interaction_terms.md` | Alta |
| Regresión polinómica | `knowledge/feature_engineering/polynomial_features.md` | Media |
| Diagnóstico con residuos, outliers y leverage | `knowledge/validation/regression_diagnostics.md` | Alta |
| Heteroscedasticidad, dependencia y weighted least squares | `knowledge/models/linear_regression_assumption_repairs.md` | Media |
| Colinealidad y VIF | `knowledge/feature_engineering/multicollinearity_vif.md` | Alta |
| Selección forward, backward y mixed | `knowledge/models/classical_variable_selection.md` | Media |
| KNN para regresión y dimensionalidad | `knowledge/models/knn_regression.md` | Alta |
| Implementación y diagnóstico con `statsmodels` | `knowledge/models/python_statsmodels_linear_regression.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Introducción; pp. 69–70 |
| 2. Regresión simple y mínimos cuadrados | 3.1–3.1.1; pp. 70–72; ecuaciones 3.1–3.4 |
| 2. Incertidumbre e inferencia | 3.1.2; pp. 72–77; ecuaciones 3.5–3.14; Tabla 3.1 |
| 2. Calidad del ajuste | 3.1.3; pp. 77–80; ecuaciones 3.15–3.18; Tabla 3.2 |
| 2–5. Regresión múltiple y efectos parciales | 3.2–3.2.1; pp. 80–83; ecuaciones 3.19–3.22; Tablas 3.3–3.5 |
| 2–6. Test (F) y selección de variables | 3.2.2; pp. 83–88; ecuaciones 3.23–3.24 |
| 2. Ajuste y predicción | 3.2.2; pp. 88–90; ecuación 3.25; Figura 3.5 |
| 2 y 5. Predictores cualitativos | 3.3.1; pp. 91–94; ecuaciones 3.26–3.30; Tablas 3.7–3.8 |
| 2 y 5. Interacciones | 3.3.2; pp. 94–98; ecuaciones 3.31–3.35; Tabla 3.9; Figura 3.7 |
| 2 y 5. Regresión polinómica | 3.3.2; pp. 98–99; ecuación 3.36; Figura 3.8; Tabla 3.10 |
| 2, 8 y 9. No linealidad y errores correlacionados | 3.3.3; pp. 100–102; Figuras 3.9–3.10 |
| 2, 8 y 9. Heteroscedasticidad y outliers | 3.3.3; pp. 103–104; Figuras 3.11–3.12 |
| 2, 6 y 9. Leverage | 3.3.3; pp. 104–106; ecuación 3.37; Figura 3.13 |
| 2, 6 y 9. Colinealidad y VIF | 3.3.3; pp. 106–109; Figuras 3.14–3.15; Tabla 3.11 |
| 3–4. Flujo y reglas aplicadas al caso Advertising | 3.4; pp. 109–111 |
| 2, 4–6. Regresión lineal frente a KNN | 3.5; pp. 111–115; Figuras 3.16–3.20 |
| 7. Ajuste, predicción y diagnósticos en Python | 3.6.1–3.6.4; pp. 116–124 |
| 7. Interacciones, polinomios y categóricas en Python | 3.6.5–3.6.7; pp. 124–126 |
| 10–11. Aplicación y checklist | Síntesis práctica derivada de 3.1–3.6; pp. 69–126 |
| Ejercicios revisados pero no incorporados como contenido | 3.7; pp. 127–134 |

Todas las páginas del PDF (páginas impresas 69–134) pudieron procesarse. Los ejercicios de las pp. 127–134 se revisaron solo para delimitar el capítulo y no se incorporaron por carecer de solución. No se usaron fuentes externas.
