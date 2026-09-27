---
title: "Linear Model Selection and Regularization"
topics:
  - "Subset selection"
  - "Ridge regression and lasso"
  - "Principal components regression and partial least squares"
  - "High-dimensional regression"
target_knowledge_files:
  - "knowledge/models/linear_model_selection.md"
  - "knowledge/models/regularized_linear_models.md"
  - "knowledge/feature_engineering/dimension_reduction.md"
  - "knowledge/validation/model_selection.md"
---

# 6. Linear Model Selection and Regularization

## 1. Propósito del capítulo

El capítulo estudia cómo mejorar un modelo lineal frente a mínimos cuadrados ordinarios cuando hay demasiados predictores, alta varianza, multicolinealidad o necesidad de interpretación. Organiza las alternativas en tres familias: seleccionar un subconjunto de variables, contraer coeficientes mediante regularización y proyectar los predictores a un espacio de menor dimensión.

El objetivo práctico es mejorar la predicción fuera de muestra y, cuando el método lo permite, producir modelos más simples. El capítulo enfatiza que el error de entrenamiento no sirve para elegir complejidad: la selección debe apoyarse en una estimación del error de prueba, preferiblemente validación o cross-validation (CV).

## 2. Conceptos esenciales

- **Por qué no basta con mínimos cuadrados.** Si la relación es aproximadamente lineal y \(n\gg p\), mínimos cuadrados suele tener poco sesgo y varianza moderada. Cuando \(p\) se acerca a \(n\), las estimaciones se vuelven muy variables; si \(p>n\), no existe una solución única y puede lograrse ajuste perfecto de entrenamiento con mala generalización.
- **Selección de variables (feature/variable selection).** Busca excluir predictores mediante coeficientes exactamente iguales a cero. Best subset y los métodos stepwise lo hacen de forma explícita; lasso lo hace mediante una penalización.
- **Regularización (shrinkage).** Ajusta todos los predictores pero penaliza coeficientes grandes. Introduce sesgo para reducir varianza. Ridge conserva todos los predictores; lasso puede generar una solución dispersa (*sparse*).
- **Reducción de dimensión.** Sustituye los \(p\) predictores por \(M<p\) combinaciones lineales y ajusta mínimos cuadrados sobre ellas. PCR construye componentes sin usar \(Y\); PLS usa \(Y\) para orientar las componentes.
- **Parámetro de ajuste.** \(\lambda\) controla la intensidad de ridge y lasso; \(M\) controla el número de componentes en PCR y PLS. Ambos deben elegirse con validación.
- **Trade-off sesgo-varianza.** Mayor regularización o menos componentes reducen flexibilidad y varianza, pero elevan sesgo. El mínimo de error de prueba suele aparecer en un nivel intermedio.
- **Regla de un error estándar.** Si varios modelos tienen errores CV similares, se elige el modelo más simple cuyo error estimado esté dentro de un error estándar del mínimo.
- **Alta dimensión.** El problema no comienza únicamente en \(p>n\); las mismas precauciones son relevantes cuando \(p\) es solo algo menor que \(n\). Agregar variables de ruido puede elevar el error de prueba: es una manifestación de la maldición de la dimensionalidad.
- **Predicción no equivale a identificación causal.** En alta dimensión pueden existir muchos subconjuntos predictivos equivalentes. Un buen modelo de prueba no demuestra que las variables seleccionadas sean las únicas ni las verdaderamente causales.

## 3. Flujo o procedimiento presentado

### Selección de subconjuntos

1. Construir una secuencia de modelos candidatos: todos los subconjuntos, una ruta forward o una ruta backward.
2. Para cada tamaño \(k\), identificar el mejor modelo usando RSS mínimo, \(R^2\) máximo o deviance mínima en modelos de clasificación.
3. Estimar el error fuera de muestra de cada tamaño mediante validation set o CV. Como alternativa de baja dimensión, usar \(C_p\), AIC, BIC o \(R^2\) ajustado.
4. Si se usa CV, repetir **la selección completa** dentro de cada fold; el mejor subconjunto de tamaño \(k\) puede cambiar entre folds.
5. Elegir el tamaño con menor error estimado o aplicar la regla de un error estándar para favorecer simplicidad.
6. Repetir la selección sobre todos los datos de entrenamiento usando el tamaño elegido y ajustar el modelo final.

### Ridge o lasso

1. Separar los datos destinados a evaluación final antes de cualquier selección, si se necesita una estimación honesta del rendimiento final.
2. Definir una rejilla de valores de \(\lambda\) que cubra desde fuerte regularización hasta una solución cercana a mínimos cuadrados.
3. Dentro de cada fold, estimar el centrado y escalado exclusivamente con el bloque de entrenamiento.
4. Ajustar ridge o lasso para cada \(\lambda\) y medir el error en el fold retenido.
5. Elegir \(\lambda\) por CV; si la curva es plana, considerar una solución más simple o más regularizada.
6. Reajustar todo el pipeline con los datos de entrenamiento disponibles y el \(\lambda\) elegido.
7. Evaluar una sola vez en el conjunto externo si se reservó uno.

### PCR o PLS

1. Estandarizar predictores dentro del pipeline, salvo que compartan unidades y exista una razón para conservar su escala.
2. Para PCR, calcular componentes principales de \(X\); para PLS, obtener direcciones relacionadas simultáneamente con \(X\) y \(Y\).
3. Ajustar una regresión lineal sobre las primeras \(M\) componentes.
4. Repetir para una rejilla de \(M\) y elegir el número de componentes mediante CV.
5. Reajustar transformación y regresión con todos los datos de entrenamiento.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| \(p\) es pequeño o moderado y se desea el mejor subconjunto por tamaño | Best subset selection | Examina todas las combinaciones y da el menor RSS para cada tamaño | Requiere evaluar \(2^p\) modelos y puede sobreajustar por búsqueda extensa |
| \(p\) impide búsqueda exhaustiva | Forward stepwise | Requiere aproximadamente \(1+p(p+1)/2\) ajustes | Es greedy y no garantiza el mejor subconjunto global |
| Se puede ajustar el modelo completo y se desea eliminación progresiva | Backward stepwise | Tiene coste similar a forward y parte de todas las variables | Requiere \(n>p\); tampoco garantiza el óptimo global |
| Varios tamaños presentan error CV casi indistinguible | Regla de un error estándar | Favorece el modelo más simple compatible con el mínimo | Requiere estimar la incertidumbre del error CV |
| Muchas variables aportan señales pequeñas de magnitud comparable | Ridge | Contrae todas las direcciones y puede reducir mucho la varianza | No selecciona variables; hay que estandarizar y ajustar \(\lambda\) |
| Se espera que pocas variables tengan efectos importantes | Lasso | Puede fijar coeficientes exactamente en cero y producir un modelo sparse | La selección puede ser inestable con predictores correlacionados; validar predicción |
| La información predictiva se concentra en direcciones de alta varianza de \(X\) | PCR | Resume predictores correlacionados con pocas componentes | PCA no usa \(Y\); alta varianza en \(X\) no garantiza relevancia para \(Y\) |
| Se buscan componentes orientadas a predecir \(Y\) | PLS | Usa la respuesta al construir las direcciones | La supervisión puede reducir sesgo pero aumentar varianza; a menudo no supera ridge o PCR |
| \(p\ge n\) o \(p\) está cerca de \(n\) | Ridge, lasso, forward stepwise o PCR con validación | Reducen la flexibilidad frente a mínimos cuadrados | No usar métricas de entrenamiento como evidencia y no confiar en \(C_p\), AIC, BIC o \(R^2\) ajustado en alta dimensión |
| Solo importa predicción y se dispone de tiempo de cómputo | Comparar ridge, lasso y métodos de componentes con CV común | Ningún método domina universalmente | Mantener los mismos splits y encapsular transformaciones dentro de cada fold |

## 5. Métodos y técnicas

### Best subset selection

- **Objetivo:** obtener, para cada tamaño \(k\), el subconjunto con mejor ajuste de entrenamiento y después elegir el tamaño por error estimado fuera de muestra.
- **Entradas:** matriz \(X\), respuesta \(Y\), métrica de ajuste por tamaño y criterio de selección final.
- **Salida:** una ruta de \(p+1\) modelos y un modelo final.
- **Funcionamiento:** ajusta todas las \(\binom{p}{k}\) combinaciones para cada \(k\), en total \(2^p\).
- **Hiperparámetros:** tamaño final \(k\), elegido con CV, validación o un criterio ajustado.
- **Ventajas:** óptimo de entrenamiento dentro de cada tamaño; concepto simple.
- **Limitaciones:** crecimiento exponencial; una búsqueda enorme puede encontrar asociaciones accidentales y aumentar varianza.
- **Casos adecuados:** \(p\) pequeño o moderado y fuerte necesidad de un subconjunto explícito.
- **No conviene:** alta dimensión o presupuesto computacional reducido.

### Forward, backward e híbridos stepwise

- **Objetivo:** aproximar la selección de subconjuntos con una búsqueda guiada.
- **Entradas:** datos, dirección de búsqueda, criterio local y criterio externo para el tamaño.
- **Salida:** modelos anidados a lo largo de una ruta.
- **Funcionamiento:** forward agrega la variable que más mejora el ajuste; backward elimina la menos útil; un híbrido permite retirar variables después de agregarlas.
- **Hiperparámetros:** dirección, número máximo de pasos y tamaño elegido.
- **Ventajas:** alrededor de \(1+p(p+1)/2\) ajustes; forward funciona cuando \(p>n\), aunque solo puede alcanzar modelos ajustables.
- **Limitaciones:** decisiones tempranas condicionan la ruta y pueden impedir llegar al mejor subconjunto de un tamaño.
- **Casos adecuados:** búsqueda exhaustiva inviable y necesidad de variables originales.
- **No conviene:** backward cuando \(p\ge n\); tampoco si se interpretará la ruta como evidencia causal.

### Ridge regression

- **Objetivo:** estabilizar predicciones contrayendo coeficientes y reduciendo varianza.
- **Entradas:** \(X\) numérica codificada, \(Y\), escala estimada solo en entrenamiento y rejilla de \(\lambda\).
- **Salida:** un coeficiente para cada predictor; normalmente ninguno es cero.
- **Funcionamiento:** minimiza RSS más una penalización \(L_2\).
- **Hiperparámetro:** \(\lambda\ge0\); \(\lambda=0\) recupera mínimos cuadrados y \(\lambda\to\infty\) lleva las pendientes hacia cero.
- **Ventajas:** útil con coeficientes distribuidos entre muchas variables, multicolinealidad y \(p\) grande; eficiente computacionalmente.
- **Limitaciones:** no selecciona variables y depende de la escala si no se estandariza.
- **Casos adecuados:** muchas señales pequeñas o mínimos cuadrados de alta varianza.
- **No conviene:** cuando la salida debe ser un subconjunto pequeño e interpretable de variables originales.

### Lasso

- **Objetivo:** regularizar y seleccionar variables simultáneamente.
- **Entradas y salida:** iguales a ridge, pero la salida puede contener numerosos coeficientes exactamente cero.
- **Funcionamiento:** minimiza RSS más una penalización \(L_1\); la geometría con esquinas de la restricción favorece soluciones sobre los ejes.
- **Hiperparámetro:** \(\lambda\), elegido por CV.
- **Ventajas:** modelos sparse, interpretabilidad y coste muy inferior a best subset para \(p\) grande.
- **Limitaciones:** no supera siempre a ridge; su ventaja predictiva depende de que la señal sea relativamente dispersa.
- **Casos adecuados:** pocos predictores con coeficientes sustanciales y muchos coeficientes nulos o pequeños.
- **No conviene:** asumir de antemano que será mejor cuando la señal se distribuye entre muchas variables de tamaño similar.

### Principal components regression (PCR)

- **Objetivo:** reducir varianza ajustando la regresión sobre \(M<p\) componentes de \(X\).
- **Entradas:** predictores centrados/escalados, respuesta y candidatos para \(M\).
- **Salida:** predicciones y coeficientes equivalentes sobre combinaciones lineales de todos los predictores.
- **Funcionamiento:** PCA maximiza sucesivamente varianza de \(X\) bajo ortogonalidad; luego mínimos cuadrados predice \(Y\) desde las primeras \(M\) componentes.
- **Hiperparámetro:** \(M\), elegido por CV.
- **Ventajas:** comprime información redundante y puede reducir mucho la varianza.
- **Limitaciones:** es no supervisado; las direcciones de mayor varianza pueden no ser predictivas. No selecciona variables originales.
- **Casos adecuados:** predictores correlacionados y señal alineada con las primeras componentes.
- **No conviene:** cuando la señal está en direcciones de baja varianza o se exige selección explícita.

### Partial least squares (PLS)

- **Objetivo:** construir \(M\) componentes que expliquen predictores y respuesta.
- **Entradas:** \(X\) y \(Y\) estandarizadas y candidatos para \(M\).
- **Salida:** componentes supervisadas y una regresión lineal sobre ellas.
- **Funcionamiento:** la primera dirección pondera cada predictor según su regresión simple/correlación con \(Y\); las siguientes se construyen sobre residuos ortogonalizados.
- **Hiperparámetro:** \(M\), elegido por CV.
- **Ventajas:** puede reducir el sesgo de PCR al incorporar la respuesta.
- **Limitaciones:** puede aumentar varianza y en la práctica a menudo no mejora ridge o PCR.
- **Casos adecuados:** muchas variables correlacionadas y potencial alineación débil entre varianza de \(X\) y señal de \(Y\).
- **No conviene:** asumir que la supervisión garantiza mejor error de prueba.

## 6. Fórmulas esenciales

### Criterio \(C_p\)

\[
C_p=\frac{1}{n}\left(RSS+2d\hat{\sigma}^2\right)
\]

- \(n\): observaciones; \(d\): predictores del modelo; \(RSS\): suma de residuos cuadrados; \(\hat{\sigma}^2\): varianza del error, normalmente estimada con el modelo completo.
- Ajusta el error de entrenamiento con una penalización por complejidad. Se prefiere un valor pequeño.
- Su justificación depende de una estimación adecuada de \(\sigma^2\); no es apropiado en alta dimensión.

### BIC para mínimos cuadrados

\[
BIC=\frac{1}{n}\left(RSS+\log(n)d\hat{\sigma}^2\right)
\]

- Usa las mismas variables que \(C_p\). Para \(n>7\), penaliza el tamaño más que \(C_p\) y suele elegir modelos menores.
- Se minimiza. El capítulo omite constantes irrelevantes para comparar modelos.

### \(R^2\) ajustado

\[
R^2_{adj}=1-\frac{RSS/(n-d-1)}{TSS/(n-1)}
\]

- \(TSS=\sum_i(y_i-\bar y)^2\); \(d\) es el número de variables.
- Penaliza predictores innecesarios mediante grados de libertad. A diferencia de \(C_p\) y BIC, se maximiza.
- No usarlo como evidencia en \(p\ge n\), donde puede resultar engañosamente perfecto.

### Ridge

\[
\hat\beta^{R}_{\lambda}=\arg\min_{\beta_0,\beta}
\left\{\sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\right)^2
+\lambda\sum_{j=1}^{p}\beta_j^2\right\}
\]

- \(\beta_0\): intercepto no penalizado; \(\beta_j\): pendiente; \(\lambda\): intensidad de regularización.
- La penalización \(L_2\) reduce la norma \(\|\beta\|_2=(\sum_j\beta_j^2)^{1/2}\).
- La interpretación de una misma \(\lambda\) depende de la escala; estandarizar antes del ajuste.

### Lasso

\[
\hat\beta^{L}_{\lambda}=\arg\min_{\beta_0,\beta}
\left\{\sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\right)^2
+\lambda\sum_{j=1}^{p}|\beta_j|\right\}
\]

- La penalización \(L_1\), \(\|\beta\|_1=\sum_j|\beta_j|\), puede llevar coeficientes exactamente a cero.
- En el caso diagonal simple del capítulo, aplica *soft-thresholding*: valores pequeños se anulan y los restantes se acercan a cero en una cantidad fija.

### Estandarización usada por el capítulo

\[
\tilde{x}_{ij}=\frac{x_{ij}}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\bar{x}_j)^2}}
\]

- El texto presupone centrado y divide por una estimación de dispersión para que los predictores queden en escala comparable.
- En implementación, centrado y escala deben aprenderse solo con el entrenamiento de cada fold.

### Reducción de dimensión

\[
Z_m=\sum_{j=1}^{p}\phi_{jm}X_j,\qquad
y_i=\theta_0+\sum_{m=1}^{M}\theta_m z_{im}+\varepsilon_i
\]

- \(\phi_{jm}\): carga del predictor \(j\) en la componente \(m\); \(M<p\); \(\theta_m\): coeficiente de la regresión sobre componentes.
- Reduce de \(p+1\) a \(M+1\) parámetros estimados y restringe los coeficientes originales a \(\beta_j=\sum_m\theta_m\phi_{jm}\).
- La reducción puede introducir sesgo, pero disminuye varianza cuando \(p\) es grande respecto de \(n\).

## 7. Implementación práctica

### Ridge y lasso sin leakage

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
alpha_grid = np.logspace(-4, 4, 80)

ridge_search = GridSearchCV(
    Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ]),
    {"model__alpha": alpha_grid},
    scoring="neg_mean_squared_error",
    cv=inner_cv,
)

lasso_search = GridSearchCV(
    Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(max_iter=20_000, random_state=42)),
    ]),
    {"model__alpha": alpha_grid},
    scoring="neg_mean_squared_error",
    cv=inner_cv,
)

ridge_search.fit(X_train, y_train)
lasso_search.fit(X_train, y_train)
```

**Nota práctica (adaptación moderna):** el laboratorio usa `ElasticNet(..., l1_ratio=0)` para ridge y `ElasticNetCV`/`Lasso.path` para rutas. La implementación anterior expresa ridge y lasso con estimadores específicos y coloca `StandardScaler` dentro de `Pipeline`, de modo que cada fold aprende su propio escalado. Esta adaptación conserva la idea del capítulo; no es una afirmación textual del autor.

### Evaluación anidada del procedimiento completo

```python
from sklearn.model_selection import cross_val_score, KFold

outer_cv = KFold(n_splits=5, shuffle=True, random_state=7)
outer_mse = -cross_val_score(
    ridge_search,
    X,
    y,
    scoring="neg_mean_squared_error",
    cv=outer_cv,
)

mean_mse = outer_mse.mean()
```

El fold externo estima el error del procedimiento que incluye escalado y búsqueda de \(\lambda\); el fold interno elige \(\lambda\). No debe usarse el mismo resultado CV empleado para seleccionar \(\lambda\) como si fuera una evaluación final independiente.

### PCR con selección del número de componentes

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

pcr_search = GridSearchCV(
    Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA()),
        ("model", LinearRegression()),
    ]),
    {"pca__n_components": range(1, min(X_train.shape) + 1)},
    scoring="neg_mean_squared_error",
    cv=inner_cv,
)
pcr_search.fit(X_train, y_train)
```

### PLS con selección del número de componentes

```python
from sklearn.cross_decomposition import PLSRegression

pls_search = GridSearchCV(
    PLSRegression(scale=True),
    {"n_components": range(1, min(X_train.shape) + 1)},
    scoring="neg_mean_squared_error",
    cv=inner_cv,
)
pls_search.fit(X_train, y_train)
```

El capítulo también implementa rutas stepwise y best subset con utilidades de `ISLP.models` y `l0bnb`. Para un coding test, la idea transferible es más importante que esas dependencias: la búsqueda de variables debe ejecutarse dentro de cada fold y no una sola vez sobre todos los datos.

## 8. Supuestos y condiciones de aplicación

1. La relación predictiva que se modela sigue siendo aproximadamente lineal en los predictores originales o en sus componentes.
2. Ridge y lasso requieren escala comparable entre predictores para que la penalización no dependa arbitrariamente de las unidades.
3. PCR supone, para funcionar bien, que las direcciones que explican gran variación de \(X\) también contienen la relación con \(Y\).
4. PLS usa \(Y\) para construir componentes; por eso toda la construcción debe repetirse dentro de cada fold.
5. Backward stepwise necesita poder ajustar el modelo completo, en particular \(n>p\) para mínimos cuadrados con solución única.
6. Los criterios \(C_p\), AIC, BIC y \(R^2\) ajustado presentados se justifican para el contexto de baja dimensión y, en los tres primeros, requieren estimar adecuadamente la varianza del error.
7. La comparación entre modelos debe usar una métrica de validación común y splits comparables.
8. La selección por CV estima desempeño predictivo; no identifica de manera única las variables verdaderamente asociadas ni establece causalidad.
9. Variables categóricas codificadas en varias columnas pueden seleccionarse por nivel si el procedimiento opera sobre columnas individuales; la interpretación debe respetar esa codificación.
10. Un aumento de predictores solo ayuda si incorpora señal suficiente para compensar la varianza añadida.

## 9. Errores y riesgos

- Elegir el modelo por RSS mínimo o \(R^2\) máximo de entrenamiento: ambas métricas mejoran mecánicamente al agregar variables.
- Ejecutar selección de variables sobre todo el dataset y luego aplicar CV al modelo ya seleccionado: filtra información de los folds retenidos.
- Estandarizar, imputar o calcular PCA antes de dividir los folds: produce data leakage.
- Reportar el mínimo error de la misma CV usada para ajustar \(\lambda\) como evaluación final independiente.
- Usar mínimos cuadrados sin regularización cuando \(p\ge n\): puede interpolar entrenamiento y tener una solución no única.
- Interpretar \(R^2=1\), RSS cero, p-values o MSE de entrenamiento como evidencia de calidad en alta dimensión.
- Aplicar backward selection con \(p\ge n\), donde el modelo completo no puede ajustarse de forma única.
- Usar \(C_p\), AIC, BIC o \(R^2\) ajustado en alta dimensión como si sus estimaciones siguieran siendo fiables.
- Omitir la estandarización en ridge, lasso, PCR o PLS cuando las escalas difieren.
- Tratar los coeficientes cero del lasso como prueba de irrelevancia causal o como una selección estable entre variables correlacionadas.
- Confundir PCR con selección de variables: cada componente suele involucrar todos los predictores.
- Suponer que PLS siempre mejora PCR por usar \(Y\); la reducción de sesgo puede pagarse con mayor varianza.
- Aumentar indiscriminadamente \(p\): los predictores de ruido elevan el riesgo de asociaciones accidentales y error de prueba.
- Escoger automáticamente el mínimo de una curva CV casi plana cuando un modelo mucho más simple tiene rendimiento indistinguible.
- Ignorar coste computacional: best subset escala como \(2^p\), mientras stepwise, ridge, lasso y rutas de regularización son mucho más viables.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Conviene consultar este capítulo cuando el baseline lineal presenta multicolinealidad, demasiadas columnas, \(p\) cercano o superior a \(n\), alta inestabilidad entre splits o una exigencia de reducir variables. La primera decisión es separar dos objetivos: si se busca predicción estable con señal repartida, probar ridge; si también se necesita sparsity, probar lasso; si existen bloques de variables correlacionadas y se acepta perder interpretabilidad directa, comparar PCR.

Un baseline competitivo bajo tiempo limitado es `Pipeline(StandardScaler(), Ridge())` con una rejilla logarítmica de \(\lambda\) y 5-fold CV reproducible. Lasso es la siguiente comparación cuando se sospecha señal dispersa o se necesita un conjunto reducido de columnas. PCR puede mejorar si unas pocas direcciones resumen bien \(X\); PLS requiere más cautela porque su ventaja no es universal.

Para mejorar el modelo, inspeccionar la curva CV completa y no solo el mínimo: una zona plana habilita una elección más simple mediante la regla de un error estándar. Si hay evaluación final disponible, reservarla y anidar el tuning. Bajo presión de tiempo no conviene programar best subset exhaustivo, criterios \(C_p\)/BIC personalizados ni búsquedas stepwise complejas si ridge/lasso ofrecen un pipeline robusto y rápido.

Señales del dataset que activan estas técnicas:

- \(p/n\) alto o \(p\ge n\): regularización obligatoria como candidato; descartar OLS sin restricción.
- Coeficientes o predicciones muy variables entre particiones: ridge o reducción de dimensión.
- Muchas variables plausiblemente irrelevantes: lasso o forward stepwise con selección anidada.
- Predictores fuertemente correlacionados: ridge o PCR.
- Curva CV plana: modelo más simple dentro de un error estándar.
- Métricas de entrenamiento casi perfectas pero CV pobre: sobreajuste de alta dimensión.
- Necesidad de interpretación: lasso o subset selection, aclarando que la selección puede no ser única.

## 11. Checklist operativo

- [ ] Calcular \(n\), \(p\) y la razón \(p/n\).
- [ ] Separar evaluación externa antes de seleccionar variables o hiperparámetros.
- [ ] Elegir una métrica fuera de muestra coherente con el objetivo.
- [ ] Encapsular escalado y cualquier transformación en un pipeline.
- [ ] Fijar splits reproducibles y usar los mismos folds al comparar métodos.
- [ ] Incluir \(\lambda\) en una rejilla logarítmica amplia para ridge/lasso.
- [ ] Ajustar \(M\) por CV en PCR/PLS.
- [ ] Ejecutar selección de variables completa dentro de cada fold.
- [ ] Comparar el mínimo CV con una alternativa más simple mediante la regla de un error estándar.
- [ ] Reajustar el procedimiento elegido con todos los datos de entrenamiento.
- [ ] Evaluar una sola vez en datos externos o usar CV anidada.
- [ ] Reportar error de validación, no solo métricas de entrenamiento.
- [ ] Evitar afirmaciones causales sobre variables seleccionadas.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Best subset, forward, backward y selección del tamaño | `knowledge/models/linear_model_selection.md` | Alta |
| Ridge, lasso, penalizaciones \(L_1/L_2\) y elección de \(\lambda\) | `knowledge/models/regularized_linear_models.md` | Alta |
| Pipelines sin leakage para escalado y selección | `knowledge/validation/model_selection.md` | Alta |
| CV anidada para estimar el error del procedimiento ajustado | `knowledge/validation/nested_cross_validation.md` | Alta |
| PCR, PCA supervisada por regresión y elección de componentes | `knowledge/feature_engineering/principal_components_regression.md` | Media |
| PLS y componentes supervisadas | `knowledge/feature_engineering/partial_least_squares.md` | Media |
| Regresión cuando \(p\) se acerca o supera a \(n\) | `knowledge/models/high_dimensional_regression.md` | Alta |
| \(C_p\), AIC, BIC, \(R^2\) ajustado y límites | `knowledge/metrics/information_criteria.md` | Media |
| Regla de un error estándar | `knowledge/validation/one_standard_error_rule.md` | Media |
| Riesgos interpretativos de selección en alta dimensión | `knowledge/interpretability/feature_selection_stability.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Introducción, pp. 229–230 (PDF 1–2) |
| 2. Conceptos esenciales | Introducción; 6.1–6.4, pp. 229–267 (PDF 1–39) |
| 3. Flujo o procedimiento presentado | Algoritmos 6.1–6.3; 6.2.3; 6.3; laboratorio 6.5, pp. 231–282 (PDF 3–54) |
| 4. Reglas de decisión | 6.1.1–6.1.3, 6.2.1–6.2.3, 6.3.1–6.3.2 y 6.4, pp. 231–267 (PDF 3–39) |
| 5. Métodos y técnicas | 6.1–6.4, pp. 231–267 (PDF 3–39) |
| 6. Fórmulas esenciales | Ecs. 6.2–6.10 y 6.16–6.18, pp. 236–254 (PDF 8–26) |
| 7. Implementación práctica | 6.5.1–6.5.3, pp. 267–282 (PDF 39–54) |
| 8. Supuestos y condiciones | 6.1.2–6.1.3, 6.2, 6.3 y 6.4, pp. 233–267 (PDF 5–39) |
| 9. Errores y riesgos | 6.1.3, 6.2, 6.3, 6.4 y nota de CV del laboratorio, pp. 236–279 (PDF 8–51) |
| 10. Aplicación en un coding test | Síntesis práctica derivada de 6.1–6.5, pp. 231–282 (PDF 3–54) |
| 11. Checklist operativo | Síntesis práctica derivada de 6.1.3, 6.2.3, 6.3 y 6.5, pp. 236–282 (PDF 8–54) |
| 12. Mapeo hacia `knowledge/` | Clasificación documental derivada de las secciones 6.1–6.5, pp. 231–282 (PDF 3–54) |

