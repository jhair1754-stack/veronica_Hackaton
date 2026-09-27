---
title: "Support Vector Machines"
topics:
  - "Maximal margin classifier y support vector classifier"
  - "Kernels lineal, polinomial y radial"
  - "Ajuste, validación y evaluación de SVM"
  - "Clasificación multiclase y relación con regresión logística"
target_knowledge_files:
  - "knowledge/models/support_vector_machines.md"
  - "knowledge/tuning/svm_hyperparameters.md"
  - "knowledge/validation/model_selection.md"
  - "knowledge/metrics/classification_metrics.md"
---

# 9. Support Vector Machines

## 1. Propósito del capítulo

El capítulo desarrolla una familia de métodos supervisados basada en hiperplanos: parte del clasificador de margen máximo para clases linealmente separables, introduce el clasificador de vectores de soporte (support vector classifier) para permitir violaciones del margen y llega a las máquinas de vectores de soporte (support vector machines, SVM) mediante kernels que producen fronteras no lineales.

Su valor práctico está en decidir cuándo usar una frontera lineal o no lineal, cómo controlar el compromiso sesgo-varianza, cómo seleccionar hiperparámetros mediante validación cruzada y cómo evaluar el modelo fuera de entrenamiento. El foco principal es clasificación; el capítulo solo presenta brevemente la extensión a regresión.

## 2. Conceptos esenciales

### 2.1 Hiperplano y regla de clasificación

**Contenido explicado por el autor.** En un espacio de (p) variables, un hiperplano es un subespacio afín plano de dimensión (p-1):

\[
\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p = 0.
\]

El signo de

\[
f(x)=\beta_0 + \beta_1x_1+\cdots+\beta_px_p
\]

indica a qué lado del hiperplano está una observación. En clasificación binaria, codificando (y_i\in\{-1,1\}), un hiperplano separador cumple (y_i f(x_i)>0) para todas las observaciones de entrenamiento. La magnitud de (f(x)) refleja distancia respecto de la frontera en la parametrización normalizada: valores cercanos a cero indican menor seguridad en la asignación.

### 2.2 Margen máximo y vectores de soporte

**Contenido explicado por el autor.** Entre los hiperplanos que separan perfectamente las clases, el clasificador de margen máximo elige el que maximiza la menor distancia entre la frontera y las observaciones. Los puntos situados sobre los bordes del margen son los vectores de soporte (support vectors). Ellos determinan directamente la frontera; mover otros puntos sin que crucen el margen no la cambia.

Este clasificador exige separabilidad lineal y puede ser inestable frente a una sola observación. El capítulo advierte además que puede sobreajustar cuando (p) es grande.

### 2.3 Margen suave

**Contenido explicado por el autor.** El clasificador de vectores de soporte usa un margen suave (soft margin): permite que algunas observaciones queden dentro del margen e incluso del lado incorrecto del hiperplano. Las variables de holgura (epsilon_i) describen cada violación:

- (epsilon_i=0): observación en el lado correcto del margen;
- (0<\epsilon_i\le 1): observación dentro del margen, pero aún del lado correcto del hiperplano;
- (epsilon_i>1): observación mal clasificada.

Solo las observaciones sobre el margen o que lo violan afectan a la solución. Esta dependencia de un subconjunto de datos hace al método poco sensible a observaciones alejadas de la frontera.

### 2.4 De fronteras lineales a SVM mediante kernels

**Contenido explicado por el autor.** Una frontera no lineal puede obtenerse ampliando explícitamente el espacio de variables con potencias e interacciones. El kernel evita construir de forma explícita ese espacio: reemplaza productos internos por una función de similitud (K(x_i,x_{i'})). Una SVM es el clasificador de vectores de soporte combinado con un kernel no lineal.

Los kernels tratados son:

- lineal: reproduce el clasificador lineal;
- polinomial: permite fronteras asociadas a polinomios de grado (d);
- radial (radial basis function, RBF): da influencia principalmente a observaciones cercanas y genera una frontera local y flexible.

### 2.5 Complejidad, generalización y evaluación

**Contenido explicado por el autor.** Mayor flexibilidad puede reducir el error de entrenamiento sin mejorar el desempeño en datos nuevos. En el ejemplo de enfermedad cardíaca, el RBF más flexible produce una curva ROC casi perfecta en entrenamiento y el peor resultado entre los modelos comparados en test. Por eso (C), (\gamma) y el kernel deben elegirse con validación cruzada y evaluarse fuera del conjunto usado para ajustarlos.

**Conclusión práctica derivada del contenido.** Una frontera visualmente compleja o un accuracy de entrenamiento perfecto no son evidencia suficiente de calidad. Debe compararse el rendimiento de validación y revisar estabilidad entre configuraciones.

### 2.6 Multiclase y regresión

**Contenido explicado por el autor.** Como la construcción básica es binaria, el problema multiclase se resuelve mediante:

- one-versus-one (OvO): entrena \(\binom{K}{2}\) clasificadores y decide por votación;
- one-versus-all/rest (OvR): entrena (K) clasificadores, cada clase contra el resto, y elige el mayor score.

El capítulo menciona support vector regression (SVR): los residuos dentro de una banda no contribuyen a la pérdida y solo los residuos suficientemente grandes son penalizados. No desarrolla su ajuste en detalle.

## 3. Flujo o procedimiento presentado

1. Definir la tarea y separar datos de entrenamiento y evaluación.
2. Determinar si una frontera lineal es una hipótesis razonable. Si lo es, comenzar con kernel lineal.
3. Ajustar un `SVC` con un valor inicial de (C).
4. Inspeccionar error, matriz de confusión, número de vectores de soporte y, cuando sea posible, la frontera de decisión.
5. Construir validación cruzada reproducible sobre entrenamiento.
6. Buscar (C) para el modelo lineal o buscar conjuntamente (C) y (\gamma) para RBF; para kernel polinomial, incluir el grado (d).
7. Seleccionar el mejor estimador según una métrica definida antes de consultar el test.
8. Evaluar una sola vez sobre test con predicciones, matriz de confusión y, en clasificación binaria, scores de `decision_function` o curva ROC.
9. Para más de dos clases, elegir o documentar OvO u OvR.
10. Comparar el resultado de entrenamiento con validación/test para detectar sobreajuste.

**Nota práctica derivada.** En un pipeline real, cualquier transformación aprendida de los datos debe ajustarse dentro de cada fold y solo con entrenamiento. Esta precaución no es un paso explícito del laboratorio, pero es necesaria para preservar la lógica de evaluación fuera de muestra que el capítulo enfatiza.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Dos clases aproximadamente separables por una frontera lineal | `SVC(kernel="linear")` | El clasificador lineal es suficiente y evita flexibilidad innecesaria | Afinar (C); separabilidad en entrenamiento no garantiza generalización |
| Datos no separables o sensibles a puntos individuales | Clasificador de margen suave | Tolera puntos dentro del margen o mal clasificados y puede ser más estable | Elegir la tolerancia mediante validación cruzada |
| Frontera claramente no lineal | Kernel RBF o polinomial | Amplían implícitamente el espacio de variables | Ajustar (C) junto con (\gamma) o `degree`; vigilar sobreajuste |
| Patrón local sin forma polinómica evidente | Kernel RBF | La similitud cae con la distancia y produce influencia local | Un (\gamma) grande puede generar una frontera muy irregular |
| Se conoce una estructura polinómica plausible | Kernel polinomial | Representa términos polinómicos de grado (d) | El grado añade complejidad y debe validarse |
| (p\gg n) | Kernel lineal como primera opción | El espacio original ya tiene gran capacidad; el ejemplo de expresión génica no necesita flexibilidad adicional | Un ajuste perfecto es fácil y no sustituye la evaluación en test |
| Más de dos clases | OvO u OvR | Extienden el clasificador binario | OvO entrena \(\binom{K}{2}\) modelos; OvR entrena (K) |
| Clases bien separadas | SVM puede ser preferible a regresión logística | El capítulo señala mejor comportamiento frecuente en este régimen | Confirmar con validación, no asumir superioridad universal |
| Clases muy solapadas | Considerar regresión logística | El capítulo indica que suele preferirse en este régimen | Comparar bajo el mismo esquema de validación |
| Se necesita variar el umbral o construir ROC | Usar `decision_function()` | El signo produce la clase y el score permite recorrer umbrales | Evaluar ROC fuera de entrenamiento |

## 5. Métodos y técnicas

### 5.1 Clasificador de margen máximo

- **Objetivo:** separar dos clases maximizando la distancia mínima entre las observaciones y el hiperplano.
- **Entradas:** matriz (X\in\mathbb{R}^{n\times p}) y etiquetas binarias (y_i\in\{-1,1\}).
- **Salida:** hiperplano lineal y clase según el signo de (f(x)).
- **Funcionamiento:** busca el centro de la banda separadora más ancha.
- **Hiperparámetros:** ninguno en su formulación ideal.
- **Ventajas:** frontera interpretable; depende solo de los puntos que sostienen el margen.
- **Limitaciones:** requiere separabilidad; puede ser extremadamente sensible a una observación y sobreajustar con muchas variables.
- **Casos adecuados:** separación lineal limpia y estable.
- **No conviene:** clases solapadas, outliers cercanos a la frontera o separabilidad frágil.

### 5.2 Clasificador de vectores de soporte lineal

- **Objetivo:** obtener una frontera lineal robusta permitiendo violaciones controladas.
- **Entradas:** (X), etiquetas binarias y parámetro de regularización/tolerancia.
- **Salida:** frontera, clases, scores y vectores de soporte.
- **Funcionamiento:** maximiza el margen sujeto a variables de holgura.
- **Hiperparámetro principal:** (C), con una importante diferencia de convención:
  - en las ecuaciones (9.12)-(9.15), (C) es el presupuesto total de holgura; un (C) mayor permite más violaciones y un margen más ancho;
  - en `sklearn.svm.SVC`, `C` es el costo de violar el margen; un `C` mayor penaliza más las violaciones, estrecha el margen y ajusta más el entrenamiento.
- **Ventajas:** puede generalizar mejor que la separación perfecta; es robusto frente a puntos lejos de la frontera; para kernel lineal se pueden recuperar coeficientes.
- **Limitaciones:** (C) es decisivo y no debe dejarse como un valor irrelevante por defecto.
- **Casos adecuados:** clasificación lineal binaria y datos de alta dimensión.
- **No conviene:** frontera esencialmente no lineal que no pueda representarse bien en el espacio original.

### 5.3 SVM con kernel polinomial

- **Objetivo:** representar una frontera no lineal asociada a una expansión polinómica.
- **Entradas:** (X), (y), (C) y grado (d).
- **Salida:** score (f(x)=\beta_0+\sum_{i\in S}\alpha_iK(x,x_i)) y clase por su signo.
- **Funcionamiento:** usa (K(x_i,x_{i'})=(1+\sum_j x_{ij}x_{i'j})^d).
- **Hiperparámetros relevantes:** `C`, `degree` y parámetros adicionales de la implementación del kernel.
- **Ventajas:** introduce no linealidad sin construir manualmente todas las potencias.
- **Limitaciones:** un grado alto aumenta flexibilidad y riesgo de sobreajuste.
- **Casos adecuados:** patrón no lineal compatible con una expansión polinómica de grado moderado.
- **No conviene:** cuando no hay evidencia de esa estructura o un modelo lineal valida igual de bien.

### 5.4 SVM con kernel radial

- **Objetivo:** construir una frontera no lineal basada en similitud local.
- **Entradas:** (X), (y), `C` y (\gamma>0).
- **Salida:** score no lineal y clase.
- **Funcionamiento:** (K(x_i,x_{i'})=\exp(-\gamma\sum_j(x_{ij}-x_{i'j})^2)); puntos lejanos tienen influencia prácticamente nula.
- **Hiperparámetros relevantes:** `C` y `gamma` deben ajustarse conjuntamente.
- **Ventajas:** espacio implícito de gran dimensión y fronteras flexibles sin expansión manual.
- **Limitaciones:** (\gamma) alto puede producir una frontera irregular y sobreajuste; la búsqueda aumenta el costo computacional.
- **Casos adecuados:** separación no lineal y local.
- **No conviene:** (p\gg n) cuando la flexibilidad lineal ya basta, o cuando el tiempo no permite validar ambos hiperparámetros.

### 5.5 Estrategias multiclase

- **OvO:** entrena un modelo por par de clases; cada modelo vota y gana la clase con más votos.
- **OvR:** entrena un modelo por clase contra todas las demás; gana la clase con score mayor.
- **Entrada/salida:** (X), etiquetas con (K>2) niveles; una etiqueta final entre las (K) clases.
- **Ventaja:** reutilizan el clasificador binario.
- **Limitación:** aumentan el número de modelos; OvO crece cuadráticamente con (K).

### 5.6 Support vector regression

- **Objetivo:** regresión con una banda de tolerancia alrededor de la función.
- **Funcionamiento:** solo los residuos con magnitud superior a una constante positiva contribuyen a la pérdida.
- **Alcance del capítulo:** presentación conceptual y mención de la implementación; no ofrece un procedimiento completo de selección ni evaluación.
- **Precaución:** no extrapolar detalles que el capítulo no desarrolla.

## 6. Fórmulas esenciales

### 6.1 Hiperplano y score

\[
f(x)=\beta_0+\sum_{j=1}^{p}\beta_jx_j.
\]

- (x_j): variable (j) de una observación;
- (\beta_0): intercepto;
- (\beta_j): coeficiente de la variable (j);
- (f(x)=0): frontera;
- (f(x)>0) y (f(x)<0): lados opuestos del hiperplano.

Con etiquetas (±1), una observación está correctamente clasificada cuando (y_if(x_i)>0).

### 6.2 Clasificador de margen máximo

\[
\max_{\beta_0,\ldots,\beta_p,M} M
\]

sujeto a

\[
\sum_{j=1}^{p}\beta_j^2=1,
\qquad
y_i f(x_i)\ge M,\quad i=1,\ldots,n.
\]

- (M): margen mínimo;
- la normalización de los coeficientes hace que (y_if(x_i)) sea una distancia perpendicular con signo;
- el supuesto esencial es que existe una separación lineal con (M>0).

### 6.3 Margen suave

\[
\max M
\]

sujeto a

\[
\sum_{j=1}^{p}\beta_j^2=1,
\qquad
y_if(x_i)\ge M(1-\epsilon_i),
\qquad
\epsilon_i\ge 0,
\qquad
\sum_{i=1}^{n}\epsilon_i\le C.
\]

- (\epsilon_i): holgura de la observación (i);
- (C): presupuesto total de violaciones en la notación teórica del capítulo;
- un (C) teórico mayor tolera más violaciones y permite un margen más ancho.

### 6.4 Representación por vectores de soporte

\[
f(x)=\beta_0+\sum_{i\in S}\alpha_iK(x,x_i).
\]

- (S): índices de los vectores de soporte;
- (\alpha_i\): pesos no nulos de esos puntos;
- (K(x,x_i)): similitud entre la observación nueva y el vector de soporte.

La representación muestra que los demás puntos no intervienen directamente en el score final.

### 6.5 Kernels

Lineal:

\[
K(x_i,x_{i'})=\sum_{j=1}^{p}x_{ij}x_{i'j}.
\]

Polinomial de grado (d\):

\[
K(x_i,x_{i'})=\left(1+\sum_{j=1}^{p}x_{ij}x_{i'j}\right)^d.
\]

Radial:

\[
K(x_i,x_{i'})=\exp\left[-\gamma\sum_{j=1}^{p}(x_{ij}-x_{i'j})^2\right].
\]

- (d): grado entero positivo;
- (\gamma): controla qué tan rápido cae la similitud con la distancia;
- la eficiencia proviene de calcular kernels entre pares sin construir explícitamente el espacio ampliado.

### 6.6 Pérdida hinge y penalización

\[
\min_{\beta_0,\ldots,\beta_p}
\left\{
\sum_{i=1}^{n}\max(0,1-y_if(x_i))
+\lambda\sum_{j=1}^{p}\beta_j^2
\right\}.
\]

- el primer término es la pérdida hinge;
- el segundo es una penalización tipo ridge;
- (\lambda\ge0) controla regularización;
- si (y_if(x_i)\ge1), la observación tiene pérdida hinge cero;
- (\lambda) grande implica coeficientes menores, más tolerancia a violaciones, mayor sesgo y menor varianza;
- en la parametrización teórica del capítulo, (\lambda) pequeño corresponde a (C) pequeño del presupuesto de holgura.

## 7. Implementación práctica

### 7.1 Baseline lineal con selección de (C)

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=0,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
linear_pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear"))

linear_search = GridSearchCV(
    linear_pipeline,
    {"svc__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    scoring="accuracy",
    cv=cv,
    refit=True,
)
linear_search.fit(X_train, y_train)
y_pred = linear_search.predict(X_test)
```

**Adaptación moderna y práctica.** El capítulo usa `KFold`; aquí se usa `StratifiedKFold` para preservar proporciones de clase en clasificación. Se añade `StandardScaler` dentro de un `Pipeline` para que el escalado se ajuste solo con cada partición de entrenamiento y no produzca leakage. Estas dos decisiones no se presentan como instrucciones del autor.

### 7.2 SVM radial

```python
rbf_pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

rbf_search = GridSearchCV(
    rbf_pipeline,
    {
        "svc__C": [0.1, 1, 10, 100, 1000],
        "svc__gamma": [0.5, 1, 2, 3, 4],
    },
    scoring="accuracy",
    cv=cv,
    refit=True,
)
rbf_search.fit(X_train, y_train)
test_scores = rbf_search.decision_function(X_test)
```

La búsqueda conjunta replica la idea del laboratorio. Los valores concretos son candidatos, no recomendaciones universales.

### 7.3 Evaluación mínima

```python
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

ConfusionMatrixDisplay.from_estimator(rbf_search, X_test, y_test)
RocCurveDisplay.from_estimator(rbf_search, X_test, y_test)
```

La matriz de confusión permite localizar tipos de error. Para clasificación binaria, la curva ROC recorre umbrales del score. El capítulo insiste en que la evaluación relevante debe hacerse fuera del entrenamiento.

### 7.4 Multiclase

```python
multiclass_svm = SVC(
    kernel="rbf",
    C=10,
    gamma=1,
    decision_function_shape="ovo",
    random_state=0,
)
multiclass_svm.fit(X_train, y_train)
```

`random_state` no cambia el problema de optimización determinista habitual, pero deja explícita la reproducibilidad cuando la implementación activa procedimientos internos aleatorios.

## 8. Supuestos y condiciones de aplicación

1. Las observaciones se representan en un espacio numérico donde un hiperplano o una similitud por kernel son significativos.
2. La formulación básica usa etiquetas binarias; multiclase requiere OvO u OvR.
3. El clasificador de margen máximo exige separabilidad lineal estricta.
4. El clasificador de vectores de soporte admite solapamiento mediante holguras.
5. Un kernel lineal presupone que una frontera lineal en las variables disponibles es adecuada.
6. Un kernel polinomial presupone que una expansión polinómica de grado moderado puede capturar la frontera.
7. El kernel RBF usa distancia euclídea y asigna influencia local; la escala de las variables afecta esa distancia. **Conclusión práctica derivada:** las escalas deben tratarse de manera coherente y sin usar información de validación/test.
8. Los hiperparámetros no son accesorios: controlan el compromiso sesgo-varianza.
9. La métrica de selección debe corresponder al objetivo. El laboratorio usa accuracy y también ROC para comparar scores a múltiples umbrales.
10. Los resultados de entrenamiento no estiman por sí solos el desempeño futuro.

## 9. Errores y riesgos

- **Confundir las dos convenciones de (C).** El (C) del presupuesto de holgura de las ecuaciones crece con la tolerancia; el `C` de `SVC` crece con la penalización de errores.
- **Elegir por error de entrenamiento.** Un (\gamma) alto o `C` grande puede mejorar entrenamiento y empeorar test.
- **Asumir que separabilidad implica buen modelo.** Con (p\gg n) es fácil obtener separación perfecta; el ejemplo de expresión génica muestra por qué debe mirarse test.
- **Usar kernel no lineal sin necesidad.** Añade flexibilidad, búsqueda y riesgo de sobreajuste.
- **No ajustar (C).** El capítulo recalca que determina underfitting y overfitting.
- **Afinar (C) y (\gamma) mirando test.** El test debe reservarse para evaluación final; la selección corresponde a validación cruzada sobre entrenamiento.
- **Reportar solo accuracy.** Puede ocultar qué clases se confunden; la matriz de confusión y, en binario, ROC aportan información adicional.
- **Interpretar |(f(x))| como probabilidad.** El capítulo lo trata como score/distancia respecto de la frontera, no como probabilidad calibrada.
- **Ignorar costo computacional.** El entrenamiento con kernels requiere similitudes entre pares; una cuadrícula de hiperparámetros multiplica ese costo.
- **Aplicar OvO sin considerar (K).** Requiere \(\binom{K}{2}\) clasificadores y crece rápidamente con el número de clases.
- **Data leakage en transformaciones.** **Riesgo práctico derivado:** escalar o transformar antes de dividir o fuera de los folds filtra información; usar pipeline.
- **Tomar una inconsistencia impresa como resultado fiable.** En la p. 391, la salida mostrada de `grid.best_params_` es `{'C': 100, 'gamma': 1}`, mientras el texto inmediato afirma (C=1), (\gamma=0.5). El resumen conserva el principio (seleccionar ambos por CV), no uno de esos valores contradictorios.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** clasificación con variables numéricas, especialmente si se sospecha una frontera lineal clara, no linealidad local o alta dimensionalidad.
- **Decisiones que permite tomar:** kernel inicial; rango de (C); necesidad de buscar (\gamma) o `degree`; uso de OvO/OvR; comparación mediante validación y test.
- **Baseline:** `StandardScaler` + `SVC(kernel="linear")`, con una cuadrícula corta de `C` y validación estratificada.
- **Mejora del modelo:** probar RBF y ajustar conjuntamente `C`/`gamma` solo si el baseline lineal deja evidencia de underfitting o si la estructura es no lineal.
- **Qué no implementar bajo presión:** expansiones polinómicas manuales grandes, cuadrículas densas, visualizaciones elaboradas en (p>2) o múltiples kernels si el baseline ya cumple.
- **Señales que activan las técnicas:**
  - (p\gg n): probar lineal primero;
  - error alto tanto en entrenamiento como validación con lineal: considerar kernel no lineal;
  - entrenamiento casi perfecto y validación peor: reducir flexibilidad, bajar `C` de `SVC` o bajar `gamma`;
  - frontera curva visible con pocas variables: RBF o polinomial;
  - más de dos clases: documentar OvO/OvR;
  - umbral de decisión relevante: trabajar con `decision_function` y ROC.
- **Recomendación para competencias:** fijar un presupuesto pequeño de búsqueda, comparar media de CV y rendimiento por fold, y conservar el test interno intacto. Esta recomendación operacionaliza el énfasis del capítulo en validación fuera de entrenamiento.

## 11. Checklist operativo

- [ ] Identificar si el objetivo es clasificación binaria, multiclase o regresión.
- [ ] Separar entrenamiento y test antes de ajustar cualquier transformación.
- [ ] Usar validación estratificada y reproducible para clasificación.
- [ ] Probar kernel lineal como baseline, especialmente si (p\gg n).
- [ ] Ajustar `C` mediante validación cruzada.
- [ ] Si se usa RBF, ajustar `C` y `gamma` conjuntamente.
- [ ] Si se usa kernel polinomial, validar `degree` además de `C`.
- [ ] Mantener transformaciones dentro de un pipeline para evitar leakage.
- [ ] Revisar matriz de confusión y no solo accuracy.
- [ ] Evaluar ROC con scores fuera de entrenamiento cuando el problema sea binario.
- [ ] Comparar brecha entre entrenamiento y validación/test.
- [ ] Registrar estrategia OvO/OvR en multiclase.
- [ ] Verificar la convención de `C` usada por la biblioteca.
- [ ] No interpretar el score como probabilidad calibrada.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Hiperplanos, margen máximo, margen suave y vectores de soporte | `knowledge/models/support_vector_machines.md` | Alta |
| Kernels lineal, polinomial y RBF | `knowledge/models/kernel_methods.md` | Alta |
| Efecto y convención de `C`, `gamma` y `degree` | `knowledge/tuning/svm_hyperparameters.md` | Alta |
| Búsqueda con validación cruzada | `knowledge/validation/model_selection.md` | Alta |
| Evaluación con matriz de confusión, score y ROC | `knowledge/metrics/classification_metrics.md` | Alta |
| Detección de overfitting por brecha train-test | `knowledge/validation/overfitting_diagnostics.md` | Alta |
| Pipeline de escalado sin leakage | `knowledge/preprocessing/scaling.md` | Media |
| One-versus-one y one-versus-rest | `knowledge/models/multiclass_strategies.md` | Media |
| Relación entre hinge loss, regularización ridge y regresión logística | `knowledge/models/classification_losses.md` | Media |
| Support vector regression | `knowledge/models/support_vector_regression.md` | Baja |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y panorama de la familia SVM | Cap. 9; §§9.1-9.6, pp. 368-395 |
| Hiperplano y clasificación por signo | §§9.1.1-9.1.2, pp. 368-370 |
| Margen máximo y vectores de soporte | §§9.1.3-9.1.4, pp. 370-372 |
| Caso no separable | §9.1.5, pp. 372-373 |
| Margen suave, holguras y papel del (C) teórico | §§9.2.1-9.2.2, pp. 373-377 |
| Expansión no lineal del espacio de variables | §9.3.1, pp. 378-379 |
| Representación por productos internos y kernels | §9.3.2, pp. 379-382 |
| Kernels lineal, polinomial y radial | §9.3.2, pp. 380-382 |
| Ejemplo de sobreajuste con ROC train/test | §9.3.3, pp. 382-383 |
| OvO y OvR | §§9.4.1-9.4.2, pp. 383-384 |
| Hinge loss, penalización y relación con regresión logística | §9.5, pp. 384-386 |
| Support vector regression | §9.5, p. 386; §9.6.4, p. 394 |
| Implementación lineal y ajuste de `C` | §9.6.1, pp. 387-390 |
| Implementación RBF y ajuste de `C`/`gamma` | §9.6.2, pp. 390-392 |
| Scores y curvas ROC | §9.6.3, pp. 392-393 |
| Implementación multiclase | §9.6.4, pp. 393-394 |
| Caso (p\gg n) con datos de expresión génica | §9.6.5, pp. 394-395 |
| Señales prácticas usadas en el checklist | Laboratorio §9.6 y ejercicios aplicados §9.7, pp. 387-398 |

**Cobertura de lectura.** Se revisaron las 31 páginas del PDF suministrado, correspondientes a las páginas impresas 368-398. Las páginas 368-395 contienen exposición y laboratorio; las páginas 395-398 contienen ejercicios. No hubo páginas ilegibles. La extracción automática distorsionó algunos símbolos, que se contrastaron visualmente en las páginas con formulaciones principales y código.
