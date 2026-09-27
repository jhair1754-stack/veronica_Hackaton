---
title: "Representing Data and Engineering Features"
topics:
  - "variables categóricas y one-hot encoding"
  - "discretización, interacciones y características polinómicas"
  - "transformaciones no lineales univariantes"
  - "selección automática de características"
  - "ingeniería de características con conocimiento experto"
target_knowledge_files:
  - "knowledge/preprocessing/categorical_encoding.md"
  - "knowledge/feature_engineering/numerical_transformations.md"
  - "knowledge/feature_engineering/interactions_and_polynomials.md"
  - "knowledge/feature_engineering/feature_selection.md"
  - "knowledge/time_series/calendar_features.md"
---

# 4. Representing Data and Engineering Features

## 1. Propósito del capítulo

El capítulo explica cómo convertir datos en una representación compatible con el modelo y suficientemente informativa para la tarea. Se concentra en variables categóricas, discretización de variables continuas, interacciones y polinomios, transformaciones univariantes, selección de características y uso de conocimiento experto. Su tesis práctica es que la correspondencia entre representación y algoritmo puede influir más en el rendimiento que un ajuste fino de hiperparámetros.

## 2. Conceptos esenciales

### Contenido explicado explícitamente por los autores

- Una variable categórica representa estados cualitativos de un conjunto finito. Que esté almacenada como entero no la vuelve continua: si sus valores no tienen orden cuantitativo, debe tratarse como discreta.
- El one-hot encoding sustituye una variable de `k` categorías por indicadores binarios, uno por categoría. El capítulo conserva las `k` columnas; señala que la codificación dummy estadística suele usar `k - 1` para evitar una matriz de rango deficiente.
- Antes de codificar texto hay que inspeccionar categorías y frecuencias. Diferencias de ortografía, mayúsculas o sinónimos pueden crear categorías artificialmente distintas.
- Las columnas de train y test deben tener idéntico significado y orden. Codificarlas por separado puede producir distinto número de columnas o, peor, igual forma con semántica diferente.
- La variable objetivo y cualquier derivación de ella deben excluirse de `X`. Incluirlas constituye fuga de información (data leakage).
- La representación óptima depende tanto de la semántica de los datos como del modelo. Los modelos lineales ganan expresividad con bins, interacciones y polinomios; los árboles suelen aprender cortes e interacciones sin expansión explícita.
- Discretizar convierte una variable continua en pertenencia a intervalos. Tras one-hot encoding, un modelo lineal puede aprender un valor distinto por intervalo.
- Añadir la variable original a sus bins produce offsets por intervalo y una pendiente global. Multiplicar la variable por cada indicador de bin permite una pendiente distinta en cada intervalo.
- `PolynomialFeatures` genera potencias e interacciones. Puede mejorar modelos lineales, pero expande rápidamente la dimensión y los grados altos se comportan de forma extrema en fronteras o regiones con pocos datos.
- Transformaciones como `log`, `exp`, `sin` y `cos` pueden adecuar escala, distribución o periodicidad. `log(x + 1)` es útil para conteos no negativos y admite ceros.
- La selección automática de características es supervisada: usa `y` y, por tanto, debe ajustarse exclusivamente con entrenamiento.
- La selección univariante evalúa cada característica de forma aislada; es rápida, pero descarta variables útiles solo en combinación. La selección basada en modelos considera variables conjuntamente si el estimador puede capturar interacciones. La selección iterativa, como RFE, reentrena múltiples modelos y es más costosa.
- El conocimiento del dominio puede convertir una representación pobre en una tarea aprendible. En datos temporales, componentes como hora, día de semana o proximidad a festivos pueden ser más informativos que un timestamp absoluto.
- Los árboles no extrapolan fuera del rango observado. En el ejemplo temporal, un random forest con tiempo POSIX memoriza bien train, pero predice un valor casi constante en fechas futuras.

### Conclusiones prácticas derivadas del contenido

- Tratar tipo de almacenamiento y significado como cuestiones separadas: un código postal, clase de producto o estado laboral numérico suele requerir codificación categórica.
- Encapsular transformaciones supervisadas y no supervisadas en un pipeline evita inconsistencias y leakage durante validación.
- Antes de ampliar características, establecer un baseline con la representación original. La expansión solo se conserva si mejora validación fuera de muestra y su coste es aceptable.
- En variables temporales, dividir cronológicamente y crear características calendario evita confundir interpolación dentro de un rango con predicción del futuro.
- La selección de características es principalmente una herramienta de coste, generalización e interpretabilidad; el capítulo advierte que en datos reales no suele producir grandes ganancias de rendimiento.

## 3. Flujo o procedimiento presentado

1. Separar la variable objetivo de las entradas e identificar el tipo semántico de cada columna.
2. Inspeccionar categorías y frecuencias; unificar errores de escritura, capitalización y valores equivalentes.
3. Separar entrenamiento y validación de forma compatible con la tarea; usar corte temporal cuando se predice el futuro.
4. Ajustar codificadores, transformadores y selectores únicamente con entrenamiento.
5. Representar variables categóricas nominales con one-hot encoding y aplicar al resto de particiones el mismo esquema de columnas.
6. Establecer un baseline con una representación simple y un modelo apropiado.
7. Si un modelo lineal no captura no linealidad, probar discretización, variable original más bins, interacciones o polinomios de grado bajo.
8. Si hay conteos sesgados o magnitudes con cola larga, evaluar `log1p`; si hay ciclos, derivar variables periódicas o calendario.
9. Comparar la expansión con un modelo no lineal capaz de aprender relaciones complejas sin crear todas las columnas.
10. Si la dimensionalidad es excesiva, ajustar selección univariante, basada en modelo o RFE solo en train; transformar validación con el selector ya ajustado.
11. Medir rendimiento fuera de muestra, número de variables, tiempo, estabilidad e interpretabilidad antes de conservar la representación.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Variable nominal, aunque esté codificada con enteros | One-hot encoding | Evita imponer distancia u orden inexistentes | Inspeccionar categorías y mantener el mismo encoder entre particiones |
| Variable ordinal, como una valoración por estrellas | Comparar representación ordinal y categórica | La mejor representación depende de tarea, datos y algoritmo | No asumir automáticamente continuidad ni nominalidad |
| Relación no lineal con un modelo lineal | Discretización one-hot | Permite un nivel distinto por intervalo | El número y límites de bins controlan flexibilidad; no suele ayudar a árboles |
| Se requieren tendencias diferentes por región | Interacción entre bin y variable original | Aprende una pendiente por intervalo | Aumenta dimensiones y riesgo de sobreajuste |
| Modelo lineal con relaciones suaves e interacciones plausibles | Polinomios/interacciones de grado bajo | Expande la base donde el modelo sigue siendo lineal | Grados altos explotan en magnitud y en bordes; escalar y validar |
| Árboles o random forest ya capturan no linealidad | Usar variables originales como baseline | Los árboles aprenden cortes e interacciones | La expansión explícita puede no ayudar e incluso empeorar |
| Conteos no negativos muy sesgados | `log(x + 1)` | Reduce asimetría y el efecto de valores extremos | Aplicar solo a variables compatibles; interpretar en la escala transformada |
| Conteo como objetivo de regresión | Evaluar `log(y + 1)` | Puede facilitar el ajuste de relaciones multiplicativas | Revertir la transformación para interpretar; el capítulo la presenta como aproximación burda a regresión de Poisson |
| Muchas variables y muchas parecen irrelevantes | Selección univariante | Es rápida y no exige entrenar un modelo complejo | No detecta utilidad conjunta ni interacciones |
| Existe un estimador con importancia o coeficientes | `SelectFromModel` | Considera todas las variables a la vez | La selección hereda los sesgos y límites del estimador |
| Se busca un subconjunto preciso y el coste es tolerable | RFE | Elimina iterativamente la característica menos importante | Reentrena muchas veces y puede ser demasiado lento |
| Predicción futura con fecha/hora | Corte cronológico + componentes calendario | Expone periodicidad reutilizable en fechas futuras | Un timestamp con árboles no permite extrapolar fuera de train |
| Efecto conjunto de día y hora con modelo lineal | One-hot de ambos + interacción | Aprende un coeficiente por combinación | La cardinalidad crece; validar que cada combinación tenga soporte |

## 5. Métodos y técnicas

### One-hot encoding

- **Objetivo:** traducir categorías nominales a entradas numéricas sin imponer un orden.
- **Entradas:** una o más columnas categóricas con vocabulario observado en entrenamiento.
- **Salida:** matriz con indicadores binarios por categoría; las variables continuas pueden conservarse.
- **Funcionamiento:** para cada observación se activa la columna asociada con su categoría.
- **Hiperparámetros relevantes:** categorías que se codifican; conservación o descarte de una categoría; tratamiento de categorías desconocidas.
- **Ventajas:** compatible con modelos lineales y con estimadores de scikit-learn; semántica explícita.
- **Limitaciones:** aumenta dimensión con cardinalidad; categorías sucias crean columnas distintas; una categoría nueva exige una política definida.
- **Casos adecuados:** marcas, ocupaciones, género, departamento o códigos nominales.
- **No conviene:** números realmente continuos; cardinalidad extrema sin evaluar coste; variables ordinales cuando perder el orden perjudique.

### Discretización o binning

- **Objetivo:** representar una variable continua mediante intervalos.
- **Entradas:** variable numérica y límites o número de bins.
- **Salida:** pertenencia discreta y, normalmente, su one-hot encoding.
- **Funcionamiento:** asigna cada valor a un intervalo; un modelo posterior aprende un efecto por bin.
- **Hiperparámetros relevantes:** número de bins y estrategia de límites.
- **Ventajas:** introduce no linealidad controlada en modelos lineales; genera una función por tramos interpretable.
- **Limitaciones:** pierde variación dentro de cada bin; muchos bins elevan varianza; pocos producen underfitting.
- **Casos adecuados:** datasets grandes o de alta dimensión donde se desea conservar un modelo lineal, pero ciertas relaciones son no lineales.
- **No conviene:** árboles, que ya aprenden cortes; variables con pocos datos por intervalo.

### Interacciones y características polinómicas

- **Objetivo:** hacer visibles a un modelo lineal efectos conjuntos y curvatura.
- **Entradas:** variables numéricas o indicadores; grado máximo y opción de solo interacciones.
- **Salida:** potencias (x_j^p) y productos entre variables.
- **Funcionamiento:** amplía la base; los coeficientes lineales operan sobre funciones no lineales de la entrada original.
- **Hiperparámetros relevantes:** `degree`, `interaction_only`, `include_bias`.
- **Ventajas:** una regresión lineal puede representar curvas suaves y dependencias conjuntas; en el ejemplo de vivienda, Ridge mejoró con interacciones de grado 2.
- **Limitaciones:** crecimiento combinatorio, escalas extremas, multicolinealidad y sobreajuste; grados altos son inestables en bordes.
- **Casos adecuados:** Ridge u otros modelos lineales con hipótesis de interacción o curvatura.
- **No conviene:** expansión indiscriminada de muchas variables; modelos que ya capturan interacciones, como random forest, sin evidencia de validación.

### Transformaciones no lineales univariantes

- **Objetivo:** adecuar distribución y escala de una característica a modelos sensibles a ellas.
- **Entradas:** variables continuas o conteos; función compatible con su dominio.
- **Salida:** misma cantidad de columnas con valores transformados.
- **Funcionamiento:** `log1p` comprime colas derechas; `exp` cambia escalas relativas; `sin` y `cos` expresan periodicidad.
- **Hiperparámetros relevantes:** elección de función y columnas; desplazamiento necesario para respetar el dominio.
- **Ventajas:** operación simple que puede mejorar de forma marcada modelos lineales o redes neuronales.
- **Limitaciones:** no hay una transformación universal; distintas columnas pueden necesitar tratamientos distintos; suele ser irrelevante para árboles porque preserva el orden.
- **Casos adecuados:** conteos con muchos valores pequeños y pocos muy grandes; relaciones multiplicativas; ciclos.
- **No conviene:** aplicar log a negativos o transformar todo el dataset sin inspección.

### Selección univariante

- **Objetivo:** conservar variables con mayor relación individual con `y`.
- **Entradas:** `X_train`, `y_train`, prueba estadística y regla de corte.
- **Salida:** máscara y matriz reducida.
- **Funcionamiento:** calcula un estadístico por columna; `SelectKBest` mantiene `k` y `SelectPercentile` un porcentaje. El capítulo menciona `f_classif` para clasificación y `f_regression` para regresión.
- **Hiperparámetros relevantes:** prueba, `k` o `percentile`.
- **Ventajas:** muy rápida; útil cuando entrenar con todas las variables es inviable o muchas son ruido.
- **Limitaciones:** ignora dependencias conjuntas y es independiente del modelo final.
- **Casos adecuados:** filtrado inicial de datasets de muy alta dimensión.
- **No conviene:** cuando la señal surge principalmente de interacciones.

### Selección basada en modelo

- **Objetivo:** seleccionar variables según importancias aprendidas conjuntamente.
- **Entradas:** estimador supervisado con `feature_importances_` o coeficientes y un umbral.
- **Salida:** máscara y subconjunto transformado.
- **Funcionamiento:** `SelectFromModel` conserva importancias superiores al umbral; árboles aportan importancias y modelos lineales pueden aportar coeficientes absolutos. Una penalización L1 induce coeficientes dispersos.
- **Hiperparámetros relevantes:** estimador, su regularización y `threshold`.
- **Ventajas:** puede considerar interacciones si el estimador también puede; el modelo de selección no tiene que ser el modelo final.
- **Limitaciones:** depende de una importancia fiable; un umbral forzado puede conservar ruido o excluir señal.
- **Casos adecuados:** variables numerosas con un estimador razonable para estimar importancia.
- **No conviene:** importancias inestables o estimador de selección mal alineado con la estructura del problema.

### Recursive Feature Elimination (RFE)

- **Objetivo:** llegar iterativamente a un número prefijado de variables.
- **Entradas:** estimador con medida de importancia y cantidad final.
- **Salida:** ranking, máscara y subconjunto transformado.
- **Funcionamiento:** ajusta con todas las variables, elimina la menos importante y repite.
- **Hiperparámetros relevantes:** `n_features_to_select`, estimador y sus parámetros.
- **Ventajas:** reevalúa importancias después de cada eliminación.
- **Limitaciones:** coste elevado por múltiples entrenamientos; no garantiza mejora grande fuera de muestra.
- **Casos adecuados:** dimensionalidad moderada y necesidad de un subconjunto pequeño o interpretable.
- **No conviene:** coding tests con poco tiempo, datos enormes o estimadores costosos.

### Características temporales basadas en conocimiento experto

- **Objetivo:** convertir un timestamp en factores con significado predictivo y recurrente.
- **Entradas:** fecha/hora y conocimiento del dominio.
- **Salida:** hora, día de semana, indicadores de festivo u otras variables derivadas.
- **Funcionamiento:** reemplaza o complementa el tiempo absoluto con componentes que se repiten y que el modelo puede compartir entre periodos.
- **Hiperparámetros relevantes:** granularidad temporal y conjunto de interacciones.
- **Ventajas:** permite generalizar patrones periódicos y puede hacer interpretable un modelo lineal.
- **Limitaciones:** requiere conocimiento y cobertura suficientes por combinación; no captura por sí sola tendencia, shocks o cambios de régimen.
- **Casos adecuados:** demanda por hora/día, estacionalidad semanal y efectos de calendario.
- **No conviene:** usar únicamente un entero POSIX con árboles para extrapolar al futuro.

## 6. Fórmulas esenciales

### Modelo lineal sobre características transformadas

\[
\hat{y}=b+\sum_{j=1}^{p} w_j\,\phi_j(x)
\]

- (x): observación original; \(\phi_j(x)\): característica original, indicador, potencia o interacción; \(w_j\): coeficiente aprendido; \(b\): intercepto; \(\hat{y}\): predicción o score antes del umbral en clasificación.
- Mide una combinación lineal en la representación transformada. Aunque el estimador sea lineal en \(\phi\), puede ser no lineal respecto de (x).
- Supuesto importante: las transformaciones elegidas deben exponer la estructura que se desea aprender.

### Indicador one-hot

\[
z_c(x)=\begin{cases}
1, & x=c\\
0, & x\neq c
\end{cases}
\]

- (c): categoría; (x): valor observado; (z_c): columna indicadora.
- Cada coeficiente asociado representa el efecto de esa categoría dentro del modelo.
- Para una categoría nominal válida, exactamente un indicador del grupo suele estar activo.

### Interacción entre características

\[
z_{jk}=x_jx_k
\]

- (x_j, x_k): características originales; (z_{jk}): interacción.
- Permite que el efecto conjunto no sea solo la suma de efectos individuales.
- Si (x_k) es indicador de bin, el producto conserva (x_j) dentro de ese bin y vale cero fuera, produciendo una pendiente específica.

### Expansión polinómica univariante

\[
\phi_d(x)=[x,x^2,\ldots,x^d]
\]

- (d): grado máximo; cada potencia añade una forma de curvatura.
- Los valores pueden crecer muy rápido cuando \(|x|>1\); el escalado y la regularización son importantes.
- Un grado mayor incrementa flexibilidad y riesgo de comportamiento extremo donde hay pocos datos.

### Transformación logarítmica de conteos

\[
x'=\log(1+x)
\]

- (x\ge 0): conteo original; (x'): valor comprimido.
- Reduce la asimetría y la influencia relativa de conteos grandes; el `+1` hace que cero sea válido.
- No convierte automáticamente la variable en gaussiana ni reemplaza la validación.

## 7. Implementación práctica

Los fragmentos siguientes conservan la idea del capítulo con interfaces actuales. **Nota práctica - adaptación moderna:** en lugar de ejecutar `get_dummies` sobre train y test juntos, se ajusta `OneHotEncoder` solo con entrenamiento y se usa `handle_unknown="ignore"`. Esto mantiene el esquema sin incorporar información del conjunto de evaluación. También se reemplazan APIs retiradas como `.ix`, `sparse=False` y `get_feature_names`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

X = data.drop(columns="income")
y = (data["income"] == ">50K").astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

categorical = ["workclass", "education", "gender", "occupation"]
numeric = ["age", "hours-per-week"]

preprocess = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ]
)
model = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
valid_accuracy = model.score(X_valid, y_valid)
```

**Nota práctica - adaptación moderna:** `KBinsDiscretizer` integra aprendizaje de límites y codificación dentro del pipeline, evitando calcular bins con validación.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

features = ColumnTransformer(
    [
        (
            "binned",
            KBinsDiscretizer(n_bins=10, encode="onehot", strategy="uniform"),
            ["continuous_feature"],
        )
    ]
)
model = make_pipeline(features, StandardScaler(with_mean=False), Ridge())
model.fit(X_train, y_train)
```

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
valid_r2 = model.score(X_valid, y_valid)
```

```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

X_train_log = np.log1p(X_train_counts)
X_valid_log = np.log1p(X_valid_counts)

model = TransformedTargetRegressor(
    regressor=Ridge(), func=np.log1p, inverse_func=np.expm1
)
model.fit(X_train_log, y_train_counts)
y_pred = model.predict(X_valid_log)
```

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear", random_state=42),
        threshold="median",
    ),
    LogisticRegression(max_iter=1000, random_state=42),
)
model.fit(X_train, y_train)
valid_accuracy = model.score(X_valid, y_valid)
```

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# Split cronológico: todo train ocurre antes que validación.
train_mask = timestamps < cutoff
calendar = timestamps.to_frame(index=False)
calendar["dayofweek"] = timestamps.dayofweek
calendar["hour"] = timestamps.hour

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_calendar = encoder.fit_transform(calendar.loc[train_mask, ["dayofweek", "hour"]])
X_valid_calendar = encoder.transform(calendar.loc[~train_mask, ["dayofweek", "hour"]])

interactions = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False
)
X_train_interactions = interactions.fit_transform(X_train_calendar)
X_valid_interactions = interactions.transform(X_valid_calendar)

model = Ridge().fit(X_train_interactions, y[train_mask])
valid_r2 = model.score(X_valid_interactions, y[~train_mask])
```

## 8. Supuestos y condiciones de aplicación

1. La semántica de cada variable es conocida; el dtype por sí solo no distingue categoría, ordinal y continuo.
2. Las categorías observadas en producción deben mapearse al mismo espacio aprendido en entrenamiento; las desconocidas requieren una política explícita.
3. La distribución de categorías y bins de entrenamiento ofrece soporte suficiente para estimar sus efectos.
4. Discretización supone que perder detalle dentro del intervalo es aceptable y que los límites son estables fuera de muestra.
5. Polinomios e interacciones presuponen que la relación puede expresarse razonablemente mediante esos productos y que hay datos para estimar los coeficientes.
6. Transformaciones logarítmicas requieren no negatividad después del desplazamiento y son especialmente pertinentes para colas derechas o conteos.
7. La selección univariante presupone que la señal útil aparece en una relación individual con el target.
8. La selección basada en modelo y RFE requieren una medida de importancia del estimador y heredan sus supuestos.
9. Todo selector supervisado se ajusta solo con entrenamiento; de lo contrario usa el target de validación de forma indirecta.
10. Los modelos de árboles no extrapolan más allá del rango visto; un timestamp creciente no codifica por sí mismo periodicidad reutilizable.
11. En series temporales, entrenamiento debe preceder a validación y las variables calendario disponibles deben poder calcularse en el horizonte de inferencia.
12. Crear una variable de dominio no obliga al modelo a usarla, pero sí aumenta complejidad potencial y debe validarse.

## 9. Errores y riesgos

- Tratar cualquier entero como continuo e imponer una relación ordinal inexistente.
- Codificar train y test por separado, creando columnas incompatibles o semánticas desplazadas.
- Incluir en `X` la variable objetivo o una columna derivada de ella.
- Seguir literalmente el ejemplo histórico de unir train y test para `get_dummies`; una implementación moderna debe aprender el vocabulario en train y definir el tratamiento de categorías nuevas.
- No limpiar espacios, mayúsculas, errores de escritura o marcadores como `?` antes de codificar.
- Generar dummies de una columna objetivo junto con las entradas y seleccionarla accidentalmente como predictor.
- Ajustar límites de bins, transformaciones o selección con todo el dataset antes de validar.
- Usar demasiados bins: pocas observaciones por intervalo, alta varianza y sobreajuste. Usar muy pocos: pérdida de señal y underfitting.
- Aplicar binning o expansión polinómica a árboles esperando la misma mejora que en modelos lineales.
- Expandir todas las variables a grado alto: crecimiento combinatorio, memoria elevada, escalas enormes e inestabilidad en bordes.
- Interpretar una mejora en train como evidencia suficiente; las características adicionales aumentan complejidad.
- Aplicar `log` directamente cuando hay ceros o negativos; para conteos no negativos el capítulo usa `log(x + 1)`.
- Transformar indiscriminadamente todas las columnas aunque solo algunas estén sesgadas.
- Seleccionar variables antes del split o fuera del ciclo de validación: el selector ve `y_valid` y produce leakage.
- Esperar que la selección univariante conserve variables útiles solo mediante interacción.
- Interpretar importancia de un selector como causalidad o estabilidad garantizada.
- Usar RFE bajo presión sin estimar su coste: el ejemplo reentrena el random forest una vez por variable eliminada.
- Dividir aleatoriamente una tarea de futuro y obtener una evaluación optimista.
- Pasar tiempo POSIX a random forest y esperar extrapolación; fuera del rango, el árbol reutiliza predicciones terminales.
- Codificar hora y día como enteros en regresión lineal, imponiendo efectos monotónicos y una distancia artificial entre extremos del ciclo.
- Incorporar características conocidas solo después del evento, creando leakage temporal.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo cuando hay columnas `object`/`category`, códigos enteros ambiguos, conteos sesgados, timestamps, muchas variables o un baseline lineal claramente incapaz de capturar curvatura.
- Baseline tabular: identificar target, dividir datos, `ColumnTransformer` con one-hot para nominales y passthrough para numéricas, seguido de un modelo regularizado.
- Baseline temporal: ordenar por tiempo, corte cronológico y variables simples de hora/día de semana; no usar split aleatorio ni confiar en timestamp absoluto con árboles.
- Mejora rápida para modelos lineales: `log1p` en conteos sesgados, bins para una relación no lineal evidente y polinomios/interacciones de grado 2 en un subconjunto pequeño.
- Mejora rápida en alta dimensión: `SelectPercentile` o `SelectFromModel` dentro del pipeline, midiendo si reduce tiempo sin degradar validación.
- Señales activadoras: pocos valores enteros sin orden -> categórica; histogramas con cola derecha -> `log1p`; residuales curvos -> bins/polinomios; señal dependiente de pares -> interacciones; muchas columnas casi inútiles -> selección; ciclos diarios/semanales -> calendario e interacciones.
- Comparar siempre contra un árbol o ensemble cuando la expansión de un modelo lineal se vuelve grande: puede capturar la no linealidad con menos ingeniería explícita.
- No implementar bajo presión polinomios de grado alto, RFE con estimador costoso, una búsqueda extensa de bins o calendarios complejos sin evidencia del EDA.
- Conservar una transformación solo si mejora validación y su coste de memoria, latencia e interpretación es aceptable.

## 11. Checklist operativo

- [ ] Separar target y descartar columnas que lo revelen directa o indirectamente.
- [ ] Clasificar variables por significado: continua, ordinal, nominal, conteo o temporal.
- [ ] Inspeccionar valores únicos, frecuencias, espacios, mayúsculas y marcadores faltantes.
- [ ] Elegir split estratificado o cronológico según la tarea antes de ajustar transformaciones.
- [ ] Ajustar encoder, bins, polinomios y selector solo con entrenamiento.
- [ ] Garantizar el mismo espacio de columnas y una política para categorías desconocidas.
- [ ] Medir un baseline simple antes de expandir características.
- [ ] Usar `log1p` solo en columnas no negativas compatibles.
- [ ] Limitar bins, grado e interacciones para controlar dimensionalidad.
- [ ] Fijar `random_state` cuando el estimador sea estocástico.
- [ ] Comparar representación expandida contra un modelo no lineal.
- [ ] Registrar número de características, score, tiempo y memoria.
- [ ] En tiempo, comprobar que ninguna característica use información posterior al instante predicho.
- [ ] Validar que la mejora persista fuera de muestra antes de conservarla.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Tipos semánticos frente a dtypes | `knowledge/preprocessing/feature_types.md` | Alta |
| Limpieza y auditoría de categorías | `knowledge/eda/categorical_audit.md` | Alta |
| One-hot encoding y categorías desconocidas | `knowledge/preprocessing/categorical_encoding.md` | Alta |
| Prevención de target leakage en representación | `knowledge/validation/data_leakage.md` | Alta |
| Discretización para modelos lineales | `knowledge/feature_engineering/discretization.md` | Media |
| Interacciones y polinomios | `knowledge/feature_engineering/interactions_and_polynomials.md` | Alta |
| Transformaciones `log1p`, exponencial y periódicas | `knowledge/feature_engineering/numerical_transformations.md` | Alta |
| Selección univariante | `knowledge/feature_engineering/univariate_feature_selection.md` | Media |
| Selección basada en modelo y L1 | `knowledge/feature_engineering/model_based_feature_selection.md` | Alta |
| RFE y coste de selección iterativa | `knowledge/feature_engineering/recursive_feature_elimination.md` | Media |
| Relación entre representación y familia de modelos | `knowledge/models/model_representation_fit.md` | Alta |
| Cortes temporales y límites de extrapolación de árboles | `knowledge/validation/time_based_validation.md` | Alta |
| Hora, día de semana, festivos e interacciones | `knowledge/time_series/calendar_features.md` | Alta |
| Pipelines seguros de ingeniería de características | `knowledge/preprocessing/feature_pipeline.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y efecto de la representación | Apertura del capítulo, pp. 211–212 |
| Variables categóricas y dataset Adult | “Categorical Variables”, pp. 212–213 |
| Definición y variantes de one-hot encoding | “One-Hot-Encoding (Dummy Variables)”, pp. 213–214 |
| Auditoría de categorías y `get_dummies` | “Checking string-encoded categorical data”, pp. 214–216 |
| Separación del target y consistencia train/test | continuación de “One-Hot-Encoding”, pp. 216–217 |
| Categóricas almacenadas como números | “Numbers Can Encode Categoricals”, pp. 218–219 |
| Binning y comparación de modelos lineales/árboles | “Binning, Discretization, Linear Models, and Trees”, pp. 220–224 |
| Variable original, bins y pendientes por interacción | “Interactions and Polynomials”, pp. 224–226 |
| Potencias y regresión polinómica | “Interactions and Polynomials”, pp. 227–229 |
| Interacciones de grado 2 y comparación Ridge/random forest | “Interactions and Polynomials”, pp. 230–231 |
| `log`, `exp`, periodicidad y conteos | “Univariate Nonlinear Transformations”, pp. 232–235 |
| Riesgo dimensional y familias de selección | apertura de “Automatic Feature Selection”, p. 236 |
| Selección univariante | “Univariate Statistics”, pp. 236–238 |
| Selección basada en modelo | “Model-Based Feature Selection”, pp. 238–240 |
| RFE, coste y utilidad práctica | “Iterative Feature Selection”, pp. 240–241 |
| Conocimiento experto y variables de calendario | “Utilizing Expert Knowledge”, pp. 242–243 |
| Split temporal y representación POSIX | “Utilizing Expert Knowledge”, pp. 243–245 |
| Límite de extrapolación del random forest | “Utilizing Expert Knowledge”, p. 245 |
| Hora y día de semana con random forest | “Utilizing Expert Knowledge”, pp. 245–246 |
| Codificación categórica temporal con modelo lineal | “Utilizing Expert Knowledge”, pp. 247–248 |
| Interacciones día-hora e interpretación por coeficientes | “Utilizing Expert Knowledge”, pp. 248–250 |
| Conclusiones sobre representación y modelo | “Summary and Outlook”, p. 250 |

### Estado de lectura

- Páginas impresas revisadas: 211–250 (páginas PDF 1–40).
- Secciones ilegibles o no procesables: ninguna.
