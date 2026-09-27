---
title: "Introducción a las librerías de creación de modelos de Python"
topics:
  - "Interconexión pandas -> modelado (to_numpy, ingeniería de características básica)"
  - "Patsy: fórmulas y matrices de diseño para modelos lineales"
  - "statsmodels: regresión OLS y modelos de series temporales (AutoReg)"
  - "scikit-learn: flujo básico de clasificación (LogisticRegression, validación cruzada)"
target_knowledge_files:
  - "knowledge/preprocessing/pandas_to_model_arrays.md"
  - "knowledge/feature_engineering/dummy_variables_categorical.md"
  - "knowledge/models/linear_models_statsmodels.md"
  - "knowledge/time_series/autoregressive_models.md"
  - "knowledge/models/sklearn_classification_baseline.md"
  - "knowledge/validation/cross_validation_basics.md"
---

# 12. Introducción a las librerías de creación de modelos de Python

## 1. Propósito del capítulo

El capítulo cierra el libro mostrando el punto de conexión entre la preparación de datos con pandas y el ajuste de modelos estadísticos o de aprendizaje automático. El autor aclara que no pretende enseñar modelado en profundidad, sino mostrar los mecanismos prácticos para pasar de un `DataFrame` a estructuras que las librerías de modelado (Patsy, statsmodels, scikit-learn) puedan consumir. Es relevante para Machine Learning porque documenta el paso de "datos limpios" a "matriz de diseño / arrays de entrenamiento", que es exactamente el cuello de botella entre EDA/preprocesamiento y modelado en un coding test o datathon.

## 2. Conceptos esenciales

- **Ingeniería de características (feature engineering)**: cualquier transformación que extrae información útil de datos sin procesar para el modelo. El capítulo indica que la agregación y `GroupBy` de pandas se usan habitualmente con este fin, pero no profundiza en técnicas.
- **Punto de contacto pandas–librería de modelado**: casi siempre es un array NumPy (`ndarray`). El método `DataFrame.to_numpy()` es la vía estándar de conversión.
- **Homogeneidad de tipos**: `to_numpy()` da un array numérico "limpio" solo si todas las columnas son numéricas; si hay columnas de tipos mixtos (p. ej. strings), el resultado es un `ndarray` de `dtype=object`, lo cual generalmente no es apto para pasar directamente a un modelo.
- **Matriz de diseño (design matrix)**: representación numérica (variables independientes X e dependiente y) que Patsy construye a partir de una fórmula de texto y un dataset.
- **Transformación con estado (stateful transformation)**: transformación (p. ej. estandarizar, centrar) que depende de estadísticos (media, desviación estándar) calculados sobre el conjunto de entrenamiento; debe aplicarse a datos nuevos reutilizando esos mismos estadísticos, nunca recalculándolos sobre el nuevo conjunto.
- **Interceptación (Intercept)**: columna constante de unos que Patsy añade por convenio a los modelos lineales, salvo que se elimine explícitamente con `+ 0`.
- **Variables indicadoras / dummies**: codificación de variables categóricas en columnas binarias. Patsy las genera automáticamente para términos no numéricos en una fórmula.
- **Términos de interacción** (`key1:key2`): en Patsy representan la interacción entre dos variables categóricas, típicos de modelos ANOVA.

## 3. Flujo o procedimiento presentado

Procedimiento general para pasar de datos en pandas a un modelo ajustado (síntesis de las secciones 12.1–12.4):

1. Cargar y limpiar los datos con pandas (fuera del alcance de este capítulo, cubierto en capítulos previos del libro).
2. Detectar y tratar valores ausentes en las columnas que se usarán como predictoras (el capítulo usa imputación simple con la mediana calculada **solo sobre el conjunto de entrenamiento**).
3. Codificar variables categóricas como numéricas (variables indicadoras manuales con `pd.get_dummies`, o automáticamente vía fórmulas de Patsy).
4. Seleccionar el subconjunto de columnas predictoras con `df.loc[:, cols].to_numpy()` en vez de convertir todo el DataFrame.
5. Construir las matrices de entrada (X) y salida (y), ya sea manualmente (NumPy/pandas) o mediante `patsy.dmatrices('y ~ x0 + x1', data)`.
6. Instanciar el modelo (`sm.OLS`, `smf.ols`, `AutoReg`, `LogisticRegression`, etc.).
7. Ajustar el modelo con `.fit()`.
8. Inspeccionar los parámetros y diagnósticos (`results.params`, `results.summary()`, `results.tvalues`).
9. Generar predicciones sobre datos nuevos/de prueba (`results.predict(...)`, `model.predict(X_test)`).
10. (Opcional) Evaluar y afinar el modelo mediante validación cruzada (`cross_val_score`, o clases con CV integrada como `LogisticRegressionCV`).

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Todas las columnas del DataFrame son numéricas y se necesitan como array para el modelo | `df.to_numpy()` | Conversión directa y eficiente al formato que consumen las librerías de modelado | Si hay columnas no numéricas mezcladas, el resultado será `dtype=object`, no apto para la mayoría de modelos |
| Solo se necesita un subconjunto de columnas para el modelo | `df.loc[:, cols].to_numpy()` | Evita crear una copia con columnas irrelevantes y dejar el resultado homogéneo en tipo | Verificar que `cols` sean todas numéricas antes de convertir |
| Modelo lineal con variables categóricas y/o transformaciones (log, estandarizar, centrar) | Fórmulas de Patsy (`patsy.dmatrices`) | Genera automáticamente la matriz de diseño, dummies y maneja el intercepto; es "lo más sencillo y menos propenso a errores" según el autor cuando hay algo más que columnas numéricas simples | Al aplicar transformaciones con estado (estandarizar/centrar) a datos nuevos, usar `patsy.build_design_matrices` con el `design_info` original, no recalcular sobre los datos nuevos |
| Se requiere sumar columnas dentro de una fórmula Patsy | Envolver en `I(x0 + x1)` | El operador `+` en fórmulas Patsy no suma, indica términos del modelo | Confundir `+` de fórmula con suma aritmética es un error común |
| Ajuste de regresión lineal clásica (OLS) sobre datos ya en arrays NumPy | `sm.OLS(y, X)` con `sm.add_constant(X)` | Interfaz basada en arrays de statsmodels; requiere añadir manualmente la constante | Olvidar `add_constant` deja el modelo sin intercepto |
| Ajuste de regresión lineal con datos en DataFrame y nombres de columnas significativos | `smf.ols('y ~ col0 + col1 + col2', data=data).fit()` | La API de fórmulas de statsmodels conserva los nombres de columnas en los resultados (`Series`) y no requiere `add_constant` | — |
| Modelado de series temporales con estructura autorregresiva | `statsmodels.tsa.ar_model.AutoReg` | Diseñado para procesos AR; permite ajustar con más retardos (`lags`) de los necesarios cuando no se conoce el orden real | El capítulo no profundiza en cómo elegir el número óptimo de retardos ni en su interpretación |
| Problema de clasificación binaria como baseline rápido (p. ej. tipo Titanic) | `sklearn.linear_model.LogisticRegression` | Modelo simple, rápido de entrenar, buen punto de partida | El propio autor advierte que no es un ejemplo de buen diseño de características, es solo demostrativo |
| Se necesita evitar sobreajuste y estimar mejor el rendimiento fuera de muestra | Validación cruzada (`cross_val_score`, o estimadores con CV integrada como `LogisticRegressionCV`) | Divide los datos de entrenamiento para simular predicción fuera de muestra | Los modelos con CV integrada tardan más en entrenarse |

## 5. Métodos y técnicas

### `DataFrame.to_numpy()`
- **Objetivo**: convertir un DataFrame (o una selección de columnas) en un array NumPy para pasarlo a librerías de modelado.
- **Entradas**: DataFrame con columnas homogéneas (idealmente numéricas).
- **Salida**: `ndarray` 2D; si los tipos son mixtos, `dtype=object`.
- **Funcionamiento**: copia los valores subyacentes del DataFrame a un array; para reconstruir el DataFrame se pasa el array de vuelta a `pd.DataFrame(array, columns=[...])`.
- **Hiperparámetros**: ninguno relevante mencionado.
- **Ventajas**: simple, es el "punto de contacto" estándar entre pandas y librerías externas.
- **Limitaciones**: pierde metadatos (nombres de columnas, tipos por columna) si no se gestionan aparte; con tipos mixtos el resultado no es directamente utilizable por la mayoría de modelos.
- **Casos adecuados**: preparar `X`/`y` para modelos que solo aceptan arrays.
- **Casos donde no conviene**: cuando hay variables categóricas sin codificar — primero hay que generar dummies o usar Patsy.

### `pandas.get_dummies`
- **Objetivo**: convertir una columna categórica en columnas indicadoras (0/1).
- **Entradas**: Series categórica (o `Categorical`), y opcionalmente un prefijo para nombrar columnas.
- **Salida**: DataFrame con una columna por categoría.
- **Funcionamiento**: se generan las dummies, se elimina la columna original y se unen (`join`) al DataFrame.
- **Ventajas**: control manual y explícito sobre la codificación.
- **Limitaciones**: el autor señala que hay "matices" al ajustar modelos estadísticos con variables indicadoras (p. ej. colinealidad si no se elimina una categoría de referencia), y recomienda Patsy como alternativa más segura.
- **Casos adecuados**: cuando no se va a usar Patsy/fórmulas y se necesita codificación manual simple.

### Patsy (`patsy.dmatrices`, `patsy.build_design_matrices`)
- **Objetivo**: describir modelos estadísticos (especialmente lineales) mediante una sintaxis de fórmula tipo `y ~ x0 + x1` y generar automáticamente las matrices de diseño (X, y).
- **Entradas**: cadena de fórmula + dataset (DataFrame o diccionario de arrays).
- **Salida**: objetos `DesignMatrix` (subclase de `ndarray` con metadatos en `design_info`, incluidos nombres de columnas).
- **Funcionamiento**:
  - `y ~ x0 + x1` construye y con intercepto por defecto; `+ 0` lo suprime.
  - Permite incrustar funciones Python dentro de la fórmula, p. ej. `np.log(np.abs(x1) + 1)`.
  - Ofrece transformaciones integradas con estado: `standardize(x)` (media 0, varianza 1) y `center(x)` (resta la media).
  - `I(x0 + x1)` permite sumar columnas aritméticamente (ya que `+` en la fórmula no suma).
  - Variables no numéricas se convierten automáticamente en dummies; si hay intercepto, se omite un nivel para evitar colinealidad (codificación tipo `key1[T.b]`).
  - Sin intercepto (`+ 0`), se generan columnas para todas las categorías.
  - `C(variable)` fuerza a tratar una columna numérica como categórica.
  - `key1:key2` genera términos de interacción entre categóricas (útil en ANOVA).
  - `patsy.build_design_matrices([X.design_info], new_data)` aplica las mismas transformaciones (incluidas las con estado) ajustadas sobre el dataset original a datos nuevos, preservando los estadísticos usados en el entrenamiento.
- **Hiperparámetros**: no aplica en sentido estricto; la "configuración" es la propia fórmula de texto.
- **Ventajas**: evita errores manuales al construir dummies e interacciones; centraliza los metadatos del modelo (`design_info`) para reutilizarlos en predicción.
- **Limitaciones**: el capítulo no cubre en profundidad todas las transformaciones disponibles (remite a la documentación oficial); el manejo de datos categóricos complejos queda fuera del alcance del libro.
- **Casos adecuados**: cualquier escenario con variables categóricas y/o transformaciones para modelos lineales en statsmodels.
- **Casos donde no conviene**: si se trabaja directamente con scikit-learn con arrays ya preparados, Patsy no es necesario (el ejemplo de scikit-learn del capítulo no usa Patsy).

### `statsmodels` — regresión OLS (`sm.OLS`, `smf.ols`)
- **Objetivo**: ajustar modelos de regresión lineal (y otros modelos lineales/generalizados, mencionados pero no desarrollados: robustos, de efectos mixtos, ANOVA, espacio de estado, método generalizado de momentos).
- **Entradas**: 
  - Interfaz de arrays: `sm.OLS(y, X)`, con `X` normalmente incluyendo una constante añadida vía `sm.add_constant(X)`.
  - Interfaz de fórmulas: `smf.ols('y ~ col0 + col1 + col2', data=df)`.
- **Salida**: objeto de resultados tras `.fit()`, con `.params` (coeficientes), `.tvalues` (estadísticos t), `.summary()` (tabla completa de diagnóstico) y `.predict(nuevos_datos)`.
- **Funcionamiento**: ajuste por mínimos cuadrados ordinarios; la interfaz de fórmulas conserva automáticamente los nombres de columnas en los resultados (como `pandas.Series`), y no requiere `add_constant` manual.
- **Hiperparámetros**: no hay hiperparámetros de regularización en OLS puro (ese es el caso más básico); el capítulo menciona que existen variantes más complejas (mínimos cuadrados iterativamente reponderados) sin desarrollarlas.
- **Ventajas**: salida estadística rica (`summary()` incluye R², estadístico F, p-valores, intervalos de confianza, diagnósticos de residuos como Durbin-Watson, Jarque-Bera, condición number).
- **Limitaciones**: el capítulo no explica cómo interpretar cada estadístico del `summary()` en detalle (queda para un curso de estadística).
- **Casos adecuados**: problemas de regresión donde interesa la inferencia estadística (significancia de coeficientes, intervalos de confianza), no solo la predicción.
- **Casos donde no conviene**: si el objetivo es maximizar poder predictivo en datos complejos/no lineales, el capítulo remite a scikit-learn.

### `statsmodels.tsa.ar_model.AutoReg`
- **Objetivo**: ajustar procesos autorregresivos (AR) a series temporales.
- **Entradas**: array/serie de valores temporales y número máximo de retardos (`lags`).
- **Salida**: tras `.fit()`, `.params` con la interceptación seguida de las estimaciones de cada retardo.
- **Funcionamiento**: cuando no se conoce el orden real del proceso AR, se puede ajustar con un número de retardos mayor al necesario (en el ejemplo, datos con estructura real AR(2) se ajustan con `MAXLAGS = 5`).
- **Hiperparámetros**: número de retardos (`lags`) — parámetro clave, pero el capítulo no explica cómo seleccionarlo óptimamente (p. ej. AIC/BIC, PACF).
- **Ventajas**: interfaz directa para AR.
- **Limitaciones**: el capítulo indica explícitamente que la interpretación detallada de estos modelos "queda más allá del alcance del libro"; no cubre selección de orden, diagnóstico de residuos ni otros modelos de series temporales mencionados (filtrado de Kalman, espacio de estado, VAR).
- **Casos adecuados**: series temporales con autocorrelación de corto plazo.
- **Casos donde no conviene**: sin más contexto del capítulo, no se puede afirmar cuándo evitarlo; requeriría consultar documentación adicional de statsmodels.

### `sklearn.linear_model.LogisticRegression` / `LogisticRegressionCV`
- **Objetivo**: clasificación binaria (ejemplo: predecir supervivencia en el dataset Titanic de Kaggle).
- **Entradas**: `X_train` (array numérico de predictores), `y_train` (etiquetas binarias).
- **Salida**: tras `.fit()`, modelo entrenado; `.predict(X_test)` devuelve las clases predichas.
- **Funcionamiento**: 
  1. Imputar ausentes en predictores (el capítulo usa la mediana de `Age` del conjunto de **entrenamiento**, aplicada también al conjunto de prueba, evitando así fuga de información desde el test).
  2. Codificar variables categóricas simples manualmente (p. ej. `IsFemale = (Sex == 'female').astype(int)`).
  3. Construir `X_train`/`X_test` como arrays NumPy vía `to_numpy()` sobre las columnas predictoras elegidas.
  4. Instanciar y ajustar: `LogisticRegression().fit(X_train, y_train)`.
  5. Predecir: `model.predict(X_test)`.
- **Hiperparámetros**: `C` (inverso de la fuerza de regularización) es el que se menciona explícitamente; `LogisticRegressionCV(Cs=10)` realiza búsqueda en rejilla sobre una malla de 10 valores de `C` con validación cruzada integrada.
- **Ventajas**: rápido de entrenar, buen baseline para clasificación binaria.
- **Limitaciones**: el propio autor aclara que el ejemplo no representa un buen diseño de características, es solo ilustrativo del flujo de la API.
- **Casos adecuados**: baseline rápido de clasificación binaria con features numéricas simples.
- **Casos donde no conviene**: el capítulo no explora relaciones no lineales ni interacciones; para eso remite a otras fuentes (libros recomendados en la conclusión).

### Validación cruzada (`cross_val_score`)
- **Objetivo**: estimar el rendimiento del modelo fuera de muestra y mitigar el sobreajuste.
- **Entradas**: modelo (estimador de scikit-learn), `X_train`, `y_train`, número de particiones (`cv`).
- **Salida**: array de puntuaciones (una por partición/fold).
- **Funcionamiento**: divide los datos de entrenamiento en `cv` partes no superpuestas, entrena y evalúa iterativamente, simulando predicción fuera de muestra. Ejemplo del capítulo: `cross_val_score(model, X_train, y_train, cv=4)` → 4 puntuaciones de exactitud.
- **Hiperparámetros**: `cv` (número de particiones); la métrica de evaluación (por defecto depende del modelo, pero se puede especificar explícitamente).
- **Ventajas**: mejor estimación del rendimiento real que una sola partición train/test; permite comparar configuraciones de hiperparámetros (búsqueda en rejilla).
- **Limitaciones**: modelos con CV integrada (`LogisticRegressionCV`) tardan más en entrenar.
- **Casos adecuados**: cualquier modelo donde se necesite estimar generalización o afinar hiperparámetros.

## 6. Fórmulas esenciales

Este capítulo no requiere fórmulas esenciales para su aplicación práctica más allá de la especificación del modelo lineal implícita en la notación de Patsy. El propio texto la resume conceptualmente:

- **Modelo lineal simulado en el ejemplo de statsmodels**: `y = X·beta + eps`, donde:
  - `X` es la matriz de predictores (n observaciones × p variables),
  - `beta` es el vector de coeficientes "reales" usados para simular los datos de ejemplo (`[0.1, 0.3, 0.5]`),
  - `eps` es el término de error aleatorio (ruido normal),
  - `y` es la variable respuesta.
  - Interpretación: es el modelo generador de datos usado únicamente con fines didácticos para mostrar que `sm.OLS` recupera coeficientes estimados (`results.params`) cercanos a los `beta` reales.
- El capítulo no desarrolla la fórmula matemática de mínimos cuadrados, la función de verosimilitud del modelo logístico, ni la fórmula del proceso AR(2) (`x_t = b0·x_{t-1} + b1·x_{t-2} + ruido`, mencionada solo conceptualmente al describir la simulación, sin discutir condiciones de estacionariedad).

## 7. Implementación práctica

Ejemplo mínimo que combina los pasos clave del capítulo (preparación con pandas + codificación categórica vía Patsy + ajuste OLS con statsmodels), consolidando el flujo mostrado en las secciones 12.1–12.3:

```python
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Datos ya limpios en un DataFrame (supuesto: sin nulos en estas columnas)
data = pd.DataFrame({
    "x0": [1, 2, 3, 4, 5],
    "x1": [0.01, -0.01, 0.25, -4.1, 0.0],
    "y": [-1.5, 0.0, 3.6, 1.3, -2.0],
})

# 2. Construcción de la matriz de diseño con Patsy (incluye intercepto)
y, X = patsy.dmatrices("y ~ x0 + x1", data)

# 3. Ajuste con la interfaz de fórmulas de statsmodels (conserva nombres de columnas)
results = smf.ols("y ~ x0 + x1", data=data).fit()
print(results.params)
print(results.summary())

# 4. Predicción sobre datos nuevos (misma estructura de columnas)
nuevos_datos = pd.DataFrame({"x0": [6, 7], "x1": [3.1, -0.5]})
predicciones = results.predict(nuevos_datos)
```

Ejemplo mínimo del flujo de clasificación con scikit-learn (basado en el ejemplo Titanic del capítulo, con separación explícita entre imputación basada solo en entrenamiento y validación cruzada):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Supuesto: train y test ya cargados, con 'Age' imputado usando
# la mediana calculada EXCLUSIVAMENTE sobre train (evita data leakage)
predictors = ["Pclass", "IsFemale", "Age"]
X_train = train[predictors].to_numpy()
y_train = train["Survived"].to_numpy()
X_test = test[predictors].to_numpy()

model = LogisticRegression(C=10, random_state=0)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# Validación cruzada para estimar rendimiento fuera de muestra
scores = cross_val_score(model, X_train, y_train, cv=4)
```

**Nota práctica (actualización de API moderna, no contenido original del libro)**: en versiones recientes de scikit-learn conviene fijar `random_state` en `LogisticRegression` para reproducibilidad (el capítulo no lo hace explícitamente en su ejemplo original) y usar `sklearn.model_selection.GridSearchCV` o `sklearn.linear_model.LogisticRegressionCV` para búsquedas de hiperparámetros más controladas que una validación cruzada manual simple.

## 8. Supuestos y condiciones de aplicación

- `to_numpy()` asume columnas de tipo homogéneo para producir un array numérico útil; con tipos mixtos, el resultado (`dtype=object`) generalmente no sirve como entrada directa de un modelo.
- Las transformaciones con estado de Patsy (`standardize`, `center`) asumen que los estadísticos (media, desviación estándar) deben calcularse **una sola vez sobre el conjunto de entrenamiento** y reutilizarse sobre datos nuevos vía `design_info`; no se deben recalcular sobre datos de test u out-of-sample.
- El modelo OLS de ejemplo asume (implícitamente, sin discutirlo en detalle) los supuestos clásicos de regresión lineal (linealidad, errores con media 0), pero el capítulo no los enumera ni los valida explícitamente.
- `AutoReg` asume una estructura autorregresiva de orden desconocido, ajustable especificando un número máximo de retardos mayor al orden real.
- Estadísticamente, la mayoría de modelos de statsmodels y scikit-learn **no aceptan valores ausentes**; es una condición previa explícita del capítulo ("a librerías como statsmodels y scikit-learn no se les pueden pasar datos ausentes").

## 9. Errores y riesgos

- **Confundir `+` en fórmulas Patsy con suma aritmética**: en una fórmula, `x0 + x1` indica dos términos del modelo, no su suma; para sumar columnas hay que usar `I(x0 + x1)`.
- **Recalcular estadísticos de transformaciones con estado sobre datos nuevos** (en vez de reutilizar `design_info` del conjunto de entrenamiento vía `build_design_matrices`): esto es una fuente directa de inconsistencia entre entrenamiento y predicción (riesgo conceptualmente análogo a un data leakage invertido / fuga de la definición del modelo).
- **Imputar valores ausentes usando estadísticos del conjunto completo (train+test) en vez de solo del entrenamiento**: el capítulo evita explícitamente este error usando `impute_value = train['Age'].median()` y aplicándolo a ambos conjuntos, un patrón a replicar para prevenir data leakage.
- **Pasar datos con valores ausentes directamente a statsmodels o scikit-learn**: no funcionará; hay que verificar con `isna().sum()` antes de ajustar el modelo.
- **Convertir un DataFrame con columnas de tipos mixtos con `to_numpy()` esperando un array numérico limpio**: produce un array `dtype=object`, no apto para la mayoría de modelos.
- **Ajustar variables indicadoras sin cuidar la colinealidad** (incluir dummies de todas las categorías junto con el intercepto): el capítulo advierte que hay matices en este punto y recomienda Patsy, que omite automáticamente un nivel de referencia cuando hay intercepto.
- **No usar validación cruzada** y evaluar solo con una partición train/test: el capítulo la presenta como técnica para "evitar el sobreajuste... y obtener con frecuencia un mejor rendimiento predictivo o una mayor robustez".
- **Limitación computacional implícita**: modelos con búsqueda en rejilla de hiperparámetros (`LogisticRegressionCV`) "tardan más en entrenarse" que un ajuste simple; relevante bajo presión de tiempo en una competencia.

## 10. Aplicación en un coding test

*Síntesis práctica derivada del capítulo.*

- **Cuándo consultar este capítulo**: al pasar de un DataFrame limpio a la fase de modelado — es decir, en el momento exacto de construir `X`/`y`, codificar categóricas y elegir entre `statsmodels` (si se necesita inferencia estadística/interpretabilidad de coeficientes) o `scikit-learn` (si el objetivo es predicción/clasificación).
- **Qué decisiones permite tomar**: 
  - Si hay variables categóricas y se va a usar un modelo lineal en statsmodels, usar fórmulas de Patsy en vez de codificar dummies a mano.
  - Si el objetivo es puramente predictivo (clasificación/regresión) sobre datos ya numéricos, ir directo a scikit-learn con `to_numpy()`.
- **Qué partes sirven para un baseline**:
  - Selección de predictores simples + imputación con mediana del train + `LogisticRegression()` por defecto (sin ajustar `C`) es exactamente el patrón de baseline mostrado con el dataset Titanic.
  - Para regresión, `smf.ols('y ~ predictores', data=df).fit()` da un baseline interpretable rápidamente.
- **Qué partes pueden mejorar el modelo**:
  - Afinar `C` con `LogisticRegressionCV` o `cross_val_score` sobre una rejilla.
  - Revisar `results.summary()` (p-valores, R²) para descartar predictores poco informativos en un modelo lineal.
- **Qué no debería implementarse bajo presión de tiempo**:
  - Selección manual y exhaustiva de orden óptimo de un modelo `AutoReg` (el capítulo mismo no explica cómo hacerlo) — mejor usar un valor de retardos razonablemente alto o herramientas específicas de selección si el tiempo lo permite.
  - Búsquedas de hiperparámetros muy exhaustivas (`LogisticRegressionCV` con muchos valores de `Cs`, o `cv` alto), dado el coste computacional mencionado.
- **Qué señales del dataset activarían las técnicas del capítulo**:
  - Presencia de columnas categóricas junto a numéricas + necesidad de un modelo lineal interpretable → Patsy + statsmodels.
  - Presencia de valores ausentes en predictores → imputación (mediana u otra) calculada solo sobre entrenamiento antes de modelar.
  - Variable objetivo temporal con posible autocorrelación → considerar `AutoReg` como primer acercamiento simple antes de modelos más complejos.
  - Objetivo de clasificación binaria simple con pocas features → `LogisticRegression` como baseline rápido, con `cross_val_score` para validar.

## 11. Checklist operativo

- [ ] Verificar tipos de columnas antes de usar `to_numpy()` (deben ser homogéneas/numéricas).
- [ ] Comprobar valores ausentes con `isna().sum()` en train y test antes de modelar.
- [ ] Imputar valores ausentes usando estadísticos calculados **solo** sobre el conjunto de entrenamiento.
- [ ] Codificar variables categóricas: usar Patsy si se trabaja con statsmodels/modelos lineales con fórmula; usar `pd.get_dummies` o codificación manual si se trabaja directo con arrays.
- [ ] Si se usan transformaciones con estado (estandarizar/centrar), guardar el `design_info` (u objeto equivalente) del entrenamiento y reutilizarlo sobre datos nuevos.
- [ ] Elegir la librería según el objetivo: statsmodels para inferencia/interpretabilidad, scikit-learn para predicción/clasificación.
- [ ] Ajustar un modelo baseline simple antes de afinar hiperparámetros.
- [ ] Evaluar con validación cruzada (`cross_val_score` u otra) antes de confiar en una única partición train/test.
- [ ] Revisar diagnósticos disponibles (`results.summary()` en statsmodels) para detectar predictores poco significativos.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ----------------------------- | --------: |
| Conversión DataFrame → array (`to_numpy`, selección de columnas con `loc`) | `knowledge/preprocessing/pandas_to_model_arrays.md` | Media |
| Codificación de variables categóricas (dummies manuales y con Patsy, términos de interacción) | `knowledge/feature_engineering/dummy_variables_categorical.md` | Alta |
| Fórmulas de Patsy, transformaciones con estado, `design_info` y matrices de diseño | `knowledge/preprocessing/pandas_to_model_arrays.md` | Alta |
| Regresión OLS con statsmodels (interfaz arrays y fórmulas, `summary()`) | `knowledge/models/linear_models_statsmodels.md` | Alta |
| Modelos autorregresivos (`AutoReg`) para series temporales | `knowledge/time_series/autoregressive_models.md` | Media |
| Flujo de clasificación baseline con scikit-learn (Titanic: imputación, codificación, `LogisticRegression`) | `knowledge/models/sklearn_classification_baseline.md` | Alta |
| Validación cruzada (`cross_val_score`, `LogisticRegressionCV`) | `knowledge/validation/cross_validation_basics.md` | Alta |
| Riesgo de fuga de información al imputar con estadísticos de todo el dataset en vez de solo train | `knowledge/validation/cross_validation_basics.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| -------------------- | -------------------------------- |
| 2. Conceptos esenciales (to_numpy, homogeneidad, feature engineering) | 12.1 "Interconexión entre pandas y el código para la creación de modelos" |
| 2 y 5. Variables indicadoras manuales (`get_dummies`) | Final de la sección 12.1 |
| 2, 3, 5, 6. Patsy: fórmulas, matrices de diseño, transformaciones, categóricas, interacciones | 12.2 "Creación de descripciones de modelos con Patsy" (incluye subsecciones "Transformaciones de datos en fórmulas Patsy" y "Datos categóricos y Patsy") |
| 5, 6, 7. statsmodels — OLS con arrays y con fórmulas | 12.3 "Introducción a statsmodels", subsección "Estimación de modelos lineales" |
| 5. statsmodels — AutoReg / series temporales | 12.3, subsección "Estimación de procesos de series temporales" |
| 3, 5, 7, 9. scikit-learn — ejemplo Titanic completo (carga, nulos, imputación, codificación, `LogisticRegression`, predicción) | 12.4 "Introducción a scikit-learn" (hasta la introducción de validación cruzada) |
| 5, 9. Validación cruzada (`LogisticRegressionCV`, `cross_val_score`) | 12.4, tramo final ("Podemos ajustar este modelo... búsqueda en rejilla...") |
| 1. Propósito y alcance general del capítulo, y libros recomendados (no incluidos como contenido técnico, solo referenciados) | 12.5 "Conclusión" y párrafos introductorios del capítulo |

---

**Observación de control de calidad**: todas las páginas del capítulo proporcionado (texto completo del Capítulo 12, secciones 12.1 a 12.5) fueron legibles y se procesaron en su totalidad; no hubo secciones ilegibles ni vacíos que requirieran señalización. El capítulo no incluye apéndices propios necesarios para su comprensión.
