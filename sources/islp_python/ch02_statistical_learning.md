---
title: "Statistical Learning"
topics:
  - "prediction and inference"
  - "model flexibility and interpretability"
  - "training error and test error"
  - "bias-variance trade-off"
  - "classification, Bayes classifier, and K-nearest neighbors"
  - "Python foundations for numerical analysis and EDA"
target_knowledge_files:
  - "knowledge/problem_definition/prediction_vs_inference.md"
  - "knowledge/problem_definition/supervised_unsupervised_regression_classification.md"
  - "knowledge/validation/training_vs_test_error.md"
  - "knowledge/validation/bias_variance_tradeoff.md"
  - "knowledge/metrics/regression_and_classification_error.md"
  - "knowledge/models/knn_classification.md"
  - "knowledge/eda/python_tabular_eda.md"
---

# 2. Statistical Learning

## 1. Propósito del capítulo

El capítulo presenta el aprendizaje estadístico como el conjunto de métodos usados para estimar una relación desconocida entre predictores y respuesta. Su objetivo operativo es enseñar a formular una tarea como predicción o inferencia, distinguir aprendizaje supervisado y no supervisado, evaluar modelos por su rendimiento fuera de muestra y comprender por qué la flexibilidad puede mejorar o perjudicar la generalización. También introduce el clasificador de Bayes y K-nearest neighbors (KNN), y cierra con un laboratorio básico de Python, NumPy, pandas y visualización.

**Convención de procedencia:** salvo donde se indique **“Conclusión práctica derivada”**, **“Síntesis práctica derivada del capítulo”**, **“Recomendación para competencias”** o **“Nota práctica - adaptación moderna”**, el contenido siguiente parafrasea explicaciones explícitas del capítulo. Estas etiquetas separan las inferencias operativas y las actualizaciones de API de lo afirmado directamente por los autores.

## 2. Conceptos esenciales

### Relación entre entradas y salida

El capítulo representa una respuesta cuantitativa mediante \(Y=f(X)+\epsilon\), donde \(X=(X_1,\ldots,X_p)\) contiene los predictores, \(f\) recoge la información sistemática que estos aportan y \(\epsilon\) representa variación aleatoria independiente de \(X\), con media cero.

- **Predicción:** interesa obtener \(\hat{Y}=\hat{f}(X)\) con poco error. La forma exacta de \(\hat f\) puede tratarse como una caja negra si la precisión es el único objetivo.
- **Inferencia:** interesa conocer qué predictores se asocian con la respuesta, la dirección y magnitud de las relaciones, y si estas pueden describirse de forma lineal. Aquí la interpretabilidad de \(\hat f\) es central.
- **Error reducible:** procede de que \(\hat f\) no coincide perfectamente con \(f\); puede disminuir con un método o ajuste mejor.
- **Error irreducible:** procede de \(\epsilon\), incluidas variables no medidas y variación no medible. Ningún modelo basado solo en \(X\) puede eliminarlo.

### Familias de estimación

- **Métodos paramétricos:** suponen una forma para \(f\), como un modelo lineal, y luego estiman un conjunto finito de parámetros. Simplifican el ajuste, pero una forma funcional incorrecta introduce sesgo.
- **Métodos no paramétricos:** evitan imponer una forma explícita y buscan aproximarse a los datos manteniendo suficiente suavidad. Pueden representar relaciones más variadas, pero suelen requerir muchas observaciones y pueden sobreajustar.

### Flexibilidad, precisión e interpretabilidad

Una mayor flexibilidad permite representar relaciones complejas y suele reducir el sesgo, pero aumenta la varianza y dificulta la interpretación. Los modelos restrictivos, como la regresión lineal, son adecuados cuando la inferencia es prioritaria; métodos más flexibles pueden ser útiles para predicción, pero no garantizan menor error de prueba. El capítulo sitúa, de menor a mayor flexibilidad, técnicas como selección de subconjuntos y lasso, mínimos cuadrados, modelos aditivos generalizados, árboles, bagging/boosting, máquinas de vectores soporte y deep learning, señalando una tendencia inversa en interpretabilidad.

### Tipos de aprendizaje y de respuesta

- **Supervisado:** cada observación tiene predictores \(x_i\) y respuesta \(y_i\). Incluye objetivos de predicción e inferencia.
- **No supervisado:** solo se observan los \(x_i\); se buscan relaciones entre variables u observaciones. El clustering intenta identificar grupos sin etiquetas conocidas.
- **Semisupervisado:** hay respuesta para solo una parte de las observaciones. El capítulo lo define, pero no desarrolla métodos.
- **Regresión:** la respuesta es cuantitativa.
- **Clasificación:** la respuesta es cualitativa o categórica. El tipo de respuesta guía la elección del método más que el tipo de los predictores, siempre que los predictores cualitativos se codifiquen correctamente.

### Generalización y sobreajuste

El objetivo es minimizar el error en observaciones no usadas durante el ajuste. El error de entrenamiento suele disminuir al aumentar la flexibilidad, mientras que el error de prueba suele seguir una curva en U: primero cae al reducirse el sesgo y luego sube cuando domina la varianza. Hay sobreajuste cuando un modelo menos flexible habría obtenido menor error de prueba, aunque el modelo más flexible tenga mejor ajuste de entrenamiento.

## 3. Flujo o procedimiento presentado

1. **Definir la respuesta y los predictores.** Identificar \(Y\), \(X\) y la unidad de observación.
2. **Precisar el objetivo.** Decidir si importa predecir, hacer inferencia o ambas cosas.
3. **Clasificar la tarea.** Determinar si es supervisada, no supervisada o semisupervisada; y, si hay respuesta, si es regresión o clasificación.
4. **Revisar y cargar los datos.** Inspeccionar el archivo, declarar correctamente valores faltantes y comprobar forma, tipos y nombres de columnas.
5. **Explorar la estructura.** Resumir variables, visualizar distribuciones y relaciones, y tratar como categóricas las variables numéricas que representan categorías.
6. **Elegir una familia de métodos.** Considerar la forma esperada de la relación, la cantidad de datos, la necesidad de interpretación y el nivel de flexibilidad tolerable.
7. **Ajustar usando solo datos de entrenamiento.** La estimación de \(\hat f\) debe basarse en observaciones de entrenamiento.
8. **Evaluar fuera de muestra.** Comparar métodos mediante error de prueba; cuando no haya un conjunto de prueba disponible para la selección, estimarlo con técnicas como cross-validation, desarrolladas en otro capítulo.
9. **Controlar la complejidad.** Buscar el punto donde se equilibran sesgo y varianza; no elegir por el mínimo error de entrenamiento.
10. **Interpretar según el objetivo.** Para inferencia, explicar asociaciones y forma funcional; para predicción, priorizar el rendimiento de generalización sin ignorar la estabilidad.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se necesita comprender qué variables se asocian con la respuesta | Método restrictivo e interpretable, como un modelo lineal | Facilita estudiar dirección, magnitud y forma de las asociaciones | Una forma lineal incorrecta puede introducir sesgo alto |
| El objetivo principal es predecir | Comparar métodos con distinto nivel de flexibilidad mediante error fuera de muestra | La interpretabilidad puede ser secundaria, pero ningún nivel de flexibilidad domina siempre | El modelo más flexible puede sobreajustar |
| Se dispone de pocos datos | Favorecer un método paramétrico o menos flexible como punto de partida | Estimar pocos parámetros exige menos observaciones | La simplificación puede no capturar no linealidades importantes |
| La relación parece compleja y hay muchas observaciones | Considerar un método no paramétrico o flexible | Puede aproximar una gama amplia de formas sin imponer una ecuación rígida | Requiere más datos y control de suavidad o complejidad |
| La respuesta es cuantitativa | Plantear regresión y evaluar con error cuadrático medio (MSE) | El MSE mide la distancia cuadrática entre predicción y valor real | El MSE de entrenamiento no estima por sí solo la generalización |
| La respuesta es cualitativa | Plantear clasificación y evaluar la tasa de error | Mide la fracción de etiquetas incorrectas | La tasa de entrenamiento puede ser muy optimista |
| No existe respuesta etiquetada | Usar aprendizaje no supervisado, por ejemplo clustering si se buscan grupos | Permite estudiar estructura entre observaciones | Los grupos verdaderos son desconocidos y el solapamiento limita la recuperación |
| Se usa KNN para clasificación | Elegir \(K\) según rendimiento de validación o prueba estimado | \(K\) controla directamente la flexibilidad | \(K=1\) puede tener varianza alta; un \(K\) muy grande puede tener sesgo alto |

## 5. Métodos y técnicas

### Modelado paramétrico

- **Objetivo:** reducir la estimación de una función desconocida a la estimación de parámetros.
- **Entradas:** observaciones \((x_i,y_i)\) y una forma funcional supuesta.
- **Salida:** parámetros estimados y una función \(\hat f\).
- **Funcionamiento:** primero se fija una forma, por ejemplo lineal; después se ajustan sus coeficientes, comúnmente por mínimos cuadrados en el caso lineal.
- **Hiperparámetros:** el capítulo no define uno general; la complejidad depende de la familia paramétrica elegida.
- **Ventajas:** menor complejidad de estimación, necesidad de menos datos e interpretación más directa.
- **Limitaciones:** si la forma supuesta difiere mucho de la relación real, el sesgo puede ser alto.
- **Casos adecuados:** inferencia y problemas con pocos datos o relaciones aproximadamente simples.
- **No conviene:** cuando la relación es muy no lineal y una forma rígida no la representa.

### Modelado no paramétrico

- **Objetivo:** estimar \(f\) sin fijar de antemano una forma funcional explícita.
- **Entradas:** observaciones \((x_i,y_i)\).
- **Salida:** una función flexible ajustada a los datos.
- **Funcionamiento:** aproxima los puntos respetando una restricción de suavidad; el capítulo usa thin-plate splines como ejemplo.
- **Hiperparámetros:** nivel de suavidad o, de forma equivalente, flexibilidad.
- **Ventajas:** capacidad para representar formas complejas.
- **Limitaciones:** demanda más observaciones y puede seguir el ruido si se permite demasiada rugosidad.
- **Casos adecuados:** relaciones no lineales con datos suficientes.
- **No conviene:** muestras pequeñas o tareas donde la interpretabilidad sea prioritaria.

### Clasificador de Bayes

- **Objetivo:** asignar cada observación a la clase con mayor probabilidad condicional dada \(X=x_0\).
- **Entradas:** las probabilidades verdaderas \(P(Y=j\mid X=x_0)\).
- **Salida:** la clase de máxima probabilidad.
- **Funcionamiento:** define una frontera de decisión entre las regiones donde domina cada clase.
- **Hiperparámetros:** ninguno en su definición teórica.
- **Ventajas:** minimiza, en promedio, la tasa de error de prueba.
- **Limitaciones:** es inalcanzable directamente con datos reales porque la distribución condicional verdadera es desconocida.
- **Casos adecuados:** sirve como referencia teórica de rendimiento.
- **No conviene:** no puede implementarse como modelo práctico sin estimar antes las probabilidades condicionales.

### K-nearest neighbors (KNN) para clasificación

- **Objetivo:** aproximar la regla de Bayes usando clases observadas en un vecindario local.
- **Entradas:** datos de entrenamiento etiquetados, una observación \(x_0\), una noción de cercanía y un entero positivo \(K\).
- **Salida:** probabilidades de clase estimadas por frecuencia local y una etiqueta predicha.
- **Funcionamiento:** encuentra los \(K\) puntos de entrenamiento más cercanos, calcula la fracción de cada clase y elige la más frecuente.
- **Hiperparámetro relevante:** \(K\). Un \(K\) pequeño implica mayor flexibilidad; uno grande suaviza la frontera.
- **Ventajas:** simple y capaz de aproximar bien una frontera de Bayes.
- **Limitaciones:** la elección de \(K\) altera drásticamente el resultado; \(K=1\) puede memorizar el entrenamiento.
- **Casos adecuados:** clasificación cuando la estructura local contiene señal y el nivel de flexibilidad se valida.
- **No conviene:** elegirlo solo por error de entrenamiento o usar valores extremos de \(K\) sin evaluación fuera de muestra.

### Inspección y EDA tabular en Python

- **Objetivo:** comprobar estructura, calidad y relaciones básicas antes del modelado.
- **Entradas:** archivos tabulares y columnas numéricas o categóricas.
- **Salida:** data frames limpios, resúmenes y gráficos.
- **Funcionamiento:** `pandas.read_csv()` carga datos; `shape`, `columns`, `dtype`, `unique()`, `dropna()` y `describe()` ayudan a inspeccionarlos; histogramas, boxplots, scatterplots y matrices de dispersión muestran distribuciones y relaciones.
- **Parámetros relevantes:** codificación de faltantes (`na_values`), separador, selección de filas/columnas y número de bins.
- **Ventajas:** detecta tipos mal inferidos, faltantes y estructura antes del análisis.
- **Limitaciones:** la inspección no reemplaza la evaluación fuera de muestra ni establece causalidad.
- **Casos adecuados:** etapa inicial de cualquier análisis tabular.
- **No conviene:** eliminar faltantes mecánicamente sin evaluar su cantidad o mecanismo; el capítulo elimina filas solo en un ejemplo donde faltan cinco observaciones.

## 6. Fórmulas esenciales

### Modelo general

\[
Y=f(X)+\epsilon
\]

- \(Y\): respuesta; \(X=(X_1,\ldots,X_p)\): predictores; \(f\): relación sistemática desconocida; \(\epsilon\): error aleatorio independiente de \(X\), con media cero.
- Expresa que predecir \(Y\) exige estimar \(f\), aunque siempre queda variación irreducible.

### Predicción y descomposición reducible/irreducible

\[
\hat Y=\hat f(X)
\]

\[
E[(Y-\hat Y)^2]=[f(X)-\hat f(X)]^2+\operatorname{Var}(\epsilon)
\]

- El primer término es error reducible de estimación; el segundo es error irreducible.
- Supuesto mostrado en el capítulo: \(\hat f\) y \(X\) se consideran fijos para que la variación provenga de \(\epsilon\).

### Modelo lineal paramétrico

\[
f(X)=\beta_0+\beta_1X_1+\cdots+\beta_pX_p
\]

- \(\beta_0\): intercepto; \(\beta_j\): efecto lineal asociado al predictor \(X_j\).
- Reduce la estimación de una función arbitraria a \(p+1\) coeficientes, bajo el supuesto de linealidad.

### Error cuadrático medio

\[
\operatorname{MSE}_{train}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat f(x_i))^2
\]

\[
\operatorname{MSE}_{test}=\operatorname{Ave}(y_0-\hat f(x_0))^2
\]

- \(n\): número de observaciones de entrenamiento; \(y_i\): respuesta observada; \(\hat f(x_i)\): predicción; \((x_0,y_0)\): observación no usada para ajustar.
- Un valor menor indica predicciones más cercanas, pero el criterio relevante para generalización es el MSE de prueba, no el de entrenamiento.

### Descomposición sesgo-varianza

\[
E[(y_0-\hat f(x_0))^2]
=\operatorname{Var}(\hat f(x_0))+[\operatorname{Bias}(\hat f(x_0))]^2+\operatorname{Var}(\epsilon)
\]

- La varianza mide cuánto cambiaría \(\hat f\) con otro conjunto de entrenamiento.
- El sesgo mide el error introducido al aproximar una realidad compleja con un modelo más simple.
- La flexibilidad suele aumentar la varianza y reducir el sesgo; el mínimo error esperado requiere equilibrarlos.

### Tasa de error de clasificación

\[
\operatorname{Error}_{train}=\frac{1}{n}\sum_{i=1}^{n}I(y_i\neq\hat y_i)
\]

\[
\operatorname{Error}_{test}=\operatorname{Ave}(I(y_0\neq\hat y_0))
\]

- \(I(\cdot)\): indicador que vale 1 si la condición es cierta y 0 en otro caso.
- Mide la fracción de observaciones mal clasificadas; interesa minimizar la tasa de prueba.

### Regla y error de Bayes

\[
\hat y(x_0)=\arg\max_j P(Y=j\mid X=x_0)
\]

\[
\operatorname{Error}_{Bayes}=1-E\left[\max_j P(Y=j\mid X)\right]
\]

- \(j\): clase posible; la esperanza promedia sobre los valores posibles de \(X\).
- El error puede ser mayor que cero cuando las clases se solapan; es análogo al error irreducible.

### Probabilidad estimada por KNN

\[
\widehat{P}(Y=j\mid X=x_0)=\frac{1}{K}\sum_{i\in N_0}I(y_i=j)
\]

- \(K\): número de vecinos; \(N_0\): conjunto de los \(K\) puntos de entrenamiento más cercanos a \(x_0\).
- La clase predicha es la de mayor frecuencia estimada en el vecindario.

## 7. Implementación práctica

El laboratorio no implementa entrenamiento ni validación de modelos; introduce herramientas de Python para preparar una exploración. Por ello, no se añade código de modelado que el capítulo no proporciona.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

# Declarar el marcador de faltantes al leer evita inferir una columna numérica como object.
auto = pd.read_csv("Auto.data", na_values=["?"], sep=r"\s+")

print(auto.shape)
print(auto.dtypes)
print(auto[["mpg", "horsepower", "weight"]].describe())

# Una variable numérica con pocos niveles puede representar categorías.
auto["cylinders"] = pd.Series(auto["cylinders"], dtype="category")

fig, axes = subplots(ncols=2, figsize=(12, 5))
auto.plot.scatter("horsepower", "mpg", ax=axes[0])
auto.boxplot("mpg", by="cylinders", ax=axes[1])
```

**Nota práctica - adaptación moderna:** el capítulo usa `delim_whitespace=True`; el ejemplo anterior expresa la misma intención con `sep=r"\s+"`. Esta sustitución es una actualización de API y no una afirmación original del capítulo.

Para simulaciones reproducibles, el laboratorio usa un generador con semilla:

```python
rng = np.random.default_rng(1)
sample = rng.standard_normal((100, 3))
column_means = sample.mean(axis=0)
```

La separación entrenamiento/validación no aparece implementada en el laboratorio de este capítulo. La regla conceptual sí es explícita: ajustar con entrenamiento y evaluar generalización en observaciones no usadas para el ajuste.

## 8. Supuestos y condiciones de aplicación

1. En \(Y=f(X)+\epsilon\), el error se considera independiente de \(X\) y con media cero.
2. El límite de precisión viene dado por variación no explicable mediante los predictores observados.
3. Un modelo lineal supone que la respuesta puede representarse adecuadamente mediante una combinación lineal de predictores.
4. Los métodos no paramétricos necesitan suficientes observaciones para estimar una función flexible con precisión.
5. La evaluación debe hacerse en observaciones que no participaron en el ajuste.
6. El clasificador de Bayes presupone conocer las probabilidades condicionales verdaderas; en datos reales funciona como ideal teórico, no como procedimiento directamente disponible.
7. KNN requiere una definición de cercanía y datos etiquetados; su nivel de flexibilidad depende de \(K\).
8. El clustering se vuelve más difícil cuando los grupos se solapan y cuando la dimensionalidad impide inspección visual directa.
9. Los predictores cualitativos deben codificarse antes de aplicar muchos métodos, aunque este capítulo no desarrolla la codificación.
10. Los resúmenes y gráficos deben respetar el tipo real de cada variable; una columna numérica puede ser conceptualmente categórica.

## 9. Errores y riesgos

- Elegir un modelo por el menor error de entrenamiento: favorece métodos excesivamente flexibles y no mide generalización.
- Confundir buen ajuste con buen modelo: una curva que pasa por todos los puntos puede estar siguiendo ruido.
- Suponer que más flexibilidad siempre mejora la predicción: el error de prueba puede aumentar por varianza.
- Usar un modelo demasiado rígido ante una relación compleja: produce underfitting y sesgo alto.
- Interpretar una asociación predictiva como causalidad: el capítulo formula asociaciones, no identifica efectos causales.
- Tratar el clasificador de Bayes como implementable sin conocer la distribución condicional verdadera.
- Escoger \(K=1\) en KNN por lograr error de entrenamiento cero: puede generar una frontera inestable y error de prueba alto.
- Escoger un \(K\) muy grande: reduce demasiado la flexibilidad y puede imponer una frontera con sesgo alto.
- Comparar métodos usando el conjunto reservado para la estimación final: aunque el capítulo no desarrolla protocolos de competición, esto contaminaría la medición fuera de muestra. **Conclusión práctica derivada del principio de test error.**
- Cargar marcadores de faltantes como texto: puede convertir una columna numérica en `object` y afectar resúmenes o cálculos.
- Eliminar filas faltantes sin justificarlo: el laboratorio lo hace solo en un caso con cinco filas afectadas; no establece una regla general.
- Confundir índices de NumPy y pandas, o combinar listas de filas y columnas como si produjeran siempre una submatriz: el laboratorio muestra que el indexado avanzado puede devolver pares de elementos o errores de forma.
- Intentar identificar clusters solo con gráficos cuando hay muchas variables: existen \(p(p-1)/2\) pares, por lo que la inspección manual deja de ser viable.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo:** al recibir un dataset nuevo y antes de elegir algoritmo o métrica; también cuando el error de entrenamiento es mucho menor que el de validación.
- **Decisiones que permite tomar:** distinguir predicción de inferencia; clasificar la tarea; elegir una primera familia de modelos; decidir si se necesita interpretabilidad; razonar sobre flexibilidad, underfitting y overfitting.
- **Baseline útil:** para una respuesta cuantitativa, comenzar con un modelo paramétrico interpretable; para clasificación, usar una referencia simple y evaluar la tasa de error fuera de muestra. KNN es un candidato básico si la estructura local parece relevante, validando \(K\).
- **Mejoras potenciales:** aumentar flexibilidad si el baseline tiene sesgo aparente; reducirla si existe una brecha grande entre entrenamiento y validación; revisar tipos, faltantes, distribuciones y relaciones antes de ajustar.
- **Qué no implementar bajo presión:** splines u otros modelos flexibles sin un esquema fiable para estimar error de prueba; análisis exhaustivos de todos los pares de variables en alta dimensión; una búsqueda basada solo en el score de entrenamiento.
- **Señales que activan estas técnicas:** respuesta continua activa regresión y MSE; respuesta categórica activa clasificación y tasa de error; ausencia de etiquetas activa métodos no supervisados; relación visualmente no lineal y datos abundantes justifican comparar más flexibilidad; inestabilidad ante pequeñas variaciones o gran brecha train/validation sugieren varianza alta.
- **Recomendación para competencias:** reservar la evaluación fuera de muestra para comparar niveles de flexibilidad y evitar que el leaderboard o conjunto final se convierta en parte del ajuste. Esta es una derivación competitiva del principio del capítulo; el capítulo no prescribe un protocolo específico de competición.

## 11. Checklist operativo

- [ ] Definir unidad de observación, predictores y respuesta.
- [ ] Decidir si el objetivo es predicción, inferencia o ambos.
- [ ] Identificar aprendizaje supervisado, no supervisado o semisupervisado.
- [ ] Si hay respuesta, clasificarla como cuantitativa o cualitativa.
- [ ] Inspeccionar forma, tipos, nombres, valores únicos y faltantes.
- [ ] Corregir variables categóricas tratadas como numéricas y marcadores de faltantes tratados como texto.
- [ ] Establecer un baseline interpretable o poco flexible.
- [ ] Ajustar únicamente con datos de entrenamiento.
- [ ] Comparar alternativas con error fuera de muestra, no con error de entrenamiento.
- [ ] Revisar la brecha entre entrenamiento y validación como señal de overfitting.
- [ ] Aumentar flexibilidad solo si la reducción esperada de sesgo compensa la varianza.
- [ ] Para KNN, validar \(K\) y evitar elegirlo por el score de entrenamiento.
- [ ] Documentar qué parte del error parece reducible y qué limitaciones vienen de datos no observados o clases solapadas.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Distinción entre predicción e inferencia | `knowledge/problem_definition/prediction_vs_inference.md` | Alta |
| Supervisado, no supervisado, semisupervisado, regresión y clasificación | `knowledge/problem_definition/supervised_unsupervised_regression_classification.md` | Alta |
| Error reducible e irreducible | `knowledge/metrics/reducible_irreducible_error.md` | Media |
| MSE de entrenamiento frente a MSE de prueba | `knowledge/validation/training_vs_test_error.md` | Alta |
| Flexibilidad, overfitting y curva de error en U | `knowledge/validation/model_flexibility_and_overfitting.md` | Alta |
| Descomposición y diagnóstico sesgo-varianza | `knowledge/validation/bias_variance_tradeoff.md` | Alta |
| Error de clasificación y referencia de Bayes | `knowledge/metrics/classification_error_and_bayes_rate.md` | Alta |
| Funcionamiento y ajuste de \(K\) en KNN | `knowledge/models/knn_classification.md` | Alta |
| Métodos paramétricos frente a no paramétricos | `knowledge/models/parametric_vs_nonparametric.md` | Media |
| Inspección tabular, tipos, faltantes y gráficos básicos | `knowledge/eda/python_tabular_eda.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | 2.1–2.3; pp. 15–62 |
| 2. Relación entre entradas y salida; predicción e inferencia | 2.1, 2.1.1; pp. 15–20; ecuaciones 2.1–2.3 |
| 2. Familias de estimación | 2.1.2; pp. 20–23; ecuación 2.4; Figuras 2.4–2.6 |
| 2. Flexibilidad, precisión e interpretabilidad | 2.1.3; pp. 23–25; Figura 2.7 |
| 2. Tipos de aprendizaje y de respuesta | 2.1.4–2.1.5; pp. 25–27; Figura 2.8 |
| 2. Generalización y sobreajuste | 2.2.1; pp. 27–31; ecuaciones 2.5–2.6; Figuras 2.9–2.11 |
| 3–4. Flujo y reglas de decisión | Síntesis de 2.1–2.2; pp. 15–39 |
| 5. Modelado paramétrico y no paramétrico | 2.1.2; pp. 20–23 |
| 5. Bayes y KNN | 2.2.3; pp. 34–39; ecuaciones 2.8–2.12; Figuras 2.13–2.17 |
| 5 y 7. Inspección y EDA en Python | 2.3.3–2.3.9; pp. 42–62 |
| 6. Fórmulas esenciales | Ecuaciones 2.1–2.12 seleccionadas; pp. 15–39 |
| 8. Supuestos y condiciones | 2.1–2.2.3; pp. 15–39 |
| 9. Errores y riesgos | 2.1.2–2.2.3 y laboratorio 2.3.6–2.3.9; pp. 20–39 y 51–62 |
| 10–11. Aplicación y checklist | Síntesis práctica derivada de 2.1–2.3; pp. 15–62 |
| Ejercicios revisados pero no incorporados como contenido | 2.4; pp. 63–67 |

Todas las páginas del PDF (páginas impresas 15–67) pudieron procesarse. No se usaron fuentes externas ni se incorporaron soluciones de los ejercicios.
