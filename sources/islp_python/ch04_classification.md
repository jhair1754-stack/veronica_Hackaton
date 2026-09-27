---
title: "Classification"
topics:
  - "logistic and multinomial logistic regression"
  - "LDA, QDA, and naive Bayes"
  - "K-nearest neighbors classification"
  - "classification thresholds, confusion matrices, ROC, and AUC"
  - "generalized linear models and Poisson regression"
target_knowledge_files:
  - "knowledge/models/logistic_regression.md"
  - "knowledge/models/discriminant_analysis.md"
  - "knowledge/models/naive_bayes.md"
  - "knowledge/models/knn_classification.md"
  - "knowledge/metrics/classification_metrics.md"
  - "knowledge/models/poisson_regression.md"
---

# 4. Classification

## 1. Propósito del capítulo

El capítulo presenta métodos para predecir respuestas cualitativas y, en muchos casos, estimar la probabilidad de pertenencia a cada clase. Desarrolla regresión logística binaria y multinomial, modelos generativos basados en Bayes -LDA, QDA y Naive Bayes-, y compara estos enfoques con K-nearest neighbors (KNN). También explica cómo convertir probabilidades en decisiones mediante umbrales, cómo evaluar los errores con matrices de confusión, ROC y AUC, y amplía la idea de regresión a respuestas de conteo mediante modelos lineales generalizados (generalized linear models, GLM) y regresión de Poisson.

**Convención de procedencia:** salvo donde se indique **“Conclusión práctica derivada”**, **“Síntesis práctica derivada del capítulo”**, **“Recomendación para competencias”** o **“Nota práctica - adaptación moderna”**, el contenido siguiente parafrasea afirmaciones explícitas del capítulo.

## 2. Conceptos esenciales

### Clasificación y probabilidades de clase

Una respuesta cualitativa toma valores en categorías o clases. Un clasificador puede asignar directamente una clase, pero muchos métodos primero estiman las probabilidades posteriores \(P(Y=k\mid X=x)\) y luego eligen la clase con mayor probabilidad o aplican un umbral dependiente del problema.

La regresión lineal no es apropiada como clasificador general. Con más de dos clases, una codificación numérica impone un orden y distancias artificiales entre categorías. Con una respuesta binaria 0/1, el ajuste lineal puede producir valores fuera de \([0,1]\), que no son probabilidades válidas. La regresión logística evita ese problema.

### Regresión logística

La regresión logística binaria modela \(p(X)=P(Y=1\mid X)\) con una función logística. El logit o log-odds es lineal en los predictores. Un incremento de una unidad en \(X_j\), manteniendo el resto fijo, suma \(\beta_j\) al log-odds y multiplica los odds por \(e^{\beta_j}\). No produce un cambio constante en la probabilidad: ese cambio depende del punto actual de la curva.

Los coeficientes se estiman por máxima verosimilitud (maximum likelihood). El estadístico \(z=\hat\beta_j/SE(\hat\beta_j)\) contrasta \(H_0:\beta_j=0\). Un coeficiente significativo describe asociación condicionada a los demás predictores, no causalidad ni necesariamente utilidad predictiva fuera de muestra.

El ejemplo `Default` muestra confounding: `student` tiene un coeficiente positivo cuando se usa solo, pero negativo al controlar por `balance` e `income`. Los estudiantes presentan mayores balances en conjunto, y el balance está fuertemente asociado al default. Por ello, una asociación marginal puede invertir su signo en un modelo múltiple.

Para \(K>2\), la regresión logística multinomial modela los log-odds de cada clase frente a una clase base. Cambiar la referencia modifica los coeficientes, pero no las probabilidades ajustadas, los log-odds entre pares de clases ni las predicciones. La codificación softmax trata simétricamente a las clases y es equivalente en esos resultados.

### Modelos generativos y teorema de Bayes

Los modelos generativos estiman, para cada clase, la distribución de los predictores \(f_k(x)=P(X=x\mid Y=k)\) y la probabilidad previa \(\pi_k=P(Y=k)\). El teorema de Bayes transforma esas cantidades en la probabilidad posterior \(P(Y=k\mid X=x)\). El clasificador asigna la clase con la posterior más alta.

- **LDA (linear discriminant analysis):** supone normalidad multivariante dentro de cada clase, medias específicas por clase y una matriz de covarianza común. Produce fronteras lineales.
- **QDA (quadratic discriminant analysis):** mantiene la normalidad, pero permite una matriz de covarianza distinta por clase. Produce fronteras cuadráticas y es más flexible.
- **Naive Bayes:** supone independencia de los predictores condicionada a la clase y factoriza la densidad conjunta en densidades univariantes. Puede usar Gaussianas, estimaciones no paramétricas o proporciones para predictores cualitativos.

LDA puede ser más estable que regresión logística cuando existe separación fuerte entre clases, y puede ganar precisión con muestras pequeñas si la normalidad por clase es razonable. QDA reduce sesgo cuando las covarianzas difieren, pero estima muchos más parámetros. Naive Bayes introduce sesgo al ignorar dependencias, pero reduce varianza, lo que puede ser útil cuando \(p\) es grande respecto de \(n\).

### Umbrales, errores y métricas

Elegir la clase de mayor probabilidad equivale, en el caso binario y con costes simétricos, a usar umbral 0.5. El capítulo muestra que reducir el umbral aumenta la detección de positivos verdaderos, pero también los falsos positivos. La elección correcta depende del coste de ambos errores y del conocimiento del dominio.

La matriz de confusión separa verdaderos positivos (TP), falsos positivos (FP), verdaderos negativos (TN) y falsos negativos (FN). De ella se derivan:

- tasa de falsos positivos: \(FP/N=1-\text{specificity}\);
- tasa de verdaderos positivos: \(TP/P=\text{sensitivity}=\text{recall}\);
- valor predictivo positivo: \(TP/P^*=\text{precision}\);
- valor predictivo negativo: \(TN/N^*\).

La curva ROC representa sensibilidad frente a tasa de falsos positivos para todos los umbrales. El área bajo la curva (AUC) resume esa capacidad de ordenamiento: 1 es ideal y 0.5 corresponde a un clasificador sin información cuando se evalúa en datos independientes.

La exactitud global puede ser engañosa en clases desbalanceadas. En el laboratorio `Caravan`, predecir siempre la clase mayoritaria supera en error global a KNN con \(K=1\), aunque KNN identifica un subconjunto de clientes con tasa de compra mayor que la tasa base. La métrica debe representar la decisión de negocio.

### Comparación entre clasificadores

LDA y regresión logística producen log-odds lineales, pero llegan a ellos de forma distinta: LDA modela la distribución de \(X\) dentro de cada clase; la logística modela directamente la distribución condicional de \(Y\mid X\). Si la normalidad por clase es aproximadamente correcta, LDA puede ser mejor; si no, la logística puede ser más robusta.

QDA agrega términos cuadráticos e interacciones en la frontera. Naive Bayes puede crear efectos univariantes flexibles, pero su forma es aditiva y no representa interacciones entre predictores. KNN no impone forma paramétrica, por lo que puede capturar fronteras complejas si hay muchas observaciones y pocos predictores; paga esa flexibilidad con mayor varianza y pérdida de interpretabilidad.

No existe un método dominante. Fronteras lineales favorecen LDA o logística; fronteras moderadamente no lineales pueden favorecer QDA o Naive Bayes; fronteras muy complejas pueden favorecer KNN si se elige bien \(K\) y \(n\) es grande respecto de \(p\).

### GLM y regresión de Poisson

Los conteos no negativos no encajan naturalmente en regresión lineal: pueden aparecer predicciones negativas, la varianza suele crecer con la media y la respuesta es discreta. La regresión de Poisson supone una respuesta Poisson cuya media \(\lambda(X)\) se relaciona con los predictores mediante un enlace logarítmico. Así, las predicciones son no negativas y un incremento unitario en \(X_j\) multiplica la media por \(e^{\beta_j}\).

Regresión lineal, logística y Poisson son GLM. Cada una combina una distribución de la familia exponencial con una función de enlace que transforma la media condicional para hacerla lineal en los predictores: identidad para Gaussiana, logit para Bernoulli y log para Poisson.

En Poisson, \(E(Y\mid X)=Var(Y\mid X)=\lambda(X)\). Si la varianza es mucho mayor que la media existe overdispersion; el capítulo advierte que esto infla los estadísticos \(z\) y exige un tratamiento más cuidadoso, aunque no desarrolla sus soluciones.

## 3. Flujo o procedimiento presentado

1. **Identificar el tipo de respuesta.** Usar clasificación para categorías, regresión logística binaria para dos clases, multinomial o clasificadores multiclase para más clases, y Poisson para conteos no negativos cuando su relación media-varianza sea plausible.
2. **Separar entrenamiento y evaluación.** En datos temporales, respetar el orden: el laboratorio entrena con 2001-2004 y evalúa 2005. No medir generalización sobre los datos usados para ajustar.
3. **Explorar prevalencia y estructura.** Revisar balance de clases, asociaciones marginales, correlación entre predictores y posibles diferencias de covarianza entre clases.
4. **Construir un baseline probabilístico.** Ajustar regresión logística y obtener probabilidades, no solo etiquetas.
5. **Interpretar condicionalmente.** Leer coeficientes como cambios en log-odds u odds manteniendo los demás predictores fijos; comparar con asociaciones marginales para detectar confounding.
6. **Probar supuestos estructurales.** Considerar LDA si la frontera parece lineal y las covarianzas son comparables; QDA si las covarianzas difieren y hay suficiente muestra; Naive Bayes si interesa reducir varianza mediante independencia condicional.
7. **Preparar KNN.** Estandarizar predictores numéricos y escoger \(K\); la distancia no debe quedar dominada por las unidades de una variable.
8. **Convertir probabilidades en decisiones.** Evaluar varios umbrales de acuerdo con el coste de FP y FN; no adoptar 0.5 por defecto si los costes son asimétricos.
9. **Evaluar fuera de muestra.** Examinar matriz de confusión y métricas relevantes; usar ROC/AUC para comparar la discriminación a través de umbrales.
10. **Comparar complejidad.** Contrastar métodos lineales, cuadráticos y no paramétricos con la misma partición de evaluación; elegir según error de prueba y objetivo operativo.
11. **Para conteos, verificar Poisson.** Confirmar valores enteros no negativos, revisar la relación media-varianza y comprobar overdispersion antes de confiar en inferencia.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Respuesta binaria y frontera aproximadamente lineal | Regresión logística | Probabilidades válidas e interpretación por log-odds | Un efecto marginal puede cambiar al controlar confusores |
| Más de dos clases sin orden natural | Regresión logística multinomial | Modela probabilidades de todas las clases sin imponer orden | Interpretar coeficientes respecto de la clase base |
| Normalidad aproximada por clase, covarianza común y muestra modesta | LDA | Frontera lineal con menor varianza que QDA | La covarianza común puede introducir sesgo si es falsa |
| Covarianzas claramente distintas y muestra grande | QDA | Captura fronteras cuadráticas y variación específica por clase | Estima \(Kp(p+1)/2\) parámetros de covarianza |
| \(p\) grande respecto de \(n\) y dependencia condicional débil | Naive Bayes | La independencia reduce drásticamente la varianza | Correlaciones fuertes dentro de clase pueden degradarlo |
| Frontera muy no lineal, \(n\) grande y \(p\) pequeño | KNN | No impone forma a la frontera | Estandarizar y elegir \(K\); sufre con alta dimensión |
| Frontera no lineal con muestra solo moderada | QDA antes que KNN | Impone una forma paramétrica y requiere menos datos | Solo si una frontera cuadrática es razonable |
| Clases separadas casi por completo | LDA u otro generativo | La logística puede tener coeficientes inestables | Verificar si los supuestos generativos son razonables |
| Coste de falsos negativos superior al de falsos positivos | Reducir el umbral de la clase positiva | Aumenta sensibilidad/recall | Disminuye especificidad y puede elevar el error total |
| Clases desbalanceadas | Matriz de confusión, recall, precision y tasa base | Accuracy puede favorecer siempre la clase mayoritaria | Elegir métricas según la acción posterior |
| Comparar discriminación sin fijar un umbral | ROC y AUC en validación/test | Resume el ranking a través de umbrales | No calcular como evidencia final sobre entrenamiento |
| Conteo entero no negativo con varianza ligada a la media | Regresión de Poisson | Predicciones no negativas y enlace natural para conteos | Revisar overdispersion; Poisson exige media=varianza |

## 5. Métodos y técnicas

### Regresión logística binaria

- **Objetivo:** estimar \(P(Y=1\mid X)\) y clasificar una respuesta binaria.
- **Entradas:** matriz de predictores numéricos/codificados y etiquetas 0/1.
- **Salida:** probabilidades, log-odds, odds y etiquetas tras aplicar un umbral.
- **Funcionamiento:** ajusta una combinación lineal en la escala logit mediante máxima verosimilitud.
- **Hiperparámetros relevantes:** el modelo clásico no presenta uno; el umbral de decisión es una regla posterior al ajuste.
- **Ventajas:** probabilidades en \([0,1]\), interpretación de coeficientes e inferencia con errores estándar y tests \(z\).
- **Limitaciones:** frontera lineal en la representación usada; puede ser inestable bajo separación fuerte; no resuelve confounding por sí sola.
- **Casos adecuados:** clasificación binaria con necesidad de probabilidades o interpretabilidad.
- **No conviene:** usarla sin extensiones cuando la frontera es muy no lineal o interpretar coeficientes como causalidad.

### Regresión logística multinomial

- **Objetivo:** extender la logística a \(K>2\) clases no ordenadas.
- **Entradas:** predictores y etiqueta multiclase.
- **Salida:** una probabilidad por clase que suma uno y una clase predicha.
- **Funcionamiento:** modela los log-odds de \(K-1\) clases respecto de una base, o usa una parametrización softmax simétrica.
- **Hiperparámetros relevantes:** clase de referencia para interpretación; no cambia las predicciones.
- **Ventajas:** extensión natural, probabilística y multiclase.
- **Limitaciones:** coeficientes dependientes de la referencia y frontera lineal en los predictores.
- **Casos adecuados:** categorías nominales múltiples.
- **No conviene:** codificar clases nominales como 1, 2, 3 y usar regresión lineal.

### LDA

- **Objetivo:** aproximar el clasificador de Bayes con fronteras lineales.
- **Entradas:** predictores, clases y, opcionalmente, probabilidades previas.
- **Salida:** funciones discriminantes, probabilidades posteriores y clase.
- **Funcionamiento:** estima medias por clase y una covarianza común bajo normalidad multivariante.
- **Hiperparámetros relevantes:** priors de clase cuando se conocen o se desean modificar.
- **Ventajas:** baja varianza relativa, multiclase natural y estabilidad con muestras pequeñas bajo supuestos razonables.
- **Limitaciones:** sesgo si las covarianzas difieren o la normalidad es muy inadecuada.
- **Casos adecuados:** frontera lineal, covarianzas similares, muestra limitada.
- **No conviene:** covarianzas muy diferentes y suficiente evidencia de frontera no lineal.

### QDA

- **Objetivo:** permitir fronteras cuadráticas entre clases.
- **Entradas:** predictores y clases.
- **Salida:** probabilidades posteriores y clase.
- **Funcionamiento:** estima una media y una covarianza completa para cada clase.
- **Hiperparámetros relevantes:** el capítulo no introduce uno; la flexibilidad deriva de covarianzas específicas.
- **Ventajas:** captura diferencias de forma y orientación entre distribuciones de clase.
- **Limitaciones:** alta varianza y gran coste paramétrico al crecer \(p\).
- **Casos adecuados:** conjunto grande o covarianzas claramente distintas.
- **No conviene:** pocas observaciones por clase o muchos predictores.

### Naive Bayes

- **Objetivo:** estimar probabilidades posteriores mediante densidades univariantes por clase.
- **Entradas:** predictores cuantitativos o cualitativos y clases.
- **Salida:** probabilidades posteriores y clase.
- **Funcionamiento:** supone independencia condicional y multiplica las contribuciones de cada predictor.
- **Hiperparámetros relevantes:** elección de densidad por predictor: Gaussiana, histograma/kernel o proporciones categóricas.
- **Ventajas:** pocos parámetros, baja varianza y buen comportamiento potencial con \(p\) relativamente grande.
- **Limitaciones:** no modela interacciones; la dependencia condicional fuerte viola su supuesto central.
- **Casos adecuados:** muestra escasa respecto de la dimensión o predictores aproximadamente independientes dentro de clase.
- **No conviene:** interacciones o correlaciones dentro de clase esenciales para separar clases.

### KNN para clasificación

- **Objetivo:** clasificar según la pluralidad entre los vecinos más cercanos.
- **Entradas:** predictores escalados, etiquetas, métrica de distancia y \(K\).
- **Salida:** clase local y, según implementación, proporción de vecinos por clase.
- **Funcionamiento:** identifica los \(K\) puntos de entrenamiento más próximos al nuevo caso.
- **Hiperparámetro principal:** \(K\); un valor pequeño reduce sesgo y aumenta varianza, y uno grande suaviza.
- **Ventajas:** frontera flexible sin suposición paramétrica.
- **Limitaciones:** sensible a escala, alta dimensión, variables irrelevantes y tamaño de muestra; no entrega coeficientes de importancia.
- **Casos adecuados:** baja dimensión, abundantes observaciones y frontera compleja.
- **No conviene:** \(n\) pequeño respecto de \(p\), predictores sin escalar o necesidad prioritaria de interpretación.

### Regresión de Poisson

- **Objetivo:** modelar la media de una respuesta de conteo.
- **Entradas:** conteos no negativos y predictores.
- **Salida:** media condicional no negativa y distribución de conteo.
- **Funcionamiento:** supone \(Y\mid X\sim Poisson(\lambda(X))\) y \(\log\lambda(X)=X\beta\); estima por máxima verosimilitud.
- **Hiperparámetros relevantes:** no se presenta uno en el capítulo; la familia Poisson y el enlace log son decisiones del modelo.
- **Ventajas:** respeta el soporte de los conteos y permite que varianza y media cambien con \(X\).
- **Limitaciones:** impone \(Var(Y\mid X)=E(Y\mid X)\); la overdispersion invalida los \(z\) ingenuos.
- **Casos adecuados:** eventos contados por unidad de observación con media dependiente de covariables.
- **No conviene:** tratar como Poisson sin revisar exceso de dispersión o usar OLS cuando genera conteos negativos.

## 6. Fórmulas esenciales

### Logística binaria, odds y logit

\[
p(X)=P(Y=1\mid X)=\frac{e^{\beta_0+\sum_{j=1}^{p}\beta_jX_j}}
{1+e^{\beta_0+\sum_{j=1}^{p}\beta_jX_j}}
\]

\[
\frac{p(X)}{1-p(X)}=e^{\beta_0+\sum_j\beta_jX_j},
\qquad
\log\left(\frac{p(X)}{1-p(X)}\right)=\beta_0+\sum_j\beta_jX_j
\]

- \(p(X)\): probabilidad posterior de la clase 1; \(\beta_0\): intercepto; \(\beta_j\): cambio en log-odds por unidad de \(X_j\), manteniendo el resto fijo.
- \(e^{\beta_j}\) es el multiplicador de los odds por una unidad adicional de \(X_j\).
- La linealidad se impone sobre el logit, no sobre la probabilidad.

### Verosimilitud logística

\[
\ell(\beta)=\prod_{i:y_i=1}p(x_i)\prod_{i:y_i=0}[1-p(x_i)]
\]

Los coeficientes se eligen para maximizar la probabilidad asignada a las clases observadas. El ajuste supone observaciones tratadas como independientes bajo el modelo.

### Logística multinomial

Con la clase \(K\) como referencia:

\[
\log\left(\frac{P(Y=k\mid X=x)}{P(Y=K\mid X=x)}\right)
=\beta_{k0}+\sum_{j=1}^{p}\beta_{kj}x_j,
\quad k=1,\ldots,K-1
\]

Cada \(\beta_{kj}\) modifica el log-odds de la clase \(k\) frente a la referencia. La referencia cambia los coeficientes, no las probabilidades ni la clasificación.

### Bayes para clasificación generativa

\[
P(Y=k\mid X=x)=\frac{\pi_k f_k(x)}{\sum_{l=1}^{K}\pi_l f_l(x)}
\]

- \(\pi_k\): probabilidad previa de la clase \(k\).
- \(f_k(x)\): densidad de los predictores dentro de la clase \(k\).
- Se asigna la clase con mayor probabilidad posterior si los costes son simétricos.

### Función discriminante de LDA

Para \(p>1\), con media \(\mu_k\) y covarianza común \(\Sigma\):

\[
\delta_k(x)=x^T\Sigma^{-1}\mu_k-
\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+\log(\pi_k)
\]

La clase elegida maximiza \(\delta_k(x)\). La frontera es lineal porque los términos cuadráticos comunes se cancelan.

### Función discriminante de QDA

\[
\delta_k(x)=-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
-\frac{1}{2}\log|\Sigma_k|+\log(\pi_k)
\]

Cada clase tiene su propia \(\Sigma_k\); por ello permanecen términos cuadráticos en \(x\) y la frontera puede ser curva.

### Factorización de Naive Bayes

\[
f_k(x)=\prod_{j=1}^{p}f_{kj}(x_j),
\qquad
P(Y=k\mid X=x)=
\frac{\pi_k\prod_j f_{kj}(x_j)}
{\sum_l\pi_l\prod_j f_{lj}(x_j)}
\]

\(f_{kj}\) es la densidad o probabilidad del predictor \(j\) dentro de la clase \(k\). La fórmula depende del supuesto de independencia condicional.

### Métricas de clasificación

\[
TPR=\frac{TP}{P}=\frac{TP}{TP+FN},
\quad
FPR=\frac{FP}{N}=\frac{FP}{FP+TN},
\quad
Precision=\frac{TP}{TP+FP}
\]

TPR es sensibilidad/recall; FPR es \(1-\)especificidad; precision mide qué proporción de los positivos predichos es realmente positiva.

### Poisson y enlace log

\[
P(Y=k)=\frac{e^{-\lambda}\lambda^k}{k!},
\qquad E(Y)=Var(Y)=\lambda
\]

\[
\log\lambda(X)=\beta_0+\sum_{j=1}^{p}\beta_jX_j,
\qquad
\lambda(X)=e^{\beta_0+\sum_j\beta_jX_j}
\]

\(Y\) es un conteo, \(k\in\{0,1,2,\ldots\}\), y \(\lambda(X)\) es su media condicional. \(e^{\beta_j}\) es el factor multiplicativo sobre la media por una unidad de \(X_j\).

### Estructura general de un GLM

\[
\eta(E[Y\mid X])=\beta_0+\sum_{j=1}^{p}\beta_jX_j
\]

\(\eta\) es la función de enlace: identidad para regresión lineal, logit para logística y log para Poisson.

## 7. Implementación práctica

El laboratorio usa `statsmodels.api.GLM` con familias `Binomial` y `Poisson`; `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `GaussianNB`, `KNeighborsClassifier` y `StandardScaler` de `scikit-learn`.

**Nota práctica - adaptación moderna:** el siguiente ejemplo encapsula el escalado de KNN dentro de un `Pipeline`, de modo que el escalador se ajusta solo con entrenamiento. El capítulo muestra `StandardScaler` y la división de datos, pero los aplica en ese orden fuera de un pipeline; encapsularlos evita leakage en un flujo reutilizable.

```python
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "lda": LinearDiscriminantAnalysis(),
    "qda": QuadraticDiscriminantAnalysis(),
    "naive_bayes": GaussianNB(),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    positive_probability = model.predict_proba(X_valid)[:, 1]
    predicted = (positive_probability >= 0.30).astype(int)
    print(name, confusion_matrix(y_valid, predicted))
    print("AUC:", roc_auc_score(y_valid, positive_probability))
```

El umbral `0.30` es ilustrativo: debe elegirse a partir del coste de errores y con datos de validación. Para series temporales, el laboratorio respalda una partición cronológica en lugar de una división aleatoria.

Poisson con `statsmodels`:

```python
import statsmodels.api as sm

X_train_with_intercept = sm.add_constant(X_train, has_constant="add")
poisson_model = sm.GLM(
    y_train,
    X_train_with_intercept,
    family=sm.families.Poisson(),
).fit()

X_valid_with_intercept = sm.add_constant(X_valid, has_constant="add")
expected_count = poisson_model.predict(X_valid_with_intercept)
```

Los ejemplos modernos usan `random_state` donde hay una partición aleatoria. En métodos deterministas como LDA, QDA o regresión logística, la semilla no modifica el estimador una vez fijados los datos.

## 8. Supuestos y condiciones de aplicación

1. Las observaciones usadas en las verosimilitudes logística y Poisson se tratan como independientes.
2. La regresión logística impone linealidad de los log-odds en las columnas de diseño; la probabilidad no es lineal.
3. La interpretación de cada coeficiente es condicional a los demás predictores incluidos.
4. LDA supone normalidad multivariante dentro de clase y una covarianza común.
5. QDA supone normalidad multivariante dentro de clase, pero permite covarianza específica por clase.
6. Naive Bayes supone independencia de predictores dentro de cada clase.
7. KNN requiere una métrica de distancia significativa, escalas comparables, pocos predictores respecto de la muestra y suficientes vecinos locales.
8. El umbral 0.5 minimiza el error global solo bajo la regla Bayes de costes simétricos; otros objetivos justifican otros umbrales.
9. ROC/AUC y métricas de rendimiento deben evaluarse en observaciones no utilizadas para ajustar el modelo.
10. La regresión de Poisson exige una respuesta entera no negativa y supone igualdad condicional de media y varianza.
11. La sobre-dispersión vuelve demasiado optimistas los estadísticos \(z\) del Poisson básico.
12. Las probabilidades previas estimadas con frecuencias de entrenamiento representan a la población solo si la muestra es adecuada; el capítulo permite introducir priors conocidos.

## 9. Errores y riesgos

- Codificar clases nominales múltiples como números y aplicar regresión lineal: impone orden y distancias inexistentes.
- Interpretar una predicción lineal para una respuesta 0/1 como probabilidad aunque esté fuera de \([0,1]\).
- Confundir un odds ratio con un cambio constante en probabilidad.
- Interpretar asociaciones como efectos causales.
- Ajustar un modelo univariado ignorando confusores correlacionados: el signo puede cambiar en el modelo múltiple.
- Evaluar sobre entrenamiento: el laboratorio muestra que 52.2% de accuracy aparente cae a 48% en el período futuro.
- Seleccionar variables o umbrales usando el conjunto final de prueba. **Conclusión práctica derivada:** esas decisiones deben hacerse con entrenamiento/validación y reservar test para la estimación final.
- Elegir 0.5 por costumbre cuando los costes de FP y FN son diferentes.
- Usar accuracy en clases desbalanceadas sin compararla con la tasa nula y sin revisar precision/recall.
- Comparar AUC calculada en entrenamiento como si fuera rendimiento de generalización.
- Usar QDA con pocos casos: las covarianzas por clase introducen alta varianza o pueden estimarse mal.
- Usar Naive Bayes cuando la dependencia entre predictores dentro de clase es fuerte y decisiva.
- Estandarizar antes de dividir los datos. **Nota práctica - adaptación moderna:** aunque el laboratorio escala antes de `train_test_split`, en una evaluación estricta el escalador debe ajustarse solo con entrenamiento para evitar leakage.
- Aplicar KNN a variables con unidades muy diferentes o con \(p\) grande: la distancia pierde significado y aparece la maldición de la dimensionalidad.
- Elegir \(K=1\) porque ajusta mejor entrenamiento: puede producir varianza alta; el capítulo muestra que un suavizado mal elegido arruina incluso una frontera no lineal.
- Inferir importancia de predictores desde KNN: el método no entrega una tabla de coeficientes.
- Aplicar OLS a conteos y aceptar predicciones negativas o varianza constante.
- Aplicar logaritmo directo a conteos con ceros: la transformación no está definida.
- Ignorar overdispersion en Poisson: infla \(z\) y vuelve poco fiable la inferencia.
- Tomar los ejercicios sin solución como evidencia adicional: se revisaron solo para delimitar el capítulo y no se usaron para afirmar resultados.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo:** ante clasificación tabular binaria o multiclase, clases desbalanceadas, necesidad de probabilidades, ajuste de umbral o respuestas de conteo.
- **Decisiones que permite tomar:** elegir entre logística, LDA, QDA, Naive Bayes y KNN; seleccionar métricas; justificar un umbral; reconocer cuándo Poisson es preferible a OLS.
- **Baseline:** regresión logística con una separación correcta y matriz de confusión. Añadir ROC/AUC si importa el ranking y reportar siempre la tasa base.
- **Mejoras rápidas:** controlar predictores correlacionados relevantes; probar LDA para una frontera lineal, QDA para covarianzas diferentes con suficiente muestra, Naive Bayes con \(p/n\) alto y KNN escalado en baja dimensión.
- **Clases desbalanceadas:** entregar probabilidades, variar el umbral y reportar precision/recall además de accuracy. La utilidad depende de la clase que activa la acción.
- **Datos temporales:** dividir cronológicamente. Una división aleatoria podría mezclar futuro y pasado.
- **Qué no implementar bajo presión:** derivar densidades manualmente, estimar QDA con alta dimensión y pocos casos, barrer muchos umbrales sobre test o desplegar KNN sin pipeline de escalado.
- **Señales que activan técnicas:** respuesta nominal activa clasificación; más de dos clases activa multinomial/LDA/QDA/NB; covarianzas distintas activan QDA; independencia aproximada y muestra pequeña activan Naive Bayes; frontera compleja con muchos datos y baja dimensión activa KNN; conteos no negativos con varianza creciente activan Poisson.
- **Recomendación para competencias:** comparar varios modelos con exactamente la misma validación y conservar probabilidades. Elegir el modelo por la métrica alineada con el objetivo, no por accuracy aislada.

## 11. Checklist operativo

- [ ] Confirmar si la respuesta es binaria, multiclase o conteo.
- [ ] Medir prevalencia de clases y tasa nula.
- [ ] Separar entrenamiento y validación antes de cualquier transformación aprendida.
- [ ] Usar partición cronológica si existe orden temporal.
- [ ] Ajustar una regresión logística como baseline probabilístico.
- [ ] Revisar asociaciones marginales y condicionadas para detectar confounding.
- [ ] Estandarizar dentro del pipeline antes de KNN.
- [ ] Probar valores de \(K\) sin consultar el test final.
- [ ] Comparar LDA y QDA según muestra y covarianzas.
- [ ] Usar Naive Bayes cuando la reducción de varianza compense su supuesto de independencia.
- [ ] Examinar matriz de confusión, precision, recall y tasa de falsos positivos.
- [ ] Comparar ROC/AUC solo fuera de muestra.
- [ ] Elegir umbral según costes de FP/FN y documentarlo.
- [ ] Para conteos, comprobar predicciones no negativas y overdispersion.
- [ ] Reservar el conjunto de test para una única evaluación final.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Regresión logística binaria, odds y logit | `knowledge/models/logistic_regression.md` | Alta |
| Regresión logística multinomial y softmax | `knowledge/models/multinomial_logistic_regression.md` | Alta |
| Confounding e interpretación condicional | `knowledge/interpretability/conditional_vs_marginal_effects.md` | Alta |
| LDA y supuestos Gaussianos | `knowledge/models/linear_discriminant_analysis.md` | Alta |
| QDA y sesgo-varianza | `knowledge/models/quadratic_discriminant_analysis.md` | Alta |
| Naive Bayes e independencia condicional | `knowledge/models/naive_bayes.md` | Alta |
| KNN para clasificación y escalado | `knowledge/models/knn_classification.md` | Alta |
| Matriz de confusión y métricas derivadas | `knowledge/metrics/classification_metrics.md` | Alta |
| Selección de umbral según costes | `knowledge/metrics/classification_thresholds.md` | Alta |
| Curvas ROC y AUC | `knowledge/metrics/roc_auc.md` | Alta |
| Validación temporal y error de entrenamiento | `knowledge/validation/holdout_and_temporal_splits.md` | Alta |
| Comparación de fronteras lineales, cuadráticas y no paramétricas | `knowledge/models/classifier_selection.md` | Alta |
| GLM y funciones de enlace | `knowledge/models/generalized_linear_models.md` | Media |
| Regresión de Poisson y overdispersion | `knowledge/models/poisson_regression.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1–2. Definición de clasificación | 4.1; pp. 135–136 |
| 2 y 9. Por qué no usar regresión lineal | 4.2; pp. 136–138; Figura 4.2 |
| 2, 5 y 6. Modelo logístico, odds, logit y máxima verosimilitud | 4.3.1–4.3.2; pp. 138–141; ecuaciones 4.1–4.5; Tabla 4.1 |
| 2 y 5. Predicción y predictores cualitativos | 4.3.3; pp. 141–142; Tabla 4.2 |
| 2, 5 y 9. Logística múltiple y confounding | 4.3.4; pp. 142–144; ecuaciones 4.6–4.9; Tabla 4.3; Figura 4.3 |
| 2, 5 y 6. Logística multinomial y softmax | 4.3.5; pp. 144–146; ecuaciones 4.10–4.14 |
| 2 y 6. Modelos generativos y Bayes | 4.4; pp. 146–147; ecuación 4.15 |
| 2, 5 y 6. LDA univariante | 4.4.1; pp. 147–149; ecuaciones 4.16–4.22; Figura 4.4 |
| 2, 5 y 6. LDA multivariante | 4.4.2; pp. 150–153; ecuaciones 4.23–4.25; Figuras 4.5–4.6; Tablas 4.4–4.5 |
| 2, 4 y 6. Umbrales, matriz de confusión, ROC y AUC | 4.4.2; pp. 153–156; ecuaciones 4.26–4.27; Figuras 4.7–4.8; Tablas 4.5–4.7 |
| 2, 4–6. QDA y sesgo-varianza | 4.4.3; pp. 156–157; ecuación 4.28; Figura 4.9 |
| 2, 4–6. Naive Bayes | 4.4.4; pp. 158–161; ecuaciones 4.29–4.30; Figura 4.10; Tablas 4.8–4.9 |
| 2, 4 y 5. Comparación analítica de métodos | 4.5.1; pp. 161–164; ecuaciones 4.31–4.34 |
| 2, 4, 5 y 10. Comparación empírica | 4.5.2; pp. 164–167; Figuras 4.11–4.12 |
| 2, 5, 6 y 8. Conteos, Poisson y GLM | 4.6–4.6.3; pp. 167–173; ecuaciones 4.35–4.42; Tablas 4.10–4.11; Figuras 4.13–4.15 |
| 3, 7 y 9. Logistic regression y separación temporal en Python | 4.7.1–4.7.2; pp. 173–179 |
| 3 y 7. LDA, QDA y Naive Bayes en Python | 4.7.3–4.7.5; pp. 179–183 |
| 3, 7, 9 y 10. KNN, escalado, tasa nula y tuning | 4.7.6; pp. 183–188 |
| 7. Regresión lineal y Poisson en Python | 4.7.7; pp. 188–193 |
| 10–11. Aplicación y checklist | Síntesis práctica derivada de 4.1–4.7; pp. 135–193 |
| Ejercicios revisados pero no incorporados como contenido | 4.8; pp. 193–199 |

Todas las páginas del PDF (páginas impresas 135–199) pudieron procesarse. Los ejercicios de las pp. 193–199 se revisaron únicamente para delimitar el capítulo y no se incorporaron por carecer de solución. No se usaron fuentes externas.
