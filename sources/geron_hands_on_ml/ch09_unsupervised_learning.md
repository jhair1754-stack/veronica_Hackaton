---
title: "Unsupervised Learning Techniques"
topics:
  - "Clustering"
  - "K-Means y Mini-Batch K-Means"
  - "DBSCAN y clustering basado en densidad"
  - "Gaussian mixture models"
  - "Anomaly detection y novelty detection"
  - "Semi-supervised learning y active learning"
target_knowledge_files:
  - "knowledge/models/clustering.md"
  - "knowledge/models/anomaly_detection.md"
  - "knowledge/preprocessing/clustering_as_transformation.md"
  - "knowledge/validation/unsupervised_model_selection.md"
  - "knowledge/models/semi_supervised_learning.md"
---

# 9. Unsupervised Learning Techniques

## 1. Propósito del capítulo

El capítulo presenta técnicas para aprender de datos sin etiquetas mediante tres tareas principales: **clustering**, **anomaly detection** y **density estimation**. Desarrolla K-Means, DBSCAN y Gaussian mixture models (GMM), y muestra cómo reutilizar el clustering para reducción no lineal de dimensionalidad, preprocesamiento supervisado, selección de ejemplos para etiquetar y propagación de etiquetas.

Su valor práctico está en responder preguntas frecuentes de proyectos y competiciones: cómo descubrir grupos, cómo elegir el número de clusters, cómo tratar clusters no esféricos, cómo crear variables de distancia a centroides, cómo explotar muchos datos no etiquetados y cómo detectar observaciones de baja densidad.

## 2. Conceptos esenciales

### 2.1 Tareas no supervisadas

- **Clustering:** asigna instancias similares a grupos sin usar etiquetas de clase. Una etiqueta de cluster es solo un índice generado por el algoritmo; no equivale a una clase conocida.
- **Anomaly detection (outlier detection):** aprende el patrón de normalidad en datos que pueden estar contaminados y señala observaciones que se apartan de él.
- **Novelty detection:** se entrena suponiendo un conjunto limpio de anomalías y decide si nuevas observaciones son novedosas.
- **Density estimation:** estima la función de densidad del proceso generador. Las regiones de densidad muy baja son candidatas a contener anomalías.
- **Hard clustering:** entrega un único cluster por instancia.
- **Soft clustering:** entrega una afinidad, distancia o probabilidad por cluster. Estas salidas pueden convertirse en variables para otro modelo.

No existe una definición universal de cluster. Distintos algoritmos capturan estructuras diferentes: grupos alrededor de centroides, regiones densas conectadas, jerarquías o formas elipsoidales.

### 2.2 K-Means

K-Means busca `k` centroides y asigna cada observación al centroide más cercano. Alterna dos operaciones hasta converger: asignación de instancias y actualización de cada centroide con la media de su cluster. Produce una partición tipo Voronoi y converge en un número finito de iteraciones, pero puede terminar en un óptimo local dependiente de la inicialización.

Conceptos operativos:

- `labels_`: cluster asignado a cada instancia de entrenamiento.
- `cluster_centers_`: centroides aprendidos.
- `predict(X_new)`: cluster más cercano para observaciones nuevas.
- `transform(X)`: distancia de cada instancia a cada centroide; convierte `n` variables en `k` variables de afinidad/distancia.
- `inertia_`: suma de distancias cuadradas al centroide más cercano en la implementación mostrada; el texto la describe como distancia cuadrática media, pero los valores y la API corresponden al criterio agregado de K-Means.
- `score(X)`: negativo de la inercia para respetar la convención “mayor es mejor”.

K-Means++ elige centroides iniciales separados y reduce el riesgo de una solución local deficiente. Ejecutar varias inicializaciones (`n_init`) y conservar la de menor inercia mitiga el mismo riesgo.

### 2.3 Elección de `k`

La inercia siempre disminuye al aumentar `k`, por lo que no debe minimizarse sin más. El capítulo propone:

1. **Elbow method:** localizar el punto a partir del cual añadir clusters aporta reducciones marginales de inercia.
2. **Silhouette score:** promediar el coeficiente silhouette de todas las instancias.
3. **Silhouette diagram:** revisar por cluster la distribución de coeficientes, su tamaño y cuántas observaciones quedan por debajo del promedio.
4. **Métrica downstream:** cuando K-Means es un transformador dentro de un pipeline supervisado, elegir `k` por validación cruzada de la métrica final, no por inercia o silhouette.

### 2.4 DBSCAN

DBSCAN define clusters como regiones continuas de alta densidad:

- el vecindario de una instancia contiene los puntos a distancia máxima `eps`;
- una **core instance** tiene al menos `min_samples` observaciones en su vecindario, contándose a sí misma;
- core instances conectadas y sus vecinos forman un cluster;
- una observación sin core instance en su vecindario se marca como anomalía (`label = -1`).

No exige fijar el número de clusters, admite formas arbitrarias y es robusto a outliers. Falla cuando las densidades varían mucho entre clusters. La implementación descrita no ofrece `predict()` para nuevos datos; se puede entrenar un clasificador posterior sobre las instancias core y añadir un umbral de distancia para conservar la categoría de anomalía.

### 2.5 Gaussian mixture models

Un GMM supone que los datos provienen de una mezcla de `k` distribuciones gaussianas. Cada componente tiene:

- peso `phi_j`, que representa su frecuencia relativa;
- media `mu_j`;
- matriz de covarianza `Sigma_j`, que determina tamaño, forma y orientación.

El algoritmo Expectation-Maximization (EM) alterna:

1. **Expectation:** estima la responsabilidad, es decir, la probabilidad de pertenencia de cada instancia a cada componente.
2. **Maximization:** actualiza pesos, medias y covarianzas ponderando cada instancia por sus responsabilidades.

El modelo permite hard clustering (`predict`), soft clustering (`predict_proba`), generación de muestras (`sample`) y densidad logarítmica (`score_samples`). Una densidad no es una probabilidad puntual y puede superar 1; para obtener una probabilidad hay que integrar la densidad sobre una región.

### 2.6 Usos del clustering más allá de segmentar

- **Reducción no lineal de dimensionalidad:** reemplazar cada instancia por sus distancias a `k` centroides.
- **Feature engineering:** concatenar las distancias a centroides con las variables originales o usarlas como entrada de un modelo supervisado.
- **Image segmentation por color:** agrupar píxeles RGB y sustituir cada color por el centroide de su cluster.
- **Semi-supervised learning:** etiquetar observaciones representativas, propagar su etiqueta dentro del cluster y limitar la propagación a instancias cercanas al centroide.
- **Active learning:** pedir a una persona etiquetas para observaciones donde el modelo presenta mayor incertidumbre u otra señal de alto valor informativo.

## 3. Flujo o procedimiento presentado

### 3.1 Flujo general de clustering

1. Definir el propósito: segmentación, variables para un modelo, exploración, propagación de etiquetas o detección de anomalías.
2. Preparar las variables y escalarlas cuando se utilicen distancias, especialmente con K-Means.
3. Elegir el supuesto geométrico: centroides aproximadamente esféricos, densidad conectada o componentes elipsoidales.
4. Ajustar el algoritmo solo con el conjunto de entrenamiento si su salida alimentará un modelo supervisado.
5. Evaluar estabilidad, geometría, tamaños y calidad de clusters con la métrica apropiada.
6. Validar el efecto downstream si el clustering actúa como preprocesamiento.
7. Examinar casos fronterizos y anomalías antes de interpretar los clusters como segmentos reales.

### 3.2 K-Means

1. Elegir `k` centroides iniciales, preferentemente con K-Means++.
2. Asignar cada instancia al centroide más cercano.
3. Recalcular cada centroide como la media de las instancias asignadas.
4. Repetir asignación y actualización hasta que los centroides dejen de moverse o se alcance el criterio de parada.
5. Repetir desde distintas inicializaciones y conservar la solución de menor inercia.
6. Comparar valores de `k` mediante elbow, silhouette o la métrica downstream, según el objetivo.

### 3.3 Etiquetado semisupervisado con clusters

1. Ajustar K-Means sobre el conjunto de entrenamiento no etiquetado.
2. Encontrar por cluster la instancia más cercana al centroide.
3. Etiquetar manualmente esas instancias representativas.
4. Entrenar un baseline solo con las instancias representativas.
5. Propagar cada etiqueta a las demás instancias de su cluster.
6. Para reducir errores de frontera, conservar solo un percentil de instancias cercanas al centroide.
7. Entrenar el modelo supervisado con las etiquetas propagadas y medirlo en validación independiente.
8. Si compensa el coste, iterar con active learning sobre los casos de mayor incertidumbre.

### 3.4 DBSCAN

1. Definir `eps` y `min_samples`.
2. Contar vecinos dentro del radio `eps`.
3. Marcar como core las instancias que cumplen `min_samples`.
4. Conectar vecindarios de core instances y formar clusters.
5. Asignar `-1` a puntos no conectados a ninguna core instance.
6. Si se requieren predicciones para datos nuevos, entrenar un clasificador sobre core instances y rechazar como anomalía cualquier predicción cuya distancia mínima supere un umbral.

### 3.5 GMM y anomalías

1. Definir `n_components` y la restricción de covarianza.
2. Ejecutar EM desde varias inicializaciones.
3. Verificar `converged_` y `n_iter_`.
4. Comparar combinaciones de componentes y `covariance_type` mediante AIC/BIC.
5. Obtener `score_samples(X)`.
6. Fijar un umbral de densidad coherente con la tasa esperada de anomalías o con el balance de falsos positivos y falsos negativos.
7. Si muchos outliers sesgan el ajuste, eliminar los casos más extremos y volver a ajustar, o elegir un estimador robusto.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Clusters compactos, aproximadamente esféricos y de tamaño/densidad semejantes | K-Means | Es rápido, escalable y permite predecir clusters nuevos | Escalar variables; repetir inicializaciones; fijar `k` |
| Dataset muy grande o que no cabe cómodamente en memoria | MiniBatchKMeans | Usa mini-batches y reduce mucho el tiempo | Suele producir una inercia algo peor que K-Means, sobre todo con `k` alto |
| `k` no es evidente para K-Means | Elbow más silhouette | La inercia aislada favorece siempre más clusters; silhouette mide cohesión y separación | Elbow es impreciso; silhouette es más costoso y debe inspeccionarse por cluster |
| K-Means es preprocesamiento para clasificación | Seleccionar `k` por validación cruzada del modelo final | El objetivo real es maximizar la métrica supervisada | Ajustar el clustering dentro del pipeline para evitar leakage |
| Clusters de forma arbitraria separados por zonas de baja densidad | DBSCAN | No exige `k`, captura conectividad y marca outliers | Un solo `eps` puede fallar con densidades muy distintas; vigilar memoria con `eps` grande |
| Clusters elipsoidales con distinta forma, tamaño u orientación | GaussianMixture con covarianza adecuada | Modela medias, covarianzas, pesos y pertenencia probabilística | EM puede caer en soluciones pobres; usar varias inicializaciones y comprobar convergencia |
| GMM con demasiadas variables o pocos datos | Restringir `covariance_type` o reducir dimensionalidad | Disminuye el número de parámetros y facilita convergencia | La restricción puede impedir representar la geometría real |
| Elegir número de componentes de un GMM | Minimizar BIC o AIC | Penalizan complejidad y premian ajuste | Inercia y silhouette no son fiables para componentes no esféricos o de tamaños distintos |
| Solo se conoce un límite superior razonable de componentes | BayesianGaussianMixture | Puede llevar a pesos casi nulos los componentes innecesarios | El prior influye más con pocos datos; es más lento que GMM estándar |
| Detectar observaciones raras en un modelo de densidad | Umbral sobre `score_samples` | Baja densidad indica observaciones poco compatibles con el modelo | El umbral controla falsos positivos/negativos y un GMM contaminado puede normalizar outliers |
| Pocas etiquetas y muchas instancias sin etiquetar | Etiquetar representantes por cluster y propagar con cautela | Las etiquetas cubren regiones diversas con menor coste | No propagar indiscriminadamente cerca de fronteras |
| Clustering jerárquico y distancia arbitraria | Agglomerative clustering | Produce un árbol flexible de clusters | Sin matriz de conectividad no escala bien a datasets grandes |
| Dataset enorme con menos de unas 20 variables | BIRCH | Usa una estructura compacta y memoria limitada | La ventaja descrita depende de una dimensionalidad no muy alta |
| Estructuras complejas o corte de grafos | Spectral clustering | Usa un embedding de similitud antes de agrupar | No escala bien y falla con clusters de tamaños muy distintos |

## 5. Métodos y técnicas

### 5.1 K-Means

- **Objetivo:** minimizar la dispersión cuadrática respecto de `k` centroides.
- **Entradas:** matriz numérica `X` y número de clusters `k`.
- **Salida:** índice de cluster, centroides, inercia y distancias a todos los centroides.
- **Funcionamiento:** alterna asignación al centroide más cercano y actualización por medias.
- **Hiperparámetros relevantes:** `n_clusters`, `init`, `n_init`; el capítulo también discute la variante acelerada y el número de iteraciones implícito en la convergencia.
- **Ventajas:** muy rápido; complejidad usual aproximadamente lineal en instancias, clusters y dimensiones; transforma datos nuevos.
- **Limitaciones:** requiere `k`; sensible a inicialización y escala; favorece grupos aproximadamente esféricos, de tamaño y densidad similares; no representa bien formas no convexas.
- **Casos adecuados:** baseline de clustering, cuantización de color, variables de distancia, preprocesamiento supervisado.
- **No conviene:** clusters con formas, densidades o tamaños muy diferentes sin evidencia de que la aproximación por centroides sea útil.

### 5.2 Mini-Batch K-Means

- **Objetivo:** aproximar K-Means con menor tiempo y memoria.
- **Entradas/salida:** equivalentes a K-Means; procesa subconjuntos en cada iteración.
- **Hiperparámetros relevantes:** `n_clusters` y tamaño/flujo de mini-batches; admite entrenamiento incremental mediante `partial_fit`.
- **Ventajas:** aproximadamente tres o cuatro veces más rápido en el ejemplo discutido; apto para datos que no caben en memoria.
- **Limitaciones:** inercia generalmente algo peor; con `partial_fit` manual hay que gestionar inicializaciones y selección de la mejor solución.
- **Casos adecuados:** clustering de gran escala y restricciones de memoria.
- **No conviene:** cuando la máxima calidad de la partición importa más que el tiempo y K-Means completo es viable.

### 5.3 DBSCAN

- **Objetivo:** encontrar regiones densas conectadas y separar ruido.
- **Entradas:** `X`, radio `eps` y mínimo `min_samples`.
- **Salida:** clusters, índices/componentes core y etiqueta `-1` para anomalías.
- **Funcionamiento:** expande clusters a través de cadenas de core instances vecinas.
- **Hiperparámetros:** `eps`, `min_samples`.
- **Ventajas:** formas arbitrarias, número de clusters automático, robustez a outliers, pocos hiperparámetros.
- **Limitaciones:** densidades variables; no incluye predicción nativa en la implementación tratada; el uso de memoria puede llegar a orden cuadrático cuando `eps` es grande.
- **Casos adecuados:** moons, geometrías no convexas, clustering con ruido.
- **No conviene:** grupos con escalas de densidad incompatibles con un único radio.

### 5.4 GaussianMixture

- **Objetivo:** estimar una mezcla probabilística de gaussianas y usarla para clustering, densidad, generación o anomalías.
- **Entradas:** `X`, `n_components`, forma de covarianza e inicializaciones.
- **Salida:** pesos, medias, covarianzas, responsabilidades, clusters y log-densidades.
- **Funcionamiento:** EM alterna responsabilidades y actualización ponderada de parámetros.
- **Hiperparámetros:** `n_components`, `n_init`, `covariance_type` (`spherical`, `diag`, `tied`, `full`).
- **Ventajas:** soft clustering; modela elipsoides con diferente tamaño/orientación; modelo generativo.
- **Limitaciones:** necesita el número de componentes; óptimos locales; dificultad con muchas dimensiones, muchos componentes o pocas instancias; falla como clustering para formas no elipsoidales.
- **Casos adecuados:** segmentos elipsoidales, probabilidades de pertenencia, muestreo, density-based anomaly detection.
- **No conviene:** clusters tipo “moons” si la interpretación buscada es recuperar esas formas; datos muy contaminados sin estrategia robusta.

Complejidad indicada por el capítulo:

- `spherical` o `diag`: aproximadamente `O(kmn)`.
- `tied` o `full`: aproximadamente `O(kmn^2 + kn^3)`; no escala bien con muchas variables.

### 5.5 BayesianGaussianMixture

- **Objetivo:** partir de un límite superior de componentes y reducir el peso de los innecesarios.
- **Entradas/salida:** similares a GMM, añadiendo priors sobre variables latentes.
- **Funcionamiento:** inferencia variacional; el prior de concentración controla la preferencia por pocos o muchos clusters.
- **Hiperparámetro destacado:** `weight_concentration_prior`.
- **Ventajas:** evita una búsqueda manual estricta del número exacto de clusters.
- **Limitaciones:** requiere un límite superior razonable; los priors pesan más con pocos datos; suele ser significativamente más lento; conserva el supuesto geométrico gaussiano.
- **Casos adecuados:** número de componentes incierto con estructura aproximadamente elipsoidal.
- **No conviene:** formas arbitrarias no gaussianas o presupuestos de cómputo muy ajustados.

### 5.6 Otros algoritmos resumidos por el capítulo

- **Agglomerative clustering:** fusiona clusters de abajo arriba; flexible en distancia y forma; una matriz de conectividad sparse mejora la escalabilidad.
- **BIRCH:** resume datos en un árbol y está diseñado para datasets enormes con dimensionalidad moderada.
- **Mean-Shift:** desplaza ventanas hacia máximos locales de densidad; encuentra número y forma de clusters, pero puede fragmentar densidades internas variables y tiene coste cuadrático.
- **Affinity propagation:** las instancias votan representantes; admite tamaños variados, pero su coste cuadrático limita la escala.
- **Spectral clustering:** crea un embedding desde una matriz de similitud y agrupa en ese espacio; capta estructuras complejas, pero escala mal.
- **PCA con error de reconstrucción:** anomalías suelen reconstruirse peor que casos normales.
- **Fast-MCD / EllipticEnvelope:** estima robustamente una única envolvente gaussiana ignorando contaminación probable.
- **Isolation Forest:** aísla anomalías en menos divisiones aleatorias; eficiente especialmente en alta dimensionalidad.
- **Local Outlier Factor:** compara la densidad local de una instancia con la de sus vecinos.
- **One-Class SVM:** aprende una región que contiene los datos normales; útil para novelty detection y alta dimensionalidad, pero no escala a datasets grandes.

## 6. Fórmulas esenciales

### 6.1 Inercia de K-Means

\[
J = \sum_{i=1}^{m} \left\|x^{(i)} - \mu_{c(i)}\right\|_2^2
\]

- `m`: número de instancias.
- `x^(i)`: instancia `i`.
- `c(i)`: cluster asignado a `i`.
- `mu_{c(i)}`: centroide de ese cluster.

Mide la dispersión interna respecto al centroide. Menor es mejor para un `k` fijo, pero no sirve por sí sola para elegir `k` porque decrece de forma monótona al aumentar los clusters.

### 6.2 Probabilidad de selección en K-Means++

\[
P\left(x^{(i)}\text{ sea el siguiente centroide}\right)
= \frac{D\left(x^{(i)}\right)^2}{\sum_{j=1}^{m} D\left(x^{(j)}\right)^2}
\]

`D(x)` es la distancia de `x` al centroide ya elegido más cercano. Elevar la distancia al cuadrado hace más probable seleccionar puntos alejados, promoviendo semillas separadas.

### 6.3 Coeficiente silhouette

\[
s(i) = \frac{b(i)-a(i)}{\max(a(i), b(i))}
\]

- `a(i)`: distancia media de la instancia `i` a las demás instancias de su cluster.
- `b(i)`: menor distancia media de `i` a otro cluster.

Interpretación: cerca de `+1`, asignación compacta y separada; cerca de `0`, frontera; cerca de `-1`, posible asignación incorrecta. El silhouette score es la media de `s(i)`.

### 6.4 Modelo generativo de un GMM

\[
z^{(i)} \sim \operatorname{Categorical}(\phi), \qquad
x^{(i)} \mid z^{(i)}=j \sim \mathcal{N}(\mu_j, \Sigma_j)
\]

- `z^(i)`: componente latente que generó la instancia.
- `phi`: vector de pesos, con pesos no negativos que suman 1.
- `mu_j`, `Sigma_j`: media y covarianza del componente `j`.

El supuesto central es que cada observación procede de una de varias gaussianas y que cada cluster se representa mediante una geometría elipsoidal.

### 6.5 BIC y AIC

\[
\operatorname{BIC} = (\log m)p - 2\log \hat{L}
\]

\[
\operatorname{AIC} = 2p - 2\log \hat{L}
\]

- `m`: número de instancias.
- `p`: número de parámetros aprendidos.
- `L_hat`: valor máximo de la función de verosimilitud.

Ambos premian ajuste y penalizan complejidad; se elige el modelo con valor menor. Cuando difieren, BIC suele favorecer un modelo más simple, especialmente con datasets grandes.

### 6.6 Umbral de densidad para anomalías

\[
t = Q_q\left(\{\log p(x^{(i)})\}_{i=1}^{m}\right), \qquad
x^{(i)}\text{ es anomalía si }\log p(x^{(i)}) < t
\]

`Q_q` es el percentil `q` de las log-densidades. El capítulo ejemplifica `q = 4` cuando se espera aproximadamente 4% de defectos. Subir el umbral aumenta detecciones y posibles falsos positivos; bajarlo reduce detecciones y puede aumentar falsos negativos.

## 7. Implementación práctica

Los siguientes fragmentos condensan las operaciones del capítulo. **Nota práctica derivada:** se añade `random_state` para reproducibilidad y se fija `n_init` explícitamente; estas dos decisiones de presentación no forman parte de todos los fragmentos originales.

### 7.1 K-Means, selección de `k` y transformación

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = {}
models = {}
for k in range(2, 11):
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_train_scaled)
    scores[k] = silhouette_score(X_train_scaled, labels)
    models[k] = model

best_k = max(scores, key=scores.get)
kmeans = models[best_k]
X_train_cluster_dist = kmeans.transform(X_train_scaled)
X_valid_cluster_dist = kmeans.transform(X_valid_scaled)
```

El escalador y K-Means deben ajustarse solo con entrenamiento. Si las variables de distancia alimentan un predictor, `best_k` debe decidirse con validación cruzada de la métrica final cuando sea posible.

### 7.2 Pipeline supervisado sin leakage

```python
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scale", StandardScaler()),
    ("cluster_features", KMeans(n_clusters=50, n_init=10, random_state=42)),
    ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
])
model.fit(X_train, y_train)
valid_score = model.score(X_valid, y_valid)
```

**Adaptación moderna:** se aumenta `max_iter` para reducir fallos de convergencia en regresión logística; es una nota operativa, no una conclusión del capítulo.

### 7.3 DBSCAN con rechazo por distancia para datos nuevos

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

dbscan = DBSCAN(eps=0.2, min_samples=5).fit(X_train_scaled)
core_labels = dbscan.labels_[dbscan.core_sample_indices_]

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, core_labels)

nearest_dist, nearest_idx = knn.kneighbors(X_valid_scaled, n_neighbors=1)
pred = core_labels[nearest_idx].ravel()
pred[nearest_dist.ravel() > 0.2] = -1
```

El umbral `0.2` reproduce la lógica del ejemplo; debe validarse para el dataset real y no asumirse universal.

### 7.4 GMM, selección por BIC y anomalías

```python
import numpy as np
from sklearn.mixture import GaussianMixture

candidates = []
for covariance_type in ("spherical", "diag", "tied", "full"):
    for n_components in range(1, 11):
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=10,
            random_state=42,
        ).fit(X_train_scaled)
        candidates.append((gm.bic(X_train_scaled), gm))

_, best_gm = min(candidates, key=lambda item: item[0])
train_log_density = best_gm.score_samples(X_train_scaled)
threshold = np.percentile(train_log_density, 4)
valid_is_anomaly = best_gm.score_samples(X_valid_scaled) < threshold
```

El percentil debe corresponder a una tasa esperada o a un trade-off medido. No debe calibrarse mirando las etiquetas del conjunto de prueba.

## 8. Supuestos y condiciones de aplicación

1. K-Means usa distancias; las variables deben tener escalas comparables.
2. K-Means funciona mejor con clusters aproximadamente esféricos, de dispersión y tamaño parecidos.
3. El valor de `k` debe conocerse o seleccionarse; la inercia no basta para hacerlo.
4. DBSCAN requiere que los clusters sean suficientemente densos y estén separados por regiones de baja densidad.
5. Un único `eps` supone una escala de densidad razonablemente compartida.
6. GMM supone que los datos se explican mediante una mezcla de gaussianas; sus clusters son elipsoidales.
7. GMM completo necesita suficientes datos para estimar covarianzas, especialmente con muchas variables.
8. EM y K-Means dependen de la inicialización y pueden converger a soluciones locales.
9. Density-based anomaly detection presupone que las anomalías ocupan regiones de baja densidad y que la contaminación no domina el ajuste.
10. Novelty detection presupone entrenamiento limpio; anomaly detection no.
11. La propagación de etiquetas presupone coherencia entre clusters y clases; esta relación debe comprobarse, sobre todo cerca de fronteras.
12. La selección de `k` por métrica downstream requiere un conjunto o esquema de validación independiente.

## 9. Errores y riesgos

- **Confundir cluster con clase:** los índices de cluster son arbitrarios y no contienen semántica por sí mismos.
- **No escalar variables:** una variable de gran magnitud puede dominar las distancias.
- **Elegir `k` por mínima inercia:** conduce a favorecer cada vez más clusters.
- **Aceptar un silhouette promedio sin inspección:** puede ocultar clusters desbalanceados o con muchos casos fronterizos.
- **Interpretar clusters inestables:** distintas inicializaciones pueden producir particiones diferentes.
- **Usar K-Means en geometrías inadecuadas:** tamaños, densidades u orientaciones distintas pueden generar divisiones erróneas incluso con menor inercia.
- **Filtrar antes del split:** escalar, seleccionar `k`, ajustar centroides o calibrar umbrales con todos los datos produce data leakage.
- **Propagar todas las etiquetas:** las observaciones fronterizas son candidatas a recibir etiquetas incorrectas; limitar por distancia puede mejorar la calidad.
- **Forzar predicciones DBSCAN:** un clasificador auxiliar siempre asignará un cluster si no se implementa rechazo por distancia.
- **`eps` demasiado grande:** puede fusionar clusters, perder anomalías y disparar el consumo de memoria de la implementación descrita.
- **Densidades heterogéneas en DBSCAN:** un solo par `eps`/`min_samples` puede no recuperar todos los clusters.
- **No revisar convergencia de GMM:** `converged_` y `n_iter_` deben inspeccionarse.
- **Pocas inicializaciones en EM:** aumenta el riesgo de conservar una solución deficiente.
- **Confundir densidad con probabilidad:** `exp(score_samples)` es una densidad puntual, no la probabilidad de una observación exacta.
- **GMM contaminado por outliers:** el modelo intenta explicar también los outliers y puede incorporarlos a su definición de normalidad.
- **Covarianza demasiado flexible:** con muchas variables aumenta drásticamente el número de parámetros y el coste.
- **Tratar BIC/AIC como garantía:** comparan candidatos bajo los supuestos del modelo; no corrigen una familia gaussiana mal especificada.
- **Usar GMM para formas no elipsoidales:** puede fragmentar una estructura real en múltiples componentes.
- **Coste cuadrático:** Mean-Shift y Affinity Propagation no son adecuados para datasets grandes; Spectral Clustering tampoco escala bien.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

### Cuándo consultar este capítulo

- El target no existe o solo una pequeña fracción está etiquetada.
- Se piden segmentos de clientes, compresión/segmentación de imágenes, detección de fraude o de defectos.
- El dataset presenta grupos visibles, formas no convexas o posibles observaciones raras.
- Un baseline supervisado parece limitado y las distancias a prototipos pueden aportar variables no lineales.

### Decisiones que permite tomar

- K-Means frente a DBSCAN frente a GMM según geometría y densidad.
- Método de selección de `k`: silhouette/elbow para estructura no supervisada, métrica de validación para un pipeline supervisado, BIC/AIC para GMM.
- Covarianza de GMM según flexibilidad requerida y coste disponible.
- Uso de clustering para priorizar etiquetado o propagar etiquetas.
- Umbral de anomalía a partir de una tasa esperada y del coste de falsos positivos/negativos.

### Baseline recomendable

1. Escalar variables numéricas.
2. Probar K-Means con un rango pequeño de `k` y `n_init` explícito.
3. Comparar silhouette y tamaños de clusters.
4. Si hay target, añadir distancias a centroides dentro de un pipeline y medir validación cruzada.
5. Para anomalías, comenzar con log-densidad de GMM solo si la geometría parece aproximadamente elipsoidal.

### Mejoras con potencial

- MiniBatchKMeans cuando el coste domina.
- DBSCAN para formas arbitrarias con ruido.
- GMM con búsqueda conjunta de `n_components` y `covariance_type` por BIC/AIC.
- Etiquetado de representantes y propagación parcial cuando el presupuesto de etiquetas es pequeño.
- Reentrenamiento del detector de anomalías después de retirar solo los casos más extremos.

### Qué evitar bajo presión de tiempo

- Implementar inferencia variacional o derivar ELBO manualmente.
- Hacer una búsqueda extensa de muchos algoritmos cuadráticos.
- Construir un predictor complejo para DBSCAN si no se evaluarán datos nuevos.
- Sobreinterpretar clusters sin inspeccionar tamaño, estabilidad y ejemplos representativos.
- Usar active learning si no existe un ciclo real de etiquetado humano durante la prueba.

### Señales del dataset que activan técnicas

- **Clusters redondeados y escala moderada:** K-Means.
- **Muchas filas o memoria ajustada:** MiniBatchKMeans/BIRCH.
- **Formas curvas con ruido:** DBSCAN.
- **Nubes elipsoidales y pertenencia ambigua:** GMM.
- **Pocas etiquetas y muchos datos:** representantes, label propagation y active learning.
- **Outliers esperados con tasa conocida:** percentil de log-densidad.
- **Alta dimensionalidad para GMM:** reducción dimensional o covarianza `diag`/`spherical`.

## 11. Checklist operativo

- [ ] Definir si el objetivo es segmentar, crear variables, propagar etiquetas o detectar anomalías.
- [ ] Separar entrenamiento, validación y prueba antes de aprender transformaciones.
- [ ] Escalar variables usadas por distancias.
- [ ] Revisar forma, densidad y tamaños esperados de los clusters.
- [ ] Ejecutar varias inicializaciones de K-Means o GMM.
- [ ] Elegir `k` con el criterio alineado al objetivo.
- [ ] Inspeccionar silhouette por cluster, no solo el promedio.
- [ ] Comprobar `converged_` en GMM.
- [ ] Validar `covariance_type` y coste dimensional.
- [ ] Calibrar `eps`/`min_samples` o el umbral de anomalía sin tocar test.
- [ ] Implementar rechazo por distancia si se predicen nuevos casos tras DBSCAN.
- [ ] Limitar label propagation a instancias confiables.
- [ ] Examinar representantes, fronteras y anomalías antes de asignar significado de negocio.
- [ ] Medir el impacto downstream dentro de un pipeline sin leakage.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Selección de algoritmo según geometría, densidad y escala | `knowledge/models/clustering.md` | Alta |
| K-Means, K-Means++, MiniBatchKMeans e inercia | `knowledge/models/kmeans.md` | Alta |
| Elbow, silhouette, BIC y AIC | `knowledge/validation/unsupervised_model_selection.md` | Alta |
| DBSCAN, core instances y rechazo para datos nuevos | `knowledge/models/density_based_clustering.md` | Alta |
| GMM, EM, covarianzas y responsabilidades | `knowledge/models/gaussian_mixtures.md` | Alta |
| Umbrales de densidad y diferencia anomaly/novelty detection | `knowledge/models/anomaly_detection.md` | Alta |
| Distancias a centroides como variables y reducción no lineal | `knowledge/preprocessing/clustering_as_transformation.md` | Alta |
| Pipelines de clustering más clasificación sin leakage | `knowledge/validation/pipelines_and_leakage.md` | Alta |
| Etiquetado representativo, propagación parcial y active learning | `knowledge/models/semi_supervised_learning.md` | Media |
| Segmentación de imágenes por cuantización de color | `knowledge/feature_engineering/image_color_quantization.md` | Media |
| Métodos alternativos: BIRCH, aglomerativo, espectral, Mean-Shift | `knowledge/models/clustering_alternatives.md` | Media |
| PCA, Isolation Forest, LOF, Fast-MCD y One-Class SVM para anomalías | `knowledge/models/anomaly_detection_alternatives.md` | Media |

## 13. Trazabilidad

La columna de páginas usa la numeración impresa del libro; entre paréntesis se indica la página correspondiente dentro del PDF suministrado.

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Introducción y catálogo de tareas, pp. 235-236 (PDF 1-2) |
| 2.1 Tareas no supervisadas | Introducción; “Clustering”; aplicaciones, pp. 236-238 (PDF 2-4) |
| 2.2 K-Means | “K-Means”; “The K-Means algorithm”; inicialización, pp. 238-245 (PDF 4-11) |
| 2.3 Elección de `k` | “Finding the optimal number of clusters”, pp. 245-248 (PDF 11-14) |
| 2.4 DBSCAN | “DBSCAN”, pp. 255-258 (PDF 21-24) |
| 2.5 Gaussian mixture models | “Gaussian Mixtures”, pp. 260-265 (PDF 26-31) |
| 2.6 Usos del clustering | Aplicaciones; segmentación, preprocesamiento y semi-supervisión, pp. 237-238 y 249-255 (PDF 3-4 y 15-21) |
| 3.1-3.2 Flujo y K-Means | Algoritmo, inicialización y selección de clusters, pp. 241-248 (PDF 7-14) |
| 3.3 Etiquetado semisupervisado | “Using Clustering for Semi-Supervised Learning” y “Active Learning”, pp. 253-255 (PDF 19-21) |
| 3.4 DBSCAN | “DBSCAN”, pp. 255-258 (PDF 21-24) |
| 3.5 GMM y anomalías | EM, anomaly detection y selección de componentes, pp. 262-270 (PDF 28-36) |
| 4. Reglas de decisión | Límites de K-Means; DBSCAN; otros algoritmos; GMM, pp. 245-275 (PDF 11-41) |
| 5.1 K-Means | pp. 238-249 (PDF 4-15) |
| 5.2 Mini-Batch K-Means | “Accelerated K-Means and mini-batch K-Means”, pp. 244-245 (PDF 10-11) |
| 5.3 DBSCAN | pp. 255-258 (PDF 21-24) |
| 5.4 GaussianMixture | pp. 260-269 (PDF 26-35) |
| 5.5 BayesianGaussianMixture | “Bayesian Gaussian Mixture Models”, pp. 270-273 (PDF 36-39) |
| 5.6 Otros algoritmos | “Other Clustering Algorithms” y “Other Algorithms for Anomaly and Novelty Detection”, pp. 258-260 y 274-275 (PDF 24-26 y 40-41) |
| 6.1-6.3 Inercia, K-Means++ y silhouette | pp. 243-248 (PDF 9-14) |
| 6.4 Modelo generativo GMM | pp. 260-263 (PDF 26-29) |
| 6.5 BIC y AIC | “Selecting the Number of Clusters” y recuadro “Likelihood Function”, pp. 267-270 (PDF 33-36) |
| 6.6 Umbral de densidad | “Anomaly Detection Using Gaussian Mixtures”, pp. 266-267 (PDF 32-33) |
| 7. Implementación práctica | Ejemplos de K-Means, pipelines, DBSCAN y GMM, pp. 239-264 y 266-270 (PDF 5-30 y 32-36) |
| 8. Supuestos y condiciones | Límites de K-Means, DBSCAN y GMM; anomaly vs novelty, pp. 248-249, 258, 264-267 y 273-275 (PDF 14-15, 24, 30-33 y 39-41) |
| 9. Errores y riesgos | Inicialización, geometría, densidad, propagación y contaminación, pp. 242-248, 254-258 y 263-275 (PDF 8-14, 20-24 y 29-41) |
| 10. Aplicación en coding test | Síntesis práctica derivada de pp. 238-275 (PDF 4-41) |
| 11. Checklist operativo | Síntesis práctica derivada de pp. 238-275 (PDF 4-41) |
| 12. Mapeo hacia `knowledge/` | Clasificación documental derivada del capítulo completo, pp. 235-276 (PDF 1-42) |

### Estado de lectura

- Páginas revisadas: 235-276 (las 42 páginas del PDF).
- Páginas o secciones ilegibles: ninguna.
- Los ejercicios de las pp. 275-276 se revisaron para identificar aplicaciones, pero no se reprodujeron ni resolvieron, conforme al criterio de reducir ejercicios sin solución.
- No se consultó el libro completo, apéndices ni fuentes externas.
