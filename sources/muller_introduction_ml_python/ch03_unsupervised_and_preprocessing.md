---
title: "Unsupervised Learning and Preprocessing"
topics:
  - "preprocesamiento y escalado"
  - "reducción de dimensionalidad y extracción de características"
  - "PCA, NMF y t-SNE"
  - "clustering y evaluación no supervisada"
target_knowledge_files:
  - "knowledge/preprocessing/feature_scaling.md"
  - "knowledge/feature_engineering/dimensionality_reduction.md"
  - "knowledge/models/clustering.md"
  - "knowledge/metrics/clustering_metrics.md"
  - "knowledge/validation/unsupervised_evaluation.md"
---

# 3. Unsupervised Learning and Preprocessing

## 1. Propósito del capítulo

El capítulo presenta aprendizaje no supervisado como extracción de estructura a partir de entradas sin una salida conocida. Desarrolla dos familias: transformaciones de la representación - escalado, reducción de dimensionalidad y descomposición - y clustering. Su utilidad práctica es explorar datos, construir visualizaciones, reducir memoria o tiempo y preparar representaciones que puedan mejorar modelos supervisados.

## 2. Conceptos esenciales

### Contenido explicado explícitamente por los autores

- En aprendizaje no supervisado no hay etiquetas que definan una respuesta correcta. Por ello, evaluar si el resultado es útil suele requerir inspección manual y conocimiento del problema.
- Una transformación no supervisada aprende una nueva representación de `X`. Puede reducir dimensiones, comprimir, visualizar o separar componentes latentes. Un algoritmo de clustering asigna observaciones similares a grupos.
- El escalado es no supervisado porque se ajusta con `X`, sin usar `y`, aunque normalmente forma parte de un flujo supervisado.
- Toda transformación debe ajustarse con entrenamiento y aplicarse sin reajuste a validación o test. Los valores de test pueden quedar fuera del intervalo aprendido por `MinMaxScaler`; eso es correcto porque se reutiliza la transformación de entrenamiento.
- `StandardScaler` centra cada variable y la lleva a varianza unitaria; `RobustScaler` usa mediana y cuartiles para reducir la influencia de outliers; `MinMaxScaler` lleva el rango de entrenamiento a un intervalo, normalmente `[0, 1]`; `Normalizer` normaliza cada fila a longitud euclídea 1.
- PCA rota datos centrados hacia direcciones ortogonales de máxima varianza. Conservando solo algunas componentes reduce dimensión, pero sus ejes suelen mezclar muchas variables y ser difíciles de interpretar.
- NMF representa cada observación como suma ponderada de componentes no negativas. Requiere entradas no negativas, funciona especialmente bien con estructura aditiva y suele producir patrones más interpretables que PCA.
- t-SNE busca una representación, normalmente bidimensional, que preserve vecindades locales. No ofrece `transform` para datos nuevos en la interfaz descrita y sirve principalmente para exploración visual, no como paso de producción supervisada.
- k-means alterna entre asignar cada punto al centro más cercano y recalcular cada centro como la media de sus puntos. Obliga a asignar todos los puntos y favorece clusters convexos, de tamaño y densidad parecidos.
- El clustering aglomerativo empieza con una observación por cluster y fusiona grupos según un criterio de enlace. Genera una jerarquía inspeccionable con dendrogramas, pero el estimador descrito no predice para observaciones nuevas.
- DBSCAN define clusters como regiones densas, admite formas complejas, no exige fijar el número de clusters y etiqueta ruido con `-1`. Sus parámetros centrales son `eps` y `min_samples`.
- Los identificadores de cluster son arbitrarios. `accuracy_score` es inapropiado para comparar particiones; con ground truth deben usarse métricas invariantes a permutación como ARI o NMI.
- Sin ground truth, el coeficiente silhouette favorece grupos compactos y puede valorar peor una estructura semánticamente adecuada de forma compleja. Ninguna métrica interna garantiza que los clusters respondan a la pregunta de negocio.

### Conclusiones prácticas derivadas del contenido

- El preprocesamiento pertenece al modelo: en validación debe encapsularse con el estimador para impedir leakage y mantener idéntica transformación.
- En datos tabulares, conviene escalar antes de métodos basados en distancia, kernels o densidad; de otro modo, las variables de mayor magnitud dominan.
- Para clustering, comparar varias granularidades, tamaños, ruido y ejemplos representativos es más informativo que reportar una sola métrica.
- Reducir con PCA antes de clustering puede acelerar el cálculo y producir una representación más semántica, pero altera la geometría y debe validarse como parte del flujo.

## 3. Flujo o procedimiento presentado

1. Definir si el objetivo es preprocesar, visualizar, extraer componentes o descubrir grupos.
2. Separar entrenamiento y evaluación antes de aprender cualquier transformación que vaya a alimentar un modelo supervisado.
3. Elegir el escalador según la geometría relevante: estadísticos estándar, robustez a outliers, rango fijo o dirección de cada fila.
4. Ejecutar `fit` solo sobre entrenamiento; aplicar `transform` a entrenamiento, validación y test con el mismo objeto.
5. Para reducción de dimensión, escalar cuando las magnitudes de las variables no sean comparables y elegir `n_components` según visualización, compresión o representación.
6. Inspeccionar componentes y, cuando exista, reconstrucción o ejemplos con coeficientes altos; no asumir interpretabilidad por defecto.
7. Para t-SNE, usar `fit_transform` sobre el conjunto que se desea visualizar y leer vecindades locales, no distancias globales ni posiciones absolutas.
8. Para clustering, estandarizar variables si la distancia depende de escala y probar un algoritmo compatible con la forma esperada de los grupos.
9. Ajustar granularidad: `n_clusters` en k-means/agglomerative; `eps` y `min_samples` en DBSCAN.
10. Evaluar con ARI/NMI si existen etiquetas de referencia; sin ellas, combinar silhouette, estabilidad ante perturbaciones e inspección cualitativa.
11. Revisar tamaños, puntos típicos, extremos, ruido y significado de cada cluster antes de usarlo en decisiones posteriores.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Variables aproximadamente continuas con escalas distintas | `StandardScaler` | Igualar magnitudes mediante media 0 y varianza 1 | Ajustar solo con entrenamiento; es sensible a outliers |
| Outliers fuertes o errores de medición | `RobustScaler` | Usa mediana y cuartiles en lugar de media y varianza | No elimina outliers ni garantiza un rango fijo |
| Se necesita un rango de entrenamiento fijo | `MinMaxScaler` | Lleva mínimos y máximos de entrenamiento al intervalo elegido | Test puede caer fuera del rango; no reajustar con test |
| Importa la dirección de cada observación, no su magnitud | `Normalizer` | Lleva cada vector-fila a norma euclídea 1 | Opera por muestra, no por variable |
| Visualizar o comprimir datos continuos correlacionados | PCA | Conserva direcciones ortogonales de mayor varianza | Escalar antes si las unidades difieren; componentes difíciles de interpretar |
| Extraer partes aditivas interpretables de datos no negativos | NMF | Componentes y coeficientes no negativos evitan cancelaciones | No admite entradas negativas; solución y orden dependen de inicialización |
| Explorar vecindades complejas en dos dimensiones | t-SNE | Preserva estructura local mejor que una proyección lineal | No transforma datos nuevos; no usar como representación de producción descrita |
| Se conoce `k` y se esperan grupos compactos/convexos | k-means | Simple, escalable y centros fáciles de inspeccionar | Falla con densidades, tamaños o formas muy diferentes; asigna incluso ruido |
| Se desea jerarquía y elección visual de granularidad | Clustering aglomerativo + dendrograma | Muestra fusiones y distancias entre grupos | No predice puntos nuevos; Ward tiende a tamaños parecidos |
| Hay ruido y clusters de forma compleja | DBSCAN | Detecta regiones densas y puntos atípicos sin fijar `k` | Muy sensible a `eps`, escala y densidades heterogéneas |
| Hay etiquetas de referencia para auditar clustering | ARI o NMI | Son invariantes a la numeración arbitraria de clusters | Las etiquetas convierten la evaluación en supervisada; no usar accuracy |
| No hay etiquetas | Inspección + estabilidad; silhouette como apoyo | Ningún score interno asegura significado semántico | Silhouette favorece compactación y puede castigar formas complejas |

## 5. Métodos y técnicas

### Escaladores

- **Objetivo:** hacer comparables las magnitudes antes de métodos sensibles a escala.
- **Entradas:** matriz numérica `X`; `fit` aprende estadísticas por columna, salvo `Normalizer`, que actúa por fila.
- **Salida:** matriz de igual forma con valores reescalados.
- **Funcionamiento:** `StandardScaler` usa media/desviación; `RobustScaler`, mediana/cuartiles; `MinMaxScaler`, mínimo/rango; `Normalizer`, norma de cada muestra.
- **Hiperparámetros:** rango en `MinMaxScaler`; norma en `Normalizer` (el capítulo muestra norma euclídea).
- **Ventajas:** pueden mejorar de forma drástica modelos sensibles a escala; comparten la interfaz `fit`/`transform`.
- **Limitaciones:** no agregan información; una elección inadecuada puede deformar la geometría útil.
- **Adecuados:** SVM, redes neuronales y algoritmos basados en distancias o densidad.
- **No convienen:** no deben ajustarse por separado en train y test ni sobre todo el dataset antes de evaluar.

### Principal Component Analysis (PCA)

- **Objetivo:** rotar hacia ejes no correlacionados ordenados por varianza y, opcionalmente, conservar solo algunos.
- **Entradas:** `X` numérica; suele requerir escalado si las unidades difieren.
- **Salida:** puntuaciones por componente; `components_` contiene un vector por componente en el espacio original.
- **Funcionamiento:** centra, encuentra la dirección de máxima varianza y sucesivas direcciones ortogonales; `inverse_transform` aproxima el espacio original con la información retenida.
- **Hiperparámetros:** `n_components`; `whiten=True` reescala componentes a la misma escala.
- **Ventajas:** visualización, compresión, eliminación de ruido, reducción de cómputo y extracción de características.
- **Limitaciones:** transformación lineal; los componentes mezclan variables; el signo es arbitrario; la máxima varianza no garantiza relevancia para la tarea.
- **Casos adecuados:** variables correlacionadas, imágenes y representación previa a un modelo o clustering.
- **No conviene:** si se necesitan factores directamente interpretables o la estructura esencial es no lineal.

### Non-negative Matrix Factorization (NMF)

- **Objetivo:** descomponer observaciones en combinaciones aditivas de componentes no negativas.
- **Entradas:** matriz con todas las entradas `>= 0`.
- **Salida:** coeficientes no negativos por muestra y componentes no negativas en `components_`.
- **Funcionamiento:** aproxima cada fila como suma ponderada de patrones. Al variar `n_components`, los patrones se vuelven a aprender; no es simplemente truncar una lista fija.
- **Hiperparámetros:** `n_components`; `random_state` controla la inicialización aleatoria mostrada.
- **Ventajas:** patrones frecuentemente más interpretables; útil en audio, expresión génica y texto con estructura aditiva.
- **Limitaciones:** reconstruye peor que PCA en el criterio donde PCA es óptima; no hay orden natural de componentes; puede variar con la semilla.
- **Casos adecuados:** datos no negativos donde “partes que se suman” tenga significado.
- **No conviene:** entradas con valores negativos o cuando solo se busca la mejor reconstrucción lineal por varianza.

### t-distributed Stochastic Neighbor Embedding (t-SNE)

- **Objetivo:** visualizar datos de alta dimensión preservando vecindades locales.
- **Entradas:** conjunto completo que se desea representar.
- **Salida:** coordenadas, normalmente 2D, solo para esas observaciones.
- **Funcionamiento:** parte de posiciones aleatorias y acerca vecinos originales mientras separa puntos lejanos, con mayor énfasis en cercanía local.
- **Hiperparámetros:** `perplexity`, `early_exaggeration` y `random_state`; el capítulo indica que los defaults suelen funcionar y que los dos primeros pueden probarse.
- **Ventajas:** puede revelar separación local que PCA no muestra.
- **Limitaciones:** no tiene `transform` para casos nuevos; es exploratorio y no una etapa habitual de un predictor supervisado.
- **Casos adecuados:** inspección 2D de clases o estructura latente.
- **No conviene:** producción, inferencia sobre nuevos registros o interpretación de distancias globales.

### k-means

- **Objetivo:** particionar en `k` regiones representadas por sus medias.
- **Entradas:** matriz numérica y número de clusters.
- **Salida:** `labels_`, `cluster_centers_`; `predict` asigna datos nuevos al centro más cercano.
- **Funcionamiento:** alterna asignación al centro más cercano y actualización de medias hasta que las asignaciones dejan de cambiar.
- **Hiperparámetros:** `n_clusters`; la inicialización aleatoria requiere ejecuciones/semilla controladas.
- **Ventajas:** fácil de entender, permite resumir cada grupo por su centro y puede verse como cuantización vectorial o nueva representación discreta.
- **Limitaciones:** clusters convexos, fronteras lineales entre centros, sensibilidad a escala e inicialización; prefiere tamaños y densidades similares; no modela ruido.
- **Casos adecuados:** grupos compactos aproximadamente esféricos y necesidad de asignar nuevos puntos.
- **No conviene:** medias lunas, estructuras no convexas, densidades muy distintas o abundantes outliers.

### Clustering aglomerativo

- **Objetivo:** construir una jerarquía de clusters mediante fusiones sucesivas.
- **Entradas:** matriz numérica, `n_clusters` y criterio de enlace.
- **Salida:** etiquetas del conjunto ajustado mediante `fit_predict`; dendrograma si se calcula la jerarquía con SciPy.
- **Funcionamiento:** cada punto comienza solo; se fusionan los dos grupos más similares hasta alcanzar la parada.
- **Hiperparámetros:** `n_clusters` y `linkage`. `ward` minimiza el aumento de varianza intra-cluster; `average` minimiza distancia media; `complete` minimiza la máxima distancia entre pares.
- **Ventajas:** la jerarquía permite inspeccionar varias granularidades; Ward funciona bien en muchos datasets.
- **Limitaciones:** no ofrece `predict` para puntos nuevos; Ward tiende a grupos de tamaño parecido; también falla en algunas formas complejas.
- **Casos adecuados:** exploración jerárquica; `average` o `complete` si los tamaños son muy desiguales.
- **No conviene:** inferencia online de membresía o estructuras complejas que el enlace no capture.

### DBSCAN

- **Objetivo:** identificar regiones densas separadas por regiones vacías y marcar ruido.
- **Entradas:** datos numéricos y una métrica de distancia implícita en `eps`.
- **Salida:** etiquetas; `-1` significa ruido; distingue puntos core, frontera y ruido.
- **Funcionamiento:** un punto es core si tiene al menos `min_samples` observaciones dentro de `eps`; los core conectados expanden el cluster y sus vecinos de frontera se incorporan.
- **Hiperparámetros:** `eps` domina la noción de cercanía; `min_samples` controla la densidad mínima y el tratamiento de regiones pequeñas.
- **Ventajas:** no exige `k`, descubre formas complejas y detecta observaciones atípicas.
- **Limitaciones:** no predice casos nuevos; algo más lento; `eps` depende fuertemente de escala; una sola densidad global funciona mal con densidades muy distintas.
- **Casos adecuados:** clusters no convexos con ruido y escala previamente controlada.
- **No conviene:** densidades extremadamente heterogéneas o necesidad de asignar todos los puntos.

### Métricas de clustering

- **ARI (adjusted Rand index) y NMI (normalized mutual information):** comparan una partición con ground truth sin depender de los números asignados a los clusters. En la interpretación presentada, 0 corresponde a asignación aleatoria y 1 a coincidencia perfecta.
- **Silhouette:** mide compactación/separación sin etiquetas; valores más altos son mejores y 1 es perfecto. Puede preferir clusters compactos aunque una forma compleja sea más plausible.
- **Inspección cualitativa:** revisar ejemplos, centros, extremos, ruido y tamaños sigue siendo necesaria para establecer significado.
- **Robustez:** el capítulo propone como estrategia mejor comparar resultados tras perturbar datos o parámetros, aunque señala que no estaba implementada directamente en scikit-learn al escribirse el libro.

## 6. Fórmulas esenciales

Las fórmulas siguientes son una **formalización operativa derivada de las descripciones verbales del capítulo**; el capítulo presenta principalmente algoritmos, figuras y código, no derivaciones algebraicas extensas.

### Estandarización por variable

\[
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
\]

- \(x_{ij}\): valor de la variable \(j\) en la muestra \(i\); \(\mu_j\) y \(\sigma_j\): media y desviación aprendidas solo en entrenamiento; \(z_{ij}\): valor transformado.
- Produce media aproximada 0 y varianza 1 en entrenamiento.
- Supuesto práctico: media y varianza son resúmenes adecuados; con outliers fuertes considerar estadísticas robustas.

### Escalado min-max

\[
z_{ij}=\frac{x_{ij}-\min_{\text{train}}(x_j)}{\max_{\text{train}}(x_j)-\min_{\text{train}}(x_j)}
\]

- Los extremos se estiman en entrenamiento. Un valor nuevo puede producir \(z<0\) o \(z>1\); no implica error.
- Mide la posición relativa respecto del rango de entrenamiento.

### Normalización por muestra

\[
z_i=\frac{x_i}{\lVert x_i\rVert_2},\qquad
\lVert x_i\rVert_2=\sqrt{\sum_j x_{ij}^2}
\]

- \(x_i\): vector de una observación; \(z_i\): vector de longitud 1.
- Conserva dirección y elimina magnitud; aplica cuando el ángulo es más importante que la longitud.

### Reconstrucción por componentes

\[
X \approx WH
\]

- \(X\): datos; \(W\): coeficientes de cada muestra; \(H\): componentes.
- En NMF se exige \(W\ge 0\) y \(H\ge 0\). En PCA, las componentes son ortogonales y se ordenan por varianza explicada.
- La calidad de reconstrucción mejora al aumentar componentes, pero una reconstrucción mejor no garantiza mayor utilidad para la tarea.

### Objetivo geométrico de k-means

\[
\sum_{k=1}^{K}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert_2^2
\]

- \(K\): número de clusters; \(C_k\): observaciones asignadas al grupo \(k\); \(\mu_k\): centro medio.
- Resume la idea del capítulo de minimizar la variación dentro de clusters. Menor valor no prueba significado semántico y decrece al aumentar \(K\).

## 7. Implementación práctica

Los fragmentos mantienen las ideas del capítulo con API actual. **Nota práctica - adaptación moderna:** se usa `Pipeline` para que el escalador se ajuste dentro de cada partición y parámetros actuales explícitos como `n_init` en `KMeans`.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(
    MinMaxScaler(),
    SVC(C=100),
)
model.fit(X_train, y_train)
valid_accuracy = model.score(X_valid, y_valid)
```

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, random_state=42),
)
X_train_2d = pca_pipeline.fit_transform(X_train)
X_valid_2d = pca_pipeline.transform(X_valid)
```

```python
from sklearn.decomposition import NMF

# Requisito: X_train_nonnegative y X_valid_nonnegative no contienen negativos.
nmf = NMF(n_components=15, init="nndsvda", random_state=42, max_iter=1000)
X_train_nmf = nmf.fit_transform(X_train_nonnegative)
X_valid_nmf = nmf.transform(X_valid_nonnegative)
components = nmf.components_
```

```python
from sklearn.manifold import TSNE

# Solo exploración del conjunto incluido; no existe transform para casos nuevos.
embedding = TSNE(n_components=2, random_state=42).fit_transform(X_sample)
```

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_train)

kmeans_labels = KMeans(
    n_clusters=3, n_init=10, random_state=42
).fit_predict(X_scaled)

dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)

ari = adjusted_rand_score(y_reference, dbscan_labels)  # solo si existe referencia
if len(set(dbscan_labels)) > 1:
    silhouette = silhouette_score(X_scaled, dbscan_labels)
```

```python
from scipy.cluster.hierarchy import dendrogram, ward

linkage_matrix = ward(X_scaled)
dendrogram(linkage_matrix)
```

## 8. Supuestos y condiciones de aplicación

1. La distancia entre observaciones debe ser significativa para k-means, clustering aglomerativo, DBSCAN y t-SNE; esto exige representación y escalas adecuadas.
2. Estadísticas de preprocesamiento se aprenden solo con entrenamiento si habrá evaluación supervisada posterior.
3. PCA asume que una proyección lineal por varianza capta estructura útil; las variables de gran escala dominarán si no se estandarizan.
4. `whiten=True` iguala escalas de componentes y puede beneficiar métodos posteriores, a costa de eliminar sus varianzas relativas.
5. NMF requiere no negatividad y una interpretación aditiva razonable; sus componentes no tienen orden natural.
6. t-SNE requiere analizar conjuntamente los puntos que aparecerán en el mapa y no permite inferencia fuera de muestra en la interfaz descrita.
7. k-means presupone clusters representables por medias y geometría aproximadamente convexa.
8. Ward busca minimizar aumento de varianza y tiende a clusters de tamaño similar; `average` o `complete` pueden adaptarse mejor a tamaños desiguales.
9. DBSCAN presupone una escala de densidad global expresable mediante `eps` y `min_samples`.
10. ARI/NMI requieren etiquetas de referencia; silhouette no requiere etiquetas, pero solo evalúa compactación/separación geométrica.
11. Ningún resultado no supervisado adquiere significado semántico sin inspección o validación ligada al objetivo.

## 9. Errores y riesgos

- Ajustar un scaler, PCA o NMF con todos los datos antes de separar validación: filtra estadísticas del conjunto de evaluación.
- Ajustar transformaciones independientes en train y test: ambos quedan en sistemas de coordenadas incompatibles.
- Interpretar valores de test fuera de `[0, 1]` tras `MinMaxScaler` como fallo y reajustar el scaler.
- Omitir escalado antes de algoritmos de distancia/densidad, permitiendo que una variable domine.
- Tratar PCA como selección de variables originales: cada componente suele combinar muchas.
- Interpretar el signo de una componente PCA como absoluto; su orientación puede invertirse sin cambiar el modelo.
- Usar NMF con negativos o asumir que “componente 1” es intrínsecamente más importante.
- Convertir t-SNE en feature engineering de producción o comparar literalmente distancias globales del mapa.
- Elegir `n_clusters=8` en k-means por ser un default histórico sin justificación del problema.
- Suponer que conocer el número correcto de clusters garantiza que k-means recupere la estructura.
- Ignorar inicialización aleatoria y no fijar `random_state` al comparar resultados.
- Usar accuracy para clustering: dos particiones idénticas con etiquetas permutadas pueden dar accuracy 0.
- Optimizar solo silhouette: puede favorecer grupos compactos y penalizar clusters curvos correctos.
- Leer `-1` de DBSCAN como un cluster ordinario o usarlo para indexar arrays sin tratamiento explícito.
- Aumentar `eps` hasta eliminar ruido: puede fusionar todo en un solo cluster. Un `eps` demasiado bajo etiqueta todo como ruido.
- Tratar estabilidad o un score alto como prueba de significado. El capítulo exige análisis manual del contenido de los clusters.
- Seleccionar ejemplos “interesantes” sin declarar la selección; puede producir una visión sesgada de la calidad del clustering.
- En datos grandes, ignorar el coste de t-SNE, clustering jerárquico o inspección exhaustiva; PCA puede reducir cálculo, pero modifica la representación.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo cuando un modelo sensible a escala rinde mal, hay muchas variables correlacionadas, se pide una visualización 2D o se necesita segmentación sin etiquetas.
- Baseline supervisado: `Pipeline(StandardScaler/MinMaxScaler, estimator)` para SVM o métodos sensibles a escala. No implementar escalado manual.
- Baseline no supervisado: `StandardScaler` + k-means si se espera geometría compacta; `StandardScaler` + DBSCAN si hay ruido o formas complejas.
- Mejora rápida: comparar dos o tres valores de `n_clusters`, inspeccionar `np.bincount(labels)` y centros/ejemplos; para DBSCAN, variar `eps` y observar ruido y número de clusters.
- PCA con 2 componentes sirve para EDA; con más componentes puede acelerar modelos o clustering. Debe ajustarse dentro del pipeline de validación.
- NMF se activa ante matrices no negativas con semántica aditiva, como conteos o intensidades; revisar componentes y muestras con coeficientes altos.
- t-SNE se activa para una visualización exploratoria final sobre una muestra manejable. Bajo presión, evitar usarlo para seleccionar un modelo o como paso de inferencia.
- Señales del dataset: escalas heterogéneas -> scaler; outliers fuertes -> `RobustScaler`; correlación/alta dimensión -> PCA; no negatividad aditiva -> NMF; grupos curvos y ruido -> DBSCAN; jerarquía esperada -> aglomerativo/dendrograma.
- No implementar bajo presión una evaluación no supervisada sofisticada basada solo en una métrica. Priorizar visualización, estabilidad básica, tamaños y ejemplos representativos.

## 11. Checklist operativo

- [ ] Definir si la meta es escalar, visualizar, comprimir, extraer partes o agrupar.
- [ ] Separar entrenamiento y validación antes de `fit` de cualquier transformación.
- [ ] Ajustar el preprocesamiento solo con entrenamiento y reutilizarlo sin reajuste.
- [ ] Confirmar que las escalas y la distancia tienen sentido para el algoritmo.
- [ ] Verificar no negatividad antes de NMF.
- [ ] Fijar `random_state` en métodos estocásticos.
- [ ] Elegir `n_components` o granularidad por objetivo, no por default.
- [ ] Comparar tamaños de clusters y proporción de ruido.
- [ ] Usar ARI/NMI, no accuracy, si hay etiquetas de referencia.
- [ ] Tratar silhouette como apoyo, no como prueba semántica.
- [ ] Inspeccionar centros, ejemplos típicos, extremos y puntos de ruido.
- [ ] Confirmar si el método puede transformar o predecir casos nuevos.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Ajuste seguro de transformaciones y prevención de leakage | `knowledge/preprocessing/train_only_fit.md` | Alta |
| Standard, Robust, MinMax y normalización por fila | `knowledge/preprocessing/feature_scaling.md` | Alta |
| PCA: visualización, compresión, whitening e interpretación | `knowledge/feature_engineering/pca.md` | Alta |
| NMF y representaciones aditivas no negativas | `knowledge/feature_engineering/nmf.md` | Media |
| t-SNE para visualización exploratoria | `knowledge/eda/tsne_visualization.md` | Media |
| Selección entre k-means, aglomerativo y DBSCAN | `knowledge/models/clustering.md` | Alta |
| k-means y cuantización vectorial | `knowledge/models/kmeans.md` | Alta |
| Clustering jerárquico, enlaces y dendrogramas | `knowledge/models/hierarchical_clustering.md` | Media |
| DBSCAN, ruido, `eps` y `min_samples` | `knowledge/models/dbscan.md` | Alta |
| ARI, NMI, silhouette y etiquetas permutables | `knowledge/metrics/clustering_metrics.md` | Alta |
| Inspección semántica y estabilidad de clusters | `knowledge/validation/unsupervised_evaluation.md` | Alta |
| Reducción previa a clustering para coste y representación | `knowledge/feature_engineering/dimensionality_reduction.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito, tipos y desafíos del aprendizaje no supervisado | “Types of Unsupervised Learning” y “Challenges in Unsupervised Learning”, pp. 131–132 |
| Escaladores y sus diferencias | “Preprocessing and Scaling” / “Different Kinds of Preprocessing”, pp. 132–134 |
| `fit`/`transform`, train/test y leakage | “Applying Data Transformations” y “Scaling Training and Test Data the Same Way”, pp. 134–138 |
| Efecto del escalado en SVC | “The Effect of Preprocessing on Supervised Learning”, pp. 138–140 |
| PCA: mecanismo y reducción | “Principal Component Analysis (PCA)”, pp. 140–142 |
| PCA para visualización e interpretación de componentes | “Applying PCA to the cancer dataset for visualization”, pp. 142–147 |
| PCA para extracción, whitening y reconstrucción | “Eigenfaces for feature extraction”, pp. 147–155 |
| NMF: restricciones, componentes y aplicaciones | “Non-Negative Matrix Factorization (NMF)”, pp. 156–163 |
| t-SNE y límites fuera de muestra | “Manifold Learning with t-SNE”, pp. 163–168 |
| k-means: algoritmo, API y fallos | “k-Means Clustering” y “Failure cases of k-means”, pp. 168–176 |
| k-means como descomposición/cuantización vectorial | “Vector quantization, or seeing k-means as decomposition”, pp. 176–181 |
| Clustering aglomerativo y criterios de enlace | “Agglomerative Clustering”, pp. 182–184 |
| Jerarquía y dendrogramas | “Hierarchical clustering and dendrograms”, pp. 184–187 |
| DBSCAN, puntos core/frontera/ruido y ajuste | “DBSCAN”, pp. 187–191 |
| ARI/NMI, error de accuracy y silhouette | “Comparing and Evaluating Clustering Algorithms”, pp. 191–195 |
| Comparación cualitativa en datos reales | “Comparing algorithms on the faces dataset”, pp. 195–207 |
| Fortalezas relativas de clustering y cierre | “Summary of Clustering Methods” y “Summary and Outlook”, pp. 207–208 |
| Interfaz de estimadores `fit`, `predict`, `transform`, `score` | “Summary of the Estimator Interface”, p. 209 |

### Estado de lectura

- Páginas impresas revisadas: 131–209 (páginas PDF 1–79).
- Página PDF 80: en blanco, sin contenido del capítulo.
- Secciones ilegibles o no procesables: ninguna.
