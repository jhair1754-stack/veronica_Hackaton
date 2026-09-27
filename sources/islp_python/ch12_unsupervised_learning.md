---
title: "12. Unsupervised Learning"
topics:
* "Principal Components Analysis (PCA)"
* "Missing values and matrix completion"
* "K-means and hierarchical clustering"
target_knowledge_files:
* "knowledge/feature_engineering/pca.md"
* "knowledge/preprocessing/missing_values_matrix_completion.md"
* "knowledge/models/clustering.md"
* "knowledge/eda/unsupervised_exploration.md"
---

# 12. Unsupervised Learning

## 1. Propósito del capítulo

El capítulo presenta técnicas para analizar datos sin una variable respuesta observada. Su objetivo no es predecir `Y`, sino descubrir estructura en las variables u observaciones, visualizar datos de muchas dimensiones, reducir dimensionalidad y encontrar subgrupos. El foco está en PCA, imputación/matrix completion y clustering.

## 2. Conceptos esenciales

- En aprendizaje no supervisado solo se dispone de una matriz de predictores `X`; no existe una etiqueta que permita medir directamente el acierto de un modelo.
- PCA transforma variables correlacionadas en componentes ortogonales. Cada componente es una combinación lineal de las variables originales y se ordena por varianza explicada.
- La escala importa: una variable con unidades o varianza mucho mayores puede dominar PCA y las distancias de clustering. El capítulo usa variables centradas y, cuando corresponde, escaladas.
- La proporción de varianza explicada (PVE) ayuda a decidir cuántos componentes conservar, pero no equivale a desempeño predictivo.
- Clustering asigna observaciones a grupos similares. Los grupos descubiertos son una construcción del método, no necesariamente categorías reales.
- K-means requiere elegir `K` y optimiza una medida de variación intra-grupo; diferentes inicializaciones pueden producir soluciones distintas.
- Clustering jerárquico construye un dendrograma mediante fusiones sucesivas. El número final de grupos se obtiene cortando el dendrograma a una altura elegida.
- La elección de distancia, linkage, escalado y número de grupos cambia el resultado; no hay una partición única garantizada por los datos.
- En datos incompletos, la matrix completion estima entradas faltantes usando estructura de bajo rango; la imputación debe distinguir valores observados de valores estimados.

## 3. Flujo o procedimiento presentado

1. Inspeccionar la matriz: tipos de variables, escalas, correlaciones y valores faltantes.
2. Definir si el objetivo es visualización, reducción de dimensionalidad, imputación o descubrimiento de grupos.
3. Centrar y, si las escalas lo exigen, estandarizar las variables.
4. Ajustar PCA sobre la matriz completa o sobre una matriz previamente completada.
5. Examinar loadings, scores, biplots y PVE; decidir cuántos componentes conservar según el uso.
6. Para clustering, elegir una distancia y un algoritmo; probar valores razonables de `K` o cortes del dendrograma.
7. Evaluar estabilidad y plausibilidad de la estructura encontrada, incluyendo visualización con PCA cuando sea útil.
8. Interpretar grupos y componentes con cautela: usar variables originales, contexto del problema y, si existen, etiquetas externas solo para interpretar, no para ajustar.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Muchas variables correlacionadas y necesidad de visualización | PCA | Resume variación en pocos ejes | Escalar cuando las unidades difieran |
| Se necesita una representación compacta antes de un método supervisado | PCA | Reduce dimensión y puede eliminar ruido | Ajustar PCA solo con el conjunto de entrenamiento |
| Se buscan grupos con número fijado de antemano | K-means | Optimiza asignaciones a `K` centroides | Repetir inicializaciones y comprobar sensibilidad |
| No se conoce el número de grupos y se desea explorar varios cortes | Clustering jerárquico | Un dendrograma muestra fusiones a varias alturas | El linkage y la distancia afectan el árbol |
| Datos con entradas faltantes y estructura aproximadamente de bajo rango | Matrix completion | Estima valores a partir de patrones compartidos | No confundir imputaciones con observaciones |
| Variables con escalas muy distintas | Estandarización antes de PCA/clustering | Evita que una escala domine | La decisión debe corresponder al significado de las unidades |

## 5. Métodos y técnicas

### PCA

- **Objetivo:** encontrar direcciones ortogonales que expliquen la mayor variación posible.
- **Entradas:** matriz numérica `X`, normalmente centrada y opcionalmente escalada.
- **Salida:** loadings (`phi`), scores/proyecciones de observaciones y varianza explicada por componente.
- **Funcionamiento:** el primer componente maximiza la varianza proyectada; los siguientes maximizan la varianza restante bajo ortogonalidad.
- **Hiperparámetros/decisiones:** número de componentes (`n_components`) y si se escala (`scale`/`StandardScaler`).
- **Ventajas:** visualización de alta dimensión, compresión, reducción de colinealidad y preprocesamiento.
- **Limitaciones:** componentes lineales, sensibles al escalado y a outliers; la mayor varianza no necesariamente es la señal útil para predecir.
- **Adecuado:** variables numéricas relacionadas y objetivos exploratorios o de reducción.
- **No conviene:** datos predominantemente categóricos sin una codificación apropiada, o cuando la interpretabilidad directa de variables sea prioritaria.

### Matrix completion e imputación

- **Objetivo:** completar entradas faltantes explotando una representación de bajo rango.
- **Entradas:** matriz parcialmente observada y máscara de valores conocidos.
- **Salida:** matriz completada; los valores observados deben conservarse o controlarse explícitamente.
- **Funcionamiento:** inicia con una matriz simple y actualiza entradas faltantes mediante componentes principales/estructura de bajo rango hasta converger.
- **Decisiones:** número de componentes o rango efectivo, regla de actualización y criterio de convergencia.
- **Ventajas:** aprovecha relaciones entre variables y puede ser mejor que imputar cada columna independientemente.
- **Limitaciones:** presupone estructura compartida; puede suavizar señales reales y producir resultados inestables con demasiados faltantes.
- **Adecuado:** matrices con patrones aproximadamente de bajo rango, como ratings o mediciones relacionadas.
- **No conviene:** ausencia de estructura común, missingness muy extrema o cuando el mecanismo de faltantes no sea compatible con una imputación simple.

### K-means

- **Objetivo:** dividir `n` observaciones en `K` grupos compactos.
- **Entradas:** observaciones numéricas, distancia euclídea implícita y `K`.
- **Salida:** etiquetas de cluster y centroides.
- **Funcionamiento:** asigna cada observación al centroide más cercano y recalcula centroides; repite hasta que las asignaciones o la función objetivo se estabilicen.
- **Hiperparámetros:** `n_clusters`, inicialización, `n_init`, máximo de iteraciones y tolerancia.
- **Ventajas:** sencillo, rápido y útil como baseline exploratorio.
- **Limitaciones:** exige `K`, favorece grupos aproximadamente esféricos y de escala comparable, y depende de la inicialización.
- **Adecuado:** grupos compactos y variables numéricas escaladas.
- **No conviene:** grupos no convexos, densidades muy diferentes, muchos outliers o cuando no se pueda justificar `K`.

### Clustering jerárquico

- **Objetivo:** obtener una jerarquía de agrupaciones.
- **Entradas:** observaciones, medida de distancia y método de linkage.
- **Salida:** matriz de fusiones y dendrograma; un corte produce etiquetas.
- **Funcionamiento:** comienza con cada observación como cluster y fusiona iterativamente los clusters más próximos.
- **Hiperparámetros:** distancia, linkage (single, complete, average, centroid/ward según implementación) y altura o número del corte.
- **Ventajas:** permite explorar distintos números de grupos en un solo dendrograma y no requiere fijar `K` al inicio.
- **Limitaciones:** decisiones tempranas son difíciles de deshacer; sensible a distancia, linkage, escala y outliers; puede ser costoso en muestras grandes.
- **Adecuado:** análisis exploratorio y muestras donde un dendrograma sea interpretable.
- **No conviene:** grandes volúmenes sin una estrategia de coste, o cuando la distancia elegida no represente similitud útil.

## 6. Fórmulas esenciales

### Componente principal

`Z_m = sum_j phi_{jm} X_j`.

`X_j` es la variable `j`, `phi_{jm}` es el loading de esa variable en el componente `m`, y `Z_m` es el score de una observación en dicho componente. El vector de loadings tiene norma uno y los componentes son ortogonales. El signo de un componente es arbitrario: invertir todos sus signos no cambia la solución.

### PVE

`PVE_m = Var(Z_m) / sum_l Var(Z_l)`.

Mide la fracción de variación total explicada por el componente `m`; la PVE acumulada indica cuánta variación retiene una representación con varios componentes. Supone que la variación es el criterio relevante para el objetivo.

### Objetivo de K-means

`sum_{k=1}^K sum_{i in C_k} ||x_i - xbar_k||^2`.

`C_k` es el conjunto de observaciones del cluster `k`, `x_i` una observación y `xbar_k` su centroide. Mide variación intra-cluster y se minimiza iterativamente; no demuestra que los clusters sean verdaderos ni que el modelo generalice.

## 7. Implementación práctica

El laboratorio usa Python y APIs de `scikit-learn` para PCA, imputación y clustering. La siguiente adaptación moderna conserva las ideas del capítulo y separa ajuste de transformación para evitar leakage:

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# X_train/X_valid deben contener únicamente variables numéricas.
pca = make_pipeline(StandardScaler(), PCA(n_components=2))
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)

kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
train_labels = kmeans.fit_predict(X_train_pca)
valid_labels = kmeans.predict(X_valid_pca)

hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = hierarchical.fit_predict(X_train_pca)
```

**Nota práctica (adaptación moderna):** los nombres de argumentos y valores por defecto de `scikit-learn` pueden diferir de los usados en el libro. `n_init=20` y `random_state=42` hacen explícita la repetición y reproducibilidad; no son contenido nuevo atribuido al autor. Para un pipeline con datos faltantes, colocar el imputador antes del escalador y PCA, ajustándolo solo en entrenamiento.

## 8. Supuestos y condiciones de aplicación

- PCA y los métodos mostrados operan sobre entradas numéricas; la codificación de categóricas debe justificarse.
- Centrado y escalado deben reflejar la pregunta y las unidades; estandarizar no es automáticamente correcto.
- Distancias euclídeas y K-means suponen que la geometría de la escala elegida representa similitud.
- K-means funciona mejor con grupos aproximadamente compactos/esféricos y no está diseñado para outliers fuertes.
- Clustering jerárquico depende de la métrica y del linkage; el dendrograma no es independiente de esas decisiones.
- PCA es lineal y sensible a variables de gran varianza y observaciones extremas.
- Matrix completion presupone una estructura de bajo rango o patrones compartidos.
- La validación no supervisada no tiene una respuesta objetivo por defecto; la estabilidad, visualización y conocimiento del dominio son evidencias complementarias.

## 9. Errores y riesgos

- Aplicar escalado, PCA o imputación usando todo el dataset antes de separar entrenamiento y validación produce leakage en una tarea supervisada posterior.
- Interpretar cada cluster como una clase natural o causal sin validación externa.
- Elegir `K` solo porque genera una visualización atractiva.
- Usar una única inicialización de K-means y asumir que la solución es global.
- Leer un dendrograma como si la altura fuera una probabilidad o como si el linkage no importara.
- Omitir el escalado cuando una unidad domina la distancia, o escalar automáticamente sin considerar significado y outliers.
- Tratar imputaciones de matrix completion como datos observados y subestimar la incertidumbre.
- Confundir PVE alta con capacidad predictiva alta.
- Aplicar PCA para interpretación directa cuando los loadings combinan muchas variables o cambian con la muestra.
- En muestras grandes, clustering jerárquico puede tener costes de memoria y tiempo relevantes.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- Consultar este capítulo cuando no haya target, cuando se necesite explorar estructura, visualizar muchas variables o construir features compactas.
- Para un baseline, revisar escalas y faltantes, usar `StandardScaler` + PCA para una visualización de dos componentes y ejecutar K-means con varios valores pequeños de `K`.
- Mejorar el modelo comparando PVE acumulada, estabilidad de asignaciones, interpretabilidad de loadings y coherencia de los grupos.
- No implementar bajo presión una rutina compleja de matrix completion si una imputación simple y validada resuelve el baseline; tampoco presentar clusters como verdad de negocio sin evidencia.
- Activar estas técnicas ante alta colinealidad, cientos de variables, matriz de ratings, necesidad de segmentación o ausencia de etiquetas.
- Recordar que en una competición con target, PCA/clustering son transformaciones o features: deben ajustarse dentro de cada fold o pipeline.

## 11. Checklist operativo

- [ ] Confirmar si existe target y si el objetivo es exploratorio, compresión, imputación o segmentación.
- [ ] Revisar escalas, tipos, outliers y patrón de valores faltantes.
- [ ] Separar entrenamiento y validación antes de ajustar transformaciones.
- [ ] Elegir explícitamente centrado/escalado y documentar la decisión.
- [ ] Para PCA, revisar loadings y PVE acumulada, no solo el gráfico.
- [ ] Para K-means, probar varias inicializaciones y justificar `K`.
- [ ] Para clustering jerárquico, fijar distancia/linkage y documentar el corte.
- [ ] Comprobar estabilidad y plausibilidad de los grupos.
- [ ] Mantener separadas las imputaciones de los valores observados.
- [ ] No convertir una estructura exploratoria en una afirmación causal o de negocio sin evidencia adicional.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Objetivo y límites del aprendizaje no supervisado | `knowledge/problem_definition/unsupervised_learning.md` | Alta |
| Escalado, PCA y reducción de dimensionalidad | `knowledge/feature_engineering/pca.md` | Alta |
| Visualización mediante scores, loadings y PVE | `knowledge/eda/unsupervised_exploration.md` | Alta |
| Matrix completion e imputación estructural | `knowledge/preprocessing/missing_values_matrix_completion.md` | Media |
| K-means: objetivo, inicialización y elección de K | `knowledge/models/kmeans.md` | Alta |
| Clustering jerárquico, distancias y linkage | `knowledge/models/hierarchical_clustering.md` | Alta |
| Ajuste de transformaciones sin leakage | `knowledge/validation/unsupervised_pipelines.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------ | ------------------------------ |
| Propósito y conceptos esenciales | §12, §12.1, pp. 503–505 |
| PCA y PVE | §12.2, §§12.2.1–12.2.5, pp. 505–515 |
| Missing values y matrix completion | §12.3, pp. 515–521 |
| K-means | §12.4.1, pp. 521–526 |
| Clustering jerárquico | §12.4.2, pp. 526–533 |
| Cuestiones prácticas de clustering | §12.4.3, pp. 531–535 |
| Implementación y ejemplos | §12.5, pp. 535–552 |
| NCI60 y comparación exploratoria | §12.5.4, pp. 546–552 |
| Ejercicios revisados como contexto del capítulo | §12.6, pp. 552–562 |

No se detectaron páginas ilegibles en la extracción del PDF. El resumen se basa únicamente en las 54 páginas del capítulo proporcionado (páginas impresas 503–562); no se consultaron fuentes externas ni otros capítulos.
