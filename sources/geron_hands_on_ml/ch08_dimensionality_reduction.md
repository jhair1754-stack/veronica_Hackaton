---
title: "Dimensionality Reduction"
topics:
  - "Curse of dimensionality"
  - "PCA, Kernel PCA y LLE"
  - "Manifold Learning"
target_knowledge_files:
  - "knowledge/preprocessing/dimensionality_reduction.md"
  - "knowledge/visualization/manifold_learning.md"
  - "knowledge/tuning/kernel_pca.md"
---

# 8. Dimensionality Reduction

## 1. Propósito del capítulo

El capítulo presenta técnicas para reducir el número de variables de un conjunto de datos. La reducción puede acelerar el entrenamiento, disminuir el tamaño de los datos y facilitar la visualización, aunque introduce pérdida de información y complejidad adicional en el pipeline. El foco está en la maldición de la dimensionalidad, la proyección, Manifold Learning, PCA, Kernel PCA y LLE.

## 2. Conceptos esenciales

- **Maldición de la dimensionalidad (curse of dimensionality):** al aumentar las dimensiones, los datos se vuelven muy dispersos, las instancias quedan lejos unas de otras y las predicciones dependen de extrapolaciones mayores. El número de instancias necesario para alcanzar una densidad dada crece exponencialmente con las dimensiones, aumentando también el riesgo de sobreajuste.
- **Reducción de dimensionalidad:** transforma los datos a un espacio de menor dimensión. Normalmente conserva parte de la estructura o varianza, pero no toda la información.
- **Proyección:** busca un subespacio de menor dimensión cercano a los datos y proyecta las instancias sobre él. Es adecuada cuando los datos están cerca de un subespacio lineal.
- **Manifold Learning:** modela una variedad (manifold) de baja dimensión doblada o retorcida dentro de un espacio de mayor dimensión. Depende de la hipótesis de que muchos datos reales están próximos a una variedad de menor dimensión.
- **Hipótesis de la variedad:** no garantiza que la tarea supervisada sea más sencilla después de reducir dimensiones. Una frontera de decisión simple en el espacio original puede volverse compleja en la variedad y viceversa.
- **PCA (Principal Component Analysis):** encuentra ejes ortogonales que maximizan sucesivamente la varianza conservada. Proyectar sobre los primeros componentes minimiza, en el sentido descrito por el capítulo, la distancia cuadrática media entre datos y proyecciones.
- **Varianza explicada:** proporción de la varianza total situada en cada componente principal. Permite elegir el número de dimensiones según un umbral de conservación.
- **Error de reconstrucción:** distancia entre los datos originales y los datos tras reducir y volver a expandir. Cuantifica información perdida en una reconstrucción.
- **Kernel PCA:** extiende PCA mediante el kernel trick para obtener proyecciones no lineales; puede conservar clusters o desenrollar variedades.
- **LLE (Locally Linear Embedding):** preserva relaciones lineales locales entre vecinos cercanos para obtener una representación de menor dimensión. Es especialmente útil para desenrollar variedades retorcidas con poco ruido.

Conclusión práctica derivada: la reducción debe evaluarse por su efecto en el objetivo final y no solo por la varianza o la apariencia de una visualización.

## 3. Flujo o procedimiento presentado

1. Identificar si el número de variables está causando entrenamiento lento, dispersión excesiva o dificultades de visualización.
2. Intentar primero entrenar con los datos originales si el coste es aceptable; la reducción no suele mejorar el rendimiento por sí misma.
3. Determinar si los datos parecen cercanos a un subespacio lineal o a una variedad no lineal.
4. Ajustar el transformador únicamente con el conjunto de entrenamiento y transformar validación y prueba con el transformador ajustado.
5. En PCA, elegir dimensiones mediante un umbral de varianza, un gráfico de varianza acumulada o 2–3 dimensiones si el objetivo es visualizar.
6. Integrar la reducción y el modelo supervisado en un pipeline cuando haya que seleccionar hiperparámetros sin leakage.
7. Comparar tiempo, métrica del modelo y, cuando corresponda, error de reconstrucción.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Datos aproximadamente lineales y necesidad de conservar varianza | PCA | Proyección ortogonal sobre los ejes de mayor varianza | Centrar los datos; Scikit-Learn lo hace automáticamente |
| Dataset muy grande o que no cabe en memoria | Incremental PCA | Permite mini-lotes o arrays mapeados en disco | Usar `partial_fit`; controlar el tamaño de lote |
| Se necesitan los primeros componentes con muchas dimensiones y `d` mucho menor que `n` | Randomized PCA | Aproxima rápidamente los primeros componentes | Es una aproximación; comparar si la precisión es crítica |
| Estructura no lineal, clusters o variedad retorcida | Kernel PCA | Proyección no lineal mediante kernels | Ajustar kernel y `gamma`; la selección no es obvia sin un objetivo |
| Variedad retorcida con poco ruido y relaciones locales importantes | LLE | Conserva relaciones con vecinos y puede desenrollar la variedad | Escala mal por el término cuadrático en el número de instancias |
| Visualización de datos de alta dimensión | PCA, t-SNE u otra técnica citada | Permite representar en 2D/3D y buscar patrones | Una visualización no demuestra por sí sola una estructura causal o útil para el modelo |

## 5. Métodos y técnicas

### PCA

- **Objetivo:** reducir a `d` dimensiones conservando tanta varianza como sea posible.
- **Entradas:** matriz `X` de instancias por variables y `n_components` como número de componentes o fracción de varianza.
- **Salida:** `X_reduced` con `d` componentes; `components_` contiene los ejes; `explained_variance_ratio_` contiene la varianza explicada por componente.
- **Funcionamiento:** centra los datos, obtiene los componentes mediante SVD y proyecta sobre los primeros ejes.
- **Hiperparámetros:** `n_components`; `svd_solver` (`"full"`, `"randomized"` o `"auto"`).
- **Ventajas:** general, rápido frente a problemas originales de alta dimensión, útil para compresión y visualización.
- **Limitaciones:** lineal; pierde información; maximizar varianza no equivale necesariamente a maximizar la métrica supervisada.
- **Casos adecuados:** variables correlacionadas o datos cercanos a un subespacio lineal.
- **No conviene:** si la geometría relevante es fuertemente no lineal o la reducción no aporta un beneficio medible.

### Randomized PCA

Calcula una aproximación de los primeros `d` componentes con menor coste cuando `d` es mucho menor que la dimensión original. El capítulo indica complejidad aproximada `O(m d²) + O(d³)`, frente a `O(m n²) + O(n³)` para SVD completa. Scikit-Learn puede seleccionarlo automáticamente con `svd_solver="auto"` en ciertos tamaños.

### Incremental PCA

Recibe mini-lotes mediante `partial_fit()` y permite trabajar con grandes conjuntos o datos que llegan progresivamente. También puede usar `numpy.memmap` para mantener los datos en disco y limitar el uso de memoria. Su salida y transformación son análogas a PCA, pero el ajuste es incremental.

### Kernel PCA

- **Objetivo:** realizar una reducción no lineal usando una representación implícita en un espacio de características.
- **Entradas:** `X`, `n_components`, `kernel` y parámetros como `gamma`.
- **Salida:** representación reducida; con `fit_inverse_transform=True`, también una preimagen aproximada mediante `inverse_transform()`.
- **Funcionamiento:** aplica el kernel trick y PCA en el espacio de características.
- **Hiperparámetros:** kernel (`"rbf"`, `"sigmoid"`, entre otros), `gamma`, `n_components`, `fit_inverse_transform`.
- **Ventajas:** puede preservar clusters y desenrollar estructuras no lineales.
- **Limitaciones:** mayor coste y selección de hiperparámetros menos directa; la reconstrucción exacta no está disponible en general.
- **Casos adecuados:** reducción como preparación de una tarea supervisada o para variedades no lineales.
- **No conviene:** bajo presión de tiempo si PCA lineal ya ofrece una métrica suficiente o si el coste de ajuste es excesivo.

### LLE

- **Objetivo:** representar los datos en pocas dimensiones preservando relaciones locales.
- **Entradas:** `X`, `n_components` y `n_neighbors`.
- **Salida:** embedding de baja dimensión.
- **Funcionamiento:** calcula pesos para reconstruir cada instancia a partir de sus vecinos y luego busca posiciones de baja dimensión que mantengan esos pesos.
- **Hiperparámetros:** principalmente `n_neighbors` y `n_components`.
- **Ventajas:** desenrolla variedades retorcidas y conserva bien la estructura local.
- **Limitaciones:** las distancias globales pueden deformarse; sensible al ruido; escala mal para datasets muy grandes.
- **Casos adecuados:** variedades no lineales con vecindarios locales informativos y poco ruido.
- **No conviene:** cuando la escala del dataset hace prohibitivo el coste o cuando se necesita preservar distancias globales.

El capítulo también menciona proyecciones aleatorias, MDS, Isomap, t-SNE y LDA como alternativas: preservan, respectivamente, distancias aproximadas, distancias, distancias geodésicas, vecindarios de similares/disímiles o ejes discriminativos entre clases. No se desarrollan aquí porque el fragmento solo las resume.

## 6. Fórmulas esenciales

1. **Proyección PCA:** `X_d-proj = X W_d`.
   - `X` es la matriz centrada de datos, `W_d` contiene los primeros `d` vectores principales y `X_d-proj` es la representación reducida.
   - Mide la coordenada de cada instancia en los ejes seleccionados.
   - Retiene la máxima varianza posible entre proyecciones lineales de dimensión `d`.
   - Supuesto importante: PCA opera sobre datos centrados; la clase de Scikit-Learn centra automáticamente.

2. **Descomposición SVD:** `X = U Σ Vᵀ`.
   - `X` es la matriz centrada; `U`, `Σ` y `Vᵀ` son los factores SVD; las columnas de `V` definen los componentes principales.
   - Permite obtener los ejes de PCA.
   - El signo de un vector principal no es estable: invertirlo no cambia el eje ni la solución geométrica.

3. **Reconstrucción PCA:** `X_recovered = X_d-proj W_dᵀ`.
   - `X_d-proj` es la representación reducida y `W_dᵀ` devuelve la aproximación al espacio original.
   - Permite medir la pérdida mediante el error cuadrático entre `X` y `X_recovered`.

4. **LLE, relación local:** `x(i) ≈ Σ_j w(i,j) x(j)`, con pesos cero fuera de los `k` vecinos y `Σ_j w(i,j)=1`.
   - `x(i)` es una instancia, `x(j)` sus vecinos y `w(i,j)` los pesos de reconstrucción.
   - Mide cómo se expresa localmente cada instancia a partir de sus vecinos; después se buscan coordenadas `z(i)` que conserven esos pesos.
   - Requiere que la estructura local sea informativa; el capítulo advierte de su mala escalabilidad.

## 7. Implementación práctica

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_valid_reduced = pca.transform(X_valid)
```

La separación `fit_transform`/`transform` evita ajustar la reducción con validación. Para visualizar, puede usarse `PCA(n_components=2)`. Para datos grandes:

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=154)
for X_batch in batches:
    ipca.partial_fit(X_batch)
X_reduced = ipca.transform(X_train)
```

Para una reducción no lineal:

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = kpca.fit_transform(X_train)
```

Para LLE:

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X_train)
```

Para seleccionar Kernel PCA según una tarea supervisada, el capítulo propone un `Pipeline` con `KernelPCA` y regresión logística, y `GridSearchCV` sobre `kernel` y `gamma`. Esta estructura es aplicable para que la validación ajuste conjuntamente la transformación y el modelo. La sintaxis se conserva conceptualmente del libro; no se requiere una actualización de API para estos ejemplos.

## 8. Supuestos y condiciones de aplicación

- Hay redundancia, correlación o una estructura de baja dimensión que permita comprimir la información.
- PCA supone que una proyección lineal es una representación útil y necesita datos centrados.
- La hipótesis de variedad supone que los datos de alta dimensión están cerca de una variedad de menor dimensión.
- La hipótesis adicional de que la tarea será más simple en la variedad puede fallar.
- LLE requiere vecinos significativos y funciona mejor con poco ruido.
- Los componentes principales pueden cambiar de signo, y componentes con varianzas cercanas pueden rotar o intercambiarse sin cambiar esencialmente el subespacio.
- La transformación debe ajustarse usando solo entrenamiento para evitar leakage.
- Kernel PCA requiere una elección razonable de kernel y parámetros; no tiene una métrica no supervisada obvia para seleccionarlos.

## 9. Errores y riesgos

- Reducir dimensiones automáticamente esperando una mejora de precisión: el beneficio general señalado es normalmente velocidad, no rendimiento.
- Ajustar PCA, Kernel PCA o LLE con todo el dataset antes de validar: filtra información de validación al entrenamiento.
- Elegir dimensiones solo por varianza explicada cuando el objetivo es supervisado: la varianza no necesariamente contiene la señal predictiva.
- Proyectar un Swiss roll con PCA lineal y aplastar capas que deberían mantenerse separadas.
- Interpretar un embedding 2D como preservación de todas las distancias o relaciones originales.
- Usar LLE en un dataset masivo sin considerar su coste, especialmente el término `O(d m²)` indicado en el capítulo.
- Ignorar la pérdida de información y el error de reconstrucción.
- Añadir una transformación que vuelve el pipeline más difícil de mantener sin que compense el coste de entrenamiento.
- Olvidar que la reducción puede modificar negativamente el modelo aunque la visualización parezca más clara.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Consulta este capítulo cuando el dataset tenga muchas variables correlacionadas, el entrenamiento sea demasiado lento, haya riesgo de dispersión o necesites visualizar clusters. Para un baseline, prueba PCA dentro de un pipeline, con un umbral de varianza como 95% o con un número pequeño de componentes si la tarea es visual. Compara tiempo y métrica contra el modelo sin reducción.

PCA suele ser la primera opción por simplicidad. Considera Incremental PCA si la memoria es el cuello de botella; Kernel PCA o LLE solo si hay evidencia de estructura no lineal y tiempo para validarlos. Bajo presión, no implementes técnicas no lineales complejas sin una mejora medible. Señales del dataset: muchas columnas, alta correlación, entrenamiento lento, datos visuales de gran dimensión, clusters curvos o una variedad retorcida.

## 11. Checklist operativo

- [ ] Medir el coste y rendimiento del baseline sin reducción.
- [ ] Separar entrenamiento, validación y prueba antes de ajustar el transformador.
- [ ] Comprobar correlación, redundancia y escala de las variables.
- [ ] Probar PCA como baseline de reducción lineal.
- [ ] Elegir `n_components` por varianza acumulada o por el objetivo de visualización.
- [ ] Usar Incremental PCA si el conjunto no cabe cómodamente en memoria.
- [ ] Comparar Kernel PCA o LLE solo si la geometría parece no lineal.
- [ ] Integrar transformación y estimador en un pipeline durante la validación.
- [ ] Comparar métrica, tiempo, memoria y, cuando proceda, error de reconstrucción.
- [ ] Confirmar que la reducción no ha introducido leakage ni ha empeorado el modelo.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Maldición de la dimensionalidad y criterios de uso | `knowledge/problem_definition/dimensionality_reduction.md` | Alta |
| PCA, centrado, componentes y varianza explicada | `knowledge/preprocessing/pca.md` | Alta |
| Selección de dimensiones y error de reconstrucción | `knowledge/validation/dimensionality_reduction_selection.md` | Alta |
| Incremental PCA y restricciones de memoria | `knowledge/preprocessing/incremental_pca.md` | Media |
| Kernel PCA y ajuste mediante pipeline/grid search | `knowledge/tuning/kernel_pca.md` | Media |
| LLE y Manifold Learning | `knowledge/feature_engineering/manifold_learning.md` | Media |
| Reducción para visualización | `knowledge/eda/dimensionality_reduction_visualization.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------ | ------------------------------ |
| Propósito y maldición de la dimensionalidad | “Dimensionality Reduction”, pp. 213–214; “The Curse of Dimensionality”, pp. 214–215 |
| Proyección y Manifold Learning | “Main Approaches for Dimensionality Reduction”, pp. 215–218 |
| PCA y selección de componentes | “PCA”, pp. 219–224 |
| Compresión, reconstrucción y PCA aleatorizado | pp. 224–225 |
| Incremental PCA | pp. 225–226 |
| Kernel PCA y selección de hiperparámetros | pp. 226–229 |
| LLE | pp. 230–232 |
| Técnicas adicionales | pp. 232–233 |
| Preguntas de aplicación y comparaciones | Exercises, pp. 233–234 |

Nota de lectura: el fragmento revisado contiene las páginas impresas 213–234, incluyendo ejercicios sin solución. No se necesitó consultar el resto del libro ni fuentes externas. Todas las afirmaciones específicas anteriores se limitan al contenido del fragmento; las indicaciones marcadas como conclusión práctica o síntesis derivada son inferencias operativas basadas en él.
