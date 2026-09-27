---
title: "Fundamentos de NumPy: arrays y computación vectorizada"
topics:
  - "ndarray, forma y tipos de datos"
  - "Indexado, vectorización y funciones universales"
  - "Agregación, ordenación y operaciones de conjuntos"
  - "Generación pseudoaleatoria, persistencia y álgebra lineal"
target_knowledge_files:
  - "knowledge/preprocessing/numpy_arrays_and_dtypes.md"
  - "knowledge/feature_engineering/vectorized_transformations.md"
  - "knowledge/eda/numpy_descriptive_operations.md"
  - "knowledge/models/numerical_linear_algebra.md"
---

# 4. Fundamentos de NumPy: arrays y computación vectorizada

## 1. Propósito del capítulo

El capítulo introduce `ndarray` como contenedor homogéneo y multidimensional para cálculo numérico eficiente. Explica cómo crear, tipar, seleccionar, transformar, resumir, ordenar, guardar y combinar arrays sin depender de bucles de Python. Este conocimiento sustenta el uso eficaz de pandas y muchas operaciones de preparación de datos para Machine Learning.

**Contenido explícito del autor:** NumPy obtiene velocidad y menor consumo de memoria al almacenar datos en bloques contiguos y ejecutar algoritmos compilados sobre arrays completos. El capítulo sitúa el pensamiento orientado a arrays como una habilidad básica para el análisis de datos, aunque NumPy no sea por sí solo una librería de modelado.

## 2. Conceptos esenciales

- **`ndarray`:** array n-dimensional de elementos homogéneos. `shape` expresa el tamaño de cada eje, `ndim` el número de ejes y `dtype` la representación de cada elemento.
- **Vectorización:** expresión de cálculos sobre arrays completos. Sustituye bucles explícitos de Python por operaciones de array, que suelen ser más rápidas y compactas.
- **Broadcasting (difusión):** mecanismo que permite operar con arrays de formas distintas cuando sus dimensiones son compatibles. El capítulo lo presenta, pero remite el tratamiento detallado al apéndice A.
- **Vista frente a copia:** los cortes básicos suelen devolver vistas del array original; modificar una vista puede cambiar los datos de origen. El indexado booleano y el indexado sofisticado (`fancy indexing`) producen copias.
- **Indexado básico:** selecciona elementos, filas, columnas o bloques mediante enteros y cortes. Al mezclar enteros con cortes puede reducirse el número de dimensiones.
- **Indexado booleano:** usa una máscara de igual longitud que el eje seleccionado para filtrar o asignar valores.
- **Indexado sofisticado:** usa arrays de enteros para seleccionar posiciones. Varios arrays de índices seleccionan pares de coordenadas, no necesariamente una región rectangular.
- **Función universal (`ufunc`):** función vectorizada rápida que opera elemento a elemento. Puede ser unaria (`sqrt`, `exp`) o binaria (`add`, `maximum`). Algunas devuelven varios arrays, como `modf`.
- **Reducción y acumulación:** `sum`, `mean`, `min`, `max`, `var` y `std` reducen datos; `cumsum` y `cumprod` conservan resultados parciales. El parámetro `axis` determina el eje de operación.
- **Generador pseudoaleatorio:** `np.random.default_rng(seed)` crea un generador reproducible y aislado de otros generadores. Sus resultados son deterministas para la misma semilla, no verdaderamente aleatorios.
- **Transposición y ejes:** `.T`, `transpose` y `swapaxes` reordenan ejes y suelen devolver vistas. Para matrices, `@` realiza producto matricial.
- **Persistencia binaria:** `np.save`/`np.load` trabajan con `.npy`; `np.savez` y `np.savez_compressed` agrupan varios arrays en `.npz`.

**Conclusión práctica derivada:** antes de una transformación, comprobar `shape`, `dtype` y si la selección devuelve vista o copia evita gran parte de los errores silenciosos de NumPy.

## 3. Flujo o procedimiento presentado

1. Convertir datos de entrada con `np.array` o `np.asarray`, o inicializar memoria con funciones como `zeros`, `ones`, `full`, `empty`, `arange` o `eye`.
2. Inspeccionar `ndim`, `shape` y `dtype` para verificar dimensionalidad y representación.
3. Convertir tipos con `astype` cuando la operación lo requiera, teniendo en cuenta pérdida de precisión, truncamiento o errores de conversión.
4. Seleccionar datos mediante cortes, máscaras booleanas o índices enteros según la geometría deseada.
5. Expresar cálculos con operadores, `ufuncs`, `np.where`, agregaciones y operaciones por eje.
6. Reorganizar dimensiones con `reshape`, transposición o intercambio de ejes cuando una operación matricial lo exija.
7. Ordenar, deduplicar o aplicar lógica de conjuntos con `sort`, `unique`, `in1d` y operaciones de conjuntos.
8. Guardar resultados intermedios en `.npy` o `.npz` si deben reutilizarse.
9. Para simulaciones, crear un generador local con semilla y generar todos los ensayos en una sola matriz cuando sea viable.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| La entrada ya puede ser un `ndarray` y no se necesita copiar | `np.asarray` | Evita una copia innecesaria | El resultado puede compartir memoria con la entrada |
| Se necesita memoria que será rellenada inmediatamente | `np.empty` | Evita inicialización redundante | Contiene valores no inicializados; no asumir ceros |
| Se necesita modificar el array original mediante una selección contigua | Corte básico | Normalmente devuelve una vista | La asignación también altera el origen |
| Se necesita una selección independiente | Máscara booleana o indexado sofisticado | Devuelven una copia | La máscara debe coincidir con el eje; los cambios no vuelven al origen |
| Hay una condición elemento a elemento | `np.where` | Vectoriza la elección entre dos valores o arrays | Las entradas deben ser compatibles por difusión |
| Se requieren resultados reproducibles | `np.random.default_rng(seed)` | Generador local, reproducible y aislado | Una semilla fija repite la secuencia; no implica aleatoriedad real |
| Se necesita producto matricial | Operador `@` | Expresa directamente la multiplicación de matrices | Verificar dimensiones internas compatibles |
| Se guardará un solo array | `np.save` / `np.load` | Formato binario simple `.npy` | No es un formato tabular interoperable |
| Se guardarán varios arrays | `np.savez` o `np.savez_compressed` | Archivo `.npz` con arrays nombrados | La compresión añade coste de CPU |
| Se necesita ordenar sin alterar la entrada | `np.sort(array)` | Devuelve una copia ordenada | `array.sort()` modifica el array en el sitio |

## 5. Métodos y técnicas

### Creación y tipado de arrays

- **Objetivo:** representar datos homogéneos en una estructura compacta.
- **Entradas:** secuencias, otros arrays, una forma o valores de relleno.
- **Salida:** `ndarray` con `shape` y `dtype` definidos o inferidos.
- **Funcionamiento:** `array` convierte y copia por omisión; `asarray` evita copiar si ya recibe un array compatible. `zeros`, `ones`, `full` y variantes `_like` construyen arrays predecibles; `empty` solo reserva memoria.
- **Parámetros relevantes:** `dtype`, `shape`; en `arange`, inicio, fin y paso.
- **Ventajas:** control de memoria, precisión y compatibilidad con operaciones vectorizadas.
- **Limitaciones:** todos los elementos comparten tipo; cadenas de longitud fija pueden truncarse; `astype` de float a entero elimina la parte decimal.
- **Casos adecuados:** matrices numéricas, tensores de características, simulaciones y transformaciones en lote.
- **No conviene:** datos heterogéneos o tabulares con semántica de columnas, para los que el capítulo recomienda considerar pandas.

### Selección mediante indexado

- **Objetivo:** extraer o modificar subconjuntos.
- **Entradas:** enteros, cortes, máscaras booleanas o arrays de índices.
- **Salida:** escalar, vista o copia, según el método.
- **Funcionamiento:** `array[i, j]` accede a coordenadas; los cortes delimitan bloques; las máscaras conservan posiciones verdaderas; los índices enteros permiten reordenar y repetir.
- **Parámetros relevantes:** índices y `axis` implícito en la forma de la expresión.
- **Ventajas:** selección compacta y asignación masiva.
- **Limitaciones:** máscaras desalineadas provocan error; mezclar arrays de índices puede seleccionar coordenadas por pares en vez de un bloque.
- **Casos adecuados:** filtrado, recodificación, clipping por condición y extracción de filas/columnas.
- **No conviene:** usar cortes sin comprender que comparten memoria cuando se necesita aislamiento.

### Funciones universales y lógica vectorizada

- **Objetivo:** aplicar funciones y condiciones elemento a elemento.
- **Entradas:** uno o más arrays compatibles.
- **Salida:** array(s) con la forma resultante de la difusión.
- **Funcionamiento:** `ufuncs` ejecutan operaciones compiladas; `np.where(cond, x, y)` elige entre `x` e `y` por posición.
- **Parámetros relevantes:** arrays de entrada y condición; la forma compatible sustituye a un hiperparámetro.
- **Ventajas:** velocidad, concisión y eliminación de bucles.
- **Limitaciones:** expresiones complejas pueden crear arrays temporales grandes; las formas incompatibles fallan.
- **Casos adecuados:** transformaciones, funciones matemáticas, reglas de sustitución y creación de variables.
- **No conviene:** lógica con estado entre elementos o efectos laterales difíciles de expresar como operación de array.

### Agregaciones, ordenación y conjuntos

- **Objetivo:** resumir, acumular, ordenar y comparar colecciones de valores.
- **Entradas:** array y, cuando corresponde, `axis`.
- **Salida:** escalar o array reducido/acumulado/ordenado.
- **Funcionamiento:** las reducciones colapsan un eje; acumulaciones mantienen su longitud; `any`/`all` resumen booleanos; `unique` devuelve valores únicos ordenados; las funciones de conjuntos operan con valores únicos.
- **Parámetros relevantes:** `axis`; en percentiles, el cuantil solicitado.
- **Ventajas:** primitives útiles para EDA y validaciones rápidas.
- **Limitaciones:** un `axis` equivocado cambia el significado; ordenar en el sitio modifica datos; las operaciones de conjuntos pierden multiplicidades.
- **Casos adecuados:** estadísticas descriptivas, detección de categorías, ranking y controles de calidad.
- **No conviene:** sustituir métricas específicas del problema por resúmenes genéricos.

### Generación pseudoaleatoria y simulación vectorizada

- **Objetivo:** muestrear distribuciones y simular muchos procesos a la vez.
- **Entradas:** generador, distribución, parámetros, `size` y semilla opcional.
- **Salida:** array de muestras.
- **Funcionamiento:** un `Generator` produce muestras; el ejemplo de caminos aleatorios genera pasos `-1/+1`, aplica `cumsum` por fila y obtiene máximos, mínimos y tiempos de cruce.
- **Parámetros relevantes:** `seed`, `size` y parámetros de distribución.
- **Ventajas:** reproducibilidad y gran mejora frente a generar escalares dentro de bucles.
- **Limitaciones:** memoria proporcional al número de simulaciones por longitud; `argmax` devuelve `0` cuando todo es falso, por lo que primero debe verificarse que el evento ocurrió.
- **Casos adecuados:** remuestreo, simulación, pruebas y generación de datos sintéticos.
- **No conviene:** tratar secuencias pseudoaleatorias como entropía criptográfica.

### Álgebra lineal

- **Objetivo:** productos matriciales, descomposiciones y resolución de sistemas.
- **Entradas:** vectores o matrices con dimensiones compatibles.
- **Salida:** escalar, vector, matriz, descomposición o solución.
- **Funcionamiento:** `@`/`np.dot` calculan productos; `numpy.linalg` incluye inversa, determinante, QR, SVD, autovalores, solución de sistemas y mínimos cuadrados.
- **Parámetros relevantes:** dimensiones y propiedades de la matriz.
- **Ventajas:** operaciones compactas respaldadas por implementaciones numéricas eficientes.
- **Limitaciones:** el capítulo enumera funciones, pero no desarrolla estabilidad numérica ni criterios de selección.
- **Casos adecuados:** transformaciones matriciales y algoritmos numéricos que ya se expresan en álgebra lineal.
- **No conviene:** usar una inversa explícita como regla universal; el capítulo no la recomienda para todo problema.

## 6. Fórmulas esenciales

### Media aritmética

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

- `x_i`: observación `i`; `n`: número de observaciones; `\bar{x}`: media.
- Mide el nivel promedio. En un array multidimensional, `axis` define qué eje se reduce.
- Es sensible a la elección del eje y a valores extremos.

### Producto matricial

\[
C_{ij} = \sum_{k=1}^{p} A_{ik}B_{kj}
\]

- `A` tiene forma `(m, p)`, `B` forma `(p, n)` y `C` forma `(m, n)`.
- Combina filas de `A` con columnas de `B`; en NumPy se expresa como `A @ B`.
- Supuesto estructural: las dimensiones internas deben coincidir.

### Camino aleatorio acumulado

\[
W_t = \sum_{i=1}^{t} S_i, \qquad S_i \in \{-1, 1\}
\]

- `S_i`: paso en el instante `i`; `W_t`: posición acumulada hasta `t`.
- Se implementa con `steps.cumsum(axis=1)` al simular varios caminos por filas.
- En el ejemplo, los pasos se generan con igual probabilidad; otras distribuciones cambian el proceso.

### Superficie del ejemplo vectorizado

\[
z = \sqrt{x^2 + y^2}
\]

- `x`, `y`: coordenadas producidas por `meshgrid`; `z`: distancia euclídea al origen.
- Ilustra la evaluación de una función sobre toda una cuadrícula sin bucles.

## 7. Implementación práctica

Los fragmentos siguientes condensan operaciones que el capítulo implementa. No incluyen entrenamiento de modelos; por tanto, la separación entrenamiento/validación se muestra únicamente donde una transformación aprendida podría causar *data leakage*.

```python
import numpy as np

features = np.asarray([[1.5, -0.1, 3.0], [0.0, -3.0, 6.5]], dtype=np.float64)
assert features.ndim == 2
assert features.shape == (2, 3)

positive_features = np.where(features > 0, features, 0.0)
column_means = features.mean(axis=0)
```

```python
import numpy as np

rng = np.random.default_rng(seed=12345)
steps = np.where(rng.integers(0, 2, size=(5_000, 1_000)) == 0, 1, -1)
walks = steps.cumsum(axis=1)

reached_30 = (np.abs(walks) >= 30).any(axis=1)
first_crossing = (np.abs(walks[reached_30]) >= 30).argmax(axis=1)
mean_crossing_time = first_crossing.mean()
```

**Nota práctica derivada:** comprobar `any(axis=1)` antes de `argmax(axis=1)` evita interpretar el índice `0` como un cruce cuando la condición nunca fue verdadera, una advertencia explícita del ejemplo.

```python
import numpy as np

# Separación previa; una estadística de ajuste se calcula solo con entrenamiento.
train = np.asarray([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
validation = np.asarray([[7.0, 20.0]])

train_mean = train.mean(axis=0)
train_std = train.std(axis=0)
train_scaled = (train - train_mean) / train_std
validation_scaled = (validation - train_mean) / train_std
```

**Síntesis práctica derivada del capítulo:** el cálculo por ejes permite aplicar las mismas estadísticas a ambos conjuntos. La regla de ajustar solo con entrenamiento es una precaución para ML derivada de estas operaciones; no se formula como protocolo de validación en el capítulo.

## 8. Supuestos y condiciones de aplicación

- Los elementos de un `ndarray` comparten `dtype`.
- Las secuencias anidadas regulares permiten inferir una forma multidimensional coherente.
- Las operaciones elemento a elemento requieren formas iguales o compatibles mediante difusión.
- Una máscara booleana debe tener longitud compatible con el eje que indexa.
- El producto matricial requiere dimensiones internas compatibles.
- Los cortes básicos pueden compartir memoria con el origen; el indexado booleano y el sofisticado crean copias.
- `np.empty` solo es seguro si cada posición será escrita antes de leerse.
- Las conversiones de tipo deben admitir todos los valores; convertir flotantes a enteros trunca la parte decimal.
- Las estadísticas por eje dependen de que filas y columnas tengan una semántica estable.
- La reproducibilidad pseudoaleatoria requiere conservar la semilla y la configuración del generador.
- La simulación vectorizada requiere memoria suficiente para materializar los arrays completos.

## 9. Errores y riesgos

- Leer o usar los valores de `np.empty` antes de inicializarlos.
- Modificar accidentalmente el array original a través de una vista obtenida por corte.
- Suponer que el indexado booleano o sofisticado devuelve una vista.
- Usar una máscara con longitud incorrecta o combinar condiciones sin paréntesis.
- Confundir selección por pares de coordenadas con selección de una región rectangular en indexado sofisticado.
- Convertir floats a enteros sin advertir el truncamiento o elegir un `dtype` demasiado pequeño.
- Usar cadenas NumPy de longitud fija cuando el truncamiento silencioso sea inaceptable.
- Aplicar una reducción sobre el eje equivocado y obtener un resultado plausible pero semánticamente incorrecto.
- Usar `array.sort()` cuando se necesitaba conservar el orden original.
- Interpretar `argmax` de un array booleano como evento encontrado sin comprobar antes `any`.
- Crear arrays temporales o simulaciones demasiado grandes y agotar memoria.
- **Riesgo de leakage, síntesis derivada:** calcular medias, desviaciones u otras estadísticas sobre todo el dataset antes de separar validación incorpora información de validación.
- **Riesgo de sobreajuste, síntesis derivada:** el capítulo no trata entrenamiento de modelos; una semilla reproducible facilita comparar experimentos, pero no sustituye un esquema de validación.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo cuando el problema exija manipular matrices, aplicar transformaciones numéricas, filtrar observaciones, resumir por ejes o acelerar bucles.
- Para un baseline, usar `asarray`, validar `shape`/`dtype`, limpiar mediante máscaras y `np.where`, y obtener resúmenes con agregaciones por eje.
- Para mejorar rendimiento, reemplazar bucles independientes por operadores vectorizados, `ufuncs`, indexado booleano y generación aleatoria por lotes.
- Señales activadoras: matrices densas numéricas, bucles sobre filas, condiciones elemento a elemento, simulaciones repetidas, necesidad de reproducibilidad o productos matriciales.
- Bajo presión de tiempo, evitar optimizaciones de memoria demasiado finas, combinaciones complejas de indexado sofisticado y álgebra lineal avanzada si no son necesarias para la solución.
- Comprobar siempre si una operación cambia el array original, qué eje reduce y qué forma produce.
- Para un pipeline de ML, separar primero entrenamiento y validación; cualquier estadística usada para transformar datos debe estimarse solo con entrenamiento. Esta última regla es una recomendación de competencia derivada, no una instrucción explícita del capítulo.

## 11. Checklist operativo

- [ ] Convertir la entrada a un `ndarray` apropiado.
- [ ] Verificar `ndim`, `shape` y `dtype`.
- [ ] Confirmar si la selección devuelve vista o copia.
- [ ] Validar que máscaras y formas sean compatibles.
- [ ] Especificar y revisar `axis` en cada agregación.
- [ ] Evitar leer arrays creados con `empty` antes de rellenarlos.
- [ ] Revisar pérdida de precisión o truncamiento al usar `astype`.
- [ ] Usar `np.random.default_rng(seed)` cuando se requiera reproducibilidad.
- [ ] Verificar `any` antes de aplicar `argmax` a una condición booleana.
- [ ] Medir memoria antes de materializar simulaciones grandes.
- [ ] En ML, ajustar estadísticas solo con entrenamiento.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| `ndarray`, formas, ejes y `dtype` | `knowledge/preprocessing/numpy_arrays_and_dtypes.md` | Alta |
| Vistas, copias, cortes y máscaras | `knowledge/preprocessing/numpy_indexing_and_memory.md` | Alta |
| Transformaciones con `ufuncs` y `np.where` | `knowledge/feature_engineering/vectorized_transformations.md` | Alta |
| Agregaciones y estadística descriptiva por eje | `knowledge/eda/numpy_descriptive_operations.md` | Alta |
| Separación de estadísticas entre entrenamiento y validación | `knowledge/validation/preprocessing_without_leakage.md` | Alta |
| Generación pseudoaleatoria reproducible | `knowledge/validation/reproducible_randomness.md` | Media |
| Ordenación, únicos y operaciones de conjuntos | `knowledge/eda/numpy_sorting_and_sets.md` | Media |
| Producto matricial y `numpy.linalg` | `knowledge/models/numerical_linear_algebra.md` | Media |
| Persistencia `.npy` y `.npz` | `knowledge/preprocessing/numpy_persistence.md` | Baja |
| Simulación vectorizada de caminos aleatorios | `knowledge/models/vectorized_simulation.md` | Baja |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y eficiencia de NumPy | Introducción, pp. 117–120 |
| `ndarray`, creación, forma y tipos | §4.1, pp. 120–128; tablas 4.1 y 4.2 |
| Aritmética, difusión e indexado básico | §4.1, pp. 128–136; figura 4.1 |
| Cortes, indexado booleano y sofisticado | §4.1, pp. 134–143; figura 4.2 |
| Transposición e intercambio de ejes | §4.1, pp. 143–145 |
| Generación pseudoaleatoria | §4.2, pp. 145–147; tabla 4.3 |
| Funciones universales | §4.3, pp. 147–151; tablas 4.4 y 4.5 |
| Programación orientada a arrays y `meshgrid` | §4.4, pp. 151–154; figura 4.3 |
| Lógica condicional con `np.where` | §4.4, pp. 153–155 |
| Métodos matemáticos, estadísticos y booleanos | §4.4, pp. 155–159; tabla 4.6 |
| Ordenación, únicos y lógica de conjuntos | §4.4, pp. 159–163; tabla 4.7 |
| Entrada y salida de archivos | §4.5, pp. 163–164 |
| Álgebra lineal | §4.6, pp. 164–166; tabla 4.8 |
| Caminos aleatorios y simulación vectorizada | §4.7, pp. 166–170; figura 4.4 |
| Conclusión | §4.8, p. 170 |

