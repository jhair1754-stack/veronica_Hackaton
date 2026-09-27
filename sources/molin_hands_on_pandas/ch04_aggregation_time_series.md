---
title: "Aggregating Pandas DataFrames"
topics:
  - "DataFrame joins and set-based validation"
  - "Vectorized transformations, binning, and window calculations"
  - "Aggregation, groupby, pivot tables, and crosstabs"
  - "Time-series selection, resampling, lags, and temporal joins"
target_knowledge_files:
  - "knowledge/preprocessing/dataframe_joins_and_aggregation.md"
  - "knowledge/feature_engineering/tabular_and_window_features.md"
  - "knowledge/time_series/pandas_time_series_operations.md"
  - "knowledge/eda/grouped_summaries_and_crosstabs.md"
---

# 4. Aggregating Pandas DataFrames

## 1. Propósito del capítulo

El capítulo enseña a enriquecer, transformar y resumir `DataFrame` de pandas: consultar filas, unir tablas, crear variables, discretizar, aplicar cálculos vectorizados, usar ventanas, agregar por grupos y trabajar con índices temporales. Para Machine Learning, estas operaciones sirven para construir tablas analíticas, producir features históricos y reducir datos a la granularidad correcta antes del modelado.

## 2. Conceptos esenciales

- **Consulta (`query`)**: expresa filtros complejos con una sintaxis próxima a una cláusula SQL `WHERE`. Las máscaras booleanas producen el mismo tipo de resultado; la elección suele depender de legibilidad.
- **Unión (`merge`/`join`)**: alinea filas mediante una o varias claves. `inner` conserva coincidencias; `left` y `right` preservan la tabla indicada; `outer` conserva todas las claves y puede introducir valores ausentes.
- **Validación por conjuntos**: `intersection()`, `difference()`, `symmetric_difference()` y `union()` permiten anticipar qué claves se conservarán o perderán antes de una unión costosa.
- **Operaciones vectorizadas**: pandas y NumPy operan columnas o filas completas en código optimizado. Deben preferirse a `iterrows()`, `itertuples()` o iteraciones elemento a elemento cuando exista una alternativa vectorizada.
- **Discretización**: `pd.cut()` crea intervalos por valor, mientras `pd.qcut()` crea grupos por cuantiles. Ambos reducen granularidad y, por tanto, información.
- **Umbrales (`clip`)**: reemplaza valores por límites inferior o superior; puede conservar el tipo numérico y producir indicadores compactos.
- **Aplicación de funciones**: `apply()` ejecuta una función sobre columnas o filas; una función válida para columnas completas debe mantenerse vectorizada.
- **Ventanas**: `rolling()` usa una ventana móvil, `expanding()` acumula toda la historia disponible y `ewm()` asigna mayor peso a observaciones recientes.
- **Agregación**: `agg()` resume columnas con una o varias funciones; `groupby()` aplica resúmenes por pertenencia a grupos; `transform()` devuelve el mismo número de filas que la entrada y permite añadir estadísticas del grupo a cada observación.
- **Tablas de resumen**: `pivot_table()` admite claves repetidas porque agrega; `crosstab()` construye tablas de frecuencia o agregaciones cruzadas.
- **Índice temporal (`DatetimeIndex`)**: habilita slicing parcial, selección por hora, agrupación temporal, rezagos, diferencias, remuestreo y uniones por proximidad.
- **Granularidad temporal**: `resample()` reduce frecuencia mediante agregación o la aumenta, en cuyo caso aparecen huecos que deben tratarse explícitamente.

## 3. Flujo o procedimiento presentado

1. Cargar los datos con fechas interpretadas y, para series temporales, colocar la fecha u hora en un `DatetimeIndex`.
2. Consultar las observaciones relevantes con `query()`, máscaras o slicing temporal.
3. Antes de unir, identificar claves, cardinalidad y duplicados; comparar conteos y conjuntos de claves.
4. Elegir `inner`, `left`, `right` u `outer` según las filas que deban conservarse; ejecutar la unión y auditar forma, nulos y procedencia.
5. Crear variables mediante operaciones vectorizadas: cambios porcentuales, rankings, Z-scores, límites o discretizaciones.
6. Para features históricas, ordenar por tiempo y elegir entre ventana móvil, acumulativa o exponencial.
7. Resumir con `agg()`; si hay grupos, usar `groupby()` y definir una agregación coherente para cada variable.
8. Usar `transform()` cuando el resultado agregado deba conservar la longitud original.
9. Reorganizar resultados mediante `unstack()`, `pivot_table()` o `crosstab()` cuando la forma tabular facilite el análisis.
10. En series temporales, seleccionar el rango y la frecuencia correctos; aplicar `shift()`, `diff()` o `resample()` según el objetivo.
11. Para relojes no perfectamente alineados, usar `merge_asof()` con dirección y tolerancia explícitas, o `merge_ordered()` si se requiere conservar y ordenar todas las marcas.
12. Verificar que las transformaciones no hayan eliminado claves, multiplicado filas, creado nulos inesperados o usado información futura.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Solo interesan claves presentes en ambas tablas | `merge(..., how='inner')` | Conserva coincidencias | Elimina claves exclusivas de cualquiera de los lados |
| Deben preservarse todas las filas de una tabla principal | `left` o `right` join | Mantiene el lado elegido | Las faltas del otro lado aparecen como `NaN` |
| Debe conservarse toda la información de ambas tablas | `outer` con `indicator=True` | Hace visible la procedencia de cada fila | Puede aumentar filas y nulos |
| Se quiere anticipar el efecto de un join | Operaciones de conjuntos sobre claves | Revelan solapamientos y pérdidas antes de unir | Usar claves únicas cuando corresponda |
| Se necesitan rangos con ancho comparable | `pd.cut()` | Divide el espacio de valores | Outliers pueden dejar grupos muy desbalanceados |
| Se necesitan grupos con conteos comparables | `pd.qcut()` | Divide por cuantiles | Los anchos varían y los empates pueden dificultar los cortes |
| Solo importa si un valor supera límites | `clip()` | Impone umbrales sin cambiar necesariamente el tipo | Se pierde magnitud fuera de los límites |
| El cálculo depende de las últimas observaciones | `rolling()` | Mantiene una ventana móvil | Definir correctamente tamaño, unidad y orden temporal |
| El cálculo debe usar toda la historia hasta cada fila | `expanding()` | Produce estadísticas acumulativas | Los resultados iniciales usan menos observaciones; revisar `min_periods` |
| Las observaciones recientes deben pesar más | `ewm()` | Implementa ponderación exponencial | El parámetro de decaimiento cambia la sensibilidad |
| La estadística del grupo debe volver a cada fila | `groupby(...).transform()` | Conserva dimensiones | No confundir con `agg()`, que reduce filas |
| Hay claves temporales cercanas, no idénticas | `merge_asof()` | Empareja por proximidad | Ordenar por clave y fijar `direction` y `tolerance` |
| Se quieren todas las marcas temporales ordenadas | `merge_ordered()` | Conserva y ordena claves según el tipo de join | Los tiempos no coincidentes producen nulos |
| Se reduce granularidad temporal | `resample(...).agg(...)` | Resume periodos de forma declarativa | Cada columna requiere una agregación semánticamente válida |
| Se aumenta granularidad temporal | `resample(...).asfreq()` | Expone la nueva cuadrícula temporal | Introduce `NaN`; no rellenar sin justificación |

## 5. Métodos y técnicas

### `DataFrame.query()`

- **Objetivo**: filtrar filas con expresiones legibles.
- **Entradas**: expresión sobre nombres de columnas y operadores booleanos.
- **Salida**: subconjunto del `DataFrame`.
- **Funcionamiento**: evalúa la condición de forma equivalente a una máscara booleana.
- **Hiperparámetros relevantes**: no aplica; la expresión determina el filtro.
- **Ventajas**: evita repetir nombres largos del `DataFrame`; facilita filtros compuestos.
- **Limitaciones**: nombres o expresiones especiales pueden ser menos directos que una máscara.
- **Casos adecuados**: filtros exploratorios y reglas con varias condiciones.
- **No conviene**: cuando una máscara explícita sea más clara o reutilizable.

### `merge()` y `join()`

- **Objetivo**: enriquecer datos alineando filas por claves.
- **Entradas**: dos tablas, claves (`on`, `left_on`, `right_on` o índices), tipo de join y sufijos.
- **Salida**: `DataFrame` combinado.
- **Funcionamiento**: aplica semántica de join relacional; `indicator=True` identifica `left_only`, `right_only` o `both`.
- **Hiperparámetros relevantes**: `how`, claves, `left_index`, `right_index`, `suffixes`, `indicator`.
- **Ventajas**: incorpora variables de otras fuentes y admite claves múltiples.
- **Limitaciones**: puede ser intensivo en recursos; claves repetidas pueden multiplicar filas; claves no coincidentes generan pérdidas o nulos.
- **Casos adecuados**: anexar metadatos, reparar información separada y construir la tabla de modelado.
- **No conviene**: sin conocer cardinalidad, unicidad o conservación requerida.

### `cut()`, `qcut()` y `clip()`

- **Objetivo**: discretizar variables o imponer umbrales.
- **Entradas**: serie numérica; número/límites de bins o cuantiles; etiquetas; límites de recorte.
- **Salida**: categorías/intervalos para `cut()` y `qcut()`; valores acotados para `clip()`.
- **Funcionamiento**: `cut()` segmenta el rango; `qcut()` segmenta la distribución; `clip()` sustituye extremos por límites.
- **Hiperparámetros relevantes**: `bins`, `q`, `labels`, `retbins`, límites inferior y superior.
- **Ventajas**: simplifica análisis, crea categorías ordinales y puede reducir influencia de extremos.
- **Limitaciones**: pierde granularidad; bins de igual ancho pueden quedar desbalanceados; cuantiles tienen amplitud desigual.
- **Casos adecuados**: bandas interpretables, análisis de frecuencia e indicadores numéricos por umbral.
- **No conviene**: cuando la magnitud exacta contiene señal importante.

### Operaciones vectorizadas y `apply()`

- **Objetivo**: transformar columnas o filas eficientemente.
- **Entradas**: columnas/filas y una función válida para cada eje.
- **Salida**: serie o tabla transformada, sin modificar en sitio en el ejemplo del capítulo.
- **Funcionamiento**: combina operaciones como `sub`, `div`, `abs`, `rank` y `pct_change`; `apply()` generaliza funciones por eje.
- **Hiperparámetros relevantes**: `axis` y argumentos de la función aplicada.
- **Ventajas**: rendimiento y claridad superiores a bucles Python.
- **Limitaciones**: una función no vectorizada puede seguir siendo lenta; debe ser válida para todas las columnas seleccionadas.
- **Casos adecuados**: estandarización, rankings, tasas de cambio y transformaciones homogéneas.
- **No conviene**: iteración fila a fila si existe una expresión vectorizada.

### `rolling()`, `expanding()` y `ewm()`

- **Objetivo**: generar estadísticas dependientes del orden.
- **Entradas**: serie/tabla ordenada, definición de ventana y función de agregación.
- **Salida**: objeto con la misma alineación temporal y estadísticas de ventana.
- **Funcionamiento**: `rolling()` desliza un intervalo; `expanding()` amplía desde el inicio; `ewm()` reduce exponencialmente el peso del pasado.
- **Hiperparámetros relevantes**: tamaño/unidad de `window`, `min_periods`, `span` u otro parámetro de decaimiento; agregación por columna.
- **Ventajas**: produce features de tendencia, acumulación, extremos y suavizado.
- **Limitaciones**: sensible a orden, frecuencia y valores faltantes; los primeros resultados tienen historia incompleta.
- **Casos adecuados**: series temporales, métricas móviles y features retrospectivos.
- **No conviene**: sobre datos desordenados o si la ventana incorpora observaciones posteriores al instante predicho.

### `agg()`, `groupby()` y `transform()`

- **Objetivo**: resumir toda la tabla o subconjuntos definidos por grupos.
- **Entradas**: claves de agrupación y funciones por columna.
- **Salida**: `agg()` normalmente reduce filas; múltiples funciones pueden crear columnas jerárquicas; `transform()` conserva la forma.
- **Funcionamiento**: divide los datos por claves y combina estadísticas predefinidas o personalizadas.
- **Hiperparámetros relevantes**: columnas de grupo, funciones de agregación y `pd.Grouper(freq=...)` para grupos temporales.
- **Ventajas**: resume grandes tablas y crea variables contextuales por grupo.
- **Limitaciones**: agregaciones incorrectas pueden destruir semántica; múltiples métricas producen `MultiIndex` que quizá deba aplanarse.
- **Casos adecuados**: estadísticas por entidad, fecha o categoría; normalización relativa al grupo.
- **No conviene**: usar `agg()` si el resultado debe mantener una fila por observación.

### `pivot_table()` y `crosstab()`

- **Objetivo**: presentar agregaciones en formato matricial.
- **Entradas**: filas, columnas, valores y función de agregación.
- **Salida**: tabla cruzada, opcionalmente con márgenes o normalización.
- **Funcionamiento**: `pivot_table()` agrega combinaciones repetidas; `crosstab()` cuenta por defecto o aplica `aggfunc` sobre `values`.
- **Hiperparámetros relevantes**: `index`, `columns`, `values`, `aggfunc`, `normalize`, `margins` y `margins_name`.
- **Ventajas**: comparación rápida entre categorías y conversión de formato largo a ancho.
- **Limitaciones**: una matriz ancha puede crecer mucho; los huecos representan combinaciones ausentes.
- **Casos adecuados**: EDA, tablas de contingencia y resúmenes por entidad/variable.
- **No conviene**: como matriz de features sin controlar cardinalidad y valores ausentes.

### Operaciones temporales

- **Objetivo**: seleccionar, desplazar, diferenciar y cambiar frecuencia.
- **Entradas**: `DatetimeIndex`, intervalos/horas, número de periodos o frecuencia.
- **Salida**: subconjuntos temporales, rezagos, diferencias o datos remuestreados.
- **Funcionamiento**: slicing parcial y `first()`/`last()` seleccionan rangos; `at_time()` y `between_time()` usan la hora; `shift()` crea lags; `diff()` calcula diferencias; `resample()` cambia frecuencia; `asof()` devuelve la última observación válida hasta una fecha.
- **Hiperparámetros relevantes**: periodo/frecuencia, rango horario y agregación.
- **Ventajas**: API concisa para features y resúmenes temporales.
- **Limitaciones**: calendarios irregulares, cierres y huecos afectan la interpretación.
- **Casos adecuados**: forecasting, señales financieras, sensores y eventos.
- **No conviene**: sin índice temporal correctamente interpretado y ordenado.

### `merge_asof()` y `merge_ordered()`

- **Objetivo**: unir series con marcas temporales no perfectamente coincidentes.
- **Entradas**: tablas ordenadas, claves temporales, dirección, tolerancia y tipo de join.
- **Salida**: observaciones alineadas por cercanía o por orden temporal.
- **Funcionamiento**: `merge_asof()` busca la clave previa, posterior o más cercana permitida; `merge_ordered()` conserva coincidencias exactas y ordena claves no coincidentes.
- **Hiperparámetros relevantes**: `direction`, `tolerance`, índices/claves, `how`, `fill_method`.
- **Ventajas**: alinea fuentes con relojes o frecuencias diferentes.
- **Limitaciones**: una tolerancia amplia puede emparejar eventos que no deberían relacionarse; una unión ordenada genera nulos.
- **Casos adecuados**: sensores, cotizaciones y logs asíncronos.
- **No conviene**: sin una regla de proximidad defendible.

## 6. Fórmulas esenciales

### Z-score

\[
z_i = \frac{x_i - \bar{x}}{s}
\]

- \(x_i\): observación.
- \(\bar{x}\): media de la muestra o periodo analizado.
- \(s\): desviación estándar del mismo conjunto.
- **Qué mide**: distancia estandarizada de una observación respecto de la media.
- **Interpretación**: el capítulo usa \(|z| > 3\) como señal de observaciones potencialmente anómalas, no como prueba universal.
- **Supuesto importante**: media y desviación deben calcularse sobre un grupo comparable; el ejemplo meteorológico limita el cálculo a un mes por estacionalidad.

### Cambio porcentual

\[
r_t = \frac{x_t - x_{t-1}}{x_{t-1}}
\]

- \(x_t\): valor actual.
- \(x_{t-1}\): valor del periodo anterior.
- **Qué mide**: variación relativa entre periodos, implementada con `pct_change()`.
- **Interpretación**: valores positivos indican aumento y negativos, disminución.
- **Supuesto importante**: los periodos deben estar correctamente ordenados y el denominador no debe ser cero.

### Diferencia temporal

\[
\Delta x_t = x_t - x_{t-1}
\]

- \(x_t\): valor actual.
- \(x_{t-1}\): valor rezagado.
- **Qué mide**: cambio absoluto entre periodos, implementado con `diff()` o `x - x.shift()`.
- **Interpretación**: el signo indica dirección; la magnitud conserva las unidades originales.
- **Supuesto importante**: el significado del periodo depende del índice y de posibles huecos del calendario.

### Media móvil simple

\[
\operatorname{SMA}_t = \frac{1}{n_t}\sum_{i \in W_t} x_i
\]

- \(W_t\): observaciones incluidas en la ventana que termina en \(t\).
- \(n_t\): número de observaciones válidas en esa ventana.
- \(x_i\): valor de cada observación.
- **Qué mide**: nivel medio reciente.
- **Interpretación**: suaviza variación local; una ventana mayor responde más lentamente.
- **Supuesto importante**: una ventana temporal como `'3D'` no equivale siempre a tres filas; depende de las fechas disponibles.

## 7. Implementación práctica

```python
import pandas as pd

# Separar antes de aprender cualquier estadística de transformación.
train = observations.loc[observations["date"] < "2024-01-01"].copy()
valid = observations.loc[observations["date"] >= "2024-01-01"].copy()

# Auditar claves antes de enriquecer la tabla.
missing_train_keys = train["entity_id"].drop_duplicates().difference(
    entities["entity_id"]
)
assert missing_train_keys.empty

train = train.merge(
    entities,
    on="entity_id",
    how="left",
    indicator=True,
    validate="many_to_one",  # Nota práctica: validación moderna de cardinalidad.
)
assert train["_merge"].eq("both").all()
train = train.drop(columns="_merge")
```

El capítulo recomienda comparar filas y claves antes/después del join. La opción `validate="many_to_one"` es una **nota práctica de API actual** que formaliza esa comprobación; no aparece en el ejemplo original.

```python
# Features retrospectivos: ordenar para impedir que una ventana use el futuro.
train = train.sort_values(["entity_id", "date"])
train["value_lag_1"] = train.groupby("entity_id")["value"].shift(1)
train["value_diff_1"] = train.groupby("entity_id")["value"].diff(1)
train["value_roll_mean_7"] = (
    train.groupby("entity_id")["value"]
    .transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
)
```

El `shift(1)` antes de la ventana es una **síntesis práctica derivada del capítulo** para predicción: asegura que la fila actual no contribuya a su propia feature.

```python
# Reducir datos intradía a granularidad diaria con semántica por columna.
daily = intraday.sort_index().resample("1D").agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volume=("volume", "sum"),
)
```

```python
# Alinear fuentes cercanas en el tiempo, no idénticas.
aligned = pd.merge_asof(
    left.sort_values("timestamp"),
    right.sort_values("timestamp"),
    on="timestamp",
    direction="backward",
    tolerance=pd.Timedelta(seconds=30),
)
```

Para modelado, `direction="backward"` es una **síntesis práctica derivada** cuando solo puede usarse información ya disponible; el capítulo demuestra también `direction="nearest"`, que puede asociar una observación futura.

**Notas prácticas de API actual:** el capítulo menciona `DataFrame.applymap()` para funciones elemento a elemento; en pandas actuales debe preferirse `DataFrame.map()`. También muestra `read_csv(..., date_parser=...)`; hoy conviene usar `date_format=` cuando el formato es conocido. `tshift()` ya no debe emplearse: desplazar el índice se expresa con `shift(freq=...)` o aritmética con `Timedelta`, según la intención. Estas son actualizaciones, no contenido original del capítulo.

## 8. Supuestos y condiciones de aplicación

1. Las claves de unión representan la misma entidad y tienen tipos/formato compatibles.
2. La cardinalidad del join es conocida; claves repetidas pueden crear relaciones muchos-a-muchos y multiplicar filas.
3. La tabla está ordenada por la variable temporal antes de `shift()`, `diff()`, ventanas o `merge_asof()`.
4. Las fechas fueron interpretadas como datetimes; para aprovechar toda la API temporal, se usa `DatetimeIndex`.
5. La función de agregación corresponde a la semántica de la variable: por ejemplo, apertura=`first`, máximo=`max`, mínimo=`min`, cierre=`last`, volumen=`sum`.
6. Los grupos comparados son estadísticamente pertinentes; el capítulo evita estandarizar clima de todo el año por estacionalidad.
7. Los bins representan una pérdida deliberada de resolución aceptable para el análisis.
8. Una ventana basada en tiempo y una basada en número de filas no son intercambiables si faltan fechas.
9. El relleno tras aumentar frecuencia supone una regla válida del dominio; los nuevos `NaN` no deben ocultarse automáticamente.
10. La tolerancia de `merge_asof()` refleja una proximidad temporal con significado real.

## 9. Errores y riesgos

- Elegir un inner join sin revisar diferencias de claves y perder observaciones silenciosamente.
- Usar un outer/left/right join y no investigar los `NaN` introducidos.
- No revisar conteos antes y después: duplicados en la clave pueden inflar el dataset.
- Confundir columnas homónimas tras el join; pandas agrega sufijos y estos deben interpretarse o renombrarse.
- Ejecutar joins grandes sin filtrar columnas/filas relevantes, con coste innecesario de memoria y tiempo.
- Usar `cut()` con outliers: bins de igual ancho pueden concentrar casi todas las filas en un solo grupo.
- Discretizar o recortar sin conservar la implicación de pérdida de información.
- Calcular Z-scores sobre periodos heterogéneos, mezclando regímenes o estacionalidad.
- Iterar filas cuando existe una operación vectorizada; el coste crece mucho más que con operaciones pandas/NumPy.
- Aplicar una función a columnas incompatibles o asumir que `apply()` convierte automáticamente código Python en una operación eficiente.
- Crear ventanas o lags sin ordenar por entidad y tiempo.
- **Data leakage**: calcular medias, desviaciones, cuantiles o límites de bins con entrenamiento y validación combinados.
- **Data leakage temporal**: usar ventanas centradas, `direction="nearest"` o relleno hacia atrás cuando una observación futura no estaba disponible al predecir.
- Usar `agg()` cuando se necesita una salida alineada fila a fila; en ese caso corresponde `transform()`.
- Agregar todas las columnas con la misma función aunque sus significados difieran.
- Upsampling sin estrategia explícita para los huecos creados.
- Tratar días ausentes como valores cero; un cierre de mercado o una medición faltante no equivale a cero.
- Crear pivots/crosstabs de categorías de cardinalidad alta y producir tablas excesivamente anchas.
- Sobreajuste indirecto al probar muchas ventanas, cortes o agregaciones contra la misma validación sin un esquema temporal sólido.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo**: cuando hay varias tablas, observaciones repetidas por entidad, fechas, necesidad de features agregados o distinta granularidad temporal.
- **Decisiones que habilita**: escoger el join, verificar su cardinalidad, elegir entre `cut` y `qcut`, decidir si una estadística reduce filas (`agg`) o se replica (`transform`) y definir ventana/frecuencia.
- **Baseline**: left join auditado; conteos y medias por grupo; `shift(1)` y una media móvil; resampling con agregaciones semánticas; crosstab para variables categóricas.
- **Mejoras**: múltiples horizontes de lag/rolling, estadísticas acumulativas o exponenciales, ratios respecto al grupo y alineación temporal con tolerancia controlada.
- **Qué evitar bajo presión**: pipes excesivamente abstractos, funciones personalizadas lentas, pivots de alta cardinalidad y combinaciones extensas de ventanas sin validación adecuada.
- **Señales activadoras**: claves repartidas entre archivos, cambio del número de filas tras unir, eventos repetidos por usuario, timestamps irregulares, diferencias de frecuencia, fuerte estacionalidad, outliers o categorías con distribución muy desigual.
- En un split temporal, toda feature debe poder reproducirse usando solo el pasado disponible. En un split aleatorio, estadísticas aprendidas como cortes de cuantiles siguen debiendo ajustarse exclusivamente con entrenamiento.

## 11. Checklist operativo

- [ ] Interpretar fechas y ordenar por entidad/tiempo.
- [ ] Identificar claves y cardinalidad esperada de cada join.
- [ ] Comparar claves y número de filas antes y después de unir.
- [ ] Revisar sufijos, nulos y procedencia de filas.
- [ ] Elegir `cut`, `qcut` o `clip` según el objetivo y cuantificar la pérdida de detalle.
- [ ] Preferir operaciones vectorizadas a bucles por fila.
- [ ] Ajustar medias, desviaciones, cuantiles y bins solo con entrenamiento.
- [ ] Crear lags y ventanas sin usar la observación actual ni el futuro.
- [ ] Elegir una agregación coherente para cada columna.
- [ ] Usar `transform()` si se necesita conservar una fila por observación.
- [ ] Auditar huecos tras `resample()` y tolerancia/dirección tras `merge_asof()`.
- [ ] Validar el pipeline con el mismo orden y granularidad que tendrá en inferencia.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Semántica, auditoría y cardinalidad de joins | `knowledge/preprocessing/dataframe_joins_and_aggregation.md` | 1 |
| `groupby`, `agg`, `transform`, pivots y crosstabs | `knowledge/eda/grouped_summaries_and_crosstabs.md` | 1 |
| Lags, diferencias y features de ventana | `knowledge/feature_engineering/tabular_and_window_features.md` | 1 |
| Selección temporal, resampling y granularidad | `knowledge/time_series/pandas_time_series_operations.md` | 1 |
| Prevención de leakage en agregaciones y ventanas | `knowledge/validation/leakage_in_feature_engineering.md` | 1 |
| Discretización con `cut`/`qcut` y umbrales | `knowledge/preprocessing/numeric_discretization.md` | 2 |
| Alineación con `merge_asof` y `merge_ordered` | `knowledge/time_series/temporal_joins.md` | 2 |
| Rendimiento mediante vectorización y pipes | `knowledge/feature_engineering/pandas_vectorization.md` | 2 |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y alcance | Introducción y materiales, pp. 192-194 (PDF 1-3) |
| Consultas y joins | “Database-style operations on DataFrames”, “Querying DataFrames” y “Merging DataFrames”, pp. 194-207 (PDF 3-16) |
| Validación mediante conjuntos | Final de “Merging DataFrames”, pp. 206-207 (PDF 15-16) |
| Operaciones aritméticas, ranking y cambio porcentual | “DataFrame operations” y “Arithmetic and statistics”, pp. 208-211 (PDF 17-20) |
| Binning, cuantiles y umbrales | “Binning and thresholds”, pp. 211-216 (PDF 20-25) |
| Aplicación de funciones y vectorización | “Applying functions”, pp. 216-218 (PDF 25-27) |
| Ventanas móviles, acumulativas y exponenciales | “Window calculations”, pp. 219-222 (PDF 28-31) |
| Pipes y función reutilizable de ventanas | “Pipes”, pp. 222-225 (PDF 31-34) |
| Agregaciones globales y por grupo | “Aggregations with pandas and numpy”, “Summarizing DataFrames” y “Using groupby”, pp. 226-234 (PDF 35-43) |
| Pivot tables y crosstabs | “Pivot tables and crosstabs”, pp. 234-239 (PDF 43-48) |
| Selección y filtrado temporal | “Time series” y “Time-based selection and filtering”, pp. 240-245 (PDF 49-54) |
| Rezagos, datos diferenciados y fechas válidas | “Shifting for lagged data” y “Differenced data”, pp. 245-247 (PDF 54-56) |
| Resampling y tratamiento de nueva frecuencia | “Resampling”, pp. 247-252 (PDF 56-61) |
| Uniones temporales | “Merging”, pp. 252-254 (PDF 61-63) |
| Confirmación del alcance del capítulo | “Summary”, p. 254 (PDF 63) |
| Ejercicios que consolidan técnicas | “Exercises”, pp. 255-256 (PDF 64-65) |

Todas las páginas del PDF suministrado fueron legibles. Las notas de API actual de la sección 7 están identificadas como adaptaciones y no se atribuyen al texto original.
