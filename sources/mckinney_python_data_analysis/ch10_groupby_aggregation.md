---
title: "Agregación de datos y operaciones con grupos"
topics:
  - "pandas groupby y split-apply-combine"
  - "Agregación, transformación y apply por grupos"
  - "Tablas dinámicas y tabulaciones cruzadas"
target_knowledge_files:
  - "knowledge/eda/grouped_analysis.md"
  - "knowledge/preprocessing/groupwise_transformations.md"
  - "knowledge/feature_engineering/group_aggregations.md"
  - "knowledge/validation/data_leakage.md"
---

# 10. Agregación de datos y operaciones con grupos

## 1. Propósito del capítulo

El capítulo presenta el paradigma **dividir-aplicar-combinar** (*split-apply-combine*) de pandas para segmentar datos por una o varias claves, aplicar cálculos dentro de cada grupo y recombinar los resultados. Cubre agregaciones, transformaciones que conservan la forma, funciones generales con `apply`, análisis por cuantiles, imputación por grupo, muestreo, estadística por segmentos, tablas dinámicas y frecuencias cruzadas.

Para Machine Learning, estas operaciones permiten explorar heterogeneidad entre segmentos, construir variables agregadas, imputar o normalizar por grupo y producir diagnósticos tabulares antes del modelado.

## 2. Conceptos esenciales

### 2.1 Dividir-aplicar-combinar

**Contenido del capítulo.** Una operación agrupada tiene tres etapas:

1. **Dividir:** separar una `Series` o un `DataFrame` según una o varias claves.
2. **Aplicar:** ejecutar una agregación, transformación o función arbitraria sobre cada grupo.
3. **Combinar:** ensamblar los resultados en una `Series`, un `DataFrame` o un valor por grupo.

Las claves pueden ser columnas, arrays de longitud compatible, diccionarios o `Series` que mapean etiquetas, funciones aplicadas a las etiquetas del índice, o niveles de un `MultiIndex`. Varias claves producen normalmente un índice jerárquico.

Un objeto `GroupBy` representa la partición y difiere el cálculo hasta que se solicita una operación como `mean`, `agg`, `transform` o `apply`.

### 2.2 Tamaño, recuento y valores ausentes

**Contenido del capítulo.** `size()` cuenta filas del grupo; `count()` cuenta valores no nulos por columna. Las claves nulas se excluyen de forma predeterminada; `dropna=False` conserva un grupo para ellas.

**Conclusión práctica derivada.** Comparar `size()` y `count()` ayuda a detectar ausencia de datos dentro de cada segmento. La decisión sobre `dropna` debe ser explícita porque puede cambiar la población representada.

### 2.3 Selección de columnas y forma del resultado

Seleccionar una columna con un nombre escalar produce un `SeriesGroupBy`; seleccionar una lista produce un `DataFrameGroupBy`. `as_index=False` devuelve las claves como columnas y evita tener que aplicar después `reset_index()`. `group_keys=False` evita añadir las claves como nivel exterior cuando `apply` concatena los resultados.

### 2.4 Tres familias de operación

- **Agregación (`agg`):** reduce cada grupo a uno o varios valores resumen.
- **Transformación (`transform`):** devuelve un escalar difundible o un objeto con la misma forma que el grupo; no debe mutar la entrada.
- **Aplicación general (`apply`):** acepta una función que devuelve un escalar u objeto pandas y concatena los resultados; es flexible, pero suele tener más sobrecarga.

### 2.5 Tablas dinámicas y tabulaciones cruzadas

`pivot_table` organiza una agregación rectangular con claves en filas y columnas. Puede elegir valores, función de agregación, relleno, categorías observadas y márgenes. `crosstab` es la interfaz directa para frecuencias entre una o más variables.

## 3. Flujo o procedimiento presentado

1. Identificar las columnas o etiquetas que definen los grupos.
2. Decidir si las claves nulas deben excluirse o conservarse mediante `dropna`.
3. Seleccionar solo las columnas necesarias antes de calcular, especialmente en datos grandes.
4. Elegir la clase de operación:
   1. `agg` si se necesita una fila o varios escalares por grupo;
   2. `transform` si el resultado debe alinearse con las filas originales;
   3. `apply` si la lógica devuelve estructuras variables o combina varias columnas.
5. Preferir operaciones optimizadas (`mean`, `sum`, `count`, `std`, `min`, `max`, `quantile`, etc.) frente a funciones Python personalizadas cuando expresen el mismo cálculo.
6. Controlar la forma del resultado con selección de columnas, `as_index`, `group_keys`, `unstack` o una tabla dinámica.
7. Verificar tamaños, nulos y combinaciones vacías antes de interpretar resultados.

**Recomendación para competencias.** En un pipeline predictivo, calcular cualquier estadística usada para imputar, normalizar o crear variables solo con el conjunto de entrenamiento; después mapearla sobre validación o test. Esta precaución es una derivación para evitar *data leakage*, no una instrucción explícita del capítulo.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se necesita un resumen escalar por grupo | `groupby(...).agg(...)` o método optimizado | Reduce cada grupo y aprovecha implementaciones rápidas | `count` ignora nulos; `size` cuenta filas |
| Se necesitan varias métricas | `agg(["count", "mean", "std"])` | Calcula múltiples resúmenes en una operación | Puede producir columnas jerárquicas |
| Cada columna requiere métricas distintas | `agg({"col_a": [...], "col_b": "sum"})` | Expresa un mapeo columna-función | Comprobar nombres y niveles del resultado |
| El resultado debe conservar una fila por observación | `transform(...)` | Mantiene alineación y tamaño | La función no debe mutar la entrada |
| La operación devuelve filas variables o combina columnas | `apply(...)` | Es la interfaz más general | Mayor sobrecarga y forma de salida menos predecible |
| Se necesitan intervalos de amplitud fija | `pd.cut` + `groupby` | Segmenta mediante límites del rango | Los grupos pueden quedar muy desbalanceados |
| Se necesitan grupos de tamaño parecido | `pd.qcut` + `groupby` | Segmenta según cuantiles muestrales | Valores repetidos pueden complicar los cortes |
| Se quiere resumir por filas y columnas | `pivot_table` | Combina agrupación y remodelado | La media es la agregación predeterminada |
| Se quieren frecuencias entre categorías | `pd.crosstab` | Es el caso especializado de conteos cruzados | Distinguir conteo de filas de conteo no nulo |
| Hay claves de agrupación ausentes relevantes | `groupby(..., dropna=False)` | Conserva el segmento nulo | El valor nulo pasa a ser una categoría analítica |

## 5. Métodos y técnicas

### 5.1 `groupby`

- **Objetivo:** dividir datos según una o varias claves.
- **Entradas:** nombres de columnas, arrays, funciones, mapeos o niveles del índice.
- **Salida:** objeto `GroupBy` de evaluación diferida.
- **Funcionamiento:** conserva la información necesaria para aplicar operaciones posteriores a cada fragmento.
- **Parámetros relevantes:** `dropna`, `as_index`, `group_keys`, `level`.
- **Ventajas:** composición flexible; claves múltiples; compatibilidad con índices jerárquicos.
- **Limitaciones:** la forma final depende de la operación; índices jerárquicos pueden complicar integraciones posteriores.
- **Casos adecuados:** EDA segmentado, resúmenes, creación de variables, imputación o normalización por grupo.
- **No conviene:** cuando no existe una clave semántica útil o cuando una operación vectorizada global expresa mejor el objetivo.

### 5.2 Agregaciones y `agg`

- **Objetivo:** convertir cada grupo en escalares de resumen.
- **Entradas:** una función, nombre de función, lista de funciones o diccionario columna-función.
- **Salida:** `Series` o `DataFrame`, posiblemente con índices o columnas jerárquicos.
- **Funcionamiento:** aplica cada agregador a cada grupo y combina los resultados.
- **Operaciones destacadas:** `any`, `all`, `count`, `size`, `first`, `last`, `mean`, `median`, `min`, `max`, `sum`, `prod`, `std`, `var`, `quantile`, `rank`, acumulados y `ohlc`.
- **Ventajas:** las funciones internas suelen estar optimizadas; permite varias métricas y especificaciones por columna.
- **Limitaciones:** las funciones Python personalizadas introducen llamadas y reordenación de datos adicionales.
- **Casos adecuados:** perfiles por clase, cliente, región, lote o período; estadísticas para variables agregadas.
- **No conviene:** si se requiere devolver una observación por fila original; usar `transform`.

### 5.3 `apply`

- **Objetivo:** ejecutar lógica general por grupo.
- **Entradas:** función que recibe el fragmento, más argumentos posicionales o nominales.
- **Salida:** escalar u objeto pandas; pandas concatena los resultados.
- **Funcionamiento:** divide, invoca la función en cada fragmento y recombina.
- **Parámetros relevantes:** los de la función; `group_keys` controla las claves externas del índice.
- **Ventajas:** permite top-N por grupo, estadística multicolumna, muestreo, correlación y regresión por segmento.
- **Limitaciones:** suele ser más lento y la forma final exige más cuidado.
- **Casos adecuados:** lógica no expresable con agregaciones o transformaciones integradas.
- **No conviene:** para cálculos simples ya disponibles como métodos optimizados.

### 5.4 `transform`

- **Objetivo:** producir valores alineados con las filas originales.
- **Entradas:** función o nombre de operación integrada.
- **Salida:** objeto del mismo tamaño que la entrada agrupada, o escalar difundido a ese tamaño.
- **Funcionamiento:** calcula dentro del grupo y restaura el orden/alineación original.
- **Ventajas:** útil para medias de grupo, rangos, imputación y estandarización; las operaciones internas ofrecen una vía rápida.
- **Limitaciones:** la salida debe ser compatible en forma y la función no debe mutar su entrada.
- **Casos adecuados:** centrar o escalar por grupo, crear desvíos respecto de la media grupal y rellenar por estadística grupal.
- **No conviene:** cuando cada grupo debe devolver un número diferente de filas.

### 5.5 `cut` y `qcut`

- **Objetivo:** discretizar una variable antes de analizarla por grupos.
- **Entradas:** serie y límites/número de intervalos (`cut`) o número de cuantiles (`qcut`).
- **Salida:** categórica de intervalos o etiquetas.
- **Funcionamiento:** `cut` divide el rango; `qcut` usa cuantiles de la muestra.
- **Ventajas:** facilita perfiles por bandas y detección de relaciones no lineales.
- **Limitaciones:** amplitud fija no garantiza tamaños iguales; cuantiles dependen de la distribución observada.
- **Casos adecuados:** análisis por deciles/cuartiles y diagnóstico de una variable continua.
- **No conviene:** si discretizar elimina señal continua necesaria para el modelo.

### 5.6 Imputación por grupo

- **Objetivo:** rellenar nulos con una media u otro valor específico de cada grupo.
- **Entradas:** serie, clave de grupo y regla de relleno.
- **Salida:** serie con valores imputados.
- **Funcionamiento:** cada grupo aplica `fillna` con su propia estadística o valor predefinido.
- **Ventajas:** respeta diferencias sistemáticas entre segmentos.
- **Limitaciones:** grupos sin observaciones válidas no producen una media utilizable; grupos pequeños dan estimaciones inestables.
- **Casos adecuados:** cuando la pertenencia al grupo justifica estadísticamente distintos centros.
- **No conviene:** cuando el grupo es un identificador casi único o no estará disponible en inferencia.

### 5.7 Estadística y muestreo por grupo

El capítulo demuestra muestreo con `sample`, media ponderada con `np.average`, correlaciones con `corrwith`/`corr` y regresión MCO por grupo con `statsmodels`. La función aplicada debe devolver un escalar, una serie de parámetros u otro objeto pandas combinable.

**Limitación práctica derivada.** Antes de comparar correlaciones o coeficientes entre grupos, revisar tamaño muestral y estabilidad. El capítulo muestra el mecanismo de cálculo, no un protocolo de inferencia ni validación.

### 5.8 `pivot_table` y `crosstab`

- **Objetivo:** resumir combinaciones de categorías en una matriz.
- **Entradas:** `index`, `columns`, `values`, `aggfunc`; opcionalmente `fill_value`, `dropna`, `margins`, `margins_name`, `observed`.
- **Salida:** tabla rectangular, potencialmente con índices jerárquicos.
- **Funcionamiento:** `pivot_table` agrega y remodela; `crosstab` calcula frecuencias.
- **Ventajas:** EDA compacto y comparaciones entre dimensiones.
- **Limitaciones:** combinaciones escasas generan nulos; rellenarlas con cero puede confundir ausencia de combinación con valor observado cero.
- **Casos adecuados:** auditoría de distribución de clases y variables categóricas.
- **No conviene:** como sustituto automático de una evaluación estadística o predictiva.

## 6. Fórmulas esenciales

### 6.1 Porcentaje o razón entre columnas

\[
r_i = \frac{a_i}{b_i}
\]

- \(a_i\): cantidad de interés de la observación \(i\), como la propina.
- \(b_i\): cantidad base, como la factura total.
- \(r_i\): proporción resultante.
- **Interpretación:** normaliza una cantidad respecto de su base para comparar observaciones de distinta escala.
- **Supuesto:** \(b_i\) no debe ser cero.

### 6.2 Media ponderada por grupo

\[
\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
\]

- \(x_i\): valor de la observación.
- \(w_i\): peso asociado.
- \(n\): número de observaciones del grupo.
- **Interpretación:** los valores con mayor peso influyen más en el promedio.
- **Supuesto:** la suma de pesos debe ser distinta de cero; los pesos deben tener significado válido.

### 6.3 Estandarización dentro del grupo

\[
z_{ig} = \frac{x_{ig} - \bar{x}_g}{s_g}
\]

- \(x_{ig}\): valor de la observación \(i\) en el grupo \(g\).
- \(\bar{x}_g\): media del grupo.
- \(s_g\): desviación estándar muestral del grupo.
- \(z_{ig}\): distancia estandarizada respecto de la media del grupo.
- **Interpretación:** cero indica el centro del grupo; el signo marca la dirección y la magnitud se expresa en desviaciones estándar.
- **Supuesto:** \(s_g > 0\); un grupo constante o demasiado pequeño no puede estandarizarse de forma útil.

## 7. Implementación práctica

### 7.1 Resumen eficiente por grupo

```python
summary = (
    train_df
    .groupby(["segment", "class"], dropna=False, as_index=False)
    .agg(
        rows=("target", "size"),
        observed_targets=("target", "count"),
        feature_mean=("feature", "mean"),
        feature_std=("feature", "std"),
    )
)
```

**Nota práctica moderna.** Se usa agregación con nombre para producir columnas planas y legibles. Es una adaptación de las variantes de `agg` mostradas en el capítulo.

### 7.2 Imputación por grupo sin fuga entre entrenamiento y validación

```python
group_medians = train_df.groupby("segment")["feature"].median()
global_median = train_df["feature"].median()

train_fill = train_df["segment"].map(group_medians).fillna(global_median)
valid_fill = valid_df["segment"].map(group_medians).fillna(global_median)

train_df["feature"] = train_df["feature"].fillna(train_fill)
valid_df["feature"] = valid_df["feature"].fillna(valid_fill)
```

**Síntesis práctica derivada del capítulo.** El cálculo por grupo procede del capítulo; ajustar el mapeo solo con entrenamiento y definir un respaldo global para grupos nuevos son medidas contra *data leakage* y fallos en inferencia.

### 7.3 Estandarización por grupo mediante operaciones simplificadas

```python
grouped = train_df.groupby("segment")["feature"]
group_mean = grouped.transform("mean")
group_std = grouped.transform("std")

train_df["feature_group_z"] = (
    (train_df["feature"] - group_mean) / group_std.replace(0, np.nan)
)
```

Para validación o test, no volver a estimar las estadísticas: guardar las medias y desviaciones del entrenamiento y mapearlas por grupo.

### 7.4 Top-N por grupo

```python
top_rows = (
    train_df
    .sort_values("score", ascending=False)
    .groupby("segment", group_keys=False)
    .head(3)
)
```

**Nota práctica moderna.** Esta forma vectorizada expresa el objetivo del ejemplo `apply(top)` con menos sobrecarga cuando solo se necesita top-N por ordenación.

### 7.5 Agrupación de columnas mediante un mapeo

```python
column_groups = {"a": "red", "b": "red", "c": "blue", "d": "blue"}
grouped_columns = frame.T.groupby(column_groups).sum().T
```

**Nota práctica moderna.** El capítulo usa `groupby(..., axis="columns")`; la transposición explícita evita depender de ese estilo antiguo de API.

## 8. Supuestos y condiciones de aplicación

1. Las claves externas deben tener la misma longitud que el eje agrupado.
2. Una agrupación múltiple produce combinaciones de claves y normalmente un `MultiIndex`.
3. Las funciones de agregación deben reducir cada grupo a escalares por salida.
4. `transform` debe devolver un escalar difundible o una salida de la misma forma y no debe mutar el grupo.
5. `apply` debe devolver valores u objetos pandas que puedan combinarse coherentemente.
6. `count` excluye valores nulos; `size` cuenta las filas.
7. `cut` usa intervalos del rango; `qcut` usa cuantiles observados.
8. Una media ponderada exige pesos compatibles y suma de pesos no nula.
9. La estandarización necesita desviación estándar no nula.
10. En tablas dinámicas, `mean` es la agregación predeterminada; categorías no observadas dependen de `observed`.
11. **Conclusión práctica derivada:** para usar estadísticas grupales como *features*, las claves deben existir en inferencia y debe haber una política para grupos desconocidos.

## 9. Errores y riesgos

- Confundir `size()` con `count()` y ocultar nulos por grupo.
- Olvidar que las claves nulas se excluyen de manera predeterminada.
- Agregar todas las columnas numéricas cuando solo se necesitan unas pocas.
- No prever índices o columnas jerárquicos tras múltiples claves o funciones.
- Usar `apply` para una operación que ya tiene método optimizado, con coste innecesario.
- Devolver desde `transform` una forma incompatible o mutar el fragmento recibido.
- Rellenar combinaciones vacías con cero sin distinguir “no observado” de cero real.
- Imputar con la media de un grupo vacío, constante o demasiado pequeño.
- Interpretar correlaciones o regresiones por grupo sin considerar tamaños y estabilidad.
- **Riesgo derivado de ML:** calcular medias, desviaciones, cuantiles, imputaciones o agregados con validación/test antes del ajuste produce *data leakage*.
- **Riesgo derivado de validación:** si varias filas pertenecen a la misma entidad, una división aleatoria puede compartir información grupal entre entrenamiento y validación.
- **Riesgo computacional:** las funciones personalizadas y `apply` añaden sobrecarga; seleccionar columnas y usar métodos integrados suele ser más eficiente.
- **Riesgo de sobreajuste derivado:** grupos muy granulares pueden generar estadísticas ruidosas o casi identificadores; no se debe convertir cada ID en una señal agregada sin validación adecuada.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** al encontrar variables categóricas, entidades repetidas, datos longitudinales, nulos dependientes de segmento o necesidad de perfiles por clase.
- **Decisiones que permite tomar:** conservar o excluir claves nulas; escoger entre agregación, transformación y aplicación general; elegir intervalos fijos o cuantiles; decidir la forma de tablas de diagnóstico.
- **Baseline útil:** `groupby(...).agg(["size", "count", "mean", "std", "min", "max"])` para revisar cobertura, nulos, centros, dispersión y extremos por grupo.
- **Mejoras posibles:** diferencias respecto de la media del grupo, *z-scores* por grupo, conteos/frecuencias, ratios y estadísticas agregadas de entidades, siempre ajustadas dentro del entrenamiento.
- **Qué evitar bajo presión:** `apply` complejo, regresiones por numerosos grupos, funciones Python fila a fila y tablas enormes con muchas dimensiones si no existe una hipótesis clara.
- **Señales activadoras:** IDs repetidos, jerarquías geográficas, lotes, tiendas, usuarios, categorías con distribuciones diferentes, proporciones de nulos variables o necesidad de top-N por segmento.
- **Validación:** si la clave representa entidad, tiempo o procedencia, considerar particiones que respeten esa estructura antes de crear agregados.

## 11. Checklist operativo

- [ ] Definir la unidad de observación y las claves de grupo.
- [ ] Comparar `size` y `count` para localizar nulos.
- [ ] Decidir explícitamente `dropna` y `observed`.
- [ ] Seleccionar únicamente las columnas necesarias.
- [ ] Usar `agg` para resúmenes y `transform` para salida alineada.
- [ ] Reservar `apply` para lógica que no tenga alternativa integrada.
- [ ] Revisar índices/columnas jerárquicos y nombres de salida.
- [ ] Calcular estadísticas de preprocesamiento solo con entrenamiento.
- [ ] Definir respaldo para grupos desconocidos o sin datos válidos.
- [ ] Verificar grupos pequeños, constantes y combinaciones vacías.
- [ ] Medir coste si hay muchos grupos o funciones personalizadas.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Split-apply-combine, claves y forma de resultados | `knowledge/eda/grouped_analysis.md` | Alta |
| `size`, `count`, `dropna` y auditoría de nulos | `knowledge/eda/missingness_by_group.md` | Alta |
| Agregaciones como variables | `knowledge/feature_engineering/group_aggregations.md` | Alta |
| Imputación y estandarización por grupo | `knowledge/preprocessing/groupwise_transformations.md` | Alta |
| Prevención de fuga al ajustar estadísticas grupales | `knowledge/validation/data_leakage.md` | Alta |
| División por entidades o grupos | `knowledge/validation/group_splitting.md` | Alta |
| `cut`, `qcut` y análisis por cuantiles | `knowledge/feature_engineering/binning.md` | Media |
| Tablas dinámicas y frecuencias cruzadas | `knowledge/eda/pivot_tables_and_crosstabs.md` | Media |
| Correlación y regresión por segmento | `knowledge/interpretability/segment_analysis.md` | Media |
| Coste de `apply` frente a operaciones integradas | `knowledge/preprocessing/pandas_performance.md` | Baja |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y alcance | p. 1; introducción del capítulo 10 |
| Split-apply-combine y tipos de claves | pp. 2-5; sección 10.1 |
| Nulos, `size`, `count` e iteración | pp. 6-9; sección 10.1 |
| Selección de columnas, mapeos, funciones y niveles | pp. 9-14; sección 10.1 |
| Agregaciones optimizadas y personalizadas | pp. 14-17; sección 10.2 y tabla 10.1 |
| Varias funciones, columnas y forma de salida | pp. 17-22; sección 10.2 |
| `apply`, argumentos y claves de grupo | pp. 22-25; sección 10.3 |
| `cut`, `qcut` y estadísticas por contenedor | pp. 25-28; sección 10.3 |
| Relleno de ausentes por grupo | pp. 28-32; sección 10.3 |
| Muestreo y permutación por grupo | pp. 32-34; sección 10.3 |
| Media ponderada y correlación por grupo | pp. 34-37; sección 10.3 |
| Regresión lineal por grupo | pp. 37-38; sección 10.3 |
| `transform` y operaciones simplificadas | pp. 38-43; sección 10.4 |
| Tablas dinámicas y opciones | pp. 43-47; sección 10.5 y tabla 10.2 |
| Tabulaciones cruzadas | pp. 47-49; sección 10.5 |
| Conclusión | p. 49; sección 10.6 |

**Estado de lectura:** se revisaron las 49 páginas del archivo suministrado. No se detectaron páginas ilegibles ni secciones que requirieran completar vacíos mediante suposiciones. Las notas sobre separación entrenamiento-validación, grupos desconocidos y APIs modernas están marcadas como derivaciones o notas prácticas y no se atribuyen al texto original.
