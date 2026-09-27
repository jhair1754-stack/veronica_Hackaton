---
title: "Empezar a trabajar con pandas"
topics:
  - "Series, DataFrame e índices de pandas"
  - "Reindexación, selección y alineación por etiquetas"
  - "Aplicación de funciones, ordenación y rangos"
  - "Estadística descriptiva, correlación y frecuencias"
target_knowledge_files:
  - "knowledge/preprocessing/pandas_indexacion_y_alineacion.md"
  - "knowledge/eda/estadistica_descriptiva_con_pandas.md"
  - "knowledge/feature_engineering/transformaciones_con_pandas.md"
  - "knowledge/validation/prevencion_de_leakage_en_transformaciones.md"
---

# 5. Empezar a trabajar con pandas

## 1. Propósito del capítulo

El capítulo presenta las estructuras y operaciones fundamentales de pandas para trabajar con datos tabulares o heterogéneos. Explica cómo representar datos con `Series` y `DataFrame`, manipularlos mediante índices etiquetados, reordenarlos, seleccionarlos, combinarlos mediante alineación automática y obtener resúmenes descriptivos.

Para Machine Learning, estas operaciones forman la base de la preparación de matrices de características, la inspección de variables, el filtrado de observaciones y el control de correspondencia entre filas, columnas y objetivos. El capítulo no desarrolla modelos predictivos ni estrategias de validación; su aportación es la manipulación fiable de los datos que alimentan esos procesos.

## 2. Conceptos esenciales

### `Series`

Una `Series` es una estructura unidimensional con:

* una secuencia de valores de un mismo tipo;
* un índice de etiquetas asociado;
* un atributo `name` opcional tanto para los valores como para el índice.

Si no se especifica un índice, pandas crea un `RangeIndex` desde `0` hasta `N - 1`. Los valores pueden consultarse mediante `.array` y el índice mediante `.index`. Una `Series` puede construirse desde una secuencia o desde un diccionario; al usar un diccionario, sus claves pasan a ser etiquetas. Si además se proporciona un índice explícito, este determina el orden y puede introducir valores faltantes para claves ausentes.

Las operaciones de estilo NumPy -filtrado booleano, multiplicación escalar y funciones universales- conservan la relación entre etiquetas y valores. La pertenencia con `in` se evalúa sobre el índice, no sobre los valores.

### `DataFrame`

Un `DataFrame` representa una tabla rectangular con una colección ordenada de columnas. Cada columna puede tener un tipo distinto y comparte un índice de filas. Puede construirse, entre otras opciones, desde:

* un diccionario de listas o arrays de igual longitud;
* un diccionario anidado;
* una lista de diccionarios;
* otro `DataFrame`.

La selección `frame["columna"]` o `frame.columna` devuelve una `Series` cuya referencia comparte los mismos datos subyacentes. La notación de atributo solo es segura si el nombre es un identificador válido y no entra en conflicto con un atributo existente; la notación con corchetes es más general.

Asignar una columna escalar propaga el valor a todas las filas. Asignar una `Series` alinea por índice, por lo que las etiquetas que no coinciden generan valores faltantes. `del frame["columna"]` elimina una columna. `frame.columns` e `frame.index` exponen los ejes, y `.to_numpy()` produce una representación NumPy homogénea, posiblemente con conversión a un tipo común.

### Índices (`Index`)

Los índices contienen las etiquetas de eje y otros metadatos. Son inmutables: no se puede modificar un elemento individual, aunque sí se puede sustituir el objeto de índice completo. Esta inmutabilidad permite compartir índices entre estructuras.

Un índice puede contener etiquetas duplicadas. Cuando una etiqueta se repite, seleccionarla devuelve todas sus apariciones y puede cambiar el tipo del resultado: una etiqueta única puede producir un escalar o una `Series`, mientras una duplicada produce una `Series` o un `DataFrame`. La propiedad `index.is_unique` permite detectar esta condición.

Los índices también ofrecen operaciones de conjuntos y utilidades como `append`, `difference`, `intersection`, `union`, `isin`, `delete`, `drop`, `insert`, `is_monotonic`, `is_unique` y `unique`.

### Alineación por etiquetas

pandas alinea automáticamente los operandos por sus etiquetas antes de operar. En una suma entre objetos con índices diferentes, el resultado usa la unión de etiquetas y deja `NaN` donde no existe una pareja. En `DataFrame`, la alineación ocurre tanto en filas como en columnas.

Esta semántica evita depender del orden físico, pero también puede crear datos faltantes de manera silenciosa si los ejes no coinciden. Los métodos aritméticos (`add`, `sub`, `div`, `floordiv`, `mul`, `pow` y sus variantes inversas con prefijo `r`) permiten usar `fill_value` para sustituir valores ausentes antes de operar.

### Selección por etiqueta y por posición

* `.loc[...]` selecciona por etiquetas.
* `.iloc[...]` selecciona por posiciones enteras.
* `.at[...]` y `.iat[...]` son accesos escalares por etiqueta y posición, respectivamente.
* Las máscaras booleanas permiten filtrar filas o valores.

Los cortes por etiqueta con `.loc` incluyen el extremo final, a diferencia de los cortes posicionales de Python. El capítulo advierte que mezclar enteros como etiquetas y como posiciones puede resultar ambiguo; usar `.loc` y `.iloc` explicita la intención.

## 3. Flujo o procedimiento presentado

1. Construir una `Series` o un `DataFrame` desde secuencias, diccionarios u otras estructuras.
2. Definir o inspeccionar `index` y `columns`; verificar si las etiquetas son únicas cuando esa propiedad importe.
3. Seleccionar filas y columnas con `.loc` para etiquetas o `.iloc` para posiciones.
4. Aplicar filtros booleanos para aislar observaciones o valores de interés.
5. Reindexar con `.reindex(...)` cuando sea necesario imponer un nuevo orden o conjunto de etiquetas.
6. Decidir explícitamente cómo tratar las etiquetas nuevas: mantener `NaN`, usar `fill_value` o aplicar un método de propagación como `ffill`/`bfill` cuando el orden lo justifique.
7. Realizar operaciones aritméticas sabiendo que pandas alineará ambos ejes por etiqueta.
8. Aplicar funciones por columna o fila con `.apply(...)`; reservar el mapeo elemento a elemento para transformaciones que realmente lo necesiten.
9. Ordenar etiquetas con `.sort_index(...)`, valores con `.sort_values(...)` o convertir valores en posiciones relativas con `.rank(...)`.
10. Resumir los datos mediante reducciones como `sum`, `mean`, `min`, `max`, `count`, `quantile` o `describe`.
11. Examinar asociaciones con `corr`, `cov` o `corrwith`, y distribuciones categóricas con `unique`, `value_counts` e `isin`.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Selección basada en nombres de filas o columnas | `.loc` | Hace explícita la semántica por etiqueta | El corte por etiqueta incluye ambos extremos |
| Selección basada en posición | `.iloc` | Evita confundir posiciones enteras con etiquetas enteras | Respeta la semántica de corte posicional de Python |
| Acceso a un único valor | `.at` o `.iat` | Expresa un acceso escalar por etiqueta o posición | Elegir el método según el tipo de referencia |
| Imponer un nuevo conjunto u orden de etiquetas | `.reindex` | Reordena y alinea sin modificar el original | Las etiquetas nuevas producen `NaN` salvo que se indique relleno |
| Rellenar huecos en una serie ordenada | `reindex(..., method="ffill")` o `bfill` | Propaga el valor conocido anterior o posterior | Solo es razonable cuando el orden tiene significado; puede usarse `limit` |
| Eliminar filas o columnas por etiqueta | `.drop(..., axis=...)` o `.drop(columns=...)` | Devuelve un nuevo objeto sin las etiquetas indicadas | Comprobar el eje; el comportamiento por defecto no modifica el original |
| Filtrar observaciones | Máscara booleana con `.loc` | Mantiene etiquetas y permite combinar condiciones | Alinear la máscara con el índice correcto |
| Asignar sobre una selección | Una sola operación `.loc[filas, columnas] = valor` | Evita la ambigüedad de vista frente a copia | No usar indexación encadenada para asignar |
| Operar con objetos de ejes distintos | Operaciones alineadas y, si procede, métodos con `fill_value` | pandas une etiquetas antes de calcular | La unión puede introducir `NaN` y cambiar el conjunto de filas/columnas |
| Función que reduce cada columna o fila | `.apply(func, axis=...)` | Devuelve un resumen por eje | Confirmar si la función devuelve escalar o serie |
| Transformación elemento a elemento | `Series.map(...)`; para `DataFrame`, mapeo elemento a elemento | Aplica una función a cada celda | Suele ser más costoso que una operación vectorizada |
| Ordenar por etiquetas | `.sort_index(axis=...)` | Organiza el eje sin alterar los valores relativos | En columnas, especificar `axis="columns"` |
| Ordenar por contenido | `.sort_values(by=...)` | Ordena según una o varias columnas/valores | Definir cómo tratar descendencia y faltantes |
| Resolver empates en rangos | `.rank(method=...)` | Permite elegir `average`, `min`, `max`, `first` o `dense` | El método cambia la interpretación del rango |
| Detectar etiquetas problemáticas | `index.is_unique` | Anticipa resultados de forma variable | Varias operaciones, como `reindex`, requieren unicidad |
| Contar categorías | `.value_counts()` | Resume frecuencias de forma directa | El orden de salida es por frecuencia de forma predeterminada |
| Comprobar pertenencia | `.isin(conjunto)` | Genera una máscara booleana | No confundir con `in`, que sobre una `Series` consulta el índice |

## 5. Métodos y técnicas

### Reindexación

* **Objetivo:** crear un objeto nuevo cuyos datos estén reordenados y alineados con nuevas etiquetas.
* **Entradas:** nuevas etiquetas mediante `index`, `columns` o `axis`; opcionalmente `method`, `fill_value`, `limit` y `tolerance`.
* **Salida:** una nueva `Series` o `DataFrame`.
* **Funcionamiento:** conserva valores de etiquetas existentes, descarta las no incluidas e introduce faltantes para etiquetas nuevas.
* **Parámetros relevantes:** `method="ffill"`/`"bfill"`, `fill_value`, `limit` y `tolerance`.
* **Ventajas:** orden y correspondencia explícitos; útil para homologar esquemas.
* **Limitaciones:** puede crear `NaN`; los métodos de relleno dependen de un orden significativo y normalmente de un índice ordenado.
* **Casos adecuados:** imponer el orden esperado de features, alinear observaciones o expandir un índice temporal.
* **No conviene:** usar propagación de valores cuando el orden no representa continuidad o cuando implicaría usar información no disponible.

### Eliminación con `drop`

* **Objetivo:** eliminar etiquetas de uno de los ejes.
* **Entradas:** etiqueta o secuencia de etiquetas y el eje (`index`, `columns` o `axis`).
* **Salida:** un nuevo objeto sin esas entradas, salvo que se solicite modificación in-place.
* **Funcionamiento:** retira las filas o columnas seleccionadas por etiqueta.
* **Ventajas:** operación explícita y legible.
* **Limitaciones:** un eje equivocado puede retirar elementos distintos de los pretendidos.
* **Casos adecuados:** descartar columnas auxiliares o filas identificadas.
* **No conviene:** eliminar por posición cuando las posiciones puedan cambiar; en ese caso conviene resolver primero las etiquetas.

### Indexación y filtrado

* **Objetivo:** seleccionar subconjuntos o actualizar posiciones concretas.
* **Entradas:** etiquetas, posiciones, listas, cortes o máscaras booleanas.
* **Salida:** escalar, `Series` o `DataFrame`, según la selección y la unicidad del índice.
* **Funcionamiento:** `.loc` opera por etiqueta y `.iloc` por posición; ambas admiten selección bidimensional.
* **Ventajas:** expresividad y preservación de metadatos.
* **Limitaciones:** etiquetas duplicadas cambian la cardinalidad y el tipo del resultado; los cortes por etiqueta son inclusivos.
* **Casos adecuados:** separar columnas predictoras, seleccionar observaciones y aplicar reglas de limpieza.
* **No conviene:** asignar mediante operaciones encadenadas como `data[mask]["col"] = valor`.

### Aritmética alineada

* **Objetivo:** combinar objetos pandas respetando etiquetas.
* **Entradas:** dos `Series`/`DataFrame` o un objeto y un escalar.
* **Salida:** objeto con la unión de los ejes y resultados alineados.
* **Funcionamiento:** empareja por etiquetas; donde falta un operando, genera `NaN`, salvo tratamiento con `fill_value`.
* **Parámetros relevantes:** `fill_value` en métodos aritméticos y `axis` al operar entre `DataFrame` y `Series`.
* **Ventajas:** evita depender del orden físico.
* **Limitaciones:** una desalineación no detectada puede expandir el resultado o introducir faltantes.
* **Casos adecuados:** restar baselines por columna, normalizar por una `Series` o combinar tablas homologadas.
* **No conviene:** asumir que objetos de igual longitud tienen las mismas observaciones sin verificar etiquetas.

### Aplicación y mapeo de funciones

* **Objetivo:** ejecutar funciones sobre columnas, filas o elementos.
* **Entradas:** función callable y, para `apply`, el eje.
* **Salida:** escalar, `Series` o `DataFrame`, según el valor devuelto por la función.
* **Funcionamiento:** `DataFrame.apply` entrega cada columna por defecto y cada fila con `axis="columns"`; `Series.map` trabaja elemento a elemento.
* **Ventajas:** permite encapsular transformaciones no cubiertas por una operación directa.
* **Limitaciones:** el mapeo Python por elemento puede ser más lento que operaciones vectorizadas.
* **Casos adecuados:** extraer rango (`max - min`) por columna, generar resúmenes por eje o formatear valores.
* **No conviene:** reemplazar operaciones vectorizadas simples con funciones Python.

**Nota práctica de API actual:** el capítulo usa `DataFrame.applymap` para el mapeo elemento a elemento. En versiones modernas de pandas, la operación equivalente se expresa con `DataFrame.map`. Esta actualización no forma parte del texto original; se conserva la idea del autor y se adapta solo el nombre de API.

### Ordenación y rangos

* **Objetivo:** ordenar ejes o valores y asignar posiciones ordinales.
* **Entradas:** eje, columnas de orden, dirección y método de desempate.
* **Salida:** objeto ordenado o rangos numéricos.
* **Funcionamiento:** `sort_index` ordena etiquetas; `sort_values` ordena contenido; `rank` transforma valores en rangos.
* **Hiperparámetros relevantes:** `ascending`, `axis`, `by`, `na_position` y `rank(method=...)`.
* **Ventajas:** hace determinista el orden de presentación y permite construir variables ordinales.
* **Limitaciones:** el método de desempate puede cambiar resultados posteriores.
* **Casos adecuados:** inspección, selección de extremos y generación de rankings.
* **No conviene:** interpretar el rango como distancia numérica entre valores.

### Estadística descriptiva y reducciones

* **Objetivo:** condensar valores por eje.
* **Entradas:** objeto, eje y opciones de tratamiento de faltantes.
* **Salida:** escalar o `Series` de resúmenes.
* **Funcionamiento:** métodos como `sum`, `mean`, `median`, `min`, `max`, `quantile`, `var`, `std`, `skew`, `kurt` y `describe` reducen un eje. `idxmin`/`idxmax` devuelven etiquetas; `cumsum`, `cummin`, `cummax`, `cumprod`, `diff` y `pct_change` producen transformaciones acumuladas o de cambio.
* **Parámetros relevantes:** `axis`, `skipna` y, en algunos métodos, `numeric_only`.
* **Ventajas:** diagnóstico rápido de escala, tendencia central, dispersión y faltantes.
* **Limitaciones:** el tratamiento predeterminado de `NaN` puede ocultar que hay datos ausentes; los resúmenes no sustituyen una inspección de distribución.
* **Casos adecuados:** EDA inicial y controles de calidad.
* **No conviene:** interpretar correlación o estadísticos agregados como causalidad o como garantía de utilidad predictiva.

### Correlación, covarianza y frecuencias

* **Objetivo:** medir asociación lineal y resumir valores discretos.
* **Entradas:** series o columnas numéricas para `corr`/`cov`; series discretas para `unique`, `value_counts` e `isin`.
* **Salida:** escalar, vector, matriz o máscara booleana.
* **Funcionamiento:** `corr` y `cov` emparejan observaciones no faltantes por etiqueta; `DataFrame.corr` y `.cov` generan matrices; `corrwith` compara columnas o filas coincidentes. `unique` obtiene valores distintos, `value_counts` sus frecuencias e `isin` prueba pertenencia.
* **Ventajas:** diagnóstico compacto de relación y distribución.
* **Limitaciones:** la alineación previa puede reducir o alterar el conjunto efectivo de pares; la correlación no establece causalidad.
* **Casos adecuados:** detectar redundancia lineal, revisar variables categóricas y construir filtros.
* **No conviene:** calcular asociaciones sin verificar primero índice, faltantes y significado de las variables.

## 6. Fórmulas esenciales

El capítulo no desarrolla fórmulas matemáticas como parte de la explicación. Las relaciones operativas esenciales son:

### Alineación aritmética

Para dos series `x` e `y` con índices `I_x` e `I_y`, el índice conceptual del resultado de una suma es:

\[
I_{resultado} = I_x \cup I_y
\]

Para cada etiqueta \(i\):

\[
z_i = x_i + y_i
\]

si ambos valores existen; en caso contrario, el resultado es faltante, salvo que se utilice un valor de relleno. Aquí \(x_i\) e \(y_i\) son los valores asociados a la misma etiqueta \(i\), no necesariamente a la misma posición física.

### Rango de una serie

Una función aplicada en el capítulo calcula:

\[
rango(x) = \max(x) - \min(x)
\]

donde `x` es una fila o columna, `max(x)` su valor máximo y `min(x)` su valor mínimo. Mide la amplitud observada, pero es sensible a valores extremos.

### Cambio porcentual

La operación `pct_change` representa conceptualmente:

\[
\Delta_t = \frac{x_t - x_{t-1}}{x_{t-1}}
\]

donde \(x_t\) es el valor actual y \(x_{t-1}\) el valor anterior según el orden del eje. Su interpretación exige que ese orden sea significativo; si \(x_{t-1}=0\), el cociente no es finito.

## 7. Implementación práctica

Ejemplo mínimo para seleccionar características y objetivo sin ambigüedad:

```python
import pandas as pd

data = pd.DataFrame(
    {
        "edad": [24, 39, 31, 47],
        "ingreso": [1800, 3100, 2600, 4200],
        "abandono": [0, 1, 0, 1],
    },
    index=["c1", "c2", "c3", "c4"],
)

feature_columns = ["edad", "ingreso"]
X = data.loc[:, feature_columns]
y = data.loc[:, "abandono"]
```

Ejemplo de separación previa y cálculo de una transformación solo con entrenamiento. Esta separación es una **síntesis práctica derivada del capítulo**, no un procedimiento de validación explicado en él:

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

train_mean = X_train.mean()
X_train_centered = X_train.sub(train_mean, axis="columns")
X_valid_centered = X_valid.sub(train_mean, axis="columns")
```

El segundo bloque usa la alineación por nombres de columnas que explica el capítulo. La media se calcula solo en entrenamiento para evitar leakage; esta precaución no se presenta como regla de validación en el capítulo, sino como aplicación en Machine Learning.

Ejemplo de asignación segura en una sola operación:

```python
missing_income = data["ingreso"].isna()
data.loc[missing_income, "ingreso"] = 0
```

Ejemplo de EDA compacto:

```python
numeric_summary = data[["edad", "ingreso"]].describe()
target_counts = data["abandono"].value_counts()
numeric_correlation = data[["edad", "ingreso"]].corr()
```

## 8. Supuestos y condiciones de aplicación

1. `Series` presupone una secuencia unidimensional cuyos valores comparten un tipo compatible.
2. `DataFrame` presupone una estructura rectangular; columnas construidas desde arrays deben tener longitudes compatibles.
3. Las operaciones alineadas presuponen que las etiquetas identifican correctamente observaciones o variables equivalentes.
4. `reindex(..., method="ffill"/"bfill")` requiere un orden con significado para que la propagación sea interpretable.
5. La selección debe distinguir explícitamente etiquetas (`loc`) de posiciones (`iloc`), en especial cuando el índice contiene enteros.
6. Las operaciones que requieren ejes únicos deben precederse de una comprobación con `index.is_unique`.
7. Las reducciones estadísticas deben revisar el tratamiento de faltantes; muchas omiten `NaN` de forma predeterminada.
8. `corr`, `cov` y `corrwith` utilizan observaciones emparejadas y alineadas por etiqueta; el conjunto efectivo puede ser menor que el original.
9. Las operaciones por fila o columna con `axis` presuponen que se ha identificado correctamente qué eje representa observaciones y cuál variables.
10. `diff` y `pct_change` presuponen que el orden actual de las filas es el orden analítico correcto.

## 9. Errores y riesgos

* **Desalineación silenciosa:** sumar o asignar objetos con etiquetas diferentes puede introducir `NaN` o ampliar el resultado.
* **Indexación encadenada:** una asignación en dos pasos puede operar sobre una copia y provocar `SettingWithCopyWarning`; debe reescribirse como una única asignación con `.loc`.
* **Confundir etiquetas y posiciones:** usar enteros sin `.loc`/`.iloc` hace el código ambiguo y frágil.
* **Cortes inclusivos inesperados:** un corte por etiquetas incluye el extremo final.
* **Índices duplicados:** una misma expresión puede devolver un escalar, una `Series` o un `DataFrame`, lo que complica código posterior.
* **Relleno injustificado:** `ffill` o `bfill` puede inventar continuidad donde no la hay y, en datos temporales, puede introducir información indebida si se usa en dirección incorrecta.
* **Faltantes ocultos por reducciones:** al omitir `NaN`, un resumen puede parecer completo aunque se base en menos observaciones.
* **Conversión de tipos:** `to_numpy()` necesita un tipo común y puede perder la heterogeneidad original del `DataFrame`.
* **Coste computacional:** `apply` y el mapeo elemento a elemento con funciones Python pueden ser más lentos que expresiones vectorizadas.
* **Orden incorrecto:** `diff` y `pct_change` producen resultados engañosos si las filas no están en el orden analítico debido.
* **Correlación mal interpretada:** una correlación alta no implica causalidad; una baja no descarta relaciones no lineales.
* **Data leakage - síntesis práctica derivada:** cualquier estadístico usado para transformar datos de validación debe estimarse solo con entrenamiento. La capacidad de alineación de pandas facilita aplicar el mismo vector de estadísticas, pero no impone esa separación automáticamente.
* **Leakage de objetivo - síntesis práctica derivada:** seleccionar columnas por exclusión descuidada puede dejar el objetivo o variables derivadas de él dentro de `X`; conviene declarar explícitamente las columnas de features.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Conviene consultar este capítulo cuando el reto entrega datos tabulares y exige construir rápidamente una canalización fiable de inspección y preparación. Permite decidir cómo representar los datos, seleccionar features y objetivo, alinear tablas, filtrar filas, ordenar, resumir y detectar categorías o asociaciones.

Un baseline puede apoyarse en:

* comprobación de forma, nombres de columnas, tipos e índices;
* separación explícita de `X` e `y` con `.loc`;
* `describe`, `count`, `value_counts` y estadísticas de faltantes;
* selección de columnas numéricas/categóricas sin asignaciones encadenadas;
* alineación explícita de las columnas de validación con las de entrenamiento mediante `reindex(columns=...)`;
* transformaciones vectorizadas y estadísticas calculadas en entrenamiento.

Para mejorar el modelo, el capítulo aporta operaciones útiles para crear diferencias, cambios porcentuales, rangos por fila/columna, frecuencias, indicadores de pertenencia y variables de ranking. Esas transformaciones solo deben usarse cuando su significado corresponda al problema.

Bajo presión de tiempo no conviene escribir funciones por celda si existe una operación vectorizada, reconstruir índices complejos sin necesidad, ni aplicar rellenos direccionales sin entender el orden. Tampoco conviene dedicar tiempo a correlaciones exhaustivas antes de verificar faltantes, duplicados y correspondencia entre `X` e `y`.

Señales del dataset que activan estas técnicas:

* columnas o filas en distinto orden entre train y test: `reindex` y alineación por etiquetas;
* identificadores repetidos: `index.is_unique` y revisión de duplicados;
* variables categóricas de baja o media cardinalidad: `value_counts`, `unique`, `isin`;
* series ordenadas en el tiempo: `diff`, `pct_change`, y relleno direccional solo cuando sea válido;
* múltiples tablas con claves comunes: comprobar índices antes de operar;
* valores faltantes tras una operación: revisar si provienen de desalineación y no solo del origen;
* columnas muy correlacionadas: `corr` como señal exploratoria de posible redundancia.

## 11. Checklist operativo

* [ ] Confirmar forma, columnas, tipos e índice del conjunto de datos.
* [ ] Verificar si el índice debe ser único con `index.is_unique`.
* [ ] Definir explícitamente columnas predictoras y objetivo.
* [ ] Usar `.loc` para etiquetas y `.iloc` para posiciones.
* [ ] Evitar asignaciones mediante indexación encadenada.
* [ ] Alinear columnas de validación/test con las de entrenamiento.
* [ ] Investigar todo `NaN` aparecido después de reindexar u operar.
* [ ] Calcular estadísticas de transformación solo con entrenamiento.
* [ ] Preferir operaciones vectorizadas frente a funciones por elemento.
* [ ] Ordenar correctamente antes de usar `diff` o `pct_change`.
* [ ] Revisar distribuciones con `describe` y `value_counts`.
* [ ] Interpretar correlaciones como asociación exploratoria, no causalidad.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Semántica de `Series`, `DataFrame` e índices | `knowledge/preprocessing/estructuras_pandas.md` | Alta |
| Selección segura con `loc`, `iloc`, `at` e `iat` | `knowledge/preprocessing/pandas_indexacion_y_alineacion.md` | Alta |
| Reindexación y control de columnas entre conjuntos | `knowledge/preprocessing/pandas_indexacion_y_alineacion.md` | Alta |
| Riesgos de índices duplicados y desalineación | `knowledge/preprocessing/calidad_de_indices.md` | Alta |
| Prevención de leakage al calcular transformaciones | `knowledge/validation/prevencion_de_leakage_en_transformaciones.md` | Alta |
| Transformaciones con `apply`, `map`, `diff`, `pct_change` y `rank` | `knowledge/feature_engineering/transformaciones_con_pandas.md` | Media |
| Estadísticos descriptivos y tratamiento de faltantes | `knowledge/eda/estadistica_descriptiva_con_pandas.md` | Alta |
| Correlación, covarianza y `corrwith` | `knowledge/eda/asociacion_entre_variables.md` | Media |
| Valores únicos, frecuencias y pertenencia | `knowledge/eda/perfiles_de_variables_categoricas.md` | Alta |
| Operaciones ordenadas y cambios porcentuales | `knowledge/time_series/transformaciones_basicas_con_pandas.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y contraste pandas/NumPy | Inicio del capítulo, pp. 171-172 |
| `Series` | 5.1, pp. 172-179 |
| `DataFrame` | 5.1, pp. 179-188 |
| Objetos índice | 5.1, pp. 188-191 |
| Reindexación | 5.2, pp. 192-196 |
| Eliminación de entradas de un eje | 5.2, pp. 196-198 |
| Indexación, selección y filtrado | 5.2, pp. 199-213 |
| Selección con `loc` e `iloc` y asignación | 5.2, pp. 201-213 |
| Aritmética y alineación de datos | 5.2, pp. 213-221 |
| Aplicación y mapeo de funciones | 5.2, pp. 221-225 |
| Ordenación y asignación de rangos | 5.2, pp. 225-231 |
| Índices con etiquetas duplicadas | 5.2, pp. 232-233 |
| Estadísticas descriptivas y reducciones | 5.3, pp. 234-238 |
| Correlación y covarianza | 5.3, pp. 238-241 |
| Valores únicos, recuentos y pertenencia | 5.3, pp. 241-245 |
| Conclusión | 5.4, p. 246 |
| Síntesis para coding tests y precauciones de leakage | Derivación práctica basada principalmente en 5.2-5.3, pp. 192-245 |

