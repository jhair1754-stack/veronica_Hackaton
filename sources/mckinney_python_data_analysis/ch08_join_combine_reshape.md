---
title: "Disputa de datos: unión, combinación y remodelación"
topics:
  - "Indexación jerárquica con MultiIndex"
  - "Uniones y fusiones de DataFrames"
  - "Concatenación y combinación de datos solapados"
  - "Remodelación entre formatos largo y ancho"
target_knowledge_files:
  - "knowledge/preprocessing/data_integration_with_pandas.md"
  - "knowledge/preprocessing/tabular_reshaping.md"
  - "knowledge/feature_engineering/hierarchical_indexing.md"
  - "knowledge/time_series/long_wide_formats.md"
---

# 8. Disputa de datos: unión, combinación y remodelación

## 1. Propósito del capítulo

**Contenido explicado por el autor.** El capítulo presenta las operaciones de pandas necesarias para reunir datos repartidos entre tablas y reorganizarlos en una forma adecuada para el análisis. Parte de la indexación jerárquica (`MultiIndex`), desarrolla las fusiones relacionales con `merge` y `join`, la concatenación con `concat`, el relleno de datos solapados con `combine_first` y la remodelación con `stack`, `unstack`, `pivot` y `melt`.

**Conclusión práctica derivada.** En Machine Learning, estas operaciones forman la capa de integración previa a la selección de variables y al entrenamiento. Una unión incorrecta puede cambiar el número de observaciones, duplicar entidades, introducir valores ausentes o desalinear la variable objetivo; una remodelación correcta permite construir una matriz de modelado sin perder la identidad de cada observación.

## 2. Conceptos esenciales

- **Indexación jerárquica (`MultiIndex`).** Permite representar varias dimensiones mediante múltiples niveles de índice en un objeto de menor dimensión. Admite selección parcial, selección por niveles internos, nombres de niveles y jerarquías tanto en filas como en columnas (pp. 347-352).
- **Orden lexicográfico del índice.** `sort_index()` puede ordenar uno o varios niveles. El autor señala que la selección sobre índices jerárquicos funciona mejor cuando el índice está ordenado desde el nivel más externo (pp. 352-354).
- **Conversión entre columnas e índice.** `set_index()` mueve una o varias columnas al índice; `reset_index()` devuelve niveles del índice a columnas. `drop=False` conserva además las columnas originales al establecer el índice (pp. 355-356).
- **Fusión relacional (`pandas.merge`).** Conecta filas mediante una o varias claves, de forma análoga a una unión SQL. Las claves pueden ser columnas, índices o combinaciones de ambos (pp. 357-372).
- **Cardinalidad de una unión.** El capítulo muestra relaciones muchos-a-uno y muchos-a-muchos. Si una clave aparece tres veces a la izquierda y dos a la derecha, una fusión muchos-a-muchos produce seis filas para esa clave: el producto cartesiano de las coincidencias (pp. 358-363).
- **Tipo de unión.** `inner` conserva claves observadas en ambos lados; `left`, todas las claves de la izquierda; `right`, todas las de la derecha; `outer`, la unión de claves de ambos lados. Las filas sin correspondencia reciben valores ausentes en las columnas del otro objeto (pp. 360-361).
- **Trazabilidad de la unión.** `indicator=True` agrega `_merge` con `left_only`, `right_only` o `both`; `validate` comprueba que la cardinalidad declarada sea uno-a-uno, uno-a-muchos, muchos-a-uno o muchos-a-muchos (pp. 365-366).
- **Unión por índice (`DataFrame.join`).** Simplifica las uniones basadas en índices. A diferencia de `merge`, su comportamiento predeterminado es una unión izquierda. También puede enlazar el índice del objeto derecho con una columna del objeto izquierdo mediante `on` (pp. 366-372).
- **Concatenación (`pandas.concat`).** Apila objetos a lo largo de filas o columnas y alinea las etiquetas en el otro eje. `join="outer"` toma la unión de etiquetas; `join="inner"`, la intersección (pp. 373-381).
- **Identidad de los fragmentos concatenados.** `keys` añade un nivel jerárquico que identifica la procedencia de cada bloque. `ignore_index=True` descarta etiquetas sin valor semántico y crea un índice entero nuevo. `verify_integrity=True` detecta duplicados en el nuevo eje (pp. 376-381).
- **Combinación de datos solapados.** `combine_first` conserva los valores presentes del objeto que invoca el método y rellena sus ausencias con valores alineados por índice y columnas del otro objeto (pp. 381-383).
- **Apilado y desapilado.** `stack` mueve columnas a filas; `unstack` mueve un nivel de filas a columnas. Al desapilar pueden aparecer ausencias; `stack` excluye los valores ausentes de forma predeterminada en la API mostrada por el capítulo (pp. 384-389).
- **Formato largo.** Cada fila representa una observación y una columna identifica la variable medida. Es flexible para almacenar series cuyo conjunto de variables puede crecer y facilita integridad relacional mediante claves como fecha e ítem (pp. 389-392).
- **Formato ancho.** Cada variable ocupa una columna. `pivot` convierte una tabla larga en ancha; es equivalente conceptualmente a `set_index(...).unstack(...)` en el ejemplo del capítulo (pp. 392-394).
- **Conversión de ancho a largo.** `pandas.melt` combina varias columnas de valores en dos columnas: una identifica la variable y otra contiene el valor. `id_vars` conserva identificadores y `value_vars` limita las columnas remodeladas (pp. 394-397).

## 3. Flujo o procedimiento presentado

1. Identificar qué representa una fila en cada tabla y cuáles son las claves que enlazan observaciones.
2. Si la identidad requiere varias dimensiones, crear o nombrar un `MultiIndex` con `set_index()` o `MultiIndex.from_arrays()`.
3. Ordenar el índice jerárquico con `sort_index()` cuando se necesiten selecciones frecuentes o por niveles.
4. Elegir la operación de integración:
   1. `merge` para relacionar filas mediante claves;
   2. `join` para uniones principalmente basadas en índices;
   3. `concat` para apilar bloques por filas o columnas;
   4. `combine_first` para rellenar huecos con otra fuente alineada.
5. Especificar las claves de forma explícita mediante `on`, `left_on`/`right_on` o `left_index`/`right_index`.
6. Elegir `how` según las entidades que deban sobrevivir a la unión: intersección (`inner`), universo izquierdo (`left`), universo derecho (`right`) o unión completa (`outer`).
7. Examinar la cardinalidad de las claves antes de una fusión, especialmente si hay duplicados en ambos lados.
8. Resolver columnas solapadas con `suffixes` y conservar índices relevantes convirtiéndolos antes en columnas con `reset_index()` cuando sea necesario.
9. En concatenaciones, decidir el eje, la política de alineación (`join`), si se conserva el índice (`ignore_index`) y si debe registrarse la procedencia (`keys`).
10. Remodelar según la unidad de análisis: `stack`/`melt` para producir formato largo; `unstack`/`pivot` para producir formato ancho.
11. Revisar el resultado: número de filas, claves sin correspondencia, valores ausentes introducidos, duplicados y estructura de índices/columnas.

**Recomendación para competencias, derivada del capítulo.** Realizar la integración antes de separar características y objetivo, pero auditar las claves para no multiplicar ni desalinear filas. Cualquier tabla auxiliar cuyas estadísticas o contenido dependan del conjunto completo debe construirse solo con entrenamiento y aplicarse después a validación o test; el capítulo explica las operaciones tabulares, no un protocolo de validación.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Una observación se identifica mediante varios niveles | `MultiIndex` / `set_index([...])` | Representa claves compuestas sin crear una estructura de mayor dimensión | Nombrar y ordenar niveles para evitar selecciones ambiguas |
| Se necesita volver de un índice compuesto a columnas | `reset_index()` | Recupera los niveles como variables ordinarias | Puede crear columnas no deseadas si el índice carecía de significado |
| Dos tablas comparten nombres de clave | `pd.merge(..., on=...)` | Explicita la relación y evita depender de la intersección automática de nombres | Verificar unicidad y cardinalidad antes de unir |
| Las claves tienen nombres distintos | `left_on` y `right_on` | Permite enlazar columnas equivalentes con etiquetas diferentes | Confirmar que ambas columnas tienen la misma semántica |
| Solo interesan entidades presentes en ambas tablas | `how="inner"` | Conserva la intersección de claves | Puede eliminar silenciosamente entidades sin correspondencia |
| La tabla izquierda define el universo de observaciones | `how="left"` | Conserva todas sus filas | Las columnas derechas tendrán NA cuando falte coincidencia |
| Se deben conservar todas las claves de ambas tablas | `how="outer"` | Facilita detectar cobertura y discrepancias | Introduce NA y puede ampliar mucho el resultado |
| La relación usa el índice de una o ambas tablas | `merge(..., *_index=True)` o `join()` | Evita convertir necesariamente el índice a columna | `join()` usa unión izquierda por defecto |
| Hay nombres de columnas no clave repetidos | `suffixes=(...)` | Mantiene ambas versiones identificables | Sufijos genéricos pueden ocultar diferencias semánticas |
| Se apilan lotes con las mismas variables | `pd.concat(..., axis="index")` | Añade observaciones por filas | Usar `ignore_index=True` solo si el índice original no contiene información |
| Se combinan bloques de variables alineados por observación | `pd.concat(..., axis="columns")` | Añade columnas y alinea por índice | Una desalineación genera NA; decidir entre `inner` y `outer` |
| Debe conservarse la procedencia de cada bloque | `concat(..., keys=...)` | Añade un nivel jerárquico de origen | Documentar el significado de las claves |
| Se quiere impedir un nuevo eje duplicado | `verify_integrity=True` | Hace fallar la concatenación ante etiquetas duplicadas | La comprobación añade coste |
| Una fuente debe completar solo huecos de otra | `combine_first()` | Rellena NA respetando valores existentes y alineando etiquetas | La precedencia corresponde al objeto que invoca el método |
| Se necesita una fila por entidad-variable | `melt()` o `stack()` | Produce formato largo | Conservar identificadores en `id_vars` o en el índice |
| Se necesita una columna por variable | `pivot()` o `unstack()` | Produce formato ancho apto para muchas matrices analíticas | Las combinaciones índice-columna deben identificar valores de forma coherente |

## 5. Métodos y técnicas

### 5.1 `MultiIndex`, selección y niveles

- **Objetivo:** representar claves compuestas y trabajar con datos de varias dimensiones en `Series` o `DataFrame`.
- **Entradas:** arrays de etiquetas, listas de niveles, columnas existentes o un índice ya construido.
- **Salida:** objeto con varios niveles de índice en filas o columnas.
- **Funcionamiento:** la selección parcial fija niveles externos; `.loc` también permite seleccionar valores de niveles internos. `swaplevel` intercambia niveles y `sort_index(level=...)` los ordena.
- **Hiperparámetros relevantes:** nombres y orden de niveles; `level` en ordenación, agregación y remodelación; `drop` en `set_index`.
- **Ventajas:** expresa claves compuestas, facilita agregaciones por nivel y conecta directamente con `stack`/`unstack`.
- **Limitaciones:** una jerarquía sin nombres o sin orden consistente dificulta selección, unión y depuración.
- **Casos adecuados:** entidades definidas por pares como región-año, cliente-fecha o variable-medición.
- **No conviene:** cuando un índice entero simple y columnas explícitas hacen más clara la tabla de modelado.

### 5.2 `pandas.merge`

- **Objetivo:** fusionar dos tablas mediante una o varias claves relacionales.
- **Entradas:** `left`, `right`; claves en columnas o índices; tipo de unión y opciones de control.
- **Salida:** un nuevo `DataFrame` con filas emparejadas y columnas de ambos lados.
- **Funcionamiento:** para cada valor o tupla de claves forma todas las combinaciones coincidentes permitidas por `how`.
- **Hiperparámetros relevantes:** `on`, `left_on`, `right_on`, `left_index`, `right_index`, `how`, `sort`, `suffixes`, `validate`, `indicator` y `copy`.
- **Ventajas:** semántica relacional completa; soporta claves compuestas y diagnóstico de procedencia/cardinalidad.
- **Limitaciones:** las claves duplicadas pueden multiplicar filas; una unión interna puede descartar datos; una externa crea ausencias.
- **Casos adecuados:** enriquecer una tabla principal con atributos, combinar predicciones o enlazar eventos con entidades.
- **No conviene:** para apilar lotes homogéneos sin relación por clave; en ese caso corresponde `concat`.

### 5.3 `DataFrame.join`

- **Objetivo:** simplificar uniones basadas en índices.
- **Entradas:** uno o varios `DataFrame`, `how` y opcionalmente `on` para enlazar una columna del llamador con el índice del objeto derecho.
- **Salida:** tabla con columnas combinadas y alineadas por clave.
- **Funcionamiento:** une por índice y utiliza `left` como política predeterminada.
- **Hiperparámetros relevantes:** `on` y `how`.
- **Ventajas:** sintaxis breve para índices comunes o parecidos; acepta una lista de tablas.
- **Limitaciones:** el valor predeterminado difiere del `inner` predeterminado de `merge`; las columnas superpuestas requieren tratamiento.
- **Casos adecuados:** matrices de características con el mismo índice de entidad.
- **No conviene:** relaciones complejas entre varias columnas, donde `merge` comunica mejor la intención.

### 5.4 `pandas.concat`

- **Objetivo:** apilar objetos pandas a lo largo de un eje.
- **Entradas:** lista o diccionario de objetos; eje, política de alineación, claves y política del índice.
- **Salida:** `Series` o `DataFrame` concatenado.
- **Funcionamiento:** concatena en `axis="index"` por defecto y alinea el eje restante por unión (`outer`) o intersección (`inner`).
- **Hiperparámetros relevantes:** `axis`, `join`, `keys`, `levels`, `names`, `verify_integrity` e `ignore_index`.
- **Ventajas:** integra lotes homogéneos, bloques de columnas y objetos con etiquetas distintas.
- **Limitaciones:** la alineación puede introducir NA; ignorar índices puede borrar identidad; permitir duplicados puede ocultar colisiones.
- **Casos adecuados:** consolidar particiones, folds exportados, periodos o bloques de características.
- **No conviene:** cuando las filas deben emparejarse mediante una clave relacional en vez de apilarse.

### 5.5 `combine_first`

- **Objetivo:** completar valores ausentes con una segunda fuente.
- **Entradas:** dos `Series` o `DataFrame` parcialmente solapados.
- **Salida:** objeto alineado con la unión de etiquetas; conserva el valor del objeto principal cuando no es NA.
- **Funcionamiento:** aplica una selección condicional alineada por etiquetas, a diferencia de `numpy.where`, que no comprueba la alineación de índices.
- **Hiperparámetros:** no presenta hiperparámetros principales; la dirección de la llamada define la precedencia.
- **Ventajas:** parcheo conciso y seguro respecto a etiquetas.
- **Limitaciones:** no resuelve conflictos entre dos valores presentes ni valida cuál fuente es correcta.
- **Casos adecuados:** respaldo de una fuente primaria con una secundaria.
- **No conviene:** cuando hay que reconciliar discrepancias entre valores no ausentes.

### 5.6 `stack` y `unstack`

- **Objetivo:** mover niveles entre filas y columnas.
- **Entradas:** objeto con columnas o índice jerárquico y el nivel que se desea mover.
- **Salida:** `Series` o `DataFrame` remodelado.
- **Funcionamiento:** `stack` convierte columnas en un nivel de filas; `unstack` convierte un nivel del índice en columnas. El nivel interno es el predeterminado.
- **Hiperparámetros relevantes:** `level`; en la API mostrada, `dropna` para controlar si `stack` conserva combinaciones ausentes.
- **Ventajas:** remodelación reversible cuando la estructura contiene todas las combinaciones necesarias.
- **Limitaciones:** `unstack` puede introducir NA; el filtrado predeterminado de ausencias puede hacer que la forma no sea estrictamente reversible.
- **Casos adecuados:** paneles, métricas por entidad y tablas con encabezados jerárquicos.
- **No conviene:** cuando identificadores y valores están en columnas ordinarias y `melt`/`pivot` resultan más claros.

### 5.7 `pivot`

- **Objetivo:** transformar datos largos en anchos.
- **Entradas:** columna para el índice, columna para los encabezados y una columna de valores opcional.
- **Salida:** `DataFrame` cuyas filas y columnas proceden de los identificadores indicados.
- **Funcionamiento:** reorganiza valores; si se omite `values`, todas las columnas de valores producen columnas jerárquicas. El capítulo lo relaciona con `set_index(...).unstack(...)`.
- **Hiperparámetros relevantes:** `index`, `columns`, `values`.
- **Ventajas:** crea rápidamente una matriz entidad-variable o tiempo-variable.
- **Limitaciones:** requiere una relación coherente entre las combinaciones de índice, columna y valor; puede producir una estructura de columnas multinivel.
- **Casos adecuados:** crear una matriz de variables a partir de observaciones largas.
- **No conviene:** si se necesita agregar múltiples registros por celda; el capítulo no desarrolla agregación mediante tablas dinámicas.

### 5.8 `pandas.melt`

- **Objetivo:** transformar datos anchos en largos.
- **Entradas:** tabla, columnas identificadoras y subconjunto opcional de columnas de valor.
- **Salida:** tabla con identificadores, una columna `variable` y una columna `value`.
- **Funcionamiento:** repite los identificadores para cada columna medida y apila los valores.
- **Hiperparámetros relevantes:** `id_vars` y `value_vars`.
- **Ventajas:** normaliza tablas anchas y prepara datos para operaciones por variable.
- **Limitaciones:** aumenta el número de filas; si se omiten identificadores se pierde la relación explícita con la entidad original.
- **Casos adecuados:** métricas almacenadas en columnas, datos para visualización o procesamiento uniforme por variable.
- **No conviene:** cuando la forma ancha ya coincide con la unidad de observación requerida por el modelo.

## 6. Fórmulas esenciales

El capítulo no presenta desarrollo matemático. La única relación cuantitativa necesaria es la cardinalidad de una fusión muchos-a-muchos:

\[
N_{\text{clave}} = n_L \times n_R
\]

donde \(n_L\) es el número de apariciones de una clave en la tabla izquierda, \(n_R\) es su número de apariciones en la tabla derecha y \(N_{\text{clave}}\) es el número de filas coincidentes generadas para esa clave. Mide la expansión producida por el producto cartesiano; por ejemplo, tres filas a la izquierda y dos a la derecha generan seis combinaciones. Presupone que todas las apariciones comparten exactamente la clave de unión. Se interpreta como una advertencia de volumen, no como una métrica de calidad (pp. 361-363).

## 7. Implementación práctica

### 7.1 Unión auditable de una tabla principal con atributos

```python
import pandas as pd

train = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "target": [0, 1, 0],
})
attributes = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "segment": ["A", "B", "A"],
})

train_enriched = train.merge(
    attributes,
    on="customer_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)
```

`validate` e `indicator` están documentados en el capítulo. La elección de una unión izquierda representa una **síntesis práctica derivada**: conserva el universo de entrenamiento y hace visibles las claves sin atributos.

### 7.2 Concatenación de particiones homogéneas

```python
all_rows = pd.concat(
    [batch_a, batch_b],
    axis="index",
    ignore_index=True,
    verify_integrity=False,
)
```

Usar `ignore_index=True` solo cuando los índices de entrada sean etiquetas posicionales sin semántica. Si contienen identificadores, deben conservarse o trasladarse a una columna.

### 7.3 Conversión de formato largo a matriz de variables

```python
wide_features = long_features.pivot(
    index="entity_id",
    columns="feature_name",
    values="feature_value",
)
```

La matriz resultante tiene una fila por entidad y una columna por característica. Antes de usarla, debe comprobarse que cada par `(entity_id, feature_name)` corresponda a un único valor.

### 7.4 Conversión de ancho a largo

```python
long_scores = pd.melt(
    wide_scores,
    id_vars=["entity_id"],
    value_vars=["model_a", "model_b"],
    var_name="model",
    value_name="score",
)
```

**Nota práctica de API actual.** El capítulo muestra que `melt` produce por defecto las columnas `variable` y `value`. Los nombres explícitos `var_name` y `value_name` son una adaptación moderna para hacer el resultado autoexplicativo; no forman parte del ejemplo original del capítulo.

### 7.5 Separación correcta de entrenamiento y validación

```python
train_features = train_entities.merge(
    lookup_from_training,
    on="entity_id",
    how="left",
    validate="one_to_one",
)
valid_features = valid_entities.merge(
    lookup_from_training,
    on="entity_id",
    how="left",
    validate="one_to_one",
)
```

**Síntesis práctica derivada del capítulo.** Si la tabla auxiliar se aprende o agrega a partir de datos, debe construirse únicamente con el fold de entrenamiento. El capítulo no presenta particiones de ML, pero sus operaciones de unión son el punto donde puede ocurrir fuga de información.

## 8. Supuestos y condiciones de aplicación

1. Las claves de unión representan la misma entidad y tienen semántica compatible en ambas tablas.
2. La cardinalidad observada de las claves coincide con la relación esperada: uno-a-uno, uno-a-muchos, muchos-a-uno o muchos-a-muchos.
3. El orden y los nombres de los niveles de un `MultiIndex` están definidos cuando se selecciona, agrega o remodela por nivel.
4. Los índices se encuentran alineados por etiquetas para `join`, `concat` por columnas y `combine_first`.
5. Una concatenación por filas presupone que las columnas con el mismo nombre representan la misma variable.
6. `ignore_index=True` solo es válido si el índice de entrada no contiene identidad o información analítica.
7. El objeto principal de `combine_first` tiene precedencia y sus valores presentes se consideran preferibles.
8. En formato largo, las columnas identificadoras conservan la unidad de observación.
9. Para `pivot`, la combinación de identificadores usada como fila y columna debe determinar de forma coherente el valor de cada celda.
10. En datos temporales, la conversión de año y trimestre a `PeriodIndex` presupone una frecuencia trimestral y una convención temporal definida (pp. 389-391).

## 9. Errores y riesgos

- **Producto cartesiano accidental.** Duplicados a ambos lados de una clave convierten una fusión en muchos-a-muchos y pueden disparar filas y memoria.
- **Pérdida por unión interna.** El `inner` predeterminado de `merge` excluye claves que solo aparecen en un lado.
- **Confusión de valores predeterminados.** `merge` usa `inner`; `DataFrame.join` usa `left`.
- **Clave implícita equivocada.** Si no se pasa `on`, `merge` emplea la intersección de nombres de columna. Una coincidencia nominal accidental puede producir una unión incorrecta.
- **Descarte del índice.** En fusiones columna-a-columna, los índices de entrada se descartan. Si contienen identidad, deben pasarse a columnas antes de unir.
- **Columnas solapadas opacas.** Los sufijos predeterminados `_x` y `_y` no explican por sí mismos la procedencia o semántica.
- **Valores ausentes introducidos.** Uniones externas, concatenación con etiquetas desalineadas, `unstack` y remodelaciones incompletas generan NA que deben cuantificarse.
- **Índice duplicado tras concatenar.** Sin `verify_integrity`, la concatenación puede producir etiquetas repetidas en el nuevo eje.
- **Pérdida de identidad.** `ignore_index=True` o `melt` sin identificadores pueden borrar la relación con la observación original.
- **Desalineación posicional.** `numpy.where` no alinea por etiquetas; el capítulo recomienda `combine_first` cuando la alineación por índice importa.
- **Reversibilidad aparente.** `stack` filtra ausencias por defecto en la API mostrada; por ello, `unstack().stack()` puede omitir combinaciones ausentes.
- **Fuga de información (data leakage), síntesis derivada.** Unir atributos calculados con toda la muestra, agregados del objetivo o registros futuros puede contaminar validación y test aunque la operación de pandas sea técnicamente correcta.
- **Desalineación de la variable objetivo, síntesis derivada.** Concatenar o fusionar `X` y `y` por posición después de cambiar índices puede asignar objetivos a entidades equivocadas.
- **Coste computacional.** Las fusiones muchos-a-muchos, los `outer join` amplios y las conversiones a formato largo pueden aumentar sustancialmente filas, columnas y memoria.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultar este capítulo:** cuando los datos llegan en varias tablas, particiones, periodos o formatos; cuando una fila no coincide todavía con la unidad de predicción; o cuando las variables están codificadas como filas en lugar de columnas.
- **Qué decisiones permite tomar:** elegir clave y cardinalidad de unión, decidir qué universo de entidades conservar, establecer precedencia entre fuentes, alinear por índice y transformar entre largo y ancho.
- **Baseline útil:** una unión izquierda explícita desde la tabla de entrenamiento, con `validate` e `indicator`; concatenación de lotes homogéneos; `pivot` mínimo para obtener una fila por entidad.
- **Mejoras de modelo:** incorporar tablas auxiliares mediante claves auditadas, construir características por entidad-variable y conservar procedencia con `keys` cuando se combinan fuentes.
- **Qué evitar bajo presión de tiempo:** jerarquías profundas de columnas si no aportan claridad, uniones muchos-a-muchos no auditadas y remodelaciones repetidas entre largo y ancho.
- **Señales del dataset que activan estas técnicas:** múltiples CSV con una clave común; identificadores compuestos; duplicados por clave; columnas que representan periodos o métricas repetidas; tablas `id-variable-valor`; NA que aparecen después de unir; número de filas que cambia inesperadamente.
- **Control mínimo tras cada unión:** comparar filas antes/después, contar claves únicas y duplicadas, inspeccionar `_merge`, medir NA nuevos y comprobar que el objetivo sigue alineado con el identificador.

## 11. Checklist operativo

- [ ] Definir qué representa una fila en cada tabla.
- [ ] Identificar claves simples o compuestas y revisar sus tipos.
- [ ] Contar duplicados por clave en ambos lados.
- [ ] Especificar claves y `how` de forma explícita.
- [ ] Declarar la cardinalidad esperada con `validate` cuando se use `merge`.
- [ ] Activar `indicator=True` durante la auditoría de cobertura.
- [ ] Comparar número de filas y claves únicas antes y después.
- [ ] Revisar sufijos, columnas solapadas y valores ausentes nuevos.
- [ ] Conservar índices con significado; usar `ignore_index` solo si son posicionales.
- [ ] Elegir formato largo o ancho según la unidad de análisis.
- [ ] Verificar unicidad antes de `pivot` y pérdida de combinaciones tras `stack`.
- [ ] Construir tablas derivadas solo con entrenamiento y reutilizarlas en validación/test.
- [ ] Confirmar la alineación entre identificadores, características y objetivo.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Selección de claves, cardinalidad y tipos de unión | `knowledge/preprocessing/data_integration_with_pandas.md` | Alta |
| Auditoría con `validate`, `indicator`, recuentos y NA | `knowledge/preprocessing/data_integration_with_pandas.md` | Alta |
| Concatenación, alineación de ejes e integridad del índice | `knowledge/preprocessing/data_integration_with_pandas.md` | Alta |
| `stack`, `unstack`, `pivot` y `melt` | `knowledge/preprocessing/tabular_reshaping.md` | Alta |
| Índices jerárquicos, selección y agregación por nivel | `knowledge/feature_engineering/hierarchical_indexing.md` | Media |
| Formatos largo y ancho para series temporales | `knowledge/time_series/long_wide_formats.md` | Media |
| Riesgos de fuga al integrar tablas derivadas | `knowledge/validation/data_leakage.md` | Alta |
| Riesgos computacionales de productos cartesianos | `knowledge/preprocessing/data_integration_with_pandas.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito general | Introducción del capítulo 8, p. 347; conclusión, p. 397 |
| `MultiIndex`, selección parcial y nombres de niveles | 8.1, pp. 347-352 |
| Reordenación, ordenación y agregación por nivel | 8.1, pp. 352-355 |
| `set_index` y `reset_index` | 8.1, pp. 355-356 |
| Panorama de `merge`, `concat` y `combine_first` | 8.2, p. 357 |
| Uniones de DataFrames al estilo de bases de datos | 8.2, pp. 357-366; tablas 8.1 y 8.2, pp. 361 y 365-366 |
| Fusiones basadas en índices y `DataFrame.join` | 8.2, pp. 366-372 |
| Concatenación a lo largo de un eje | 8.2, pp. 373-381; tabla 8.3, pp. 380-381 |
| Combinación de datos solapados con `combine_first` | 8.2, pp. 381-383 |
| `stack` y `unstack` | 8.3, pp. 384-389 |
| Formato largo y `pivot` | 8.3, pp. 389-394 |
| Formato ancho y `melt` | 8.3, pp. 394-397 |
| Aplicación a coding tests, leakage y controles posteriores | Síntesis práctica derivada de las operaciones y riesgos descritos en pp. 357-397 |
