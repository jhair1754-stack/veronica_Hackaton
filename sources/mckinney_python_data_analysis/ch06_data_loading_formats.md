---
title: "Carga de datos, almacenamiento y formatos de archivo"
topics:
  - "Ingesta y exportación de datos con pandas"
  - "Formatos de archivo, API web y bases de datos"
target_knowledge_files:
  - "knowledge/preprocessing/data_ingestion.md"
  - "knowledge/preprocessing/missing_values.md"
  - "knowledge/feature_engineering/datetime_parsing.md"
  - "knowledge/models/data_interfaces.md"
---

# 6. Carga de datos, almacenamiento y formatos de archivo

## 1. Propósito del capítulo

El capítulo explica cómo hacer accesibles los datos para analizarlos con pandas. Cubre la lectura y escritura de texto tabular, JSON, HTML y XML; formatos binarios como pickle, Parquet y HDF5; archivos Excel; API web; y bases de datos SQL. Su relevancia para Machine Learning está en construir una entrada de datos reproducible, interpretar correctamente tipos y ausencias, y elegir una representación que no convierta la E/S en el cuello de botella del flujo.

## 2. Conceptos esenciales

### 2.1 La carga forma parte del análisis

El autor incluye dentro de la carga tanto el acceso físico a los datos como su interpretación en tablas y tipos. Las fuentes se agrupan en archivos de texto o binarios, bases de datos y fuentes de red. En pandas, el resultado habitual es un `DataFrame`.

Las opciones de lectura de texto se concentran en cinco problemas:

- **Indexación:** decidir encabezados, nombres de columnas e índices simples o jerárquicos.
- **Tipos y conversión:** inferir tipos o transformar valores mediante funciones definidas por el usuario.
- **Fechas y horas:** reconocer columnas temporales y, si procede, combinar componentes.
- **Iteración:** limitar filas o procesar archivos grandes por fragmentos (`chunks`).
- **Datos no limpios:** omitir filas, comentarios o pies; reconocer separadores y convenciones numéricas.

CSV no incorpora un esquema de tipos. Por ello, la inferencia puede requerir comprobación y ajustes. HDF5, ORC y Parquet sí almacenan información de tipos.

### 2.2 Configuración explícita de texto tabular

`pandas.read_csv` es la interfaz principal presentada. Sus controles prácticos incluyen:

- `header=None` cuando el archivo no contiene encabezado;
- `names=[...]` para asignar nombres;
- `index_col` para usar una o varias columnas como índice;
- `sep` o `delimiter`, incluida una expresión regular como `r"\s+"`;
- `skiprows` y `comment` para ignorar contenido ajeno a la tabla;
- `nrows` para una lectura exploratoria limitada;
- `encoding`, `thousands` y `decimal` para interpretar el texto;
- `parse_dates` y `dayfirst` para datos temporales;
- `converters` para transformaciones por columna;
- `engine` para elegir entre los motores `c`, `python` y `pyarrow`.

El capítulo indica que el motor `c` es el predeterminado, `python` es más lento pero admite funciones no soportadas por otros motores y `pyarrow` puede ser considerablemente más rápido.

### 2.3 Valores ausentes

pandas reconoce por defecto distintas cadenas centinela, como `NA` y `NULL`, además de campos vacíos, y representa los faltantes como `NaN`. `na_values` amplía o define los marcadores; puede ser una secuencia global o un diccionario por columna. `keep_default_na=False` desactiva la lista predeterminada.

**Conclusión práctica derivada del contenido:** una cadena como `NA` puede ser una categoría legítima. Antes de entrenar, conviene declarar la semántica de faltantes por columna y comprobarla con `isna()`; aceptar silenciosamente los valores predeterminados puede alterar las categorías y los tipos.

### 2.4 Procesamiento por fragmentos

`nrows` permite probar la configuración sin cargar todo el archivo. `chunksize` devuelve un `TextFileReader` iterable y `get_chunk` permite solicitar fragmentos de tamaño arbitrario. El ejemplo del capítulo agrega recuentos de categorías por fragmento mediante `Series.add(..., fill_value=0)`.

Esta estrategia limita la memoria, pero obliga a que el cálculo pueda combinar resultados parciales. El capítulo advierte también que transponer un archivo completo con `zip(*values)` puede consumir mucha memoria.

### 2.5 Escritura y round trip

`DataFrame.to_csv` permite configurar delimitador, representación de faltantes (`na_rep`), inclusión de índice y encabezado, selección y orden de columnas. El módulo estándar `csv` ofrece `reader`, `writer` y dialectos para delimitadores de un carácter, comillas, terminadores y escapes. El análisis manual queda reservado para casos que `read_csv` no resuelva, como estructuras especialmente irregulares o delimitadores complejos.

### 2.6 Datos semiestructurados

- **JSON:** `json.loads` convierte texto JSON a objetos Python y `json.dumps` realiza la conversión inversa. Una lista de diccionarios puede convertirse en `DataFrame`; `read_json` interpreta directamente disposiciones compatibles y `to_json(orient="records")` genera registros por fila. La adaptación de JSON anidado a tabla depende de la estructura concreta.
- **HTML:** `read_html` localiza las etiquetas `<table>` y devuelve una lista de `DataFrame`; hay que elegir la tabla adecuada.
- **XML:** `lxml.objectify` permite recorrer nodos, filtrar etiquetas y construir registros; `read_xml` simplifica el caso tabular. Para XML complejo, la selección de nodos y la tabla objetivo deben declararse.

### 2.7 Formatos binarios y hojas de cálculo

- **pickle:** serializa objetos Python y puede leerse con `read_pickle`. El autor lo recomienda solo para almacenamiento a corto plazo porque su estabilidad entre versiones no está garantizada y su portabilidad fuera de Python es limitada.
- **Parquet y ORC:** pandas ofrece soporte integrado; Parquet se presenta como alternativa apropiada para almacenamiento distribuido y datos remotos.
- **Excel:** `ExcelFile` resulta conveniente y más eficiente al leer varias hojas; `read_excel` basta para una lectura directa. `index_col` evita importar como dato una columna que representa el índice. `to_excel` exporta tablas.
- **HDF5:** almacena múltiples conjuntos de datos y metadatos, soporta compresión y acceso eficiente a secciones. `HDFStore` ofrece una interfaz tipo diccionario. El esquema `fixed` es el predeterminado; `table` suele ser más lento, pero admite consultas con `select` y `where`. Es adecuado para grandes datos locales de una escritura y muchas lecturas, no sustituye a una base de datos y puede corromperse con escrituras simultáneas.

### 2.8 API web y SQL

Para una API, el flujo presentado es `requests.get`, `raise_for_status`, `Response.json` y selección de campos al construir el `DataFrame`. La respuesta puede ser una lista o un diccionario y, si la fuente es dinámica, los resultados varían entre ejecuciones.

En SQL, los controladores suelen devolver tuplas y metadatos de columnas mediante `cursor.description`. pandas evita esa conversión manual con `read_sql` sobre una conexión o motor SQLAlchemy. La elección de una base de datos depende de rendimiento, integridad y escalabilidad.

## 3. Flujo o procedimiento presentado

1. Identificar la fuente: texto tabular, JSON/XML/HTML, binario, Excel, API o SQL.
2. Inspeccionar una muestra o limitar la lectura con `nrows` antes de procesar el conjunto completo.
3. Configurar estructura: encabezado, nombres, delimitador, columnas e índice.
4. Configurar interpretación: codificación, convenciones numéricas, fechas, conversores y marcadores de ausencia.
5. Cargar en un `DataFrame` con la función específica del formato.
6. Verificar forma, columnas, tipos, faltantes y una muestra de filas.
7. Si el archivo es grande, iterar con `chunksize` y combinar resultados parciales.
8. En JSON/XML/HTML, seleccionar expresamente los registros, nodos o tablas que constituyen la tabla analítica.
9. En API, validar el estado HTTP antes de interpretar JSON; en SQL, preferir `read_sql` para preservar nombres de columnas sin ensamblaje manual.
10. Elegir el formato de persistencia según portabilidad, volumen, patrón de acceso y entorno local o distribuido.
11. Al exportar, declarar índice, encabezado, orden de columnas y representación de faltantes; después comprobar que una relectura conserva la estructura esperada.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Texto tabular delimitado | `pd.read_csv` con opciones explícitas | Resuelve la mayoría de variantes y carga directamente un `DataFrame` | CSV no incorpora tipos; revisar inferencia, nulos y separadores |
| Campos separados por espacios variables | `read_csv(..., sep=r"\s+")` | Una expresión regular representa el separador variable | Una columna extra puede ser inferida como índice; verificar columnas |
| Archivo grande o configuración aún incierta | Probar con `nrows` y luego usar `chunksize` | Reduce tiempo de prueba y memoria | El cálculo debe poder agregarse por fragmentos |
| CSV de un solo carácter con dialecto especial | `csv.reader`/`csv.writer` o parámetros de dialecto | Controla comillas, escapes y terminadores | El procesamiento manual puede consumir mucha memoria y complicar tipos |
| JSON tabular simple | `pd.read_json` | Convierte registros compatibles directamente | JSON anidado exige seleccionar o transformar campos |
| Página HTML con tablas | `pd.read_html` | Devuelve todas las tablas como una lista | Elegir la tabla correcta y validar fechas y tipos después |
| XML tabular | `pd.read_xml` | Simplifica el recorrido manual | XML complejo requiere selección o filtrado de nodos |
| Varias hojas del mismo libro Excel | `pd.ExcelFile` y `parse` | Evita reabrir repetidamente el archivo | Declarar hoja e índice; vigilar columnas `Unnamed` |
| Persistencia Python temporal | `to_pickle`/`read_pickle` | Conserva objetos pandas de forma directa | No usar como formato estable de largo plazo ni interoperable |
| Grandes arrays locales, una escritura y muchas lecturas | HDF5/HDFStore | Compresión y lectura de subconjuntos eficiente | No es base de datos; evitar escrituras simultáneas |
| Datos remotos o almacenamiento distribuido | Parquet | Diseñado para ese entorno y conserva tipos | Requiere motor compatible, como `pyarrow` |
| API JSON | `requests.get` + `raise_for_status()` + `.json()` | Separa validación HTTP de interpretación | La respuesta es dinámica y su estructura debe comprobarse |
| Consulta relacional | `pd.read_sql` con SQLAlchemy | Abstrae diferencias de controladores y conserva columnas | Rendimiento, integridad y escalabilidad dependen del sistema elegido |

## 5. Métodos y técnicas

### 5.1 `pandas.read_csv`

- **Objetivo:** convertir texto tabular delimitado en `DataFrame`.
- **Entradas:** ruta, URL u objeto de archivo; configuración de estructura, tipos, fechas, nulos y motor.
- **Salida:** `DataFrame` o `TextFileReader` si se usa iteración por fragmentos.
- **Funcionamiento:** separa campos, determina encabezados e índice, reconoce ausencias y convierte valores.
- **Hiperparámetros operativos:** `sep`, `header`, `names`, `index_col`, `skiprows`, `na_values`, `keep_default_na`, `parse_dates`, `converters`, `nrows`, `chunksize`, `encoding`, `thousands`, `decimal`, `engine`.
- **Ventajas:** interfaz flexible para la mayoría de textos tabulares.
- **Limitaciones:** inferencia falible; archivos mal formados pueden necesitar ajustes o análisis manual.
- **Casos adecuados:** CSV, delimitadores alternativos y texto separado por espacios.
- **No conviene:** estructuras jerárquicas o formatos que ya ofrecen almacenamiento tipado y acceso selectivo más eficiente.

### 5.2 Lectura por fragmentos

- **Objetivo:** procesar archivos que no conviene cargar completos.
- **Entrada:** archivo y `chunksize` en filas.
- **Salida:** secuencia de `DataFrame` parciales y un resultado agregado por el usuario.
- **Funcionamiento:** `TextFileReader` itera por bloques; cada resultado parcial se combina.
- **Parámetro relevante:** tamaño del fragmento.
- **Ventajas:** controla el uso de memoria y acelera la validación inicial.
- **Limitaciones:** no todos los algoritmos se expresan como agregaciones parciales; hay que mantener consistencia entre bloques.
- **Casos adecuados:** recuentos, sumas y estadísticas combinables.
- **No conviene:** operaciones que requieren simultáneamente todas las filas, salvo que se rediseñen.

### 5.3 JSON, HTML y XML

- **Objetivo:** convertir datos semiestructurados o tablas web en estructuras analizables.
- **Entradas:** texto/archivo JSON, documento HTML o XML.
- **Salida:** objetos Python, lista de `DataFrame` o `DataFrame`.
- **Funcionamiento:** JSON mapea objetos y arrays; HTML extrae `<table>`; XML recorre o selecciona nodos.
- **Parámetros relevantes:** orientación JSON y selección/filtro de nodos XML.
- **Ventajas:** acceso directo desde formatos web comunes.
- **Limitaciones:** el anidamiento no define por sí solo una tabla; la selección depende del dominio.
- **Casos adecuados:** registros JSON homogéneos, tablas HTML y XML con elementos repetidos.
- **No conviene:** asumir que cualquier jerarquía compleja se aplana correctamente sin reglas explícitas.

### 5.4 HDF5 con `HDFStore`

- **Objetivo:** persistir grandes objetos pandas y recuperar subconjuntos.
- **Entradas:** clave, `Series` o `DataFrame`, esquema `fixed` o `table` y condición `where` cuando corresponda.
- **Salida:** objeto pandas completo o selección filtrada.
- **Funcionamiento:** un archivo almacena varios conjuntos bajo claves; `table` permite consultas.
- **Hiperparámetros operativos:** `format` y condición `where`.
- **Ventajas:** compresión, acceso parcial y buen rendimiento local de E/S.
- **Limitaciones:** dependencia de PyTables; `table` es más lento; riesgo con escritura concurrente.
- **Casos adecuados:** grandes datos locales, una escritura y muchas lecturas.
- **No conviene:** almacenamiento distribuido, concurrencia de escritura o uso como base de datos.

### 5.5 API con `requests`

- **Objetivo:** recuperar un feed web y convertir los campos relevantes en tabla.
- **Entrada:** URL de API.
- **Salida:** objeto Python derivado de JSON y después `DataFrame`.
- **Funcionamiento:** petición GET, validación HTTP, análisis JSON y selección de columnas.
- **Parámetros relevantes:** el capítulo no desarrolla parámetros adicionales de la petición.
- **Ventajas:** flujo pequeño y directo.
- **Limitaciones:** datos variables y posible anidamiento.
- **Casos adecuados:** API que devuelve registros JSON.
- **No conviene:** ignorar el estado HTTP o asumir un esquema sin inspeccionarlo.

### 5.6 `pandas.read_sql`

- **Objetivo:** cargar resultados de una consulta SQL en un `DataFrame`.
- **Entradas:** consulta y conexión/motor SQLAlchemy.
- **Salida:** `DataFrame` con nombres de columnas.
- **Funcionamiento:** encapsula la ejecución y el ensamblaje manual de tuplas y metadatos.
- **Hiperparámetros:** el capítulo no presenta parámetros de ajuste.
- **Ventajas:** interfaz uniforme sobre distintos sistemas SQL.
- **Limitaciones:** el rendimiento y la escalabilidad siguen dependiendo de la base y de la consulta.
- **Casos adecuados:** extracción tabular desde bases relacionales.
- **No conviene:** traer datos sin filtrar cuando la selección puede realizarse en SQL.

## 6. Fórmulas esenciales

Este capítulo no requiere fórmulas esenciales para su aplicación práctica.

## 7. Implementación práctica

### 7.1 Carga robusta de CSV

```python
import pandas as pd

datos = pd.read_csv(
    "train.csv",
    index_col="id",
    na_values={"ingresos": ["NA", "NULL"], "segmento": ["desconocido"]},
    keep_default_na=False,
    parse_dates=["fecha_evento"],
    encoding="utf-8",
)

print(datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
```

El ejemplo combina mecanismos explicados en el capítulo. La comprobación explícita evita que una decisión silenciosa de lectura llegue al modelado.

### 7.2 Agregación por fragmentos

```python
import pandas as pd

recuentos = pd.Series(dtype="int64")
for fragmento in pd.read_csv("eventos.csv", chunksize=100_000):
    parcial = fragmento["categoria"].value_counts()
    recuentos = recuentos.add(parcial, fill_value=0)

recuentos = recuentos.sort_values(ascending=False).astype("int64")
```

Este patrón reproduce la idea del capítulo: calcular por bloque y combinar sin cargar el archivo completo.

### 7.3 API JSON a tabla

```python
import pandas as pd
import requests

respuesta = requests.get("https://api.example.com/registros")
respuesta.raise_for_status()
registros = respuesta.json()

datos_api = pd.DataFrame(registros, columns=["id", "fecha", "valor"])
```

La URL es ilustrativa; el procedimiento respaldado por el capítulo es validar la respuesta, analizar JSON y seleccionar campos.

### 7.4 Lectura SQL

```python
import pandas as pd
import sqlalchemy as sa

motor = sa.create_engine("sqlite:///datos.sqlite")
datos_sql = pd.read_sql(
    "SELECT id, fecha, objetivo FROM observaciones",
    motor,
)
```

### 7.5 Nota práctica de actualización de API

El capítulo muestra `writer.save()` al exportar Excel. En código actual conviene administrar el escritor con un contexto, sin cambiar la idea original de `ExcelWriter` + `to_excel`:

```python
import pandas as pd

with pd.ExcelWriter("salida.xlsx") as writer:
    datos.to_excel(writer, sheet_name="datos", index=False)
```

Esta es una adaptación moderna, no una instrucción textual del capítulo. Los ejemplos no incluyen separación entrenamiento/validación porque el capítulo se limita a E/S y no ajusta modelos. **Síntesis práctica derivada:** cualquier inferencia de esquema, vocabulario de nulos o transformación aprendida a partir de datos debe definirse sin consultar el conjunto de evaluación oculto.

## 8. Supuestos y condiciones de aplicación

1. El archivo tabular tiene una estructura de filas y campos que el lector puede separar con un delimitador o regla conocida.
2. La codificación, el separador decimal y el separador de miles son conocidos o verificables.
3. Los marcadores de ausencia tienen significado consistente, posiblemente específico por columna.
4. La inferencia automática de tipos se valida antes del análisis.
5. La lectura por fragmentos exige una operación cuyos resultados parciales se puedan combinar correctamente.
6. `read_html` supone que la información objetivo está en etiquetas `<table>` y devuelve todas las candidatas.
7. La conversión de JSON o XML a tabla exige identificar qué objetos o nodos representan registros y qué campos representan columnas.
8. pickle supone un entorno Python compatible y un horizonte corto de almacenamiento.
9. HDF5 es especialmente apropiado para datos locales, acceso parcial y muchas lecturas; no para escrituras concurrentes.
10. Las llamadas a API pueden devolver datos diferentes con el tiempo y deben responder sin error HTTP.
11. `read_sql` requiere una conexión válida y una consulta cuya salida sea tabular.

## 9. Errores y riesgos

- Tratar la primera fila de datos como encabezado o importar el encabezado como una fila.
- Confundir una columna de identificador con una característica o, al contrario, perderla al usarla como índice.
- Aceptar la inferencia especial de índice cuando hay menos nombres de columna que campos.
- Usar el delimitador, la codificación o las convenciones numéricas equivocadas.
- Convertir cadenas legítimas como `NA` en faltantes por aceptar la lista predeterminada sin revisarla.
- No reconocer centinelas específicos de una columna y dejar “faltantes” como categorías ordinarias.
- Confiar en la inferencia de tipos de CSV sin comprobar `dtypes` y nulos.
- Leer un archivo completo solo para probar la configuración; usar `nrows` reduce ese coste.
- Elegir un `chunksize` sin diseñar cómo se combinarán los resultados parciales.
- Materializar todas las filas y transponerlas con `zip(*values)` en archivos grandes.
- Tomar la primera tabla de `read_html` sin confirmar que es la correcta.
- Aplanar JSON o XML anidado sin una definición explícita del registro analítico.
- Usar pickle como contrato estable o interoperable de largo plazo.
- Elegir HDF5 para almacenamiento distribuido o permitir varias escrituras simultáneas.
- Omitir `raise_for_status()` y procesar como datos una respuesta HTTP fallida.
- Suponer que una API en tiempo real es reproducible sin guardar la extracción.
- Recuperar columnas innecesarias de SQL y trasladar trabajo que podría resolver la consulta.
- **Data leakage, síntesis práctica derivada:** inferir reglas de limpieza o categorías combinando entrenamiento con evaluación puede transmitir información. El capítulo no discute validación de modelos, pero sus decisiones de carga deben fijarse antes de aplicar el mismo esquema a cada partición.
- **Overfitting/underfitting:** el capítulo no trata ajuste de modelos; el riesgo directo está en alterar la representación de los datos antes del modelado.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** cuando los archivos no cargan limpiamente, aparecen columnas `object` inesperadas, faltantes incoherentes, fechas como texto, memoria insuficiente o fuentes JSON/SQL.
- **Decisiones que permite tomar:** seleccionar el lector, declarar separador/encabezado/índice/nulos/fechas, decidir entre carga completa y fragmentada y elegir un formato de persistencia.
- **Baseline:** cargar con opciones explícitas; comprobar `shape`, columnas, `dtypes`, nulos y primeras filas; separar identificador y objetivo; conservar una extracción reproducible.
- **Mejoras posibles:** lectores por fragmentos, `pyarrow` para CSV cuando sea compatible, Parquet para datos tipados o distribuidos, HDF5 para acceso local parcial y filtrado en SQL antes de transferir.
- **Qué evitar bajo presión:** un parser manual completo, aplanamiento genérico de XML/JSON complejo, conversión masiva a HDF5 sin necesidad o ingeniería de una capa de API de alto nivel.
- **Señales activadoras:** errores de tokenización, número irregular de campos, filas de comentarios, `Unnamed: 0`, categorías `NA` desaparecidas, memoria agotada, múltiples hojas, tablas web, nodos repetidos en XML, respuesta JSON o credenciales/conexión SQL.

No debe atribuirse una mejora de métrica a un formato por sí mismo: estas técnicas mejoran corrección, reproducibilidad y rendimiento de la preparación, no el modelo directamente.

## 11. Checklist operativo

**Síntesis práctica derivada del capítulo.**

- [ ] Identificar la fuente y el formato real, no solo la extensión.
- [ ] Leer una muestra con `nrows` y revisar encabezado, delimitador e índice.
- [ ] Declarar codificación y convenciones numéricas cuando sean relevantes.
- [ ] Definir los marcadores de ausencia globales o por columna.
- [ ] Comprobar `shape`, nombres, `dtypes`, duplicados de índice y `isna().sum()`.
- [ ] Interpretar fechas y confirmar su resultado.
- [ ] Usar `chunksize` si el volumen no cabe cómodamente en memoria.
- [ ] Seleccionar de forma explícita tablas HTML, campos JSON o nodos XML.
- [ ] Validar el estado HTTP antes de procesar una API.
- [ ] Filtrar columnas y filas en SQL cuando sea posible.
- [ ] Evitar pickle para persistencia duradera y HDF5 para escritura concurrente.
- [ ] Exportar con índice, encabezado, columnas y nulos definidos.
- [ ] Releer una muestra del archivo exportado y comparar estructura y tipos.
- [ ] Aplicar el mismo contrato de carga a entrenamiento y evaluación sin aprender reglas de esta última.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Contrato de ingestión tabular y validación de esquema | `knowledge/preprocessing/data_ingestion.md` | Alta |
| Marcadores de ausencia predeterminados y por columna | `knowledge/preprocessing/missing_values.md` | Alta |
| Lectura de fechas y convenciones ambiguas | `knowledge/feature_engineering/datetime_parsing.md` | Alta |
| Procesamiento de archivos grandes por fragmentos | `knowledge/preprocessing/out_of_core_processing.md` | Alta |
| Selección entre CSV, Parquet, pickle y HDF5 | `knowledge/preprocessing/storage_formats.md` | Alta |
| Ingesta de JSON, HTML y XML | `knowledge/preprocessing/semistructured_data.md` | Media |
| Extracción desde API y SQL hacia `DataFrame` | `knowledge/models/data_interfaces.md` | Media |
| Controles de leakage en la fase de carga | `knowledge/validation/data_leakage.md` | Media |

## 13. Trazabilidad

Las páginas se indican como numeración impresa del libro; entre paréntesis figura la página dentro del PDF suministrado.

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura del capítulo 6, p. 247 (PDF 1); conclusión 6.5, p. 281 (PDF 35) |
| 2.1 La carga forma parte del análisis | 6.1 y tabla 6.1, pp. 247–249 (PDF 1–3) |
| 2.2 Configuración explícita de texto tabular | 6.1, ejemplos `read_csv` y tabla 6.2, pp. 249–256 (PDF 3–10) |
| 2.3 Valores ausentes | 6.1, ejemplos `ex5.csv`, pp. 252–255 (PDF 6–9) |
| 2.4 Procesamiento por fragmentos | “Leer archivos de texto por partes”, pp. 256–258 (PDF 10–12) |
| 2.5 Escritura y round trip | “Escribir datos en formato de texto” y “Trabajar con otros formatos delimitados”, pp. 258–262 (PDF 12–16) |
| 2.6 Datos semiestructurados | “Datos JSON” y “XML y HTML: raspado web”, pp. 262–271 (PDF 16–25) |
| 2.7 Formatos binarios y hojas de cálculo | 6.2, pp. 271–276 (PDF 25–30) |
| 2.8 API web y SQL | 6.3–6.4, pp. 276–281 (PDF 30–35) |
| 3. Flujo o procedimiento presentado | Síntesis de 6.1–6.4, pp. 247–281 (PDF 1–35) |
| 4. Reglas de decisión | Comparaciones y advertencias de 6.1–6.4, especialmente pp. 256, 271–276 y 279–281 (PDF 10, 25–30, 33–35) |
| 5. Métodos y técnicas | 6.1–6.4, pp. 247–281 (PDF 1–35) |
| 6. Fórmulas esenciales | El capítulo no presenta fórmulas matemáticas, pp. 247–281 (PDF 1–35) |
| 7. Implementación práctica | Ejemplos de `read_csv`, fragmentos, API, SQL y Excel, pp. 249–281 (PDF 3–35); nota moderna identificada como adaptación |
| 8. Supuestos y condiciones | Condiciones y advertencias de 6.1–6.4, pp. 249–281 (PDF 3–35) |
| 9. Errores y riesgos | Advertencias sobre memoria, pickle, HDF5, API e inferencia, pp. 255–280 (PDF 9–34); leakage marcado como síntesis derivada |
| 10. Aplicación en un coding test | Síntesis práctica derivada de 6.1–6.4, pp. 247–281 (PDF 1–35) |
| 11. Checklist operativo | Síntesis práctica derivada de 6.1–6.4, pp. 247–281 (PDF 1–35) |
| 12. Mapeo hacia `knowledge/` | Clasificación documental derivada del contenido completo del capítulo, pp. 247–281 (PDF 1–35) |

No se detectaron páginas ilegibles ni secciones incompletas en el PDF suministrado.
