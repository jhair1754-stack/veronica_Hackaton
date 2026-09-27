---
title: "Limpieza y preparación de los datos"
topics:
  - "Valores ausentes y tipos anulables"
  - "Transformación, discretización y valores atípicos"
  - "Limpieza de texto y expresiones regulares"
  - "Datos categóricos y variables indicadoras"
target_knowledge_files:
  - "knowledge/preprocessing/missing_values.md"
  - "knowledge/preprocessing/data_cleaning_with_pandas.md"
  - "knowledge/feature_engineering/categorical_encoding.md"
  - "knowledge/feature_engineering/discretization.md"
  - "knowledge/feature_engineering/text_cleaning.md"
---

# 7. Limpieza y preparación de los datos

## 1. Propósito del capítulo

**Contenido explicado por el autor.** El capítulo presenta herramientas de pandas y Python para convertir datos sin procesar en estructuras utilizables para análisis y modelado: detectar y tratar valores ausentes, eliminar duplicados, reemplazar y mapear valores, discretizar variables, detectar valores atípicos, muestrear filas, crear variables indicadoras, usar tipos anulables, limpiar texto y trabajar con datos categóricos.

**Conclusión práctica derivada.** Para Machine Learning, estas operaciones forman la capa de preparación previa al ajuste del modelo. Su valor no está en aplicar todas las transformaciones, sino en elegir una política reproducible para cada problema de calidad de datos y conservar la semántica de las variables.

## 2. Conceptos esenciales

- **Datos ausentes (NA).** pandas representa la ausencia mediante valores centinela. En datos `float64` suele usar `np.nan`; `None` también se reconoce como ausente. Los tipos de extensión anulables usan `pd.NA`. `isna()` detecta ausencias y `notna()` devuelve su negación. Las estadísticas descriptivas de pandas omiten los NA de forma predeterminada (pp. 282-285).
- **Filtrado frente a imputación.** `dropna()` elimina observaciones o columnas según la presencia de NA; `fillna()` conserva la estructura y rellena los huecos con constantes, valores por columna, propagación hacia delante o atrás, o una estadística como la media (pp. 285-291).
- **Duplicado.** `duplicated()` identifica una fila que repite los valores de una fila previa. `drop_duplicates()` la elimina. El subconjunto de columnas y la elección de conservar la primera o la última aparición determinan la identidad del duplicado (pp. 292-294).
- **Transformación por mapeo.** `Series.map()` aplica una función o diccionario elemento a elemento. `replace()` sustituye valores concretos, incluidos centinelas no estándar, por uno o varios valores nuevos (pp. 294-298).
- **Discretización.** `pd.cut()` asigna valores continuos a intervalos definidos por bordes o por igual amplitud. `pd.qcut()` usa cuantiles y produce grupos con cantidades aproximadamente iguales de observaciones (pp. 300-304).
- **Valor atípico (outlier).** El capítulo muestra detección mediante umbrales sobre el valor absoluto y dos respuestas operativas: filtrar filas o limitar los valores al umbral conservando el signo (pp. 304-306). No presenta ese umbral como regla universal.
- **Muestreo.** Una permutación reordena observaciones. `sample()` extrae una muestra sin reemplazo de forma predeterminada y con reemplazo si `replace=True` (pp. 306-309).
- **Variable indicadora (dummy/one-hot).** `pd.get_dummies()` crea una columna binaria por categoría. `Series.str.get_dummies()` resuelve membresías múltiples codificadas en una cadena delimitada (pp. 309-313, 345-346).
- **Tipos de extensión.** pandas incorpora tipos como `Int64`, `boolean`, `string`, `category` y tipos temporales para representar valores ausentes sin forzar tipos inadecuados y, en ciertos casos, reducir memoria o mejorar rendimiento (pp. 313-318).
- **Limpieza vectorizada de texto.** El accesor `Series.str` aplica operaciones de cadena y expresiones regulares a una serie, propagando los NA. `str.extract()` devuelve los grupos capturados como columnas; `str.contains()`, `str.findall()`, `str.split()` y otras operaciones evitan bucles explícitos (pp. 318-331).
- **Representación categórica.** `Categorical` almacena un conjunto de categorías y códigos enteros. Puede reducir sustancialmente la memoria y acelerar operaciones como `groupby` o `value_counts()` cuando hay pocos valores distintos repetidos muchas veces (pp. 331-345).
- **Categoría ordenada.** El orden solo tiene significado si se declara explícitamente mediante `ordered=True` o `as_ordered()`. Los códigos son una representación interna y no sustituyen la semántica de las etiquetas (pp. 336-338).

## 3. Flujo o procedimiento presentado

1. Inspeccionar los datos ausentes con `isna()` y considerar si la ausencia puede revelar problemas de recogida o sesgos.
2. Decidir si eliminar filas/columnas (`dropna`) o rellenar (`fillna`), especificando eje, umbral o método cuando corresponda.
3. Detectar duplicados con `duplicated()` y eliminarlos con una clave explícita en `subset` cuando la igualdad de todas las columnas no represente la identidad real.
4. Normalizar valores mediante `map()` y `replace()`: traducir categorías, reemplazar centinelas por NA y unificar representaciones.
5. Normalizar etiquetas de filas o columnas con `rename()` o con una función de transformación.
6. Cuando sea útil agrupar una variable continua, elegir entre intervalos definidos (`cut`) y cuantiles (`qcut`), revisar límites e inclusividad y asignar etiquetas legibles.
7. Detectar valores extremos con una máscara booleana; decidir si se conservan, filtran o limitan. El ejemplo del capítulo usa `abs() > 3` en datos normales simulados.
8. Reordenar o muestrear observaciones con una permutación o `sample()`, indicando reemplazo si se requiere.
9. Convertir categorías a indicadores con `get_dummies()`; para categorías múltiples en una cadena, separar mediante `str.get_dummies(delimitador)`.
10. Convertir columnas a tipos de extensión adecuados con `astype()` para preservar enteros, booleanos, texto o categorías con valores ausentes.
11. Limpiar texto con métodos `str` y usar expresiones regulares compiladas si el mismo patrón se aplica repetidamente.
12. Para columnas repetitivas y de baja cardinalidad, evaluar `category`, declarar el orden si existe y eliminar categorías no usadas tras filtrar.

**Recomendación para competencias, derivada del capítulo.** Ajustar decisiones dependientes de los datos -estadísticas de imputación, bordes de cuantiles, vocabularios y categorías- solo con entrenamiento, y reutilizarlas sin recalcular en validación o test. El capítulo muestra las operaciones de pandas, pero no desarrolla un pipeline de validación; esta precaución es una consecuencia necesaria para evitar fuga de información.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Solo deben desaparecer filas totalmente vacías | `dropna(how="all")` | Conserva filas parcialmente informativas | El valor predeterminado elimina cualquier fila con al menos un NA |
| Se exige un mínimo de observaciones presentes | `dropna(thresh=n)` | Expresa un umbral de datos no nulos | Elegir `n` según las columnas relevantes |
| Cada columna requiere un relleno distinto | `fillna({columna: valor})` | Permite política específica por variable | El valor debe ser compatible con la semántica y el tipo |
| Los huecos representan continuidad secuencial | Relleno hacia delante/atrás | Propaga una observación vecina | El orden debe tener significado; limitar la longitud propagada cuando proceda |
| Hay códigos centinela como `-999` | `replace(..., np.nan)` | Convierte ausencia no estándar en NA reconocido | Verificar que el valor no sea una observación válida |
| Un diccionario define una traducción uno a uno | `Series.map(diccionario)` | Transformación elemento a elemento concisa | Claves no cubiertas pueden producir NA |
| Los intervalos tienen significado de dominio | `pd.cut()` con bordes explícitos | Mantiene límites interpretables | Revisar `right`, los extremos y las etiquetas |
| Se quieren grupos de tamaño parecido | `pd.qcut()` | Usa cuantiles de muestra | Los bordes dependen de la distribución observada |
| Se necesitan localizar filas con algún extremo | `(df.abs() > umbral).any(axis="columns")` | Combina máscaras por columnas | El umbral del ejemplo no es universal |
| Se necesita muestra sin repetición | `DataFrame.sample(n=...)` | Es la conducta predeterminada | El capítulo no fija semilla en este ejemplo |
| Se necesita muestra con repetición | `sample(..., replace=True)` | Permite seleccionar una fila varias veces | Cambia la distribución de frecuencias observada |
| Una columna tiene una categoría por fila | `pd.get_dummies()` | Produce indicadores binarios | Puede aumentar mucho la dimensionalidad |
| Una celda contiene varias categorías delimitadas | `Series.str.get_dummies(sep)` | Crea indicadores multietiqueta | El autor advierte que no es rápido para datos muy grandes |
| Texto con NA requiere limpieza vectorizada | `Series.str` | Omite o propaga los valores ausentes | Preferir `dtype="string"` para semántica anulable consistente |
| Hay muchas repeticiones y baja cardinalidad | `astype("category")` | Puede reducir memoria y acelerar agrupaciones | La conversión tiene un coste inicial; declarar el orden solo si existe |

## 5. Métodos y técnicas

### 5.1 Ausencia: `isna`, `dropna` y `fillna`

- **Objetivo:** detectar, eliminar o rellenar valores ausentes.
- **Entradas:** `Series` o `DataFrame`; eje, política (`how`), umbral (`thresh`), valor o método de relleno y límite opcional.
- **Salida:** máscara booleana o nuevo objeto filtrado/rellenado; las funciones no modifican el original por defecto.
- **Funcionamiento:** `dropna()` filtra etiquetas según NA; `fillna()` sustituye huecos; `ffill` y `bfill` propagan valores adyacentes.
- **Hiperparámetros relevantes:** `axis`, `how`, `thresh`, `value`, dirección de relleno y `limit`.
- **Ventajas:** API directa, vectorizada y consistente entre tipos.
- **Limitaciones:** eliminar puede descartar información; rellenar puede alterar la distribución. El capítulo no propone una estrategia universal.
- **Casos adecuados:** limpieza de tablas, series ordenadas y conversión de centinelas.
- **No conviene:** aplicar propagación cuando el orden no tiene significado o imputar sin examinar el patrón de ausencia.

### 5.2 Duplicados, mapeo, reemplazo y renombrado

- **Objetivo:** eliminar registros repetidos y normalizar valores o etiquetas.
- **Entradas:** columnas clave (`subset`), política `keep`, diccionario/función de mapeo, pares de reemplazo o mapeos de etiquetas.
- **Salida:** máscara, objeto sin duplicados o valores/etiquetas transformados.
- **Funcionamiento:** `duplicated`/`drop_duplicates` comparan combinaciones de columnas; `map` transforma elementos; `replace` sustituye valores; `rename` cambia etiquetas sin reconstruir manualmente el objeto.
- **Hiperparámetros relevantes:** `subset`, `keep`, diccionario de valores, `index` y `columns`.
- **Ventajas:** operaciones declarativas y reproducibles.
- **Limitaciones:** la deduplicación es correcta solo si la clave representa realmente una entidad; un mapeo incompleto puede introducir NA.
- **Casos adecuados:** identificadores repetidos, categorías heterogéneas, centinelas y nombres inconsistentes.
- **No conviene:** eliminar registros únicamente porque comparten algunas características no identificadoras.

### 5.3 Discretización: `cut` y `qcut`

- **Objetivo:** convertir una variable continua en una variable categórica por intervalos.
- **Entradas:** valores, bordes o número de intervalos, etiquetas, lado cerrado y precisión; en `qcut`, número o lista de cuantiles.
- **Salida:** `Categorical` con intervalos/etiquetas y códigos.
- **Funcionamiento:** `cut` divide por límites o amplitudes; `qcut` calcula bordes a partir de cuantiles de muestra.
- **Hiperparámetros relevantes:** `bins`/`q`, `labels`, `right`, `precision`.
- **Ventajas:** facilita análisis por grupos y puede alimentar variables indicadoras.
- **Limitaciones:** discretizar pierde resolución; `cut` no equilibra recuentos y `qcut` produce bordes dependientes de la muestra.
- **Casos adecuados:** rangos de dominio, análisis de cuantiles e informes agrupados.
- **No conviene:** cuando la señal depende de variaciones finas o cuando los bordes aprendidos no pueden estabilizarse.

### 5.4 Valores atípicos

- **Objetivo:** localizar observaciones que exceden un umbral y, si procede, filtrarlas o acotarlas.
- **Entradas:** columnas numéricas y umbral absoluto.
- **Salida:** máscara/fila seleccionada o datos limitados.
- **Funcionamiento:** `abs()` crea magnitudes; comparaciones producen máscaras; `any(axis="columns")` agrega por fila; `np.sign()` conserva el signo al acotar.
- **Hiperparámetro:** umbral, mostrado como 3 en el ejemplo simulado.
- **Ventajas:** enfoque vectorizado y transparente.
- **Limitaciones:** el criterio es un ejemplo operativo, no un detector general; puede borrar extremos válidos.
- **Casos adecuados:** revisión rápida de columnas numéricas con escala conocida.
- **No conviene:** tratar automáticamente como error cualquier extremo sin contexto.

### 5.5 Permutación y muestreo

- **Objetivo:** reordenar filas/columnas o extraer subconjuntos aleatorios.
- **Entradas:** tamaño del eje, `n`, eje y `replace`.
- **Salida:** objeto reordenado o muestra.
- **Funcionamiento:** `np.random.permutation` genera posiciones; `take`/`iloc` las aplica; `sample` selecciona observaciones.
- **Ventajas:** implementación breve para aleatorización y muestreo.
- **Limitaciones:** una muestra aleatoria simple no garantiza estratificación.
- **Casos adecuados:** inspección, prototipos y muestreo con o sin reemplazo.
- **No conviene:** sustituir una estrategia de validación que deba respetar clases, grupos o tiempo.

### 5.6 Variables indicadoras

- **Objetivo:** transformar categorías en columnas binarias para modelado.
- **Entradas:** serie categórica o de texto, prefijo y, para multietiqueta, delimitador.
- **Salida:** `DataFrame` con una columna por categoría.
- **Funcionamiento:** `get_dummies` codifica una categoría por fila; `str.get_dummies` separa y codifica múltiples membresías.
- **Ventajas:** salida directamente utilizable por numerosos modelos.
- **Limitaciones:** expansión dimensional; el enfoque multietiqueta del capítulo es lento para conjuntos muy grandes.
- **Casos adecuados:** variables nominales de cardinalidad controlada y multietiquetas delimitadas.
- **No conviene:** cardinalidad extremadamente alta sin una estrategia adicional.

### 5.7 Tipos de extensión y `Categorical`

- **Objetivo:** conservar tipos anulables y representar eficientemente valores repetidos.
- **Entradas:** datos y tipo de destino en `astype`, categorías/códigos y orden opcional.
- **Salida:** series `Int64`, `boolean`, `string`, `category` u otros tipos de extensión.
- **Funcionamiento:** los tipos anulables usan `pd.NA`; `Categorical` separa categorías de códigos enteros.
- **Hiperparámetros relevantes:** dtype, lista de categorías y `ordered`.
- **Ventajas:** evita convertir enteros con NA a `float64`; puede reducir memoria y acelerar agrupaciones.
- **Limitaciones:** convertir tiene coste; una categoría ordenada mal declarada introduce una semántica falsa.
- **Casos adecuados:** enteros/booleanos con ausencias, texto repetitivo y baja cardinalidad.
- **No conviene:** usar códigos internos como magnitudes numéricas para un modelo nominal.

### 5.8 Cadenas y expresiones regulares

- **Objetivo:** dividir, normalizar, buscar, reemplazar y extraer patrones de texto.
- **Entradas:** cadenas/series, delimitadores, patrón regex, banderas y grupos de captura.
- **Salida:** cadenas, listas, máscaras o un `DataFrame` de grupos extraídos.
- **Funcionamiento:** los métodos internos sirven para tareas simples; `re` ofrece `split`, `findall`, `search`, `match` y `sub`; `Series.str` vectoriza estas operaciones y maneja NA.
- **Hiperparámetros relevantes:** patrón, flags como `re.IGNORECASE`, delimitador y grupos.
- **Ventajas:** extracción estructurada y vectorizada; `re.compile()` ahorra trabajo cuando un patrón se reutiliza.
- **Limitaciones:** `match` solo prueba el inicio; `search` busca en cualquier posición; patrones deficientes producen coincidencias incorrectas.
- **Casos adecuados:** correos, tokens, campos delimitados y normalización de texto estructurado.
- **No conviene:** usar regex complejo sin validar sus coincidencias y casos negativos.

## 6. Fórmulas esenciales

Este capítulo no requiere fórmulas esenciales para su aplicación práctica.

## 7. Implementación práctica

El siguiente ejemplo reúne solo operaciones mostradas por el capítulo. La separación entre entrenamiento y validación y la reutilización de estadísticas/categorías se marcan como **adaptación moderna y recomendación para competencias**, no como código original del libro.

```python
import numpy as np
import pandas as pd

# Supuesto: train_df y valid_df ya fueron separados antes de aprender parámetros.
numeric_columns = ["age", "income"]

# Adaptación moderna: estadísticas calculadas únicamente con entrenamiento.
train_medians = train_df[numeric_columns].median()
train_df[numeric_columns] = train_df[numeric_columns].fillna(train_medians)
valid_df[numeric_columns] = valid_df[numeric_columns].fillna(train_medians)

# Contenido del capítulo: convertir centinelas y normalizar texto.
for frame in (train_df, valid_df):
    frame["score"] = frame["score"].replace([-999, -1000], np.nan)
    frame["city"] = frame["city"].astype("string").str.strip().str.lower()

# Adaptación moderna: fijar categorías desde entrenamiento y alinear columnas.
train_encoded = pd.get_dummies(train_df, columns=["city"], dtype="int8")
valid_encoded = pd.get_dummies(valid_df, columns=["city"], dtype="int8")
valid_encoded = valid_encoded.reindex(columns=train_encoded.columns, fill_value=0)
```

Ejemplo mínimo de discretización con bordes de dominio:

```python
age_edges = [18, 25, 35, 60, 100]
age_labels = ["youth", "young_adult", "middle_aged", "senior"]

train_df["age_group"] = pd.cut(
    train_df["age"], bins=age_edges, labels=age_labels, right=True
)
valid_df["age_group"] = pd.cut(
    valid_df["age"], bins=age_edges, labels=age_labels, right=True
)
```

Ejemplo mínimo de extracción de texto:

```python
import re

email_pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,})"
email_parts = data["email"].astype("string").str.extract(
    email_pattern, flags=re.IGNORECASE
)
email_parts.columns = ["user", "domain", "suffix"]
```

**Nota práctica de API actual.** El capítulo usa `fillna(method="ffill")`; en pandas actual puede expresarse de manera directa como `df.ffill(limit=...)` o `df.bfill(limit=...)`. También usa la función superior `pd.value_counts`; la forma habitual actual es `serie.value_counts()`. Estas son actualizaciones de API, no afirmaciones originales del capítulo.

## 8. Supuestos y condiciones de aplicación

- Los valores marcados como NA o centinelas representan realmente ausencia y no valores válidos del dominio.
- Eliminar filas o columnas es aceptable solo cuando la pérdida de información no invalida el análisis.
- `ffill` y `bfill` requieren un orden significativo; la observación vecina debe ser una aproximación defendible.
- La deduplicación necesita una definición correcta de identidad mediante todas las columnas o un `subset` explícito.
- `cut` supone que los bordes elegidos tienen sentido; la inclusión derecha/izquierda debe coincidir con la definición del intervalo.
- `qcut` depende de cuantiles de la muestra y busca tamaños de grupo aproximadamente iguales.
- El umbral de outliers debe tener una escala interpretable; el valor 3 del capítulo pertenece a un ejemplo con datos normales simulados.
- El muestreo aleatorio simple no conserva necesariamente la proporción de clases o grupos.
- One-hot encoding es manejable cuando el número de categorías no genera una matriz excesivamente ancha.
- `Categorical` resulta especialmente ventajoso cuando existe repetición y baja cardinalidad.
- Una variable categórica solo debe marcarse como ordenada cuando el orden sea real y conocido.
- Un patrón regex debe validarse contra ejemplos positivos y negativos; el patrón de correo del capítulo identifica “la mayoría” de direcciones, no todas las permitidas por una especificación completa.
- El tipo `string` basado en ciertas implementaciones puede requerir `pyarrow`, según la configuración indicada por el capítulo.

## 9. Errores y riesgos

- **Fuga de información (conclusión práctica derivada):** calcular medias, medianas, cuantiles, límites o vocabularios con todo el dataset antes de separar validación.
- **Pérdida silenciosa:** usar `dropna()` sin recordar que, por defecto, elimina cualquier fila con al menos un valor ausente.
- **Imputación sin semántica:** rellenar con cero cuando cero significa una medición real, o propagar valores en datos no secuenciales.
- **Duplicados mal definidos:** deduplicar por columnas que no forman una clave o conservar arbitrariamente la primera/última aparición.
- **Mapeo incompleto:** aplicar un diccionario que no contiene todas las claves y convertir valores no reconocidos en NA.
- **Intervalos ambiguos:** ignorar qué extremo está abierto/cerrado en `cut` y clasificar mal valores frontera.
- **Outliers válidos eliminados:** confundir observaciones raras con errores; el capítulo no justifica una regla global de tres unidades.
- **Validación incorrecta:** usar una muestra aleatoria en datos temporales o agrupados, aunque el capítulo solo la presenta como operación de muestreo.
- **No reproducibilidad:** omitir semilla en procesos aleatorios. El capítulo fija una semilla en ejemplos concretos, pero no desarrolla una política de reproducibilidad.
- **Explosión dimensional:** crear miles de dummies para variables de alta cardinalidad.
- **Desalineación train/test (conclusión práctica derivada):** crear dummies por separado y obtener columnas diferentes; hay que fijar/alinear el vocabulario.
- **Ordinalidad falsa:** entregar códigos de `Categorical` a un modelo como si las distancias entre códigos fueran significativas.
- **Categorías obsoletas:** filtrar datos y conservar niveles no usados, lo que puede afectar recuentos; `remove_unused_categories()` los elimina.
- **Coste computacional:** `str.get_dummies()` para multietiquetas no es especialmente rápido en grandes volúmenes; el autor sugiere escribir directamente en un array NumPy.
- **Regex incorrecta:** confundir `match` con `search`, omitir raw strings o escribir grupos que capturen componentes equivocados.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** al recibir tablas con NA, centinelas, duplicados, texto sucio, categorías repetidas, columnas continuas que podrían agruparse o posibles extremos.
- **Decisiones que permite tomar:** eliminar frente a rellenar; `cut` frente a `qcut`; categoría nativa frente a dummies; métodos de texto simples frente a regex; muestreo con o sin reemplazo.
- **Baseline:** convertir centinelas, cuantificar NA, imputar con una estadística de entrenamiento, limpiar texto básico, deduplicar con clave defendible y aplicar one-hot de cardinalidad controlada.
- **Mejoras posibles:** tipos anulables, `category` para memoria/rendimiento, bins de dominio o cuantiles, extracción regex de campos estructurados y tratamiento explícito de extremos.
- **Qué evitar bajo presión:** regex sofisticadas sin tests, discretización sin hipótesis, recodificación multietiqueta masiva y reglas automáticas de outliers no justificadas.
- **Señales activadoras:** `object` con pocos valores repetidos sugiere `category`; códigos como `-999` sugieren `replace`; porcentajes altos de NA exigen política explícita; delimitadores dentro de una celda sugieren multietiqueta; colas extremas exigen inspección; cardinalidad alta desaconseja dummies indiscriminadas.
- **Prioridad de competencia:** asegurar primero un flujo consistente entre entrenamiento y validación; optimizar memoria o ingeniería fina después de tener un baseline evaluable.

## 11. Checklist operativo

- [ ] Identificar NA nativos y posibles centinelas de dominio.
- [ ] Medir ausencias antes de decidir entre eliminación e imputación.
- [ ] Separar entrenamiento y validación antes de aprender estadísticas o categorías.
- [ ] Definir la clave de duplicado y la política `keep`.
- [ ] Normalizar valores y etiquetas sin introducir NA inesperados.
- [ ] Revisar bordes e inclusividad si se usa `cut` o `qcut`.
- [ ] Justificar cualquier regla de filtrado o recorte de outliers.
- [ ] Fijar semilla cuando la aleatoriedad deba ser reproducible.
- [ ] Comprobar cardinalidad antes de crear variables indicadoras.
- [ ] Alinear columnas codificadas entre entrenamiento y validación.
- [ ] Validar regex con casos positivos, negativos y valores ausentes.
- [ ] Usar tipos anulables o categóricos cuando preserven semántica o memoria.
- [ ] Eliminar categorías no usadas después de filtrar si afectan los resultados.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Detección, filtrado y relleno de valores ausentes | `knowledge/preprocessing/missing_values.md` | Alta |
| Duplicados, centinelas, mapeos y renombrado | `knowledge/preprocessing/data_cleaning_with_pandas.md` | Alta |
| Tipos anulables y conversiones con `astype` | `knowledge/preprocessing/pandas_nullable_dtypes.md` | Media |
| Discretización por bordes y cuantiles | `knowledge/feature_engineering/discretization.md` | Alta |
| Detección vectorizada y recorte de valores atípicos | `knowledge/preprocessing/outlier_handling.md` | Media |
| One-hot y categorías multietiqueta | `knowledge/feature_engineering/categorical_encoding.md` | Alta |
| Limpieza vectorizada de texto y regex | `knowledge/feature_engineering/text_cleaning.md` | Alta |
| Categorías, orden, memoria y rendimiento | `knowledge/preprocessing/categorical_dtypes.md` | Media |
| Riesgo de aprender transformaciones antes del split | `knowledge/validation/data_leakage.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Introducción, pp. 282-283; conclusión, p. 346 |
| 2. Datos ausentes | 7.1, pp. 282-291 |
| 2. Duplicados, mapeo, reemplazo y renombrado | 7.2, pp. 291-300 |
| 2. Discretización y outliers | 7.2, pp. 300-306 |
| 2. Muestreo y variables indicadoras | 7.2, pp. 306-313 |
| 2. Tipos de extensión | 7.3, pp. 313-318 |
| 2. Texto y expresiones regulares | 7.4, pp. 318-331 |
| 2. Datos categóricos | 7.5, pp. 331-346 |
| 3. Flujo operativo | Síntesis de 7.1-7.5, pp. 282-346 |
| 4. Reglas de decisión | `dropna`/`fillna`, pp. 285-291; `cut`/`qcut`, pp. 300-304; muestreo y dummies, pp. 306-313; texto y categorías, pp. 318-345 |
| 5. Métodos y técnicas | 7.1-7.5, pp. 282-346 |
| 6. Fórmulas esenciales | El capítulo no presenta fórmulas necesarias, pp. 282-346 |
| 7. Implementación práctica | Ejemplos del capítulo, pp. 285-291, 297-298, 300-304, 309-313 y 326-330; adaptaciones modernas identificadas en el propio resumen |
| 8. Supuestos y condiciones | Derivados de ejemplos y advertencias, pp. 283-291, 300-306, 312-318, 322-330 y 336-345 |
| 9. Errores y riesgos | Advertencias explícitas, pp. 283, 287, 312, 322-326, 337 y 341-345; riesgos de validación marcados como derivados |
| 10. Aplicación en un coding test | Síntesis práctica derivada del conjunto del capítulo, pp. 282-346 |
| 11. Checklist operativo | Síntesis ejecutable derivada del conjunto del capítulo, pp. 282-346 |
| 12. Mapeo hacia `knowledge/` | Correspondencia temática derivada de 7.1-7.5, pp. 282-346 |
