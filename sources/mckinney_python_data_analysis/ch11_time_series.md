---
title: "Capítulo 11 - Series temporales"
topics:
  - "series temporales"
  - "datetime y timestamps en pandas"
  - "rangos de fechas y frecuencias"
  - "remuestreo (resampling)"
  - "funciones de ventana móvil (rolling / expanding / ewm)"
  - "zonas horarias"
  - "periodos (Period / PeriodIndex)"
target_knowledge_files:
  - "knowledge/time_series/pandas_datetime_basics.md"
  - "knowledge/time_series/resampling.md"
  - "knowledge/time_series/rolling_windows.md"
  - "knowledge/time_series/timezones_and_periods.md"
  - "knowledge/feature_engineering/time_based_features.md"
---

# 11. Series temporales

## 1. Propósito del capítulo

El capítulo explica cómo representar, indexar, manipular, remuestrear y suavizar series temporales con pandas. Cubre los tipos de datos de fecha/hora de Python y pandas, la construcción de índices temporales (`DatetimeIndex`, `PeriodIndex`), el manejo de zonas horarias, la conversión entre frecuencias (`resample`) y las funciones de ventana móvil (`rolling`, `expanding`, `ewm`). Es relevante para ML porque estas operaciones son la base de cualquier feature engineering y preprocesamiento en problemas de forecasting, series financieras, datos de sensores o logs, donde el índice temporal condiciona el split de validación y la generación de variables.

## 2. Conceptos esenciales

- **Serie temporal (time series)**: cualquier dato registrado repetidamente en el tiempo. El autor distingue cuatro formas de referenciar el tiempo: marcas temporales (timestamps), periodos fijos (p. ej. un mes o un año completo), intervalos de tiempo (definidos por inicio y fin) y tiempo transcurrido/tiempo de experimento (índice entero o flotante desde un origen 0). El capítulo se centra en las tres primeras.
- **Frecuencia fija vs. irregular**: una serie de frecuencia fija tiene observaciones a intervalos regulares (cada 15 s, cada 5 min, mensual); una serie irregular no tiene unidad de tiempo fija. pandas soporta ambas, pero varias operaciones (p. ej. `shift` con `freq`, `asfreq`) requieren o asumen frecuencia conocida.
- **`datetime`, `timedelta` (módulo estándar `datetime`)**: `datetime` almacena fecha y hora hasta microsegundos; `timedelta` representa una diferencia entre dos `datetime` y se puede sumar/restar a un `datetime` para desplazarlo.
- **`pandas.Timestamp`**: escalar que reemplaza a `datetime` dentro de pandas, con precisión de nanosegundos y consciencia opcional de zona horaria; es intercambiable con `datetime` en casi todos los contextos (pero no al revés).
- **`DatetimeIndex`**: índice de pandas compuesto por objetos `Timestamp`, almacenado internamente como `datetime64[ns]`. Es el tipo de índice estándar para series temporales indexadas por marca temporal.
- **`NaT` (Not a Time)**: valor nulo equivalente a `NaN` pero específico para datos de tipo fecha/hora.
- **Alineación por fecha**: al operar aritméticamente entre dos series indexadas por fecha, pandas alinea automáticamente por las marcas temporales, generando `NaN`/`NaT` donde no hay coincidencia.
- **Índices duplicados**: una `DatetimeIndex` puede contener marcas temporales repetidas (`is_unique == False`); indexar por una marca duplicada devuelve varias filas en vez de un escalar. Se agregan con `groupby(level=0)`.
- **Desfase de fecha (`DateOffset`)**: objetos como `Day`, `Hour`, `Minute`, `MonthEnd` que representan una unidad de desplazamiento temporal; se pueden sumar entre sí y aplicar sobre `datetime`/`Timestamp` para desplazar fechas respetando reglas de calendario (p. ej. fin de mes).
- **Desfase anclado (anchored offset)**: desfases como `MonthEnd`/`BM` que no representan intervalos de igual duración, sino puntos "anclados" a una regla de calendario (último día del mes, etc.).
- **`Period` / `PeriodIndex`**: representan un lapso de tiempo completo (p. ej. el año 2011 entero), no un instante. Se construyen con una frecuencia soportada (`"A-DEC"`, `"M"`, `"Q-DEC"`, etc.). Sumar/restar enteros a un `Period` desplaza el periodo según su frecuencia.
- **Conversión de frecuencia de periodos (`asfreq`)**: cambia la granularidad de un `Period`/`PeriodIndex`, eligiendo si el subperiodo corresponde al inicio (`how="start"`) o al final (`how="end"`) del periodo original.
- **Remuestreo (`resample`)**: proceso de convertir una serie temporal de una frecuencia a otra. *Submuestreo* (downsampling) agrega de alta a baja frecuencia; *sobremuestreo* (upsampling) va de baja a alta frecuencia (requiere rellenar o interpolar, no agregar). No toda conversión de frecuencia es up/down (p. ej. de `W-WED` a `W-FRI`).
- **Ventana móvil (rolling window)**: transformación que aplica una función estadística (media, desviación, correlación, etc.) sobre una ventana deslizante de tamaño fijo (en número de periodos o en un desplazamiento de tiempo tipo `"20D"`).
- **Ventana expansiva (`expanding`)**: como `rolling`, pero la ventana crece desde el inicio de la serie hasta cubrirla completa, en vez de tener tamaño fijo.
- **Media móvil ponderada exponencialmente (`ewm`, EWMA)**: alternativa a la ventana de tamaño fijo; pondera más las observaciones recientes mediante un factor de decaimiento (comúnmente parametrizado por `span`), reaccionando más rápido a cambios recientes que una media simple.
- **Zona horaria (timezone)**: por defecto las series temporales de pandas no son "conscientes" de zona horaria (`tz` es `None`). Se pueden **localizar** (`tz_localize`, asignar una zona horaria a datos no conscientes) y **convertir** (`tz_convert`, cambiar de una zona horaria a otra manteniendo el instante UTC interno).

## 3. Flujo o procedimiento presentado

**A. Construcción y uso básico de una serie temporal**
1. Construir una lista/array de objetos `datetime` (o convertir cadenas con `pd.to_datetime`).
2. Crear una `Series`/`DataFrame` usando esas fechas como índice → pandas genera automáticamente un `DatetimeIndex`.
3. Indexar/segmentar por: un `Timestamp` exacto, una cadena de fecha (`"2011-01-10"`), un año (`"2001"`), un año-mes (`"2001-05"`), un objeto `datetime`, o un slice de fechas (`ts[fecha1:fecha2]`) — el slicing produce una vista, no una copia.
4. Usar `truncate(before=, after=)` como alternativa explícita a segmentar por fecha.

**B. Generación de rangos de fechas de frecuencia fija**
1. Definir inicio y/o fin y una frecuencia con `pd.date_range(start, end, periods=, freq=)`.
2. Si solo se da inicio o fin, especificar `periods` para fijar la longitud.
3. Opcional: `normalize=True` para forzar las marcas a medianoche.
4. Combinar frecuencias sumando desfases (`Hour(2) + Minute(30)`) o usando cadenas compuestas (`"1h30min"`, `"4H"`).

**C. Remuestreo (downsampling)**
1. Llamar a `serie.resample(regla)` para crear un objeto resampler (API similar a `groupby`).
2. Encadenar una función de agregación (`.mean()`, `.sum()`, `.ohlc()`, `.count()`, etc.).
3. Decidir `closed=` ("left"/"right": qué extremo del intervalo es inclusive) y `label=` ("left"/"right": con qué borde se etiqueta el resultado).
4. Opcional: desplazar el índice resultante sumando/restando un offset pequeño (`to_offset("-1s")`) para clarificar a qué intervalo corresponde cada etiqueta.

**D. Remuestreo (upsampling) y relleno**
1. Usar `resample(freq).asfreq()` para generar la nueva frecuencia sin agregación (introduce `NaN`).
2. Rellenar con `.ffill()` (forward fill) o limitar el relleno con `ffill(limit=n)`.
3. Tener en cuenta que el nuevo índice no tiene por qué coincidir con el original (p. ej. remuestrear semanal miércoles a semanal jueves).

**E. Remuestreo agrupado (varias series con clave)**
1. Establecer la columna de tiempo como índice (`set_index("time")`).
2. Crear un objeto `pd.Grouper(freq=...)`.
3. Agrupar por `[columna_clave, grouper_de_tiempo]` con `.groupby([...]).agg()`.
4. `reset_index()` si se necesita formato tabular plano.

**F. Funciones de ventana móvil**
1. Elegir el tamaño de ventana (`rolling(n)` en número de periodos, o `rolling("20D")` en tiempo, útil para series irregulares).
2. Definir `min_periods` si se quiere tolerar ventanas incompletas al inicio de la serie (por defecto todos los valores de la ventana deben ser no nulos).
3. Encadenar la función de agregación (`.mean()`, `.std()`, `.corr(otra_serie)`, `.apply(func_personalizada)`, `.ohlc()`).
4. Para ponderación exponencial, usar `.ewm(span=n).mean()` en lugar de `.rolling(n).mean()`.
5. Para ventana creciente desde el inicio: `.expanding().mean()`.

**G. Manejo de zonas horarias**
1. Verificar si el índice es consciente de zona horaria (`serie.index.tz`).
2. Localizar: `serie.tz_localize("America/New_York")` (o `"UTC"`).
3. Convertir entre zonas ya localizadas: `serie.tz_convert("Europe/Berlin")`.
4. Al combinar dos series con distinta zona horaria, pandas convierte automáticamente el resultado a UTC; combinar datos conscientes con no conscientes de zona horaria produce error.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
|---|---|---|---|
| Se necesita convertir texto a fechas de forma robusta y vectorizada | `pandas.to_datetime` | Analiza formatos estándar (ISO 8601) y gestiona valores ausentes como `NaT` automáticamente | `dateutil.parser` (usado internamente) puede interpretar cadenas ambiguas como fechas no deseadas (p. ej. "42" → año 2042) |
| Serie con más observaciones que las que necesita el análisis (alta frecuencia → baja frecuencia) | Submuestreo con `resample(freq).agg()` | Reduce granularidad agregando de forma controlada (media, suma, ohlc, etc.) | Definir bien `closed`/`label`; el valor por defecto no siempre es intuitivo (ver nota 2 del capítulo) |
| Serie con menos observaciones que las requeridas (baja frecuencia → alta) | Sobremuestreo con `resample(freq).asfreq()` + `ffill()` | Genera el índice de mayor frecuencia y permite decidir cómo rellenar huecos | No agregar información nueva; el relleno hacia delante puede introducir sesgo de "look-ahead" si se usa mal en un split temporal |
| Se quiere suavizar una serie ruidosa dando igual peso a cada observación dentro de una ventana | `rolling(window).mean()` (u otra estadística) | Media/estadística de ventana fija, fácil de interpretar | Al inicio de la serie faltan datos suficientes; usar `min_periods` si se acepta una ventana parcial |
| Se quiere que las observaciones recientes pesen más que las antiguas | `ewm(span=n).mean()` | Se adapta más rápido a cambios recientes que la media móvil simple | El resultado depende fuertemente del parámetro `span`; no excluye per se datos futuros, hay que evitar filtrarlos en validación temporal |
| Se necesita correlación/covarianza móvil entre dos series (p. ej. activo vs. índice) | `serie.rolling(window, min_periods=).corr(otra_serie)` | Calcula la estadística binaria sobre la ventana deslizante de forma vectorizada | Requiere ambas series alineadas en el mismo índice temporal |
| Se necesita una estadística de ventana no cubierta por los métodos integrados | `rolling(...).apply(func_personalizada)` | Permite aplicar cualquier función de reducción a la ventana | La función debe devolver un único valor por ventana; puede ser lenta si se aplica sobre ventanas grandes o series largas |
| Varias series distinguidas por una columna clave requieren el mismo remuestreo | `groupby([clave, pd.Grouper(freq=...)])` | Combina agrupación categórica y temporal en una sola operación | El índice de tiempo debe establecerse explícitamente antes de agrupar |
| Los datos combinan información de calendario repartida en columnas (año, trimestre) | Construir `PeriodIndex` a partir de arrays (`pd.PeriodIndex(year=..., quarter=..., freq=...)`) | Reconstituye una serie temporal coherente a partir de columnas separadas | La frecuencia debe ser consistente con la granularidad real de los datos |
| Se trabaja con datos financieros/de negocio con distintos años fiscales | Frecuencias trimestrales/anuales ancladas (`Q-JAN` a `Q-DEC`, `A-JAN` a `A-DEC`) | Cada alias fija el mes de cierre de año, evitando errores de alineación temporal | Sobremuestreo/submuestreo entre frecuencias periodo-a-periodo exige que una sea sub/superperiodo válido de la otra, o falla |
| Se combinan series con zonas horarias distintas | Localizar con `tz_localize` y convertir con `tz_convert` antes de operar | pandas almacena el instante en UTC internamente; combinar series con distinta zona horaria produce resultado en UTC automáticamente | Operar entre una serie consciente y otra no consciente de zona horaria lanza error; localizar cerca de transiciones de horario de verano puede producir horas ambiguas/inexistentes |

## 5. Métodos y técnicas

### `pandas.to_datetime`
- **Objetivo**: convertir cadenas de texto (o listas de cadenas) en objetos `Timestamp`/`DatetimeIndex`.
- **Entradas**: cadena o lista/array de cadenas de fecha; opcionalmente `None`/cadenas vacías para valores ausentes.
- **Salida**: `Timestamp` (escalar) o `DatetimeIndex` (colección), con `NaT` en las posiciones no interpretables.
- **Funcionamiento**: reconoce formatos estándar (ISO 8601) directamente; internamente aprovecha `dateutil` para formatos menos estrictos.
- **Ventajas**: vectorizado, gestiona nulos de forma nativa (`pd.isna`).
- **Limitaciones**: puede interpretar cadenas ambiguas como fechas (riesgo de falsos positivos de parseo).
- **Casos adecuados**: ingesta de datos con columnas de fecha en texto (típico en CSV de datathons).

### `Series.shift(periods, freq=None)`
- **Objetivo**: desplazar valores hacia adelante o atrás en el tiempo.
- **Entradas**: número de periodos (positivo o negativo); opcionalmente una frecuencia (`freq`).
- **Salida**: nueva `Series` desplazada.
- **Funcionamiento**: sin `freq`, desplaza los *valores* dejando el índice fijo (introduce `NaN` en los extremos); con `freq`, desplaza las *marcas temporales del índice* en lugar de los datos.
- **Uso típico**: calcular cambios porcentuales (`ts / ts.shift(1) - 1`), features de lag para forecasting.
- **Precaución**: mezclar el desplazamiento de índice con el de valores es una fuente de errores conceptuales; hay que decidir cuál se necesita.

### `DateOffset` (`Day`, `Hour`, `Minute`, `MonthEnd`, etc.)
- **Objetivo**: representar y aplicar desplazamientos de calendario a objetos `datetime`/`Timestamp`.
- **Entradas**: instancia con multiplicador opcional (`Hour(4)`) y un `datetime`/`Timestamp` sobre el que sumar.
- **Salida**: nuevo `Timestamp` desplazado.
- **Funcionamiento**: los desfases anclados (p. ej. `MonthEnd`) usan `rollforward`/`rollback` para adelantar/atrasar explícitamente a la siguiente/anterior fecha ancla.
- **Aplicación creativa**: combinarlos con `groupby` (`ts.groupby(MonthEnd().rollforward).mean()`) para agregación manual, aunque `resample` cumple la misma función de forma más directa y eficiente.

### `Series/DataFrame.resample(rule, ...)`
- **Objetivo**: convertir una serie temporal de una frecuencia a otra (down o upsampling).
- **Entradas**: `rule` (alias de frecuencia o `DateOffset`); argumentos clave: `closed`, `label`, `kind` ("timestamp"/"period"), `convention`, `fill_method`, `limit`, `origin`, `offset`.
- **Salida**: objeto resampler; requiere encadenar una agregación (`.mean()`, `.sum()`, `.ohlc()`, `.count()`, `.ffill()`, `.asfreq()`).
- **Funcionamiento**: agrupa los datos en intervalos ("bins") mitad-abiertos definidos por la frecuencia objetivo, análogo a `groupby`.
- **Hiperparámetros relevantes**: `closed` (qué extremo del bin es inclusive), `label` (con qué borde se etiqueta el resultado), `convention` (solo para `PeriodIndex`, decide si el subperiodo corresponde al inicio o fin).
- **Ventajas**: unifica agregación y relleno para distintas granularidades temporales en una sola API.
- **Limitaciones**: los valores por defecto de `closed`/`label` no son uniformes (para "M","A","Q","BM","BQ","W" el cierre por defecto es a la derecha; para el resto, a la izquierda), lo que puede llevar a errores de interpretación si no se revisa explícitamente.
- **Casos adecuados**: agregación de datos financieros/logs a frecuencia diaria/mensual; generación de features agregadas por periodo para modelos.

### `rolling(window, min_periods=None)`
- **Objetivo**: calcular estadísticas sobre una ventana deslizante de tamaño fijo.
- **Entradas**: `window` (entero = nº de periodos, o cadena de offset de tiempo, p. ej. `"20D"`, útil en series irregulares); `min_periods` (mínimo de observaciones no nulas requeridas para producir un resultado).
- **Salida**: objeto rolling; se encadena `.mean()`, `.std()`, `.sum()`, `.corr(otra_serie)`, `.apply(func)`, `.ohlc()`, etc.
- **Funcionamiento**: por defecto excluye automáticamente `NaN`; si no se especifica `min_periods`, requiere que la ventana esté completa (sin nulos) para producir un valor.
- **Ventajas**: suavizado interpretable, aplicable a `DataFrame` completo (aplica la transformación columna a columna).
- **Limitaciones**: introduce `NaN` al inicio de la serie mientras la ventana no está completa (salvo que se use `min_periods`); con `apply` y funciones personalizadas puede ser lento.
- **Casos adecuados**: medias móviles, desviaciones móviles, correlaciones móviles (features clásicas para series financieras o sensores).

### `expanding()`
- **Objetivo**: calcular una estadística acumulativa desde el inicio de la serie.
- **Entradas**: ninguna adicional (a diferencia de `rolling`, no requiere tamaño de ventana).
- **Salida**: objeto expanding; se encadena `.mean()`, etc.
- **Funcionamiento**: la ventana comienza en el mismo punto que `rolling` pero crece hasta cubrir toda la serie.
- **Caso adecuado**: métricas acumuladas tipo "promedio hasta la fecha" (útil como feature sin fuga de información futura, ya que solo usa el pasado hasta el punto actual).

### `ewm(span=n)`
- **Objetivo**: media/estadística móvil con ponderación exponencial (más peso a observaciones recientes).
- **Entradas**: `span` (u otros parámetros de decaimiento no detallados en profundidad en el capítulo).
- **Salida**: objeto ewm; se encadena `.mean()`.
- **Funcionamiento**: pondera exponencialmente decreciente hacia el pasado; comparado con `rolling` de tamaño similar, reacciona más rápido a cambios recientes.
- **Caso adecuado**: suavizado de series financieras que deben reaccionar rápido a cambios (EWMA); alternativa a la media móvil simple.

### `tz_localize` / `tz_convert`
- **Objetivo**: asignar (`tz_localize`) o cambiar (`tz_convert`) la zona horaria de una serie/índice/`Timestamp`.
- **Entradas**: nombre de zona horaria (cadena, p. ej. `"UTC"`, `"America/New_York"`) o un objeto de zona horaria de `pytz`.
- **Salida**: serie/índice/`Timestamp` con zona horaria asignada o convertida; el instante UTC interno no cambia al convertir.
- **Funcionamiento**: aplicable tanto a `Series`/`DataFrame` como a objetos `Timestamp` individuales o a `DatetimeIndex`.
- **Limitaciones**: la localización de marcas no conscientes puede producir horas ambiguas o inexistentes en transiciones de horario de verano/invierno; combinar series con y sin zona horaria produce error.

### `Period` / `pandas.period_range` / `asfreq`
- **Objetivo**: representar y operar con lapsos de tiempo completos (no instantes).
- **Entradas**: cadena o entero + frecuencia soportada (p. ej. `"2011"`, `freq="A-DEC"`).
- **Salida**: objeto `Period` o `PeriodIndex`.
- **Funcionamiento**: sumar/restar enteros desplaza el `Period` según su frecuencia; restar dos `Period` de igual frecuencia da un desfase (número de unidades entre ellos); `asfreq(nueva_freq, how="start"/"end")` reconvierte el periodo a otra granularidad, escogiendo el sub/superperiodo correspondiente.
- **Casos adecuados**: datos contables/financieros reportados por trimestre o año fiscal; construir `PeriodIndex` combinando columnas separadas de año y trimestre (`pd.PeriodIndex(year=..., quarter=..., freq="Q-DEC")`).
- **Conversión a/desde timestamp**: `Series.to_period()` convierte un índice de marcas temporales a periodos; `to_timestamp(how=)` hace la conversión inversa.

## 6. Fórmulas esenciales

Este capítulo no presenta fórmulas matemáticas formales (no incluye demostraciones estadísticas). El contenido es principalmente de API y manipulación de estructuras de datos. Las únicas expresiones "operativas" con valor de fórmula son:

- **Cambio porcentual (percent change) mediante `shift`**:
  `retorno_t = ts / ts.shift(1) - 1`
  Donde `ts` es la serie de precios/valores y `ts.shift(1)` es la misma serie desplazada un periodo hacia adelante (el valor del periodo anterior). Mide la variación relativa entre observaciones consecutivas. Se interpreta como retorno simple periodo a periodo; asume que el índice está ordenado cronológicamente y sin huecos relevantes para el periodo usado.

- **Diferencia entre periodos de igual frecuencia**:
  `Period(f2) - Period(f1) = n * offset_de_la_frecuencia`
  Mide cuántas unidades de la frecuencia dada separan a dos periodos; solo es válida si ambos `Period` comparten la misma frecuencia.

## 7. Implementación práctica

```python
import pandas as pd
import numpy as np

# --- 1. Construcción de índice temporal y selección ---
dates = pd.date_range("2011-01-01", periods=6, freq="D")
ts = pd.Series(np.random.default_rng(0).standard_normal(6), index=dates)

ts["2011-01"]            # selección por año-mes
ts[dates[1]:dates[3]]     # slicing por fecha (vista, no copia)

# --- 2. Downsampling: agregación mensual ---
daily = pd.Series(
    np.random.default_rng(0).standard_normal(100),
    index=pd.date_range("2000-01-01", periods=100, freq="D"),
)
monthly_mean = daily.resample("ME").mean()  # "ME" = fin de mes (alias moderno de "M")

# --- 3. Upsampling con relleno hacia adelante ---
weekly = pd.DataFrame(
    np.random.default_rng(0).standard_normal((2, 2)),
    index=pd.date_range("2000-01-01", periods=2, freq="W-WED"),
    columns=["a", "b"],
)
daily_filled = weekly.resample("D").ffill()

# --- 4. Media móvil simple vs. exponencial (features típicas de forecasting) ---
serie_precio = daily.cumsum() + 100  # serie simulada tipo "precio"
media_movil_30 = serie_precio.rolling(30, min_periods=15).mean()
media_exp_30 = serie_precio.ewm(span=30).mean()

# --- 5. Correlación móvil entre dos series ---
otra_serie = pd.Series(
    np.random.default_rng(1).standard_normal(100),
    index=daily.index,
)
correlacion_movil = daily.rolling(30, min_periods=20).corr(otra_serie)

# --- 6. Zonas horarias ---
ts_utc = ts.tz_localize("UTC")
ts_ny = ts_utc.tz_convert("America/New_York")
```

**Notas de adaptación moderna** (no forman parte literal del texto del capítulo, que usa la versión de pandas de 2022 con alias `"M"`, `"T"`, `"H"`, `"A-DEC"`):
- Los alias de frecuencia `"M"` (fin de mes), `"T"`/`"min"` (minuto), `"H"` (hora) y `"A"` (año) han sido deprecados/renombrados en versiones recientes de pandas en favor de `"ME"`, `"min"`, `"h"`, `"YE"`, respectivamente; al reutilizar el código del capítulo conviene verificar el alias vigente en la versión de pandas instalada.
- `resample("BM")` (fin de mes laborable) y offsets equivalentes también han cambiado de alias en versiones recientes; revisar la documentación de `pandas.tseries.offsets` antes de usarlos en un entorno productivo o de competición.

## 8. Supuestos y condiciones de aplicación

- Las operaciones de slicing por fecha (`ts[fecha1:fecha2]`) asumen que la serie está ordenada cronológicamente; el capítulo señala explícitamente que la mayoría de datos de series temporales cumple este supuesto y que por eso es válido "consultar por rango" incluso con marcas no presentes exactamente en el índice.
- `resample` para submuestreo requiere decidir explícitamente `closed`/`label`; el comportamiento por defecto varía según la frecuencia (ver tabla de reglas de decisión), por lo que no debe asumirse un único comportamiento universal.
- `rolling`/`expanding` requieren, por defecto, que todos los valores de la ventana sean no nulos; si los datos tienen huecos, hay que decidir explícitamente `min_periods`.
- El remuestreo up/down de `PeriodIndex` exige compatibilidad de frecuencias: en submuestreo, la frecuencia destino debe ser subperiodo de la de origen; en sobremuestreo, debe ser superperiodo. Si no se cumple, la operación puede fallar (afecta especialmente a frecuencias trimestral/anual/semanal con distintos anclajes).
- Las conversiones de zona horaria asumen datos correctamente localizados; operar entre datos conscientes y no conscientes de zona horaria no está soportado.
- El manejo de zonas horarias depende de la base de datos Olson expuesta vía `pytz`, relevante especialmente para datos históricos donde las reglas de horario de verano han cambiado.

## 9. Errores y riesgos

- **Interpretación incorrecta de `closed`/`label` en `resample`**: usar el valor por defecto sin verificarlo puede desplazar la asignación de valores a un periodo incorrecto (el propio capítulo advierte, en nota al pie, que el default no es uniforme entre frecuencias).
- **Confusión entre `shift` de valores y `shift` con `freq`**: sin `freq`, se desplazan los datos dejando el índice fijo (para calcular retornos); con `freq`, se desplazan las fechas del índice dejando los datos fijos. Usar el modo equivocado produce features de lag mal alineadas.
- **Riesgo de leakage en validación temporal con `ffill`/`bfill`**: rellenar hacia atrás (`bfill`) usaría información futura para completar el pasado; en un pipeline de forecasting esto puede introducir fuga de información si se aplica sin cuidado sobre el conjunto completo antes de separar train/validación.
- **`rolling`/`ewm` calculados sobre toda la serie antes de separar train/test**: aunque estas operaciones solo miran hacia atrás (no hacia adelante) dentro de una misma serie, si se recalculan después de mezclar información de test en el DataFrame (p. ej. tras un merge), pueden filtrar información no disponible en el momento de predicción real.
- **Parseo automático de fechas ambiguas** (`dateutil`/`pd.to_datetime`): cadenas cortas o ambiguas (ej. "42") pueden interpretarse como fechas no intencionadas; conviene validar el resultado del parseo antes de usarlo como índice.
- **Índices duplicados no detectados**: si una `DatetimeIndex` tiene marcas repetidas y no se comprueba `is_unique`, la indexación puede devolver inesperadamente una `Series` en lugar de un escalar, rompiendo código downstream.
- **Incompatibilidad de frecuencias en remuestreo de `PeriodIndex`**: intentar sobremuestrear/submuestrear entre frecuencias no relacionadas (p. ej. trimestre con año fiscal distinto al anclaje) puede fallar o dar resultados sin sentido si no se elige la frecuencia anclada correcta.
- **Mezcla de zonas horarias**: combinar series de distinta zona horaria sin localizar puede lanzar error o (si están localizadas) el resultado se normaliza automáticamente a UTC, lo cual debe ser un comportamiento esperado y no accidental.
- **Coste computacional de `rolling(...).apply(func_personalizada)`**: al no ser vectorizado, puede ser notablemente más lento que las agregaciones integradas (`mean`, `std`, `corr`), relevante si la serie es larga o la ventana es grande.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultar este capítulo**: cuando el dataset de un coding test/datathon incluye una columna de fecha/hora que debe convertirse en índice temporal, cuando se requiere agregar datos a una frecuencia distinta (por ejemplo, de eventos por segundo a totales diarios), o cuando se necesitan features de tipo "media móvil", "lag", "tendencia" o "correlación con una referencia" para un modelo de forecasting o de series temporales.
- **Decisiones que permite tomar**: 
  - Elegir si conviene submuestrear (reducir ruido/tamaño) o sobremuestrear (rellenar huecos) según la granularidad requerida por el modelo.
  - Elegir entre ventana fija (`rolling`) y ponderación exponencial (`ewm`) para suavizar o crear features de tendencia reciente.
  - Decidir si localizar/convertir zonas horarias es necesario antes de combinar fuentes de datos con distinto origen geográfico.
- **Partes útiles para un baseline rápido**:
  - `pd.to_datetime` + fijar la columna de fecha como índice.
  - `resample(freq).mean()/sum()` para generar una serie agregada lista para modelar.
  - `shift(1)` para crear un lag simple y calcular variación porcentual como feature básica.
  - `rolling(window, min_periods=1).mean()` como feature de tendencia con tolerancia a datos incompletos al inicio.
- **Partes que pueden mejorar el modelo (si hay tiempo)**:
  - Media móvil ponderada exponencialmente (`ewm`) como alternativa más reactiva a la media simple.
  - Correlación móvil con una serie de referencia (p. ej. un índice de mercado o una variable exógena) como feature adicional.
  - Construcción de `PeriodIndex` a partir de columnas año/trimestre si el dataset viene fragmentado así, para habilitar agregaciones temporales coherentes.
- **Qué no debería implementarse bajo presión de tiempo**:
  - Lógica manual de desfases anclados combinados con `groupby` (el propio capítulo indica que `resample` resuelve lo mismo de forma más simple y rápida).
  - Gestión fina de zonas horarias si el dataset ya viene en una sola zona horaria consistente (evitar complejidad innecesaria).
  - Funciones de ventana móvil personalizadas vía `.apply()` si existe un agregado integrado equivalente (más lento y más propenso a errores bajo presión).
- **Señales del dataset que activan las técnicas del capítulo**: presencia de una columna de fecha/hora como variable de ordenación natural; necesidad explícita de predecir un valor futuro (forecasting); múltiples series identificadas por una clave categórica que comparten el mismo eje temporal (uso de `pd.Grouper`); datos financieros/contables organizados por año y trimestre en columnas separadas.

## 11. Checklist operativo

- [ ] Convertir la(s) columna(s) de fecha a `datetime` con `pd.to_datetime` y validar que no haya `NaT` inesperados.
- [ ] Establecer la columna de fecha como índice (`set_index`) si se van a usar `resample`/`rolling` directamente sobre la serie/DataFrame.
- [ ] Verificar si el índice temporal es único (`index.is_unique`) y decidir cómo agregar duplicados si los hay.
- [ ] Decidir explícitamente la frecuencia objetivo y los parámetros `closed`/`label` al usar `resample` para submuestreo.
- [ ] Elegir entre `ffill`/`asfreq`/interpolación al hacer sobremuestreo, evitando rellenar con información futura (`bfill`) si se busca simular un escenario de forecasting real.
- [ ] Definir `min_periods` en `rolling`/`ewm` si se acepta trabajar con ventanas incompletas al inicio de la serie.
- [ ] Separar cualquier feature de ventana móvil/lag calculada estrictamente sobre información pasada respecto al punto de predicción (evitar leakage temporal).
- [ ] Si hay varias series por clave categórica, usar `groupby([clave, pd.Grouper(freq=...)])` en vez de bucles manuales.
- [ ] Si el dataset mezcla zonas horarias, localizar y convertir explícitamente antes de combinar o graficar series.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
|---|---|---:|
| Conceptos de `datetime`, `Timestamp`, `DatetimeIndex`, `to_datetime`, indexación por fecha | `knowledge/time_series/pandas_datetime_basics.md` | Alta |
| Generación de rangos de fechas, frecuencias y desfases (`date_range`, `DateOffset`) | `knowledge/time_series/pandas_datetime_basics.md` | Media |
| Remuestreo (`resample`, downsampling/upsampling, OHLC, remuestreo agrupado con `Grouper`) | `knowledge/time_series/resampling.md` | Alta |
| Funciones de ventana móvil (`rolling`, `expanding`, `ewm`, correlación móvil, `apply` personalizado) | `knowledge/time_series/rolling_windows.md` | Alta |
| Zonas horarias (`tz_localize`, `tz_convert`, operaciones entre zonas) | `knowledge/time_series/timezones_and_periods.md` | Media |
| `Period`, `PeriodIndex`, conversión de frecuencia de periodos, frecuencias trimestrales/fiscales | `knowledge/time_series/timezones_and_periods.md` | Media |
| Creación de features de lag/retorno porcentual (`shift`) y medias móviles como variables de modelo | `knowledge/feature_engineering/time_based_features.md` | Alta |
| Riesgos de leakage temporal al rellenar/agregar antes de separar train/validación | `knowledge/validation/time_series_validation.md` (si existe categoría equivalente) | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
|---|---|
| 2. Conceptos esenciales (tipos de series, `datetime`/`Timestamp`, `DatetimeIndex`, `NaT`, índices duplicados, `DateOffset`, `Period`) | 11.1 "Tipos de datos de fecha y hora y herramientas asociadas"; 11.2 "Fundamentos de las series temporales" (incluye subsección "Series temporales con índices duplicados") |
| 3.A Construcción y uso básico de una serie temporal | 11.2 "Fundamentos de las series temporales", subsección "Indexación, selección y creación de subconjuntos" |
| 3.B Generación de rangos de fechas | 11.3 "Rangos de fechas, frecuencias y desplazamiento", subsecciones "Generación de rangos de fecha" y "Frecuencias y desfases de fechas" |
| 2/3 Periodos y aritmética de periodos, conversión de frecuencias de periodos, frecuencias trimestrales | 11.5 "Periodos y aritmética de periodos" (incluye "Conversión de frecuencias de periodos", "Frecuencias de periodos trimestrales", "Conversión de marcas temporales a periodos (y viceversa)", "Creación de un objeto PeriodIndex a partir de arrays") |
| 3.C/3.D/3.E Remuestreo (downsampling, upsampling, OHLC, remuestreo agrupado) | 11.6 "Remuestreo y conversión de frecuencias" (incluye "Submuestreo", "Remuestreo OHLC", "Sobremuestreo e interpolación", "Remuestreo con periodos", "Remuestreo de tiempo agrupado") |
| 3.F Funciones de ventana móvil (`rolling`, `expanding`, `ewm`, funciones binarias y definidas por el usuario) | 11.7 "Funciones de ventana móvil" (incluye "Funciones ponderadas exponencialmente", "Funciones binarias de ventana móvil", "Funciones de ventana móvil definidas por el usuario") |
| 3.G / 5. Zonas horarias | 11.4 "Manipulación de zonas horarias" (incluye "Localización y conversión de zonas horarias", "Operaciones con objetos de marca temporal conscientes de la zona horaria", "Operaciones entre distintas zonas horarias") |
| 6. Fórmulas esenciales (cambio porcentual con `shift`, diferencia entre periodos) | 11.3 "Desplazamiento de los datos (adelantar y retrasar)"; 11.5 "Periodos y aritmética de periodos" |
| 9. Errores y riesgos (nota sobre `closed`/`label` por defecto) | Nota al pie 2, al final del capítulo (11.8 "Conclusión") |

**Nota de cobertura**: se revisó la totalidad del texto del capítulo tal como fue proporcionado (páginas correspondientes a las secciones 11.1 a 11.8, incluyendo tablas 11.1 a 11.5 y figuras 11.1 a 11.10). No se detectaron páginas ilegibles ni contenido truncado en el material recibido.
