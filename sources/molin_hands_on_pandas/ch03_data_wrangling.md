---
title: "Data Wrangling with Pandas"
topics:
  - "data cleaning"
  - "data reshaping"
  - "data enrichment and APIs"
  - "missing values and duplicates"
  - "datetime and categorical data"
target_knowledge_files:
  - "knowledge/eda/data_quality_checks.md"
  - "knowledge/preprocessing/pandas_cleaning_and_reshaping.md"
  - "knowledge/feature_engineering/datetime_and_derived_features.md"
  - "knowledge/time_series/calendar_alignment_and_interpolation.md"
---

# 3. Data Wrangling with Pandas

## 1. Propósito del capítulo

El capítulo presenta el trabajo de convertir datos recolectados en datos confiables y analizables con pandas. Cubre recolección desde una API, limpieza de tipos, fechas, valores faltantes, infinitos y duplicados, enriquecimiento con columnas derivadas, alineación temporal y transformación entre formatos long y wide.

## 2. Conceptos esenciales

- **Data wrangling**: inspeccionar, limpiar, transformar, combinar o enriquecer datos para que representen correctamente el problema analítico. Es iterativo: una visualización o análisis puede revelar que se necesita volver a limpiar.
- **Wide vs. long**: en wide, cada variable suele ocupar una columna; en long, cada observación de una variable ocupa una fila y las columnas identifican la variable y su valor. Long facilita `groupby`, `seaborn` y operaciones por variable; wide suele ser más cómodo para lectura y algunas visualizaciones.
- **Limpieza guiada por el dominio**: los rangos imposibles, unidades, códigos especiales y reglas de negocio deben detectarse comparando los datos con lo que deberían significar, no solo con métodos automáticos.
- **Valores ausentes**: `NaN`, `None` e infinitos requieren diagnósticos distintos. No se debe imputar antes de entender por qué falta el dato.
- **Índice temporal**: para series de tiempo, un índice regular permite detectar huecos y alinear observaciones; reindexar también puede introducir faltantes que deben tratarse explícitamente.
- **Enriquecimiento**: crear columnas derivadas, por ejemplo Fahrenheit desde Celsius, indicadores booleanos o categorías, puede hacer explícita información útil para análisis posteriores.
- **Tipos semánticos**: fechas deben ser `datetime`; categorías repetidas pueden representarse como `category`; strings numéricos deben convertirse a tipos numéricos antes de calcular.

## 3. Flujo o procedimiento presentado

1. Formular qué representa cada fila, columna, unidad y rango válido.
2. Obtener datos de una fuente local o API y conservar una respuesta reproducible.
3. Inspeccionar forma, columnas, tipos, valores únicos, resúmenes y calidad general.
4. Renombrar columnas para que sean inequívocas y convertir tipos (`to_datetime`, `to_numeric`, `astype`).
5. Estandarizar fechas, zonas horarias, índice y frecuencia cuando el análisis sea temporal.
6. Buscar ausentes, infinitos, valores imposibles, duplicados y columnas inútiles.
7. Elegir, con evidencia del dominio, si se corrige, elimina, conserva o imputa cada problema.
8. Crear variables derivadas y enriquecer los datos sin perder la trazabilidad de las originales.
9. Reestructurar con `pivot`, `unstack`, `melt` o `stack` según la operación posterior.
10. Validar nuevamente tipos, rangos, cardinalidad, duplicados y coherencia temporal; documentar las decisiones.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se necesita una fila por variable y observación | Formato long; `melt` para convertir desde wide | Facilita agrupaciones y gráficos por variable | Mantener identificadores en `id_vars` |
| Se necesita una columna por variable | `pivot` o `unstack` | Hace el dataset más legible y usable por columnas | Comprobar claves únicas y faltantes creados |
| Hay índice jerárquico y se desea mover un nivel a columnas | `unstack` | Es la operación natural para MultiIndex | Revisar el orden de niveles y `fill_value` |
| Se incorporan datos con periodicidad distinta | `reindex` a un calendario/frecuencia común | Permite comparar fechas alineadas | No interpretar un hueco como cero automáticamente |
| Faltan valores por cierre de mercado | `ffill` para precios y cero para volumen, si el dominio lo justifica | Mantiene el último precio y refleja ausencia de operaciones | No propagar variables que no deban mantenerse |
| Se detectan valores extremos o imposibles | Filtrar/corregir con reglas de dominio antes de imputar | Evita que estadísticas y modelos se contaminen | No reemplazar sin identificar la causa |
| Hay varias observaciones para una clave al pivotar | Agregar primero o usar una operación explícita | `pivot` exige una combinación única | No ocultar duplicados con una agregación arbitraria |

## 5. Métodos y técnicas

### Recolección y enriquecimiento mediante API

- **Objetivo**: consultar metadatos, estaciones y observaciones meteorológicas del NCEI y enriquecer el DataFrame con datos de temperatura.
- **Entradas**: endpoint, parámetros de consulta, identificadores de dataset/localización, rango de fechas y token cuando la API lo requiera.
- **Salida**: respuesta HTTP/JSON que se normaliza a filas y columnas, en el ejemplo principalmente en formato long.
- **Funcionamiento**: construir parámetros, enviar la solicitud, comprobar `response.ok`/código de estado, explorar resultados y limitar consultas.
- **Ventajas**: datos actualizables y filtrables por fuente.
- **Limitaciones**: límites de tasa, paginación, cambios de esquema y dependencia de red.
- **Adecuado**: cuando la fuente oficial es la autoridad y se necesitan observaciones seleccionadas.
- **No conviene**: repetir consultas costosas sin caché ni registrar parámetros.
- **Adaptación moderna**: el capítulo usa una función de solicitud y ejemplos de `requests`; en código actual conviene añadir timeouts, reintentos controlados, validación de esquema y guardar la respuesta cruda. Es una recomendación práctica, no contenido original.

### Conversión de tipos y fechas

- **Objetivo**: hacer que operaciones, ordenamiento y resúmenes tengan semántica correcta.
- **Entradas/salida**: strings numéricos o de fecha -> columnas numéricas/`datetime`.
- **Funcionamiento**: `pd.to_datetime`, `pd.to_numeric`, `.dt` y conversión de zona horaria; `assign` permite derivar columnas sin mutar innecesariamente.
- **Ventajas**: permite extraer componentes, comparar fechas, convertir unidades y detectar errores.
- **Limitaciones**: fechas ambiguas, zonas horarias y errores de parsing pueden producir resultados silenciosamente incorrectos.

### Limpieza de valores faltantes, infinitos y duplicados

- **Objetivo**: aislar problemas y elegir una estrategia respaldada por el dominio.
- **Entradas**: DataFrame con `NaN`, `inf`, `-inf`, duplicados o valores fuera de rango.
- **Salida**: dataset corregido o un diagnóstico documentado cuando no es seguro corregir.
- **Funcionamiento**: `isna`, `notna`, `replace([np.inf, -np.inf], np.nan)`, `dropna`, `fillna`, `duplicated` y `drop_duplicates`.
- **Ventajas**: hace explícitas las decisiones y evita que operaciones estadísticas fallen o se distorsionen.
- **Limitaciones**: eliminar filas puede sesgar; imputar puede reducir variabilidad; `ffill`/interpolación puede inventar comportamiento.
- **Casos adecuados**: `ffill` en precios durante cierres, cero en volumen cuando el dominio lo establece, mediana para ciertos huecos cuando la distribución y el contexto lo justifican, e interpolación en series con continuidad razonable.

### Reindexación temporal

- **Objetivo**: alinear observaciones a un calendario común.
- **Entrada**: índice temporal existente y un nuevo `DatetimeIndex`/frecuencia.
- **Salida**: DataFrame con filas para todas las fechas objetivo.
- **Funcionamiento**: `set_index` y `reindex`; después, tratar los huecos según cada columna.
- **Ventajas**: comparaciones correctas entre activos o fuentes con calendarios distintos.
- **Limitaciones**: puede generar faltantes artificiales y exige distinguir días sin mercado de observaciones perdidas.

### Reshape: `pivot`, `unstack`, `melt` y `stack`

- `pivot`: wide desde columnas de índice/columnas/valores; requiere combinaciones únicas.
- `unstack`: mueve un nivel del índice jerárquico a columnas y admite `fill_value`.
- `melt`: wide a long, conservando identificadores con `id_vars`.
- `stack`: mueve columnas al índice y produce una estructura más larga, usualmente una `Series` o DataFrame con MultiIndex.
- **Riesgo común**: perder nombres de niveles, crear índices jerárquicos inesperados o confundir faltantes generados por el reshape con faltantes originales.

### Ordenamiento y categorías

`sort_values` ordena por columnas y `sort_index` por índice; una lista de claves rompe empates según su orden. `pd.Categorical` permite representar categorías repetidas y, si se define un orden, ordenar de acuerdo con la semántica del dominio.

## 6. Fórmulas esenciales

Este capítulo no requiere fórmulas esenciales para su aplicación práctica. Sí utiliza transformaciones directas de unidades, por ejemplo:

\[
T_F = 9/5 \cdot T_C + 32
\]

`T_C` es la temperatura Celsius y `T_F` la temperatura Fahrenheit. Mide la misma temperatura en otra escala; no es una métrica de modelo y supone que la unidad de entrada está correctamente identificada.

## 7. Implementación práctica

Ejemplos mínimos adaptados de las operaciones del capítulo; las adaptaciones modernas están marcadas.

```python
import numpy as np
import pandas as pd

df = pd.read_csv("weather.csv")
df = df.rename(columns={"DATE": "date", "TMAX": "temp_C"})
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df["temp_C"] = pd.to_numeric(df["temp_C"], errors="coerce")
df["temp_F"] = df["temp_C"].mul(9 / 5).add(32)
df = df.replace([np.inf, -np.inf], np.nan)
```

```python
# Diagnóstico antes de decidir una corrección.
missing = df.isna().sum().sort_values(ascending=False)
duplicates = df.duplicated(subset=["date", "station"], keep=False)
```

```python
# Long -> wide: una fila por fecha y columnas por tipo de temperatura.
wide = long_df.pivot(index="date", columns="datatype", values="temp_C")

# Wide -> long: conservar la fecha como identificador.
long = wide.reset_index().melt(
    id_vars="date", var_name="datatype", value_name="temp_C"
)
```

```python
# Adaptación práctica: separar entrenamiento y validación antes de aprender
# imputaciones o transformaciones que dependan de datos.
train = df.loc[df["date"] < "2019-01-01"].copy()
valid = df.loc[df["date"] >= "2019-01-01"].copy()
median_temp = train["temp_C"].median()
train["temp_C"] = train["temp_C"].fillna(median_temp)
valid["temp_C"] = valid["temp_C"].fillna(median_temp)
```

La separación explícita antes de calcular la mediana es una salvaguarda para ML derivada del principio de evitar contaminación de validación; el capítulo se centra en pandas y no presenta un protocolo de evaluación de modelos.

## 8. Supuestos y condiciones de aplicación

- Las columnas y sus unidades se conocen; una fecha parseada no garantiza que su zona horaria sea correcta.
- Una clave usada para `pivot` identifica como máximo una observación, salvo que se agregue previamente.
- La propagación hacia adelante solo es válida cuando el valor debe mantenerse durante el intervalo.
- La interpolación lineal supone una evolución razonablemente continua entre puntos conocidos.
- Imputar con mediana supone que esa estadística es una representación aceptable para el uso concreto y que no destruye una señal relevante.
- Un calendario temporal común debe reflejar la disponibilidad real de cada fuente.
- Las categorías ordenadas solo deben definirse cuando el orden tenga significado.
- La calidad de una API depende de identificadores, límites de fecha, paginación y respuestas exitosas.

## 9. Errores y riesgos

- Tratar `NaN` como cero o comparar directamente con `np.nan` (las comparaciones con `NaN` no funcionan como igualdad ordinaria).
- Imputar o rellenar antes de investigar si el faltante significa cierre, sensor defectuoso o ausencia real.
- Copiar el último precio a todas las variables de una fila; el volumen, por ejemplo, puede requerir cero.
- No reindexar series con calendarios diferentes y atribuir a los huecos cambios económicos inexistentes.
- Ignorar `inf` y `-inf`, que pueden sobrevivir a conversiones y romper resúmenes.
- Conservar duplicados sin decidir si son repeticiones, múltiples estaciones u observaciones válidas.
- Aceptar valores imposibles sin validación de dominio; el ejemplo detecta temperaturas absurdas y una columna de nieve dominada por infinitos.
- Crear columnas derivadas a partir de datos que todavía tienen unidades o tipos incorrectos.
- En ML, calcular imputaciones, escalados o codificaciones usando todo el dataset antes del split produce leakage; esta advertencia es una síntesis práctica derivada.
- Las consultas recursivas o masivas a una API pueden exceder límites de tasa o producir costos computacionales innecesarios.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Consultar este capítulo cuando el dataset llegue con fechas como texto, códigos especiales, columnas inconsistentes, filas duplicadas, frecuencia temporal irregular o una forma wide/long incómoda. Para un baseline, bastan inspección, conversión de tipos, detección de faltantes/infinitos/duplicados y una transformación de forma mínima. Para mejorar el modelo, pueden añadirse variables de fecha, indicadores de ausencia, alineación temporal y transformaciones justificadas por el dominio.

Bajo presión no conviene implementar un cliente API complejo, una estrategia de imputación sofisticada o una cadena extensa de reshape sin pruebas. Primero hay que preservar una copia, separar validación y comprobar la forma final. Señales como `object` donde se espera número, `datetime` inválido, valores extremos, más filas que claves esperadas, múltiples calendarios o columnas repetidas activan las técnicas del capítulo.

## 11. Checklist operativo

- [ ] Definir el significado de fila, columna, unidad y clave.
- [ ] Inspeccionar forma, tipos, rangos, valores únicos y duplicados.
- [ ] Convertir fechas y números con manejo explícito de errores.
- [ ] Identificar `NaN`, `None`, infinitos y valores imposibles por separado.
- [ ] Verificar si el dataset debe estar en formato long o wide.
- [ ] Alinear índices y calendarios antes de comparar series.
- [ ] Elegir `dropna`, `fillna`, `ffill` o interpolación según el dominio.
- [ ] Crear variables derivadas después de corregir unidades y tipos.
- [ ] Si se usará ML, aprender imputaciones solo con entrenamiento.
- [ ] Validar cardinalidad, índice, columnas, rangos y trazabilidad de cambios.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Diagnóstico de tipos, rangos, nulos, infinitos y duplicados | `knowledge/eda/data_quality_checks.md` | Alta |
| Conversión de tipos, limpieza y estrategias de faltantes | `knowledge/preprocessing/pandas_cleaning_and_reshaping.md` | Alta |
| `pivot`, `unstack`, `melt`, `stack`, long/wide | `knowledge/preprocessing/pandas_cleaning_and_reshaping.md` | Alta |
| Conversión Celsius/Fahrenheit y variables de fecha | `knowledge/feature_engineering/datetime_and_derived_features.md` | Media |
| Reindexación, frecuencia, cierres de mercado e interpolación | `knowledge/time_series/calendar_alignment_and_interpolation.md` | Alta |
| Separación de transformaciones para evitar leakage | `knowledge/validation/transformations_without_leakage.md` | Alta |
| Consulta y normalización de datos de APIs | `knowledge/eda/data_collection_from_apis.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------ | ------------------------------ |
| Propósito y conceptos esenciales | PDF pp. 133–137; páginas PDF 1–5 |
| Wide/long y reshape | PDF pp. 137–148; páginas PDF 5–16 |
| API, consulta y enriquecimiento | PDF pp. 145–158; páginas PDF 13–26 |
| Tipos, fechas, categorías y ordenamiento | PDF pp. 158–170; páginas PDF 26–38 |
| Índices, reindexación y datos financieros | PDF pp. 169–178; páginas PDF 37–46 |
| `pivot`, `unstack`, `melt` y `stack` | PDF pp. 178–189; páginas PDF 46–57 |
| Datos sucios, nulos, infinitos y duplicados | PDF pp. 189–199; páginas PDF 57–67 |
| Mitigación, imputación e interpolación | PDF pp. 197–206; páginas PDF 65–74 |
| Resumen y ejercicios | PDF pp. 206–207; páginas PDF 74–75 |

No se procesaron apéndices externos: no fueron necesarios para comprender el capítulo. La extracción textual presentó sustituciones aisladas de caracteres tipográficos, pero no se identificó una página ilegible ni un vacío que requiriera suposición.
