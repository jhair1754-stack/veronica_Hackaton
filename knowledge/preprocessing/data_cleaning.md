---
title: "Limpieza de datos para Machine Learning"
knowledge_area: "preprocessing"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Limpieza de datos para Machine Learning

## 1. Propósito y alcance

La limpieza corrige o prepara problemas de calidad identificados antes del modelado, procurando conservar información útil y no introducir errores. No todo dataset necesita todas las técnicas: cada intervención debe responder a un hallazgo, a la semántica de la variable y al uso posterior. [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) y [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) presentan alternativas, no una receta universal.

- **Auditoría:** detecta, cuantifica y documenta problemas. Un hallazgo no determina automáticamente su tratamiento.
- **Limpieza:** aplica una decisión justificada, como convertir, unificar, imputar, filtrar o conservar explícitamente un caso.
- **Transformación:** cambia la representación para análisis o modelado, por ejemplo mediante codificación o escalamiento. La creación de nuevas señales corresponde principalmente a *feature engineering*.

**Síntesis integrada:** limpiar es convertir hallazgos de auditoría en decisiones reproducibles, verificables y compatibles con la validación; no consiste en hacer desaparecer toda irregularidad.

## 2. Principios para limpiar datos

1. **Entender antes de modificar.** Hay que conocer qué representa cada fila, columna, unidad, código y rango. Un extremo puede ser error o caso válido; un faltante puede tener significado. [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)
2. **Separar detección de tratamiento.** `isna()`, `duplicated()` o una regla de rango identifican candidatos; no justifican por sí mismos eliminar, imputar o recortar. [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md)
3. **Evitar pérdidas innecesarias.** Eliminar puede sesgar; imputar puede reducir variabilidad; propagar o interpolar puede inventar comportamiento. [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)
4. **Aprender solo de entrenamiento.** Medianas, escalas, cuantiles, bins y vocabularios se ajustan con entrenamiento y se reutilizan sin reajuste. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md)
5. **Aplicar el mismo flujo.** Validación, prueba e inferencia reciben la misma secuencia y los parámetros aprendidos en entrenamiento. [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)
6. **Comprobar el resultado.** Después de limpiar se revisan otra vez tipos, rangos, cardinalidad, faltantes, duplicados y coherencia. [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)
7. **Conservar trazabilidad.** Operaciones declarativas y pipelines registran orden y parámetros y facilitan reproducir el tratamiento. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)

**Conclusión práctica derivada:** para cada cambio conviene registrar hallazgo, hipótesis, tratamiento, parámetros aprendidos y comprobación posterior.

## 3. Flujo general de limpieza

1. Revisar el objetivo y los hallazgos de auditoría sin modificar aún los datos.
2. Confirmar unidad de observación, claves, significado, tipo semántico, unidades y rangos esperados.
3. Separar la evaluación final antes de explorar en detalle o aprender transformaciones; explorar y decidir con entrenamiento. [Géron, cap. 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)
4. Clasificar cada hallazgo: ausencia, duplicación, tipo/formato, categoría, escala, valor atípico o transformación requerida.
5. Elegir entre conservar, corregir, convertir, imputar, codificar, escalar, filtrar o posponer; registrar supuesto y riesgo.
6. Aplicar primero correcciones semánticas que no aprendan de la distribución, como unidades confirmadas o centinelas validados.
7. Ajustar con entrenamiento los tratamientos que aprenden parámetros; aplicar esos mismos objetos a validación o prueba.
8. Encadenar en un pipeline las transformaciones dependientes de datos que participen en validación cruzada.
9. Comparar antes y después: forma, tipos, faltantes, duplicados, rangos, cardinalidad y columnas resultantes.
10. Guardar decisiones, limitaciones y casos no resueltos.

Los pasos 4, 5, 9 y 10 son **síntesis práctica derivada de las fuentes**.

## 4. Tipos de problemas y tratamiento

### 4.1 Valores faltantes

`isna()`/`notna()` identifican ausencias reconocidas. Códigos como `-999`, infinitos o cadenas especiales solo deben convertirse a NA tras confirmar que no son válidos. Las fuentes presentan eliminación, relleno por constante o estadística, propagación y, con continuidad defendible, interpolación; ninguna opción es automática. [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md); [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)

| Problema observado | Preguntas antes de actuar | Tratamiento respaldado por las fuentes | Riesgo o precaución |
|---|---|---|---|
| NA en variable útil | ¿Por qué falta? ¿Qué significa? ¿Eliminar pierde casos relevantes? | Conservar, `dropna()` condicionado o `fillna()` por columna/estadística | Eliminar puede sesgar; imputar altera distribución |
| Fila totalmente vacía | ¿Es registro sin información o fallo de ingesta? | `dropna(how="all")` | No confundir con una fila parcialmente informativa |
| Mínimo de datos presentes | ¿Qué columnas son indispensables? | `dropna(thresh=n)` | Justificar `n` según columnas relevantes |
| Centinela o infinito | ¿El código puede ser legítimo? | `replace(..., np.nan)` tras validación | Una sustitución incorrecta destruye datos válidos |
| Hueco secuencial | ¿El orden importa? ¿El valor debe mantenerse? | `ffill`, `bfill` o interpolación si el dominio lo permite | Puede inventar continuidad; limitar propagación |
| Imputación numérica baseline | ¿La estadística representa la variable? | Mediana aprendida con entrenamiento | No capta relaciones; no usar validación/prueba para calcularla |

La decisión puede depender del tipo y significado de la variable, el contexto de ausencia y el método posterior. Las fuentes no ofrecen un diagnóstico completo de mecanismos estadísticos de ausencia.

### 4.2 Duplicados

`duplicated()` localiza repeticiones y `drop_duplicates()` aplica una política. La identidad depende de todas las columnas o de un `subset` explícito. Antes de eliminar hay que distinguir registros redundantes de múltiples mediciones, eventos válidos o entidades que comparten atributos. [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md); [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)

| Problema observado | Preguntas antes de actuar | Tratamiento respaldado por las fuentes | Riesgo o precaución |
|---|---|---|---|
| Filas idénticas | ¿La unidad de observación admite repeticiones? | Inspeccionar con `duplicated(keep=False)`; eliminar solo si son redundantes | Dos eventos reales pueden tener los mismos valores |
| Clave repetida | ¿Identifica entidad o evento? ¿Hay tiempo? | Definir `subset` y `keep` explícitos | Conservar primero/último arbitrariamente puede perder información |
| Varias filas impiden `pivot` | ¿Son duplicados o varias observaciones válidas? | Agregar solo con operación justificada | Una agregación arbitraria oculta el problema |

### 4.3 Tipos y formatos inconsistentes

Los tipos deben reflejar la semántica: strings numéricos se convierten antes de calcular; fechas a `datetime`; categorías repetidas pueden usar `category`; enteros o booleanos con ausencia pueden usar tipos anulables. `to_numeric`, `to_datetime` y `astype` explicitan la conversión. Fechas ambiguas, zonas horarias y errores de *parsing* pueden producir resultados plausibles pero incorrectos. [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md); [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md)

| Problema observado | Preguntas antes de actuar | Tratamiento respaldado por las fuentes | Riesgo o precaución |
|---|---|---|---|
| Número almacenado como texto | ¿Separadores, símbolos o centinelas tienen significado? | `pd.to_numeric(..., errors="coerce")` tras normalizar | `coerce` crea NA; revisar fallos |
| Fecha como texto | ¿Formato, zona y granularidad son conocidos? | `pd.to_datetime()` y zona horaria explícita | Parsear no garantiza semántica correcta |
| Unidades mezcladas | ¿Cuál es la unidad de cada registro? | Convertir con relación confirmada | Derivar antes de corregir propaga errores |
| Etiquetas inconsistentes | ¿Son realmente la misma categoría? | Métodos `str`, `map()` o `replace()` | Un mapeo incompleto crea NA o fusiones incorrectas |
| Enteros/booleanos con NA | ¿Debe preservarse el tipo? | `Int64`, `boolean` u otro tipo anulable | Verificar compatibilidad posterior |

La conversión se verifica comparando fallos, rangos, valores únicos, unidades y ejemplos antes/después; cambiar el dtype no valida el significado.

### 4.4 Variables categóricas

Primero se inspeccionan frecuencias para hallar espacios, capitalización, errores, sinónimos y marcadores de ausencia. Un entero puede seguir siendo nominal. Para categorías nominales las fuentes respaldan one-hot; una categoría solo debe marcarse ordenada cuando el orden sea real. [Müller, cap. 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md); [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md)

| Problema observado | Preguntas antes de actuar | Tratamiento respaldado por las fuentes | Riesgo o precaución |
|---|---|---|---|
| Variantes de una etiqueta | ¿Son equivalentes según dominio? | Normalizar o mapear equivalencias verificadas | No fusionar estados distintos |
| Categoría nominal | ¿Hay orden real? ¿Cardinalidad manejable? | `OneHotEncoder` o `get_dummies()` | Explosión dimensional; vocabulario desde train |
| Categoría ordinal | ¿Orden y niveles están definidos? | `Categorical` ordenado o codificación ordinal | Los códigos no prueban distancias cuantitativas |
| Categoría nueva | ¿Cuál es la política del sistema? | Encoder aprendido en train; adaptación moderna con `handle_unknown="ignore"` | Codificar particiones por separado desalinearía columnas |
| Baja cardinalidad repetida | ¿Se busca semántica o ahorro de memoria? | `astype("category")` | La conversión cuesta; revisar niveles no usados |

No se desarrolla una estrategia completa de alta cardinalidad: las fuentes advierten del coste de one-hot, pero las alternativas avanzadas pertenecen a representación de variables.

### 4.5 Variables numéricas y escalamiento

Escalar es pertinente para modelos sensibles a magnitudes, distancias, kernels o densidad; no es universal. `StandardScaler` usa media y desviación; `RobustScaler`, mediana y cuartiles; `MinMaxScaler`, extremos de entrenamiento; `Normalizer` actúa por fila. [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md)

| Problema observado | Preguntas antes de actuar | Tratamiento respaldado por las fuentes | Riesgo o precaución |
|---|---|---|---|
| Escalas heterogéneas en método sensible a distancia | ¿La distancia es significativa? | Estandarización o min-max | Ajustar solo con train; cambia la geometría |
| Outliers influyen en media/desviación | ¿Son errores o extremos válidos? | Considerar `RobustScaler` | No elimina outliers ni fija rango |
| Se necesita intervalo de referencia | ¿Es aceptable depender de extremos? | `MinMaxScaler` | Sensible a outliers; nuevos valores pueden salir del intervalo |
| Importa dirección de la muestra | ¿Debe descartarse magnitud? | `Normalizer` | Normaliza filas, no variables |
| Modelo de árboles | ¿Hay evidencia de beneficio? | Variables originales como baseline | No se prescribe escalamiento universal |

### 4.6 Valores atípicos o fuera de rango

Son decisiones distintas detectar un valor inusual, confirmar que contradice el dominio y elegir tratamiento. McKinney muestra máscaras, filtrado y acotamiento; Molin prioriza reglas de dominio. El umbral `3` de un ejemplo simulado no es universal. [McKinney, cap. 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md); [Molin, cap. 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)

| Problema observado | Preguntas antes de actuar | Tratamiento respaldado por las fuentes | Riesgo o precaución |
|---|---|---|---|
| Valor imposible | ¿Unidad, captura o conversión lo explican? | Corregir con evidencia; filtrar o marcar si no es recuperable | No reemplazar antes de investigar |
| Valor extremo posible | ¿Es señal, subpoblación o error? | Conservar, revisar, filtrar o limitar según contexto | Recortar extremos válidos elimina señal |
| Estadísticos afectados | ¿Está confirmada la fuente del extremo? | Regla de dominio; considerar escala robusta | Robustez de escala no equivale a limpiar |

No hay umbrales universales ni política general de recorte. La eficacia debe verificarse con datos reales y evaluación fuera de muestra.

### 4.7 Transformaciones de datos

Las transformaciones de compatibilidad incluyen convertir tipos, unificar etiquetas, reemplazar centinelas, codificar y escalar. McKinney también cubre `cut`/`qcut`, texto y regex; Müller cubre `log1p`, bins, interacciones y polinomios. Estas últimas cambian capacidad predictiva y pertenecen principalmente a *feature engineering*.

- `cut` usa límites definidos; `qcut` aprende cuantiles. Ambos pierden resolución y deben aplicarse consistentemente.
- `log1p` está respaldado para conteos no negativos sesgados; no se aplica indiscriminadamente.
- Regex y métodos `str` limpian o extraen texto, pero deben validarse con casos positivos, negativos y NA.
- Bins, interacciones, polinomios y calendario son representaciones candidatas, no correcciones obligatorias.

**Síntesis integrada:** una transformación necesaria para conservar significado o compatibilidad forma parte de limpieza/preparación; una expansión destinada a crear señal corresponde al documento de *feature engineering*.

## 5. Qué ajustar solo con los datos de entrenamiento

Se ajustan solo con entrenamiento —o con el fold de entrenamiento en validación cruzada— los tratamientos que aprenden de la muestra:

- estadísticas de imputación;
- media/desviación, mediana/cuartiles o mínimos/máximos de escaladores;
- vocabulario y orden de columnas categóricas;
- cuantiles o bins aprendidos;
- parámetros de transformaciones no supervisadas usadas por el modelo;
- selección de variables, especialmente si usa `y`.

Usar validación o prueba para calcularlos transmite información del conjunto evaluado. Ajustar por separado cada conjunto también crea coordenadas o columnas incompatibles. [Müller, cap. 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md); [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)

1. `fit` aprende con entrenamiento.
2. `transform` reutiliza exactamente lo aprendido en entrenamiento, validación, prueba e inferencia.

El desarrollo completo de la fuga corresponde a `knowledge/preprocessing/data_leakage.md`.

## 6. Uso de pipelines

Un `Pipeline` encapsula transformaciones y estimador bajo una interfaz única. En validación cruzada, cada transformación se ajusta dentro del fold de entrenamiento y solo transforma el fold de validación. Esto conserva orden, reduce diferencias entre entrenamiento e inferencia y permite ajustar parámetros con `<paso>__<parametro>`. [Müller, cap. 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)

`ColumnTransformer` aplica ramas diferentes a columnas numéricas y categóricas y concatena resultados. Géron lo usa para combinar imputación, escalado y one-hot. Los pasos intermedios deben transformar entradas compatibles y conservar filas cuando se requiere correspondencia con `y`.

El pipeline no decide si una limpieza es semánticamente correcta ni corrige una mala partición. Hace reproducible una política ya justificada y evita ajustar accidentalmente transformadores antes de validar.

## 7. Reglas de decisión

| Situación | Tratamiento que puede considerarse | Motivo | Precaución |
|---|---|---|---|
| Faltante de significado desconocido | Conservar y diagnosticar | Puede contener contexto del proceso | No rellenar solo para completar |
| Faltante numérico y mediana defendible | Imputador ajustado con train | Conserva filas | Puede reducir variabilidad |
| Repetición bajo clave válida | Deduplicar con `subset` y `keep` explícitos | Elimina redundancia confirmada | Verificar eventos distintos |
| Número/fecha como texto | Conversión y auditoría de fallos | Recupera semántica operativa | Revisar NA creados y ambigüedad |
| Nominal de cardinalidad manejable | One-hot ajustado con train | Evita orden artificial | Categorías desconocidas y coste |
| Ordinal real | Representación ordenada | Conserva orden | No inferir distancias |
| Algoritmo sensible a escala | Escalador en pipeline | Evita dominio por magnitud | Elegir según geometría y outliers |
| Extremo imposible confirmado | Corregir, filtrar o marcar con evidencia | Evita contaminación | Sin umbral universal |
| Extremo posible | Conservar inicialmente y comparar | Puede ser señal | No modificar por rareza |
| Tratamiento dependiente de datos | Ajustar en train y reutilizar | Evita leakage | Dentro de cada fold en CV |
| Limpieza compleja sin evidencia | Posponer y registrar | Protege baseline reproducible | Resolver si impide entrenar |

Las dos últimas reglas son **conclusiones prácticas derivadas**.

## 8. Implementación mínima en Python

Diagnóstico sin tratamiento automático:

```python
import numpy as np
import pandas as pd


def quality_report(data: pd.DataFrame, duplicate_key=None) -> dict:
    duplicates = data.duplicated(subset=duplicate_key, keep=False)
    numeric = data.select_dtypes(include="number")
    return {
        "shape": data.shape,
        "dtypes": data.dtypes.astype(str),
        "missing_by_column": data.isna().sum(),
        "infinite_numeric": numeric.isin([np.inf, -np.inf]).sum(),
        "duplicate_rows": int(duplicates.sum()),
    }
```

Separación previa y ajuste exclusivo con entrenamiento; las listas de columnas deben definirse por semántica, no solo por dtype:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_steps = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_steps = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_steps, numeric_columns),
    ("categorical", categorical_steps, categorical_columns),
])

X_train_ready = preprocessor.fit_transform(X_train)
X_valid_ready = preprocessor.transform(X_valid)
```

La mediana está respaldada como baseline por Géron. Este ejemplo no impone un tratamiento para faltantes categóricos: debe elegirse antes de usarlo según la semántica de la ausencia y el respaldo disponible.

**Nota práctica: adaptación moderna de la idea descrita en la fuente.** Se usa `handle_unknown="ignore"` y un objeto ajustado solo con entrenamiento, como en las adaptaciones modernas de los resúmenes. Para validación cruzada, `preprocessor` debe incluirse junto al estimador en otro `Pipeline`, no ejecutarse globalmente antes de los folds.

## 9. Errores frecuentes y riesgos

- **Explícito:** imputar sin entender la ausencia; eliminar información útil; tratar NA como cero; propagar sin orden significativo.
- **Explícito:** deduplicar con clave incorrecta o asumir que toda repetición es error.
- **Explícito:** aceptar conversiones de fecha, zona, unidad o número sin revisar fallos y rangos.
- **Explícito:** codificar nominales como enteros ordinales; declarar orden inexistente; crear demasiadas columnas one-hot.
- **Explícito:** codificar o escalar train y validación por separado.
- **Explícito:** ajustar imputación, escala, bins, codificación o selección antes del split o fuera de CV.
- **Explícito:** aplicar a evaluación otra secuencia o volver a ajustar parámetros.
- **Explícito:** eliminar/limitar outliers sin investigar o universalizar un umbral de ejemplo.
- **Explícito:** aplicar log a valores incompatibles, discretizar sin soporte o expandir polinomios sin validar coste.
- **Síntesis práctica:** documentar solo el código, sin motivo y verificación, impide auditar la decisión.
- **Síntesis práctica:** asumir que limpiar siempre mejora el rendimiento confunde calidad semántica con capacidad predictiva.

## 10. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Atender primero lo que impide un baseline o invalida significado: split/target, tipos incompatibles, infinitos, faltantes no admitidos, columnas reveladoras y desalineación.
2. Confirmar unidad, claves, tipos semánticos y rangos antes de borrar o rellenar.
3. Crear una separación reproducible y un pipeline corto; mantener el holdout fuera de decisiones iterativas.
4. No deduplicar, recortar outliers, interpolar, discretizar o normalizar agresivamente sin evidencia.
5. Ajustar cada transformación con entrenamiento —o dentro de cada fold— y reutilizarla en validación.
6. Comparar una alternativa a la vez y registrar score, columnas, forma, tiempo y decisión.
7. Posponer limpiezas complejas si no impiden el baseline y no hay hipótesis comprobable; registrar el riesgo.
8. Anotar qué cambió, por qué, con qué parámetros, qué quedó pendiente y cómo se verificó.

## 11. Checklist operativo

- [ ] Revisé los hallazgos de auditoría antes de limpiar.
- [ ] Confirmé unidad de observación, claves, significado, unidades y tipo semántico.
- [ ] Separé detección de faltantes, duplicados o extremos de su tratamiento.
- [ ] Justifiqué cada eliminación, imputación, conversión o recodificación.
- [ ] Separé entrenamiento y validación antes de ajustar transformaciones.
- [ ] Ajusté imputadores, escaladores, encoders y límites solo con entrenamiento.
- [ ] Apliqué los mismos objetos, sin reajuste, a validación o prueba.
- [ ] Incluí en el pipeline los pasos que deben ajustarse en cada fold.
- [ ] Verifiqué tipos, rangos, faltantes, duplicados, cardinalidad y columnas después.
- [ ] Comprobé la alineación de categorías y columnas codificadas.
- [ ] Registré decisiones, supuestos, parámetros, riesgos y casos pospuestos.

## 12. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con limpieza de datos |
|---|---|
| `knowledge/eda/dataset_audit.md` | Identifica problemas de calidad que pueden requerir tratamiento |
| `knowledge/preprocessing/data_leakage.md` | Desarrolla riesgos de fuga durante preparación y transformación |
| `knowledge/feature_engineering/feature_engineering.md` | Trata creación y selección de variables |
| `knowledge/validation/validation_strategy.md` | Define cómo separar y evaluar los datos |
| `knowledge/workflows/competition_workflow.md` | Integra la limpieza dentro del flujo de competición |

## 13. Vacíos y limitaciones de las fuentes

- No desarrollan un marco estadístico completo para mecanismos de ausencia ni una estrategia universal de imputación.
- No ofrecen porcentajes universales para eliminar filas o columnas con faltantes.
- No dan una taxonomía completa de duplicados de entidades, eventos o transacciones; la clave depende del dominio.
- La cobertura de texto se concentra en cadenas y regex, no en limpieza lingüística integral.
- One-hot está cubierto para cardinalidad controlada; no hay respaldo suficiente aquí para recomendar alternativas de alta cardinalidad.
- No hay umbrales universales ni evidencia para prescribir recorte, eliminación o imputación general de outliers.
- Interpolación y propagación dependen de continuidad y orden y deben probarse con datos reales.
- Transformaciones avanzadas, selección y variables calendario corresponden a *feature engineering*.
- Partición temporal/por grupos y leakage exhaustivo deben desarrollarse en sus documentos específicos.
- Algunas APIs originales son antiguas; las actualizaciones usadas están marcadas como adaptaciones prácticas.
- Ninguna fuente garantiza que una limpieza aumente la métrica; depende de datos, modelo y evaluación.

## 14. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Aurélien Géron, *Hands-On Machine Learning* | [Resumen, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Split, imputación, codificación, escalamiento, `Pipeline` y `ColumnTransformer` |
| Andreas C. Müller y Sarah Guido, *Introduction to Machine Learning with Python* | [Resumen, capítulo 3](../../sources/muller_introduction_ml_python/ch03_unsupervised_and_preprocessing.md) | Escaladores; `fit` en train y `transform` en evaluación |
| Andreas C. Müller y Sarah Guido, *Introduction to Machine Learning with Python* | [Resumen, capítulo 4](../../sources/muller_introduction_ml_python/ch04_feature_engineering.md) | Semántica categórica, one-hot, consistencia y frontera con feature engineering |
| Andreas C. Müller y Sarah Guido, *Introduction to Machine Learning with Python* | [Resumen, capítulo 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Pipelines, ajuste por fold y prevención de leakage |
| Wes McKinney, *Python for Data Analysis* | [Resumen, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) | Faltantes, duplicados, reemplazos, tipos, texto, categorías, bins y outliers |
| Stefanie Molin, *Hands-On Data Analysis with Pandas* | [Resumen, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) | Dominio, tipos, fechas, unidades, infinitos, duplicados y verificación |

## 15. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | McKinney cap. 7; Molin cap. 3 | Síntesis integrada |
| 2. Principios | Molin cap. 3; McKinney cap. 7; Géron cap. 2; Müller caps. 3 y 6 | Síntesis integrada |
| 3. Flujo | Molin cap. 3; Géron cap. 2; Müller cap. 6 | Conclusión práctica derivada |
| 4.1 Valores faltantes | McKinney cap. 7; Molin cap. 3; Géron cap. 2 | Explícito de la fuente |
| 4.2 Duplicados | McKinney cap. 7; Molin cap. 3 | Explícito de la fuente |
| 4.3 Tipos y formatos | Molin cap. 3; McKinney cap. 7 | Explícito de la fuente |
| 4.4 Categóricas | Müller cap. 4; McKinney cap. 7; Géron cap. 2 | Síntesis integrada |
| 4.5 Numéricas y escalamiento | Müller cap. 3; Géron cap. 2 | Explícito de la fuente |
| 4.6 Atípicos | McKinney cap. 7; Molin cap. 3 | Síntesis integrada |
| 4.7 Transformaciones | McKinney cap. 7; Müller cap. 4 | Síntesis integrada |
| 5. Ajuste con entrenamiento | Géron cap. 2; Müller caps. 3, 4 y 6 | Síntesis integrada |
| 6. Pipelines | Müller cap. 6; Géron cap. 2 | Explícito de la fuente |
| 7. Reglas | Las seis fuentes | Síntesis integrada |
| 8. Implementación | Géron cap. 2; Müller caps. 4 y 6; McKinney cap. 7 | Conclusión práctica derivada |
| 9. Errores y riesgos | Las seis fuentes | Síntesis integrada |
| 10. Coding test/datathon | Las seis fuentes | Recomendación para coding test |
| 11. Checklist | Las seis fuentes | Conclusión práctica derivada |
| 12. Documentos relacionados | Estructura solicitada | Síntesis integrada |
| 13. Vacíos | Alcance observado en las seis fuentes | Conclusión práctica derivada |
| 14. Fuentes | Resúmenes leídos y citados | Explícito de la fuente |
