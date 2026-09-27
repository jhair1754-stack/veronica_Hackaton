---
title: "Auditoría de datasets para Machine Learning"
knowledge_area: "eda"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Auditoría de datasets para Machine Learning

## 1. Propósito y alcance

Una auditoría determina si un dataset es comprensible y utilizable para el objetivo planteado, qué señales requieren investigación y qué decisiones deben quedar resueltas antes de modelar. Revisa la unidad de observación, el esquema, la calidad, las distribuciones, el objetivo y las dependencias que condicionan la partición. El resultado es un registro de evidencia, riesgos, preguntas y decisiones pendientes; no es un dataset ya limpiado ni una garantía de calidad. Esta función previa al modelado está respaldada por los flujos de inspección y proyecto de [Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), [ISLP, capítulo 2](../../sources/islp_python/ch02_statistical_learning.md) y [Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md).

Conviene separar cuatro actividades:

1. **Describir:** registrar lo observable, por ejemplo dimensiones, tipos, frecuencias, rangos o ausencias.
2. **Diagnosticar:** interpretar una observación como señal de un posible problema, sin darla aún por confirmada.
3. **Decidir:** elegir, con contexto de dominio y del uso futuro, si el hallazgo exige tratamiento, una partición especial, datos adicionales o ninguna acción.
4. **Transformar:** ejecutar imputación, eliminación, conversión, codificación u otra modificación. Esta fase queda fuera de la auditoría detallada y corresponde a documentos posteriores.

**Conclusión práctica derivada:** conservar el dataset sin modificar durante la auditoría facilita distinguir evidencia original de una consecuencia del preprocesamiento. Cuando el análisis exploratorio vaya a guiar transformaciones o variables, debe protegerse la evaluación fuera de muestra; Géron recomienda separar test antes de la exploración detallada ([capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md)).

## 2. Qué debe responder una auditoría

La auditoría debe producir respuestas verificables o marcar explícitamente que falta evidencia:

- ¿Qué representa una fila y cuál es la granularidad de la observación?
- ¿Cuántas filas y columnas hay, y puede ese conteo explicarse por el proceso de recopilación?
- ¿Qué significa cada columna y cuál es su tipo semántico, no solo su `dtype`?
- ¿Qué columna es el objetivo, qué representa y está correctamente tipada?
- ¿Qué columnas son identificadores, predictores candidatos o metadatos del proceso?
- ¿Dónde hay valores faltantes, infinitos, centinelas o formatos inconsistentes?
- ¿Existen filas completamente repetidas o claves repetidas? ¿Son duplicados reales o eventos válidos?
- ¿Qué rangos, colas, topes, huecos o valores inusuales requieren contexto?
- ¿Cómo se distribuyen variables numéricas, categóricas y el objetivo?
- En clasificación, ¿cómo se reparten las clases? En un objetivo continuo, ¿hay topes, concentración o extremos relevantes?
- ¿Qué asociaciones, agrupamientos o redundancias merecen investigación, sin inferir causalidad?
- ¿Hay entidades repetidas, grupos o dependencia temporal que impidan tratar las filas como independientes?
- ¿Alguna variable usa información posterior al resultado, deriva del objetivo o no estaría disponible al predecir?
- ¿Qué limitaciones de cobertura, medición, etiquetado o representatividad impiden interpretar el dataset?

Estas preguntas integran la inspección tabular de [ISLP, capítulo 2](../../sources/islp_python/ch02_statistical_learning.md), la exploración de [Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md), la calidad descrita en [ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md) y los riesgos de grupos de [Müller y Guido, capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md). No responden nada sobre un dataset concreto: para ello se necesitan los datos y su documentación.

## 3. Inspección inicial del dataset

### 3.1 Dimensiones y unidad de observación

Registrar `shape` solo responde cuántas filas y columnas hay. Antes de interpretar conteos, faltantes o duplicados hay que declarar qué constituye una fila: persona, transacción, sensor-tiempo, imagen, episodio u otra unidad. Si una entidad puede aparecer varias veces, registrar la clave de entidad, el motivo y la dimensión temporal o de evento. [Molin, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) propone inspeccionar forma, columnas, tipos, valores únicos y calidad; [Müller y Guido, capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) muestra que filas de una misma entidad pueden condicionar la separación.

### 3.2 Columnas, tipos y estructura

Revisar nombres, orden, `dtype`, valores de ejemplo, cardinalidad y significado. Un tipo físico no basta: un entero con pocos niveles puede ser una categoría; una fecha cargada como texto o un marcador como `?` puede convertir una columna numérica en `object` ([ISLP, capítulo 2](../../sources/islp_python/ch02_statistical_learning.md)). Comparar la inferencia del software con el diccionario de datos, las unidades y el proceso de captura. Nombres ambiguos, columnas casi vacías, unidades mezcladas o esquemas distintos entre archivos son señales para investigar, no correcciones automáticas ([ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md)).

### 3.3 Identificadores y variables candidatas

Clasificar provisionalmente cada columna como identificador, predictor candidato, objetivo o metadato de recopilación. Verificar unicidad de índices o claves y cardinalidad; pandas permite revisar `index.is_unique`, `unique()` y `value_counts()` ([McKinney, capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md)). Un identificador puede ser necesario para mantener entidades juntas, crear un split estable o investigar joins. Por ello, su nombre o alta cardinalidad no justifican eliminarlo. Primero debe averiguarse qué codifica, cómo se genera y si estaría disponible en inferencia.

### 3.4 Variable objetivo

Confirmar cuál es el target, qué evento o medición representa, cuándo se conoce y si su tipo corresponde a clasificación o regresión. Revisar faltantes, valores imposibles, categorías, soporte por clase y posibles topes. Géron muestra que un objetivo truncado puede imponer un límite artificial al aprendizaje ([capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)); [ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md) añade que labels erróneos, inconsistentes o sin cobertura distorsionan el aprendizaje. No redefinir ni reparar el objetivo durante esta fase: registrar la evidencia y remitir la decisión a formulación, limpieza o métricas.

| Elemento | Qué revisar | Señal que requiere investigación | Registro recomendado |
|---|---|---|---|
| Dimensiones | Número de filas y columnas | Conteo incompatible con la fuente o que cambia tras una unión | Forma por archivo y por etapa |
| Unidad de observación | Significado de una fila y granularidad | Varias filas por entidad sin explicación | Definición, clave y ejemplo |
| Columnas | Nombre, significado, unidad y procedencia | Nombres ambiguos, unidades o esquemas incompatibles | Diccionario y fuente de cada campo |
| Tipos | `dtype` y tipo semántico | Números/fechas como texto; códigos tratados como magnitudes | Tipo observado, tipo esperado y ejemplos |
| Identificadores | Unicidad, estabilidad y generación | Claves repetidas o que parecen codificar el resultado | Cardinalidad, duplicados y uso previsto |
| Predictores candidatos | Disponibilidad y significado | Campo generado después del evento de predicción | Momento y proceso de disponibilidad |
| Objetivo | Definición, tipo, faltantes, rango y distribución | Labels ausentes/inconsistentes, clases escasas o tope artificial | Definición, conteos y dudas abiertas |
| Metadatos de captura | Fuente, lote, operador o tiempo de ingesta | Campo asociado al target por cómo se recolectó | Proceso que lo produce y riesgo |

**Tipo de contenido:** la tabla es una **síntesis integrada** de las fuentes citadas en esta sección.

## 4. Calidad y consistencia de los datos

### 4.1 Valores faltantes

Inspeccionar por columna y, si es pertinente, por fila, grupo, clase o periodo. Distinguir `NaN`, `None`, infinitos y centinelas de dominio: no todos significan lo mismo ([Molin, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)). Registrar conteo, proporción, ejemplos, patrón y si la ausencia aparece también en el target. Preguntar por qué falta, si la ausencia tiene significado, si el proceso de captura cambió y si el valor podría conocerse al predecir.

La decisión posterior podría ser corregir el origen, excluir, imputar, crear una señal de ausencia o no actuar. Las fuentes no establecen una regla universal. [McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) advierte que eliminar puede perder información y rellenar puede alterar la distribución; [Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) exige aprender estadísticas de imputación solo con entrenamiento. El tratamiento pertenece a `knowledge/preprocessing/data_cleaning.md`.

### 4.2 Filas duplicadas y claves repetidas

Medir por separado filas idénticas, duplicados según una clave candidata y multiplicidad por entidad. Registrar las columnas usadas, el criterio `keep`, los conteos y ejemplos. Preguntar si una repetición es error, reintento, actualización, medición simultánea o varias observaciones legítimas. [McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) indica que deduplicar solo es correcto si la clave representa la entidad; [Molin, capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md) advierte que claves repetidas pueden multiplicar filas en un join. La acción posterior puede ser corregir claves, consolidar eventos o eliminar repeticiones confirmadas, nunca borrar automáticamente.

### 4.3 Tipos incorrectos y formatos inconsistentes

Comparar el tipo inferido con el significado esperado. Revisar números almacenados como texto, fechas inválidas, categorías con mayúsculas o espacios, unidades incompatibles, códigos centinela y columnas nominales declaradas como ordenadas. Registrar valores que fallan al convertir y su origen. [Molin, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) respalda `to_datetime`, `to_numeric` y `astype`, pero la auditoría solo identifica el problema; la normalización pertenece a limpieza.

### 4.4 Valores fuera de rango y posibles anomalías

Contrastar mínimos, máximos, cuantiles, histogramas y box plots con límites de dominio y con el proceso de medición. Un extremo puede ser error, evento raro válido, cambio de unidad, tope del instrumento o población distinta. [McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) presenta umbrales en un ejemplo simulado, pero declara que no son un detector general y que pueden eliminar extremos válidos. Por tanto, registrar valor, fila, variable, regla que lo señaló, contexto y decisión pendiente; no etiquetarlo automáticamente como error u outlier.

Para cada problema de calidad, el registro mínimo es: **evidencia observada → alcance → hipótesis → pregunta al dominio → riesgo si no se resuelve → responsable/decisión pendiente**. Esta plantilla es una **conclusión práctica derivada** de las fuentes de calidad y documentación.

## 5. Distribuciones y relaciones

### 5.1 Variables numéricas

Usar conteo, media, desviación, mínimos, máximos, cuantiles y `describe()` para una primera lectura ([McKinney, capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md)). Complementar con histograma, ECDF o box plot cuando la forma importe. Histogramas pueden revelar sesgo, colas, multimodalidad y topes, pero dependen de `bins`; un box plot resume cuartiles y extremos, pero no conserva toda la forma ([McKinney, capítulo 9](../../sources/mckinney_python_data_analysis/ch09_visualization.md); [Molin, capítulo 5](../../sources/molin_hands_on_pandas/ch05_visualization_matplotlib.md)).

### 5.2 Variables categóricas

Revisar cardinalidad, categorías únicas, frecuencias y faltantes mediante `unique()` y `value_counts()`. Las barras muestran conteos; proporciones normalizadas ayudan a comparar composición entre grupos, pero ocultan volumen absoluto. Registrar categorías raras o inesperadas como señales. No consolidarlas ni imponer orden durante la auditoría. La alta cardinalidad anticipa coste para codificaciones como one-hot, pero no determina por sí sola el tratamiento ([McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md)).

### 5.3 Relaciones entre variables

Usar tablas cruzadas para pares categóricos; dispersión o `hexbin` para pares numéricos; box plots o resúmenes por grupo para una variable numérica frente a una categórica. La correlación es un cribado de asociación lineal: una correlación baja no descarta dependencia no lineal y una alta no establece causalidad ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [McKinney, capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md)). Matrices de pares crecen rápidamente con el número de variables, por lo que deben limitarse a un subconjunto justificado ([Molin, capítulo 5](../../sources/molin_hands_on_pandas/ch05_visualization_matplotlib.md)).

### 5.4 Distribución y balance del objetivo

En clasificación, registrar conteos y proporciones por clase y comprobar si la composición cambia por grupo, tiempo o partición. El desbalance es una propiedad que condiciona evaluación y partición; no confirma por sí solo que deba remuestrearse. [Müller y Guido, capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) muestra que accuracy puede ser engañosa y que la estratificación conserva aproximadamente proporciones. En regresión, revisar rango, cuantiles, concentración, colas y topes; un objetivo truncado requiere entender si el sistema debe predecir fuera de ese límite ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)).

| Tipo de variable o relación | Inspección posible | Qué puede revelar | Precaución |
|---|---|---|---|
| Numérica | `describe`, cuantiles, histograma, ECDF, box plot | Escala, dispersión, sesgo, colas, huecos y topes | La visual depende de bins/resumen; un extremo no es automáticamente error |
| Categórica | `unique`, `value_counts`, barras | Cardinalidad, clases raras, categorías inesperadas | Mostrar conteo además de proporción |
| Numérica-numérica | Correlación, scatter, `hexbin` | Asociación lineal, forma, densidad y observaciones influyentes | Asociación no implica causalidad; Pearson no cubre toda no linealidad |
| Categórica-categórica | `crosstab`, conteos y proporciones | Composición y combinaciones escasas | La normalización oculta tamaño de muestra |
| Numérica-categórica | Resumen por grupo y box plot | Diferencias de centro, dispersión y extremos | Comparar también tamaños y forma completa |
| Objetivo de clasificación | Conteos/proporciones globales y por grupo/tiempo | Desbalance y cobertura desigual | No decidir métrica o remuestreo solo por el conteo |
| Objetivo continuo | `describe`, histograma, ECDF, topes | Escala, concentración, colas y truncamiento | No confundir rareza con error ni forma con causalidad |
| Serie temporal | Línea, huecos, `lag_plot`, autocorrelación | Cambios, estacionalidad o dependencia | Ordenar; un lag observacional no siempre equivale a tiempo calendario |

**Tipo de contenido:** la tabla es una **síntesis integrada** de los capítulos de pandas y visualización citados.

## 6. Riesgos de partición, tiempo y grupos

La auditoría no define la validación final, pero debe registrar las señales que la condicionan:

- **Entidades repetidas:** varias filas por paciente, persona, dispositivo o hablante pueden filtrar información si una entidad aparece a ambos lados. `GroupKFold` mantiene grupos enteros cuando se busca generalizar a entidades nuevas ([Müller y Guido, capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).
- **Orden temporal:** timestamps, lags, ventanas, estacionalidad o cambios de instrumentación invalidan asumir intercambio aleatorio. Conservar el tiempo del evento y el de disponibilidad ayuda a detectar leakage temporal ([ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md)).
- **Disponibilidad histórica:** ventanas, `shift`, `diff` y joins temporales requieren ordenar por entidad y tiempo; `direction="nearest"`, ventanas centradas o relleno hacia atrás pueden usar futuro ([Molin, capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md)).
- **Estratos críticos:** si una clase o subpoblación importante está desbalanceada, registrar proporciones y tamaño; demasiados estratos o estratos pequeños tampoco son viables ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)).
- **Split estable:** si el dataset se actualiza, un ID único e inmutable puede mantener estable la pertenencia; una semilla aislada no estabiliza necesariamente filas nuevas ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)).

La elección final debe desarrollarse en `knowledge/validation/validation_strategy.md`. Aquí se entrega: unidad de generalización, claves de grupo, columnas temporales, momento de disponibilidad, granularidad y evidencia de cambios entre periodos o subpoblaciones.

## 7. Detección inicial de data leakage

La auditoría busca señales, no confirma fuga solo por el nombre de una columna:

- variable medida o creada después del resultado;
- columna derivada directa o indirectamente del target;
- identificador, lote o metadato que codifica el proceso de etiquetado;
- misma entidad o evento relacionado en entrenamiento y validación;
- imputación, escalado, selección, cuantiles, categorías o agregaciones aprendidos antes de separar;
- join, lag, ventana o relleno que incorpora información futura;
- exploración reiterada de test para decidir variables o transformaciones.

Para cada sospecha, investigar **cómo se generó**, **cuándo estuvo disponible**, **qué campos utiliza** y **si podría reproducirse en inferencia**. Solo después puede confirmarse o descartarse. Las transformaciones dependientes de datos deben ajustarse dentro de cada fold mediante un pipeline; hacerlo globalmente puede inflar la validación ([Müller y Guido, capítulo 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)). Los riesgos temporales y de joins están descritos en [Molin, capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md). El tratamiento completo corresponde a `knowledge/preprocessing/data_leakage.md`.

## 8. Procedimiento de auditoría paso a paso

1. **Recibir datos y contexto.** Registrar procedencia, periodo, población, archivos, diccionario, forma de recolección y objetivo. No transformar. ([ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md))
2. **Definir la unidad de observación.** Escribir qué representa una fila, su granularidad, claves candidatas y si una entidad puede repetirse. ([ISLP, capítulo 2](../../sources/islp_python/ch02_statistical_learning.md))
3. **Inventariar estructura.** Registrar forma, nombres, `dtype`, valores de ejemplo, cardinalidad e índice; contrastar tipos físicos y semánticos. ([ISLP, capítulo 2](../../sources/islp_python/ch02_statistical_learning.md); [Molin, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md))
4. **Clasificar columnas provisionalmente.** Separar objetivo, identificadores, predictores candidatos y metadatos; anotar generación y disponibilidad. ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md))
5. **Auditar el target.** Confirmar definición, tipo, faltantes, rango/conteos, topes y cobertura; no alterar labels. ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md))
6. **Cuantificar calidad.** Contar faltantes, infinitos, centinelas conocidos, duplicados por varios criterios, conversiones fallidas y rangos sospechosos. ([McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md); [Molin, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md))
7. **Describir distribuciones.** Combinar estadísticas, frecuencias y un conjunto pequeño de visualizaciones pertinentes; anotar sensibilidad a bins y tamaños de grupo. ([McKinney, capítulo 9](../../sources/mckinney_python_data_analysis/ch09_visualization.md); [Molin, capítulo 5](../../sources/molin_hands_on_pandas/ch05_visualization_matplotlib.md))
8. **Explorar relaciones seleccionadas.** Usar correlaciones, dispersión, tablas cruzadas o resúmenes por grupo según tipos; tratarlas como asociación exploratoria. ([McKinney, capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md))
9. **Identificar estructura de partición.** Localizar tiempo, grupos, entidades repetidas, estratos y condiciones de disponibilidad. ([Müller y Guido, capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md); [Molin, capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md))
10. **Buscar señales de leakage.** Revisar procedencia y tiempo de cada campo, joins y cualquier estadística o selección aprendida globalmente. ([Müller y Guido, capítulo 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md))
11. **Priorizar sin limpiar.** Clasificar hallazgos por impacto potencial, evidencia y coste de investigación; formular decisiones condicionales para limpieza y validación.
12. **Cerrar el registro.** Documentar evidencia, preguntas, limitaciones, responsable, siguiente documento y trazabilidad. ([Géron, apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md))

Los pasos 11 y 12 son una **conclusión práctica derivada** que integra la exigencia de documentar hallazgos y conservar una evaluación independiente.

## 9. Implementación mínima en Python

El siguiente bloque inspecciona un `DataFrame` ya recibido como `data`. No asume archivo, columnas ni target; no modifica los datos ni ajusta transformaciones. Las operaciones están respaldadas por la inspección de [ISLP, capítulo 2](../../sources/islp_python/ch02_statistical_learning.md), las reducciones de [McKinney, capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md) y duplicados/faltantes de [McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md).

```python
import pandas as pd


def audit_table(data: pd.DataFrame, target_column: str | None = None) -> dict:
    """Devuelve vistas de auditoría; no transforma el DataFrame."""
    numeric_columns = data.select_dtypes(include="number").columns

    report = {
        # Dimensiones, columnas y tipos inferidos.
        "shape": data.shape,
        "dtypes": data.dtypes.astype("string"),
        # Ausencia por columna; los infinitos requieren revisión separada.
        "missing_count": data.isna().sum().sort_values(ascending=False),
        "missing_rate": data.isna().mean().sort_values(ascending=False),
        # Filas idénticas; no equivale a duplicado de entidad.
        "fully_duplicated_rows": int(data.duplicated(keep=False).sum()),
        # Cardinalidad incluye NA para hacer visible esa categoría de estado.
        "unique_count": data.nunique(dropna=False).sort_values(),
        # Resumen numérico; no decide qué valores son errores.
        "numeric_summary": data[numeric_columns].describe().T,
    }

    if target_column is not None:
        if target_column not in data.columns:
            raise KeyError(f"Target no encontrado: {target_column}")
        # Útil para clasificación; en targets continuos puede ser muy extenso.
        report["target_counts"] = data[target_column].value_counts(
            dropna=False
        )

    return report
```

Para una clave candidata, inspeccionar sin eliminar:

```python
def inspect_candidate_key(data: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Devuelve todas las filas cuya clave candidata se repite."""
    repeated_key = data.duplicated(subset=key_columns, keep=False)
    return data.loc[repeated_key, key_columns].sort_values(key_columns)
```

**Adaptación moderna:** se usan anotaciones `str | None`, `list[str]` y el dtype pandas `"string"` para claridad; son convenciones actuales, no requisitos de los resúmenes originales. Si el target es continuo o de alta cardinalidad, sustituir la tabla exhaustiva por un resumen acorde a su significado. Las visualizaciones y correlaciones se omiten del bloque mínimo para evitar producir automáticamente análisis masivos o inadecuados.

## 10. Reglas de decisión

| Hallazgo | Pregunta de seguimiento | Decisión que puede activarse | Documento relacionado |
|---|---|---|---|
| Valores faltantes | ¿Por qué faltan y qué significan? | Corregir origen, excluir, imputar, señalar ausencia o no actuar | `knowledge/preprocessing/data_cleaning.md` |
| Infinitos o centinelas | ¿Son valores válidos, errores o códigos de ausencia? | Normalizar representación después de confirmarlo | `knowledge/preprocessing/data_cleaning.md` |
| Filas o claves repetidas | ¿La clave representa entidad, evento o versión? | Conservar, consolidar, corregir clave o eliminar duplicado confirmado | `knowledge/preprocessing/data_cleaning.md` |
| Tipo o formato inconsistente | ¿Cuál es el tipo semántico y la unidad correcta? | Convertir y normalizar con registro de fallos | `knowledge/preprocessing/data_cleaning.md` |
| Valor fuera de rango | ¿Viola una regla de dominio o es un caso raro válido? | Corregir, excluir, acotar, usar tratamiento robusto o conservar | `knowledge/preprocessing/data_cleaning.md` |
| Clases desbalanceadas | ¿Qué clases importan y qué representa el uso futuro? | Estratificación y revisión de métrica; tratamiento posterior condicionado | `knowledge/metrics/classification_metrics.md` |
| Entidad repetida | ¿Se generaliza a filas nuevas o entidades nuevas? | Mantener grupos íntegros en la partición | `knowledge/validation/validation_strategy.md` |
| Dependencia temporal | ¿Qué estaba disponible en cada instante? | Preservar orden y definir corte temporal compatible | `knowledge/validation/validation_strategy.md` |
| Columna posterior o derivada del target | ¿Podría existir al predecir y cómo fue creada? | Excluirla del modelado o reconstruirla sin fuga, si se confirma | `knowledge/preprocessing/data_leakage.md` |
| Estadística aprendida globalmente | ¿Se calculó antes de separar o fuera del fold? | Reubicar transformación dentro del pipeline | `knowledge/preprocessing/data_leakage.md` |
| Cardinalidad alta | ¿Es identificador, categoría real o texto libre? | Elegir tratamiento posterior compatible; no eliminar por defecto | `knowledge/feature_engineering/feature_engineering.md` |

**Tipo de contenido:** tabla de **síntesis integrada**; todas las decisiones son condicionales y remiten al documento que debe desarrollarlas.

## 11. Errores frecuentes y riesgos de interpretación

- Asumir que un faltante es aleatorio, irrelevante o equivalente a cero antes de entender su origen ([Molin, capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md)).
- Borrar filas repetidas sin definir la unidad de observación ni una clave válida ([McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md)).
- Eliminar una columna por llamarse `id` o por tener alta cardinalidad sin investigar su uso en grupos, joins o splits. **Síntesis práctica derivada.**
- Confundir un valor extremo con error; los umbrales de ejemplos no son universales ([McKinney, capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md)).
- Tomar correlación, una recta visual o una diferencia entre grupos como causalidad ([McKinney, capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md); [McKinney, capítulo 9](../../sources/mckinney_python_data_analysis/ch09_visualization.md)).
- Evaluar el objetivo de clasificación solo con accuracy o sin mostrar soporte por clase cuando hay desbalance ([Müller y Guido, capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md)).
- Usar información que se genera después del resultado o que no estaría disponible en producción ([Molin, capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md)).
- Calcular imputación, escala, cuantiles, selección o categorías antes de separar entrenamiento y validación ([Müller y Guido, capítulo 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md)).
- Explorar test y convertir sus patrones en decisiones: produce *data snooping* ([Géron, capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md)).
- Generalizar una distribución o ECDF a datos futuros sin justificar representatividad ([Molin, capítulo 5](../../sources/molin_hands_on_pandas/ch05_visualization_matplotlib.md)).
- Ignorar que joins con claves repetidas multiplican filas o que resampling introduce huecos ([Molin, capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md)).
- Ocultar el tamaño de los grupos mostrando solo proporciones normalizadas ([McKinney, capítulo 9](../../sources/mckinney_python_data_analysis/ch09_visualization.md)).

## 12. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

Con tiempo limitado, la prioridad es evitar construir un baseline sobre una unidad de observación mal entendida, un target defectuoso o una partición contaminada. Conviene producir una ficha corta con forma, tipos, target, faltantes, duplicados, clases/extremos, claves, tiempo, grupos y sospechas de leakage. Un hallazgo justifica análisis adicional cuando puede cambiar el target, la unidad de generalización, la disponibilidad de una variable o la validez del split; detalles cosméticos pueden esperar.

Flujo rápido:

1. Confirmar qué representa cada fila.
2. Revisar dimensiones, columnas y tipos.
3. Identificar el target y su distribución.
4. Inspeccionar faltantes, duplicados y valores inusuales.
5. Buscar grupos, tiempo e identificadores relevantes.
6. Investigar posibles señales de leakage.
7. Registrar hallazgos y decisiones pendientes.

Registrar al menos: evidencia reproducible, columnas/filas afectadas, posible impacto, pregunta abierta y siguiente acción. No dedicar esta fase a imputación sofisticada, ingeniería extensa, modelos o tuning. Géron prioriza inspección, split protegido y documentación ([capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md); [apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md)); [ML Design Patterns, capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md) prioriza target, duplicados, faltantes, labels, unidades, rangos y cobertura.

## 13. Checklist operativo

- [ ] Definí dimensiones, unidad de observación y granularidad.
- [ ] Registré significado, origen, tipo semántico y unidad de cada columna relevante.
- [ ] Identifiqué objetivo, tipo, faltantes, rango o conteos y posibles topes.
- [ ] Cuantifiqué faltantes, infinitos y centinelas sin tratarlos automáticamente.
- [ ] Revisé filas idénticas y claves repetidas con criterios explícitos.
- [ ] Inspeccioné rangos y valores inusuales con contexto de dominio.
- [ ] Describí numéricas y categóricas con estadísticas/conteos y vistas pertinentes.
- [ ] Traté relaciones observadas como asociación, no causalidad.
- [ ] Identifiqué entidades repetidas, grupos, orden temporal y momentos de disponibilidad.
- [ ] Busqué columnas posteriores, derivadas del target y transformaciones globales.
- [ ] Separé hallazgos confirmados de sospechas que requieren investigación.
- [ ] Documenté limitaciones, preguntas, responsables y decisiones pendientes.
- [ ] Añadí fuente y tipo de contenido a los hallazgos importantes.

## 14. Relación con otros documentos de `knowledge/`

| Documento relacionado | Qué recibe de la auditoría |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Contrasta la formulación con la estructura real de los datos |
| `knowledge/preprocessing/data_cleaning.md` | Recibe problemas de calidad que requieren tratamiento |
| `knowledge/preprocessing/data_leakage.md` | Desarrolla los riesgos de fuga detectados |
| `knowledge/feature_engineering/feature_engineering.md` | Recibe preguntas sobre variables y transformaciones |
| `knowledge/validation/validation_strategy.md` | Recibe señales de tiempo, grupos o dependencia entre observaciones |
| `knowledge/metrics/classification_metrics.md` | Recibe hallazgos sobre la distribución de un objetivo de clasificación |
| `knowledge/metrics/regression_metrics.md` | Recibe hallazgos sobre la escala y distribución de un objetivo continuo |

Esta sección solo define el traspaso de información; no desarrolla ni crea esos documentos.

## 15. Vacíos y limitaciones de las fuentes

- No se proporcionó un dataset ni un diccionario de datos; ninguna señal puede confirmarse para un caso real.
- Las fuentes no ofrecen un umbral universal para porcentaje de faltantes, cardinalidad, correlación, duplicados u outliers. Los valores usados en ejemplos dependen de su contexto.
- Los resúmenes describen herramientas para detectar y tratar problemas, pero no un estándar único de severidad o un formato formal de reporte de auditoría. La plantilla de evidencia de este documento es una síntesis.
- La inspección de faltantes está cubierta, pero los mecanismos estadísticos de ausencia y métodos avanzados de imputación no se desarrollan de forma suficiente para incluirlos aquí.
- Hay cobertura de grupos y riesgos temporales, pero la estrategia completa de validación temporal, por grupos o combinada queda fuera de alcance y debe validarse en su documento específico.
- Las fuentes permiten detectar señales de leakage, pero confirmar una fuga exige conocer el proceso de generación y el momento real de disponibilidad de cada dato.
- Los resúmenes cubren datos tabulares principalmente. No aportan un protocolo de auditoría equivalente y detallado para imagen, audio, video, grafos o texto libre.
- Calidad de labels subjetivos, representatividad y drift aparecen a nivel conceptual en ML Design Patterns, pero no se ofrecen pruebas universales para certificarlos.
- La reproducibilidad completa de artefactos, versiones y ejecución se menciona, pero este documento solo conserva trazabilidad de la auditoría; la implementación operativa queda para documentación posterior.
- Las limitaciones previas sobre ranking autónomo, forecasting completo e implementación probabilística avanzada no se trasladan automáticamente aquí. Las fuentes sí respaldan revisar orden temporal, disponibilidad y dependencia; no respaldan convertir esta auditoría en un procedimiento completo de forecasting.

## 16. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning* | [Capítulo 2](../../sources/geron_hands_on_ml/ch02_end_to_end_ml_project.md) | Flujo EDA, inspección, target, split protegido, correlación, pipelines y leakage |
| Géron, *Hands-On Machine Learning* | [Apéndice B](../../sources/geron_hands_on_ml/appendix_b_machine_learning_project_checklist.md) | Checklist, exploración documentada y separación previa de test |
| James et al., *ISLP with Python* | [Capítulo 2](../../sources/islp_python/ch02_statistical_learning.md) | Unidad de observación, inspección tabular, tipos semánticos, faltantes y cautela causal |
| McKinney, *Python for Data Analysis* | [Capítulo 5](../../sources/mckinney_python_data_analysis/ch05_pandas_fundamentals.md) | Índices, unicidad, estadísticas descriptivas, frecuencias y correlación |
| McKinney, *Python for Data Analysis* | [Capítulo 7](../../sources/mckinney_python_data_analysis/ch07_data_cleaning.md) | Faltantes, duplicados, tipos categóricos y precauciones con outliers |
| McKinney, *Python for Data Analysis* | [Capítulo 9](../../sources/mckinney_python_data_analysis/ch09_visualization.md) | Histogramas, barras, scatter, box plots y precauciones visuales |
| Molin, *Hands-On Data Analysis with Pandas* | [Capítulo 3](../../sources/molin_hands_on_pandas/ch03_data_wrangling.md) | Forma, tipos, fechas, faltantes, infinitos, duplicados y coherencia temporal |
| Molin, *Hands-On Data Analysis with Pandas* | [Capítulo 4](../../sources/molin_hands_on_pandas/ch04_aggregation_time_series.md) | Claves, joins, grupos, granularidad temporal, ventanas y leakage futuro |
| Molin, *Hands-On Data Analysis with Pandas* | [Capítulo 5](../../sources/molin_hands_on_pandas/ch05_visualization_matplotlib.md) | Distribuciones, ECDF, correlación visual, lags y límites de interpretación |
| Müller y Guido, *Introduction to Machine Learning with Python* | [Capítulo 5](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | Desbalance, estratificación, grupos y separación train/validation/test |
| Müller y Guido, *Introduction to Machine Learning with Python* | [Capítulo 6](../../sources/muller_introduction_ml_python/ch06_pipelines.md) | Transformaciones dentro de folds, pipelines y prevención de leakage |
| Lakshmanan et al., *Machine Learning Design Patterns* | [Capítulo 1](../../sources/ml_design_patterns/ch01_ml_design_challenges.md) | Calidad, consistencia, cobertura, timestamps de evento/ingesta, drift y reproducibilidad |

Solo se listan resúmenes leídos y utilizados; no se consultaron fuentes externas.

## 17. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Géron ch. 2 y ap. B; ISLP ch. 2 | Síntesis integrada |
| 2. Preguntas de auditoría | Géron ch. 2; ISLP ch. 2; ML Design Patterns ch. 1; Müller ch. 5 | Síntesis integrada |
| 3.1 Dimensiones y unidad | Molin ch. 3; Müller ch. 5 | Síntesis integrada |
| 3.2 Columnas y tipos | ISLP ch. 2; Molin ch. 3; ML Design Patterns ch. 1 | Explícito de la fuente |
| 3.3 Identificadores | McKinney ch. 5; Müller ch. 5; Géron ch. 2 | Síntesis integrada |
| 3.4 Variable objetivo | Géron ch. 2; ML Design Patterns ch. 1 | Síntesis integrada |
| 4.1 Valores faltantes | Molin ch. 3; McKinney ch. 7; Géron ch. 2 | Síntesis integrada |
| 4.2 Duplicados y claves | McKinney ch. 7; Molin ch. 4 | Síntesis integrada |
| 4.3 Tipos y formatos | ISLP ch. 2; Molin ch. 3 | Explícito de la fuente |
| 4.4 Rangos y anomalías | McKinney ch. 7; McKinney ch. 9 | Síntesis integrada |
| 5.1–5.3 Distribuciones y relaciones | McKinney ch. 5 y 9; Molin ch. 5; Géron ch. 2 | Síntesis integrada |
| 5.4 Balance del objetivo | Müller ch. 5; Géron ch. 2 | Síntesis integrada |
| 6. Partición, tiempo y grupos | Müller ch. 5; Molin ch. 4; ML Design Patterns ch. 1; Géron ch. 2 | Síntesis integrada |
| 7. Data leakage | Müller ch. 6; Molin ch. 4; Géron ch. 2 | Síntesis integrada |
| 8. Procedimiento | Todas las fuentes técnicas anteriores | Conclusión práctica derivada |
| 9. Python mínimo | ISLP ch. 2; McKinney ch. 5 y 7 | Conclusión práctica derivada |
| 10. Reglas de decisión | McKinney ch. 7; Müller ch. 5 y 6; Molin ch. 4 | Síntesis integrada |
| 11. Errores frecuentes | Géron ch. 2; McKinney ch. 5, 7 y 9; Molin ch. 3–5; Müller ch. 5–6 | Síntesis integrada |
| 12. Coding test/datathon | Géron ch. 2 y ap. B; ML Design Patterns ch. 1 | Recomendación para coding test |
| 13. Checklist | Conjunto de fuentes utilizadas | Conclusión práctica derivada |
| 14. Relación con otros documentos | Estructura solicitada y hallazgos de la auditoría | Conclusión práctica derivada |
| 15. Vacíos y limitaciones | Cobertura observada en los resúmenes utilizados | Conclusión práctica derivada |
| 16. Fuentes utilizadas | Archivos Markdown verificados | Explícito de la fuente |
