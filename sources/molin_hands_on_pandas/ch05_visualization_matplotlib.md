---
title: "Visualizing Data with Pandas and Matplotlib"
topics:
  - "Visualización exploratoria con pandas y Matplotlib"
  - "Selección de gráficos para relaciones, distribuciones, conteos y series temporales"
  - "Diagnóstico visual de correlación, autocorrelación e incertidumbre"
target_knowledge_files:
  - "knowledge/eda/visualization_selection.md"
  - "knowledge/eda/distribution_and_relationship_plots.md"
  - "knowledge/time_series/visual_diagnostics.md"
  - "knowledge/feature_engineering/correlation_diagnostics.md"
---

# 5. Visualizing Data with Pandas and Matplotlib

## 1. Propósito del capítulo

El capítulo enseña a convertir datos tabulares en visualizaciones útiles para análisis exploratorio (EDA) y comunicación. Presenta la relación entre `matplotlib`, la API de gráficos de `pandas` y `pandas.plotting`; explica la estructura `Figure`/`Axes`; y ofrece criterios para visualizar evolución temporal, relaciones entre variables, distribuciones, conteos, frecuencias, autocorrelación e incertidumbre.

Para Machine Learning, su valor principal es diagnóstico: permite detectar forma de distribuciones, posibles outliers, transformaciones útiles, variables correlacionadas, densidades ocultas por solapamiento y dependencia temporal antes de modelar. El capítulo no cubre entrenamiento, validación ni métricas predictivas.

## 2. Conceptos esenciales

- **Capa de visualización:** `matplotlib` aporta la infraestructura de bajo nivel. `pandas` envuelve parte de esa funcionalidad con `Series.plot()` y `DataFrame.plot()`, reduciendo código a cambio de menor flexibilidad. Muchos ajustes posteriores se realizan sobre objetos de Matplotlib.
- **Jerarquía de objetos:** `Figure` es el contenedor superior; contiene uno o más objetos `Axes`, y estos contienen líneas, marcas, etiquetas y demás elementos. Conservar referencias a `Figure` y `Axes` permite modificar o guardar gráficos concretos, no solo el último gráfico activo.
- **Estado frente a objetos:** `plt.figure()` crea una figura; `plt.subplots()` crea y devuelve `(Figure, Axes)`; `fig.savefig()` guarda una figura específica; `plt.close()` libera recursos.
- **API de pandas:** `plot(kind=...)` selecciona el gráfico. Entre los parámetros transversales están `x`, `y`, `ax`, `subplots`, `layout`, `figsize`, `title`, `legend`, `style`, `color`, `colormap`, `logx`, `logy`, `loglog`, límites, rotación, ejes compartidos, tamaño de fuente y cuadrícula.
- **Evolución temporal:** un gráfico de líneas conecta observaciones ordenadas y es apropiado cuando la continuidad entre puntos tiene sentido. Un gráfico de barras evita sugerir interpolación cuando se muestran totales o conteos por periodos discretos.
- **Relaciones entre variables:** un diagrama de dispersión (scatter plot) ayuda a detectar correlación y relaciones no lineales. La transparencia (`alpha`) revela parcialmente el solapamiento; un gráfico hexagonal (`hexbin`) resume la densidad como un histograma bidimensional cuando hay demasiados puntos.
- **Correlación:** una matriz de correlación con mapa de color divergente muestra magnitud y dirección. Es una herramienta exploratoria; una asociación visual o un coeficiente alto no establece causalidad.
- **Distribuciones:** el histograma depende de la elección de intervalos (`bins`); una estimación de densidad por kernels (KDE) aproxima una función de densidad; una ECDF muestra la proporción observada menor o igual que cada valor; un box plot resume cuartiles y posibles outliers, pero oculta densidad y detalles de forma.
- **Conteos y categorías:** las barras son adecuadas para frecuencias o agregados discretos. Las barras horizontales facilitan etiquetas largas; las verticales pueden funcionar mejor con muchas categorías compactas. El capítulo desaconseja los gráficos circulares por su dificultad de lectura.
- **Diagnóstico temporal:** un `lag_plot` compara una serie con una versión rezagada; un patrón indica dependencia. `autocorrelation_plot()` muestra la correlación para múltiples rezagos y bandas de confianza.
- **Bootstrap gráfico:** `bootstrap_plot()` toma muestras con reemplazo y visualiza la variabilidad de estadísticas resumen como media, mediana y rango medio (`midrange`).
- **Contenido explícito del autor:** las visualizaciones deben usarse durante el EDA, no solo al comunicar conclusiones; escoger el tipo de gráfico adecuado es parte esencial del análisis.

## 3. Flujo o procedimiento presentado

1. **Cargar y tipar los datos.** Parsear fechas y, cuando resulte natural, usar el tiempo como índice.
2. **Definir la pregunta visual.** Determinar si se busca evolución, relación, distribución, conteo, frecuencia, dependencia temporal o incertidumbre.
3. **Elegir el gráfico según la estructura de los datos.** Usar línea para continuidad temporal; dispersión o `hexbin` para relaciones; histograma, KDE, ECDF o box plot para distribuciones; y barras para categorías o periodos discretos.
4. **Preparar solo las transformaciones necesarias.** Ejemplos del capítulo: logaritmo del volumen, diferencia `high - low`, discretización con `pd.cut()`, redondeo hacia abajo con `np.floor()`, agregación con `groupby()` o `resample()`, y tabla pivote para barras apiladas.
5. **Crear `Figure` y `Axes` cuando se necesite control.** Usar `plt.subplots()` para composiciones regulares; `add_gridspec()` y `add_subplot()` para tamaños desiguales; `add_axes()` para posiciones manuales o gráficos insertados.
6. **Dibujar con pandas o Matplotlib.** Reutilizar el mismo `Axes` mediante `ax=` cuando se superponen distribuciones o capas comparables.
7. **Comprobar que la codificación visual no distorsiona.** Probar más de un valor de `bins`, revisar escalas lineales frente a logarítmicas, controlar solapamiento y comprobar que los ejes comparados tienen escalas compatibles.
8. **Interpretar en el contexto de la muestra.** Una ECDF empírica describe los datos observados; extrapolarla a la población requiere que la muestra sea representativa.
9. **Guardar y liberar recursos.** Guardar antes de `plt.show()` cuando se usa el flujo con estado; preferir `fig.savefig()` si se conserva la referencia; aplicar `tight_layout()` si se cortan etiquetas; cerrar figuras cuando ya no se necesitan.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Variable medida a lo largo del tiempo con continuidad significativa | Gráfico de líneas | Hace visible tendencia, variación y comparación entre series | No conectar categorías o agregados discretos si la línea implicaría interpolación |
| Conteos o totales por mes/categoría | Barras | Compara magnitudes discretas sin sugerir valores intermedios | Ordenar categorías y elegir orientación legible |
| Relación entre dos variables numéricas con pocos puntos | Dispersión | Expone dirección, forma y no linealidad | El solapamiento puede ocultar concentración |
| Dispersión con solapamiento moderado | Dispersión con `alpha < 1` | Hace visibles zonas con puntos superpuestos | La percepción de densidad sigue siendo aproximada |
| Relación bivariada con alta densidad | `hexbin` | Resume conteos en una cuadrícula bidimensional | `gridsize` altera el nivel de agregación; debe probarse un equilibrio |
| Variable positiva muy sesgada en un eje | Escala logarítmica (`logx`/`logy`) | Puede revelar estructura comprimida por la escala original | No confundir cambio de escala visual con transformación almacenada de los datos |
| Inspección conjunta de muchas variables numéricas | `scatter_matrix()` | Muestra cada combinación y la distribución diagonal | Se vuelve difícil de leer con demasiadas variables; no sustituye evaluación predictiva |
| Forma de una distribución discreta o conteos por intervalo | Histograma | Resume frecuencia por intervalos | La conclusión puede cambiar con `bins`; revisar varias resoluciones |
| Forma suavizada de una variable continua | KDE | Estima la densidad y facilita comparar forma | Es una estimación; al superponerla a un histograma usar `density=True` |
| Probabilidad acumulada empírica o cuantiles visuales | ECDF | Permite leer la proporción `P(X <= x)` sin elegir bins | Generalizar a población solo si la muestra es representativa |
| Comparar mediana, cuartiles y posibles outliers | Box plot | Resume la distribución con cinco números | Oculta multimodalidad y densidad interna |
| Categorías con nombres largos | Barras horizontales | Reserva espacio para las etiquetas | Mantener orden de lectura y limitar categorías |
| Composición de una categoría por otra | Barras apiladas | Muestra total y contribución de subgrupos | Un grupo dominante puede ocultar los pequeños |
| Comparar proporciones dentro de cada grupo | Barras apiladas normalizadas | Elimina el efecto del tamaño total y muestra composición | Ya no comunica conteos absolutos; considerar mostrar ambos gráficos |
| Dependencia de una serie respecto de un rezago concreto | `lag_plot(..., lag=k)` | Revela patrón entre `x_t` y `x_(t-k)` | El rezago corresponde a observaciones, no necesariamente a tiempo calendario |
| Dependencia a través de muchos rezagos | `autocorrelation_plot()` | Resume autocorrelación por lag con bandas de confianza | No demuestra por sí sola estacionariedad ni causalidad |
| Incertidumbre de estadísticas resumen | `bootstrap_plot()` | Visualiza variación bajo remuestreo con reemplazo | Depende del tamaño y número de muestras; no es validación de modelo |

## 5. Métodos y técnicas

### `plt.subplots()` y composición de figuras

- **Objetivo:** crear una figura y una cuadrícula de ejes controlables.
- **Entradas:** número de filas y columnas; opcionalmente `figsize`.
- **Salida:** tupla `(fig, axes)`.
- **Funcionamiento:** cada `Axes` actúa como destino de uno o más gráficos. `add_gridspec()` permite asignar áreas de tamaño desigual; `add_axes()` acepta `[left, bottom, width, height]` como proporciones de la figura.
- **Hiperparámetros relevantes:** forma de la cuadrícula, `figsize`, ejes compartidos.
- **Ventajas:** control explícito, reutilización, superposición y guardado fiable.
- **Limitaciones:** más código que una llamada directa a `DataFrame.plot()`.
- **Casos adecuados:** comparaciones lado a lado, distribuciones superpuestas o diseños irregulares.
- **No conviene:** para un gráfico aislado si la llamada de pandas ya ofrece todo el control necesario.

### `Series.plot()` y `DataFrame.plot()`

- **Objetivo:** generar visualizaciones comunes desde objetos pandas.
- **Entradas:** columnas o índice, `kind` y opciones de presentación.
- **Salida:** normalmente un `Axes` o un arreglo de ejes si se solicitan subplots.
- **Funcionamiento:** pandas delega el dibujo en Matplotlib.
- **Hiperparámetros relevantes:** `kind`, `x`, `y`, `ax`, `subplots`, `layout`, `figsize`, `style`, `alpha`, `bins`, `gridsize`, `stacked`, escalas logarítmicas y ejes compartidos.
- **Ventajas:** código breve e integración directa con índice y columnas.
- **Limitaciones:** menor flexibilidad; algunos gráficos de `pandas.plotting` devuelven composiciones difíciles de personalizar.
- **Casos adecuados:** EDA rápido y gráficos estándar.
- **No conviene:** cuando se requiere una composición o personalización no expuesta por el wrapper.

### Dispersión, transparencia y `hexbin`

- **Objetivo:** estudiar relación y concentración entre dos variables numéricas.
- **Entradas:** `x`, `y`; para dispersión, `alpha`; para `hexbin`, `gridsize` y `colormap`.
- **Salida:** gráfico bivariado sobre un `Axes`.
- **Funcionamiento:** scatter dibuja cada observación; `hexbin` agrupa observaciones en celdas hexagonales y codifica el conteo mediante color.
- **Hiperparámetros:** `alpha` entre 0 y 1; `gridsize` escalar o tupla; escala lineal/logarítmica.
- **Ventajas:** scatter conserva observaciones; `hexbin` revela densidad que el solapamiento oculta.
- **Limitaciones:** scatter satura con muchos datos; `hexbin` pierde detalle individual y depende del tamaño de celda.
- **Casos adecuados:** relaciones de features, diagnóstico de transformaciones y heterogeneidad.
- **No conviene:** para variables categóricas sin codificación significativa.

### Matriz de correlación y `scatter_matrix()`

- **Objetivo:** inspeccionar relaciones por pares y redundancia potencial entre variables.
- **Entradas:** columnas numéricas; en `scatter_matrix`, `figsize` y `diagonal='hist'` o `'kde'`.
- **Salida:** mapa matricial de coeficientes o cuadrícula de dispersiones.
- **Funcionamiento:** la matriz de correlación codifica coeficientes con color; la matriz de dispersión cruza cada variable con todas las demás y coloca una distribución en la diagonal.
- **Ventajas:** cribado visual rápido de relaciones fuertes, no linealidad y variables posiblemente útiles.
- **Limitaciones:** crece cuadráticamente con el número de columnas; correlación lineal no captura toda dependencia ni causalidad.
- **Casos adecuados:** EDA previo a selección o ingeniería de features.
- **No conviene:** como única justificación para descartar variables o afirmar importancia predictiva.

### Histograma, KDE, ECDF y box plot

- **Objetivo:** describir forma, rango, concentración, cuantiles y colas de una variable.
- **Entradas:** serie numérica; `bins` para histograma; datos continuos para KDE; observaciones para ECDF; opcionalmente grupos para box plots.
- **Salida:** frecuencias/densidad, densidad estimada, probabilidad acumulada empírica o resumen de cinco números.
- **Funcionamiento:** histograma agrupa; KDE suaviza; ECDF ordena observaciones y acumula proporciones; box plot resume cuartiles y extremos/outliers potenciales.
- **Hiperparámetros:** `bins`, normalización `density`, transparencia y, aunque el capítulo no lo ajusta, el suavizado implícito de KDE.
- **Ventajas:** son vistas complementarias de la distribución.
- **Limitaciones:** histograma depende de bins; KDE puede ocultar detalle; box plot oculta densidad; ECDF exige interpretación acumulada.
- **Casos adecuados:** sesgo, colas, multimodalidad, rangos por grupo y umbrales percentiles.
- **No conviene:** interpretar cualquiera de ellos aisladamente cuando la decisión depende de detalles que ese gráfico no muestra.

### Barras simples, apiladas y normalizadas

- **Objetivo:** comparar conteos, totales o composición por categoría.
- **Entradas:** serie agregada o tabla pivote; `kind='bar'/'barh'`; `stacked=True`.
- **Salida:** barras verticales u horizontales.
- **Funcionamiento:** `value_counts()`, `groupby()` o `pivot_table()` prepara los valores; dividir cada fila por su suma produce proporciones para apilado normalizado.
- **Hiperparámetros:** orientación, orden, rotación, apilamiento y normalización.
- **Ventajas:** lectura directa de magnitudes discretas; la normalización revela composición.
- **Limitaciones:** apilar dificulta comparar segmentos sin línea base común; normalizar elimina información de volumen.
- **Casos adecuados:** distribución de clases, frecuencia de categorías y composición de subgrupos.
- **No conviene:** demasiadas categorías o segmentos sin una selección/agrupación previa.

### `lag_plot()` y `autocorrelation_plot()`

- **Objetivo:** detectar dependencia temporal.
- **Entradas:** una serie ordenada; `lag=k` para `lag_plot()`.
- **Salida:** dispersión entre observaciones separadas por `k` posiciones o curva de autocorrelación por múltiples rezagos.
- **Funcionamiento:** el lag plot empareja `x_t` con `x_(t-k)`; la autocorrelación mide la relación de la serie con versiones rezagadas. El gráfico incluye bandas de confianza del 95% y 99%.
- **Hiperparámetros:** rezago; orden y frecuencia efectiva de la serie.
- **Ventajas:** diagnóstico rápido de señal temporal frente a ruido.
- **Limitaciones:** un patrón no explica su causa; en calendarios irregulares, un rezago observacional no equivale a una unidad fija de tiempo.
- **Casos adecuados:** forecasting, features rezagadas y evaluación inicial de dependencia.
- **No conviene:** aplicar sin ordenar por tiempo o ignorando huecos en la frecuencia.

### `bootstrap_plot()`

- **Objetivo:** evaluar visualmente la incertidumbre de estadísticas resumen.
- **Entradas:** serie; número de remuestras (`samples`) y tamaño de cada remuestra (`size`).
- **Salida:** distribuciones y secuencias de estadísticas bootstrap.
- **Funcionamiento:** extrae muestras con reemplazo y recalcula media, mediana y rango medio.
- **Hiperparámetros:** cantidad y tamaño de muestras.
- **Ventajas:** ofrece una vista empírica de estabilidad sin derivación paramétrica explícita.
- **Limitaciones:** refleja la muestra disponible y no corrige falta de representatividad o dependencia no tratada.
- **Casos adecuados:** comparar estabilidad de estadísticas descriptivas.
- **No conviene:** confundirlo con estimación de incertidumbre de un pipeline predictivo completo.

## 6. Fórmulas esenciales

### Función de distribución acumulada

\[
F_X(x) = P(X \leq x)
\]

- `X`: variable aleatoria.
- `x`: valor concreto de comparación.
- `F_X(x)`: probabilidad de observar un valor menor o igual que `x`, entre 0 y 1.
- **Interpretación:** en una ECDF, el eje vertical indica la proporción acumulada de observaciones hasta cada valor del eje horizontal.
- **Supuesto importante:** interpretar la ECDF como aproximación de la población requiere una muestra representativa.

### Normalización por fila para barras apiladas

\[
p_{ij} = \frac{c_{ij}}{\sum_j c_{ij}}
\]

- `c_ij`: conteo de la subcategoría `j` dentro del grupo `i`.
- `p_ij`: proporción de esa subcategoría dentro del grupo.
- **Qué mide:** composición relativa del grupo; cada fila suma 1.
- **Interpretación:** facilita comparar mezcla de categorías aun cuando los tamaños totales difieren.
- **Precaución:** se pierde la comparación de conteos absolutos.

### Emparejamiento de un rezago

\[
(x_t, x_{t-k})
\]

- `x_t`: observación en la posición temporal `t`.
- `k`: número de periodos u observaciones de rezago.
- `x_(t-k)`: observación previa emparejada.
- **Qué mide:** un lag plot representa estos pares; una estructura visible sugiere dependencia temporal.
- **Supuesto importante:** la serie debe estar correctamente ordenada y el significado de `k` depende de su frecuencia observacional.

## 7. Implementación práctica

Los ejemplos siguientes condensan las operaciones del capítulo. No constituyen un pipeline de entrenamiento. Cuando una visualización se use para decidir transformaciones o features, esa decisión debe evaluarse después con una partición de validación independiente.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

prices = pd.read_csv(
    "data/fb_stock_prices_2018.csv",
    index_col="date",
    parse_dates=True,
)

fig, ax = plt.subplots(figsize=(10, 4))
prices["open"].plot(ax=ax, title="Evolution of opening price", legend=False)
ax.set_ylabel("price")
fig.tight_layout()
```

Relación bivariada con densidad:

```python
plot_data = prices.assign(
    log_volume=np.log(prices["volume"]),
    daily_range=prices["high"] - prices["low"],
)

ax = plot_data.plot.hexbin(
    x="log_volume",
    y="daily_range",
    gridsize=20,
    cmap="gray_r",
    sharex=False,
)
ax.set_title("Daily range vs. log(volume)")
```

Comparación consistente de histograma y KDE:

```python
fig, ax = plt.subplots(figsize=(8, 4))
prices["high"].plot.hist(ax=ax, density=True, alpha=0.4, bins=20)
prices["high"].plot.kde(ax=ax)
ax.set_xlabel("daily high price")
fig.tight_layout()
```

Barras apiladas normalizadas:

```python
pivot = (
    earthquakes.assign(magnitude_bin=lambda frame: np.floor(frame["mag"]))
    .pivot_table(
        index="magnitude_bin",
        columns="magType",
        values="mag",
        aggfunc="count",
    )
    .fillna(0)
)

row_totals = pivot.sum(axis=1)
normalized = pivot.div(row_totals, axis=0)
normalized.plot.bar(stacked=True, rot=0, figsize=(10, 5))
```

Diagnóstico de dependencia temporal:

```python
from pandas.plotting import autocorrelation_plot, lag_plot

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lag_plot(prices["close"], lag=1, ax=axes[0])
autocorrelation_plot(prices["close"], ax=axes[1])
fig.tight_layout()
```

**Nota práctica moderna (no presentada como recomendación original del libro):** conviene preferir referencias explícitas `fig, ax = plt.subplots()` y métodos de `Axes` en código reutilizable, y cerrar figuras en procesos por lotes. Si `DataFrame.corr()` encuentra columnas no numéricas en versiones actuales de pandas, seleccionar primero las columnas numéricas o indicar el comportamiento numérico explícitamente. Estas adaptaciones conservan la idea del capítulo sin atribuirle APIs posteriores.

**Prevención de leakage, síntesis práctica derivada:** el EDA completo sobre la variable objetivo o sobre todo el dataset puede influir en decisiones de features y transformaciones. En una competencia, fijar la partición de validación antes de convertir observaciones visuales en decisiones de modelado; ajustar transformaciones aprendidas únicamente con entrenamiento. El capítulo no formula esta regla, pero sus gráficos sirven como diagnóstico que puede afectar el pipeline.

## 8. Supuestos y condiciones de aplicación

1. Las columnas deben tener tipos compatibles con el gráfico: numéricas para correlación, densidad, scatter y autocorrelación; categorías o agregados para barras; fechas ordenadas para evolución temporal.
2. El índice temporal debe estar ordenado. Un rezago cuenta posiciones observadas; en datos bursátiles, `lag=5` representa cinco sesiones, no necesariamente cinco días calendario.
3. Una escala logarítmica requiere valores positivos en el eje transformado.
4. Histogramas y hexbins requieren una discretización visual adecuada (`bins` o `gridsize`); no existe una resolución neutral en todos los casos.
5. Para superponer histograma y KDE en el mismo eje vertical, el histograma debe usar densidad (`density=True`).
6. Comparar series en un mismo eje presupone escalas compatibles; de lo contrario, usar subplots o ejes separados.
7. La interpretación poblacional de ECDF y bootstrap depende de que los datos observados sean informativos y suficientemente representativos.
8. Una matriz de correlación describe asociación, principalmente lineal en el uso mostrado; no establece causalidad ni utilidad fuera de muestra.
9. Los box plots resumen cuartiles y posibles valores extremos, pero no conservan forma completa, densidad ni orden temporal.
10. Las barras normalizadas requieren totales de fila distintos de cero.

## 9. Errores y riesgos

- **Elegir bins convenientes:** cambiar `bins` puede hacer que una distribución parezca unimodal o bimodal. Revisar sensibilidad antes de concluir.
- **Ocultar densidad por sobreimpresión:** un scatter opaco puede parecer disperso aunque concentre muchos puntos. Probar `alpha` o `hexbin`.
- **Interpretar una línea entre categorías:** conectar puntos discretos sugiere continuidad inexistente. Preferir barras para conteos y agregados discretos.
- **Confundir escala con transformación:** `logx=True` cambia la representación del eje; no crea una feature logarítmica en el dataframe.
- **Comparar histograma y KDE en escalas distintas:** sin `density=True`, la curva puede quedar visualmente aplastada.
- **Sobreinterpretar correlación:** patrones de scatter, mapas de correlación y lag plots son exploratorios; no prueban causalidad ni rendimiento predictivo.
- **Ignorar la resolución de `hexbin`:** una cuadrícula demasiado fina produce celdas difíciles de leer; una demasiado gruesa borra estructura.
- **Perder conteos al normalizar:** una barra apilada al 100% hace comparables las proporciones, pero puede ocultar que los grupos tienen tamaños muy distintos.
- **Usar demasiadas variables en `scatter_matrix`:** el número de paneles crece rápidamente y la lectura se degrada.
- **Interpretar lag como calendario:** datos con fines de semana, huecos o frecuencia irregular rompen la equivalencia entre posiciones y duración.
- **Generalizar una ECDF sin representatividad:** el capítulo condiciona explícitamente la lectura poblacional a que la distribución observada represente a la población.
- **Guardar después de `plt.show()`:** en el flujo descrito, `show()` puede cerrar la figura y producir un archivo vacío. Guardar antes o usar `fig.savefig()`.
- **No cerrar figuras:** conservar muchas figuras abiertas desperdicia memoria; usar `plt.close(fig)` o `plt.close("all")` cuando corresponda.
- **Etiquetas recortadas:** nombres largos pueden salir cortados al guardar. Aplicar `tight_layout()` antes de exportar.
- **Data leakage (síntesis práctica derivada):** usar patrones del conjunto de validación/test para escoger transformaciones, bins o features puede contaminar el proceso de modelado, aunque el capítulo no discute validación predictiva.
- **Overfitting visual (síntesis práctica derivada):** reaccionar a cada patrón aparente en una muestra pequeña puede producir ingeniería de features específica del ruido. Confirmar mejoras fuera de muestra.
- **Costo computacional:** scatter matrices escalan aproximadamente con el cuadrado del número de variables; scatter plots masivos y muchas figuras abiertas consumen tiempo y memoria.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo:** al iniciar EDA, al decidir transformaciones, al detectar dependencia temporal o cuando una tabla de estadísticas no explica la forma de los datos.
- **Decisiones que permite tomar:** línea frente a barras; scatter frente a `hexbin`; escala original frente a logarítmica; histograma/KDE/ECDF/box plot según la pregunta; conteos frente a proporciones; rezagos que merecen validación posterior.
- **Baseline visual mínimo:** distribución del target y features principales; frecuencia de clases/categorías; scatter o correlación para relaciones numéricas; línea y autocorrelación si existe tiempo.
- **Mejoras potenciales:** transformar variables fuertemente sesgadas; crear features de rango o logaritmos sugeridas por el dominio; explorar rezagos; identificar variables redundantes; estratificar análisis por grupos.
- **Qué no implementar bajo presión:** composiciones complejas con `add_axes()`/`GridSpec`, matrices de dispersión enormes, personalización estética extensa o múltiples gráficos redundantes. No dedicar tiempo a gráficos circulares.
- **Señales activadoras:** cola larga -> histograma más KDE y posible escala log; solapamiento intenso -> `alpha` o `hexbin`; categorías largas -> `barh`; grupo dominante -> vista normalizada además de conteos; timestamp -> evolución y lags; outliers aparentes -> box plot más una vista de densidad; correlaciones fuertes -> revisar redundancia sin asumir causalidad.
- **Separación entrenamiento/validación:** fijar primero una estrategia de validación. Usar EDA para proponer, no para confirmar, decisiones predictivas; confirmar cualquier feature o transformación con la métrica del problema.

## 11. Checklist operativo

- [ ] Formular la pregunta: evolución, relación, distribución, conteo, dependencia o incertidumbre.
- [ ] Verificar tipos, nulos, orden temporal y frecuencia efectiva.
- [ ] Elegir el gráfico que no sugiera continuidad o precisión inexistentes.
- [ ] Probar sensibilidad a `bins`, `gridsize`, transparencia y escala logarítmica.
- [ ] Mostrar conteos absolutos además de proporciones cuando ambos importen.
- [ ] Combinar box plot con histograma, KDE o ECDF si se necesita conocer la forma.
- [ ] Tratar correlación y autocorrelación como señales exploratorias, no como causalidad.
- [ ] Fijar validación antes de convertir hallazgos visuales en features.
- [ ] Etiquetar ejes, título y unidades; revisar que no se corten al guardar.
- [ ] Guardar mediante la referencia de `Figure` y cerrar figuras en procesos por lotes.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Selección de gráfico por tipo de pregunta y dato | `knowledge/eda/visualization_selection.md` | Alta |
| Histogramas, sensibilidad a bins, KDE, ECDF y box plots | `knowledge/eda/distribution_and_relationship_plots.md` | Alta |
| Scatter, transparencia y hexbin para concentración bivariada | `knowledge/eda/distribution_and_relationship_plots.md` | Alta |
| Matrices de correlación y scatter matrices | `knowledge/feature_engineering/correlation_diagnostics.md` | Alta |
| Lag plots y autocorrelación | `knowledge/time_series/visual_diagnostics.md` | Alta |
| Conteos, barras apiladas y proporciones normalizadas | `knowledge/eda/categorical_visualization.md` | Media |
| Bootstrap de estadísticas resumen | `knowledge/metrics/bootstrap_uncertainty.md` | Media |
| Gestión de Figure/Axes, subplots y exportación | `knowledge/eda/matplotlib_workflow.md` | Media |
| Riesgo de leakage al convertir EDA en decisiones de modelado | `knowledge/validation/data_leakage.md` | Media |

## 13. Trazabilidad

La numeración “p. impresa” corresponde al número visible del libro; “p. PDF” corresponde a este archivo recortado.

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura y temas del capítulo, p. impresa 257-258 (p. PDF 1-2) |
| 2. Capas Matplotlib/pandas y conceptos Figure/Axes | “An introduction to matplotlib”, “The basics” y “Plot components”, p. impresa 259-270 (p. PDF 3-14) |
| 2-3. API de `plot()` y flujo general | “Plotting with pandas”, p. impresa 271-276 (p. PDF 15-20) |
| 4-5. Evolución temporal | “Evolution over time”, p. impresa 273-276 (p. PDF 17-20) |
| 4-5. Scatter, log, transparencia y hexbin | “Relationships between variables”, p. impresa 277-280 (p. PDF 21-24) |
| 4-5. Matriz de correlación | “Relationships between variables”, p. impresa 281-282 (p. PDF 25-26) |
| 4-7. Histogramas, KDE, ECDF y box plots | “Distributions”, p. impresa 283-290 (p. PDF 27-34) |
| 6. Fórmula de CDF | Recuadro y ejemplo ECDF, p. impresa 287-288 (p. PDF 31-32) |
| 4-7. Barras, conteos, apilamiento y normalización | “Counts and frequencies”, p. impresa 291-299 (p. PDF 35-43) |
| 5. `scatter_matrix()` | “The pandas.plotting subpackage - Scatter matrices”, p. impresa 300-302 (p. PDF 44-46) |
| 5-7. Lag plots | “Lag plots”, p. impresa 303-304 (p. PDF 47-48) |
| 5-7. Autocorrelación | “Autocorrelation plots”, p. impresa 305-306 (p. PDF 49-50) |
| 5. Bootstrap | “Bootstrap plots”, p. impresa 306-307 (p. PDF 50-51) |
| 8-9. Supuestos, precauciones y riesgos explícitos | Advertencias sobre bins, escalas, ECDF, guardado y cierre de figuras, p. impresa 262-270, 277-299 (p. PDF 6-14, 21-43) |
| 10-12. Aplicación, checklist y mapeo | Síntesis práctica derivada de todo el capítulo, especialmente el resumen de p. impresa 307 (p. PDF 51) |

**Cobertura:** se revisaron las 53 páginas del PDF, que abarcan las páginas impresas 257-309. No hubo páginas ilegibles ni secciones que requirieran completar vacíos por suposición. Los ejercicios y la bibliografía se revisaron para delimitar el capítulo, pero no se incorporaron como conocimiento técnico adicional.
