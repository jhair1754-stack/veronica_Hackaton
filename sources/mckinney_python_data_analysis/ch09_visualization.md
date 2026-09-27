---
title: "Gráficos y visualización"
topics:
  - "Visualización con matplotlib"
  - "Gráficos con pandas y seaborn"
  - "Distribuciones, relaciones y datos categóricos"
  - "Composición, anotación y exportación de figuras"
target_knowledge_files:
  - "knowledge/eda/visualization_selection.md"
  - "knowledge/eda/distribution_plots.md"
  - "knowledge/eda/relationship_plots.md"
  - "knowledge/eda/categorical_visualization.md"
  - "knowledge/interpretability/model_result_visualization.md"
---

# 9. Gráficos y visualización

## 1. Propósito del capítulo

**Contenido explicado por el autor.** El capítulo introduce la visualización de datos con `matplotlib` y sus integraciones de mayor nivel en pandas y seaborn. Explica cómo construir y organizar figuras, personalizar ejes y leyendas, guardar resultados y elegir gráficos básicos para series, distribuciones, relaciones entre variables y comparaciones categóricas (pp. 398-440).

**Conclusión práctica derivada.** Para Machine Learning, estas herramientas permiten detectar valores atípicos, escalas problemáticas, distribuciones multimodales, relaciones entre variables y diferencias entre grupos antes de modelar. También ayudan a comunicar resultados, pero un gráfico exploratorio no sustituye una medición ni demuestra causalidad.

## 2. Conceptos esenciales

- **Figura y ejes.** En matplotlib, `Figure` es el contenedor global y cada `Axes` es una región de trazado. El capítulo recomienda trabajar con métodos del eje (`ax.plot`, `ax.hist`, `ax.scatter`) para controlar con precisión cada subgráfico (pp. 400-405).
- **Cuadrícula de subgráficos.** `plt.subplots(nrows, ncols)` crea conjuntamente la figura y un array de ejes. `sharex` y `sharey` permiten comparar paneles en una escala común; `subplots_adjust` controla el espacio entre ellos (pp. 404-406).
- **Codificación visual.** Color, estilo de línea, marcadores, transparencia (`alpha`) y estilo de dibujo distinguen series y hacen visibles los puntos observados. Una leyenda requiere etiquetas (`label`) y una llamada a `ax.legend()` (pp. 406-413).
- **Escalas y rotulación.** Los límites, marcas y etiquetas de los ejes se consultan o modifican con métodos `get_*` y `set_*`. Títulos, nombres de eje y etiquetas rotadas aportan contexto y legibilidad (pp. 409-413).
- **Anotaciones.** `text`, `annotate` y las figuras geométricas de `matplotlib.patches` permiten señalar eventos, regiones o elementos concretos del gráfico (pp. 413-416).
- **Exportación.** `Figure.savefig` deduce el formato por la extensión y admite opciones como `dpi`, color de fondo y formato explícito. SVG y PDF son formatos vectoriales; PNG permite controlar la resolución (pp. 416-417).
- **Configuración.** `plt.rc`, `plt.rcParams`, `plt.rcdefaults` y los archivos `matplotlibrc` permiten definir estilos globales. seaborn también utiliza el sistema de configuración de matplotlib (pp. 417-418).
- **Pandas como interfaz de alto nivel.** `Series.plot` y `DataFrame.plot` delegan en matplotlib. El índice se usa por defecto en el eje x; un `DataFrame` traza una línea por columna y genera una leyenda (pp. 418-422).
- **Gráficos de barras.** `plot.bar` y `plot.barh` representan valores categóricos; `stacked=True` apila componentes. `value_counts().plot.bar()` visualiza frecuencias. Una tabulación cruzada normalizada permite comparar proporciones entre grupos (pp. 422-428).
- **Resumen estadístico con seaborn.** `sns.barplot` agrega múltiples observaciones por categoría mediante la media y, en el ejemplo, muestra intervalos de confianza del 95 %. `hue` añade una variable categórica mediante color (pp. 428-430).
- **Histograma y densidad.** El histograma cuenta observaciones en intervalos discretos; un gráfico KDE aproxima una distribución continua como mezcla de kernels. El número de contenedores (`bins`) modifica la resolución visual del histograma (pp. 430-433).
- **Dispersión y regresión visual.** Un diagrama de dispersión examina la relación entre dos variables. `sns.regplot` añade una recta de regresión lineal; `sns.pairplot` construye una matriz de relaciones por pares con distribuciones en la diagonal (pp. 433-436).
- **Facetas.** Una cuadrícula de facetas divide los datos en paneles según los valores de variables categóricas. `sns.catplot` puede combinar ejes, color (`hue`), columnas (`col`) y filas (`row`) para comparar subgrupos (pp. 435-439).
- **Diagrama de cajas.** Resume mediana, cuartiles y valores atípicos, y resulta útil para comparar distribuciones entre categorías (pp. 438-439).
- **Herramientas interactivas.** El autor menciona Altair, Bokeh y Plotly para gráficos dinámicos orientados al navegador, mientras recomienda matplotlib, pandas y seaborn para gráficos estáticos (pp. 439-440).

## 3. Flujo o procedimiento presentado

1. Definir la pregunta visual: evolución, frecuencia, distribución, relación entre variables o comparación entre categorías.
2. Preparar los datos que se van a representar; cuando sea necesario, calcular frecuencias, proporciones, transformaciones o resúmenes.
3. Crear una figura y uno o varios ejes con `plt.subplots`, especificando tamaño y, si la comparación lo requiere, ejes compartidos.
4. Elegir el gráfico según la estructura de los datos: línea, barra, histograma/KDE, dispersión, matriz de pares, facetas o caja.
5. Dibujar sobre un eje explícito mediante `ax=...` o los métodos del objeto `Axes`.
6. Ajustar límites, marcas, etiquetas, título, leyenda, colores y transparencia para que la codificación sea interpretable.
7. Añadir anotaciones únicamente cuando ayuden a relacionar puntos o periodos con hechos relevantes.
8. Revisar escalas, solapamientos y legibilidad; modificar espaciado o rotación de etiquetas cuando sea necesario.
9. Guardar la figura con `fig.savefig`, seleccionando un formato y una resolución apropiados al medio de salida.

**Recomendación para competencias, derivada del capítulo.** Construir primero un conjunto pequeño de gráficos diagnósticos ligados a preguntas concretas. Comparar entrenamiento y validación/test solo con agregados permitidos y sin usar la variable objetivo del conjunto de evaluación para tomar decisiones de entrenamiento.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se observa una variable ordenada o temporal | Gráfico de líneas | Muestra evolución y cambios a lo largo del índice | La línea interpola entre puntos; usar marcadores si importa distinguir observaciones reales |
| Se comparan varios paneles en la misma escala | `plt.subplots(..., sharex=True, sharey=True)` | Facilita comparaciones directas | Verificar que compartir escala no oculte variación relevante |
| Se comparan valores o frecuencias por categoría | Barras verticales u horizontales | Hace explícita la magnitud por categoría | Muchas categorías o etiquetas largas pueden saturar el eje |
| Se comparan composiciones dentro de grupos | Barras apiladas con proporciones normalizadas | Muestra la fracción de cada componente | Los segmentos intermedios son difíciles de comparar; comprobar que cada fila sume 1 |
| Se quiere inspeccionar la forma de una distribución | Histograma | Muestra frecuencias por intervalo | El resultado depende de `bins`; revisar más de una resolución |
| Se busca una aproximación continua de la distribución | KDE / `plot.density` | Suaviza la distribución observada | El suavizado puede ocultar o crear apariencia de estructura |
| Se estudia la relación entre dos variables numéricas | Dispersión | Hace visibles asociación, dispersión y observaciones extremas | La superposición de puntos puede ocultar densidad; usar `alpha` |
| Se desea una referencia lineal sobre la dispersión | `sns.regplot` | Añade una recta de regresión al gráfico | Una recta visual no prueba causalidad ni garantiza linealidad adecuada |
| Se exploran varias variables numéricas a la vez | `sns.pairplot` | Reúne relaciones por pares y distribuciones marginales | Su coste y tamaño crecen rápidamente con el número de variables |
| Hay una o más variables categóricas de agrupación | `sns.catplot` con `hue`, `row` o `col` | Separa comparaciones por subgrupo | Demasiadas categorías producen una cuadrícula difícil de leer |
| Se comparan centro, dispersión y atípicos por categoría | Diagrama de cajas | Resume mediana, cuartiles y valores atípicos | No muestra por sí solo el tamaño de muestra ni toda la forma de la distribución |
| Se prepara una figura estática para publicación | matplotlib/pandas/seaborn + `savefig` | Ofrece control fino y formatos de salida comunes | Elegir DPI suficiente en formatos rasterizados |
| Se necesita interacción en navegador | Altair, Bokeh o Plotly | El capítulo las señala como opciones para gráficos dinámicos | Quedan fuera del desarrollo práctico del capítulo |

## 5. Métodos y técnicas

### 5.1 Figuras, ejes y subgráficos de matplotlib

- **Objetivo:** componer una o varias visualizaciones con control explícito de cada panel.
- **Entradas:** número de filas y columnas, tamaño de figura, datos x/y y opciones de estilo.
- **Salida:** un objeto `Figure` y uno o varios objetos `Axes`.
- **Funcionamiento:** `plt.subplots` crea la cuadrícula; cada eje recibe llamadas como `plot`, `hist` o `scatter`.
- **Hiperparámetros relevantes:** `nrows`, `ncols`, `figsize`, `sharex`, `sharey`, `wspace`, `hspace`.
- **Ventajas:** control preciso, composición flexible y base común para pandas y seaborn.
- **Limitaciones:** exige gestionar manualmente detalles de presentación y no detecta solapamientos de etiquetas.
- **Casos adecuados:** paneles comparativos, figuras para informes y gráficos que requieren personalización.
- **No conviene:** cuando basta un gráfico exploratorio inmediato y la personalización no aporta valor.

### 5.2 Gráficos de líneas

- **Objetivo:** representar una variable frente a un índice ordenado.
- **Entradas:** coordenadas x/y o una `Series`/`DataFrame` indexada.
- **Salida:** una o varias líneas, opcionalmente con marcadores y leyenda.
- **Funcionamiento:** conecta observaciones consecutivas; pandas usa el índice como x de forma predeterminada.
- **Hiperparámetros relevantes:** `color`, `linestyle`, `marker`, `drawstyle`, `label`, `use_index`, `xlim`, `ylim`.
- **Ventajas:** lectura rápida de tendencia y comparación de series.
- **Limitaciones:** la conexión visual puede sugerir continuidad donde solo hay puntos discretos.
- **Casos adecuados:** series temporales, curvas de aprendizaje y métricas por iteración.
- **No conviene:** categorías sin orden natural o nubes de observaciones independientes.

### 5.3 Barras, tablas de frecuencia y proporciones

- **Objetivo:** comparar magnitudes o composiciones entre categorías.
- **Entradas:** valores por categoría, conteos de `value_counts` o tablas de `pd.crosstab`.
- **Salida:** barras verticales, horizontales, agrupadas o apiladas.
- **Funcionamiento:** cada categoría se asigna a una barra; la normalización por suma de fila convierte conteos en proporciones.
- **Hiperparámetros relevantes:** orientación, `stacked`, `alpha`, color y orden de categorías.
- **Ventajas:** interpretación directa de conteos y comparaciones categóricas.
- **Limitaciones:** las barras apiladas dificultan comparar segmentos sin una línea base común.
- **Casos adecuados:** balance de clases, frecuencias de categorías y composición por grupo.
- **No conviene:** variables numéricas continuas sin discretización justificada.

### 5.4 Histogramas y estimación de densidad (KDE)

- **Objetivo:** examinar la distribución de una variable numérica.
- **Entradas:** observaciones unidimensionales.
- **Salida:** conteos por intervalo o una curva de densidad suavizada.
- **Funcionamiento:** el histograma divide el rango en contenedores; KDE aproxima una distribución continua mediante kernels.
- **Hiperparámetros relevantes:** `bins` en histogramas; las opciones de suavizado no se desarrollan en el capítulo.
- **Ventajas:** revela asimetría, colas, multimodalidad y valores extremos.
- **Limitaciones:** la apariencia depende de la discretización o el suavizado.
- **Casos adecuados:** inspección de variables y residuos, y decisión exploratoria sobre transformaciones.
- **No conviene:** variables categóricas nominales.

### 5.5 Dispersión, `regplot` y `pairplot`

- **Objetivo:** explorar relaciones entre variables numéricas.
- **Entradas:** dos columnas para dispersión/regresión o varias columnas para matriz de pares.
- **Salida:** nube de puntos, recta lineal opcional y distribuciones en la diagonal.
- **Funcionamiento:** `regplot` superpone un ajuste lineal; `pairplot` aplica gráficos a cada combinación de variables.
- **Hiperparámetros relevantes:** `x`, `y`, `diag_kind`, `plot_kws` y `alpha`.
- **Ventajas:** detecta patrones, no linealidad, agrupamientos y observaciones influyentes.
- **Limitaciones:** `pairplot` escala mal con muchas variables y la recta de `regplot` puede simplificar demasiado la relación.
- **Casos adecuados:** EDA de un conjunto pequeño de características numéricas.
- **No conviene:** matrices de alta dimensionalidad sin selección previa.

### 5.6 `barplot`, facetas y diagramas de cajas en seaborn

- **Objetivo:** comparar una variable entre grupos y subgrupos categóricos.
- **Entradas:** un `DataFrame` y nombres de columnas para x, y, color o facetas.
- **Salida:** estimaciones agregadas con incertidumbre, cuadrículas categóricas o resúmenes de caja.
- **Funcionamiento:** `barplot` agrega observaciones; `catplot` distribuye grupos mediante `hue`, `row` y `col`, y cambia el tipo con `kind`.
- **Hiperparámetros relevantes:** `hue`, `row`, `col`, `kind`, orientación y filtro de datos.
- **Ventajas:** expresa varias dimensiones categóricas con poco código.
- **Limitaciones:** demasiados grupos reducen legibilidad; el resumen agregado puede ocultar la distribución subyacente.
- **Casos adecuados:** comparación de métricas por clase, segmento o partición.
- **No conviene:** cardinalidad categórica alta o grupos con tamaños muy desiguales sin mostrar también su tamaño.

### 5.7 Anotación, configuración y exportación

- **Objetivo:** hacer una figura interpretable, consistente y reutilizable.
- **Entradas:** títulos, etiquetas, posiciones, estilos globales, ruta y formato de salida.
- **Salida:** figura rotulada y archivo gráfico.
- **Funcionamiento:** métodos `set_*`, `legend`, `annotate`, `plt.rc` y `savefig` modifican presentación y salida.
- **Hiperparámetros relevantes:** límites, marcas, rotación, `loc`, `dpi`, `facecolor`, `edgecolor`, `format`.
- **Ventajas:** mejora comunicación y permite salidas vectoriales o rasterizadas.
- **Limitaciones:** la personalización manual puede ser extensa y no corrige automáticamente todos los solapamientos.
- **Casos adecuados:** resultados finales, reportes reproducibles y figuras con eventos destacados.
- **No conviene:** decoración que no contribuya a interpretar los datos.

## 6. Fórmulas esenciales

### 6.1 Proporción dentro de cada grupo

\[
p_{ij} = \frac{c_{ij}}{\sum_j c_{ij}}
\]

- \(c_{ij}\): conteo de la categoría \(j\) dentro del grupo \(i\).
- \(p_{ij}\): fracción de la categoría \(j\) en el grupo \(i\).
- **Qué mide:** composición relativa de cada grupo.
- **Interpretación:** para cada grupo con suma positiva, \(\sum_j p_{ij}=1\).
- **Supuesto:** los conteos incluidos definen el universo que se desea comparar. En el ejemplo, el autor excluye categorías poco frecuentes antes de normalizar (pp. 426-428).

### 6.2 Porcentaje de propina sobre el importe sin propina

\[
\text{tip\_pct} = \frac{\text{tip}}{\text{total\_bill} - \text{tip}}
\]

- `tip`: importe de la propina.
- `total_bill`: total registrado en la factura.
- `tip_pct`: proporción de propina respecto del importe base usado por el ejemplo.
- **Qué mide:** tamaño relativo de la propina.
- **Interpretación:** valores mayores indican una propina más grande respecto de la cuenta sin propina.
- **Supuesto:** `total_bill - tip` es positivo y representa la base correcta para el análisis (pp. 428-429).

### 6.3 Diferencia logarítmica

\[
\Delta \log(x_t) = \log(x_t) - \log(x_{t-1})
\]

- \(x_t\): valor positivo de una variable en el instante \(t\).
- \(\Delta \log(x_t)\): cambio logarítmico entre observaciones consecutivas.
- **Qué mide:** cambio relativo aproximado para variaciones pequeñas.
- **Interpretación:** signo positivo indica aumento y signo negativo, disminución.
- **Supuesto:** los valores deben ser estrictamente positivos para aplicar el logaritmo. El capítulo usa `np.log(data).diff().dropna()` antes de explorar relaciones (pp. 433-435).

## 7. Implementación práctica

### 7.1 Distribución y comparación entre grupos

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(data=train_df, x="feature", bins=30, ax=axes[0])
sns.boxplot(data=train_df, x="target", y="feature", ax=axes[1])

axes[0].set(title="Distribución de feature", xlabel="feature")
axes[1].set(title="Feature por clase", xlabel="target", ylabel="feature")
fig.tight_layout()
```

**Adaptación moderna, no contenido original del libro.** Se usa una interfaz explícita `data=...` y ejes creados con `plt.subplots`; `tight_layout()` se incorpora como ajuste práctico de presentación. En una competencia, este código debe ejecutarse primero sobre entrenamiento. No debe usarse el objetivo real de test, si no está disponible legítimamente.

### 7.2 Frecuencias relativas por partición

```python
import pandas as pd

counts = pd.crosstab(train_df["split_group"], train_df["category"])
proportions = counts.div(counts.sum(axis="columns"), axis="index")

ax = proportions.plot.bar(stacked=True, figsize=(8, 4))
ax.set(title="Composición por grupo", ylabel="Proporción", ylim=(0, 1))
ax.legend(title="category")
```

Este patrón reproduce el procedimiento del capítulo: tabular, normalizar por fila y representar barras apiladas. **Conclusión práctica derivada:** es útil para comparar balance de clases o categorías entre folds, siempre que los grupos se hayan definido sin información futura.

### 7.3 Relaciones entre un conjunto pequeño de variables

```python
numeric_columns = ["feature_a", "feature_b", "feature_c"]

sns.pairplot(
    train_df[numeric_columns].dropna(),
    diag_kind="hist",
    plot_kws={"alpha": 0.25},
)
```

**Adaptación moderna, no contenido original del libro.** Se limita explícitamente el número de columnas y se usa histograma en la diagonal para reducir dependencias y coste. El capítulo muestra `diag_kind="kde"`.

### 7.4 Guardar una figura reproducible

```python
fig.savefig("eda_distribution.png", dpi=200, bbox_inches="tight")
```

`dpi` está explicado en el capítulo. **Adaptación moderna, no contenido original del libro:** `bbox_inches="tight"` se añade como nota práctica para reducir recortes o márgenes innecesarios.

## 8. Supuestos y condiciones de aplicación

1. Los ejes y unidades deben corresponder a la semántica real de las variables.
2. Un gráfico de líneas presupone un orden significativo entre observaciones; los segmentos conectan puntos consecutivos.
3. El logaritmo requiere valores positivos; `diff()` genera una ausencia inicial que el ejemplo elimina.
4. Las proporciones por fila requieren denominadores positivos y categorías comparables entre grupos.
5. Un histograma depende del ancho o número de contenedores; KDE depende del suavizado, aunque el capítulo no desarrolla su ajuste.
6. La recta de `regplot` representa un ajuste lineal y no implica causalidad.
7. Las comparaciones mediante ejes compartidos requieren unidades y escalas compatibles.
8. `barplot` resume múltiples observaciones por categoría; el lector debe interpretar la barra y su intervalo como un resumen, no como cada observación individual.
9. Las facetas son útiles cuando el número de niveles categóricos es manejable.
10. Los diagramas de cajas resumen la distribución y los atípicos, pero no sustituyen la inspección del tamaño de muestra.
11. Una resolución rasterizada adecuada depende del medio final; los formatos vectoriales son preferibles cuando se necesita escalado sin pérdida.

## 9. Errores y riesgos

- Usar una línea para categorías sin orden y sugerir continuidad inexistente.
- Comparar paneles con escalas automáticas distintas sin advertirlo; o compartir escalas incompatibles.
- Reducir a cero el espacio entre subgráficos y dejar etiquetas superpuestas; matplotlib no lo corrige por sí solo (pp. 405-406).
- Omitir títulos, unidades, etiquetas o leyendas y producir un gráfico imposible de interpretar fuera del notebook.
- Añadir `label` pero olvidar `ax.legend()`, o crear una leyenda ambigua.
- Elegir `bins` una sola vez y tratar la forma resultante del histograma como definitiva.
- Interpretar una KDE como observaciones reales o ignorar que es una estimación suavizada.
- Interpretar la línea de `regplot` como evidencia causal o pasar por alto relaciones no lineales.
- Usar `pairplot` con demasiadas columnas: el número de paneles crece cuadráticamente y puede agotar tiempo y memoria.
- Crear facetas con demasiadas categorías, paneles vacíos o grupos con pocos datos.
- Comparar barras agregadas sin revisar la distribución o el tamaño de cada grupo.
- Eliminar valores extremos solo porque aparecen en un gráfico; primero se debe comprobar si son errores o casos válidos.
- **Riesgo de leakage, derivado para ML:** explorar repetidamente el conjunto de test o usar sus etiquetas para decidir transformaciones, variables o hiperparámetros convierte el test en parte del proceso de ajuste.
- **Riesgo de validación, derivado para ML:** calcular transformaciones o límites con todos los datos antes de dividir puede transferir información. El capítulo muestra visualización y preparación, no un protocolo de validación.
- Guardar PNG con DPI insuficiente o con etiquetas recortadas.
- Modificar parámetros globales (`rcParams`) y afectar figuras posteriores sin intención.
- La extracción textual del PDF presenta artefactos tipográficos en caracteres acentuados, pero las páginas y el contenido técnico pudieron leerse; no se detectaron páginas ilegibles.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** al comenzar el EDA, al investigar una métrica inesperada, al comparar folds o segmentos y al preparar una explicación final.
- **Decisiones que permite tomar:** si una variable necesita transformación; si hay asimetría, multimodalidad o valores extremos; si dos variables parecen relacionadas; si las categorías o clases están desbalanceadas; y si el comportamiento cambia entre subgrupos.
- **Baseline:** histograma de variables numéricas clave, barras de la variable objetivo y categorías principales, dispersión de pares seleccionados y cajas por clase cuando la cardinalidad sea baja.
- **Mejoras posibles:** aplicar diferencias logarítmicas a series positivas cuando la pregunta sea sobre cambios relativos, comparar distribuciones por fold y usar facetas para detectar heterogeneidad entre segmentos.
- **Qué no implementar bajo presión:** matrices de pares con todas las variables, personalización estética extensa, anotaciones decorativas, figuras con demasiadas facetas o una biblioteca interactiva adicional si un gráfico estático responde la pregunta.
- **Señales activadoras:** colas largas o escalas muy diferentes activan histogramas/KDE; puntos superpuestos activan transparencia; muchas categorías activan barras horizontales o selección de niveles; relaciones entre dos variables activan dispersión; múltiples agrupadores categóricos activan facetas; observaciones extremas activan caja y revisión de datos.
- **Control de leakage:** ajustar decisiones usando únicamente entrenamiento y validación. El test final solo debe utilizarse según las reglas de la evaluación; si se grafica sin etiquetas, debe evitarse adaptar iterativamente el pipeline a patrones exclusivos del test.

## 11. Checklist operativo

- [ ] Definir la pregunta que debe responder cada gráfico.
- [ ] Confirmar tipo, unidad, orden y rango de las variables.
- [ ] Separar entrenamiento y validación antes de decisiones dependientes de datos.
- [ ] Revisar conteos y proporciones de la variable objetivo y categorías principales.
- [ ] Examinar distribuciones numéricas con histogramas y, cuando aporte valor, KDE o cajas.
- [ ] Probar más de un número de `bins` antes de concluir sobre la forma.
- [ ] Usar dispersión solo para pares relevantes y `alpha` si hay superposición.
- [ ] Limitar `pairplot` a pocas variables seleccionadas.
- [ ] Compartir ejes cuando la comparación requiera la misma escala.
- [ ] Añadir título, etiquetas, unidades y leyenda cuando correspondan.
- [ ] Revisar solapamientos, categorías vacías y tamaños de grupo.
- [ ] No eliminar atípicos sin verificar su origen.
- [ ] Guardar la figura con formato y resolución adecuados.
- [ ] Registrar qué decisión de modelado, si alguna, está respaldada por el gráfico.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Selección de gráfico según tipo de pregunta y variable | `knowledge/eda/visualization_selection.md` | Alta |
| Histogramas, KDE, cajas y sensibilidad a parámetros visuales | `knowledge/eda/distribution_plots.md` | Alta |
| Dispersión, regresión visual y matrices de pares | `knowledge/eda/relationship_plots.md` | Alta |
| Barras, proporciones, `hue` y cuadrículas de facetas | `knowledge/eda/categorical_visualization.md` | Alta |
| Comparación visual entre folds y control de leakage | `knowledge/validation/fold_diagnostics.md` | Alta |
| Diferencias logarítmicas para explorar cambios relativos | `knowledge/time_series/log_differences.md` | Media |
| Composición reproducible de figuras con matplotlib | `knowledge/eda/matplotlib_workflow.md` | Media |
| Comunicación y exportación de resultados de modelos | `knowledge/interpretability/model_result_visualization.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y papel de la visualización | Apertura del capítulo, pp. 398-399 |
| Figura, ejes y creación de subgráficos | 9.1, “Una breve introducción a la API matplotlib” y “Figuras y subgráficos”, pp. 399-405 |
| Espaciado, colores, marcadores y estilos de línea | 9.1, pp. 405-409 |
| Marcas, etiquetas, títulos y leyendas | 9.1, “Marcas, etiquetas y leyendas”, pp. 409-413 |
| Anotaciones y figuras geométricas | 9.1, “Anotaciones y dibujos en un subgráfico”, pp. 413-416 |
| Guardado y configuración de matplotlib | 9.1, “Almacenamiento de gráficos en archivo” y “Configuración de matplotlib”, pp. 416-418 |
| Gráficos de líneas con pandas | 9.2, “Gráficos de líneas”, pp. 418-422 |
| Gráficos de barras, frecuencias y proporciones | 9.2, “Gráficos de barras”, pp. 422-428 |
| `barplot`, estilos y paletas de seaborn | 9.2, pp. 428-430 |
| Histogramas y gráficos de densidad | 9.2, “Histogramas y gráficos de densidad”, pp. 430-433 |
| Dispersión, `regplot` y `pairplot` | 9.2, “Gráficos de dispersión o de puntos”, pp. 433-436 |
| Facetas, datos categóricos y cajas | 9.2, “Cuadrícula de facetas y datos categóricos”, pp. 435-439 |
| Otras bibliotecas y conclusión | 9.3-9.4, pp. 439-440 |
| Fórmula de proporciones por grupo | Ejemplo `crosstab` y normalización, pp. 426-428 |
| Fórmula de porcentaje de propina | Ejemplo de `tip_pct`, pp. 428-429 |
| Fórmula de diferencia logarítmica | Ejemplo `macrodata`, pp. 433-435 |
| Aplicación en coding test y controles de leakage | Síntesis práctica derivada de los procedimientos del capítulo; no es una recomendación explícita del autor |

