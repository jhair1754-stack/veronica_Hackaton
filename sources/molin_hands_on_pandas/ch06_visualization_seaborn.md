---
title: "Plotting with Seaborn and Customization Techniques"
topics:
  - "Visualización multivariada y categórica con Seaborn"
  - "Diagnóstico gráfico de correlación, regresión, residuos y distribuciones"
  - "Faceting, formato, anotaciones, referencias y color en Matplotlib"
target_knowledge_files:
  - "knowledge/eda/seaborn_visualization_selection.md"
  - "knowledge/eda/regression_and_residual_diagnostics.md"
  - "knowledge/eda/plot_formatting_and_annotation.md"
  - "knowledge/interpretability/visual_communication.md"
---

# 6. Plotting with Seaborn and Customization Techniques

## 1. Propósito del capítulo

El capítulo amplía la visualización exploratoria con Seaborn, especialmente para datos en formato largo (*long format*) y comparaciones entre subconjuntos. Presenta gráficos categóricos, mapas de calor, matrices de pares, gráficos conjuntos, regresión y residuos, distribuciones y facetas. Después muestra cómo hacer una figura interpretable mediante títulos, etiquetas, leyendas, escalas, marcas, líneas y regiones de referencia, anotaciones y mapas de color.

Para Machine Learning, su utilidad principal es diagnóstica y comunicativa: permite detectar solapamiento, relaciones entre variables, heterocedasticidad aparente, diferencias entre segmentos, forma de distribuciones y puntos o umbrales relevantes. El capítulo no enseña a entrenar ni validar modelos predictivos.

## 2. Conceptos esenciales

- **Seaborn sobre Matplotlib:** Seaborn simplifica gráficos estadísticos y el trabajo con datos largos; los objetos y funciones de Matplotlib siguen disponibles para guardar, ajustar y anotar las figuras.
- **Datos categóricos frente a numéricos:** `stripplot()` y `swarmplot()` muestran una variable numérica por categoría. El primero usa *jitter* para reducir solapamiento; el segundo intenta colocar los puntos sin que se superpongan y deja entrever la distribución.
- **Correlación y estructura bivariada:** `heatmap()` combina una matriz numérica con codificación por color; `annot=True` conserva los coeficientes y `center=0` hace que cero sea el punto medio visual. `pairplot()` muestra relaciones por pares y distribuciones marginales.
- **Gráfico conjunto (*joint plot*):** `jointplot()` combina una relación central con distribuciones marginales. El argumento `kind` cambia la vista entre dispersión, hexágonos, regresión, residuos y densidad KDE.
- **Diagnóstico de regresión:** `regplot()` dibuja datos y ajuste; `residplot()` muestra los residuos. Los residuos son `observado - predicho`. Si su dispersión cambia sistemáticamente con el predictor o se alejan de cero siguiendo un patrón, la relación lineal puede ser inadecuada.
- **Segmentación:** `lmplot()` permite separar regresiones mediante `hue`, `col` y `row`. `FacetGrid` generaliza esta idea: particiona por categorías y aplica una función a cada subconjunto con `map()`.
- **Distribuciones por grupo:** `boxenplot()` muestra más cuantiles que un box plot tradicional. `violinplot()` combina un resumen central con una KDE simétrica; aporta forma de distribución, pero esta parte es una estimación suavizada.
- **Jerarquía de títulos:** `plt.suptitle()` titula la figura completa; `Axes.set_title()` titula un subplot. `plt.title()` actúa sobre el eje activo y puede terminar en el último subplot.
- **Formato de ejes:** límites, escala, posiciones de marcas y formato de etiquetas son decisiones distintas. Los *locators* determinan dónde aparecen las marcas; los *formatters*, cómo se representan.
- **Referencias y anotaciones:** `axhline()`/`axvline()` marcan umbrales; `axhspan()`/`axvspan()` resaltan intervalos; `fill_between()` rellena entre curvas; `annotate()` conecta texto con coordenadas y puede incluir flechas.
- **Color con semántica:** una paleta cualitativa distingue grupos sin orden; una secuencial representa magnitud ordenada; una divergente destaca un centro significativo entre dos extremos. Una elección incorrecta puede sugerir orden o centralidad inexistentes.
- **Iterables y generadores:** `itertools.permutations()`, `cycle()`, `repeat()` y `product()` ayudan a construir múltiples gráficos o combinaciones de estilo sin materializar listas grandes. Un iterador se agota; un iterable reutilizable puede recorrerse de nuevo; un generador produce valores bajo demanda.
- **Contenido explícito del autor:** escoger el gráfico correcto y etiquetarlo con claridad es parte de hacerlo interpretable; el formato debe aplicarse antes de `plt.show()` o en la misma celda con el backend *inline*.

## 3. Flujo o procedimiento presentado

1. **Cargar y tipar los datos.** Parsear fechas cuando correspondan y conservar columnas categóricas y numéricas necesarias.
2. **Definir la pregunta visual.** Determinar si se busca comparar categorías, explorar correlaciones, evaluar una relación bivariada, revisar residuos, comparar distribuciones o contrastar subconjuntos.
3. **Preparar variables sin alterar innecesariamente la fuente.** El capítulo usa `assign()` para crear, por ejemplo, `log(volume)`, el rango diario `high - low` y el trimestre derivado del índice.
4. **Elegir la vista adecuada.** Usar `swarmplot()` si el solapamiento oculta puntos categóricos; `heatmap()` para una matriz; `pairplot()` para varias variables; `jointplot()` para dos; `regplot()`/`residplot()` para ajuste y diagnóstico; `boxenplot()`/`violinplot()` para distribuciones; facetas para subconjuntos.
5. **Comparar segmentos cuando la relación pueda variar.** Separar mediante `hue`, `row` o `col`; no asumir que un patrón agregado se mantiene en todos los grupos.
6. **Crear o recuperar los objetos `Figure` y `Axes`.** Usar el `Axes` como destino explícito cuando se superponen capas o se personalizan subplots.
7. **Añadir contexto mínimo suficiente.** Incorporar título de figura, títulos de subplots, etiquetas con unidades y una leyenda descriptiva cuando existan varias capas.
8. **Ajustar ejes y marcas.** Revisar límites para no inducir comparaciones engañosas, aplicar escala logarítmica solo si ayuda a la lectura y usar *locators*/*formatters* acordes con la variable.
9. **Añadir referencias solo si tienen significado.** Marcar umbrales, intervalos, curvas o eventos, y etiquetarlos sin tapar los datos.
10. **Elegir colores por el tipo de variable.** Cualitativos para categorías, secuenciales para magnitud y divergentes cuando existe un centro interpretable como cero.
11. **Verificar legibilidad y exportar.** Aplicar `tight_layout()`; si falla al guardar, usar `bbox_inches='tight'`. Confirmar que títulos, leyendas, anotaciones y escalas no ocultan ni distorsionan el patrón.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Variable numérica por categoría con pocos solapamientos | `sns.stripplot()` | Conserva cada observación y separa categorías | Ajustar `jitter` o `alpha` si los puntos se amontonan |
| Variable numérica por categoría con fuerte solapamiento | `sns.swarmplot()` | Reduce la superposición y muestra la concentración | Puede resultar pesado o congestionado con muestras grandes |
| Matriz de correlaciones | `sns.heatmap(..., annot=True, center=0)` | Une coeficientes y patrón cromático | Usar un mapa divergente; correlación no implica causalidad |
| Varias variables numéricas y pocos atributos | `sns.pairplot()` | Reúne relaciones por pares y distribuciones diagonales | Crece cuadráticamente y se vuelve ilegible con muchas columnas |
| Dos variables con marginales | `sns.jointplot()` | Muestra relación central y distribuciones laterales | Escoger `kind` según solapamiento y pregunta, no por estética |
| Dispersión bivariada muy solapada | `jointplot(kind='hex')` | Revela concentraciones que el scatter oculta | Agrega observaciones; ya no se ven puntos individuales |
| Evaluar si un ajuste lineal es razonable | `regplot()` junto con `residplot()` | Permite contrastar ajuste y estructura de errores | Una banda o línea convincente no sustituye la validación del modelo |
| Relación potencialmente distinta por segmento | `lmplot()` con `hue`, `col` o `row` | Compara ajustes por grupo | Grupos pequeños pueden producir patrones inestables |
| Comparar muchos cuantiles entre grupos | `boxenplot()` | Expone más estructura de colas que el box plot | No muestra por sí solo toda la densidad |
| Comparar forma y resumen de distribuciones | `violinplot()` | Combina KDE y resumen central | La KDE es una estimación y puede suavizar detalles |
| Aplicar el mismo gráfico a subconjuntos | `FacetGrid` + `map()` | Hace comparaciones consistentes entre facetas | Mantener escalas comparables y limitar el número de paneles |
| Título para toda una cuadrícula | `plt.suptitle()` o `fig.suptitle()` | Titula la figura, no solo el eje activo | `plt.title()` puede quedar en el último subplot |
| Valores que deben mostrarse como porcentajes | `ticker.PercentFormatter` | Separa datos de su representación textual | Configurar correctamente `xmax`; una base errónea produce porcentajes falsos |
| Eje que solo admite enteros | `ticker.MultipleLocator(base=1)` | Controla posiciones válidas de las marcas | Es un *locator*, no un *formatter* |
| Umbral horizontal o vertical | `axhline()` / `axvline()` | Hace visible un límite o punto de referencia | El umbral debe provenir del análisis, no elegirse para reforzar una narrativa |
| Intervalo relevante | `axhspan()` / `axvspan()` | Destaca una banda completa | Usar transparencia para no ocultar los datos |
| Diferencia entre dos curvas | `fill_between()` / `fill_betweenx()` | Codifica visualmente la separación | Las series deben estar alineadas en el eje compartido |
| Categorías sin orden | Mapa cualitativo | Distingue grupos sin sugerir jerarquía | Evitar gradientes que impliquen magnitud |
| Magnitud ordenada | Mapa secuencial | La intensidad representa progresión | Mantener dirección intuitiva y contraste suficiente |
| Centro significativo y dos extremos | Mapa divergente | Destaca desviaciones a ambos lados del centro | Centrar en el valor con significado real, como cero en correlación |

## 5. Métodos y técnicas

### `stripplot()` y `swarmplot()`

- **Objetivo:** comparar una variable numérica entre categorías conservando observaciones individuales.
- **Entradas:** `data`, columna categórica en `x` o `y`, columna numérica en el otro eje y, opcionalmente, una categoría adicional en `hue`.
- **Salida:** un `Axes` con puntos agrupados por categoría.
- **Funcionamiento:** `stripplot()` desplaza puntos ligeramente; `swarmplot()` calcula posiciones para reducir solapamiento.
- **Hiperparámetros relevantes:** `hue`, `jitter`, `alpha`, orientación, tamaño de marcador.
- **Ventajas:** muestran observaciones y permiten detectar concentración, rango y grupos.
- **Limitaciones:** el strip plot puede ocultar puntos; el swarm plot escala peor con muestras densas.
- **Casos adecuados:** EDA de una feature numérica frente a clase, fuente, tipo o segmento.
- **No conviene:** muestras masivas sin muestreo o agregación.

### `heatmap()` y `pairplot()`

- **Objetivo:** explorar relaciones entre múltiples columnas numéricas.
- **Entradas:** para `heatmap()`, una matriz como `DataFrame.corr()`; para `pairplot()`, un DataFrame y opcionalmente `hue`, `diag_kind` y `kind`.
- **Salida:** un mapa matricial o una cuadrícula de distribuciones y relaciones por pares.
- **Funcionamiento:** el mapa de calor codifica cada valor con color; la matriz de pares dibuja una relación por combinación y una distribución en la diagonal.
- **Hiperparámetros:** `annot`, `center`, mapa de color; `hue`, `diag_kind='kde'`, `kind='reg'`.
- **Ventajas:** cribado rápido de redundancia, cambios por grupo y relaciones candidatas.
- **Limitaciones:** correlación resume principalmente asociación lineal; `pairplot()` consume espacio y cómputo conforme aumentan columnas y filas.
- **Casos adecuados:** conjunto numérico pequeño después de una selección razonable de features.
- **No conviene:** cientos de variables o como criterio único de selección de features.

### `jointplot()`, `regplot()`, `residplot()` y `lmplot()`

- **Objetivo:** estudiar una relación bivariada, su ajuste y sus errores, globalmente o por segmentos.
- **Entradas:** predictor `x`, respuesta `y`, `data`; `kind` para `jointplot()`; `order` o `logistic` en el soporte de regresión; `hue`, `row`, `col` para segmentación con `lmplot()`.
- **Salida:** cuadrícula conjunta, `Axes` de ajuste/residuos o cuadrícula de regresiones.
- **Funcionamiento:** `regplot()` estima y dibuja la relación; `residplot()` grafica `y_i - ŷ_i`; `lmplot()` repite ajustes en subconjuntos.
- **Hiperparámetros:** `kind={'scatter','hex','reg','resid','kde'}`, `order`, `logistic`, `hue`, `row`, `col`.
- **Ventajas:** enlaza la exploración visual con el diagnóstico de especificación.
- **Limitaciones:** el gráfico no demuestra generalización ni reemplaza métricas; probar muchas permutaciones produce `p(p-1)` pares dirigidos y puede ser costoso.
- **Casos adecuados:** elegir transformaciones, detectar no linealidad aparente, heterocedasticidad o diferencias entre grupos.
- **No conviene:** concluir calidad predictiva solo por apariencia o ajustar polinomios de orden alto sin validación.

### `boxenplot()` y `violinplot()`

- **Objetivo:** comparar distribuciones numéricas entre categorías.
- **Entradas:** columnas `x`, `y`, `data`; control de ancho en violines.
- **Salida:** cuantiles extendidos o forma KDE con resumen central.
- **Funcionamiento:** el boxen plot añade niveles de cuantiles; el violín refleja una KDE simétrica alrededor de un eje central.
- **Hiperparámetros:** categoría, variable numérica y normalización/ancho del violín.
- **Ventajas:** boxen enfatiza colas; violín revela asimetría y multimodalidad aparente.
- **Limitaciones:** el violín depende de una estimación suavizada; el boxen puede ser difícil con muestras pequeñas.
- **Casos adecuados:** comparar features continuas por clase o segmento.
- **No conviene:** grupos con muy pocas observaciones sin mostrar también el tamaño muestral.

### `FacetGrid`

- **Objetivo:** aplicar una visualización idéntica a cada subconjunto de datos.
- **Entradas:** `data`, variables de partición en `row` y `col`; función y columnas pasadas a `map()`.
- **Salida:** una cuadrícula de ejes.
- **Funcionamiento:** construye subconjuntos por combinaciones categóricas y ejecuta la función de dibujo en cada uno.
- **Hiperparámetros:** `row`, `col`, orden de categorías y función mapeada.
- **Ventajas:** facilita comparaciones controladas entre segmentos.
- **Limitaciones:** demasiadas categorías generan paneles vacíos o una cuadrícula inmanejable.
- **Casos adecuados:** revisar distribuciones o relaciones por región, clase, periodo o fuente.
- **No conviene:** categorías de cardinalidad alta.

### Formato de títulos, leyendas y ejes

- **Objetivo:** hacer que la figura pueda interpretarse sin contexto oral adicional.
- **Entradas:** texto, unidades, límites, escala, posiciones y formato de marcas.
- **Salida:** la misma figura con metadatos visuales claros.
- **Funcionamiento:** `suptitle`, `set_title`, `set_xlabel`, `set_ylabel` y `legend` describen; `xlim`/`ylim` y `xscale`/`yscale` transforman la vista; `ticker` localiza y formatea marcas.
- **Hiperparámetros:** `loc`, coordenadas, límites, tipo de escala, base de locator, máximo de porcentaje.
- **Ventajas:** reduce ambigüedad y evita etiquetas o marcas engañosas.
- **Limitaciones:** límites truncados y escalas mal elegidas pueden exagerar diferencias.
- **Casos adecuados:** toda figura destinada a revisión o comunicación.
- **No conviene:** omitir unidades, dejar leyendas encima de los datos o titular solo el eje activo en una cuadrícula.

### Referencias, regiones y anotaciones

- **Objetivo:** resaltar umbrales, intervalos, diferencias entre curvas y eventos concretos.
- **Entradas:** coordenadas, límites, máscara booleana para `where`, texto, `xy`, `xytext` y propiedades de flecha.
- **Salida:** capas adicionales sobre un `Axes`.
- **Funcionamiento:** líneas y bandas usan coordenadas del eje; `fill_between()` recibe un conjunto de `x` y dos conjuntos de `y`; `annotate()` coloca texto y opcionalmente lo conecta a un punto.
- **Hiperparámetros:** `color`, `linestyle`, `alpha`, `label`, `where`, `arrowprops`.
- **Ventajas:** orientan la lectura hacia condiciones relevantes.
- **Limitaciones:** demasiadas capas ocultan datos y pueden inducir una narrativa selectiva.
- **Casos adecuados:** límites de control, rangos aceptables, cambios de régimen y eventos conocidos.
- **No conviene:** cuando el umbral no tiene justificación o cuando la anotación tapa evidencia contraria.

### Mapas de color personalizados, ciclos y generadores

- **Objetivo:** aplicar una semántica cromática consistente o producir estilos bajo demanda.
- **Entradas:** nombres de color, códigos hexadecimales, tuplas RGB/RGBA normalizadas a `[0,1]` o listas de colores.
- **Salida:** colores RGBA, `ListedColormap`, iterador o generador.
- **Funcionamiento:** se interpolan componentes RGB con `np.linspace()` para construir gradientes; `itertools.cycle()` repite una secuencia; `yield` produce colores sin crear una lista completa.
- **Hiperparámetros:** colores extremos, número de entradas `N`, número y tamaño de segmentos.
- **Ventajas:** consistencia, control fino y menor uso de memoria en secuencias grandes.
- **Limitaciones:** un gradiente personalizado puede tener contraste o semántica deficientes; un iterador se agota salvo que sea infinito.
- **Casos adecuados:** identidad visual, mapas secuenciales/divergentes y asignación condicional de colores.
- **No conviene:** crear una paleta nueva cuando una paleta estándar expresa mejor el tipo de datos.

## 6. Fórmulas esenciales

### Residuo de una observación

\[
e_i = y_i - \hat{y}_i
\]

- `e_i`: residuo de la observación `i`.
- `y_i`: valor observado.
- `ŷ_i`: valor predicho por la línea o modelo ajustado.
- **Qué mide:** error firmado de la predicción en una observación.
- **Interpretación:** residuos alrededor de cero sin patrón sistemático apoyan el uso del ajuste examinado; dispersión creciente o estructura visible indican que puede no ser adecuado.
- **Supuesto importante:** esta es una comprobación visual de especificación, no una prueba de generalización.

### Cambio porcentual usado en una anotación

\[
r = \frac{p_{apertura}-p_{cierre\ previo}}{p_{cierre\ previo}}
\]

- `p_apertura`: precio de apertura posterior al evento.
- `p_cierre previo`: precio de cierre anterior.
- `r`: variación relativa, representable como porcentaje.
- **Qué mide:** cambio respecto de la base previa.
- **Interpretación:** un valor negativo indica caída; el formato `.2%` lo muestra como porcentaje con dos decimales.
- **Supuesto importante:** la base no puede ser cero y los dos precios deben corresponder al intervalo que se pretende comparar.

### Líneas a desviaciones estándar de la media

\[
x_k^{\pm} = \mu \pm k\sigma, \qquad k \in \{1,2,3\}
\]

- `μ`: media muestral.
- `σ`: desviación estándar muestral.
- `k`: número de desviaciones estándar.
- `x_k±`: posición de cada referencia vertical.
- **Qué mide:** distancia respecto de la media en unidades de desviación estándar.
- **Interpretación:** facilita comparar la ubicación de la densidad estimada con referencias de escala.
- **Supuesto importante:** el capítulo usa estas líneas como referencias visuales; no afirma que la distribución sea normal ni asigna probabilidades a las bandas.

### Conversión RGB para Matplotlib

\[
c_{normalizado} = \frac{c_{8\ bits}}{255}
\]

- `c_8 bits`: componente rojo, verde o azul en `[0,255]`.
- `c_normalizado`: componente requerido por Matplotlib en `[0,1]`.
- **Qué mide:** la misma intensidad expresada en otra escala.
- **Interpretación:** se aplica de forma independiente a R, G y B; alpha vale 1 si no se especifica.
- **Supuesto importante:** los valores de entrada deben estar dentro del rango válido.

## 7. Implementación práctica

### Diagnóstico mínimo de regresión y residuos

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.regplot(data=train_df, x="feature", y="target", ax=axes[0])
sns.residplot(data=train_df, x="feature", y="target", ax=axes[1])
axes[0].set_title("Ajuste en entrenamiento")
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Residuos en entrenamiento")
fig.tight_layout()
```

El capítulo demuestra el diagnóstico sobre datos exploratorios, no un pipeline predictivo. **Síntesis práctica derivada:** si el objetivo es evaluar un modelo, ajustar sobre `train_df` y revisar métricas en validación por separado evita convertir una inspección visual del conjunto completo en evidencia de generalización.

### Comparación de una feature por clase

```python
import seaborn as sns

ax = sns.swarmplot(
    data=train_df,
    x="target_class",
    y="feature_value",
    hue="data_source",
)
ax.set(title="Distribución por clase", xlabel="clase", ylabel="valor")
```

Si la muestra es grande, el capítulo advierte indirectamente el coste del solapamiento; bajo esa condición conviene usar una vista agregada o una muestra reproducible solo para visualizar.

### Correlación con centro significativo

```python
import seaborn as sns

corr = train_df.select_dtypes("number").corr()
ax = sns.heatmap(corr, annot=True, center=0, cmap="vlag")
ax.set_title("Correlaciones en entrenamiento")
```

**Síntesis práctica derivada:** calcular decisiones de selección o transformación solo con entrenamiento reduce leakage. El capítulo enseña el gráfico, pero no formula esta regla de validación.

### Facetas para comprobar heterogeneidad por grupo

```python
import seaborn as sns

grid = sns.FacetGrid(train_df, row="segment", col="period")
grid.map_dataframe(sns.histplot, x="feature", bins=20)
grid.set_axis_labels("feature", "conteo")
```

**Nota práctica de API actual:** el libro usa `FacetGrid.map(plt.hist, ...)`; `map_dataframe()` y `sns.histplot()` conservan mejor la semántica de columnas en código moderno. Esta adaptación no es contenido original del capítulo.

### Anotar un umbral y una región

```python
import matplotlib.pyplot as plt

ax = validation_curve.plot(title="Métrica por umbral")
ax.axvline(best_threshold, color="red", linestyle="--", label="umbral")
ax.axvspan(lower_bound, upper_bound, alpha=0.15, label="zona aceptable")
ax.set(xlabel="umbral", ylabel="métrica")
ax.legend()
plt.tight_layout()
```

El capítulo respalda las operaciones visuales. **Síntesis práctica derivada:** el umbral debe fijarse con datos de validación y no ajustarse mirando el test final.

### Adaptaciones modernas identificadas

- El libro menciona `sns.distplot()`. **Nota práctica de API actual:** usar `sns.histplot(..., kde=True)` para una figura sobre un `Axes` o `sns.displot()` para una interfaz de figura.
- El ejemplo usa `violinplot(scale='width')`. **Nota práctica de API actual:** en versiones recientes, expresar la normalización del ancho con `density_norm='width'`.
- El libro obtiene mapas con `cm.datad.keys()` y `cm.get_cmap()`. **Nota práctica de API actual:** usar `matplotlib.colormaps` y `matplotlib.colormaps[name]`.
- El libro usa llamadas posicionales en algunas funciones de Seaborn. **Nota práctica de API actual:** pasar `data=`, `x=` y `y=` explícitamente para evitar ambigüedad y cambios de firma.

## 8. Supuestos y condiciones de aplicación

1. Los datos deben estar en una estructura que Seaborn pueda mapear por nombres de columnas; el formato largo facilita `hue`, `row` y `col`.
2. Las variables utilizadas en correlación, regresión, KDE y diferencias aritméticas deben ser numéricas y estar tratadas respecto de valores faltantes.
3. Un gráfico de regresión supone que la forma solicitada (`lineal`, `order`, `logistic`) corresponde a una hipótesis razonable; los residuos se usan para cuestionarla.
4. Comparar facetas requiere categorías con tamaños suficientes y escalas que permitan una comparación honesta.
5. KDE y violines estiman densidad; su forma no equivale a observar directamente la distribución verdadera.
6. Las líneas `μ ± kσ` son referencias descriptivas salvo que se justifiquen supuestos distributivos adicionales.
7. Los porcentajes requieren una base (`xmax` o denominador) coherente y distinta de cero.
8. `fill_between()` requiere un eje compartido y series alineadas; la máscara `where` debe tener longitud compatible.
9. RGB/RGBA debe estar normalizado a `[0,1]`; un hexadecimal debe tener una forma válida de tres o seis dígitos.
10. Un colormap divergente necesita un centro con significado; uno secuencial necesita orden; uno cualitativo se reserva para grupos sin orden.
11. Los iteradores consumibles no pueden reutilizarse después de agotarse; `itertools.cycle()` y `repeat()` sin límite son infinitos y deben combinarse con un iterable finito.
12. El número de permutaciones dirigidas de `p` columnas es `p(p-1)`; dibujar ajuste y residuos para todas puede ser inviable cuando `p` crece.

## 9. Errores y riesgos

- Usar `stripplot()` sin suficiente `jitter` o transparencia y concluir que hay pocos puntos cuando en realidad están superpuestos.
- Interpretar una correlación alta, una línea ajustada o una banda de confianza como causalidad o buen rendimiento fuera de muestra.
- Ignorar residuos con abanico, curvatura o desplazamiento respecto de cero; el ejemplo del capítulo muestra dispersión creciente a mayor volumen.
- Probar muchas variables y segmentos visualmente sin controlar el sesgo de selección; un patrón atractivo puede ser espurio.
- Generar `pairplot()` o todas las permutaciones de regresión con demasiadas columnas: el coste y el número de paneles crecen aproximadamente de forma cuadrática.
- Comparar facetas con escalas incompatibles o grupos muy desbalanceados sin mostrar tamaños de muestra.
- Interpretar un violín como datos observados exactos; sus contornos provienen de KDE.
- Usar `plt.title()` esperando un título global en una cuadrícula y terminar titulando solo el último subplot.
- Mover límites del eje para dramatizar cambios o aplicar logaritmos sin informar la escala.
- Confundir *locator* con *formatter*: el primero decide posiciones; el segundo cambia la representación.
- Configurar mal `PercentFormatter(xmax=...)`, produciendo etiquetas porcentuales incorrectas.
- Añadir referencias o regiones sin explicar su origen; la visualización puede convertir una elección arbitraria en aparente evidencia.
- Colocar anotaciones sobre los datos; `xytext` y flechas deben usarse para preservar la visibilidad.
- Elegir una paleta secuencial para categorías nominales o una divergente sin centro real.
- Usar demasiados colores, bajo contraste o una paleta personalizada no comprobada; el gráfico pierde jerarquía e interpretación.
- Reutilizar un iterador agotado y obtener silenciosamente cero elementos en una segunda pasada.
- Pasar iterables de longitudes diferentes a `zip()` sin advertir que la salida se corta al más corto.
- **Posible leakage - síntesis derivada:** calcular matrices, transformaciones o umbrales con el conjunto de test antes de decidir el pipeline contamina la evaluación, aunque el capítulo no desarrolla validación.
- **Posible overfitting - síntesis derivada:** escoger polinomios, segmentos o anotaciones por su apariencia en todos los datos puede sobreajustar la narrativa exploratoria.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **Cuándo consultarlo:** cuando se pide EDA, interpretación de variables, comparación por clase/segmento, diagnóstico de una regresión o producción de una figura clara para justificar decisiones.
- **Qué decisiones permite tomar:** elegir entre puntos individuales y agregación; decidir si conviene segmentar; detectar transformaciones candidatas; escoger un gráfico de distribución; seleccionar una semántica de color y un formato de ejes.
- **Baseline útil:** una matriz de correlación sobre entrenamiento, uno o dos gráficos de distribución por variable objetivo y un gráfico bivariado/residual para las features más plausibles. Cada figura debe tener título, ejes con unidades y leyenda solo si aporta información.
- **Mejoras del modelo:** residuos estructurados pueden activar transformaciones, interacciones o un modelo no lineal; diferencias por faceta pueden sugerir interacciones, estratificación o validación por grupo; distribuciones muy distintas por clase pueden señalar features útiles.
- **Qué evitar bajo presión:** `pairplot()` sobre todas las columnas, permutaciones completas de regresión, colormaps personalizados, anotaciones decorativas y rejillas con cardinalidad alta. Son costosos y rara vez mejoran un score directamente.
- **Señales que activan técnicas:** categorías más una variable continua -> strip/swarm/violin; muchas variables numéricas -> heatmap y selección previa; nube densa -> hex; error que crece con el predictor -> residuales; sospecha de heterogeneidad -> facetas; umbral de negocio o métrica -> línea/banda anotada.
- **Separación de datos:** realizar EDA estructural y ajustes visuales sobre entrenamiento. El test final debe reservarse para evaluación; si se visualiza, hacerlo solo después de cerrar decisiones o con un propósito explícito que no retroalimente el pipeline.
- **Criterio de entrega:** preferir pocas figuras legibles y justificadas. Una visualización sin interpretación operativa consume tiempo y no aporta evidencia al notebook.

## 11. Checklist operativo

- [ ] Identificar si la pregunta es categórica, bivariada, multivariada, distributiva o segmentada.
- [ ] Separar entrenamiento y validación antes de tomar decisiones que afecten al pipeline.
- [ ] Seleccionar pocas columnas relevantes antes de usar `pairplot()` o regresiones múltiples.
- [ ] Cambiar de puntos a hexágonos, agregación o muestra si existe solapamiento severo.
- [ ] Revisar ajuste y residuos juntos antes de aceptar una relación lineal.
- [ ] Comprobar tamaños y escalas de los grupos antes de comparar facetas.
- [ ] Añadir título, nombres de ejes, unidades y leyenda cuando sea necesaria.
- [ ] Verificar límites, escala, posiciones y formato de las marcas.
- [ ] Usar paleta cualitativa, secuencial o divergente según la semántica real.
- [ ] Justificar cada umbral, región sombreada y anotación.
- [ ] Evitar que las capas tapen puntos o patrones relevantes.
- [ ] Aplicar `tight_layout()` y revisar la figura exportada.
- [ ] Explicar en una frase qué decisión o riesgo revela cada gráfico.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Selección entre strip, swarm, boxen y violin por tipo y densidad de datos | `knowledge/eda/seaborn_visualization_selection.md` | Alta |
| Heatmaps, pair plots y joint plots para relaciones multivariadas | `knowledge/eda/multivariate_visual_diagnostics.md` | Alta |
| Regresión, residuos y señales visuales de mala especificación | `knowledge/eda/regression_and_residual_diagnostics.md` | Alta |
| Faceting y comparación consistente entre segmentos | `knowledge/eda/segmented_analysis.md` | Alta |
| Riesgos de usar EDA del test para decidir transformaciones o umbrales | `knowledge/validation/eda_and_leakage.md` | Alta |
| Formato de ejes, títulos, leyendas y unidades | `knowledge/interpretability/visual_communication.md` | Media |
| Umbrales, regiones, diferencias entre curvas y anotaciones | `knowledge/interpretability/plot_annotations.md` | Media |
| Selección semántica de mapas cualitativos, secuenciales y divergentes | `knowledge/eda/color_semantics.md` | Media |
| Optimización de coste visual ante alta dimensionalidad o cardinalidad | `knowledge/eda/scalable_visualization.md` | Media |
| Diagnóstico por periodo mediante facetas | `knowledge/time_series/visual_diagnostics.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura y temas del capítulo, pp. 310-311 |
| 2. Conceptos esenciales | *Utilizing seaborn for advanced plotting*, pp. 312-332; *Formatting*, pp. 332-341; *Customizing visualizations*, pp. 341-365 |
| 3. Flujo o procedimiento presentado | Secuencia de notebooks y ejemplos, pp. 311-365; pasos 9-11 son síntesis operativa de las técnicas mostradas |
| 4. Reglas de decisión | Gráficos categóricos, pp. 312-314; correlación y relaciones, pp. 315-328; distribuciones y facetas, pp. 329-332; formato y personalización, pp. 332-365 |
| 5. `stripplot()` y `swarmplot()` | *Categorical data*, pp. 312-314 |
| 5. `heatmap()` y `pairplot()` | *Correlations and heatmaps*, pp. 315-318 |
| 5. `jointplot()`, regresión y residuos | pp. 319-328, incluidos *Regression plots* y `lmplot()` |
| 5. `boxenplot()` y `violinplot()` | *Distributions*, pp. 329-330 |
| 5. `FacetGrid` | *Faceting*, pp. 330-332 |
| 5. Formato de títulos, leyendas y ejes | *Formatting*, pp. 332-341 |
| 5. Referencias, regiones y anotaciones | *Adding reference lines*, pp. 342-347; *Shading regions*, pp. 347-350; *Annotations*, pp. 350-352 |
| 5. Color, colormaps y generadores | *Colors*, pp. 352-365 |
| 6. Fórmulas esenciales | Residuos, pp. 321-322; media y desviación estándar, pp. 344-346; cambio porcentual, pp. 351-352; RGB, pp. 353-354 |
| 7. Implementación práctica | Ejemplos del capítulo, pp. 313-332 y 342-360; las adaptaciones modernas están marcadas explícitamente como notas prácticas |
| 8. Supuestos y condiciones | Condiciones expresas o necesarias para las APIs mostradas, pp. 312-365 |
| 9. Errores y riesgos | Solapamiento, p. 314; residuos, p. 322; subplots y títulos, pp. 333-335; ejes, pp. 337-341; color, pp. 352-365; leakage y overfitting marcados como síntesis derivada |
| 10. Aplicación en un coding test | Síntesis práctica derivada de pp. 312-365; no presentada como sección original del libro |
| 11. Checklist operativo | Síntesis ejecutable derivada de pp. 312-365 |
| 12. Mapeo hacia `knowledge/` | Clasificación documental derivada del contenido de pp. 312-365 |

