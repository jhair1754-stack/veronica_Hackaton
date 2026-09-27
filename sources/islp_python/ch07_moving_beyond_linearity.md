---
title: "Moving Beyond Linearity"
topics:
  - "Polynomial regression and step functions"
  - "Regression splines, smoothing splines, and local regression"
  - "Generalized additive models"
target_knowledge_files:
  - "knowledge/models/nonlinear_basis_models.md"
  - "knowledge/models/generalized_additive_models.md"
  - "knowledge/tuning/nonlinear_smoothers.md"
  - "knowledge/interpretability/additive_effects.md"
---

# 7. Moving Beyond Linearity

## 1. Propósito del capítulo

El capítulo amplía los modelos lineales cuando la relación entre predictores y respuesta no es lineal, procurando conservar interpretabilidad. Presenta regresión polinómica, funciones escalón, funciones base, splines de regresión y suavizado, regresión local y modelos aditivos generalizados (GAM) para regresión y clasificación.

## 2. Conceptos esenciales

- Una **función base** (*basis function*) transforma un predictor `X` en variables conocidas `b_j(X)`; el modelo sigue siendo lineal en sus coeficientes y puede ajustarse por mínimos cuadrados.
- La **regresión polinómica** introduce potencias de `X`. Su estructura es global: un cambio local afecta la curva en todo el dominio. Grados altos pueden producir formas inestables, sobre todo en los extremos; el capítulo indica que rara vez se usan grados mayores que 3 o 4.
- Una **función escalón** (*step function*) divide el rango de `X` y asigna una constante a cada intervalo. Es interpretable, pero puede ocultar tendencias dentro de los bins si no existen puntos de corte naturales.
- Un **spline de regresión** ajusta polinomios de bajo grado por regiones y los une en nudos (*knots*) bajo restricciones de continuidad. Un spline de grado `d` mantiene continuas las derivadas hasta orden `d-1`.
- Un **spline cúbico natural** obliga a la función a ser lineal más allá de los nudos extremos, lo que reduce la varianza en los bordes.
- Un **spline de suavizado** (*smoothing spline*) equilibra error de ajuste y rugosidad mediante `lambda`. Tiene un nudo en cada valor único de entrenamiento, pero la penalización reduce sus grados de libertad efectivos.
- La **regresión local** ajusta un modelo ponderado nuevo alrededor de cada punto objetivo. El *span* controla qué proporción de observaciones cercanas participa.
- Un **GAM** suma efectos `f_j(X_j)`. Permite formas no lineales interpretables por variable, pero la aditividad puede omitir interacciones.
- Los grados de libertad resumen flexibilidad: más grados suelen reducir sesgo y aumentar varianza. El capítulo propone validación cruzada para elegirlos y favorece el modelo más simple cuando el desempeño es similar.
- Las bandas mostradas son intervalos puntuales, no bandas simultáneas. En clasificación con eventos raros o categorías sin casos positivos, la incertidumbre puede ser muy grande y el ajuste difícil.

## 3. Flujo o procedimiento presentado

1. Graficar la respuesta frente al predictor y comprobar si una recta parece insuficiente.
2. Elegir una familia compatible con el problema: polinomio, bins, spline, regresión local o GAM.
3. Definir la complejidad: grado, cortes/nudos, grados de libertad, `lambda` o *span*.
4. Ajustar toda transformación exclusivamente con los datos de entrenamiento y reutilizarla sin recalcular parámetros al transformar validación o test.
5. Seleccionar complejidad mediante validación cruzada; para modelos anidados ajustados por mínimos cuadrados, ANOVA/F-test también permite evaluar si la extensión es necesaria.
6. Examinar curvas, extremos, intervalos puntuales y estabilidad. Preferir splines naturales si preocupan los bordes.
7. Para varios predictores, construir un GAM con un efecto por variable y factores para variables cualitativas.
8. Inspeccionar efectos parciales manteniendo los demás términos fijos y evaluar si alguna relación puede simplificarse a lineal.
9. Añadir interacciones de forma explícita solo cuando la estructura aditiva sea insuficiente.
10. En clasificación, revisar conteos por clase y categoría antes de interpretar el ajuste.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Curvatura suave y moderada en un predictor | Polinomio de grado bajo | Es simple y conserva herramientas de regresión lineal | Evitar grados altos y revisar extremos |
| Existen umbrales naturales y se busca máxima interpretabilidad | Función escalón | Produce medias o probabilidades por intervalos | Cortes arbitrarios pueden perder tendencia intrabin |
| Se necesita flexibilidad local estable | Spline de regresión de bajo grado | Aumenta flexibilidad con nudos sin elevar mucho el grado | Elegir nudos o grados de libertad con validación |
| Hay extrapolación o alta incertidumbre en los extremos | Spline cúbico natural | Impone linealidad en los bordes y suele estabilizarlos | Sigue dependiendo de datos suficientes cerca de los límites |
| No se desean seleccionar nudos | Spline de suavizado | `lambda` controla la rugosidad con nudos en valores únicos | Un `lambda` pequeño puede interpolar y sobreajustar |
| La relación es local y hay pocos predictores | Regresión local | Adapta el ajuste a la vecindad del punto objetivo | Predicción costosa; se degrada por encima de 3 o 4 dimensiones |
| Hay varios predictores con efectos no lineales principalmente aditivos | GAM | Combina flexibilidad con efectos individuales interpretables | Puede omitir interacciones importantes |
| Dos complejidades rinden de forma similar | Modelo con menos grados de libertad | El capítulo favorece la solución más simple sin evidencia de mejora | La comparación debe hacerse fuera de muestra |

## 5. Métodos y técnicas

### Regresión polinómica

- **Objetivo:** modelar una curva global usando potencias de un predictor.
- **Entradas:** predictor cuantitativo, respuesta continua o binaria y grado `d`.
- **Salida:** predicción continua o probabilidad logística; curva e intervalos puntuales.
- **Funcionamiento:** ajusta un modelo lineal sobre `X, X^2, ..., X^d`.
- **Hiperparámetro:** grado `d`.
- **Ventajas:** implementación directa; admite inferencia de modelos lineales.
- **Limitaciones:** coeficientes individuales poco informativos; alta inestabilidad con grados elevados, especialmente en bordes.
- **Adecuado:** curvatura global suave de baja complejidad.
- **No conviene:** patrones locales complejos o necesidad de extrapolación estable con grado alto.

### Funciones escalón

- **Objetivo:** representar efectos constantes por intervalos.
- **Entradas:** predictor continuo, puntos de corte y respuesta.
- **Salida:** una constante, media o probabilidad por bin.
- **Funcionamiento:** crea indicadores mutuamente excluyentes y los usa como variables dummy.
- **Hiperparámetros:** número y ubicación de cortes.
- **Ventajas:** lectura directa; útil con umbrales sustantivos.
- **Limitaciones:** discontinuidad y pérdida de información dentro de cada intervalo.
- **Adecuado:** cortes naturales o reglas de negocio conocidas.
- **No conviene:** una tendencia continua sin umbrales claros.

### Splines de regresión

- **Objetivo:** modelar curvatura local mediante polinomios por tramos unidos suavemente.
- **Entradas:** predictor cuantitativo, respuesta, grado y nudos o grados de libertad.
- **Salida:** curva continua; con splines cúbicos, también son continuas las derivadas primera y segunda.
- **Funcionamiento:** construye una matriz de funciones base y estima sus coeficientes por mínimos cuadrados. Un spline cúbico con `K` nudos usa `K+4` grados de libertad, incluido el intercepto.
- **Hiperparámetros:** grado, número y ubicación de nudos; alternativamente, grados de libertad.
- **Ventajas:** más estabilidad que un polinomio global de alto grado; flexibilidad local.
- **Limitaciones:** puede tener alta varianza en los extremos; la parametrización requiere selección.
- **Adecuado:** relaciones suaves no lineales; usar nudos por cuantiles como opción uniforme o más nudos donde la función cambia rápido.
- **No conviene:** saltos reales sin suavidad o datos demasiado escasos por región.

### Spline cúbico natural

- **Objetivo:** estabilizar un spline cúbico en los extremos.
- **Entradas y salida:** iguales a un spline de regresión.
- **Funcionamiento:** agrega restricciones de linealidad fuera de los nudos de frontera.
- **Ventajas:** menor varianza y bandas más estrechas en bordes.
- **Limitaciones:** restringe la forma extrema.
- **Adecuado:** opción general cuando importa el comportamiento en los límites.
- **No conviene:** cuando existe evidencia sólida de curvatura no lineal más allá de los nudos extremos.

### Splines de suavizado

- **Objetivo:** obtener una curva suave sin seleccionar manualmente los nudos.
- **Entradas:** `X`, `y` y penalización `lambda` o grados de libertad efectivos deseados.
- **Salida:** spline cúbico natural penalizado.
- **Funcionamiento:** minimiza RSS más una penalización sobre la segunda derivada.
- **Hiperparámetro:** `lambda`; al crecer, disminuyen los grados de libertad efectivos desde `n` hacia 2.
- **Ventajas:** control continuo del compromiso sesgo-varianza; LOOCV puede calcularse eficientemente.
- **Limitaciones:** necesita ajustar la penalización; `lambda=0` interpola y puede sobreajustar.
- **Adecuado:** suavizado univariado y como término de un GAM.
- **No conviene:** relaciones con discontinuidades reales.

### Regresión local

- **Objetivo:** estimar la respuesta en `x0` usando observaciones cercanas.
- **Entradas:** datos de entrenamiento, punto objetivo, función de pesos, orden local y *span* `s=k/n`.
- **Salida:** predicción local en cada punto objetivo.
- **Funcionamiento:** toma los `k` vecinos, pondera por cercanía y ajusta mínimos cuadrados ponderados.
- **Hiperparámetro principal:** *span*; uno pequeño genera mayor flexibilidad.
- **Ventajas:** adaptación local sin imponer una forma global.
- **Limitaciones:** requiere conservar los datos y reajustar para cada predicción; sufre en alta dimensión.
- **Adecuado:** uno o pocos predictores; también modelos con coeficientes variables localmente en tiempo.
- **No conviene:** muchos predictores o inferencia de baja latencia sobre grandes volúmenes.

### Modelos aditivos generalizados (GAM)

- **Objetivo:** modelar varias variables mediante la suma de efectos lineales, suaves y categóricos.
- **Entradas:** matriz de predictores, respuesta, tipo de término y complejidad por variable.
- **Salida:** predicciones y funciones parciales `f_j` con incertidumbre puntual.
- **Funcionamiento:** con bases de splines naturales se reduce a una gran regresión lineal; con splines de suavizado puede ajustarse por *backfitting* sobre residuos parciales.
- **Hiperparámetros:** base, grados de libertad o `lambda` por término; términos de interacción opcionales.
- **Ventajas:** precisión no lineal con interpretación variable por variable; sirve para regresión y clasificación logística.
- **Limitaciones:** asume aditividad; interacciones no incluidas se pierden.
- **Adecuado:** relaciones no lineales suaves y principalmente aditivas.
- **No conviene:** interacción de alta dimensión dominante sin términos explícitos.

## 6. Fórmulas esenciales

### Modelo de funciones base

$$y_i = \beta_0 + \sum_{j=1}^{K}\beta_j b_j(x_i) + \epsilon_i$$

`b_j` son transformaciones fijas elegidas antes del ajuste; `beta_j` son coeficientes estimados y `epsilon_i` es el error. Permite tratar polinomios, bins y splines como regresión lineal sobre variables transformadas.

### Base de potencia truncada para spline cúbico

$$h(x,\xi)=(x-\xi)^3_+=\begin{cases}(x-\xi)^3,&x>\xi\\0,&x\leq\xi\end{cases}$$

`xi` es un nudo. Agregar una función `h` por nudo permite cambiar la curvatura después del nudo manteniendo continuidad de la función y de sus dos primeras derivadas.

### Criterio del spline de suavizado

$$\min_g\left\{\sum_{i=1}^{n}(y_i-g(x_i))^2 + \lambda\int[g''(t)]^2dt\right\}$$

El primer término mide falta de ajuste; el segundo mide rugosidad. `lambda>=0` controla el compromiso: `lambda=0` interpola y `lambda->infinity` converge a la recta de mínimos cuadrados.

### Grados de libertad efectivos

$$\hat{g}_{\lambda}=S_{\lambda}y,\qquad df_{\lambda}=\operatorname{tr}(S_{\lambda})=\sum_{i=1}^{n}(S_{\lambda})_{ii}$$

`S_lambda` es la matriz de suavizado y `y` el vector de respuestas. La traza mide la flexibilidad efectiva pese a que el spline tenga nominalmente muchos parámetros.

### LOOCV eficiente para spline de suavizado

$$RSS_{CV}(\lambda)=\sum_{i=1}^{n}\left[\frac{y_i-\hat g_{\lambda}(x_i)}{1-(S_{\lambda})_{ii}}\right]^2$$

Permite elegir `lambda` sin reajustar explícitamente `n` modelos. Cada denominador corrige la influencia de la observación `i`.

### Regresión local lineal

$$\min_{\beta_0,\beta_1}\sum_{i=1}^{n}K_{i0}(y_i-\beta_0-\beta_1x_i)^2,\qquad \hat f(x_0)=\hat\beta_0+\hat\beta_1x_0$$

`K_i0` es el peso de la observación `i` respecto de `x0`; fuera de la vecindad vale cero. Supone que, localmente, una recta aproxima adecuadamente la relación.

### GAM de regresión y clasificación

$$y_i=\beta_0+\sum_{j=1}^{p}f_j(x_{ij})+\epsilon_i$$

$$\log\frac{p(X)}{1-p(X)}=\beta_0+\sum_{j=1}^{p}f_j(X_j)$$

Cada `f_j` es un efecto univariado. La primera ecuación modela una respuesta cuantitativa; la segunda modela el *log-odds* de una respuesta binaria. El supuesto estructural central es la aditividad.

## 7. Implementación práctica

El laboratorio usa `ISLP`, `statsmodels` y `pygam`. Los fragmentos siguientes expresan las mismas ideas con una separación explícita entre entrenamiento y validación.

### Spline de regresión con validación

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

preprocess = ColumnTransformer([
    ("age_spline", SplineTransformer(
        n_knots=5, degree=3, knots="quantile", include_bias=False
    ), ["age"]),
    ("education", OneHotEncoder(handle_unknown="ignore"), ["education"]),
])

model = Pipeline([
    ("features", preprocess),
    ("regressor", Ridge(alpha=1.0)),
])
model.fit(X_train, y_train)
validation_rmse = mean_squared_error(
    y_valid, model.predict(X_valid)
) ** 0.5
```

**Nota práctica - adaptación moderna:** `SplineTransformer`, `Pipeline` y `ColumnTransformer` no son la API presentada por el capítulo. Se usan aquí para conservar su principio esencial: aprender nudos, codificación y coeficientes solo en entrenamiento, y aplicar exactamente la misma transformación a validación.

### GAM de regresión con selección de suavizado

```python
import numpy as np
from pygam import LinearGAM, s, f

X_train_gam = np.column_stack([
    X_train["age"],
    X_train["year"],
    X_train["education_code"],
])
X_valid_gam = np.column_stack([
    X_valid["age"],
    X_valid["year"],
    X_valid["education_code"],
])

gam = LinearGAM(s(0) + s(1, n_splines=7) + f(2))
gam.gridsearch(X_train_gam, y_train)
valid_predictions = gam.predict(X_valid_gam)
```

`gridsearch` operacionaliza la selección de la penalización descrita en el capítulo. La codificación de categorías debe ser consistente y aprenderse sin consultar validación.

### Regresión local

```python
import statsmodels.api as sm

predictions = sm.nonparametric.lowess(
    endog=y_train,
    exog=X_train["age"],
    frac=0.5,
    xvals=X_valid["age"],
)
```

`frac` es el *span*. El capítulo muestra que un valor mayor produce una curva más suave.

## 8. Supuestos y condiciones de aplicación

1. La regresión polinómica, los splines y los GAM son lineales en sus coeficientes aunque sean no lineales en los predictores.
2. Los errores del modelo base conservan los supuestos necesarios para la inferencia lineal; la aproximación de intervalos al 95 % usa normalidad del error.
3. Los splines suponen que la relación es suave; el orden de continuidad depende del grado.
4. El spline natural asume linealidad fuera de los nudos extremos.
5. El spline de suavizado penaliza la curvatura medida por la segunda derivada.
6. La regresión local necesita densidad suficiente alrededor de cada punto objetivo y funciona principalmente en baja dimensión.
7. Un GAM supone que los efectos se suman; las interacciones deben añadirse explícitamente.
8. Para comparar modelos mediante ANOVA/F-test, los modelos deben ser anidados.
9. Los efectos parciales se interpretan manteniendo los demás términos fijos; no son efectos causales.
10. En clasificación, cada categoría necesita suficiente representación de ambas clases para una estimación estable.

## 9. Errores y riesgos

- Ajustar transformaciones o cuantiles con el conjunto completo produce **data leakage**. Los parámetros de base deben aprenderse en entrenamiento.
- Elegir grado, número de nudos, `lambda` o *span* observando validación repetidamente sobreajusta la validación; la selección debe formar parte del esquema de CV.
- Los polinomios de grado alto presentan oscilaciones y alta varianza en fronteras.
- Demasiados nudos, grados de libertad altos, `lambda` demasiado pequeño o *span* demasiado pequeño producen sobreajuste.
- Complejidad insuficiente, `lambda` grande o *span* grande produce underfitting cercano a una recta o ajuste global.
- Los bins arbitrarios pierden información y pueden esconder cambios dentro del intervalo.
- Inferir estabilidad a partir de una muestra grande total es engañoso si el evento positivo es raro; el ejemplo tiene 3,000 casos pero solo 79 ingresos altos y bandas amplias.
- Una categoría sin positivos dificulta o impide estimar correctamente un GAM logístico; revisar tablas de contingencia antes del ajuste.
- Interpretar coeficientes polinómicos o de una base spline por separado suele ser poco útil; debe interpretarse la función completa.
- Las bandas de confianza puntuales no garantizan cobertura simultánea de toda la curva.
- Los GAM pueden parecer interpretables mientras omiten interacciones relevantes.
- La regresión local conserva los datos y reajusta por punto; tiene mayor coste de predicción y sufre la maldición de dimensionalidad.
- Extrapolar fuera del rango observado es riesgoso; incluso un spline natural solo impone una continuación lineal.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo cuando los residuos o gráficos univariados sugieran curvatura, una recta quede claramente corta o se necesite mantener interpretación.
- Para un baseline rápido, comparar una relación lineal con un polinomio de grado 2 o 3 y con un spline natural de pocos grados de libertad.
- Para mejorar el modelo, validar grados de libertad o penalización, usar splines naturales en variables continuas y combinar términos en un GAM.
- Los bins son razonables solo con umbrales naturales o cuando la explicación por intervalos sea un requisito.
- Señales activadoras: curvatura residual, cambio de pendiente por regiones, efectos de borde inestables, predictor temporal con evolución local y varios efectos univariados suaves.
- En clasificación, inspeccionar primero prevalencia y cruces categoría-clase; los grupos sin positivos o negativos invalidan una lectura confiada del efecto.
- Bajo presión de tiempo, no implementar bases de potencia truncada a mano, LOOCV matricial, selección manual extensa de nudos ni regresión local multidimensional.
- Un GAM es una mejora razonable cuando la validación respalda no linealidad y no hay evidencia fuerte de interacciones dominantes.

## 11. Checklist operativo

- [ ] Confirmar que la no linealidad aparece en entrenamiento, no solo en test o en una inspección filtrada.
- [ ] Crear una partición o CV antes de ajustar transformaciones.
- [ ] Comparar contra un baseline lineal con la misma métrica.
- [ ] Probar primero polinomios de grado bajo o splines naturales con pocos grados de libertad.
- [ ] Ajustar nudos, grados de libertad, `lambda` o *span* dentro de CV.
- [ ] Revisar extremos, intervalos y densidad de datos por región.
- [ ] Preferir el modelo más simple si la mejora validada es marginal.
- [ ] En GAM, graficar efectos parciales y comprobar interacciones plausibles.
- [ ] En clasificación, revisar prevalencia y categorías sin representación de una clase.
- [ ] Aplicar a validación/test exactamente la transformación aprendida en entrenamiento.
- [ ] Documentar cualquier actualización de API por separado del contenido del capítulo.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Funciones base, polinomios y funciones escalón | `knowledge/models/nonlinear_basis_models.md` | Alta |
| Splines de regresión y splines naturales | `knowledge/models/regression_splines.md` | Alta |
| Penalización y grados de libertad efectivos | `knowledge/tuning/nonlinear_smoothers.md` | Alta |
| Selección por CV de grado, nudos, `lambda` y *span* | `knowledge/validation/nonlinear_model_selection.md` | Alta |
| GAM para regresión y clasificación | `knowledge/models/generalized_additive_models.md` | Alta |
| Efectos parciales e interpretación aditiva | `knowledge/interpretability/additive_effects.md` | Media |
| Regresión local y modelos con coeficientes variables | `knowledge/models/local_regression.md` | Media |
| Riesgos de ajuste de transformaciones fuera del entrenamiento | `knowledge/preprocessing/leakage_safe_transforms.md` | Alta |
| Inestabilidad por eventos raros o categorías vacías | `knowledge/models/classification_edge_cases.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y panorama de métodos | Cap. 7, pp. 289–290; PDF pp. 1–2 |
| Regresión polinómica e intervalos | §7.1, pp. 290–292; PDF pp. 2–4 |
| Funciones escalón | §7.2, pp. 292–293; PDF pp. 4–5 |
| Funciones base | §7.3, pp. 293–294; PDF pp. 5–6 |
| Splines por tramos, restricciones y bases | §§7.4.1–7.4.3, pp. 294–297; PDF pp. 6–9 |
| Nudos, grados de libertad, CV y comparación con polinomios | §§7.4.4–7.4.5, pp. 297–300; PDF pp. 9–12 |
| Splines de suavizado, penalización, grados de libertad y LOOCV | §§7.5.1–7.5.2, pp. 300–303; PDF pp. 12–15 |
| Regresión local y algoritmo | §7.6, pp. 303–305; PDF pp. 15–17 |
| GAM de regresión, *backfitting*, ventajas y limitaciones | §7.7.1, pp. 305–308; PDF pp. 17–20 |
| GAM de clasificación | §7.7.2, pp. 308–310; PDF pp. 20–22 |
| Implementación de polinomios y funciones escalón | §7.8.1, pp. 310–315; PDF pp. 22–27 |
| Implementación de splines | §7.8.2, pp. 315–317; PDF pp. 27–29 |
| Implementación de splines de suavizado y GAM | §7.8.3, pp. 317–324; PDF pp. 29–36 |
| Implementación de regresión local | §7.8.4, pp. 324–325; PDF pp. 36–37 |
| Ejercicios revisados y excluidos por carecer de solución | §7.9, pp. 325–329; PDF pp. 37–41 |

