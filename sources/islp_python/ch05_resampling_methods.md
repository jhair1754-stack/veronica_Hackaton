---
title: "Resampling Methods"
topics:
  - "Cross-validation"
  - "Model assessment and model selection"
  - "Bootstrap"
  - "Bias-variance trade-off in validation"
target_knowledge_files:
  - "knowledge/validation/cross_validation.md"
  - "knowledge/validation/holdout_validation.md"
  - "knowledge/validation/bootstrap.md"
  - "knowledge/models/model_selection.md"
---

# 5. Resampling Methods

## 1. Propósito del capítulo

El capítulo presenta métodos de remuestreo que reutilizan observaciones del conjunto de entrenamiento para obtener información que un único ajuste no proporciona. Desarrolla dos familias: **cross-validation (CV)** para estimar error de prueba, evaluar modelos (*model assessment*) y elegir su flexibilidad (*model selection*); y **bootstrap** para cuantificar la incertidumbre de un estimador o método de aprendizaje.

Su valor práctico está en responder dos preguntas distintas: qué rendimiento cabe esperar en datos independientes y cuánta variabilidad tiene una estimación. El coste es computacional, porque ambos enfoques requieren ajustes repetidos.

## 2. Conceptos esenciales

- **Error de entrenamiento frente a error de prueba.** El primero se calcula sobre datos usados para ajustar el modelo y suele subestimar el segundo. La validación debe evaluar observaciones excluidas del ajuste.
- **Model assessment.** Estimación del rendimiento predictivo de un procedimiento.
- **Model selection.** Elección entre métodos o niveles de flexibilidad mediante error estimado fuera de muestra.
- **Validation set approach.** Divide aleatoriamente los datos una vez en entrenamiento y validación. Es simple, pero su estimación depende mucho de la partición y suele ser pesimista respecto del modelo reajustado con todos los datos, porque entrena con menos observaciones.
- **Leave-one-out cross-validation (LOOCV).** Entrena (n) veces; en cada ajuste deja fuera una observación. Tiene poco sesgo y particiones deterministas, pero puede ser costoso y su estimación tiene mayor varianza que CV con (k<n), debido a la alta correlación entre conjuntos de entrenamiento casi idénticos.
- **(k)-fold cross-validation.** Divide aleatoriamente en (k) grupos no solapados; cada grupo actúa una vez como validación. Los valores (k=5) o (k=10) suelen equilibrar coste, sesgo y varianza.
- **Elección por CV.** Aunque el valor absoluto del error CV puede subestimar o sobreestimar el error real, la curva suele localizar razonablemente el nivel de flexibilidad que minimiza el error.
- **CV en clasificación.** El mecanismo no cambia; se reemplaza MSE por el número o proporción de observaciones mal clasificadas.
- **Bootstrap.** Construye (B) muestras del mismo tamaño que el conjunto original, muestreando filas **con reemplazo**. Calcula el estimador en cada réplica y usa la dispersión entre réplicas para estimar su error estándar.
- **Unidad de remuestreo.** Cuando una observación contiene varias variables, se remuestrea la fila completa; no se muestrean sus columnas por separado.

## 3. Flujo o procedimiento presentado

### Cross-validation

1. Definir el procedimiento predictivo y la métrica apropiada: MSE para la regresión ilustrada o tasa de error para clasificación.
2. Crear una partición reproducible. Para (k)-fold CV, dividir los (n) casos en (k) folds aproximadamente iguales y no solapados.
3. Para cada fold, ajustar todo el procedimiento con los otros (k-1) folds.
4. Predecir exclusivamente el fold retenido y calcular su error.
5. Promediar los (k) errores.
6. Repetir el mismo esquema de folds al comparar niveles de flexibilidad, para que la comparación no quede confundida por particiones diferentes.
7. Elegir el método o nivel de flexibilidad con menor error CV; después, ajustar el procedimiento elegido sobre todos los datos disponibles para entrenamiento.

LOOCV aplica el mismo flujo con (k=n). El enfoque de validación simple lo aplica una sola vez con un bloque de validación grande.

### Bootstrap para un error estándar

1. Definir una función que reciba datos e índices y devuelva el estadístico de interés.
2. Para cada réplica (b=1,\ldots,B), extraer (n) índices con reemplazo desde las observaciones originales.
3. Calcular el estadístico sobre la muestra remuestreada completa.
4. Conservar los (B) valores del estadístico.
5. Estimar el error estándar mediante la desviación estándar muestral de las réplicas.
6. Usar una semilla para reproducibilidad y un (B) suficientemente grande para estabilizar el resultado.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Se necesita una comprobación rápida y barata del error de prueba | Validation set approach | Solo requiere un ajuste por candidato y es fácil de implementar | La estimación puede variar mucho con el split y suele sobreestimar el error del modelo ajustado con todos los datos |
| Se busca poco sesgo y (n) ajustes son viables | LOOCV | Cada ajuste usa (n-1) observaciones y el resultado no depende de una partición aleatoria | Puede ser costoso y presentar mayor varianza que (k)-fold CV |
| Se necesita una estimación práctica y general del error | 5-fold o 10-fold CV | Buen compromiso empírico entre sesgo, varianza y coste | La división aleatoria introduce variación; fijar la semilla |
| Se comparan niveles de flexibilidad o hiperparámetros | (k)-fold CV con folds comunes | La localización del mínimo de la curva CV suele aproximar la del error de prueba | No seleccionar por error de entrenamiento, que tiende a bajar al aumentar la flexibilidad |
| La respuesta es cuantitativa | CV con MSE | Mide el error cuadrático predictivo ilustrado en el capítulo | Calcularlo solo sobre observaciones retenidas |
| La respuesta es cualitativa | CV con tasa de clasificación errónea | Adapta el mismo procedimiento al resultado de clase | La regla de clasificación, incluido el umbral, debe permanecer fija durante la comparación |
| Se necesita la incertidumbre de un estadístico sin fórmula sencilla | Bootstrap no paramétrico | Aproxima el proceso de obtener nuevas muestras usando remuestreo con reemplazo | Preservar juntas todas las variables de cada observación |
| Un ajuste es caro o (n) es grande | (k)-fold CV en vez de LOOCV | Reduce los ajustes de (n) a (k) | Un (k) menor aumenta el sesgo porque cada entrenamiento usa menos datos |

## 5. Métodos y técnicas

### Validation set approach

- **Objetivo:** estimar error de prueba con una única partición.
- **Entradas:** datos, procedimiento de ajuste, tamaño de validación, semilla y métrica.
- **Salida:** error sobre el bloque retenido.
- **Funcionamiento:** ajusta con el subconjunto de entrenamiento y evalúa el subconjunto de validación.
- **Hiperparámetros:** fracción o tamaño de validación y semilla.
- **Ventajas:** sencillo y barato.
- **Limitaciones:** alta variabilidad entre splits; menor conjunto de ajuste; estimación generalmente pesimista para el modelo entrenado con todos los datos.
- **Casos adecuados:** exploración rápida o coste de ajuste muy alto.
- **No conviene:** cuando pequeñas variaciones del split pueden cambiar la decisión o se necesita una estimación estable.

### Leave-one-out cross-validation

- **Objetivo:** estimar el error de prueba usando casi todos los datos en cada ajuste.
- **Entradas:** (n) observaciones, procedimiento predictivo y métrica.
- **Salida:** promedio de (n) errores de una observación.
- **Funcionamiento:** cada observación se excluye exactamente una vez.
- **Hiperparámetros:** ninguno relativo a la partición; equivale a (k=n).
- **Ventajas:** bajo sesgo; resultado reproducible sin aleatoriedad en folds.
- **Limitaciones:** hasta (n) ajustes; mayor varianza que (k)-fold CV con (k<n).
- **Casos adecuados:** conjuntos pequeños y ajustes baratos.
- **No conviene:** (n) grande o modelos lentos, salvo atajos analíticos disponibles.

### (k)-fold cross-validation

- **Objetivo:** estimar error y seleccionar modelos con un coste controlable.
- **Entradas:** datos, estimador, (k), esquema de división, métrica y semilla si se barajan datos.
- **Salida:** (k) errores y su media.
- **Funcionamiento:** cada fold se valida una vez y se entrena sobre los restantes.
- **Hiperparámetros:** (k), `shuffle` y `random_state`.
- **Ventajas:** general; 5 o 10 folds ofrecen un compromiso útil; menor varianza que LOOCV según el razonamiento del capítulo.
- **Limitaciones:** conserva algo de sesgo porque cada ajuste usa ((k-1)n/k) observaciones; cambia con la partición aleatoria.
- **Casos adecuados:** evaluación y selección rutinarias en regresión o clasificación.
- **No conviene:** si la estructura de muestreo impide dividir observaciones aleatoriamente; el capítulo no desarrolla esquemas especiales para datos agrupados o temporales.

### Bootstrap no paramétrico

- **Objetivo:** estimar la variabilidad de un estadístico, coeficiente o predicción.
- **Entradas:** conjunto de (n) filas, función estadística, número (B) de réplicas y semilla.
- **Salida:** distribución de (B) estimaciones y un error estándar bootstrap.
- **Funcionamiento:** obtiene muestras de tamaño (n) con reemplazo y recalcula el estadístico.
- **Hiperparámetros:** (B) y semilla.
- **Ventajas:** amplia aplicabilidad y ausencia de cálculos analíticos complicados; útil cuando el software no entrega una medida de variabilidad.
- **Limitaciones:** coste de (B) evaluaciones; la aproximación depende de que el conjunto observado represente adecuadamente la población.
- **Casos adecuados:** errores estándar de estimadores sin fórmula sencilla y variabilidad de coeficientes o predicciones.
- **No conviene:** cuando se remuestrean componentes de una fila de forma independiente o cuando el coste de reajustar (B) veces no es asumible.

## 6. Fórmulas esenciales

### LOOCV para regresión

\[
CV_{(n)}=\frac{1}{n}\sum_{i=1}^{n}MSE_i,
\qquad MSE_i=(y_i-\hat y_i^{(-i)})^2.
\]

- (n): número de observaciones.
- (y_i): respuesta observada.
- \(\hat y_i^{(-i)}\): predicción para (i) del modelo ajustado sin la observación (i).
- Mide el error cuadrático medio fuera de muestra estimado. Un valor menor favorece al procedimiento comparado.

Para mínimos cuadrados existe el atajo:

\[
CV_{(n)}=\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i-\hat y_i}{1-h_i}\right)^2,
\]

donde \(\hat y_i\) es el valor ajustado usando todos los datos y (h_i) es el *leverage* de la observación. El factor (1-h_i) infla correctamente los residuos de alta influencia. Esta igualdad no vale para modelos predictivos en general.

### (k)-fold CV para regresión

\[
CV_{(k)}=\frac{1}{k}\sum_{j=1}^{k}MSE_j.
\]

- (k): número de folds.
- (MSE_j): MSE calculado en el fold (j), excluido de su ajuste.
- Estima el error de prueba; se minimiza para seleccionar el procedimiento o su flexibilidad.

### CV para clasificación

\[
CV_{(n)}=\frac{1}{n}\sum_{i=1}^{n}Err_i,
\qquad Err_i=I(y_i\ne\hat y_i^{(-i)}).
\]

- (I(\cdot)): indicador que vale 1 si la predicción es incorrecta y 0 en otro caso.
- El promedio es la proporción estimada de clasificaciones erróneas. La versión (k)-fold se define de manera análoga.

### Error estándar bootstrap

Sean \(\hat\theta^{*1},\ldots,\hat\theta^{*B}\) las estimaciones obtenidas en (B) muestras bootstrap y

\[
\bar{\theta}^{*}=\frac{1}{B}\sum_{b=1}^{B}\hat\theta^{*b}.
\]

Entonces:

\[
SE_B(\hat\theta)=
\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}
\left(\hat\theta^{*b}-\bar{\theta}^{*}\right)^2}.
\]

- (B): número de réplicas.
- \(\hat\theta^{*b}\): estadístico en la réplica (b).
- \(SE_B\): dispersión bootstrap que aproxima el error estándar del estimador original.
- Supuesto operativo: remuestrear observaciones completas, con reemplazo, desde la muestra observada.

El capítulo también usa como ejemplo la asignación de mínima varianza entre dos activos:

\[
\alpha=\frac{\sigma_Y^2-\sigma_{XY}}
{\sigma_X^2+\sigma_Y^2-2\sigma_{XY}},
\]

donde \(\sigma_X^2\) y \(\sigma_Y^2\) son varianzas y \(\sigma_{XY}\) es la covarianza. Al sustituirlas por estimaciones muestrales se obtiene \(\hat\alpha\), cuya incertidumbre se estima por bootstrap. La fórmula ilustra el método, no una recomendación de inversión.

## 7. Implementación práctica

### (k)-fold CV sin filtrar transformaciones

```python
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

cv = KFold(n_splits=10, shuffle=True, random_state=0)
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_squared_error",
)
mean_cv_mse = -scores["test_score"].mean()
```

**Nota práctica (adaptación moderna):** el capítulo usa `ISLP.models` y un wrapper de `statsmodels`. Aquí se emplea una `Pipeline` de scikit-learn para que la expansión polinómica se ajuste dentro de cada fold. También se hace explícito que scikit-learn devuelve MSE negativo para este `scoring`.

### Validación simple reproducible

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.5, random_state=0
)
model.fit(X_train, y_train)
valid_mse = mean_squared_error(y_valid, model.predict(X_valid))
```

La transformación, si existe, debe formar parte de `model` o ajustarse solo con `X_train` antes de transformar validación.

### Bootstrap genérico para un estadístico

```python
import numpy as np

def bootstrap_standard_error(data, statistic, B=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    estimates = np.empty(B)
    n = len(data)

    for b in range(B):
        indices = rng.choice(n, size=n, replace=True)
        estimates[b] = statistic(data.iloc[indices])

    return estimates.std(ddof=1)
```

La función `statistic` debe reajustar cualquier modelo o transformación necesaria en cada réplica. **Nota práctica (adaptación moderna):** conservar las réplicas en un arreglo y usar `std(ddof=1)` expresa directamente la ecuación 5.8; el laboratorio acumula primeros y segundos momentos.

## 8. Supuestos y condiciones de aplicación

1. El error se evalúa en observaciones no usadas para ajustar el modelo correspondiente.
2. En el esquema descrito, las observaciones pueden dividirse aleatoriamente en folds; el capítulo no extiende el método a dependencias temporales o por grupos.
3. La métrica corresponde al tipo de respuesta: MSE en regresión o error de clasificación en el ejemplo cualitativo.
4. Todos los candidatos se comparan bajo un protocolo de validación coherente.
5. En bootstrap se muestrean (n) observaciones con reemplazo desde las (n) originales.
6. Las variables pertenecientes a una misma observación permanecen unidas al remuestrear.
7. La muestra observada actúa como sustituto empírico de la población para aproximar la variabilidad del estimador.
8. El atajo de LOOCV basado en leverage se limita a regresión por mínimos cuadrados; otros métodos requieren reajustes.
9. Las fórmulas analíticas de errores estándar de regresión pueden depender de supuestos sobre el ruido y predictores fijos; el ejemplo muestra que bootstrap puede ser más adecuado cuando el modelo lineal no describe bien la relación.

## 9. Errores y riesgos

- Usar error de entrenamiento para elegir flexibilidad: favorece modelos demasiado flexibles y no estima generalización.
- Ajustar transformaciones antes de separar folds: incorpora información de validación al entrenamiento y produce *data leakage*.
- Interpretar un único holdout como una estimación estable: distintos splits pueden cambiar el valor y hasta la selección.
- Comparar candidatos usando splits distintos: añade ruido de partición a la comparación.
- Confundir (K) de KNN con (k) de (k)-fold CV; son parámetros diferentes.
- Creer que LOOCV tiene necesariamente menor varianza por promediar más errores: sus ajustes están altamente correlacionados.
- Tratar la desviación entre varios `ShuffleSplit` como error estándar muestral válido: los entrenamientos se solapan y los resultados están correlacionados; el capítulo dice que solo refleja variación Monte Carlo de las particiones.
- Evaluar en el fold retenido durante decisiones internas y reutilizarlo sin control: la selección debe estar contenida dentro del procedimiento evaluado.
- Remuestrear columnas por separado en bootstrap: destruye las relaciones entre variables de cada caso.
- Omitir `replace=True`: deja de ser el bootstrap descrito.
- Usar demasiado pocas réplicas (B): añade variación Monte Carlo a la estimación del error estándar.
- Aplicar el atajo de leverage a modelos distintos de mínimos cuadrados: la identidad no es general.
- Ignorar coste: LOOCV necesita (n) ajustes y bootstrap (B); métodos lentos pueden volverlos impracticables.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

Consultar este capítulo cuando no exista un test designado, haya que escoger entre modelos o hiperparámetros, o se solicite incertidumbre de un estimador. Un baseline rápido puede usar un holdout reproducible, pero 5-fold o 10-fold CV es la elección más sólida cuando el presupuesto lo permite. La mejora principal consiste en comparar candidatos con folds comunes y encapsular todas las transformaciones dentro del ajuste de cada fold.

Señales que activan estas técnicas:

- pocos datos o resultados muy sensibles al split: pasar de holdout a (k)-fold CV;
- varios grados de polinomio, valores de (K) en KNN u otros niveles de flexibilidad: seleccionar por el mínimo de la curva CV, no por entrenamiento;
- conjunto pequeño y modelo barato: considerar LOOCV;
- estadístico sin error estándar disponible o incertidumbre de coeficientes/predicciones: bootstrap;
- coste elevado por ajuste: preferir 5 o 10 folds a LOOCV y limitar (B) a un valor justificable.

Bajo presión de tiempo no conviene implementar LOOCV manual con bucles, repetir muchos holdouts como sustituto informal de CV ni construir un sistema bootstrap complejo si solo se pide rendimiento predictivo. Tampoco conviene optimizar diferencias mínimas de error sin considerar la variación debida a la partición.

## 11. Checklist operativo

- [ ] Definir si el objetivo es estimar rendimiento, seleccionar flexibilidad o cuantificar incertidumbre.
- [ ] Elegir una métrica acorde con regresión o clasificación.
- [ ] Reservar cada fold antes de ajustar cualquier transformación.
- [ ] Usar 5 o 10 folds como opción práctica por defecto respaldada por el capítulo.
- [ ] Fijar `random_state` en divisiones aleatorias.
- [ ] Reutilizar los mismos folds al comparar candidatos.
- [ ] Revisar la media de errores fuera de muestra, no el error de entrenamiento.
- [ ] Si se usa bootstrap, muestrear filas completas con reemplazo y tamaño (n).
- [ ] Recalcular todo el estimador dentro de cada réplica bootstrap.
- [ ] Ajustar el modelo final elegido con todos los datos de entrenamiento disponibles.
- [ ] Registrar (k), semilla, métrica, (B) y coste computacional.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Holdout: procedimiento, ventajas y alta variabilidad | `knowledge/validation/holdout_validation.md` | Alta |
| LOOCV y atajo para mínimos cuadrados | `knowledge/validation/loocv.md` | Media |
| (k)-fold CV, elección de (k) y compromiso sesgo-varianza | `knowledge/validation/cross_validation.md` | Alta |
| Selección de flexibilidad mediante mínimo de error CV | `knowledge/models/model_selection.md` | Alta |
| MSE y error de clasificación dentro de CV | `knowledge/metrics/cross_validated_metrics.md` | Alta |
| Bootstrap para errores estándar de estadísticas y modelos | `knowledge/validation/bootstrap.md` | Alta |
| Riesgo de leakage al ajustar transformaciones fuera del fold | `knowledge/validation/data_leakage.md` | Alta |
| Coste de LOOCV, (k)-fold y bootstrap | `knowledge/tuning/computational_tradeoffs.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura del cap. 5, p. 201 |
| 2. Conceptos esenciales | Sec. 5.1, pp. 202–211; Sec. 5.2, pp. 212–214 |
| 3. Flujo o procedimiento presentado | Secs. 5.1.1–5.1.5, pp. 202–211; Sec. 5.2, pp. 212–214 |
| 4. Reglas de decisión | Secs. 5.1.1–5.1.4, pp. 202–209; Sec. 5.2, pp. 212–214 |
| 5. Métodos y técnicas | Secs. 5.1.1–5.1.5, pp. 202–211; Sec. 5.2, pp. 212–214 |
| 6. Fórmulas esenciales | Ecs. 5.1–5.4, pp. 204–209; Ecs. 5.6–5.8, pp. 212–214 |
| 7. Implementación práctica | Lab 5.3.1–5.3.3, pp. 215–224 |
| 8. Supuestos y condiciones | Secs. 5.1.1–5.1.5, pp. 202–211; Sec. 5.2, pp. 212–214; Lab, pp. 215–224 |
| 9. Errores y riesgos | Secs. 5.1.1–5.1.4, pp. 202–209; Lab, pp. 216–224 |
| 10. Aplicación en un coding test | Síntesis práctica derivada de Secs. 5.1–5.3, pp. 202–224 |
| 11. Checklist operativo | Síntesis práctica derivada de Secs. 5.1–5.3, pp. 202–224 |
| 12. Mapeo hacia `knowledge/` | Clasificación documental derivada del capítulo completo, pp. 201–224 |

**Cobertura:** se revisaron las 28 páginas del PDF, correspondientes a las páginas impresas 201–228. Las páginas 224–228 contienen ejercicios y se revisaron para confirmar el alcance, pero no se incorporaron como contenido sustantivo porque son problemas sin solución. No hubo secciones explicativas imposibles de procesar; algunas figuras generan texto extraído ruidoso, pero sus títulos, ejes y explicaciones en el cuerpo fueron legibles.
