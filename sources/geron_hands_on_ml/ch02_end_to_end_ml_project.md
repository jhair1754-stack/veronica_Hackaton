---
title: "End-to-End Machine Learning Project"
topics:
  - "flujo end-to-end de Machine Learning"
  - "regresión supervisada"
  - "partición estratificada y validación"
  - "EDA y feature engineering"
  - "pipelines de preprocesamiento"
  - "selección, ajuste y despliegue de modelos"
target_knowledge_files:
  - "knowledge/problem_definition/framing_ml_projects.md"
  - "knowledge/eda/tabular_eda.md"
  - "knowledge/preprocessing/tabular_pipelines.md"
  - "knowledge/validation/train_validation_test.md"
  - "knowledge/metrics/regression_metrics.md"
  - "knowledge/tuning/hyperparameter_search.md"
  - "knowledge/models/regression_baselines.md"
---

# 2. End-to-End Machine Learning Project

## 1. Propósito del capítulo

El capítulo recorre un proyecto de regresión supervisada de principio a fin usando datos censales de vivienda de California. El objetivo operativo es predecir el valor mediano de las viviendas de un distrito a partir de sus atributos. El caso conecta definición del problema, adquisición de datos, análisis exploratorio, partición, preprocesamiento reproducible, comparación de modelos, ajuste, evaluación final y operación en producción.

**Explicado explícitamente por el autor:** el trabajo no termina al entrenar un modelo; comienza con el objetivo de negocio y continúa hasta el despliegue, la monitorización y el reentrenamiento.

**Conclusión práctica derivada:** el valor principal del capítulo es el orden del proceso. Cada decisión posterior depende de proteger el conjunto de prueba, comprender los datos y automatizar las transformaciones.

## 2. Conceptos esenciales

### Encuadre y métrica

- El objetivo de negocio determina la formulación, los algoritmos, la métrica y el esfuerzo de ajuste.
- El caso es aprendizaje **supervisado**, **regresión múltiple** y **univariada**: usa varias variables para predecir una salida continua.
- El aprendizaje por lotes (*batch learning*) basta cuando no hay flujo continuo, no se necesita adaptación rápida y los datos caben en memoria.
- Debe verificarse el uso posterior de la predicción. Si el consumidor solo necesita categorías, una regresión puede ser una formulación equivocada.
- **RMSE (Root Mean Square Error)** expresa el error en las unidades del objetivo y penaliza especialmente errores grandes. **MAE (Mean Absolute Error)** es menos sensible a outliers.
- RMSE corresponde a la norma euclídea o \(\ell_2\); MAE, a \(\ell_1\). Cuanto mayor es el orden de la norma, más pesan los errores grandes.

### Inspección y EDA

- `head()`, `info()`, `value_counts()` y `describe()` permiten revisar estructura, tipos, categorías, nulos, rangos, dispersión y percentiles.
- Los histogramas descubren escalas distintas, topes artificiales y distribuciones sesgadas.
- Un objetivo truncado impide aprender valores superiores al límite. El capítulo propone obtener etiquetas correctas o excluir los casos afectados si el sistema debe predecir más allá del tope.
- La exploración se hace solo con entrenamiento para evitar *data snooping bias*.
- La correlación de Pearson mide asociación lineal entre -1 y 1; no detecta relaciones no lineales ni implica causalidad.
- Los scatter plots revelan densidad, relaciones con el objetivo, topes y artefactos en las etiquetas.

### Partición y preparación

- Una división aleatoria reproducible usa `random_state`, pero puede cambiar al refrescar el dataset.
- Una partición por hash de un identificador único e inmutable mantiene estable la pertenencia train/test.
- El muestreo estratificado conserva proporciones de una variable importante; no deben existir demasiados estratos ni estratos pequeños.
- Para nulos: eliminar filas, eliminar la variable o imputar. La mediana debe aprenderse solo en entrenamiento y reutilizarse en validación, prueba y producción.
- `OrdinalEncoder` introduce orden y proximidad; para categorías nominales corresponde `OneHotEncoder`.
- One-hot suele generar matrices dispersas y puede explotar en dimensionalidad si la cardinalidad es alta.
- Transformadores personalizados encapsulan limpieza o atributos derivados. `BaseEstimator` y `TransformerMixin` los integran con Scikit-Learn.
- Min-max lleva valores a un intervalo, usualmente [0, 1], pero es sensible a outliers. La estandarización centra y divide por la desviación estándar; no acota, pero resiste mejor valores extremos.
- `Pipeline` encadena operaciones y `ColumnTransformer` aplica transformaciones distintas a columnas numéricas y categóricas.

### Modelado, ajuste y operación

- `LinearRegression` ofrece un baseline. Error alto en train sugiere underfitting por falta de capacidad o de atributos útiles.
- `DecisionTreeRegressor` captura relaciones no lineales, pero un error nulo en train puede revelar overfitting.
- K-fold cross-validation produce una distribución de scores; deben reportarse media y dispersión, a cambio de mayor coste.
- `neg_mean_squared_error` usa signo negativo porque el API de scoring maximiza; se debe negar antes de aplicar la raíz.
- Random Forest mejora la validación en el ejemplo, aunque la brecha frente a train todavía indica overfitting.
- Antes del tuning intensivo se comparan varias familias y se conservan entre dos y cinco candidatos.
- `GridSearchCV` es apropiado para espacios pequeños; `RandomizedSearchCV`, para espacios grandes y presupuesto fijo.
- Pasos de preparación inciertos también pueden tratarse como hiperparámetros.
- Importancias y errores específicos ayudan a eliminar atributos, crear otros y detectar outliers.
- Test se usa una sola vez al final con `transform()`, no `fit_transform()`.
- Producción requiere monitorizar rendimiento y calidad de inputs, automatizar reentrenamiento y conservar modelos, datasets y rollback.

## 3. Flujo o procedimiento presentado

1. Definir el objetivo de negocio y cómo se consumirá la predicción.
2. Formular la tarea: supervisada/no supervisada, clasificación/regresión y batch/online.
3. Elegir RMSE o MAE según el coste de errores grandes y la presencia de outliers.
4. Verificar supuestos con los responsables del sistema posterior.
5. Automatizar la obtención y carga de datos.
6. Inspeccionar esquema, nulos, categorías, estadísticas e histogramas.
7. Crear y aislar test mediante división aleatoria, por ID o estratificada.
8. Explorar únicamente entrenamiento.
9. Crear combinaciones de atributos con sentido y reevaluar su asociación con el target.
10. Separar predictores y etiqueta.
11. Encapsular imputación, codificación, feature engineering y escalado.
12. Entrenar baselines y modelos candidatos.
13. Comparar con cross-validation usando media y desviación estándar.
14. Conservar una shortlist de dos a cinco modelos.
15. Afinar con grid search o randomized search.
16. Analizar importancias y errores del mejor modelo.
17. Evaluar una vez en test sin reajustar transformaciones.
18. Comunicar resultados, límites y supuestos.
19. Desplegar, monitorizar, versionar y preparar rollback y reentrenamiento.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Errores grandes especialmente costosos y outliers raros | RMSE | Penaliza cuadráticamente errores grandes | Sensible a outliers |
| Muchos outliers relevantes | MAE | Penalización lineal más robusta | Enfatiza menos los fallos grandes |
| Dataset grande sin subgrupo crítico | Split aleatorio reproducible | Simple y normalmente representativo | Puede sesgar muestras pequeñas |
| Dataset que se actualizará con ID estable | Split por hash | Mantiene estable train/test | El ID debe ser único e inmutable |
| Variable crítica desbalanceada | Muestreo estratificado | Conserva proporciones | Pocos estratos y suficientemente grandes |
| Nulos en una variable útil | Imputación mediana | Conserva filas | Ajustar solo con train |
| Categoría nominal | One-hot encoding | Evita orden artificial | Alta cardinalidad aumenta coste |
| Variables en escalas distintas | Estandarización o min-max | Ayuda a algoritmos sensibles a escala | Ajustar solo con train; min-max es sensible a outliers |
| Espacio pequeño y discreto | `GridSearchCV` | Búsqueda exhaustiva | Coste combinaciones × folds |
| Espacio grande | `RandomizedSearchCV` | Controla presupuesto y explora más valores | Definir distribuciones razonables |
| Error alto en entrenamiento | Más capacidad o mejores variables | Compatible con underfitting | Confirmar con validación |
| Train casi perfecto y CV peor | Simplificar, regularizar o aportar datos | Compatible con overfitting | No seleccionar por score de train |
| Modelo final elegido | Evaluación única en test | Estimación independiente | No retocar después mirando test |

## 5. Métodos y técnicas

### Partición estratificada

- **Objetivo:** conservar proporciones de subgrupos importantes.
- **Entradas:** filas, etiqueta de estrato y proporción de test.
- **Salida:** índices train/test.
- **Funcionamiento:** discretiza una variable relevante y divide preservando proporciones.
- **Hiperparámetros:** `n_splits`, `test_size`, `random_state` y límites de estratos.
- **Ventajas:** reduce sesgo de muestreo.
- **Limitaciones:** exige criterio de dominio; demasiados estratos producen grupos pequeños.
- **Caso adecuado:** variable predictiva importante y distribución desigual.
- **No conviene:** estratificación arbitraria o con categorías escasas.

### `SimpleImputer`

- **Objetivo:** sustituir nulos con una estadística aprendida.
- **Entradas/salida:** matriz con nulos / matriz imputada.
- **Funcionamiento:** `fit()` aprende por columna; `transform()` reutiliza.
- **Hiperparámetro:** `strategy="median"` en el capítulo.
- **Ventajas:** simple, automatizable y consistente.
- **Limitaciones:** no modela relaciones entre variables.
- **Caso adecuado:** baseline tabular numérico.
- **No conviene:** ajustarlo con datos de validación o test.

### Codificación categórica

- **Objetivo:** convertir categorías a números.
- **Entrada/salida:** columnas categóricas / matriz ordinal o one-hot.
- **Funcionamiento:** aprende categorías en `fit()` y transforma después.
- **Ventaja de one-hot:** no impone distancias falsas.
- **Limitación:** cardinalidad alta produce muchas columnas y mayor coste.
- **Caso adecuado:** categorías nominales como proximidad al océano.
- **No conviene:** ordinal encoding para categorías sin orden natural.

### Transformadores, escalado y pipelines

- **Objetivo:** aplicar preparación idéntica en train, validación, test e inferencia.
- **Entradas/salida:** DataFrame heterogéneo / matriz lista para el modelo.
- **Funcionamiento:** `Pipeline` encadena; `ColumnTransformer` ramifica por columnas y concatena.
- **Hiperparámetros:** parámetros de pasos mediante sus nombres; `sparse_threshold` controla salida dispersa.
- **Ventajas:** reproducibilidad, composición y tuning conjunto.
- **Limitaciones:** todos los transformadores deben conservar filas; la salida puede ser densa o sparse.
- **Casos adecuados:** imputación, escalado, one-hot y ratios tabulares.
- **No conviene:** lógica manual diferente entre entrenamiento e inferencia.

### Modelos comparados

| Método | Objetivo | Hiperparámetros destacados | Ventajas | Limitaciones |
| --- | --- | --- | --- | --- |
| `LinearRegression` | Relación lineal | No se ajustan en el capítulo | Baseline rápido | Underfitting en relaciones complejas |
| `DecisionTreeRegressor` | Relaciones no lineales por particiones | No se detallan aquí | Flexible | Memoriza con facilidad |
| `RandomForestRegressor` | Promedio de múltiples árboles | `n_estimators`, `max_features`, `bootstrap` | Mejor CV del ejemplo | Más coste y posible overfitting |

### Cross-validation y tuning

- **Objetivo:** estimar generalización y seleccionar configuraciones sin tocar test.
- **Entradas:** estimador, X, y, métrica, folds y espacio de parámetros.
- **Salida:** score por fold, mejor configuración y estimador reajustado.
- **Hiperparámetros:** `cv`, `scoring`, `refit`; `n_iter` en búsqueda aleatoria.
- **Ventajas:** comparación sistemática y medida de variabilidad.
- **Limitaciones:** coste multiplicado por folds y candidatos.
- **Caso adecuado:** shortlist ya definida.
- **No conviene:** usar test como criterio o hacer grids enormes sin filtro previo.

## 6. Fórmulas esenciales

### RMSE

\[
\operatorname{RMSE}(X,h)=\sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(h(x^{(i)})-y^{(i)}\right)^2}
\]

- \(m\): número de instancias; \(x^{(i)}\): atributos de la instancia; \(y^{(i)}\): valor real; \(h\): función de predicción; \(X\): matriz de atributos.
- Mide la raíz del promedio de errores cuadrados. Menor es mejor y queda en unidades del target.
- Penaliza errores grandes; resulta apropiada cuando los outliers son raros.

### MAE

\[
\operatorname{MAE}(X,h)=\frac{1}{m}\sum_{i=1}^{m}\left|h(x^{(i)})-y^{(i)}\right|
\]

- Las variables tienen el significado anterior.
- Mide la magnitud absoluta media del error. Menor es mejor.
- Es más robusta a outliers que RMSE.

### Norma \(\ell_k\)

\[
\lVert v\rVert_k=\left(|v_1|^k+\cdots+|v_n|^k\right)^{1/k}
\]

- \(v\): vector de errores; \(v_j\): componente; \(n\): cantidad de componentes; \(k\): orden.
- Al crecer \(k\), los valores grandes dominan más la medida.

## 7. Implementación práctica

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

income_strata = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(housing, income_strata))

train_data = housing.iloc[train_idx]
test_data = housing.iloc[test_idx]
X_train = train_data.drop(columns="median_house_value")
y_train = train_data["median_house_value"]
X_test = test_data.drop(columns="median_house_value")
y_test = test_data["median_house_value"]

numeric_columns = X_train.select_dtypes(include="number").columns
categorical_columns = X_train.select_dtypes(exclude="number").columns

preprocessing = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_columns),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
])
model = Pipeline([
    ("preprocessing", preprocessing),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

cv_mse = -cross_val_score(
    model, X_train, y_train,
    scoring="neg_mean_squared_error", cv=5, n_jobs=-1,
)
cv_rmse = np.sqrt(cv_mse)
print(cv_rmse.mean(), cv_rmse.std())

model.fit(X_train, y_train)
test_rmse = root_mean_squared_error(y_test, model.predict(X_test))
```

**Nota práctica actualizada (no es contenido original del libro):** se usa `root_mean_squared_error`; el capítulo emplea `np.sqrt(mean_squared_error(...))`. También se añaden `handle_unknown="ignore"`, `random_state`, `n_jobs` y el predictor dentro del pipeline. Esto evita leakage en cada fold. Si la versión instalada no contiene `root_mean_squared_error`, debe usarse la forma original.

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    model,
    param_distributions={
        "regressor__n_estimators": [100, 300, 600],
        "regressor__max_features": [0.5, 0.8, 1.0],
        "regressor__bootstrap": [True, False],
    },
    n_iter=8,
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
```

**Nota práctica actualizada:** el prefijo `regressor__` ajusta parámetros del estimador dentro del pipeline; el capítulo ajusta el bosque después del preprocesamiento separado.

## 8. Supuestos y condiciones de aplicación

1. La salida numérica es lo que necesita el consumidor posterior.
2. Las etiquetas representan el fenómeno y no solo una versión truncada.
3. Los datos son ejemplos supervisados con un objetivo continuo.
4. El régimen batch satisface volumen y velocidad de actualización.
5. La variable de estratificación es relevante y forma estratos poblados.
6. El ID para split estable es único e inmutable.
7. Pearson se interpreta solo como asociación lineal.
8. Imputación, escalado y categorías se aprenden solo con train.
9. One-hot se usa para categorías sin orden y cardinalidad manejable.
10. Los transformadores conservan el número de filas.
11. Los folds son compatibles con la estructura; el capítulo no cubre series temporales ni grupos.
12. Test permanece intacto hasta el modelo final.
13. Producción permite observar rendimiento, proxies o calidad de entradas.

## 9. Errores y riesgos

- Explorar test o decidir tras ver su score causa *data snooping*.
- Ajustar imputación o escalado con todo el dataset produce leakage; en CV deben ajustarse dentro de cada fold.
- Una división aleatoria pequeña puede deformar subgrupos.
- Refrescar datos puede mover filas entre train y test si el split no es estable.
- Un target truncado enseña un límite artificial.
- Correlación cercana a cero no descarta relación no lineal; correlación no es causalidad.
- Enteros para categorías nominales crean ordinalidad falsa.
- One-hot de alta cardinalidad consume memoria.
- Error alto en train indica posible underfitting; train perfecto con CV peor, posible overfitting.
- `neg_mean_squared_error` exige invertir signo antes de la raíz.
- K-fold y grid search multiplican el coste de entrenamiento.
- Un mejor valor en el borde del grid sugiere ampliar el rango.
- Ajustar tras ver test destruye su independencia.
- Drift, datos obsoletos, fuentes rotas y categorías nuevas degradan producción.
- Sin versiones y rollback no se pueden restaurar ni comparar modelos y datasets.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo en regresión tabular donde importe justificar el flujo completo.
- Baseline: split protegido, `ColumnTransformer`, imputación, one-hot, un modelo lineal o Random Forest y CV con la métrica pedida.
- Mejoras de alto retorno: ratios con sentido, estratificación, transformación de sesgos, comparación lineal/no lineal y búsqueda aleatoria moderada.
- Evitar bajo presión: despliegue, monitorización completa, ensembles extensos y grids grandes salvo requisito explícito.
- Señales: nulos → imputación; texto nominal → one-hot; escalas diferentes → escalado; variable crítica desbalanceada → estratificación; target limitado → advertencia; brecha train-CV → overfitting.
- **Recomendación para competencias:** reservar un holdout local que imite la evaluación y no ajustarlo iterativamente. El capítulo no cubre leaderboard; esta regla deriva de mantener test intacto.
- **Recomendación para competencias:** usar un pipeline único para que la CV no filtre estadísticas entre folds.
- **Recomendación para competencias:** comparar primero dos a cinco familias y afinar después.

## 11. Checklist operativo

- [ ] Confirmar objetivo y uso de la predicción.
- [ ] Formular tarea y régimen de aprendizaje.
- [ ] Elegir RMSE o MAE con justificación.
- [ ] Inspeccionar esquema, nulos, categorías y topes.
- [ ] Crear test antes del EDA detallado.
- [ ] Justificar split aleatorio, por ID o estratificado.
- [ ] Explorar y crear atributos solo con train.
- [ ] Separar predictores y objetivo.
- [ ] Encapsular preparación en pipelines.
- [ ] Ajustar estadísticas solo con train o cada fold.
- [ ] Comparar baseline y modelos no lineales con CV.
- [ ] Revisar media, dispersión y brecha train-CV.
- [ ] Afinar solo una shortlist.
- [ ] Analizar importancias y errores.
- [ ] Evaluar test una sola vez.
- [ ] Documentar límites, monitorización y rollback.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Formulación de proyectos | `knowledge/problem_definition/framing_ml_projects.md` | Alta |
| EDA tabular y correlación | `knowledge/eda/tabular_eda.md` | Alta |
| Topes, sesgos y artefactos | `knowledge/eda/data_quality_signals.md` | Alta |
| Imputación, codificación y escalado | `knowledge/preprocessing/tabular_preprocessing.md` | Alta |
| Pipelines y transformadores | `knowledge/preprocessing/tabular_pipelines.md` | Alta |
| Ratios tabulares | `knowledge/feature_engineering/tabular_ratios.md` | Media |
| Split, estratificación y test | `knowledge/validation/train_validation_test.md` | Alta |
| K-fold | `knowledge/validation/cross_validation.md` | Alta |
| RMSE y MAE | `knowledge/metrics/regression_metrics.md` | Alta |
| Baselines de regresión | `knowledge/models/regression_baselines.md` | Media |
| Diagnóstico bias-varianza | `knowledge/models/bias_variance_diagnostics.md` | Alta |
| Grid y randomized search | `knowledge/tuning/hyperparameter_search.md` | Alta |
| Ensembles | `knowledge/ensembles/ensemble_basics.md` | Baja |
| Importancias y errores | `knowledge/interpretability/feature_importance_and_error_analysis.md` | Media |
| Monitorización y rollback | `knowledge/validation/production_monitoring.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| Propósito y flujo | Apertura, pp. 35–36; desarrollo, pp. 37–84 |
| Encuadre y supuestos | “Look at the Big Picture”, “Frame the Problem” y “Check the Assumptions”, pp. 37–42 |
| Métricas y fórmulas | “Select a Performance Measure”, pp. 39–41 |
| Inspección y EDA | “Take a Quick Look…”, pp. 47–51; “Discover and Visualize…”, pp. 55–62 |
| Partición | “Create a Test Set”, pp. 51–55 |
| Preparación | “Prepare the Data for Machine Learning Algorithms”, pp. 62–72 |
| Modelos y CV | “Select and Train a Model”, pp. 72–75 |
| Ajuste y análisis | “Fine-Tune Your Model”, pp. 75–80 |
| Implementación | Split, pp. 52–55; preparación, pp. 63–71; CV, pp. 72–75; tuning/test, pp. 76–80. Las adaptaciones modernas están marcadas |
| Errores y riesgos | Muestreo, pp. 51–55; artefactos, pp. 50–62; leakage/overfitting, pp. 63–80; producción, pp. 80–83 |
| Coding test y checklist | Síntesis práctica derivada de pp. 35–84 |
| Despliegue y mantenimiento | “Launch, Monitor, and Maintain Your System”, pp. 80–83 |
| Cierre | “Try It Out!” y “Exercises”, pp. 83–84; no se incorporaron ejercicios sin resolver |

**Estado de lectura:** se revisaron las 50 páginas del PDF, correspondientes a las páginas impresas 35–84. No se detectaron páginas ilegibles ni secciones que exigieran completar vacíos mediante suposiciones. Los apéndices mencionados no fueron necesarios y no se consultaron.
