---
title: "Training Models"
topics:
  - "linear regression and optimization"
  - "gradient descent"
  - "polynomial regression and regularization"
  - "logistic and softmax regression"
target_knowledge_files:
  - "knowledge/models/linear_models.md"
  - "knowledge/tuning/gradient_descent.md"
  - "knowledge/validation/learning_curves.md"
  - "knowledge/models/regularized_regression.md"
  - "knowledge/models/logistic_softmax_regression.md"
---

# 4. Training Models

## 1. Propósito del capítulo

El capítulo abre la caja negra del entrenamiento de modelos. Explica cómo ajustar modelos lineales mediante soluciones cerradas y optimización iterativa, cómo representar relaciones no lineales con Polynomial Regression, cómo diagnosticar underfitting/overfitting y cómo regularizar. Termina con clasificación probabilística binaria y multiclase mediante Logistic Regression y Softmax Regression.

Su utilidad práctica es elegir un algoritmo de entrenamiento, interpretar su convergencia, controlar la complejidad y construir baselines reproducibles para regresión y clasificación.

Las afirmaciones se distinguen así: **Explicado por el autor** es contenido explícito; **Conclusión práctica derivada** es una consecuencia operativa; **Recomendación para competencias** adapta el contenido al coding test.

## 2. Conceptos esenciales

### Modelos lineales y función de coste

- **Explicado por el autor:** Linear Regression calcula una suma ponderada de features más un intercepto. El entrenamiento busca los parámetros que minimizan el error; minimizar MSE produce los mismos parámetros que minimizar RMSE y es más conveniente para optimización.
- La forma vectorizada es `ŷ = θᵀx`, incluyendo `x₀ = 1` para el bias. `θ` contiene intercepto y pesos.
- La Normal Equation obtiene una solución cerrada. La pseudoinversa basada en SVD es más robusta cuando `XᵀX` es singular (features redundantes o menos instancias que features).

### Gradient Descent (GD)

- **Explicado por el autor:** GD inicializa parámetros y los mueve iterativamente en la dirección opuesta al gradiente. `learning rate (η)` controla el tamaño del paso: demasiado bajo ralentiza; demasiado alto puede divergir.
- Para Linear Regression con MSE, la función es convexa: existe un único mínimo global. El escalado de features es necesario para converger con rapidez.
- **Batch GD** usa todo el training set por paso; **Stochastic GD** usa una instancia y puede entrenar out-of-core; **Mini-batch GD** usa pequeños grupos y aprovecha operaciones matriciales/GPU.
- SGD oscila alrededor del mínimo; un learning schedule decreciente permite asentarse. Las instancias deben ser IID o barajarse conjuntamente con sus targets.

### Polynomial Regression y diagnóstico

- **Explicado por el autor:** `PolynomialFeatures` añade potencias y combinaciones de features; después se ajusta un modelo lineal sobre las features expandidas.
- El número de features crece combinatoriamente: con `n` features y grado `d` es `(n+d)!/(d!n!)` (sin contar ajustes de `include_bias`).
- **Learning curves** trazan error de entrenamiento y validación contra el tamaño de entrenamiento. Curvas próximas y altas sugieren underfitting; una brecha grande con training mucho mejor sugiere overfitting.
- La complejidad eleva típicamente la varianza y reduce el bias; simplificar hace lo contrario. El error irreducible procede del ruido de los datos.

### Regularización

- **Ridge (L2):** penaliza el cuadrado de los pesos y los reduce sin llevarlos normalmente a cero. `alpha` controla la fuerza.
- **Lasso (L1):** puede llevar pesos a cero, realizando selección de features; su coste no es diferenciable en cero y usa subgradientes.
- **Elastic Net:** mezcla L1 y L2 con `l1_ratio`; `0` equivale a Ridge y `1` a Lasso.
- La regularización no se añade a la métrica final de evaluación. Los modelos regularizados son sensibles a escala, por lo que las features deben escalarse.
- **Early stopping:** detiene un algoritmo iterativo cuando el error de validación alcanza su mínimo y recupera los mejores parámetros si la curva es ruidosa.

### Clasificación probabilística

- **Logistic Regression:** aplica la sigmoid a `xᵀθ` para obtener `P(y=1|x)`. Con umbral 0.5 produce una frontera lineal; `C` en Scikit-Learn es la inversa de la fuerza de regularización (mayor `C`, menor regularización). El capítulo indica que L2 es la opción predeterminada de la clase.
- **Softmax Regression (Multinomial Logistic Regression):** calcula un score por clase, normaliza con softmax y elige la clase de mayor probabilidad. Es multiclase de etiqueta única, no multioutput: las clases deben ser mutuamente excluyentes.

## 3. Flujo o procedimiento presentado

1. Definir la tarea, el target y la métrica de evaluación.
2. Construir una matriz de features con intercepto cuando se use una formulación manual.
3. Elegir una solución cerrada (Normal Equation/SVD) o GD según tamaño, número de features y memoria.
4. Escalar features antes de GD o de modelos regularizados.
5. Separar training y validación; reservar test para la evaluación final.
6. Ajustar un baseline lineal y, si hay no linealidad, expandir con `PolynomialFeatures` dentro de un pipeline.
7. Inspeccionar learning curves y errores de training/validación para diagnosticar bias/variance.
8. Reducir complejidad, aumentar datos/features o regularizar según el diagnóstico.
9. Seleccionar Ridge, Lasso o Elastic Net; ajustar `alpha`/`l1_ratio` sin consultar el test.
10. Para entrenamiento iterativo, definir learning rate, iteraciones/tolerancia, schedule y early stopping.
11. Para clasificación, usar Logistic Regression binaria o Softmax multiclase según la exclusividad de las clases.
12. Reentrenar la configuración elegida con los datos de entrenamiento disponibles y evaluar una vez en test.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Pocas features, dataset en memoria y solución lineal | Normal Equation o `LinearRegression` (SVD) | Solución directa y predicción rápida | La inversión se vuelve cara con muchas features; la Normal Equation puede ser singular |
| Muchas features o dataset grande | Gradient Descent | Escala mejor con `n` y puede ser incremental | Escalar features y ajustar `η`/tolerancia |
| Dataset no cabe en memoria | Stochastic o Mini-batch GD | Admiten entrenamiento out-of-core | Requieren orden/mezcla adecuada y schedule |
| Relación no lineal | `PolynomialFeatures` + modelo lineal | Convierte la no linealidad en features | Explosión combinatoria y riesgo de overfitting |
| Curvas train/validation altas y cercanas | Modelo más complejo o mejores features | Señal de underfitting/high bias | Más datos por sí solos no solucionan el underfitting |
| Training mucho mejor que validation | Más datos, regularización o menor complejidad | Señal de overfitting/high variance | No ajustar mirando repetidamente el test |
| Muchos pesos, se quieren reducir sin selección dura | Ridge | L2 suaviza todos los pesos | Escalar features; `alpha` alto puede underfit |
| Se sospechan pocas features útiles | Lasso o Elastic Net | L1 puede producir modelo sparse | Lasso puede ser inestable con features correlacionadas o `n > m`; Elastic Net suele ser preferible |
| Entrenamiento iterativo empieza a empeorar en validación | Early stopping | Detiene antes de memorizar el training set | Con SGD/Mini-batch esperar varios pasos y guardar el mejor estado |
| Dos clases | Logistic Regression | Probabilidad binaria y frontera lineal | Ajustar regularización/umbral a la tarea |
| Varias clases mutuamente excluyentes | Softmax Regression | Probabilidades normalizadas por clase | No sirve para etiquetas simultáneas/multioutput |

## 5. Métodos y técnicas

### Linear Regression: Normal Equation y SVD

- **Objetivo:** minimizar MSE en una relación lineal.
- **Entradas:** `X` con intercepto y vector `y` numérico.
- **Salida:** `θ`, predicciones y, con Scikit-Learn, `intercept_`/`coef_`.
- **Funcionamiento:** Normal Equation resuelve directamente; `LinearRegression` usa least squares con SVD/pseudoinversa.
- **Hiperparámetros:** ninguno esencial en la solución cerrada.
- **Ventajas:** simple, exacto para el objetivo cuadrático, predicción rápida.
- **Limitaciones:** linealidad, memoria y coste cuadrático/cúbico en el número de features; la Normal Equation falla si la matriz es singular.
- **Adecuado:** baselines lineales y datasets que caben en memoria.
- **No conviene:** cientos de miles de features o no linealidad sin transformación.

### Gradient Descent

- **Objetivo:** minimizar una función de coste mediante pasos iterativos.
- **Entradas:** datos escalados, parámetros iniciales, learning rate y criterio de parada.
- **Salida:** parámetros aproximados al mínimo.
- **Hiperparámetros:** `eta`, número de iteraciones/épocas, tolerancia, schedule, tamaño de mini-batch.
- **Ventajas:** maneja muchas features; SGD y Mini-batch permiten out-of-core.
- **Limitaciones:** sensible a escala, `eta` y orden; SGD no se asienta sin schedule adecuado.
- **Casos adecuados:** modelos grandes o funciones sin solución cerrada.
- **No conviene:** si una solución cerrada es barata y el dataset pequeño.

### Polynomial Regression

- **Objetivo:** ajustar relaciones no lineales con un estimador lineal sobre features polinómicas.
- **Entradas:** `X` numérica y grado `d`.
- **Salida:** matriz expandida y predicciones.
- **Hiperparámetros:** `degree`, `include_bias` y regularización del estimador posterior.
- **Ventajas:** expresa curvaturas e interacciones.
- **Limitaciones:** crecimiento combinatorio, extrapolación peligrosa y overfitting con grados altos.
- **Adecuado:** no linealidad moderada con pocas features.
- **No conviene:** grados altos sin regularización, validación o control de memoria.

### Ridge, Lasso y Elastic Net

- **Objetivo:** limitar pesos para reducir varianza.
- **Entradas:** features escaladas, target, `alpha` y, en Elastic Net, `l1_ratio`.
- **Salida:** modelo lineal regularizado; Lasso/Elastic Net pueden ser sparse.
- **Funcionamiento:** añaden penalizaciones L2, L1 o una mezcla al coste de entrenamiento.
- **Ventajas:** controlan overfitting; Lasso selecciona features; Elastic Net combina estabilidad y sparsity.
- **Limitaciones:** `alpha` mal elegido causa underfitting; Lasso puede comportarse erráticamente con correlación fuerte.
- **Adecuado:** polynomial features, alta dimensionalidad y modelos con riesgo de overfitting.
- **No conviene:** comparar pesos sin escalar features o usar la penalización como métrica final.

### Logistic Regression

- **Objetivo:** estimar probabilidad de una clase positiva y clasificar binariamente.
- **Entradas:** features, labels 0/1 y `C`/penalty.
- **Salida:** probabilidad, clase y frontera lineal.
- **Funcionamiento:** sigmoid del score; entrena minimizando log loss mediante GD u otro optimizador.
- **Ventajas:** probabilidades interpretables y decisión lineal eficiente.
- **Limitaciones:** frontera lineal; el umbral 0.5 no es universal.
- **Adecuado:** dos clases y features con relación aproximadamente lineal en log-odds.
- **No conviene:** clases no linealmente separables sin features adecuadas o etiquetas simultáneas.

### Softmax Regression

- **Objetivo:** clasificar una instancia en una única clase entre `K` clases.
- **Entradas:** features y una etiqueta por instancia.
- **Salida:** vector de probabilidades que suma 1 y clase `argmax`.
- **Funcionamiento:** score por clase, softmax y cross entropy.
- **Hiperparámetros:** `C`, solver compatible (el ejemplo usa `lbfgs`) y configuración multinomial.
- **Ventajas:** modelo multiclase directo con fronteras lineales.
- **Limitaciones:** no modela multilabel/multioutput; requiere clases mutuamente excluyentes.
- **Adecuado:** una sola especie/clase por instancia.
- **No conviene:** cuando varias etiquetas pueden ser ciertas simultáneamente.

## 6. Fórmulas esenciales

### Predicción y coste lineales

`ŷ = θᵀx`, donde `θ` es el vector de parámetros, `x` el vector de features con `x₀=1` y `ŷ` la predicción. El MSE es `MSE(θ) = (1/m) Σᵢ(θᵀxᵢ − yᵢ)²`; mide el promedio del error cuadrático y penaliza más los errores grandes. Minimizar MSE y RMSE produce el mismo óptimo; se supone target numérico y observaciones representadas por el modelo.

La Normal Equation es `θ = (XᵀX)⁻¹Xᵀy`. `X` incluye el intercepto y `y` los targets. Requiere invertibilidad; si hay singularidad, la pseudoinversa SVD es la alternativa robusta.

### Gradient Descent

`∇θ MSE = (2/m)Xᵀ(Xθ − y)` es el gradiente completo. La actualización es `θ_next = θ − η∇θMSE`. `η` es el learning rate: bajo converge lentamente; alto puede divergir. En GD convexo la solución se aproxima al mínimo global si se espera suficiente y `η` no es demasiado grande.

### Regularización

- Ridge: `J(θ)=MSE(θ)+ (α/2)Σᵢ₌₁ⁿ θᵢ²`. `α` controla la penalización; no se regulariza `θ₀`. Reduce magnitudes y varianza.
- Lasso: `J(θ)=MSE(θ)+ αΣᵢ₌₁ⁿ|θᵢ|`. Puede hacer pesos exactamente cero y seleccionar features.
- Elastic Net: `J(θ)=MSE(θ)+rαΣ|θᵢ|+((1−r)/2)αΣθᵢ²`, con `r` entre 0 y 1. Interpola Lasso/Ridge.

### Logistic Regression

`p=σ(xᵀθ)=1/(1+exp(−xᵀθ))`; `p` es la probabilidad de la clase positiva. Se predice 1 si `p≥0.5`; el umbral corresponde a `xᵀθ≥0`. El log loss es `J(θ)=−(1/m)Σ[yᵢlog pᵢ +(1−yᵢ)log(1−pᵢ)]`; penaliza probabilidades seguras pero equivocadas. El logit es `log(p/(1−p))`.

### Softmax

`sₖ(x)=xᵀθₖ` es el score de la clase `k`; `pₖ=exp(sₖ)/Σⱼexp(sⱼ)` es su probabilidad; la predicción es `argmaxₖ pₖ` (equivalente a `argmaxₖ sₖ`). La cross entropy es `J(Θ)=−(1/m)ΣᵢΣₖ yₖⁱ log pₖⁱ`, donde `yₖⁱ` vale 1 para la clase correcta y 0 para las demás. Requiere una clase objetivo por instancia.

## 7. Implementación práctica

Los fragmentos son mínimos y actualizan APIs antiguas del libro (por ejemplo, `np.int` y argumentos de versiones antiguas) como **nota práctica**, no como contenido original.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
valid_predictions = model.predict(X_valid)
```

La transformación se ajusta dentro del pipeline y solo con training, evitando leakage. Para un baseline lineal se puede sustituir el pipeline por `LinearRegression()`; para clasificación binaria o multiclase:

```python
from sklearn.linear_model import LogisticRegression

binary_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
binary_model.fit(X_train, y_train)
probabilities = binary_model.predict_proba(X_valid)
```

**Nota práctica de API:** el capítulo usa `LogisticRegression(multi_class="multinomial", solver="lbfgs")`; comprobar la versión instalada antes de fijar opciones de `multi_class`, pues la API moderna puede cambiar defaults. No copiar `np.int`; usar `int` o un dtype explícito.

## 8. Supuestos y condiciones de aplicación

1. La relación lineal es una aproximación útil para Linear/Ridge/Lasso/Logistic/Softmax; Polynomial Regression amplía la representación, pero sigue usando un estimador lineal.
2. MSE es diferenciable y convexo para Linear Regression; GD converge cerca del óptimo con learning rate apropiado.
3. GD y regularización requieren escalas comparables; de lo contrario la convergencia y la penalización quedan dominadas por unidades grandes.
4. SVD/pseudoinversa necesita que los datos quepan en memoria, aunque evita la singularidad de `XᵀX`.
5. SGD presupone instancias IID o barajadas conjuntamente; ordenarlas por clase puede impedir convergencia adecuada.
6. PolynomialFeatures presupone que potencias/interacciones representan la señal y que el tamaño expandido es computable.
7. Las learning curves requieren una partición de validación independiente y un error comparable entre tamaños.
8. La evaluación debe usar la métrica sin penalización; las penalizaciones solo forman parte del entrenamiento.
9. Logistic Regression estima log-odds lineales y Softmax exige clases mutuamente excluyentes.
10. Early stopping necesita una validación fiable y guardar/restaurar el mejor estado.

## 9. Errores y riesgos

- No escalar features antes de GD, Ridge, Lasso, Elastic Net o polynomial models.
- Usar un `learning rate` alto y confundir divergencia con falta de capacidad del modelo; usar uno muy bajo y detenerse prematuramente.
- Aplicar la Normal Equation con features redundantes; preferir SVD/pseudoinversa o regularización.
- Expandir a grados altos sin controlar la explosión combinatoria, memoria y overfitting.
- Interpretar una gran brecha train/validation como falta de datos únicamente; también puede requerir regularización o menor complejidad.
- Esperar que más datos solucionen underfitting cuando ambas curvas son altas y cercanas.
- Regularizar el intercepto o evaluar usando la función de coste penalizada, distorsionando la métrica final.
- Ajustar `PolynomialFeatures`, `StandardScaler` o cualquier transformación usando validación/test (data leakage); usar pipeline.
- Confundir `C` con `alpha`: en `LogisticRegression`, mayor `C` significa menor regularización.
- Usar Softmax para multilabel/multioutput; sus probabilidades compiten y suman 1.
- Confiar en umbral 0.5 sin considerar la función de costes del problema.
- Parar SGD/Mini-batch en la primera subida de validación: la curva puede oscilar; guardar el mejor estado o aplicar paciencia.
- No barajar datos de SGD cuando están ordenados por clase.
- Ignorar que Lasso puede ser inestable con features muy correlacionadas; considerar Elastic Net.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo cuando el target sea numérico/categórico y haya que elegir un baseline lineal, tratar no linealidad, diagnosticar overfitting o justificar regularización.
- Baseline rápido: separar train/validación, pipeline con escalado cuando corresponda, `LinearRegression`/`Ridge` para regresión o `LogisticRegression` para clasificación, y una métrica de validación clara.
- Activar Polynomial Regression si los residuos o curvas sugieren curvatura; empezar con grado bajo y regularización, no con grados altos.
- Activar Ridge como regularizador por defecto de un modelo lineal; probar Lasso/Elastic Net si se necesita sparsity o selección de features.
- Usar GD/SGD cuando el número de features o el volumen haga inviable la solución cerrada; controlar escalado, schedule y convergencia.
- Señales de overfitting: training muy bueno y validación peor, brecha en learning curves. Señales de underfitting: ambos errores altos y cercanos.
- Logistic Regression es adecuada para un baseline binario probabilístico; Softmax para una única clase entre varias.
- Bajo presión, no implementar manualmente toda la optimización si una clase de Scikit-Learn satisface el objetivo. Tampoco buscar hiperparámetros contra el test.
- **Recomendación para competencias:** mantener el test de la competición intacto y elegir grado, `alpha`, `C`, solver y umbral solo con validación/cross-validation; esto es una adaptación del protocolo del capítulo.

## 11. Checklist operativo

- [ ] Separar training, validación y test antes de ajustar transformaciones.
- [ ] Identificar regresión, clasificación binaria o clasificación multiclase exclusiva.
- [ ] Construir un baseline lineal y registrar la métrica.
- [ ] Escalar features antes de GD o regularización.
- [ ] Elegir solución cerrada frente a GD según `m`, `n` y memoria.
- [ ] Si hay curvatura, usar `PolynomialFeatures` dentro de un pipeline.
- [ ] Inspeccionar learning curves y brecha train/validación.
- [ ] Ajustar complejidad, `alpha`, `l1_ratio` o `C` sin tocar el test.
- [ ] Para SGD, mezclar datos conjuntamente y revisar schedule/tolerancia.
- [ ] Aplicar early stopping guardando el mejor modelo cuando el error de validación oscile.
- [ ] Confirmar que Softmax no se usa para multilabel.
- [ ] Evaluar finalmente con la métrica no penalizada y documentar supuestos.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
|---|---|---:|
| Predicción lineal, MSE y Normal Equation | `knowledge/models/linear_models.md` | Alta |
| SVD, pseudoinversa y coste computacional | `knowledge/models/linear_models.md` | Media |
| Batch, Stochastic y Mini-batch GD | `knowledge/tuning/gradient_descent.md` | Alta |
| Escalado y learning schedules | `knowledge/preprocessing/feature_scaling.md` | Alta |
| PolynomialFeatures e interacciones | `knowledge/feature_engineering/polynomial_features.md` | Alta |
| Learning curves y bias/variance | `knowledge/validation/learning_curves.md` | Alta |
| Ridge, Lasso y Elastic Net | `knowledge/models/regularized_regression.md` | Alta |
| Early stopping | `knowledge/validation/early_stopping.md` | Media |
| Sigmoid, log loss y frontera binaria | `knowledge/models/logistic_softmax_regression.md` | Alta |
| Softmax y cross entropy multiclase | `knowledge/models/logistic_softmax_regression.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
|---|---|
| 1. Propósito | Apertura del capítulo, pp. 111–112 |
| 2. Modelos lineales y MSE | “Linear Regression”, pp. 112–117 |
| 2–3. Gradient Descent | “Gradient Descent”, pp. 118–127 |
| 2, 5. Polynomial Regression | “Polynomial Regression”, pp. 128–130 |
| 2, 5. Learning curves y bias/variance | “Learning Curves” y “The Bias/Variance Trade-off”, pp. 130–134 |
| 2, 5. Regularización | “Regularized Linear Models”, pp. 134–141 |
| 2, 5. Early stopping | “Early Stopping”, pp. 141–142 |
| 2, 5. Logistic Regression | “Logistic Regression”, pp. 142–147 |
| 2, 5. Softmax Regression | “Softmax Regression”, pp. 148–150 |
| 6. Fórmulas | Ecuaciones 4-1–4-23, pp. 112–150 |
| 7. Implementación | Ejemplos de código, pp. 114–117, 121–126, 128–129, 132–133, 137–142, 146–150 |
| 8–9. Supuestos y riesgos | Discusión de escalado, convergencia, bias/variance, regularización y clases, pp. 117–150 |
| 10–11. Síntesis práctica | Conclusiones operativas derivadas de pp. 111–150 |

### Cobertura y legibilidad

- Páginas revisadas: 1–42 del PDF, equivalentes a las páginas impresas 111–152.
- No se consultaron fuentes externas ni otros capítulos; el apéndice de soluciones no era necesario para comprender este capítulo.
- Se revisaron los ejercicios de las pp. 151–152 solo para delimitar el cierre; se omitieron como contenido principal porque el encargo excluye ejercicios sin solución.
- Secciones no procesables: ninguna. El texto, ecuaciones, tablas, ejemplos y figuras relevantes fueron legibles; las figuras se usaron para interpretar tendencias (convergencia y curvas), no para inventar valores.