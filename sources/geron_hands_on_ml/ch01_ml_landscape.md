---
title: "The Machine Learning Landscape"
topics:
  - "machine learning problem types"
  - "supervised and unsupervised learning"
  - "generalization"
  - "overfitting and underfitting"
  - "data quality"
target_knowledge_files:
  - "knowledge/problem_definition/problem_types.md"
  - "knowledge/problem_definition/ml_problem_assessment.md"
  - "knowledge/validation/generalization.md"
  - "knowledge/validation/overfitting_underfitting.md"
---

# 1. The Machine Learning Landscape

**Fuente:** Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, tercera edición, capítulo 1.

## 1. Propósito del capítulo

El capítulo define Machine Learning (ML), organiza sus principales familias y presenta el ciclo mínimo de un proyecto: estudiar los datos, elegir un modelo, entrenarlo, inferir sobre casos nuevos y estimar su capacidad de generalización. También identifica los fallos dominantes -datos insuficientes, no representativos o de mala calidad, atributos irrelevantes, overfitting y underfitting- y explica por qué la selección de modelos debe separarse de la evaluación final.

Su relevancia práctica es servir como mapa para formular un problema antes de elegir un algoritmo y para diseñar una evaluación que no produzca una estimación optimista del rendimiento.

## 2. Conceptos esenciales

Las marcas usadas en este documento distinguen el origen de cada afirmación:

- **Explicado por el autor:** contenido explícito del capítulo.
- **Conclusión práctica derivada:** consecuencia operativa directa, no formulada necesariamente como regla literal.
- **Recomendación para competencias:** adaptación del contenido a coding tests o datathons.

### Definición y unidad de aprendizaje

- **Explicado por el autor:** ML es programar sistemas capaces de aprender de los datos. En la formulación de Mitchell, un programa aprende de la experiencia `E` respecto de una tarea `T` y una medida `P` si su rendimiento en `T`, medido por `P`, mejora con `E`.
- El **training set** contiene los ejemplos usados para aprender; cada ejemplo es una **training instance** o **sample**.
- Una descarga de datos no constituye aprendizaje si el sistema no mejora en una tarea medida.
- ML resulta especialmente útil cuando las reglas manuales serían numerosas o cambiantes, el problema no tiene un algoritmo tradicional satisfactorio, el entorno fluctúa o se quieren descubrir patrones en grandes volúmenes de datos (**data mining**).

### Tipos según supervisión

- **Supervised learning:** usa ejemplos con la solución deseada (**labels**). Sus tareas típicas son clasificación y regresión. La regresión predice valores numéricos; un algoritmo llamado “regresión”, como Logistic Regression, también puede resolver clasificación.
- **Unsupervised learning:** trabaja con datos sin etiquetas. Incluye clustering, detección de anomalías o novedades, visualización, reducción de dimensionalidad y reglas de asociación.
- **Semisupervised learning:** combina pocos ejemplos etiquetados con muchos no etiquetados; suele ensamblar componentes supervisados y no supervisados.
- **Reinforcement Learning:** un agente observa un entorno, ejecuta acciones y recibe recompensas o penalizaciones. Aprende una **policy** que selecciona acciones para maximizar la recompensa acumulada.

### Tipos según actualización

- **Batch learning:** entrena con todos los datos disponibles y no aprende durante inferencia. Adaptarlo exige entrenar y desplegar una nueva versión; el proceso puede automatizarse.
- **Online learning:** actualiza incrementalmente con instancias o **mini-batches**. También permite **out-of-core learning**, procesando porciones de un dataset que no cabe en memoria.
- El **learning rate** controla la velocidad de adaptación: alto implica adaptación y olvido rápidos; bajo aporta inercia y menor sensibilidad al ruido reciente.

### Tipos según generalización

- **Instance-based learning:** memoriza ejemplos y predice comparando nuevos casos mediante una medida de similitud.
- **Model-based learning:** selecciona una familia de modelos, aprende parámetros que optimizan una función de rendimiento o coste y usa el modelo ajustado para inferir.
- Un **model parameter** se aprende durante el entrenamiento. Un **hyperparameter** gobierna el algoritmo de aprendizaje, se fija antes del entrenamiento y permanece constante durante este.

### Generalización y evaluación

- El objetivo no es ajustar perfectamente el entrenamiento, sino rendir en instancias nuevas.
- El **generalization error** u **out-of-sample error** se estima sobre un test set no usado para entrenar ni seleccionar.
- **Holdout validation:** separa una validation/dev set del entrenamiento para comparar modelos e hiperparámetros. Tras elegir, se reentrena el modelo seleccionado con todo el training set, incluida la validación, y se evalúa una sola vez en test.
- **Repeated cross-validation:** usa varias particiones pequeñas de validación y promedia resultados; mejora la precisión de la estimación a cambio de multiplicar el tiempo de entrenamiento.
- Ante **data mismatch**, validation y test deben representar los datos de producción. Un **train-dev set** extraído de la misma distribución que training ayuda a separar overfitting de desajuste entre distribuciones.
- El teorema **No Free Lunch** implica que, sin supuestos sobre los datos, ningún modelo es preferible a priori. En la práctica se prueban unas pocas familias razonables según supuestos explícitos.

## 3. Flujo o procedimiento presentado

1. Definir la tarea `T`, la experiencia o datos `E` y la medida de rendimiento `P`.
2. Estudiar los datos y comprobar cantidad, calidad, representatividad y relevancia de atributos.
3. Determinar los ejes del sistema: supervisión disponible, actualización batch/online y generalización por instancias/modelo.
4. Reservar test para estimar el error de generalización; no usarlo para decisiones iterativas.
5. Dentro del training set, crear validation/dev o aplicar validación cruzada para comparar candidatos e hiperparámetros.
6. Seleccionar el tipo y la arquitectura del modelo.
7. Entrenar: el algoritmo busca parámetros que mejoren una utilidad o minimicen un coste sobre training.
8. Diagnosticar con errores de training y validation: una brecha alta apunta a overfitting; error alto incluso en training apunta a underfitting.
9. Corregir datos, features, complejidad o regularización según el diagnóstico.
10. Reentrenar la opción elegida con todo el training set disponible y evaluarla en test.
11. Desplegar para inferencia y, si el aprendizaje es online, monitorizar rendimiento e inputs, con capacidad de detener o revertir actualizaciones.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
|---|---|---|---|
| Hay labels y se predice una clase | Clasificación supervisada | Aprende la relación entre atributos y clases conocidas | Elegir `P` de acuerdo con la tarea; accuracy es solo un ejemplo del capítulo |
| Hay labels y el objetivo es numérico | Regresión supervisada | Produce una estimación cuantitativa | Un nombre de algoritmo no determina por sí solo el tipo de tarea |
| No hay labels y se buscan grupos | Clustering | Descubre grupos por similitud sin asignaciones previas | Los grupos requieren interpretación posterior |
| Se buscan casos raros dentro de datos posiblemente contaminados | Anomaly detection | Aprende el patrón mayoritario normal y marca desviaciones | Un caso raro puede ser válido, no necesariamente un error |
| Se quieren detectar casos nuevos respecto de una referencia limpia | Novelty detection | Compara contra un conjunto conocido sin las novedades objetivo | El conjunto de entrenamiento debe estar limpio |
| Hay muchas dimensiones correlacionadas | Reducción de dimensionalidad / feature extraction | Simplifica, reduce almacenamiento y puede acelerar el aprendizaje | Puede perder información; no se garantiza mejorar el rendimiento |
| Hay pocas etiquetas y muchos ejemplos sin etiqueta | Semisupervised learning | Aprovecha la estructura no etiquetada y refina con labels | La calidad de clusters o componentes no supervisados condiciona el resultado |
| Las decisiones alteran un entorno y hay recompensa diferida | Reinforcement Learning | Aprende una policy por interacción y recompensa acumulada | No equivale a clasificación supervisada de acciones |
| Los datos cambian despacio y cabe reentrenar | Batch learning | Es simple y automatizable | Reentrenar desde cero puede ser caro y poco reactivo |
| Hay flujo continuo, deriva rápida o memoria limitada | Online/incremental learning | Actualiza barato con instancias o mini-batches | Datos malos degradan el sistema; exige monitorización y rollback |
| El dataset no cabe en memoria | Out-of-core learning | Procesa fragmentos secuencialmente | Es incremental, aunque normalmente se ejecute offline |
| Se dispone de una similitud útil y ejemplos representativos | Instance-based learning | Generaliza por cercanía a instancias conocidas | Predicción y almacenamiento pueden crecer con el número de ejemplos |
| Se busca una representación compacta y una regla de inferencia | Model-based learning | Resume patrones en parámetros aprendidos | La familia elegida incorpora supuestos que deben validarse |
| Validation/test difieren de training pero reflejan producción | Train-dev set adicional | Distingue overfitting de data mismatch | Evitar duplicados y casi duplicados entre particiones |
| Validation es demasiado pequeña o inestable | Validación cruzada repetida | Promedia varias evaluaciones | Multiplica el coste de entrenamiento |

## 5. Métodos y técnicas

### Clasificación supervisada

- **Objetivo:** asignar una clase a una instancia nueva.
- **Entradas:** atributos y labels de clase.
- **Salida:** clase; algunos modelos también entregan probabilidad.
- **Funcionamiento:** ajusta una relación predictiva usando ejemplos etiquetados.
- **Hiperparámetros:** dependen del algoritmo; el capítulo no los detalla.
- **Ventajas:** objetivo directo y evaluación definida cuando existen labels fiables.
- **Limitaciones:** depende de cobertura y calidad de las etiquetas.
- **Adecuado:** spam/ham, texto, imagen o comentario ofensivo.
- **No conviene:** cuando no existen labels y el objetivo real es explorar estructura.

### Regresión supervisada

- **Objetivo:** predecir un valor numérico.
- **Entradas:** predictores y target cuantitativo.
- **Salida:** valor estimado.
- **Funcionamiento:** aprende parámetros que minimizan una función de coste; el ejemplo usa un modelo lineal.
- **Hiperparámetros:** familia, complejidad y regularización; no confundirlos con los coeficientes aprendidos.
- **Ventajas:** permite inferencia cuantitativa y, en modelos simples, interpretación directa.
- **Limitaciones:** una forma funcional demasiado simple hace underfitting; una demasiado flexible puede sobreajustar.
- **Adecuado:** precios, ingresos o satisfacción estimada.
- **No conviene:** para targets categóricos sin una formulación apropiada.

### Clustering y reducción de dimensionalidad

- **Objetivo:** descubrir grupos o construir una representación de menor dimensión.
- **Entradas:** datos sin etiquetas.
- **Salida:** asignaciones/grupos o nuevas features compactas.
- **Funcionamiento:** preserva semejanzas o estructura relevante; feature extraction puede combinar atributos correlacionados.
- **Hiperparámetros:** el capítulo enumera K-Means, DBSCAN, HCA, PCA, kernel PCA, LLE y t-SNE, pero no desarrolla sus parámetros.
- **Ventajas:** exploración, visualización, segmentación, menor memoria y entrenamiento potencialmente más rápido.
- **Limitaciones:** puede perder información; una visualización no constituye evidencia supervisada.
- **Adecuado:** segmentación de clientes, exploración de alta dimensión y preprocesamiento.
- **No conviene:** como sustituto automático de labels requeridos por una decisión supervisada.

### Anomaly detection y novelty detection

- **Objetivo:** identificar observaciones diferentes del patrón normal.
- **Entradas:** principalmente ejemplos normales; novelty detection requiere una referencia limpia.
- **Salida:** normal/anómalo o nuevo.
- **Funcionamiento:** aprende la estructura predominante y mide desviación.
- **Hiperparámetros:** no desarrollados en el capítulo.
- **Ventajas:** fraude, defectos, vigilancia de inputs y posible limpieza de outliers.
- **Limitaciones:** rareza no implica error; la contaminación de referencia afecta la detección.
- **Adecuado:** transacciones, sensores y control de datos online.
- **No conviene:** si las clases de interés están bien etiquetadas y se necesita discriminación supervisada ordinaria.

### Aprendizaje online / out-of-core

- **Objetivo:** actualizar el modelo incrementalmente o entrenar con datos que no caben en memoria.
- **Entradas:** instancias o mini-batches secuenciales.
- **Salida:** modelo actualizado tras cada paso.
- **Funcionamiento:** realiza pasos de aprendizaje baratos y descarta datos ya procesados si no se requiere replay.
- **Hiperparámetro central:** learning rate.
- **Ventajas:** adaptación, autonomía, bajo uso de memoria.
- **Limitaciones:** sensibilidad a entradas corruptas; requiere observabilidad y reversión.
- **Adecuado:** streams, deriva rápida y datasets enormes.
- **No conviene:** si no existe monitorización o si el reentrenamiento batch satisface coste y latencia.

### Instance-based frente a model-based

- **Objetivo:** generalizar por similitud o por un modelo aprendido.
- **Entradas:** instancias de entrenamiento; instance-based requiere una medida de similitud, model-based una familia y una función objetivo.
- **Salida:** predicción para casos nuevos.
- **Funcionamiento:** el primero consulta vecinos/ejemplos; el segundo estima parámetros y aplica la función resultante.
- **Hiperparámetros relevantes:** número de vecinos `k` en el ejemplo de k-NN; arquitectura y regularización en modelos.
- **Ventajas:** instance-based es directo cuando la cercanía es informativa; model-based comprime el patrón en parámetros.
- **Limitaciones:** instance-based depende críticamente de la similitud; model-based depende de sus supuestos.
- **Adecuado:** k-NN para vecindarios comparables; modelo lineal cuando la relación lineal es razonable.
- **No conviene:** ninguna familia debe elegirse como universalmente superior; aplica No Free Lunch.

## 6. Fórmulas esenciales

### Criterio de aprendizaje

Un sistema aprende si su rendimiento medido por `P` en la tarea `T` mejora con la experiencia `E`.

- `T`: tarea que debe ejecutar el sistema.
- `E`: datos o interacción de la que aprende.
- `P`: medida de rendimiento.
- **Qué mide:** no es una ecuación numérica, sino una condición operacional para afirmar que hubo aprendizaje.
- **Interpretación:** más datos por sí solos no bastan; debe observarse mejora medible en la tarea.
- **Supuesto:** `P` representa adecuadamente el objetivo real.

### Modelo lineal simple

\[
\text{life\_satisfaction} = \theta_0 + \theta_1 \times \text{GDP\_per\_capita}
\]

- `life_satisfaction`: predicción del target.
- `GDP_per_capita`: predictor de entrada.
- `θ₀`: intercepto aprendido.
- `θ₁`: pendiente aprendida, cambio previsto en el target por unidad del predictor.
- **Qué mide:** una relación lineal entre un predictor y el objetivo.
- **Interpretación:** `θ₁` positivo produce predicciones mayores al crecer el predictor; entrenamiento elige `θ₀` y `θ₁` para ajustar los ejemplos.
- **Supuesto:** la estructura relevante puede aproximarse linealmente y las desviaciones son ruido descartable. El propio capítulo muestra que ampliar la muestra puede invalidar este supuesto.

### Accuracy

\[
\text{accuracy} = \frac{\text{predicciones correctas}}{\text{predicciones totales}}
\]

- El numerador cuenta clasificaciones correctas y el denominador todos los casos evaluados.
- **Qué mide:** proporción de aciertos.
- **Interpretación:** valores mayores indican más aciertos en la muestra evaluada.
- **Supuesto importante:** el capítulo la presenta como ejemplo de `P` para clasificación, no como métrica universal.

## 7. Implementación práctica

El único código completo del capítulo compara Linear Regression con k-Nearest Neighbors regression. El siguiente fragmento conserva esa idea y añade la separación de evaluación descrita más adelante en el capítulo.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
predictions = model.predict(new_cases)
```

**Nota práctica de API actual:** `train_test_split` y `random_state` hacen explícita una partición reproducible. Esta forma compacta no aparece en el ejemplo original del capítulo; materializa la separación train/test que el autor explica conceptualmente. Si se comparan modelos o hiperparámetros, `X_test` debe permanecer intacto y la selección debe hacerse con validation o cross-validation dentro de `X_train`.

Sustitución mínima por aprendizaje basado en instancias:

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(new_cases)
```

`n_neighbors=3` corresponde al ejemplo del capítulo: se promedian tres países cercanos en GDP per capita. No constituye un valor recomendado universal.

## 8. Supuestos y condiciones de aplicación

1. Training, validation y test deben representar los casos sobre los que se generalizará; validation y test deben aproximar especialmente la distribución de producción.
2. Las particiones deben excluir duplicados o casi duplicados entre ellas cuando estos inflen artificialmente el rendimiento.
3. La muestra debe ser suficientemente grande para limitar **sampling noise** y obtenida mediante un método que no introduzca **sampling bias** o **nonresponse bias**.
4. Los datos deben contener suficientes features relevantes y no demasiadas irrelevantes.
5. Los errores de medición, missing values, ruido y outliers requieren una decisión explícita: corregir, eliminar, imputar, descartar atributos o comparar variantes.
6. Instance-based learning presupone que la medida de similitud alinea cercanía con comportamiento predictivo.
7. Model-based learning presupone que la familia elegida descarta detalles que no generalizan y conserva la estructura útil.
8. Novelty detection requiere un conjunto de referencia sin las novedades que se desean detectar.
9. Online learning requiere entradas suficientemente controladas, monitorización y capacidad de detener o revertir el aprendizaje.
10. La complejidad apropiada depende del tamaño y ruido del training set; no existe un modelo superior para todos los datasets.

## 9. Errores y riesgos

- **Cantidad insuficiente:** muchos algoritmos necesitan miles de ejemplos y tareas complejas pueden exigir millones, salvo reutilización de partes de un modelo existente.
- **Muestra no representativa:** una muestra pequeña sufre sampling noise; incluso una enorme puede sufrir sampling bias por un procedimiento de recolección defectuoso.
- **Datos pobres:** errores, ruido, missing values y outliers ocultan patrones. Eliminar outliers sin investigar también puede borrar casos válidos.
- **Features irrelevantes:** degradan el aprendizaje; feature engineering incluye selección, extracción y creación mediante nueva recolección.
- **Overfitting:** error bajo en training y alto fuera de muestra. Ocurre cuando la complejidad es excesiva respecto de cantidad y ruido. Respuestas: simplificar o restringir el modelo, reducir atributos, conseguir más datos o disminuir ruido.
- **Regularización excesiva:** evita sobreajuste, pero puede producir un modelo casi plano e incapaz de capturar la señal.
- **Underfitting:** el modelo es demasiado simple incluso para training. Respuestas: aumentar potencia/parámetros, mejorar features o reducir restricciones.
- **Leakage de evaluación:** ajustar modelos o hiperparámetros mirando repetidamente el test adapta el sistema a ese test y sesga la estimación de generalización.
- **Validation mal dimensionada:** demasiado pequeña produce comparaciones imprecisas; demasiado grande deja candidatos entrenados con una muestra poco comparable con el entrenamiento final.
- **Data mismatch:** buen resultado en training y malo en validation puede ser overfitting o cambio de distribución. El train-dev set permite separar ambos diagnósticos.
- **Online poisoning o fallo de sensor:** entradas malas degradan gradualmente un modelo vivo. Se requieren monitorización de rendimiento, vigilancia de inputs y rollback.
- **Coste computacional:** batch puede exigir CPU, memoria, disco, I/O y reentrenamientos costosos; cross-validation multiplica el tiempo por el número de particiones.
- **Interpretación incorrecta:** un buen ajuste del training set no prueba generalización; correlaciones o patrones espurios pueden surgir por azar.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- **1. ¿El problema requiere ML?** Usar ML cuando las reglas manuales serían numerosas, frágiles o cambiantes; cuando no existe un algoritmo tradicional satisfactorio; cuando el sistema debe adaptarse a datos nuevos; o cuando el objetivo es descubrir patrones. Si una regla determinista corta resuelve la tarea de forma estable, este capítulo no aporta una razón para reemplazarla por ML.
- **2. ¿Qué tipo de tarea es?** Target categórico con labels -> clasificación. Target numérico con labels -> regresión. Sin labels y con interés en grupos -> clustering. Interés en casos raros o desviaciones del patrón normal -> anomaly detection; usar novelty detection cuando la referencia de entrenamiento está limpia y se buscan casos nuevos.
- **3. ¿Batch u online?** Preferir batch cuando el dataset cabe en el proceso de entrenamiento y la actualización periódica basta. Considerar online cuando llegan streams, hay cambios rápidos, se necesita autonomía o el volumen exige aprendizaje incremental/out-of-core. Online solo es defendible si se pueden vigilar inputs y rendimiento y revertir actualizaciones.
- **4. ¿Qué puede impedir la generalización?** Pocos ejemplos, muestra no representativa, sampling bias, errores de medición, ruido, missing values, outliers, features irrelevantes, duplicados entre particiones o mismatch entre training y producción.
- **5. ¿Overfitting o underfitting?** Error bajo en training y claramente peor en validation -> overfitting. Error alto incluso en training -> underfitting o señal insuficiente. La comparación debe hacerse con una partición no usada para ajustar parámetros.
- **6. ¿Cuál es el baseline conceptual?** Definir `T`, `E` y `P`; reservar test; crear validation dentro de training; comprobar calidad y representatividad; y entrenar una familia simple coherente con el tipo de target. Este baseline establece una referencia antes de añadir complejidad, regularización o nuevas features.
- **Decisiones que permite tomar:** clasificar la tarea por supervisión, tipo de target, régimen batch/online y forma de generalización; decidir particiones y diagnóstico de errores.
- **Mejoras:** más/mejores datos, features relevantes, regularización, modelo más o menos potente según el patrón de errores, y cross-validation cuando una única validation es inestable.
- **Qué evitar bajo presión:** RL si la tarea no implica interacción/recompensas; aprendizaje online sin necesidad de actualización; búsquedas masivas de hiperparámetros contra test; modelos complejos antes de verificar representatividad y calidad.
- **Señales que activan técnicas:** target numérico -> regresión; target categórico -> clasificación; ausencia de labels -> no supervisado; flujo/deriva -> online; dataset mayor que memoria -> out-of-core; pocos labels y muchos datos -> semisupervisado; casos raros -> anomalías/novedades; muchas features correlacionadas -> reducción de dimensionalidad; gran brecha train-validation -> overfitting; errores altos en ambos -> underfitting o señal insuficiente.
- **Recomendación para competencias:** tratar el test de la competición como evaluación final inaccesible; toda decisión reproducible debe proceder de validation/cross-validation. Esta es una traducción competitiva del principio del capítulo, no una instrucción literal del autor sobre plataformas de competición.

## 11. Checklist operativo

- [ ] Definir `T`, `E` y `P`.
- [ ] Identificar clasificación, regresión, clustering, anomalías u otro objetivo.
- [ ] Comprobar labels, tamaño, representatividad, ruido, missing values y features irrelevantes.
- [ ] Separar test antes de modelar y no usarlo para tuning.
- [ ] Diseñar validation o cross-validation dentro de training.
- [ ] Elegir un baseline compatible con los supuestos de los datos.
- [ ] Comparar errores de training y validation para diagnosticar overfitting/underfitting.
- [ ] Ajustar complejidad, regularización, features o datos según el diagnóstico.
- [ ] Si hay data mismatch, crear train-dev desde la distribución de training.
- [ ] Reentrenar el candidato elegido y evaluar una vez en test.
- [ ] Si el aprendizaje es online, monitorizar inputs/rendimiento y preparar rollback.

## 12. Mapeo hacia knowledge/

| Contenido extraído | Archivo de destino propuesto | Prioridad |
|---|---|---|
| Formulación `T`, `E`, `P` y taxonomía de tareas | `knowledge/problem_definition/problem_types.md` | Alta |
| Evaluación inicial de si el problema requiere ML | `knowledge/problem_definition/ml_problem_assessment.md` | Alta |
| Separación train/validation/test y error de generalización | `knowledge/validation/generalization.md` | Alta |
| Holdout, cross-validation y riesgo de ajustar contra test | `knowledge/validation/model_selection_protocol.md` | Alta |
| Train-dev y diagnóstico de data mismatch | `knowledge/validation/data_mismatch.md` | Alta |
| Cantidad, calidad, representatividad y sampling bias | `knowledge/preprocessing/data_quality_and_representativeness.md` | Alta |
| Feature selection, extraction y creación | `knowledge/feature_engineering/feature_engineering_overview.md` | Media |
| Overfitting, underfitting y regularización | `knowledge/validation/overfitting_underfitting.md` | Alta |
| Instance-based frente a model-based | `knowledge/models/instance_vs_model_based.md` | Media |
| Batch, online y out-of-core learning | `knowledge/models/incremental_learning.md` | Media |
| Accuracy como ejemplo de medida `P` | `knowledge/metrics/classification_metrics.md` | Baja |
| Clustering, anomalías y reducción dimensional | `knowledge/models/unsupervised_learning_overview.md` | Media |
| Learning rate y control de adaptación online | `knowledge/tuning/learning_rate.md` | Baja |

## 13. Trazabilidad

La columna de páginas usa la numeración impresa del capítulo, que coincide con las 34 páginas del PDF suministrado.

| Sección del resumen | Sección o páginas del capítulo |
|---|---|
| 1. Propósito del capítulo | Introducción, pp. 1-2; “Stepping Back”, p. 30 |
| 2. Definición y utilidad de ML | “What Is Machine Learning?”, p. 2; “Why Use Machine Learning?”, pp. 2-5 |
| 2. Ejemplos y formulación de tareas | “Examples of Applications”, pp. 5-7 |
| 2. Tipos según supervisión | “Supervised/Unsupervised Learning”, pp. 7-14 |
| 2. Batch, online y out-of-core | “Batch and Online Learning”, pp. 14-17 |
| 2. Instance-based y model-based | “Instance-Based Versus Model-Based Learning”, pp. 17-23 |
| 3. Flujo presentado | Resumen del flujo, p. 23; “Testing and Validating”, pp. 30-32 |
| 4. Reglas de decisión | Tipos de sistemas, pp. 7-18; desafíos y validación, pp. 23-33 |
| 5. Métodos y técnicas | Supervisión, pp. 8-14; actualización, pp. 15-17; generalización, pp. 17-23 |
| 6. Criterio `T`, `E`, `P` y accuracy | p. 2 |
| 6. Modelo lineal simple | Ecuación 1-1 y explicación, pp. 19-22 |
| 7. Implementación práctica | Ejemplo 1-1 y alternativa k-NN, pp. 21-22; separación train/test, pp. 30-31 |
| 8. Supuestos y condiciones | “Main Challenges of Machine Learning”, pp. 23-29; “Data Mismatch”, p. 32; “No Free Lunch Theorem”, p. 33 |
| 9. Datos insuficientes/no representativos | pp. 23-26 |
| 9. Calidad y features | pp. 26-27 |
| 9. Overfitting, regularización y underfitting | pp. 27-29 |
| 9. Riesgos de validación y data mismatch | pp. 30-32 |
| 10. Aplicación en coding test | Síntesis derivada de pp. 7-33 |
| 11. Checklist operativo | Síntesis derivada del flujo y desafíos, pp. 23-33 |
| 12. Mapeo hacia knowledge/ | Organización documental derivada del capítulo completo, pp. 1-33 |

### Cobertura y legibilidad

- Páginas revisadas: 1-34 del capítulo 1.
- Secciones no procesables: ninguna; el texto, figuras, tablas, ecuación y notas relevantes fueron legibles.
- Ejercicios: se revisaron las preguntas de las pp. 33-34 para comprobar cobertura, pero se omitieron como contenido principal porque el encargo excluye ejercicios sin solución. El apéndice de respuestas no fue consultado por no ser necesario para comprender el capítulo.
