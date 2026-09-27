---
title: "Model Training Patterns"
book: "Machine Learning Design Patterns"
chapter: "4"
topics:
  - "Training loops"
  - "Overfitting"
  - "Checkpointing"
  - "Transfer learning"
  - "Distributed training"
  - "Hyperparameter optimization"
patterns:
  - "Useful Overfitting"
  - "Checkpoints"
  - "Transfer Learning"
  - "Distribution Strategy"
  - "Hyperparameter Tuning"
target_knowledge_files:
  - "knowledge/models/useful_overfitting.md"
  - "knowledge/mlops/checkpoints.md"
  - "knowledge/models/transfer_learning.md"
  - "knowledge/mlops/distributed_training.md"
  - "knowledge/tuning/hyperparameter_tuning.md"
---

# 4. Model Training Patterns

## 1. Propósito del capítulo

El capítulo estudia modificaciones deliberadas del bucle de entrenamiento habitual basado en mini-batches y stochastic gradient descent (SGD). Los problemas aparecen durante el desarrollo, entrenamiento, recuperación, escalado y optimización de modelos: a veces conviene ajustar perfectamente datos deterministas; los entrenamientos largos necesitan estado recuperable; los datasets pequeños pueden aprovechar representaciones ya aprendidas; los modelos grandes exigen distribuir cómputo; y los hiperparámetros requieren una optimización exterior al entrenamiento.

Los patrones se justifican porque el ciclo convencional `train → validate → test → export` no cubre por sí solo estos casos. Cada patrón cambia una parte concreta del bucle sin eliminar la necesidad de medir correctamente el resultado.

# 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Useful Overfitting | El objetivo no es generalizar a datos desconocidos, sino reproducir exactamente una función determinista o verificar que el sistema puede ajustar un lote pequeño. | Training | Media |
| Checkpoints | Un entrenamiento largo puede fallar, sobreajustar o necesitar reanudación y selección de estados intermedios. | Training / Reproducibility / Pipeline | Alta |
| Transfer Learning | No hay suficientes datos propios para aprender desde cero una representación de alta calidad. | Training / Feature engineering | Alta |
| Distribution Strategy | El volumen de datos o el tamaño del modelo vuelve demasiado lento o imposible el entrenamiento en un solo dispositivo. | Training / Scalability | Media |
| Hyperparameter Tuning | La selección manual o exhaustiva de hiperparámetros es costosa y no aprende de ensayos anteriores. | Hyperparameter tuning / Validation | Alta |

# 3. Patrones

## Pattern 11: Useful Overfitting

### 3.1 Problema

La regla habitual de evitar el overfitting deja de ser apropiada cuando el conjunto de entradas posibles puede enumerarse y cada entrada tiene una salida exacta, determinista y sin ruido. Separar parte de esa tabla como validación obligaría al modelo a aprender regiones que se le han ocultado, aunque no exista un verdadero dominio futuro distinto.

El patrón también responde a un problema de diagnóstico: que un modelo compile y ejecute no demuestra que la arquitectura, la función de pérdida o el pipeline de entrada estén bien configurados. Un modelo con capacidad suficiente debería poder llevar a pérdida cercana a cero un lote pequeño y cuidadosamente verificado.

### 3.2 Contexto de aplicación

- Aproximación rápida de sistemas físicos o dinámicos gobernados por ecuaciones, cuando el cálculo numérico exacto es demasiado lento para serving.
- Espacios de entrada finitos o discretizados con etiquetas calculables sin ruido ni variables ocultas.
- Knowledge distillation: un modelo pequeño aprende las salidas suaves generadas por un modelo grande.
- Desarrollo y troubleshooting de redes neuronales sobre un lote pequeño con labels comprobados.
- Si el dominio solo se muestrea por Monte Carlo, el caso deja de ser exhaustivo y requiere limitar complejidad o aplicar regularización leve.

### 3.3 Señales que indican que el patrón puede ser necesario

- Todas las combinaciones de entrada relevantes pueden tabularse o generarse a una resolución conocida.
- Para una misma entrada existe una única salida calculable, sin ruido estadístico.
- La tabla completa sería demasiado grande para servirla como lookup table, pero puede usarse para entrenar.
- Un modelo o pipeline nuevo no logra memorizar un lote pequeño verificado; esto señala un posible bug de datos, loss o arquitectura.
- **Inferencia práctica:** el coste de inferencia del simulador exacto domina al de una aproximación ML y la tolerancia de error está acotada.

### 3.4 Solución propuesta

1. Verificar que no habrá entradas genuinamente fuera del dominio cubierto y que los labels son exactos.
2. Construir el dataset sobre todo el espacio de entrada o sobre una discretización suficientemente fina.
3. Entrenar sin mecanismos cuyo propósito sea generalizar a datos no vistos: no reservar validación para early stopping y, en el caso exhaustivo, prescindir de regularización y dropout.
4. Ajustar el modelo tan precisamente como sea posible a la tabla entrada-salida.
5. En inferencia, redondear las entradas a la resolución con la que se discretizó el espacio.
6. Para diagnóstico, repetir un lote pequeño verificado hasta que un modelo con capacidad suficiente alcance pérdida cercana a cero; después reutilizar la arquitectura con el dataset completo.

### 3.5 Cómo funciona

El modelo sustituye una tabla de consulta o un cálculo numérico caro por una función paramétrica compacta. Si todos los puntos posibles forman parte del entrenamiento, una predicción no requiere extrapolar: reproduce puntos conocidos. Si hay puntos intermedios, el modelo interpola según su cercanía a ejemplos entrenados; esto exige que la discretización respete la resolución del fenómeno.

En distillation, el modelo pequeño aprende la función suave del teacher sobre muchos ejemplos generados. En el sanity check de un lote, la repetición elimina la escasez de pasos como explicación del error y prueba conjuntamente modelo, loss y entrada.

### 3.6 Por qué funciona

Con etiquetas deterministas y cobertura completa, la varianza debida a ruido y el error de generalización dejan de ser el objetivo central: ajustar la tabla equivale a aproximar la función que la generó. El capítulo apoya esta intuición con la capacidad de una red neuronal de aproximar funciones bien comportadas. En datasets muestreados, el riesgo procede de la complejidad del modelo entre puntos, no de ajustar ruido en los labels.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Combinaciones exhaustivas o finamente muestreadas del dominio; para diagnóstico, un lote pequeño con labels verificados. |
| Transformación | Entrenamiento intencional hasta ajustar con gran precisión las correspondencias entrada-salida. |
| Salida | Aproximador rápido de la función determinista o evidencia de que el pipeline puede aprender correctamente. |

### 3.8 Procedimiento operativo

1. Confirmar ausencia de ruido y unicidad de la salida por entrada.
2. Delimitar el dominio y la resolución de cada variable.
3. Comprobar si una lookup table directa sería suficientemente pequeña; si lo es, preferirla.
4. Generar o verificar los labels con el modelo físico, teacher o fuente determinista.
5. Elegir una arquitectura con capacidad para representar la función.
6. Entrenar buscando error de entrenamiento mínimo.
7. Si el espacio fue muestreado, controlar complejidad y usar regularización leve.
8. Validar explícitamente las zonas interpoladas y los límites de resolución.
9. Para depuración, repetir un batch verificado y revisar código, pipeline y loss si no se puede ajustar.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Resolución de discretización | Define qué entradas se consideran representadas. | Una malla gruesa puede fallar entre puntos o en sistemas caóticos. |
| Complejidad del modelo | Aumenta la capacidad para reproducir la función. | Con muestreo incompleto puede generar interpolaciones erróneas. |
| Regularización / dropout | Se eliminan en cobertura exhaustiva; pueden ser leves con Monte Carlo. | Aplicarlos sin necesidad impide el ajuste exacto; omitirlos con cobertura parcial puede sobreajustar. |
| Tamaño del lote de diagnóstico | Debe ser pequeño y verificable. | Un lote arbitrario con labels erróneos invalida la prueba. |

### 3.10 Ventajas

- Convierte simulaciones o cálculos deterministas caros en inferencia rápida.
- Comprime una tabla demasiado grande para producción en parámetros de modelo.
- Permite knowledge distillation hacia un modelo más pequeño.
- Ofrece un sanity check directo para arquitectura, loss e input pipeline.

### 3.11 Limitaciones

- Requiere cobertura completa o una discretización suficientemente fina del dominio.
- La interpolación puede fallar en sistemas caóticos o dominios no acotados.
- La existencia de una red capaz de aproximar la función no garantiza que SGD encuentre sus pesos.
- No sustituye el control de generalización en datos observacionales ruidosos.

### 3.12 Riesgos y trade-offs

- Tratar como exhaustivo un espacio solo muestreado introduce error en puntos ausentes.
- Un modelo de alta complejidad sobre muestras Monte Carlo puede producir estimaciones incorrectas entre puntos.
- Redondear de forma distinta en serving rompe la correspondencia con la grilla de entrenamiento.
- Si la tabla es pequeña, ML añade complejidad sin beneficio frente a lookup directo.
- **Inferencia práctica:** usar el patrón sobre datos reales con ruido elimina protecciones necesarias y puede destruir generalización.

### 3.13 Cuándo utilizarlo

- Cuando las entradas posibles están enumeradas y los labels son deterministas y exactos.
- Cuando una tabla exhaustiva es demasiado grande para serving, pero entrenable.
- Cuando se destila el comportamiento de un teacher.
- Como prueba controlada de que un modelo puede ajustar un pequeño conjunto correcto.

### 3.14 Cuándo no utilizarlo

- Cuando existirán entradas fuera del dominio entrenado.
- Cuando hay ruido, variables no observadas o diferentes labels plausibles para una misma entrada.
- Cuando el sistema es caótico y la resolución disponible no acota la interpolación.
- Cuando una lookup table simple satisface tamaño y latencia.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Lookup table | Reproduce exactamente valores conocidos. | Puede ser demasiado grande o difícil de servir. | Cuando la tabla es pequeña. |
| Método numérico clásico | Conserva la formulación física. | Puede ser demasiado lento o escalar mal con dimensionalidad. | Cuando su coste de inferencia es aceptable. |
| Monte Carlo + modelo de baja complejidad | Evita enumerar combinaciones imposibles. | Pierde cobertura exhaustiva. | Cuando no puede tabularse todo el dominio. |
| Regularización leve | Controla errores entre muestras. | Reduce capacidad de ajuste exacto. | Con cobertura muestreada, no exhaustiva. |

### 3.16 Implementación práctica

Idea mínima del sanity check descrito en el capítulo:

```python
batch = train_dataset.batch(batch_size).take(1)
model.fit(batch.repeat(), steps_per_epoch=steps, epochs=epochs)
```

El batch debe contener ejemplos cuyas etiquetas hayan sido revisadas. La condición útil no es solo que el código ejecute, sino que una arquitectura suficientemente grande pueda llevar la pérdida del lote cerca de cero.

### 3.17 Ejemplo mínimo conceptual

Un simulador físico asigna una salida exacta a cada combinación de entradas discretizadas, pero tarda demasiado en producción. Se generan todas las parejas entrada-salida, se entrena una red hasta reproducirlas y en serving se redondean entradas a la misma resolución. El resultado esperado es una aproximación compacta y rápida dentro del dominio cubierto.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En un coding test o datathon, el uso principal es diagnóstico: tomar un batch pequeño con labels inspeccionados y comprobar que el pipeline puede memorizarlo. Si no lo hace, conviene corregir primero loss, labels, preprocessing o capacidad antes de buscar mejores hiperparámetros. El uso físico exhaustivo será poco frecuente. La prueba es rápida y puede evitar entrenamientos completos sobre código defectuoso, pero memorizar el train completo no demuestra capacidad competitiva ni autoriza evaluar sobre train.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar que la función objetivo es determinista y sin ruido.
- [ ] Confirmar cobertura del dominio y resolución.
- [ ] Comparar contra una lookup table directa.
- [ ] Mantener igual redondeo en entrenamiento e inferencia.
- [ ] Si es diagnóstico, usar labels verificados.
- [ ] Si el muestreo es parcial, limitar complejidad y validar interpolación.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Checkpoints | Complementario | Un modelo grande puede entrenarse hasta sobreajustar y conservar estados intermedios antes de regularizar o seleccionar. |
| Transfer Learning | Puede combinarse | La distillation y el fine-tuning reutilizan conocimiento, aunque con mecanismos distintos. |
| Cascade | Dependencia conceptual | El capítulo remite a Cascade para encadenar optimizaciones al aproximar soluciones de PDE. |

## Pattern 12: Checkpoints

### 3.1 Problema

Los modelos y datasets grandes alargan el entrenamiento y elevan la probabilidad de fallo. Exportar solo el predictor final no conserva epoch, batch, estado del optimizer, learning-rate schedule, aleatoriedad o historia necesaria para continuar. Además, el modelo final no siempre es el que mejor generaliza.

### 3.2 Contexto de aplicación

- Entrenamientos de horas o días con riesgo de interrupción.
- Redes profundas con estado de optimizer y schedules.
- Early stopping o selección posterior del mejor estado según validación.
- Retraining y fine-tuning con datos nuevos sin cambiar arquitectura.
- Datasets grandes, streaming o distribuidos donde un epoch físico es poco útil.

### 3.3 Señales que indican que el patrón puede ser necesario

- Reiniciar desde cero implicaría perder mucho tiempo o dinero.
- La validación mejora y luego empeora durante el entrenamiento.
- Se necesita reentrenar periódicamente con datos recientes.
- Un epoch contiene tantos ejemplos que el intervalo entre guardados es demasiado largo.
- El tamaño del dataset o batch cambia entre ejecuciones y hace incomparables los epochs.

### 3.4 Solución propuesta

1. Guardar periódicamente el estado completo del entrenamiento, no solo el modelo exportado.
2. Incluir pesos, optimizer, epoch/paso y cualquier estado requerido para reanudar.
3. Elegir una frecuencia que equilibre trabajo potencialmente perdido e I/O.
4. Reanudar desde el último checkpoint válido tras un fallo.
5. Conservar métricas de validación para seleccionar posteriormente el checkpoint con mejor generalización.
6. En datasets grandes, definir epochs virtuales mediante número de ejemplos procesados y usar esas fronteras para checkpointing.

### 3.5 Cómo funciona

Cada checkpoint serializa el estado mutable del bucle. Al restaurarlo, el optimizer y el modelo continúan desde el mismo punto lógico. Los estados sucesivos capturan fases distintas: aprendizaje de estructura general, ajuste de detalles y posible overfitting. Esto permite recuperación, early stopping, checkpoint selection y fine-tuning desde un estado anterior más general.

### 3.6 Por qué funciona

El entrenamiento iterativo produce modelos parcialmente entrenados válidos. Los primeros capturan patrones amplios; después se ajustan casos menos frecuentes; finalmente pueden empezar a memorizar perturbaciones. Guardar la secuencia desacopla la duración del entrenamiento de la elección del modelo y hace recuperable un proceso costoso.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Modelo, optimizer, contador de pasos/epoch, métricas y demás estado del entrenamiento. |
| Transformación | Serialización periódica en fronteras de epoch real o virtual. |
| Salida | Estados restaurables para reanudación, selección o fine-tuning. |

### 3.8 Procedimiento operativo

1. Enumerar todo el estado necesario para reiniciar de forma equivalente.
2. Estimar duración tolerable de trabajo perdido.
3. Definir número total de ejemplos a mostrar al modelo y número de checkpoints.
4. Calcular `steps_per_epoch = total_examples / (batch_size * checkpoints)`.
5. Repetir el dataset para que el control dependa de pasos, no de agotarlo una vez.
6. Guardar estado completo en cada frontera virtual.
7. Registrar la métrica de evaluación asociada.
8. Tras un fallo, restaurar el último estado consistente.
9. Para producción, elegir early stopping si prima coste o checkpoint selection si prima precisión.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Frecuencia de checkpoint | Reduce trabajo perdido y da selección más granular. | Más I/O, almacenamiento y overhead. |
| Patience de early stopping | Tolera oscilaciones de validación. | Muy baja detiene prematuramente; muy alta gasta cómputo. |
| Número total de ejemplos | Hace comparables ejecuciones con distinto batch size. | Un presupuesto insuficiente impide convergencia. |
| Checkpoint para fine-tuning | Un estado anterior conserva mayor generalidad. | Uno demasiado temprano no contiene suficiente aprendizaje. |
| Early stopping vs selection | Control de coste frente a búsqueda del mejor estado. | Selection requiere entrenar más tiempo. |

### 3.10 Ventajas

- Recuperación ante fallos sin recomenzar.
- Selección del estado con mejor validación después de entrenar.
- Soporte para early stopping y fine-tuning.
- Comparabilidad al usar pasos o ejemplos constantes en lugar de epochs dependientes del tamaño del dataset.
- Mayor resiliencia en entrenamiento distribuido.

### 3.11 Limitaciones

- Los checkpoints son mucho mayores que un modelo exportado.
- Guardar con demasiada frecuencia introduce I/O significativo.
- Fine-tuning desde checkpoint exige mantener la arquitectura.
- La selección repetida sobre evaluación puede convertirla de hecho en señal de entrenamiento.

### 3.12 Riesgos y trade-offs

- Un output directory con checkpoints antiguos puede provocar una reanudación no deseada.
- Detenerse ante la primera subida de error puede perder mejoras posteriores o deep double descent.
- Early stopping reserva datos para decidir cuándo detenerse; regularización permite usar más ejemplos para actualizar pesos.
- Un checkpoint por epoch puede ser demasiado infrecuente en datasets enormes.
- Cambiar batch size manteniendo pasos fijos cambia el número de ejemplos vistos; por eso el patrón propone epochs virtuales basados en ejemplos.

### 3.13 Cuándo utilizarlo

- Siempre que perder una parte del entrenamiento sea costoso.
- Cuando se quiera seleccionar el mejor estado según validación.
- En retraining periódico con datos frescos.
- En entrenamientos distribuidos o datasets cuyo tamaño cambia.

### 3.14 Cuándo no utilizarlo

- En entrenamientos triviales que pueden repetirse a bajo coste.
- Para fine-tuning si cambió la arquitectura.
- Cuando solo se necesita serving: en ese caso basta el artefacto exportado, aunque no permita reanudar.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Early stopping | Reduce coste deteniendo entrenamiento. | Puede detenerse por una oscilación temporal. | Cuando el presupuesto domina. |
| Checkpoint selection | Permite entrenar más y elegir después. | Mayor coste. | Cuando prima precisión. |
| L2 regularization | Busca una curva estable sin fase de overfitting. | Introduce otro hiperparámetro. | Cuando se desea usar más datos para actualizar pesos. |
| Dropout / modelo menos complejo | Limita overfitting. | Puede reducir capacidad. | Cuando el modelo memoriza demasiado pronto. |
| Exportar solo el modelo | Artefacto pequeño para inferencia. | No restaura el entrenamiento. | Después de elegir el modelo final. |

### 3.16 Implementación práctica

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=False,
)

steps_per_epoch = total_training_examples // (batch_size * num_checkpoints)
model.fit(
    train_dataset.repeat(),
    validation_data=eval_dataset,
    epochs=num_checkpoints,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint_cb],
)
```

En PyTorch, el estado mínimo mostrado incluye `epoch`, `model_state_dict`, `optimizer_state_dict` y `loss`, además de cualquier estado propio necesario.

### 3.17 Ejemplo mínimo conceptual

Un entrenamiento de dos días falla después de 30 horas. Con checkpoints completos cada intervalo virtual, el proceso restaura el último estado, incluido optimizer y schedule, y repite solo el tramo posterior. El mismo historial permite elegir un checkpoint anterior si la validación empeoró al final.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En datathons con sesiones limitadas, checkpointing evita perder entrenamientos y permite conservar el mejor estado, no solo el último. Tiene prioridad alta para deep learning y baja para modelos que entrenan en segundos. Bajo presión conviene guardar por mejora de la métrica o por intervalos suficientemente amplios para no saturar I/O. Debe mantenerse un conjunto de validación coherente y no usar el test final para seleccionar checkpoints.

### 3.19 Checklist operativo del patrón

- [ ] Guardar modelo y optimizer, no solo pesos de serving.
- [ ] Definir una frecuencia según coste de pérdida e I/O.
- [ ] Asociar cada checkpoint con paso y métrica.
- [ ] Probar restauración antes de un entrenamiento largo.
- [ ] Mantener batch size y número de ejemplos comparables.
- [ ] Separar checkpoint final de artefacto exportado.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Useful Overfitting | Complementario | Durante experimentación puede verificarse capacidad de ajuste antes de regularizar y seleccionar estados. |
| Transfer Learning | Complementario | Un checkpoint previo puede ser la base para fine-tuning sin cambiar arquitectura. |
| Distribution Strategy | Dependencia | El entrenamiento distribuido necesita resiliencia y favorece epochs virtuales. |
| Hyperparameter Tuning | Complementario | Los trials largos se benefician de checkpoints y presupuestos comparables. |

## Pattern 13: Transfer Learning

### 3.1 Problema

Los modelos de imagen y texto de alta calidad necesitan grandes datasets etiquetados. En dominios especializados suele haber solo cientos o miles de ejemplos, insuficientes para aprender desde cero todas las representaciones, pero un modelo genérico no produce directamente las etiquetas específicas requeridas.

### 3.2 Contexto de aplicación

- Datos propios limitados y existencia de un modelo preentrenado sobre la misma modalidad.
- Tarea igual o similar: clasificación de imágenes, object detection, style transfer, generación, clasificación de texto o traducción.
- Idealmente, dominio de preentrenamiento similar; como mínimo, modalidad y tipo de tarea compatibles.
- Uso más maduro en imágenes y texto; la transferencia entre problemas tabulares es más difícil por heterogeneidad de features y objetivos.

### 3.3 Señales que indican que el patrón puede ser necesario

- Solo hay cientos o pocos miles de ejemplos etiquetados.
- Entrenar desde cero sobreajusta o no aprende representaciones útiles.
- Existe un modelo preentrenado con la misma modalidad y una tarea comparable.
- El coste de preentrenar una representación propia es prohibitivo.
- **Inferencia práctica:** el dominio tiene patrones de bajo nivel reutilizables, como bordes en imágenes o estructura semántica en texto.

### 3.4 Solución propuesta

1. Elegir un modelo preentrenado para la misma modalidad y una tarea lo más parecida posible.
2. Retirar su capa de salida específica.
3. Usar una bottleneck layer como representación compacta.
4. Añadir nuevas capas de salida para las etiquetas propias.
5. Congelar el modelo base para feature extraction o descongelar algunas capas para fine-tuning.
6. Entrenar las capas nuevas con el dataset especializado.
7. Decidir progresivamente cuántas capas descongelar según datos, similitud, pérdida y presupuesto.

### 3.5 Cómo funciona

Las primeras capas de una red aprenden rasgos reutilizables; las últimas combinan esos rasgos para las etiquetas originales. El patrón conserva las representaciones hasta el bottleneck y reemplaza la cabeza. En feature extraction, solo aprende la nueva cabeza. En fine-tuning, también actualiza parte o todo el backbone con un learning rate bajo.

### 3.6 Por qué funciona

El preentrenamiento sobre un dataset grande ya resolvió el costoso aprendizaje de características. En una CNN, las capas tempranas capturan bordes y formas, y las posteriores objetos y configuraciones más específicas. Reutilizarlas reduce la cantidad de datos necesaria para la nueva tarea. El beneficio aumenta cuando modalidad, imágenes y objetivo son similares.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Modelo preentrenado, dataset especializado y nuevas etiquetas. |
| Transformación | Eliminación de la cabeza original, congelado o fine-tuning del backbone y entrenamiento de una nueva cabeza. |
| Salida | Modelo adaptado a la tarea específica. |

### 3.8 Procedimiento operativo

1. Verificar compatibilidad de modalidad, forma de entrada y tarea.
2. Cargar el backbone sin su capa superior.
3. Seleccionar la bottleneck layer.
4. Congelar inicialmente el backbone.
5. Añadir pooling/flattening y una salida con la dimensión adecuada.
6. Entrenar la cabeza y evaluar.
7. Si hay suficientes datos y la tarea es similar, descongelar progresivamente desde las capas finales.
8. Mantener learning rate bajo y pocas iteraciones durante progressive fine-tuning.
9. Detener el descongelado cuando la pérdida se estabilice o empeore.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Modelo fuente | Determina qué representaciones se reutilizan. | Dominio o tarea incompatibles reducen el valor de la transferencia. |
| Bottleneck layer | Fija la representación transferida. | Una capa demasiado temprana es genérica; una demasiado tardía puede ser específica. |
| Capas congeladas | Reduce coste y overfitting. | Congelar demasiado limita adaptación. |
| Learning rate de fine-tuning | Controla cuánto cambian pesos útiles. | Un valor alto destruye conocimiento preentrenado. |
| Iteraciones | Permiten adaptación. | Exceso sobre dataset pequeño induce overfitting. |

### 3.10 Ventajas

- Funciona con muchos menos datos etiquetados que el entrenamiento desde cero.
- Reduce tiempo y cómputo.
- Reutiliza representaciones aprendidas sobre grandes corpus.
- Feature extraction ofrece una opción barata; fine-tuning permite mayor adaptación.

### 3.11 Limitaciones

- Depende de encontrar un modelo compatible.
- Es menos directo para datos tabulares heterogéneos.
- La entrada debe adaptarse a forma y preprocessing esperados por el backbone.
- La bottleneck no garantiza contener la información necesaria para una tarea muy distinta.

### 3.12 Riesgos y trade-offs

- Fine-tuning con pocos datos puede sobreajustar y borrar información general.
- Feature extraction es barata, pero puede quedarse corta si el dominio difiere.
- Descongelar más capas eleva tiempo, memoria y sensibilidad a hiperparámetros.
- Usar un modelo solo para sus etiquetas originales no constituye transfer learning; hay que adaptar la salida.
- **Inferencia práctica:** preprocessing distinto al esperado por el modelo fuente produce degradación aunque la arquitectura sea correcta.

### 3.13 Cuándo utilizarlo

- Dataset pequeño con modelo fuente compatible.
- Imágenes o texto con patrones reutilizables.
- Feature extraction si hay pocos datos, tarea diferente o presupuesto bajo.
- Fine-tuning si hay más datos, tarea similar y presupuesto alto.

### 3.14 Cuándo no utilizarlo

- No existe un modelo preentrenado de modalidad o tarea compatible.
- La tarea y el dominio difieren tanto que las representaciones no son útiles.
- En tabular, cuando no hay una representación preentrenada transferible adecuada.
- Si se dispone de suficiente dato y entrenar desde cero es preferible y viable.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Feature extraction | Menor coste y menor riesgo de overfitting. | Menor adaptación. | Dataset pequeño o tarea distinta. |
| Fine-tuning | Mayor adaptación y posible precisión. | Más coste y riesgo de olvidar conocimiento general. | Dataset mayor y tarea similar. |
| Progressive fine-tuning | Busca cuántas capas adaptar de manera controlada. | Requiere varias ejecuciones. | Cuando no es obvio dónde congelar. |
| Entrenar desde cero | Control total del modelo. | Exige muchos datos y cómputo. | Dataset propio grande o ausencia de modelo compatible. |

### 3.16 Implementación práctica

```python
base = tf.keras.applications.VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=(150, 150, 3),
)
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])
```

**Nota práctica de actualización:** las URLs y versiones concretas de módulos mostradas por el libro pertenecen a 2020; al implementar debe verificarse la API instalada, sin cambiar el mecanismo del patrón.

### 3.17 Ejemplo mínimo conceptual

Hay 400 radiografías etiquetadas para una clasificación binaria. Se toma un clasificador de imágenes preentrenado, se elimina su salida, se congelan sus capas y se añade una cabeza de dos clases. El resultado esperado es aprovechar representaciones visuales ya aprendidas sin intentar aprenderlas desde 400 ejemplos.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En visión o NLP, un backbone preentrenado puede convertir rápidamente un baseline débil en uno competitivo. Bajo presión, feature extraction tiene prioridad alta: pocas capas entrenables y menor riesgo. Fine-tuning progresivo tiene prioridad media cuando hay GPU, tiempo y suficiente dato. Debe compararse contra el backbone congelado y vigilar overfitting; la semejanza de modalidad y tarea es una señal más importante que la popularidad del modelo.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar misma modalidad y tarea comparable.
- [ ] Reproducir forma y preprocessing de entrada.
- [ ] Sustituir la cabeza original.
- [ ] Empezar con el backbone congelado.
- [ ] Comparar feature extraction y fine-tuning.
- [ ] Descongelar progresivamente con learning rate bajo.
- [ ] Vigilar overfitting sobre el dataset pequeño.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Checkpoints | Complementario | Permite iniciar fine-tuning desde un estado previamente entrenado y conservar estados intermedios. |
| Embeddings | Complementario | La bottleneck es una representación compacta; embeddings preentrenados pueden servir como base transferida. |
| Useful Overfitting | Complementario | Distillation transfiere la función de un modelo grande a uno pequeño mediante datos generados. |

## Pattern 14: Distribution Strategy

### 3.1 Problema

Al crecer datos y parámetros, aumentan memoria, cómputo, tiempo de entrenamiento y coste. Una sola GPU deja de ser suficiente; esperar días o semanas frena iteración y en algunos problemas no es válido reducir datos o dividir el objetivo.

### 3.2 Contexto de aplicación

- Redes grandes entrenadas sobre datasets masivos.
- Uno o varios hosts con múltiples GPU/TPU.
- Modelos que no caben en un dispositivo.
- Redes fiables y rápidas para sincronización, o flotas de workers lentos/no fiables para ejecución asíncrona.
- Pipelines donde I/O puede convertirse en el cuello de botella.

### 3.3 Señales que indican que el patrón puede ser necesario

- El modelo no cabe en la memoria de un solo dispositivo.
- El tiempo de entrenamiento bloquea iteraciones útiles.
- Un acelerador queda ocioso esperando datos.
- La comunicación entre workers domina el paso de entrenamiento.
- Un worker lento detiene a todos los demás en modo síncrono.
- **Inferencia práctica:** aumentar hardware no eleva throughput de forma proporcional, indicando un cuello de botella de red, I/O o batch size.

### 3.4 Solución propuesta

1. Distribuir el bucle entre múltiples workers.
2. Elegir data parallelism para repartir slices del batch y agregar gradientes, o model parallelism para repartir el propio modelo.
3. Elegir sincronización síncrona o asíncrona según topología y fiabilidad.
4. Ajustar batch size y learning rate para equilibrar utilización y convergencia.
5. Usar aceleradores y comunicaciones adecuadas.
6. Optimizar el input pipeline con formatos eficientes y prefetch para que los aceleradores no esperen.

### 3.5 Cómo funciona

En data parallelism síncrono, cada worker mantiene una copia del modelo, procesa una fracción del mini-batch y participa en all-reduce; el gradiente agregado produce una actualización común. En modo asíncrono, los workers envían gradientes a parameter servers sin esperarse; esto aumenta throughput, aunque algunos gradientes pueden calcularse sobre estado antiguo. En model parallelism, cada dispositivo almacena y calcula una parte distinta de la red sobre el mismo mini-batch.

### 3.6 Por qué funciona

El trabajo que antes era secuencial se reparte. Más workers elevan throughput y reducen el tiempo hasta convergencia si el coste de coordinación no domina. El método apropiado alinea la unidad comunicada con la arquitectura: data parallelism es eficiente cuando hay mucho cómputo por peso, como convoluciones; model parallelism cuando hay mucho cómputo por activación o el modelo es demasiado ancho para un dispositivo.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Modelo, dataset shardeable, workers, aceleradores y topología de red. |
| Transformación | Reparto de datos o modelo, cálculo paralelo y coordinación de gradientes/parámetros. |
| Salida | Modelo entrenado en menos tiempo, sujeto a overhead y equivalencia estadística de la estrategia. |

### 3.8 Procedimiento operativo

1. Medir memoria, cómputo, I/O y comunicación del entrenamiento actual.
2. Si el modelo cabe por dispositivo, probar data parallelism; si no, model parallelism.
3. En un host con enlace GPU-GPU rápido, usar estrategia mirrored; con CPU-GPU más rápido, central storage puede convenir.
4. En múltiples hosts fiables y rápidos, usar sincronización; con workers poco fiables, considerar parameter server asíncrono.
5. Shardear correctamente datos y batches.
6. Incrementar batch de forma controlada y ajustar learning rate cuando corresponda.
7. Usar epochs virtuales/checkpoints porque el concepto de epoch puede ser ambiguo en asíncrono.
8. Prefetch y formatos optimizados para solapar preparación y ejecución.
9. Medir throughput, tiempo a métrica y calidad final; no solo utilización.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Síncrono vs asíncrono | Consistencia frente a tolerancia a stragglers. | Esperas globales o gradientes stale. |
| Data vs model parallelism | Reparte ejemplos o capas. | Elegir mal aumenta comunicación. |
| Número de workers | Eleva throughput potencial. | Rendimientos decrecientes por coordinación. |
| Batch size global | Reduce pasos y mejora utilización. | Batch muy grande empeora convergencia y generalización. |
| Learning rate | Puede escalarse con batch grande. | Escalado incorrecto eleva error. |
| Prefetch | Solapa CPU/I/O con acelerador. | Consume memoria y no corrige almacenamiento lento por sí solo. |

### 3.10 Ventajas

- Reduce entrenamientos de semanas a horas en casos adecuados.
- Permite modelos o datasets que exceden un dispositivo.
- Asíncrono tolera workers lentos o fallidos.
- Model parallelism también puede reducir latencia de serving para modelos enormes.

### 3.11 Limitaciones

- Requiere hardware, red y orquestación adicionales.
- La aceleración no es lineal por comunicación e I/O.
- El modo asíncrono dificulta contar epochs y puede perder slices.
- Los aceleradores rápidos exponen nuevos cuellos de botella de entrada.

### 3.12 Riesgos y trade-offs

- Síncrono se bloquea por el worker más lento.
- Asíncrono usa estados posiblemente obsoletos y sacrifica contabilidad exacta.
- Batch grande mejora eficiencia de hardware, pero puede empeorar calidad final.
- Model parallelism agrega comunicación de activaciones; data parallelism comunica pesos/gradientes.
- Más workers aumentan coste económico aunque reduzcan tiempo de pared.

### 3.13 Cuándo utilizarlo

- Cuando entrenamiento en un dispositivo es inviable por tiempo o memoria.
- Data parallelism para modelos replicables y datos shardeables.
- Model parallelism cuando el modelo no cabe o tiene capas muy anchas.
- Asíncrono con muchos workers heterogéneos o poco fiables.
- Síncrono con dispositivos rápidos y enlaces fuertes.

### 3.14 Cuándo no utilizarlo

- Si un solo dispositivo cumple el presupuesto de tiempo y memoria.
- Si el pipeline de datos no alimenta ni siquiera un acelerador.
- Si la sobrecarga de red supera el ahorro computacional.
- Si no se puede validar que batch y learning rate preserven calidad.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Synchronous data parallelism | Gradiente agregado consistente. | Sensible a stragglers y red lenta. | Hardware homogéneo y bien conectado. |
| Asynchronous parameter server | Mayor throughput y tolerancia a fallos. | Gradientes stale y epochs ambiguos. | Workers numerosos o poco fiables. |
| Model parallelism | Permite modelos mayores que un dispositivo. | Comunicación entre partes del modelo. | El modelo no cabe o tiene capas muy anchas. |
| ASIC/TPU/FPGA | Gran aceleración matricial. | Dependencia de hardware y adaptación. | Cargas compatibles y escala suficiente. |
| Optimizar I/O/prefetch | Mejora utilización sin cambiar el modelo. | No resuelve falta de memoria del modelo. | Aceleradores ociosos esperando datos. |

### 3.16 Implementación práctica

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(loss="mse", optimizer="sgd")

model.fit(train_dataset, epochs=epochs)
```

En PyTorch, el capítulo combina `DistributedDataParallel`, `DistributedSampler` e inicialización del process group. El mecanismo esencial es que cada proceso reciba su shard y sincronice gradientes mediante el backend.

### 3.17 Ejemplo mínimo conceptual

Un modelo convolucional tarda semanas en una GPU. Se replica en varias GPU, cada una procesa una fracción del batch y los gradientes se agregan en cada paso. Con un input pipeline que prefetch, el resultado esperado es menor tiempo a convergencia sin cambiar la tarea del modelo.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

La prioridad suele ser baja en coding tests y media en datathons con datasets o redes grandes. Antes de distribuir, conviene comprobar que I/O no sea el cuello de botella. Data parallelism es la opción más directa cuando hay varias GPU. El riesgo competitivo es gastar tiempo configurando infraestructura o usar batches tan grandes que empeoren validación. Debe juzgarse por tiempo a una métrica estable, no solo por ejemplos/segundo.

### 3.19 Checklist operativo del patrón

- [ ] Medir cuello de botella de cómputo, memoria, red e I/O.
- [ ] Elegir data o model parallelism según arquitectura.
- [ ] Elegir síncrono o asíncrono según topología.
- [ ] Shardear datos sin duplicación accidental.
- [ ] Ajustar batch size y learning rate conjuntamente.
- [ ] Prefetch para evitar aceleradores ociosos.
- [ ] Usar checkpoints y epochs virtuales.
- [ ] Comparar calidad final además de throughput.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Checkpoints | Dependencia | Los jobs distribuidos largos necesitan recuperación y epochs virtuales. |
| Hyperparameter Tuning | Complementario | Los trials pueden paralelizarse, pero demasiados trials simultáneos reducen aprendizaje entre ensayos. |

## Pattern 15: Hyperparameter Tuning

### 3.1 Problema

Los hiperparámetros -learning rate, batch size, profundidad, neuronas, kernels o epochs- no se aprenden como los pesos. El ensayo manual consume tiempo, cambia varias variables sin aislar su efecto y no demuestra optimalidad. Grid search sufre explosión combinatoria y no aprende de resultados; random search reduce coste, pero puede omitir la mejor región.

### 3.2 Contexto de aplicación

- Cualquier modelo con decisiones de arquitectura o entrenamiento que afecten la métrica.
- Entrenamientos donde cada trial es caro y el espacio es amplio o mixto.
- Objetivo evaluable en validación, posiblemente distinto de la loss de entrenamiento.
- Infraestructura capaz de ejecutar trials secuenciales o paralelos.

### 3.3 Señales que indican que el patrón puede ser necesario

- La calidad cambia mucho con learning rate, profundidad, batch o arquitectura.
- La búsqueda manual consume muchas ejecuciones y no deja un procedimiento reproducible.
- Grid search crece exponencialmente con cada parámetro añadido.
- Se desea optimizar una métrica de negocio distinta de la loss.
- Cada entrenamiento tarda horas, por lo que reducir trials completos tiene gran valor.

### 3.4 Solución propuesta

1. Encapsular la construcción y el entrenamiento del modelo como función objetivo.
2. Definir nombres, tipos, rangos, escalas y pasos de los hiperparámetros.
3. Elegir una métrica calculada sobre validation/test de evaluación, no training.
4. Usar Bayesian optimization para construir una surrogate function barata.
5. Proponer una combinación, ejecutar un trial completo y devolver su métrica.
6. Actualizar el surrogate con el resultado y repetir hasta agotar el presupuesto.
7. Para escala, usar un servicio resiliente y controlar cuántos trials corren en paralelo.

### 3.5 Cómo funciona

La búsqueda forma un bucle exterior alrededor del entrenamiento. El modelo entrenado es una función caja negra que asigna hiperparámetros a una métrica. Bayesian optimization aprende un surrogate -por ejemplo, Gaussian process o tree-structured Parzen estimator- y lo consulta muchas veces para elegir trials informativos. Solo las combinaciones prometedoras pagan el coste del entrenamiento completo; sus resultados actualizan el surrogate.

### 3.6 Por qué funciona

El problema exterior suele ser no diferenciable y cada evaluación es cara. A diferencia de grid/random, el método bayesiano usa evidencia de trials previos para concentrar el presupuesto en regiones prometedoras. Además, permite optimizar una métrica de validación o negocio que no sea la loss diferenciable del bucle interior.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Función de entrenamiento, espacio de búsqueda, métrica y presupuesto de trials. |
| Transformación | Optimización exterior que propone, evalúa y aprende de combinaciones. |
| Salida | Mejor combinación observada y registro de trials; opcionalmente una familia de modelos para ensemble. |

### 3.8 Procedimiento operativo

1. Fijar un baseline sin tuning.
2. Seleccionar solo hiperparámetros con impacto plausible en arquitectura o entrenamiento.
3. Definir rango, escala lineal/logarítmica, tipo y restricciones.
4. Definir la métrica de validación alineada con el objetivo.
5. Fijar máximo de trials y paralelismo.
6. Ejecutar Bayesian optimization.
7. Registrar parámetros, métrica y estado de cada trial.
8. Comparar el mejor trial contra el baseline.
9. Reentrenar o confirmar la configuración elegida sin contaminar el test final.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Espacio de búsqueda | Limita combinaciones explorables. | Excluir la zona útil hace imposible encontrarla. |
| Escala lineal/log | Cambia densidad de propuestas. | Escala inadecuada desperdicia trials. |
| Métrica objetivo | Orienta toda la búsqueda. | Una métrica desalineada optimiza el comportamiento equivocado. |
| `max_trials` | Define presupuesto. | Muy bajo explora poco; muy alto eleva coste y adaptación a validación. |
| Trials paralelos | Reduce tiempo de pared. | Reduce cuánto puede aprender cada propuesta de resultados previos. |
| Early stopping de trials | Elimina candidatos débiles. | Puede descartar configuraciones de convergencia lenta. |

### 3.10 Ventajas

- Aprende de trials anteriores.
- Requiere menos entrenamientos completos que búsquedas ciegas en espacios costosos.
- Admite hiperparámetros de arquitectura y entrenamiento.
- Puede optimizar métricas técnicas o de negocio.
- Puede ejecutarse en infraestructura distribuida y resiliente.

### 3.11 Limitaciones

- Cada trial sigue requiriendo entrenar un modelo.
- El usuario define manualmente espacio, métrica y presupuesto.
- El surrogate añade coste de decisión, aunque menor que el entrenamiento.
- La búsqueda no puede recuperar valores fuera del espacio especificado.

### 3.12 Riesgos y trade-offs

- Optimizar repetidamente la misma validación puede sobreajustar las decisiones a ese conjunto.
- Más paralelismo termina antes, pero aprovecha menos feedback secuencial.
- Un espacio excesivo encarece; uno estrecho introduce sesgo humano.
- Grid search es reproducible pero bruto; random search es barato pero puede no cubrir la zona óptima.
- Genetic algorithms usan resultados previos, pero requieren muchos entrenamientos y su complejidad crece con el espacio.

### 3.13 Cuándo utilizarlo

- Cuando los hiperparámetros influyen materialmente y hay presupuesto para varios trials.
- Cuando cada trial es caro y conviene usar una búsqueda informada.
- Cuando existe una métrica de validación estable y alineada.
- Cuando el espacio no puede explorarse razonablemente por grid.

### 3.14 Cuándo no utilizarlo

- Antes de disponer de un baseline funcional y una validación confiable.
- Si un único entrenamiento ya excede el presupuesto total.
- Si el espacio contiene parámetros sin efecto o la métrica es muy ruidosa.
- Para compensar errores de datos, leakage o una formulación incorrecta del problema.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Manual search | Simple y guiada por conocimiento. | Costosa, poco sistemática y no aislada. | Muy pocos parámetros y modelos rápidos. |
| Grid search | Exhaustiva sobre una grilla finita. | Explosión combinatoria; no aprende. | Espacio pequeño y discreto. |
| Random search | Más rápida que grid. | Puede omitir la región óptima. | Presupuesto pequeño y espacio amplio. |
| Bayesian optimization | Usa surrogate y feedback. | Mayor complejidad de coordinación. | Trials caros. |
| Genetic algorithms | Aprende mediante selección de trials aptos. | Muchos entrenamientos y menor soporte. | Casos donde esa estrategia esté disponible y justificada. |

### 3.16 Implementación práctica

```python
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(
            hp.Int("hidden", 32, 256, step=32),
            activation="relu",
        ),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 0.005, 0.01, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

**Nota práctica de actualización:** el libro usa el nombre de paquete `kerastuner` vigente en 2020. Una implementación actual debe adaptar el import a la versión instalada, manteniendo la definición explícita del espacio, objetivo y presupuesto.

### 3.17 Ejemplo mínimo conceptual

Un clasificador tarda horas y depende de learning rate, anchura y profundidad. En vez de evaluar toda la grilla, un optimizador bayesiano propone un trial, observa accuracy de validación, actualiza su surrogate y concentra nuevos trials en combinaciones prometedoras. El resultado esperado es una configuración mejor dentro del presupuesto, no una garantía de óptimo global.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.**

En datathons, primero debe existir un baseline y un split estable. Random search puede ser suficiente bajo presión; Bayesian optimization gana prioridad si los trials son caros pero repetibles. Conviene acotar el espacio con pocas decisiones de impacto y guardar todos los trials. La mejora de leaderboard razonable proviene de encontrar configuraciones que el default no captura, pero no debe prometerse: una validación ruidosa o distinta del leaderboard puede dirigir la búsqueda al lugar equivocado.

### 3.19 Checklist operativo del patrón

- [ ] Construir y registrar un baseline.
- [ ] Congelar la estrategia de validación.
- [ ] Definir rangos, escalas y tipos explícitos.
- [ ] Elegir una métrica alineada con el objetivo.
- [ ] Fijar presupuesto y paralelismo.
- [ ] Medir la métrica fuera del training set.
- [ ] Registrar todos los trials.
- [ ] Confirmar el mejor resultado sin reutilizar el test final.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Checkpoints | Complementario | Hace recuperables trials largos y permite presupuestos comparables. |
| Distribution Strategy | Puede combinarse | Permite distribuir un trial o ejecutar varios; el paralelismo cambia el feedback disponible. |
| Ensembles | Puede combinarse | El framework de tuning puede producir una familia de modelos para combinar. |

# 4. Reglas de decisión del capítulo

| Situación | Pattern recomendado | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Dominio finito, labels exactos y simulación lenta | Useful Overfitting | Ajustar toda la tabla reemplaza un cálculo caro. | Confirmar cobertura y resolución. |
| El modelo no memoriza un batch pequeño verificado | Useful Overfitting | Diagnostica arquitectura, loss o pipeline. | No confundir la prueba con generalización. |
| Entrenamiento largo con riesgo de interrupción | Checkpoints | Conserva estado completo restaurable. | Probar restore y equilibrar frecuencia/I/O. |
| La validación empeora tras varias iteraciones | Checkpoints | Permite early stopping o seleccionar un estado anterior. | No reaccionar a una sola oscilación pequeña. |
| Dataset especializado pequeño y backbone compatible | Transfer Learning | Reutiliza representaciones aprendidas con muchos datos. | Congelar primero; fine-tuning puede sobreajustar. |
| Modelo o datos exceden un dispositivo | Distribution Strategy | Reparte cómputo, memoria o datos. | Medir red, I/O y calidad con batch grande. |
| Workers rápidos y bien conectados | Distribution Strategy síncrona | Mantiene actualizaciones agregadas consistentes. | Un straggler puede frenar todo el paso. |
| Workers heterogéneos o poco fiables | Distribution Strategy asíncrona | Evita bloqueo global. | Acepta gradientes stale y epochs ambiguos. |
| Espacio de hiperparámetros amplio y trials caros | Hyperparameter Tuning bayesiano | Aprende de trials y reduce evaluaciones ciegas. | Definir bien métrica, espacio y presupuesto. |

# 5. Comparación entre patrones del capítulo

| Pattern | Problema que resuelve | Coste | Beneficio | Complejidad |
| ------- | --------------------- | ----- | --------- | ----------- |
| Useful Overfitting | Ajuste exacto de funciones deterministas o diagnóstico de aprendizaje. | Entrenamiento potencialmente largo; generación exhaustiva. | Inferencia rápida y detección de bugs. | Media |
| Checkpoints | Pérdida de progreso y elección del estado final. | Almacenamiento e I/O. | Resiliencia, selección y fine-tuning. | Baja-Media |
| Transfer Learning | Escasez de labels propios. | Adaptación y posible fine-tuning. | Menos datos y tiempo para alta calidad. | Media |
| Distribution Strategy | Tiempo o memoria de entrenamiento insuficientes. | Hardware, red y operación. | Escala y menor tiempo de pared. | Alta |
| Hyperparameter Tuning | Selección ineficiente de configuración. | Múltiples entrenamientos. | Mejor configuración dentro del presupuesto. | Media-Alta |

# 6. Riesgos globales del capítulo

- **Overfitting:** puede ser objetivo solo bajo condiciones deterministas; aparece como riesgo en checkpoint selection, transfer learning y tuning.
- **Validación incorrecta:** elegir checkpoints o hiperparámetros sobre training produce métricas engañosas; reutilizar demasiado validation adapta decisiones a ese conjunto.
- **Distribución incorrecta:** cobertura parcial disfrazada de exhaustiva invalida Useful Overfitting; workers y sharding incorrectos afectan entrenamiento distribuido.
- **Reproducibilidad y recuperación:** exportar solo el predictor no conserva el estado necesario para reanudar.
- **Complejidad operacional:** distribución y tuning agregan coordinación, fallos, almacenamiento y costes.
- **Coste computacional:** fine-tuning, trials y distribución deben justificarse por datos, tiempo y métrica.
- **Training-serving consistency:** Useful Overfitting requiere el mismo redondeo de entradas; Transfer Learning requiere el preprocessing esperado por el backbone.
- **I/O:** checkpoints frecuentes y aceleradores distribuidos pueden quedar limitados por lectura, transferencia o escritura.

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

Comprobar tamaño y modalidad del dataset, disponibilidad de modelos preentrenados, coste estimado de un fit y estabilidad del entorno. Si hay una red o pipeline nuevo, reservar un pequeño batch verificado para probar que puede aprenderlo. Definir el split de validación antes de tuning o checkpoint selection.

### Durante el baseline

Usar un backbone congelado en visión/NLP si el dataset es pequeño; evita entrenar representaciones desde cero. Para modelos rápidos, no añadir distribución. Guardar un baseline con su métrica antes de explorar hiperparámetros.

### Durante la mejora del modelo

Aplicar fine-tuning progresivo cuando el dominio/tarea sean similares y haya presupuesto. Usar tuning acotado sobre las decisiones de mayor impacto. Distribuir solo si una medición confirma que cómputo o memoria son el límite; optimizar I/O antes si los aceleradores esperan datos.

### Durante validación

Seleccionar checkpoints por la métrica de evaluación, conservar el historial y no tomar el último estado por defecto. Mantener constante el número de ejemplos vistos al comparar batch sizes. Evitar elegir hiperparámetros sobre el test final.

### Bajo presión de tiempo

- **Alta prioridad:** sanity check de un batch, checkpoint del mejor modelo, Transfer Learning con backbone congelado y tuning pequeño sobre una validación fija.
- **Prioridad media:** fine-tuning progresivo y Bayesian optimization si los trials caben en el presupuesto.
- **Baja prioridad:** infraestructura distribuida compleja, salvo que el modelo no pueda ejecutarse de otro modo; Useful Overfitting exhaustivo, salvo problema determinista explícito.

# 8. Checklist global del capítulo

- [ ] Confirmar si el objetivo es generalización o reproducción determinista.
- [ ] Verificar que el pipeline puede ajustar un lote pequeño correcto.
- [ ] Definir validación y baseline antes de seleccionar estados o tunear.
- [ ] Guardar estado restaurable en entrenamientos costosos.
- [ ] Considerar un modelo preentrenado si faltan labels.
- [ ] Medir cómputo, memoria, red e I/O antes de distribuir.
- [ ] Mantener comparable el número de ejemplos vistos.
- [ ] Acotar espacio, métrica y presupuesto de tuning.
- [ ] Comparar toda mejora contra el baseline.

# 9. Mapeo hacia `knowledge/`

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Useful Overfitting | `knowledge/models/useful_overfitting.md` | Media |
| Sanity check por overfitting de un batch | `knowledge/validation/overfit_one_batch_check.md` | Alta |
| Checkpoints y restore | `knowledge/mlops/checkpoints.md` | Alta |
| Early stopping y checkpoint selection | `knowledge/validation/model_selection_during_training.md` | Alta |
| Virtual epochs | `knowledge/reproducibility/virtual_epochs.md` | Alta |
| Transfer Learning | `knowledge/models/transfer_learning.md` | Alta |
| Feature extraction vs fine-tuning | `knowledge/models/fine_tuning.md` | Alta |
| Distribution Strategy | `knowledge/mlops/distributed_training.md` | Media |
| Batch size en distribución | `knowledge/models/distributed_batch_size.md` | Media |
| Hyperparameter Tuning | `knowledge/tuning/hyperparameter_tuning.md` | Alta |
| Search algorithms y budgets | `knowledge/tuning/search_strategies.md` | Alta |

# 10. Índice de patrones reutilizables

| Pattern | Activador principal | Acción principal | Knowledge destino |
| ------- | ------------------- | ---------------- | ----------------- |
| Useful Overfitting | Función determinista exhaustiva o batch que debería memorizarse. | Entrenar hasta ajuste exacto bajo condiciones controladas. | `knowledge/models/` / `knowledge/validation/` |
| Checkpoints | Entrenamiento largo, frágil o con sobreajuste posterior. | Guardar estado completo y seleccionar/restaurar. | `knowledge/mlops/` / `knowledge/reproducibility/` |
| Transfer Learning | Pocos datos y modelo fuente compatible. | Reutilizar backbone y adaptar la cabeza. | `knowledge/models/` |
| Distribution Strategy | Tiempo o memoria de un dispositivo insuficientes. | Repartir datos o modelo y optimizar comunicación/I/O. | `knowledge/mlops/` |
| Hyperparameter Tuning | Configuración sensible y espacio caro. | Optimizar un bucle exterior sobre validación. | `knowledge/tuning/` |

# 11. Trazabilidad

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Propósito y training loop típico | pp. 139-141; Figure 4-1 |
| Useful Overfitting - problema y solución | pp. 141-144; Figures 4-2 a 4-4 |
| Useful Overfitting - condiciones, interpolación, Monte Carlo y PDE | pp. 145-148; Figure 4-5 |
| Useful Overfitting - distillation y overfitting de un batch | pp. 148-149 |
| Checkpoints - problema, estado completo e implementación | pp. 149-152; Figure 4-6 |
| Checkpoints - fases del entrenamiento | pp. 152-154; Figures 4-7 y 4-8 |
| Checkpoints - early stopping, selection y regularización | pp. 154-157; Figures 4-9 y 4-10 |
| Checkpoints - fine-tuning | pp. 157-158; Figure 4-11 |
| Checkpoints - steps y virtual epochs | pp. 158-160; Figure 4-12 |
| Transfer Learning - problema, solución y bottleneck | pp. 161-169; Figure 4-13 |
| Transfer Learning - por qué funciona | pp. 169-171; Figure 4-14 |
| Transfer Learning - feature extraction y fine-tuning | pp. 172-173; Table 4-1 |
| Transfer Learning - texto, imágenes, tabular y sentence embeddings | pp. 173-174 |
| Distribution Strategy - problema y data parallelism | pp. 175-180; Figures 4-15 y 4-16; Table 4-2 |
| Distribution Strategy - por qué funciona | pp. 181-182; Figures 4-17 y 4-18 |
| Distribution Strategy - model parallelism y aceleradores | pp. 183-185; Figure 4-19 |
| Distribution Strategy - batch size e I/O | pp. 185-187; Figures 4-20 a 4-22 |
| Hyperparameter Tuning - problema, manual/grid/random search | pp. 187-190 |
| Hyperparameter Tuning - solución con Bayesian optimization | pp. 190-194; Figures 4-23 y 4-24 |
| Hyperparameter Tuning - servicio gestionado y paralelismo | pp. 194-197; Figure 4-25 |
| Hyperparameter Tuning - genetic algorithms | pp. 197-198 |
| Resumen del capítulo | pp. 198-199 |
