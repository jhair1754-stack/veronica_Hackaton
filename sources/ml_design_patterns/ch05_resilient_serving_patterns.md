---
title: "Design Patterns for Resilient Serving"
book: "Machine Learning Design Patterns"
chapter: "5"
topics:
  - "Resilient serving"
  - "Online and batch inference"
  - "Production monitoring"
  - "Edge deployment"
  - "Prediction traceability"
patterns:
  - "Stateless Serving Function"
  - "Batch Serving"
  - "Continued Model Evaluation"
  - "Two-Phase Predictions"
  - "Keyed Predictions"
target_knowledge_files:
  - "knowledge/mlops/stateless_serving_function.md"
  - "knowledge/mlops/batch_serving.md"
  - "knowledge/mlops/continued_model_evaluation.md"
  - "knowledge/mlops/two_phase_predictions.md"
  - "knowledge/mlops/keyed_predictions.md"
  - "knowledge/validation/production_model_monitoring.md"
---

# 5. Design Patterns for Resilient Serving

## 1. Propósito del capítulo

El capítulo aborda problemas que aparecen después del entrenamiento, cuando un modelo debe servir predicciones con resiliencia, escala y poca intervención humana. Los patrones separan el artefacto de inferencia del entorno de entrenamiento, distinguen latencia de throughput, vigilan la vigencia del modelo, reparten inferencia entre edge y cloud, y mantienen la correspondencia entre entradas y salidas en sistemas distribuidos.

# 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Stateless Serving Function | Servir un modelo a baja latencia, a escala y desde clientes heterogéneos | Serving | Alta |
| Batch Serving | Inferir de forma asíncrona sobre millones o miles de millones de instancias | Serving / Pipeline | Alta |
| Continued Model Evaluation | Detectar cuándo un modelo desplegado deja de ser apto por data drift o concept drift | Validation / MLOps | Alta |
| Two-Phase Predictions | Conservar capacidad predictiva bajo restricciones de edge, conectividad y latencia | Serving | Media |
| Keyed Predictions | Asociar correctamente entradas y salidas cuando el procesamiento distribuido altera el orden | Serving / Pipeline | Alta |

# 3. Patrones

## Pattern 16: Stateless Serving Function

### 3.1 Problema

Invocar `predict()` sobre un objeto entrenable en memoria obliga a cargar el modelo completo, limita concurrencia y latencia, acopla el cliente al framework y expone interfaces pensadas para training, como logits o registros binarios. Además, el estado mutable hace costoso compartir componentes entre clientes y dificulta el escalado.

### 3.2 Contexto de aplicación

Modelos que necesitan predicción síncrona y de baja latencia, múltiples lenguajes cliente, autoscaling o una interfaz estable distinta de la representación de entrenamiento. También aplica a modelos entrenados en sistemas orientados a consultas largas que deben ofrecer inferencia online.

### 3.3 Señales que indican que el patrón puede ser necesario

- El servicio carga el modelo o sus activos en cada petición.
- Los clientes deben conocer TensorFlow, Keras u otro framework de entrenamiento.
- La salida natural del modelo no es apropiada para consumidores, por ejemplo logits en vez de probabilidades.
- Se requieren miles o millones de solicitudes síncronas por segundo.
- **Inferencia práctica:** cambios de preprocesamiento o postprocesamiento fuerzan cambios coordinados en muchos clientes.

### 3.4 Solución propuesta

1. Exportar el núcleo matemático del modelo en un formato independiente del lenguaje, como SavedModel u ONNX.
2. Restaurar en producción la función forward como una función sin estado, con pesos constantes y una firma explícita.
3. Incluir en la firma el preprocesamiento y postprocesamiento sin estado que deban ser consistentes.
4. Desplegar la función en un framework que proporcione un endpoint REST y autoscaling.
5. Cargar la función una sola vez por proceso o durante el cold start, no en cada petición.

### 3.5 Cómo funciona

El artefacto exportado contiene la fórmula, pesos, vocabularios y firma de entrada/salida necesarios para inferencia, pero excluye estado exclusivo del entrenamiento, como época o learning rate. El servidor comparte instancias de esa función entre solicitudes; el cliente transfiere en cada llamada todo el contexto necesario. Una firma personalizada puede transformar entradas o convertir logits a probabilidades, y varias firmas pueden exponer cálculos con distinto coste.

### 3.6 Por qué funciona

Una función cuyo resultado depende solo de sus entradas puede replicarse y repartirse entre servidores sin sincronizar estado conversacional. Esto permite reutilizar décadas de ingeniería de servicios web: balanceo, autoscaling, monitorización, throttling y APIs neutrales al lenguaje. La firma exportada desacopla el ritmo de evolución del modelo del de las aplicaciones consumidoras.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Solicitud autocontenida, normalmente JSON, conforme a una firma tipada |
| Transformación | Preprocesamiento sin estado, forward pass y postprocesamiento sin estado |
| Salida | Predicción con nombres y tipos estables, por ejemplo logits y/o probabilidad |

### 3.8 Procedimiento operativo

1. Definir qué información necesita realmente la inferencia y mover el contexto de secuencia a la entrada si es necesario.
2. Exportar el modelo y sus activos en un formato portable.
3. Inspeccionar y probar la firma de serving.
4. Crear una firma personalizada si el contrato externo requiere pre/postprocesamiento.
5. Desplegarla en un servidor de modelos o framework web administrado.
6. Probar solicitudes individuales y por lotes pequeños.
7. Ajustar batching, aceleradores, threads y autoscaling según latencia, throughput y coste.
8. Monitorizar el endpoint con herramientas operacionales estándar.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Firma única o múltiples firmas | Evita cálculos caros para clientes que no los requieren | Contrato de API más complejo |
| Tamaño del batch por solicitud | Más throughput y menos llamadas | Más latencia por petición y posibles hot spots |
| CPU, GPU, TPU y threads | Cambia latencia y capacidad | Mayor coste o recursos ociosos |
| Contexto enviado por el cliente | Mantiene el servidor sin estado | Más complejidad y ancho de banda del cliente |
| REST o librería embebida | Centralización frente a ejecución local | Red frente a dificultad de actualización |

### 3.10 Ventajas

- Escalado horizontal y baja latencia mediante infraestructura web madura.
- Contrato neutral al lenguaje y al framework de entrenamiento.
- Centraliza pre/postprocesamiento y simplifica clientes.
- Permite monitorización, control de cuota y monetización con herramientas existentes.
- Facilita cambiar el modelo detrás de un endpoint estable.

### 3.11 Limitaciones

- Los modelos inherentemente secuenciales requieren que el cliente transporte el contexto completo.
- La capa de exportación y serving añade indirección y un contrato que debe mantenerse.
- Un endpoint síncrono no es la opción económica para inferencias masivas no urgentes.
- El escalado de una librería embebida queda limitado por el dispositivo cliente.

### 3.12 Riesgos y trade-offs

- Mover estado al cliente aumenta complejidad y puede exigir reenviar secuencias.
- Calcular todas las salidas para todos los clientes desperdicia cómputo; varias firmas reducen ese coste a cambio de más superficie de API.
- Batches grandes aumentan throughput, pero también latencia y riesgo de hot spots.
- Una librería evita red y traslada coste al cliente, pero complica actualizaciones y restringe lenguajes.
- **Inferencia práctica:** preprocesamiento fuera de la firma puede producir training-serving skew.

### 3.13 Cuándo utilizarlo

Cuando se necesita inferencia online, síncrona, escalable y accesible desde distintos lenguajes; cuando conviene encapsular el contrato de entrada/salida y el postprocesamiento; o cuando un modelo entrenado en un sistema batch debe ofrecer baja latencia.

### 3.14 Cuándo no utilizarlo

No es la mejor opción para millones de predicciones periódicas sin urgencia, donde Batch Serving es más económico. Una librería local puede ser preferible sin conectividad, con restricciones de red estrictas o cuando el cómputo debe recaer en el cliente.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Librería de predicción embebida | Sin llamada de red; coste en el cliente | Actualización difícil, lenguajes limitados y escalabilidad finita | Sin conectividad o con latencia de red inaceptable |
| Batch Serving | Alto throughput y menor coste para grandes volúmenes | Resultado asíncrono | Predicciones masivas no sensibles a latencia |
| Múltiples firmas | Evita postprocesamiento caro innecesario | Mayor complejidad contractual | Pocos clientes requieren una salida costosa |

### 3.16 Implementación práctica

```python
@tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
def serve(text):
    logits = model(text, training=False)
    return {"logits": logits, "probability": tf.sigmoid(logits)}

model.save(export_path, signatures={"serving_default": serve})
```

Idea esencial del capítulo: exportar una función de inferencia con firma estable y postprocesamiento incluido. **Nota práctica de actualización:** los nombres concretos de servicios y CLI del libro corresponden a 2020; la lógica del patrón no depende de ellos.

### 3.17 Ejemplo mínimo conceptual

Clasificador que entrega logits → firma de serving aplica sigmoid y expone JSON → cualquier cliente recibe una probabilidad interpretable sin cargar Keras ni duplicar postprocesamiento.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En un coding test orientado a producción, conviene demostrar una función de inferencia determinista, un contrato tipado y carga única del artefacto. En un datathon, este patrón rara vez mejora el leaderboard, pero evita que el notebook sea el único modo de inferencia y reduce inconsistencias entre validación y demo. Bajo presión, priorizar una única firma que encapsule exactamente las transformaciones indispensables.

### 3.19 Checklist operativo del patrón

- [ ] Separar estado de entrenamiento del artefacto de inferencia.
- [ ] Definir nombres, tipos y formas de entrada/salida.
- [ ] Incluir pre/postprocesamiento sin estado necesario.
- [ ] Evitar cargar el modelo en cada solicitud.
- [ ] Probar concurrencia, batching, latencia y cold starts.
- [ ] Verificar que el cliente envía todo el contexto requerido.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Batch Serving | Complementario | Forma una arquitectura Lambda que intercambia latencia por throughput |
| Continued Model Evaluation | Dependencia | La firma puede exponer y registrar entradas, salidas y scores evaluables |
| Keyed Predictions | Puede combinarse | La firma puede aceptar y devolver una clave sin usarla como feature |
| Two-Phase Predictions | Puede combinarse | El modelo cloud de la segunda fase puede servirse como función sin estado |

## Pattern 17: Batch Serving

### 3.1 Problema

Un endpoint online diseñado para pocas instancias y respuesta inmediata se satura o resulta caro si recibe millones o miles de millones de predicciones periódicas. En casos como reposición horaria o playlists diarias, la latencia por instancia no es el objetivo; lo importante es completar un volumen enorme.

### 3.2 Contexto de aplicación

Inferencia asíncrona sobre datos estructurados o no estructurados, trabajos ocasionales o periódicos, generación masiva de recomendaciones y pipelines con pre/postprocesamiento. El cálculo debe poder particionarse porque la predicción de una instancia no depende de la de otra.

### 3.3 Señales que indican que el patrón puede ser necesario

- Millones de registros requieren predicción al mismo tiempo o según una agenda.
- El consumidor puede esperar a que termine un job y leer un archivo o tabla.
- El coste de mantener capacidad online para picos es desproporcionado.
- Se necesita procesamiento stateful, por ventanas o sobre audio, imagen o video.
- El espacio de resultados puede precalcularse y refrescarse periódicamente.

### 3.4 Solución propuesta

1. Ejecutar el modelo dentro de infraestructura distribuida como MapReduce, Spark, BigQuery o Beam.
2. Dividir los datos en shards independientes y procesarlos en paralelo.
3. Combinar y escribir los resultados en una ubicación especificada.
4. Usar un pipeline si el pre/postprocesamiento no cabe bien en SQL o requiere estado.
5. Si se necesita baja latencia de lectura, precalcular resultados y servirlos desde una base de datos, refrescándolos en segundo plano.

### 3.5 Cómo funciona

El sistema distribuye grupos de instancias entre workers, cada uno restaura o invoca el modelo y produce predicciones parciales. Operaciones posteriores, como ordenar y obtener un top-k, también se jerarquizan entre shards antes de combinar el resultado. Para recomendaciones, el batch llena anticipadamente el espacio usuario-item y la capa online solo recupera filas ya calculadas.

### 3.6 Por qué funciona

La inferencia independiente por instancia es un problema embarazosamente paralelo. Al no exigir respuesta inmediata, el sistema optimiza utilización y throughput, reparte el trabajo y evita dimensionar un microservicio online para cargas masivas esporádicas.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Tabla, archivos o stream con gran número de instancias |
| Transformación | Sharding, inferencia paralela y agregación/postprocesamiento |
| Salida | Tabla, archivo o caché de predicciones, disponible de forma asíncrona |

### 3.8 Procedimiento operativo

1. Confirmar que la tarea tolera asincronía y es particionable.
2. Elegir SQL para datos estructurados y lógica simple, o Spark/Beam para datos no estructurados, estado o transformaciones complejas.
3. Hacer accesible el artefacto del modelo a los workers.
4. Particionar entradas e incluir claves para recomponer resultados.
5. Ejecutar inferencia y postprocesamiento distribuido.
6. Escribir resultados durables.
7. Si se sirven online, cargar la caché o base de datos y definir la frecuencia de refresco.
8. Medir throughput, coste, frescura y fallos parciales.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Número/tamaño de shards | Paralelismo y utilización | Overhead o workers desbalanceados |
| SQL frente a pipeline general | Simplicidad frente a expresividad | Herramienta inadecuada para datos no estructurados o estado |
| Frecuencia de refresco | Frescura de resultados | Mayor coste de cómputo |
| Cohorte a recalcular | Reduce trabajo incremental | Resultados obsoletos para usuarios omitidos |
| Batch, online o Lambda | Posición en el eje latencia-throughput | Dos sistemas aumentan complejidad operacional |

### 3.10 Ventajas

- Escala horizontal para volúmenes masivos.
- Mejor coste que serving online cuando la latencia no importa.
- Integra inferencia con filtros, joins, agregaciones y top-k.
- Permite precalcular resultados de lectura inmediata.

### 3.11 Limitaciones

- Requiere que el trabajo sea divisible entre workers.
- Los resultados no están disponibles de inmediato.
- La frescura depende de la agenda de ejecución.
- El orden de salida puede no coincidir con el de entrada.

### 3.12 Riesgos y trade-offs

- Más throughput implica aceptar latencia del job y resultados potencialmente obsoletos.
- Cachés masivas requieren almacenamiento, actualización y coherencia.
- Pipelines stateful son más complejos que una consulta SQL.
- Una arquitectura Lambda cubre ambos extremos, pero duplica infraestructura.
- Sin claves, el procesamiento distribuido puede perder la asociación entrada-salida.

### 3.13 Cuándo utilizarlo

Predicciones periódicas o eventuales a gran escala, scoring de tablas completas, recomendaciones precalculadas, medios no estructurados o lógica stateful sin requisito de respuesta inmediata.

### 3.14 Cuándo no utilizarlo

No usar como sustituto de un endpoint cuando cada evento necesita respuesta inmediata. Tampoco cuando la tarea no puede particionarse con seguridad o el resultado depende del orden global sin una estrategia distribuida adecuada.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Stateless Serving Function | Baja latencia online | Caro para picos masivos | Predicción interactiva |
| Resultados batch en caché | Lectura online muy rápida | Menor frescura | Espacio de entradas limitado o precalculable |
| Lambda architecture | Combina latencia y throughput | Dos rutas operacionales | Se requieren ambos modos |
| Pipeline de stream | Estado y ventanas temporales | Mayor complejidad | Features o reglas dependen de eventos recientes |

### 3.16 Implementación práctica

```sql
SELECT *
FROM ML.PREDICT(MODEL proyecto.modelo,
  (SELECT feature_1, feature_2 FROM proyecto.tabla_pendiente));
```

Idea mínima: acercar el modelo a un motor distribuido y dejar que particione la tabla. **Nota práctica de actualización:** las APIs cloud concretas del capítulo pueden haber cambiado.

### 3.17 Ejemplo mínimo conceptual

Diez millones de usuarios requieren cinco recomendaciones → un job calcula scores por shards y guarda el top-5 por usuario → la aplicación consulta una fila en baja latencia y el batch la refresca periódicamente.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es útil cuando el test set no cabe cómodamente en memoria o la generación de features/predicciones puede particionarse. En un datathon, priorizarlo solo si el volumen bloquea el baseline o si conviene precalcular embeddings/predicciones; su valor es capacidad y tiempo total, no una mejora directa de métrica. Una implementación distribuida excesiva puede consumir el tiempo disponible.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar que no se exige respuesta inmediata.
- [ ] Verificar independencia o particionabilidad de instancias.
- [ ] Elegir SQL, batch o stream según datos y estado.
- [ ] Incorporar claves estables.
- [ ] Definir salida durable y política de refresco.
- [ ] Comparar coste y frescura con serving online.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Stateless Serving Function | Complementario | Juntos forman una arquitectura Lambda |
| Keyed Predictions | Dependencia | Permite recomponer salidas desordenadas |
| Continued Model Evaluation | Puede combinarse | El batch puede evaluar muestras y métricas periódicamente |
| Two-Phase Predictions | Puede combinarse | La segunda fase puede ejecutarse en batch si no es urgente |

## Pattern 18: Continued Model Evaluation

### 3.1 Problema

Un buen resultado en un test histórico no garantiza desempeño futuro. Concept drift cambia la relación entre entradas y target; data drift cambia esquema, distribuciones o significado de los datos. Sin evaluación posterior al despliegue no se sabe cuándo el modelo deja de ser útil ni cuándo justifica el coste de reentrenarlo.

### 3.2 Contexto de aplicación

Modelos en producción, especialmente en dominios dinámicos o adversariales como fraude, spam, trading, publicidad o ciberseguridad. Requiere capturar solicitudes, predicciones y, cuando esté disponible, ground truth; este último puede llegar con retraso.

### 3.3 Señales que indican que el patrón puede ser necesario

- Caen precision, recall, F1 u otra métrica de producción.
- Aparecen valores inesperados, features ausentes o cambios de esquema/distribución.
- El comportamiento del adversario, producto o población cambia.
- Se desconoce la frecuencia de reentrenamiento adecuada.
- **Inferencia práctica:** alta confianza del modelo coexiste con más correcciones humanas.

### 3.4 Solución propuesta

1. Muestrear y guardar inputs, outputs, score, versión y timestamp sin degradar serving.
2. Capturar ground truth mediante labels posteriores, interacción, correcciones o anotación humana.
3. Unir predicción y verdad cuando esta esté disponible.
4. Calcular las mismas métricas utilizadas durante desarrollo, por tiempo, clase, versión o segmento.
5. Comparar versiones en el entorno actual.
6. Usar umbrales o agenda para activar reentrenamiento.
7. Desplegar la nueva versión solo si supera a la actual sobre datos recientes.
8. Complementar la evaluación post hoc con validación de distribuciones de entrada.

### 3.5 Cómo funciona

Un log de evaluación conserva la solicitud cruda, la predicción, su score, versión y hora. Al llegar la verdad, se completa el registro y se computan métricas y matrices de confusión sobre ventanas recientes. Los resultados revelan deterioro global o por slice y pueden disparar un pipeline reproducible de reentrenamiento y comparación.

### 3.6 Por qué funciona

Sustituye la suposición estática de que futuro y pasado comparten distribución por evidencia cuantitativa sobre datos nuevos. Evaluar exclusivamente tráfico reciente permite detectar staleness, estimar la tasa de degradación y decidir retraining con relación explícita entre calidad, coste y frescura.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Muestra de requests, predicciones, scores, versión, timestamp y ground truth tardío |
| Transformación | Join, slicing temporal/segmentado y cálculo de métricas |
| Salida | Tendencias de calidad, alertas, comparación de versiones y señal de retraining |

### 3.8 Procedimiento operativo

1. Seleccionar métricas ligadas al caso de negocio.
2. Definir tasa de muestreo que preserve rendimiento del servicio.
3. Registrar versión y tiempo junto a input/output.
4. Diseñar una fuente válida de ground truth y su clave de unión.
5. Revisar si la intervención causada por la predicción invalida esa verdad observada.
6. Programar evaluación por ventanas y slices.
7. Acordar umbrales absolutos o de tendencia y validar cada trigger.
8. Reentrenar mediante pipeline controlado.
9. Comparar candidato y versión actual con un test de datos recientes.
10. Monitorizar además anomalías, drift y training-serving skew en inputs.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Porcentaje de muestreo | Cobertura frente a carga de serving | Muestra insuficiente o coste excesivo |
| Ventana temporal | Sensibilidad al cambio | Ruido en ventanas cortas o demora en largas |
| Umbral de retraining | Frescura del modelo | Reentrenamiento caro o modelo obsoleto |
| Métrica y slices | Alineación con el problema | Promedios pueden ocultar fallos locales |
| Agenda de retraining | Previsibilidad operacional | No coincide con la velocidad real del drift |
| Fuente de ground truth | Validez de evaluación | Feedback sesgado o contaminado por intervención |

### 3.10 Ventajas

- Detecta staleness y degradación sobre datos reales nuevos.
- Cuantifica cuándo reentrenar o reemplazar una versión.
- Permite A/B testing y comparación por versión, tiempo o clase.
- Hace observables data drift, concept drift y training-serving skew.

### 3.11 Limitaciones

- Ground truth puede demorarse meses o no ser observable directamente.
- Muestrear todo puede afectar serving; muestrear poco reduce certeza.
- El patrón detecta deterioro, pero no elimina el coste de pipelines, labeling y retraining.
- Una intervención basada en la predicción puede destruir el contrafactual necesario para evaluarla.

### 3.12 Riesgos y trade-offs

- Umbral sensible mantiene calidad pero eleva coste y cambios de versión; uno laxo reduce coste pero tolera staleness.
- Feedback de usuario puede crear bucles y sesgo de selección.
- Reentrenamiento automático sin registrar y validar triggers dificulta debugging y control.
- Métricas agregadas pueden ocultar degradación en segmentos.
- Usar como ground truth el resultado intervenido produce una evaluación inválida.

### 3.13 Cuándo utilizarlo

Para cualquier modelo cuyo entorno, población o relación entrada-target pueda cambiar, y especialmente cuando el error productivo tiene coste relevante o se automatizará el retraining.

### 3.14 Cuándo no utilizarlo

No debe automatizarse un ciclo de retraining si no existe ground truth confiable, métricas acordadas o un pipeline validado. En dominios muy estáticos puede bastar una cadencia menos frecuente, pero el capítulo no propone omitir completamente la observación.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Retraining programado | Simple y predecible | Puede entrenar demasiado o demasiado tarde | Labels llegan en ciclos naturales |
| Trigger por umbral | Responde al deterioro observado | Sensible al ruido y exige control | Métrica y coste están bien definidos |
| Validación de datos | Detecta anomalías antes de obtener labels | No mide directamente calidad predictiva | Ground truth tarda o se busca alerta temprana |
| Simulación offline con datos antiguos/recientes | Estima intervalo de degradación a bajo coste | Aproximación histórica | Diseñar inicialmente la cadencia |
| Online learning | Actualización muy rápida | Mayor complejidad y riesgo operacional | Decisiones continuas con flujo abundante de labels |

### 3.16 Implementación práctica

```python
recent = evaluations[evaluations["time"] >= window_start]
score = metric(recent["ground_truth"], recent["prediction"])
if score < agreed_threshold:
    trigger_retraining_pipeline()
```

La lógica mínima exige versión, tiempo, predicción y verdad alineadas. No basta vigilar confianza o distribución sin medir el resultado cuando el label esté disponible.

### 3.17 Ejemplo mínimo conceptual

Clasificador de titulares desplegado → se muestrean input, clase, confianza y versión → al obtener la fuente real se calcula F1 por semana y clase → una caída sostenida activa reentrenamiento con datos recientes y comparación contra la versión vigente.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En un datathon tradicional no hay producción continua, pero el patrón inspira validación temporal: entrenar con datos viejos y medir en datos recientes revela drift y evita un split engañoso. En un coding test de MLOps, son prioritarios logging con versión/timestamp, delayed labels, slices, umbrales y control del trigger. No usar el leaderboard como sustituto de ground truth productivo.

### 3.19 Checklist operativo del patrón

- [ ] Elegir métrica y slices relevantes.
- [ ] Registrar muestra de inputs, outputs, score, versión y tiempo.
- [ ] Diseñar unión con ground truth tardío.
- [ ] Revisar interferencia de las acciones del modelo.
- [ ] Definir ventanas, umbrales y cadencia.
- [ ] Validar y auditar triggers de retraining.
- [ ] Comparar candidato y producción sobre datos actuales.
- [ ] Monitorizar drift y training-serving skew.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Stateless Serving Function | Complementario | El endpoint genera los eventos de predicción observados |
| Batch Serving | Puede combinarse | Evaluación y agregación periódicas pueden ejecutarse en batch |
| Keyed Predictions | Dependencia | La clave permite unir predicción, metadata y ground truth |
| Two-Phase Predictions | Complementario | Los modelos offline también necesitan evaluación o una réplica online |
| Workflow Pipeline | Dependencia | El capítulo lo señala como mecanismo para automatizar retraining y despliegue |

## Pattern 19: Two-Phase Predictions

### 3.1 Problema

Un modelo complejo en el edge compite por almacenamiento, CPU, energía y latencia, puede perder accuracy al cuantizarse y es difícil de actualizar sin conectividad. Un modelo solo en cloud, en cambio, depende de una red potencialmente lenta, cara o inexistente.

### 3.2 Contexto de aplicación

Móviles, sensores, dispositivos embebidos y aplicaciones offline donde el problema puede dividirse en una decisión simple, frecuente y local seguida de una decisión compleja, menos frecuente y remota. Ejemplos del capítulo: wake word y comprensión de consulta, anomalía acústica y diagnóstico, detección de animal y clasificación de especie.

### 3.3 Señales que indican que el patrón puede ser necesario

- El dispositivo no tiene conectividad confiable o la llamada cloud es cara.
- El modelo debe responder localmente con baja latencia.
- Existe un filtro binario o detección de candidatos antes de una tarea rica.
- La cuantización o reducción del modelo completo perjudica demasiado la calidad.
- Solo una fracción de eventos necesita el modelo complejo.

### 3.4 Solución propuesta

1. Dividir el problema en dos tareas con distinta complejidad.
2. Entrenar un modelo pequeño y barato para la primera fase en el dispositivo.
3. Optimizarlo o cuantizarlo para restricciones de edge.
4. Usarlo como filtro o activador.
5. Enviar solo candidatos al modelo complejo hospedado en cloud.
6. Almacenar solicitudes localmente cuando no haya red y procesarlas al recuperar conectividad.
7. Evaluar regularmente también la fase local.

### 3.5 Cómo funciona

La fase 1 ejecuta una clasificación simple y frecuente, por ejemplo “instrumento/no instrumento”. Solo una salida positiva activa la fase 2, que recibe el ejemplo y ejecuta una arquitectura mayor para clasificar entre muchas categorías. La primera fase reduce llamadas, latencia y ancho de banda; la segunda conserva sofisticación donde aporta valor.

### 3.6 Por qué funciona

Se asigna cada parte del problema al entorno que mejor soporta sus restricciones. El edge aporta disponibilidad y respuesta rápida; el cloud aporta cómputo, actualización y modelos complejos. Al filtrar localmente, el sistema reserva el coste remoto para una fracción informativa de los eventos.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Evento crudo capturado por el dispositivo |
| Transformación | Fase 1 local filtra; fase 2 cloud refina o completa |
| Salida | Respuesta básica local o predicción compleja remota |

### 3.8 Procedimiento operativo

1. Verificar que la tarea admite una frontera lógica entre fases.
2. Definir el objetivo binario/simple de fase 1 y el criterio que activa fase 2.
3. Entrenar una arquitectura compatible con edge.
4. Medir tamaño, latencia y pérdida de accuracy al cuantizar.
5. Elegir métrica de fase 1 según el coste de falsos positivos y falsos negativos.
6. Entrenar y desplegar el modelo complejo en cloud.
7. Implementar cola o caché local para interrupciones de red.
8. Medir calidad end-to-end, tasa de activación y coste de llamadas.
9. Diseñar actualización y evaluación del modelo local.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Umbral de activación | Controla tráfico a cloud | Omitir casos útiles o enviar demasiados |
| Precision frente a recall en fase 1 | Cambia falsos positivos/negativos | Optimizar el error equivocado para el coste real |
| Arquitectura y cuantización | Reduce tamaño y latencia | Pérdida de accuracy |
| Tarea de cada fase | Determina separabilidad | Frontera artificial que degrada el sistema |
| Política offline | Mejora disponibilidad | Predicciones tardías y almacenamiento local |

### 3.10 Ventajas

- Inferencia básica con baja latencia y sin red.
- Menos llamadas, ancho de banda y coste cloud.
- Mantiene un modelo sofisticado para casos que lo necesitan.
- Permite que cada fase tenga una arquitectura y objetivo adecuados.

### 3.11 Limitaciones

- Solo aplica si el problema puede dividirse coherentemente.
- El error de fase 1 puede impedir que fase 2 vea un caso.
- Hay que versionar, actualizar y observar dos modelos.
- La cuantización y simplificación local pueden reducir accuracy.

### 3.12 Riesgos y trade-offs

- Tamaño, complejidad, frecuencia de actualización, accuracy y latencia se negocian entre sí.
- Una fase 1 demasiado restrictiva reduce coste pero pierde candidatos.
- Conectividad muy escasa puede volver inviable la segunda fase.
- Las predicciones que nunca llegan a cloud quedan fuera del monitoreo si no se diseña una muestra local.
- **Inferencia práctica:** diferencias de preprocesamiento entre dispositivo y cloud pueden introducir inconsistencias.

### 3.13 Cuándo utilizarlo

Cuando existe una detección local simple que puede filtrar o activar una tarea cloud más compleja, la conectividad no es constante y una sola versión reducida del modelo no cumple simultáneamente latencia y calidad.

### 3.14 Cuándo no utilizarlo

No usar si todos los casos requieren la predicción compleja, si la tarea no se separa o si el dispositivo casi nunca podrá contactar al cloud. En este último caso, el capítulo prefiere un modelo local autosuficiente aunque sea menos preciso.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Modelo local monofásico | Funciona totalmente offline | Menor capacidad/accuracy | Conectividad mínima o nula |
| Funciones offline limitadas | Menor tamaño y complejidad | Experiencia parcial | Solo algunos casos deben funcionar offline |
| Caché de predicciones/consultas | Reutiliza resultados y difiere llamadas | Menor frescura | Consultas repetibles o reconexión posterior |
| Filtro local del mismo objetivo | Reduce miles de llamadas | Añade doble evaluación | Muchos eventos, pocos candidatos relevantes |
| Réplica online del modelo edge | Facilita evaluación continua | Cómputo adicional | El modelo local no puede transmitir todo su tráfico |

### 3.16 Implementación práctica

```python
local_score = edge_model.predict(event)
if local_score >= activation_threshold:
    result = cloud_model.predict(event)
else:
    result = {"candidate": False}
```

El capítulo demuestra exportar el modelo local a TensorFlow Lite y cuantizar pesos. **Nota práctica de actualización:** las APIs y servicios cloud concretos son históricos; conservar la separación edge/cloud y validar el formato actual.

### 3.17 Ejemplo mínimo conceptual

Audio continuo → modelo local decide si contiene un instrumento → solo los positivos se envían al cloud → modelo grande identifica uno de 18 instrumentos.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En un datathon estándar rara vez es prioritario porque el leaderboard no mide deployment, pero puede inspirar una cascada: filtro barato sobre todos los casos y modelo costoso solo sobre candidatos. En coding tests de edge/MLOps, explicar el umbral, el error propagado, la cuantización y el monitoreo de casos no enviados es más importante que implementar dos modelos completos bajo presión.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar que el problema se divide en dos tareas útiles.
- [ ] Definir coste de falsos positivos y negativos en fase 1.
- [ ] Medir tamaño, latencia y accuracy del modelo edge.
- [ ] Establecer umbral y tasa esperada de activación.
- [ ] Diseñar comportamiento sin conectividad.
- [ ] Versionar y evaluar ambas fases.
- [ ] Vigilar calidad end-to-end, no solo cada modelo aislado.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Stateless Serving Function | Complementario | La fase cloud puede exponerse con este patrón |
| Batch Serving | Puede combinarse | Eventos guardados offline pueden enviarse y procesarse juntos |
| Continued Model Evaluation | Dependencia | La fase edge también sufre drift y requiere evaluación |
| Keyed Predictions | Puede combinarse | Identifica eventos diferidos entre dispositivo y cloud |

## Pattern 20: Keyed Predictions

### 3.1 Problema

En procesamiento distribuido o batching, las salidas llegan desordenadas. Forzar orden global impide paralelismo y resulta computacionalmente caro; devolver resultados sin identificación impide al cliente saber qué predicción corresponde a qué entrada. Batches online grandes también crean hot spots.

### 3.2 Contexto de aplicación

Batch serving, arrays de instancias online, servidores con múltiples clientes, acumulación asíncrona para GPU/TPU y evaluación continua que necesita unir predicciones con labels o metadata. La clave no es una feature del modelo.

### 3.3 Señales que indican que el patrón puede ser necesario

- La entrada y salida son archivos con muchas instancias.
- Workers paralelos alteran el orden original.
- Se mezclan solicitudes de varios clientes para mejorar utilización.
- Se necesita unir una predicción posterior con ground truth o segmentos.
- El servidor exige ordenar o limitar batches para evitar hot spots.

### 3.4 Solución propuesta

1. Hacer que el cliente asigne una clave única a cada instancia.
2. Enviar `(key, features)` al serving system.
3. Separar la clave antes de invocar el modelo para que no influya en la predicción.
4. Adjuntar la misma clave a la salida `(key, prediction)`.
5. Permitir procesamiento y retorno fuera de orden.
6. Diseñar claves sin colisiones entre clientes y, cuando sea útil, con metadata de evaluación.

### 3.5 Cómo funciona

La firma de serving acepta features más `key`; copia las features, elimina la clave, llama al modelo y devuelve un diccionario con clave y predicción. Pueden conservarse dos firmas: la predeterminada compatible sin clave y otra para keyed prediction. El cliente hace join por clave en vez de confiar en posición.

### 3.6 Por qué funciona

La identidad viaja con cada elemento a través de shards, colas y microbatches. Así desaparece la necesidad de serializar todo el trabajo o realizar un ordenamiento global, preservando paralelismo, batching eficiente y trazabilidad.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Clave única más features del modelo |
| Transformación | La clave pasa alrededor del modelo; solo las features alimentan la inferencia |
| Salida | Misma clave más predicción |

### 3.8 Procedimiento operativo

1. Definir el ámbito de unicidad, incluido identificador de cliente si hay multiplexación.
2. Generar la clave en el cliente, donde puede asociarse al registro original.
3. Añadirla a la firma sin incorporarla a las features.
4. Devolverla sin modificación junto con la predicción.
5. Probar reordenamiento, duplicados, colisiones y retries.
6. Hacer join de salidas con entradas o ground truth por clave.
7. Mantener, si es necesario, una firma anterior sin clave para compatibilidad.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Estructura de clave | Trazabilidad y slicing | Colisiones, exposición de datos o claves voluminosas |
| Ámbito por cliente | Evita colisiones al mezclar requests | Convención más compleja |
| Firma nueva o default | Compatibilidad | Clientes invocan la firma equivocada |
| Metadata embebida | Facilita evaluación segmentada | Privacidad y acoplamiento semántico |
| Tamaño de microbatch/latencia máxima | Eficiencia de acelerador | Mayor tiempo de espera |

### 3.10 Ventajas

- Mantiene correspondencia exacta sin preservar orden.
- Habilita paralelismo distribuido y microbatching eficiente.
- Facilita joins con ground truth y análisis por segmentos.
- Reduce necesidad de ordenar resultados en el servidor.

### 3.11 Limitaciones

- Exige modificar contrato y clientes.
- Las claves deben ser únicas en un sistema multiusuario.
- Metadata sensible dentro de la clave puede crear riesgos de privacidad.
- Añade bytes y lógica de join.

### 3.12 Riesgos y trade-offs

- Claves generadas solo por el servidor no resuelven batch si no pueden unirse de vuelta a la entrada.
- Ordenar en el servidor conserva una interfaz posicional, pero es caro a escala.
- Devolver todas las features permite matching, pero consume más ancho de banda y cómputo cliente.
- Retries sin una política de idempotencia pueden duplicar claves/resultados.
- Segmentos en la clave ayudan a fairness/evaluación, pero no deben convertirse accidentalmente en features.

### 3.13 Cuándo utilizarlo

Siempre que las predicciones puedan procesarse fuera de orden, mezclarse entre clientes, ejecutarse asincrónicamente o deban enlazarse después con el registro original, metadata o ground truth.

### 3.14 Cuándo no utilizarlo

En una llamada estrictamente individual, síncrona y sin joins posteriores puede ser innecesario. No incrustar atributos sensibles en texto claro si no existe una política apropiada de privacidad y acceso.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Preservar/ordenar resultados | Cliente simple | Ordenamiento caro y menor escalabilidad | Volumen pequeño y orden contractual imprescindible |
| ID asignado por servidor | Útil para una solicitud online | No enlaza por sí solo al registro batch original | Trazabilidad de request simple |
| Devolver todas las entradas | Matching directo | Más ancho de banda y cómputo | Entradas pequeñas y contrato ya tolera eco |
| Posición de array | Cero campo adicional | Frágil ante paralelismo y retries | Procesamiento local estrictamente ordenado |

### 3.16 Implementación práctica

```python
@tf.function(input_signature=[{
    "key": tf.TensorSpec([None], tf.string),
    "x": tf.TensorSpec([None], tf.float32),
}])
def keyed_prediction(inputs):
    features = inputs.copy()
    key = features.pop("key")
    return {"key": key, "prediction": model(features)}
```

La clave pasa por la función, pero no por el modelo.

### 3.17 Ejemplo mínimo conceptual

Archivo con un millón de filas → cada fila recibe un ID estable → workers producen resultados desordenados → el cliente hace join por ID y recupera la predicción correcta sin ordenar globalmente.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es una defensa barata contra submissions desalineadas: conservar el identificador original durante particionado, paralelización y recombinación. En coding tests, demuestra comprensión de sistemas distribuidos si la clave queda fuera de features y se validan unicidad y orden final. No mejora la métrica, pero evita una entrega inválida por correspondencia rota.

### 3.19 Checklist operativo del patrón

- [ ] Generar claves estables y únicas en el cliente.
- [ ] Excluir la clave de las features del modelo.
- [ ] Devolverla sin modificación.
- [ ] Probar reordenamiento, retries, duplicados y colisiones.
- [ ] Hacer join por clave, no por posición.
- [ ] Revisar privacidad de metadata embebida.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Batch Serving | Dependencia | Recompone resultados producidos por workers desordenados |
| Stateless Serving Function | Puede combinarse | La firma exportada puede pasar la clave |
| Continued Model Evaluation | Dependencia | Une predicción, contexto y ground truth |
| Two-Phase Predictions | Puede combinarse | Identifica eventos almacenados y enviados posteriormente |

# 4. Reglas de decisión del capítulo

| Situación | Pattern recomendado | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Respuesta interactiva y muchos clientes | Stateless Serving Function | Autoscaling y contrato neutral al lenguaje | Enviar todo el contexto requerido |
| Millones de predicciones no urgentes | Batch Serving | Paralelismo y mejor coste | Resultado asíncrono y frescura |
| Predicción masiva con lectura online rápida | Batch Serving con caché | Precalcula y sirve por lookup | Definir frecuencia de refresco |
| Calidad productiva puede degradarse | Continued Model Evaluation | Mide datos nuevos con ground truth | Evitar labels contaminados por intervención |
| Dispositivo restringido y problema divisible | Two-Phase Predictions | Edge filtra; cloud refina | Error de fase 1 se propaga |
| Salidas pueden llegar fuera de orden | Keyed Predictions | Join correcto sin ordenar globalmente | Claves únicas y no usadas como features |
| Se requieren online y batch | Stateless Serving Function + Batch Serving | Arquitectura Lambda equilibra latencia y throughput | Mayor complejidad operacional |

# 5. Comparación entre patrones del capítulo

| Pattern | Problema que resuelve | Coste | Beneficio | Complejidad |
| ------- | --------------------- | ----- | --------- | ----------- |
| Stateless Serving Function | Serving online escalable | Endpoint y operación continua | Baja latencia y desacoplamiento | Media |
| Batch Serving | Inferencia masiva | Job distribuido y almacenamiento | Alto throughput y menor coste unitario | Media |
| Continued Model Evaluation | Staleness y drift | Logging, labels, evaluación y retraining | Calidad observable en producción | Alta |
| Two-Phase Predictions | Restricciones edge/cloud | Dos modelos y dos rutas | Disponibilidad local con capacidad cloud | Alta |
| Keyed Predictions | Correspondencia en paralelismo | Campo y join adicionales | Escala sin orden global | Baja |

# 6. Riesgos globales del capítulo

- **Training-serving skew:** transformaciones fuera del artefacto o cambios de esquema pueden separar training y producción.
- **Data y concept drift:** el modelo estático pierde vigencia en un mundo dinámico.
- **Validación incorrecta:** feedback intervenido no representa el contrafactual y métricas agregadas ocultan slices.
- **Complejidad operacional:** Lambda architecture, retraining automatizado y dos fases multiplican componentes/versiones.
- **Distribución y orden:** batch y microbatch alteran orden; sin claves se pierde correspondencia.
- **Coste, latencia y throughput:** optimizar uno puede deteriorar los otros.
- **Pérdida de calidad en edge:** cuantización y simplificación reducen tamaño a posible coste de accuracy.
- **Reproducibilidad y mantenimiento:** triggers no auditados y firmas incompatibles dificultan explicar qué versión produjo una salida.
- **Fairness e interpretabilidad operacional:** el capítulo propone slicing por segmentos para localizar degradación; no desarrolla explicabilidad del modelo.

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

Identificar el modo de inferencia esperado: interactivo o masivo; confirmar si existe un ID estable; revisar si los datos son temporales, si habrá drift y si el entorno objetivo es edge. Estas decisiones activan, respectivamente, Stateless/Batch Serving, Keyed Predictions, Continued Model Evaluation y Two-Phase Predictions.

### Durante el baseline

Conservar IDs desde el primer pipeline y encapsular preprocesamiento más inferencia en una función determinista. Si el test es grande, predecir por batches o shards sin perder claves. Estas acciones reducen errores de entrega, aunque no elevan por sí mismas la métrica.

### Durante la mejora del modelo

Usar una cascada de dos fases solo si el modelo caro debe aplicarse a pocos candidatos. Simular drift con split temporal y evaluar por ventanas/slices si el dataset representa un sistema cambiante.

### Durante validación

Evitar que posición de filas sustituya a una clave. En problemas temporales, comparar entrenamiento antiguo con datos recientes. Mantener la misma transformación en evaluación e inferencia y separar score offline de métricas obtenibles con ground truth productivo.

### Bajo presión de tiempo

- **Alta prioridad:** Keyed Predictions; elección correcta entre inferencia online y batch; función de inferencia autocontenida.
- **Prioridad media:** simulación de Continued Model Evaluation mediante split temporal y slices.
- **Baja prioridad:** infraestructura Lambda completa o Two-Phase Predictions, salvo que sean parte explícita del reto.

# 8. Checklist global del capítulo

- [ ] Clasificar la carga por latencia, volumen y frecuencia.
- [ ] Exportar un contrato de inferencia sin estado cuando se requiera online serving.
- [ ] Usar procesamiento distribuido para predicción masiva asíncrona.
- [ ] Conservar claves únicas de extremo a extremo.
- [ ] Registrar versión, timestamp, inputs, outputs y scores necesarios.
- [ ] Diseñar una fuente válida de ground truth.
- [ ] Medir drift y calidad sobre datos recientes y slices.
- [ ] Evaluar tamaño, latencia y pérdida de calidad si existe edge.
- [ ] Comparar coste operacional con el beneficio de cada patrón.

# 9. Mapeo hacia `knowledge/`

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Stateless Serving Function | `knowledge/mlops/stateless_serving_function.md` | Alta |
| Batch Serving | `knowledge/mlops/batch_serving.md` | Alta |
| Lambda architecture y latencia-throughput | `knowledge/mlops/serving_architectures.md` | Media |
| Continued Model Evaluation | `knowledge/mlops/continued_model_evaluation.md` | Alta |
| Drift y evaluación temporal | `knowledge/validation/production_model_monitoring.md` | Alta |
| Two-Phase Predictions | `knowledge/mlops/two_phase_predictions.md` | Media |
| Edge quantization y serving | `knowledge/mlops/edge_inference.md` | Media |
| Keyed Predictions | `knowledge/mlops/keyed_predictions.md` | Alta |
| Slicing de evaluación | `knowledge/metrics/production_slicing.md` | Media |
| Reentrenamiento por trigger | `knowledge/mlops/retraining_triggers.md` | Alta |

# 10. Índice de patrones reutilizables

| Pattern | Activador principal | Acción principal | Knowledge destino |
| ------- | ------------------- | ---------------- | ----------------- |
| Stateless Serving Function | Serving síncrono escalable | Exportar función sin estado con firma estable | `knowledge/mlops/` |
| Batch Serving | Volumen masivo no urgente | Particionar e inferir en workers | `knowledge/mlops/` |
| Continued Model Evaluation | Riesgo de drift/staleness | Unir predicciones con verdad y monitorizar métricas | `knowledge/mlops/`, `knowledge/validation/` |
| Two-Phase Predictions | Edge restringido y tarea divisible | Filtrar localmente y refinar en cloud | `knowledge/mlops/` |
| Keyed Predictions | Resultados fuera de orden | Pasar una clave cliente entrada-salida | `knowledge/mlops/` |

# 11. Trazabilidad

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Propósito y mapa del capítulo | p. 201; resumen p. 248 |
| Stateless Serving Function: concepto y problema | pp. 201-204 |
| Stateless Serving Function: solución, exportación y endpoint | pp. 205-209 |
| Stateless Serving Function: firmas, alternativas y librería | pp. 209-213 |
| Batch Serving: problema y solución | pp. 213-215 |
| Batch Serving: paralelismo | pp. 215-217; Figuras 5-1 y 5-2 |
| Batch Serving: pipelines, caché y Lambda architecture | pp. 217-220 |
| Continued Model Evaluation: problema, drift y solución | pp. 220-223 |
| Continued Model Evaluation: captura de predicciones y ground truth | pp. 223-225; Tablas 5-1 y 5-2 |
| Continued Model Evaluation: métricas y justificación | pp. 225-227; Figuras 5-3 y 5-4 |
| Continued Model Evaluation: triggers y retraining programado | pp. 227-230; Figuras 5-5 y 5-6 |
| Continued Model Evaluation: validación de datos y estimación offline | pp. 230-232; Figuras 5-7 y 5-8 |
| Two-Phase Predictions: problema y solución | pp. 232-235; Figuras 5-9 y 5-10 |
| Two-Phase Predictions: modelo edge y cuantización | pp. 235-239; Figura 5-11 |
| Two-Phase Predictions: modelo cloud | pp. 239-241 |
| Two-Phase Predictions: alternativas y evaluación offline | pp. 241-243; Figura 5-12 |
| Keyed Predictions: problema y solución | pp. 243-245; Figura 5-13 |
| Keyed Predictions: firmas y modelo existente | pp. 245-247; Figura 5-14 |
| Keyed Predictions: alternativas, asincronía y evaluación | pp. 247-248 |
