---
title: "Reproducibility Design Patterns"
book: "Machine Learning Design Patterns"
chapter: "6"
topics:
  - "Reproducibility"
  - "Training-serving consistency"
  - "Data splitting"
  - "Schema evolution"
  - "Streaming inference"
  - "ML pipelines"
  - "Feature management"
  - "Model lifecycle"
patterns:
  - "Transform"
  - "Repeatable Splitting"
  - "Bridged Schema"
  - "Windowed Inference"
  - "Workflow Pipeline"
  - "Feature Store"
  - "Model Versioning"
target_knowledge_files:
  - "knowledge/preprocessing/transform.md"
  - "knowledge/validation/repeatable_splitting.md"
  - "knowledge/reproducibility/bridged_schema.md"
  - "knowledge/time_series/windowed_inference.md"
  - "knowledge/mlops/workflow_pipeline.md"
  - "knowledge/feature_engineering/feature_store.md"
  - "knowledge/mlops/model_versioning.md"
---

# 6. Reproducibility Design Patterns

## 1. Propósito del capítulo

El capítulo aborda la reproducibilidad más allá de fijar una semilla: cómo conservar las mismas transformaciones entre entrenamiento e inferencia, producir particiones estables y sin contaminación, adaptar datos históricos a nuevos esquemas, repetir cálculos temporales en producción, ejecutar flujos completos de ML de manera trazable, reutilizar features y actualizar modelos sin romper consumidores. Estos problemas aparecen desde la preparación de datos y la validación hasta serving, reentrenamiento y mantenimiento. Los patrones formalizan dependencias, estado, artefactos y contratos que de otro modo quedarían implícitos en notebooks, scripts o infraestructura ad hoc.

# 2. Mapa de patrones del capítulo

| Pattern | Problema principal | Área de ML | Prioridad práctica |
| ------- | ------------------ | ---------- | ------------------ |
| Transform | Evitar diferencias entre las transformaciones de training y serving | Feature engineering / Serving | Alta |
| Repeatable Splitting | Obtener splits estables que respeten correlaciones y eviten leakage | Validation / Reproducibility | Alta |
| Bridged Schema | Entrenar con pocos datos nuevos y abundantes datos de un esquema anterior | Data representation / Training | Media |
| Windowed Inference | Servir modelos o features que requieren contexto temporal o secuencial | Pipeline / Serving / Time series | Media |
| Workflow Pipeline | Reproducir, desacoplar y orquestar un proceso de ML de extremo a extremo | Pipeline / Reproducibility | Alta |
| Feature Store | Reutilizar features consistentes en training y serving | Feature engineering / Serving | Media |
| Model Versioning | Actualizar modelos preservando compatibilidad y capacidad de comparación | Serving / MLOps | Alta |

# 3. Patrones

## Pattern 21: Transform

### 3.1 Problema

Los inputs que recibe una aplicación no suelen ser las features numéricas que consume el modelo. Si la lógica que convierte inputs en features queda mezclada con la consulta de entrenamiento, en un notebook o en código distinto al de serving, el consumidor debe reconstruir detalles como escalas, vocabularios, embeddings, extracción de fechas o tamaño de imágenes. Cualquier diferencia causa *training-serving skew*. El ejemplo del libro muestra un modelo de bicicletas entrenado con `start_station_name`, `dayofweek` y `hourofday`, aunque la información original era `start_station_name` y `start_date`; servir el modelo exigía conocer cómo se derivaron las dos últimas features.

### 3.2 Contexto de aplicación

Aplica a datos tabulares, texto e imágenes; a transformaciones por instancia y a transformaciones que requieren estadísticas del dataset. Es especialmente importante cuando entrenamiento y serving se ejecutan en lenguajes, frameworks o sistemas distintos, o cuando el modelo se despliega mucho después de haber sido entrenado.

### 3.3 Señales que indican que el patrón puede ser necesario

- El endpoint exige features derivadas en vez de inputs naturales del dominio.
- Hay código de preprocesamiento duplicado en training y serving.
- Escalas, vocabularios, tablas de consulta o embeddings se guardan aparte sin vínculo inequívoco con el modelo.
- Las predicciones offline y online difieren para el mismo input.
- **Inferencia práctica:** cambiar una transformación obliga a coordinar manualmente varios repositorios o servicios.

### 3.4 Solución propuesta

1. Definir explícitamente los inputs crudos.
2. Separar la capa de transformación de la arquitectura predictiva.
3. Capturar tanto la lógica como los artefactos aprendidos por las transformaciones.
4. Incorporar esa transformación al grafo o artefacto servido, o mantener un mecanismo compartido que la reproduzca.
5. En transformaciones de dataset, ejecutar un pase escalable previo, conservar estadísticas/vocabularios y adjuntarlos al modelo para serving.

BigQuery ML materializa la idea mediante `TRANSFORM`: el `SELECT` conserva inputs y la cláusula de transformación produce las features que también se calcularán en `ML.PREDICT`. En Keras, el libro separa `Input`, transformaciones y `DenseFeatures`. Para estadísticas globales propone `tf.transform`.

### 3.5 Cómo funciona

El flujo es `input crudo → transformación versionada → feature → modelo → predicción`. Las transformaciones por instancia pueden vivir en capas del modelo. Las de dataset requieren analizar todos los datos para calcular media, varianza, fronteras de buckets o vocabularios; el pase de análisis genera datos transformados para training y un artefacto reutilizable por serving. Así, serving recibe el mismo input semántico y ejecuta exactamente la transformación asociada al modelo.

### 3.6 Por qué funciona

El patrón elimina una fuente de divergencia al convertir preprocesamiento y parámetros auxiliares en parte del contrato reproducible del modelo. La igualdad no depende de que dos equipos reimplementen la misma operación, sino de reutilizar la lógica y los artefactos que produjeron las features de entrenamiento.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Valores crudos disponibles también durante inferencia |
| Transformación | Lógica más artefactos: escalas, vocabularios, embeddings, lookup tables o reglas de imagen/texto |
| Salida | Features en la representación exacta esperada por el modelo |

### 3.8 Procedimiento operativo

1. Inventariar inputs crudos y features derivadas.
2. Mover toda derivación posterior al input a una capa de transformación explícita.
3. Clasificar cada operación como de instancia o de dataset.
4. Para operaciones de dataset, analizar únicamente training y persistir los artefactos resultantes.
5. Construir training con las features transformadas.
6. Exportar transformación y modelo como una unidad servible o con una dependencia versionada.
7. Probar que un mismo input produce las mismas features offline y online.
8. Excluir del camino de inferencia las transformaciones exclusivas de training, como ciertos aumentos aleatorios de imagen.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Transformación dentro del grafo | Portabilidad y consistencia directa | Repetir cómputo en cada época |
| Pase previo con `tf.transform` | Evita recomputar estadísticas en training | Añade pipeline y artefactos que gestionar |
| Feature Store | Permite lógica fuera del grafo y reutilización | Dependencia externa y serving más complejo |
| Contenedor común | Desacopla lenguaje de transformación y modelo | Mayor coste operacional |

### 3.10 Ventajas

- Reduce *training-serving skew*.
- Simplifica la firma de predicción: el cliente envía inputs naturales.
- Conserva parámetros auxiliares junto con la lógica que los usa.
- Permite serving batch o en línea sin reimplementar preprocessing.
- Distingue transformaciones de instancia de análisis globales costosos.

### 3.11 Limitaciones

- No toda transformación cabe en un grafo de modelo.
- Las transformaciones dentro del modelo pueden repetirse en cada iteración.
- Las operaciones de dataset exigen un pase previo escalable.
- Transformaciones aleatorias exclusivas de training no deben reproducirse en serving.

### 3.12 Riesgos y trade-offs

- Calcular estadísticas con datos de validación o test introduce leakage.
- Una transformación cara dentro del loop aumenta tiempo de entrenamiento.
- Una Feature Store reduce acoplamiento al grafo, pero aumenta infraestructura y dependencia.
- Texto e imágenes requieren conservar exactamente limpieza, tokenización, orientación, crop, padding o resampling.
- Capturar transformaciones incorrectas garantiza consistencia, no corrección.

### 3.13 Cuándo utilizarlo

Cuando las features no coinciden con los inputs crudos, cuando hay parámetros aprendidos de preprocessing o cuando training y serving podrían implementar rutas distintas.

### 3.14 Cuándo no utilizarlo

No se debe incluir en el artefacto de inferencia una transformación usada solo para regularizar training, como aumento aleatorio que no forma parte del contrato de entrada. Si todas las features se obtienen de una Feature Store coherente, la transformación puede residir fuera del grafo.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Feature Store | Reutiliza features y admite lógica en otros lenguajes | Infraestructura y dependencia adicional | Features compartidas por muchos modelos |
| Contenedores de preprocessing | Aíslan lenguaje y dependencias | Orquestación más compleja | Transformaciones no expresables en el grafo |
| Transformación en el grafo | Serving autocontenido | Coste repetido en training | Operaciones por instancia y baratas |
| `tf.transform` | Análisis global escalable y artefactos servibles | Paso adicional | Media, varianza, vocabularios o buckets globales |

### 3.16 Implementación práctica

```python
# Idea mínima; la API concreta depende del framework.
raw = Input(name="raw_value")
feature = TransformLayer(saved_statistics)(raw)
prediction = Predictor()(feature)
model = Model(raw, prediction)
```

**Nota práctica de actualización:** las APIs de `tf.feature_column` y `DenseFeatures` mostradas en el libro corresponden a la época de publicación. Al actualizar el ejemplo, debe preservarse la arquitectura conceptual `inputs → preprocessing versionado → modelo`, sin presentar la API moderna como texto original del capítulo.

### 3.17 Ejemplo mínimo conceptual

Fecha cruda y estación → el artefacto extrae día de semana y hora con reglas versionadas → training y serving usan las mismas tres features → se evita que el cliente interprete de forma distinta qué número corresponde a cada día.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En un coding test conviene encapsular imputación, escalado y encoding en un pipeline único antes de comparar modelos. En un datathon, el patrón tiene prioridad alta cuando hay validación repetida o generación compleja de features: evita ajustar transformaciones sobre todo el dataset y evita que la ruta de test difiera de training. Puede mejorar un baseline indirectamente al eliminar leakage e inconsistencias; no garantiza una ganancia de métrica por sí solo.

### 3.19 Checklist operativo del patrón

- [ ] Separar inputs crudos de features.
- [ ] Ajustar artefactos solo con training.
- [ ] Versionar lógica y artefactos juntos.
- [ ] Reutilizar exactamente la transformación en validación y serving.
- [ ] Excluir transformaciones exclusivas de training del camino online.
- [ ] Comparar features offline y online para inputs idénticos.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Feature Store | Alternativo / complementario | Centraliza features cuando la transformación no cabe en el grafo y añade reutilización |
| Workflow Pipeline | Complementario | Orquesta el análisis, transformación, training y despliegue de artefactos |
| Windowed Inference | Complementario | Reproduce features con estado temporal |

## Pattern 22: Repeatable Splitting

### 3.1 Problema

Un split aleatorio por fila puede cambiar entre ejecuciones y colocar observaciones correlacionadas en training y test. En vuelos, dos vuelos del mismo día comparten condiciones; separarlos produce leakage. Fijar una semilla o guardar el resultado no resuelve bien el crecimiento del dataset, el uso entre lenguajes ni las múltiples particiones requeridas por validación, bootstrap o tuning.

### 3.2 Contexto de aplicación

Datasets tabulares o no estructurados con grupos correlacionados por persona, fecha, dispositivo, ubicación o evento; comparación repetida de modelos; datos crecientes; series temporales; clases o features raras; pretraining y embeddings.

### 3.3 Señales que indican que el patrón puede ser necesario

- La métrica cambia al regenerar el split.
- La misma entidad, fecha o evento aparece en training y test.
- Hay duplicados o vecinos temporales casi idénticos entre particiones.
- Nuevos datos reasignan filas históricas.
- **Inferencia práctica:** el ranking de modelos depende más del split que del modelo.

### 3.4 Solución propuesta

1. Identificar la columna o combinación de columnas que representa la correlación entre filas.
2. Aplicar una función hash determinista y bien distribuida.
3. Usar rangos del módulo del hash para asignar train, validation y test.
4. Verificar suficientes valores únicos y distribuciones comparables de labels/features.
5. Versionar la lógica de partición y actualizar la versión del modelo cuando cambie.

### 3.5 Cómo funciona

El mismo valor de grupo produce siempre el mismo hash y, por tanto, la misma partición. Todas las filas correlacionadas permanecen juntas. Rangos del residuo, por ejemplo 0–7, 8 y 9, crean aproximadamente 80/10/10 sin depender de orden, seed o lenguaje, siempre que se use el mismo algoritmo.

### 3.6 Por qué funciona

El determinismo estabiliza la evaluación entre ejecuciones, mientras el hash distribuye grupos sin almacenar una lista de asignaciones. Agrupar antes de dividir corta la vía de leakage entre observaciones correlacionadas y permite que el dataset crezca sin mover ejemplos existentes.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Filas y una clave de agrupación estable |
| Transformación | Hash determinista más módulo y rangos |
| Salida | Identificador estable de train, validation o test |

### 3.8 Procedimiento operativo

1. Formular qué dependencia debe quedar confinada a un split.
2. Elegir clave de grupo; cruzar columnas si la dependencia es conjunta.
3. Confirmar que la clave no excluye del training valores necesarios de una feature del modelo.
4. Confirmar suficiente cardinalidad; el libro sugiere como regla de seguridad 3–5 veces el denominador del módulo.
5. Aplicar hash y asignación determinista.
6. Comparar distribución del label y variables importantes; el capítulo propone CDF y distancia Kolmogorov–Smirnov.
7. Para series o comportamiento adaptativo, usar split secuencial y respetar estacionalidad.
8. Para datos desbalanceados, estratificar antes de dividir.
9. Calcular embeddings o autoencoders después del split y solo con training, salvo pretraining sobre un dataset totalmente independiente.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Clave de hash | Define independencia entre splits | Clave equivocada deja leakage o excluye categorías |
| Denominador y rangos | Fijan proporciones | Poca cardinalidad produce splits inestables o sesgados |
| Split temporal | Evalúa futuro real | Menos mezcla y posibles diferencias estacionales |
| Estratificación | Conserva grupos raros | Puede ignorar correlación si se aplica sin agrupación |
| Hash de fila completa | Split aleatorio reproducible | No protege contra correlaciones semánticas |

### 3.10 Ventajas

- Independiente de seeds y orden de filas.
- Reproducible entre ejecuciones y lenguajes compatibles.
- Estable al crecer el dataset.
- Mantiene duplicados o grupos correlacionados en una sola partición.
- No requiere persistir copias completas de cada split.

### 3.11 Limitaciones

- Requiere una clave que represente la dependencia real.
- Un split por entidad impide usar esa entidad como feature si sus valores quedan totalmente separados.
- La distribución de labels puede quedar sesgada.
- Muestreo y partición con módulos relacionados requieren cuidado.

### 3.12 Riesgos y trade-offs

- Hash del texto, filename o fila equivale a random split y puede ocultar correlaciones por autor, captura o evento.
- Si muestreo y split usan módulos incompatibles, la segunda condición puede ser redundante.
- En series temporales, random split filtra información del futuro.
- Embeddings calculados antes de dividir pueden contaminar evaluación.
- Un único query reduce cómputo, pero agrega una columna o tabla materializada.

### 3.13 Cuándo utilizarlo

Cuando deben compararse modelos con la misma evaluación, el dataset crece o existen grupos correlacionados que no pueden cruzar particiones.

### 3.14 Cuándo no utilizarlo

No usar hash aleatorio cuando el objetivo exige extrapolar al futuro o cuando el orden temporal y la adaptación del entorno son parte del problema; allí corresponde split secuencial. Tampoco usar una clave con muy pocos valores o distribución de label incompatible.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Seed aleatoria | Simple | Dependiente de implementación y datos | Experimento local transitorio sin grupos |
| Persistir splits | Reproduce exactamente | Duplica datos y envejece con el dataset | Muestra pequeña congelada |
| Hash de fila completa | No exige clave natural | No agrupa correlaciones | Filas realmente independientes |
| Split secuencial | Simula inferencia futura | Distribución cambia con el tiempo | Forecasting, fraude adaptativo, alta autocorrelación |
| Split estratificado | Preserva categorías raras | Debe combinarse con agrupación | Labels o features muy desbalanceadas |

### 3.16 Implementación práctica

```sql
CASE ABS(MOD(stable_hash(group_key), 10))
  WHEN 9 THEN 'test'
  WHEN 8 THEN 'validation'
  ELSE 'training'
END AS split
```

Para múltiples columnas, `group_key` es una concatenación no ambigua. Para tomar 1/70 y luego 80% de esa muestra, el capítulo usa módulos compatibles con el producto (70 y 700), no 70 y 10.

### 3.17 Ejemplo mínimo conceptual

Vuelos del mismo día correlacionados → hash estable de la fecha → todos los vuelos del día caen en un solo split → la evaluación no se beneficia de condiciones del mismo día vistas durante training.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es uno de los primeros patrones que conviene evaluar. En coding tests, usar `GroupKFold`, split temporal o equivalente conceptual cuando una entidad se repite. En datathons, inspeccionar IDs, usuarios, sesiones, fechas y fuentes antes del baseline. Puede bajar una métrica inicialmente inflada, pero ofrece una estimación más fiable y evita seleccionar modelos por leakage. El coste de implementación es bajo si existe una clave clara.

### 3.19 Checklist operativo del patrón

- [ ] Identificar grupos y correlaciones.
- [ ] Elegir hash determinista y clave estable.
- [ ] Mantener cada grupo en un solo split.
- [ ] Revisar cardinalidad y distribución del label.
- [ ] Usar split temporal si el futuro es el objetivo.
- [ ] Ajustar transformaciones y embeddings solo con training.
- [ ] Versionar la lógica de split.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Transform | Dependencia | Las transformaciones aprendidas deben ajustarse después del split |
| Workflow Pipeline | Complementario | Automatiza y versiona la generación de particiones |
| Model Versioning | Complementario | Un cambio de split debe quedar asociado a una nueva versión |

## Pattern 23: Bridged Schema

### 3.1 Problema

Un proveedor mejora el esquema y ofrece información más granular o nuevas features, pero todavía hay pocos registros nuevos para entrenar un buen modelo. Entrenar solo con lo nuevo sacrifica volumen; usar los datos viejos sin adaptación introduce valores que jamás aparecerán en producción. El ejemplo transforma `card` del esquema antiguo en `gift card`, `debit card` y `credit card` del nuevo.

### 3.2 Contexto de aplicación

Migraciones de esquema donde serving ya recibe exclusivamente la versión nueva, pero training dispone de un gran histórico antiguo y una muestra pequeña nueva. Incluye categorías desagregadas, nuevas features y mayor precisión de medición.

### 3.3 Señales que indican que el patrón puede ser necesario

- Una fuente cambia su contrato de datos.
- Las nuevas variables parecen útiles, pero no hay volumen suficiente.
- Un valor antiguo agregado no existirá en inferencia futura.
- El modelo con solo datos nuevos es inestable o inferior por falta de muestra.
- **Inferencia práctica:** el equipo planea retrasar una mejora valiosa únicamente para acumular datos.

### 3.4 Solución propuesta

1. Adaptar el esquema antiguo al nuevo mediante imputación.
2. Preferir imputación estática: representar el valor antiguo con las probabilidades a priori estimadas únicamente en training nuevo.
3. Reservar una evaluación suficiente compuesta solo por datos del esquema nuevo.
4. Entrenar con los datos nuevos restantes más una cantidad ajustable de ejemplos antiguos puenteados.
5. Ajustar cuántos ejemplos antiguos incorporar y reducirlos conforme crece el dataset nuevo.
6. Comparar el modelo puenteado con el modelo de producción sin cambios sobre la evaluación nueva.

### 3.5 Cómo funciona

En one-hot, un `card` antiguo se convierte, por ejemplo, en `[0, 0.1, 0.3, 0.6]`, mientras un débito observado en el nuevo esquema es `[0, 0, 1, 0]`. La representación fraccionaria equivale al promedio que produciría una asignación probabilística tras muchas épocas, pero es determinista y barata. El histórico aporta señal general; los registros nuevos aportan la granularidad que existirá en producción.

### 3.6 Por qué funciona

El patrón alinea la representación del histórico con el contrato futuro sin fingir conocer la categoría exacta de cada observación antigua. El volumen antiguo reduce la varianza del modelo, mientras la evaluación exclusivamente nueva mide el entorno real de despliegue. La utilidad del histórico se trata como hiperparámetro con rendimientos decrecientes.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Pocos datos nuevos y abundantes datos bajo el esquema antiguo |
| Transformación | Imputación estática/probabilística o modelo Cascade |
| Salida | Dataset de training híbrido expresado bajo el esquema nuevo; evaluación solo nueva |

### 3.8 Procedimiento operativo

1. Definir el contrato que estará disponible en producción.
2. Estimar distribuciones o estadísticos de imputación solo con training nuevo.
3. Determinar el tamaño mínimo de evaluación observando cuándo se estabiliza la variabilidad de la métrica del modelo actual.
4. Reservar esa evaluación exclusivamente del esquema nuevo.
5. Adaptar el histórico mediante imputación estática.
6. Probar cantidades crecientes de datos antiguos y elegir la menor cantidad que alcance el rendimiento necesario.
7. Evitar usar evaluación para early stopping, selección de checkpoints o tuning; el capítulo propone regularización y pérdida de training cuando solo pueden conservarse dos splits.
8. Comparar modelo antiguo, modelo solo nuevo y modelo puenteado.
9. Retirar gradualmente el histórico a medida que crece la nueva fuente.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Tamaño de evaluación | Estabilidad de la estimación | Restar demasiados datos al training nuevo |
| Número de ejemplos puenteados | Balance volumen/relevancia | Dominar training con información aproximada |
| Estadístico de imputación | Representación del valor desconocido | Supuesto distributivo incorrecto |
| Static vs Cascade | Simplicidad frente a precisión | Cascade añade complejidad temporal |

El libro ilustra, no universaliza, una estabilización alrededor de 2 000–3 000 ejemplos de evaluación y rendimientos decrecientes después de unos 20 000 ejemplos puenteados.

### 3.10 Ventajas

- Usa antes una mejora del esquema.
- Aprovecha histórico sin introducir una categoría imposible en producción.
- La imputación estática es determinista, simple y barata.
- Permite retirar progresivamente el puente.

### 3.11 Limitaciones

- La distribución global puede ser incorrecta para subgrupos específicos.
- El histórico sigue siendo una aproximación del esquema nuevo.
- La evaluación nueva consume datos escasos.
- Tuning sin validation separada restringe técnicas como early stopping.

### 3.12 Riesgos y trade-offs

- Usar evaluación en tuning produce sobreajuste y una estimación optimista.
- Un esquema unión (`cash`, `card`, `gift`, `debit`, `credit`) entrena sobre `card`, valor ausente en producción.
- Demasiado histórico puede diluir la señal nueva.
- Cascade puede mejorar imputación, pero es excesivo para una solución temporal.
- Una distribución altamente sesgada y mal muestreada puede invalidar la imputación de precisión.

### 3.13 Cuándo utilizarlo

Cuando el nuevo contrato ya rige inference, el histórico es valioso y los datos nuevos aún no bastan.

### 3.14 Cuándo no utilizarlo

Si el modelo solo con datos nuevos ya supera al puenteado, si la imputación no mejora al modelo anterior o si el valor antiguo no puede mapearse razonablemente al nuevo. Debe desaparecer cuando haya suficientes datos nuevos.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Método probabilístico | Simula categorías según frecuencia | Aleatorio y costoso; mismo promedio a largo plazo | Fines experimentales |
| Imputación estática | Determinista y barata | Ignora contexto por fila | Opción recomendada por el capítulo |
| Cascade | Usa otras variables para imputar | Complejidad alta para solución temporal | Static resulta insuficiente |
| Union schema | No transforma históricos | Entrena un valor ausente en producción | No recomendada para este caso de ML |
| Esperar nuevos datos | Sin aproximación histórica | Retrasa beneficio | El puente no aporta valor |

### 3.16 Implementación práctica

```python
# Frecuencias estimadas solo en training del esquema nuevo.
bridge = {"cash": [1, 0, 0, 0], "card": [0, 0.1, 0.3, 0.6]}
old_payment_vector = old_payment.map(bridge)
```

Para nuevas features: media si es numérica aproximadamente normal; mediana si es numérica sesgada o con outliers; mediana si la categoría es ordenable; moda si no lo es; frecuencia de `true` si es booleana.

### 3.17 Ejemplo mínimo conceptual

Histórico solo distingue cash/card → `card` se representa con probabilidades de los tipos nuevos → se entrena con una evaluación reservada solo del esquema nuevo → el modelo utiliza información moderna sin desperdiciar todo el histórico.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Es útil si train y test reflejan versiones distintas de una fuente o si una feature solo existe en datos recientes. Bajo presión tiene prioridad media: primero debe verificarse que el problema sea realmente evolución de esquema. La imputación estática es viable; Cascade probablemente consume demasiado tiempo. La validación debe parecerse al test nuevo y quedar aislada del tuning.

### 3.19 Checklist operativo del patrón

- [ ] Confirmar que producción usa solo el esquema nuevo.
- [ ] Reservar evaluación exclusivamente nueva.
- [ ] Calcular imputación solo con training nuevo.
- [ ] Comparar static, solo nuevo y modelo anterior.
- [ ] Ajustar la menor cantidad útil de histórico.
- [ ] No usar evaluación para tuning.
- [ ] Planificar retiro del puente.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Transform | Dependencia | Materializa de forma reproducible el mapeo entre esquemas |
| Repeatable Splitting | Complementario | Separa la evaluación nueva antes de estimar transformaciones |
| Workflow Pipeline | Complementario | Versiona y automatiza el puente temporal |
| Model Versioning | Complementario | Permite comparar y retirar versiones durante la migración |

## Pattern 24: Windowed Inference

### 3.1 Problema

Algunas predicciones no dependen de una sola instancia, sino de una secuencia o de agregados recientes. Una anomalía en retrasos de vuelos solo puede juzgarse respecto a las dos horas anteriores. En histórico se dispone del dataframe completo; en producción llegan eventos individuales. Recalcular la ventana de manera distinta causa *training-serving skew* y conservar todos los eventos puede agotar memoria.

### 3.2 Contexto de aplicación

Anomaly detection, forecasting, sequence models, RNN/LSTM, traducción con contexto, features con promedios móviles, streams de alta frecuencia y batching de solicitudes tolerantes a latencia.

### 3.3 Señales que indican que el patrón puede ser necesario

- Una predicción requiere eventos anteriores.
- Una feature usa `rolling`, ventanas SQL o agregados temporales.
- Training calcula contexto en batch y serving recibe un evento por vez.
- Reentrenar o recalcular por cada elemento es demasiado caro.
- **Inferencia práctica:** el servicio “stateless” exige al cliente reconstruir un historial inconsistente.

### 3.4 Solución propuesta

1. Ejecutar inferencia desde un pipeline de streaming con estado.
2. Definir ventana, deslizamiento y frecuencia de cierre.
3. Actualizar estado interno con cada evento.
4. Al cerrar la ventana, calcular o actualizar parámetros del modelo/agregado.
5. Externalizar un estado compacto.
6. Aplicar ese estado a cada nuevo elemento.
7. Usar la misma lógica de ventana para histórico y producción.

### 3.5 Cómo funciona

En el ejemplo, una ventana de dos horas se cierra cada diez minutos. El pipeline calcula media y desviación de retrasos y publica `prediction` y `acceptable_deviation`. Cada vuelo se compara con ese estado sin transportar toda la ventana al modelo. Si el modelo admite actualización online, basta mantener suma, suma de cuadrados y conteo en vez de millones de eventos.

### 3.6 Por qué funciona

Externalizar el estado desacopla entrenamiento/agregación costosa de la inferencia por evento. El pipeline reproduce el orden temporal, conserva contexto y permite escalar un predictor stateless. El mismo código de streaming puede procesar histórico y eventos nuevos.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Stream ordenado con timestamp y clave de agrupación |
| Transformación | Ventana, actualización de estado y modelo/agregado |
| Salida | Predicción por evento más estado externo versionable |

### 3.8 Procedimiento operativo

1. Definir qué historia necesita la predicción y por qué.
2. Elegir ventana temporal, paso y clave de partición.
3. Excluir el evento actual del agregado cuando corresponda.
4. Implementar el cálculo en un motor batch/stream compartido.
5. Separar actualización cara del estado e inferencia barata por evento.
6. Reducir estado a estadísticas suficientes si el algoritmo lo permite.
7. Verificar tratamiento de eventos tardíos, límites y orden.
8. Medir memoria, throughput y latencia.
9. Comparar features históricas y online en los mismos timestamps.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Longitud de ventana | Cantidad de contexto | Ventana corta pierde señal; larga mezcla regímenes |
| Paso/cierre | Frescura frente a coste | Paso grande deja estado obsoleto |
| Estado completo vs online | Flexibilidad frente a memoria | Estadísticas suficientes no existen para todo modelo |
| Beam vs SQL streaming | Escalabilidad/latencia | SQL analítico puede tardar segundos |
| Batch de requests | Reduce overhead de red | Añade demora máxima del intervalo |

### 3.10 Ventajas

- Reproduce features temporales entre training y serving.
- Permite inferencia por evento con modelos stateless.
- Separa cómputo caro de scoring barato.
- Las actualizaciones online reducen drásticamente memoria.
- Admite secuencias, agregados y batching.

### 3.11 Limitaciones

- Requiere infraestructura de streaming y manejo de estado.
- No todo modelo permite actualización incremental.
- Ventanas aproximadas pueden diferir de un cálculo centrado por evento.
- Batching solo sirve si el caso tolera latencia.

### 3.12 Riesgos y trade-offs

- Incluir el evento actual o futuro en su propia feature introduce leakage.
- Mantener dataframes por ventana escala mal en streams intensos.
- Un motor de consulta puede ser preciso pero insuficiente para control en tiempo real.
- Diferentes implementaciones batch/stream reintroducen skew.
- Estado atrasado produce predicciones con contexto obsoleto.

### 3.13 Cuándo utilizarlo

Cuando el modelo o alguna feature necesita una historia móvil y los eventos llegan incrementalmente.

### 3.14 Cuándo no utilizarlo

No aporta valor si cada instancia es independiente y todas las features están disponibles al momento de inferencia. No usar batching si el SLA no tolera la espera.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Ventana SQL | Misma consulta en batch/stream | Latencia potencial de segundos | Analítica no crítica en tiempo real |
| Estado completo | Soporta modelos arbitrarios | Memoria elevada | Volumen moderado |
| Actualización online | Estado mínimo | Solo algoritmos actualizables | Streams de alto throughput |
| Two-Phase Predictions | Agrupa llamadas remotas | Latencia adicional | Cliente remoto tolerante a espera |

### 3.16 Implementación práctica

```python
# Estado suficiente para media y varianza.
state = (sum_x, sum_x2, count)
mean = sum_x / count
variance = sum_x2 / count - mean * mean
is_anomaly = abs(x - mean) > 4 * sqrt(max(variance, 0))
```

El multiplicador cuatro pertenece al ejemplo del libro, no es una recomendación universal.

### 3.17 Ejemplo mínimo conceptual

Evento de retraso aislado → pipeline conserva estadísticas de las dos horas previas → compara el nuevo retraso con el estado externo → marca anomalía según el contexto vigente.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En datathons temporales, crear rolling features con ventanas que terminen antes del registro actual. Es prioridad alta si la métrica depende de tiempo y media si solo mejora marginalmente un baseline. En coding tests, mostrar explícitamente orden, partición por entidad y exclusión del presente suele ser más valioso que construir streaming real. Evita el error frecuente de calcular agregados con toda la serie.

### 3.19 Checklist operativo del patrón

- [ ] Ordenar por tiempo y particionar por entidad.
- [ ] Definir ventana y paso.
- [ ] Excluir presente/futuro cuando corresponda.
- [ ] Reusar lógica batch y online.
- [ ] Externalizar estado mínimo.
- [ ] Medir memoria, latencia y eventos tardíos.
- [ ] Validar paridad histórica/online.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Transform | Complementario | La ventana es una transformación dependiente del tiempo |
| Workflow Pipeline | Dependencia | Orquesta estado, agregación e inferencia |
| Feature Store | Puede combinarse | Publica features temporales para acceso online/offline |

## Pattern 25: Workflow Pipeline

### 3.1 Problema

Un notebook monolítico deja implícitos orden, dependencias, entornos y artefactos. Al crecer el equipo y el proceso, un cambio pequeño obliga a ejecutar o desplegar todo; es difícil depurar pasos, programar retraining, reaccionar a eventos y reproducir meses después el entorno exacto.

### 3.2 Contexto de aplicación

Flujos con ingestión, validación, transformación, entrenamiento, evaluación y despliegue; equipos múltiples; reentrenamiento programado o por evento; requerimientos de lineage, auditoría y ejecución en local, on-premises o cloud.

### 3.3 Señales que indican que el patrón puede ser necesario

- Solo una persona puede ejecutar el flujo completo.
- El orden de notebooks se comunica manualmente.
- Dependencias de pasos colisionan.
- No puede rastrearse qué datos o transformación produjeron un modelo.
- Retraining y deployment requieren intervención manual.
- **Inferencia práctica:** corregir un componente obliga a repetir todo sin caché ni trazabilidad.

### 3.4 Solución propuesta

1. Convertir cada etapa en un componente aislado, preferentemente containerizado.
2. Declarar inputs, outputs y dependencias como un DAG.
3. Orquestar el DAG con un framework de pipelines.
4. Registrar logs, metadatos y artefactos por componente y ejecución.
5. Permitir ejecutar un paso o el flujo completo.
6. Añadir triggers de calendario, datos o cambios de código.
7. Incorporar condiciones de promoción basadas en evaluación.

### 3.5 Cómo funciona

Cada componente ejecuta código y dependencias propias y entrega artefactos tipados al siguiente. El orquestador decide orden, scheduling, retries y ramas. El ejemplo TFX cubre `ExampleGen → StatisticsGen/SchemaGen/ExampleValidator/Transform → Trainer → Pusher`. La containerización reproduce runtime; el almacén de metadatos conserva lineage.

### 3.6 Por qué funciona

Un DAG hace explícita la topología del proceso. Los contenedores fijan dependencias y entorno; los contratos entre componentes reducen acoplamiento. Artefactos, logs y metadatos permiten repetir, depurar y comparar ejecuciones. Separar pasos posibilita trabajo paralelo y reutilización.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Datos, código, configuración y versiones de componentes |
| Transformación | DAG orquestado de ingestión a despliegue |
| Salida | Modelo desplegado y lineage de artefactos/logs por ejecución |

### 3.8 Procedimiento operativo

1. Dibujar las etapas y sus contratos.
2. Separar código, runtime y dependencias por componente.
3. Containerizar o usar componentes preconstruidos.
4. Declarar el DAG y sus ubicaciones de artefactos.
5. Añadir validación de datos antes de training.
6. Registrar estadísticas, schema, modelo y evaluación.
7. Ejecutar localmente y luego en un orquestador.
8. Añadir triggers solo cuando el flujo sea estable.
9. Condicionar deployment a criterios explícitos.
10. Conservar lineage entre datos, código, ejecución y modelo.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Granularidad de componentes | Aislamiento y reutilización | Demasiados pasos elevan overhead |
| Prebuilt vs custom | Rapidez frente a control | Custom exige mantenimiento |
| TFX/KFP/Airflow | Abstracción ML frente a flexibilidad | Lock-in operativo o complejidad |
| Trigger temporal/datos/código | Frescura | Retraining innecesario o tardío |
| Umbral de promoción | Controla deployment | Una métrica aislada puede ser insuficiente |

### 3.10 Ventajas

- Reproducibilidad end-to-end.
- Componentes ejecutables y depurables por separado.
- Entornos heterogéneos sin mezclar dependencias.
- Scheduling, triggers y CI/CD.
- Logs, monitoreo, artefactos y lineage centralizados.
- Colaboración paralela entre equipos.

### 3.11 Limitaciones

- Convertir notebooks en componentes tiene coste inicial.
- Operar contenedores y orquestador añade infraestructura.
- Un pipeline no garantiza que cada paso sea estadísticamente correcto.
- Desarrollo interactivo y producción requieren formas de invocación distintas.

### 3.12 Riesgos y trade-offs

- Componentes demasiado acoplados recrean un monolito distribuido.
- Automatizar sin validación puede desplegar regresiones.
- Dependencias o artefactos no versionados rompen reproducibilidad.
- Umbrales rígidos ignoran otras dimensiones de calidad.
- La portabilidad del framework no elimina diferencias de infraestructura.

### 3.13 Cuándo utilizarlo

Cuando el flujo se repite, participa más de una persona, se despliega a producción o se necesita auditoría y retraining automático.

### 3.14 Cuándo no utilizarlo

Para una exploración desechable de una sola persona, un notebook puede iterar más rápido. El patrón cobra valor al estabilizarse etapas o necesitar repetición.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Notebooks/scripts ad hoc | Iteración inmediata | Poca escala y trazabilidad | Prototipo inicial |
| TFX | Componentes ML de alto nivel | Menos flexibilidad | Flujo TensorFlow convencional |
| Kubeflow Pipelines | Flexible y orientado a ML | Más trabajo de definición | Componentes personalizados |
| Airflow | Maduro para ETL | No diseñado específicamente para ML | Organización ya centrada en Airflow |
| Contenedores custom | Control total | Mayor mantenimiento | Runtime o herramientas especiales |

### 3.16 Implementación práctica

```python
pipeline = Pipeline(components=[
    ingest, validate, transform, train, evaluate, deploy
])
# Cada componente declara inputs, outputs, imagen y dependencias.
```

**Nota práctica de actualización:** los nombres de servicios cloud y comandos del libro reflejan 2020. La receta reusable es DAG + componentes aislados + artefactos/lineage + orquestación, no una versión concreta de la CLI.

### 3.17 Ejemplo mínimo conceptual

Notebook único difícil de repetir → seis componentes containerizados con contratos y DAG → una ejecución registra datasets, estadísticas, modelo y evaluación → el flujo puede repetirse o depurarse por etapa.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En un coding test, basta un pipeline claro y determinista; containerizar suele ser excesivo. En un datathon de equipo, separar `prepare → features → train → evaluate → predict` evita pisar artefactos y permite comparar ejecuciones. Prioridad media al inicio y alta cuando ya existen múltiples modelos, seeds o entregables. No mejora el leaderboard directamente, pero evita resultados irreproducibles y submission skew.

### 3.19 Checklist operativo del patrón

- [ ] Definir DAG e interfaces.
- [ ] Aislar dependencias por componente.
- [ ] Versionar código, datos y artefactos.
- [ ] Registrar logs y lineage.
- [ ] Validar antes de entrenar y desplegar.
- [ ] Probar pasos individuales y flujo completo.
- [ ] Automatizar triggers y promoción con criterios explícitos.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Transform | Complementario | El componente de transformación produce artefactos reproducibles |
| Repeatable Splitting | Complementario | El split versionado es una etapa del DAG |
| Feature Store | Puede combinarse | El pipeline ingiere y recupera features compartidas |
| Model Versioning | Dependencia | Deployment crea y registra una versión nueva |
| Windowed Inference | Puede combinarse | El DAG puede incluir componentes streaming |

## Pattern 26: Feature Store

### 3.1 Problema

Features costosas y dependientes del dominio se recrean en notebooks, no se comparten ni documentan y se calculan con rutas diferentes en training y serving. Esto duplica esfuerzo, dificulta gobernanza, puede introducir leakage y obliga a resolver simultáneamente throughput histórico y latencia online.

### 3.2 Contexto de aplicación

Organizaciones con múltiples modelos/equipos, fuentes batch y streaming, features complejas o caras, modelos online de baja latencia y entrenamiento sobre grandes históricos.

### 3.3 Señales que indican que el patrón puede ser necesario

- Equipos calculan la misma feature de forma distinta.
- Embeddings o agregados caros se recomputan por proyecto.
- Training usa batch y serving reimplementa online.
- No hay catálogo, owner, schema ni versión de features.
- **Inferencia práctica:** una feature útil está atrapada en un notebook o pipeline privado.

### 3.4 Solución propuesta

1. Desacoplar creación de features de consumo por modelos.
2. Centralizar datasets de features, documentación, versiones y controles de acceso.
3. Construir una capa de procesamiento batch/stream.
4. Mantener almacenes offline de alto throughput y online de baja latencia.
5. Proveer API común de ingestión y recuperación.
6. Usar entidades y timestamps para joins *point-in-time correct*.

### 3.5 Cómo funciona

Pipelines de feature engineering escriben la misma definición en stores online y offline. Model training consulta históricos por entidad y timestamp; serving recupera el valor más reciente con baja latencia. Una capa de metadatos registra schema, fuente, versión y documentación. En el ejemplo Feast, Beam procesa, BigQuery sirve offline y Redis online.

### 3.6 Por qué funciona

La única canalización de creación reduce divergencia entre training y serving. Separar almacenes optimiza necesidades incompatibles de throughput y latencia sin cambiar semántica. El catálogo convierte features en activos reutilizables y gobernables.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Fuentes crudas batch/stream, entidades, timestamps y transformaciones |
| Transformación | Pipelines de features y escritura coordinada online/offline |
| Salida | Features históricas point-in-time correct y features online de baja latencia |

### 3.8 Procedimiento operativo

1. Definir entidad, timestamp, schema y owner de cada feature set.
2. Implementar una sola lógica de creación para batch y stream.
3. Registrar el feature set sin datos.
4. Ingerir y validar schema.
5. Poblar stores online y offline de forma coordinada.
6. Recuperar training por entidades y timestamps.
7. Recuperar serving por entidad con SLA de latencia.
8. Versionar y documentar cambios; controlar accesos.
9. Verificar consistencia y fallos parciales entre stores.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Entidad | Clave de joins y lookup | Entidad inmutable o granularidad incorrecta |
| Timestamp/max age | Corrección temporal | Leakage o features obsoletas |
| Store online/offline | Latencia y escala | Divergencia por fallos parciales |
| Feature set/schema | Reutilización | Cambios incompatibles |
| Cadencia de ingestión | Frescura | Coste y retraso |

### 3.10 Ventajas

- Reutilización y descubrimiento de features.
- Menos *training-serving skew*.
- Joins históricos temporalmente correctos.
- Gobernanza, documentación, acceso y versionado.
- Optimización separada para batch y online.
- Menor tiempo de desarrollo de nuevos modelos.

### 3.11 Limitaciones

- Infraestructura considerable para proyectos pequeños.
- Debe coordinar dos stores y fallos parciales.
- No ejecuta por sí sola toda transformación; recibe features de pipelines upstream.
- Algunas operaciones dentro del grafo pueden aprovechar mejor hardware acelerado.

### 3.12 Riesgos y trade-offs

- Timestamps incorrectos provocan leakage silencioso.
- Online y offline pueden divergir ante carreras o ingestión parcial.
- Cambiar entidades o nombres puede ser incompatible.
- Features sensibles requieren controles uniformes.
- Centralización crea dependencia crítica del store.

### 3.13 Cuándo utilizarlo

Cuando varias soluciones comparten features, se necesitan valores online y históricos consistentes o el coste de recreación/gobernanza ya es significativo.

### 3.14 Cuándo no utilizarlo

Para un único modelo pequeño con transformaciones simples y serving autocontenido, el overhead puede superar el beneficio; `Transform` puede bastar.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| `tf.transform` | Paridad training-serving y aceleración en grafo | Features no se comparten fácilmente | Un modelo o ecosistema TensorFlow |
| Transform en modelo | Artefacto autocontenido | No sirve como catálogo común | Transformaciones por instancia |
| Feature stores alternativos | Distintos stacks y despliegues | Operación específica | Ajuste a infraestructura existente |

### 3.16 Implementación práctica

```python
feature_set = FeatureSet(
    name="entity_features",
    entities=[Entity("entity_id")],
    features=[Feature("rolling_count")],
    source=source,
)
register(feature_set)
ingest(feature_set, feature_data)
```

**Nota práctica de actualización:** la API de Feast del libro es histórica. Debe actualizarse solo la sintaxis, conservando entidades, timestamps, schema, ingestión coordinada y recuperación online/offline como núcleo del patrón.

### 3.17 Ejemplo mínimo conceptual

Tres equipos calculan pedidos del último mes → un pipeline gobernado publica la feature por cliente y timestamp → training recupera historia correcta y serving el último valor → todos usan la misma definición.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** Construir una Feature Store real tiene prioridad baja en un datathon corto. Sí conviene imitar sus principios: catálogo simple, nombres estables, timestamps, funciones reutilizables y caché de features caras. Puede acelerar iteraciones y evitar inconsistencias, pero el overhead de una plataforma completa rara vez compensa bajo presión.

### 3.19 Checklist operativo del patrón

- [ ] Definir entidades y timestamps.
- [ ] Asegurar joins point-in-time correct.
- [ ] Unificar creación batch/stream.
- [ ] Documentar owner, schema y versión.
- [ ] Validar paridad online/offline.
- [ ] Controlar acceso a features sensibles.
- [ ] Detectar ingestión parcial y staleness.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Transform | Alternativo / complementario | Transform conserva lógica en el artefacto; Feature Store añade reutilización |
| Windowed Inference | Complementario | Produce y sirve agregados temporales |
| Workflow Pipeline | Dependencia | Orquesta creación, validación e ingestión |
| Model Versioning | Complementario | Vincula una versión a definiciones concretas de features |

## Pattern 27: Model Versioning

### 3.1 Problema

Los modelos cambian por drift, nuevos datos, arquitectura o contratos de salida. Reemplazar un único endpoint rompe consumidores, impide comparar desempeño histórico, dificulta A/B testing y mezcla errores de versiones distintas. Algunos usuarios necesitan mantener temporalmente el comportamiento anterior.

### 3.2 Contexto de aplicación

Modelos desplegados como servicios, múltiples consumidores internos o externos, actualizaciones frecuentes, monitoring, rollout gradual, cambios de framework y necesidad de backward compatibility.

### 3.3 Señales que indican que el patrón puede ser necesario

- Un update puede alterar predicciones o formato de respuesta.
- Clientes actualizan a ritmos distintos.
- Se desea A/B test o canary release.
- No puede atribuirse una regresión a un modelo concreto.
- **Inferencia práctica:** rollback significa reconstruir manualmente un artefacto anterior.

### 3.4 Solución propuesta

1. Desplegar cada cambio como versión separada con endpoint identificable.
2. Mantener varias versiones disponibles durante la transición.
3. Definir una versión por defecto sin eliminar acceso explícito a anteriores.
4. Enrutar tráfico por usuario o grupo para pruebas.
5. Monitorizar precisión, errores, uso y tráfico por versión.
6. Retirar versiones según evidencia y política.
7. Crear un modelo nuevo, no una versión, cuando cambia la tarea predictiva.

### 3.5 Cómo funciona

Cada versión es un microservicio stateless asociado a un artefacto, runtime y contrato. El cliente o gateway elige endpoint. Un gateway puede enviar un porcentaje estable de usuarios a la nueva versión. Las métricas se etiquetan por versión, lo que permite comparación, rollback y deprecación controlada.

### 3.6 Por qué funciona

Versionar desacopla el ciclo de vida del modelo del frontend y preserva contratos existentes. La coexistencia convierte un reemplazo irreversible en una migración observable. El identificador de versión conecta predicción, datos, código y métricas.

### 3.7 Entradas y salidas

| Elemento | Descripción |
| -------- | ----------- |
| Entrada | Artefacto, runtime, schema de request/response y metadatos de training |
| Transformación | Despliegue aislado y reglas de routing |
| Salida | Endpoint versionado, métricas separadas y capacidad de rollback |

### 3.8 Procedimiento operativo

1. Definir contrato y compatibilidad esperada.
2. Empaquetar artefacto y dependencias de forma inmutable.
3. Asignar versión; incluir rango temporal de training cuando sea útil.
4. Desplegar sin sobrescribir la versión anterior.
5. Ejecutar pruebas de contrato y rendimiento.
6. Enrutar un subconjunto estable de tráfico.
7. Comparar métricas técnicas y de producto.
8. Promover a default o revertir.
9. Deprecar versiones antiguas con evidencia y comunicación.

### 3.9 Decisiones e hiperparámetros relevantes

| Decisión | Efecto | Riesgo |
| -------- | ------ | ------ |
| Porcentaje de tráfico | Velocidad de aprendizaje | Exponer demasiados usuarios a regresión |
| Default | Afecta clientes sin versión explícita | Upgrade involuntario |
| Retención de versiones | Rollback y compatibilidad | Coste y complejidad |
| Versión vs modelo nuevo | Define continuidad de tarea | Romper clientes con contrato distinto |
| Multiple signatures | Varias interfaces en un artefacto | Todas cambian al desplegar ese artefacto |

### 3.10 Ventajas

- Backward compatibility.
- Rollout gradual, A/B testing y rollback.
- Monitoring y atribución por versión.
- Separación entre modelo y aplicación.
- Cambio de framework sin cambiar necesariamente el contrato.

### 3.11 Limitaciones

- Ejecutar varias versiones consume recursos.
- Matriz de versiones y signatures puede volverse compleja.
- El patrón no decide automáticamente cuándo retirar una versión.
- Cambios de tarea requieren nuevo recurso, no solo versión.

### 3.12 Riesgos y trade-offs

- Un default nuevo puede cambiar comportamiento de clientes que no fijaron versión.
- Mantener versiones indefinidamente acumula deuda y coste.
- A/B routing no estable por usuario contamina comparación.
- Cambios de response schema exigen coordinación con aplicaciones.
- Versionar sin lineage de datos/código no asegura reproducibilidad.

### 3.13 Cuándo utilizarlo

Siempre que un modelo de producción deba actualizarse sin romper consumidores, compararse con versiones previas o probarse sobre tráfico parcial.

### 3.14 Cuándo no utilizarlo

No tratar como nueva versión un cambio de tarea predictiva que altera el significado de la salida y obliga al consumidor a decidir entre productos distintos; crear un modelo nuevo.

### 3.15 Alternativas

| Alternativa | Ventaja | Desventaja | Cuándo preferirla |
| ----------- | ------- | ---------- | ----------------- |
| Servicio ML administrado | Versiones, dependencias y hardware integrados | Menor flexibilidad | Serving estándar |
| Contenedor serverless | Runtime y preprocessing flexibles | Gestión propia | Entorno personalizado |
| TensorFlow Serving/Seldon/MLflow | Open source y control de hardware | Operación propia | On-premises o multi-cloud |
| Multiple signatures | Varias interfaces de un mismo modelo | No separa arquitecturas/versiones | Interfaces permanentes del mismo artefacto |
| Modelo nuevo | Contrato y tarea independientes | Más recursos que gestionar | Cambia la tarea o significado de salida |

### 3.16 Implementación práctica

```text
/models/flight_delay/versions/v1
/models/flight_delay/versions/v2
```

Un gateway asigna de forma estable una fracción de usuarios a `v2`; métricas y errores incluyen la versión. Si una clasificación de “retrasado” pasa a regresión de minutos, se crea otro recurso de modelo.

**Nota práctica de actualización:** los comandos y nombres de AI Platform del libro son históricos; el principio es endpoint/version inmutable, routing explícito y observabilidad por versión.

### 3.17 Ejemplo mínimo conceptual

Modelo XGBoost v1 → TensorFlow v2 conserva inputs/outputs → ambos endpoints coexisten y v2 recibe tráfico gradual → el equipo compara, promueve o revierte sin romper v1.

### 3.18 Aplicación en competencias

**Síntesis práctica derivada del capítulo.** En competiciones, mapear cada experimento/submission a código, split, features y artefacto equivale a versionar. Tiene prioridad alta aunque no haya REST endpoints: evita perder el mejor modelo y permite ensamblar o reproducir una submission. En coding tests basta nombrado consistente y registro de parámetros; desplegar múltiples servicios sería excesivo.

### 3.19 Checklist operativo del patrón

- [ ] Versionar artefacto, datos, código y contrato.
- [ ] No sobrescribir la versión activa.
- [ ] Probar compatibilidad de request/response.
- [ ] Etiquetar métricas por versión.
- [ ] Definir rollout, rollback y deprecación.
- [ ] Crear modelo nuevo si cambia la tarea.
- [ ] Evitar combinaciones innecesarias de versiones y signatures.

### 3.20 Relación con otros patrones

| Patrón relacionado | Tipo de relación | Explicación |
| ------------------ | ---------------- | ----------- |
| Workflow Pipeline | Dependencia | Automatiza construcción, evaluación y despliegue de versiones |
| Transform | Complementario | Cada versión debe incluir sus transformaciones y artefactos |
| Feature Store | Complementario | La versión debe apuntar a definiciones de features compatibles |
| Repeatable Splitting | Complementario | Permite comparar versiones sobre la misma evaluación |

# 4. Reglas de decisión del capítulo

| Situación | Pattern recomendado | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| Training y serving derivan features por rutas distintas | Transform | Captura lógica y artefactos una sola vez | Ajustar estadísticas solo con training |
| Filas correlacionadas cruzan particiones | Repeatable Splitting | Agrupa mediante hash estable | Validar distribución y cardinalidad |
| Producción ya usa un esquema nuevo con poco histórico | Bridged Schema | Combina señal nueva con volumen antiguo | Evaluación solo nueva y fuera del tuning |
| Una predicción necesita eventos recientes | Windowed Inference | Externaliza estado temporal | Evitar incluir presente/futuro |
| El flujo depende de notebooks y pasos manuales | Workflow Pipeline | Declara y orquesta el DAG | No automatizar deployment sin validación |
| Varias soluciones recrean las mismas features | Feature Store | Centraliza cálculo, catálogo y acceso | Garantizar point-in-time correctness |
| Un modelo cambia sin poder romper clientes | Model Versioning | Coexistencia, routing y rollback | Nuevo recurso si cambia la tarea |

# 5. Comparación entre patrones del capítulo

| Pattern | Problema que resuelve | Coste | Beneficio | Complejidad |
| ------- | --------------------- | ----- | --------- | ----------- |
| Transform | Paridad de preprocessing | Bajo–medio | Evita skew y simplifica serving | Media |
| Repeatable Splitting | Leakage e inestabilidad del split | Bajo | Evaluación fiable y comparable | Baja–media |
| Bridged Schema | Escasez tras cambio de esquema | Medio | Usa antes datos enriquecidos | Media |
| Windowed Inference | Estado temporal en producción | Medio–alto | Features/predicciones contextuales reproducibles | Alta |
| Workflow Pipeline | Repetición del proceso completo | Alto inicial | Automatización, aislamiento y lineage | Alta |
| Feature Store | Reuso y consistencia de features | Alto | Escala organizacional y serving dual | Alta |
| Model Versioning | Actualizaciones compatibles | Medio | Rollout, comparación y rollback | Media |

# 6. Riesgos globales del capítulo

- **Data leakage:** dividir filas correlacionadas, calcular embeddings antes del split, usar evaluación para tuning, o hacer joins temporales con información futura.
- **Distribución incorrecta:** hash sobre claves poco diversas, datos puenteados dominantes o imputación global no válida para subgrupos.
- **Validación incorrecta:** evaluación antigua cuando serving usa schema nuevo, random split en escenarios temporales o default cambiado sin comparación.
- **Training-serving skew:** lógica duplicada de preprocessing, ventanas distintas o stores online/offline divergentes.
- **Reproducibilidad incompleta:** fijar solo una seed sin versionar split, datos, artefactos, pipeline y modelo.
- **Complejidad operacional:** streaming, orquestadores, stores dobles y múltiples endpoints añaden puntos de fallo.
- **Dependencia excesiva de infraestructura:** una solución reproducible puede quedar atada a un store, pipeline o servicio concreto.
- **Mantenimiento:** versiones, schemas, features y componentes necesitan política de compatibilidad y retiro.

# 7. Aplicación global en coding tests y datathons

**Síntesis práctica derivada del capítulo.**

### Durante los primeros minutos

Buscar IDs repetidos, fechas, sesiones, fuentes, schemas distintos entre train/test, features derivadas y cualquier variable que pueda usar futuro. Esto activa primero Repeatable Splitting y Transform; si el orden es esencial, Windowed Inference.

### Durante el baseline

Construir una única ruta de transformación, congelar un split por grupo/tiempo y registrar el artefacto. Bridged Schema solo se aplica si hay evidencia clara de evolución de fuente. Guardar configuración y predicciones equivale a una forma ligera de Model Versioning.

### Durante la mejora del modelo

Cachear features caras con principios de Feature Store, agregar rolling features sin mirar el futuro y estructurar etapas repetibles. Comparar modelos siempre sobre el mismo split y la misma transformación.

### Durante validación

Confirmar que grupos correlacionados no cruzan folds, que las estadísticas se ajustan dentro de cada fold, que la evaluación representa el schema de producción y que joins temporales respetan el tiempo del evento.

### Bajo presión de tiempo

- **Alta prioridad:** Repeatable Splitting, Transform y versionado ligero de experimentos.
- **Prioridad media:** Windowed Inference cuando hay tiempo, Bridged Schema cuando train/test reflejan una migración, y estructura de Workflow Pipeline sin infraestructura pesada.
- **Baja prioridad:** desplegar una Feature Store completa o un orquestador distribuido durante una competición corta.

# 8. Checklist global del capítulo

- [ ] Identificar todas las fuentes de aleatoriedad y estado.
- [ ] Diseñar el split antes de ajustar features o embeddings.
- [ ] Capturar transformaciones y artefactos con el modelo.
- [ ] Evitar información futura en ventanas y joins.
- [ ] Hacer explícitos inputs/outputs de cada etapa.
- [ ] Versionar datos, código, features, modelo y contrato.
- [ ] Comparar contra una evaluación estable y representativa.
- [ ] Verificar paridad training-serving.
- [ ] Mantener rollback y lineage.
- [ ] Adoptar solo la infraestructura proporcional al caso.

# 9. Mapeo hacia `knowledge/`

| Pattern / conocimiento | Archivo de destino propuesto | Prioridad |
| ---------------------- | ---------------------------- | --------- |
| Transform | `knowledge/preprocessing/transform.md` | Alta |
| Transform: skew | `knowledge/mlops/training_serving_skew.md` | Alta |
| Repeatable Splitting | `knowledge/validation/repeatable_splitting.md` | Alta |
| Repeatable Splitting: determinismo | `knowledge/reproducibility/deterministic_splits.md` | Alta |
| Bridged Schema | `knowledge/reproducibility/bridged_schema.md` | Media |
| Windowed Inference | `knowledge/time_series/windowed_inference.md` | Media |
| Workflow Pipeline | `knowledge/mlops/workflow_pipeline.md` | Alta |
| Workflow lineage | `knowledge/reproducibility/ml_lineage.md` | Alta |
| Feature Store | `knowledge/feature_engineering/feature_store.md` | Media |
| Feature Store: temporal joins | `knowledge/validation/point_in_time_correctness.md` | Alta |
| Model Versioning | `knowledge/mlops/model_versioning.md` | Alta |

# 10. Índice de patrones reutilizables

| Pattern | Activador principal | Acción principal | Knowledge destino |
| ------- | ------------------- | ---------------- | ----------------- |
| Transform | Inputs ≠ features | Empaquetar lógica y artefactos de transformación | `knowledge/preprocessing/` |
| Repeatable Splitting | Grupos correlacionados o splits cambiantes | Hash estable por grupo o split temporal | `knowledge/validation/` |
| Bridged Schema | Poco dato bajo un schema nuevo | Imputar histórico y evaluar solo en nuevo | `knowledge/reproducibility/` |
| Windowed Inference | Contexto reciente requerido | Externalizar estado de una ventana | `knowledge/time_series/` |
| Workflow Pipeline | Flujo manual o monolítico | Componentizar y orquestar como DAG | `knowledge/mlops/` |
| Feature Store | Features duplicadas y serving dual | Centralizar catálogo y acceso online/offline | `knowledge/feature_engineering/` |
| Model Versioning | Updates con consumidores activos | Desplegar endpoints versionados | `knowledge/mlops/` |

# 11. Trazabilidad

Las páginas siguientes son las páginas impresas del capítulo, no el índice del archivo PDF dividido.

| Pattern / sección del resumen | Sección o páginas del capítulo |
| ----------------------------- | ------------------------------ |
| Propósito y mapa del capítulo | pp. 249–250; resumen pp. 317–318 |
| Pattern 21: Transform — problema y solución | pp. 250–252; Figura 6-1, pp. 250–251 |
| Transform — Keras y `tf.transform` | pp. 252–257; Figura 6-2, p. 253 |
| Transform — texto, imágenes y alternativas | pp. 257–258 |
| Pattern 22: Repeatable Splitting — problema y hash | pp. 258–260 |
| Repeatable Splitting — variantes y muestreo | pp. 260–263 |
| Repeatable Splitting — secuencial, estratificado y no estructurado | pp. 263–265 |
| Pattern 23: Bridged Schema — problema, métodos probabilístico/estático | pp. 266–268 |
| Bridged Schema — tamaño de evaluación y volumen puenteado | pp. 268–270; Figuras 6-3 y 6-4, pp. 269–270 |
| Bridged Schema — alternativas y nuevas features | pp. 271–273 |
| Pattern 24: Windowed Inference — problema y solución | pp. 273–277; Figura 6-5, p. 274 |
| Windowed Inference — eficiencia, online update y SQL | pp. 277–280; Tabla 6-1, p. 280 |
| Windowed Inference — secuencias, stateful features y batching | pp. 280–282 |
| Pattern 25: Workflow Pipeline — problema y solución | pp. 282–289; Figuras 6-6 a 6-8, pp. 283, 286, 288 |
| Workflow Pipeline — custom components, CI/CD y alternativas | pp. 289–294; Figuras 6-9 a 6-11, pp. 291, 293–294 |
| Pattern 26: Feature Store — problema, solución y arquitectura | pp. 295–300; Figuras 6-12 a 6-14, pp. 297–299 |
| Feature Store — ingestión y recuperación | pp. 300–306; Tabla 6-2, p. 301; Figura 6-15, p. 304 |
| Feature Store — por qué funciona y alternativas | pp. 306–310; Figuras 6-16 a 6-18, pp. 307–308 |
| Pattern 27: Model Versioning — problema y solución | pp. 310–314; Figura 6-19, p. 313 |
| Model Versioning — tracking de datos, alternativas y signatures | pp. 314–317 |
| Model Versioning — modelo nuevo vs versión | p. 317 |
