# Contrato de decisiones v1 — subfases 2A–2D

Entrada: `python research_agent.py --decisions-json examples/decisions.pending.json --csv data.csv --target resultado --task binary_classification --audit-only`

El ejemplo es una **plantilla pendiente de revisión**, no una aprobación humana.
Las rutas se resuelven desde el directorio de ejecución. `schema_version: 1` es
obligatorio. Los demás campos pueden omitirse; `null` representa una decisión sin
respuesta en los campos escalares. No se infieren confirmaciones del CSV.

`configuration` usa los nombres y campos de `ResearchRequest`: `csv_path`, `target`,
`task_type`, `metric`, `seed`, `context`, `metric_context`, `target_unit`,
`split_strategy`, `group_column`, `time_column`, `test_size`, `time_format`,
`validation_strategy`, `observation_structure`, `cv_folds`, `max_cv_fits`.
Se reutiliza esa solicitud como fuente única de configuración; no se mantienen
dos configuraciones internas. JSON aporta valores cuando CLI no los declara;
si ambos declaran valores diferentes se rechaza la entrada, incluso si CLI declara
un valor igual al predeterminado. `null` no anula un argumento explícito.
Los argumentos previos conservan sus valores predeterminados.

Campos adicionales:

| Campo | Contenido |
| --- | --- |
| `review_binding` | Hashes de CSV, configuración relevante y señales; obligatorio para entrenar; lo genera la auditoría |
| `dataset_sha256` | SHA-256 de los bytes del CSV; null si aún no se revisó su identidad |
| `positive_class` | Etiqueta como texto, normalizada con `strip()`, solo clasificación binaria |
| `row_meaning` | Qué representa una fila |
| `prediction_scenario` | Momento o escenario de predicción |
| `metric_reason` | Justificación humana de la métrica |
| `predictor_availability` | Mapa columna → `available`, `unavailable`, `unknown` |
| `excluded_columns` | Mapa columna → motivo; null deja el motivo pendiente |
| `responses` | Lista con `column_name`, `rule_id` o `signal_id` opcionales, `decision` (`keep`, `exclude`, `pending`), `reason` y `column_meaning` al conservar una columna señalada |

`observation_structure` admite `independent`, `grouped`, `temporal`, `unknown`.
Una respuesta sin `rule_id` ni `signal_id` se refiere a la columna; con referencia,
a una señal emitida por la auditoría actual. Si se proporcionan ambos identificadores,
deben referirse a la misma señal y columna. Una señal no evaluada no puede confirmarse con esta carga.
Se rechazan campos, columnas y señales desconocidos, tipos incorrectos, versiones
no admitidas, claves duplicadas, identidad CSV distinta y contradicciones directas.
Una justificación ausente es una decisión pendiente, no un error de formato.

El resultado JSON incluye `decision_contract`: configuración efectiva,
`declared_fields` (permite distinguir declaraciones de defaults), identidad
declarada y observada del CSV, hash canónico de configuración y `pending_decisions`.
Las respuestas se proyectan también al campo existente `DatasetAudit.human_review`;
su estado distingue `pending`, `resolved` y `not_required`. Solo describe la revisión
de señales bajo la política 2C, no certifica todas las decisiones ni ausencia de fuga.
Este resultado es un artefacto de salida; el archivo de entrada usa únicamente
los campos documentados arriba.

La auditoría necesita CSV, objetivo y tarea explícita. Puede continuar sin métrica,
motivo, clase positiva o declaraciones de disponibilidad. Si el split no es viable,
se conserva la auditoría global y se registran limitaciones para señales de train.
`completed` significa auditoría ejecutada, no decisiones completas ni entrenamiento
autorizado. Formato inválido devuelve código 1; faltan datos mínimos, código 2.

## Validación semántica 2B

Para entrenar, el mismo comando se ejecuta sin `--audit-only`, con decisiones
completas. También las llamadas Python a `run_baseline`/`run_model_comparison`
exigen las condiciones semánticas. Los argumentos CLI existentes se conservan,
pero **ya no es suficiente omitir el contrato para entrenar**.

Se requiere significado de fila, escenario de predicción y motivo de métrica como
texto no vacío. Una cadena de espacios se rechaza. La estructura `unknown` requiere
respuesta; `split_strategy` debe estar declarado explícitamente en JSON o CLI.
Se conservan las restricciones de grupos, tiempo y CV. No se cambia un protocolo
inválido a aleatorio. La auditoría incompleta omite relaciones predictor-target si
falta el protocolo o la estructura y explica el análisis pendiente en JSON y Markdown.

Clasificación exige dos clases válidas y una `positive_class` observada. Tanto las
etiquetas del CSV como la declaración se normalizan con `str.strip()`: se quitan
espacios exteriores, se conservan mayúsculas, espacios interiores y representación
textual (`01` no equivale a `1`). Los marcadores faltantes siguen la lectura existente
(vacío, na, n/a, nan, null, none, <na>, sin distinguir mayúsculas): no son clases válidas.
La clase declarada se codifica como 1 y la otra como 0; el orden alfabético no decide
la clase positiva. Se usa el mismo mapeo para constante, modelo lineal, holdout y CV.
El resultado registra `positive_class` y `class_mapping`; precision/recall/F1 se
refieren al valor 1. En regresión el target debe ser numérico y finito y no admite
clase positiva. Los errores no generan métricas.

La selección existente de métricas continúa proporcionando recomendaciones.
`metric_decision` registra métrica efectiva, origen, motivo técnico, motivo humano y
estado `recommendation` o `confirmed`. Una recomendación por `metric_context` o el
F1 predeterminado no es aprobación humana: se necesita `metric` explícita y
`metric_reason`. La selección queda visible aun cuando el entrenamiento esté bloqueado.

Cada predictor efectivo necesita `available` explícito para el escenario declarado.
`unknown`, `unavailable` o ausencia bloquean su uso. Nombre, tipo y correlación no
confirman disponibilidad. Una exclusión justificada retira la columna antes del
preprocesamiento; ya no se exige disponibilidad predictiva a la columna excluida.

Códigos: 0 para auditoría/entrenamiento completado, 2 para decisiones pendientes o
solicitudes no admitidas, 1 para contrato/CSV inválido o fallo de ejecución.
La auditoría `completed` no significa entrenamiento autorizado: consulte
`semantic_issues`, `pending_decisions` y `signal_evaluation_limitations`.

## Revisión humana 2C: política v1

Cada `ColumnReviewSignal` contiene `signal_id` y `review_required`. La referencia
`signal-v1-<sha256>` se deriva de columna y regla (JSON UTF-8) y es estable para esa
pareja. CSV y configuración se identifican por separado en el contrato: el ID no
certifica que una respuesta antigua sea válida para otros datos.

| Señales | Política para entrenar |
| --- | --- |
| `possible_target_proxy` evaluable: nombre de riesgo, correlación y pureza categórica | Respuesta obligatoria |
| Identificadores, fechas, cardinalidad, constantes, extremos | Informativas; no excluyen automáticamente |
| `not_evaluable`, incluida muestra train insuficiente | Informativas: expresan límites, no una fuga detectada |

No existe confirmación global. Cada respuesta identifica una columna y, opcionalmente,
una señal. Conservar una columna señalada exige `reason` y `column_meaning` no vacíos,
además de `predictor_availability[columna] = "available"`. El sistema comprueba que
hay una explicación escrita; no certifica su veracidad ni que tenga suficiente
calidad de dominio. Un “confirmado” genérico en `reason` no sustituye el significado
ni puede sobrescribir disponibilidad `unknown`/`unavailable`.

La respuesta por columna cubre sus señales; una respuesta específica tiene prioridad
sobre esa respuesta general. Por ejemplo, una señal específica `pending` sigue
bloqueando aunque exista `keep` por columna. `keep` y `exclude` sobre una misma
columna son contradictorios y se rechazan. Las referencias duplicadas también se
rechazan, incluso cuando una usa `rule_id` y otra `signal_id`.

`exclude` con motivo, ya sea en `responses` o en `excluded_columns`, excluye toda
la columna y resuelve su uso como predictor en todas sus señales. No elimina las
señales originales ni las convierte en fugas confirmadas. Sin motivo, la exclusión
sigue pendiente y no se aplica. Excluir todos los predictores bloquea con explicación.
El target no puede conservarse ni excluirse como predictor porque nunca lo es.

La lista efectiva conserva el orden del CSV y se calcula una vez para el contexto
de revisión (datos, configuración y decisiones); se reutiliza en todos los candidatos
y folds antes de `_encode` (imputación, escalado y one-hot). Las filas originales se
conservan para el split: una columna excluida puede seguir siendo `group_column` o
`time_column`. Esto no autoriza a emplearla como predictor.

`human_review` registra candidatas, exclusiones y motivos, predictores efectivos,
resolución por señal, columnas de partición y problemas pendientes. Cada candidato
registra `predictor_columns`; JSON y Markdown conservan las señales originales.
`--audit-only` muestra esta resolución sin ajustar modelos.

Las heurísticas con target conservan la partición preliminar de revisión existente:
solo train del holdout declarado (estratificado, por grupo o temporal). Se registran
estrategia, número, índices de filas de datos desde cero y SHA-256 de esos índices.
**No se amplían a validación/test ni se unen los train de los folds.** Si se solicita
CV, esa revisión preliminar no se presenta como una revisión independiente por fold.
Si falta protocolo, no se inventa: el análisis de relaciones queda pendiente.
La revisión utiliza datos de desarrollo y **no convierte el score en una evaluación
final independiente**. La correlación y las demás señales siguen siendo sospechas.

Ejemplo ilustrativo de respuesta (debe sustituirse por una explicación real):

```json
{
  "column_name": "temperatura_inicial",
  "decision": "keep",
  "reason": "El sensor registra el valor antes de iniciar el proceso objetivo",
  "column_meaning": "Temperatura medida al recibir la pieza"
}
```

Además debe existir `"temperatura_inicial": "available"` en
`predictor_availability`. Este ejemplo no confirma ninguna columna del CSV del usuario.

## Recorrido completo 2D

1. Audite indicando target, tarea, métrica y protocolo previstos. Genere un archivo
   nuevo con `--write-decisions-template`. La opción implica auditoría sin retrieval,
   modelo de lenguaje, preprocesamiento de modelos ni entrenamiento.
2. Revise el JSON: complete significado de fila, escenario, clase positiva si
   corresponde, motivo de métrica, disponibilidades y respuestas requeridas. La
   plantilla no confirma ninguno de estos datos por la persona.
3. Cargue el archivo revisado con `--decisions-json`. `run_research`, `run_baseline`
   y `run_model_comparison` usan la misma validación previa y un CSV recién leído.
   Los objetos audit/rows suministrados por Python no permiten omitir comprobaciones;
   rows debe coincidir con el snapshot actual. Las funciones con prefijo `_` son
   internas, no una API de entrenamiento soportada.
4. Solo con revisión válida se ejecutan los candidatos. En entrenamiento bloqueado
   se devuelve `needs_input`, sin baseline ni comparación, con acciones concretas.

Ejemplo de CLI (sustituya las rutas por su CSV y revisión):

```powershell
python research_agent.py --csv data.csv --target target --task binary_classification --metric f1 --split-strategy random --observation-structure independent --write-decisions-template decisions.pending.json --output-dir .veronica/audit
# Revise el archivo y guarde una copia como decisions.reviewed.json.
python research_agent.py --decisions-json decisions.reviewed.json --output-dir .veronica/training
```

La auditoría puede terminar con código 0 y `completed` aunque `review_status` sea
`needs_input`: solo terminó la inspección. El comando de entrenamiento devuelve 2
con `needs_input` para decisiones pendientes. Solicitudes no admitidas conservan
`unsupported`; formato/CSV inválidos producen un fallo sin métricas.

`review_binding` contiene `dataset_sha256`, `configuration_sha256` y `signals_sha256`.
La configuración incluye todos los campos efectivos de `ResearchRequest` salvo
la ruta del CSV: target, tarea, métrica, semilla, estructura, split, columnas de grupo/
tiempo, tamaños y CV, entre otros. Las señales incluyen evidencia, alcance e índices
de train. La verificación se vuelve a ejecutar en cada entrada pública de entrenamiento.

Un cambio de bytes, configuración o señales invalida conservadoramente las
confirmaciones: se solicita una nueva auditoría y plantilla, se marca `binding_status`
como `stale` y se enumeran `invalidated_confirmations`. No edite los hashes para
forzar aceptación: son comprobaciones de coherencia, no firmas ni prueba de autoría.
Si inicialmente faltaban tarea/protocolo/métrica, complete esa configuración y vuelva
a generar una plantilla antes de resolver las decisiones. Cambiar justificaciones,
disponibilidad o exclusiones no altera el vínculo al contexto, pero vuelve a ejecutar
las validaciones semánticas y de revisión.

La plantilla se crea con modo exclusivo: ni una pendiente ni una completada se
sobrescriben. Los informes sí describen la última solicitud en el directorio de salida:
un bloqueo o error de carga/conflicto CLI reemplaza JSON/Markdown con un resultado sin
métricas anteriores. `review_summary` en JSON se reproduce literalmente como bloque
JSON en Markdown, con contexto, clase/mapeo, métrica/motivo, disponibilidad, exclusiones,
predictores, protocolo solicitado/efectivo, hashes, versión, estado y pendientes.
Un protocolo efectivo nulo significa que no hubo entrenamiento; el split preliminar
para revisar señales se informa por separado en su alcance.

### Demostración reproducible ejecutada

```powershell
python examples/demo_decision_workflow.py
```

Genera exclusivamente datos y respuestas **sintéticos**, simulando al autor del caso;
no confirma datos reales. En la ejecución registrada:

| Paso | Código | Estado | Baseline |
| --- | --- | --- | --- |
| Auditoría y plantilla | 0 | completed | no |
| Plantilla pendiente | 2 | needs_input | no |
| Revisión sintética completada | 0 | completed | sí |
| CSV modificado después | 2 | needs_input | no |

Comandos exactos, logs, JSON y Markdown quedan en `.veronica/demo_2d/` (ignorado por
Git); `execution_summary.json` guarda los argumentos y resultados. El CSV termina
modificado intencionalmente: reutilizar aquella revisión debe seguir bloqueado.
Para repetir, indique un directorio nuevo, por ejemplo:
`python examples/demo_decision_workflow.py .veronica/demo_2d_repeat`.

### Limitaciones que permanecen

Las métricas son preliminares: no existe una reserva final independiente para evaluar
el modelo seleccionado. La revisión con target sigue usando el holdout preliminar de
train, incluso cuando después se usa CV; no se presenta como revisión independiente
por fold. Los textos humanos se validan estructuralmente, no se certifica su veracidad.
No hay autenticación de revisores ni firma de decisiones. Los informes se escriben como
dos archivos locales, sin transacción entre ambos ante interrupciones del sistema.
Se mantiene NumPy, sin scikit-learn, Nebius ni interfaz web.
