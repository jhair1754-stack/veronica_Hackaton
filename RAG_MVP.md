# FASE 4 — RAG MVP local

Retriever local para los Markdown que estén bajo `knowledge/` y `playbook/` o `playbooks/`. No carga `sources/`, `tmp/` ni documentos fuera de esas carpetas. No cambia los documentos fuente.

## Uso

Requiere Python 3.10 o posterior y NumPy. Instala la dependencia con `python -m pip install -r requirements-rag.txt`.

```powershell
python rag_mvp.py build
python rag_mvp.py search "GroupKFold para evitar leakage entre entidades" --top-k 5
python rag_mvp.py search "métricas de clasificación" --type knowledge --topic metrics
python rag_mvp.py evaluate
python rag_mvp.py evaluate --per-section-limit 1
python rag_mvp.py evaluate --per-section-limit 0
```

`build` siempre reconstruye el índice desde los Markdown. La búsqueda compara un manifiesto local de hashes y lo regenera si falta o cambió algún documento. `evaluate` registra los resultados JSON y Markdown en `.veronica/`.

## Reproducción y comparación

Desde la raíz del proyecto, instala los requisitos una vez y ejecuta los pasos en este orden:

```powershell
python -m pip install -r requirements-rag.txt
python rag_mvp.py build
python rag_mvp.py search "GroupKFold para evitar leakage entre entidades" --top-k 5
python rag_mvp.py evaluate
python rag_mvp.py compare
```

`build` reconstruye `.veronica/rag_index.json` desde `knowledge/` y `playbook(s)/`. La búsqueda muestra los cinco primeros fragmentos. `evaluate` vuelve a comprobar que el índice corresponda al manifiesto actual y escribe `.veronica/retrieval_evaluation.json` y `.veronica/retrieval_evaluation.md`; ejecutarlo después de `build` evita depender de un índice anterior. La comparación vuelve a evaluar y escribe `.veronica/retrieval_comparison.json`.

`evaluate` devuelve código 1 si alguna expectativa falla, aunque deja los informes completos. Revisa los casos individuales además de las métricas y ejecuta `compare` como comando separado.

La referencia inicial está en `rag_reference_baseline.json`, fuera de `.veronica/`. Incluye métricas agregadas, resultado y ranking por consulta, posiciones relevantes, configuración efectiva, versión de Python y NumPy, SHA-256 de las consultas y manifiesto/hash de documentos indexados. El reporte de evaluación también registra estos datos. Los filtros negativos se comparan en una sección separada y no entran en Recall@k ni MRR.

La referencia `rag_reference_baseline.json` está versionada en la raíz. Los informes generados en `.veronica/` están excluidos por `.gitignore`.

`compare` muestra Recall@k, MRR, aciertos por consulta y posiciones relevantes. `retrieval_changed` indica que cambió si el fragmento relevante aparece dentro del mayor k; `reorder_only` indica que ese acierto se mantuvo mientras cambió el orden de los fragmentos. Revisa `compatibility` antes de interpretar los deltas: corpus, consultas, k o configuración distintos se señalan como no equivalentes. La configuración incluye fragmentación, tokenización, vocabulario, BM25/LSA y pesos, filtros usados, diversidad y deduplicación; por eso cambios de versión de Python/NumPy también quedan visibles.

Una ejecución normal nunca reemplaza la referencia. Solo actualízala después de revisar deliberadamente los cambios con:

```powershell
python rag_mvp.py baseline --update
```

Este comando evalúa el estado actual y reemplaza explícitamente `rag_reference_baseline.json`; conserva la medición resultante como una nueva revisión del archivo cuando se registre en control de versiones. No uses esa actualización para comparar: primero ejecuta `compare` contra la referencia anterior.

## Diversidad de resultados

Después de calcular las puntuaciones y ordenar todos los candidatos, `search()` limita a **dos fragmentos por sección** entre los resultados devueltos. El límite predeterminado conserva partes complementarias de una sección sin dejar que esta domine la lista. La identidad de sección es el par `(source_path, section_path)`: títulos iguales en documentos diferentes no se agrupan. El filtro recorre el ranking ordenado y conserva el orden relativo de los fragmentos que pasan.

El argumento `per_section_limit` permite cambiar el límite; `None` desactiva la política. La CLI acepta `--per-section-limit 1` o `--per-section-limit 2`; `--per-section-limit 0` desactiva el límite. `--deduplicate-exact` omite únicamente textos idénticos. Es una opción independiente y queda desactivada por defecto.

```powershell
python rag_mvp.py search "validación cruzada K-Fold" --per-section-limit 1
python rag_mvp.py search "validación cruzada K-Fold" --per-section-limit 0 --deduplicate-exact
python rag_mvp.py evaluate --per-section-limit 2
```

El evaluador compara el ranking sin límite, deduplicación exacta, límites 1 y 2, con las mismas consultas y valores de `ranking_k`. Mantiene el criterio existente de relevancia por fragmento (ruta, sección y todos los términos en un mismo fragmento), y calcula las métricas sobre el ranking filtrado; no reúne evidencia entre fragmentos. Los informes muestran Recall@k, MRR, secciones distintas y fragmentos repetidos por consulta y en agregado. La línea base inicial versionable queda en `rag_reference_baseline.json`; el archivo anterior bajo `.veronica/`, si existe, es un artefacto local histórico y no se usa como referencia de comparación.

## Consultas de referencia y evaluación

`rag_reference_queries.json` contiene un objeto con `ranking_k` y una lista `cases`. `ranking_k` es una lista explícita de enteros positivos para Recall@k (por defecto se documenta `[1, 3, 5, 10]`). Cada caso tiene un `id`, `query` y puede añadir `document_type` (`knowledge` o `playbook`) y `topic`, que se aplican como filtros reales de `search()`.

Los casos positivos incluyen `expected_path`, `expected_section` y `expected_terms`. La ruta debe coincidir exactamente; el texto de sección esperado debe aparecer en la ruta de encabezados del fragmento; y **cada** término debe aparecer en el texto de ese mismo fragmento. El fragmento solo es relevante cuando cumple las tres condiciones a la vez. No se agregan términos de otros resultados ni de otros fragmentos del mismo documento. `expected_terms` puede representar términos técnicos exactos o conceptos expresados mediante vocabulario presente en el pasaje esperado.

Los casos negativos omiten esos tres campos y usan `expected_zero_results: true`. Se informa por separado si `search()` devolvió cero resultados y si ese hecho coincide con lo esperado. Estos casos no entran en las métricas de ranking ni en su denominador.

La evaluación usa el ranking devuelto por `search()`. El `top_k` evaluado es al menos el mayor k declarado (o un `top_k` mayor configurado en el caso):

- **Recall@k:** proporción de consultas positivas con un fragmento relevante entre las primeras k posiciones. Como cada caso define un fragmento objetivo, es 1 si aparece y 0 si no.
- **MRR:** media del recíproco de la posición del primer fragmento relevante en los casos positivos; si no aparece, aporta 0.
- El JSON y Markdown presentan valores agregados de Recall@k y MRR, resultados de cada caso, ranking, secciones, puntuaciones y marcas de relevancia. Los filtros negativos tienen un resumen de cero resultados independiente.

Ejecuta `python rag_mvp.py evaluate` para regenerar `.veronica/retrieval_evaluation.json` y `.veronica/retrieval_evaluation.md`. La salida conserva la lista de k y el detalle suficiente para interpretar la posición y el resultado de cada consulta. El resultado no modifica `rag_reference_baseline.json`.

## Pruebas unitarias

Ejecuta la suite aislada con `python -m unittest discover -s tests -v`. Las pruebas crean documentos pequeños en directorios temporales y no escriben en el corpus ni en `.veronica/`. Una consulta sin términos presentes en el vocabulario indexado devuelve `[]`.

## Revisión de trazabilidad

Para auditar enlaces locales, anclas Markdown, citas de capítulo/sección, metadatos, mapeos del catálogo y el inventario de playbooks, ejecuta desde la raíz: `python knowledge_traceability.py`. El comando devuelve estado distinto de cero ante errores verificables. Las citas a un archivo completo sin ancla se informan como ambiguas; los destinos `target_knowledge_files` ausentes se informan como propuestas futuras y no como errores. El auditor conserva las listas YAML válidas y las valida como listas; la lectura escalar limitada de `rag_mvp.py` es independiente. Usa `python knowledge_traceability.py --verbose` para ver cada referencia y destino.

Los playbooks `playbook/classification.md`, `playbook/ml_workflow.md`, `playbook/regression.md` y `playbook/time_series.md` se comprobaron vacíos y permanecen pendientes. El catálogo y `sources/README.md` explican el conocimiento o la decisión operativa que falta. No generan fragmentos en el índice.

## Método

- El fragmentador conserva cada sección y su ruta de encabezados; agrupa párrafos, tablas o bloques de código hasta 1.400 caracteres y corta bloques mayores.
- La tokenización normaliza Unicode y acentos, y reduce un conjunto pequeño de sufijos españoles productivos (`-ar/-er/-ir` y `-ación/-aciones`) a claves léxicas comunes. Por ejemplo, `eliminar` y `eliminación` comparten clave. Es una heurística local para recuperación, no un analizador lingüístico; la evaluación sigue comprobando los términos esperados literalmente dentro del mismo fragmento.
- BM25 cubre coincidencias léxicas; una proyección LSA de los vectores TF-IDF aporta similitud semántica local. NumPy es la única dependencia adicional; los componentes semánticos se aprenden al construir el índice, sin modelo remoto.
- Filtros disponibles: tipo (`knowledge` o `playbook`) y tema. `knowledge_area`, `topic` o `subject` se toman del YAML solo cuando hay un valor escalar. Los campos ausentes no se inventan.
- Cada resultado contiene ID estable basado en ruta, encabezados y texto, posición del fragmento, tipo, título, encabezados, ruta de origen y puntuaciones léxica, semántica y combinada. La puntuación sirve para ordenar, no expresa confianza probabilística.
- El índice está en `.veronica/rag_index.json`, fuera de las carpetas documentales. No depende de un índice anterior y se reemplaza en cada reconstrucción.

## Cobertura observada

Se encontraron 12 Markdown no vacíos en `knowledge/` y 3 no vacíos en `playbook/`: 15 documentos y 530 fragmentos. Los cuatro playbooks pendientes son `playbook/classification.md`, `playbook/ml_workflow.md`, `playbook/regression.md` y `playbook/time_series.md`; se mantienen en 0 bytes y no generan fragmentos. Ningún archivo de `sources/` o `tmp/` se incorpora.

Las consultas de referencia, derivadas de estos documentos, se ejecutan con `python rag_mvp.py evaluate`. Incluyen consultas directas, paráfrasis, vocabulario técnico, filtros por tipo/tema y filtros sin resultados esperados. El detalle, métricas y posiciones queda en `.veronica/retrieval_evaluation.md` y `.veronica/retrieval_evaluation.json`.

## Límites

LSA aprende relaciones del corpus indexado; no equivale a embeddings de un modelo lingüístico y puede rendir peor en paráfrasis alejadas del vocabulario. Los documentos sin metadata de tema no se pueden filtrar por tema. La extracción de YAML es deliberadamente limitada a escalares de una sola línea. Los enlaces Markdown relativos se resuelven desde el archivo que los contiene; las anclas se basan en los encabezados Markdown.

## Fase 5 — agente local para clasificación y regresión tabular

`research_agent.py` implementa una secuencia local: valida la solicitud, audita un CSV, recupera fragmentos mediante la interfaz actual `rag_mvp.search()`, prepara un plan determinista, compara un predictor constante y un modelo lineal en clasificación binaria o regresión y revisa resultado y citas. No hay proveedor de LLM configurado en este proyecto; el planificador actual implementa una interfaz `Planner` inyectable sin llamadas remotas. Una futura integración debe implementar esa interfaz, aceptar solo datos estructurados de auditoría/evidencia, leer credenciales de variables de entorno y evitar guardar credenciales o prompts con datos privados. La suite no usa modelo ni red.

Para conectar un proveedor real en el futuro, el adaptador podrá recibir `VERONICA_LLM_PROVIDER`, `VERONICA_LLM_BASE_URL` y `VERONICA_LLM_API_KEY` desde el entorno (los nombres son una convención propuesta, no una configuración que este agente lea actualmente). La dependencia SDK del proveedor se declarará por separado solo cuando se implemente ese adaptador. Nunca se escriben estos valores en código ni en artefactos.

No se añadió una dependencia: NumPy ya forma parte de requirements-rag.txt; CSV, serialización, hash y split usan la librería estándar. El preprocesamiento usa mediana y estandarización numérica, moda categórica y codificación one-hot, ajustados exclusivamente con las filas de entrenamiento de cada partición. Por defecto, clasificación conserva holdout estratificado y regresión holdout aleatorio; las columnas nunca eligen una estrategia. La solicitud explícita group asigna grupos completos; time corta cronológicamente con timestamps empatados completos. Los protocolos registran semilla y tamaños solicitados/reales. Cada corrida compara dos candidatos fijos: la clase más frecuente aprendida en train frente a regresión logística L2 para clasificación; la media de train frente a ridge con alpha=1.0 para regresión. No se ajustan hiperparámetros ni umbrales. El target de regresión se audita con estadísticas agregadas calculadas desde el CSV.

### Ejecución

```powershell
python -m pip install -r requirements-rag.txt
python research_agent.py --csv .\datos.csv --target etiqueta --task binary_classification --metric f1 --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric-context large_errors_cost_more --seed 42
python research_agent.py --csv .\clientes.csv --target renovacion --task binary_classification --metric f1 --split-strategy group --group-column cliente_id --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --split-strategy time --time-column fecha --seed 42
```

Métricas admitidas: clasificación `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`; regresión `mae`, `rmse`, `r2`. Métrica no compatible con la tarea devuelve `unsupported`; una solicitud incompleta devuelve `needs_input`; error del archivo, split o revisión de validez devuelve `failed`. R² puede ser negativo. Si el target evaluado es constante y R² no se define, el agente devuelve `failed` con explicación; no inventa un score. Los códigos de salida de CLI son 0 para `completed`, 2 para `needs_input`/`unsupported` y 1 para `failed` o error de guardado.

La opción --metric es opcional: clasificación conserva F1; regresión exige métrica u objetivo estructurado. --context es texto libre y no elige protocolo ni métrica. --target-unit declara opcionalmente la unidad del target.

### Estrategias explícitas de validación

random conserva el comportamiento previo: clasificación estratificada y regresión aleatoria. group exige --group-column y mantiene cada grupo entero a un lado; valida que la intersección sea vacía. No garantiza estratificación por clase. time exige --time-column y elige un corte entre timestamps distintos; garantiza max(time(train)) < min(time(validation)) y mantiene juntas las filas empatadas.

#### Holdout y validación cruzada de candidatos

`--validation-strategy holdout` es el valor predeterminado para conservar compatibilidad con las solicitudes existentes. Ejecuta una sola partición, con `--test-size` y `--seed`; el resultado registra el protocolo exacto, el hash SHA-256 del CSV, la cantidad de filas y un score. Una sola partición no mide la variabilidad entre particiones y su score no garantiza generalización. Si ese holdout representa el escenario final de producción, debe mantenerse reservado y no reutilizarse para elegir configuración.

`--validation-strategy cross_validation` solicita K-fold explícitamente y requiere `--observation-structure independent`. VERONICA nunca infiere independencia por ausencia de una columna de grupo/fecha ni por nombres de columnas. Para grupos use el protocolo de holdout existente `--split-strategy group --group-column ...`; para tiempo use `--split-strategy time --time-column ...`. K-fold aleatorio ordinario puede mezclar entidades o períodos relacionados entre train y validación, por lo que no se admite con estructura `grouped` o `temporal`. Si se solicita CV sin declarar independencia devuelve `needs_input`; si la estructura es incompatible informa la limitación y no cae silenciosamente a holdout.

`--validation-strategy auto` es determinista: elige K-fold solo si se declara `independent`, hay al menos 20 filas, existen por lo menos 2 folds y el presupuesto permite ajustarlos; en caso contrario conserva holdout e informa el motivo. K solicitado se limita, únicamente en `auto`, por el total de filas, el soporte de la clase minoritaria y `max_cv_fits`; la reducción y su causa quedan en el resultado. El valor predeterminado de K es 5 y el límite predeterminado de ajustes CV es 5. Una solicitud explícita de CV no reduce K ni cambia a holdout si no es factible: informa el error. F1, accuracy, balanced accuracy, precision y recall se calculan en cada fold conservando su semántica; balanced accuracy requiere ambas clases en cada validación. MAE y RMSE requieren folds no vacíos. R² requiere target no constante en cada fold; si no está definido en alguno, se informa el fold y no se omite ni inventa el score. Elija MAE/RMSE o una configuración factible.

La comparación genera las particiones una vez y comparte los mismos índices train/validación y su orden entre ambos candidatos. En cada fold, imputación, escalado y codificación se ajustan solo con train; la validación usa esas transformaciones. Holdout mantiene exactamente el protocolo solicitado (aleatorio/estratificado, grupos o tiempo); la CV actual se acepta solo para independencia declarada y no sustituye un protocolo explícito por grupos o tiempo. Ningún candidato recibe métricas del otro durante el ajuste.

El JSON conserva el campo histórico `baseline` como el candidato lineal, y agrega `model_comparison` con esquema propio versión 1. El esquema del informe JSON es versión 3; los lectores pueden seguir tratando informes previos versión 2 como informes de baseline sin comparación. Para cada candidato se registran identificador, familia, tarea, parámetros, preprocesamiento, protocolo, identificadores de partición con huellas de índices, score por fold, promedio y desviación para CV, fallos, tiempo transcurrido y cantidad de ajustes. El tiempo total suma ajuste, predicción y cálculo de score del modelo en segundos; excluye el preprocesamiento compartido y es una medida local, no un benchmark entre máquinas.

La métrica primaria gobierna la selección según su dirección (mayor es mejor para F1/accuracy y menor para MAE/RMSE). CV reporta promedio no ponderado y desviación estándar poblacional `ddof=0`, además de mínimo/máximo. La agregación no ponderada describe el promedio de scores por fold, no el score combinado de todas las predicciones. Un empate exacto favorece al predictor constante por simplicidad. Si falta el score de un candidato/fold, se conserva el error y no se declara ganador. Una diferencia pequeña no prueba significancia estadística.

El flujo actual no separa un holdout final de la validación utilizada para comparar: el informe declara esa limitación y no afirma que hubo evaluación final. Si en el futuro se añade un holdout final, debe evaluarse una sola vez después de seleccionar el procedimiento y no influir en la selección. La comparación es preliminar para el dataset y protocolo actuales; el score no garantiza generalización y la selección repetida puede sobreajustar la validación.

Ejemplos CV explícitos:

```powershell
python research_agent.py --csv .\datos.csv --target etiqueta --task binary_classification --metric f1 --validation-strategy cross_validation --observation-structure independent --cv-folds 5 --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --validation-strategy cross_validation --observation-structure independent --cv-folds 5 --max-cv-fits 5 --seed 42
python research_agent.py --csv .\datos.csv --target etiqueta --task binary_classification --metric f1 --validation-strategy auto --observation-structure independent --cv-folds 5 --seed 42
```

La opción `--observation-structure` acepta `unknown` (predeterminado), `independent`, `grouped` o `temporal`. `unknown` conserva holdout salvo solicitud explícita de CV, que necesita respuesta del usuario. `--cv-folds` y `--max-cv-fits` son opcionales; cuando se solicita CV se usan 5 si se omiten. El límite se aplica al número de folds por candidato; el costo total es dos ajustes por fold porque hay dos candidatos fijos. En los resultados, `baseline.score` es el score lineal del holdout o el promedio de sus folds; `baseline.validation.folds` conserva sus scores por fold y `model_comparison.candidates` contiene ambos candidatos y el resumen agregado. La variabilidad entre folds no debe interpretarse como intervalo de confianza.

Ejemplos de comparación por defecto (no hace falta activar una opción adicional):

```powershell
python research_agent.py --csv .\clientes.csv --target renovacion --task binary_classification --metric f1 --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --seed 42
```

Para reproducir una corrida, conserve el CSV original, el valor de `--seed`, el protocolo y sus columnas de grupo/tiempo, además del JSON generado. El informe incluye hash del CSV e identificadores de partición; no incluye una copia del dataset ni valores individuales.

Ejemplos válidos:

```powershell
python research_agent.py --csv .\clientes.csv --target renovacion --task binary_classification --metric f1 --split-strategy group --group-column cliente_id --test-size 0.25 --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --split-strategy time --time-column fecha --test-size 0.2 --seed 42
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --split-strategy time --time-column fecha_local --time-format "%d/%m/%Y" --seed 42
```

Ejemplos inválidos:

```powershell
# needs_input: falta --group-column
python research_agent.py --csv .\clientes.csv --target renovacion --task binary_classification --split-strategy group
# needs_input: falta --time-column
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric rmse --split-strategy time
# needs_input: una columna no elige estrategia
python research_agent.py --csv .\clientes.csv --target renovacion --group-column cliente_id
```

--test-size acepta una fracción estrictamente entre 0 y 1 y su valor predeterminado es 0.2. --time-format acepta datetime.strptime y solo con time. Fechas ISO 8601 se interpretan directamente; fechas ambiguas como 01/02/2024 requieren formato explícito. Se rechazan valores vacíos/inválidos, timestamps mezclados con/sin zona horaria, y un único timestamp. Grupos vacíos o menos de dos grupos también fallan. El tamaño real puede diferir de test_size por tamaños desiguales de grupo o timestamps empatados; JSON y Markdown registran fracciones, conteos, advertencias y garantías. En clasificación, train debe contener todas las clases; balanced_accuracy también requiere todas las clases en validación. R² falla si el target de validación es constante. Falta una estrategia o columna requerida: needs_input. Combinación grupo+tiempo no soportada: unsupported. Columna inexistente o imposibilidad de garantizar el protocolo: failed. Ninguna estrategia demuestra ausencia total de leakage; una feature puede incluir información futura incluso con split temporal.

Ejemplos de regresión:

```powershell
# Métrica explícita: prevalece aunque el objetivo estructurado sugiera otra.
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric mae --metric-context large_errors_cost_more
# Recomendación estructurada: RMSE porque los errores grandes reciben penalización adicional.
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --metric-context large_errors_cost_more
# Sin métrica ni objetivo: termina en needs_input; elija --metric o --metric-context.
python research_agent.py --csv .\ventas.csv --target ingresos --task regression
```

La CLI imprime Markdown y escribe research_report.md y research_result.json en .veronica/research_runs/<UTC>/. El JSON guarda estrategia, columnas de split, test_size, semilla, hash SHA-256, perfil agregado y diagnósticos; no copia filas, valores de grupo ni el dataset. Rangos temporales y nombres de columnas pueden ser sensibles y se guardan como diagnóstico local. Markdown y JSON derivan del mismo resultado.

Los riesgos se clasifican por tarea. Regresión incluye hechos calculados, señales heurísticas y límites del protocolo; no emite riesgos de clases. Una columna con valores únicos solo se presenta como una señal que merece revisión humana. Clasificación binaria conserva la distribución de clases y puede alertar para revisar proporciones muy dispares.

#### Señales de columnas para revisión humana

El reporte agrega `column_review_signals`, una lista estructurada que también se representa en la sección **Columnas y señales para revisión humana** del Markdown. Cada elemento conserva columna, tipo de señal, estado, `rule_id`, descripción, evidencia agregada, motivo, pregunta, alcance y limitación. `human_review` comienza con `status: pending` y respuestas vacías; las señales pendientes no bloquean la corrida. JSON y Markdown se generan desde el mismo objeto de resultado.

Todas son heurísticas. No confirman errores ni leakage, y no eliminan, transforman ni cambian el rol de predictores. Una fecha no selecciona validación temporal; un candidato a identificador no selecciona validación por grupos. Los predictores del baseline antes y después de la auditoría se registran y permanecen idénticos. No se publican filas ni ejemplos/valores de identificadores.

Reglas actuales y límites:

- **Posible identificador:** el nombre activa una señal débil si algún token coincide con `id`, `identifier`, `key`, `uuid`, `code`, `codigo` o `clave`. Por separado, unicidad ≥ 90% con al menos 10 valores observados activa otra señal; se informa tipo inferido, cardinalidad y si los números parecen una secuencia. La señal de secuencia exige unicidad y rango ≤ 1.25×n o consistencia ≥ 90% de los pasos respecto al paso mediano. Con menos de 10 observaciones y unicidad ≥ 50% (o nombre sugestivo) se informa `not_evaluable`, sin conclusión firme. Una variable legítima puede ser única.
- **Posible fecha/hora:** cuenta valores ISO/inequívocos parseables, ambiguos, inválidos y faltantes. Fechas numéricas como `01/02/2024` no se ordenan por adivinación; se marcan ambiguas. Una forma numérica donde solo una lectura día/mes es válida puede parsearse inequívocamente. `--time-format` permite comprobar adicionalmente un `strptime` explícito y conserva los formatos ISO reconocidos. Una fecha no demuestra que la predicción sea temporal ni cambia el split.
- **Alta cardinalidad:** señala cardinalidad de 20 o más valores distintos o proporción única mayor de 50%. Es una pista para interpretar categorías/texto/claves, no una razón de exclusión.
- **Columna constante:** cero o un valor distinto no vacío activa revisión descriptiva; no la elimina automáticamente.
- **Posibles extremos:** para target numérico y predictores numéricos usa cercas de Tukey `Q1 − 1.5×IQR` y `Q3 + 1.5×IQR`. Reporta n finito, Q1/mediana/Q3/IQR, límites, cantidad/proporción afuera, faltantes y no finitos. Con menos de 4 valores finitos o IQR cero queda `not_evaluable`; sin valores fuera no se emite señal de outlier. Son estadísticas exploratorias, no una instrucción de recorte.
- **Posible proxy del target:** correlación de Pearson absoluta ≥ 0.98 en regresión; correlación punto-biserial absoluta ≥ 0.98 o pureza por categoría ≥ 98% en clasificación. Para no confundir categorías únicas con una relación estable, la regla de pureza exige que cada categoría tenga al menos 2 observaciones y limita cardinalidad a `max(20, 50% de train)`. También registra vocabulario de riesgo en nombres (`target`, `label`, `outcome`, `result`, `post`, `after`, `response` y equivalentes en español) solo como indicio léxico. El mínimo es 10 filas. Todas estas reglas se calculan con los índices de entrenamiento del split efectivo, después de construirlo, y cada señal declara alcance `train_only` y estrategia; la validación no participa. Asociación alta puede ser válida y no prueba leakage.

Ejemplo abreviado del JSON y de la tabla Markdown: una señal `possible_outlier` puede incluir `{ "rule_id": "outlier.tukey_1_5_iqr", "valid_observations": 40, "outside_count": 1, "outside_ratio": 0.025, "method": "Tukey 1.5×IQR" }`, acompañada de la pregunta sobre plausibilidad/unidades. El informe muestra resúmenes cuantitativos sin filas individuales.

Para inspeccionar sin entrenar, `--audit-only` valida el dataset, construye el split determinista requerido para las señales predictor-target y guarda los mismos artefactos, pero no hace retrieval, no ajusta modelo y declara `score_computed: false`:

```powershell
python research_agent.py --csv .\datos.csv --target etiqueta --task binary_classification --audit-only
python research_agent.py --csv .\ventas.csv --target ingresos --task regression --audit-only --split-strategy time --time-column fecha --time-format "%d/%m/%Y"
```

La lista humana se lee columna por columna: revisar la regla activada, la evidencia agregada y la pregunta concreta; luego registrar el significado en el contexto del dominio. No se infieren respuestas ni se bloquea una corrida por tener elementos pendientes.

Ejemplo resumido (el valor depende del CSV):

```text
Estado: completed
Auditoría: 60 filas, 3 columnas; target ingresos: media …, min …, max …; SHA-256: …
Plan: regresión lineal ridge (alpha=1.0); holdout aleatorio 80/20; semilla 42
Baseline: rmse=…; 48 train / 12 validación
Evidencia: <chunk_id> — knowledge/metrics/regression_metrics.md#…
Límite: el score describe este split; no garantiza generalización
```

### Pruebas del agente y aceptación

Las consultas de `rag_reference_queries.json` solo evalúan retrieval. `tests/test_research_agent.py` cubre validación, perfil, errores, separación target, score, citas, estados, serialización, abstención documental, integración completa y repetición determinista en CSV sintético. No usa datos descargados, modelo real ni red. Las pruebas del agente están separadas de las expectativas de retrieval.

```powershell
python -m unittest discover -s tests -v
python rag_mvp.py evaluate
python knowledge_traceability.py
```

### Evidencia, citas y revisión manual

Los datos numéricos del perfil se identifican como estadísticas calculadas por el agente. Las afirmaciones documentales se limitan al contenido y localización de los fragmentos recuperados. Las decisiones del agente y las recomendaciones se identifican aparte. La validación automática comprueba pertenencia e identidad de cada cita (`chunk_id`, `source_path`, `section_path`); no pretende probar equivalencia semántica por coincidencia de palabras.

Lista de casos para revisión humana de soporte semántico (revisar afirmación, fragmento completo, matiz y alcance; registrar respaldada, parcial o no respaldada):

| Afirmación candidata | Fragmento a inspeccionar | Criterio de aceptación |
|---|---|---|
| MAE resume el error absoluto medio de predicción | `knowledge/metrics/regression_metrics.md#41-mae` | El pasaje define MAE con error absoluto y promedio; no extrapola calidad general. |
| RMSE penaliza más errores grandes que MAE por elevar residuales al cuadrado antes de promediar y extraer raíz | `knowledge/metrics/regression_metrics.md#43-rmse` | La relación debe estar expresada en el texto recuperado, no inferida del título. |
| R² puede ser negativo cuando el modelo es peor que el predictor de media | `knowledge/metrics/regression_metrics.md#44-r2` | El pasaje debe indicar comparación con media y admitir valores negativos. |
| Un holdout mide el protocolo concreto y no garantiza generalización | `knowledge/validation/validation_strategy.md#41-holdout` | Confirmar que el pasaje advierte sobre limitaciones del split y no adjudicarle garantías. |

Los anclajes de esta tabla son una guía de revisión: verificar el encabezado y el texto actuales si cambian los documentos. No se usa prueba léxica automática como verificación semántica.

### Límites y hoja de ruta posterior

| Mejora | Beneficio | Dependencias y riesgos | Pruebas necesarias | Condición de término |
|---|---|---|---|---|
| Admisión de regresión tabular | Implementada: constante y ridge con MAE/RMSE/R² | La calidad depende de target, predictores y protocolo | Tests sintéticos de regresión y repetibilidad | Comparación local reproducible |
| Validación por grupos | Implementada por solicitud explícita con `--split-strategy group --group-column` | No infiere grupos por nombre. Rechaza columna ausente, nulos y menos de dos grupos. No estratifica por clase. | Verifica intersección vacía, tamaños, clases y repetibilidad | Implementado; falla si no puede cumplir garantías |
| Validación temporal | Implementada por solicitud explícita con `--split-strategy time --time-column` | ISO 8601 por defecto; formato explícito para otras fechas; rechaza nulos, errores y mezcla de zonas; mantiene timestamps empatados | Verifica `max(train_time) < min(validation_time)`, tamaños, métrica y repetibilidad | Implementado; falla si no puede cumplir garantías |
| Selección explícita de protocolo | Implementada: random predeterminado; group/time requieren flags y columna explícitos | Los nombres de columnas y el texto libre no seleccionan protocolo. Grupo+tiempo combinado devuelve unsupported. | Tests de protocolo explícito, ausente, inválido y default no inferido | Implementado para random, group y time; protocolo combinado fuera de alcance |
| Análisis explícito de leakage | Detectar predictores sospechosos antes de interpretar métricas | Heurísticas dan falsos positivos y no prueban ausencia de leakage | Columnas posteriores, IDs, proxies y abstención | Señales explicadas, sin declarar seguridad automática |
| Retrieval iterativo y crítica de resultados | Completar evidencia y señalar contradicciones | Más llamadas/latencia y riesgo de cita no verificable | Retriever vacío/contradictorio y citas manipuladas | Iteraciones acotadas y afirmaciones ligadas a fragmentos comprobables |
| Comparación inicial de candidatos constantes y lineales | Implementada para clasificación binaria y regresión; mismo holdout/folds por candidato | Preliminar; no hay holdout final separado; selección repetida puede sobreajustar validación | Scores, errores y configuración por candidato; particiones compartidas y determinismo | Implementado con selección por métrica y empate determinista |
| Más familias de modelos, tuning, calibración y búsqueda anidada | Extender opciones después de validar la comparación inicial | Aumenta costo y puede sobreajustar validación; nested CV requiere protocolo explícito | Presupuesto por modelo, control de fuga, comparación con mismo split y evaluación final reservada | Futuro; no implementado |
| Registro de experimentos y memoria | Comparar corridas y recuperar hallazgos acotados al contexto | Versionado del esquema; confundir conocimiento documental con resultado empírico | Hashes, migraciones, procedencia y filtros por dataset | Corridas comparables y hallazgos con contexto/procedencia |
| API o interfaz de usuario | Facilitar solicitudes repetidas | Autenticación, privacidad y operación | Contratos, permisos y flujo end-to-end | Privacidad revisada y funcionamiento reproducible |
| Más herramientas y posible multiagente | Separar tareas repetitivas justificadas | Permisos explícitos y más estados de fallo | Límites de herramientas e integración comparativa | Beneficio medido frente al agente individual |
| Benchmarks comparables | Medir el flujo completo, no solo retrieval | Datos con permiso, costo y protocolo fijo | Ejecución, score, reproducción, evidencia, citas, abstención e inválidos | Casos congelados antes de medir; expectativas cambiadas con justificación |

La validación por grupos, temporal y la elección explícita de esos protocolos están implementadas; la comparación constante/lineal también está implementada. Más familias de modelos, tuning, calibración y búsqueda anidada siguen fuera de alcance. Los scores no sustituyen la revisión del target, leakage, validación adecuada ni estabilidad del resultado.
