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

Esta copia de trabajo no contiene directorio Git ni archivos `.gitignore`; por tanto, la referencia está preparada en la raíz para incorporarse al control de versiones, pero todavía no aparece rastreada por Git aquí. Ninguna regla `.gitignore` excluye los informes de `.veronica/` en esta copia.

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
