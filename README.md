# VERÓNICA

VERÓNICA es hoy una **CLI local** para problemas tabulares de clasificación binaria y regresión. Lee y audita un CSV, recupera evidencia de documentos Markdown mediante RAG local, prepara un plan determinista y compara un predictor constante con un modelo lineal. El caso reproducible de este repositorio es una regresión de ventas simuladas por acción promocional.

## Requisitos e instalación

La subfase **2A** añade `--decisions-json` para registrar decisiones explícitas antes
del entrenamiento. Consulte [el contrato v1](DECISION_CONTRACT.md) y la
[plantilla pendiente de revisión](examples/decisions.pending.json).
La subfase **2B** exige decisiones semánticas explícitas para entrenar: clase positiva
(binaria), contexto, protocolo, métrica justificada y disponibilidad de cada predictor.
Los ejemplos de entrenamiento anteriores deben acompañarse de `--decisions-json`
completado para su dataset; `--audit-only` permite inspección con decisiones pendientes.
La subfase **2C** aplica exclusiones justificadas antes del preprocesamiento y exige
resolver señales evaluables de posible relación con el target. Ambos candidatos
comparten los predictores efectivos; las columnas de split se conservan aunque se
excluyan del modelo. La subfase **2D** centraliza la validación previa de CLI/API, vincula decisiones a
CSV/configuración/señales y genera plantillas con `--write-decisions-template`.
Recorrido reproducible: `python examples/demo_decision_workflow.py` (solo fixture
sintético; crea `.veronica/demo_2d/` sin sobrescribirlo). No existe evaluación final independiente.

- Python 3.10 o posterior.
- NumPy `>=1.24,<3`, declarado en `requirements-rag.txt`. Es la única dependencia externa utilizada por el agente y el recuperador; la lectura CSV, el generador y las pruebas usan la biblioteca estándar. No se usa scikit-learn.

Desde la raíz del repositorio, en PowerShell, se recomienda instalar en un entorno virtual:

```powershell
$installTemp = Join-Path (Get-Location) 'tmp\readme_install_temp'
New-Item -ItemType Directory -Path $installTemp -Force | Out-Null
$env:TEMP = $installTemp
$env:TMP = $installTemp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
```

En los comandos siguientes, `python` significa el intérprete donde se instaló ese requisito. Si se usa el entorno virtual anterior, sustituya `python` por `.\.venv\Scripts\python.exe`.

## Demo local reproducible

El [generador](scripts/generate_demo_ventas.py) crea `demo_ventas_sinteticas.csv` con 320 filas usando una semilla fija. La [ficha del dataset](DATASET_DEMO.md) explica columnas, unidades, fórmula, independencia simulada, procedencia y permiso de publicación del CSV y del generador. No reutiliza datos de terceros.

La fase 2 tiene una [verificación integral ejecutada](PHASE2_ACCEPTANCE.md). El
[JSON de demostración](examples/decisions.demo_ventas.json) está fundamentado en el
simulador; **no representa una aprobación humana** ni debe reutilizarse en datos reales.
Ejecute en este orden desde la raíz, usando rutas de salida nuevas:

```powershell
Get-FileHash .\demo_ventas_sinteticas.csv -Algorithm SHA256
python -B research_agent.py --csv demo_ventas_sinteticas.csv --target ventas --task regression --metric rmse --metric-context large_errors_cost_more --target-unit unidades_simuladas --split-strategy random --validation-strategy cross_validation --observation-structure independent --cv-folds 5 --seed 42 --write-decisions-template .veronica/ventas.pending.json --output-dir .veronica/ventas_audit
# Bloqueo esperado: needs_input, código de salida 2, sin métricas.
python -B research_agent.py --decisions-json .veronica/ventas.pending.json --output-dir .veronica/ventas_blocked
# Solo para este CSV y simulador: completa las decisiones justificadas en DATASET_DEMO.md.
python -B scripts/complete_demo_decisions.py .veronica/ventas.pending.json .veronica/ventas.simulator.json
python -B rag_mvp.py build
python -B research_agent.py --decisions-json .veronica/ventas.simulator.json --output-dir .veronica/ventas_run_1
python -B research_agent.py --decisions-json .veronica/ventas.simulator.json --output-dir .veronica/ventas_run_2
```

`--audit-only` escribe el perfil y las señales para revisión humana **sin entrenar ni calcular score**. La corrida completa recupera fragmentos de `knowledge/`, compara la media de entrenamiento con regresión ridge y calcula RMSE en cinco folds. La independencia se declara porque cada fila del simulador representa una acción nueva; el agente no puede deducirla del CSV.

`--write-decisions-template` implica auditoría y nunca sobrescribe un archivo existente.
Para otro dataset, copie la plantilla a un archivo nuevo y complete manualmente
`row_meaning`, `prediction_scenario`, `configuration.observation_structure`, métrica y
`metric_reason`, disponibilidad y respuestas justificadas. No copie confirmaciones de
esta demo. Si aparece `needs_input`, consulte `next_actions`: por ejemplo, una fila sin
significado exige completar `row_meaning`; una disponibilidad `unknown` exige evidencia
para declararla `available` o excluir la columna con motivo. No basta una confirmación
genérica. Cambiar CSV, target, protocolo o señales invalida la vinculación y requiere
auditar de nuevo y revisar las decisiones. No edite los hashes para saltarse la revisión.
Los argumentos CLI que contradigan el JSON producen un error, no una sobrescritura.

`rag_mvp.py build` escribe el índice en `.veronica/rag_index.json`. El agente escribe
`research_result.json` y `research_report.md` en cada directorio indicado, ignorado por
Git. Ambos incluyen el mismo resumen de revisión, identidad, decisiones y protocolo;
un bloqueo no conserva métricas de una ejecución anterior. Si se omite `--output-dir`,
los informes van a `.veronica/research_runs/<fecha-UTC>/`.

**RMSE** está en las unidades del objetivo (`ventas`: unidades vendidas simuladas); menor es mejor. Se elige para penalizar más los errores grandes en esta demostración, no como coste empresarial validado. `baseline.score` es la media de RMSE del candidato lineal en los cinco folds. `model_comparison` incluye también el predictor constante, los scores por fold y la desviación entre folds. Los dos últimos comandos repiten la medición con el mismo CSV, decisiones y semilla `42`. Compare el hash, las particiones y los scores por fold de ambos JSON; el tiempo de ejecución puede variar. La repetibilidad numérica se verificó en el mismo entorno Python/NumPy.

El recuperador tiene una evaluación separada: `python -B rag_mvp.py evaluate` mide Recall@k y MRR de consultas documentales; `python -B rag_mvp.py compare` las compara con `rag_reference_baseline.json`. Esas métricas **no son** el RMSE de ventas. Antes de interpretar los deltas de `compare`, revise `compatibility`: configuraciones distintas no permiten afirmar una mejora o regresión comparable. La verificación de fase 2 usó Python 3.14.0 y NumPy 2.5.0; no reevalúa la calidad del RAG. Consulte [RAG_MVP.md](RAG_MVP.md).

## Clase positiva, revisión y pruebas

En clasificación binaria declare `positive_class` en el JSON: esa etiqueta se mapea a
1 y la otra a 0; precision, recall y F1 se refieren a la clase declarada. Se eliminan
espacios exteriores como en la lectura CSV y se conserva la distinción entre mayúsculas
y minúsculas. Deben existir exactamente dos etiquetas válidas. El orden alfabético no
elige la clase positiva. El verificador incluye `accept=1`, `reject=0` y la suite
contrasta precision, recall y F1 con predicciones conocidas.

Las señales evaluables `possible_target_proxy` requieren respuesta individual antes
del ajuste. Un `keep` exige motivo, significado de la columna y disponibilidad
`available`; un `exclude` exige motivo y retira la columna antes de imputar, escalar o
codificar. Las señales no evaluables y las de unicidad, cardinalidad y extremos son
informativas: no causan exclusión automática. Las señales originales permanecen en el
informe. Para excluir, registre `excluded_columns` y haga coherentes las respuestas de
esa columna; `keep` y `exclude` contradictorios se rechazan. Ambos candidatos comparten
la lista efectiva; una columna excluida del modelo puede seguir sirviendo para grupos
o tiempo. El target nunca es predictor. Véase [la política completa](DECISION_CONTRACT.md).

Recorrido automatizado con ocho criterios y artefactos por paso (directorio nuevo):

```powershell
python -B scripts/verify_phase2.py --output-dir .veronica/phase2_acceptance_new
$testTemp = Join-Path (Get-Location) 'tmp\phase2_tests'
New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
python -B -m unittest discover -s tests -q
```

## Alcance y límites

Los dos candidatos y el preprocesamiento están implementados con **NumPy**. El planificador es determinista y usa evidencia recuperada, pero no hace una llamada a un modelo de lenguaje. No hay interfaz web, despliegue público ni integración con Nebius/NVIDIA. La misma validación sirve para elegir candidato y comunicar su score; no hay un conjunto final independiente. La fórmula sintética favorece una relación lineal y el resultado no estima desempeño con ventas reales.

Las señales de auditoría son preguntas para revisión, no fugas o errores confirmados.
Las decisiones aplicadas determinan qué predictores se conservan. La configuración
principal de la demo incluye `presupuesto_estimado` por su procedencia simulada; la
prueba de ablación lo excluye realmente. Las heurísticas que usan el target se limitan
al train del holdout preliminar registrado en el informe, con índices y huella de filas;
no inspeccionan validación/test. Esa revisión no convierte la CV posterior en evaluación
final independiente. Los informes y el índice de `.veronica/`, los archivos de `tmp/`
y `chapter.txt` son trabajo local ignorado por Git. La publicación de los resúmenes
derivados de libros en `sources/`, `knowledge/` y `playbook/` requiere revisar sus derechos
por separado.
