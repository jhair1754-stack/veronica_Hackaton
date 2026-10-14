# VERÓNICA

VERÓNICA es hoy una **CLI local** para problemas tabulares de clasificación binaria y regresión. Lee y audita un CSV, recupera evidencia de documentos Markdown mediante RAG local, prepara un plan determinista y compara un predictor constante con un modelo lineal. El caso reproducible de este repositorio es una regresión de ventas simuladas por acción promocional.

## Requisitos e instalación

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

Ejecute en este orden desde la raíz:

```powershell
python -B scripts/generate_demo_ventas.py
$testTemp = Join-Path (Get-Location) 'tmp\readme_test_temp'
New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
python -B -m unittest discover -s tests -q
python -B rag_mvp.py build
python -B research_agent.py --csv .\demo_ventas_sinteticas.csv --target ventas --task regression --metric rmse --audit-only --output-dir .\tmp\readme_demo_audit
python -B research_agent.py --csv .\demo_ventas_sinteticas.csv --target ventas --task regression --metric rmse --validation-strategy cross_validation --observation-structure independent --cv-folds 5 --seed 42 --output-dir .\tmp\readme_demo_run_1
```

`--audit-only` escribe el perfil y las señales para revisión humana **sin entrenar ni calcular score**. La corrida completa recupera fragmentos de `knowledge/`, compara la media de entrenamiento con regresión ridge y calcula RMSE en cinco folds. La independencia se declara porque cada fila del simulador representa una acción nueva; el agente no puede deducirla del CSV.

`rag_mvp.py build` escribe el índice en `.veronica/rag_index.json`. Los dos comandos del agente escriben `research_result.json` y `research_report.md` en los directorios `tmp/` indicados; ambos directorios están ignorados por Git. Si se omite `--output-dir`, los informes van a `.veronica/research_runs/<fecha-UTC>/`, también ignorado. El JSON conserva el hash SHA-256 del CSV, semilla, particiones, métricas y señales; el Markdown presenta el mismo resultado para lectura humana. Ninguno incluye las filas originales.

**RMSE** está en las unidades del objetivo (`ventas`: unidades vendidas simuladas); menor es mejor. `baseline.score` es la media de RMSE del candidato lineal en los cinco folds. `model_comparison` incluye también el predictor constante, los scores por fold y la desviación entre folds. Para repetir la medición con el mismo CSV, la misma versión de Python/NumPy y la semilla `42`, ejecute otra vez el último comando cambiando solo `--output-dir` por `.\tmp\readme_demo_run_2`. Compare `dataset_sha256`, `baseline.score` y los scores por fold de ambos JSON; el tiempo de ejecución puede variar. Regenerar el CSV con el script produce el mismo contenido en el mismo entorno.

El recuperador tiene una evaluación separada: `python -B rag_mvp.py evaluate` mide Recall@k y MRR de consultas documentales; `python -B rag_mvp.py compare` las compara con `rag_reference_baseline.json`. Esas métricas **no son** el RMSE de ventas. Antes de interpretar los deltas de `compare`, revise `compatibility`: la referencia actual se creó con Python 3.12.14, NumPy 2.3.5 y otra configuración de tokenización. Con la instalación limpia verificada aquí (Python 3.14.0 y NumPy 2.5.4), `same_configuration=NO`; el comando funciona, pero sus deltas no son una mejora o regresión comparable. Consulte [RAG_MVP.md](RAG_MVP.md) para las opciones completas.

## Alcance y límites

Los dos candidatos y el preprocesamiento están implementados con **NumPy**. El planificador es determinista y usa evidencia recuperada, pero no hace una llamada a un modelo de lenguaje. No hay interfaz web, despliegue público ni integración con Nebius/NVIDIA. La misma validación sirve para elegir candidato y comunicar su score; no hay un conjunto final independiente. La fórmula sintética favorece una relación lineal y el resultado no estima desempeño con ventas reales.

Las señales de auditoría son preguntas para revisión humana, no fugas o errores confirmados. El agente conserva las columnas señaladas como predictores; por ejemplo, el score de la demo **incluye** `presupuesto_estimado`. Los informes y el índice de `.veronica/`, los archivos de `tmp/` y `chapter.txt` son trabajo local ignorado por Git. La publicación de los resúmenes derivados de libros en `sources/`, `knowledge/` y `playbook/` requiere revisar sus derechos por separado.
