# VERÓNICA

Prototipo local para investigar problemas de aprendizaje automático tabular. Audita un CSV, recupera evidencia de una base de conocimiento en Markdown y compara un predictor constante con un modelo lineal para clasificación binaria o regresión.

## Requisitos y ejecución

Python 3.10 o posterior y NumPy (declarado en `requirements-rag.txt`). Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements-rag.txt
python rag_mvp.py build
python research_agent.py --csv .\demo_regresion.csv --target ventas --task regression --metric rmse --seed 42
```

El agente guarda `research_result.json` y `research_report.md` bajo `.veronica/research_runs/`. Los resultados son una comparación preliminar sobre la partición indicada; no son una evaluación final independiente. El ejemplo contiene solo 20 filas.

Para revisar un CSV sin entrenar:

```powershell
python research_agent.py --csv .\demo_regresion.csv --target ventas --task regression --audit-only
```

## Verificación

```powershell
python -B -m unittest discover -s tests -q
python -B rag_mvp.py evaluate
python -B knowledge_traceability.py
```

`RAG_MVP.md` documenta las opciones del recuperador y del agente. `sources/` guarda resúmenes por fuente; `knowledge/` contiene síntesis temáticas y `playbook/` procedimientos. El índice y los informes locales bajo `.veronica/` no se incluyen en Git. `chapter.txt` es texto extraído para trabajo local y tampoco se publica.

## Estado y límites

El proyecto ofrece una CLI local. No incluye interfaz web, despliegue público ni integración real con un modelo NVIDIA o Nebius. El preprocesamiento y los modelos lineales actuales están implementados con NumPy. Antes de distribuir o reutilizar los resúmenes derivados de libros en otro contexto, corresponde verificar los derechos de ese material. La auditoría de octubre de 2026 está en `VERONICA_AUDITORIA_2026-10-12.md` y describe pendientes de producto y trazabilidad.
