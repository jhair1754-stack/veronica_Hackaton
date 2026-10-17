# Fase 2: verificación integral ejecutada

Fecha: 2026-10-16. **Fase 2 completa dentro del alcance local 2A–2E**: contrato,
validación semántica, revisión aplicada a predictores, bloqueo centralizado y recorrido
reproducible verificados. Esto no significa evaluación final independiente ni preparación
para producción. No se migró el motor NumPy ni se incorporaron Nebius, LLM o interfaz web.

## Entorno y procedencia

- Intérprete realmente usado: `C:\Users\jhair\AppData\Local\Programs\Python\Python314\python.exe`.
- Python 3.14.0; NumPy 2.5.0; Windows/PowerShell; comandos desde la raíz del repositorio.
- CSV: `demo_ventas_sinteticas.csv`, 320 filas, semilla de generación 20261012.
- SHA-256: `43d55690830f8958216117a919f400b12bfa5e6b94e03557ce38de641640ea50`.
- Se regeneró una copia y se compararon todos los bytes: iguales. El CSV original no se modificó.
- [Decisiones de demostración](examples/decisions.demo_ventas.json) fundamentadas en
  [generador](scripts/generate_demo_ventas.py) y [ficha](DATASET_DEMO.md), sin aprobación
  humana inventada. El completador se restringe al hash/configuración de este simulador.
  No completa automáticamente señales nuevas sin fundamento.

## Criterios de aceptación

Los artefactos siguientes son locales e ignorados por Git, bajo
`.veronica/phase2_acceptance_verified/`. El [verificador](scripts/verify_phase2.py)
permite regenerarlos en un directorio nuevo. `commands.json` registra los argumentos,
intérprete y códigos de salida reales; cada paso conserva su log, JSON y Markdown.

| Criterio | Estado | Evidencia ejecutada |
| --- | --- | --- |
| 1. Auditoría sin entrenamiento ni score | Cumplido | `01_audit`: completed, baseline/model_comparison nulos; mocks de recuperación, codificación y ajuste en `test_phase2_acceptance.py`. |
| 2. Bloqueo con decisiones pendientes | Cumplido | `02_pending`: needs_input, código 2, acciones concretas, ninguna métrica; mocks confirman ausencia de ajuste. |
| 3. Configuración completa | Cumplido | `03_complete` y `03_checked_in_example`: completed, mismos resultados con el archivo generado y el ejemplo versionable. |
| 4. Exclusión efectiva | Cumplido | `04_exclusion`: presupuesto ausente de predictores y columnas codificadas; ambos candidatos reciben la misma lista. Pruebas de 2C inspeccionan entrada al preprocesamiento. |
| 5. Binaria con clase positiva explícita | Cumplido | `05_binary`: accept=1/reject=0, F1=1; suite semántica verifica manualmente precision=1/2, recall=1/3 y F1=2/5 con predicciones conocidas. |
| 6. Decisiones de otro CSV | Cumplido | `06_wrong_csv`: needs_input, binding stale, sin métricas; pruebas 2D cubren también target, protocolo, señales y API pública. |
| 7. Repetibilidad | Cumplido | `07_repeat`: igualdad exacta de particiones, score, scores por fold y resúmenes para los dos candidatos, mismo entorno. |
| 8. JSON/Markdown coherentes | Cumplido | Se extrajo el bloque JSON del Markdown de nueve informes y se comparó con `review_summary`. Las pruebas 2D verifican además que un bloqueo sustituye resultados anteriores. |

## Comandos y resultados reales

Recorrido CLI integrado ejecutado con éxito (código 0):

```powershell
python -B scripts/verify_phase2.py --output-dir .veronica/phase2_acceptance_verified
```

Este comando ejecutó regeneración, construcción del índice RAG, auditoría y plantilla,
solicitud incompleta, completado fundamentado en el simulador, entrenamiento, ejemplo
incluido, ablación, fixture binario, CSV alterado y repetición. Su `acceptance.json`
contiene las métricas y resultados de las aserciones. Los comandos manuales completos
están en [README](README.md#demo-local-reproducible) y [DATASET_DEMO](DATASET_DEMO.md#reproducción-y-límites).

Se ejecutaron primero las suites pertinentes y después la completa, con este entorno:

```powershell
$testTemp = Join-Path (Get-Location) 'tmp\phase2_tests'
New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
python -B -m unittest discover -s tests -p 'test_*decision*.py' -q
python -B -m unittest discover -s tests -p 'test_*review.py' -q
python -B -m unittest discover -s tests -p test_phase2_acceptance.py -v
python -B -m unittest discover -s tests -q
```

Resultados: **27 OK**, **11 OK**, **4 OK**; suite completa **120 pruebas OK en 19,230 s**.
Las cantidades de suites focalizadas son subconjuntos de las 120, no pruebas adicionales.
`git diff --check` no encontró errores de whitespace; Git avisó de conversión LF/CRLF
para archivos de texto. La huella del CSV original permaneció igual.

El primer intento del nuevo verificador falló al decodificar salida de un subproceso
Windows (codificación de consola frente a UTF-8); no se contabilizó como aprobado.
Se fijó `PYTHONIOENCODING=utf-8` en los subprocesos y se ejecutó de nuevo en el directorio
`phase2_acceptance_verified`, con éxito. Los artefactos del intento fallido quedaron
en `.veronica/phase2_acceptance/`. No se detectó un defecto adicional del motor 2A–2D
en los casos ejecutados; se corrigieron instrucciones obsoletas que omitían decisiones
y afirmaciones de que toda columna señalada siempre se conserva.

## Métricas obtenidas

Regresión: target `ventas`, métrica RMSE, justificación ilustrativa de penalizar errores
grandes, estructura independiente fundamentada en el simulador, split aleatorio,
CV de cinco folds y semilla 42. Contrato v1 ligado al CSV/configuración/señales.

| Configuración | Candidato | RMSE medio | Desviación entre folds (ddof=0) |
| --- | --- | ---: | ---: |
| Horas, experiencia, presupuesto, canal | Lineal | 4,3578221906892125 | 0,646862720577589 |
| Misma configuración | Constante | 14,250729083538081 | 1,5724575311312121 |
| Ablación: horas, experiencia y canal | Lineal | 9,749532352600102 | Consultar JSON de `04_exclusion` |

La ablación es una decisión experimental explícita, no la declaración de una fuga.
El fixture binario de 80 filas usa `x=i%2` y deriva su etiqueta de esa entrada; obtiene
**F1=1,0 para accept**, con holdout aleatorio y semilla 42. Es una comprobación de
orientación y recorrido, no evidencia de calidad en un problema real.

## Alcance pendiente fuera de esta fase

Las métricas son **preliminares**: se selecciona y evalúa candidato con la misma
validación; todavía falta una reserva final independiente. La revisión de predictores
no elimina esa limitación. Las heurísticas con target usan solo train del holdout
preliminar registrado; no se amplían a validación/test aunque el entrenamiento posterior
use CV. La fórmula sintética favorece al modelo lineal y no valida costes ni desempeño
comercial. La repetibilidad exacta se comprobó en el mismo entorno, no entre versiones
de NumPy o plataformas. El contrato registra declaraciones; no autentica una firma
humana ni puede comprobar por sí solo disponibilidad real en producción.

## Archivos de esta subfase

- Actualizados: `README.md`, `DATASET_DEMO.md`, `scripts/generate_demo_ventas.py`
  (opción `--output` para reproducir una copia sin tocar el original).
- Añadidos: `PHASE2_ACCEPTANCE.md`, `examples/decisions.demo_ventas.json`,
  `scripts/complete_demo_decisions.py`, `scripts/verify_phase2.py`,
  `tests/test_phase2_acceptance.py`.
- Se conservaron los cambios previos de contrato, agente y pruebas 2A–2D.
  No se hizo stage, commit ni push.
