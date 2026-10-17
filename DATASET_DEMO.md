# Dataset de demostración: ventas por acción promocional simulada

## Propósito y procedencia

Este CSV permite demostrar auditoría tabular, recuperación de evidencia, comparación de dos candidatos de regresión y reportes locales de VERÓNICA. `demo_ventas_sinteticas.csv` fue generado por Codex específicamente para este proyecto mediante `scripts/generate_demo_ventas.py`; no contiene datos personales ni filas tomadas de una fuente externa. Se permite publicar, copiar y adaptar **este CSV y su generador** como material de demostración del proyecto. Ese permiso no se extiende a los libros, resúmenes u otros archivos del repositorio.

Los nombres de variables y cifras son ficticios: no representan una empresa, campaña ni mercado reales. No use el score para hacer promesas de ventas.

## Unidad de observación y columnas

Cada una de las 320 filas representa **una acción promocional simulada**. No hay identificador, entidad compartida ni secuencia temporal.

| Columna | Papel | Significado y unidad |
| --- | --- | --- |
| `horas_promocion` | Predictor numérico | Duración planificada de la acción, horas enteras de 3 a 18. |
| `experiencia_equipo` | Predictor numérico | Experiencia simulada del equipo, años enteros de 0 a 12; aproximadamente 6 % de las celdas se ocultan. |
| `presupuesto_estimado` | Predictor numérico | Presupuesto previsto, unidades monetarias ficticias; base de 80 a 400 en pasos de 10. Aproximadamente 2,5 % de las acciones reciben un incremento independiente de 650 a 1000, y aproximadamente 4 % de las celdas se ocultan. Sus valores extremos permiten una **señal de revisión humana**: son plausibles por construcción, no errores ni fugas confirmadas. |
| `canal` | Predictor categórico | Medio planificado: `email`, `redes` o `tienda` (tres categorías). |
| `ventas` | Objetivo numérico | Unidades vendidas simuladas, enteras no negativas. No hay valores faltantes. |

## Generación y supuestos

El script usa `random.Random(20261012)` y genera 320 filas en orden fijo. Para cada fila sortea de nuevo las cuatro variables de entrada, el indicador de presupuesto excepcional, el ruido gaussiano y dos indicadores de ocultación. Calcula el objetivo antes de ocultar celdas:

```text
ventas = max(0, redondear(15 + 1,9×horas_promocion
                         + 1,4×experiencia_equipo
                         + 0,065×presupuesto_estimado
                         + efecto_canal + ruido))
efecto_canal = {email: 2, redes: 5, tienda: 8}
ruido ~ Normal(0, 4)
```

Las filas son independientes **por construcción del simulador**: no reutilizan clientes, equipos, campañas, estados latentes ni efectos temporales, y cada sorteo de una fila usa nuevas extracciones del generador seudoaleatorio. Esta independencia simulada permite declarar `--observation-structure independent` para este caso; no es una conclusión que el agente infiera del CSV. La ocultación de predictores es aleatoria e independiente de sus valores y del objetivo. El valor oculto sí se usó en la fórmula de ventas, como magnitud latente que el modelo no observa.

## Decisiones fundamentadas en el simulador

[La configuración de demostración](examples/decisions.demo_ventas.json) no representa
aprobación humana. Declara un escenario ilustrativo de planificación de una acción,
antes de calcular sus ventas, utilizando los predictores exportados. Su fundamento
es el orden de generación de variables, no su correlación con las ventas.

| Predictor | Disponibilidad en ese escenario |
| --- | --- |
| `horas_promocion` | Duración planificada, sorteada antes de calcular el objetivo. |
| `experiencia_equipo` | Atributo simulado previo al objetivo. Se entrega con los faltantes exportados, sin recuperar el valor latente. |
| `presupuesto_estimado` | Importe previsto, incluido su incremento excepcional, generado antes de ventas. Se entrega con los faltantes exportados. |
| `canal` | Medio elegido antes del objetivo; su efecto se aplica después en la fórmula. |

`available` describe la disponibilidad de la variable, no garantiza que cada celda
tenga valor. Los indicadores de ocultación se sortean después del objetivo, pero son
independientes de él: este escenario entrega los datos enmascarados; el modelo imputa
solo con información de train. No se afirma que una empresa real recoja así estos datos.

Se usa RMSE para penalizar más los errores grandes conservando las unidades de ventas.
Es una elección explícita del experimento ilustrativo; el generador no establece
costes empresariales ni prueba que RMSE sea la métrica correcta para un uso real.

La auditoría actual encuentra extremos y alta cardinalidad en `presupuesto_estimado`,
ambos informativos según la política de [revisión](DECISION_CONTRACT.md). Se responde
`keep` con su significado y disponibilidad: el incremento excepcional es deliberado
y los importes numéricos distintos son legítimos. La señal de extremos del target
`ventas` permanece en el informe; el target nunca se convierte en predictor. No hay
señales evaluables de posible proxy pendientes en esta demo. El completador solo
resuelve esas señales conocidas; una señal nueva queda pendiente de justificación.

Una sospecha evaluable `possible_target_proxy` sí requiere respuesta antes de entrenar.
Conservarla exige significado, motivo y disponibilidad confirmada; excluirla requiere
motivo y la retira antes del preprocesamiento, sin borrar la señal. Una columna de
grupos o tiempo puede seguir particionando aunque se excluya del modelo. No existe
una opción global para ignorar todas las sospechas.

## Reproducción y límites

SHA-256 verificado el 2026-10-16:

```text
43d55690830f8958216117a919f400b12bfa5e6b94e03557ce38de641640ea50
```

El verificador regenera una copia con `--output` y comprueba igualdad byte por byte,
sin reemplazar el CSV original. Cambios de contenido o finales de línea cambian el
hash e invalidan las decisiones; no se deben actualizar hashes para evitar la revisión.

Desde la raíz del repositorio:

```powershell
Get-FileHash .\demo_ventas_sinteticas.csv -Algorithm SHA256
python -B research_agent.py --csv demo_ventas_sinteticas.csv --target ventas --task regression --metric rmse --metric-context large_errors_cost_more --target-unit unidades_simuladas --split-strategy random --validation-strategy cross_validation --observation-structure independent --cv-folds 5 --seed 42 --write-decisions-template .veronica/demo_ventas.pending.json --output-dir .veronica/demo_ventas_audit
# Resultado esperado: needs_input, código 2, sin entrenamiento ni métricas.
python -B research_agent.py --decisions-json .veronica/demo_ventas.pending.json --output-dir .veronica/demo_ventas_blocked
python -B scripts/complete_demo_decisions.py .veronica/demo_ventas.pending.json .veronica/demo_ventas.simulator.json
python -B rag_mvp.py build
python -B research_agent.py --decisions-json .veronica/demo_ventas.simulator.json --output-dir .veronica/demo_ventas_run
# Alternativa: ejemplo ya preparado, vinculado al mismo CSV y configuración.
python -B research_agent.py --decisions-json examples/decisions.demo_ventas.json --output-dir .veronica/demo_ventas_example
# Ocho criterios, copia regenerada, fixture binario, ablación y repetición:
python -B scripts/verify_phase2.py --output-dir .veronica/demo_ventas_verification
```

Use rutas nuevas: ni la plantilla ni el completador sobrescriben archivos existentes.
La auditoría funciona sin índice RAG, LLM ni entrenamiento. La plantilla deja contexto,
disponibilidad, justificación y respuestas pendientes. El bloqueo se resuelve completando
los campos señalados en `next_actions`; el ayudante anterior lo hace solo para este
simulador y hash. Para otros datos debe documentarse la evidencia real, o dejar pendiente
lo que no se conoce. Las contradicciones CLI/JSON se rechazan.

Esta demo es regresión: `positive_class` no aplica. En clasificación binaria hay que
declarar la etiqueta positiva; se mapea a 1 y define precision, recall y F1. Se recortan
espacios exteriores y se conserva mayúsculas/minúsculas. El fixture del verificador
declara `accept=1`, `reject=0`, aunque `accept` no sea la última etiqueta alfabética.

Resultados ejecutados con Python 3.14.0, NumPy 2.5.0, CV aleatoria de cinco folds,
semilla 42: RMSE lineal **4,3578221906892125** (desviación entre folds **0,646862720577589**),
constante **14,250729083538081**. La ablación que excluye realmente `presupuesto_estimado`
obtiene RMSE lineal **9,749532352600102**; conserva horas, experiencia y canal. No se
excluye por sospecha confirmada, sino para verificar el efecto de una decisión explícita.
Véase [la evidencia de aceptación](PHASE2_ACCEPTANCE.md).

La fórmula lineal favorece deliberadamente al candidato lineal; el caso comprueba el
recorrido, no la generalización a ventas reales. Las heurísticas dependientes del target
usan solo train del holdout preliminar documentado, con sus filas y hash. La revisión
de variables no convierte el score posterior en evaluación independiente. La comparación
usa la misma validación para elegir candidato y reportar su score; **no existe una reserva
final independiente**. Todas estas métricas son preliminares.
