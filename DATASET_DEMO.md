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

## Reproducción y límites

Desde la raíz del repositorio:

```powershell
python -B scripts/generate_demo_ventas.py
python -B research_agent.py --csv .\demo_ventas_sinteticas.csv --target ventas --task regression --metric rmse --audit-only
python -B rag_mvp.py build
python -B research_agent.py --csv .\demo_ventas_sinteticas.csv --target ventas --task regression --metric rmse --validation-strategy cross_validation --observation-structure independent --cv-folds 5 --seed 42
```

La fórmula lineal favorece deliberadamente al candidato lineal; el caso comprueba el recorrido, no la capacidad de generalizar a ventas reales. Los faltantes se generan al azar y los presupuestos altos son parte válida del simulador. La señal de valores extremos solicita revisión humana y **no confirma** error, fuga ni necesidad de excluir una columna. El agente conserva todas las columnas señaladas como predictores: el score incluye `presupuesto_estimado`, de modo que no mide una versión del modelo sin esa variable. La comparación usa la misma validación para elegir candidato y reportar su score; no existe un test final independiente.
