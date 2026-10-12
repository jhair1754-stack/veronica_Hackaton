# Auditoría de VERÓNICA — 12 de octubre de 2026

## 1. Veredicto ejecutivo

**VERÓNICA es hoy un prototipo local de consola para ML tabular, no una aplicación lista para presentar en Best Apps and Agents.** Sí puede leer un CSV local, auditarlo, recuperar fragmentos de una base de conocimiento y calcular una comparación preliminar entre un predictor constante y un modelo lineal. Verifiqué ese recorrido con el CSV de ejemplo: produjo métricas reales y reportes JSON/Markdown.

Las brechas que impiden presentarla son concretas: **no hay experiencia de carga y revisión para un evaluador, no hay llamada real a un modelo NVIDIA en Nebius y no hay demo accesible**. Además, el entrenamiento usa NumPy propio, mientras que el MVP solicitado exige pipelines y modelos de referencia de scikit-learn. El módulo que ejecuta el recorrido, su prueba y el CSV de ejemplo están **sin seguimiento en Git**; la copia pública no puede darse por equivalente a esta carpeta.

## 2. Arquitectura y flujo actual

| Componente | Función comprobada |
|---|---|
| [research_agent.py](./research_agent.py#L1650) | Entrada CLI. Recibe ruta del CSV, target, tarea, métrica y opciones de validación. |
| [research_agent.py](./research_agent.py#L669) | Lee el CSV y calcula tipos, faltantes, duplicados y distribución o estadísticas del objetivo. |
| [rag_mvp.py](./rag_mvp.py#L130) | Indexa `knowledge/` y `playbook/`; recupera fragmentos mediante BM25 y una proyección LSA local. **No indexa `sources/`.** |
| [research_agent.py](./research_agent.py#L274) | Genera un plan determinista a partir de la auditoría y los fragmentos recuperados. No utiliza un modelo de lenguaje. |
| [research_agent.py](./research_agent.py#L1108) | Compara un predictor constante y un modelo lineal implementados con NumPy, sobre las mismas particiones. |
| [research_agent.py](./research_agent.py#L1575) | Guarda `research_result.json` y `research_report.md`; por defecto bajo `.veronica/research_runs/`. |
| [sources/README.md](./sources/README.md#L3) | Describe `sources/` como resúmenes por libro, `knowledge/` como síntesis temática y `playbook/` como procedimientos. |

La única dependencia declarada en [requirements-rag.txt](./requirements-rag.txt#L1) es NumPy. No encontré servidor, interfaz web, configuración de proveedor de IA ni uso de scikit-learn. `AGENTS.md` existe y está vacío.

## 3. Matriz de capacidades

| Capacidad | Estado | Evidencia en archivo y línea | Prueba realizada | Brecha |
|---|---|---|---|---|
| CSV y columna objetivo | **Implementada y verificada por CLI** | [Validación](./research_agent.py#L319), [lectura](./research_agent.py#L669) | Suite y corrida local con `demo_regresion.csv` | Falta carga, selección y confirmación en una interfaz. |
| Clasificación y regresión | **Implementadas; probadas principalmente con datos sintéticos** | [Tareas y métricas](./research_agent.py#L30), [target](./research_agent.py#L697) | 78 pruebas; regresión comprobada además con CSV de ejemplo | La clasificación es solo binaria. F1 toma como clase positiva la última etiqueta ordenada, sin elección del usuario: [cálculo](./research_agent.py#L1082). |
| Diagnóstico calculado | **Implementado y verificado** | [Auditoría](./research_agent.py#L716) | Corrida real: 20 filas; suite | Los tipos se infieren con reglas simples; falta una revisión visible antes de entrenar. |
| Señales de fuga y estructura | **Parcial** | [Señales heurísticas](./research_agent.py#L590), [revisión pendiente](./research_agent.py#L241) | Pruebas sintéticas | Se distinguen indicios de problemas confirmados, pero no hay mecanismo para registrar la respuesta humana; las columnas señaladas siguen entrando al modelo. |
| Validación y preprocesamiento | **Implementados; verificados en pruebas** | [Splits](./research_agent.py#L946), [transformación ajustada con train](./research_agent.py#L775) | Pruebas de holdout, grupos, tiempo y CV | La independencia debe declararse; no se reserva una evaluación final separada: [limitación explícita](./research_agent.py#L1219). |
| Baseline y métricas reales | **Implementados y verificados, con otra tecnología** | [Candidatos NumPy](./research_agent.py#L1078), [comparación](./research_agent.py#L1142) | Corrida real y suite | No son `Pipeline`, `DummyClassifier` ni `DummyRegressor` de scikit-learn. La selección usa la misma validación que se informa. |
| Recomendaciones con RAG | **Parcial** | [Recuperación](./research_agent.py#L1263), [plan](./research_agent.py#L274) | Índice real: 15 documentos, 530 fragmentos; 21/21 consultas de referencia | Las recomendaciones son reglas fijas. Las citas identifican fragmentos de `knowledge/`, pero mucha trazabilidad hacia las fuentes originales carece de sección precisa. |
| Reporte reproducible | **Parcial** | [Guardado](./research_agent.py#L1575), [contenido](./research_agent.py#L1401) | Se generaron ambos archivos en ubicación temporal | Guarda semilla, hash y protocolo, pero no genera código ejecutable ni incluye el CSV necesario para repetir la corrida. |
| Modelo NVIDIA en Nebius | **Ausente** | [Planificador local](./research_agent.py#L274); [integración descrita como futura](./RAG_MVP.md#L109) | Inspección estática; no se hizo llamada de pago | Falta adaptador, configuración del modelo, llamada real y prueba de integración. |
| Interfaz y demo accesible | **Ausentes** | [Único punto de entrada de producto](./research_agent.py#L1650) | Inventario e inspección | Un evaluador debe disponer de Python y un CSV local; no existe URL de demo. |
| Instalación y publicación | **Parciales** | [Instrucciones técnicas](./RAG_MVP.md#L117), [dependencia](./requirements-rag.txt#L1) | Se ejecutó con el entorno ya disponible; no se instaló desde cero | Faltan README principal, licencia y comprobación de instalación limpia. El módulo principal aún no está registrado en Git. |

**Solo documentado:** las variables de entorno propuestas para un futuro proveedor en [RAG_MVP.md](./RAG_MVP.md#L113) no son leídas por el código. Cuatro playbooks están vacíos y registrados como pendientes en [sources/README.md](./sources/README.md#L78); no forman parte del MVP tabular necesario para la demo.

## 4. Resultados de verificación

| Comprobación | Resultado y límite |
|---|---|
| `python -B -m unittest discover -s tests -q` | **78 pruebas, OK**, al dirigir `TMP` y `TEMP` a `tmp/`. El primer intento falló por permisos del directorio temporal del sandbox; no fue un fallo demostrado del producto. |
| Ejecución local real con `demo_regresion.csv`, `target=ventas`, regresión, RMSE y semilla 42 | **Completada.** 20 filas; predictor constante **RMSE 16,641439** y modelo lineal **RMSE 0,724897** en ese holdout. Se seleccionó el lineal; hubo 12 fragmentos citados, evidencia de métricas y validación, y se generaron JSON y Markdown. Una segunda ejecución dio el mismo score. Estos números describen **esa partición de un CSV pequeño**, no rendimiento esperado en datos nuevos. |
| Construcción y evaluación del índice con corpus real | 15 documentos, 530 fragmentos; **21/21** consultas de referencia acertadas, Recall@5 de 1,0 en 19 casos positivos. Son consultas preparadas para ese corpus, no una evaluación independiente de calidad de las recomendaciones. |
| `python -B knowledge_traceability.py` | Código 0, **0 errores verificables**, pero **530 referencias ambiguas** a archivos completos y **204 destinos propuestos aún ausentes**. |
| Git | `research_agent.py`, su prueba y `demo_regresion.csv` figuran sin seguimiento; `RAG_MVP.md` tiene cambios locales. La revisión de Git usó `git -c safe.directory=...` porque la propiedad del checkout produjo una advertencia; no se cambió la configuración. |

Las pruebas de integración del agente inyectan fragmentos de recuperación simulados en varios casos, por ejemplo en [test_research_agent.py](./tests/test_research_agent.py#L162). La corrida local anterior sí utilizó el corpus real y los modelos locales. **No se verificaron** una instalación limpia, una URL pública, una llamada Nebius ni el funcionamiento con datos representativos de usuarios.

## 5. Hallazgos prioritarios

| Prioridad | Problema y consecuencia | Evidencia | Corrección propuesta |
|---|---|---|---|
| **P0** | Falta la ejecución en Nebius de un modelo abierto NVIDIA: no satisface el requisito técnico central del hackathon. | [Planificador determinista](./research_agent.py#L274). Las [reglas oficiales](https://nebiusglobalaihackathon.devpost.com/rules) exigen llamada de inferencia en Token Factory o ejecución en AI Cloud. | Integrar un modelo NVIDIA servido por Nebius para interpretar estadísticas y evidencia estructuradas; validar su respuesta y conservar los cálculos en Python. Probar una llamada real antes de grabar la demo. |
| **P0** | No hay producto accesible para evaluación. | [CLI](./research_agent.py#L1650). | Construir un solo recorrido web: CSV → target y supuestos → diagnóstico → confirmación → baseline → informe y descargas; alojar una demo funcional. |
| **P0** | La implementación principal está fuera de Git; un repositorio publicado desde el estado registrado no contendría el recorrido auditado. | `git status --short`; [archivo principal](./research_agent.py#L1). | Registrar y revisar los archivos necesarios antes de verificar el repositorio que se entregará. |
| **P1** | El baseline no usa los pipelines ni los `Dummy*` pedidos; el preprocesamiento NumPy propio aumenta el mantenimiento y dificulta mostrar el método solicitado. | [Codificación manual](./research_agent.py#L775), [modelos](./research_agent.py#L1078). | Sustituir el interior del ejecutor por `ColumnTransformer`/`Pipeline` y `DummyClassifier` o `DummyRegressor`, conservando la interfaz de resultados. |
| **P1** | Las señales de identificador o posible proxy no detienen el entrenamiento ni cambian predictores; podría mostrarse un score engañoso. | [Aviso sin exclusión](./research_agent.py#L223), [todas las columnas salvo target](./research_agent.py#L1142). | Exigir confirmación de columnas disponibles al predecir y permitir excluirlas; mostrar si el score es preliminar. |
| **P1** | No hay evaluación final independiente tras escoger candidato. | [Declaración del propio código](./research_agent.py#L1219). | Para la demo, fijar un protocolo antes de comparar y reservar un test final una sola vez, o etiquetar claramente el resultado como validación preliminar. |
| **P1** | Un CSV grande o una columna categórica de enorme cardinalidad puede consumir memoria o tiempo sin límite. | [Lectura completa](./research_agent.py#L674), [one-hot manual](./research_agent.py#L790). | Imponer límites de bytes, filas, columnas, categorías, tiempo y trabajos concurrentes; rechazar entradas excesivas con mensajes claros. |
| **P1** | Faltan licencia, README principal y revisión de permisos de los resúmenes derivados de libros. | [Catálogo de fuentes](./sources/source_catalog.md#L1); no se hallaron archivos `LICENSE*` ni README de raíz. | Documentar instalación y atribución; comprobar derechos de distribución antes de hacer públicos `sources/` y el contenido derivado. No puede inferirse permiso de las referencias bibliográficas. |
| **P2** | En clasificación, el significado de la clase positiva no se confirma. | [Ordenación de etiquetas](./research_agent.py#L1114). | Pedir la clase positiva y reflejarla en métricas e informe. |

La aplicación actual no ejecuta código generado. Tampoco contiene servidor ni credenciales de proveedor: la protección de claves del lado servidor y los límites de consumo **quedan por diseñar con la integración**, no pueden declararse verificados. El CSV permanece en la ruta local indicada; el reporte guarda su hash y no una copia, pero no existe todavía una política de carga, retención o borrado para una futura demo alojada.

## 6. Plan mínimo para llegar a la presentación

| Orden | Trabajo y módulos | Criterio de aceptación verificable |
|---|---|---|
| 1 | **Consolidar el recorrido existente**: registrar `research_agent.py`, su prueba y un CSV de demo permitido; añadir README y dependencia de scikit-learn. | Un checkout limpio instala dependencias y reproduce una corrida documentada sin archivos locales omitidos. |
| 2 | **Cerrar el contrato metodológico** en `research_agent.py`: confirmación de target, tarea, clase positiva cuando aplique, estructura de observaciones, columnas disponibles y estrategia de validación. | Casos con target inválido, proxy señalado y grupo/tiempo reciben una decisión visible antes de entrenar; no se anuncia evaluación final si no existe. |
| 3 | **Implementar baseline scikit-learn** detrás del resultado actual: pipelines ajustados solo en train, modelo simple y `Dummy*` con mismas particiones. | Pruebas comprueban ausencia de contaminación de validación y métricas recalculadas desde predicciones; salida registra versiones, semilla, columnas y protocolo. |
| 4 | **Integrar Nebius Token Factory y un modelo NVIDIA** como adaptador del planificador. Añadir tiempo de espera, manejo de errores y límite de consumo. | Una prueba real autorizada muestra identificador del modelo y respuesta fundamentada en estadísticas calculadas, sin inventar números; fallo del proveedor deja una explicación clara. |
| 5 | **Exponer una interfaz única** sobre ese flujo y generar descarga de informe, JSON y script reproducible. | Un evaluador completa el caso de demo en el navegador y descarga artefactos que reproducen el score con el CSV original. |
| 6 | **Preparar entrega**: licencia, revisión del material derivado, README en inglés o traducido, demo alojada, video y texto Devpost. | URL probada desde una sesión nueva; repositorio público completo; video público de menos de tres minutos; explicación del uso de NVIDIA/Nebius, feedback y cambios significativos del proyecto previo. |

Según las [reglas oficiales de Devpost](https://nebiusglobalaihackathon.devpost.com/rules), la fecha indicada de **30 de octubre de 2026, 12:00 de Lima** coincide con el cierre a las 10:00 PDT. La demo debe permanecer accesible gratuitamente hasta el fin de la evaluación, **15 de diciembre de 2026, 15:00 de Lima** (12:00 hora del Pacífico en diciembre). La categoría anima a usar Nemotron en Token Factory; los servicios Serverless se **recomiendan, pero no son obligatorios**. Las reglas también exigen repositorio público con licencia visible, README, video público, feedback y explicación de cambios significativos si el proyecto ya existía. No se publicó ni desplegó nada durante esta auditoría.

## 7. Propuesta de demo

Usar inicialmente [demo_regresion.csv](./demo_regresion.csv), con `ventas` como objetivo: mostrar sus 20 filas, el diagnóstico, la decisión explícita de regresión y RMSE, la revisión de `horas` y `experiencia`, y la comparación real observada (**RMSE 16,641439 constante; 0,724897 lineal**, semilla 42). Etiquetar ambos como resultados de un holdout pequeño y preliminar. El modelo NVIDIA explicaría *por qué* se eligió el protocolo y cómo leer los resultados, citando los hechos calculados; nunca calcularía el score. Para la presentación final conviene un CSV sintético más amplio y publicable que permita enseñar faltantes y una señal de posible fuga sin afirmar que esta queda confirmada.

## 8. Decisiones pendientes

1. ¿Qué credenciales o cuenta de Nebius estarán disponibles para la prueba real y para mantener la demo activa durante la evaluación?
2. ¿Qué material de `sources/` y `knowledge/` tiene permiso comprobable para distribuirse públicamente? Hasta resolverlo, su inclusión en el repositorio abierto sigue siendo incierta.
3. ¿Qué dataset publicable representará el caso final y qué significado de negocio tendrá su objetivo?

**Siguiente paso recomendado:** consolidar en Git el recorrido local ya funcional y convertir su ejecutor en un pipeline scikit-learn con una prueba reproducible desde checkout limpio. Eso deja una base comprobable para añadir después la llamada Nebius y la interfaz, sin rehacer la auditoría ni la capa de recuperación.
