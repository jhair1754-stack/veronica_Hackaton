# Fuentes de la base de conocimiento de Machine Learning

## 1. Propósito de `sources/`

`sources/` conserva resúmenes fieles y trazables de capítulos o patrones individuales. Cada archivo representa una fuente concreta; los resúmenes no deben mezclar libros y deben conservar la procedencia. La integración temática entre fuentes se realizará posteriormente en `knowledge/`.

## 2. Diferencia entre `sources/`, `knowledge/` y `playbook/`

| Capa         | Función                                               | ¿Puede combinar fuentes? | Resultado            |
| ------------ | ----------------------------------------------------- | -----------------------: | -------------------- |
| `sources/`   | Extraer y conservar contenido de capítulos o patrones |                       No | Resúmenes por fuente |
| `knowledge/` | Integrar conocimiento por tema                        |                       Sí | Guías temáticas      |
| `playbook/`  | Convertir conocimiento en procedimientos              |                       Sí | Flujos operativos    |

## 3. Organización de carpetas

```text
sources/
├── geron_hands_on_ml/
├── muller_introduction_ml_python/
├── mckinney_python_data_analysis/
├── islp_python/
├── molin_hands_on_pandas/
└── ml_design_patterns/
```

Las seis carpetas de la primera fase existen. `README.md` y `source_catalog.md` documentan la colección y no son resúmenes.

## 4. Fuentes de la primera fase

| Carpeta | Fuente identificada | Cantidad de resúmenes | Temas principales |
| ------- | ------------------- | --------------------: | ----------------- |
| `geron_hands_on_ml/` | *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* | 9 | Fundamentos, proyecto end-to-end, clasificación, modelos, ensembles, reducción dimensional y aprendizaje no supervisado |
| `muller_introduction_ml_python/` | *Introduction to Machine Learning with Python* | 6 | Aprendizaje supervisado y no supervisado, preprocesamiento, feature engineering, evaluación, pipelines y texto |
| `mckinney_python_data_analysis/` | *Python for Data Analysis* | 9 | NumPy, pandas, ingesta, limpieza, integración, visualización, agregación, series temporales y modelado |
| `islp_python/` | *An Introduction to Statistical Learning with Applications in Python* | 9 | Aprendizaje estadístico, regresión, clasificación, remuestreo, regularización, árboles, SVM y aprendizaje no supervisado |
| `molin_hands_on_pandas/` | *Hands-On Data Analysis with Pandas* | 7 | Data wrangling, agregación, series temporales, visualización, anomalías, pipelines y optimización |
| `ml_design_patterns/` | *Machine Learning Design Patterns* | 7 | Representación de datos y problemas, entrenamiento, serving, reproducibilidad y responsible AI |

## 5. Estructura de los resúmenes

La estructura observada con mayor frecuencia incluye propósito, conceptos esenciales, flujo o procedimiento, reglas de decisión, métodos, fórmulas, implementación, supuestos, riesgos, aplicación en coding tests, checklist, mapeo hacia `knowledge/` y trazabilidad. No todas las fuentes usan idénticos encabezados: los documentos de patrones se organizan por problema, contexto, solución, procedimiento, ventajas, limitaciones, trade-offs y relaciones.

## 6. Convenciones de nombres

- `chNN_descripcion.md` identifica capítulos numerados, como `ch02_end_to_end_ml_project.md` y `ch05_model_evaluation.md`.
- `appendix_letra_descripcion.md` identifica un apéndice, como `appendix_b_machine_learning_project_checklist.md`.
- Los nombres observados usan minúsculas, guiones bajos y una descripción breve en inglés.
- En esta fase no se observaron archivos con la convención `moduleN_chNN_descripcion.md`.

## 7. Estados de procesamiento

- `complete`: contiene propósito, conceptos, contenido operativo, riesgos o limitaciones, aplicación práctica, mapeo hacia `knowledge/` y trazabilidad.
- `partial`: aporta contenido útil, pero omite elementos importantes.
- `needs_review`: presenta inconsistencias de metadatos, rutas, mezcla de fuentes, afirmaciones sin trazabilidad u otros problemas estructurales; prevalece sobre `partial`.
- `empty`: está vacío o contiene únicamente títulos o marcadores de posición.

Los estados describen el resumen, no la calidad de la fuente original.

## 8. Flujo de incorporación de una fuente

1. Crear un Markdown por capítulo, apéndice o conjunto delimitado de patrones.
2. Identificar título, temas y destinos propuestos en `knowledge/` sin mezclar fuentes.
3. Extraer propósito, conceptos, procedimientos, decisiones, limitaciones y aplicación práctica.
4. Separar contenido explícito, inferencias, adaptaciones modernas y recomendaciones operativas.
5. Añadir mapeo temático y trazabilidad a secciones o páginas.
6. Revisar estructura, enlaces y estado antes de incorporarlo al catálogo.

## 9. Reglas de uso

- Usar los resúmenes como capa de procedencia, no como síntesis temática final.
- No combinar libros dentro de un archivo de `sources/`.
- Mantener separadas las afirmaciones de la fuente y las inferencias prácticas.
- No presentar rutas futuras de `knowledge/` como documentos ya creados.
- Registrar datos ausentes o inconsistentes sin corregir silenciosamente el resumen.
- Regenerar el catálogo cuando se añadan o modifiquen resúmenes.

## 10. Estado de los playbooks

`playbook/` contiene tres procedimientos con contenido y cuatro archivos vacíos que se conservan como pendientes. Los vacíos no se cuentan como guías terminadas ni se rellenan para mejorar la cobertura del retriever.

| Archivo | Estado comprobado | Motivo para mantenerlo pendiente |
| ------- | ----------------- | -------------------------------- |
| `playbook/classification.md` | Pendiente; 0 bytes | Falta decidir su alcance general frente al flujo tabular ya documentado en `tabular_classification.md` y cómo evitar duplicación. |
| `playbook/ml_workflow.md` | Pendiente; 0 bytes | Falta definir el público y los límites frente al flujo de competición existente; no se debe asumir que ambos procedimientos son intercambiables. |
| `playbook/regression.md` | Pendiente; 0 bytes | Falta decidir el alcance general y los diagnósticos requeridos frente al flujo tabular ya documentado en `tabular_regression.md`. |
| `playbook/time_series.md` | Pendiente; 0 bytes | Las fuentes cubren operaciones con fechas, agregaciones y ventanas, pero no una ruta completa de modelado predictivo y validación temporal. |

Los destinos `target_knowledge_files` del front matter de `sources/` son mapeos propuestos. Un destino que todavía no existe se registra como trabajo futuro; no se debe presentar como documento ya creado.

Para repetir la revisión de enlaces, anclas, metadatos y catálogo: `python knowledge_traceability.py`. Añade `--verbose` para ver cada cita ambigua y cada destino propuesto ausente.

## 10. Navegación

[Consultar el catálogo completo](source_catalog.md)
