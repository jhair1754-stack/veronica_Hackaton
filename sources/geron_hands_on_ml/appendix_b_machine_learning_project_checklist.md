---
title: "Machine Learning Project Checklist"
topics:
  - "Ciclo de vida de proyectos de Machine Learning"
  - "Preparación, selección y ajuste de modelos"
  - "Despliegue, monitorización y mantenimiento"
target_knowledge_files:
  - "knowledge/problem_definition/project_framing.md"
  - "knowledge/validation/model_selection_and_test_set.md"
  - "knowledge/preprocessing/reproducible_data_preparation.md"
  - "knowledge/tuning/hyperparameter_search.md"
  - "knowledge/ensembles/model_combination.md"
---

# B. Machine Learning Project Checklist

## 1. Propósito del capítulo

El apéndice ofrece una lista de control adaptable para conducir un proyecto de Machine Learning de extremo a extremo: formular el problema, obtener y explorar datos, prepararlos, comparar modelos, ajustar la solución, comunicarla y operarla. Su valor práctico es reducir omisiones que producen métricas irrelevantes, *data leakage*, sobreajuste al conjunto de prueba o fallos en producción.

## 2. Conceptos esenciales

- **Alineación con el objetivo:** el problema, la métrica y el rendimiento mínimo aceptable deben derivarse del objetivo de negocio y del uso previsto de la solución.
- **Supuestos verificables:** hay que enumerar y comprobar, cuando sea posible, los supuestos sobre el problema, los datos y la solución.
- **Conjunto de prueba reservado:** debe muestrearse al obtener los datos, apartarse y no inspeccionarse para evitar *data snooping*.
- **Exploración documentada:** se estudian tipo, valores ausentes, ruido, utilidad y distribución de cada atributo, además del objetivo, las visualizaciones, correlaciones y transformaciones prometedoras.
- **Preparación reproducible:** se trabaja con copias y se encapsulan las transformaciones para reutilizarlas con datos nuevos, prueba y producción, y para poder tratarlas como hiperparámetros.
- **Comparación amplia antes del ajuste:** se prueban rápidamente modelos de categorías distintas y se comparan con validación cruzada, usando tanto la media como la desviación estándar de la métrica.
- **Análisis de errores:** las variables relevantes y los tipos de error orientan nuevas rondas de selección e ingeniería de características.
- **Diversidad para el conjunto final:** conviene preseleccionar tres a cinco modelos prometedores que cometan errores distintos.
- **Ajuste sin contaminar la prueba:** hiperparámetros y decisiones de preparación se ajustan mediante validación cruzada; el conjunto de prueba se usa una vez para estimar el error de generalización.
- **Operación continua:** tras el lanzamiento se monitorizan rendimiento y calidad de entradas, se generan alertas y se reentrena periódicamente con datos frescos.

## 3. Flujo o procedimiento presentado

1. **Enmarcar el problema:** definir el objetivo de negocio, el uso, las soluciones actuales, el tipo de aprendizaje, la métrica, el mínimo aceptable, los recursos expertos y los supuestos.
2. **Obtener los datos:** especificar necesidades y fuentes, revisar volumen, permisos y obligaciones legales, preparar almacenamiento, convertir sin alterar el contenido, proteger información sensible y reservar el conjunto de prueba.
3. **Explorar los datos:** trabajar sobre una copia manejable, registrar el análisis, caracterizar atributos, identificar el objetivo, visualizar, estudiar correlaciones, proponer transformaciones y datos adicionales, y documentar hallazgos.
4. **Preparar los datos:** limpiar valores ausentes y, opcionalmente, *outliers*; seleccionar atributos; crear, descomponer, discretizar o agregar características; y escalar mediante estandarización o normalización.
5. **Preseleccionar modelos:** entrenar modelos rápidos de familias distintas, evaluarlos con validación cruzada, analizar variables y errores, iterar sobre características y conservar de tres a cinco candidatos diversos.
6. **Ajustar el sistema:** optimizar hiperparámetros y decisiones de transformación con validación cruzada, probar *ensembles* y, al final, medir una sola vez sobre prueba.
7. **Presentar la solución:** documentar el trabajo, comenzar por la visión global, conectar resultados con el objetivo, explicar éxitos, fallos, supuestos y limitaciones, y comunicar hallazgos con visualizaciones o mensajes memorables.
8. **Lanzar y mantener:** integrar entradas productivas, añadir pruebas unitarias, monitorizar rendimiento y calidad de entradas, alertar ante caídas y reentrenar regularmente.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| El conjunto es demasiado grande para comparar muchos modelos | Muestrear subconjuntos de entrenamiento más pequeños | Reduce el tiempo de exploración de candidatos | Penaliza modelos complejos, como redes neuronales grandes o *Random Forests* |
| Hay pocas combinaciones de hiperparámetros | Búsqueda en rejilla (*grid search*) | El espacio reducido puede recorrerse directamente | El apéndice solo la admite implícitamente para muy pocos valores |
| Hay más que unos pocos valores de hiperparámetros | Búsqueda aleatoria (*random search*) | El autor la prefiere a la búsqueda en rejilla en espacios más amplios | Debe evaluarse mediante validación cruzada |
| Cada entrenamiento tarda mucho | Optimización bayesiana | Puede dirigir la exploración hacia configuraciones prometedoras | Se presenta como opción, no como garantía de superioridad |
| Varios modelos buenos cometen errores diferentes | Métodos de conjunto (*ensemble methods*) | La combinación suele rendir mejor que cada modelo por separado | Primero hay que seleccionar modelos sólidos y diversos |
| No está clara una decisión de preparación | Tratarla como hiperparámetro | Permite comparar, por ejemplo, distintas estrategias de imputación | La decisión debe ajustarse dentro de la validación, no con prueba |
| El modelo final ya está decidido | Evaluar una vez en el conjunto de prueba | Estima el error de generalización | No reajustar después: se sobreajustaría el conjunto de prueba |
| El sistema está en producción | Monitorizar rendimiento y calidad de entradas; reentrenar con datos frescos | Los modelos y datos se degradan con el tiempo | El rendimiento real puede requerir etiquetas o evaluación humana |

## 5. Métodos y técnicas

### Validación cruzada de N particiones (N-fold cross-validation)

- **Objetivo:** comparar candidatos y ajustar hiperparámetros sin usar el conjunto de prueba.
- **Entradas:** datos de entrenamiento, modelo, métrica y número de particiones `N`.
- **Salida:** media y desviación estándar de la métrica en las `N` particiones.
- **Funcionamiento:** se repite el entrenamiento y la evaluación rotando la partición de validación.
- **Hiperparámetros:** `N`; el apéndice no fija un valor concreto.
- **Ventajas:** estima rendimiento y variabilidad con los datos de entrenamiento.
- **Limitaciones:** aumenta el coste al entrenar varias veces.
- **Casos adecuados:** comparación rápida de familias de modelos y ajuste de modelos/transformaciones.
- **Casos donde no conviene usarlo:** el apéndice no define una exclusión; advierte indirectamente del coste cuando entrenar es lento.

### Búsqueda de hiperparámetros

- **Objetivo:** elegir configuraciones del modelo y de la preparación.
- **Entradas:** candidatos, espacio de búsqueda, datos de entrenamiento, métrica y esquema de validación cruzada.
- **Salida:** configuración preferida según la evaluación de validación.
- **Funcionamiento:** la rejilla enumera combinaciones; la búsqueda aleatoria muestrea configuraciones; la optimización bayesiana usa resultados previos para orientar nuevas pruebas.
- **Hiperparámetros:** los rangos o distribuciones del espacio de búsqueda y el presupuesto de evaluaciones; el apéndice no prescribe valores.
- **Ventajas:** permite incluir como decisiones ajustables la imputación, eliminación de filas u otras transformaciones inciertas.
- **Limitaciones:** puede ser costosa; el texto solo menciona la optimización bayesiana como alternativa cuando entrenar tarda mucho.
- **Casos adecuados:** rejilla para muy pocos valores, aleatoria para espacios mayores y bayesiana para entrenamientos largos.
- **Casos donde no conviene usarlo:** no se debe ajustar usando el conjunto de prueba.

### Métodos de conjunto (ensemble methods)

- **Objetivo:** combinar los mejores modelos para mejorar el rendimiento.
- **Entradas:** modelos candidatos ya entrenados o sus predicciones.
- **Salida:** una predicción combinada.
- **Funcionamiento:** el apéndice no especifica el mecanismo de combinación; solo recomienda probarlo.
- **Hiperparámetros:** no detallados.
- **Ventajas:** la combinación suele superar a modelos individuales, sobre todo si sus errores difieren.
- **Limitaciones:** no se describen coste, pesos ni arquitectura del conjunto.
- **Casos adecuados:** varios candidatos sólidos con patrones de error distintos.
- **Casos donde no conviene usarlo:** el apéndice no establece una prohibición; no respalda construir un conjunto sin evaluar antes sus componentes.

### Monitorización y reentrenamiento

- **Objetivo:** detectar degradación y sostener la calidad en producción.
- **Entradas:** métricas periódicas de rendimiento, señales de calidad de entradas y datos frescos.
- **Salida:** alertas, diagnóstico operativo y nuevas versiones reentrenadas.
- **Funcionamiento:** comprobar regularmente el desempeño, vigilar entradas anómalas o desactualizadas y reentrenar de forma periódica y automatizada.
- **Hiperparámetros:** frecuencia y umbrales de alerta; no se fijan valores.
- **Ventajas:** aborda la degradación lenta causada por la evolución de los datos.
- **Limitaciones:** medir rendimiento puede requerir intervención humana; la monitorización de entradas es especialmente crítica en aprendizaje *online*.
- **Casos adecuados:** cualquier sistema desplegado, en particular sistemas *online*.
- **Casos donde no conviene usarlo:** el apéndice no identifica excepciones.

## 6. Fórmulas esenciales

Este capítulo no requiere fórmulas esenciales para su aplicación práctica.

## 7. Implementación práctica

El apéndice no proporciona código ni una implementación de API. Su recomendación explícita es automatizar tanto como sea posible la obtención de datos, las transformaciones, la comparación y el ajuste, además de escribir funciones reutilizables para cada transformación. También exige separar el conjunto de prueba antes de la exploración y reservarlo hasta que el modelo final esté decidido.

## 8. Supuestos y condiciones de aplicación

- La lista es una guía adaptable, no una secuencia rígida e idéntica para todos los proyectos.
- El objetivo de negocio, el uso previsto y la métrica pueden definirse y alinearse.
- Las fuentes, permisos, obligaciones legales y requisitos de protección de datos se conocen antes del modelado.
- Los datos originales pueden conservarse intactos y el trabajo puede realizarse sobre copias.
- Existe un conjunto de prueba que puede reservarse sin inspección.
- En aprendizaje supervisado, el atributo o atributos objetivo pueden identificarse.
- La validación cruzada y la métrica elegida son apropiadas para comparar candidatos; el apéndice no prescribe esquemas especiales para series temporales, grupos u otras dependencias.
- Hay capacidad computacional suficiente o puede emplearse una muestra para la exploración inicial.
- En producción es posible observar señales de rendimiento o establecer una vía humana para obtenerlas.
- Se dispone periódicamente de datos frescos para reentrenar.

## 9. Errores y riesgos

- Elegir una métrica que no esté alineada con el objetivo de negocio.
- Mantener supuestos sin enumerarlos ni verificarlos.
- Inspeccionar el conjunto de prueba durante exploración, selección o ajuste (*data snooping*).
- Ajustar el modelo después de observar la métrica de prueba; esto sobreajusta la prueba y invalida su papel como estimación de generalización.
- Transformar los datos de forma manual o irreproducible, dificultando aplicar exactamente el mismo proceso a prueba y producción.
- Alterar o perder el conjunto original en lugar de trabajar sobre copias.
- Omitir la protección de información sensible, autorizaciones u obligaciones legales.
- Comparar modelos solo por la media y omitir la desviación estándar entre particiones.
- Preseleccionar modelos redundantes que cometen los mismos tipos de errores.
- Muestrear datos para acelerar la comparación sin reconocer que perjudica a modelos complejos.
- Ignorar degradación lenta del modelo o entradas defectuosas/desactualizadas en producción.
- **Conclusión práctica derivada:** toda transformación aprendida de los datos debe ajustarse dentro de cada partición de entrenamiento; de otro modo, la validación puede contaminarse. El apéndice no usa el término *pipeline*, pero su exigencia de separar prueba y reutilizar transformaciones respalda esta precaución.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo.**

- **Cuándo consultarlo:** al inicio para estructurar el trabajo, antes de comparar modelos, antes de tocar el conjunto de prueba y al cerrar la solución.
- **Decisiones que permite tomar:** tipo de problema, métrica, estrategia de reserva de prueba, transformaciones candidatas, familias iniciales, esquema de comparación y alcance del ajuste.
- **Baseline:** separar prueba; identificar objetivo y tipos de atributos; tratar valores ausentes; escalar cuando corresponda; entrenar modelos rápidos de familias distintas; comparar media y desviación estándar mediante validación cruzada.
- **Mejoras:** analizar errores, iterar sobre selección e ingeniería de características, buscar hiperparámetros y combinar candidatos con errores distintos.
- **Qué evitar bajo presión:** optimización bayesiana, búsquedas amplias, conjuntos complejos o infraestructura de producción cuando el ejercicio solo exige un modelo validado; son extensiones justificadas únicamente si el presupuesto y el objetivo lo permiten.
- **Señales del dataset:** faltantes activan decisiones de imputación/eliminación; *outliers* requieren revisión; distribuciones sesgadas sugieren transformaciones como `log(x)` o `sqrt(x)`; escalas heterogéneas activan estandarización/normalización; fechas o categorías requieren descomposición; gran volumen puede justificar una muestra exploratoria; errores complementarios entre modelos sugieren un *ensemble*.
- **Límite de esta síntesis:** la priorización temporal y el uso de un *baseline* son adaptaciones para competición; el apéndice aporta el flujo y las técnicas, pero no define un calendario de coding test.

## 11. Checklist operativo

- [ ] Traducir el objetivo a tipo de problema, uso, métrica y mínimo aceptable.
- [ ] Enumerar y verificar supuestos críticos.
- [ ] Documentar fuentes, permisos, volumen y protección de datos.
- [ ] Reservar el conjunto de prueba antes de explorar.
- [ ] Explorar una copia y registrar tipos, faltantes, ruido, utilidad y distribuciones.
- [ ] Identificar objetivo, correlaciones, transformaciones y datos adicionales útiles.
- [ ] Encapsular limpieza, selección, ingeniería y escalado en transformaciones reproducibles.
- [ ] Comparar modelos de familias distintas con validación cruzada, media y desviación estándar.
- [ ] Analizar variables relevantes y tipos de error; iterar una o dos veces.
- [ ] Conservar de tres a cinco candidatos con errores diversos.
- [ ] Ajustar modelos y transformaciones sin usar prueba; probar un *ensemble* si procede.
- [ ] Evaluar una vez en prueba y no reajustar después.
- [ ] Documentar objetivo, resultados, fallos, supuestos y limitaciones.
- [ ] Si se despliega, añadir pruebas, monitorización de rendimiento/entradas, alertas y reentrenamiento.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| Formulación, objetivo, uso, métrica, mínimo y supuestos | `knowledge/problem_definition/project_framing.md` | Alta |
| Reserva del conjunto de prueba y prevención de *data snooping* | `knowledge/validation/model_selection_and_test_set.md` | Alta |
| Perfilado de atributos, faltantes, ruido, distribuciones y correlaciones | `knowledge/eda/structured_eda_checklist.md` | Alta |
| Transformaciones reutilizables y preparación sin alterar originales | `knowledge/preprocessing/reproducible_data_preparation.md` | Alta |
| Discretización, descomposición, transformaciones y agregaciones | `knowledge/feature_engineering/feature_transformation_options.md` | Media |
| Comparación con media y desviación estándar en N particiones | `knowledge/validation/cross_validation_comparison.md` | Alta |
| Rejilla, búsqueda aleatoria y optimización bayesiana | `knowledge/tuning/hyperparameter_search.md` | Alta |
| Selección de modelos con errores distintos y combinación | `knowledge/ensembles/model_diversity_and_combination.md` | Media |
| Presentación de resultados, supuestos y limitaciones | `knowledge/interpretability/solution_communication.md` | Media |
| Monitorización de rendimiento/entradas y reentrenamiento | `knowledge/models/production_monitoring.md` | Alta |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Introducción y ocho pasos principales, p. 755 |
| 2. Conceptos esenciales | Todo el apéndice, pp. 755-760 |
| 3. Flujo o procedimiento presentado | Encabezados principales desde “Frame the Problem...” hasta “Launch!”, pp. 755-760 |
| 4. Reglas de decisión | “Shortlist Promising Models”, p. 758; “Fine-Tune the System”, p. 759; “Launch!”, p. 760 |
| 5. Métodos y técnicas | “Shortlist Promising Models”, p. 758; “Fine-Tune the System”, p. 759; “Launch!”, p. 760 |
| 6. Fórmulas esenciales | No hay fórmulas en pp. 755-760 |
| 7. Implementación práctica | Notas de “Get the Data”, p. 756; “Prepare the Data”, p. 757; “Shortlist Promising Models”, p. 758; “Fine-Tune the System”, p. 759; “Launch!”, p. 760 |
| 8. Supuestos y condiciones | “Frame the Problem...”, pp. 755-756; “Get the Data”, p. 756; “Prepare the Data”, pp. 757-758; “Launch!”, p. 760 |
| 9. Errores y riesgos | “Get the Data”, p. 756; “Prepare the Data”, p. 757; “Shortlist Promising Models”, p. 758; advertencia de prueba, p. 759; “Launch!”, p. 760 |
| 10. Aplicación en un coding test | Síntesis derivada de pp. 755-760, especialmente preparación, preselección y ajuste, pp. 757-759 |
| 11. Checklist operativo | Condensación de la lista completa, pp. 755-760 |
| 12. Mapeo hacia `knowledge/` | Clasificación derivada de los contenidos de pp. 755-760 |

