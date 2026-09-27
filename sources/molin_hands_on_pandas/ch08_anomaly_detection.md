---
title: "Rule-Based Anomaly Detection"
topics:
  - "Simulacion reproducible de eventos"
  - "Deteccion de anomalias basada en reglas"
  - "Agregacion temporal y construccion de baselines"
  - "Evaluacion de clasificacion desbalanceada"
target_knowledge_files:
  - "knowledge/eda/anomaly_detection_eda.md"
  - "knowledge/feature_engineering/temporal_aggregations.md"
  - "knowledge/models/rule_based_anomaly_detection.md"
  - "knowledge/metrics/imbalanced_classification.md"
  - "knowledge/validation/anomaly_threshold_validation.md"
---

# 8. Rule-Based Anomaly Detection

## 1. Propósito del capítulo

El capítulo construye un detector de actividad de autenticación sospechosa a partir de reglas. Para obtener datos controlados, primero simula accesos legítimos y ataques de fuerza bruta; después realiza análisis exploratorio, agrega los eventos por IP y hora, estima el comportamiento normal y aplica umbrales basados en cambio porcentual, cercas de Tukey y puntuaciones Z.

Su valor para Machine Learning está en el flujo completo: formular supuestos, convertir eventos en variables agregadas, establecer un baseline reproducible, evitar que las anomalías contaminen ese baseline y evaluar un problema desbalanceado con precisión, recall y F1 en lugar de confiar solo en accuracy.

## 2. Conceptos esenciales

### Simulación de eventos

- Una simulación aproxima un sistema complejo mediante componentes más simples y supuestos explícitos.
- Las llegadas de usuarios válidos se modelan como un proceso de Poisson con tasa horaria variable. El número de llegadas por hora sigue una distribución de Poisson y los tiempos entre llegadas siguen una distribución exponencial.
- La tasa cambia según día y hora para representar actividad laboral, nocturna y del resto del día.
- La reproducibilidad exige fijar semillas tanto en `random` como en NumPy.
- El simulador separa una interfaz pública pequeña (`simulate()`, guardado de logs) de métodos internos para generar llegadas, intentos y ataques.

### Detección de anomalías basada en reglas

- Una anomalía es una observación generada por un proceso distinto del considerado normal.
- El caso se reduce a clasificar cada IP como legítima o maliciosa mediante reglas de umbral.
- Una regla útil combina varias señales: muchos usuarios distintos, muchos intentos y alta tasa de fallos dentro de una ventana horaria.
- Detectar pronto importa: una agregación mensual puede revelar ataques, pero llega demasiado tarde; el capítulo elige ventanas de una hora.
- Los umbrales dependen de un baseline por hora porque el comportamiento normal cambia a lo largo del día.

### Variables operativas

Para cada par `(source_ip, hora)` se calculan:

- `username`: número de nombres de usuario distintos;
- `success`: cantidad de accesos correctos;
- `failures`: cantidad de accesos fallidos;
- `attempts`: `success + failures`;
- `success_rate`: `success / attempts`;
- `failure_rate`: `1 - success_rate`.

En los datos simulados, las IP atacantes presentan más intentos, más usuarios distintos y una tasa de fallos mucho mayor. El capítulo advierte que una IP puede variar durante un ataque, por lo que ninguna señal aislada es universal.

### Baselines robustos

- El promedio es sensible a outliers. Si los ataques entran en la muestra usada como referencia, elevan el baseline y dificultan la detección.
- El capítulo propone muestreo con reemplazo por hora y recorte previo de observaciones por encima del percentil 95 en variables críticas.
- La mediana es robusta mientras menos de la mitad de las observaciones sean outliers.
- La cerca de Tukey usa cuartiles y funciona bajo el supuesto práctico de que menos del 25% de los datos son outliers.
- La puntuación Z depende de media y desviación estándar, por lo que también sufre contaminación por outliers.

### Evaluación con clases desbalanceadas

- La matriz de confusión distingue verdaderos positivos (`TP`), verdaderos negativos (`TN`), falsos positivos (`FP`) y falsos negativos (`FN`).
- Accuracy puede ser engañosa: si solo 13% de las IP son atacantes, predecir siempre “legítima” produce aproximadamente 87% de accuracy sin detectar ataques.
- Precision responde qué fracción de las alertas era realmente maliciosa.
- Recall responde qué fracción de las IP maliciosas fue detectada.
- F1 equilibra precision y recall mediante su media armónica.
- La selección final del umbral debe reflejar el costo relativo de `FP` y `FN`.

## 3. Flujo o procedimiento presentado

1. Definir los procesos legítimo y malicioso, sus supuestos y las probabilidades relevantes.
2. Generar usuarios e IP, fijar semillas y simular intentos de autenticación por hora.
3. Registrar fecha-hora, IP, usuario, éxito y motivo de fallo; conservar también la verdad de terreno de los ataques para investigación.
4. Explorar volumen temporal, intentos por IP, tasas de éxito/fallo, motivos de error y usuarios distintos por IP.
5. Elegir una granularidad útil para alerta; el capítulo usa una hora.
6. Agregar eventos por IP y hora para obtener conteos y tasas.
7. Construir baselines por hora mediante muestras con reemplazo; si se usa la media, recortar primero valores extremos.
8. Aplicar una regla conjunta basada en cambio porcentual, cerca superior de Tukey o puntuación Z.
9. Deduplicar IP alertadas cuando aparecen en varias horas.
10. Comparar alertas con etiquetas mediante matriz de confusión, precision, recall y F1.
11. Ajustar porcentajes, multiplicador `k` o cutoff Z según el costo de falsos positivos y falsos negativos.

## 4. Reglas de decisión

| Situación | Técnica recomendada | Motivo | Precaución |
| --------- | ------------------- | ------ | ---------- |
| El comportamiento normal cambia según la hora | Baseline separado por hora | Evita comparar madrugada y horario laboral con una única referencia | Debe haber observaciones suficientes en cada hora |
| La media del baseline puede contener ataques | Recorte sobre el percentil 95 antes de promediar | Reduce la influencia de actividad extrema | El percentil es un parámetro a validar; puede retirar actividad legítima alta |
| Se desea evitar el recorte explícito y hay menos de 50% de outliers | Mediana por hora | Es robusta a valores extremos | Pierde robustez si los outliers alcanzan la mitad de la muestra |
| Hay menos de 25% de outliers y el IQR es informativo | Cerca superior de Tukey | Usa cuartiles en vez de la media | Si `IQR = 0`, aumentar `k` no eleva el límite; el capítulo añade un margen porcentual |
| La distribución de referencia es estable después del recorte | Puntuación Z | Expresa la desviación en unidades de desviación estándar | Media y desviación estándar son sensibles a outliers y una desviación cero invalida el cociente |
| Las anomalías son una minoría | Precision, recall y F1 | Accuracy puede premiar una regla que nunca alerta | Elegir la métrica según el costo de `FP` frente a `FN` |
| Se necesita alertar pronto | Agregación por IP y hora | Reduce el retraso frente a un resumen mensual | Una ventana demasiado corta puede producir baselines escasos o tasas inestables |
| Varias señales débiles describen el ataque | Regla conjunta sobre usuarios, intentos y tasa de fallos | Reduce alertas por picos legítimos de una sola señal | Una conjunción estricta puede aumentar falsos negativos |

## 5. Métodos y técnicas

### Proceso de Poisson para llegadas

- **Objetivo:** simular cuántos usuarios válidos llegan por hora.
- **Entradas:** hora, tipo de día y una tasa `lambda` escogida según el periodo.
- **Salida:** número entero de llegadas y tiempos entre llegadas.
- **Funcionamiento:** `Poisson(lambda)` genera el conteo; `Exponential(1/lambda)` genera intervalos.
- **Hiperparámetros:** rangos usados para `lambda` en horario laboral, nocturno y restante.
- **Ventajas:** produce eventos discretos y tiempos de llegada coherentes con un proceso sin memoria.
- **Limitaciones:** simplifica estacionalidad, dependencia entre usuarios y picos externos.
- **Adecuado:** creación de datos sintéticos de llegadas bajo supuestos explícitos.
- **No conviene:** cuando el proceso real tiene dependencia fuerte, ráfagas coordinadas o tasas que no pueden aproximarse por tramos horarios.

### Cambio porcentual respecto del baseline

- **Objetivo:** alertar cuando las señales exceden una proporción de su referencia horaria.
- **Entradas:** agregados por IP-hora, baseline por hora y multiplicadores por variable.
- **Salida:** IP que cumplen simultáneamente todas las condiciones.
- **Funcionamiento:** compara cada señal con `baseline * multiplicador`.
- **Hiperparámetros:** multiplicadores de `username`, `attempts` y `failure_rate`; el ejemplo principal usa `1.25`.
- **Ventajas:** interpretable y fácil de implementar.
- **Limitaciones:** un baseline muy bajo o contaminado vuelve el umbral inestable.
- **Adecuado:** reglas operativas donde se conoce qué significa “por encima de lo normal”.
- **No conviene:** si el baseline es cero o casi cero sin un tratamiento específico.

### Cerca superior de Tukey

- **Objetivo:** detectar valores por encima de una referencia robusta basada en cuartiles.
- **Entradas:** primer cuartil, tercer cuartil, multiplicador `k` y, opcionalmente, margen porcentual.
- **Salida:** límite superior por señal y hora; después, IP alertadas.
- **Funcionamiento:** calcula `Q3 + k * IQR`.
- **Hiperparámetros:** `k` (el capítulo prueba `3`; menciona `1.5` como alternativa menos restrictiva) y margen porcentual (ejemplo: `1.05`).
- **Ventajas:** menor sensibilidad a extremos que la media.
- **Limitaciones:** si `Q1 = Q3`, el IQR es cero y `k` no cambia el límite.
- **Adecuado:** menos de 25% de outliers y dispersión intercuartílica no nula.
- **No conviene:** datos muy discretos y concentrados sin añadir una tolerancia coherente.

### Puntuación Z

- **Objetivo:** alertar observaciones situadas varias desviaciones estándar sobre la media horaria.
- **Entradas:** valor observado, media, desviación estándar y cutoff.
- **Salida:** IP cuyas tres señales alcanzan el cutoff.
- **Funcionamiento:** estandariza cada señal y aplica una conjunción de condiciones.
- **Hiperparámetros:** cutoff; el capítulo usa `3`.
- **Ventajas:** escala comparable entre variables.
- **Limitaciones:** sensible a outliers; requiere desviación estándar distinta de cero.
- **Adecuado:** baseline recortado y variabilidad suficiente.
- **No conviene:** distribuciones degeneradas, fuertemente contaminadas o con muestras horarias muy pequeñas.

### Precision, recall y F1

- **Objetivo:** evaluar alertas binarias bajo desbalance.
- **Entradas:** `TP`, `FP`, `TN`, `FN`.
- **Salida:** tasas entre 0 y 1.
- **Hiperparámetros:** ninguno; el umbral del detector controla el compromiso entre métricas.
- **Ventajas:** separan cobertura de ataques y confiabilidad de alertas.
- **Limitaciones:** requieren etiquetas confiables y no incorporan por sí solas costos monetarios u operativos.
- **Adecuado:** comparación de reglas y umbrales.
- **No conviene:** usar F1 como criterio único cuando los costos de `FP` y `FN` son muy distintos.

## 6. Fórmulas esenciales

### Proceso de llegada

\[
N_h \sim \operatorname{Poisson}(\lambda_h), \qquad T_i \sim \operatorname{Exponential}(1/\lambda_h)
\]

- `N_h`: número de llegadas en la hora `h`.
- `lambda_h`: tasa media de llegadas para esa hora.
- `T_i`: tiempo entre llegadas consecutivas.
- Mide el volumen discreto de llegadas y su separación temporal.
- Supuesto clave: los intervalos son independientes y sin memoria; la tasa se considera constante dentro de cada hora.

### Tasas de éxito y fallo

\[
\text{attempts}=\text{success}+\text{failures}
\]

\[
\text{success\_rate}=\frac{\text{success}}{\text{attempts}}, \qquad
\text{failure\_rate}=1-\text{success\_rate}
\]

- `success` y `failures`: conteos en una ventana IP-hora.
- Las tasas miden la proporción de resultados correctos e incorrectos.
- Requieren `attempts > 0`; el capítulo solo conserva ventanas con actividad.

### Umbral porcentual

\[
x \ge b_h p
\]

- `x`: valor observado de una señal.
- `b_h`: baseline de esa señal para la hora `h`.
- `p`: multiplicador porcentual, por ejemplo `1.25`.
- Una observación se considera alta cuando alcanza o supera el límite.
- Si `b_h` es cero o muy bajo, el criterio puede volverse demasiado permisivo.

### Cerca superior de Tukey

\[
IQR=Q_3-Q_1, \qquad U=Q_3+k\,IQR
\]

- `Q1`, `Q3`: cuartiles primero y tercero.
- `IQR`: rango intercuartílico.
- `k`: multiplicador de la cerca.
- `U`: límite superior.
- Un valor mayor que `U` es candidato a anomalía. La robustez práctica presupone menos de 25% de outliers.

### Puntuación Z

\[
z=\frac{x-\mu_h}{\sigma_h}
\]

- `x`: valor observado.
- `mu_h`: media de la señal para la hora `h`.
- `sigma_h`: desviación estándar para esa hora.
- `z` mide cuántas desviaciones estándar separan al valor de la media.
- Requiere `sigma_h > 0`; media y desviación deben estimarse con datos no contaminados o recortados.

### Métricas de clasificación

\[
\text{accuracy}=\frac{TP+TN}{TP+FP+TN+FN}
\]

\[
\text{precision}=\frac{TP}{TP+FP}, \qquad
\text{recall}=\frac{TP}{TP+FN}
\]

\[
F_1=2\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}
\]

- `TP`: atacante alertado; `TN`: legítimo no alertado; `FP`: legítimo alertado; `FN`: atacante no alertado.
- Accuracy mide aciertos totales, precision la pureza de las alertas, recall la cobertura de atacantes y F1 el equilibrio armónico de precision y recall.
- Los denominadores deben ser distintos de cero.

## 7. Implementación práctica

### Agregación horaria sin leakage

```python
import numpy as np
import pandas as pd

events = pd.read_csv("logs.csv", parse_dates=["datetime"])

hourly = (
    events.assign(failures=~events["success"].astype(bool))
    .set_index("datetime")
    .groupby("source_ip")
    .resample("1h")
    .agg(
        username=("username", "nunique"),
        success=("success", "sum"),
        failures=("failures", "sum"),
    )
    .query("success + failures > 0")
    .assign(
        attempts=lambda frame: frame["success"] + frame["failures"],
        success_rate=lambda frame: frame["success"] / frame["attempts"],
        failure_rate=lambda frame: 1 - frame["success_rate"],
    )
    .reset_index()
)

# Síntesis práctica derivada del capítulo: el periodo de referencia debe
# terminar antes del periodo evaluado para que las observaciones puntuadas
# no definan sus propios umbrales.
baseline_rows = hourly[hourly["datetime"] < "2018-11-22"]
scoring_rows = hourly[hourly["datetime"] >= "2018-11-22"]
```

### Baseline robusto por hora y regla porcentual

```python
signals = ["username", "attempts", "failure_rate"]

def trim_by_hour(frame: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    limits = frame.groupby(frame["datetime"].dt.hour)[signals].quantile(quantile)
    hours = frame["datetime"].dt.hour
    keep = np.logical_and.reduce([
        frame[column].le(hours.map(limits[column]))
        for column in signals
    ])
    return frame.loc[keep]

trimmed = trim_by_hour(baseline_rows)
baselines = trimmed.assign(hour=trimmed["datetime"].dt.hour).groupby("hour")[signals].mean()

multipliers = {"username": 1.25, "attempts": 1.25, "failure_rate": 1.25}
scored = scoring_rows.assign(hour=scoring_rows["datetime"].dt.hour).join(
    baselines.add_suffix("_baseline"), on="hour"
)
flags = scored[
    (scored["username"] >= scored["username_baseline"] * multipliers["username"])
    & (scored["attempts"] >= scored["attempts_baseline"] * multipliers["attempts"])
    & (scored["failure_rate"] >= scored["failure_rate_baseline"] * multipliers["failure_rate"])
]["source_ip"].drop_duplicates()
```

**Nota práctica de API actual:** el libro usa `DataFrame.append()` para acumular filas durante la simulación. En versiones modernas de pandas esa API ya no es la opción aplicable; conviene acumular diccionarios en una lista y construir un `DataFrame` una vez, o concatenar lotes con `pd.concat()`. Esta actualización no forma parte del contenido original del capítulo.

### Evaluación correcta por conjuntos

```python
def classification_counts(alerted_ips, attack_ips, all_ips):
    alerted = set(alerted_ips)
    attacks = set(attack_ips)
    observed = set(all_ips)
    return {
        "tp": len(alerted & attacks),
        "fp": len(alerted - attacks),
        "tn": len(observed - alerted - attacks),
        "fn": len(attacks - alerted),
    }
```

**Nota de fidelidad:** esta implementación conserva las definiciones textuales de la matriz de confusión del capítulo. El código `evaluate()` impreso en las páginas 492-493 filtra las IP alertadas al calcular `tn`, lo que no coincide con la definición de verdadero negativo dada inmediatamente antes. Debe verificarse antes de reutilizarlo.

## 8. Supuestos y condiciones de aplicación

- Las llegadas legítimas pueden aproximarse por un proceso de Poisson con tasa dependiente de hora y día.
- Los tiempos entre llegadas son exponenciales y tienen la propiedad sin memoria.
- Un usuario válido opera desde una a tres IP; dos usuarios pueden compartir IP, aunque la simulación lo considera improbable.
- Los usuarios legítimos cometen pocos errores de credenciales.
- Cada ataque simulado pertenece a un solo atacante, usa una IP fija en el escenario principal, realiza intentos secuenciales sin demora y no comparte información con otros atacantes.
- Los atacantes conocen el número y formato aproximado de las cuentas, pero adivinan los usuarios exactos.
- La detección observa intentos registrados; no cubre robo o descifrado offline de contraseñas.
- Existe actividad suficiente por hora para estimar baselines; el bootstrap por hora busca evitar horas sin estadístico.
- Las señales seleccionadas separan razonablemente actividad legítima y maliciosa en este escenario simulado.
- Mediana, Tukey y Z-score requieren respetar sus condiciones de contaminación y variabilidad descritas en las secciones anteriores.
- La evaluación supervisada requiere etiquetas de ataque, un lujo que el capítulo reconoce como poco común fuera de investigación.

## 9. Errores y riesgos

- **Contaminación del baseline:** incluir ataques en la muestra normal eleva umbrales y oculta anomalías.
- **Data leakage de evaluación:** el capítulo advierte que no se debería evaluar la regla sobre las mismas entradas que influyeron en el baseline.
- **Ventana inadecuada:** esperar un mes retrasa la respuesta; una ventana horaria es más accionable en este caso.
- **Confiar en una señal:** muchos intentos pueden corresponder a actividad legítima; variar IP puede eludir reglas centradas en IP.
- **Media no robusta:** unos pocos ataques elevan el promedio.
- **IQR nulo:** cuando `Q1` y `Q3` coinciden, cambiar `k` no modifica la cerca superior.
- **Desviación cero:** hace indefinida la puntuación Z.
- **Accuracy engañosa:** el desbalance puede producir una cifra alta sin detectar positivos.
- **Umbrales sin costo de negocio:** maximizar recall puede molestar a usuarios legítimos; maximizar precision puede dejar ataques sin detectar.
- **Sobreajuste de reglas:** ajustar multiplicadores al mismo conjunto etiquetado puede producir resultados optimistas. Esta es una conclusión práctica derivada de la advertencia del capítulo sobre separar baseline y datos evaluados.
- **Supuestos sintéticos demasiado favorables:** los atacantes son deliberadamente visibles; resultados perfectos en estos datos no implican desempeño real equivalente.
- **Código inconsistente:** el cálculo de `tn` de `evaluate()` no concuerda con su definición textual; reutilizarlo sin pruebas puede distorsionar accuracy.
- **Manejo de excepciones:** un `except` general puede ocultar errores no previstos; el capítulo recomienda capturar primero excepciones específicas.
- **Costo computacional:** `groupby` + `resample` + `apply` y bootstrap por hora crecen con el volumen; el capítulo no presenta una estrategia de escalado distribuido.

## 10. Aplicación en un coding test

**Síntesis práctica derivada del capítulo**

- Consultar este capítulo cuando haya logs o eventos temporales y se necesite un baseline interpretable sin entrenar un modelo complejo.
- Empezar con un baseline de agregación por entidad y ventana: conteos, número de categorías distintas y tasas de error.
- Graficar conteos, histogramas, boxplots y pares de señales para verificar separación y outliers.
- Separar cronológicamente un periodo de referencia de otro de validación antes de calcular umbrales.
- Como baseline rápido, usar mediana horaria o media tras recorte y una regla conjunta con multiplicadores explícitos.
- Para mejorar, comparar cambio porcentual, Tukey y Z-score y seleccionar umbrales con precision/recall según el costo del error.
- No dedicar tiempo bajo presión a construir un simulador o paquete completo si ya existe un dataset etiquetado; conservar la idea de supuestos reproducibles y generación sintética solo si faltan datos.
- No implementar bootstrap complejo si el tiempo es corto y una mediana por grupo ofrece una referencia suficientemente robusta.
- Señales que activan estas técnicas: timestamps, entidad repetida (IP, usuario, dispositivo), fallos/éxitos, fuerte desbalance, estacionalidad intradía, colas largas y picos locales.
- Explicar siempre la posible evasión: una regla por IP pierde cobertura si el atacante distribuye los intentos entre muchas IP.

## 11. Checklist operativo

- [ ] Definir qué comportamiento es normal y qué evento debe alertarse.
- [ ] Declarar los supuestos de generación y observación de eventos.
- [ ] Fijar semilla si se simulan o muestrean datos.
- [ ] Elegir entidad y ventana temporal acordes al tiempo de respuesta.
- [ ] Calcular conteos, categorías distintas y tasas por entidad-ventana.
- [ ] Separar el periodo usado para baseline del periodo puntuado.
- [ ] Revisar outliers antes de usar la media.
- [ ] Probar una referencia robusta cuando corresponda.
- [ ] Tratar explícitamente baselines o desviaciones iguales a cero.
- [ ] Deduplicar entidades alertadas si la evaluación es por entidad.
- [ ] Calcular `TP`, `FP`, `TN` y `FN` con pruebas simples.
- [ ] Comparar precision, recall y F1; no decidir solo con accuracy.
- [ ] Elegir umbral según el costo relativo de falsas alarmas y ataques omitidos.
- [ ] Documentar cómo podría evadirse la regla.

## 12. Mapeo hacia `knowledge/`

| Contenido extraído | Archivo de destino propuesto | Prioridad |
| ------------------ | ---------------------------- | --------: |
| EDA de logs, picos, distribuciones y separación visual | `knowledge/eda/anomaly_detection_eda.md` | Alta |
| Agregación por entidad y ventana temporal | `knowledge/feature_engineering/temporal_aggregations.md` | Alta |
| Conteos, cardinalidad y tasas como señales | `knowledge/feature_engineering/event_log_features.md` | Alta |
| Reglas de cambio porcentual, Tukey y Z-score | `knowledge/models/rule_based_anomaly_detection.md` | Alta |
| Construcción de baseline robusto y contaminación | `knowledge/validation/anomaly_threshold_validation.md` | Alta |
| Precision, recall y F1 con clases desbalanceadas | `knowledge/metrics/imbalanced_classification.md` | Alta |
| Selección de umbral según costo de `FP` y `FN` | `knowledge/tuning/decision_thresholds.md` | Media |
| Simulación reproducible con Poisson y exponencial | `knowledge/problem_definition/synthetic_event_simulation.md` | Media |
| Interpretación de reglas y límites de evasión | `knowledge/interpretability/rule_based_alerts.md` | Media |

## 13. Trazabilidad

| Sección del resumen | Sección o páginas del capítulo |
| ------------------- | ------------------------------ |
| 1. Propósito del capítulo | Apertura y temas del capítulo, pp. 450-451; Summary, p. 498 |
| 2. Simulación de eventos | “Simulating login attempts” y “Assumptions”, pp. 451-453; clase del simulador, pp. 456-465 |
| 2. Variables y EDA | “Exploratory data analysis”, pp. 471-482 |
| 2. Baselines robustos | “Percent difference”, pp. 484-488; “Tukey fence”, pp. 488-490; “Z-score”, pp. 490-491 |
| 2. Evaluación desbalanceada | “Evaluating performance”, pp. 491-497 |
| 3. Flujo o procedimiento | Simulación, pp. 451-471; EDA y reglas, pp. 471-491; evaluación, pp. 491-497 |
| 4. Reglas de decisión | Comparación y discusión de métodos, pp. 484-497 |
| 5. Proceso de Poisson | pp. 452, 462-464 |
| 5. Cambio porcentual | pp. 484-488 |
| 5. Cerca de Tukey | pp. 488-490, 497 |
| 5. Puntuación Z | pp. 490-491, 497 |
| 5. Métricas | pp. 492-496 |
| 6. Fórmulas | Poisson/exponencial, pp. 452 y 462-463; tasas, pp. 478 y 483; Tukey, pp. 488-489; Z-score, pp. 490-491; métricas, pp. 493-495 |
| 7. Implementación práctica | Agregación, p. 483; baseline y reglas, pp. 484-491; evaluación, pp. 492-495; actualización moderna marcada como nota práctica |
| 8. Supuestos | “Assumptions”, pp. 451-453; limitaciones prácticas, pp. 471-479 |
| 9. Errores y riesgos | pp. 469-470, 476-479, 485-497 |
| 10. Aplicación en coding test | Síntesis derivada de pp. 471-498 |
| 11. Checklist operativo | Síntesis derivada del flujo completo, pp. 451-498 |
| 12. Mapeo hacia `knowledge/` | Síntesis documental derivada del capítulo completo, pp. 450-500 |

