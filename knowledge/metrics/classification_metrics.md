---
title: "Métricas de clasificación"
knowledge_area: "metrics"
status: "draft"
version: "1.0"
created_from:
  - "sources/"
target_use:
  - "coding tests"
  - "datathons"
  - "proyectos de Machine Learning"
---

# Métricas de clasificación

## 1. Propósito y alcance

Este documento ayuda a elegir, calcular e interpretar métricas de clasificación según el objetivo de evaluación, la distribución de clases y la importancia relativa de los distintos errores. Organiza el tema por decisiones: qué salida se evalúa, qué error importa, si el análisis corresponde a un umbral concreto o a múltiples umbrales y cómo cambia la lectura en problemas desbalanceados, multiclase o multilabel.

No reemplaza la formulación del problema ni la definición del uso futuro de las predicciones. Tampoco define por sí solo una estrategia de partición: una métrica solo resulta informativa cuando se calcula sobre observaciones no usadas para ajustar el modelo y con un esquema que represente la generalización buscada. El desarrollo general de esta separación corresponde a `knowledge/validation/validation_strategy.md`. [Géron, ch. 3, §§2.2 y 8; Müller, ch. 5, §§1–3 y 8]

**Convención de procedencia.** Las afirmaciones acompañadas por una referencia de capítulo son contenido explícito de los resúmenes. Los párrafos rotulados como **Síntesis integrada**, **Conclusión práctica derivada** o **Recomendación para coding test** combinan o aplican ese contenido sin atribuir la redacción resultante a una fuente individual.

## 2. Qué significa evaluar un clasificador

Una evaluación compara el **resultado observado** (`y_true`) con una salida del clasificador obtenida fuera de muestra. Esa salida puede adoptar dos formas diferentes:

- **Clase predicha:** etiqueta discreta asignada después de aplicar una regla de decisión. Alimenta la matriz de confusión, accuracy, precision, recall, especificidad y F1 en un punto operativo concreto. [Géron, ch. 3, §§2.3–2.5; Müller, ch. 5, §§2 y 5.6]
- **Puntuación o probabilidad predicha:** valor continuo usado para ordenar casos y recorrer umbrales. `decision_function()` puede tener escala arbitraria; `predict_proba()` produce valores entre 0 y 1, pero los resúmenes advierten que eso no demuestra que estén calibrados. [Géron, ch. 3, §§5.1–5.2 y 8; Müller, ch. 5, §§5.7 y 8]
- **Decisión basada en un umbral:** en binario, un caso se declara positivo cuando su puntuación supera el punto de corte. Los valores habituales de `predict` —0 para una función de decisión o 0.5 para una probabilidad— son valores predeterminados de la interfaz, no umbrales universalmente adecuados. [Müller, ch. 5, §2; ISLP, ch. 4, §§2 y 8]

**Síntesis integrada.** Evaluar etiquetas responde “¿qué ocurrió con esta regla de decisión?”. Evaluar puntuaciones mediante curvas o métricas de ranking responde “¿cómo ordena el modelo los positivos y negativos a través de posibles reglas?”. Elegir el umbral es una decisión posterior que debe alinearse con los costos de falsos positivos (`FP`) y falsos negativos (`FN`) y realizarse en validación, no en el conjunto final de prueba. [Géron, ch. 3, §§2.5 y 5.4; Müller, ch. 5, §§2, 5.7 y 8; ISLP, ch. 4, §§2–4]

## 3. Matriz de confusión

Antes de calcular métricas binarias se debe declarar qué etiqueta es la **clase positiva**: es el evento cuya detección activa el conteo de positivos. Cambiar esa elección cambia la interpretación de precision, recall, especificidad y de las curvas. [Géron, ch. 3, §8; Müller, ch. 5, §§5.6 y 8]

| Resultado observado \ Predicción | Positiva | Negativa |
|---|---:|---:|
| Positiva | `TP`: verdadero positivo | `FN`: falso negativo |
| Negativa | `FP`: falso positivo | `TN`: verdadero negativo |

- `TP`: caso positivo detectado correctamente.
- `FP`: caso negativo marcado incorrectamente como positivo.
- `TN`: caso negativo descartado correctamente.
- `FN`: caso positivo omitido.

Las filas representan clases reales y las columnas clases predichas. La diagonal contiene aciertos y las demás celdas muestran confusiones específicas. [Géron, ch. 3, §2.3; Müller, ch. 5, §§2 y 5.6]

En multiclase, la matriz conserva la misma orientación y permite localizar pares de clases confundidos. Normalizar cada fila por el número de ejemplos de la clase real facilita comparar tasas cuando los soportes difieren, pero conviene conservar también los conteos para no perder el volumen de errores. [Géron, ch. 3, §§2.3, 4 y 7.4]

## 4. Métricas principales

Sea `TP`, `FP`, `TN` y `FN` los conteos definidos arriba. Las fórmulas siguientes están explícitamente respaldadas por los resúmenes de Géron, Müller y, según la métrica, ISLP o Molin.

| Métrica | Qué mide | Fórmula respaldada | Cuándo puede ser útil | Limitación |
|---|---|---|---|---|
| Accuracy | Fracción total de aciertos | `(TP + TN) / (TP + TN + FP + FN)` | Cuando clases y costos son comparables y se contrasta con un baseline | Puede ser alta aunque el modelo ignore una clase rara |
| Precision | Pureza de las predicciones positivas | `TP / (TP + FP)` | Cuando los `FP` son costosos | Puede aumentarse sacrificando recall |
| Recall / sensibilidad / TPR | Cobertura de los positivos reales | `TP / (TP + FN)` | Cuando los `FN` son costosos | Puede aumentarse generando más `FP` |
| Especificidad / TNR | Cobertura de los negativos reales | `TN / (TN + FP)` | Cuando interesa descartar correctamente negativos | No describe la detección de positivos |
| F1 | Equilibrio armónico de precision y recall | `2 × precision × recall / (precision + recall)` | Cuando ambas importan de forma comparable | Resume un solo umbral y no representa costos asimétricos |
| FPR | Fracción de negativos reales marcada como positiva | `FP / (FP + TN) = 1 - TNR` | Para leer falsos positivos y el eje horizontal de ROC | No describe por sí sola la cobertura positiva |

Fuentes de la tabla: [Géron, ch. 3, §§2.4 y 6], [Müller, ch. 5, §§4–6], [ISLP, ch. 4, §§2 y 6] y [Molin, ch. 8, §§2, 5 y 6].

### 4.1 Accuracy

\[
\mathrm{accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

Mide la proporción total de decisiones correctas. Es fácil de interpretar si las clases y los costos son comparables. Sin embargo, en un target desbalanceado un predictor constante puede alcanzar una cifra alta: los resúmenes muestran ejemplos en los que predecir siempre la clase mayoritaria supera 90 %, 99 % o aproximadamente 87 % sin detectar la clase de interés. [Géron, ch. 3, §2.4; Müller, ch. 5, §§4 y 9; Molin, ch. 8, §§2 y 9]

**Conclusión práctica derivada.** Reportar accuracy exige acompañarla al menos de la distribución del objetivo, un baseline trivial y la matriz de confusión cuando existe desbalance.

### 4.2 Precision

\[
\mathrm{precision}=\frac{TP}{TP+FP}
\]

Responde: “de todos los casos que el sistema declaró positivos, ¿qué fracción era realmente positiva?”. Una precision alta implica pocos falsos positivos. Puede ser apropiada cuando cada alarma, revisión o acción incorrecta resulta costosa. [Géron, ch. 3, §§2.4 y 6; Müller, ch. 5, §§4 y 6; Molin, ch. 8, §§2 y 6]

No debe interpretarse sin recall: una regla extremadamente restrictiva puede acertar casi todas sus pocas alertas y, al mismo tiempo, omitir la mayoría de los positivos.

### 4.3 Recall o sensibilidad

\[
\mathrm{recall}=\mathrm{TPR}=\frac{TP}{TP+FN}
\]

Responde: “de todos los positivos reales, ¿qué fracción detectó el sistema?”. Un recall alto implica pocos falsos negativos. Es relevante cuando omitir un positivo es el error prioritario. [Géron, ch. 3, §§2.4 y 6; Müller, ch. 5, §§4 y 6; ISLP, ch. 4, §§2 y 6]

Reducir el umbral suele aumentar recall, pero también produce más falsos positivos. Por eso tampoco se interpreta de forma aislada.

### 4.4 Especificidad

\[
\mathrm{specificity}=\mathrm{TNR}=\frac{TN}{TN+FP}
\]

La especificidad mide qué fracción de los negativos reales se descarta correctamente. Su complemento es la tasa de falsos positivos:

\[
\mathrm{FPR}=\frac{FP}{FP+TN}=1-\mathrm{TNR}
\]

Es útil cuando interesa describir el comportamiento sobre la clase negativa y para interpretar ROC, que grafica TPR frente a FPR. Bajar el umbral para ganar sensibilidad normalmente reduce especificidad. [Géron, ch. 3, §6; ISLP, ch. 4, §§2, 4 y 6]

### 4.5 F1 y otras métricas

\[
F_1=2\frac{\mathrm{precision}\times\mathrm{recall}}
{\mathrm{precision}+\mathrm{recall}}
=\frac{2TP}{2TP+FP+FN}
\]

F1 es la media armónica de precision y recall; solo será alta si ambas lo son. Puede servir como resumen cuando los dos tipos de error importan de forma comparable, pero representa un único umbral y no incorpora costos monetarios u operativos. [Géron, ch. 3, §§2.4 y 6; Müller, ch. 5, §6; Molin, ch. 8, §§5–6]

Para múltiples clases o etiquetas, Müller distingue tres agregaciones: `macro` da el mismo peso a cada clase; `weighted` pondera por soporte; `micro` agrega `TP`, `FP` y `FN` antes del cálculo. Un promedio ponderado puede ocultar rendimiento deficiente en una clase minoritaria, mientras que `macro` puede dar gran influencia a clases raras. [Müller, ch. 5, §§2, 4, 5.6 y 9]

Géron documenta F1 macro para etiquetas igualmente importantes y F1 ponderado cuando sus soportes difieren; advierte que una etiqueta rara pero crítica puede quedar oculta en el promedio ponderado. [Géron, ch. 3, §§4 y 8–9]

## 5. Curvas y métricas basadas en puntuaciones

Estas herramientas requieren `y_true` y puntuaciones continuas —por ejemplo, `decision_function()` o la columna de la clase positiva de `predict_proba()`—, no etiquetas ya umbralizadas. Ordenan los casos por score y evalúan diferentes puntos de corte. [Müller, ch. 5, §§5.7 y 9]

### 5.1 ROC y ROC-AUC

La curva ROC representa `TPR` frente a `FPR` para todos los umbrales. El área ROC-AUC resume la capacidad de ranking: 1 corresponde a separación ideal y 0.5 a un ranking aleatorio en datos independientes. No selecciona un umbral y un valor alto no garantiza que el punto operativo por defecto sea útil. [Géron, ch. 3, §§2.4–2.5; Müller, ch. 5, §§2, 5.7 y 8–9; ISLP, ch. 4, §§2 y 4]

Géron advierte que ROC-AUC puede parecer optimista cuando la clase positiva es rara; en ese escenario recomienda examinar la curva precision-recall. Además, AP o AUC pueden ocultar precisamente la región operativa que importa. [Géron, ch. 3, §§2.5 y 4; Müller, ch. 5, §5.7]

### 5.2 Precisión-recall y PR-AUC

La curva precision-recall muestra cómo cambian ambas métricas al recorrer umbrales. Resulta especialmente informativa cuando la clase positiva es rara o interesan de manera especial los falsos positivos. Deben reportarse precision y recall en el umbral finalmente elegido. [Géron, ch. 3, §§2.5 y 4; Müller, ch. 5, §§2 y 5.7]

El resumen de Müller respalda **average precision (`AP`)** como resumen escalar de la curva PR y muestra `average_precision_score`. No aporta una fórmula para un área geométrica trapezoidal bajo PR ni establece que `AP` y cualquier implementación denominada “PR-AUC” sean intercambiables. Por fidelidad, este documento usa `AP` en el código y registra esa diferencia como limitación. [Müller, ch. 5, §§5.7 y 7]

### 5.3 Umbral de decisión

Elevar el umbral tiende a aumentar precision y reducir recall; bajarlo tiende a aumentar recall, reducir especificidad y generar más falsos positivos. El umbral adecuado depende del costo de `FP` y `FN`, de una restricción operativa —por ejemplo, precision mínima— y del objetivo del problema. No existe un valor universal. [Géron, ch. 3, §§2.5, 4 y 5.4; Müller, ch. 5, §§2 y 7; ISLP, ch. 4, §§2–4]

La selección debe hacerse con scores fuera de muestra o en validación. Optimizar y reportar el mismo conjunto, o barrer umbrales sobre test, produce una evaluación optimista. Tras fijar el umbral se comprueba la métrica complementaria: una precision alta con recall ínfimo, o viceversa, puede no ser útil. [Géron, ch. 3, §§5.4 y 9; Müller, ch. 5, §§8–9]

## 6. Clasificación desbalanceada

Cuando una clase domina, accuracy mezcla muchos aciertos fáciles de la mayoría con el desempeño sobre el evento de interés. La matriz de confusión, precision, recall y F1 exponen los errores que ese agregado oculta; PR/AP y ROC-AUC permiten estudiar puntuaciones a través de umbrales. [Géron, ch. 3, §§2.4–2.5 y 4; Müller, ch. 5, §§4–5.7; Molin, ch. 8, §§2 y 4–6]

**Síntesis integrada.** Un flujo mínimo consiste en: medir prevalencias, comparar con un predictor trivial, declarar la clase positiva, revisar conteos y tasas por clase y elegir la métrica según el costo del error. Si la positiva es rara, la curva PR puede mostrar mejor el compromiso relevante; ROC-AUC todavía mide ranking, pero no reemplaza el análisis del punto operativo.

Las técnicas de re-muestreo, balanceo o pesos de clase cambian el entrenamiento y quedan fuera del alcance de este documento. Deben documentarse por separado y validarse sin mezclar información entre particiones; aquí solo se evalúan sus resultados con las mismas métricas y el mismo esquema de validación.

## 7. Clasificación multiclase y multilabel

En **multiclase**, cada instancia recibe exactamente una clase entre más de dos. La matriz de confusión permite examinar errores por par de clases y su normalización por fila compara tasas por clase real. Para métricas derivadas, los resúmenes respaldan promedios `macro`, `weighted` y `micro`; la elección depende de si importan por igual las clases, las observaciones o los soportes. [Géron, ch. 3, §§2.1, 2.6 y 7.4; Müller, ch. 5, §§2 y 5.6]

En **multilabel**, una instancia puede tener varias etiquetas binarias simultáneas. Géron usa F1 macro cuando todas las etiquetas importan igual y F1 ponderado por soporte cuando sus frecuencias difieren, con la precaución de que el ponderado puede ocultar una etiqueta rara crítica. No todos los clasificadores admiten esta salida. [Géron, ch. 3, §§2.1, 4, 5.6 y 8]

**Limitación.** Los resúmenes consultados no desarrollan con suficiente detalle métricas multilabel por instancia, *subset accuracy*, Hamming loss ni reglas completas para elegir entre todas las agregaciones. Tampoco respaldan trasladar sin cambios la interpretación binaria de la clase positiva a cada problema multiclase o multilabel.

## 8. Cómo elegir una métrica

| Necesidad de evaluación | Métrica que puede considerarse | Motivo | Precaución |
|---|---|---|---|
| Clases y costos comparables | Accuracy | Resume la proporción total de aciertos | Compararla con prevalencia y baseline |
| Falsos positivos costosos | Precision | Penaliza positivos predichos incorrectamente | Revisar el recall resultante |
| Falsos negativos costosos | Recall | Mide cuántos positivos reales se recuperan | Revisar `FP` y especificidad |
| Descartar correctamente negativos | Especificidad | Mide cobertura de negativos reales | No informa la cobertura positiva |
| Precision y recall igualmente relevantes | F1 | Exige que ambas sean altas | No modela costos asimétricos |
| Positiva rara o interés especial en `FP` | Curva PR y AP | Expone el intercambio precision-recall a través de umbrales | AP puede ocultar la región operativa |
| Comparar ranking global sin fijar umbral | ROC y ROC-AUC | Resume TPR frente a FPR a través de umbrales | Puede parecer optimista con positivos raros; no fija umbral |
| Cada clase importa por igual | Promedio macro | Pesa cada clase de igual manera | Las clases raras pueden dominar el promedio |
| Cada observación importa por igual | Promedio micro | Agrega conteos antes de calcular | Puede quedar dominado por clases frecuentes |
| Reflejar el soporte de cada clase | Promedio weighted | Pondera clases por frecuencia | Puede ocultar una minoría crítica |
| Competición con métrica oficial | Métrica oficial, más diagnósticos coherentes | Hace comparable el resultado local con el criterio del reto | Confirmar tipo de entrada, promedio y signo; no sustituye el diagnóstico |

**Síntesis integrada.** La secuencia de decisión es: (1) definir la acción y el error costoso; (2) declarar la clase positiva y revisar prevalencias; (3) decidir si se evalúan etiquetas a un umbral o ranking a través de umbrales; (4) escoger la agregación en multiclase/multilabel; (5) mantener constante el conjunto y el método de evaluación al comparar modelos. Esta regla combina Géron, ch. 3, §§3–4 y 8; Müller, ch. 5, §§2–5 y 8; ISLP, ch. 4, §§2–4.

**Conclusión práctica derivada.** Si el objetivo o la competición fija una métrica, esta es el criterio principal de selección, pero la matriz de confusión y métricas complementarias siguen siendo necesarias para entender cómo se obtuvo el resultado y qué errores oculta.

## 9. Implementación mínima en Python

El ejemplo separa etiquetas (`predicted_labels`) de puntuaciones (`positive_scores`). Calcula únicamente métricas cubiertas por las fuentes y no presupone un dataset ni una estrategia de validación concreta.

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    observed_labels,
    predicted_labels,
    positive_scores,
):
    tn, fp, fn, tp = confusion_matrix(
        observed_labels,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": accuracy_score(observed_labels, predicted_labels),
        "precision": precision_score(observed_labels, predicted_labels),
        "recall": recall_score(observed_labels, predicted_labels),
        "specificity": specificity,
        "f1": f1_score(observed_labels, predicted_labels),
        "average_precision": average_precision_score(
            observed_labels,
            positive_scores,
        ),
        "roc_auc": roc_auc_score(observed_labels, positive_scores),
    }


def labels_from_threshold(positive_scores, decision_threshold):
    return (np.asarray(positive_scores) >= decision_threshold).astype(int)
```

Nota práctica: adaptación moderna de la idea descrita en la fuente.

El contrato `labels=[0, 1]` hace explícita la clase positiva y estabiliza el orden de la matriz. `average_precision_score` y `roc_auc_score` reciben scores continuos; las demás métricas reciben etiquetas. El umbral debe elegirse en validación y registrarse por separado. La gestión de denominadores nulos para precision, recall o F1 depende de la política del proyecto y no está desarrollada por los resúmenes, por lo que el ejemplo no impone `zero_division`. Base: Géron, ch. 3, §§7.1–7.2; Müller, ch. 5, §7; Molin, ch. 8, §§6–7.

## 10. Errores frecuentes y riesgos de interpretación

**Riesgos explícitos de las fuentes:**

- Usar accuracy como única evidencia en datos desbalanceados; un predictor mayoritario puede parecer competente sin detectar positivos. [Géron, ch. 3, §9; Müller, ch. 5, §9; Molin, ch. 8, §9]
- Confundir precision —pureza de los positivos predichos— con recall —cobertura de los positivos reales—. [Géron, ch. 3, §6; Müller, ch. 5, §6]
- Omitir qué clase se definió como positiva; las métricas binarias cambian de significado. [Géron, ch. 3, §8; Müller, ch. 5, §8]
- Comparar resultados obtenidos con umbrales, conjuntos o splitters diferentes como si solo cambiara el modelo. [Géron, ch. 3, §§3 y 8; Müller, ch. 5, §§8–9]
- Pasar etiquetas de `predict` a AP o ROC-AUC; estas métricas requieren scores continuos. [Müller, ch. 5, §§5.7 y 9]
- Tratar una puntuación o salida de `predict_proba` como probabilidad necesariamente calibrada. [Géron, ch. 3, §§5.2 y 9; Müller, ch. 5, §8]
- Interpretar AUC como selección de umbral o garantía del rendimiento operativo. [Müller, ch. 5, §§5.7 y 9]
- Elegir la métrica sin traducir el costo relativo de `FP` y `FN`. [Géron, ch. 3, §§2.4–2.5; ISLP, ch. 4, §§2 y 4; Molin, ch. 8, §§4–5]
- Seleccionar el umbral sobre test o reportar el mismo conjunto usado para elegirlo. [Géron, ch. 3, §§5.4 y 9; Müller, ch. 5, §9]
- Reportar un promedio weighted sin revisar clases minoritarias; puede ocultar desempeño nulo o insuficiente. [Müller, ch. 5, §9]

**Conclusiones prácticas derivadas:**

- Toda cifra debe acompañarse del método de evaluación, conjunto, definición de positivo y umbral cuando corresponda.
- Una métrica agregada no reemplaza la matriz de confusión ni el análisis de errores relevantes.
- Un valor mejor solo es comparable si la salida requerida, el conjunto evaluado y el procedimiento de cálculo permanecen constantes.

## 11. Aplicación en un coding test o datathon

**Síntesis práctica derivada de las fuentes.**

1. Confirmar la métrica oficial y si espera etiquetas, probabilidades o scores; reproducir exactamente ese contrato de cálculo.
2. Identificar si la tarea es binaria, multiclase o multilabel y declarar la clase positiva cuando corresponda.
3. Revisar la distribución del objetivo y construir un baseline trivial para contextualizar accuracy y otros agregados.
4. Elegir una métrica de diagnóstico según el error relevante: precision para `FP`, recall para `FN`, F1 si ambos importan de forma comparable, y matriz de confusión para conservar los conteos.
5. Separar métricas basadas en etiquetas de las basadas en puntuaciones; conservar scores si la métrica oficial evalúa ranking.
6. Registrar el umbral y elegirlo solo con validación. No reutilizar el test para explorar puntos de corte.
7. En multiclase o multilabel, documentar el promedio (`macro`, `micro` o `weighted`) y revisar métricas por clase o etiqueta.
8. Reportar la métrica principal junto con al menos un diagnóstico complementario; no concluir a partir de una sola cifra.

Estas recomendaciones derivan de Géron, ch. 3, §§10–11; Müller, ch. 5, §§10–11; ISLP, ch. 4, §§10–11, y no constituyen citas literales de esas fuentes.

## 12. Checklist operativo

- [ ] Confirmé si la tarea es binaria, multiclase o multilabel.
- [ ] Identifiqué la clase positiva, si corresponde.
- [ ] Revisé la distribución de las clases y un baseline trivial.
- [ ] Confirmé la métrica oficial del problema o competición.
- [ ] Elegí métricas coherentes con los errores relevantes.
- [ ] Distinguí etiquetas de clase de puntuaciones o probabilidades.
- [ ] Registré el umbral cuando afecta la evaluación.
- [ ] Elegí el umbral sin consultar el test final.
- [ ] Interpreté las métricas junto con la matriz de confusión.
- [ ] Documenté el promedio usado en multiclase o multilabel.
- [ ] Comparé modelos con el mismo conjunto y método de evaluación.
- [ ] Reporté la métrica junto con el esquema de evaluación empleado.

## 13. Relación con otros documentos de `knowledge/`

| Documento relacionado | Relación con las métricas de clasificación |
|---|---|
| `knowledge/problem_definition/problem_formulation.md` | Define el objetivo y la métrica inicial coherente con el problema |
| `knowledge/eda/dataset_audit.md` | Aporta la distribución y calidad inicial del objetivo |
| `knowledge/validation/validation_strategy.md` | Define cómo obtener una evaluación válida |
| `knowledge/preprocessing/data_leakage.md` | Advierte sobre evaluaciones optimistas por leakage |
| `knowledge/ensembles/boosting_models.md` | Evalúa los modelos de clasificación que se desarrollen |
| `knowledge/tuning/hyperparameter_tuning.md` | Utiliza la métrica para comparar configuraciones |

## 14. Vacíos y limitaciones de las fuentes

- Los resúmenes respaldan `average precision` como resumen escalar de PR, pero no una definición o fórmula específica de área trapezoidal “PR-AUC”; no se equiparan ambas sin una fuente adicional.
- No hay cobertura suficiente de calibración de probabilidades, Brier score, log loss, MCC, balanced accuracy, Cohen's kappa ni métricas sensibles a costos con fórmula explícita; se omiten.
- La cobertura multiclase explica la matriz y los promedios `macro`, `micro` y `weighted`, pero no desarrolla fórmulas completas por estrategia ni ROC/PR multiclase.
- La cobertura multilabel se limita principalmente a la definición del problema y F1 macro o ponderado; faltan métricas por instancia y reglas detalladas de selección.
- Los resúmenes explican el efecto del umbral y algunos criterios —precision mínima o costos relativos—, pero no proporcionan un procedimiento universal de optimización, estimación formal de costos ni análisis de estabilidad del umbral.
- La elección final depende del objetivo, de la métrica oficial de una competición y de experimentación con datos representativos; ninguna métrica se presenta como universal.
- La evaluación de incertidumbre estadística, intervalos de confianza y pruebas de diferencias entre modelos no está desarrollada en los resúmenes consultados.
- La estrategia completa de validación, incluidas particiones temporales o por grupos, queda fuera de alcance y debe resolverse en `knowledge/validation/validation_strategy.md`.

## 15. Fuentes utilizadas

| Fuente | Archivo | Aporte utilizado |
|---|---|---|
| Géron, *Hands-On Machine Learning*, cap. 3 | [Resumen](../../sources/geron_hands_on_ml/ch03_classification.md) | Tipos de clasificación, matriz, fórmulas, desbalance, PR/ROC, umbral, multiclase y multilabel |
| Müller y Guido, *Introduction to Machine Learning with Python*, cap. 5 | [Resumen](../../sources/muller_introduction_ml_python/ch05_model_evaluation.md) | Evaluación fuera de muestra, métricas, AP/ROC-AUC, scores continuos, umbral y promedios macro/micro/weighted |
| James et al., *An Introduction to Statistical Learning with Applications in Python*, cap. 4 | [Resumen](../../sources/islp_python/ch04_classification.md) | Probabilidades y clases, matriz, sensibilidad, especificidad, ROC/AUC y costos de errores |
| Molin, *Hands-On Data Analysis with Pandas*, cap. 8 | [Resumen](../../sources/molin_hands_on_pandas/ch08_anomaly_detection.md) | Ejemplo de desbalance, precisión, recall, F1 y selección de umbral según costos |

## 16. Trazabilidad

| Sección de este documento | Fuente de origen | Tipo de contenido |
|---|---|---|
| 1. Propósito y alcance | Géron ch. 3 §§1–3, 8; Müller ch. 5 §§1–3, 8 | Síntesis integrada |
| 2. Clases, scores/probabilidades y umbral | Géron ch. 3 §§2.5, 5.1–5.4, 8; Müller ch. 5 §§2, 5.7, 8; ISLP ch. 4 §2 | Síntesis integrada |
| 3. TP, FP, TN, FN y matriz | Géron ch. 3 §§2.3, 7.4; Müller ch. 5 §§2, 5.6 | Explícito de la fuente |
| 4.1 Accuracy y fórmula | Géron ch. 3 §§2.4, 6; Müller ch. 5 §6; Molin ch. 8 §6 | Explícito de la fuente |
| 4.2 Precision y fórmula | Géron ch. 3 §§2.4, 6; Müller ch. 5 §6; Molin ch. 8 §6 | Explícito de la fuente |
| 4.3 Recall y fórmula | Géron ch. 3 §§2.4, 6; Müller ch. 5 §6; ISLP ch. 4 §6 | Explícito de la fuente |
| 4.4 Especificidad y FPR | Géron ch. 3 §6; ISLP ch. 4 §§2, 6 | Explícito de la fuente |
| 4.5 F1 y fórmula | Géron ch. 3 §6; Müller ch. 5 §6; Molin ch. 8 §6 | Explícito de la fuente |
| 4.5 Agregaciones macro, micro y weighted | Müller ch. 5 §§2, 4, 5.6; Géron ch. 3 §§4, 8 | Explícito de la fuente |
| 5.1 ROC y ROC-AUC | Géron ch. 3 §§2.4–2.5, 4; Müller ch. 5 §§2, 5.7; ISLP ch. 4 §§2, 4 | Síntesis integrada |
| 5.2 PR y average precision | Géron ch. 3 §§2.5, 4; Müller ch. 5 §§2, 5.7, 7 | Síntesis integrada |
| 5.3 Efecto y selección del umbral | Géron ch. 3 §§2.5, 5.4, 9; Müller ch. 5 §§2, 7–9; ISLP ch. 4 §§2–4 | Síntesis integrada |
| 6. Desbalance | Géron ch. 3 §§2.4–2.5, 4; Müller ch. 5 §§4–5.7; Molin ch. 8 §§2, 4–6 | Síntesis integrada |
| 7. Multiclase y multilabel | Géron ch. 3 §§2.1, 2.6, 4, 5.6, 7.4; Müller ch. 5 §§2, 5.6 | Síntesis integrada |
| 8. Tabla de elección | Géron ch. 3 §§3–4; Müller ch. 5 §§2–5; ISLP ch. 4 §§2–4 | Conclusión práctica derivada |
| 9. Código Python | Géron ch. 3 §§7.1–7.2; Müller ch. 5 §7; Molin ch. 8 §§6–7 | Conclusión práctica derivada |
| 10. Riesgos explícitos | Géron ch. 3 §9; Müller ch. 5 §9; Molin ch. 8 §9 | Explícito de la fuente |
| 10. Reglas de reporte | Fuentes anteriores combinadas | Conclusión práctica derivada |
| 11. Proceso de coding test/datathon | Géron ch. 3 §§10–11; Müller ch. 5 §§10–11; ISLP ch. 4 §§10–11 | Recomendación para coding test |
| 12. Checklist | Géron ch. 3 §11; Müller ch. 5 §11; ISLP ch. 4 §11; Molin ch. 8 §11 | Recomendación para coding test |
| 13. Relaciones documentales | Estructura solicitada y mapeos de los resúmenes | Síntesis integrada |
| 14. Vacíos | Comparación de la cobertura efectiva de las cuatro fuentes | Conclusión práctica derivada |
