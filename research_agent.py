"""Agente local y determinista para investigación tabular.

El contenido recuperado se trata como evidencia inerte. La ejecución se limita a
las funciones explícitas de auditoría, retrieval y entrenamiento de este módulo.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
import hashlib
import json
import math
import platform
import random
import sys
from time import perf_counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import rag_mvp

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / ".veronica" / "research_runs"
TASK = "binary_classification"
REGRESSION = "regression"
CLASSIFICATION_METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}
REGRESSION_METRICS = {"mae", "rmse", "r2"}
# Alias conservado para consumidores previos que importaban METRICS.
METRICS = CLASSIFICATION_METRICS
TASK_METRICS = {TASK: CLASSIFICATION_METRICS, REGRESSION: REGRESSION_METRICS}
_MISSING = {"", "na", "n/a", "nan", "null", "none", "<na>"}
REGRESSION_OBJECTIVES = {"absolute_error_cost", "large_errors_cost_more", "variance_explanation", "relative_error", "official_metric", "unspecified"}

# Umbrales públicos de las reglas heurísticas de auditoría.
ID_MIN_OBSERVATIONS = 10
ID_UNIQUENESS_RATIO = 0.90
ID_SEQUENCE_STEP_CONSISTENCY = 0.90
ID_SEQUENCE_RANGE_PER_OBSERVATION = 1.25
HIGH_CARDINALITY_MIN = 20
HIGH_CARDINALITY_RATIO = 0.50
OUTLIER_MIN_OBSERVATIONS = 4
TARGET_PROXY_MIN_TRAIN = 10
TARGET_PROXY_ASSOCIATION = 0.98


def metric_direction(metric: str) -> str:
    if metric in {"mae", "rmse"}: return "lower_is_better"
    if metric == "r2": return "higher_is_better"
    if metric in CLASSIFICATION_METRICS: return "higher_is_better"
    raise ValueError(f"Métrica sin dirección conocida: {metric}")


def _metric_evidence_ids(metric: str, evidence: list[dict[str, Any]]) -> list[str]:
    """Return only retrieved chunks whose text states the metric properties used."""
    patterns = {
        "mae": ("magnitud absoluta media", "error absoluto", "menos sensible a outliers"),
        "rmse": ("errores grandes", "mayor peso relativo"),
        "r2": ("adimensional", "puede ser negativo", "magnitud del error", "r^2=1", "r²=1", "rss/tss"),
    }[metric]
    result = []
    for item in evidence:
        text = str(item.get("text", "")).casefold()
        section = str(item.get("section_path", "")).casefold()
        if "fuentes utilizadas" in section or "sources used" in section:
            continue
        if any(pattern in text for pattern in patterns):
            result.append(item["chunk_id"])
    return result


def select_regression_metric(request: ResearchRequest,
                             evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Deterministic metric choice; free-form context never selects a metric."""
    objective = request.metric_context or "unspecified"
    requested = request.metric.lower() if request.metric else None
    supported = {"mae", "rmse", "r2"}
    if requested and requested not in supported:
        return {"requested_metric": request.metric, "selected_metric": None,
                "selection_source": "unsupported_request", "objective": objective,
                "target_unit": request.target_unit,
                "rationale": f"La métrica oficial o solicitada {request.metric!r} no está soportada; las opciones disponibles son MAE, RMSE y R².",
                "tradeoffs": [], "alternatives": [], "evidence_chunk_ids": []}
    choices = {"absolute_error_cost": "mae", "large_errors_cost_more": "rmse",
               "variance_explanation": "r2"}
    if requested:
        selected, source = requested, "user_explicit"
        rationale = f"Se conserva la métrica explícita {requested.upper()} solicitada por el usuario."
        if objective not in REGRESSION_OBJECTIVES:
            rationale += f" El metric_context {objective!r} no se reconoce y no altera la métrica explícita."
    elif objective not in REGRESSION_OBJECTIVES:
        selected, source = None, "needs_input"
        rationale = f"El objetivo de métrica {objective!r} no está reconocido; elija un objetivo estructurado compatible."
    elif objective in choices:
        selected, source = choices[objective], "structured_context"
        rationale = {
            "absolute_error_cost": "MAE resume la magnitud absoluta media con penalización lineal; resulta coherente cuando el coste crece aproximadamente de forma lineal con el error absoluto.",
            "large_errors_cost_more": "RMSE eleva al cuadrado los residuales antes de promediar, por lo que da mayor peso relativo a errores grandes.",
            "variance_explanation": "R² es una medida adimensional de variabilidad explicada; puede ser negativo y no expresa por sí solo la magnitud del error en unidades del target.",
        }[objective]
    elif objective == "official_metric":
        selected, source = None, "needs_input"
        rationale = "Se indicó que hay una métrica oficial, pero no se proporcionó una métrica compatible. Indique mae, rmse o r2."
    elif objective == "relative_error":
        selected, source = None, "needs_input"
        rationale = "El objetivo requiere errores relativos y no hay una métrica relativa compatible implementada. Elija MAE, RMSE o R² si alguna responde a su objetivo."
    else:
        selected, source = None, "needs_input"
        rationale = "No se indicó una métrica ni un objetivo explícito suficiente para elegir entre MAE, RMSE y R². Elija una métrica o metric_context."
    tradeoffs: list[str] = []
    if requested and objective in choices and choices[objective] != requested:
        tradeoffs.append(f"El objetivo {objective} suele sugerir {choices[objective].upper()}, pero se conserva {requested.upper()} como se solicitó; revise el trade-off.")
    if selected == "r2":
        tradeoffs.append("R² es adimensional, puede ser negativo y no comunica por sí solo la magnitud del error en la unidad del target.")
    if selected in {"mae", "rmse"}:
        tradeoffs.append("La métrica conserva la escala del target; sin una unidad declarada no se asigna interpretación física al score.")
    if selected and objective not in REGRESSION_OBJECTIVES and requested:
        tradeoffs.append("Se ignoró metric_context porque su valor no está reconocido; la métrica explícita conserva prioridad.")
    alternatives = [{"metric": metric, "appropriate_when": description,
                     "evidence_chunk_ids": _metric_evidence_ids(metric, evidence or [])} for metric, description in [
        ("mae", "la magnitud absoluta media y una penalización aproximadamente lineal representan el objetivo"),
        ("rmse", "los errores grandes deben recibir mayor peso relativo"),
        ("r2", "interesa complementar el error con una medida adimensional de variabilidad explicada")
    ] if metric != selected]
    ids = _metric_evidence_ids(selected, evidence or []) if selected else []
    if selected and not ids:
        rationale += " No se recuperó un fragmento que respalde propiedades de esta métrica; la elección se basa solo en la solicitud y el score es una medición de la corrida."
    return {"requested_metric": request.metric, "selected_metric": selected,
            "selection_source": source, "objective": objective,
            "target_unit": request.target_unit, "rationale": rationale,
            "tradeoffs": tradeoffs, "alternatives": alternatives,
            "evidence_chunk_ids": ids}


def build_risk_assessment(task_type: str, audit: DatasetAudit,
                          baseline: dict[str, Any] | None,
                          split_strategy: str | None = None) -> dict[str, Any]:
    observed: list[str] = [f"Dataset: {audit.rows} filas y {audit.columns} columnas; {audit.duplicate_rows} filas duplicadas."]
    review = list(audit.review_signals)
    protocol: list[str] = []
    validation = baseline.get("validation", {}) if baseline else {}
    is_cv = validation.get("effective_folds") is not None
    if task_type == REGRESSION:
        stats = audit.target_statistics or {}
        if stats:
            observed.append("Estadísticas calculadas del target: " + json.dumps(stats, ensure_ascii=False, sort_keys=True) + ".")
        if baseline and baseline.get("metric") == "rmse":
            review.append("RMSE da mayor peso relativo a errores grandes; revisar errores individuales para determinar si esa penalización representa el objetivo.")
        strategy = baseline.get("split_strategy", split_strategy or "random_holdout") if baseline else split_strategy or "random_holdout"
        if is_cv:
            protocol.append("La media y desviación estándar resumen únicamente los folds de desarrollo observados; no son intervalo de confianza ni garantía de generalización.")
        else:
            protocol.append(f"El protocolo {strategy} usa una sola partición; sus garantías se limitan al split declarado y no demuestran generalización ni ausencia total de leakage.")
        test_rows = baseline.get("test_rows") if baseline else max(1, round(audit.rows * 0.2))
        if test_rows < 10:
            protocol.append(f"Validación pequeña: {test_rows} caso(s); el score de una partición puede variar con los casos incluidos.")
            protocol.append("La comparación local incluye un predictor constante de la media del target aprendido solo con train.")
        if not is_cv:
            protocol.append("Una sola partición de validación no mide estabilidad entre splits ni garantiza generalización.")
    else:
        observed.append("Distribución calculada de clases: " + json.dumps(audit.target_distribution, ensure_ascii=False, sort_keys=True) + ".")
        counts = list(audit.target_distribution.values())
        if len(counts) == 2 and min(counts) / max(counts) < 0.5:
            review.append("La proporción observada entre clases es inferior a 1:2; revisar si la distribución y la métrica reflejan el objetivo del problema.")
        strategy = baseline.get("split_strategy", split_strategy or "stratified_holdout") if baseline else split_strategy or "stratified_holdout"
        if is_cv:
            protocol.append("La media y desviación estándar resumen únicamente los folds observados; no son intervalo de confianza ni garantía de generalización.")
        else:
            protocol.append(f"Un único split {strategy} no mide estabilidad entre particiones ni demuestra generalización o ausencia total de leakage.")
        if strategy == "time":
            protocol.append("El orden cronológico evita solapamiento entre períodos, pero una feature aún puede contener información futura.")
    return {"observed_facts": observed, "review_signals": review,
            "protocol_limitations": protocol, "unsupported_claims": []}


def _planned_validation(request: ResearchRequest) -> str:
    if request.validation_strategy == "cross_validation":
        return "K-fold aleatorio solicitado explícitamente; requiere independencia declarada"
    if request.validation_strategy == "auto":
        return "selección auto determinista entre holdout y K-fold según independencia y factibilidad"
    if request.split_strategy != "random":
        return f"split {request.split_strategy} con test_size={request.test_size}, semilla fija"
    return "holdout aleatorio con test_size solicitado, semilla fija" if request.task_type == REGRESSION else "holdout estratificado con test_size solicitado, semilla fija"


@dataclass
class ResearchRequest:
    csv_path: str | None
    target: str | None
    task_type: str | None = "binary_classification"
    metric: str | None = None
    seed: int = 42
    context: str | None = None
    metric_context: str | None = None
    target_unit: str | None = None
    split_strategy: str = "random"
    group_column: str | None = None
    time_column: str | None = None
    test_size: float = 0.2
    time_format: str | None = None
    validation_strategy: str = "holdout"
    observation_structure: str = "unknown"
    cv_folds: int | None = None
    max_cv_fits: int | None = None


@dataclass
class ColumnReviewSignal:
    column_name: str
    signal_type: str
    status: str
    rule_id: str
    rule_description: str
    evidence: dict[str, Any]
    why_review: str
    review_question: str
    data_scope: dict[str, Any]
    limitation: str = "Señal heurística: no confirma error ni leakage y no excluye ni transforma la columna."


@dataclass
class DatasetAudit:
    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    categorical_cardinality: dict[str, int]
    target_distribution: dict[str, int]
    constant_columns: list[str]
    review_signals: list[str]
    dataset_sha256: str
    target_statistics: dict[str, float | int] | None = None
    column_review_signals: list[dict[str, Any]] = field(default_factory=list)
    human_review: dict[str, Any] = field(default_factory=lambda: {"status": "pending", "responses": []})


@dataclass
class ResearchResult:
    status: str
    summary: str
    dataset_audit: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    next_actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    split_diagnostics: dict[str, Any] | None = None
    model_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Retriever(Protocol):
    def __call__(self, query: str, **kwargs: Any) -> list[dict[str, Any]]: ...


class Planner(Protocol):
    def plan(self, request: ResearchRequest, audit: DatasetAudit,
             evidence: list[dict[str, Any]]) -> dict[str, Any]: ...


class DeterministicPlanner:
    """Plan local sin un proveedor externo ni generación no controlada."""

    def plan(self, request: ResearchRequest, audit: DatasetAudit,
             evidence: list[dict[str, Any]]) -> dict[str, Any]:
        cites = [{"chunk_id": e["chunk_id"], "source_path": e["source_path"],
                  "section_path": e["section_path"]} for e in evidence]
        metric_file = "/metrics/regression_metrics.md" if request.task_type == REGRESSION else "/metrics/classification_metrics.md"
        has_metric_context = any(metric_file in e["source_path"] for e in evidence)
        has_validation_context = any("/validation/validation_strategy.md" in e["source_path"] for e in evidence)
        sufficient = has_metric_context and has_validation_context
        is_regression = request.task_type == REGRESSION
        metric_selection = select_regression_metric(request, evidence) if is_regression else {
            "requested_metric": request.metric, "selected_metric": request.metric or "f1",
            "selection_source": "user_explicit" if request.metric else "classification_default",
            "objective": request.metric_context or "unspecified", "target_unit": request.target_unit,
            "rationale": "Se conserva la métrica compatible de clasificación solicitada." if request.metric else "Se conserva F1 como métrica predeterminada de clasificación binaria.",
            "tradeoffs": [], "alternatives": [], "evidence_chunk_ids": []}
        selected_metric = metric_selection["selected_metric"]
        risks = build_risk_assessment(request.task_type, audit, None, request.split_strategy)
        return {
            "task_type": request.task_type, "target": request.target, "metric": selected_metric,
            "metric_selection": metric_selection,
            "metric_reason": metric_selection["rationale"],
            "validation": _planned_validation(request),
            "preprocessing": "imputación mediana y estandarización para columnas numéricas; imputación por categoría más one-hot para categóricas, ajustadas solo con train",
            "model": "regresión lineal ridge fija implementada con NumPy" if is_regression else "regresión logística binaria L2 implementada con NumPy",
            "seed": request.seed,
            "risks_to_review": [x for key in ("review_signals", "protocol_limitations") for x in risks[key]],
            "evidence_basis": cites,
            "recommendations": [
                {"text": f"Interpretar {selected_metric} solo dentro del split declarado.",
                 "evidence_chunk_ids": [e["chunk_id"] for e in evidence if "/validation/validation_strategy.md" in e["source_path"]]},
                {"text": metric_selection["rationale"],
                 "evidence_chunk_ids": metric_selection["evidence_chunk_ids"]},
            ],
            "evidence_sufficient": sufficient,
            "evidence_limitation": None if sufficient else f"La recuperación no aportó evidencia localizada tanto de métricas de {'regresión' if is_regression else 'clasificación'} como de validación; no se presentan conclusiones documentales como respaldadas.",
            "tool_facts": {"rows": audit.rows, "columns": audit.columns,
                           "target_distribution": audit.target_distribution,
                           "target_statistics": audit.target_statistics},
            "agent_decisions": ["Se comparan dos candidatos fijos (constante y lineal) con las mismas particiones; no se ajustan hiperparámetros ni se usa un holdout final para selección."],
        }


def validate_request(request: ResearchRequest) -> list[str]:
    issues: list[str] = []
    if not request.csv_path:
        issues.append("Falta la ruta del archivo CSV; indique csv_path.")
    else:
        path = Path(request.csv_path).expanduser()
        if path.suffix.casefold() != ".csv":
            issues.append("La ruta debe apuntar a un archivo .csv; exporte o seleccione un CSV.")
        elif not path.is_file():
            issues.append("No existe el archivo CSV indicado; compruebe la ruta y los permisos.")
    if not request.target:
        issues.append("Falta la columna objetivo; indique target.")
    if request.task_type not in TASK_METRICS:
        issues.append(f"Tarea no soportada: {request.task_type!r}; use una de {sorted(TASK_METRICS)}.")
    elif not request.metric and request.task_type == TASK:
        pass  # F1 remains the established default for binary classification.
    elif not request.metric and request.task_type == REGRESSION:
        if not request.metric_context or request.metric_context == "unspecified":
            issues.append("needs_input: falta una métrica o un objetivo metric_context; indique mae/rmse/r2 o un objetivo estructurado.")
        elif request.metric_context not in REGRESSION_OBJECTIVES:
            issues.append(f"Objetivo metric_context no soportado; use uno de {sorted(REGRESSION_OBJECTIVES)}.")
        elif request.metric_context in {"relative_error", "official_metric"}:
            issues.append(select_regression_metric(request)["rationale"])
    elif request.metric.lower() not in TASK_METRICS[request.task_type]:
        issues.append(f"Métrica incompatible o no soportada para {request.task_type}: {request.metric!r}; use una de {sorted(TASK_METRICS[request.task_type])}.")
    if not isinstance(request.seed, int) or isinstance(request.seed, bool):
        issues.append("La semilla debe ser un entero.")
    if request.split_strategy not in {"random", "group", "time"}:
        issues.append("Estrategia de split no soportada; use random, group o time.")
    if request.group_column and request.time_column:
        issues.append("No se admite combinar split por grupo y tiempo en esta iteración; solicite un protocolo combinado cuando esté implementado.")
    if request.split_strategy == "group" and not request.group_column:
        issues.append("needs_input: split_strategy=group requiere group_column.")
    if request.split_strategy == "time" and not request.time_column:
        issues.append("needs_input: split_strategy=time requiere time_column.")
    if request.group_column and request.split_strategy != "group":
        issues.append("group_column solo se admite con split_strategy=group; indique la estrategia explícitamente.")
    if request.time_column and request.split_strategy != "time":
        issues.append("time_column solo se admite con split_strategy=time; indique la estrategia explícitamente.")
    if not isinstance(request.test_size, (int, float)) or isinstance(request.test_size, bool) or not 0 < request.test_size < 1:
        issues.append("test_size debe ser una fracción estrictamente entre 0 y 1.")
    if request.time_format is not None and (not isinstance(request.time_format, str) or not request.time_format):
        issues.append("time_format debe ser un formato strptime no vacío.")
    if request.time_format and request.split_strategy != "time":
        issues.append("time_format solo se admite con split_strategy=time.")
    if request.validation_strategy not in {"holdout", "cross_validation", "auto"}:
        issues.append("validation_strategy debe ser holdout, cross_validation o auto.")
    if request.observation_structure not in {"independent", "grouped", "temporal", "unknown"}:
        issues.append("observation_structure debe ser independent, grouped, temporal o unknown.")
    if request.cv_folds is not None and (not isinstance(request.cv_folds, int) or isinstance(request.cv_folds, bool) or request.cv_folds < 2):
        issues.append("cv_folds debe ser un entero de al menos 2.")
    if request.max_cv_fits is not None and (not isinstance(request.max_cv_fits, int) or isinstance(request.max_cv_fits, bool) or request.max_cv_fits < 1):
        issues.append("max_cv_fits debe ser un entero positivo.")
    if request.validation_strategy == "cross_validation":
        if request.split_strategy in {"group", "time"} or request.observation_structure in {"grouped", "temporal"}:
            issues.append("unsupported: CV aleatorio ordinario no admite observaciones grouped/temporal; conserve el split explícito por grupo/tiempo o use holdout.")
        elif request.observation_structure != "independent":
            issues.append("needs_input: cross_validation requiere declarar --observation-structure independent; no se infiere independencia del CSV.")
    if request.validation_strategy == "auto" and request.observation_structure in {"grouped", "temporal"} and request.split_strategy == "random":
        issues.append("unsupported: auto no puede usar CV aleatorio en estructura grouped/temporal; indique un split por grupo/tiempo o holdout adecuado.")
    if request.observation_structure == "grouped" and request.split_strategy != "group":
        issues.append("needs_input: observation_structure=grouped requiere conservar un split explícito por grupo con --split-strategy group y --group-column.")
    if request.observation_structure == "temporal" and request.split_strategy != "time":
        issues.append("needs_input: observation_structure=temporal requiere conservar un split explícito por tiempo con --split-strategy time y --time-column.")
    return issues


def _missing(value: str | None) -> bool:
    return value is None or value.strip().casefold() in _MISSING


def _review_signal(column: str, kind: str, rule: str, description: str,
                   evidence: dict[str, Any], why: str, question: str,
                   scope: dict[str, Any], status: str = "review_recommended") -> dict[str, Any]:
    return asdict(ColumnReviewSignal(column, kind, status, rule, description,
                                     evidence, why, question, scope))


def _date_parse(value: str, explicit_format: str | None = None) -> tuple[bool, bool]:
    """Return (parsed, ambiguous); never choose day/month ordering implicitly."""
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True, False
    except ValueError:
        pass
    numeric_date = re.fullmatch(r"(\d{1,2})([/-])(\d{1,2})\2(\d{4})", value)
    if numeric_date:
        first, separator, second, year = numeric_date.groups()
        first_num, second_num = int(first), int(second)
        if first_num > 12 and second_num <= 12:
            try:
                datetime.strptime(value, f"%d{separator}%m{separator}%Y")
                return True, False
            except ValueError:
                return False, False
        if second_num > 12 and first_num <= 12:
            try:
                datetime.strptime(value, f"%m{separator}%d{separator}%Y")
                return True, False
            except ValueError:
                return False, False
        if first_num > 12 and second_num > 12:
            return False, False
        if explicit_format:
            try:
                datetime.strptime(value, explicit_format)
                return True, False
            except ValueError:
                pass
        return False, True
    if explicit_format:
        try:
            datetime.strptime(value, explicit_format)
            return True, False
        except ValueError:
            pass
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            datetime.strptime(value, fmt)
            return True, False
        except ValueError:
            continue
    return False, False


def _structure_signals(column_names: list[str], rows: list[dict[str, str]],
                       column_types: dict[str, str], target: str,
                       time_format: str | None = None) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    row_count = len(rows)
    for col in column_names:
        observed = [row[col].strip() for row in rows if not _missing(row[col])]
        unique = len(set(observed))
        ratio = unique / len(observed) if observed else 0.0
        scope = {"data_scope": "full_csv", "rows_available": row_count}
        normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", col).casefold()
        tokens = set(re.split(r"[^a-z0-9]+", normalized))
        suggestive = bool(tokens & {"id", "identifier", "key", "uuid", "code", "codigo", "clave"})
        if col != target and suggestive:
            signals.append(_review_signal(col, "possible_identifier", "identifier.name_token",
                "El nombre contiene un token que suele usarse para identificadores.",
                {"name_token_detected": True, "observed_non_empty": len(observed),
                 "unique_values": unique, "uniqueness_ratio": round(ratio, 6)},
                "El nombre sugiere una clave, pero no determina si la columna representa una entidad, una medición u otra cosa.",
                "¿Qué entidad o concepto representa esta columna en el dominio?", scope))
        if col != target and len(observed) >= ID_MIN_OBSERVATIONS and ratio >= ID_UNIQUENESS_RATIO:
            numeric = bool(observed) and all(_is_float(value) for value in observed)
            sequence = False
            sequence_step_consistency = None
            if numeric:
                numbers = sorted(float(value) for value in observed)
                differences = np.diff(numbers)
                if len(differences) and float(np.median(differences)) > 0:
                    step = float(np.median(differences))
                    sequence_step_consistency = sum(math.isclose(float(diff), step, rel_tol=0.01, abs_tol=1e-9)
                                                    for diff in differences) / len(differences)
                sequence = len(set(numbers)) == len(numbers) and (
                    max(numbers) - min(numbers) <= len(numbers) * ID_SEQUENCE_RANGE_PER_OBSERVATION
                    or (sequence_step_consistency is not None
                        and sequence_step_consistency >= ID_SEQUENCE_STEP_CONSISTENCY))
            signals.append(_review_signal(col, "possible_identifier", "identifier.high_uniqueness",
                f"Al menos {ID_MIN_OBSERVATIONS} observaciones y unicidad ≥ {ID_UNIQUENESS_RATIO:.0%}.",
                {"observed_non_empty": len(observed), "unique_values": unique,
                 "uniqueness_ratio": round(ratio, 6), "inferred_type": column_types[col],
                 "numeric_sequence_like": sequence,
                 "sequence_step_consistency": sequence_step_consistency},
                "La unicidad alta puede corresponder a una clave o a una medición válida; conviene confirmar su semántica y disponibilidad.",
                "¿Cada valor identifica una entidad/observación o mide una propiedad relevante?", scope))
        elif col != target and suggestive and 0 < len(observed) < ID_MIN_OBSERVATIONS:
            signals.append(_review_signal(col, "possible_identifier", "identifier.insufficient_sample",
                f"La regla de unicidad requiere al menos {ID_MIN_OBSERVATIONS} valores observados.",
                {"observed_non_empty": len(observed), "unique_values": unique,
                 "uniqueness_ratio": round(ratio, 6), "minimum_required": ID_MIN_OBSERVATIONS},
                "La muestra es demasiado pequeña para una conclusión firme sobre unicidad.",
                "¿Cuál es el significado de esta columna y cuántos valores distintos espera en producción?",
                scope, "not_evaluable"))
        elif col != target and 0 < len(observed) < ID_MIN_OBSERVATIONS and ratio >= HIGH_CARDINALITY_RATIO:
            signals.append(_review_signal(col, "possible_identifier", "identifier.insufficient_sample",
                f"La regla de unicidad requiere al menos {ID_MIN_OBSERVATIONS} valores observados.",
                {"observed_non_empty": len(observed), "unique_values": unique,
                 "uniqueness_ratio": round(ratio, 6), "minimum_required": ID_MIN_OBSERVATIONS},
                "La proporción observada es alta, pero la muestra no permite una conclusión firme.",
                "¿La unicidad es estructural o casual en esta muestra?", scope, "not_evaluable"))

        if col != target and observed:
            date_results = [_date_parse(value, time_format) for value in observed]
            parsed = sum(ok for ok, _ in date_results)
            ambiguous = sum(is_ambiguous for _, is_ambiguous in date_results)
            if parsed or ambiguous or tokens & {"date", "datetime", "timestamp", "time", "fecha", "hora"}:
                invalid = len(observed) - parsed - ambiguous
                status = "not_evaluable" if ambiguous and parsed == 0 else "review_recommended"
                signals.append(_review_signal(col, "possible_datetime", "datetime.parseability",
                "Reconoce ISO y formatos inequívocos; un formato strptime explícito puede resolver formatos adicionales.",
                    {"observed_non_empty": len(observed), "parseable_count": parsed,
                     "ambiguous_count": ambiguous, "invalid_count": invalid,
                     "missing_count": row_count - len(observed),
                     "parseable_ratio": round(parsed / len(observed), 6)},
                    "Una fecha puede aportar contexto temporal, pero no demuestra que la tarea requiera validación temporal.",
                    "¿Está disponible al predecir y representa el orden temporal relevante para la tarea?", scope, status))

        numeric_values: list[float] = []
        numeric_parse_count = 0
        nonfinite_count = 0
        for row in rows:
            value = row[col].strip()
            if _missing(value):
                continue
            try:
                number = float(value)
                numeric_parse_count += 1
                if math.isfinite(number):
                    numeric_values.append(number)
                else:
                    nonfinite_count += 1
            except ValueError:
                pass
        non_missing_count = sum(not _missing(row[col]) for row in rows)
        is_numeric = non_missing_count > 0 and numeric_parse_count == non_missing_count
        if is_numeric and len(numeric_values) >= OUTLIER_MIN_OBSERVATIONS:
            q1, median, q3 = np.quantile(numeric_values, [0.25, 0.5, 0.75]).tolist()
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outside = sum(value < lower or value > upper for value in numeric_values)
                if outside:
                    signals.append(_review_signal(col, "possible_outlier", "outlier.tukey_1_5_iqr",
                        "Cerca de Tukey: fuera de [Q1 − 1.5×IQR, Q3 + 1.5×IQR].",
                        {"valid_observations": len(numeric_values), "q1": float(q1),
                         "median": float(median), "q3": float(q3), "iqr": float(iqr),
                         "lower_fence": float(lower), "upper_fence": float(upper),
                         "outside_count": outside, "outside_ratio": outside / len(numeric_values),
                         "method": "Tukey 1.5×IQR", "missing_count": row_count - numeric_parse_count,
                         "non_finite_count": nonfinite_count},
                        "Un valor fuera de las cercas puede ser un error, un caso raro válido o una característica real de la distribución.",
                        "¿Estos valores son plausibles para el proceso y la unidad de esta variable?", scope))
            else:
                signals.append(_review_signal(col, "possible_outlier", "outlier.zero_iqr",
                    "La cerca de Tukey no es informativa porque IQR = 0.",
                    {"valid_observations": len(numeric_values), "q1": float(q1),
                     "median": float(median), "q3": float(q3), "iqr": 0.0,
                     "method": "Tukey 1.5×IQR", "outside_count": 0,
                     "evaluation_stability": "not_evaluable: IQR cero"},
                    "La distribución concentrada impide usar la cerca de Tukey de forma informativa.",
                    "¿La concentración de valores es esperada para esta variable?", scope, "not_evaluable"))
        elif is_numeric:
            signals.append(_review_signal(col, "possible_outlier", "outlier.small_sample",
                f"La regla de Tukey requiere al menos {OUTLIER_MIN_OBSERVATIONS} observaciones finitas.",
                {"valid_observations": len(numeric_values), "minimum_required": OUTLIER_MIN_OBSERVATIONS,
                 "missing_count": row_count - numeric_parse_count, "non_finite_count": nonfinite_count,
                 "evaluation_stability": "not_evaluable: muestra pequeña"},
                "Una muestra breve puede ocultar o exagerar valores extremos.",
                "¿Hay límites operativos conocidos para esta variable?", scope, "not_evaluable"))

        if unique <= 1:
            signals.append(_review_signal(col, "constant_column", "column.constant",
                "La columna tiene como máximo un valor no vacío distinto.",
                {"observed_non_empty": len(observed), "unique_values": unique,
                 "missing_count": row_count - len(observed)},
                "Una columna constante aporta poca variación en esta muestra, aunque su utilidad puede depender del proceso.",
                "¿Esta columna debería variar en el universo de uso previsto?", scope))
        elif col != target and (unique >= HIGH_CARDINALITY_MIN or ratio > HIGH_CARDINALITY_RATIO):
            signals.append(_review_signal(col, "high_cardinality", "category.high_cardinality",
                f"Cardinalidad ≥ {HIGH_CARDINALITY_MIN} o unicidad > {HIGH_CARDINALITY_RATIO:.0%}.",
                {"observed_non_empty": len(observed), "unique_values": unique,
                 "uniqueness_ratio": round(ratio, 6), "inferred_type": column_types[col]},
                "Muchas categorías pueden requerir interpretación de dominio y una estrategia de codificación adecuada.",
                "¿La variedad representa categorías legítimas, texto libre o claves?", scope))
    return signals


def _target_relation_signals(rows: list[dict[str, str]], columns: list[str], target: str,
                             task_type: str, train_indices: list[int],
                             split_strategy: str) -> list[dict[str, Any]]:
    """Target-related heuristics intentionally see training rows only."""
    signals: list[dict[str, Any]] = []
    n = len(train_indices)
    scope = {"data_scope": "train_only", "split_strategy": split_strategy,
             "training_rows": n}
    target_tokens = {"target", "label", "outcome", "result", "post", "after", "response",
                     "objetivo", "resultado", "posterior"}
    if n < TARGET_PROXY_MIN_TRAIN:
        return [_review_signal(target, "possible_target_proxy", "target_relation.small_train",
            f"Las reglas de relación requieren al menos {TARGET_PROXY_MIN_TRAIN} filas de entrenamiento.",
            {"training_rows": n, "minimum_required": TARGET_PROXY_MIN_TRAIN},
            "La relación predictor-target no puede estimarse con estabilidad en este split.",
            "¿Puede confirmarse disponibilidad temporal y origen de los predictores?", scope, "not_evaluable")]
    y_strings = [rows[i][target].strip() for i in train_indices]
    y_numeric = None
    if task_type == REGRESSION:
        y_numeric = np.asarray([float(value) for value in y_strings], dtype=float)
    for col in columns:
        if col == target:
            continue
        tokens = set(re.split(r"[^a-z0-9]+", col.casefold()))
        if tokens & target_tokens:
            signals.append(_review_signal(col, "possible_target_proxy", "target_relation.risk_name",
                "El nombre contiene vocabulario que puede aludir al resultado o a información posterior; es solo un indicio léxico.",
                {"risk_tokens_present": sorted(tokens & target_tokens), "training_rows": n},
                "El nombre no prueba fuga, pero el momento de disponibilidad y la forma de construcción merecen confirmación.",
                "¿Esta variable existía al momento de predecir o se construyó usando el resultado?", scope))
        values = [rows[i][col].strip() for i in train_indices]
        if task_type == REGRESSION:
            try:
                x = np.asarray([float(value) for value in values], dtype=float)
                if not np.all(np.isfinite(x)) or np.std(x) == 0 or np.std(y_numeric) == 0:
                    continue
                correlation = float(np.corrcoef(x, y_numeric)[0, 1])
                if abs(correlation) >= TARGET_PROXY_ASSOCIATION:
                    signals.append(_review_signal(col, "possible_target_proxy", "target_relation.numeric_correlation",
                        f"Correlación de Pearson absoluta ≥ {TARGET_PROXY_ASSOCIATION:.0%} en entrenamiento.",
                        {"training_rows": n, "pearson_correlation": correlation,
                         "absolute_correlation": abs(correlation)},
                        "Una asociación alta puede ser legítima; conviene revisar si la variable deriva directa o temporalmente del target.",
                        "¿La relación es una característica disponible antes de la predicción o una consecuencia del resultado?", scope))
            except (ValueError, TypeError):
                pass
        else:
            try:
                x = np.asarray([float(value) for value in values], dtype=float)
                encoded_y = np.asarray([1.0 if value == sorted(set(y_strings))[-1] else 0.0 for value in y_strings])
                if np.all(np.isfinite(x)) and np.std(x) > 0 and np.std(encoded_y) > 0:
                    correlation = float(np.corrcoef(x, encoded_y)[0, 1])
                    if abs(correlation) >= TARGET_PROXY_ASSOCIATION:
                        signals.append(_review_signal(col, "possible_target_proxy", "target_relation.numeric_correlation",
                            f"Correlación punto-biserial absoluta ≥ {TARGET_PROXY_ASSOCIATION:.0%} en entrenamiento.",
                            {"training_rows": n, "point_biserial_correlation": correlation,
                             "absolute_correlation": abs(correlation)},
                            "Una asociación alta puede ser legítima; conviene revisar si la variable deriva directa o temporalmente del target.",
                            "¿La relación es una característica disponible antes de la predicción o una consecuencia del resultado?", scope))
            except (ValueError, TypeError):
                pass
            # Pureza de asociación: máxima fracción de una clase por categoría.
            groups: dict[str, Counter[str]] = {}
            for value, label in zip(values, y_strings):
                groups.setdefault(value, Counter())[label] += 1
            max_categories = max(HIGH_CARDINALITY_MIN, int(n * HIGH_CARDINALITY_RATIO))
            if 1 < len(groups) <= max_categories and all(sum(counts.values()) >= 2 for counts in groups.values()):
                matches = sum(max(counts.values()) for counts in groups.values())
                purity = matches / n
                if purity >= TARGET_PROXY_ASSOCIATION:
                    signals.append(_review_signal(col, "possible_target_proxy", "target_relation.category_purity",
                        f"Pureza por categoría ≥ {TARGET_PROXY_ASSOCIATION:.0%} en entrenamiento.",
                        {"training_rows": n, "distinct_values": len(groups),
                         "majority_class_purity": purity},
                        "La asociación puede reflejar una relación legítima o una variable construida desde la etiqueta.",
                        "¿La categoría estaba disponible antes de observar la clase objetivo?", scope))
    return signals


def audit_dataset(csv_path: str | Path, target: str, task_type: str = TASK,
                  time_format: str | None = None) -> tuple[DatasetAudit, list[dict[str, str]]]:
    path = Path(csv_path).expanduser()
    if path.suffix.casefold() != ".csv" or not path.is_file():
        raise ValueError("Se requiere un archivo CSV existente; verifique extensión y ruta.")
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError("El CSV está vacío; agregue encabezados y filas de datos.")
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines(), strict=True)
        headers = reader.fieldnames
        if not headers or any(not h.strip() for h in headers):
            raise ValueError("El CSV no tiene encabezados válidos; añada nombres de columna únicos.")
        if len(set(headers)) != len(headers):
            raise ValueError("El CSV contiene nombres de columna duplicados; renómbrelos antes de continuar.")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"CSV mal formado ({exc}); guárdelo como CSV UTF-8 con filas consistentes.") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Hay filas con más campos que el encabezado; corrija el CSV.")
    if not rows:
        raise ValueError("El CSV no contiene filas de datos; agregue observaciones.")
    if target not in headers:
        raise ValueError(f"No existe la columna objetivo {target!r}; seleccione uno de {headers}.")
    if any(_missing(row.get(target, "")) for row in rows):
        raise ValueError(f"El target {target!r} tiene valores faltantes; elimine o complete esas filas con criterio explícito.")
    distribution: dict[str, int] = {}
    target_statistics = None
    if task_type == TASK:
        for row in rows:
            label = row[target].strip()
            distribution[label] = distribution.get(label, 0) + 1
        if len(distribution) != 2:
            if len(distribution) == 1:
                raise ValueError(f"El target {target!r} es constante; se requieren exactamente dos clases para clasificación binaria.")
            raise ValueError(f"El target {target!r} tiene {len(distribution)} clases; esta fase admite exactamente dos.")
    elif task_type == REGRESSION:
        target_values = [float(row[target]) for row in rows if _is_float(row[target].strip())]
        if len(target_values) != len(rows):
            raise ValueError(f"El target {target!r} debe contener solo valores numéricos finitos para regresión.")
        arr = np.asarray(target_values, dtype=float)
        target_statistics = {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                             "min": float(np.min(arr)), "median": float(np.median(arr)),
                             "max": float(np.max(arr)), "unique_values": len(set(target_values))}
    else:
        raise ValueError(f"Tarea no soportada para auditoría: {task_type!r}.")
    names = list(headers)
    missing = {col: sum(_missing(row[col]) for row in rows) for col in names}
    types: dict[str, str] = {}
    cardinality: dict[str, int] = {}
    constants: list[str] = []
    signals: list[str] = []
    for col in names:
        vals = [row[col].strip() for row in rows if not _missing(row[col])]
        unique = set(vals)
        if len(unique) <= 1:
            constants.append(col)
        if vals and all(_is_float(v) for v in vals):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
            cardinality[col] = len(unique)
            if len(unique) > max(20, int(len(rows) * 0.5)):
                signals.append(f"{col}: cardinalidad alta ({len(unique)} valores); señal para revisión humana, según el significado de la columna")
        if col != target and unique and len(unique) == len(rows):
            signals.append(f"{col}: todos los valores son únicos; columna que merece revisión. La unicidad no confirma que sea un identificador; depende del significado del dato")
    if len(rows) < 20:
        signals.append("Dataset pequeño (<20 filas); el score del holdout tendrá alta variabilidad")
    row_counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(row[c].strip() for c in names)
        row_counts[key] = row_counts.get(key, 0) + 1
    audit = DatasetAudit(len(rows), len(names), names, types, missing,
                         sum(count - 1 for count in row_counts.values()),
                         cardinality, distribution, constants, signals,
                         hashlib.sha256(raw).hexdigest(), target_statistics)
    audit.column_review_signals = _structure_signals(names, rows, types, target, time_format)
    return audit, rows


def _is_float(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stratified_split(labels: list[str], seed: int, test_size: float = 0.2) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    if any(len(indices) < 2 for indices in by_class.values()):
        raise ValueError("Cada clase necesita al menos 2 filas para aparecer en train y test; agregue datos de la clase minoritaria.")
    train: list[int] = []
    test: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, round(len(indices) * test_size)))
        test.extend(indices[:n_test]); train.extend(indices[n_test:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def _encode(rows: list[dict[str, str]], feature_names: list[str], types: dict[str, str],
            train_indices: list[int], indices: list[int]) -> tuple[np.ndarray, list[str]]:
    """Ajusta imputación/escalado/categorías con train y transforma el split dado."""
    matrices: list[np.ndarray] = []
    output_names: list[str] = []
    for col in feature_names:
        if types[col] == "numeric":
            values = np.array([float(rows[i][col]) if not _missing(rows[i][col]) and _is_float(rows[i][col]) else np.nan for i in indices], dtype=float)
            train_values = np.array([float(rows[i][col]) if not _missing(rows[i][col]) and _is_float(rows[i][col]) else np.nan for i in train_indices], dtype=float)
            finite = train_values[np.isfinite(train_values)]
            median = float(np.median(finite)) if finite.size else 0.0
            values[~np.isfinite(values)] = median
            mean = float(np.mean(np.where(np.isfinite(train_values), train_values, median)))
            scale = float(np.std(np.where(np.isfinite(train_values), train_values, median))) or 1.0
            matrices.append(((values - mean) / scale).reshape(-1, 1)); output_names.append(col)
        else:
            observed = [rows[i][col].strip() for i in train_indices if not _missing(rows[i][col])]
            cats = sorted(set(observed))
            if not cats:
                cats = ["__MISSING__"]
            mode = Counter(observed).most_common(1)[0][0] if observed else cats[0]
            for cat in cats:
                matrices.append(np.array([1.0 if (rows[i][col].strip() if not _missing(rows[i][col]) else mode) == cat else 0.0 for i in indices]).reshape(-1, 1))
                output_names.append(f"{col}={cat}")
    if not matrices:
        raise ValueError("No hay predictores utilizables tras excluir el target; agregue columnas de entrada.")
    return np.concatenate(matrices, axis=1), output_names


def _metric_value(metric: str, truth: list[int], pred: list[int]) -> float:
    tp = sum(y == 1 and p == 1 for y, p in zip(truth, pred))
    tn = sum(y == 0 and p == 0 for y, p in zip(truth, pred))
    fp = sum(y == 0 and p == 1 for y, p in zip(truth, pred))
    fn = sum(y == 1 and p == 0 for y, p in zip(truth, pred))
    acc = (tp + tn) / len(truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if metric == "accuracy": return acc
    if metric == "balanced_accuracy": return ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    if metric == "precision": return precision
    if metric == "recall": return recall
    if metric == "f1": return f1
    raise ValueError(f"Métrica no soportada: {metric}")


def _regression_metric_value(metric: str, truth: list[float], pred: list[float]) -> float | None:
    """Calcula métricas regresivas; R² no definido se representa como None."""
    if not truth or len(truth) != len(pred):
        raise ValueError("Se requieren vectores no vacíos de igual longitud para evaluar regresión.")
    residual = np.asarray(truth, dtype=float) - np.asarray(pred, dtype=float)
    if metric == "mae": return float(np.mean(np.abs(residual)))
    if metric == "rmse": return float(np.sqrt(np.mean(residual ** 2)))
    if metric == "r2":
        actual = np.asarray(truth, dtype=float)
        denominator = float(np.sum((actual - np.mean(actual)) ** 2))
        if denominator == 0.0: return None
        return float(1.0 - np.sum(residual ** 2) / denominator)
    raise ValueError(f"Métrica de regresión no soportada: {metric}")


def _random_split(n_rows: int, seed: int) -> tuple[list[int], list[int]]:
    if n_rows < 2:
        raise ValueError("Regresión requiere al menos 2 filas para separar train y validación.")
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_test = min(n_rows - 1, max(1, round(n_rows * 0.2)))
    return indices[n_test:], indices[:n_test]


def _requested_split(n_rows: int, seed: int, test_size: float) -> tuple[list[int], list[int]]:
    if n_rows < 2:
        raise ValueError("Se requieren al menos 2 filas para separar train y validación.")
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_test = min(n_rows - 1, max(1, round(n_rows * test_size)))
    return indices[n_test:], indices[:n_test]


def _group_split(rows: list[dict[str, str]], column: str, seed: int,
                 test_size: float) -> tuple[list[int], list[int], dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        value = row.get(column)
        if _missing(value):
            raise ValueError(f"La columna de grupo {column!r} tiene valores vacíos o nulos; complete o elimine esas filas antes del split.")
        groups.setdefault(str(value).strip(), []).append(i)
    if len(groups) < 2:
        raise ValueError("Split por grupos imposible: se requieren al menos dos grupos distintos.")
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    target_rows = len(rows) * test_size
    chosen: list[str] = []
    current = 0
    for group in ordered[:-1]:
        size = len(groups[group])
        if abs(current + size - target_rows) < abs(current - target_rows):
            chosen.append(group); current += size
    if not chosen:
        chosen = [min(ordered[:-1], key=lambda g: (abs(len(groups[g]) - target_rows), g))]
    if len(chosen) == len(ordered):
        chosen.remove(max(chosen, key=lambda g: (len(groups[g]), g)))
    test_set = set(chosen)
    test = [i for group in sorted(test_set) for i in groups[group]]
    train = [i for group in sorted(set(groups) - test_set) for i in groups[group]]
    train_groups = {str(rows[i][column]).strip() for i in train}
    validation_groups = {str(rows[i][column]).strip() for i in test}
    intersection = sorted(train_groups & validation_groups)
    if not train or not test or intersection:
        raise ValueError("Split por grupos imposible: la asignación dejó una partición vacía o grupos compartidos.")
    diagnostics = {"strategy": "group", "group_column": column,
        "requested_fraction": test_size, "actual_fraction": len(test) / len(rows),
        "train_rows": len(train), "validation_rows": len(test),
        "train_groups": len(train_groups), "validation_groups": len(validation_groups),
        "group_intersection": intersection, "seed": seed, "warnings": [], "valid": True}
    if diagnostics["actual_fraction"] != test_size:
        diagnostics["warnings"].append("La fracción real difiere de test_size porque los grupos se mantienen íntegros.")
    return train, test, diagnostics


def _parse_timestamp(value: str, time_format: str | None, column: str) -> datetime:
    try:
        if time_format:
            return datetime.strptime(value.strip(), time_format)
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        # ISO parsing is deliberate: ambiguous local date conventions require an explicit format.
        return parsed
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Fecha inválida o ambigua en {column!r}; use ISO 8601 o especifique --time-format. No se guardan valores individuales.") from exc


def _time_split(rows: list[dict[str, str]], column: str, seed: int, test_size: float,
                time_format: str | None) -> tuple[list[int], list[int], dict[str, Any]]:
    parsed: list[datetime] = []
    for row in rows:
        value = row.get(column)
        if _missing(value):
            raise ValueError(f"La columna temporal {column!r} contiene valores vacíos o nulos.")
        parsed.append(_parse_timestamp(str(value), time_format, column))
    aware = [dt.utcoffset() is not None for dt in parsed]
    if any(aware) and not all(aware):
        raise ValueError("La columna temporal mezcla timestamps con y sin zona horaria; normalice la columna antes de continuar.")
    normalized = [dt.astimezone(timezone.utc) if dt.utcoffset() is not None else dt for dt in parsed]
    unique = sorted(set(normalized))
    if len(unique) < 2:
        raise ValueError("Split temporal imposible: se requiere más de un timestamp distinto.")
    target_rows = len(rows) * test_size
    counts = Counter(normalized)
    cuts = []
    cumulative = 0
    for timestamp in unique[:-1]:
        cumulative += counts[timestamp]
        cuts.append((abs(target_rows - (len(rows) - cumulative)), timestamp))
    _, cut_timestamp = min(cuts)
    train = [i for i, dt in enumerate(normalized) if dt <= cut_timestamp]
    test = [i for i, dt in enumerate(normalized) if dt > cut_timestamp]
    if not train or not test or max(normalized[i] for i in train) >= min(normalized[i] for i in test):
        raise ValueError("Split temporal imposible: no se obtuvo una partición no vacía con orden cronológico estricto.")
    fmt = lambda dt: dt.isoformat()
    diagnostics = {"strategy": "time", "time_column": column,
        "requested_fraction": test_size, "actual_fraction": len(test) / len(rows),
        "train_rows": len(train), "validation_rows": len(test),
        "train_period": [fmt(min(normalized[i] for i in train)), fmt(max(normalized[i] for i in train))],
        "validation_period": [fmt(min(normalized[i] for i in test)), fmt(max(normalized[i] for i in test))],
        "cut": fmt(cut_timestamp), "strict_chronology": True, "seed": seed,
        "warnings": [], "valid": True}
    if diagnostics["actual_fraction"] != test_size:
        diagnostics["warnings"].append("La fracción real difiere de test_size porque el corte conserva juntos timestamps empatados.")
    return train, test, diagnostics


def _split_indices(request: ResearchRequest, rows: list[dict[str, str]], labels: list[str] | None = None):
    if request.split_strategy == "group":
        if request.group_column not in rows[0]:
            raise ValueError(f"La columna de grupo {request.group_column!r} no existe en el CSV; corrija group_column.")
        train, test, diagnostics = _group_split(rows, str(request.group_column), request.seed, request.test_size)
    elif request.split_strategy == "time":
        if request.time_column not in rows[0]:
            raise ValueError(f"La columna temporal {request.time_column!r} no existe en el CSV; corrija time_column.")
        train, test, diagnostics = _time_split(rows, str(request.time_column), request.seed, request.test_size, request.time_format)
    else:
        if labels is not None:
            train, test = _stratified_split(labels, request.seed, request.test_size)
        else:
            train, test = _requested_split(len(rows), request.seed, request.test_size)
        diagnostics = {"strategy": "stratified_holdout" if labels is not None else "random_holdout",
            "requested_fraction": request.test_size, "actual_fraction": len(test) / len(rows),
            "train_rows": len(train), "validation_rows": len(test), "seed": request.seed,
            "warnings": [], "valid": True}
    return train, test, diagnostics


def _unverified_split_diagnostics(request: ResearchRequest, issues: list[str]) -> dict[str, Any]:
    """Describe a requested protocol that could not be verified or executed."""
    return {"strategy": request.split_strategy,
        "group_column": request.group_column if request.split_strategy == "group" else None,
        "time_column": request.time_column if request.split_strategy == "time" else None,
        "requested_fraction": request.test_size, "actual_fraction": None,
        "train_rows": None, "validation_rows": None,
        "train_groups": None, "validation_groups": None, "group_intersection": None,
        "train_period": None, "validation_period": None, "cut": None,
        "strict_chronology": None, "seed": request.seed,
        "warnings": list(issues), "valid": False}


def _kfold_indices(n_rows: int, folds: int, seed: int,
                   labels: list[str] | None = None) -> list[tuple[list[int], list[int]]]:
    """Deterministic shuffled K-fold; stratifies labels when supplied."""
    buckets: list[list[int]] = [[] for _ in range(folds)]
    rng = random.Random(seed)
    if labels is None:
        order = list(range(n_rows))
        rng.shuffle(order)
        for position, idx in enumerate(order):
            buckets[position % folds].append(idx)
    else:
        for label in sorted(set(labels)):
            indices = [i for i, value in enumerate(labels) if value == label]
            rng.shuffle(indices)
            for position, idx in enumerate(indices):
                buckets[position % folds].append(idx)
    result = []
    all_indices = set(range(n_rows))
    for bucket in buckets:
        validation = sorted(bucket)
        train = sorted(all_indices - set(validation))
        if not train or not validation:
            raise ValueError("K-fold imposible: se generó un fold vacío.")
        result.append((train, validation))
    return result


def _validation_choice(request: ResearchRequest, audit: DatasetAudit,
                       labels: list[str] | None,
                       rows: list[dict[str, str]],
                       split_cache: dict[int, list[tuple[list[int], list[int]]]] | None = None) -> tuple[str, int, list[str]]:
    """Return effective protocol, folds requested/effective, and explicit rationale."""
    strategy = request.validation_strategy
    requested = request.cv_folds if request.cv_folds is not None else 5
    fit_limit = request.max_cv_fits if request.max_cv_fits is not None else 5
    notes: list[str] = []
    if strategy == "holdout":
        return "holdout", 0, ["Se conserva holdout porque validation_strategy=holdout (predeterminado compatible)."]
    if request.observation_structure != "independent":
        message = "K-fold requiere independencia declarada explícitamente; observation_structure es unknown o no independiente."
        if strategy == "cross_validation":
            raise ValueError(message)
        return "holdout", 0, [f"auto eligió holdout: {message}"]
    if request.split_strategy in {"group", "time"} or request.observation_structure in {"grouped", "temporal"}:
        message = "la estructura grouped/temporal requiere conservar su split explícito; K-fold aleatorio ordinario no es compatible."
        if strategy == "cross_validation":
            raise ValueError(message)
        return "holdout", 0, [f"auto eligió holdout: {message}"]
    n = audit.rows
    if strategy == "auto" and n < 20:
        return "holdout", 0, ["auto eligió holdout: el umbral documentado para CV es al menos 20 filas."]
    maximum = min(requested, fit_limit, n)
    if labels is not None:
        minimum_support = min(Counter(labels).values())
        maximum = min(maximum, minimum_support)
    if strategy == "cross_validation" and maximum < requested:
        why = f"CV solicitada con {requested} folds excede soporte por clase/filas o max_cv_fits={fit_limit}; no se reduce K en una solicitud explícita."
        raise ValueError(why)
    if strategy == "auto" and maximum < requested:
        notes.append(f"auto redujo folds de {requested} a {maximum}: mínimo entre filas, soporte por clase y max_cv_fits={fit_limit}.")
    candidate = maximum
    while candidate >= 2:
        if split_cache is not None:
            if candidate not in split_cache:
                split_cache[candidate] = _kfold_indices(n, candidate, request.seed, labels)
            splits = split_cache[candidate]
        else:
            splits = _kfold_indices(n, candidate, request.seed, labels)
        infeasible_reason = None
        if request.metric == "balanced_accuracy" and labels is not None and any(
                {labels[i] for i in valid} != set(labels) for _, valid in splits):
            feasible = False
            infeasible_reason = "balanced_accuracy requiere ambas clases en cada fold de validación"
        elif request.task_type == REGRESSION and request.metric == "r2":
            target = str(request.target)
            undefined = next((number for number, (_, valid) in enumerate(splits, 1)
                              if len({rows[i][target] for i in valid}) <= 1), None)
            feasible = undefined is None
            if undefined is not None:
                infeasible_reason = f"R² no está definido en el fold {undefined}: target constante en validación"
        else:
            feasible = True
        if feasible:
            return "cross_validation", candidate, notes
        if strategy != "auto":
            raise ValueError(f"Métrica imposible en al menos un fold: {infeasible_reason}.")
        candidate -= 1
        notes.append(f"auto redujo folds a {candidate}: {infeasible_reason}.")
    if strategy == "auto":
        return "holdout", 0, notes + ["auto eligió holdout: no quedó una partición K-fold válida con la métrica y restricciones indicadas."]
    raise ValueError("CV imposible: no hay al menos dos folds no vacíos factibles.")


def run_baseline(request: ResearchRequest, audit: DatasetAudit,
                 rows: list[dict[str, str]]) -> dict[str, Any]:
    return run_model_comparison(request, audit, rows)["baseline"]


def _candidate_prediction(candidate_id: str, task_type: str, x_train: np.ndarray,
                          x_valid: np.ndarray, y_train: list[Any],
                          classes: list[str]) -> list[Any]:
    """Fit one fixed candidate using only its fold's training data."""
    if task_type == TASK:
        encoded_train = np.asarray([1.0 if value == classes[1] else 0.0 for value in y_train])
        if candidate_id == "constant":
            counts = Counter(y_train)
            # Stable tie break: first label in the sorted class list.
            chosen = min(classes, key=lambda label: (-counts[label], label))
            return [1 if chosen == classes[1] else 0] * len(x_valid)
        design = np.column_stack([np.ones(len(x_train)), x_train])
        weights = np.zeros(design.shape[1], dtype=float)
        for _ in range(600):
            probs = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
            grad = design.T @ (probs - encoded_train) / len(encoded_train)
            grad[1:] += 0.01 * weights[1:]
            weights -= 0.1 * grad
        valid_design = np.column_stack([np.ones(len(x_valid)), x_valid])
        return (valid_design @ weights >= 0.0).astype(int).tolist()
    targets = np.asarray(y_train, dtype=float)
    if candidate_id == "constant":
        return [float(np.mean(targets))] * len(x_valid)
    design = np.column_stack([np.ones(len(x_train)), x_train])
    regularizer = np.eye(design.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return (np.column_stack([np.ones(len(x_valid)), x_valid]) @ weights).tolist()


def run_model_comparison(request: ResearchRequest, audit: DatasetAudit,
                         rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compare a constant reference and the existing linear model on shared splits."""
    target = str(request.target)
    classification = request.task_type == TASK
    labels = [row[target].strip() for row in rows] if classification else None
    classes = sorted(set(labels or []))
    split_cache: dict[int, list[tuple[list[int], list[int]]]] = {}
    effective, fold_count, notes = _validation_choice(request, audit, labels, rows, split_cache)
    if effective == "cross_validation":
        raw_splits = split_cache[fold_count]
        split_method = "veronica_kfold_v1"
        protocol = "stratified_kfold" if classification else "kfold"
    else:
        train, valid, diagnostic = _split_indices(request, rows, labels)
        raw_splits = [(train, valid)]
        split_method = "veronica_holdout_v1"
        protocol = diagnostic["strategy"]

    if classification:
        for fold_number, (train_indices, valid_indices) in enumerate(raw_splits, 1):
            classes_in_train = {labels[index] for index in train_indices}
            if classes_in_train != set(classes):
                absent = sorted(set(classes) - classes_in_train)
                raise ValueError(f"El split {request.split_strategy!r} deja al entrenamiento sin clases {absent}; agregue grupos/fechas suficientes o ajuste test_size.")
            validation_classes = {labels[index] for index in valid_indices}
            if request.metric == "balanced_accuracy" and validation_classes != set(classes):
                absent = sorted(set(classes) - validation_classes)
                raise ValueError(f"La métrica balanced_accuracy no es calculable: faltan clases {absent} en validación para el split {request.split_strategy!r}.")
    elif request.metric == "r2":
        for fold_number, (_, valid_indices) in enumerate(raw_splits, 1):
            if len({rows[index][target] for index in valid_indices}) <= 1:
                raise ValueError(f"R² no está definido en el fold {fold_number}: el target de validación es constante. Elija MAE/RMSE o revise el split.")

    features = [column for column in audit.column_names if column != target]
    candidate_specs = [
        {"id": "constant", "name": "Predictor constante", "family": "constant",
         "task": request.task_type, "parameters": {"strategy": "most_frequent" if classification else "mean"}, "complexity": 0},
        {"id": "linear", "name": "Modelo lineal", "family": "logistic_regression_l2" if classification else "ridge_regression",
         "task": request.task_type,
         "parameters": ({"learning_rate": 0.1, "iterations": 600, "l2": 0.01, "threshold": 0.5}
                         if classification else {"alpha": 1.0, "fit_intercept": True}), "complexity": 1},
    ]
    state = {spec["id"]: {**spec, "preprocessing": "numeric median imputation + standardization; categorical train mode + one-hot; fitted per train partition",
                           "folds": [], "errors": [], "fit_count": 0, "elapsed_seconds": 0.0}
             for spec in candidate_specs}
    partitions = []
    for fold_number, (train_indices, valid_indices) in enumerate(raw_splits, 1):
        part_hash = hashlib.sha256(json.dumps([train_indices, valid_indices], separators=(",", ":")).encode()).hexdigest()[:16]
        partitions.append({"id": f"partition-{fold_number}-{part_hash}", "fold": fold_number,
                           "train_rows": len(train_indices), "validation_rows": len(valid_indices),
                           "train_index_sha256": hashlib.sha256(json.dumps(train_indices).encode()).hexdigest(),
                           "validation_index_sha256": hashlib.sha256(json.dumps(valid_indices).encode()).hexdigest()})
        # All candidates use this same train-fitted transform and identical row order.
        x_train, encoded_columns = _encode(rows, features, audit.column_types, train_indices, train_indices)
        x_valid, _ = _encode(rows, features, audit.column_types, train_indices, valid_indices)
        y_train = ([labels[i] for i in train_indices] if classification else
                   [float(rows[i][target]) for i in train_indices])
        y_valid = ([1 if labels[i] == classes[1] else 0 for i in valid_indices] if classification else
                   [float(rows[i][target]) for i in valid_indices])
        for candidate_id in ("constant", "linear"):
            candidate = state[candidate_id]
            started = perf_counter()
            stage = "fit"
            candidate["fit_count"] += 1
            try:
                predictions = _candidate_prediction(candidate_id, request.task_type, x_train, x_valid, y_train, classes)
                stage = "evaluate"
                score = (_metric_value(str(request.metric), y_valid, predictions) if classification else
                         _regression_metric_value(str(request.metric), y_valid, predictions))
                if score is None:
                    raise ValueError(f"{request.metric} no está definida para las observaciones de validación de esta partición.")
                candidate["folds"].append({"fold": fold_number, "partition_id": partitions[-1]["id"],
                    "score": float(score), "train_rows": len(train_indices), "validation_rows": len(valid_indices), "status": "completed"})
            except Exception as exc:
                candidate["errors"].append({"candidate_id": candidate_id, "fold": fold_number,
                    "partition_id": partitions[-1]["id"], "stage": stage, "message": str(exc)})
                candidate["folds"].append({"fold": fold_number, "partition_id": partitions[-1]["id"],
                    "score": None, "train_rows": len(train_indices), "validation_rows": len(valid_indices), "status": "failed"})
            finally:
                candidate["elapsed_seconds"] += perf_counter() - started

    direction = metric_direction(str(request.metric))
    all_complete = all(not item["errors"] and len(item["folds"]) == len(raw_splits)
                       and all(fold["score"] is not None for fold in item["folds"]) for item in state.values())
    for candidate in state.values():
        scores = [fold["score"] for fold in candidate["folds"] if fold["score"] is not None]
        candidate["summary"] = ({"mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=0)),
                                 "std_ddof": 0, "weighted": False, "min": float(min(scores)), "max": float(max(scores)),
                                 "partial": bool(candidate["errors"])}
                                if scores and effective == "cross_validation" else
                                {"score": float(scores[0]), "partial": bool(candidate["errors"])} if scores else None)
        candidate["elapsed_seconds"] = float(candidate["elapsed_seconds"])
        candidate["elapsed_scope"] = "total model fitting, prediction and scoring across folds/partitions; excludes shared preprocessing; seconds; local wall time, not a cross-machine benchmark"
        candidate["status"] = "completed" if not candidate["errors"] else "incomplete"
    eligible = [item for item in state.values() if item["summary"]]
    winner = None
    selection_rule = "La métrica primaria decide; en empate exacto se elige el predictor constante por ser más sencillo."
    if all_complete and len(eligible) == len(state):
        winner = sorted(eligible, key=lambda item: ((-item["summary"].get("mean", item["summary"].get("score")))
                           if direction == "higher_is_better" else item["summary"].get("mean", item["summary"].get("score")), item["complexity"], item["id"]))[0]["id"]
    comparison = {"schema_version": 1, "status": "comparable" if winner else "incomplete_or_not_comparable",
        "task": request.task_type, "target": target, "metric": request.metric, "metric_direction": direction,
        "protocol": protocol, "requested_protocol": request.validation_strategy,
        "split_method": split_method, "seed": request.seed, "dataset_sha256": audit.dataset_sha256,
        "row_order": "CSV row positions starting at 0, excluding the header",
        "partition_construction": "constructed once and shared by all candidates; same train/validation indices and row order",
        "partitions": partitions, "preprocessing_fit_scope": "train rows only, separately within each shared partition",
        "candidates": list(state.values()), "selection_rule": selection_rule,
        "selected_candidate_id": winner, "selected_candidate_name": next((item["name"] for item in eligible if item["id"] == winner), None),
        "selection_scope": "Selección preliminar sobre el conjunto y protocolo de validación actuales; no afirma que sea el mejor modelo en general",
        "final_evaluation": {"available": False, "used_for_selection": False,
                             "limitation": "El flujo actual no distingue un holdout final separado; no se presenta la validación actual como evaluación final."},
        "warning": "El score no garantiza generalización; seleccionar modelos repetidamente puede sobreajustar la validación.",
        "notes": notes + (["CV resume scores por fold con media no ponderada y desviación estándar poblacional; no es intervalo de confianza."] if effective == "cross_validation" else ["El holdout actual funciona como validación del flujo y no mide variabilidad entre particiones."]),
        "adjustments_per_candidate": {key: item["fit_count"] for key, item in state.items()}}

    linear = state["linear"]
    successful = [fold for fold in linear["folds"] if fold["score"] is not None]
    linear_score = (linear["summary"]["mean"] if effective == "cross_validation" and linear["summary"] else
                    linear["summary"].get("score") if linear["summary"] else None)
    if linear_score is None and successful:
        linear_score = successful[0]["score"]
    first_partition = raw_splits[0]
    train_indices, valid_indices = first_partition
    split_diag = diagnostic if effective != "cross_validation" else {"strategy": protocol, "valid": True, "seed": request.seed,
        "train_rows": None, "validation_rows": None, "folds": fold_count, "dataset_sha256": audit.dataset_sha256,
        "warnings": notes}
    validation_folds = [{"fold": fold["fold"], "train_rows": fold["train_rows"], "validation_rows": fold["validation_rows"],
                         "score": fold["score"], "status": fold["status"], **({"validation_class_counts": {c: sum(labels[i] == c for i in raw_splits[fold["fold"]-1][1]) for c in classes}} if classification else {})}
                        for fold in linear["folds"]]
    linear_validation = {"strategy": protocol, "observation_structure": request.observation_structure,
        "requested_folds": request.cv_folds if request.cv_folds is not None else (5 if effective == "cross_validation" else None),
        "effective_folds": fold_count if effective == "cross_validation" else None, "seed": request.seed,
        "shuffle": effective == "cross_validation" or request.split_strategy == "random", "split_algorithm": split_method,
        "row_order": comparison["row_order"], "dataset_sha256": audit.dataset_sha256,
        "fit_count": linear["fit_count"], "score": linear_score, "folds": validation_folds,
        "aggregate": linear["summary"] if effective == "cross_validation" else None,
        "notes": comparison["notes"], "failed_folds": linear["errors"]}
    baseline = {"task_type": request.task_type, "split_strategy": protocol,
        "split_fraction": diagnostic.get("actual_fraction") if effective != "cross_validation" else None,
        "split_diagnostics": split_diag, "train_rows": len(train_indices), "test_rows": len(valid_indices),
        "seed": request.seed, "metric": request.metric, "score": linear_score,
        "model": "numpy_logistic_regression_l2" if classification else "numpy_ridge_regression",
        "parameters": candidate_specs[1]["parameters"], "predictor_columns": features,
        "encoded_columns": encoded_columns, "target_excluded": target not in features,
        "warnings": [comparison["warning"]], "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "validation": linear_validation, "model_comparison": comparison}
    if classification:
        baseline.update({"classes": classes,
            "train_class_counts": {c: sum(labels[i] == c for i in train_indices) for c in classes},
            "test_class_counts": {c: sum(labels[i] == c for i in valid_indices) for c in classes}})
    return {"baseline": baseline, "comparison": comparison}


def _retrieve(retriever: Retriever, request: ResearchRequest) -> list[dict[str, Any]]:
    is_regression = request.task_type == REGRESSION
    query = ("regresión tabular target continuo formulación del problema auditoría métricas validación leakage baseline lineal" if is_regression
             else "clasificación binaria formulación del problema auditoría de datos métricas validación leakage baseline logístico")
    if is_regression:
        metric_name = request.metric or {"absolute_error_cost": "mae", "large_errors_cost_more": "rmse",
                                         "variance_explanation": "r2"}.get(request.metric_context, "regresión")
        metric_query = {
            "mae": "MAE magnitud absoluta media error absoluto unidades objetivo penalización lineal",
            "rmse": "RMSE raíz error cuadrático da mayor peso relativo errores grandes residuales",
            "r2": "R² R2 fórmula RSS TSS adimensional variabilidad explicada magnitud del error",
        }.get(metric_name, "métricas regresión MAE RMSE R² target continuo")
    else:
        metric_query = "métrica F1 clasificación desbalance evaluación"
    model_query = "baseline regresión lineal ridge tabular" if is_regression else "baseline clasificación regresión logística"
    found: list[dict[str, Any]] = []
    validation_query = ("holdout aleatorio semilla evaluación sin leakage" if is_regression
                        else "validación estratificada leakage entrenamiento prueba")
    for q, topic in [(query, None), (metric_query, "metrics"),
                     (validation_query, "validation"),
                     (model_query, "models"),
                     ("auditoría valores faltantes cardinalidad target", "eda")]:
        try:
            for chunk in retriever(q, top_k=5, document_type="knowledge", topic=topic):
                if chunk.get("chunk_id") and chunk.get("text") and chunk.get("source_path"):
                    found.append(chunk)
        except Exception:
            continue
    unique: dict[str, dict[str, Any]] = {}
    for item in found:
        unique[item["chunk_id"]] = item
    return list(unique.values())[:12]


def validate_citations(citations: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> bool:
    available = {e.get("chunk_id"): e for e in evidence}
    for citation in citations:
        source = available.get(citation.get("chunk_id"))
        if not source or citation.get("source_path") != source.get("source_path"):
            return False
        if citation.get("section_path") != source.get("section_path"):
            return False
        if citation.get("text") != source.get("text"):
            return False
    return True


def validate_plan(plan: dict[str, Any], request: ResearchRequest,
                  evidence: list[dict[str, Any]]) -> bool:
    expected_validation = _planned_validation(request)
    selected_metric = (select_regression_metric(request, evidence).get("selected_metric")
                       if request.task_type == REGRESSION else request.metric or "f1")
    if (plan.get("task_type") != request.task_type or plan.get("target") != request.target
            or plan.get("metric") != selected_metric or plan.get("seed") != request.seed
            or plan.get("validation") != expected_validation):
        return False
    available = {e.get("chunk_id"): e for e in evidence}
    basis = plan.get("evidence_basis")
    if not isinstance(basis, list):
        return False
    valid_basis = all(item.get("chunk_id") in available
               and item.get("source_path") == available[item.get("chunk_id")].get("source_path")
               and item.get("section_path") == available[item.get("chunk_id")].get("section_path")
               for item in basis if isinstance(item, dict)) and all(isinstance(item, dict) for item in basis)
    recommendations = plan.get("recommendations", [])
    valid_recommendations = isinstance(recommendations, list) and all(
        isinstance(rec, dict) and isinstance(rec.get("evidence_chunk_ids"), list)
        and all(chunk_id in available for chunk_id in rec["evidence_chunk_ids"])
        for rec in recommendations)
    selection = plan.get("metric_selection", {})
    valid_metric_cites = isinstance(selection.get("evidence_chunk_ids"), list) and all(
        chunk_id in available for chunk_id in selection["evidence_chunk_ids"])
    valid_alternative_cites = all(isinstance(alt.get("evidence_chunk_ids"), list)
                                  and all(chunk_id in available for chunk_id in alt["evidence_chunk_ids"])
                                  for alt in selection.get("alternatives", []))
    return valid_basis and valid_recommendations and valid_metric_cites and valid_alternative_cites


def _md(result: ResearchResult) -> str:
    lines = ["# Informe de investigación VERONICA", "", f"**Estado:** {result.status}", "", result.summary]
    if result.dataset_audit:
        a = result.dataset_audit
        lines += ["", "## Auditoría", f"- Filas: {a['rows']}; columnas: {a['columns']}; duplicados: {a['duplicate_rows']}."]
        if a.get("target_statistics"):
            lines.append(f"- Target: `{a['target']}` (regresión continua).")
        else:
            lines.append(f"- Target: `{a['target']}`; distribución: `{json.dumps(a['target_distribution'], ensure_ascii=False)}`.")
        if a.get("target_statistics"):
            lines.append(f"- Estadísticas del target (calculadas desde el CSV): `{json.dumps(a['target_statistics'], ensure_ascii=False)}`.")
        lines += [f"- SHA-256 del CSV: `{a['dataset_sha256']}`."]
        lines += [f"- Señales: {', '.join(a['review_signals']) if a['review_signals'] else 'ninguna detectada por reglas básicas'}."]
        signals = a.get("column_review_signals", [])
        lines += ["", "## Columnas y señales para revisión humana",
                  "Estas señales ayudan a revisar el significado de las columnas. No confirman errores ni leakage y no modifican las variables usadas por el modelo."]
        if signals:
            lines += ["", "| Columna | Señal | Evidencia observada | Por qué revisarla | Pregunta para la persona |", "|---|---|---|---|---|"]
            for signal in signals:
                evidence_text = (json.dumps(signal["evidence"], ensure_ascii=False, sort_keys=True)
                                 + f"; alcance={json.dumps(signal['data_scope'], ensure_ascii=False, sort_keys=True)}; {signal['limitation']}").replace("|", "\\|")
                cells = [signal["column_name"], f"{signal['signal_type']} — {signal['rule_id']} ({signal['status']}): {signal['rule_description']}",
                         evidence_text, signal["why_review"], signal["review_question"]]
                lines.append("| " + " | ".join(" ".join(str(cell).split()).replace("|", "\\|") for cell in cells) + " |")
        else:
            lines.append("- No se generaron señales de columnas.")
        lines.append(f"- Checklist humana: `{a.get('human_review', {'status': 'pending'})}`")
    if result.plan:
        lines += ["", "## Plan", f"- Modelo: {result.plan['model']}; validación: {result.plan['validation']}.", f"- Métrica: {result.plan['metric']} — {result.plan['metric_reason']}", f"- Preprocesamiento: {result.plan['preprocessing']}.", f"- Semilla: {result.plan['seed']}."]
        selection = result.plan.get("metric_selection")
        if selection:
            lines += ["", "### Selección de métrica",
                      f"- Solicitada: {selection.get('requested_metric') or 'ninguna'}; seleccionada: {selection.get('selected_metric') or 'pendiente'}; origen: {selection.get('selection_source')}; objetivo: {selection.get('objective')}; unidad declarada: {selection.get('target_unit') or 'no declarada'}.",
                      f"- Justificación: {selection.get('rationale')}"]
            for tradeoff in selection.get("tradeoffs", []): lines.append(f"- Trade-off: {tradeoff}")
            for alternative in selection.get("alternatives", []):
                linked = ", ".join(f"`{cid}`" for cid in alternative.get("evidence_chunk_ids", [])) or "sin fragmento recuperado que respalde esta propiedad"
                lines.append(f"- Alternativa {alternative['metric'].upper()}: apropiada cuando {alternative['appropriate_when']} (evidencia: {linked}).")
            if selection.get("evidence_chunk_ids"):
                lines.append("- Fragmentos que respaldan propiedades: " + ", ".join(f"`{cid}`" for cid in selection["evidence_chunk_ids"]) + ".")
        if result.plan.get("evidence_limitation"): lines.append(f"- Evidencia documental: {result.plan['evidence_limitation']}")
        for rec in result.plan.get("recommendations", []):
            linked = ", ".join(f"`{chunk_id}`" for chunk_id in rec["evidence_chunk_ids"]) or "sin fragmento suficiente"
            lines.append(f"- Recomendación del agente: {rec['text']} (evidencia: {linked}).")
    if result.baseline:
        score = result.baseline.get("score")
        metric = result.baseline['metric']
        direction = "menor es mejor" if metric_direction(metric) == "lower_is_better" else "mayor es mejor"
        unit = result.plan.get("metric_selection", {}).get("target_unit") if result.plan else None
        lines += ["", "## Baseline", f"- Tarea: {result.baseline.get('task_type', TASK)}; modelo: {result.baseline['model']}; parámetros: `{json.dumps(result.baseline['parameters'], ensure_ascii=False)}`.", f"- Métrica primaria: {metric.upper()}; score: {score if score is None else f'{score:.6f}'} ({direction}; resultado medido en esta corrida)." + (f" Unidad del target declarada: {unit}." if unit else ""), f"- Split: {result.baseline['split_strategy']} ({result.baseline['train_rows']} train / {result.baseline['test_rows']} validación); semilla: {result.baseline['seed']}.", f"- Columnas predictoras: {', '.join(result.baseline['predictor_columns'])}."]
        if result.baseline.get("warnings"):
            lines.append("- Advertencias: " + " ".join(result.baseline["warnings"]))
        split = result.baseline.get("split_diagnostics", {})
        lines += ["", "### Diagnóstico del split", f"- Estrategia: {split.get('strategy')}; fracción solicitada: {split.get('requested_fraction')}; fracción real: {split.get('actual_fraction')}; validez: {split.get('valid')}; semilla: {split.get('seed')}."]
        if split.get("group_column"):
            lines.append(f"- Columna de grupo: `{split['group_column']}`; grupos train/validación: {split['train_groups']}/{split['validation_groups']}; intersección: `{split['group_intersection']}`.")
        if split.get("time_column"):
            lines.append(f"- Columna temporal: `{split['time_column']}`; período train: `{split['train_period']}`; período validación: `{split['validation_period']}`; corte: `{split['cut']}`; orden estricto: `{split['strict_chronology']}`.")
        for warning in split.get("warnings", []):
            lines.append(f"- Limitación del split: {warning}")
        validation = result.baseline.get("validation", {})
        if validation:
            lines += ["", "### Validación y reproducibilidad",
                      f"- Protocolo efectivo: `{validation.get('strategy')}`; estructura declarada: `{validation.get('observation_structure')}`; semilla: `{validation.get('seed')}`; ajustes: {validation.get('fit_count')}.",
                      f"- SHA-256 del CSV: `{validation.get('dataset_sha256')}`; algoritmo: `{validation.get('split_algorithm', 'split holdout existente')}`; barajado: `{validation.get('shuffle')}`."]
            if validation.get("effective_folds") is not None:
                aggregate = validation.get("aggregate")
                if aggregate:
                    lines.append(f"- Folds solicitados/efectivos: {validation.get('requested_folds')}/{validation.get('effective_folds')}; media no ponderada: {aggregate['mean']:.6f}; desviación estándar poblacional (ddof=0): {aggregate['std']:.6f}; mínimo/máximo: {aggregate['min']:.6f}/{aggregate['max']:.6f}.")
                else:
                    lines.append(f"- Folds solicitados/efectivos: {validation.get('requested_folds')}/{validation.get('effective_folds')}; sin resumen lineal porque no hubo scores válidos.")
                lines.append("- Scores por fold/partición (train/validación):")
                for fold in validation["folds"]:
                    score_text = f"{fold['score']:.6f}" if fold.get("score") is not None else "fallido"
                    lines.append(f"  - Fold {fold['fold']}: {score_text}; {fold['train_rows']}/{fold['validation_rows']} filas.")
                lines.append("- La media y desviación estándar describen estos folds; no son un intervalo de confianza ni una garantía de generalización.")
            else:
                holdout_score = validation.get("score")
                lines.append(f"- Score holdout: {holdout_score:.6f}" if holdout_score is not None else "- Score holdout: no disponible por fallo del candidato lineal")
                for fold in validation.get("folds", []):
                    score_text = f"{fold['score']:.6f}" if fold.get("score") is not None else "fallido"
                    lines.append(f"- Fold {fold['fold']}: {score_text}; {fold['train_rows']}/{fold['validation_rows']} filas.")
            lines.extend(f"- Decisión: {note}" for note in validation.get("notes", []))
    if not result.baseline and result.split_diagnostics:
        split = result.split_diagnostics
        lines += ["", "## Diagnóstico del split", f"- Estrategia solicitada: {split.get('strategy')}; fracción solicitada: {split.get('requested_fraction')}; fracción real: no disponible; validez: {split.get('valid')}; semilla: {split.get('seed')}."]
        if split.get("group_column"):
            lines.append(f"- Columna de grupo: `{split['group_column']}`; train/validación: no calculados; intersección de grupos: no verificada.")
        if split.get("time_column"):
            lines.append(f"- Columna temporal: `{split['time_column']}`; períodos y corte: no calculados; orden estricto: no verificado.")
        lines.extend(f"- Limitación del split: {warning}" for warning in split.get("warnings", []))
    if result.model_comparison:
        comparison = result.model_comparison
        direction = "menor es mejor" if comparison["metric_direction"] == "lower_is_better" else "mayor es mejor"
        lines += ["", "## Comparación de modelos", f"- Tarea: {comparison['task']}; target: `{comparison['target']}`; métrica principal: {str(comparison['metric']).upper()} ({direction}).",
                  f"- Protocolo: `{comparison['protocol']}`; particiones construidas una vez y compartidas por los candidatos. Preprocesamiento ajustado solo en train de cada partición.",
                  "", "| Candidato | Familia | Configuración | Scores por fold/partición | Resumen | Ajustes | Tiempo total (s) | Estado |",
                  "|---|---|---|---|---|---:|---:|---|"]
        for candidate in comparison["candidates"]:
            folds = "; ".join(f"{item['fold']}: {item['score']:.6f}" if item["score"] is not None else f"{item['fold']}: error" for item in candidate["folds"])
            summary = candidate["summary"]
            if summary and "mean" in summary:
                partial_label = "; parcial, folds fallidos excluidos" if summary.get("partial") else ""
                summary_text = f"media={summary['mean']:.6f}, desviación={summary['std']:.6f} (ddof=0){partial_label}"
            elif summary:
                partial_label = "; parcial" if summary.get("partial") else ""
                summary_text = f"score={summary['score']:.6f}{partial_label}"
            else:
                summary_text = "sin score"
            config = json.dumps(candidate["parameters"], ensure_ascii=False, sort_keys=True).replace("|", "\\|")
            lines.append(f"| {candidate['name']} (`{candidate['id']}`) | {candidate['family']} | `{config}` | {folds} | {summary_text} | {candidate['fit_count']} | {candidate['elapsed_seconds']:.6f} | {candidate['status']} |")
            for error in candidate["errors"]:
                lines.append(f"- Error {candidate['id']} fold {error['fold']} ({error['stage']}): {error['message']}")
        selected = comparison["selected_candidate_name"] or "ninguno: comparación incompleta o no comparable"
        lines += [f"- Estado de comparabilidad: `{comparison['status']}`; candidato seleccionado por validación: **{selected}**.",
                  f"- Regla: {comparison['selection_rule']}",
                  f"- {comparison['selection_scope']}. {comparison['warning']}",
                  f"- Evaluación final separada: disponible={comparison['final_evaluation']['available']}; {comparison['final_evaluation']['limitation']}",
                  "- Los tiempos suman ajuste, predicción y score por candidato en segundos, excluyen el preprocesamiento compartido y no son benchmarks comparables entre máquinas."]
        lines.append("- Particiones compartidas: " + "; ".join(f"{item['id']} ({item['train_rows']} train/{item['validation_rows']} validación)" for item in comparison["partitions"]) + ".")
    if result.risk_assessment:
        lines += ["", "## Riesgos clasificados"]
        for key, label in [("observed_facts", "Hechos observados"), ("review_signals", "Señales para revisión humana"), ("protocol_limitations", "Límites del protocolo"), ("unsupported_claims", "Afirmaciones no respaldadas")]:
            entries = result.risk_assessment.get(key, [])
            lines.append(f"### {label}")
            lines.extend([f"- {entry}" for entry in entries] if entries else ["- Ninguno."])
    if result.evidence:
        lines += ["", "Las estadísticas de auditoría proceden de herramientas; las afirmaciones documentales se limitan a los fragmentos citados; el score procede de esta corrida. Recomendaciones y supuestos del agente se identifican como tales."]
    if result.citations:
        lines += ["", "## Evidencia recuperada"]
        for c in result.citations:
            excerpt = " ".join(str(c.get("text", "")).split())[:240]
            if len(" ".join(str(c.get("text", "")).split())) > 240: excerpt += "…"
            section = " › ".join(str(c["section_path"]).splitlines())
            lines.append(f"- `{c['chunk_id']}` — `{c['source_path']}#{section}` (score ranking={c.get('score')}; no es confianza). {excerpt}")
    for title, values in [("Supuestos", result.assumptions), ("Riesgos y límites", result.risks), ("Siguientes acciones", result.next_actions), ("Errores", result.errors)]:
        if values: lines += ["", f"## {title}", *[f"- {v}" for v in values]]
    lines += ["", "El detalle estructurado completo está en el archivo JSON complementario.", ""]
    return "\n".join(lines)


def run_research(request: ResearchRequest, *, retriever: Retriever | None = None,
                 planner: Planner | None = None) -> ResearchResult:
    if request.metric is not None and request.metric.lower() != request.metric:
        request = replace(request, metric=request.metric.lower())
    issues = validate_request(request)
    if issues:
        status = "unsupported" if (request.task_type not in TASK_METRICS or request.split_strategy not in {"random", "group", "time"} or (request.metric and request.task_type in TASK_METRICS and request.metric not in TASK_METRICS[request.task_type]) or any("No se admite combinar" in issue or issue.startswith("unsupported:") for issue in issues)) else "needs_input"
        return ResearchResult(status, "La solicitud necesita corrección antes de ejecutar el baseline.", errors=issues,
                              split_diagnostics=_unverified_split_diagnostics(request, issues))
    try:
        audit, rows = audit_dataset(str(request.csv_path), str(request.target), str(request.task_type), request.time_format)
    except Exception as exc:
        error = str(exc)
        return ResearchResult("failed", "No se pudo auditar el dataset; no se entrenó ningún modelo.", errors=[error],
                              split_diagnostics=_unverified_split_diagnostics(request, [error]))
    retriever = retriever or rag_mvp.search
    planner = planner or DeterministicPlanner()
    evidence = _retrieve(retriever, request)
    plan = planner.plan(request, audit, evidence)
    try:
        if not validate_plan(plan, request, evidence):
            raise ValueError("El plan no coincide con la solicitud o cita fragmentos fuera del contexto recuperado.")
        selected_metric = plan["metric"]
        effective_request = replace(request, metric=selected_metric)
        labels = [row[str(request.target)].strip() for row in rows] if request.task_type == TASK else None
        train_indices, _, audit_split = _split_indices(effective_request, rows, labels)
        audit.column_review_signals.extend(_target_relation_signals(
            rows, audit.column_names, str(request.target), str(request.task_type),
            train_indices, audit_split["strategy"]))
        baseline = run_baseline(effective_request, audit, rows)
        if (not baseline["target_excluded"] or (baseline["score"] is not None and not math.isfinite(float(baseline["score"])))
                or baseline["metric"] != selected_metric or baseline["seed"] != request.seed
                or baseline.get("task_type", TASK) != request.task_type
                or baseline["model"] != ("numpy_ridge_regression" if request.task_type == REGRESSION else "numpy_logistic_regression_l2")):
            raise ValueError("La revisión de validez falló: separación target/predictores o score incompatible.")
        if baseline["train_rows"] == 0 or baseline["test_rows"] == 0 or (request.task_type == TASK and min(baseline["train_class_counts"].values()) == 0):
            raise ValueError("La revisión de validez falló: split vacío o sin ambas clases.")
        split_diag = baseline.get("split_diagnostics", {})
        if not split_diag.get("valid"):
            raise ValueError("La revisión de validez falló: el protocolo de split solicitado no cumplió sus garantías.")
        if request.split_strategy == "group" and split_diag.get("group_intersection"):
            raise ValueError("La revisión de validez falló: hay grupos compartidos entre train y validación.")
        if request.split_strategy == "time" and not split_diag.get("strict_chronology"):
            raise ValueError("La revisión de validez falló: no se cumple max(time(train)) < min(time(validation)).")
        citations = [{"chunk_id": e["chunk_id"], "source_path": e["source_path"],
                      "section_path": e["section_path"], "text": e["text"], "score": e.get("score"),
                      "lexical_score": e.get("lexical_score"), "semantic_score": e.get("semantic_score")} for e in evidence]
        if not validate_citations(citations, evidence):
            raise ValueError("La revisión de validez falló: cita no corresponde a un fragmento recuperado.")
    except Exception as exc:
        error = str(exc)
        diagnostics = baseline.get("split_diagnostics") if "baseline" in locals() and isinstance(baseline, dict) else None
        if diagnostics is None:
            diagnostics = _unverified_split_diagnostics(request, [error])
        audit_payload = asdict(audit)
        audit_payload.update({"target": request.target, "task_type": request.task_type,
                              "score_computed": False,
                              "predictor_columns_before_audit": [c for c in audit.column_names if c != request.target],
                              "predictor_columns_after_audit": [c for c in audit.column_names if c != request.target]})
        return ResearchResult("failed", "La ejecución no superó la revisión de validez.", audit_payload, plan,
                              errors=[error], evidence=evidence, split_diagnostics=diagnostics)
    target_stats = asdict(audit)
    target_stats["target"] = request.target
    target_stats["task_type"] = request.task_type
    target_stats["predictor_columns_before_audit"] = [c for c in audit.column_names if c != request.target]
    target_stats["predictor_columns_after_audit"] = list(baseline["predictor_columns"])
    if target_stats["predictor_columns_before_audit"] != target_stats["predictor_columns_after_audit"]:
        raise ValueError("La auditoría no debe modificar los predictores usados por el baseline.")
    risk_assessment = build_risk_assessment(request.task_type, audit, baseline)
    plan["risk_assessment"] = risk_assessment
    score_label = baseline["metric"]
    direction = ("menor es mejor" if metric_direction(score_label) == "lower_is_better" else "mayor es mejor") if request.task_type == REGRESSION else "mayor es mejor"
    selected_name = baseline["model_comparison"]["selected_candidate_name"]
    constant = next(candidate for candidate in baseline["model_comparison"]["candidates"] if candidate["id"] == "constant")
    if baseline["validation"].get("effective_folds") is not None:
        constant_score = constant["summary"].get("mean") if constant["summary"] and "mean" in constant["summary"] else (constant["summary"].get("score") if constant["summary"] else None)
        linear_text = f"{baseline['score']:.4f}" if baseline["score"] is not None else "no disponible"
        constant_text = f"{constant_score:.4f}" if constant_score is not None else "no disponible"
        summary = f"Se compararon dos candidatos para `{request.target}` con {baseline['validation']['strategy']} ({baseline['validation']['effective_folds']} folds). Mejor candidato en esta validación: {selected_name or 'no seleccionado'}; media lineal {score_label.upper()}={linear_text} y score constante={constant_text}; comparación preliminar, no garantía de generalización."
    else:
        linear_text = f"{baseline['score']:.4f}" if baseline["score"] is not None else "no disponible"
        task_label = "baseline binario" if request.task_type == TASK else "baseline de regresión"
        summary = f"Se compararon un candidato constante y uno lineal ({task_label}) para `{request.target}` con split {baseline['split_strategy']}. Candidato seleccionado por validación: {selected_name or 'ninguno por resultados incompletos'}; score lineal={linear_text} ({direction}). Resultado preliminar de esta partición, no garantía de generalización."
    risks = [item for key in ("review_signals", "protocol_limitations") for item in risk_assessment[key]]
    return ResearchResult("completed", summary, target_stats, plan, baseline, evidence, citations,
                          assumptions=(["El split aleatorio presupone observaciones independientes y representativas; el dataset no permite confirmar por sí solo esa condición."] if request.split_strategy == "random" else ["La garantía se limita al protocolo de partición solicitado; no demuestra ausencia total de leakage. Una feature puede contener información futura incluso con split cronológico."]),
                          risks=risks,
                          risk_assessment=risk_assessment,
                          next_actions=["Revisar el significado de cada predictor y cualquier señal heurística."],
                          split_diagnostics=baseline["split_diagnostics"],
                          model_comparison=baseline["model_comparison"])


def save_result(result: ResearchResult, request: ResearchRequest, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    destination = Path(output_dir) if output_dir else ARTIFACT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 3,
                "request": {"dataset_filename": Path(request.csv_path).name if request.csv_path else None,
                            "target": request.target, "task_type": request.task_type, "requested_metric": request.metric,
                            "metric_context": request.metric_context, "target_unit": request.target_unit,
                            "seed": request.seed, "context_provided": bool(request.context),
                            "split_strategy": request.split_strategy, "group_column": request.group_column,
                            "time_column": request.time_column, "test_size": request.test_size,
                            "time_format": request.time_format,
                            "validation_strategy": request.validation_strategy,
                            "observation_structure": request.observation_structure,
                            "cv_folds": request.cv_folds, "max_cv_fits": request.max_cv_fits},
                "result": result.to_dict()}
    json_path = destination / "research_result.json"
    md_path = destination / "research_report.md"
    markdown = _md(result)
    if not validate_report_consistency(result, metadata, markdown):
        raise ValueError("Markdown y JSON no concuerdan en los datos del resultado.")
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def validate_report_consistency(result: ResearchResult, payload: dict[str, Any],
                                markdown: str) -> bool:
    """Check task, selected metric, score, split, risks and citations across artifacts."""
    data = payload.get("result")
    if data != result.to_dict() or markdown != _md(result):
        return False
    if not result.baseline:
        return True
    validation = data["baseline"].get("validation", {})
    validation_text_ok = (validation.get("strategy") in markdown
                          and str(validation.get("fit_count")) in markdown
                          and all(f"Fold {fold['fold']}" in markdown for fold in validation.get("folds", [])))
    return (validation_text_ok
            and data.get("plan", {}).get("task_type") == data["baseline"].get("task_type", data.get("plan", {}).get("task_type"))
            and data["plan"].get("metric") == data["baseline"].get("metric")
            and data["plan"].get("seed") == data["baseline"].get("seed")
            and data["risk_assessment"] == result.risk_assessment
            and data.get("citations") == result.citations
            and all(warning in markdown for warning in data["baseline"].get("warnings", []))
            and (data["baseline"]["score"] is None or f"{data['baseline']['score']:.6f}" in markdown)
            and data.get("model_comparison") == data["baseline"].get("model_comparison")
            and (not data.get("model_comparison") or data["model_comparison"].get("status") in markdown)
            and str(data["baseline"]["seed"]) in markdown)


def run_audit_only(request: ResearchRequest) -> ResearchResult:
    """Run dataset checks and train-only heuristics without retrieval or model fitting."""
    validation_request = replace(request, metric=request.metric or ("mae" if request.task_type == REGRESSION else "f1"))
    issues = validate_request(validation_request)
    if issues:
        return ResearchResult("needs_input", "La auditoría necesita una solicitud válida.", errors=issues)
    try:
        audit, rows = audit_dataset(str(request.csv_path), str(request.target), str(request.task_type), request.time_format)
        labels = [row[str(request.target)].strip() for row in rows] if request.task_type == TASK else None
        train_indices, _, split = _split_indices(validation_request, rows, labels)
        audit.column_review_signals.extend(_target_relation_signals(
            rows, audit.column_names, str(request.target), str(request.task_type),
            train_indices, split["strategy"]))
    except Exception as exc:
        return ResearchResult("failed", "No se pudo completar la auditoría; no se entrenó ningún modelo.",
                              errors=[str(exc)])
    payload = asdict(audit)
    payload.update({"target": request.target, "task_type": request.task_type,
                    "score_computed": False,
                    "predictor_columns_before_audit": [c for c in audit.column_names if c != request.target],
                    "predictor_columns_after_audit": [c for c in audit.column_names if c != request.target]})
    return ResearchResult("completed", "Auditoría completada. No se calculó ningún score ni se entrenó un modelo.",
                          dataset_audit=payload, split_diagnostics=split)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Agente local VERONICA para clasificación binaria y regresión tabular")
    parser.add_argument("--csv", dest="csv_path")
    parser.add_argument("--target")
    parser.add_argument("--task", dest="task_type", default=TASK)
    parser.add_argument("--metric", help="Métrica explícita: clasificación (f1, accuracy, precision, recall...) o regresión (mae, rmse, r2). Si se omite en regresión, use --metric-context.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-strategy", choices=("holdout", "cross_validation", "auto"), default="holdout",
                        help="Protocolo; holdout conserva el comportamiento actual, cross_validation requiere independencia declarada y auto usa reglas deterministas.")
    parser.add_argument("--observation-structure", choices=("independent", "grouped", "temporal", "unknown"), default="unknown",
                        help="Estructura declarada; no se infiere del CSV.")
    parser.add_argument("--cv-folds", type=int, default=None, help="Folds solicitados (predeterminado 5 al usar CV).")
    parser.add_argument("--max-cv-fits", type=int, default=None, help="Máximo de ajustes CV (predeterminado 5).")
    parser.add_argument("--split-strategy", choices=("random", "group", "time"), default="random")
    parser.add_argument("--group-column", help="Columna de entidad; requiere --split-strategy group.")
    parser.add_argument("--time-column", help="Columna de tiempo; requiere --split-strategy time.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracción solicitada para validación, estrictamente entre 0 y 1.")
    parser.add_argument("--time-format", help="Formato datetime.strptime explícito para fechas no ISO inequívocas; solo con split time.")
    parser.add_argument("--context")
    parser.add_argument("--metric-context", choices=sorted(REGRESSION_OBJECTIVES), default=None,
                        help="Objetivo estructurado de métrica para regresión; el texto libre --context no selecciona una métrica.")
    parser.add_argument("--target-unit", help="Unidad declarada del target, solo informativa.")
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-only", action="store_true",
                        help="Valida y escribe la auditoría sin retrieval, entrenamiento ni score.")
    args = parser.parse_args(argv)
    request = ResearchRequest(args.csv_path, args.target, args.task_type, args.metric, args.seed, args.context,
                              args.metric_context, args.target_unit, args.split_strategy,
                              args.group_column, args.time_column, args.test_size, args.time_format,
                              args.validation_strategy, args.observation_structure,
                              args.cv_folds, args.max_cv_fits)
    result = run_audit_only(request) if args.audit_only else run_research(request)
    try:
        json_path, md_path = save_result(result, request, args.output_dir)
        print(_md(result))
        print(f"Artefactos: {json_path} | {md_path}")
    except OSError as exc:
        print(f"No se pudieron guardar artefactos: {exc}", file=sys.stderr)
        return 1
    return 0 if result.status == "completed" else 2 if result.status in {"needs_input", "unsupported"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
