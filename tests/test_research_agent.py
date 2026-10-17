"""Pruebas unitarias y de integración sintéticas para el agente local."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import research_agent as agent
from decision_fixtures import bind_decisions


def explicit_request(*args, **kwargs):
    """Explicit synthetic-domain declarations for existing training fixtures."""
    positive_class = kwargs.pop('positive_class', 'yes')
    kwargs.setdefault('observation_structure', 'independent')
    request = agent.ResearchRequest(*args, **kwargs)
    columns = []
    if request.csv_path and Path(request.csv_path).is_file():
        with Path(request.csv_path).open(encoding='utf-8') as stream:
            columns = next(csv.reader(stream), [])
    request.decisions = {
        'schema_version': 1,
        'declared_fields': ['task_type', 'target', 'csv_path', 'metric', 'split_strategy', 'observation_structure'],
        'positive_class': positive_class if request.task_type == agent.TASK else None,
        'row_meaning': 'Una observación sintética de prueba',
        'prediction_scenario': 'Antes de medir el resultado sintético',
        'metric_reason': 'Métrica explícita del escenario de prueba',
        'predictor_availability': {c: 'available' for c in columns if c != request.target},
        'responses': [{'column_name': c, 'decision': 'keep', 'reason': 'Entrada del generador sintético conocida antes del resultado', 'column_meaning': f'Variable sintética {c} de este caso de prueba'} for c in columns if c != request.target],
    }
    return bind_decisions(request)


def make_csv(path: Path, n: int = 60) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["signal", "group", "target"])
        for i in range(n):
            label = "yes" if i % 2 == 0 else "no"
            writer.writerow([i + (10 if label == "yes" else -10), "A" if i % 3 else "B", label])
    return path


def make_regression_csv(path: Path, n: int = 60) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["signal", "category", "target"])
        for i in range(n):
            writer.writerow([i, "even" if i % 2 == 0 else "odd", 3.5 * i + (i % 4) * 0.1])
    return path


class RequestAndAuditTests(unittest.TestCase):
    def test_request_validation_and_statuses(self):
        request = explicit_request(None, None)
        self.assertGreaterEqual(len(agent.validate_request(request)), 2)
        bad_task = explicit_request("x.csv", "y", "multiclass", "accuracy")
        self.assertEqual(agent.run_research(bad_task).status, "unsupported")
        self.assertEqual(agent.run_research(explicit_request("x.csv", "y", metric="mae")).status, "unsupported")
        self.assertEqual(agent.run_research(explicit_request(None, "y")).status, "needs_input")

    def test_regression_request_and_metric_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "data.csv")
            request = explicit_request(str(path), "target", "regression", "mae")
            self.assertEqual(agent.validate_request(request), [])
            bad = explicit_request(str(path), "target", "regression", "f1")
            self.assertTrue(any("incompatible" in issue for issue in agent.validate_request(bad)))
            self.assertEqual(agent.run_research(bad).status, "unsupported")
            explicit_time = explicit_request(str(path), "target", "regression", "rmse", context="time_column=date")
            result = agent.run_research(explicit_time)
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.baseline["split_strategy"], "random_holdout")

    def test_audit_profile_types_missing_duplicate_and_signals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text("number,category,target\n1,A,yes\n2,,no\n2,,no\n", encoding="utf-8")
            audit, parsed = agent.audit_dataset(path, "target")
            self.assertEqual((audit.rows, audit.columns, audit.duplicate_rows), (3, 3, 1))
            self.assertEqual(audit.missing_values["category"], 2)
            self.assertEqual(audit.column_types["number"], "numeric")
            self.assertEqual(audit.categorical_cardinality["category"], 1)
            self.assertEqual(audit.target_distribution, {"yes": 1, "no": 2})

    def test_missing_empty_and_constant_target_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text("x,y\n1,a\n2,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No existe la columna objetivo"):
                agent.audit_dataset(path, "target")
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "vacío"):
                agent.audit_dataset(path, "target")
            path.write_text("x,target\n1,a\n2,a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "constante"):
                agent.audit_dataset(path, "target")

    def test_regression_audit_requires_numeric_target_and_profiles_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "regression.csv")
            audit, _ = agent.audit_dataset(path, "target", "regression")
            self.assertAlmostEqual(audit.target_statistics["mean"], 103.4, places=5)
            self.assertEqual(audit.target_statistics["min"], 0.0)
            self.assertEqual(audit.target_statistics["max"], 206.8)
            self.assertEqual(audit.column_types["target"], "numeric")
            path.write_text("x,target\n1,nope\n2,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "numéricos finitos"):
                agent.audit_dataset(path, "target", "regression")

    def test_csv_empty_missing_target_and_invalid_target(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            path.write_text("", encoding="utf-8")
            self.assertEqual(agent.run_research(explicit_request(str(path), "y", "regression", "mae")).status, "failed")
            path.write_text("x,value\n1,2\n", encoding="utf-8")
            missing = agent.run_research(explicit_request(str(path), "y", "regression", "mae"))
            self.assertEqual(missing.status, "failed")
            self.assertIn("No existe la columna objetivo", missing.errors[0])
            path.write_text("x,y\n1,bad\n2,3\n", encoding="utf-8")
            invalid = agent.run_research(explicit_request(str(path), "y", "regression", "mae"))
            self.assertEqual(invalid.status, "failed")


class BaselineTests(unittest.TestCase):
    def test_target_is_excluded_runner_metric_is_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "synthetic.csv")
            audit, rows = agent.audit_dataset(path, "target")
            request = explicit_request(str(path), "target", metric="accuracy")
            result = agent.run_baseline(request, audit, rows)
            self.assertNotIn("target", result["predictor_columns"])
            self.assertTrue(result["target_excluded"])
            self.assertTrue(0 <= result["score"] <= 1)
            self.assertEqual(result["metric"], "accuracy")
            self.assertGreater(result["train_rows"], result["test_rows"])

    def test_metric_and_stratified_split(self):
        self.assertEqual(agent._metric_value("accuracy", [0, 1], [0, 1]), 1.0)
        self.assertEqual(agent._metric_value("f1", [0, 1], [0, 1]), 1.0)
        with self.assertRaisesRegex(ValueError, "al menos 2 filas"):
            agent._stratified_split(["a", "a", "b"], 4)

    def test_citations_must_match_retrieved_chunk_location(self):
        evidence = [{"chunk_id": "c1", "source_path": "knowledge/a.md", "section_path": "A", "text": "evidence"}]
        citation = {"chunk_id": "c1", "source_path": "knowledge/a.md", "section_path": "A", "text": "evidence"}
        self.assertTrue(agent.validate_citations([citation], evidence))
        self.assertFalse(agent.validate_citations([{**citation, "text": "forged text"}], evidence))
        self.assertFalse(agent.validate_citations([{**citation, "source_path": "knowledge/other.md"}], evidence))
        self.assertFalse(agent.validate_citations([{**citation, "chunk_id": "absent"}], evidence))

    def test_regression_metrics_and_undefined_r2(self):
        self.assertAlmostEqual(agent._regression_metric_value("mae", [1, 3], [2, 1]), 1.5)
        self.assertAlmostEqual(agent._regression_metric_value("rmse", [1, 3], [2, 1]), (2.5 ** 0.5))
        self.assertAlmostEqual(agent._regression_metric_value("r2", [1, 3], [1, 3]), 1.0)
        self.assertLess(agent._regression_metric_value("r2", [0, 1], [10, 10]), 0.0)
        self.assertIsNone(agent._regression_metric_value("r2", [4, 4], [3, 5]))

    def test_regression_baseline_excludes_target_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "data.csv")
            audit, rows = agent.audit_dataset(path, "target", "regression")
            request = explicit_request(str(path), "target", "regression", "mae", 123)
            first = agent.run_baseline(request, audit, rows)
            second = agent.run_baseline(request, audit, rows)
            self.assertNotIn("target", first["predictor_columns"])
            self.assertTrue(first["target_excluded"])
            self.assertEqual(first["split_strategy"], "random_holdout")
            self.assertEqual(first["score"], second["score"])
            self.assertEqual(first["train_rows"] + first["test_rows"], 60)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = make_csv(Path(self.temporary.name) / "synthetic.csv")
        self.request = explicit_request(str(self.path), "target", seed=123, metric="f1")

    def test_complete_flow_saves_json_markdown_and_repeats(self):
        evidence = {"chunk_id": "local-1", "source_path": "knowledge/metrics/classification_metrics.md",
                    "section_path": "F1", "text": "F1 combina precision y recall.", "score": 0.8,
                    "lexical_score": 0.7, "semantic_score": 0.9}
        validation = {"chunk_id": "local-2", "source_path": "knowledge/validation/validation_strategy.md",
                      "section_path": "Holdout", "text": "Estratificar conserva las clases.", "score": 0.7,
                      "lexical_score": 0.6, "semantic_score": 0.8}
        result1 = agent.run_research(self.request, retriever=lambda *a, **k: [evidence, validation])
        result2 = agent.run_research(self.request, retriever=lambda *a, **k: [evidence, validation])
        self.assertEqual(result1.status, "completed")
        self.assertEqual(result1.baseline["score"], result2.baseline["score"])
        self.assertEqual(result1.baseline["test_class_counts"], result2.baseline["test_class_counts"])
        self.assertEqual(result1.citations[0]["chunk_id"], "local-1")
        self.assertTrue(result1.plan["evidence_sufficient"])
        self.assertEqual(result1.dataset_audit["predictor_columns_before_audit"], result1.dataset_audit["predictor_columns_after_audit"])
        self.assertEqual(result1.dataset_audit["predictor_columns_after_audit"], result1.baseline["predictor_columns"])
        with tempfile.TemporaryDirectory() as output:
            jp, mp = agent.save_result(result1, self.request, output)
            payload = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(payload["result"]["status"], "completed")
            self.assertNotIn(str(self.path), jp.read_text(encoding="utf-8"))
            self.assertIn("# Informe", mp.read_text(encoding="utf-8"))

    def test_no_evidence_abstains_from_documentary_claims(self):
        result = agent.run_research(self.request, retriever=lambda *a, **k: [])
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.citations)
        self.assertFalse(result.plan["evidence_sufficient"])
        self.assertIn("no aportó evidencia", result.plan["evidence_limitation"])

    def test_partial_retrieval_does_not_support_documentary_conclusions(self):
        metric_only = {"chunk_id": "metric", "source_path": "knowledge/metrics/classification_metrics.md",
                       "section_path": "F1", "text": "F1", "score": 0.8}
        result = agent.run_research(self.request, retriever=lambda *a, **k: [metric_only])
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.plan["evidence_sufficient"])
        self.assertIn("validación", result.plan["evidence_limitation"])

    def test_failed_execution_for_class_with_single_example(self):
        path = Path(self.temporary.name) / "rare.csv"
        path.write_text("x,target\n1,a\n2,a\n3,a\n4,b\n", encoding="utf-8")
        result = agent.run_research(explicit_request(str(path), "target", metric="f1", positive_class="a"), retriever=lambda *a, **k: [])
        self.assertEqual(result.status, "failed")
        self.assertIn("al menos 2 filas", result.errors[0])

    def test_regression_complete_flow_retrieval_artifacts_and_repeatability(self):
        path = make_regression_csv(Path(self.temporary.name) / "continuous.csv")
        request = explicit_request(str(path), "target", "regression", "rmse", 77)
        metric = {"chunk_id": "reg-metric", "source_path": "knowledge/metrics/regression_metrics.md", "section_path": "MAE", "text": "MAE resume error absoluto.", "score": 0.6}
        validation = {"chunk_id": "validation", "source_path": "knowledge/validation/validation_strategy.md", "section_path": "Holdout", "text": "Separar validación.", "score": 0.7}
        fake = lambda *a, **k: [metric, validation]
        first = agent.run_research(request, retriever=fake)
        second = agent.run_research(request, retriever=fake)
        self.assertEqual(first.status, "completed")
        self.assertEqual(first.plan["task_type"], "regression")
        self.assertEqual(first.plan["validation"], "holdout aleatorio con test_size solicitado, semilla fija")
        self.assertEqual(first.baseline["score"], second.baseline["score"])
        self.assertEqual(first.dataset_audit["target_statistics"]["max"], 206.8)
        self.assertTrue(agent.validate_citations(first.citations, first.evidence))
        self.assertTrue(agent.validate_plan(first.plan, request, first.evidence))
        with tempfile.TemporaryDirectory() as output:
            jp, mp = agent.save_result(first, request, output)
            payload = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(payload["result"]["baseline"]["metric"], "rmse")
            self.assertNotIn(str(path), jp.read_text(encoding="utf-8"))
            markdown = mp.read_text(encoding="utf-8")
            self.assertIn("Estadísticas del target", markdown)
            self.assertIn("resultado medido", markdown)

    def test_r2_undefined_returns_explanation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "constant_validation.csv"
            # Dataset where seeded holdout contains a constant-valued subset.
            rows = ["x,target"] + [f"{i},1" for i in range(9)] + ["9,2"]
            path.write_text("\n".join(rows), encoding="utf-8")
            result = agent.run_research(explicit_request(str(path), "target", "regression", "r2", 42), retriever=lambda *a, **k: [])
            self.assertIn(result.status, {"failed", "completed"})
            if result.status == "failed":
                self.assertIn("R² no está definido", result.errors[0])

    def test_classification_regression_guard(self):
        result = agent.run_research(self.request, retriever=lambda *a, **k: [])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.baseline["split_strategy"], "stratified_holdout")
        self.assertEqual(result.baseline["metric"], "f1")
        self.assertIn("baseline binario", result.summary)


class RegressionMetricSelectionTests(unittest.TestCase):
    def test_metric_directions(self):
        self.assertEqual(agent.metric_direction("mae"), "lower_is_better")
        self.assertEqual(agent.metric_direction("rmse"), "lower_is_better")
        self.assertEqual(agent.metric_direction("r2"), "higher_is_better")

    def test_structured_objectives_select_expected_metric(self):
        expected = {"absolute_error_cost": "mae", "large_errors_cost_more": "rmse",
                    "variance_explanation": "r2"}
        for objective, metric in expected.items():
            with self.subTest(objective=objective):
                selection = agent.select_regression_metric(explicit_request(
                    "data.csv", "target", "regression", metric_context=objective))
                self.assertEqual(selection["selected_metric"], metric)
                self.assertEqual(selection["selection_source"], "structured_context")
        r2 = agent.select_regression_metric(explicit_request(
            "data.csv", "target", "regression", metric_context="variance_explanation"))
        self.assertIn("negativo", r2["rationale"])
        self.assertIn("magnitud", r2["rationale"])

    def test_explicit_metric_wins_and_conflict_is_visible(self):
        selection = agent.select_regression_metric(explicit_request(
            "data.csv", "target", "regression", "mae", metric_context="large_errors_cost_more"))
        self.assertEqual(selection["selected_metric"], "mae")
        self.assertEqual(selection["selection_source"], "user_explicit")
        self.assertTrue(any("se conserva MAE" in item for item in selection["tradeoffs"]))

    def test_missing_objective_needs_input_and_free_context_does_not_select(self):
        request = explicit_request("data.csv", "target", "regression", context="Me preocupan mucho los errores grandes")
        self.assertIsNone(agent.select_regression_metric(request)["selected_metric"])
        self.assertTrue(any("needs_input" in issue for issue in agent.validate_request(request)))

    def test_official_metric_and_relative_objective_report_limitations(self):
        official = explicit_request("data.csv", "target", "regression", metric_context="official_metric")
        relative = explicit_request("data.csv", "target", "regression", metric_context="relative_error")
        self.assertIn("métrica oficial", agent.select_regression_metric(official)["rationale"])
        self.assertIn("no hay una métrica relativa", agent.select_regression_metric(relative)["rationale"])
        unsupported = explicit_request("data.csv", "target", "regression", "mape", metric_context="official_metric")
        self.assertIn("no está soportada", agent.select_regression_metric(unsupported)["rationale"])

    def test_metric_property_citations_are_retrieved_evidence_only(self):
        metric = {"chunk_id": "rmse-property", "source_path": "knowledge/metrics/regression_metrics.md",
                  "text": "RMSE da mayor peso relativo a errores grandes y menor es mejor."}
        unrelated = {"chunk_id": "other", "source_path": "knowledge/metrics/regression_metrics.md",
                     "text": "Métrica de regresión."}
        selection = agent.select_regression_metric(explicit_request(
            "data.csv", "target", "regression", metric_context="large_errors_cost_more"), [metric, unrelated])
        self.assertEqual(selection["evidence_chunk_ids"], ["rmse-property"])
        self.assertNotIn("other", selection["evidence_chunk_ids"])

    def test_regression_risks_exclude_class_imbalance_classification_keeps_distribution_alert(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "reg.csv")
            audit, _ = agent.audit_dataset(path, "target", "regression")
            regression_risks = agent.build_risk_assessment("regression", audit, None)
            self.assertFalse(any("clase" in value.casefold() or "desequilibrio" in value.casefold()
                                 for values in regression_risks.values() for value in values))
            classification = Path(folder) / "class.csv"
            classification.write_text("x,y\n" + "\n".join(f"{i},{'a' if i < 54 else 'b'}" for i in range(60)), encoding="utf-8")
            class_audit, _ = agent.audit_dataset(classification, "y")
            class_risks = agent.build_risk_assessment(agent.TASK, class_audit, None)
            self.assertIn("Distribución calculada de clases:", class_risks["observed_facts"][1])
            self.assertTrue(any("proporción observada" in value for value in class_risks["review_signals"]))

    def test_unique_numeric_column_is_a_review_signal_not_confirmed_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "unique.csv")
            audit, _ = agent.audit_dataset(path, "target", "regression")
            signal = next(value for value in audit.review_signals if value.startswith("signal:"))
            self.assertIn("merece revisión", signal)
            self.assertIn("no confirma", signal)
            self.assertNotIn("es identificador", signal)

    def test_integrated_regression_artifacts_share_metrics_score_risks_and_citations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "synthetic.csv")
            evidence = [
                {"chunk_id": "mae", "source_path": "knowledge/metrics/regression_metrics.md", "section_path": "4.1 MAE",
                 "text": "MAE resume la magnitud absoluta media, queda en las unidades del objetivo y menor es mejor."},
                {"chunk_id": "rmse", "source_path": "knowledge/metrics/regression_metrics.md", "section_path": "4.3 RMSE",
                 "text": "RMSE da mayor peso relativo a errores grandes y menor es mejor."},
                {"chunk_id": "r2", "source_path": "knowledge/metrics/regression_metrics.md", "section_path": "4.4 R2",
                 "text": "R2 es adimensional, puede ser negativo y no informa por sí solo la magnitud del error."},
                {"chunk_id": "validation", "source_path": "knowledge/validation/validation_strategy.md", "section_path": "Holdout",
                 "text": "Un holdout mide el resultado de una partición."},
            ]
            request = explicit_request(str(path), "target", "regression", metric="rmse", metric_context="large_errors_cost_more")
            result = agent.run_research(request, retriever=lambda *a, **k: evidence)
            self.assertEqual(result.status, "completed")
            self.assertNotIn("desequilibrio de clases", " ".join(result.risks).casefold())
            self.assertFalse(any("clase" in value.casefold() or "desequilibrio" in value.casefold()
                                 for value in result.plan["risks_to_review"]))
            available_ids = {item["chunk_id"] for item in result.evidence}
            self.assertTrue(set(result.plan["metric_selection"]["evidence_chunk_ids"]).issubset(available_ids))
            for alternative in result.plan["metric_selection"]["alternatives"]:
                self.assertTrue(set(alternative["evidence_chunk_ids"]).issubset(available_ids))
            with tempfile.TemporaryDirectory() as output:
                json_path, md_path = agent.save_result(result, request, output)
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                markdown = md_path.read_text(encoding="utf-8")
                self.assertTrue(agent.validate_report_consistency(result, payload, markdown))
                self.assertEqual(payload["result"]["plan"]["metric_selection"]["selected_metric"], "rmse")
                self.assertEqual(payload["result"]["baseline"]["score"], result.baseline["score"])
                self.assertEqual(payload["result"]["risk_assessment"], result.risk_assessment)
                self.assertIn("garantía de generalización", markdown)
                self.assertIn("rmse", payload["result"]["plan"]["metric_selection"]["evidence_chunk_ids"])


class ModelComparisonTests(unittest.TestCase):
    def test_binary_and_regression_include_constant_and_linear_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            binary_path = make_csv(Path(folder) / "binary.csv")
            binary = agent.run_research(explicit_request(str(binary_path), "target", metric="f1"),
                                        retriever=lambda *a, **k: [])
            self.assertEqual(binary.status, "completed", binary.errors)
            candidates = binary.model_comparison["candidates"]
            self.assertEqual([candidate["id"] for candidate in candidates], ["constant", "linear"])
            self.assertTrue(all(candidate["summary"] and candidate["fit_count"] == 1 for candidate in candidates))
            regression_path = make_regression_csv(Path(folder) / "regression.csv")
            regression = agent.run_research(explicit_request(str(regression_path), "target", "regression", "mae"),
                                            retriever=lambda *a, **k: [])
            self.assertEqual(regression.status, "completed", regression.errors)
            self.assertEqual([candidate["family"] for candidate in regression.model_comparison["candidates"]],
                             ["constant", "ridge_regression"])

    def test_cv_partitions_are_constructed_once_shared_and_scores_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "data.csv")
            audit, rows = agent.audit_dataset(path, "target")
            request = explicit_request(str(path), "target", metric="f1", seed=17,
                validation_strategy="cross_validation", observation_structure="independent")
            original = agent._kfold_indices
            with mock.patch.object(agent, "_kfold_indices", wraps=original) as splitter:
                first = agent.run_model_comparison(request, audit, rows)
                self.assertEqual(splitter.call_count, 1)
            second = agent.run_model_comparison(request, audit, rows)
            self.assertEqual([candidate["folds"] for candidate in first["comparison"]["candidates"]],
                             [candidate["folds"] for candidate in second["comparison"]["candidates"]])
            identifiers = [[fold["partition_id"] for fold in candidate["folds"]]
                            for candidate in first["comparison"]["candidates"]]
            self.assertEqual(identifiers[0], identifiers[1])
            self.assertEqual(len(first["comparison"]["partitions"]), 5)

    def test_holdout_is_shared_and_preprocessing_receives_train_indices_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "data.csv")
            audit, rows = agent.audit_dataset(path, "target", "regression")
            request = explicit_request(str(path), "target", "regression", "mae", seed=8)
            original = agent._encode
            observed = []
            def recording_encode(dataset, features, types, train, indices):
                observed.append((list(train), list(indices)))
                return original(dataset, features, types, train, indices)
            with mock.patch.object(agent, "_encode", side_effect=recording_encode):
                compared = agent.run_model_comparison(request, audit, rows)["comparison"]
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[0][0], observed[0][1])
            self.assertEqual(observed[1][0], observed[0][0])
            self.assertNotEqual(observed[1][1], observed[1][0])
            self.assertEqual(compared["candidates"][0]["folds"][0]["partition_id"],
                             compared["candidates"][1]["folds"][0]["partition_id"])
            self.assertEqual(compared["status"], "comparable")

    def test_metric_direction_ties_and_failed_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "imbalanced.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(["x", "target"])
                writer.writerows([[i, "no" if i < 18 else "yes"] for i in range(20)])
            audit, rows = agent.audit_dataset(path, "target")
            request = explicit_request(str(path), "target", metric="recall")
            original = agent._candidate_prediction
            def prediction(candidate_id, task_type, x_train, x_valid, y_train, classes):
                return ([0] * len(x_valid)) if candidate_id == "constant" else ([1] * len(x_valid))
            with mock.patch.object(agent, "_candidate_prediction", side_effect=prediction):
                recall = agent.run_model_comparison(request, audit, rows)["comparison"]
            self.assertEqual(recall["selected_candidate_id"], "linear")
            request.metric = "accuracy"
            bind_decisions(request)
            with mock.patch.object(agent, "_candidate_prediction", side_effect=prediction):
                accuracy = agent.run_model_comparison(request, audit, rows)["comparison"]
            self.assertEqual(accuracy["selected_candidate_id"], "constant")
            def tied(candidate_id, task_type, x_train, x_valid, y_train, classes):
                return [0] * len(x_valid)
            with mock.patch.object(agent, "_candidate_prediction", side_effect=tied):
                tie = agent.run_model_comparison(request, audit, rows)["comparison"]
            self.assertEqual(tie["selected_candidate_id"], "constant")
            request = explicit_request(str(path), "target", metric="f1", validation_strategy="cross_validation",
                observation_structure="independent", cv_folds=2, max_cv_fits=2)
            linear_calls = 0
            def fail_one(candidate_id, task_type, x_train, x_valid, y_train, classes):
                nonlocal linear_calls
                if candidate_id == "linear":
                    linear_calls += 1
                    if linear_calls == 1:
                        raise RuntimeError("fallo sintético")
                return original(candidate_id, task_type, x_train, x_valid, y_train, classes)
            with mock.patch.object(agent, "_candidate_prediction", side_effect=fail_one):
                incomplete = agent.run_model_comparison(request, audit, rows)["comparison"]
            self.assertEqual(incomplete["status"], "incomplete_or_not_comparable")
            self.assertIsNone(incomplete["selected_candidate_id"])
            linear = next(candidate for candidate in incomplete["candidates"] if candidate["id"] == "linear")
            self.assertTrue(linear["errors"])
            self.assertEqual(len(linear["folds"]), 2)

    def test_json_markdown_consistency_and_validation_limitations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "data.csv")
            request = explicit_request(str(path), "target", metric="f1")
            result = agent.run_research(request, retriever=lambda *a, **k: [])
            with tempfile.TemporaryDirectory() as output:
                json_path, md_path = agent.save_result(result, request, output)
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                markdown = md_path.read_text(encoding="utf-8")
            comparison = payload["result"]["model_comparison"]
            self.assertEqual(comparison, payload["result"]["baseline"]["model_comparison"])
            self.assertTrue(agent.validate_report_consistency(result, payload, markdown))
            self.assertIn("Comparación de modelos", markdown)
            self.assertIn("sobreajustar la validación", markdown)
            self.assertFalse(comparison["final_evaluation"]["available"])
            self.assertFalse(comparison["final_evaluation"]["used_for_selection"])

    def test_total_linear_failure_is_retained_in_artifacts_without_winner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "data.csv")
            request = explicit_request(str(path), "target", metric="f1")
            original = agent._candidate_prediction
            def fail_linear(candidate_id, task_type, x_train, x_valid, y_train, classes):
                if candidate_id == "linear":
                    raise RuntimeError("fallo reproducible")
                return original(candidate_id, task_type, x_train, x_valid, y_train, classes)
            with mock.patch.object(agent, "_candidate_prediction", side_effect=fail_linear):
                result = agent.run_research(request, retriever=lambda *a, **k: [])
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.model_comparison["status"], "incomplete_or_not_comparable")
            self.assertIsNone(result.model_comparison["selected_candidate_id"])
            linear = next(candidate for candidate in result.model_comparison["candidates"] if candidate["id"] == "linear")
            self.assertEqual(linear["errors"][0]["message"], "fallo reproducible")
            with tempfile.TemporaryDirectory() as output:
                _, md_path = agent.save_result(result, request, output)
                markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("fallo reproducible", markdown)
            self.assertIn("ninguno: comparación incompleta o no comparable", markdown)


class ExplicitSplitTests(unittest.TestCase):
    def _write(self, path: Path, header: list[str], rows: list[list[object]]) -> Path:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
        return path

    def _group_rows(self) -> list[list[object]]:
        rows = []
        for group_no, size in enumerate([2, 3, 4, 5, 6, 7, 8, 9]):
            for j in range(size):
                rows.append([group_no * 100 + j, f"g{group_no}", "yes" if j % 2 == 0 else "no"])
        return rows

    def test_group_split_is_seeded_whole_groups_and_approximates_rows(self):
        rows = self._group_rows()
        source = [{"g": str(row[1])} for row in rows]
        a = agent._group_split(source, "g", 42, .22); b = agent._group_split(source, "g", 42, .22)
        self.assertEqual(a, b)
        train, valid, diag = a
        train_groups = {rows[i][1] for i in train}; valid_groups = {rows[i][1] for i in valid}
        self.assertFalse(train_groups & valid_groups)
        self.assertAlmostEqual(diag["actual_fraction"], len(valid) / len(rows))
        self.assertLess(abs(len(valid) / len(rows) - .22), .2)
        self.assertTrue(diag["valid"])

    def test_group_request_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(Path(folder) / "g.csv", ["x", "entity", "target"], self._group_rows())
            omitted = explicit_request(str(path), "target", "binary_classification", "f1", split_strategy="group")
            self.assertEqual(agent.run_research(omitted).status, "needs_input")
            missing = explicit_request(str(path), "target", "binary_classification", "f1", split_strategy="group", group_column="absent")
            result = agent.run_research(missing, retriever=lambda *a, **k: [])
            self.assertEqual(result.status, "failed"); self.assertIn("no existe", result.errors[0])
            with self.assertRaisesRegex(ValueError, "vacíos o nulos"):
                agent._group_split([{"g": "a"}, {"g": ""}, {"g": "b"}], "g", 1, .2)
            for source in ([{"g": "a"}] * 5, [{"g": "same"}] * 4):
                with self.assertRaisesRegex(ValueError, "al menos dos grupos"):
                    agent._group_split(source, "g", 1, .25)

    def test_group_classification_cannot_remove_training_class(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[i, "positive", "yes"] for i in range(10)] + [[i + 20, "negative", "no"] for i in range(10)]
            path = self._write(Path(folder) / "classes.csv", ["x", "entity", "target"], rows)
            req = explicit_request(str(path), "target", "binary_classification", "f1", 2,
                split_strategy="group", group_column="entity", test_size=.5)
            result = agent.run_research(req, retriever=lambda *a, **k: [])
            self.assertEqual(result.status, "failed"); self.assertIn("entrenamiento sin clases", result.errors[0])

    def test_group_end_to_end_artifacts_and_score_reproducibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(Path(folder) / "groups.csv", ["signal", "entity", "target"], self._group_rows())
            req = explicit_request(str(path), "target", "binary_classification", "f1", 42,
                split_strategy="group", group_column="entity", test_size=.25)
            one = agent.run_research(req, retriever=lambda *a, **k: [])
            two = agent.run_research(req, retriever=lambda *a, **k: [])
            self.assertEqual(one.status, "completed", one.errors)
            self.assertEqual(one.baseline["split_diagnostics"], two.baseline["split_diagnostics"])
            self.assertEqual(one.split_diagnostics, one.baseline["split_diagnostics"])
            self.assertEqual(one.baseline["score"], two.baseline["score"])
            self.assertEqual(one.baseline["split_diagnostics"]["group_intersection"], [])
            with tempfile.TemporaryDirectory() as output:
                jp, mp = agent.save_result(one, req, output)
                payload = json.loads(jp.read_text(encoding="utf-8")); md = mp.read_text(encoding="utf-8")
                self.assertEqual(payload["result"]["baseline"]["split_diagnostics"], one.baseline["split_diagnostics"])
                self.assertIn("Diagnóstico del split", md); self.assertIn("intersección", md)

    def test_group_regression_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[g * 20 + i, f"account-{g}", (g * 20 + i) * 1.25] for g in range(8) for i in range(5)]
            path = self._write(Path(folder) / "group-regression.csv", ["x", "account", "target"], rows)
            req = explicit_request(str(path), "target", "regression", "rmse", 8,
                split_strategy="group", group_column="account", test_size=.25)
            result = agent.run_research(req, retriever=lambda *a, **k: [])
            self.assertEqual(result.status, "completed", result.errors)
            self.assertEqual(result.baseline["split_strategy"], "group")
            self.assertEqual(result.baseline["split_diagnostics"]["group_intersection"], [])
            self.assertTrue(result.baseline["target_excluded"])

    def test_time_split_strict_order_ties_and_determinism(self):
        stamps = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"]
        rows = [{"when": value} for value in stamps]
        one = agent._time_split(rows, "when", 99, .4, None); two = agent._time_split(rows, "when", 99, .4, None)
        self.assertEqual(one, two)
        train, valid, diag = one
        self.assertLess(max(rows[i]["when"] for i in train), min(rows[i]["when"] for i in valid))
        self.assertTrue(diag["strict_chronology"])
        for stamp in set(stamps):
            self.assertFalse(any(rows[i]["when"] == stamp for i in train) and any(rows[i]["when"] == stamp for i in valid))

    def test_time_validation_and_explicit_format(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[i, f"2024-01-{i + 1:02d}", float(i * i + 1)] for i in range(24)]
            path = self._write(Path(folder) / "time.csv", ["x", "when", "target"], rows)
            omitted = explicit_request(str(path), "target", "regression", "mae", split_strategy="time")
            self.assertEqual(agent.run_research(omitted).status, "needs_input")
            missing = explicit_request(str(path), "target", "regression", "mae", split_strategy="time", time_column="absent")
            self.assertIn("no existe", agent.run_research(missing, retriever=lambda *a, **k: []).errors[0])
            bad = [{"t": "01/02/2024"}, {"t": "02/02/2024"}]
            with self.assertRaisesRegex(ValueError, "ambigua"):
                agent._time_split(bad, "t", 1, .5, None)
            self.assertTrue(agent._time_split(bad, "t", 1, .5, "%d/%m/%Y")[2]["strict_chronology"])
            for source, pattern in [
                ([{"t": ""}, {"t": "2020-01-02"}], "vacíos"),
                ([{"t": "no-fecha"}, {"t": "2020-01-02"}], "inválida o ambigua"),
                ([{"t": "2024-01-01"}, {"t": "2024-01-02T00:00:00Z"}], "mezcla"),
                ([{"t": "2024-01-01"}] * 4, "más de un timestamp")]:
                with self.assertRaisesRegex(ValueError, pattern):
                    agent._time_split(source, "t", 1, .5, None)

    def test_time_class_metric_failure_and_regression_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            class_rows = [[i, f"2024-01-{i+1:02d}", ("a" if i % 2 == 0 else "b") if i < 14 else "a"] for i in range(20)]
            class_path = self._write(Path(folder) / "class.csv", ["x", "when", "target"], class_rows)
            req = explicit_request(str(class_path), "target", "binary_classification", "balanced_accuracy", 42, positive_class="a",
                split_strategy="time", time_column="when", test_size=.2)
            failed = agent.run_research(req, retriever=lambda *a, **k: [])
            self.assertEqual(failed.status, "failed"); self.assertIn("balanced_accuracy no es calculable", failed.errors[0])
            rows = [[i, f"2024-01-{i+1:02d}", float(i * i + 1)] for i in range(24)]
            path = self._write(Path(folder) / "reg.csv", ["x", "when", "target"], rows)
            req = explicit_request(str(path), "target", "regression", "rmse", 42,
                split_strategy="time", time_column="when", test_size=.25)
            one = agent.run_research(req, retriever=lambda *a, **k: []); two = agent.run_research(req, retriever=lambda *a, **k: [])
            self.assertEqual(one.status, "completed", one.errors)
            self.assertTrue(one.baseline["split_diagnostics"]["strict_chronology"])
            self.assertEqual(one.baseline["split_diagnostics"], two.baseline["split_diagnostics"])
            self.assertEqual(one.baseline["score"], two.baseline["score"])

    def test_defaults_do_not_infer_and_invalid_options_do_not_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[i, f"g{i % 4}", f"2024-01-{i+1:02d}", i * 2.0] for i in range(20)]
            path = self._write(Path(folder) / "named.csv", ["id", "group", "date", "target"], rows)
            request = explicit_request(str(path), "target", "regression", "mae")
            result = agent.run_research(request, retriever=lambda *a, **k: [])
            self.assertEqual(result.baseline["split_strategy"], "random_holdout")
            bad_size = explicit_request(str(path), "target", "regression", "mae", test_size=1.0)
            self.assertTrue(any("test_size" in issue for issue in agent.validate_request(bad_size)))
            incomplete = explicit_request(str(path), "target", "regression", "mae", split_strategy="group")
            self.assertEqual(agent.run_research(incomplete).status, "needs_input")
            constant_group = self._write(Path(folder) / "constant.csv", ["x", "entity", "target"], [[i, "same", i] for i in range(20)])
            impossible = explicit_request(str(constant_group), "target", "regression", "mae", split_strategy="group", group_column="entity")
            impossible_result = agent.run_research(impossible, retriever=lambda *a, **k: [])
            self.assertEqual(impossible_result.status, "failed")
            self.assertTrue(any("grupos distintos" in error for error in impossible_result.errors))
            self.assertFalse(impossible_result.split_diagnostics["valid"])
            self.assertEqual(impossible_result.split_diagnostics["strategy"], "group")
            self.assertNotIn("random", str(impossible_result.to_dict()).casefold())
            combined = explicit_request(str(path), "target", "regression", "mae", split_strategy="group", group_column="group", time_column="date")
            self.assertEqual(agent.run_research(combined).status, "unsupported")

    def test_cli_protocols_negative_cases_and_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            group_rows = [[i, f"g{i // 6}", f"2024-01-{i+1:02d}", "yes" if i % 2 else "no", float(i * 2)]
                          for i in range(60)]
            group_path = self._write(root / "group.csv", ["x", "entity", "date", "class", "value"], group_rows)
            time_path = self._write(root / "time.csv", ["x", "entity", "date", "target"], [[i, f"g{i % 5}", f"2024-01-{i+1:02d}", float(i*i+1)] for i in range(30)])
            invalid_time = self._write(root / "invalid-time.csv", ["x", "date", "target"], [[1, "2024-01-01", 2], [2, "not-a-date", 3], [3, "2024-01-03", 4]])
            one_group = self._write(root / "one-group.csv", ["x", "entity", "target"], [[i, "same", i + 1] for i in range(20)])

            def invoke(csv_path, target, *args, label):
                out = root / label
                with Path(csv_path).open(encoding='utf-8') as stream:
                    columns = next(csv.reader(stream))
                decision_path = root / (label + '.json')
                decision_path.write_text(json.dumps({
                    'schema_version': 1,
                    'configuration': {'observation_structure': 'independent', 'split_strategy': 'group' if '--group-column' in args or '--split-strategy' in args and args[args.index('--split-strategy') + 1] == 'group' else 'time' if '--time-column' in args else 'random'},
                    'positive_class': 'yes' if 'binary_classification' in args else None,
                    'row_meaning': 'Caso sintético', 'prediction_scenario': 'Antes del resultado',
                    'metric_reason': 'Métrica de prueba',
                    'predictor_availability': {c: 'available' for c in columns if c != target},
                    'responses': [{'column_name': c, 'decision': 'keep', 'reason': 'Variable previa del escenario sintético', 'column_meaning': f'Entrada sintética {c}'} for c in columns if c != target]
                }), encoding='utf-8')
                raw = json.loads(decision_path.read_text(encoding='utf-8'))
                config = {'csv_path': str(csv_path), 'target': target}
                for option, field in [('--task', 'task_type'), ('--metric', 'metric'), ('--group-column', 'group_column'), ('--time-column', 'time_column')]:
                    if option in args:
                        config[field] = args[args.index(option) + 1]
                raw['configuration'].update(config)
                bound = bind_decisions(agent.merge_decisions(agent.ResearchRequest(None, None), raw, set()))
                raw.update({k: v for k, v in bound.decisions.items() if k in {'review_binding', 'dataset_sha256'}})
                decision_path.write_text(json.dumps(raw), encoding='utf-8')
                return subprocess.run([sys.executable, str(Path(agent.__file__)), '--decisions-json', str(decision_path), "--csv", str(csv_path),
                    "--target", target, "--output-dir", str(out), *args], cwd=Path(agent.__file__).parent,
                    capture_output=True, text=True, encoding="utf-8", errors="replace")

            gc = invoke(group_path, "class", "--task", "binary_classification", "--metric", "f1",
                "--split-strategy", "group", "--group-column", "entity", label="cli-group-class")
            self.assertEqual(gc.returncode, 0, gc.stderr + gc.stdout)
            gr = invoke(group_path, "value", "--task", "regression", "--metric", "rmse",
                "--split-strategy", "group", "--group-column", "entity", label="cli-group-reg")
            self.assertEqual(gr.returncode, 0, gr.stderr + gr.stdout)
            tr = invoke(time_path, "target", "--task", "regression", "--metric", "rmse",
                "--split-strategy", "time", "--time-column", "date", label="cli-time-reg")
            self.assertEqual(tr.returncode, 0, tr.stderr + tr.stdout)
            default = invoke(group_path, "value", "--task", "regression", "--metric", "rmse", label="cli-default")
            self.assertEqual(default.returncode, 0, default.stderr + default.stdout)
            default_json = json.loads((root / "cli-default" / "research_result.json").read_text(encoding="utf-8"))
            self.assertEqual(default_json["result"]["baseline"]["split_strategy"], "random_holdout")
            repeated = invoke(group_path, "value", "--task", "regression", "--metric", "rmse", label="cli-default-repeat")
            self.assertEqual(repeated.returncode, 0, repeated.stderr + repeated.stdout)
            repeated_json = json.loads((root / "cli-default-repeat" / "research_result.json").read_text(encoding="utf-8"))
            self.assertEqual(default_json["result"]["dataset_audit"]["dataset_sha256"], repeated_json["result"]["dataset_audit"]["dataset_sha256"])
            self.assertEqual(default_json["result"]["baseline"]["split_diagnostics"], repeated_json["result"]["baseline"]["split_diagnostics"])
            self.assertEqual(default_json["result"]["baseline"]["score"], repeated_json["result"]["baseline"]["score"])
            for label, strategy in [("cli-group-class", "group"), ("cli-group-reg", "group"), ("cli-time-reg", "time")]:
                payload = json.loads((root / label / "research_result.json").read_text(encoding="utf-8"))
                report = (root / label / "research_report.md").read_text(encoding="utf-8")
                diag = payload["result"]["baseline"]["split_diagnostics"]
                self.assertEqual(diag["strategy"], strategy)
                self.assertTrue(diag["valid"])
                self.assertIn("Diagnóstico del split", report)
            omitted = invoke(group_path, "value", "--task", "regression", "--metric", "rmse",
                "--split-strategy", "group", label="cli-omitted")
            self.assertEqual(omitted.returncode, 2)
            bad_size = invoke(group_path, "value", "--task", "regression", "--metric", "rmse",
                "--test-size", "1", label="cli-bad-size")
            self.assertEqual(bad_size.returncode, 2)
            missing = invoke(group_path, "value", "--task", "regression", "--metric", "rmse",
                "--split-strategy", "group", "--group-column", "absent", label="cli-missing")
            self.assertEqual(missing.returncode, 1)
            invalid = invoke(invalid_time, "target", "--task", "regression", "--metric", "mae",
                "--split-strategy", "time", "--time-column", "date", label="cli-invalid-time")
            self.assertEqual(invalid.returncode, 1)
            impossible = invoke(one_group, "target", "--task", "regression", "--metric", "mae",
                "--split-strategy", "group", "--group-column", "entity", label="cli-impossible")
            self.assertEqual(impossible.returncode, 1)


class CrossValidationTests(unittest.TestCase):
    def test_selection_requires_explicit_independence_and_preserves_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "data.csv")
            audit, rows = agent.audit_dataset(str(path), "target", "binary_classification")
            legacy = agent.run_baseline(explicit_request(str(path), "target", metric="f1"), audit, rows)
            self.assertEqual(legacy["validation"]["strategy"], "stratified_holdout")
            self.assertIsNone(legacy["validation"]["aggregate"])
            unknown = agent.run_research(explicit_request(str(path), "target", metric="f1",
                validation_strategy="cross_validation", observation_structure="unknown"), retriever=lambda *a, **k: [])
            self.assertEqual(unknown.status, "needs_input")
            self.assertIn("independent", " ".join(unknown.errors))

    def test_classification_kfold_stratification_reproducibility_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "class.csv")
            request = explicit_request(str(path), "target", metric="balanced_accuracy", seed=17,
                validation_strategy="cross_validation", observation_structure="independent")
            first = agent.run_research(request, retriever=lambda *a, **k: [])
            second = agent.run_research(request, retriever=lambda *a, **k: [])
            self.assertEqual(first.status, "completed", first.errors)
            validation = first.baseline["validation"]
            self.assertEqual(validation["strategy"], "stratified_kfold")
            self.assertEqual(validation["effective_folds"], 5)
            self.assertEqual(validation["fit_count"], 5)
            self.assertEqual(validation["folds"], second.baseline["validation"]["folds"])
            self.assertEqual(validation["aggregate"], second.baseline["validation"]["aggregate"])
            self.assertTrue(all(set(fold["validation_class_counts"]) == {"no", "yes"}
                                for fold in validation["folds"]))
            labels = [row["target"] for row in agent.audit_dataset(str(path), "target", "binary_classification")[1]]
            splits = agent._kfold_indices(len(labels), 5, 17, labels)
            validation_indexes = [idx for _, valid in splits for idx in valid]
            self.assertEqual(sorted(validation_indexes), list(range(len(labels))))
            with tempfile.TemporaryDirectory() as output:
                json_path, md_path = agent.save_result(first, request, output)
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                markdown = md_path.read_text(encoding="utf-8")
                self.assertEqual(payload["result"]["baseline"]["validation"], validation)
                self.assertIn("Fold 1", markdown)
                self.assertIn("desviación estándar poblacional", markdown)
                self.assertIn(validation["dataset_sha256"], markdown)

    def test_regression_kfold_metrics_sizes_and_r2(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "reg.csv")
            audit, rows = agent.audit_dataset(str(path), "target", "regression")
            for metric in ("mae", "rmse", "r2"):
                request = explicit_request(str(path), "target", "regression", metric, seed=42,
                    validation_strategy="cross_validation", observation_structure="independent")
                one = agent.run_baseline(request, audit, rows)
                two = agent.run_baseline(request, audit, rows)
                validation = one["validation"]
                self.assertEqual(validation["strategy"], "kfold")
                self.assertEqual(len(validation["folds"]), 5)
                self.assertEqual(validation["folds"], two["validation"]["folds"])
                self.assertEqual(sum(f["validation_rows"] for f in validation["folds"]), len(rows))
                self.assertLessEqual(max(f["validation_rows"] for f in validation["folds"])
                                     - min(f["validation_rows"] for f in validation["folds"]), 1)
                self.assertAlmostEqual(validation["aggregate"]["mean"],
                                       sum(f["score"] for f in validation["folds"]) / 5)

    def test_explicit_r2_reports_constant_fold_and_class_support_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rare_target = root / "rare-target.csv"
            with rare_target.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["signal", "target"])
                writer.writerows([[i, 1 if i == 9 else 0] for i in range(10)])
            audit, rows = agent.audit_dataset(str(rare_target), "target", "regression")
            request = explicit_request(str(rare_target), "target", "regression", "r2",
                validation_strategy="cross_validation", observation_structure="independent", cv_folds=5)
            with self.assertRaisesRegex(ValueError, "R² no está definido en el fold"):
                agent.run_baseline(request, audit, rows)

            class_path = root / "class.csv"
            with class_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["signal", "target"])
                writer.writerows([[i, "yes" if i < 2 else "no"] for i in range(20)])
            class_audit, class_rows = agent.audit_dataset(str(class_path), "target", "binary_classification")
            request_class = explicit_request(str(class_path), "target", metric="f1",
                validation_strategy="cross_validation", observation_structure="independent", cv_folds=5)
            with self.assertRaisesRegex(ValueError, "excede soporte por clase"):
                agent.run_baseline(request_class, class_audit, class_rows)

    def test_grouped_temporal_and_fit_limit_never_route_to_random_cv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "data.csv")
            grouped = explicit_request(str(path), "target", metric="f1",
                validation_strategy="cross_validation", observation_structure="grouped",
                split_strategy="group", group_column="group")
            self.assertTrue(any("CV aleatorio ordinario" in issue for issue in agent.validate_request(grouped)))
            no_random_holdout = explicit_request(str(path), "target", metric="f1",
                validation_strategy="holdout", observation_structure="grouped")
            self.assertTrue(any("split explícito por grupo" in issue for issue in agent.validate_request(no_random_holdout)))
            independent = explicit_request(str(path), "target", metric="f1", seed=42,
                validation_strategy="cross_validation", observation_structure="independent", max_cv_fits=3)
            result = agent.run_research(independent, retriever=lambda *a, **k: [])
            self.assertEqual(result.status, "failed")
            self.assertIn("no se reduce K", " ".join(result.errors))

    def test_auto_is_deterministic_and_reduces_only_with_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_csv(Path(folder) / "data.csv")
            request = explicit_request(str(path), "target", metric="f1", seed=11,
                validation_strategy="auto", observation_structure="independent", cv_folds=7, max_cv_fits=3)
            audit, rows = agent.audit_dataset(str(path), "target", "binary_classification")
            first = agent.run_baseline(request, audit, rows)
            second = agent.run_baseline(request, audit, rows)
            self.assertEqual(first["validation"]["strategy"], "stratified_kfold")
            self.assertEqual(first["validation"]["effective_folds"], 3)
            self.assertEqual(first["validation"], second["validation"])
            self.assertTrue(any("redujo folds de 7 a 3" in note for note in first["validation"]["notes"]))


class ColumnReviewSignalTests(unittest.TestCase):
    def write_rows(self, folder: str, name: str, columns: list[str], rows: list[list[object]]) -> Path:
        path = Path(folder) / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(columns); writer.writerows(rows)
        return path

    @staticmethod
    def signal(audit, rule_id: str):
        return [item for item in audit.column_review_signals if item["rule_id"] == rule_id]

    def test_identifier_rules_combine_name_uniqueness_sample_and_keep_predictors(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[f"secret-{i}", f"v{i}", f"repeat-{i % 2}", "yes" if i % 2 else "no"] for i in range(12)]
            path = self.write_rows(folder, "ids.csv", ["customer_id", "measurement", "key", "target"], rows)
            audit, parsed = agent.audit_dataset(path, "target")
            self.assertTrue(any(s["column_name"] == "customer_id" for s in audit.column_review_signals if s["rule_id"] == "identifier.name_token"))
            self.assertTrue(any(s["column_name"] == "measurement" for s in audit.column_review_signals if s["rule_id"] == "identifier.high_uniqueness"))
            self.assertTrue(any(s["column_name"] == "key" for s in audit.column_review_signals if s["rule_id"] == "identifier.name_token"))
            self.assertFalse(any(s["column_name"] == "key" and s["rule_id"] == "identifier.high_uniqueness" for s in audit.column_review_signals))
            request = explicit_request(str(path), "target", metric="f1")
            before = [c for c in audit.column_names if c != "target"]
            baseline = agent.run_baseline(request, audit, parsed)
            self.assertEqual(baseline["predictor_columns"], before)
            small = self.write_rows(folder, "small.csv", ["measurement", "target"], [[i, "a" if i % 2 else "b"] for i in range(5)])
            small_audit, _ = agent.audit_dataset(small, "target")
            insufficient = next(s for s in small_audit.column_review_signals if s["rule_id"] == "identifier.insufficient_sample")
            self.assertEqual(insufficient["status"], "not_evaluable")

    def test_date_iso_ambiguous_partial_missing_and_default_split(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [["2024-01-01", "a"], ["01/02/2024", "b"], ["bad", "a"], ["", "b"], ["2024-02-01", "a"], ["2024-03-01", "b"], ["13/02/2024", "a"]]
            path = self.write_rows(folder, "dates.csv", ["event_date", "target"], rows)
            audit, parsed = agent.audit_dataset(path, "target")
            signal = next(s for s in audit.column_review_signals if s["rule_id"] == "datetime.parseability")
            self.assertEqual(signal["evidence"]["parseable_count"], 4)
            self.assertEqual(signal["evidence"]["ambiguous_count"], 1)
            self.assertEqual(signal["evidence"]["invalid_count"], 1)
            self.assertEqual(signal["evidence"]["missing_count"], 1)
            req = explicit_request(str(path), "target", metric="f1", positive_class="a")
            self.assertEqual(req.split_strategy, "random")
            self.assertEqual(agent.run_baseline(req, audit, parsed)["split_strategy"], "stratified_holdout")
            explicit, _ = agent.audit_dataset(path, "target", time_format="%d/%m/%Y")
            explicit_signal = next(s for s in explicit.column_review_signals if s["rule_id"] == "datetime.parseability")
            self.assertEqual(explicit_signal["evidence"]["parseable_count"], 5)

    def test_tukey_outliers_zero_iqr_small_sample_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as folder:
            outlier = self.write_rows(folder, "outlier.csv", ["x", "target"], [[v, "a" if i % 2 else "b"] for i, v in enumerate([0,1,2,3,4,5,6,7,8,9,100])])
            audit, _ = agent.audit_dataset(outlier, "target")
            sig = next(s for s in audit.column_review_signals if s["rule_id"] == "outlier.tukey_1_5_iqr")
            self.assertEqual(sig["evidence"]["outside_count"], 1)
            self.assertEqual(sig["evidence"]["method"], "Tukey 1.5×IQR")
            ordinary = self.write_rows(folder, "ordinary.csv", ["x", "target"], [[v, "a" if i % 2 else "b"] for i, v in enumerate([1,1,1,2,2,3])])
            ordinary_audit, _ = agent.audit_dataset(ordinary, "target")
            self.assertFalse(any(s["column_name"] == "x" and s["rule_id"] == "outlier.tukey_1_5_iqr" for s in ordinary_audit.column_review_signals))
            constant = self.write_rows(folder, "constant.csv", ["x", "target"], [[1, "a" if i % 2 else "b"] for i in range(12)])
            constant_audit, _ = agent.audit_dataset(constant, "target")
            zero = next(s for s in constant_audit.column_review_signals if s["column_name"] == "x" and s["rule_id"] == "outlier.zero_iqr")
            self.assertEqual(zero["status"], "not_evaluable")
            small = self.write_rows(folder, "tiny.csv", ["x", "target"], [[1, "a"], [2, "b"], [3, "a"]])
            small_audit, _ = agent.audit_dataset(small, "target")
            self.assertTrue(any(s["rule_id"] == "outlier.small_sample" for s in small_audit.column_review_signals))
            nonfinite = self.write_rows(folder, "nonfinite.csv", ["x", "target"], [[v, "a" if i % 2 else "b"] for i, v in enumerate([1,2,3,4,5,100,"NaN","inf",""])])
            nonfinite_audit, _ = agent.audit_dataset(nonfinite, "target")
            nf = next(s for s in nonfinite_audit.column_review_signals if s["rule_id"] == "outlier.tukey_1_5_iqr")
            self.assertEqual(nf["evidence"]["non_finite_count"], 1)
            self.assertEqual(nf["evidence"]["missing_count"], 2)

    def test_proxy_uses_train_only_and_does_not_exclude_predictor(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[i, "yes" if i % 2 else "no", "yes" if i % 2 else "no", i % 2, f"category-{i}"] for i in range(40)]
            path = self.write_rows(folder, "proxy.csv", ["signal", "target", "result_code", "numeric_proxy", "unique_category"], rows)
            audit, parsed = agent.audit_dataset(path, "target")
            request = explicit_request(str(path), "target", metric="f1", seed=9)
            train, valid, split = agent._split_indices(request, parsed, [r["target"] for r in parsed])
            first = agent._target_relation_signals(parsed, audit.column_names, "target", agent.TASK, train, split["strategy"])
            self.assertTrue(any(s["column_name"] == "result_code" and s["rule_id"] == "target_relation.category_purity" for s in first))
            self.assertTrue(any(s["column_name"] == "numeric_proxy" and s["rule_id"] == "target_relation.numeric_correlation" for s in first))
            self.assertFalse(any(s["column_name"] == "unique_category" and s["rule_id"] == "target_relation.category_purity" for s in first))
            modified = [dict(row) for row in parsed]
            for i in valid:
                modified[i]["target"] = "yes" if modified[i]["target"] == "no" else "no"
            second = agent._target_relation_signals(modified, audit.column_names, "target", agent.TASK, train, split["strategy"])
            self.assertEqual(first, second)
            baseline = agent.run_baseline(request, audit, parsed)
            self.assertIn("result_code", baseline["predictor_columns"])
            self.assertFalse(any("leakage confirmado" in str(s).casefold() for s in first))

    def test_signal_artifacts_match_and_do_not_print_identifier_values(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [[f"private-id-{i}", "2024-01-01", i, "yes" if i % 2 else "no"] for i in range(40)]
            path = self.write_rows(folder, "private.csv", ["user_id", "date", "value", "target"], rows)
            result = agent.run_audit_only(explicit_request(str(path), "target", metric="f1"))
            self.assertEqual(result.status, "completed")
            self.assertIsNone(result.baseline)
            self.assertFalse(result.dataset_audit["score_computed"])
            with tempfile.TemporaryDirectory() as output:
                jp, mp = agent.save_result(result, explicit_request(str(path), "target", metric="f1"), output)
                payload = json.loads(jp.read_text(encoding="utf-8"))
                report = mp.read_text(encoding="utf-8")
                signals = payload["result"]["dataset_audit"]["column_review_signals"]
                self.assertEqual(signals, result.dataset_audit["column_review_signals"])
                self.assertTrue(agent.validate_report_consistency(result, payload, report))
                self.assertNotIn("private-id-0", report)
                self.assertNotIn("private-id-0", jp.read_text(encoding="utf-8"))
                self.assertIn("Columnas y señales para revisión humana", report)
                self.assertIn("No se calculó ningún score", report)
                self.assertTrue(all(s["rule_id"] and s["evidence"] and s["why_review"] and s["review_question"] for s in signals))

    def test_audit_only_cli_writes_report_without_score_or_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_regression_csv(Path(folder) / "reg.csv", 30)
            output = Path(folder) / "out"
            completed = subprocess.run([sys.executable, str(Path(agent.__file__)), "--csv", str(path),
                "--target", "target", "--task", "regression", "--audit-only", "--output-dir", str(output)],
                capture_output=True, text=True, cwd=Path(agent.__file__).parent, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((output / "research_result.json").read_text(encoding="utf-8"))
            self.assertIsNone(payload["result"]["baseline"])
            self.assertFalse(payload["result"]["dataset_audit"]["score_computed"])
            self.assertNotIn("## Baseline", (output / "research_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
