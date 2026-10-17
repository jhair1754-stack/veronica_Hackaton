"""Behavioral tests for explicit semantics, without external providers."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import research_agent as agent
from decision_fixtures import bind_decisions


class SemanticDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data.csv'
        self.path.write_text('x,target\n' + ''.join(
            f'{i},{" accept " if i % 2 else "reject"}\n' for i in range(40)))
        self.data = {
            'schema_version': 1,
            'configuration': {'csv_path': str(self.path), 'target': 'target',
                              'task_type': agent.TASK, 'metric': 'f1',
                              'observation_structure': 'independent', 'split_strategy': 'random'},
            'positive_class': ' accept ', 'row_meaning': 'Una solicitud',
            'prediction_scenario': 'Al recibir la solicitud',
            'metric_reason': 'Equilibrar precisión y recuperación de aceptaciones',
            'predictor_availability': {'x': 'available'},
            'responses': [{'column_name': 'x', 'decision': 'keep', 'reason': 'Medición previa a la solicitud', 'column_meaning': 'Valor sintético de entrada'}],
        }

    def request(self, data=None):
        return bind_decisions(agent.merge_decisions(agent.ResearchRequest(None, None), data or self.data, set()))

    def run_request(self, request):
        return agent.run_research(request, retriever=lambda *a, **k: [])

    def test_declared_non_alphabetical_positive_and_serialization(self):
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertEqual(result.baseline['class_mapping'], {'reject': 0, 'accept': 1})
        self.assertEqual(result.baseline['positive_class'], 'accept')
        self.assertEqual(result.plan['metric_decision']['status'], 'confirmed')
        output, _ = agent.save_result(result, self.request(), Path(self.temp.name) / 'out')
        self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['result']['baseline']['class_mapping'],
                         {'reject': 0, 'accept': 1})

    def test_manual_precision_recall_f1_on_known_predictions(self):
        # Validation truth: accept, accept, accept, reject.
        # Predictions:       accept, reject, reject, accept => TP=1, FP=1, FN=2.
        self.path.write_text('x,target\n1,accept\n2,reject\n3, accept \n4,accept\n5,accept\n6,reject\n')
        audit, rows = agent.audit_dataset(self.path, 'target')
        for metric, expected in [('precision', 1/2), ('recall', 1/3), ('f1', 2/5)]:
            self.data['configuration']['metric'] = metric
            request = self.request()
            def predict(candidate, task, train, valid, labels, classes):
                self.assertEqual(classes, ['reject', 'accept'])
                return [1, 0, 0, 1]
            with patch.object(agent, '_split_indices', return_value=(
                    [0, 1], [2, 3, 4, 5], {'strategy': 'stratified_holdout', 'valid': True})), \
                    patch.object(agent, '_candidate_prediction', side_effect=predict):
                bind_decisions(request)
                result = agent.run_baseline(request, audit, rows)
            self.assertAlmostEqual(result['score'], expected)

    def test_missing_semantics_block_before_fit_but_allow_inspection(self):
        for field in ['positive_class', 'row_meaning', 'prediction_scenario', 'metric_reason', 'predictor_availability']:
            data = copy.deepcopy(self.data)
            del data[field]
            request = self.request(data)
            with self.subTest(field=field), patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')):
                audit = agent.run_audit_only(request)
                result = self.run_request(request)
                self.assertEqual(audit.status, 'completed', audit.errors)
                self.assertEqual(result.status, 'needs_input', result.errors)
                self.assertIsNone(result.baseline)

    def test_unknown_or_unavailable_predictor_block(self):
        for availability in ['unknown', 'unavailable']:
            self.data['predictor_availability']['x'] = availability
            result = self.run_request(self.request())
            self.assertEqual(result.status, 'needs_input')
            self.assertIn("'x'", ' '.join(result.errors))
            self.assertIsNone(result.baseline)

    def test_no_protocol_or_unknown_structure_does_not_invent_split(self):
        for field, value in [('split_strategy', None), ('observation_structure', 'unknown')]:
            data = copy.deepcopy(self.data)
            data['configuration'][field] = value
            with self.subTest(field=field), patch.object(agent, '_split_indices', side_effect=AssertionError('invented split')):
                audit = agent.run_audit_only(self.request(data))
                result = self.run_request(self.request(data))
            self.assertEqual(audit.status, 'completed')
            self.assertTrue(audit.decision_contract['signal_evaluation_limitations'])
            self.assertFalse(any(s['rule_id'].startswith('target_relation.') for s in audit.dataset_audit['column_review_signals']))
            self.assertEqual(result.status, 'needs_input')

    def test_recommendation_is_not_confirmation(self):
        self.path.write_text('x,target\n' + ''.join(f'{i},{i * 3}\n' for i in range(40)))
        self.data.pop('positive_class')
        self.data['configuration'].update(task_type='regression', metric=None, metric_context='large_errors_cost_more')
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'needs_input')
        selection = result.decision_contract['metric_decision']
        self.assertEqual(selection['effective_metric'], 'rmse')
        self.assertEqual(selection['status'], 'recommendation')
        self.data['configuration']['metric'] = 'rmse'
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertEqual(result.plan['metric_decision']['human_reason'], self.data['metric_reason'])

    def test_whitespace_and_case_normalization(self):
        for field in ['row_meaning', 'prediction_scenario', 'metric_reason', 'positive_class']:
            data = copy.deepcopy(self.data)
            data[field] = ' \t '
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.request(data)
        self.data['positive_class'] = 'Accept'
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'failed')
        self.assertIn('Corrija positive_class', result.errors[0])

    def test_invalid_targets_never_produce_metrics(self):
        for task, values in [(agent.TASK, ['a'] * 4), (agent.TASK, ['a', 'b', 'c', 'b']),
                             (agent.TASK, ['a', '', 'b', 'a']), ('regression', ['1', 'inf', '2', '3']),
                             ('regression', ['1', 'NaN', '2', '3']), ('regression', ['1', 'text', '2', '3'])]:
            with self.subTest(task=task, values=values):
                self.path.write_text('x,target\n' + ''.join(f'{i},{v}\n' for i, v in enumerate(values)))
                data = copy.deepcopy(self.data)
                data['configuration'].update(task_type=task, metric='mae' if task == 'regression' else 'f1')
                data['positive_class'] = None if task == 'regression' else 'a'
                with patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')):
                    result = self.run_request(self.request(data))
                self.assertEqual(result.status, 'failed', result.to_dict())
                self.assertTrue(result.errors)
                self.assertIsNone(result.baseline)

    def test_no_contract_cannot_bypass_and_all_exclusions_block(self):
        legacy = agent.ResearchRequest(str(self.path), 'target', metric='f1')
        result = self.run_request(legacy)
        self.assertEqual(result.status, 'needs_input')
        audit, rows = agent.audit_dataset(self.path, 'target')
        with self.assertRaises(ValueError):
            agent.run_baseline(legacy, audit, rows)
        self.data['responses'] = []
        self.data['excluded_columns'] = {'x': 'No usar'}
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'needs_input')
        self.assertIn('No quedan predictores', ' '.join(result.errors))

    def test_positive_mapping_is_preserved_in_cross_validation(self):
        self.data['configuration'].update(validation_strategy='cross_validation', cv_folds=3)
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertEqual(result.baseline['class_mapping'], {'reject': 0, 'accept': 1})
        self.assertEqual(result.baseline['validation']['effective_folds'], 3)

    def test_missing_task_declaration_remains_pending(self):
        del self.data['configuration']['task_type']
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'needs_input')
        self.assertIsNone(result.baseline)

    def test_normalization_does_not_create_a_second_class(self):
        self.path.write_text('x,target\n1,accept\n2, accept \n3,accept\n')
        result = self.run_request(self.request())
        self.assertEqual(result.status, 'failed')
        self.assertIn('dos clases', result.errors[0])
        self.assertIsNone(result.baseline)


if __name__ == '__main__':
    unittest.main()
