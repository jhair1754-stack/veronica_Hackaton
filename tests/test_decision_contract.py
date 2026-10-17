import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import decision_contract as contract
import research_agent as agent
from decision_fixtures import bind_decisions


class DecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / 'data.csv'
        self.csv.write_text('x,target\n' + ''.join(f'{i},{"yes" if i % 2 else "no"}\n' for i in range(40)))
        self.data = {'schema_version': 1, 'configuration': {
            'csv_path': str(self.csv), 'target': 'target', 'task_type': agent.TASK}}

    def request(self, data=None):
        return bind_decisions(agent.merge_decisions(agent.ResearchRequest(None, None), data or self.data, set()))

    def test_incomplete_read_audit_and_serialization(self):
        path = self.root / 'decisions.json'
        path.write_text(json.dumps(self.data))
        request = self.request(contract.load_contract(path))
        with patch.object(agent, 'run_baseline', side_effect=AssertionError('training')):
            result = agent.run_audit_only(request)
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertIn('positive_class', result.decision_contract['pending_decisions'])
        self.assertIn('metric', result.decision_contract['pending_decisions'])
        self.assertEqual(result.dataset_audit['human_review']['status'], 'pending')
        self.assertEqual(result.dataset_audit['human_review']['responses'], [])
        jp, _ = agent.save_result(result, request, self.root / 'out')
        self.assertEqual(json.loads(jp.read_text(encoding='utf-8'))['result'], result.to_dict())
        self.assertEqual(agent.run_research(request).status, 'needs_input')

    def test_valid_decisions_and_identity(self):
        self.data.update(dataset_sha256=hashlib.sha256(self.csv.read_bytes()).hexdigest(),
                         positive_class='yes', row_meaning='Una medición', prediction_scenario='Antes del resultado',
                         metric_reason='Prioridad de recuperación', predictor_availability={'x': 'available'},
                         responses=[{'column_name': 'x', 'decision': 'keep', 'reason': 'Medida previa', 'column_meaning': 'Medición de entrada'}])
        self.data['configuration'].update(metric='recall', observation_structure='independent', split_strategy='random')
        result = agent.run_audit_only(self.request())
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertEqual(result.decision_contract['pending_decisions'], [])
        self.assertEqual(result.dataset_audit['human_review']['status'], 'resolved')
        self.data['dataset_sha256'] = '0' * 64
        self.assertEqual(agent.run_research(self.request()).status, 'needs_input')

    def test_bad_shapes_versions_fields(self):
        cases = [[], {'schema_version': True}, {'schema_version': 2},
                 {'schema_version': 1, 'extra': 0}, {'schema_version': 1, 'configuration': []},
                 {'schema_version': 1, 'configuration': {'seed': True}},
                 {'schema_version': 1, 'configuration': {'target': 9}},
                 {'schema_version': 1, 'configuration': {'typo': 'x'}},
                 {'schema_version': 1, 'responses': {}},
                 {'schema_version': 1, 'predictor_availability': {'x': []}},
                 {'schema_version': 1, 'excluded_columns': {'x': False}}]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                contract.validate_contract(data)

    def test_unknown_columns_signals_and_positive_class(self):
        for key, value in [('predictor_availability', {'absent': 'available'}),
                           ('excluded_columns', {'absent': 'ID'}),
                           ('responses', [{'column_name': 'x', 'rule_id': 'invented', 'decision': 'pending'}]),
                           ('positive_class', 'absent')]:
            with self.subTest(key=key):
                result = agent.run_audit_only(self.request({**self.data, key: value}))
                self.assertEqual(result.status, 'failed', result.to_dict())

    def test_missing_reason_and_unresolved_split_remain_pending(self):
        self.data['configuration']['observation_structure'] = 'grouped'
        self.data['excluded_columns'] = {'x': None}
        result = agent.run_audit_only(self.request())
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertIn('excluded_columns.x.reason', result.decision_contract['pending_decisions'])
        self.assertTrue(result.decision_contract['signal_evaluation_limitations'])

    def test_cli_conflicts_including_explicit_default(self):
        self.data['configuration']['seed'] = 7
        path = self.root / 'decisions.json'
        path.write_text(json.dumps(self.data))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(agent.main(['--decisions-json', str(path), '--seed=42']), 1)
            self.assertEqual(agent.main(['--decisions-json', str(path), '--seed', '7', '--audit-only',
                                         '--output-dir', str(self.root / 'cli')]), 0)
        self.assertEqual(self.request().seed, 7)

    def test_malformed_duplicate_and_nonfinite_json(self):
        path = self.root / 'bad.json'
        for text in ['{', '{"schema_version":1,"schema_version":1}',
                     '{"schema_version":1,"configuration":{"test_size":NaN}}']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                contract.load_contract(path)

    def test_minimal_contract_and_regression_without_metric(self):
        minimal = self.request({'schema_version': 1})
        self.assertEqual(agent.run_audit_only(minimal).status, 'needs_input')
        self.csv.write_text('x,target\n' + ''.join(f'{i},{i * 2}\n' for i in range(40)))
        self.data['configuration']['task_type'] = 'regression'
        result = agent.run_audit_only(self.request())
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertIn('metric', result.decision_contract['pending_decisions'])
        self.assertNotIn('positive_class', result.decision_contract['pending_decisions'])
        self.assertIsNone(result.decision_contract['configuration']['metric'])
        self.data['configuration']['seed'] = 9
        other = agent.run_audit_only(self.request())
        self.assertNotEqual(result.decision_contract['configuration_sha256'],
                            other.decision_contract['configuration_sha256'])


if __name__ == '__main__':
    unittest.main()
