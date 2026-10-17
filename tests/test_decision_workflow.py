"""2D integration: real CLI loading/artifacts and fresh public API guards."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import research_agent as agent


class DecisionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / 'data.csv'
        self.csv.write_text('x,target,alternate\n' + ''.join(
            f'{i % 7},{"yes" if i % 2 else "no"},{"no" if i % 2 else "yes"}\n' for i in range(60)))
        self.template = self.root / 'pending.json'
        self.reviewed = self.root / 'reviewed.json'
        self.out = self.root / 'out'
        self.audit_args = ['--csv', str(self.csv), '--target', 'target', '--task', agent.TASK,
            '--metric', 'f1', '--split-strategy', 'random', '--observation-structure', 'independent',
            '--write-decisions-template', str(self.template), '--output-dir', str(self.out)]

    def cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return agent.main(args)

    def complete_fixture(self):
        self.assertEqual(self.cli(self.audit_args), 0)
        data = json.loads(self.template.read_text(encoding='utf-8'))
        data.update(positive_class='yes', row_meaning='Una solicitud sintética independiente',
                    prediction_scenario='Antes de conocer cualquiera de las etiquetas',
                    metric_reason='Balance de precision y recall para yes',
                    predictor_availability={'x': 'available', 'alternate': 'unavailable'},
                    excluded_columns={'alternate': 'Otra etiqueta conocida después de predecir'})
        # The synthetic fixture author supplies decisions explicitly, never production defaults.
        for response in data['responses']:
            if response['column_name'] == 'alternate':
                response.update(decision='exclude', reason='Etiqueta posterior al evento')
        self.reviewed.write_text(json.dumps(data), encoding='utf-8')
        return data

    def train_args(self):
        return ['--decisions-json', str(self.reviewed), '--output-dir', str(self.out)]

    def assert_blocked_without_fit(self, request):
        with patch.object(agent, '_encode', side_effect=AssertionError('preprocessing')), \
             patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')):
            result = agent.run_research(request)
        self.assertEqual(result.status, 'needs_input', result.errors)
        self.assertIsNone(result.baseline)
        self.assertIsNone(result.model_comparison)
        self.assertTrue(result.next_actions)
        return result

    def test_roundtrip_template_valid_execution_and_report_consistency(self):
        with patch.object(agent.rag_mvp, 'search', side_effect=AssertionError('retrieval in audit')), \
             patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit in audit')):
            data = self.complete_fixture()
        self.assertEqual(data['review_binding']['dataset_sha256'], data['dataset_sha256'])
        with patch.object(agent.rag_mvp, 'search', return_value=[]):
            self.assertEqual(self.cli(self.train_args()), 0)
        result = json.loads((self.out / 'research_result.json').read_text(encoding='utf-8'))['result']
        markdown = (self.out / 'research_report.md').read_text(encoding='utf-8')
        summary = json.loads(markdown.split('```json\n', 1)[1].split('\n```', 1)[0])
        self.assertEqual(summary, result['review_summary'])
        self.assertEqual(summary['review_status'], 'ready')
        self.assertEqual(summary['class_mapping'], {'no': 0, 'yes': 1})
        self.assertEqual(summary['effective_predictors'], ['x'])
        self.assertFalse(summary['final_independent_evaluation'])
        self.assertIsNotNone(result['baseline']['score'])

    def test_incomplete_template_blocks(self):
        self.cli(self.audit_args)
        data = json.loads(self.template.read_text(encoding='utf-8'))
        request = agent.merge_decisions(agent.ResearchRequest(None, None), data, set())
        self.assert_blocked_without_fit(request)

    def test_modified_csv_invalidates_and_public_api_cannot_reuse_audit(self):
        data = self.complete_fixture()
        request = agent.merge_decisions(agent.ResearchRequest(None, None), data, set())
        old_audit, old_rows = agent.audit_dataset(self.csv, 'target')
        self.csv.write_text(self.csv.read_text() + '3,yes,no\n')
        result = self.assert_blocked_without_fit(request)
        self.assertEqual(result.decision_contract['binding_status'], 'stale')
        old_audit.human_review = {'status': 'resolved', 'effective_predictors': ['x']}
        for runner in (agent.run_baseline, agent.run_model_comparison):
            with patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')), \
                 self.assertRaises(ValueError):
                runner(request, old_audit, old_rows)

    def test_target_protocol_and_signal_binding_changes_require_review(self):
        data = self.complete_fixture()
        for key, value in [('target', 'alternate'), ('seed', 99), ('test_size', 0.3),
                           ('validation_strategy', 'cross_validation')]:
            changed = copy.deepcopy(data)
            changed['configuration'][key] = value
            request = agent.merge_decisions(agent.ResearchRequest(None, None), changed, set())
            self.assert_blocked_without_fit(request)
        data['review_binding']['signals_sha256'] = '0' * 64
        self.assert_blocked_without_fit(agent.merge_decisions(agent.ResearchRequest(None, None), data, set()))

    def test_existing_template_never_overwritten(self):
        self.complete_fixture()
        self.template.write_bytes(self.reviewed.read_bytes())
        before = self.template.read_bytes()
        self.assertEqual(self.cli(self.audit_args), 1)
        self.assertEqual(before, self.template.read_bytes())

    def test_public_api_ignores_forged_review_and_rejects_changed_python_rows(self):
        data = self.complete_fixture()
        request = agent.merge_decisions(agent.ResearchRequest(None, None), data, set())
        audit, rows = agent.audit_dataset(self.csv, 'target')
        altered = copy.deepcopy(rows)
        altered[0]['x'] = '999'
        for runner in (agent.run_baseline, agent.run_model_comparison):
            with patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')), \
                 self.assertRaisesRegex(ValueError, 'datos Python'):
                runner(request, audit, altered)
        request.decisions['row_meaning'] = None
        audit.human_review = {'status': 'resolved', 'issues': [], 'effective_predictors': ['x']}
        for runner in (agent.run_baseline, agent.run_model_comparison):
            with patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')), \
                 self.assertRaises(ValueError):
                runner(request, audit, rows)

    def test_blocked_and_invalid_input_replace_old_reports(self):
        data = self.complete_fixture()
        with patch.object(agent.rag_mvp, 'search', return_value=[]):
            self.assertEqual(self.cli(self.train_args()), 0)
        data['row_meaning'] = None
        self.reviewed.write_text(json.dumps(data), encoding='utf-8')
        self.assertEqual(self.cli(self.train_args()), 2)
        current = json.loads((self.out / 'research_result.json').read_text(encoding='utf-8'))['result']
        self.assertIsNone(current['baseline'])
        self.assertNotIn('## Baseline', (self.out / 'research_report.md').read_text(encoding='utf-8'))
        self.reviewed.write_text('{invalid', encoding='utf-8')
        self.assertEqual(self.cli(self.train_args()), 1)
        current = json.loads((self.out / 'research_result.json').read_text(encoding='utf-8'))['result']
        self.assertEqual(current['status'], 'failed')
        self.assertIsNone(current['baseline'])


if __name__ == '__main__':
    unittest.main()
