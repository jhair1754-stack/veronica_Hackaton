"""Human review changes model inputs, never the partitioning columns."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import research_agent as agent
from decision_fixtures import bind_decisions


class PredictorReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'data.csv'
        self.path.write_text('x,post_result,target\n' + ''.join(
            f'{i % 7},{"yes" if i % 2 else "no"},{"yes" if i % 2 else "no"}\n' for i in range(60)))
        self.data = {'schema_version': 1,
            'configuration': {'csv_path': str(self.path), 'target': 'target', 'task_type': agent.TASK,
                              'metric': 'f1', 'split_strategy': 'random', 'observation_structure': 'independent'},
            'positive_class': 'yes', 'row_meaning': 'Solicitud sintética',
            'prediction_scenario': 'Al recibir la solicitud', 'metric_reason': 'Balance de precisión y recall',
            'predictor_availability': {'x': 'available', 'post_result': 'unavailable'},
            'responses': []}

    def request(self):
        return bind_decisions(agent.merge_decisions(agent.ResearchRequest(None, None), self.data, set()))

    def run_request(self):
        return agent.run_research(self.request(), retriever=lambda *a, **k: [])

    def test_exclusion_precedes_encoding_and_candidates_share_effective_list(self):
        self.data['excluded_columns'] = {'post_result': 'Se conoce después del resultado'}
        original_encode = agent._encode
        seen_features = []
        def encode(rows, features, types, train, selected):
            self.assertEqual(features, ['x'])
            seen_features.append(features)
            return original_encode(rows, features, types, train, selected)
        request = self.request()
        with patch.object(agent, '_encode', side_effect=encode), \
                patch.object(agent, '_target_relation_signals', wraps=agent._target_relation_signals) as signals:
            result = agent.run_research(request, retriever=lambda *a, **k: [])
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertEqual(signals.call_count, 1)
        self.assertEqual(len(seen_features), 2)
        self.assertIs(seen_features[0], seen_features[1])
        candidates = result.model_comparison['candidates']
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertIs(candidate['predictor_columns'], seen_features[0])
        review = result.dataset_audit['human_review']
        self.assertEqual(review['candidate_predictors'], ['x', 'post_result'])
        self.assertEqual(review['excluded_predictors'], ['post_result'])
        self.assertEqual(result.baseline['predictor_columns'], ['x'])
        self.assertTrue(any(s['column_name'] == 'post_result' and s['review_required']
                            for s in result.dataset_audit['column_review_signals']))
        jp, mp = agent.save_result(result, self.request(), self.root / 'out')
        self.assertEqual(json.loads(jp.read_text(encoding='utf-8'))['result'], result.to_dict())
        self.assertIn('evaluación final independiente', mp.read_text(encoding='utf-8'))

    def test_required_pending_blocks_before_preprocessing(self):
        self.data['predictor_availability']['post_result'] = 'available'
        with patch.object(agent, '_encode', side_effect=AssertionError('must not preprocess')):
            result = self.run_request()
        self.assertEqual(result.status, 'needs_input')
        self.assertIsNone(result.baseline)
        self.assertIn('señal requerida', ' '.join(result.errors))
        self.assertEqual(result.dataset_audit['human_review']['effective_predictors'], ['x', 'post_result'])

    def test_keep_requires_meaning_reason_and_actual_availability(self):
        response = {'column_name': 'post_result', 'decision': 'keep',
                    'reason': 'Relación prevista por el generador, medida antes del resultado',
                    'column_meaning': 'Medición binaria sintética inicial'}
        self.data['responses'] = [response]
        for availability in ['unknown', 'unavailable']:
            self.data['predictor_availability']['post_result'] = availability
            self.assertEqual(self.run_request().status, 'needs_input')
        self.data['predictor_availability']['post_result'] = 'available'
        for missing in ['reason', 'column_meaning']:
            altered = copy.deepcopy(response)
            altered.pop(missing)
            self.data['responses'] = [altered]
            self.assertEqual(self.run_request().status, 'needs_input')
        self.data['responses'] = [response]
        result = self.run_request()
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertIn('post_result', result.baseline['predictor_columns'])
        self.assertEqual(result.dataset_audit['human_review']['status'], 'resolved')

    def test_signal_id_stable_and_exclusion_by_signal_resolves_column(self):
        first = agent.run_audit_only(self.request())
        second = agent.run_audit_only(self.request())
        signals = first.dataset_audit['column_review_signals']
        self.assertEqual([s['signal_id'] for s in signals],
                         [s['signal_id'] for s in second.dataset_audit['column_review_signals']])
        signal = next(s for s in signals if s['column_name'] == 'post_result' and s['review_required'])
        self.data['responses'] = [{'column_name': 'post_result', 'signal_id': signal['signal_id'],
                                  'decision': 'exclude', 'reason': 'Medición posterior'}]
        result = self.run_request()
        self.assertEqual(result.status, 'completed', result.errors)
        resolved = [s for s in result.dataset_audit['human_review']['signal_resolutions'] if s['column_name'] == 'post_result']
        self.assertTrue(all(s['resolution'] == 'excluded' for s in resolved))

    def test_partition_column_is_kept_for_group_and_time_split(self):
        for strategy in ['group', 'time']:
            with self.subTest(strategy=strategy):
                self.path.write_text('x,entity,date,target\n' + ''.join(
                    f'{i % 7},g{i // 4},2024-01-{i + 1:02d},{"yes" if i % 2 else "no"}\n' for i in range(28)))
                self.data['configuration'].update(split_strategy=strategy,
                    observation_structure='grouped' if strategy == 'group' else 'temporal',
                    group_column='entity' if strategy == 'group' else None,
                    time_column='date' if strategy == 'time' else None)
                self.data['predictor_availability'] = {'x': 'available'}
                self.data['excluded_columns'] = {'entity': 'Solo particionar', 'date': 'Solo ordenar'}
                result = self.run_request()
                self.assertEqual(result.status, 'completed', result.errors)
                self.assertEqual(result.baseline['predictor_columns'], ['x'])
                self.assertEqual(result.baseline['split_strategy'], strategy)
                self.assertTrue(result.split_diagnostics['valid'])

    def test_excluding_every_predictor_blocks_with_explanation(self):
        self.data['excluded_columns'] = {'x': 'No usar', 'post_result': 'Posterior'}
        with patch.object(agent, '_encode', side_effect=AssertionError('preprocessing')):
            result = self.run_request()
        self.assertEqual(result.status, 'needs_input')
        self.assertIn('No quedan predictores utilizables', ' '.join(result.errors))

    def test_unknown_target_and_contradictory_decisions_rejected(self):
        for column in ['missing', 'target']:
            self.data['excluded_columns'] = {column: 'No usar'}
            result = self.run_request()
            self.assertEqual(result.status, 'failed')
        self.data.pop('excluded_columns')
        self.data['responses'] = [
            {'column_name': 'post_result', 'decision': 'keep', 'reason': 'Medida previa'},
            {'column_name': 'post_result', 'rule_id': 'target_relation.risk_name', 'decision': 'exclude', 'reason': 'Posterior'}]
        with self.assertRaisesRegex(ValueError, 'contradictorias'):
            self.request()

    def test_target_signal_scope_records_only_preliminary_train_rows(self):
        request = self.request()
        result = agent.run_audit_only(request)
        audit, rows = agent.audit_dataset(self.path, 'target')
        train, valid, split = agent._split_indices(request, rows, [r['target'] for r in rows])
        signals = [s for s in result.dataset_audit['column_review_signals'] if s['data_scope']['data_scope'] == 'train_only']
        self.assertTrue(signals)
        for signal in signals:
            scope = signal['data_scope']
            self.assertEqual(scope['training_row_indices'], train)
            self.assertFalse(set(scope['training_row_indices']) & set(valid))
            self.assertEqual(scope['split_strategy'], split['strategy'])
        changed = copy.deepcopy(rows)
        for index in valid:
            changed[index]['post_result'] = 'modified-validation-only'
            changed[index]['target'] = 'modified-validation-only'
        original = agent._target_relation_signals(rows, audit.column_names, 'target', agent.TASK, train, split['strategy'])
        after = agent._target_relation_signals(changed, audit.column_names, 'target', agent.TASK, train, split['strategy'])
        self.assertEqual(original, after)

    def test_informational_signals_neither_require_answers_nor_drop_columns(self):
        self.path.write_text('x,target\n' + ''.join(f'{i},{"yes" if i % 2 else "no"}\n' for i in range(60)))
        self.data['predictor_availability'] = {'x': 'available'}
        result = self.run_request()
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertTrue(result.dataset_audit['column_review_signals'])
        self.assertFalse(any(s['review_required'] for s in result.dataset_audit['column_review_signals']))
        self.assertEqual(result.baseline['predictor_columns'], ['x'])
        self.assertEqual(result.dataset_audit['human_review']['status'], 'not_required')

    def test_same_effective_predictors_across_cv_folds(self):
        self.data['configuration'].update(validation_strategy='cross_validation', cv_folds=3)
        self.data['excluded_columns'] = {'post_result': 'Posterior al evento'}
        with patch.object(agent, '_encode', wraps=agent._encode) as encode:
            result = self.run_request()
        self.assertEqual(result.status, 'completed', result.errors)
        self.assertEqual(encode.call_count, 6)
        for call in encode.call_args_list:
            self.assertIs(call.args[1], result.baseline['predictor_columns'])
            self.assertEqual(call.args[1], ['x'])

    def test_explicit_pending_signal_is_not_overridden_by_column_keep(self):
        result = agent.run_audit_only(self.request())
        signal = next(s for s in result.dataset_audit['column_review_signals'] if s['review_required'])
        self.data['predictor_availability']['post_result'] = 'available'
        self.data['responses'] = [
            {'column_name': 'post_result', 'decision': 'keep', 'reason': 'Entrada previa', 'column_meaning': 'Medición inicial'},
            {'column_name': 'post_result', 'signal_id': signal['signal_id'], 'decision': 'pending'}]
        result = self.run_request()
        self.assertEqual(result.status, 'needs_input')
        self.assertIn(signal['signal_id'], ' '.join(result.errors))


if __name__ == '__main__':
    unittest.main()
