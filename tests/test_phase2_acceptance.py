"""Simulator provenance and no-fit acceptance checks for phase 2E."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import research_agent as agent
from scripts.complete_demo_decisions import complete, EXPECTED_HASH
from scripts.generate_demo_ventas import generate

ROOT = Path(agent.__file__).parent


class Phase2AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'examples/decisions.demo_ventas.json').read_text(encoding='utf-8'))

    def request(self, data):
        return agent.merge_decisions(agent.ResearchRequest(None, None), data, set())

    def test_generator_reproduces_original_bytes_without_overwriting_original(self):
        original = (ROOT / 'demo_ventas_sinteticas.csv').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), EXPECTED_HASH)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'regenerated.csv'
            generate(path)
            self.assertEqual(path.read_bytes(), original)

    def test_real_demo_audit_and_pending_request_never_retrieve_encode_or_fit(self):
        pending = copy.deepcopy(self.data)
        pending['row_meaning'] = None
        pending['metric_reason'] = None
        with patch.object(agent.rag_mvp, 'search', side_effect=AssertionError('retrieval')), \
             patch.object(agent, '_encode', side_effect=AssertionError('preprocess')), \
             patch.object(agent, '_candidate_prediction', side_effect=AssertionError('fit')):
            audit = agent.run_audit_only(self.request(pending))
            result = agent.run_research(self.request(pending))
        self.assertEqual(audit.status, 'completed')
        self.assertFalse(audit.dataset_audit['score_computed'])
        self.assertEqual(result.status, 'needs_input')
        self.assertIsNone(result.baseline)
        self.assertIsNone(result.model_comparison)

    def test_demo_completion_is_not_a_general_approval_tool(self):
        completed = complete(copy.deepcopy(self.data))
        self.assertIn('no es aprobación humana', completed['row_meaning'])
        self.assertIn('no un coste empresarial validado', completed['metric_reason'])
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / 'other.csv'
            other.write_text('x,target\n1,2\n', encoding='utf-8')
            changed = copy.deepcopy(self.data)
            changed['configuration']['csv_path'] = str(other)
            with self.assertRaisesRegex(ValueError, 'solo admite'):
                complete(changed)

    def test_new_unjustified_signal_remains_pending(self):
        unknown = {'column_name': 'presupuesto_estimado', 'rule_id': 'new.unreviewed.rule',
                   'signal_id': 'new-signal', 'decision': 'pending', 'reason': None}
        self.data['responses'].append(unknown)
        result = complete(self.data)
        preserved = next(r for r in result['responses'] if r.get('rule_id') == 'new.unreviewed.rule')
        self.assertEqual(preserved['decision'], 'pending')
        self.assertIsNone(preserved['reason'])


if __name__ == '__main__':
    unittest.main()
