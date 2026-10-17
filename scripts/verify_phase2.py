"""Run and assert the eight phase-2 acceptance cases; preserve every CLI artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    commands, results = [], {}

    def command(label, arguments, expected=0, agent=False):
        args = [sys.executable, '-B', *arguments]
        if agent:
            args += ['--output-dir', str(output / label)]
        process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        (output / f'{label}.log').write_text(process.stdout + process.stderr, encoding='utf-8')
        commands.append({'label': label, 'command': args, 'expected_exit': expected, 'exit_code': process.returncode})
        (output / 'commands.json').write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding='utf-8')
        if process.returncode != expected:
            raise AssertionError(f'{label}: expected {expected}, got {process.returncode}; inspect log')
        if not agent:
            return
        result = json.loads((output / label / 'research_result.json').read_text(encoding='utf-8'))['result']
        markdown = (output / label / 'research_report.md').read_text(encoding='utf-8')
        extracted = json.loads(markdown.split('```json\n', 1)[1].split('\n```', 1)[0])
        assert extracted == result['review_summary'], label
        if expected == 2 or '--write-decisions-template' in arguments:
            assert result['baseline'] is None and result['model_comparison'] is None, label
        results[label] = result
        return result

    generated = output / 'regenerated.csv'
    command('provenance', ['scripts/generate_demo_ventas.py', '--output', str(generated)])
    original = ROOT / 'demo_ventas_sinteticas.csv'
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert generated.read_bytes() == original.read_bytes(), 'CSV differs from generator; do not silently replace it'
    command('rag_build', ['rag_mvp.py', 'build'])
    pending, reviewed = output / 'decisions.pending.json', output / 'decisions.simulator.json'
    common = ['research_agent.py', '--csv', 'demo_ventas_sinteticas.csv', '--target', 'ventas',
        '--task', 'regression', '--metric', 'rmse', '--metric-context', 'large_errors_cost_more',
        '--target-unit', 'unidades_simuladas', '--split-strategy', 'random', '--validation-strategy',
        'cross_validation', '--observation-structure', 'independent', '--cv-folds', '5', '--seed', '42']
    command('01_audit', common + ['--write-decisions-template', str(pending)], agent=True)
    command('02_pending', ['research_agent.py', '--decisions-json', str(pending)], 2, True)
    command('complete_simulator_fixture', ['scripts/complete_demo_decisions.py', str(pending), str(reviewed)])
    valid = command('03_complete', ['research_agent.py', '--decisions-json', str(reviewed)], agent=True)
    assert valid['status'] == 'completed'
    checked_in = command('03_checked_in_example', ['research_agent.py', '--decisions-json',
                         'examples/decisions.demo_ventas.json'], agent=True)
    assert checked_in['baseline']['score'] == valid['baseline']['score']
    exclusion = json.loads(reviewed.read_text(encoding='utf-8'))
    exclusion['excluded_columns'] = {'presupuesto_estimado': 'Ablación de demostración para verificar exclusión; no es una fuga confirmada.'}
    for response in exclusion['responses']:
        if response['column_name'] == 'presupuesto_estimado':
            response.update(decision='exclude', reason=exclusion['excluded_columns']['presupuesto_estimado'])
    exclusion_path = output / 'decisions.without_budget.json'
    exclusion_path.write_text(json.dumps(exclusion, ensure_ascii=False, indent=2), encoding='utf-8')
    excluded = command('04_exclusion', ['research_agent.py', '--decisions-json', str(exclusion_path)], agent=True)
    assert 'presupuesto_estimado' not in excluded['baseline']['predictor_columns']
    assert 'presupuesto_estimado' not in excluded['baseline']['encoded_columns']
    assert all(c['predictor_columns'] == excluded['baseline']['predictor_columns'] for c in excluded['model_comparison']['candidates'])
    # Fixture: x exists first, target is deterministically derived from x. Positive is not alphabetically last.
    binary_csv = output / 'binary.csv'
    binary_csv.write_text('x,target\n' + ''.join(f'{i % 2},{"accept" if i % 2 else "reject"}\n' for i in range(80)), encoding='utf-8')
    binary_pending = output / 'binary.pending.json'
    command('05_binary_audit', ['research_agent.py', '--csv', str(binary_csv), '--target', 'target',
        '--task', 'binary_classification', '--metric', 'f1', '--split-strategy', 'random',
        '--observation-structure', 'independent', '--write-decisions-template', str(binary_pending)], agent=True)
    binary = json.loads(binary_pending.read_text(encoding='utf-8'))
    binary.update(positive_class='accept', row_meaning='Fixture sintético: una entrada binaria independiente, no aprobación humana',
        prediction_scenario='x existe antes de derivar la etiqueta en el fixture', metric_reason='F1 de accept para verificar la orientación de clase',
        predictor_availability={'x': 'available'})
    for response in binary['responses']:
        response.update(decision='keep', reason='La etiqueta se deriva de la entrada x del fixture, no al contrario',
                        column_meaning='Entrada binaria i módulo 2 de este fixture')
    binary_reviewed = output / 'binary.simulator.json'
    binary_reviewed.write_text(json.dumps(binary, ensure_ascii=False, indent=2), encoding='utf-8')
    classified = command('05_binary', ['research_agent.py', '--decisions-json', str(binary_reviewed)], agent=True)
    assert classified['baseline']['class_mapping'] == {'reject': 0, 'accept': 1}
    other = output / 'other.csv'
    other.write_bytes(original.read_bytes() + b'10,5,200,email,60\n')
    mismatch = json.loads(reviewed.read_text(encoding='utf-8'))
    mismatch['configuration']['csv_path'] = str(other)
    mismatch_path = output / 'decisions.wrong_csv.json'
    mismatch_path.write_text(json.dumps(mismatch, ensure_ascii=False, indent=2), encoding='utf-8')
    rejected = command('06_wrong_csv', ['research_agent.py', '--decisions-json', str(mismatch_path)], 2, True)
    assert rejected['decision_contract']['binding_status'] == 'stale'
    repeated = command('07_repeat', ['research_agent.py', '--decisions-json', str(reviewed)], agent=True)
    assert valid['model_comparison']['partitions'] == repeated['model_comparison']['partitions']
    assert valid['baseline']['score'] == repeated['baseline']['score']
    for one, two in zip(valid['model_comparison']['candidates'], repeated['model_comparison']['candidates']):
        assert one['summary'] == two['summary']
        assert [f['score'] for f in one['folds']] == [f['score'] for f in two['folds']]
    summary = {'python': sys.version, 'interpreter': sys.executable, 'numpy': np.__version__,
        'dataset_sha256': digest, 'generator_reproduces_bytes': True,
        'configuration': json.loads(reviewed.read_text(encoding='utf-8'))['configuration'],
        'criteria': {str(i): 'cumplido' for i in range(1, 9)},
        'regression': {'linear_rmse': valid['baseline']['score'],
                       'candidates': [{k: c[k] for k in ('id', 'summary', 'predictor_columns')} for c in valid['model_comparison']['candidates']]},
        'without_budget_rmse': excluded['baseline']['score'], 'binary_f1': classified['baseline']['score'],
        'binary_mapping': classified['baseline']['class_mapping'], 'repeat_identical_partitions_and_metrics': True,
        'consistent_reports': list(results), 'independent_final_evaluation': False}
    (output / 'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.veronica' / 'phase2_acceptance')
    verify(parser.parse_args().output_dir.resolve())
