"""Reproducible 2D CLI demonstration with explicitly synthetic review answers.

This fixture simulates a review; it never confirms decisions for real user data.
Run from the repository: python examples/demo_decision_workflow.py
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
destination = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / '.veronica' / 'demo_2d'
destination.mkdir(parents=True, exist_ok=False)
csv_path = destination / 'synthetic.csv'
csv_path.write_text('x,post_result,target\n' + ''.join(
    f'{i % 7},{"yes" if i % 2 else "no"},{"yes" if i % 2 else "no"}\n' for i in range(60)), encoding='utf-8')
pending = destination / 'pending.json'
reviewed = destination / 'reviewed.synthetic.json'
records = []


def run(label, arguments, expected):
    command = [sys.executable, str(ROOT / 'research_agent.py'), *arguments,
               '--output-dir', str(destination / label)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (destination / f'{label}.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
    payload = json.loads((destination / label / 'research_result.json').read_text(encoding='utf-8'))
    result = payload['result']
    records.append({'step': label, 'command': command, 'exit_code': completed.returncode,
                    'status': result['status'], 'has_baseline': result['baseline'] is not None})
    if completed.returncode != expected:
        raise RuntimeError(f'{label}: expected {expected}, got {completed.returncode}; inspect log')
    print(json.dumps(records[-1], ensure_ascii=False))


run('01_audit', ['--csv', str(csv_path), '--target', 'target', '--task', 'binary_classification',
    '--metric', 'f1', '--observation-structure', 'independent', '--split-strategy', 'random',
    '--write-decisions-template', str(pending)], 0)
run('02_pending', ['--decisions-json', str(pending)], 2)

data = json.loads(pending.read_text(encoding='utf-8'))
data.update(positive_class='yes', row_meaning='Una observación del generador sintético de esta demo',
            prediction_scenario='Antes de calcular las etiquetas del generador',
            metric_reason='Fixture: balance de precision y recall para yes',
            predictor_availability={'x': 'available', 'post_result': 'unavailable'},
            excluded_columns={'post_result': 'Copia sintética del target, calculada después del resultado'})
for response in data['responses']:
    if response['column_name'] == 'post_result':
        response.update(decision='exclude', reason='Copia de la etiqueta; no existe al predecir')
    else:
        response.update(decision='keep', reason='Entrada inicial explícita del generador',
                        column_meaning='Residuo del índice sintético módulo siete')
reviewed.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
run('03_train', ['--decisions-json', str(reviewed)], 0)
csv_path.write_text(csv_path.read_text(encoding='utf-8') + '4,yes,yes\n', encoding='utf-8')
run('04_stale', ['--decisions-json', str(reviewed)], 2)
(destination / 'execution_summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Synthetic review fixture only. Evidence: {destination}')
