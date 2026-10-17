"""Complete only the documented synthetic simulator demo, not a human approval."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_HASH = '43d55690830f8958216117a919f400b12bfa5e6b94e03557ce38de641640ea50'
MEANINGS = {
    'horas_promocion': 'Duración planificada; rng.randint(3,18), antes de calcular ventas.',
    'experiencia_equipo': 'Años simulados conocidos al planificar; rng.randint(0,12). El 6% de ocultación no depende del target.',
    'presupuesto_estimado': 'Presupuesto previsto: base 80..400 e incremento excepcional independiente, ambos antes de ventas. El 4% de ocultación es aleatorio.',
    'canal': 'Medio planificado email/redes/tienda, elegido antes de ventas.',
}
REASONS = {
    'horas_promocion': 'Entrada del simulador y duración disponible al planificar, no derivada de ventas.',
    'experiencia_equipo': 'Entrada previa del simulador; disponibilidad de la variable no significa ausencia de celdas faltantes. Se permite imputación ajustada solo en train.',
    'presupuesto_estimado': 'El generador crea presupuestos excepcionales antes del objetivo, no son errores ni fuga confirmada. Su cardinalidad refleja importes numéricos legítimos. Faltantes aleatorios imputados solo con train.',
    'canal': 'Decisión de canal previa a la predicción; su efecto entra después en la fórmula del objetivo.',
}


def complete(data):
    """A narrowly scoped demo fixture, with no automatic general-purpose approval."""
    path = Path(data['configuration']['csv_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_HASH or data['dataset_sha256'] != EXPECTED_HASH:
        raise ValueError('Este completador solo admite el CSV original del simulador; vuelva a revisar la procedencia.')
    for key, value in {'target': 'ventas', 'task_type': 'regression', 'metric': 'rmse',
                       'seed': 42, 'validation_strategy': 'cross_validation',
                       'split_strategy': 'random', 'observation_structure': 'independent', 'cv_folds': 5}.items():
        if data['configuration'].get(key) != value:
            raise ValueError(f'Configuración de demostración esperada: {key}={value!r}')
    data.update(positive_class=None,
        row_meaning='CONFIGURACIÓN DE DEMOSTRACIÓN fundamentada en scripts/generate_demo_ventas.py y DATASET_DEMO.md; no es aprobación humana. Una acción promocional sintética independiente.',
        prediction_scenario='Planificación de una acción simulada, antes de calcular ventas; se observan los predictores exportados, con las celdas ocultas como faltantes.',
        metric_reason='Objetivo de esta demostración: penalizar más los errores grandes en unidades vendidas simuladas; RMSE conserva esas unidades. Es una elección ilustrativa, no un coste empresarial validado ni una exigencia del generador.',
        predictor_availability={c: 'available' for c in MEANINGS}, excluded_columns={})
    addressed = set()
    for response in data['responses']:
        column = response['column_name']
        if column == 'presupuesto_estimado' and response.get('rule_id') in {'outlier.tukey_1_5_iqr', 'category.high_cardinality'}:
            response.update(decision='keep', reason=REASONS[column], column_meaning=MEANINGS[column])
            addressed.add(column)
        else:
            response.update(decision='pending', reason=None, column_meaning=None)
            addressed.add(column)
    for column in MEANINGS:
        if column in addressed:
            continue
        data['responses'].append({'column_name': column, 'decision': 'keep',
                                  'reason': REASONS[column], 'column_meaning': MEANINGS[column]})
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pending', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = complete(json.loads(args.pending.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    print(f'Configuración de demostración del simulador, NO aprobación humana: {args.output}')
