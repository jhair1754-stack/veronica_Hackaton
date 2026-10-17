"""Version 1 decision input: strict syntax, explicit pending human decisions."""
from __future__ import annotations

import json
import re
from pathlib import Path


CONFIG_TYPES = {
    **dict.fromkeys(('csv_path', 'target', 'task_type', 'metric', 'context',
                     'metric_context', 'target_unit', 'split_strategy', 'group_column',
                     'time_column', 'time_format', 'validation_strategy',
                     'observation_structure'), str),
    **dict.fromkeys(('seed', 'cv_folds', 'max_cv_fits'), int),
    'test_size': (int, float),
}
ENUMS = {
    'task_type': {'binary_classification', 'regression'},
    'observation_structure': {'independent', 'grouped', 'temporal', 'unknown'},
    'split_strategy': {'random', 'group', 'time'},
    'validation_strategy': {'holdout', 'cross_validation', 'auto'},
    'metric': {'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'mae', 'rmse', 'r2'},
}


def _object(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(f'{path}: se requiere un objeto JSON')
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'{path}: campos desconocidos {sorted(unknown)}')


def _text(value, path):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f'{path}: se requiere texto no vacío o null')


def validate_contract(data):
    _object(data, {'schema_version', 'configuration', 'dataset_sha256', 'positive_class',
                  'row_meaning', 'prediction_scenario', 'metric_reason',
                  'predictor_availability', 'excluded_columns', 'responses', 'review_binding'}, 'decisions')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('schema_version: versión no admitida; se requiere 1')
    if 'review_binding' in data:
        binding = data['review_binding']
        _object(binding, {'dataset_sha256', 'configuration_sha256', 'signals_sha256'}, 'review_binding')
        for key in ('dataset_sha256', 'configuration_sha256', 'signals_sha256'):
            value = binding.get(key)
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError(f'review_binding.{key}: SHA-256 obligatorio')
    config = data.get('configuration', {})
    _object(config, CONFIG_TYPES.keys(), 'configuration')
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, CONFIG_TYPES[key]):
            raise ValueError(f'configuration.{key}: tipo incorrecto')
        if isinstance(value, str):
            _text(value, key)
        if key in ENUMS and value not in ENUMS[key]:
            raise ValueError(f'configuration.{key}: valor desconocido')
    if config.get('test_size') is not None and not 0 < config['test_size'] < 1:
        raise ValueError('test_size debe estar entre 0 y 1')
    for key, minimum in [('cv_folds', 2), ('max_cv_fits', 1)]:
        if config.get(key) is not None and config[key] < minimum:
            raise ValueError(f'{key}: debe ser >= {minimum}')
    for key in ('dataset_sha256', 'positive_class', 'row_meaning', 'prediction_scenario', 'metric_reason'):
        _text(data.get(key), key)
    digest = data.get('dataset_sha256')
    if digest is not None and not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('dataset_sha256: se requieren 64 caracteres hexadecimales minúsculos')
    for key in ('predictor_availability', 'excluded_columns'):
        entries = data.get(key, {})
        _object(entries, entries.keys() if isinstance(entries, dict) else set(), key)
        for column, value in entries.items():
            _text(column, key)
            if key == 'predictor_availability':
                if not isinstance(value, str) or value not in {'available', 'unavailable', 'unknown'}:
                    raise ValueError(f'{key}.{column}: disponibilidad inválida')
            else:
                _text(value, f'{key}.{column}')
    responses = data.get('responses', [])
    if not isinstance(responses, list):
        raise ValueError('responses: se requiere una lista')
    seen = set()
    for response in responses:
        _object(response, {'column_name', 'rule_id', 'signal_id', 'decision', 'reason', 'column_meaning'}, 'responses[]')
        for key in ('column_name', 'rule_id', 'signal_id', 'reason', 'column_meaning'):
            _text(response.get(key), key)
        if not response.get('column_name'):
            raise ValueError('responses: falta column_name')
        if not isinstance(response.get('decision'), str) or response['decision'] not in {'keep', 'exclude', 'pending'}:
            raise ValueError('responses: decision debe ser keep, exclude o pending')
        identity = (response['column_name'], response.get('signal_id') or response.get('rule_id'))
        if identity in seen:
            raise ValueError('responses: respuesta duplicada para columna/señal')
        seen.add(identity)
        if response['decision'] == 'keep' and response['column_name'] in data.get('excluded_columns', {}):
            raise ValueError('responses: conservar contradice excluded_columns')
    for column in {r['column_name'] for r in responses}:
        if {'keep', 'exclude'} <= {r['decision'] for r in responses if r['column_name'] == column}:
            raise ValueError(f'responses: decisiones contradictorias para {column}')
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Clave JSON duplicada: {key}')
        result[key] = value
    return result


def load_contract(path):
    def invalid_constant(value):
        raise ValueError(f'Constante JSON inválida: {value}')
    return validate_contract(json.loads(Path(path).read_text(encoding='utf-8-sig'),
                                        object_pairs_hook=_unique_object,
                                        parse_constant=invalid_constant))
