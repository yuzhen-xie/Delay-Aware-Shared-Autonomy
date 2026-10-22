from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from dasa.config import manifest, write_json


_UNITS = {
    'task.control_dt': 's', 'task.decision_dt': 's',
    'task.force_limit': 'N', 'task.robot_radius': 'm',
    'task.workspace_min': 'm', 'task.workspace_max': 'm', 'task.max_seconds': 's',
    'task.grasp_tolerance': 'm', 'task.place_tolerance': 'm', 'task.dwell_seconds': 's',
    'task.max_grasp_speed': 'm/s', 'task.lift_height': 'm',
    'task.feedback_delay': 's, one-way feedback', 'task.command_delay': 's, one-way command',
    'scene.goals': 'm', 'scene.start': 'm', 'scene.place': 'm',
    'scene.object_size_m': 'm',
    'scene.novel_translation': 'm, Euclidean xy displacement', 'scene.added_obstacle_radius': 'm',
    'motion.attractive_gain': 'N/m', 'motion.repulsive_gain': 'N m',
    'motion.repulsive_sigma': 'm', 'motion.repulsive_limit': 'N per obstacle',
    'motion.attractive_weight': '1', 'motion.repulsive_weight': '1',
    'motion.distance_min': 'm, configured risk-distance convention',
    'motion.distance_max': 'm, configured risk-distance convention',
    'motion.lambda_gain': '1', 'motion.kalman_process': 'N^2/sample',
    'motion.kalman_measurement': 'N^2', 'motion.kalman_initial_variance': 'N^2',
    'arbitration.stiffness': 'N/m', 'arbitration.damping': 'N s/m',
    'arbitration.covariance_lower': 'm^2', 'arbitration.covariance_upper': 'm^2',
    'arbitration.roughness_gamma': 's/N', 'arbitration.roughness_window': 'samples',
    'arbitration.epsilon': '1', 'arbitration.rho_window_seconds': 's',
    'arbitration.low_confidence_threshold': '1', 'arbitration.correction_distance': 'm',
    'arbitration.fixed_weight': '1', 'arbitration.guarded_corrective_max': '1',
    'intent.beta': 'rad^-1', 'intent.confidence_threshold': '1', 'intent.sustained_seconds': 's',
    'intent.memory_decay': '1', 'intent.learning_rate': '1/update',
    'intent.kl_weight': '1', 'intent.deviation_weight': '1',
    'skill.covariance_floor': 'm^2', 'skill.duration_floor': 'resampled samples',
    'skill.lqt_control_cost': 'relative cost', 'skill.original_weight': '1 per episode',
    'skill.new_weight': '1 per episode',
    'skill.prediction_horizon_samples': 'HSMM reference-grid samples',
    'experiments.intent_delays': 's, one-way feedback',
    'experiments.sharing_delays': 's, one-way feedback', 'study.max_seconds': 's',
    'study.hardware_scene_tolerance_m': 'm',
    'analysis.metrics.resample_dt_seconds': 's',
    'analysis.metrics.smoothing_window_seconds': 's',
    'analysis.metrics.gamma_st_s3_per_m': 's^3/m',
    'analysis.metrics.gamma_sa_s2_per_m': 's^2/m',
}
_MEASURED = {'UR10_measured_teleoperation', 'UR10_measured_autonomous',
             'UR10_workstation_measurement', 'actual_participant_measurement',
             'measured_workstation_acquisition', 'physical_robot_measurement'}


def flatten(value: dict, prefix=''):
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            yield from flatten(item, name)
        else:
            yield name, item


def parameter_rows(config: dict):
    rows = []
    for path, value in flatten(config):
        unit = _UNITS.get(path)
        if path in {'motion.distance_min', 'motion.distance_max'}:
            unit = 'm, obstacle center distance' if config['motion']['risk_distance'] == 'center' \
                else 'm, obstacle and TCP surface clearance'
        if unit is None:
            suffix = path.rsplit('.', 1)[-1]
            unit = ('J' if suffix.endswith('_j') else 's' if suffix.endswith('_seconds')
                    else 'm/s' if suffix.endswith('_m_s') else 'count' if type(value) is int else '1')
        rows.append({'parameter': path, 'value': value, 'unit': unit,
                     'selection': config.get('analysis', {}).get(
                         'parameter_provenance', 'configured_operational_values')})
    return rows


def export_protocol(config: dict, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = parameter_rows(config)
    software = manifest(config, 'protocol')
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    config_digest = hashlib.sha256(encoded).hexdigest()
    study = config['study']
    protocol_path = output / 'study_protocol.yaml'
    if protocol_path.exists():
        protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
        if protocol.get('config_sha256') != config_digest:
            raise ValueError('A frozen protocol cannot be overwritten by another configuration')
    else:
        protocol = {
            'schema_version': 3, 'record_status': 'prospective',
            'study_id': study.get('study_id', 'dasa-study-v2'),
            'analysis_plan_frozen_on': datetime.now(timezone.utc).isoformat(),
            'config_sha256': config_digest, 'design': 'within_subjects',
            'source_sha256': software['source_sha256'],
            'source_git_commit': software['git_commit'],
            'expected_participants': study['participants'],
            'conditions': config['experiments']['conditions'],
            'feedback_delays_seconds': config['experiments']['sharing_delays'],
            'command_delay_seconds': config['task']['command_delay'],
            'tlx_delays_seconds': [0.8], 'workload_figure_delay_seconds': 0.8,
            'trial_timeout_seconds': study['max_seconds'], 'timing_tolerance_seconds': .02,
            'practice_trials': study['practice_trials'],
            'practice_trial_timeout_seconds': study['max_seconds'],
            'practice_sequence': study.get('practice_sequence',
                [['C1', 0.]] * 4 + [['C3', 0.]] * 4),
            'randomization_seed': config['seed'], 'trial_repeats': study['trial_repeats'],
            'data_provenance': 'actual_participant_measurement',
            'scoring': {'success': 'correct_grasp_lift_and_place_without_collision_before_timeout',
                        'completion_seconds': 'successful_trials_only',
                        'safety_index': 'duration_weighted_exp_negative_gamma_speed_squared_over_clearance',
                        'stability_index': 'duration_weighted_exp_negative_absolute_gamma_second_speed_derivative'},
            'exclusion_rules': {
                'excluded': ['practice', 'documented_device_fault', 'documented_corrupt_record',
                             'participant_not_included'],
                'retained_as_failures': ['collision', 'bounds_violation', 'wrong_goal', 'timeout']},
            'primary_outcomes': ['success_rate', 'completion_seconds'],
            'secondary_outcomes': ['safety_index', 'stability_index', 'minimum_clearance_m',
                                   'jerk_rms', 'timeout_penalized_seconds'],
            'unit_of_analysis': 'participant', 'test': 'paired_two_sided_sign_permutation',
            'multiple_comparison_correction': 'Holm',
            'confidence_intervals': 'paired_participant_bootstrap_95_percent_pointwise',
            'task_family': 'all_conditions_delays_and_task_metrics',
            'workload_family': 'RTLX_and_six_scales_all_conditions_and_declared_TLX_delays',
            'missingness': 'failures_retained_and_success_conditioned_completion_reported',
            'questionnaire': config['analysis']['questionnaire'],
        }
        protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding='utf-8')
    write_json(output / 'parameters.json', rows)
    (output / 'resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    with (output / 'parameters.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, 'value': json.dumps(r['value'])} for r in rows)
    write_json(output / 'provenance.json', software)


def audit_run(config: dict, root: Path):
    root = Path(root)
    records, malformed, unaccepted = [], [], []
    for path in sorted(root.rglob('*.npz')):
        if 'audit' in path.relative_to(root).parts:
            continue
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if 'metadata' not in arrays:
                    continue
                metadata = json.loads(str(arrays['metadata']))
            if 'scene' not in metadata:
                continue
            record = {'path': str(path.relative_to(root)),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'trial_id': metadata.get('trial_id'), 'episode_id': metadata.get('episode_id'),
                      'unit': metadata.get('operator_id'),
                      'goal': metadata.get('intended_goal', metadata.get('goal')),
                      'condition': metadata.get('condition', metadata.get('source_condition')),
                      'provenance': metadata.get('provenance'), 'scene': metadata['scene']}
            if metadata.get('provenance') not in _MEASURED:
                unaccepted.append(record)
                continue
            records.append(record)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            malformed.append({'path': str(path.relative_to(root)), 'error': str(error)})
    for path in sorted(root.rglob('recording.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('provenance') != 'UR10_workstation_measurement' or data.get('execute') is not True:
                unaccepted.append({'path': str(path.relative_to(root)),
                                   'reason': 'not_executed_measured_acquisition'})
                continue
            records.append({'path': str(path.relative_to(root)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'n_samples': len(data.get('samples', [])),
                            'provenance': data['provenance'],
                            'assignment': data.get('experiment_assignment', {}),
                            'timing': data.get('timing', {})})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            malformed.append({'path': str(path.relative_to(root)), 'error': str(error)})
    write_json(root / 'audit/actual_scene_records.json', {
        'schema_version': 2, 'records': records, 'malformed_records': malformed,
        'unaccepted_records': unaccepted,
        'interpretation': 'file inventory; acquisitions and imported copies are not independent units'})
    export_protocol(config, root / 'audit')
    write_json(root / 'audit/run_inventory.json', manifest(
        config, 'audit-run', [root / record['path'] for record in records]))
    return {'record_count': len(records), 'malformed_count': len(malformed),
            'unaccepted_count': len(unaccepted)}
