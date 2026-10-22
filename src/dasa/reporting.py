from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasa-matplotlib')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dasa.config import write_json
from dasa.geometry import Scene


_MEASURED = {'UR10_measured_teleoperation', 'UR10_measured_autonomous',
             'actual_participant_measurement', 'physical_robot_measurement',
             'measured_workstation_acquisition'}
_COLORS = {'C1': '#637381', 'C2': '#d69b40', 'C3': '#2478a8', 'C4': '#459b79'}


def load_rows(path: Path):
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('A measured metrics file must contain a row list')
    if any(row.get('provenance') not in _MEASURED for row in rows):
        raise ValueError('Reports require explicitly measured observation provenance')
    return rows


def summary(rows, metrics):
    result = []
    cells = sorted({(r['condition'], r.get('delay_seconds', 0.), r.get('scenario', ''))
                    for r in rows})
    for method, delay, scenario in cells:
        selected = [r for r in rows if (r['condition'], r.get('delay_seconds', 0.),
                                       r.get('scenario', '')) == (method, delay, scenario)]
        item = {'condition': method, 'delay_seconds': delay, 'scenario': scenario,
                'n_trials': len(selected), 'n_units': len({r['unit'] for r in selected})}
        for metric in metrics:
            units = {}
            for row in selected:
                value = row.get(metric)
                if value is not None and np.isfinite(float(value)):
                    units.setdefault(str(row['unit']), []).append(float(value))
            means = np.asarray([np.mean(values) for values in units.values()])
            item[metric] = float(means.mean()) if len(means) else None
            item[f'{metric}_n_units'] = len(means)
        result.append(item)
    return result


def _save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _sphere(ax, center, radius):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 16))
    ax.plot_surface(center[0] + radius * np.cos(u) * np.sin(v),
                    center[1] + radius * np.sin(u) * np.sin(v),
                    center[2] + radius * np.cos(v), color='#7e8a98', alpha=.25, linewidth=0)


def _recording(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('provenance') != 'UR10_workstation_measurement' or data.get('execute') is not True:
        raise ValueError('An executed physical workstation recording is required')
    samples = data.get('samples', [])
    if len(samples) < 3 or not any(row.get('executed') is True for row in samples):
        raise ValueError('No executed physical trajectory is available')
    time = np.asarray([row['time'] for row in samples], float)
    position = np.asarray([row['position'] for row in samples], float)
    if (not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or position.shape != (len(time), 3) or not np.isfinite(position).all()):
        raise ValueError('Recorded trajectory coordinates and times are invalid')
    return data, samples, time, position, Scene.from_dict(data['scene'])


def _decisions(data, samples):
    commands = data.get('commands', [])
    if len(commands) == len(samples):
        return [value if isinstance(value, dict) else {} for value in commands]
    return [row.get('decision', {}) for row in samples]


def recording_figures(recording_path: Path, output: Path):
    data, samples, time, position, scene = _recording(recording_path)
    decisions = _decisions(data, samples)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    for center, radius, identifier in zip(scene.centers, scene.radii, scene.obstacle_ids):
        _sphere(ax, center, radius)
        ax.text(*center, identifier, fontsize=8)
    ax.scatter(*scene.goals.T, c='#d69b40', s=60, label='Candidate objects')
    ax.scatter(*scene.place, c='#459b79', marker='s', s=65, label='Placement zone')
    ax.plot(*position.T, c='#2478a8', linewidth=2, label='Measured TCP trajectory')
    reference = [(index, decision.get('reference')) for index, decision in enumerate(decisions)
                 if decision.get('reference') is not None]
    if reference:
        coordinates = np.asarray([item[1] for item in reference], float)
        if coordinates.shape != (len(reference), 3) or not np.isfinite(coordinates).all():
            raise ValueError('Recorded reference coordinates are invalid')
        ax.plot(*coordinates.T, c='#b0536d', linewidth=1.2, label='Recorded controller reference')
    ax.set(xlabel='Base x (m)', ylabel='Base y (m)', zlabel='Base z (m)',
           title=f"Measured physical trajectory | {data.get('condition', '')}")
    ax.legend(fontsize=8)
    _save(fig, output / 'figure7_trajectory')
    files = ['figure7_trajectory.png', 'figure7_trajectory.pdf', 'figure7_trajectory.svg']
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    channels = [('alpha', 'Autonomy weight'), ('lambda', 'Local guidance weight'),
                ('autonomy_confidence', 'Autonomy confidence'), ('human_confidence', 'Human confidence')]
    drawn = 0
    for key, label in channels:
        points = [(time[index], decision[key]) for index, decision in enumerate(decisions)
                  if decision.get(key) is not None]
        if points:
            values = np.asarray(points, float)
            if not np.isfinite(values).all():
                raise ValueError('Recorded arbitration coefficients must be finite')
            ax.plot(values[:, 0], values[:, 1], label=label)
            drawn += 1
    if drawn:
        ax.set(xlabel='Time (s)', ylabel='Coefficient', ylim=(-.03, 1.03))
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=.2)
        _save(fig, output / 'figure7_authority')
        files.extend(f'figure7_authority.{suffix}' for suffix in ('png', 'pdf', 'svg'))
    else:
        plt.close(fig)
    robot_radius = float(data.get('execution_contract', {}).get('robot_radius_m', 0.))
    if robot_radius <= 0:
        raise ValueError('The measured record must declare its collision-envelope radius')
    clearance = np.asarray([scene.clearance(p, robot_radius) for p in position])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout='constrained', sharex=True)
    time_edges = np.r_[time[0] - (time[1] - time[0]) / 2,
                       (time[1:] + time[:-1]) / 2,
                       time[-1] + (time[-1] - time[-2]) / 2]
    image = axes[0].pcolormesh(time_edges, np.arange(len(scene.centers) + 1), clearance.T,
                              shading='flat', cmap='viridis')
    axes[0].set(yticks=np.arange(len(scene.centers)) + .5,
                yticklabels=scene.obstacle_ids, ylabel='Obstacle')
    fig.colorbar(image, ax=axes[0], label='TCP surface clearance (m)')
    axes[1].plot(time, clearance.min(axis=1), c='#2478a8', label='Minimum surface clearance')
    axes[1].axhline(0., c='#b0536d', linestyle='--', label='Contact boundary')
    axes[1].set(xlabel='Time (s)', ylabel='Clearance (m)')
    axes[1].legend(frameon=False)
    _save(fig, output / 'figure7_clearance')
    files.extend(f'figure7_clearance.{suffix}' for suffix in ('png', 'pdf', 'svg'))
    energy_channels = {}
    for index, sample in enumerate(samples):
        diagnostics = sample.get('passivity', {})
        if not isinstance(diagnostics, dict):
            continue
        for key, value in diagnostics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                if key.endswith('_j') or key in {'rho', 'energy', 'storage'}:
                    energy_channels.setdefault(key, []).append((time[index], value))
    if energy_channels:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout='constrained', sharex=True)
        for key, points in sorted(energy_channels.items()):
            values = np.asarray(points)
            ax = axes[1] if key == 'rho' else axes[0]
            ax.plot(values[:, 0], values[:, 1], label=key)
        axes[0].set(ylabel='Energy (J)')
        axes[1].set(xlabel='Time (s)', ylabel='Relative removed work')
        for ax in axes:
            if ax.lines:
                ax.legend(frameon=False, fontsize=7, ncol=2)
            ax.grid(alpha=.2)
        _save(fig, output / 'measured_port_energy')
        files.extend(f'measured_port_energy.{suffix}' for suffix in ('png', 'pdf', 'svg'))
    write_json(output / 'recording_figure_manifest.json', {
        'provenance': data['provenance'], 'recording_sha256': hashlib.sha256(
            Path(recording_path).read_bytes()).hexdigest(), 'files': files,
        'distance_definition': 'spherical_obstacle_surface_minus_TCP_envelope',
        'reference_definition': 'literal_recorded_controller_reference'})
    return files


def video(recording_path: Path, output: Path):
    from matplotlib.animation import FuncAnimation, PillowWriter
    data, _, time, position, scene = _recording(recording_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for center, radius in zip(scene.centers, scene.radii):
        _sphere(ax, center, radius)
    ax.scatter(*scene.goals.T, c='#d69b40', label='Objects')
    ax.scatter(*scene.place, c='#459b79', marker='s', label='Placement')
    lower, upper = position.min(axis=0) - .1, position.max(axis=0) + .1
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
           xlabel='Base x (m)', ylabel='Base y (m)', zlabel='Base z (m)')
    line, = ax.plot([], [], [], c='#2478a8', label='Measured TCP')
    ax.legend()
    stride = max(1, len(time) // 240)
    def update(index):
        path = position[:index + 1]
        line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
        ax.set_title(f"Measured trajectory replay | {data.get('condition', '')} | t={time[index]:.2f} s")
        return (line,)
    animation = FuncAnimation(fig, update, frames=range(0, len(time), stride), interval=80)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=12))
    plt.close(fig)


def html_table(rows):
    if not rows:
        return '<p>No recorded observations in this section.</p>'
    keys = list(dict.fromkeys(key for row in rows for key in row))
    def cell(value):
        if value is None:
            return 'not estimable'
        return f'{value:.5g}' if isinstance(value, float) else str(value)
    head = ''.join(f'<th>{html.escape(key)}</th>' for key in keys)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(cell(row.get(key)))}</td>' for key in keys)
                   + '</tr>' for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def report(config: dict, root: Path):
    from dasa.figures import build, eligible
    root = Path(root)
    directory = root / 'report'
    directory.mkdir(parents=True, exist_ok=True)
    stages = {
        'Intent inference': (root / 'intent/metrics.json', ['ipa', 'early_prediction_seconds', 'nll']),
        'Shared control': (root / 'sharing/metrics.json', ['success_rate', 'completion_seconds',
                            'safety_index', 'stability_index', 'minimum_clearance_m', 'jerk_rms']),
        'Refined policy': (root / 'refinement/metrics.json', ['success_rate', 'reference_error_m']),
    }
    tables = {}
    for name, (path, metrics) in stages.items():
        rows = load_rows(path)
        if name == 'Refined policy':
            rows = [row for row in rows if eligible(row)]
        tables[name] = summary(rows, metrics)
    figure_status = build(config, root)
    trace_files = []
    recordings = sorted(root.rglob('recording.json'))
    if recordings:
        selected = next((path for path in recordings if json.loads(
            path.read_text(encoding='utf-8')).get('execute') is True), None)
        if selected is not None:
            trace_files = recording_figures(selected, directory / 'trace')
    write_json(directory / 'summary.json', tables)
    links = []
    for path in sorted(root.rglob('*statistics*.csv')):
        if path.is_file():
            relative = os.path.relpath(path, directory).replace(os.sep, '/')
            links.append(f'<li><a href="{html.escape(relative)}">{html.escape(str(path.relative_to(root)))}</a></li>')
    figures = ''.join(f'<figure><img src="trace/{html.escape(name)}" alt="{html.escape(name)}"></figure>'
                      for name in trace_files if name.endswith('.png'))
    sections = ''.join(f'<h2>{html.escape(name)}</h2>{html_table(rows)}' for name, rows in tables.items())
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Measured shared-control experiment</title>
<style>body{{font:15px system-ui;max-width:1180px;margin:32px auto;padding:16px;color:#263244}}
img{{max-width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border:1px solid #dce2e9}}
figure{{margin:24px 0}}a{{color:#2478a8}}</style>
<h1>Measured shared-control experiment</h1>
<p>Participant-level summaries from explicitly measured records. Planned acquisitions are not outcomes.</p>
<p><a href="../figures/manifest.json">Figure and data manifest</a></p>
<h2>Statistical contrasts</h2><ul>{''.join(links)}</ul>{sections}{figures}</html>'''
    (directory / 'index.html').write_text(body, encoding='utf-8')
    write_json(directory / 'report_manifest.json', {'figures': figure_status,
               'recording_count': len(recordings), 'summary_file': 'summary.json',
               'provenance': 'measured_records_only'})
    return directory / 'index.html'


def learning_trajectories(config, recordings, scene_id, output):
    from dasa.data import Demonstration

    recordings, output = Path(recordings), Path(output)
    groups, sources, actual_scene = {}, [], None
    tolerance = config.get('study', {}).get('hardware_scene_tolerance_m', .005)
    for path in sorted(recordings.rglob('*.npz')):
        demo = Demonstration.load(path)
        assignment = demo.metadata.get('experiment_assignment', {})
        if assignment.get('stage') != 'autonomous_evaluation' or assignment.get('scene_id') != scene_id:
            continue
        condition = assignment.get('training_condition')
        repeat = assignment.get('training_repeat')
        if condition not in {'C1', 'C2', 'C3'} or type(repeat) is not int:
            raise ValueError('A trajectory needs an actual policy-condition and training-repeat assignment')
        if actual_scene is None:
            actual_scene = demo.scene
        else:
            if actual_scene.obstacle_ids != demo.scene.obstacle_ids:
                raise ValueError('The selected trajectories have different obstacle identities')
            for name in ('goals', 'start', 'place', 'centers', 'radii'):
                x, y = getattr(actual_scene, name), getattr(demo.scene, name)
                if x.shape != y.shape or np.max(np.abs(x - y), initial=0.) > tolerance:
                    raise ValueError('Selected trajectories exceed the matched-layout tolerance')
        key = (condition, repeat)
        if key in groups:
            raise ValueError('A matched scene must have one actual trajectory per policy repeat')
        groups[key] = demo
        sources.append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'condition': condition, 'training_repeat': repeat,
                        'success': demo.metadata['success'], 'goal': demo.goal})
    if not groups:
        raise ValueError('No actual autonomous trajectories match the selected scene ID')
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    colors = {'C1': '#637381', 'C2': '#d69b40', 'C3': '#2478a8'}
    labelled = set()
    for (condition, _), demo in sorted(groups.items()):
        label = condition if condition not in labelled else None
        labelled.add(condition)
        ax.plot(*demo.position.T, color=colors[condition], alpha=.65,
                linewidth=1.4, label=label)
        if not demo.metadata['success']:
            ax.scatter(*demo.position[-1], color=colors[condition], marker='x', s=35)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 16))
    for center, radius in zip(actual_scene.centers, actual_scene.radii):
        ax.plot_surface(center[0] + radius * np.cos(u) * np.sin(v),
                        center[1] + radius * np.sin(u) * np.sin(v),
                        center[2] + radius * np.cos(v), color='#a9b4be', alpha=.2, linewidth=0)
    ax.scatter(*actual_scene.goals.T, marker='o', color='#263238', s=35, label='Candidate goals')
    ax.scatter(*actual_scene.place, marker='s', color='#459b79', s=45, label='Place')
    ax.scatter(*actual_scene.start, marker='^', color='#263238', s=45, label='Start')
    ax.set(xlabel='Base x (m)', ylabel='Base y (m)', zlabel='Base z (m)',
           title=f'Measured autonomous trajectories: {scene_id}')
    ax.legend(loc='best')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'learning_trajectories.{extension}', dpi=250)
    plt.close(fig)
    write_json(output / 'trajectory_sources.json', {'scene_id': scene_id, 'sources': sources,
        'selection': 'all_actual_policy_repeats_for_explicitly_selected_matched_scene',
        'failure_display': 'failed_traces_retained_with_cross_at_final_measured_position',
        'obstacle_display': 'first_actual_calibrated_scene_after_matching_all_actual_layouts',
        'provenance': 'UR10_measured_teleoperation'})
    return sources
