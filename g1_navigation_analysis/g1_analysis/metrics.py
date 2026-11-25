"""Per-trial metrics. Missing evidence yields None and an audit warning."""
import math
import numpy as np
from .numerics import pairs, rmse, wrap, consistency, distance_to_polyline
from .outcomes import resolve


def window(rows, start, end):
    return [r for r in rows if start <= r['t'] <= end]


def stable_frame(rows):
    frames = {r.get('frame') for r in rows}
    return len(frames) == 1 and None not in frames and '' not in frames


def analyze(trial, data, settings, annotation):
    result = dict(trial)
    result['settings_id'] = annotation.get('settings_id', '')
    result.update(resolve(data, annotation))
    warnings = list(data['warnings'])
    streams = data['streams']
    start = result['start_s'] if result['start_s'] is not None else data['bag_start']
    end = result['end_s'] if result['end_s'] is not None else data['bag_end']
    if start is None or end is None:
        raise ValueError('Empty bag.')
    result['analysis_start_s'], result['analysis_end_s'] = start, end
    result['window_source'] = 'mission' if result['start_s'] is not None and result['end_s'] is not None else 'bag_or_partial_mission'
    result['mission_duration_s'] = end-start if result['window_source'] == 'mission' else None
    if result['window_source'] != 'mission':
        warnings.append('Mission boundary incomplete: diagnostic window includes bag boundaries.')
    if settings['time_basis'] == 'header':
        warnings.append('Header mode requires independently verified ROS/header clock compatibility; status times are reception times.')
    selected = {topic: window(rows, start, end) for topic, rows in streams.items()}
    commands = selected.get(settings['command_topic'], [])
    active = [r for r in commands if math.hypot(r['vx'],r['vy']) >= settings['linear_activity_threshold_mps'] or abs(r['wz']) >= settings['angular_activity_threshold_radps']]
    result['active_command_window_s'] = active[-1]['t']-active[0]['t'] if active else None
    result['active_command_samples'] = len(active)
    if not active:
        warnings.append('No active command detected; trial retained in outcome counts.')
    observed = moving = in_domain = 0.0
    for first, second in zip(commands, commands[1:]):
        dt = second['t']-first['t']
        if 0 < dt <= settings['maximum_command_gap_s']:
            observed += dt
            if math.hypot(first['vx'],first['vy']) >= settings['linear_activity_threshold_mps'] or abs(first['wz']) >= settings['angular_activity_threshold_radps']:
                moving += dt
    input_rows = selected.get('/cmd_vel', [])
    input_observed = 0.0
    for a,b in zip(input_rows, input_rows[1:]):
        dt = b['t']-a['t']
        if 0 < dt <= settings['maximum_command_gap_s']:
            input_observed += dt
            if settings['compensation_min_vx'] <= a['vx'] <= settings['compensation_max_vx']:
                in_domain += dt
    result.update(command_observed_s=observed, commanded_motion_s=moving,
                  compensation_domain_time_fraction=in_domain/input_observed if input_observed else None)
    for source, target, name in [('/cmd_vel','/cmd_vel_nav_scaled','compensation'),
                                  ('/cmd_vel_nav_scaled','/cmd_vel_mux','arbitration'),
                                  ('/cmd_vel_mux','/cmd_vel_rlsar','transport')]:
        matched = pairs(selected.get(source,[]), selected.get(target,[]), settings['command_match_tolerance_s'])
        result[f'{name}_matched_samples'] = len(matched)
        for axis in ('vx','vy','wz'):
            delta = [b[axis]-a[axis] for a,b in matched]
            result[f'{name}_{axis}_delta_mean'] = float(np.mean(delta)) if delta else None
            result[f'{name}_{axis}_delta_rmse'] = rmse(delta)
    amcl = selected.get('/amcl_pose', [])
    odom = selected.get('/odom', [])
    result['odom_amcl_consistency_rmse_m'] = None
    if stable_frame(amcl) and stable_frame(odom):
        result['consistency_details'] = consistency(amcl, odom, settings['pose_match_tolerance_s'])
        result['odom_amcl_consistency_rmse_m'] = result['consistency_details']['position_rmse_m']
    else:
        warnings.append('Odometry-AMCL consistency unavailable: missing or changing frame IDs.')
    result['raw_corrected_consistency'] = {}
    raw, corrected = selected.get('/dog_odom', []), selected.get('/odom_dog_calibrated', [])
    if stable_frame(raw) and stable_frame(corrected):
        result['raw_corrected_consistency'] = consistency(corrected, raw, settings['pose_match_tolerance_s'])
    result['initial_plan_deviation_rmse_m'] = None
    plans = [p for p in streams.get('/plan', []) if start <= p['t'] <= end and p['points']]
    if plans and stable_frame(amcl) and amcl[0]['frame'] == plans[0]['frame']:
        track = [r for r in amcl if r['t'] >= plans[0]['t']]
        result['initial_plan_deviation_rmse_m'] = rmse([distance_to_polyline(r['x'],r['y'],plans[0]['points']) for r in track])
    else:
        warnings.append('Initial-plan deviation unavailable: missing plan/AMCL or incompatible frames.')
    result['plan_publications'] = len(plans)
    # A single recorded goal is acceptable. Multiple goal publications require annotation.
    goals = [g for g in streams.get('/goal_pose', []) if g['t'] <= end and g['frame'] == settings['map_frame']]
    target = None
    if all(annotation.get(k, '').strip() for k in ('goal_x','goal_y','goal_yaw_rad')):
        target = dict(x=float(annotation['goal_x']), y=float(annotation['goal_y']), yaw=float(annotation['goal_yaw_rad']))
        result['target_source'] = 'annotation_map_frame'
    elif len(goals) == 1 and result['candidate_goal_count'] <= 1:
        target = goals[0]
        result['target_source'] = 'single_goal_pose_message'
    else:
        result['target_source'] = 'unavailable_or_ambiguous'
    result['terminal_position_residual_m'] = None
    result['terminal_orientation_residual_deg'] = None
    # Use last AMCL pose at or before terminal evidence, not arbitrary bag end.
    if target is not None and result['end_s'] is not None:
        poses = [r for r in streams.get('/amcl_pose', []) if r['frame'] == settings['map_frame'] and r['t'] <= end and end-r['t'] <= settings['terminal_pose_tolerance_s']]
        if poses:
            pose = poses[-1]
            result['terminal_pose_age_s'] = end-pose['t']
            result['terminal_position_residual_m'] = math.hypot(pose['x']-target['x'], pose['y']-target['y'])
            result['terminal_orientation_residual_deg'] = abs(math.degrees(wrap(pose['yaw']-target['yaw'])))
    if result['terminal_position_residual_m'] is None:
        warnings.append('Terminal residual unavailable: no unambiguous target, terminal time, or sufficiently recent AMCL pose.')
    result['warnings'] = warnings
    result['timestamp_audit'] = {}
    for topic, rows in streams.items():
        ages = [r['bag_t']-r['header_t'] for r in rows if r.get('header_t') is not None]
        if ages:
            result['timestamp_audit'][topic] = dict(
                reception_minus_header_median_s=float(np.median(ages)),
                reception_minus_header_p05_s=float(np.percentile(ages,5)),
                reception_minus_header_p95_s=float(np.percentile(ages,95)))
    result['topic_counts'] = data['counts']
    result['topic_types'] = data['types']
    return result, selected
