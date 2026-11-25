"""Command windows and sustained stops. These describe commands, not physical motion."""
import math


def select_commands(streams, preferred='/cmd_vel_rlsar'):
    for topic in dict.fromkeys([preferred, '/cmd_vel_mux', '/cmd_vel_nav_scaled', '/cmd_vel']):
        if streams.get(topic):
            return topic, streams[topic]
    return None, []


def command_metrics(streams, settings):
    topic, rows = select_commands(streams, settings.get('command_topic', '/cmd_vel_rlsar'))
    result = dict(command_detection_topic=topic, command_publication_window_s=None,
                  active_command_window_s=None, commanded_stop_count=None,
                  commanded_stop_duration_s=None, command_observed_s=None,
                  commanded_motion_s=None, active_command_samples=0)
    if len(rows) < 2:
        return result
    active = lambda r: (math.hypot(r['vx'], r['vy']) >= settings['linear_activity_threshold_mps']
                        or abs(r['wz']) >= settings['angular_activity_threshold_radps'])
    indices = [i for i,r in enumerate(rows) if active(r)]
    result.update(command_publication_window_s=rows[-1]['t']-rows[0]['t'],
                  active_command_samples=len(indices))
    if not indices:
        # No traversal, so the count of within-traversal stops is undefined.
        return result
    result['active_command_window_s'] = rows[indices[-1]]['t']-rows[indices[0]]['t']
    run = observed = motion = stop_duration = 0.0
    count = 0
    minimum = settings.get('minimum_command_stop_s', 1.0)
    maximum_gap = settings['maximum_command_gap_s']
    # Exclude waiting before first active command and terminal idle samples.
    subset = rows[indices[0]:indices[-1]+1]
    for first,second in zip(subset, subset[1:]):
        dt = second['t']-first['t']
        if not 0 < dt <= maximum_gap or active(first):
            if run >= minimum:
                count += 1
                stop_duration += run
            run = 0.0
        if 0 < dt <= maximum_gap:
            observed += dt
            if active(first):
                motion += dt
            else:
                run += dt
    if run >= minimum:
        count += 1
        stop_duration += run
    result.update(commanded_stop_count=count, commanded_stop_duration_s=stop_duration,
                  command_observed_s=observed, commanded_motion_s=motion,
                  minimum_command_stop_s=minimum)
    return result
