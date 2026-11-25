"""Publication-window metrics compatible with the documented original protocol.

These metrics deliberately do not require action status messages. All timestamps
are bag reception times. Terminal plan residuals are explicitly marked as proxies.
"""
import math
import numpy as np
from .numerics import consistency, distance_to_polyline, rmse, wrap


def finite_pose(row):
    return all(math.isfinite(row.get(k, float('nan'))) for k in ('x', 'y', 'yaw'))


def same_frame(rows):
    return bool(rows) and len({r.get('frame') for r in rows}) == 1 and bool(rows[0].get('frame'))


def distance_metrics(rows, max_gap, rate=5., max_speed=2.):
    """Resample continuous valid segments; never bridge gaps or pose jumps."""
    segments, current, invalid = [], [], 0
    for row in rows:
        if not finite_pose(row):
            if current: segments.append(current)
            current = []; invalid += 1
            continue
        if current:
            last = current[-1]
            dt = row['t'] - last['t']
            d = math.hypot(row['x']-last['x'], row['y']-last['y'])
            if dt <= 0 or dt > max_gap or row['frame'] != last['frame'] or d/dt > max_speed:
                segments.append(current); current = []; invalid += 1
        current.append(row)
    if current: segments.append(current)
    partial, covered = 0., 0.
    for segment in segments:
        if len(segment) < 2: continue
        t = np.array([r['t'] for r in segment]); t = t-t[0]
        grid = np.unique(np.append(np.arange(0., t[-1], 1/rate), t[-1]))
        xy = np.column_stack([np.interp(grid, t, [r[k] for r in segment]) for k in ('x','y')])
        partial += float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        covered += float(t[-1])
    raw = sum(math.hypot(b['x']-a['x'], b['y']-a['y']) for a,b in zip(rows, rows[1:])
              if finite_pose(a) and finite_pose(b) and a['frame']==b['frame'])
    return dict(odom_distance_5hz_m=partial if covered and not invalid else None,
                odom_partial_distance_5hz_m=partial if covered else None,
                odom_raw_sample_distance_m=raw if len(rows)>1 else None,
                odom_invalid_segments=invalid, odom_covered_s=covered)


def publication_metrics(data, settings, annotation):
    """Return separately named metrics; never overwrite strict action residuals."""
    streams = data['streams']
    commands = streams.get('/cmd_vel', [])
    output = dict(legacy_command_topic='/cmd_vel', legacy_window_source='unavailable')
    if not commands:
        return output
    start, end = commands[0]['t'], commands[-1]['t']
    output.update(legacy_window_source='first_to_last_recorded_cmd_vel',
                  command_start_s=start, command_end_s=end, command_window_s=end-start,
                  legacy_command_count=len(commands))
    cut = lambda name: [r for r in streams.get(name,[]) if start <= r['t'] <= end]
    gap = settings['maximum_command_gap_s']
    observed=weighted=stopped=run=0.; stops=0
    for a,b in zip(commands,commands[1:]):
        dt=b['t']-a['t']
        quiet=math.hypot(a['vx'],a['vy']) < settings['linear_activity_threshold_mps'] and abs(a['wz']) < settings['angular_activity_threshold_radps']
        if 0 < dt <= gap:
            observed+=dt; weighted+=dt*a['vx']
        if 0 < dt <= gap and quiet:
            run+=dt
        else:
            if run >= settings['minimum_command_stop_s']: stops+=1; stopped+=run
            run=0.
    if run >= settings['minimum_command_stop_s']: stops+=1; stopped+=run
    output.update(legacy_commanded_stop_count=stops,commanded_stop_fraction_pct=100*stopped/observed if observed else None,
                  cmd_vx_time_mean_m_s=weighted/observed if observed else None,
                  command_observed_coverage_pct=100*observed/(end-start) if end>start else None,
                  cmd_vx_max_m_s=max(r['vx'] for r in commands),
                  max_command_gap_s=max((b['t']-a['t'] for a,b in zip(commands,commands[1:])),default=None))
    amcl=[r for r in cut('/amcl_pose') if finite_pose(r)]
    output.update(distance_metrics(cut('/odom'),gap))
    plans=[r for r in cut('/plan') if r['points'] and np.isfinite(r['points']).all()]
    plan=plans[0] if plans else None
    output['legacy_reference_plan_time_s']=plan['t'] if plan else None
    if plan and same_frame(amcl) and amcl[0]['frame']==plan['frame']:
        deviations=[distance_to_polyline(r['x'],r['y'],plan['points']) for r in amcl]
        output.update(legacy_initial_plan_deviation_rmse_m=rmse(deviations),
                      initial_plan_deviation_max_m=max(deviations),deviation_samples=len(deviations))
    final_age=end-amcl[-1]['t'] if amcl else None
    output['final_amcl_age_s']=final_age
    fresh=final_age is not None and final_age <= settings.get('legacy_final_max_age_s',2.0)
    for topic,prefix in [('/odom','used'),('/dog_odom','raw')]:
        candidate=[r for r in cut(topic) if finite_pose(r)]
        if same_frame(amcl) and same_frame(candidate):
            detail=consistency(amcl,candidate,settings.get('legacy_sync_tolerance_s',0.15))
            count=detail['matched_samples']
            output[prefix+'_odom_amcl_matched_samples']=count
            output[prefix+'_odom_amcl_match_pct']=100*count/len(amcl)
            if count>=3:
                output['legacy_'+prefix+'_odom_amcl_rmse_m']=detail['position_rmse_m']
                output[prefix+'_odom_amcl_yaw_rmse_deg']=detail['yaw_rmse_deg']
                # Final separation is only valid when the final AMCL sample is matched.
                last_matched=any(abs(r['t']-amcl[-1]['t']) <= settings.get('legacy_sync_tolerance_s',0.15) for r in candidate)
                output['legacy_'+prefix+'_odom_amcl_final_m']=detail.get('final_separation_m') if fresh and last_matched else None
    target=None
    if all(str(annotation.get(k,'')).strip() for k in ('goal_x','goal_y','goal_yaw_rad')):
        target=dict(x=float(annotation['goal_x']),y=float(annotation['goal_y']),yaw=float(annotation['goal_yaw_rad']),frame=settings['map_frame'])
        output['goal_reference']='annotated_goal'
    else:
        goals=cut('/goal_pose')
        if len(goals)==1:
            target=goals[0]; output['goal_reference']='single_recorded_goal_pose'
        elif plan:
            target=dict(x=plan['points'][-1][0],y=plan['points'][-1][1],yaw=plan.get('endpoint_yaw',float('nan')),frame=plan['frame'])
            output['goal_reference']='initial_plan_endpoint_proxy'
    if target and fresh and same_frame(amcl) and amcl[-1]['frame']==target['frame']:
        pose=amcl[-1]
        output['final_position_residual_m']=math.hypot(pose['x']-target['x'],pose['y']-target['y'])
        if math.isfinite(target['yaw']):
            output['final_orientation_residual_deg']=abs(math.degrees(wrap(pose['yaw']-target['yaw'])))
    if plan:
        output['plan_endpoint_change_max_m']=max(math.hypot(p['points'][-1][0]-plan['points'][-1][0],p['points'][-1][1]-plan['points'][-1][1]) for p in plans if p['frame']==plan['frame'])
    return output
