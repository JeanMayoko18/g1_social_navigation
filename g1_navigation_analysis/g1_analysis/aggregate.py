"""Trial-level statistics and unpaired, stratified command-compensation contrasts."""
from collections import defaultdict, Counter
import numpy as np
from .numerics import wilson

METRICS = ['mission_duration_s', 'active_command_window_s', 'terminal_position_residual_m',
           'terminal_orientation_residual_deg', 'initial_plan_deviation_rmse_m',
           'odom_amcl_consistency_rmse_m', 'compensation_domain_time_fraction',
           'compensation_vy_delta_rmse', 'compensation_wz_delta_rmse']


def groups(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['scenario'],r['speed_mps'],r['configuration'])].append(r)
    return grouped


def summarize(results):
    output = []
    for (scenario,speed,configuration), trials in sorted(groups(results).items()):
        counts = Counter(r['outcome'] for r in trials)
        n = len(trials)
        unknown = counts['UNKNOWN']
        known = n-unknown
        low,high = wilson(counts['SUCCEEDED'], known)
        row = dict(scenario=scenario,speed_mps=speed,configuration=configuration,
                   n_trials=n,n_success=counts['SUCCEEDED'],n_unknown=unknown,
                   n_aborted=counts['ABORTED'],n_canceled=counts['CANCELED'],
                   n_other_failure=counts['FAILED']+counts['TIMEOUT'],
                   success_rate_known=counts['SUCCEEDED']/known if known else None,
                   success_ci95_low=low,success_ci95_high=high,
                   success_fraction_lower_bound=counts['SUCCEEDED']/n,
                   success_fraction_upper_bound=(counts['SUCCEEDED']+unknown)/n)
        for metric in METRICS:
            for population in ('all', 'success'):
                values = [r[metric] for r in trials if r.get(metric) is not None and
                          (population == 'all' or r['outcome'] == 'SUCCEEDED')]
                prefix = f'{metric}_{population}'
                row[prefix+'_n'] = len(values)
                row[prefix+'_median'] = float(np.median(values)) if values else None
                row[prefix+'_q25'] = float(np.percentile(values,25)) if values else None
                row[prefix+'_q75'] = float(np.percentile(values,75)) if values else None
        output.append(row)
    return output


def contrasts(results, settings):
    grouped = groups(results)
    rng = np.random.default_rng(settings['random_seed'])
    output = []
    for scenario,speed in sorted({(r['scenario'],r['speed_mps']) for r in results}):
        a = grouped.get((scenario,speed,'Odometry-only'), [])
        b = grouped.get((scenario,speed,'calibrated'), [])
        ids = {r.get('settings_id','') for r in a+b}
        verified = bool(a and b) and len(ids) == 1 and '' not in ids
        for metric in ['success_rate']+METRICS:
            population = 'known_outcomes' if metric == 'success_rate' else 'success_only'
            def extract(rows):
                if metric == 'success_rate':
                    return np.array([float(r['outcome']=='SUCCEEDED') for r in rows if r['outcome']!='UNKNOWN'])
                return np.array([r[metric] for r in rows if r['outcome']=='SUCCEEDED' and r.get(metric) is not None])
            x,y = extract(a),extract(b)
            row = dict(scenario=scenario,speed_mps=speed,metric=metric,population=population,
                       direction='calibrated minus Odometry-only',n_odometry_only=len(x),n_combined=len(y),
                       matched_settings_declared=verified,delta_mean=None,ci95_low=None,ci95_high=None)
            if len(x) and len(y):
                row['delta_mean'] = float(y.mean()-x.mean())
                if len(x)>=2 and len(y)>=2:
                    draws = [rng.choice(y,len(y),replace=True).mean()-rng.choice(x,len(x),replace=True).mean()
                             for _ in range(settings['bootstrap_samples'])]
                    row['ci95_low'],row['ci95_high'] = map(float,np.percentile(draws,[2.5,97.5]))
            output.append(row)
    return output
