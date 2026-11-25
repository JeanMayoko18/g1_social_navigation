"""Pure numerical functions; units and alignment choices are explicit."""
import bisect
import math
import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def pairs(reference, candidate, tolerance):
    times = [r['t'] for r in candidate]
    result = []
    for row in reference:
        index = bisect.bisect_left(times, row['t'])
        nearby = candidate[max(0, index-1):index+1]
        if nearby:
            other = min(nearby, key=lambda r: abs(r['t']-row['t']))
            if abs(other['t']-row['t']) <= tolerance:
                result.append((row, other))
    return result


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def local_xy(rows):
    """Initial SE(2) alignment, not trajectory-fitting or ground-truth alignment."""
    first = rows[0]
    c, s = math.cos(first['yaw']), math.sin(first['yaw'])
    return np.array([[c*(r['x']-first['x'])+s*(r['y']-first['y']),
                     -s*(r['x']-first['x'])+c*(r['y']-first['y'])] for r in rows])


def consistency(reference, candidate, tolerance):
    matched = pairs(reference, candidate, tolerance)
    if len(matched) < 2:
        return {'matched_samples': len(matched), 'position_rmse_m': None, 'yaw_rmse_deg': None}
    a, b = zip(*matched)
    delta = local_xy(a) - local_xy(b)
    angles = [math.degrees(wrap((x['yaw']-a[0]['yaw'])-(y['yaw']-b[0]['yaw']))) for x,y in matched]
    return dict(matched_samples=len(matched), position_rmse_m=rmse(np.linalg.norm(delta, axis=1)),
                yaw_rmse_deg=rmse(angles), final_separation_m=float(np.linalg.norm(delta[-1])),
                alignment='independent initial SE2 alignment; no scale fitting',
                maximum_time_mismatch_s=max(abs(x['t']-y['t']) for x,y in matched))


def distance_to_polyline(x, y, points):
    p = np.asarray(points, dtype=float)
    if len(p) == 1:
        return float(np.linalg.norm(p[0]-[x,y]))
    a, d = p[:-1], np.diff(p, axis=0)
    length2 = np.sum(d*d, axis=1)
    u = np.divide(np.sum((np.array([x,y])-a)*d, axis=1), length2,
                  out=np.zeros(len(d)), where=length2 > 0)
    projected = a + np.clip(u, 0, 1)[:, None]*d
    return float(np.min(np.linalg.norm(projected-[x,y], axis=1)))


def wilson(successes, count):
    if not count:
        return None, None
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0, center-half), min(1, center+half)
