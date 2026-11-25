"""
Association similarities and quality-weighted fusion.

Receives:
    Compatible geometric, motion, appearance and size evidence.
    Pair qualities established before score calculation.
    Explicit weights and comparison scales.

Calculates:
    g: predicted-box intersection over union.
    m: distance-based motion consistency.
    a: quality-weighted upper/lower appearance similarity.
    h: quality-weighted apparent/metric size similarity.

    Global score:
        S = sum(w[r] * q[r] * S[r]) / sum(w[r] * q[r])

    Effective weights:
        effective_w[r] = w[r] * q[r] / sum(w[l] * q[l])

Returns:
    Scores, qualities, available support and effective weights.

Boundary:
    Scores do not assign identities.
    Pair admissibility, global assignment and ambiguity decisions
    belong to assignment.py.

    Zero quality means excluded evidence.
    Zero similarity with positive quality means measured disagreement.
"""


import math
from typing import Dict, Mapping, Sequence, Tuple

from .geometry import distance_3d, intersection_over_union
from .types import BoundingBox2D, CueScore, Vector3
from .validation import (
    finite,
    normalized,
    positive,
    unit_interval,
)


# ============================================================================
# COMMON QUALITY-WEIGHTED FUSION
# ============================================================================


def combine(
    cues: Mapping[str, CueScore],
    weights: Mapping[str, float],
) -> Tuple[CueScore, Dict[str, float]]:
    """
    Combine similarities using their actually available evidence.

    Equations:
        mass[r] = weight[r] * quality[r]
        support = sum(mass[r])
        effective_weight[r] = mass[r] / support
        score = sum(effective_weight[r] * score[r])

        composite_quality = support / sum(weight[r])

    Receives:
        Named cues and coefficients with exactly the same keys.
        Coefficients may be unnormalized but must be non-negative.

    Returns:
        Composite CueScore and effective weights.

    No support:
        Return CueScore(None, 0) and zero effective weights.
        Never replace absent evidence with a measured similarity of zero.
    """
    if not cues or set(cues) != set(weights):
        raise ValueError(
            "Cues and weights must have identical nonempty keys"
        )

    for name, weight in weights.items():
        finite("weight " + name, weight)

        if weight < 0.0:
            raise ValueError("Weights must be non-negative")

        if not isinstance(cues[name], CueScore):
            raise ValueError("Every cue must be CueScore")

    nominal_total = math.fsum(weights.values())
    positive("total nominal weight", nominal_total)

    masses = {
        name: weights[name] * cues[name].quality
        for name in cues
    }

    support = math.fsum(masses.values())

    if support == 0.0:
        return (
            CueScore(score=None, quality=0.0),
            {name: 0.0 for name in cues},
        )

    effective_weights = {
        name: mass / support
        for name, mass in masses.items()
    }

    # Positive mass guarantees an available score through CueScore's contract.
    score = math.fsum(
        effective_weights[name] * cues[name].score
        for name in cues
        if masses[name] > 0.0
    )

    # Bound only floating-point rounding at the interval endpoints.
    return (
        CueScore(
            score=min(1.0, max(0.0, score)),
            quality=min(1.0, support / nominal_total),
        ),
        effective_weights,
    )


# ============================================================================
# GEOMETRIC CUE: g
# ============================================================================

def geometric_score(
    predicted_bbox: BoundingBox2D,
    candidate_bbox: BoundingBox2D,
    pair_quality: float,
) -> CueScore:
    """
    Calculate geometric similarity S_g from predicted-box overlap.

    Equation:
        S_g = IoU(predicted_bbox, candidate_bbox)

    Both boxes must refer to the candidate image coordinate system
    at the candidate observation time.

    The caller supplies pair quality only after validating that prediction.
    A retained old box is not automatically a valid predicted box.
    """
    unit_interval("geometric pair quality", pair_quality)

    if pair_quality == 0.0:
        return CueScore(None, 0.0)

    return CueScore(
        score=intersection_over_union(predicted_bbox, candidate_bbox),
        quality=pair_quality,
    )

# ============================================================================
# MOTION CUE: m
# ============================================================================

def motion_score(
    predicted_position: Vector3,
    candidate_position: Vector3,
    distance_tolerance_m: float,
    pair_quality: float,
) -> CueScore:
    """
    Calculate motion consistency from the prediction residual.

    Equations:
        d_ij = norm(candidate_position - predicted_position)
        S_m = max(0, 1 - d_ij / distance_tolerance_m)

    Receives:
        Positions expressed in the same odom frame, in metres.
        A strictly positive distance tolerance, in metres.
        Reliability returned by the motion-quality calculation.

    Returns:
        One for coincident positions.
        Zero at or beyond the distance tolerance.

    The tolerance is supplied explicitly.
    This function does not invent an uncertainty-growth model.
    """
    positive("distance_tolerance_m", distance_tolerance_m)
    unit_interval("motion pair quality", pair_quality)

    if pair_quality == 0.0:
        return CueScore(None, 0.0)

    distance = distance_3d(
        predicted_position,
        candidate_position,
    )

    return CueScore(
        score=max(0.0, 1.0 - distance / distance_tolerance_m),
        quality=pair_quality,
    )

# ============================================================================
# APPEARANCE CUE: a
# ============================================================================

def histogram_similarity(
    reference: Sequence[float],
    candidate: Sequence[float],
    q_reference: float,
    q_candidate: float,
) -> CueScore:
    """
    Calculate Bhattacharyya similarity for one appearance region.

    Equations:
        S_region = sum(sqrt(reference[b] * candidate[b]))
        q_region = min(q_reference, q_candidate)

    Receives:
        Compatible normalized histograms.
        Qualities after temporal and compatibility checks.

    Returns:
        Regional similarity and reliability.

    Histogram schema compatibility must be checked before extracting bins.
    Zero quality excludes the comparison without inspecting unavailable bins.
    """
    unit_interval("reference appearance quality", q_reference)
    unit_interval("candidate appearance quality", q_candidate)

    pair_quality = min(q_reference, q_candidate)

    if pair_quality == 0.0:
        return CueScore(None, 0.0)

    reference = tuple(reference)
    candidate = tuple(candidate)

    if len(reference) != len(candidate):
        raise ValueError("Histogram bin counts must match")

    normalized("reference histogram", reference, tolerance=1e-6)
    normalized("candidate histogram", candidate, tolerance=1e-6)

    similarity = math.fsum(
        math.sqrt(first * second)
        for first, second in zip(reference, candidate)
    )

    return CueScore(
        score=min(1.0, similarity),
        quality=pair_quality,
    )

def appearance_score(
    upper: CueScore,
    lower: CueScore,
    eta_upper: float,
    eta_lower: float,
) -> Tuple[CueScore, Dict[str, float]]:
    """
    Combine upper/lower appearance evidence.

    Equations:
        S_a = (eta_upper * q_upper * S_upper
               + eta_lower * q_lower * S_lower)
              / (eta_upper * q_upper + eta_lower * q_lower)

        q_a = (eta_upper * q_upper + eta_lower * q_lower)
              / (eta_upper + eta_lower)

    Returns:
        Appearance CueScore and effective regional weights.

    A missing region contributes no mass.
    The remaining region may contribute, with reduced composite quality.
    """
    positive("eta_upper", eta_upper)
    positive("eta_lower", eta_lower)

    return combine(
        cues={"upper": upper, "lower": lower},
        weights={"upper": eta_upper, "lower": eta_lower},
    )

# ============================================================================
# SIZE CUE: h
# ============================================================================

def size_similarity(
    candidate: Sequence[float],
    reference: Sequence[float],
    sigma: Sequence[float],
    pair_quality: float,
) -> CueScore:
    """
    Compare positive width/height using logarithmic dimension ratios.

    Equations:
        residual_width = log(candidate_width / reference_width)
        residual_height = log(candidate_height / reference_height)

        S_size = exp(
            -0.5 * (
                (residual_width / sigma_width)**2
                + (residual_height / sigma_height)**2
            )
        )

    Receives:
        Candidate and reference (width, height) in identical units.
        Positive dimensionless scales for the logarithmic residuals.
        Pairwise reliability.

    Apparent size:
        Compare current pixel dimensions with predicted pixel dimensions.

    Metric size:
        Compare current metric dimensions with eligible retained dimensions.

    This function does not mix pixels and metres.
    """
    unit_interval("size pair quality", pair_quality)

    sigma = tuple(sigma)

    if len(sigma) != 2:
        raise ValueError("Size sigma must contain width and height scales")

    for scale in sigma:
        positive("size sigma", scale)

    if pair_quality == 0.0:
        return CueScore(None, 0.0)

    candidate = tuple(candidate)
    reference = tuple(reference)

    if len(candidate) != 2 or len(reference) != 2:
        raise ValueError("Size values must contain width and height")

    for value in candidate + reference:
        positive("size dimension", value)

    # Difference of logarithms avoids overflow in the raw dimension ratio.
    residuals = tuple(
        (math.log(current) - math.log(previous)) / scale
        for current, previous, scale in zip(
            candidate,
            reference,
            sigma,
        )
    )

    residual_norm = math.hypot(*residuals)

    # Multiplication permits a very large residual to yield exp(-inf) = 0.
    similarity = math.exp(
        -0.5 * residual_norm * residual_norm
    )

    return CueScore(
        score=similarity,
        quality=pair_quality,
    )

def size_score(
    apparent: CueScore,
    metric: CueScore,
    alpha: float,
    beta: float,
) -> Tuple[CueScore, Dict[str, float]]:
    """
    Combine apparent and metric size evidence.

    Equations:
        S_h = (alpha * q_app * S_app + beta * q_metric * S_metric)
              / (alpha * q_app + beta * q_metric)

        q_h = (alpha * q_app + beta * q_metric) / (alpha + beta)

    Returns:
        Size CueScore and effective apparent/metric weights.
    """
    positive("alpha", alpha)
    positive("beta", beta)

    return combine(
        cues={"apparent": apparent, "metric": metric},
        weights={"apparent": alpha, "metric": beta},
    )

# ============================================================================
# GLOBAL ASSOCIATION SCORE
# ============================================================================

def association_score(
    cues: Mapping[str, CueScore],
    weights: Mapping[str, float],
) -> dict:
    """
    Calculate the global score S_ij for one track/candidate pair.

    Required cues:
        g: geometric overlap.
        m: motion consistency.
        a: composite appearance.
        h: composite size.

    Constraints:
        w_r >= 0
        sum(w_r) = 1

    Equations:
        support = sum(w_r * q_r)
        effective_w_r = w_r * q_r / support
        S_ij = sum(effective_w_r * S_r)

    Returns:
        score: fused similarity, or None without usable support.
        support: total available nominal weight.
        scores: individual cue similarities.
        qualities: individual cue reliabilities.
        effective_weights: actual normalized contributions.

    A high score with weak support is not automatically admissible.
    Assignment must retain support and evidence mode when applying gates.
    """
    expected = {"g", "m", "a", "h"}

    if set(cues) != expected or set(weights) != expected:
        raise ValueError(
            "Association requires exactly the g, m, a and h cues"
        )

    normalized(
        "nominal association weights",
        tuple(weights[name] for name in ("g", "m", "a", "h")),
    )

    fused, effective_weights = combine(cues, weights)

    support = math.fsum(
        weights[name] * cues[name].quality
        for name in ("g", "m", "a", "h")
    )

    return {
        "score": fused.score,
        "support": support,
        "scores": {
            name: cues[name].score
            for name in ("g", "m", "a", "h")
        },
        "qualities": {
            name: cues[name].quality
            for name in ("g", "m", "a", "h")
        },
        "effective_weights": effective_weights,
    }

