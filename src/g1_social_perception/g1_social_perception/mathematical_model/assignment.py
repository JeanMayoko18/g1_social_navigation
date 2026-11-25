"""
Global one-to-one association with explicit ambiguity assessment.

Receives:
    Track/candidate pair scores, evidence support and comparison modes.
    Hard-gate decisions and mode-specific acceptance thresholds.

Processing:
    1. Exclude inadmissible pairs.
    2. Maximize total similarity with optional unmatched tracks.
    3. For each selected pair, solve again with that pair forbidden.
    4. Accept the pair only when its global margin is sufficient.

Equations:
    J(A) = sum(A[i, j] * S[i, j])

    Constraints:
        Each track receives at most one candidate.
        Each candidate updates at most one track.

    Global margin:
        margin[i, j] = J(best) - J(best_without_pair[i, j])

Returns:
    Accepted pairs, ambiguous pairs and unassigned indices.

Boundary:
    No track mutation, identity creation or confirmation occurs here.
    An unassigned detection is not automatically a new person.
"""


import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .validation import integer, nonempty, positive, unit_interval


# ============================================================================
# INPUT CONTRACTS
# ============================================================================


@dataclass(frozen=True)
class PairEvidence:
    """
    Evidence for track i and candidate j.

    Fields:
        score:
            Fused similarity S_ij, or None without usable evidence.

        support:
            Available nominal weight sum(w_r * q_r).

        mode:
            Explicit comparison mode selected upstream.
            Its name indexes the configured threshold policy.

        hard_gate_passed:
            Whether required physical and semantic constraints passed.
            A high fused score cannot override a failed hard gate.

    Indices refer to stable input lists for this association cycle.
    """

    track_index: int
    candidate_index: int
    score: Optional[float]
    support: float
    mode: str
    hard_gate_passed: bool

    def __post_init__(self):
        integer("track_index", self.track_index)
        integer("candidate_index", self.candidate_index)
        unit_interval("support", self.support)
        nonempty("mode", self.mode)

        if type(self.hard_gate_passed) is not bool:
            raise ValueError("hard_gate_passed must be a boolean")

        if self.score is None:
            if self.support != 0.0:
                raise ValueError(
                    "Unavailable score requires zero support"
                )
        else:
            unit_interval("pair score", self.score)


@dataclass(frozen=True)
class AssignmentThresholds:
    """
    Acceptance policy for one explicitly named evidence mode.

    minimum_score:
        Minimum fused similarity for an admissible pair.

    minimum_support:
        Minimum available evidence mass.

    minimum_margin:
        Minimum loss in the global objective when the pair is forbidden.

    All thresholds are positive and at most one.
    Values must be configured; this module supplies no arbitrary defaults.
    """

    minimum_score: float
    minimum_support: float
    minimum_margin: float

    def __post_init__(self):
        for name in (
            "minimum_score",
            "minimum_support",
            "minimum_margin",
        ):
            value = getattr(self, name)
            unit_interval(name, value)
            positive(name, value)


# ============================================================================
# OUTPUT CONTRACTS
# ============================================================================


@dataclass(frozen=True)
class SelectedPair:
    """One globally selected pair and its alternative-solution margin."""

    evidence: PairEvidence
    global_margin: float


@dataclass(frozen=True)
class AssignmentResult:
    """
    Association outcome before updating track memory.

    accepted:
        Selected pairs whose global margins passed.

    ambiguous:
        Selected pairs whose global margins were insufficient.
        Their candidate data must not update the selected identities.

    unassigned_tracks / unassigned_candidates:
        Indices not participating in an accepted association.
        These include ambiguous selections.

    candidates_without_admissible_pair:
        Candidates with no edge surviving the configured gates.
        New-track creation still requires a separate eligibility policy.

    best_objective:
        Objective of the optimal assignment before ambiguity rejection.
    """

    accepted: Tuple[SelectedPair, ...]
    ambiguous: Tuple[SelectedPair, ...]
    unassigned_tracks: Tuple[int, ...]
    unassigned_candidates: Tuple[int, ...]
    candidates_without_admissible_pair: Tuple[int, ...]
    best_objective: float


# ============================================================================
# RECTANGULAR MINIMUM-COST ASSIGNMENT
# ============================================================================


def _hungarian(cost):
    """
    Assign every row to a distinct column with minimum total cost.

    Receives:
        A finite rectangular cost matrix with columns >= rows.

    Returns:
        Selected column index for each row.

    Implementation:
        Hungarian augmenting-path algorithm.
        Row/column potentials maintain reduced costs while each new row
        extends or rearranges the current assignment.

    The caller adds dummy columns to represent unmatched tracks.
    """
    n = len(cost)

    if n == 0:
        return ()

    m = len(cost[0])

    # Potentials, column owners and augmenting-path predecessors.
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0

        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = float("inf"), 0

            for j in range(1, m + 1):
                if used[j]:
                    continue

                reduced = cost[i0 - 1][j - 1] - u[i0] - v[j]

                if reduced < minimum[j]:
                    minimum[j], way[j] = reduced, j0

                if minimum[j] < delta:
                    delta, j1 = minimum[j], j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        # Reverse the augmenting path to install the new assignment.
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    columns = [-1] * n

    for j in range(1, m + 1):
        if p[j]:
            columns[p[j] - 1] = j - 1

    return tuple(columns)


def _solve(track_count, candidate_count, edges, forbidden=None):
    """
    Maximize admissible similarity using a minimum-cost solver.

    Cost construction:
        Admissible pair: -score.
        Unmatched track: 0 through a dummy column.
        Forbidden pair: +2, always worse than an available dummy.

    One dummy column per track allows every track to remain unmatched.
    Real candidate columns may remain unused.
    """
    cost = [
        [2.0] * candidate_count + [0.0] * track_count
        for _ in range(track_count)
    ]

    for key, evidence in edges.items():
        if key == forbidden:
            continue

        i, j = key
        cost[i][j] = -evidence.score

    columns = _hungarian(cost)

    selected = tuple(
        (i, j)
        for i, j in enumerate(columns)
        if j < candidate_count
    )

    if any(key not in edges or key == forbidden for key in selected):
        raise RuntimeError("Solver selected a forbidden association")

    objective = math.fsum(
        edges[key].score
        for key in selected
    )

    return selected, objective


# ============================================================================
# GLOBAL ASSOCIATION AND AMBIGUITY
# ============================================================================


def assign_pairs(
    track_count: int,
    candidate_count: int,
    pairs: Sequence[PairEvidence],
    thresholds_by_mode: Mapping[str, AssignmentThresholds],
) -> AssignmentResult:
    """
    Select globally consistent and sufficiently unambiguous associations.

    Admissibility:
        Hard gate passed.
        Score available and above the mode's minimum.
        Support above the mode's minimum.

    Ambiguity:
        Recompute the optimal assignment after forbidding each selected pair.
        Accept only if the objective loss meets that mode's minimum margin.

    No reassignment is performed after rejecting ambiguous pairs.
    Otherwise removing a competitor could manufacture apparent certainty.

    Missing pairs are treated as inadmissible.
    Duplicate pair entries and unknown modes are configuration errors.
    """
    integer("track_count", track_count)
    integer("candidate_count", candidate_count)

    edges = {}
    seen = set()

    for evidence in pairs:
        if not isinstance(evidence, PairEvidence):
            raise ValueError("Every pair must be PairEvidence")

        i = evidence.track_index
        j = evidence.candidate_index
        key = (i, j)

        if i >= track_count or j >= candidate_count:
            raise ValueError("Pair index exceeds the input list size")

        if key in seen:
            raise ValueError("Duplicate track/candidate pair")

        seen.add(key)

        policy = thresholds_by_mode.get(evidence.mode)

        if not isinstance(policy, AssignmentThresholds):
            raise ValueError(
                "Missing assignment policy for mode " + evidence.mode
            )

        if (
            evidence.hard_gate_passed
            and evidence.score is not None
            and evidence.score >= policy.minimum_score
            and evidence.support >= policy.minimum_support
        ):
            edges[key] = evidence

    selected, best_objective = _solve(
        track_count,
        candidate_count,
        edges,
    )

    accepted = []
    ambiguous = []

    for key in selected:
        evidence = edges[key]
        policy = thresholds_by_mode[evidence.mode]

        _, alternative_objective = _solve(
            track_count,
            candidate_count,
            edges,
            forbidden=key,
        )

        # Clamp tiny negative differences caused by floating-point rounding.
        margin = max(
            0.0,
            best_objective - alternative_objective,
        )

        decision = SelectedPair(
            evidence=evidence,
            global_margin=margin,
        )

        if margin >= policy.minimum_margin:
            accepted.append(decision)
        else:
            ambiguous.append(decision)

    accepted_tracks = {
        pair.evidence.track_index
        for pair in accepted
    }
    accepted_candidates = {
        pair.evidence.candidate_index
        for pair in accepted
    }
    candidates_with_edges = {
        candidate_index
        for _, candidate_index in edges
    }

    return AssignmentResult(
        accepted=tuple(accepted),
        ambiguous=tuple(ambiguous),
        unassigned_tracks=tuple(
            i for i in range(track_count)
            if i not in accepted_tracks
        ),
        unassigned_candidates=tuple(
            j for j in range(candidate_count)
            if j not in accepted_candidates
        ),
        candidates_without_admissible_pair=tuple(
            j for j in range(candidate_count)
            if j not in candidates_with_edges
        ),
        best_objective=best_objective,
    )