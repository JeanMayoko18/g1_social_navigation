"""
Validate the mathematical domains used throughout the pipeline.

Receives:
    A field name and the value to validate.

Processes:
    Checks whether the value belongs to its required mathematical domain.
    Values are never clipped, normalized or replaced automatically.

Returns:
    None when valid; raises ValueError when invalid.

Model correspondence:
    w_r:
        Non-negative association weights whose sum is one.

    q[r, i, j]:
        Reliability in [0, 1]. Zero excludes a cue from fusion.

    S[r, i, j]:
        Similarity in [0, 1]. Zero may represent valid disagreement.

    eta_upper, eta_lower, alpha, beta:
        Strictly positive blending coefficients.

    n:
        Integer observation count, with the minimum selected by the caller.

    Histograms:
        Non-negative bin values whose sum is one.

These checks validate domains, not observation freshness or identity.
"""
import math

def finite(name, value):
    """
    Require a finite scalar before using it in an equation.

    Applies to:
        Coordinates, velocities, durations and other numeric quantities.

    Why:
        NaN and infinity would propagate through distances, predictions
        and scores. A boolean is not accepted as a numeric measurement.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(name + " must be a finite number")

def positive(name, value):
    """
    Require a finite value strictly greater than zero.

    Applies to:
        Box dimensions, distance scales, Gaussian scales and strictly
        positive blending coefficients such as alpha and beta.

    Why:
        Zero dimensions invalidate logarithmic size ratios.
        Zero scales invalidate division and normalized distances.
    """
    finite(name, value)

    if value <= 0:
        raise ValueError(name + " must be positive")

def unit_interval(name, value):
    """
    Require a finite value in the closed interval [0, 1].

    Applies to:
        Qualities, normalized similarities, confidences and weights.

    Interpretation:
        A quality of zero means unavailable or unusable evidence.
        A similarity of zero can mean measured disagreement.
        This function checks the range, not those different meanings.
    """
    finite(name, value)

    if not 0 <= value <= 1:
        raise ValueError(name + " must be in [0, 1]")

def integer(name, value, minimum=0):
    """
    Require an integer count or identifier at or above its minimum.

    Applies to:
        Observation counts, consecutive hits, detection indices
        and identity generations.

    Examples:
        An index may start at zero.
        An initialized track's observation count starts at one.

    Counts are checked without rounding; booleans are rejected.
    """
    if type(minimum) is not int:
        raise ValueError(name + " minimum must be an integer")

    if type(value) is not int or value < minimum:
        raise ValueError(
            name + " must be an integer >= " + str(minimum)
        )

def nonempty(name, value):
    """
    Require an explicit string identifier without surrounding whitespace.

    Applies to:
        Coordinate frames, producer sessions, person IDs
        and histogram schema IDs.

    Why:
        These identifiers establish which measurements can be compared.
        Silent trimming could hide inconsistent producer messages.
    """
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            name + " must be nonempty without surrounding whitespace"
        )

def normalized(name, values, tolerance=1e-9):
    """
    Require a nonempty distribution with values in [0, 1] and sum one.

    Applies to:
        Nominal association weights:
            sum(w_r) = 1

        Normalized appearance histograms:
            sum(histogram_bin) = 1

    Processing:
        Checks every value, then checks the sum with an absolute
        tolerance for floating-point rounding.

    Boundary:
        Does not normalize invalid input.
        Does not apply to effective masses w_r * q[r, i, j]:
        their sum may be below one or zero when evidence is unavailable.
    """
    finite("normalization tolerance", tolerance)

    if not 0 <= tolerance < 1:
        raise ValueError(
            "normalization tolerance must be in [0, 1)"
        )

    values = tuple(values)

    if not values:
        raise ValueError(name + " must not be empty")

    for index, value in enumerate(values):
        unit_interval(
            name + "[" + str(index) + "]",
            value,
        )

    total = math.fsum(values)

    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            name + " must sum to one; received " + str(total)
        )