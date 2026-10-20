"""
Evaluate configurable runtime compensation models.

Supported representations
-------------------------
polynomial:

    C(u) = c0 + c1*u + c2*u^2 + ... + cn*u^n

piecewise_linear:

    C(u) is linearly interpolated between configured points.

The module contains no robot-specific parameters and performs no ROS
communication.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# INPUT / CONFIGURATION
# =============================================================================


class CompensationModelType(str, Enum):
    """Supported runtime mathematical representations."""

    POLYNOMIAL = "polynomial"
    PIECEWISE_LINEAR = "piecewise_linear"


class OutsideDomainPolicy(str, Enum):
    """Behavior when the input lies outside the validated domain."""

    DISABLE = "disable"
    CLAMP = "clamp"


@dataclass(frozen=True)
class CompensationModel:
    """Store one validated runtime compensation model."""

    model_type: CompensationModelType

    domain_minimum: float
    domain_maximum: float

    outside_domain_policy: OutsideDomainPolicy

    polynomial_coefficients: tuple[float, ...] = ()

    input_points: tuple[float, ...] = ()
    output_points: tuple[float, ...] = ()


# =============================================================================
# VALIDATION
# =============================================================================


def _require_finite(
    value: float,
    *,
    name: str,
) -> float:
    """Require one finite numeric value."""

    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def validate_compensation_model(
    model: CompensationModel,
) -> None:
    """Validate one complete compensation model definition."""

    if model.domain_minimum > model.domain_maximum:
        raise ValueError(
            "Compensation model domain minimum must not exceed maximum."
        )

    if model.model_type is CompensationModelType.POLYNOMIAL:
        if not model.polynomial_coefficients:
            raise ValueError(
                "Polynomial model requires at least one coefficient."
            )

        for index, coefficient in enumerate(
            model.polynomial_coefficients
        ):
            _require_finite(
                coefficient,
                name=f"polynomial_coefficients[{index}]",
            )

        return

    if model.model_type is CompensationModelType.PIECEWISE_LINEAR:
        if len(model.input_points) < 2:
            raise ValueError(
                "Piecewise-linear model requires at least two points."
            )

        if len(model.input_points) != len(
            model.output_points
        ):
            raise ValueError(
                "Input and output point counts must match."
            )

        for index, input_value in enumerate(
            model.input_points
        ):
            _require_finite(
                input_value,
                name=f"input_points[{index}]",
            )

        for index, output_value in enumerate(
            model.output_points
        ):
            _require_finite(
                output_value,
                name=f"output_points[{index}]",
            )

        if any(
            right <= left
            for left, right in zip(
                model.input_points,
                model.input_points[1:],
            )
        ):
            raise ValueError(
                "Piecewise-linear input points must be strictly increasing."
            )

        return

    raise ValueError(
        f"Unsupported compensation model type: {model.model_type}"
    )


# =============================================================================
# DOMAIN PROCESSING
# =============================================================================


def resolve_domain_input(
    *,
    input_value: float,
    domain_minimum: float,
    domain_maximum: float,
    policy: OutsideDomainPolicy,
) -> float | None:
    """
    Apply the configured outside-domain policy.

    disable:
        Return None outside the validated domain.

    clamp:
        Replace values below or above the domain by the nearest boundary.
    """

    value = _require_finite(
        input_value,
        name="input_value",
    )

    if domain_minimum <= value <= domain_maximum:
        return value

    if policy is OutsideDomainPolicy.DISABLE:
        return None

    if policy is OutsideDomainPolicy.CLAMP:
        return min(
            max(
                value,
                domain_minimum,
            ),
            domain_maximum,
        )

    raise ValueError(
        f"Unsupported outside-domain policy: {policy}"
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================


def evaluate_polynomial(
    *,
    input_value: float,
    coefficients: tuple[float, ...],
) -> float:
    """
    Evaluate a polynomial using Horner's method.

    Coefficient order:

        coefficients = (c0, c1, ..., cn)

    Formula:

        C(u) = c0 + c1*u + ... + cn*u^n
    """

    result = 0.0

    for coefficient in reversed(
        coefficients
    ):
        result = (
            result * input_value
            + coefficient
        )

    return result


def evaluate_piecewise_linear(
    *,
    input_value: float,
    input_points: tuple[float, ...],
    output_points: tuple[float, ...],
) -> float:
    """
    Evaluate linear interpolation between two neighboring points.

    Formula:

        ratio = (u - u0) / (u1 - u0)

        C(u) = C0 + ratio * (C1 - C0)
    """

    right_index = bisect.bisect_right(
        input_points,
        input_value,
    )

    if right_index == 0:
        return output_points[0]

    if right_index >= len(
        input_points
    ):
        return output_points[-1]

    left_index = (
        right_index
        - 1
    )

    input_left = input_points[
        left_index
    ]

    input_right = input_points[
        right_index
    ]

    output_left = output_points[
        left_index
    ]

    output_right = output_points[
        right_index
    ]

    ratio = (
        input_value
        - input_left
    ) / (
        input_right
        - input_left
    )

    return (
        output_left
        + ratio
        * (
            output_right
            - output_left
        )
    )


def evaluate_compensation_model(
    *,
    model: CompensationModel,
    input_value: float,
) -> float:
    """Validate and evaluate one configured compensation model."""

    validate_compensation_model(
        model
    )

    resolved_input = resolve_domain_input(
        input_value=input_value,
        domain_minimum=model.domain_minimum,
        domain_maximum=model.domain_maximum,
        policy=model.outside_domain_policy,
    )

    if resolved_input is None:
        return 0.0

    if model.model_type is CompensationModelType.POLYNOMIAL:
        result = evaluate_polynomial(
            input_value=resolved_input,
            coefficients=model.polynomial_coefficients,
        )

    elif model.model_type is CompensationModelType.PIECEWISE_LINEAR:
        result = evaluate_piecewise_linear(
            input_value=resolved_input,
            input_points=model.input_points,
            output_points=model.output_points,
        )

    else:
        raise ValueError(
            f"Unsupported compensation model type: {model.model_type}"
        )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            "Compensation model produced a non-finite result."
        )

    return result