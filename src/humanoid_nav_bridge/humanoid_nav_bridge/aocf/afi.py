#!/usr/bin/env python3
"""
Adaptive Increment Fusion (AIF).
Adaptive Increment Calibration (AIC)

AIF combines measured and expected motion increments using component-wise
Motion Consistency Indices produced by AMCE.

AIF does not estimate confidence and does not maintain temporal state.
"""

from dataclasses import dataclass


@dataclass
class MotionIncrement:
    """
    Planar motion increment.
    """

    dx: float
    dy: float
    dyaw: float


@dataclass
class MotionConfidence:
    """
    Component-wise confidence associated with a motion increment.
    """

    x: float
    y: float
    yaw: float


@dataclass
class AFIResult:
    """
    AFI output for one motion increment.
    """

    increment: MotionIncrement
    enabled: bool


class AdaptiveFusionIntegrator:
    """
    Component-wise adaptive increment fusion.

    Measured and expected translational increments must be expressed
    in the same reference frame.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable AIF at runtime."""
        self.enabled = bool(enabled)

    def fuse(
        self,
        measured: MotionIncrement,
        expected: MotionIncrement,
        confidence: MotionConfidence,
    ) -> AFIResult:
        """
        Fuse measured and expected motion increments.

        When AIF is disabled, the measured increment is returned unchanged.
        This behavior supports clean ablation experiments.
        """

        if not self.enabled:
            return AFIResult(
                increment=MotionIncrement(
                    dx=measured.dx,
                    dy=measured.dy,
                    dyaw=measured.dyaw,
                ),
                enabled=False,
            )

        adapted_dx = (
            confidence.x * measured.dx
            + (1.0 - confidence.x) * expected.dx
        )

        adapted_dy = (
            confidence.y * measured.dy
            + (1.0 - confidence.y) * expected.dy
        )

        adapted_dyaw = (
            confidence.yaw * measured.dyaw
            + (1.0 - confidence.yaw) * expected.dyaw
        )

        return AFIResult(
            increment=MotionIncrement(
                dx=adapted_dx,
                dy=adapted_dy,
                dyaw=adapted_dyaw,
            ),
            enabled=True,
        )