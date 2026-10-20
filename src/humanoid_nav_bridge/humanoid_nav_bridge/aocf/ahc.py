#!/usr/bin/env python3
"""
Adaptive Heading Calibration (AHC).

AHC calibrates the planar navigation heading from the adapted yaw
increment produced by AFI.

Responsibilities
----------------
1. Bound yaw accumulation during commanded straight motion.
2. Apply the startup ramp to the accepted yaw increment.
3. Reconstruct the calibrated navigation heading.
4. Optionally support command-predicted heading reconstruction.

AHC does not estimate motion consistency and does not fuse increments.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AHCConfig:
    """Configuration parameters for Adaptive Heading Calibration."""

    enabled: bool = True

    yaw_correction_mode: str = "raw_increment"

    enable_straight_yaw_gate: bool = True
    straight_yaw_gate_wz: float = 0.008

    straight_cmd_vx_min: float = 0.08
    straight_cmd_wz_max: float = 0.08

    yaw_observation_gain: float = 0.03
    max_yaw_observation_correction: float = 0.0015


@dataclass
class AHCResult:
    """Complete AHC output for one odometry update."""

    calibrated_heading: float

    dyaw_before_gate: float
    dyaw_after_gate: float
    final_dyaw: float

    straight_gate_active: bool
    startup_alpha: float
    mode_used: str

    predicted_heading: Optional[float]
    yaw_observation_error: Optional[float]
    yaw_observation_correction: Optional[float]


class AdaptiveHeadingCalibration:
    """
    Calibrate and reconstruct the planar navigation heading.
    """

    def __init__(self, config: AHCConfig):
        self.config = config

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    def _is_straight_command(
        self,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ) -> bool:
        if not cmd_active:
            return False

        return (
            abs(cmd_vx)
            >= self.config.straight_cmd_vx_min
            and abs(cmd_wz)
            <= self.config.straight_cmd_wz_max
        )

    def apply_straight_yaw_gate(
        self,
        dyaw: float,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Bound yaw accumulation during a recent straight command.
        """

        if not self.config.enabled:
            return dyaw, False

        if not self.config.enable_straight_yaw_gate:
            return dyaw, False

        if dt is None or dt <= 1e-4:
            return dyaw, False

        straight_gate_active = self._is_straight_command(
            cmd_vx=cmd_vx,
            cmd_wz=cmd_wz,
            cmd_active=cmd_active,
        )

        if not straight_gate_active:
            return dyaw, False

        max_dyaw = (
            max(0.0, self.config.straight_yaw_gate_wz)
            * dt
        )

        gated_dyaw = self._clamp(
            dyaw,
            -max_dyaw,
            max_dyaw,
        )

        return gated_dyaw, True

    def calibrate(
        self,
        previous_heading: float,
        adapted_dyaw: float,
        startup_alpha: float,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
        raw_heading: float,
        expected_dyaw: Optional[float],
    ) -> AHCResult:
        """
        Calibrate the yaw increment and reconstruct the navigation heading.

        The active article configuration is normally ``raw_increment``.
        """

        startup_alpha = self._clamp(
            startup_alpha,
            0.0,
            1.0,
        )

        dyaw_before_gate = adapted_dyaw

        gated_dyaw, straight_gate_active = (
            self.apply_straight_yaw_gate(
                dyaw=adapted_dyaw,
                dt=dt,
                cmd_vx=cmd_vx,
                cmd_wz=cmd_wz,
                cmd_active=cmd_active,
            )
        )

        final_dyaw = startup_alpha * gated_dyaw

        predicted_heading = None
        yaw_observation_error = None
        yaw_observation_correction = None

        use_predicted_mode = (
            self.config.enabled
            and self.config.yaw_correction_mode
            == "predicted_cmd"
            and expected_dyaw is not None
        )

        if use_predicted_mode:
            predicted_heading = self._wrap_angle(
                previous_heading + expected_dyaw
            )

            yaw_observation_error = self._wrap_angle(
                raw_heading - predicted_heading
            )

            yaw_observation_correction = self._clamp(
                self.config.yaw_observation_gain
                * yaw_observation_error,
                -self.config.max_yaw_observation_correction,
                self.config.max_yaw_observation_correction,
            )

            calibrated_heading = self._wrap_angle(
                predicted_heading
                + yaw_observation_correction
            )

            mode_used = "predicted_cmd"

        else:
            calibrated_heading = self._wrap_angle(
                previous_heading + final_dyaw
            )

            mode_used = "raw_increment"

        return AHCResult(
            calibrated_heading=calibrated_heading,

            dyaw_before_gate=dyaw_before_gate,
            dyaw_after_gate=gated_dyaw,
            final_dyaw=final_dyaw,

            straight_gate_active=straight_gate_active,
            startup_alpha=startup_alpha,
            mode_used=mode_used,

            predicted_heading=predicted_heading,
            yaw_observation_error=yaw_observation_error,
            yaw_observation_correction=(
                yaw_observation_correction
            ),
        )