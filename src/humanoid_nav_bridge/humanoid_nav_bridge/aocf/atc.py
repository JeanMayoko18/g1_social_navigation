#!/usr/bin/env python3
"""
Adaptive Twist Calibration (ATC).

ATC reconstructs a navigation-compatible body twist from consecutive
AOCF-corrected poses.

Responsibilities
----------------
1. Differentiate consecutive corrected poses.
2. Reconstruct angular velocity with wrapped heading differences.
3. Transform planar velocity from corrected odometry frame to body frame.
4. Filter and clamp the published twist components.
5. Force zero twist during stationary hold when requested.

ATC does not modify the corrected pose.
"""

import math
from dataclasses import dataclass


@dataclass
class ATCConfig:
    """Configuration parameters for Adaptive Twist Calibration."""

    enabled: bool = True

    max_vx: float = 0.8
    max_vy: float = 0.4
    max_wz: float = 1.2

    filter_output_vx: bool = True
    filter_output_vy: bool = True
    filter_output_wz: bool = True

    ema_vx_alpha: float = 0.20
    ema_vy_alpha: float = 0.15
    ema_wz_alpha: float = 0.25

    max_output_vx: float = 0.25
    max_output_vy: float = 0.08
    max_output_wz: float = 0.35


@dataclass
class ATCResult:
    """Complete ATC output for one corrected-pose transition."""

    vx: float
    vy: float
    wz: float

    vx_odom: float
    vy_odom: float
    wz_pose: float

    vx_body_raw: float
    vy_body_raw: float

    valid: bool
    forced_zero: bool


class AdaptiveTwistCalibration:
    """
    Reconstruct and calibrate planar body twist.
    """

    def __init__(self, config: ATCConfig):
        self.config = config

        self.vx_filtered = 0.0
        self.vy_filtered = 0.0
        self.wz_filtered = 0.0

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

    def reset(self) -> None:
        """Reset all temporal output filters."""
        self.vx_filtered = 0.0
        self.vy_filtered = 0.0
        self.wz_filtered = 0.0

    def _filter_vx(self, value: float) -> float:
        if not self.config.filter_output_vx:
            return self._clamp(
                value,
                -self.config.max_vx,
                self.config.max_vx,
            )

        alpha = self._clamp(
            self.config.ema_vx_alpha,
            0.0,
            1.0,
        )

        self.vx_filtered = (
            alpha * value
            + (1.0 - alpha) * self.vx_filtered
        )

        return self._clamp(
            self.vx_filtered,
            -self.config.max_output_vx,
            self.config.max_output_vx,
        )

    def _filter_vy(self, value: float) -> float:
        if not self.config.filter_output_vy:
            return self._clamp(
                value,
                -self.config.max_vy,
                self.config.max_vy,
            )

        alpha = self._clamp(
            self.config.ema_vy_alpha,
            0.0,
            1.0,
        )

        self.vy_filtered = (
            alpha * value
            + (1.0 - alpha) * self.vy_filtered
        )

        return self._clamp(
            self.vy_filtered,
            -self.config.max_output_vy,
            self.config.max_output_vy,
        )

    def _filter_wz(self, value: float) -> float:
        if not self.config.filter_output_wz:
            return self._clamp(
                value,
                -self.config.max_wz,
                self.config.max_wz,
            )

        alpha = self._clamp(
            self.config.ema_wz_alpha,
            0.0,
            1.0,
        )

        self.wz_filtered = (
            alpha * value
            + (1.0 - alpha) * self.wz_filtered
        )

        return self._clamp(
            self.wz_filtered,
            -self.config.max_output_wz,
            self.config.max_output_wz,
        )

    def calibrate(
        self,
        current_x: float,
        current_y: float,
        current_heading: float,
        previous_x: float,
        previous_y: float,
        previous_heading: float,
        dt: float,
        has_previous_output: bool,
        force_zero: bool = False,
    ) -> ATCResult:
        """
        Reconstruct and calibrate body-frame twist.
        """

        valid = (
            self.config.enabled
            and has_previous_output
            and dt is not None
            and dt > 1e-4
        )

        if not valid:
            return ATCResult(
                vx=0.0,
                vy=0.0,
                wz=0.0,

                vx_odom=0.0,
                vy_odom=0.0,
                wz_pose=0.0,

                vx_body_raw=0.0,
                vy_body_raw=0.0,

                valid=False,
                forced_zero=False,
            )

        vx_odom = (
            current_x - previous_x
        ) / dt

        vy_odom = (
            current_y - previous_y
        ) / dt

        wz_pose = self._wrap_angle(
            current_heading - previous_heading
        ) / dt

        c = math.cos(current_heading)
        s = math.sin(current_heading)

        # R(-heading) * v_odom
        vx_body_raw = (
            c * vx_odom
            + s * vy_odom
        )

        vy_body_raw = (
            -s * vx_odom
            + c * vy_odom
        )

        vx = self._filter_vx(vx_body_raw)
        vy = self._filter_vy(vy_body_raw)
        wz = self._filter_wz(wz_pose)

        if force_zero:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        return ATCResult(
            vx=vx,
            vy=vy,
            wz=wz,

            vx_odom=vx_odom,
            vy_odom=vy_odom,
            wz_pose=wz_pose,

            vx_body_raw=vx_body_raw,
            vy_body_raw=vy_body_raw,

            valid=True,
            forced_zero=force_zero,
        )