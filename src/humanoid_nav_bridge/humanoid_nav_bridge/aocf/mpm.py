#!/usr/bin/env python3
"""
Motion Prediction Model (MPM).

MPM predicts command-conditioned planar body motion and maintains
adaptive G1 command-response gains.

Responsibilities
----------------
1. Learn longitudinal and angular response gains.
2. Predict expected body-frame motion.
3. Build admissible velocity margins around the prediction.

MPM does not estimate MCI and does not fuse motion increments.
"""

import math
from dataclasses import dataclass


@dataclass
class MPMConfig:
    """Configuration of the Motion Prediction Model."""

    enable_response_learning: bool = True
    learning_alpha: float = 0.02

    straight_cmd_vx_min: float = 0.08
    straight_cmd_wz_max: float = 0.08
    turn_cmd_wz_min: float = 0.08

    vx_gain_min: float = 0.35
    vx_gain_max: float = 1.20

    wz_gain_min: float = 0.30
    wz_gain_max: float = 1.30

    initial_vx_gain: float = 0.60
    initial_wz_gain: float = 0.60

    max_expected_vx_margin: float = 0.06
    max_expected_vy: float = 0.025
    max_expected_wz_margin: float = 0.08
    max_expected_wz_straight: float = 0.008


@dataclass
class MPMPrediction:
    """Command-conditioned expected body-frame motion."""

    dx_body: float
    dy_body: float
    dyaw: float

    vx: float
    vy: float
    wz: float

    straight_command: bool


@dataclass
class MPMTolerances:
    """Admissible velocity margins around the MPM prediction."""

    vx_margin: float
    vy_margin: float
    wz_margin: float


class MotionPredictionModel:
    """
    Adaptive command-conditioned motion prediction model.
    """

    def __init__(self, config: MPMConfig):
        self.config = config

        self.vx_gain_est = self._clamp(
            config.initial_vx_gain,
            config.vx_gain_min,
            config.vx_gain_max,
        )

        self.wz_gain_est = self._clamp(
            config.initial_wz_gain,
            config.wz_gain_min,
            config.wz_gain_max,
        )

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(upper, value))

    def reset(self) -> None:
        """Reset learned response gains."""
        self.vx_gain_est = self._clamp(
            self.config.initial_vx_gain,
            self.config.vx_gain_min,
            self.config.vx_gain_max,
        )

        self.wz_gain_est = self._clamp(
            self.config.initial_wz_gain,
            self.config.wz_gain_min,
            self.config.wz_gain_max,
        )

    def is_straight_command(
        self,
        cmd_vx: float,
        cmd_wz: float,
    ) -> bool:
        """Classify a command as approximately straight walking."""

        return (
            abs(cmd_vx)
            >= self.config.straight_cmd_vx_min
            and abs(cmd_wz)
            <= self.config.straight_cmd_wz_max
        )

    def update_response_model(
        self,
        measured_vx: float,
        measured_wz: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ) -> None:
        """
        Update learned G1 response gains using a recent valid command.
        """

        if not self.config.enable_response_learning:
            return

        if not cmd_active:
            return

        values = [
            measured_vx,
            measured_wz,
            cmd_vx,
            cmd_wz,
        ]

        if not all(math.isfinite(v) for v in values):
            return

        alpha = self._clamp(
            self.config.learning_alpha,
            0.0,
            1.0,
        )

        if (
            abs(cmd_vx)
            >= self.config.straight_cmd_vx_min
        ):
            ratio_vx = (
                abs(measured_vx)
                / max(abs(cmd_vx), 1e-6)
            )

            ratio_vx = self._clamp(
                ratio_vx,
                self.config.vx_gain_min,
                self.config.vx_gain_max,
            )

            self.vx_gain_est = (
                (1.0 - alpha) * self.vx_gain_est
                + alpha * ratio_vx
            )

        if (
            abs(cmd_wz)
            >= self.config.turn_cmd_wz_min
        ):
            ratio_wz = (
                abs(measured_wz)
                / max(abs(cmd_wz), 1e-6)
            )

            ratio_wz = self._clamp(
                ratio_wz,
                self.config.wz_gain_min,
                self.config.wz_gain_max,
            )

            self.wz_gain_est = (
                (1.0 - alpha) * self.wz_gain_est
                + alpha * ratio_wz
            )

    def predict(
        self,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ) -> MPMPrediction:
        """
        Predict expected body-frame motion from a validated command.
        """

        if (
            dt is None
            or dt <= 1e-4
            or not cmd_active
        ):
            return MPMPrediction(
                dx_body=0.0,
                dy_body=0.0,
                dyaw=0.0,
                vx=0.0,
                vy=0.0,
                wz=0.0,
                straight_command=False,
            )

        straight_command = self.is_straight_command(
            cmd_vx,
            cmd_wz,
        )

        expected_vx = (
            cmd_vx * self.vx_gain_est
        )

        expected_vy = 0.0

        if straight_command:
            expected_wz = 0.0
        else:
            expected_wz = (
                cmd_wz * self.wz_gain_est
            )

        return MPMPrediction(
            dx_body=expected_vx * dt,
            dy_body=expected_vy * dt,
            dyaw=expected_wz * dt,
            vx=expected_vx,
            vy=expected_vy,
            wz=expected_wz,
            straight_command=straight_command,
        )

    def tolerances(
        self,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ) -> MPMTolerances:
        """
        Return the admissible velocity envelope around the prediction.
        """

        if not cmd_active:
            return MPMTolerances(
                vx_margin=0.0,
                vy_margin=0.0,
                wz_margin=0.0,
            )

        straight_command = self.is_straight_command(
            cmd_vx,
            cmd_wz,
        )

        if straight_command:
            wz_margin = max(
                0.0,
                self.config.max_expected_wz_straight,
            )
        else:
            wz_margin = max(
                0.0,
                self.config.max_expected_wz_margin,
            )

        return MPMTolerances(
            vx_margin=max(
                0.0,
                self.config.max_expected_vx_margin,
            ),
            vy_margin=max(
                0.0,
                self.config.max_expected_vy,
            ),
            wz_margin=wz_margin,
        )