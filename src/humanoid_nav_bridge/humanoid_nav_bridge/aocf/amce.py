#!/usr/bin/env python3
"""
Adaptive Motion Consistency Estimator (AMCE).

AMCE evaluates the consistency between measured and expected planar motion.

It computes:
    - instantaneous motion innovations,
    - temporally smoothed innovation errors,
    - component-wise Motion Consistency Indices.

AMCE does not modify or fuse motion increments.
"""
import math
from dataclasses import dataclass


@dataclass
class AMCEConfig:
    """
    Configuration parameters for AMCE.
    """

    enabled: bool = True
    alpha: float = 0.02

    mci_x_min: float = 0.45
    mci_y_min: float = 0.10
    mci_yaw_min: float = 0.01

    mci_x_sensitivity: float = 8.0
    mci_y_sensitivity: float = 16.0
    mci_yaw_sensitivity: float = 25.0


@dataclass
class AMCEResult:
    """
    Complete AMCE output for one odometry sample.
    """

    mci_x: float
    mci_y: float
    mci_yaw: float

    innovation_vx: float
    innovation_vy: float
    innovation_wz: float

    filtered_error_vx: float
    filtered_error_vy: float
    filtered_error_wz: float

    absolute_error_vx: float
    absolute_error_vy: float
    absolute_error_wz: float

class AdaptiveMotionConsistencyEstimator:
    """
    Component-wise motion consistency estimator.

    The translational measured and expected velocities must be expressed
    in the same reference frame.
    """

    def __init__(self, config: AMCEConfig):
        self.config = config

        # Temporally smoothed innovation errors.
        self.err_vx = 0.0
        self.err_vy = 0.0
        self.err_wz = 0.0

        # Initial confidence.
        self.mci_x = 1.0
        self.mci_y = 1.0
        self.mci_yaw = 1.0

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        """Clamp a scalar value."""
        return max(lower, min(upper, value))

    def reset(self) -> None:
        """
        Reset all temporal AMCE states.

        This may be used after a complete odometry reinitialization.
        """
        self.err_vx = 0.0
        self.err_vy = 0.0
        self.err_wz = 0.0

        self.mci_x = 1.0
        self.mci_y = 1.0
        self.mci_yaw = 1.0

    def estimate(
        self,
        measured_vx: float,
        measured_vy: float,
        measured_wz: float,
        expected_vx: float,
        expected_vy: float,
        expected_wz: float,
        expected_vx_margin: float = 0.0,
        expected_vy_margin: float = 0.0,
        expected_wz_margin: float = 0.0,
    ) -> AMCEResult:
        """
        Estimate component-wise Motion Consistency Indices.

        The absolute error measures the full difference between measured
        and expected motion.

        The innovation measures only the part of this error that exceeds
        the admissible MPM tolerance envelope.
        """

        # Ensure non-negative tolerance margins.
        margin_vx = max(0.0, expected_vx_margin)
        margin_vy = max(0.0, expected_vy_margin)
        margin_wz = max(0.0, expected_wz_margin)

        # Full measured-versus-expected errors.
        absolute_error_vx = abs(
            measured_vx - expected_vx
        )
        absolute_error_vy = abs(
            measured_vy - expected_vy
        )
        absolute_error_wz = abs(
            measured_wz - expected_wz
        )

        # Residual innovations outside the admissible MPM envelope.
        innovation_vx = max(
            0.0,
            absolute_error_vx - margin_vx,
        )
        innovation_vy = max(
            0.0,
            absolute_error_vy - margin_vy,
        )
        innovation_wz = max(
            0.0,
            absolute_error_wz - margin_wz,
        )

        if not self.config.enabled:
            self.mci_x = 1.0
            self.mci_y = 1.0
            self.mci_yaw = 1.0

            return AMCEResult(
                mci_x=self.mci_x,
                mci_y=self.mci_y,
                mci_yaw=self.mci_yaw,

                absolute_error_vx=absolute_error_vx,
                absolute_error_vy=absolute_error_vy,
                absolute_error_wz=absolute_error_wz,

                innovation_vx=innovation_vx,
                innovation_vy=innovation_vy,
                innovation_wz=innovation_wz,

                filtered_error_vx=self.err_vx,
                filtered_error_vy=self.err_vy,
                filtered_error_wz=self.err_wz,
            )

        alpha = self.clamp(
            self.config.alpha,
            0.0,
            1.0,
        )

        self.err_vx = (
            (1.0 - alpha) * self.err_vx
            + alpha * innovation_vx
        )

        self.err_vy = (
            (1.0 - alpha) * self.err_vy
            + alpha * innovation_vy
        )

        self.err_wz = (
            (1.0 - alpha) * self.err_wz
            + alpha * innovation_wz
        )

        raw_mci_x = math.exp(
            -self.config.mci_x_sensitivity
            * self.err_vx
        )

        raw_mci_y = math.exp(
            -self.config.mci_y_sensitivity
            * self.err_vy
        )

        raw_mci_yaw = math.exp(
            -self.config.mci_yaw_sensitivity
            * self.err_wz
        )

        self.mci_x = self.clamp(
            raw_mci_x,
            self.config.mci_x_min,
            1.0,
        )

        self.mci_y = self.clamp(
            raw_mci_y,
            self.config.mci_y_min,
            1.0,
        )

        self.mci_yaw = self.clamp(
            raw_mci_yaw,
            self.config.mci_yaw_min,
            1.0,
        )

        return AMCEResult(
            mci_x=self.mci_x,
            mci_y=self.mci_y,
            mci_yaw=self.mci_yaw,

            absolute_error_vx=absolute_error_vx,
            absolute_error_vy=absolute_error_vy,
            absolute_error_wz=absolute_error_wz,

            innovation_vx=innovation_vx,
            innovation_vy=innovation_vy,
            innovation_wz=innovation_wz,

            filtered_error_vx=self.err_vx,
            filtered_error_vy=self.err_vy,
            filtered_error_wz=self.err_wz,
        )

        if not self.config.enabled:
            # AMCE ablation:
            # preserve full confidence and do not update temporal errors.
            self.mci_x = 1.0
            self.mci_y = 1.0
            self.mci_yaw = 1.0

            return AMCEResult(
                mci_x=self.mci_x,
                mci_y=self.mci_y,
                mci_yaw=self.mci_yaw,
                innovation_vx=innovation_vx,
                innovation_vy=innovation_vy,
                innovation_wz=innovation_wz,
                filtered_error_vx=self.err_vx,
                filtered_error_vy=self.err_vy,
                filtered_error_wz=self.err_wz,
            )

        alpha = self.clamp(self.config.alpha, 0.0, 1.0)

        # Exponential temporal smoothing of the innovations.
        self.err_vx = (
            (1.0 - alpha) * self.err_vx
            + alpha * innovation_vx
        )

        self.err_vy = (
            (1.0 - alpha) * self.err_vy
            + alpha * innovation_vy
        )

        self.err_wz = (
            (1.0 - alpha) * self.err_wz
            + alpha * innovation_wz
        )

        # Convert smoothed errors into confidence indices.
        raw_mci_x = math.exp(
            -self.config.mci_x_sensitivity * self.err_vx
        )

        raw_mci_y = math.exp(
            -self.config.mci_y_sensitivity * self.err_vy
        )

        raw_mci_yaw = math.exp(
            -self.config.mci_yaw_sensitivity * self.err_wz
        )

        # Preserve a configurable minimum contribution from raw odometry.
        self.mci_x = self.clamp(
            raw_mci_x,
            self.config.mci_x_min,
            1.0,
        )

        self.mci_y = self.clamp(
            raw_mci_y,
            self.config.mci_y_min,
            1.0,
        )

        self.mci_yaw = self.clamp(
            raw_mci_yaw,
            self.config.mci_yaw_min,
            1.0,
        )

        return AMCEResult(
            mci_x=self.mci_x,
            mci_y=self.mci_y,
            mci_yaw=self.mci_yaw,
            innovation_vx=innovation_vx,
            innovation_vy=innovation_vy,
            innovation_wz=innovation_wz,
            filtered_error_vx=self.err_vx,
            filtered_error_vy=self.err_vy,
            filtered_error_wz=self.err_wz,
        )
