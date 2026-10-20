#!/usr/bin/env python3

"""
locomotion_calibration_node.py

Purpose
-------
Apply configurable locomotion calibration corrections to planar Twist commands
before forwarding them to the robot locomotion interface.

Calibration architecture
------------------------
All calibration layers are independently configurable from ROS parameters.

Straight-motion corrections:
    - Forward heading compensation.
    - Forward lateral compensation.

Pure-rotation corrections:
    - Direction-dependent longitudinal compensation for clockwise and
      counterclockwise rotations.
    - Direction-dependent lateral compensation for clockwise and
      counterclockwise rotations.
    - Direction-dependent angular rotation scaling.

Global correction:
    - Translation scaling.

Command flow
------------
                        LAUNCH / YAML
                             |
        +--------------------+--------------------+
        |                    |                    |
    activation             values             thresholds
        |                    |                    |
        +--------------------+--------------------+
                             |
                             v
                 locomotion_calibration_node
                             |
       +---------------------+---------------------+
       |                     |                     |
    STRAIGHT              ROTATION              GLOBAL
       |                     |                     |
 heading comp.          CW/CCW scale       translation scale
 lateral comp.          CW/CCW vx comp.
                        CW/CCW vy comp.
                             |
                             v
                  /cmd_vel_nav_scaled

Processing principles
---------------------
The node first identifies the motion context from the incoming Twist command
and then applies only the corrections that are enabled and applicable to that
motion.

Straight-motion compensations are applied only during intended forward motion.

Rotation-translation compensations are applied only during intended pure
rotations, where the incoming longitudinal and lateral commands are below the
configured thresholds and the angular command is active.

Rotation scaling modifies only the angular command and remains independent from
the translation compensation applied during rotation.

Each correction can therefore be estimated, enabled, disabled, and validated
independently without modifying the node implementation.

Design objective
----------------
Future calibration and validation experiments should require changes only to
launch/YAML parameters, not to the locomotion calibration node itself.
"""

from __future__ import annotations

import time

import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64

from cmd_vel_qos_adapter.compensation_models import (
    CompensationModel,
    CompensationModelType,
    OutsideDomainPolicy,
    evaluate_compensation_model,
    validate_compensation_model,
)

class LocomotionCalibrationNode(Node):
    """Apply configurable locomotion calibration corrections."""

    def __init__(self) -> None:
        super().__init__(
            "locomotion_calibration_node"
        )

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/cmd_vel",
        )

        self.declare_parameter(
            "output_topic",
            "/cmd_vel_nav_scaled",
        )

        self.declare_parameter(
            "enable_calibration",
            False,
        )

        # ---------------------------------------------------------------------
        # Latency diagnostics
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "enable_latency_diagnostics",
            True,
        )

        self.declare_parameter(
            "computation_latency_topic",
            "/locomotion_calibration/computation_latency_ms",
        )

        self.declare_parameter(
            "callback_latency_topic",
            "/locomotion_calibration/callback_latency_ms",
        )

        # ---------------------------------------------------------------------
        # Calibration reference commands
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "forward_compensation_reference_vx",
            0.12,
        )

        self.declare_parameter(
            "rotation_translation_reference_wz",
            0.20,
        )

        self.declare_parameter(
            "allow_compensation_extrapolation",
            False,
        )

        # ---------------------------------------------------------------------
        # Forward heading compensation
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "enable_forward_heading_compensation",
            False,
        )

        self.declare_parameter(
            "forward_heading_compensation_wz",
            0.0,
        )

        self.declare_parameter(
            "forward_heading_compensation_min_vx",
            0.08,
        )

        # ---------------------------------------------------------------------
        # Generic forward heading compensation model
        # ---------------------------------------------------------------------
        #
        # legacy_reference_scaled preserves the previous behavior:
        #
        #     C(vx) = C_ref * min(vx / vx_ref, 1)
        #
        # polynomial evaluates:
        #
        #     C(vx) = c0 + c1*vx + c2*vx^2 + ...
        #
        # piecewise_linear interpolates configured input/output points.
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "forward_heading_compensation_model_type",
            "legacy_reference_scaled",
        )

        self.declare_parameter(
            "forward_heading_compensation_coefficients",
            [0.0],
        )

        self.declare_parameter(
            "forward_heading_compensation_input_points",
            [0.0, 1.0],
        )

        self.declare_parameter(
            "forward_heading_compensation_output_points",
            [0.0, 0.0],
        )

        self.declare_parameter(
            "forward_heading_compensation_domain_min",
            0.0,
        )

        self.declare_parameter(
            "forward_heading_compensation_domain_max",
            1.0,
        )

        self.declare_parameter(
            "forward_heading_compensation_outside_domain_policy",
            "disable",
        )

        # ---------------------------------------------------------------------
        # Forward lateral compensation
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "enable_forward_lateral_compensation",
            False,
        )

        self.declare_parameter(
            "forward_lateral_compensation_vy",
            0.0,
        )

        self.declare_parameter(
            "forward_lateral_compensation_min_vx",
            0.08,
        )

        # ---------------------------------------------------------------------
        # Generic forward lateral compensation model
        # ---------------------------------------------------------------------
        #
        # legacy_reference_scaled preserves the previous behavior:
        #
        #     C(vx) = C_ref * min(vx / vx_ref, 1)
        #
        # polynomial evaluates:
        #
        #     C(vx) = c0 + c1*vx + c2*vx^2 + ...
        #
        # piecewise_linear interpolates configured input/output points.
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "forward_lateral_compensation_model_type",
            "legacy_reference_scaled",
        )

        self.declare_parameter(
            "forward_lateral_compensation_coefficients",
            [0.0],
        )

        self.declare_parameter(
            "forward_lateral_compensation_input_points",
            [0.0, 1.0],
        )

        self.declare_parameter(
            "forward_lateral_compensation_output_points",
            [0.0, 0.0],
        )

        self.declare_parameter(
            "forward_lateral_compensation_domain_min",
            0.0,
        )

        self.declare_parameter(
            "forward_lateral_compensation_domain_max",
            1.0,
        )

        self.declare_parameter(
            "forward_lateral_compensation_outside_domain_policy",
            "disable",
        )

        # ---------------------------------------------------------------------
        # Rotation scaling
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "enable_rotation_scaling",
            False,
        )

        self.declare_parameter(
            "clockwise_rotation_scale",
            1.0,
        )

        self.declare_parameter(
            "counterclockwise_rotation_scale",
            1.0,
        )

        # ---------------------------------------------------------------------
        # Translation scaling
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "enable_translation_scaling",
            False,
        )

        self.declare_parameter(
            "translation_scale",
            1.0,
        )

        # ---------------------------------------------------------------------
        # Rotation translation compensation
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "enable_rotation_translation_compensation",
            False,
        )

        # Longitudinal compensation during pure rotation
        self.declare_parameter(
            "clockwise_rotation_vx_compensation",
            0.0,
        )

        self.declare_parameter(
            "counterclockwise_rotation_vx_compensation",
            0.0,
        )

        # Lateral compensation during pure rotation
        self.declare_parameter(
            "clockwise_rotation_vy_compensation",
            0.0,
        )

        self.declare_parameter(
            "counterclockwise_rotation_vy_compensation",
            0.0,
        )

        # Pure-rotation detection thresholds
        self.declare_parameter(
            "rotation_translation_max_linear_command",
            1e-6,
        )

        self.declare_parameter(
            "rotation_translation_min_wz",
            1e-6,
        )

        # =====================================================================
        # READ CONFIGURATION
        # =====================================================================

        self.input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )

        self.output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )

        self.enable_calibration = bool(
            self.get_parameter(
                "enable_calibration"
            ).value
        )

        self.forward_compensation_reference_vx = float(
            self.get_parameter(
                "forward_compensation_reference_vx"
            ).value
        )

        self.rotation_translation_reference_wz = float(
            self.get_parameter(
                "rotation_translation_reference_wz"
            ).value
        )

        self.allow_compensation_extrapolation = bool(
            self.get_parameter(
                "allow_compensation_extrapolation"
            ).value
        )

        if self.forward_compensation_reference_vx <= 0.0:
            raise ValueError(
                "forward_compensation_reference_vx must be > 0."
            )

        if self.rotation_translation_reference_wz <= 0.0:
            raise ValueError(
                "rotation_translation_reference_wz must be > 0."
            )

        self.enable_forward_heading_compensation = bool(
            self.get_parameter(
                "enable_forward_heading_compensation"
            ).value
        )

        self.forward_heading_compensation_wz = float(
            self.get_parameter(
                "forward_heading_compensation_wz"
            ).value
        )

        self.forward_heading_compensation_min_vx = float(
            self.get_parameter(
                "forward_heading_compensation_min_vx"
            ).value
        )

        self.enable_forward_lateral_compensation = bool(
            self.get_parameter(
                "enable_forward_lateral_compensation"
            ).value
        )

        self.forward_lateral_compensation_vy = float(
            self.get_parameter(
                "forward_lateral_compensation_vy"
            ).value
        )

        self.forward_lateral_compensation_min_vx = float(
            self.get_parameter(
                "forward_lateral_compensation_min_vx"
            ).value
        )


        (
            self.forward_heading_compensation_model_type,
            self.forward_heading_compensation_model,
        ) = self.load_compensation_model(
            parameter_prefix=(
                "forward_heading_compensation"
            ),
        )

        (
            self.forward_lateral_compensation_model_type,
            self.forward_lateral_compensation_model,
        ) = self.load_compensation_model(
            parameter_prefix=(
                "forward_lateral_compensation"
            ),
        )

        self.enable_rotation_scaling = bool(
            self.get_parameter(
                "enable_rotation_scaling"
            ).value
        )

        self.clockwise_rotation_scale = float(
            self.get_parameter(
                "clockwise_rotation_scale"
            ).value
        )

        self.counterclockwise_rotation_scale = float(
            self.get_parameter(
                "counterclockwise_rotation_scale"
            ).value
        )

        self.enable_translation_scaling = bool(
            self.get_parameter(
                "enable_translation_scaling"
            ).value
        )

        self.translation_scale = float(
            self.get_parameter(
                "translation_scale"
            ).value
        )

        self.enable_rotation_translation_compensation = bool(
            self.get_parameter(
                "enable_rotation_translation_compensation"
            ).value
        )

        self.clockwise_rotation_vx_compensation = float(
            self.get_parameter(
                "clockwise_rotation_vx_compensation"
            ).value
        )

        self.counterclockwise_rotation_vx_compensation = float(
            self.get_parameter(
                "counterclockwise_rotation_vx_compensation"
            ).value
        )

        self.clockwise_rotation_vy_compensation = float(
            self.get_parameter(
                "clockwise_rotation_vy_compensation"
            ).value
        )

        self.counterclockwise_rotation_vy_compensation = float(
            self.get_parameter(
                "counterclockwise_rotation_vy_compensation"
            ).value
        )

        self.rotation_translation_max_linear_command = float(
            self.get_parameter(
                "rotation_translation_max_linear_command"
            ).value
        )

        self.rotation_translation_min_wz = float(
            self.get_parameter(
                "rotation_translation_min_wz"
            ).value
        )

        self.enable_latency_diagnostics = bool(
            self.get_parameter(
                "enable_latency_diagnostics"
            ).value
        )

        self.computation_latency_topic = str(
            self.get_parameter(
                "computation_latency_topic"
            ).value
        )

        self.callback_latency_topic = str(
            self.get_parameter(
                "callback_latency_topic"
            ).value
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.subscription = self.create_subscription(
            Twist,
            self.input_topic,
            self.command_callback,
            10,
        )

        self.publisher = self.create_publisher(
            Twist,
            self.output_topic,
            10,
        )

        if self.enable_latency_diagnostics:
            self.computation_latency_publisher = (
                self.create_publisher(
                    Float64,
                    self.computation_latency_topic,
                    10,
                )
            )

            self.callback_latency_publisher = (
                self.create_publisher(
                    Float64,
                    self.callback_latency_topic,
                    10,
                )
            )
        else:
            self.computation_latency_publisher = None
            self.callback_latency_publisher = None

        self.get_logger().info(
            "\n"
            "=========================================================\n"
            "          ROBOT LOCOMOTION CALIBRATION NODE\n"
            "=========================================================\n"
            f" Input topic                         : {self.input_topic}\n"
            f" Output topic                        : {self.output_topic}\n"
            f" Calibration enabled                : {self.enable_calibration}\n"
            "\n"
            
            " STRAIGHT-MOTION CALIBRATION\n"
            "---------------------------------------------------------\n"

            f" Forward heading compensation       : "
            f"{self.enable_forward_heading_compensation}\n"
            f" Forward heading model              : "
            f"{self.forward_heading_compensation_model_type}\n"
            f" Forward heading compensation wz    : "
            f"{self.forward_heading_compensation_wz:+.4f} rad/s\n"
            f" Forward heading minimum vx         : "
            f"{self.forward_heading_compensation_min_vx:.4f} m/s\n"
            f" Forward lateral compensation       : "
            f"{self.enable_forward_lateral_compensation}\n"
            f" Forward lateral model              : "
            f"{self.forward_lateral_compensation_model_type}\n"
            f" Forward lateral compensation vy    : "
            f"{self.forward_lateral_compensation_vy:+.4f} m/s\n"
            f" Forward lateral minimum vx         : "
            f"{self.forward_lateral_compensation_min_vx:.4f} m/s\n"
            "\n"

            " ROTATION-TRANSLATION COMPENSATION\n"
            "---------------------------------------------------------\n"
            f" Enabled                            : "
            f"{self.enable_rotation_translation_compensation}\n"
            f" CW rotation vx compensation        : "
            f"{self.clockwise_rotation_vx_compensation:+.5f} m/s\n"
            f" CW rotation vy compensation        : "
            f"{self.clockwise_rotation_vy_compensation:+.5f} m/s\n"
            f" CCW rotation vx compensation       : "
            f"{self.counterclockwise_rotation_vx_compensation:+.5f} m/s\n"
            f" CCW rotation vy compensation       : "
            f"{self.counterclockwise_rotation_vy_compensation:+.5f} m/s\n"
            f" Maximum incoming linear command    : "
            f"{self.rotation_translation_max_linear_command:.6f} m/s\n"
            f" Minimum rotation command wz        : "
            f"{self.rotation_translation_min_wz:.6f} rad/s\n"
            "\n"
            " ROTATION SCALING\n"
            "---------------------------------------------------------\n"
            f" Enabled                            : {self.enable_rotation_scaling}\n"
            f" CW rotation scale                  : "
            f"{self.clockwise_rotation_scale:.6f}\n"
            f" CCW rotation scale                 : "
            f"{self.counterclockwise_rotation_scale:.6f}\n"
            "\n"
            " TRANSLATION SCALING\n"
            "---------------------------------------------------------\n"
            f" Enabled                            : {self.enable_translation_scaling}\n"
            f" Translation scale                  : "
            f"{self.translation_scale:.6f}\n"
            "========================================================="
        )

    # =========================================================================
    # PROCESSING
    # =========================================================================
    
    def load_compensation_model(
        self,
        *,
        parameter_prefix: str,
    ) -> tuple[str, CompensationModel | None]:
        """
        Load and validate one generic compensation model from ROS parameters.

        Parameter convention:

            <prefix>_model_type
            <prefix>_coefficients
            <prefix>_input_points
            <prefix>_output_points
            <prefix>_domain_min
            <prefix>_domain_max
            <prefix>_outside_domain_policy

        The legacy_reference_scaled model is evaluated by the historical
        runtime path and therefore does not create a CompensationModel.
        """

        model_type = str(
            self.get_parameter(
                f"{parameter_prefix}_model_type"
            ).value
        )

        if model_type == "legacy_reference_scaled":
            return model_type, None

        model = CompensationModel(
            model_type=CompensationModelType(
                model_type
            ),

            domain_minimum=float(
                self.get_parameter(
                    f"{parameter_prefix}_domain_min"
                ).value
            ),

            domain_maximum=float(
                self.get_parameter(
                    f"{parameter_prefix}_domain_max"
                ).value
            ),

            outside_domain_policy=OutsideDomainPolicy(
                self.get_parameter(
                    f"{parameter_prefix}_outside_domain_policy"
                ).value
            ),

            polynomial_coefficients=tuple(
                float(value)
                for value in self.get_parameter(
                    f"{parameter_prefix}_coefficients"
                ).value
            ),

            input_points=tuple(
                float(value)
                for value in self.get_parameter(
                    f"{parameter_prefix}_input_points"
                ).value
            ),

            output_points=tuple(
                float(value)
                for value in self.get_parameter(
                    f"{parameter_prefix}_output_points"
                ).value
            ),
        )

        validate_compensation_model(
            model
        )

        return model_type, model
    
    def calculate_compensation_ratio(
        self,
        command_magnitude: float,
        reference_magnitude: float,
    ) -> float:
        """
        Input:
            Magnitude of the current motion command.
            Magnitude of the command used during experimental calibration.

        Processing:
            Computes the ratio between the current command and the
            experimentally calibrated reference command.

            When extrapolation is disabled, the ratio is limited to 1.0.

        Output:
            Non-negative compensation ratio.

        Used by:
            calculate_forward_lateral_compensation().
            calculate_forward_heading_compensation().
            calculate_rotation_translation_compensation().

        Side effects:
            None.
        """
        if reference_magnitude <= 0.0:
            return 0.0

        ratio = (
            abs(command_magnitude)
            / reference_magnitude
        )

        if not self.allow_compensation_extrapolation:
            ratio = min(
                ratio,
                1.0,
            )

        return ratio

    def apply_translation_scale(
        self,
        linear_x: float,
    ) -> float:
        """
        Input:
            Raw forward command.

        Processing:
            Applies the configured translation scale when enabled.

        Output:
            Calibrated forward command.

        Used by:
            command_callback().

        Side effects:
            None.
        """
        if not self.enable_translation_scaling:
            return linear_x

        return (
            linear_x
            * self.translation_scale
        )

    def calculate_rotation_translation_compensation(
        self,
        raw_linear_x: float,
        raw_linear_y: float,
        raw_angular_z: float,
    ) -> tuple[float, float]:
        """
        Input:
            Raw planar Twist command.

        Processing:
            Detects intended pure rotation.

            Selects direction-dependent CW/CCW compensation values
            identified experimentally at
            rotation_translation_reference_wz.

            Compensation magnitude is scaled proportionally to the
            actual angular command magnitude.

        Output:
            Tuple containing:
                - longitudinal compensation [m/s],
                - lateral compensation [m/s].

        Used by:
            command_callback().

        Side effects:
            None.
        """
        if not self.enable_rotation_translation_compensation:
            return 0.0, 0.0

        if (
            abs(raw_linear_x)
            > self.rotation_translation_max_linear_command
        ):
            return 0.0, 0.0

        if (
            abs(raw_linear_y)
            > self.rotation_translation_max_linear_command
        ):
            return 0.0, 0.0

        if (
            abs(raw_angular_z)
            < self.rotation_translation_min_wz
        ):
            return 0.0, 0.0

        ratio = self.calculate_compensation_ratio(
            command_magnitude=raw_angular_z,
            reference_magnitude=self.rotation_translation_reference_wz,
        )

        if raw_angular_z < 0.0:
            return (
                self.clockwise_rotation_vx_compensation
                * ratio,
                self.clockwise_rotation_vy_compensation
                * ratio,
            )

        if raw_angular_z > 0.0:
            return (
                self.counterclockwise_rotation_vx_compensation
                * ratio,
                self.counterclockwise_rotation_vy_compensation
                * ratio,
            )

        return 0.0, 0.0

    def calculate_forward_lateral_compensation_old(
        self,
        linear_x: float,
        raw_angular_z: float,
    ) -> float:
        """
        Input:
            Forward command and raw angular command.

        Processing:
            Applies lateral compensation only during intended straight
            forward motion.

            The experimentally identified compensation corresponds to
            forward_compensation_reference_vx and is scaled according
            to the actual forward command.

        Output:
            Lateral velocity compensation.

        Used by:
            command_callback().

        Side effects:
            None.
        """
        if not self.enable_forward_lateral_compensation:
            return 0.0

        if (
            linear_x
            < self.forward_lateral_compensation_min_vx
        ):
            return 0.0

        if abs(raw_angular_z) > 1e-6:
            return 0.0

        ratio = self.calculate_compensation_ratio(
            command_magnitude=linear_x,
            reference_magnitude=self.forward_compensation_reference_vx,
        )

        return (
            self.forward_lateral_compensation_vy
            * ratio
        )

    def calculate_forward_lateral_compensation(
        self,
        linear_x: float,
        raw_angular_z: float,
    ) -> float:
        """
        Calculate lateral compensation during straight forward motion.

        Legacy model:

            C(vx) = C_ref * min(vx / vx_ref, 1)

        Generic polynomial model:

            C(vx) = c0 + c1*vx + ... + cn*vx^n

        Generic piecewise-linear model:

            C(vx) is interpolated between configured points.

        Compensation is not applied during rotation or reverse motion.
        """

        if not self.enable_forward_lateral_compensation:
            return 0.0

        if linear_x <= 0.0:
            return 0.0

        if abs(raw_angular_z) > 1e-6:
            return 0.0

        # Preserve complete compatibility with existing configurations.
        if (
            self.forward_lateral_compensation_model_type
            == "legacy_reference_scaled"
        ):
            if (
                linear_x
                < self.forward_lateral_compensation_min_vx
            ):
                return 0.0

            ratio = self.calculate_compensation_ratio(
                command_magnitude=linear_x,
                reference_magnitude=(
                    self.forward_compensation_reference_vx
                ),
            )

            return (
                self.forward_lateral_compensation_vy
                * ratio
            )

        if self.forward_lateral_compensation_model is None:
            raise RuntimeError(
                "Forward lateral compensation model is not initialized."
            )

        return evaluate_compensation_model(
            model=self.forward_lateral_compensation_model,
            input_value=linear_x,
        )

    
    def apply_rotation_scale(
        self,
        angular_z: float,
    ) -> float:
        """
        Input:
            Raw angular command.

        Processing:
            Applies an independent scale according to rotation direction.

        Output:
            Calibrated angular command.

        Used by:
            command_callback().

        Side effects:
            None.
        """
        if not self.enable_rotation_scaling:
            return angular_z

        if angular_z < 0.0:
            return (
                angular_z
                * self.clockwise_rotation_scale
            )

        if angular_z > 0.0:
            return (
                angular_z
                * self.counterclockwise_rotation_scale
            )

        return angular_z

    def calculate_forward_heading_compensation(
        self,
        linear_x: float,
        raw_angular_z: float,
    ) -> float:
        """
        Calculate heading compensation during straight forward motion.

        Legacy model:

            C(vx) = C_ref * min(vx / vx_ref, 1)

        Generic polynomial model:

            C(vx) = c0 + c1*vx + ... + cn*vx^n

        Generic piecewise-linear model:

            C(vx) is interpolated between configured points.

        Compensation is not applied during rotation or reverse motion.
        """

        if not self.enable_forward_heading_compensation:
            return 0.0

        if linear_x <= 0.0:
            return 0.0

        if abs(raw_angular_z) > 1e-6:
            return 0.0

        # Preserve complete compatibility with existing configurations.
        if (
            self.forward_heading_compensation_model_type
            == "legacy_reference_scaled"
        ):
            if (
                linear_x
                < self.forward_heading_compensation_min_vx
            ):
                return 0.0

            ratio = self.calculate_compensation_ratio(
                command_magnitude=linear_x,
                reference_magnitude=(
                    self.forward_compensation_reference_vx
                ),
            )

            return (
                self.forward_heading_compensation_wz
                * ratio
            )

        if self.forward_heading_compensation_model is None:
            raise RuntimeError(
                "Forward heading compensation model is not initialized."
            )

        return evaluate_compensation_model(
            model=self.forward_heading_compensation_model,
            input_value=linear_x,
        )

    # =========================================================================
    # ORCHESTRATION
    # =========================================================================

    def command_callback(
        self,
        msg: Twist,
    ) -> None:
        """
        Input:
            Incoming raw Twist command.

        Processing:
            Applies all enabled locomotion calibration layers independently:

            1. Translation scaling.
            2. Straight-motion lateral compensation.
            3. Straight-motion heading compensation.
            4. Pure-rotation longitudinal/lateral compensation.
            5. Direction-dependent rotation scaling.
            6. Composition of the final calibrated Twist.

        Output:
            Publishes the calibrated Twist command.

        Used by:
            ROS subscription callback.

        Side effects:
            Publishes one Twist message and writes diagnostic logs.
        """
        # =====================================================================
        # INPUT / INITIALIZATION
        # =====================================================================

        callback_start_ns = (
            time.perf_counter_ns()
        )

        out = Twist()

        scaled_linear_x = msg.linear.x
        scaled_angular_z = msg.angular.z

        forward_lateral_compensation_vy = 0.0
        forward_heading_compensation_wz = 0.0

        rotation_compensation_vx = 0.0
        rotation_compensation_vy = 0.0

        forward_compensation_ratio = 0.0
        rotation_compensation_ratio = 0.0

        active_corrections: list[str] = []


        # =====================================================================
        # PASSTHROUGH
        # =====================================================================

        if not self.enable_calibration:
            out.linear.x = msg.linear.x
            out.linear.y = msg.linear.y
            out.linear.z = msg.linear.z

            out.angular.x = msg.angular.x
            out.angular.y = msg.angular.y
            out.angular.z = msg.angular.z

            mode = "PASSTHROUGH"

        # =====================================================================
        # CALIBRATION
        # =====================================================================

        else:


            # -----------------------------------------------------------------
            # 1. Translation scaling
            # -----------------------------------------------------------------

            scaled_linear_x = (
                self.apply_translation_scale(
                    msg.linear.x
                )
            )

            if (
                self.enable_translation_scaling
                and abs(
                    scaled_linear_x
                    - msg.linear.x
                ) > 1e-12
            ):
                active_corrections.append(
                    "TRANSLATION_SCALING"
                )

            if (
                scaled_linear_x
                >= min(
                    self.forward_heading_compensation_min_vx,
                    self.forward_lateral_compensation_min_vx,
                )
            ):
                forward_compensation_ratio = (
                    self.calculate_compensation_ratio(
                        command_magnitude=scaled_linear_x,
                        reference_magnitude=self.forward_compensation_reference_vx,
                    )
                )

            if (
                abs(msg.linear.x)
                <= self.rotation_translation_max_linear_command
                and abs(msg.linear.y)
                <= self.rotation_translation_max_linear_command
                and abs(msg.angular.z)
                >= self.rotation_translation_min_wz
            ):
                rotation_compensation_ratio = (
                    self.calculate_compensation_ratio(
                        command_magnitude=msg.angular.z,
                        reference_magnitude=self.rotation_translation_reference_wz,
                    )
                )

            # -----------------------------------------------------------------
            # 2. Straight-motion lateral compensation
            # -----------------------------------------------------------------

            forward_lateral_compensation_vy = (
                self.calculate_forward_lateral_compensation(
                    scaled_linear_x,
                    msg.angular.z,
                )
            )

            if abs(
                forward_lateral_compensation_vy
            ) > 1e-12:
                active_corrections.append(
                    "FORWARD_LATERAL_COMPENSATION"
                )

            # -----------------------------------------------------------------
            # 3. Straight-motion heading compensation
            # -----------------------------------------------------------------

            forward_heading_compensation_wz = (
                self.calculate_forward_heading_compensation(
                    scaled_linear_x,
                    msg.angular.z,
                )
            )

            if abs(
                forward_heading_compensation_wz
            ) > 1e-12:
                active_corrections.append(
                    "FORWARD_HEADING_COMPENSATION"
                )

            # -----------------------------------------------------------------
            # 4. Pure-rotation translation compensation
            # -----------------------------------------------------------------

            (
                rotation_compensation_vx,
                rotation_compensation_vy,
            ) = self.calculate_rotation_translation_compensation(
                msg.linear.x,
                msg.linear.y,
                msg.angular.z,
            )

            if (
                abs(
                    rotation_compensation_vx
                ) > 1e-12
                or abs(
                    rotation_compensation_vy
                ) > 1e-12
            ):
                active_corrections.append(
                    "ROTATION_TRANSLATION_COMPENSATION"
                )

            # -----------------------------------------------------------------
            # 5. Direction-dependent rotation scaling
            # -----------------------------------------------------------------

            scaled_angular_z = (
                self.apply_rotation_scale(
                    msg.angular.z
                )
            )

            if (
                self.enable_rotation_scaling
                and abs(
                    scaled_angular_z
                    - msg.angular.z
                ) > 1e-12
            ):
                active_corrections.append(
                    "ROTATION_SCALING"
                )

            # -----------------------------------------------------------------
            # 6. Compose calibrated command
            # -----------------------------------------------------------------

            out.linear.x = (
                scaled_linear_x
                + rotation_compensation_vx
            )

            out.linear.y = (
                msg.linear.y
                + forward_lateral_compensation_vy
                + rotation_compensation_vy
            )

            out.linear.z = msg.linear.z

            out.angular.x = msg.angular.x
            out.angular.y = msg.angular.y

            out.angular.z = (
                scaled_angular_z
                + forward_heading_compensation_wz
            )

            mode = (
                " + ".join(
                    active_corrections
                )
                if active_corrections
                else "CALIBRATION_PASSTHROUGH"
            )

        computation_end_ns = (
            time.perf_counter_ns()
        )

        computation_latency_ms = (
            computation_end_ns
            - callback_start_ns
        ) / 1_000_000.0

        # =====================================================================
        # OUTPUT
        # =====================================================================

        self.publisher.publish(
            out
        )

        callback_end_ns = (
            time.perf_counter_ns()
        )

        callback_latency_ms = (
            callback_end_ns
            - callback_start_ns
        ) / 1_000_000.0


        if self.enable_latency_diagnostics:
            computation_latency_message = (
                Float64()
            )

            computation_latency_message.data = (
                computation_latency_ms
            )

            self.computation_latency_publisher.publish(
                computation_latency_message
            )

            callback_latency_message = (
                Float64()
            )

            callback_latency_message.data = (
                callback_latency_ms
            )

            self.callback_latency_publisher.publish(
                callback_latency_message
            )

        # =====================================================================
        # REPORTING
        # =====================================================================

        self.get_logger().info(
            (
                "\n"
                "=========================================================\n"
                "ROBOT LOCOMOTION CALIBRATION\n"
                "=========================================================\n"
                f"Mode                     : {mode}\n"
                "\n"
                "Input command\n"
                "---------------------------------------------------------\n"
                f"vx                       : {msg.linear.x:+.4f} m/s\n"
                f"vy                       : {msg.linear.y:+.4f} m/s\n"
                f"wz                       : {msg.angular.z:+.4f} rad/s\n"
                "\n"
                "Base scaling\n"
                "---------------------------------------------------------\n"
                f"Scaled vx                : {scaled_linear_x:+.4f} m/s\n"
                f"Scaled wz                : {scaled_angular_z:+.4f} rad/s\n"
                "\n"
                "Straight-motion compensation\n"
                "---------------------------------------------------------\n"
                f"Forward lateral vy       : "
                f"{forward_lateral_compensation_vy:+.4f} m/s\n"
                f"Forward heading wz       : "
                f"{forward_heading_compensation_wz:+.4f} rad/s\n"
                "\n"
                "Pure-rotation translation compensation\n"
                "---------------------------------------------------------\n"
                f"Rotation compensation vx : "
                f"{rotation_compensation_vx:+.4f} m/s\n"
                f"Rotation compensation vy : "
                f"{rotation_compensation_vy:+.4f} m/s\n"
                "\n"
                "Output command\n"
                "---------------------------------------------------------\n"
                f"vx                       : {out.linear.x:+.4f} m/s\n"
                f"vy                       : {out.linear.y:+.4f} m/s\n"
                f"wz                       : {out.angular.z:+.4f} rad/s\n"
                "\n"
                
                "Calibration adaptation\n"
                "---------------------------------------------------------\n"
                f"Forward reference vx     : "
                f"{self.forward_compensation_reference_vx:.4f} m/s\n"
                f"Forward ratio            : "
                f"{forward_compensation_ratio:.4f}\n"
                f"Rotation reference |wz|  : "
                f"{self.rotation_translation_reference_wz:.4f} rad/s\n"
                f"Rotation ratio           : "
                f"{rotation_compensation_ratio:.4f}\n"

                "\n"
                " LATENCY DIAGNOSTICS\n"
                "---------------------------------------------------------\n"
                f" Enabled                            : "
                f"{self.enable_latency_diagnostics}\n"
                f" Computation latency topic          : "
                f"{self.computation_latency_topic}\n"
                f" Callback latency topic             : "
                f"{self.callback_latency_topic}\n"
                "========================================================="
            ),
            throttle_duration_sec=0.5,
        )
        


def main() -> None:
    """Run the locomotion calibration node."""
    rclpy.init()

    node = LocomotionCalibrationNode()

    try:
        rclpy.spin(
            node
        )

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()