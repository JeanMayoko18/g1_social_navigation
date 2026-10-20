#!/usr/bin/env python3
"""
Temporal Motion Increment Representation (TMIR).

TMIR converts raw odometry poses into local temporal motion increments.

Responsibilities
----------------
1. Initialize a local odometry origin.
2. Normalize raw planar poses relative to that origin.
3. Compute motion increments between consecutive samples.
4. Express translational increments in either:
   - the local odometry frame;
   - the instantaneous robot body frame.

TMIR does not predict motion, estimate confidence, or fuse increments.
"""

import math
from dataclasses import dataclass


@dataclass
class LocalPlanarPose:
    """Planar pose expressed in the local raw-odometry frame."""

    x: float
    y: float
    yaw: float


@dataclass
class TMIRIncrement:
    """
    Temporal motion increment produced by TMIR.

    dx_odom and dy_odom are expressed in the local odometry frame.
    dyaw is the wrapped planar heading increment.
    heading_prev is the raw heading associated with the beginning
    of the increment.
    """

    has_increment: bool
    dx_odom: float
    dy_odom: float
    dyaw: float
    heading_prev: float


class TemporalMotionIncrementRepresentation:
    """
    Stateful temporal representation of raw odometry motion.
    """

    def __init__(self):
        # Local origin.
        self.have_origin = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        # Previous normalized raw pose.
        self.have_previous = False
        self.previous_x = 0.0
        self.previous_y = 0.0
        self.previous_yaw = 0.0

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    def reset(self) -> None:
        """Reset the origin and temporal increment state."""
        self.have_origin = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        self.have_previous = False
        self.previous_x = 0.0
        self.previous_y = 0.0
        self.previous_yaw = 0.0

    def normalize_pose(
        self,
        raw_x: float,
        raw_y: float,
        raw_yaw: float,
    ) -> LocalPlanarPose:
        """
        Express a raw planar pose relative to the initial raw pose.

        The resulting local frame starts at:

            x = 0
            y = 0
            yaw = 0
        """

        if not self.have_origin:
            self.origin_x = raw_x
            self.origin_y = raw_y
            self.origin_yaw = raw_yaw
            self.have_origin = True

            return LocalPlanarPose(
                x=0.0,
                y=0.0,
                yaw=0.0,
            )

        dx = raw_x - self.origin_x
        dy = raw_y - self.origin_y

        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)

        # Equivalent to R(-origin_yaw) * [dx, dy].
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        local_yaw = self.wrap_angle(
            raw_yaw - self.origin_yaw
        )

        return LocalPlanarPose(
            x=local_x,
            y=local_y,
            yaw=local_yaw,
        )

    def compute_increment(
        self,
        pose: LocalPlanarPose,
    ) -> TMIRIncrement:
        """
        Compute the temporal increment between consecutive normalized poses.
        """

        if not self.have_previous:
            self.previous_x = pose.x
            self.previous_y = pose.y
            self.previous_yaw = pose.yaw
            self.have_previous = True

            return TMIRIncrement(
                has_increment=False,
                dx_odom=0.0,
                dy_odom=0.0,
                dyaw=0.0,
                heading_prev=pose.yaw,
            )

        heading_prev = self.previous_yaw

        dx_odom = pose.x - self.previous_x
        dy_odom = pose.y - self.previous_y
        dyaw = self.wrap_angle(
            pose.yaw - self.previous_yaw
        )

        self.previous_x = pose.x
        self.previous_y = pose.y
        self.previous_yaw = pose.yaw

        return TMIRIncrement(
            has_increment=True,
            dx_odom=dx_odom,
            dy_odom=dy_odom,
            dyaw=dyaw,
            heading_prev=heading_prev,
        )
    # ΔpB=R(−θ)ΔpO
    @staticmethod
    def odom_increment_to_body(
        dx_odom: float,
        dy_odom: float,
        heading: float,
    ):
        """
        Transform a translation increment from local odometry frame
        into the robot body frame.

        Δp_body = R(-heading) Δp_odom
        """

        c = math.cos(heading)
        s = math.sin(heading)

        dx_body = c * dx_odom + s * dy_odom
        dy_body = -s * dx_odom + c * dy_odom

        return dx_body, dy_body
    
    # ΔpO=R(θ)ΔpB
    @staticmethod
    def body_increment_to_odom(
        dx_body: float,
        dy_body: float,
        heading: float,
    ):
        """
        Transform a translation increment from robot body frame
        into the local corrected odometry frame.

        Δp_odom = R(heading) Δp_body
        """

        c = math.cos(heading)
        s = math.sin(heading)

        dx_odom = c * dx_body - s * dy_body
        dy_odom = s * dx_body + c * dy_body

        return dx_odom, dy_odom