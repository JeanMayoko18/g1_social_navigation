"""
Trajectory candidate generation for socially-aware navigation.

Responsibility
--------------
TrajectoryManager generates exact candidate trajectories through Nav2.

Generation modes:

    NONE
        -> ComputePathToPose
        -> CandidateTrajectory

    LEFT / RIGHT
        -> require valid person heading
        -> construct a person-relative passage waypoint
        -> ComputePathThroughPoses
        -> CandidateTrajectory

The manager does NOT:
    - decide whether a trajectory is socially valid;
    - authorize robot motion;
    - interpret human intent;
    - execute FollowPath;
    - publish CollaborativeDecision.

Safety invariant
----------------
Every successful planning result receives a fresh immutable path_id.

A validation for one path_id must never authorize another path.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
import copy
from dataclasses import dataclass
from typing import Dict

import json
from std_msgs.msg import String

import rclpy

from geometry_msgs.msg import PoseStamped

from rclpy.duration import Duration
from rclpy.time import Time

from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)

from nav2_msgs.action import (
    ComputePathThroughPoses,
    ComputePathToPose,
)
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node

from g1_social_msgs.msg import (
    CandidateTrajectory,
    SocialZoneArray,
    TrajectoryGenerationRequest,
)


# ========================================================================
# GENERATION REASONS
# ========================================================================

GENERATION_REASON_INITIAL = "INITIAL"
GENERATION_REASON_REPLAN = "REPLAN"
GENERATION_REASON_ALTERNATIVE = "ALTERNATIVE"

VALID_GENERATION_REASONS = frozenset(
    {
        GENERATION_REASON_INITIAL,
        GENERATION_REASON_REPLAN,
        GENERATION_REASON_ALTERNATIVE,
    }
)


# ========================================================================
# DIRECTIONAL CONSTRAINTS
# ========================================================================

DIRECTION_NONE = "NONE"
DIRECTION_LEFT = "LEFT"
DIRECTION_RIGHT = "RIGHT"

VALID_DIRECTIONS = frozenset(
    {
        DIRECTION_NONE,
        DIRECTION_LEFT,
        DIRECTION_RIGHT,
    }
)


# ========================================================================
# INTERNAL REQUEST STATE
# ========================================================================

@dataclass(frozen=True)
class PendingPlanningRequest:
    """
    Immutable metadata associated with one planning request.
    """

    request_id: str
    navigation_goal_id: str

    planner_id: str
    generation_reason: str

    directional_constraint: str

    person_id: str

    received_monotonic: float


class TrajectoryManager(Node):
    """
    Generate candidate trajectories using Nav2 planning actions.

    Input
    -----
    /social_navigation/trajectory_generation_request

    Output
    ------
    /social_navigation/candidate_trajectory

    Nav2 actions
    ------------
    /compute_path_to_pose
    /compute_path_through_poses
    """

    def __init__(self) -> None:
        super().__init__("trajectory_manager")

        # ================================================================
        # PARAMETERS
        # ================================================================

        self.declare_parameter(
            "generation_request_topic",
            "/social_navigation/trajectory_generation_request",
        )

        self.declare_parameter(
            "candidate_trajectory_topic",
            "/social_navigation/candidate_trajectory",
        )

        self.declare_parameter(
            "compute_path_action",
            "/compute_path_to_pose",
        )

        self.declare_parameter(
            "compute_path_through_poses_action",
            "/compute_path_through_poses",
        )

        self.declare_parameter(
            "planner_server_wait_timeout_sec",
            2.0,
        )

        self.declare_parameter(
            "pending_request_timeout_sec",
            15.0,
        )

        self.declare_parameter(
            "social_zones_topic",
            "/social_perception/social_zones",
        )

        self.declare_parameter(
            "social_zone_max_age_sec",
            0.5,
        )

        # --------------------------------------------------------------
        # Person-relative passage geometry.
        #
        # This is deliberately a ROS parameter rather than a hard-coded
        # scientific constant.
        # --------------------------------------------------------------

        self.declare_parameter(
            "passage_lateral_offset_m",
            1.0,
        )
        
        self.last_social_stamp_ns = None

        self.declare_parameter(
            "passage_clearance_m",
            0.35,
        )

        self.passage_clearance_m = float(
            self.get_parameter("passage_clearance_m").value
        )

        if (
            not math.isfinite(self.passage_clearance_m)
            or self.passage_clearance_m <= 0.0
        ):
            raise ValueError(
                "passage_clearance_m must be finite and positive."
            )

        self.generation_request_topic = str(
            self.get_parameter(
                "generation_request_topic"
            ).value
        )

        self.candidate_trajectory_topic = str(
            self.get_parameter(
                "candidate_trajectory_topic"
            ).value
        )

        self.compute_path_action = str(
            self.get_parameter(
                "compute_path_action"
            ).value
        )

        self.compute_path_through_poses_action = str(
            self.get_parameter(
                "compute_path_through_poses_action"
            ).value
        )

        self.planner_server_wait_timeout_sec = float(
            self.get_parameter(
                "planner_server_wait_timeout_sec"
            ).value
        )

        self.pending_request_timeout_sec = float(
            self.get_parameter(
                "pending_request_timeout_sec"
            ).value
        )

        self.passage_lateral_offset_m = float(
            self.get_parameter(
                "passage_lateral_offset_m"
            ).value
        )

        if self.passage_lateral_offset_m <= 0.0:
            raise ValueError(
                "passage_lateral_offset_m must be > 0."
            )

        self.social_zones_topic = str(
            self.get_parameter(
                "social_zones_topic"
            ).value
        )

        self.social_zone_max_age_sec = float(
            self.get_parameter(
                "social_zone_max_age_sec"
            ).value
        )

        # ================================================================
        # CONCURRENCY / STATE
        # ================================================================

        #self.callback_group = ReentrantCallbackGroup()
        self.callback_group = MutuallyExclusiveCallbackGroup()

        self.state_lock = threading.RLock()

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.pending_requests: Dict[
            str,
            PendingPlanningRequest,
        ] = {}

        self.cancel_pending_requests = set()
        self.retired_social_episode_ids = {}

        # navigation_goal_id -> newest deferred TrajectoryGenerationRequest
        self.deferred_requests_by_goal = {}

        self.active_goal_handles: Dict[
            str,
            ClientGoalHandle,
        ] = {}

        # person_id -> (SocialZone, receive_monotonic)
        self.social_zones = {}

        # Historical observations are planning hints, not safety evidence.
        self._planning_zone_history = {}

        # Preserve exactly one identical waypoint retry per direction.
        self._directional_waypoints = {}

        self.timeout_notified_requests = set()

        # Requests whose planning result can no longer authorize
        # publication of a candidate trajectory.
        #
        # They remain inhibited until terminal confirmation.
        self.inhibited_request_ids = set()

        # ================================================================
        # ROS INTERFACES
        # ================================================================

        self.candidate_publisher = self.create_publisher(
            CandidateTrajectory,
            self.candidate_trajectory_topic,
            10,
        )

        self.request_subscription = self.create_subscription(
            TrajectoryGenerationRequest,
            self.generation_request_topic,
            self._trajectory_generation_request_callback,
            10,
            callback_group=self.callback_group,
        )

        self.social_zones_subscription = self.create_subscription(
            SocialZoneArray,
            self.social_zones_topic,
            self._social_zones_callback,
            10,
            callback_group=self.callback_group,
        )

        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            self.compute_path_action,
            callback_group=self.callback_group,
        )

        self.compute_path_through_poses_client = ActionClient(
            self,
            ComputePathThroughPoses,
            self.compute_path_through_poses_action,
            callback_group=self.callback_group,
        )

        self.cleanup_timer = self.create_timer(
            1.0,
            self._cleanup_stale_requests,
            callback_group=self.callback_group,
        )

        self.generation_result_pub = self.create_publisher(
            String,
            "/social_navigation/trajectory_generation_result",
            10,
        )

        self.terminal_generation_results = {}
        self.generation_reconcile_futures = {}

        self.generation_status_request_sub = self.create_subscription(
            String,
            "/social_navigation/trajectory_generation_status_request",
            self._generation_status_request_callback,
            10,
            callback_group=self.callback_group,
        )


        self.get_logger().info(
            "[TRAJECTORY MANAGER] Ready. "
            f"request_topic={self.generation_request_topic}, "
            f"candidate_topic={self.candidate_trajectory_topic}, "
            f"compute_path={self.compute_path_action}, "
            f"compute_through_poses="
            f"{self.compute_path_through_poses_action}"
        )

    # ====================================================================
    # REQUEST HANDLING
    # ====================================================================

    def _publish_generation_failure(self, request_id, reason):
        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

        if metadata is None:
            return

        msg = String()
        msg.data = json.dumps({
            "version": 1,
            "producer": "trajectory_manager",
            "navigation_goal_id": metadata.navigation_goal_id,
            "request_id": metadata.request_id,
            "status": "FAILED",
            "reason": str(reason),
        })

        self.generation_result_pub.publish(msg)

        self.get_logger().error(
            "[TRAJECTORY RESULT] "
            f"request={request_id} | "
            f"status=FAILED | reason={reason}"
        )


    def _trajectory_generation_request_callback(
        self,
        msg: TrajectoryGenerationRequest,
    ) -> None:
        """
        Accept one trajectory-generation request.

        Strong invariant:
            only one Nav2 planning generation may be alive at a time for the
            same navigation_goal_id.

        If a newer request arrives while an older generation for the same
        navigation goal is still alive, retain the newest request, cancel the
        older generation, and start the newer one only after the older Nav2
        action reaches a terminal result.
        """

        request_id = str(
            msg.request_id
        ).strip()

        navigation_goal_id = str(
            msg.navigation_goal_id
        ).strip()

        if not request_id:
            self.get_logger().warning(
                "[TRAJECTORY MANAGER] "
                "Rejected request without request_id."
            )
            return

        if not navigation_goal_id:
            self.get_logger().warning(
                "[TRAJECTORY MANAGER] "
                f"Rejected request {request_id}: "
                "navigation_goal_id is empty."
            )
            return

        with self.state_lock:
            terminal = self.terminal_generation_results.get(
                request_id
            )

        if terminal is not None:
            if terminal["navigation_goal_id"] == navigation_goal_id:
                self.generation_result_pub.publish(
                    String(data=json.dumps(terminal))
                )
            return
        
        generation_reason = (
            str(msg.generation_reason)
            .strip()
            .upper()
        )

        if generation_reason not in VALID_GENERATION_REASONS:
            self.get_logger().warning(
                "[TRAJECTORY MANAGER] "
                f"Rejected request {request_id}: "
                f"invalid generation_reason="
                f"{msg.generation_reason!r}."
            )
            return

        directional_constraint = (
            str(msg.directional_constraint)
            .strip()
            .upper()
            or DIRECTION_NONE
        )

        if directional_constraint not in VALID_DIRECTIONS:
            self.get_logger().warning(
                "[TRAJECTORY MANAGER] "
                f"Rejected request {request_id}: "
                f"invalid directional_constraint="
                f"{msg.directional_constraint!r}."
            )
            return

        if (
            directional_constraint
            in {DIRECTION_LEFT, DIRECTION_RIGHT}
            and not str(msg.person_id).strip()
        ):
            self.get_logger().warning(
                "[TRAJECTORY MANAGER] "
                f"Rejected directional request {request_id}: "
                "person_id is empty."
            )
            return

        requests_to_cancel = []

        with self.state_lock:

            if request_id in self.pending_requests:
                self.get_logger().warning(
                    "[TRAJECTORY MANAGER] "
                    f"Ignoring duplicate request_id={request_id}."
                )
                return

            requests_to_cancel = [
                existing_request_id
                for existing_request_id, existing_metadata
                in self.pending_requests.items()
                if (
                    existing_metadata.navigation_goal_id
                    == navigation_goal_id
                    and existing_request_id != request_id
                )
            ]

            if requests_to_cancel:

                # Keep only the newest requested generation.
                self.deferred_requests_by_goal[
                    navigation_goal_id
                ] = copy.deepcopy(msg)

        if requests_to_cancel:

            self.get_logger().info(
                "[TRAJECTORY MANAGER] "
                f"REQUEST_DEFERRED={request_id} | "
                f"GOAL={navigation_goal_id} | "
                "WAITING_FOR_PREVIOUS_GENERATION="
                f"{requests_to_cancel}"
            )

            for old_request_id in requests_to_cancel:
                self._cancel_and_remove_request(
                    old_request_id
                )

            return
        
        with self.state_lock:
            if request_id in self.pending_requests:
                return

            pending_for_goal = [
                other_id
                for other_id, other_metadata
                in self.pending_requests.items()
                if other_metadata.navigation_goal_id == navigation_goal_id
            ]

            if pending_for_goal:
                # Retain only the newest replacement for this public goal.
                self.deferred_requests_by_goal[navigation_goal_id] = (
                    copy.deepcopy(msg)
                )

        if pending_for_goal:
            self.get_logger().info(
                "[TRAJECTORY MANAGER] "
                f"REQUEST_DEFERRED={request_id} | "
                f"GOAL={navigation_goal_id}"
            )

            for other_id in pending_for_goal:
                self._cancel_and_remove_request(other_id)

            return
        
        metadata = PendingPlanningRequest(
            request_id=request_id,
            navigation_goal_id=navigation_goal_id,
            planner_id=str(msg.planner_id).strip(),
            generation_reason=generation_reason,
            directional_constraint=directional_constraint,
            person_id=str(msg.person_id).strip(),
            received_monotonic=time.monotonic(),
        )

        with self.state_lock:

            if request_id in self.pending_requests:
                return

            self.pending_requests[
                request_id
            ] = metadata

        self.get_logger().info(
            "[TRAJECTORY MANAGER] "
            f"REQUEST_ACTIVE={request_id} | "
            f"GOAL={navigation_goal_id} | "
            f"REASON={generation_reason} | "
            f"DIRECTION={directional_constraint}"
        )

        if directional_constraint == DIRECTION_NONE:

            self._request_standard_path(
                msg,
                metadata,
            )

            return

        self._request_directional_path(
            msg,
            metadata,
        )

    # ====================================================================
    # STANDARD PATH
    # ====================================================================

    def _publish_generation_result(self, request_id, status, reason):
        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

        if metadata is None:
            return

        payload = {
            "version": 1,
            "producer": "trajectory_manager",
            "navigation_goal_id": metadata.navigation_goal_id,
            "request_id": metadata.request_id,
            "status": str(status).upper(),
            "reason": str(reason),
        }

        message = String()
        message.data = json.dumps(
            payload,
            separators=(",", ":"),
        )

        self.generation_result_pub.publish(message)

        self.get_logger().warning(
            "[TRAJECTORY RESULT] "
            f"request={request_id} | "
            f"status={payload['status']} | "
            f"reason={payload['reason']}"
        )


    def _request_standard_path(
        self,
        msg: TrajectoryGenerationRequest,
        metadata: PendingPlanningRequest,
    ) -> None:
        """
        Submit a standard candidate to Nav2.

        Local definitive failures publish FAILED.
        Uncertain submissions retain request ownership.
        """
        request_id = metadata.request_id

        goal = ComputePathToPose.Goal()
        request_stamp = self.get_clock().now().to_msg()

        goal.goal = copy.deepcopy(msg.goal)
        goal.goal.header.stamp = request_stamp

        goal.use_start = bool(msg.use_start)
        goal.start = copy.deepcopy(msg.start)
        goal.start.header.stamp = request_stamp
        goal.planner_id = metadata.planner_id

        if not self.compute_path_client.wait_for_server(
            timeout_sec=self.planner_server_wait_timeout_sec
        ):
            self._publish_generation_result(
                request_id,
                "FAILED",
                "COMPUTE_PATH_TO_POSE_UNAVAILABLE",
            )
            self._remove_request(request_id)
            return

        try:
            future = self.compute_path_client.send_goal_async(
                goal
            )

            if future is None:
                raise RuntimeError(
                    "Planning submission future missing."
                )

        except Exception as exc:
            with self.state_lock:
                self.inhibited_request_ids.add(request_id)
                self.timeout_notified_requests.add(request_id)

            self._publish_generation_result(
                request_id,
                "TERMINATION_UNCONFIRMED",
                "STANDARD_SUBMISSION_UNCONFIRMED",
            )

            self.get_logger().error(
                "[TRAJECTORY MANAGER] "
                f"Standard submission uncertain | "
                f"request={request_id} | error={exc}"
            )
            return

        self.get_logger().info(
            "[TRAJECTORY MANAGER] ComputePathToPose | "
            f"request={request_id} | "
            f"goal={metadata.navigation_goal_id}"
        )

        future.add_done_callback(
            lambda done_future, rid=request_id:
                self._planning_goal_response_callback(
                    rid,
                    done_future,
                )
        )
        
    # ====================================================================
    # DIRECTIONAL PATH
    # ====================================================================

    def _request_directional_path(
        self,
        msg: TrajectoryGenerationRequest,
        metadata: PendingPlanningRequest,
    ) -> None:
        """
        Generate a directional trajectory through a person-relative
        waypoint.

        Every local failure publishes FAILED before releasing the
        request. An accepted Nav2 goal remains owned until its
        terminal result.
        """
        request_id = metadata.request_id

        try:
            waypoint = self._build_directional_waypoint(
                msg,
                metadata,
            )
        except Exception as exc:
            self.get_logger().error(
                "[TRAJECTORY MANAGER] Directional waypoint error | "
                f"request={request_id} | error={exc}"
            )

            self._publish_generation_result(
                request_id,
                "FAILED",
                "DIRECTIONAL_WAYPOINT_EXCEPTION",
            )
            self._remove_request(request_id)
            return

        if waypoint is None:
            self._publish_generation_result(
                request_id,
                "FAILED",
                "DIRECTIONAL_WAYPOINT_UNAVAILABLE",
            )
            self._remove_request(request_id)
            return

        goal = ComputePathThroughPoses.Goal()

        request_stamp = self.get_clock().now().to_msg()

        waypoint_for_planner = copy.deepcopy(waypoint)
        waypoint_for_planner.header.stamp = request_stamp

        final_goal_for_planner = copy.deepcopy(msg.goal)
        final_goal_for_planner.header.stamp = request_stamp

        goal.goals = [
            waypoint_for_planner,
            final_goal_for_planner,
        ]

        goal.use_start = bool(msg.use_start)
        goal.start = copy.deepcopy(msg.start)
        goal.start.header.stamp = request_stamp
        goal.planner_id = metadata.planner_id

        if not (
            self.compute_path_through_poses_client
            .wait_for_server(
                timeout_sec=self.planner_server_wait_timeout_sec
            )
        ):
            self._publish_generation_result(
                request_id,
                "FAILED",
                "COMPUTE_PATH_THROUGH_POSES_UNAVAILABLE",
            )
            self._remove_request(request_id)
            return

        try:
            future = (
                self.compute_path_through_poses_client
                .send_goal_async(goal)
            )

            if future is None:
                raise RuntimeError(
                    "Planning submission future missing."
                )

        except Exception as exc:
            # Submission failure does not prove that Nav2
            # has not received the goal.
            # Preserve request ownership until reconciled.
            with self.state_lock:
                self.inhibited_request_ids.add(request_id)
                self.timeout_notified_requests.add(request_id)

            self._publish_generation_result(
                request_id,
                "TERMINATION_UNCONFIRMED",
                "DIRECTIONAL_SUBMISSION_UNCONFIRMED",
            )

            self.get_logger().error(
                "[TRAJECTORY MANAGER] "
                f"Directional submission uncertain | "
                f"request={request_id} | error={exc}"
            )
            return

        self.get_logger().info(
            "[TRAJECTORY MANAGER] ComputePathThroughPoses | "
            f"request={request_id} | "
            f"direction={metadata.directional_constraint} | "
            f"person={metadata.person_id}"
        )

        future.add_done_callback(
            lambda done_future, rid=request_id:
                self._planning_goal_response_callback(
                    rid,
                    done_future,
                )
        )

    # ====================================================================
    # PERSON-RELATIVE WAYPOINT
    # ====================================================================

    def _build_directional_waypoint(self, msg, metadata):
        """
        Build person-relative planning waypoints with one exact retry.

        Historical geometry may be used as a planning hint only.
        It never constitutes trajectory validation or motion clearance.
        """
        now = time.monotonic()

        key = (
            str(metadata.navigation_goal_id),
            str(metadata.person_id),
            str(metadata.directional_constraint),
        )

        with self.state_lock:
            previous = self._directional_waypoints.get(key)

            if (
                previous is not None
                and now - previous["created_at"] >= 180.0
            ):
                self._directional_waypoints.pop(key, None)
                previous = None

            # Retry the same waypoint exactly once.
            if previous is not None and previous["attempts"] == 1:
                previous["attempts"] = 2

                self.get_logger().info(
                    "[DIRECTIONAL PLAN] Repeating exact waypoint | "
                    f"direction={metadata.directional_constraint}"
                )

                return copy.deepcopy(previous["waypoint"])

        direction = str(
            metadata.directional_constraint
        ).strip().upper()

        if direction not in {"LEFT", "RIGHT"}:
            return None

        target_frame = str(
            msg.goal.header.frame_id
        ).strip()

        if not target_frame:
            return None

        # Prefer current observations for a new planning proposal.
        fresh = self._get_fresh_social_zone(
            metadata.person_id
        )

        historical = False

        if fresh is None:
            with self.state_lock:
                saved = self._planning_zone_history.get(
                    str(metadata.person_id)
                )

            if saved is None or now - saved[2] > 180.0:
                return None

            zone = copy.deepcopy(saved[0])
            header = copy.deepcopy(saved[1])
            historical = True

        else:
            zone, header = fresh

        with self.state_lock:
            attempts = (
                previous["attempts"]
                if previous is not None
                else 0
            )

        # LEFT and RIGHT are relative to the person's heading.
        sign = 1.0 if direction == "LEFT" else -1.0

        offset = max(
            float(self.passage_lateral_offset_m),
            float(zone.radius_lateral) + 0.35,
        )

        # After the identical retry, explore a wider proposal.
        if attempts >= 2:
            offset += 0.40 * (attempts - 1)

        x = (
            float(zone.center.x)
            - sign * offset * math.sin(zone.heading)
        )

        y = (
            float(zone.center.y)
            + sign * offset * math.cos(zone.heading)
        )

        z = float(zone.center.z)

        if str(header.frame_id) != target_frame:
            try:
                # Historical geometry is only a planning reference.
                # Use the latest available frame transform when needed.
                timestamp = (
                    Time()
                    if historical
                    else Time.from_msg(header.stamp)
                )

                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    str(header.frame_id),
                    timestamp,
                    timeout=Duration(seconds=0.0),
                )

                x, y, z = self._transform_point(
                    x,
                    y,
                    z,
                    transform,
                )

            except (TransformException, ValueError) as exc:
                self.get_logger().warning(
                    "[DIRECTIONAL PLAN] TF unavailable | "
                    f"error={exc}"
                )
                return None

        if not all(
            math.isfinite(value)
            for value in (x, y, z)
        ):
            return None

        waypoint = PoseStamped()

        waypoint.header.frame_id = target_frame
        waypoint.header.stamp = (
            self.get_clock().now().to_msg()
        )

        waypoint.pose.position.x = x
        waypoint.pose.position.y = y
        waypoint.pose.position.z = z
        waypoint.pose.orientation.w = 1.0

        with self.state_lock:
            self._directional_waypoints[key] = {
                "waypoint": copy.deepcopy(waypoint),
                "attempts": attempts + 1,
                "created_at": (
                    previous["created_at"]
                    if previous is not None
                    else now
                ),
            }

        self.get_logger().info(
            "[DIRECTIONAL PLAN] "
            f"goal={metadata.navigation_goal_id} | "
            f"direction={direction} | "
            f"attempt={attempts + 1} | "
            f"historical_hint={historical} | "
            f"waypoint=({x:.2f}, {y:.2f})"
        )

        return waypoint

    @staticmethod
    def _transform_point(x, y, z, transform):
        """Apply a normalized rigid transform without modifying input."""
        q = transform.transform.rotation
        t = transform.transform.translation

        values = (
            x, y, z,
            q.x, q.y, q.z, q.w,
            t.x, t.y, t.z,
        )

        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Non-finite transform.")

        norm = math.hypot(q.x, q.y, q.z, q.w)

        if norm <= 1e-12:
            raise ValueError("Invalid transform quaternion.")

        qx, qy, qz, qw = (
            q.x / norm,
            q.y / norm,
            q.z / norm,
            q.w / norm,
        )

        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)

        return (
            x + qw * tx + qy * tz - qz * ty + t.x,
            y + qw * ty + qz * tx - qx * tz + t.y,
            z + qw * tz + qx * ty - qy * tx + t.z,
        )
    
    def _get_fresh_social_zone(self, person_id):
        """Return bounded planning geometry; never authorize execution."""
        person_id = str(person_id or "").strip()
        limit = self._planning_geometry_max_age()

        with self.state_lock:
            entry = self.social_zones.get(person_id)

        if entry is None:
            self.get_logger().warning(
                "[PLANNING GEOMETRY] "
                f"person={person_id} | "
                "reason=ZONE_NOT_CACHED"
            )
            return None

        zone, header, received = entry

        source_ns = (
            int(header.stamp.sec) * 1_000_000_000
            + int(header.stamp.nanosec)
        )

        source_age = (
            self.get_clock().now().nanoseconds
            - source_ns
        ) / 1e9

        receipt_age = time.monotonic() - received

        if not (
            -0.1 <= source_age <= limit
            and 0.0 <= receipt_age <= limit
        ):
            self.get_logger().warning(
                "[PLANNING GEOMETRY] Snapshot expired | "
                f"person={person_id} | "
                f"source_age={source_age:.3f}s | "
                f"receipt_age={receipt_age:.3f}s | "
                f"limit={limit:.3f}s"
            )
            return None

        self.get_logger().info(
            "[PLANNING GEOMETRY] Waypoint input available | "
            f"person={person_id} | "
            f"source_age={source_age:.3f}s | "
            "fresh_path_validation_required=True"
        )

        return (
            copy.deepcopy(zone),
            copy.deepcopy(header),
        )
    
    # ====================================================================
    # NAV2 GOAL RESPONSE
    # ====================================================================

    def _planning_goal_response_callback(self, request_id, future):
        """
        Handle planner goal acceptance/rejection.

        Late acceptance never restores ownership of an obsolete
        generation. Cancellation remains separate from terminal
        confirmation.
        """
        request_id = str(request_id or "").strip()

        try:
            goal_handle = future.result()
            if goal_handle is None:
                raise RuntimeError("Goal handle missing.")
        except Exception as exc:
            with self.state_lock:
                if request_id not in self.pending_requests:
                    return

                self.inhibited_request_ids.add(request_id)

                notify = (
                    request_id not in self.timeout_notified_requests
                )
                self.timeout_notified_requests.add(request_id)

            if notify:
                self._publish_generation_result(
                    request_id,
                    "TERMINATION_UNCONFIRMED",
                    "GOAL_RESPONSE_UNCONFIRMED",
                )

            self._cancel_and_remove_request(request_id)
            self.get_logger().error(
                "[TRAJECTORY MANAGER] Goal response uncertain | "
                f"request={request_id} | error={exc}"
            )
            return

        if not goal_handle.accepted:
            with self.state_lock:
                if request_id not in self.pending_requests:
                    return
                inhibited = (
                    request_id in self.inhibited_request_ids
                )

            self._publish_generation_result(
                request_id,
                "TERMINATED" if inhibited else "FAILED",
                "GOAL_REJECTED",
            )
            self._remove_request(request_id)
            return

        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

            if metadata is not None:
                self.active_goal_handles[request_id] = goal_handle

                cancel_required = (
                    request_id in self.cancel_pending_requests
                    or request_id in self.inhibited_request_ids
                )
            else:
                cancel_required = True

        # Always attempt to obtain the terminal result,
        # even when cancellation was already requested.
        try:
            result_future = goal_handle.get_result_async()

            if result_future is None:
                raise RuntimeError("Result future missing.")

            if metadata is not None:
                result_future.add_done_callback(
                    lambda completed, rid=request_id:
                        self._planning_result_callback(
                            rid,
                            completed,
                        )
                )

        except Exception as exc:
            if metadata is not None:
                with self.state_lock:
                    self.inhibited_request_ids.add(request_id)
                    self.cancel_pending_requests.add(request_id)

                self._publish_generation_result(
                    request_id,
                    "TERMINATION_UNCONFIRMED",
                    "PLANNER_RESULT_FUTURE_UNAVAILABLE",
                )

            cancel_required = True

            self.get_logger().error(
                "[TRAJECTORY MANAGER] Cannot observe terminal result | "
                f"request={request_id} | error={exc}"
            )

        if not cancel_required:
            return

        try:
            cancel_future = goal_handle.cancel_goal_async()

            if cancel_future is None:
                raise RuntimeError("Cancel future missing.")

            if metadata is not None:
                cancel_future.add_done_callback(
                    lambda completed, rid=request_id:
                        self._planning_cancel_response_callback(
                            rid,
                            completed,
                        )
                )

        except Exception as exc:
            if metadata is not None:
                with self.state_lock:
                    self.inhibited_request_ids.add(request_id)
                    self.cancel_pending_requests.add(request_id)

                self._publish_generation_result(
                    request_id,
                    "TERMINATION_UNCONFIRMED",
                    "LATE_CANCEL_UNCONFIRMED",
                )

            self.get_logger().error(
                "[TRAJECTORY MANAGER] Late cancellation uncertain | "
                f"request={request_id} | error={exc}"
            )

    # ====================================================================
    # NAV2 RESULT
    # ====================================================================

    def _planning_result_callback_old(
        self,
        request_id: str,
        future,
    ) -> None:
        """
        Handle the terminal result of one Nav2 planning action.

        This callback is the terminal ownership barrier for a generation.

        A request that was canceled/superseded must never publish a candidate,
        even if Nav2 races with cancellation and returns a valid path.

        Only a still-current STATUS_SUCCEEDED request is allowed to publish
        CandidateTrajectory.
        """

        with self.state_lock:

            metadata = self.pending_requests.get(
                request_id
            )

            cancellation_pending = (
                request_id
                in self.cancel_pending_requests
            )

        if metadata is None:
            return

        try:
            wrapped_result = future.result()

        except Exception as exc:

            self.get_logger().error(
                "[TRAJECTORY MANAGER] "
                f"Planning result failed "
                f"for request_id={request_id}: {exc}"
            )

            self._remove_request(
                request_id
            )

            return

        if wrapped_result is None:

            self.get_logger().error(
                "[TRAJECTORY MANAGER] "
                f"Missing planning result "
                f"for request_id={request_id}."
            )

            self._remove_request(
                request_id
            )

            return

        status = int(
            wrapped_result.status
        )

        # ------------------------------------------------------------
        # SUPERSEDED / CANCELED GENERATION
        # ------------------------------------------------------------


        if cancellation_pending:
            self.get_logger().info(
                "[TRAJECTORY MANAGER] Terminal result after cancel | "
                f"request={request_id} | status={status}"
            )

            with self.state_lock:
                previously_inhibited = (
                    request_id in self.inhibited_request_ids
                )

            if previously_inhibited:
                self._publish_generation_result(
                    request_id,
                    "TERMINATED",
                    f"TERMINAL_AFTER_CANCEL_STATUS_{status}",
                )

            self._remove_request(request_id)
            return


        # ------------------------------------------------------------
        # NON-SUCCESSFUL PLANNER TERMINATION
        # ------------------------------------------------------------

        if status != GoalStatus.STATUS_SUCCEEDED:
            self._publish_generation_failure(
                request_id,
                f"PLANNING_FAILED_STATUS_{status}",
            )
            self._remove_request(request_id)
            return


        if wrapped_result.result is None:

            self.get_logger().error(
                "[TRAJECTORY MANAGER] "
                f"Missing successful planning payload "
                f"for request_id={request_id}."
            )

            self._remove_request(
                request_id
            )

            return

        path = wrapped_result.result.path

        if not path.poses:

            self.get_logger().warning(
                "[TRAJECTORY MANAGER] "
                f"Planner returned empty path "
                f"for request_id={request_id}."
            )

            self._remove_request(
                request_id
            )

            return

        # ------------------------------------------------------------
        # FINAL GENERATION OWNERSHIP CHECK
        #
        # No deferred replacement may exist at the instant a candidate is
        # published. If one does, this result has already become obsolete.
        # ------------------------------------------------------------

        with self.state_lock:

            replacement_waiting = (
                metadata.navigation_goal_id
                in self.deferred_requests_by_goal
            )

        if replacement_waiting:

            self.get_logger().info(
                "[TRAJECTORY MANAGER] "
                f"SUCCESS_DISCARDED_AS_SUPERSEDED="
                f"{request_id} | "
                f"GOAL={metadata.navigation_goal_id}"
            )

            self._remove_request(
                request_id
            )

            return

        path_id = self._generate_path_id()

        candidate = CandidateTrajectory()

        candidate.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        candidate.header.frame_id = str(
            path.header.frame_id
        )

        candidate.request_id = (
            metadata.request_id
        )

        candidate.path_id = (
            path_id
        )

        candidate.navigation_goal_id = (
            metadata.navigation_goal_id
        )

        candidate.path = (
            path
        )

        candidate.planner_id = (
            metadata.planner_id
        )

        candidate.generation_reason = (
            metadata.generation_reason
        )

        self.candidate_publisher.publish(
            candidate
        )

        self.get_logger().info(
            "[TRAJECTORY MANAGER] "
            f"Published candidate "
            f"request_id={metadata.request_id}, "
            f"path_id={path_id}, "
            f"navigation_goal_id="
            f"{metadata.navigation_goal_id}, "
            f"direction="
            f"{metadata.directional_constraint}, "
            f"poses={len(path.poses)}."
        )

        self._remove_request(
            request_id
        )

    def _planning_result_callback(self, request_id, future):
        """
        Handle one planning result with strict request ownership.

        An obsolete or inhibited generation cannot publish a path.
        A confirmed terminal result releases the request and may
        unblock a deferred generation of the same public goal.
        """
        request_id = str(request_id or "").strip()

        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

        if metadata is None:
            return

        try:
            wrapped_result = future.result()

            if wrapped_result is None:
                raise RuntimeError("Missing planning result.")

            status = int(wrapped_result.status)

            if status not in {
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED,
            }:
                raise RuntimeError(
                    f"Nonterminal planning result status: {status}"
                )

        except Exception as exc:
            with self.state_lock:
                if request_id not in self.pending_requests:
                    return

                self.inhibited_request_ids.add(request_id)

                notify = (
                    request_id not in self.timeout_notified_requests
                )
                self.timeout_notified_requests.add(request_id)

            self.get_logger().error(
                "[TRAJECTORY RESULT] Terminal status unavailable | "
                f"request={request_id} | error={exc}"
            )

            if notify:
                self._publish_generation_result(
                    request_id,
                    "TERMINATION_UNCONFIRMED",
                    "PLANNER_RESULT_UNCONFIRMED",
                )

            # Without a terminal status, ownership cannot
            # safely be released.
            return

        with self.state_lock:
            current = self.pending_requests.get(request_id)

            if current is None:
                return

            obsolete = (
                request_id in self.cancel_pending_requests
                or request_id in self.inhibited_request_ids
                or current.navigation_goal_id
                   in self.deferred_requests_by_goal
            )

            previously_uncertain = (
                request_id in self.inhibited_request_ids
            )

        if obsolete:
            if previously_uncertain:
                self._publish_generation_result(
                    request_id,
                    "TERMINATED",
                    f"TERMINAL_STATUS_{status}",
                )

            self.get_logger().info(
                "[TRAJECTORY RESULT] Obsolete result discarded | "
                f"request={request_id} | status={status}"
            )

            self._remove_request(request_id)
            return

        if status != GoalStatus.STATUS_SUCCEEDED:
            self._publish_generation_result(
                request_id,
                "FAILED",
                f"PLANNING_FAILED_STATUS_{status}",
            )
            self._remove_request(request_id)
            return

        result = wrapped_result.result

        if result is None or not hasattr(result, "path"):
            self._publish_generation_result(
                request_id,
                "FAILED",
                "PLANNING_RESULT_PAYLOAD_MISSING",
            )
            self._remove_request(request_id)
            return

        path = result.path

        if not path.header.frame_id or not path.poses:
            self._publish_generation_result(
                request_id,
                "FAILED",
                "INVALID_OR_EMPTY_PATH",
            )
            self._remove_request(request_id)
            return

        for pose in path.poses:
            position = pose.pose.position
            orientation = pose.pose.orientation

            values = (
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )

            if not all(math.isfinite(float(v)) for v in values):
                self._publish_generation_result(
                    request_id,
                    "FAILED",
                    "NONFINITE_PATH_GEOMETRY",
                )
                self._remove_request(request_id)
                return

            quaternion_norm = math.sqrt(
                orientation.x ** 2
                + orientation.y ** 2
                + orientation.z ** 2
                + orientation.w ** 2
            )

            if quaternion_norm < 1e-6:
                self._publish_generation_result(
                    request_id,
                    "FAILED",
                    "INVALID_PATH_ORIENTATION",
                )
                self._remove_request(request_id)
                return

        path_id = self._generate_path_id()

        candidate = CandidateTrajectory()
        candidate.header.stamp = self.get_clock().now().to_msg()
        candidate.header.frame_id = str(path.header.frame_id)

        candidate.request_id = metadata.request_id
        candidate.path_id = path_id
        candidate.navigation_goal_id = metadata.navigation_goal_id
        candidate.path = copy.deepcopy(path)
        candidate.planner_id = metadata.planner_id
        candidate.generation_reason = metadata.generation_reason

        # Atomic final ownership check and publication:
        # no cancellation can invalidate this result between
        # the check and candidate publication.
        published = False

        with self.state_lock:
            current = self.pending_requests.get(request_id)

            if (
                current is not None
                and request_id not in self.cancel_pending_requests
                and request_id not in self.inhibited_request_ids
                and current.navigation_goal_id
                    not in self.deferred_requests_by_goal
            ):
                self.candidate_publisher.publish(candidate)
                published = True

        if published:
            self.get_logger().info(
                "[TRAJECTORY MANAGER] Candidate published | "
                f"request={request_id} | "
                f"path={path_id} | "
                f"goal={metadata.navigation_goal_id} | "
                f"poses={len(path.poses)}"
            )
        else:
            self.get_logger().warning(
                "[TRAJECTORY MANAGER] Candidate suppressed | "
                f"request={request_id} | "
                "reason=REQUEST_SUPERSEDED"
            )

        self._remove_request(request_id)

    # ====================================================================
    # SOCIAL ZONES CALLBACK
    # ====================================================================

    def _social_zones_callback(self, msg):
        """Cache an ordered, complete snapshot for waypoint construction."""
        now = time.monotonic()
        limit = self._planning_geometry_max_age()

        try:
            session = str(msg.tracking_session_id).strip()
            stamp = msg.header.stamp

            source_ns = (
                int(stamp.sec) * 1_000_000_000
                + int(stamp.nanosec)
            )

            age = (
                self.get_clock().now().nanoseconds
                - source_ns
            ) / 1e9

            if (
                not session
                or not str(msg.header.frame_id).strip()
                or stamp.sec < 0
                or not 0 <= stamp.nanosec < 1_000_000_000
                or source_ns <= 0
                or age < -0.1
            ):
                return

            complete = bool(
                msg.source_observation_complete
                and msg.source_person_count == len(msg.zones)
                and age <= limit
            )

            updated = {}

            for zone in msg.zones:
                person_id = str(zone.person_id).strip()
                zone_stamp = zone.observation_stamp

                zone_ns = (
                    int(zone_stamp.sec) * 1_000_000_000
                    + int(zone_stamp.nanosec)
                )

                values = (
                    zone.center.x,
                    zone.center.y,
                    zone.center.z,
                    zone.heading,
                    zone.radius_longitudinal,
                    zone.radius_lateral,
                    zone.person_speed,
                )

                if (
                    not person_id
                    or person_id in updated
                    or zone.tracking_session_id != session
                    or zone_stamp.sec < 0
                    or not (
                        0 <= zone_stamp.nanosec
                        < 1_000_000_000
                    )
                    or zone_ns != source_ns
                    or zone.tracking_state != "TRACKED"
                    or not all(
                        math.isfinite(float(value))
                        for value in values
                    )
                    or zone.radius_longitudinal <= 0.0
                    or zone.radius_lateral <= 0.0
                    or zone.person_speed < 0.0
                ):
                    complete = False
                    break

                # Preserve the original observation timestamp.
                updated[person_id] = (
                    copy.deepcopy(zone),
                    copy.deepcopy(msg.header),
                    now,
                )

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            with self.state_lock:
                self.social_zones = {}
            return

        with self.state_lock:
            previous_session = getattr(
                self,
                "_planning_zone_session",
                "",
            )

            retired = getattr(
                self,
                "_planning_retired_sessions",
                set(),
            )

            if session in retired:
                return

            if (
                session == previous_session
                and self.last_social_stamp_ns is not None
                and source_ns <= self.last_social_stamp_ns
            ):
                return

            if (
                previous_session
                and previous_session != session
            ):
                retired.add(previous_session)

            self._planning_retired_sessions = retired
            self._planning_zone_session = session
            self.last_social_stamp_ns = source_ns

            # New incomplete evidence invalidates the previous snapshot.
            self.social_zones = updated if complete else {}

            # Preserve the last valid geometry for planning only.
            # New incomplete evidence invalidates the current snapshot.
            self.social_zones = updated if complete else {}

            # Preserve the last valid geometry for planning only.
            if complete:
                for person_id, entry in updated.items():
                    self._planning_zone_history[person_id] = (
                        copy.deepcopy(entry[0]),
                        copy.deepcopy(entry[1]),
                        now,
                    )

            # Expire planning hints independently of live observations.
            self._planning_zone_history = {
                person_id: entry
                for person_id, entry in self._planning_zone_history.items()
                if 0.0 <= now - entry[2] < 180.0
            }

            # Expire directional waypoint states.
            self._directional_waypoints = {
                key: state
                for key, state in self._directional_waypoints.items()
                if 0.0 <= now - state["created_at"] < 180.0
            }

        if not complete:
            self.get_logger().warning(
                "[PLANNING GEOMETRY] Snapshot unavailable | "
                f"source_age={age:.3f}s | "
                f"limit={limit:.3f}s | "
                f"coverage_complete="
                f"{msg.source_observation_complete}",
                throttle_duration_sec=2.0,
            )

    # ====================================================================
    # REQUEST LIFECYCLE
    # ====================================================================

    def _cleanup_stale_requests(self):
        now = time.monotonic()

        with self.state_lock:
            stale_ids = [
                request_id
                for request_id, metadata in self.pending_requests.items()
                if (
                    now - metadata.received_monotonic
                    >= self.pending_request_timeout_sec
                    and request_id not in self.timeout_notified_requests
                )
            ]

            self.timeout_notified_requests.update(stale_ids)
            self.inhibited_request_ids.update(stale_ids)

        for request_id in stale_ids:
            self._publish_generation_result(
                request_id,
                "TERMINATION_UNCONFIRMED",
                "PLANNING_TIMEOUT_CANCELLATION_PENDING",
            )

            self._cancel_and_remove_request(request_id)

    def _cancel_and_remove_request(self, request_id: str) -> None:
        """
        Request cancellation while retaining ownership.

        A cancellation request or acknowledgement is not proof
        that the planning action has terminated.

        The request is removed only by a confirmed terminal
        result or a confirmed goal rejection.
        """
        request_id = str(request_id or "").strip()

        if not request_id:
            return

        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

            if metadata is None:
                return

            if request_id in self.cancel_pending_requests:
                return

            self.cancel_pending_requests.add(request_id)

            goal_handle = self.active_goal_handles.get(request_id)

        if goal_handle is None:
            self.get_logger().warning(
                "[TRAJECTORY CANCEL] Waiting for goal response | "
                f"request={request_id} | "
                f"goal={metadata.navigation_goal_id}"
            )
            return

        try:
            future = goal_handle.cancel_goal_async()

            if future is None:
                raise RuntimeError(
                    "Cancellation future was not created."
                )

            future.add_done_callback(
                lambda completed, rid=request_id:
                    self._planning_cancel_response_callback(
                        rid,
                        completed,
                    )
            )

        except Exception as exc:
            with self.state_lock:
                if request_id not in self.pending_requests:
                    return

                self.inhibited_request_ids.add(request_id)

                notify = (
                    request_id not in self.timeout_notified_requests
                )

                self.timeout_notified_requests.add(request_id)

            self.get_logger().error(
                "[TRAJECTORY CANCEL] Submission uncertain | "
                f"request={request_id} | error={exc}"
            )

            if notify:
                self._publish_generation_result(
                    request_id,
                    "TERMINATION_UNCONFIRMED",
                    "CANCEL_SUBMISSION_UNCONFIRMED",
                )

            # Retain cancellation ownership.
            # Never re-enable candidate publication here.
            return

        self.get_logger().info(
            "[TRAJECTORY CANCEL] Requested | "
            f"request={request_id} | "
            f"goal={metadata.navigation_goal_id}"
        )

    def _planning_cancel_response_callback(
        self,
        request_id: str,
        future,
    ) -> None:
        """
        Process cancellation acknowledgement.

        Cancellation acknowledgement is not terminal confirmation.

        Keep the planning request owned until the corresponding
        result callback establishes a terminal Nav2 status.
        """
        request_id = str(request_id or "").strip()

        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

        if metadata is None:
            return

        try:
            response = future.result()

            if response is None:
                raise RuntimeError(
                    "Empty cancellation response."
                )

            accepted = bool(response.goals_canceling)

        except Exception as exc:
            accepted = False
            detail = str(exc)

        else:
            detail = "NAV2_CANCEL_NOT_CONFIRMED"

        if accepted:
            self.get_logger().info(
                "[TRAJECTORY CANCEL] Acknowledged | "
                f"request={request_id} | "
                "waiting_for_terminal_result=True"
            )
            return

        with self.state_lock:
            if request_id not in self.pending_requests:
                return

            # A missing acknowledgement must not resurrect
            # a canceled or superseded generation.
            self.cancel_pending_requests.add(request_id)
            self.inhibited_request_ids.add(request_id)

            notify = (
                request_id not in self.timeout_notified_requests
            )

            self.timeout_notified_requests.add(request_id)

        self.get_logger().error(
            "[TRAJECTORY CANCEL] Unconfirmed | "
            f"request={request_id} | "
            f"reason={detail} | "
            "request_retained=True"
        )

        if notify:
            self._publish_generation_result(
                request_id,
                "TERMINATION_UNCONFIRMED",
                "CANCEL_RESPONSE_UNCONFIRMED",
            )

    def _planning_geometry_max_age(self):
        """Configure a planning-only age limit, independent of validation."""
        name = "directional_planning_geometry_max_age_sec"

        if not self.has_parameter(name):
            self.declare_parameter(name, 2.0)

        value = float(self.get_parameter(name).value)

        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and positive."
            )

        return value
    
    def _generation_status_request_callback(self, msg):
        """
        Reconcile an uncertain planning transaction.

        A missing record never proves termination.
        Querying a live uncertain request inhibits its candidate.
        """
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
            ):
                return

            request_id = str(data["request_id"]).strip()
            goal_id = str(data["navigation_goal_id"]).strip()

            if not request_id or not goal_id:
                return

        except (ValueError, TypeError, KeyError):
            return

        with self.state_lock:
            terminal = self.terminal_generation_results.get(
                request_id
            )
            metadata = self.pending_requests.get(request_id)

            if terminal is not None:
                if terminal["navigation_goal_id"] != goal_id:
                    return

                payload = dict(terminal)
                goal_handle = None

            elif metadata is None:
                payload = {
                    "version": 1,
                    "producer": "trajectory_manager",
                    "navigation_goal_id": goal_id,
                    "request_id": request_id,
                    "status": "UNKNOWN",
                    "reason": "REQUEST_NOT_RECONCILED",
                }
                goal_handle = None

            else:
                if metadata.navigation_goal_id != goal_id:
                    return

                self.inhibited_request_ids.add(request_id)

                payload = None
                goal_handle = self.active_goal_handles.get(
                    request_id
                )

                outstanding = self.generation_reconcile_futures.get(
                    request_id
                )

                if (
                    outstanding is not None
                    and not outstanding.done()
                ):
                    goal_handle = None

        if payload is not None:
            self.generation_result_pub.publish(
                String(data=json.dumps(payload))
            )
            return

        self._cancel_and_remove_request(request_id)

        if goal_handle is None:
            return

        try:
            future = goal_handle.get_result_async()

            if future is None:
                raise RuntimeError("Result future missing.")

            with self.state_lock:
                if request_id not in self.pending_requests:
                    return

                self.generation_reconcile_futures[request_id] = future

            future.add_done_callback(
                lambda completed, rid=request_id:
                    self._planning_result_callback(rid, completed)
            )

        except Exception as exc:
            self.get_logger().warning(
                "[TRAJECTORY RECONCILIATION] Result unavailable | "
                f"request={request_id} | error={exc}"
            )

    def _remove_request(self, request_id: str) -> None:
        """
        Publish terminal generation evidence before releasing ownership.

        A TERMINATED result only proves that this planning transaction
        has finished. It does not authorize trajectory execution.
        """
        deferred_request = None
        terminal_payload = None

        with self.state_lock:
            metadata = self.pending_requests.get(request_id)

            if metadata is None:
                return

            goal_id = metadata.navigation_goal_id

            terminal_payload = {
                "version": 1,
                "producer": "trajectory_manager",
                "navigation_goal_id": goal_id,
                "request_id": request_id,
                "status": "TERMINATED",
                "reason": "GENERATION_TERMINAL_RECONCILED",
            }

            self.terminal_generation_results[
                request_id
            ] = dict(terminal_payload)

            while len(self.terminal_generation_results) > 512:
                oldest = next(
                    iter(self.terminal_generation_results)
                )

                self.terminal_generation_results.pop(
                    oldest,
                    None,
                )

            self.pending_requests.pop(request_id, None)
            self.active_goal_handles.pop(request_id, None)
            self.generation_reconcile_futures.pop(
                request_id,
                None,
            )

            self.cancel_pending_requests.discard(request_id)
            self.timeout_notified_requests.discard(request_id)
            self.inhibited_request_ids.discard(request_id)

            same_goal_alive = any(
                other.navigation_goal_id == goal_id
                for other in self.pending_requests.values()
            )

            if not same_goal_alive:
                deferred_request = (
                    self.deferred_requests_by_goal.pop(
                        goal_id,
                        None,
                    )
                )

        # Publish terminal evidence for the collaboration pipeline.
        # CandidateTrajectory may arrive before or after this result.
        self.generation_result_pub.publish(
            String(
                data=json.dumps(
                    terminal_payload,
                    separators=(",", ":"),
                )
            )
        )

        self.get_logger().info(
            "[TRAJECTORY MANAGER] Generation terminal published | "
            f"request={request_id} | "
            f"goal={goal_id} | "
            "status=TERMINATED"
        )

        # Start a deferred replacement only after the previous
        # generation has reached its terminal result.
        if deferred_request is not None:
            self._trajectory_generation_request_callback(
                deferred_request
            )
            
    # ====================================================================
    # IDENTIFIERS
    # ====================================================================

    @staticmethod
    def _generate_path_id() -> str:
        return f"path_{uuid.uuid4().hex}"


def main(args=None) -> None:

    rclpy.init(args=args)

    node = TrajectoryManager()

    executor = (
        rclpy.executors.MultiThreadedExecutor()
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:

        executor.remove_node(node)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()