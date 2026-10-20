import threading
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
from g1_social_msgs.msg import CandidateTrajectory, CollaborativeDecision

from g1_nav2.navigation_goal_gate_node import NavigationGoalGateNode


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def make_gate():
    gate = object.__new__(NavigationGoalGateNode)

    gate.navigation_state_lock = threading.Lock()
    gate.trajectory_lock = threading.Lock()
    gate.internal_goal_lock = threading.Lock()

    gate.public_navigation_goals = {}
    gate.public_navigation_phases = {}
    gate.navigation_control_states = {}
    gate.navigation_control_requests = {}

    gate.candidate_trajectories = {}
    gate.authorized_resume_path_ids = {}
    gate.pending_resume_path_ids = {}
    gate.internal_goal_handles = {}
    gate.internal_cancel_pending = set()

    gate.get_logger = lambda: FakeLogger()
    gate._publish_navigation_state = lambda: None

    return gate


def active_handle():
    return SimpleNamespace(is_active=True)


def candidate(goal_id, path_id):
    msg = CandidateTrajectory()
    msg.navigation_goal_id = goal_id
    msg.path_id = path_id

    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.orientation.w = 1.0

    msg.path.poses.append(pose)

    return msg


def decision(goal_id, request_id, path_id):
    msg = CollaborativeDecision()
    msg.navigation_goal_id = goal_id
    msg.request_id = request_id
    msg.action = "RESUME"
    msg.navigation_authorized = True
    msg.path_id = path_id

    return msg


def test_unknown_path_is_rejected():
    gate = make_gate()

    goal_id = "goal_1"

    gate.public_navigation_goals[goal_id] = active_handle()
    gate.navigation_control_states[goal_id] = gate.CONTROL_WAITING

    gate._collaborative_decision_callback(
        decision(goal_id, "request_1", "unknown_path")
    )

    assert goal_id not in gate.authorized_resume_path_ids
    assert goal_id not in gate.pending_resume_path_ids
    assert gate.navigation_control_states[goal_id] == gate.CONTROL_WAITING


def test_candidate_from_another_goal_is_rejected():
    gate = make_gate()

    goal_id = "goal_1"

    gate.public_navigation_goals[goal_id] = active_handle()
    gate.navigation_control_states[goal_id] = gate.CONTROL_WAITING

    gate.candidate_trajectories["path_1"] = candidate(
        "another_goal",
        "path_1",
    )

    gate._collaborative_decision_callback(
        decision(goal_id, "request_1", "path_1")
    )

    assert goal_id not in gate.authorized_resume_path_ids
    assert goal_id not in gate.pending_resume_path_ids
    assert gate.navigation_control_states[goal_id] == gate.CONTROL_WAITING


def test_resume_is_deferred_while_old_attempt_exists():
    gate = make_gate()

    goal_id = "goal_1"
    path_id = "path_1"

    gate.public_navigation_goals[goal_id] = active_handle()
    gate.navigation_control_states[goal_id] = gate.CONTROL_WAITING

    gate.candidate_trajectories[path_id] = candidate(
        goal_id,
        path_id,
    )

    # Previous NavigateToPose attempt still exists.
    gate.internal_goal_handles[goal_id] = object()
    gate.internal_cancel_pending.add(goal_id)

    gate._collaborative_decision_callback(
        decision(goal_id, "request_1", path_id)
    )

    assert gate.pending_resume_path_ids[goal_id] == path_id
    assert goal_id not in gate.authorized_resume_path_ids

    # Critical invariant: RESUME must not start another attempt yet.
    assert gate.navigation_control_states[goal_id] == gate.CONTROL_WAITING


def test_pending_resume_activates_after_old_attempt_is_gone():
    gate = make_gate()

    goal_id = "goal_1"
    path_id = "path_1"

    gate.public_navigation_goals[goal_id] = active_handle()
    gate.navigation_control_states[goal_id] = gate.CONTROL_WAITING

    gate.candidate_trajectories[path_id] = candidate(
        goal_id,
        path_id,
    )

    gate.pending_resume_path_ids[goal_id] = path_id

    # Simulates confirmed terminal state of previous attempt.
    gate.internal_goal_handles.pop(goal_id, None)
    gate.internal_cancel_pending.discard(goal_id)

    activated = gate._activate_pending_resume(goal_id)

    assert activated is True

    assert goal_id not in gate.pending_resume_path_ids

    assert gate.authorized_resume_path_ids[goal_id] == path_id

    assert (
        gate.navigation_control_states[goal_id]
        == gate.CONTROL_RUNNING
    )

    assert (
        gate.public_navigation_phases[goal_id]
        == gate.PHASE_RESUMING
    )