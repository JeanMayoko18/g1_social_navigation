from types import SimpleNamespace

from g1_social_msgs.msg import SocialNavigationDecision

from g1_robot_collaboration.collaboration_manager_node import (
    CollaborationManager,
    STATE_OBSERVING,
    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
)


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def make_manager():
    manager = object.__new__(CollaborationManager)

    # Existing social episode: the person was already announced while
    # navigation was idle.
    manager.fsm_state = STATE_OBSERVING
    manager.state_entered_at = 0.0

    manager.active_decision_id = "social_decision_1"
    manager.active_person_id = "person_1"

    manager.human_proximity_active = True
    manager.social_observation_available = False
    manager.waiting_for_observation_reacquisition = True

    manager.conflict_observed_since = None
    manager.persistent_conflict_handled = False

    manager.announced_observe = True
    manager.current_announcement_action = ""
    manager.announced_navigation_goal_id = ""

    # Navigation started after the original OBSERVE announcement.
    manager.navigation_active = True
    manager.active_navigation_goal_id = "goal_1"

    manager.navigation_was_active_at_intrusion = False
    manager.nav2_cancel_requested = False
    manager.nav2_cancel_confirmed = False

    manager.stable_stop_confirmed = True
    manager.stable_stop_source = "test"

    # Minimal trajectory state required by invalidation.
    manager.trajectory_validation = "UNKNOWN"
    manager.validated_path_id = ""
    manager.validation_received_at = None
    manager.candidate_path_id = ""
    manager.candidate_navigation_goal_id = ""
    manager.pending_generation_request_id = ""

    manager.get_logger = lambda: FakeLogger()

    published = []

    manager._publish_collaborative_decision = (
        lambda **kwargs: published.append(kwargs)
    )

    return manager, published


def test_reacquisition_interrupts_navigation_in_existing_social_episode():
    manager, published = make_manager()

    msg = SocialNavigationDecision()
    msg.decision_id = "social_decision_2"
    msg.person_id = "person_1"

    manager._handle_observation_reacquired(msg)

    # The existing social episode must now interrupt navigation.
    assert manager.fsm_state == STATE_STOPPING_FOR_SOCIAL_CONFLICT

    assert manager.navigation_was_active_at_intrusion is True
    assert manager.nav2_cancel_requested is True
    assert manager.nav2_cancel_confirmed is False

    # A new physical-stop confirmation is required after WAIT.
    assert manager.stable_stop_confirmed is False
    assert manager.stable_stop_source == ""

    # Reacquisition has restored factual observation.
    assert manager.human_proximity_active is True
    assert manager.social_observation_available is True
    assert manager.waiting_for_observation_reacquisition is False

    # Exactly one navigation command must have been emitted.
    assert len(published) == 1

    decision = published[0]

    assert decision["action"] == "WAIT"
    assert decision["outcome"] == "SOCIAL_CONFLICT"
    assert decision["human_intent"] == "UNKNOWN"
    assert decision["directional_constraint"] == "NONE"
    assert decision["path_id"] == ""

    assert (
        decision["reason"]
        == "NAVIGATION_STARTED_DURING_SOCIAL_INTRUSION"
    )

'''
cd ~/g1_ws

source /opt/ros/humble/setup.bash
source ~/g1_ws/install/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q \
src/g1_robot_collaboration/test/test_existing_social_episode_navigation.py

'''