#!/usr/bin/env bash

# Stop on errors and pipeline failures.
# Do not enable nounset before ROS 2 setup.
set -eo pipefail

# =====================================================
# OFFLINE ROS 2 ENVIRONMENT
# =====================================================
# Load ROS 2 and the isolated offline environment.
source "$HOME/g1_ws/offline_ros_env.sh"

# Enable strict variable checking only after ROS setup.
set -u

# Generate a fresh ROS timestamp using the system clock.
# Valid when ROS nodes use use_sim_time=false.
refresh_ros_stamp() {
    local now_ns

    now_ns="$(date +%s%N)"

    ROS_STAMP_SEC=$((now_ns / 1000000000))
    ROS_STAMP_NSEC=$((now_ns % 1000000000))
}

STAMP="$(date +%s%N)"

GOAL_ID="offline_goal_${STAMP}"

# Unique offline producer session.
SOCIAL_SESSION="offline_session_${STAMP}"

# One social episode for this test.
SOCIAL_EPISODE="offline_episode_${STAMP}"

# First decision in this episode.
SOCIAL_SEQUENCE=1

DECISION_ID="${SOCIAL_SESSION}:${SOCIAL_EPISODE}:${SOCIAL_SEQUENCE}"

LOG="$HOME/g1_ws/log/offline_hri/collaboration_manager.log"

if [[ ! -f "$LOG" ]]; then
    echo "ERROR: CollaborationManager log not found:"
    echo "  $LOG"
    exit 1
fi

LOG_START=$(wc -c < "$LOG")

DIALOGUE_LOG="/tmp/offline_dialogue_request.log"
rm -f "$DIALOGUE_LOG"

PYTHONUNBUFFERED=1 timeout 120s python3 -u - <<'PY' \
    > "$DIALOGUE_LOG" 2>&1 &
import rclpy
from rclpy.node import Node
from g1_social_msgs.msg import DialogueRequest


class OfflineDialogueMonitor(Node):
    def __init__(self):
        super().__init__("offline_dialogue_monitor")

        self.subscription = self.create_subscription(
            DialogueRequest,
            "/hri/dialogue_request",
            self.on_dialogue_request,
            10,
        )

        self.get_logger().info(
            "Offline dialogue monitor ready."
        )

    def on_dialogue_request(self, msg):
        print(
            f"interaction_id: {msg.interaction_id}",
            flush=True,
        )


def main():
    rclpy.init()
    node = OfflineDialogueMonitor()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
PY


ECHO_PID=$!

sleep 1

echo "=== 1/5 GOAL CONTEXT ==="

refresh_ros_stamp

ros2 topic pub -w 1 --once \
  /social_navigation/navigation_goal_context \
  g1_social_msgs/msg/NavigationGoalContext \
  "{header: {
      stamp: {
        sec: ${ROS_STAMP_SEC},
        nanosec: ${ROS_STAMP_NSEC}
      }
    },
    navigation_goal_id: '${GOAL_ID}',
    goal: {
      header: {frame_id: 'map'},
      pose: {
        position: {x: 2.0, y: 0.0, z: 0.0},
        orientation: {w: 1.0}
      }
    }}"

sleep 2

echo "=== 2/5 NAVIGATING ==="

refresh_ros_stamp

ros2 topic pub -w 1 --once \
  /social_navigation/navigation_state \
  g1_social_msgs/msg/NavigationState \
  "{header: {
      stamp: {
        sec: ${ROS_STAMP_SEC},
        nanosec: ${ROS_STAMP_NSEC}
      }
    },
    navigation_goal_id: '${GOAL_ID}',
    navigation_state: 'NAVIGATING',
    navigation_active: true}"

sleep 2

echo "=== 3/5 INTRUSION ==="

refresh_ros_stamp

ros2 topic pub -w 1 --once \
  /social_navigation/decision \
  g1_social_msgs/msg/SocialNavigationDecision \
  "{header: {
      stamp: {
        sec: ${ROS_STAMP_SEC},
        nanosec: ${ROS_STAMP_NSEC}
      }
    },
    decision_id: '${DECISION_ID}',
    event: 'INTRUSION_CONFIRMED',
    navigation_state: 'NAVIGATING',
    navigation_goal_id: '${GOAL_ID}',
    navigation_active: true,
    social_conflict_active: true,
    person_id: 'pc_test_person',
    zone_type: 'PERSONAL',
    tracking_state: 'TRACKED',
    person_position: {x: 1.0, y: 0.5, z: 0.0},
    person_distance: 1.1,
    social_radius: 1.5,
    person_speed: 0.0,
    occupancy_ratio: 1.0,
    requires_explanation: true}"

sleep 2

echo "=== 4/5 NAVIGATION STOPPED ==="

refresh_ros_stamp

ros2 topic pub -w 1 --once \
  /social_navigation/navigation_state \
  g1_social_msgs/msg/NavigationState \
  "{header: {
      stamp: {
        sec: ${ROS_STAMP_SEC},
        nanosec: ${ROS_STAMP_NSEC}
      }
    },
    navigation_goal_id: '${GOAL_ID}',
    navigation_state: 'WAITING',
    navigation_active: false}"

sleep 2

echo "=== 5/5 PHYSICAL STOP CONFIRMED ==="

# Offline-only simulated motion monitor.
# One ROS process, one persistent publisher.
# No connection to any robot motion backend.

python3 - <<'PY'
import time

import rclpy
from g1_social_msgs.msg import RobotMotionState


rclpy.init()

node = rclpy.create_node("offline_motion_state_injector")

publisher = node.create_publisher(
    RobotMotionState,
    "/social_navigation/robot_motion_state",
    10,
)

# Allow DDS discovery before publishing.
time.sleep(1.0)

start = time.monotonic()
period = 0.1
duration = 8.0

try:
    while rclpy.ok():
        now = time.monotonic()
        elapsed = now - start

        if elapsed >= duration:
            break

        msg = RobotMotionState()

        msg.header.stamp = node.get_clock().now().to_msg()

        msg.stable_stop_confirmed = True
        msg.linear_speed = 0.0
        msg.angular_speed = 0.0
        msg.command_quiet = True
        msg.odometry_quiet = True
        msg.stable_duration = float(elapsed)
        msg.source = "OFFLINE_TEST"

        publisher.publish(msg)

        time.sleep(period)

finally:
    node.destroy_node()
    rclpy.shutdown()

print("Offline motion-state publication completed.")
PY

echo
echo "=== INJECTION DONE ==="

echo "Waiting for CollaborationManager to open the human initiative window..."
echo

TIMEOUT=35
START=$(date +%s)

while true; do

    if grep -q 'interaction_id:' "$DIALOGUE_LOG" 2>/dev/null; then
        echo
        echo "========== DIALOGUE REQUEST =========="
        cat "$DIALOGUE_LOG"
        echo
        echo "======================================"
        echo " HRI WINDOW REQUESTED"
        echo "======================================"
        kill "$ECHO_PID" 2>/dev/null || true
        wait "$ECHO_PID" 2>/dev/null || true
        exit 0
    fi

    if ! kill -0 "$ECHO_PID" 2>/dev/null; then
        echo "ERROR: Dialogue monitor stopped unexpectedly."
        cat "$DIALOGUE_LOG"
        exit 1
    fi

    if (( $(date +%s) - START >= TIMEOUT )); then
        echo
        echo "ERROR: CollaborationManager published no DialogueRequest."
        echo
        echo "=== NEW COLLABORATION EVENTS ==="
        tail -c "+$((LOG_START + 1))" "$LOG"

        kill "$ECHO_PID" 2>/dev/null || true
        wait "$ECHO_PID" 2>/dev/null || true
        exit 1
    fi

    sleep 0.2
done

echo "=== DONE ==="