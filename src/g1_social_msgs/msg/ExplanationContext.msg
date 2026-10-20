# Structured context prepared for the explainability stack.
#
# This message contains deterministic facts coming from the navigation
# decision pipeline. It is intended to be consumed by an LLM provider
# or by a deterministic fallback explanation generator.

std_msgs/Header header

# Original deterministic decision identifier.
string decision_id

# Navigation lifecycle state observed when this decision was created.
# Expected values:
#   UNKNOWN: navigation state has not been established
#   IDLE: no active navigation goal
#   WAITING_FOR_SPEECH: the gate is holding a goal before forwarding
#   NAVIGATING: the navigation action is executing
#   CANCELING: cancellation is in progress
#   SUCCEEDED: the current goal completed successfully
#   ABORTED: the current goal failed
#   CANCELED: the current goal was canceled
string navigation_state

# Public NavigateToPose goal UUID encoded as a hexadecimal string.
# Empty when no public goal is associated with this decision.
string navigation_goal_id

# True only when the navigation action is executing.
# False while waiting for speech, canceling, idle, or in a terminal state.
bool navigation_active

# Whether a confirmed social intrusion exists independently of navigation.
# A person may be inside the safety boundary while the robot has no goal.
bool social_conflict_active

# Requested output language.
# Expected values:
#   es
#   en
string language

# Navigation/collaboration action being explained.
# Expected values:
#   WAIT
#   REPLAN
#   RESUME
#   STOP
string action

# Machine-readable reason associated with the collaboration event.
# Examples:
#   SOCIAL_ZONE_INTRUSION
#   PATH_CLEAR
#   SOCIAL_CONFLICT_PERSISTENT
#   HUMAN_DENIED_PASSAGE
#   HUMAN_REQUESTED_STOP
#   TRAJECTORY_CONFLICT
#   TRAJECTORY_VALID
string reason_code

# Person associated with the event.
string person_id

# Social-zone classification.
# Expected values:
#   NONE
#   STATIC
#   DYNAMIC
string zone_type

# Source person tracking state.
string tracking_state

# Person position in the shared reference frame.
geometry_msgs/Point person_position

# Planar distance from robot to person.
float32 person_distance

# Effective social radius used by the decision layer.
# Configured intrusion entry distance in meters.
# Independent from the Gaussian costmap influence and release hysteresis.
float32 social_radius

# Current estimated person speed.
float32 person_speed

# Social-zone occupancy ratio.
float32 occupancy_ratio

# Whether the underlying deterministic decision requests replanning.
bool requires_replan

# Machine-oriented semantic summary of what happened.
string situation_summary

# Machine-oriented description of the robot's intended response.
string robot_intention

# Instruction defining what the future language model should explain.
string explanation_goal
