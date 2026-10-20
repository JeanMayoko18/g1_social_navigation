# Factual social-navigation event generated from the current social zones.
#
# This message represents a deterministic observation of the current social
# situation and the navigation context in which that observation was made.
#
# It does NOT command navigation behavior.
# STOP, WAIT, REPLAN, RESUME and dialogue policy are owned exclusively by
# CollaborationManager.
#
# An LLM may explain an event downstream, but must never modify, create,
# suppress, or reinterpret the social-navigation event.

std_msgs/Header header


# Unique identifier for this emitted social-navigation event.
#
# This identifier is unique for each published event message.
# It is not the identifier of the complete social interaction episode.
string decision_id


# Factual social-navigation event.
#
# Expected values:
#   INTRUSION_CONFIRMED:
#       A tracked person has continuously satisfied the configured intrusion
#       condition for the required confirmation duration.
#
#   OBSERVATION_LOST:
#       A previously confirmed social episode can no longer be reliably
#       observed. This does NOT mean that the social conflict has cleared.
#
#   CLEAR:
#       Every person involved in the active social episode has been observed
#       beyond the configured release boundary for the required confirmation
#       duration.
#
# CONFLICT_PERSISTENT is part of the shared social-navigation vocabulary but
# is not currently emitted by SocialNavigationDecisionNode. Persistence policy
# belongs to the collaboration FSM.
string event


# Navigation lifecycle state observed when this event was created.
#
# Expected values are defined by g1_social_common.navigation_states.
#
# Current canonical values:
#   UNKNOWN:
#       Navigation state has not been established or the latest state is stale.
#
#   IDLE:
#       No navigation goal is currently executing.
#
#   NAVIGATING:
#       A navigation goal is actively executing.
#
#   SUCCEEDED:
#       The current navigation goal completed successfully.
#
#   ABORTED:
#       The current navigation goal failed.
#
#   CANCELED:
#       The current navigation goal was canceled.
#
# This field is contextual information only. It does not authorize movement.
string navigation_state


# Public NavigateToPose goal UUID encoded as a hexadecimal string.
#
# Empty when no public navigation goal is associated with the current
# navigation context.
string navigation_goal_id


# True only when a valid navigation goal is currently executing.
#
# False when navigation is idle, unknown, stale, or in a terminal state.
#
# This field describes navigation activity at event publication time.
# It does not indicate whether movement is socially permitted.
bool navigation_active


# Whether the current social episode remains unresolved.
#
# True:
#   A confirmed social intrusion exists, or observation of an already
#   confirmed intrusion has been lost.
#
# False:
#   No confirmed conflict is active, or observed clearance has been
#   explicitly confirmed.
#
# IMPORTANT:
#   Observation loss must never set this field to false.
bool social_conflict_active


# Persistent identifier of the person currently associated with the event.
#
# For INTRUSION_CONFIRMED, this identifies the currently most constraining
# tracked person.
#
# For OBSERVATION_LOST, the last known responsible person identifier may be
# preserved without reusing stale geometry.
#
# For CLEAR, the identifier of the previously constraining person may be
# preserved as event context.
string person_id


# Social-zone type associated with the current observation.
#
# Expected values:
#   NONE
#   STATIC
#   DYNAMIC
#
# NONE is used when no current social-zone geometry can safely be attached
# to the event, for example after observation loss or confirmed clearance.
string zone_type


# Source spatial tracking state associated with the current observation.
#
# Expected values:
#   NONE
#   TENTATIVE
#   TRACKED
#   LOST
#
# NONE is used when no current tracking observation can safely be attached
# to the event.
string tracking_state


# Current observed person position relative to the robot.
#
# The position is expressed in the frame identified by header.frame_id,
# normally base_footprint.
#
# This geometry must not be interpreted as current when the event is
# OBSERVATION_LOST.
geometry_msgs/Point person_position


# Euclidean planar distance from the robot origin to the observed person,
# expressed in meters.
#
# This value is meaningful only when current person geometry is available.
float32 person_distance


# Effective social-zone radius in the direction of the robot, expressed
# in meters.
#
# This value describes the current social geometry associated with the
# observed person. It is distinct from the fixed intrusion-distance threshold
# used to confirm entry into a social conflict.
float32 social_radius


# Current estimated planar speed of the observed person, in meters per second.
#
# This value is meaningful only when current tracking information is available.
float32 person_speed


# Ratio describing the relationship between the effective social-zone radius
# and the robot-to-person distance.
#
# occupancy_ratio = social_radius / person_distance
#
# > 1.0:
#     The robot lies inside the effective social-zone radius.
#
# = 1.0:
#     The robot lies on the effective social-zone boundary.
#
# < 1.0:
#     The robot lies outside the effective social-zone radius.
#
# This value is descriptive and may be used to rank simultaneous social
# constraints. It does not directly command navigation behavior.
float32 occupancy_ratio


# True when the downstream explainability stack may generate an explanation
# for this event.
#
# This flag is event-driven and must not become true for every perception
# update.
#
# It authorizes explanation generation only. It never authorizes speech
# timing, navigation, replanning, or robot motion.
#
# In particular, CollaborationManager remains responsible for ensuring that
# any detection announcement occurs only after the required stop sequence
# and confirmed physical immobilization.
bool requires_explanation