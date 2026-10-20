std_msgs/Header header

# Unique identifier of the collaboration episode.
string interaction_id

# Navigation goal associated with this decision.
string navigation_goal_id

# Request identifier used for execution acknowledgement and deduplication.
string request_id

# Person associated with the social situation.
string person_id

# Navigation action requested from the execution layer.
#
# Allowed values:
#   WAIT
#   RESUME
#   STOP
#   REPLAN
#
# Semantics:
#
#   STOP:
#       Request navigation cancellation / stopping.
#       navigation_authorized = false
#
#   WAIT:
#       Keep navigation execution inhibited.
#       navigation_authorized = false
#
#   REPLAN:
#       Request computation of a new candidate trajectory.
#       This NEVER authorizes execution of that trajectory.
#       navigation_authorized = false
#       requires_replan = true
#
#   RESUME:
#       Authorize execution of the exact trajectory that received a
#       fresh VALID social-trajectory validation.
#       navigation_authorized = true
#       requires_replan = false
string action

# Collaborative outcome associated with this decision.
#
# Examples:
#   SOCIAL_CONFLICT
#   WAITING
#   HUMAN_CONFIRMED
#   HUMAN_REJECTED
#   HUMAN_REQUESTED_STOP
#   TRAJECTORY_VALIDATED
#   TRAJECTORY_CONFLICT
#   OBSERVATION_LOST
#   TIMEOUT
#   FALLBACK
string outcome

# Confirmed human intention contributing to this decision.
#
# Examples:
#   UNKNOWN
#   ALLOW_PASSAGE
#   DENY_PASSAGE
#   REQUEST_STOP
#   REQUEST_FOLLOW
#
# IMPORTANT:
# A pending/unconfirmed interpretation must never be published here
# as though it were confirmed.
string human_intent

# Human-relative directional constraint associated with the decision.
#
# Allowed values:
#   NONE
#   LEFT
#   RIGHT
#
# IMPORTANT:
# LEFT and RIGHT are always expressed in the PERSON'S reference frame.
string directional_constraint

# Exact trajectory/path associated with RESUME or REPLAN when available.
#
# For RESUME this must identify the exact trajectory that received
# fresh VALID social validation.
string path_id

# Reason for the decision.
string reason

# True only when execution of path_id is explicitly authorized.
#
# This may be true only for action == RESUME after fresh positive
# validation of that exact path.
bool navigation_authorized

# True only when a new candidate trajectory must be computed.
#
# REPLAN is a planning request, not movement authorization.
bool requires_replan