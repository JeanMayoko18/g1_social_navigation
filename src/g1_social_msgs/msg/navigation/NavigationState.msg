std_msgs/Header header

# Current navigation lifecycle snapshot published by the goal gate.

# Public goal UUID encoded as a hexadecimal string.
# Empty when no goal is known or concurrent goals make the state ambiguous.
string navigation_goal_id

# Expected values:
#   UNKNOWN
#   IDLE
#   WAITING_FOR_SPEECH
#   FORWARDING
#   NAVIGATING
#   CANCELING
#   SUCCEEDED
#   ABORTED
#   CANCELED
string navigation_state

# True only when the internal navigation goal is executing.
bool navigation_active