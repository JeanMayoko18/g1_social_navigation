# Request to install or revoke a person-relative navigation constraint.
# This message never authorizes robot motion.

std_msgs/Header header

# Unique request identifier for idempotency.
string request_id

# Public navigation goal UUID.
string navigation_goal_id

# Social interaction episode identifier.
string interaction_id

# Identity of the authorized person.
string person_id
string tracking_session_id
uint64 identity_generation

# INSTALL or REVOKE.
string action

# NONE, LEFT or RIGHT, relative to the person's reference frame.
string directional_constraint

# Maximum time the constraint may remain valid without renewal.
builtin_interfaces/Duration validity_duration