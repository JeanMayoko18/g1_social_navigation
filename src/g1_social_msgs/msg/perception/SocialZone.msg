# Persistent person identifier associated with this social zone.
string person_id

# Identity generation of the person represented by this zone.
uint64 identity_generation

# Center of the social zone in the parent array frame.
geometry_msgs/Point center

# Zone orientation in radians.
# For moving persons, this follows the motion heading.
float32 heading

# Longitudinal radius in meters.
# This is the forward/backward semi-axis for directional zones.
float32 radius_longitudinal

# Lateral radius in meters.
# This is the side semi-axis.
float32 radius_lateral

# Current person speed in meters per second.
float32 person_speed

# Zone behavior classification.
# Expected values:
#   STATIC
#   DYNAMIC
string zone_type

# Source spatial tracking state.
string tracking_state

# Tracking producer session associated with person_id.
string tracking_session_id

# Source timestamp of the actual observation used to build this zone.
builtin_interfaces/Time observation_stamp
