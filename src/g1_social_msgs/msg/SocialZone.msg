# Persistent person identifier associated with this social zone.
string person_id

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
