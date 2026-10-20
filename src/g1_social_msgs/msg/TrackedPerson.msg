# Persistent person identifier inherited from the perception tracker.
string id

# Semantic class information.
int32 class_id
string class_name

# Latest detector confidence.
float32 confidence

# Current person position in the parent array frame.
geometry_msgs/Point position

# Estimated planar velocity in the parent array frame.
geometry_msgs/Vector3 velocity

# Estimated planar speed in meters per second.
float32 speed

# Estimated planar heading of the motion vector, in radians.
float32 motion_heading

# Time elapsed since this person was last observed.
float32 time_since_last_seen

# Number of successful spatial observations associated with this track.
uint32 observation_count

# Spatial tracking state.
# Expected values:
#   TENTATIVE
#   TRACKED
#   LOST
string tracking_state
