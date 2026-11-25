# Persistent person identifier inherited from the perception tracker.
string id

# Generation assigned by the identity authority.
# Distinguishes successive incarnations of the same person ID.
uint64 identity_generation

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

# Identity-authority session, not the spatial node process session.
string tracking_session_id
builtin_interfaces/Time last_observation_stamp
bool observed_in_current_array
# True only for a current admissible derivative in the stated frame.
bool velocity_valid
