std_msgs/Header header

# True only after BOTH command and measured odometry have remained
# compatible with a navigation stop for a continuous dwell period.
#
# A Nav2 cancellation, a zero cmd_vel sample, or a single low-speed
# odometry sample MUST NOT confirm stable stop by itself.
bool stable_stop_confirmed

# Diagnostic measured planar speeds.
float32 linear_speed
float32 angular_speed

# Diagnostic predicates used by the temporal stop estimator.
bool command_quiet
bool odometry_quiet

# Continuous duration for which both predicates have remained true.
float32 stable_duration

# Human-readable description of the estimator/source.
string source