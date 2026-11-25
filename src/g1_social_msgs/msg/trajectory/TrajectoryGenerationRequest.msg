# TrajectoryGenerationRequest.msg

std_msgs/Header header

# Unique request identifier.
string request_id

# Public navigation goal for which a trajectory is requested.
string navigation_goal_id

# Final navigation target.
geometry_msgs/PoseStamped goal

# Optional explicit start pose.
# When use_start == false, Nav2 uses the current robot pose.
geometry_msgs/PoseStamped start
bool use_start

# Empty means Nav2 default planner.
string planner_id

# INITIAL / REPLAN / ALTERNATIVE
string generation_reason

# NONE / LEFT / RIGHT
# LEFT and RIGHT are relative to the person's reference frame.
string directional_constraint

# Person associated with the constraint, when applicable.
string person_id
