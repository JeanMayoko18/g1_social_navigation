std_msgs/Header header

# Identifier of the TrajectoryGenerationRequest that produced this candidate.
# Preserved unchanged from request to candidate for strict correlation.
string request_id

# Unique immutable identifier for this exact candidate trajectory.
# Every newly computed path MUST receive a new path_id.
string path_id

# Public navigation goal to which this trajectory belongs.
string navigation_goal_id

# Exact path returned by Nav2 ComputePathToPose.
# This exact path is the object that must later be socially validated
# and, after authorization, executed through FollowPath.
nav_msgs/Path path

# Planner used to generate this trajectory.
# Empty means that Nav2 selects its configured default planner.
string planner_id

# Reason why this candidate was generated.
#
# Expected values:
#   INITIAL
#   REPLAN
#   ALTERNATIVE
string generation_reason