# Directional constraint lifecycle and enforcement status.
# ACTIVE is valid only when both Nav2 costmaps have been verified.

std_msgs/Header header

string request_id
string navigation_goal_id
string interaction_id

string person_id
string tracking_session_id
uint64 identity_generation

# NONE, LEFT or RIGHT.
string directional_constraint

# INACTIVE, INSTALLING, PUBLISHED, ACTIVE, REVOKED or FAILED.
# PUBLISHED means that the mask was published.
# ACTIVE requires verified application by both Nav2 costmaps.
string state

# Unique geometry version.
string mask_id

# Evidence of application by the global and local costmaps.
bool global_costmap_confirmed
bool local_costmap_confirmed

# Human-readable diagnostic.
string reason