# Coordinate frame and source timestamp shared by all social zones.
std_msgs/Header header

# Tracking producer session associated with this source array.
# Must also be populated when zones is empty.
string tracking_session_id

# Number of retained persons in the exact source array.
uint32 source_person_count

# True only when every source person has a current TRACKED
# observation and a corresponding generated social zone.
#
# False does not mean that the scene is empty.
bool source_observation_complete

# Social zones derived from this exact source array.
SocialZone[] zones