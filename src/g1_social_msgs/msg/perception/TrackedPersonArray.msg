# Coordinate frame and timestamp shared by all tracked persons.
std_msgs/Header header

# Tracked persons currently retained by the spatial tracker.
TrackedPerson[] persons
# Producer session, including when persons is empty.
string tracking_session_id

# Includes source persons with unresolved identity or unavailable geometry.
uint32 source_detection_count
# False if any upstream person could not be represented reliably.
bool source_observation_complete
