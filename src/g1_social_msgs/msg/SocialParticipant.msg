# Identity of the spatial tracker producer.
string tracking_session_id

# Person identity within that tracker session.
string person_id

# True only when current actual geometry is available.
bool observed

# True only for a currently observed, confirmed episode participant
# still inside the social release boundary.
bool conflict