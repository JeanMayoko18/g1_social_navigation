# Processing state of one observation feature.
uint8 NOT_PROCESSED=0
uint8 UNAVAILABLE=1
uint8 AVAILABLE=2

uint8 status

# Reference observation timestamp and feature coordinate frame.
# AVAILABLE:
#   stamp equals the candidate's source observation timestamp.
#   frame_id identifies geometry coordinates or the source image frame.
# NOT_PROCESSED / UNAVAILABLE:
#   stamp is zero and frame_id is empty.
std_msgs/Header header

# Reliability in [0, 1].
# NOT_PROCESSED / UNAVAILABLE require quality == 0.
# AVAILABLE with quality == 0 remains inspectable but cannot support matching.
float64 quality

# Machine-readable explanation, for example:
# DEPTH_TIMEOUT, TF_UNAVAILABLE, INSUFFICIENT_PIXELS, VALID.
string reason