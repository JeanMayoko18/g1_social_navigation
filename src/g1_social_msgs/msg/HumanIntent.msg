std_msgs/Header header

# Dialogue interaction associated with this intention.
string interaction_id
string person_id

# Interpreted human intention.
#
# Expected values include:
#   ALLOW_PASSAGE
#   DENY_PASSAGE
#   REQUEST_STOP
#   REQUEST_FOLLOW
#   CONFIRM
#   REJECT
#   UNKNOWN
#
# CONFIRM and REJECT are responses to a pending semantic
# confirmation or alternative proposal. Their meaning must be
# resolved using the pending dialogue state held by
# CollaborationManager.
string intent

# Optional spatial direction associated with the intent.
#
# Examples:
#   LEFT
#   RIGHT
#   FRONT
#   BACK
#   NONE
#
# For ALLOW_PASSAGE:
#
#   NONE:
#       Passage is allowed without an explicit directional preference.
#
#   LEFT:
#       Passage is allowed on the person's left side.
#
#   RIGHT:
#       Passage is allowed on the person's right side.
#
# IMPORTANT:
# LEFT and RIGHT are ALWAYS expressed in the PERSON'S reference frame.
# They never refer to the robot's left or right.
string direction

# Optional target associated with the intention.
string target

# Optional constraint expressed by the human.
string constraint

# Confidence of the intent interpretation.
float32 confidence

# Original utterance or normalized transcription.
string utterance

# Source of the interpreted intention.
#
# Expected values include:
#   DIRECT_SPEECH
#   RESPONSE_TO_ROBOT
#   GESTURE
#   TIMEOUT
#   UNKNOWN
string source

string modality

# Exact DialogueRequest.request_id associated with this result.
# HUMAN_RESPONSE_TIMEOUT
# ROBOT_SPEECH_TIMEOUT
# ROBOT_SPEECH_FAILED
# AUDIO_TERMINATED
# AUDIO_TERMINATION_UNCONFIRMED
# DIALOGUE_REQUEST_REJECTED

string request_id