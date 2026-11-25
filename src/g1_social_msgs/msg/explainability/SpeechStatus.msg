# Status of one text-to-speech request.

std_msgs/Header header

# Identifier copied from ExplanationText.decision_id.
string decision_id

# Expected values:
#   STARTED
#   COMPLETED
#   FAILED
# CANCELED: audio cancellation confirmed.
# FAILED detail classification:
# AUDIO_RETRYABLE|... : audio state confirmed safe.
# AUDIO_UNCERTAIN|... : audio termination not confirmed.

string state

# Optional diagnostic information.
string detail