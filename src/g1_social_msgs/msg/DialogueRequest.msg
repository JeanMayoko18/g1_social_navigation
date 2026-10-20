std_msgs/Header header

# Unique identifier for the complete dialogue interaction.
#
# The same interaction_id is preserved across clarification,
# confirmation and alternative-proposal turns belonging to the
# same collaboration episode.
string interaction_id

# Type of dialogue turn.
#
# Expected values:
#   PASSAGE_CLARIFICATION
#   NAVIGATION_CLARIFICATION
#   HUMAN_ASSISTANCE
#   INTENT_CONFIRMATION
#   ALTERNATIVE_PROPOSAL
string request_type

# Situation that triggered this dialogue turn.
#
# Expected values may include:
#   PATH_BLOCKED
#   SOCIAL_CONFLICT
#   AMBIGUOUS_INTENT
#   NO_HUMAN_RESPONSE
#   INTENT_REQUIRES_CONFIRMATION
#   REQUESTED_ROUTE_UNAVAILABLE
#
# Structured semantic information must not be encoded here.
string context
string language

# Unique identifier of this dialogue turn, including retries.
string request_id

# Speech-ready question/request.
#
# This field is intended for human communication only.
# It must never be parsed downstream to recover semantic state.
string question

# Semantic intent being confirmed or proposed.
#
# Empty when not applicable.
#
# Examples:
#   ALLOW_PASSAGE
#   DENY_PASSAGE
#
# IMPORTANT:
# This is a pending semantic interpretation/proposal.
# It is not a confirmed human instruction.
string intent_to_confirm

# Direction associated with the intent being confirmed or proposed.
#
# Expected values:
#   LEFT
#   RIGHT
#   FRONT
#   BACK
#   NONE
#
# IMPORTANT:
# LEFT and RIGHT are always expressed in the PERSON'S reference frame,
# never in the robot's reference frame.
string direction_to_confirm

# Person associated with the dialogue interaction.
string person_id

# Maximum time expected for a human response, in seconds.
#
# Timeout never means confirmation, rejection, denial of passage
# or permission to move.
float32 response_timeout

# True when the robot initiated this dialogue turn.
bool robot_initiated