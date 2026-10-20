# Final natural-language robot explanation.

std_msgs/Header header

# Original deterministic decision identifier.
string decision_id

# Output language.
# Expected:
#   es
#   en
string language
string fallback_text
string fallback_language

# Final speech-ready explanation.
string text

# Source of the explanation.
# Examples:
#   local
#   openai
#   gemini
#   deterministic
#   cache
string provider

# Model identifier when applicable.
string model

# True when the deterministic fallback was used.
bool used_fallback

# True when the explanation came from the semantic cache.
bool used_cache

# Speech category.
# Expected values:
#   SOCIAL_EXPLANATION
#   NAVIGATION_GOAL
#   NAVIGATION_RESULT
#   INITIAL_POSE
#
# Empty means an unclassified legacy request.
string speech_kind

# Source context timestamp, preserved across text generation.
# This is distinct from header.stamp, which records text publication.
builtin_interfaces/Time context_stamp

# Navigation context associated with this explanation.
string navigation_state
string navigation_goal_id
bool navigation_active

# Whether the source context contains a confirmed social intrusion.
bool social_conflict_active
