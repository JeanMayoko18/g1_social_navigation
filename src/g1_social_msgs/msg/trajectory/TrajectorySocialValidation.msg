# TrajectorySocialValidation.msg

std_msgs/Header header

# Identifier of the exact trajectory/path being evaluated.
string path_id

# Social-validation result for this exact trajectory.
#
# Expected values:
#   VALID
#   CONFLICT
#   UNKNOWN
#
# VALID is positive authorization evidence only for path_id.
# It does not authorize any other trajectory.
string validation

# Human/person associated with the social validation when applicable.
string person_id

# Optional reason describing the result.
string reason