# Current spatial processing frame and original observation timestamp.
# At SOURCE stage, frame_id is the source image frame.
# After depth processing, frame_id is the camera optical frame.
# After spatial transformation, frame_id is the target spatial frame.
#
# Every AVAILABLE candidate position must use this header's stamp and frame.
# The header alone does not imply that any position is available.
std_msgs/Header header

# Original RGB observation header.
# Preserved unchanged across all processing stages.
std_msgs/Header image_header

# Session allocated by the source-candidate adapter.
# Changes when that producer restarts.
string source_session_id

# Strictly increasing number allocated by the source-candidate adapter.
# Preserved unchanged downstream.
# Together with source_session_id, identifies one source observation.
uint64 source_sequence

uint8 SOURCE=0
uint8 DEPTH=1
uint8 SPATIAL=2
uint8 APPEARANCE=3
uint8 IDENTITY=4

# Last processing stage completed or attempted for this observation.
# Stage advancement does not imply every feature became available.
uint8 processing_stage

# Session of the identity authority.
# Empty before identity evaluation.
string tracking_session_id

# False before the identity stage; true after identity evaluation.
bool identity_evaluated

# Number of person candidates selected from the original YOLO array.
# All selected candidates remain represented in candidates, including
# candidates without depth, appearance or a resolved identity.
uint32 source_detection_count

# Complete evidence under the requirements of the current stage.
# A downstream stage may clear this flag, never restore an upstream failure.
#
# SOURCE:
#   all selected person candidates are represented and structurally valid.
# DEPTH / SPATIAL:
#   additionally require usable position evidence for every candidate.
# IDENTITY:
#   additionally require confirmed identities for every candidate.
#
# Optional cue unavailability does not by itself make the array incomplete.
# Missing appearance can coexist with complete evidence when the configured
# identity policy confirms all candidates using other admissible cues.
#
# A valid empty source observation may be complete.
# A failed or missing observation must never be converted into an empty one.
bool source_observation_complete

# Actual current candidates, never retained historical tracks or predictions.
# Length must equal source_detection_count.
PersonDetection[] candidates