# Index in the original YOLO detection array.
# It need not be contiguous after filtering non-person classes.
# It remains unchanged throughout the processing chain.
uint32 source_index

# Original YOLO detection, preserved unchanged.
# Its id may belong to an upstream detector/tracker.
# It is NOT the identity authority of this pipeline.
# Downstream nodes must use the explicit features below rather than
# interpreting source_detection.bbox3d as the current spatial estimate.
yolo_msgs/Detection source_detection

# Explicit features of the current actual observation.
PositionFeature position
AppearanceFeature appearance_upper
AppearanceFeature appearance_lower
ExtentFeature apparent_extent
ExtentFeature metric_extent

# Actual depth provenance and extraction support.
DepthEvidence depth_evidence

# Identity assigned by this pipeline.
# Empty before identity evaluation or when association remains unresolved.
string person_id

string NEW=NEW
string MATCHED=MATCHED
string RETURNING=RETURNING
string AMBIGUOUS=AMBIGUOUS
string INSUFFICIENT=INSUFFICIENT

# Before identity evaluation: INSUFFICIENT.
string identity_state

# True only when the identity confirmation policy is satisfied.
bool identity_confirmed

# Unique live-segment token within the tracking session.
# Zero when no identity is assigned.
# Downstream motion processing must reset when this token changes.
uint64 identity_generation