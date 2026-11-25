# Estimated 3D position of a detected person.
# Coordinates are expressed in metadata.header.frame_id.

FeatureMetadata metadata

# Position in metres.
# Ignore numeric values unless metadata.status == AVAILABLE.
geometry_msgs/Vector3 value
