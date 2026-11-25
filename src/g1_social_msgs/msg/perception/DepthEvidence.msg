uint8 NOT_PROCESSED=0
uint8 UNAVAILABLE=1
uint8 AVAILABLE=2

uint8 status

# True when a depth image was selected, even if its ROI yielded no valid depth.
bool image_selected

# Actual selected depth-image header.
# Zero stamp and empty frame when image_selected is false.
std_msgs/Header image_header

# Signed synchronization difference:
#   selected_depth_stamp - source_rgb_stamp
# Meaningful only when image_selected is true.
int64 image_offset_ns

# Camera model used for deprojection.
# CameraInfo may describe static calibration and need not share the RGB stamp.
bool calibration_available
std_msgs/Header camera_info_header

# Pinhole intrinsics for the aligned image coordinates, in pixels.
# Meaningful only when calibration_available is true.
float64 fx
float64 fy
float64 cx
float64 cy

# Sample support in the selected depth ROI.
uint64 roi_pixel_count
uint64 valid_pixel_count

# Optical-axis depth Z in metres.
# Positive and finite only when status == AVAILABLE.
float64 median_depth_m

# Median absolute deviation of valid depths, in metres.
# Zero may represent a constant valid-depth region.
# Meaningful only when status == AVAILABLE.
float64 depth_mad_m

# Machine-readable processing result.
string reason