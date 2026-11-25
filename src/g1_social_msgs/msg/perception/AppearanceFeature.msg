# Appearance descriptor extracted from one actual source image region.
FeatureMetadata metadata

# Identifies extraction version, region definition, colour representation
# and histogram binning. Only compatible schemas may be compared.
string schema_id

# Normalized histogram when AVAILABLE:
#   bins are finite and non-negative;
#   sum(bins) equals one within the configured numerical tolerance.
# Empty when NOT_PROCESSED or UNAVAILABLE.
float64[] bins

# Intended region relative to the original YOLO bounding box.
# Order: left, top, right, bottom; values in [0, 1].
# Region definition precedes clipping to image boundaries.
float64[4] relative_region

# Pixel support used by the extraction quality policy.
# Counts refer to the full region before any optional subsampling.
uint64 intended_pixel_count
uint64 selected_pixel_count

# selected_pixel_count / intended_pixel_count when the denominator is positive.
# Zero otherwise.
float64 support_fraction

# True only when a person-specific foreground mask was applied.
bool foreground_mask_used