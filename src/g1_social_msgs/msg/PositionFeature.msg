# Visible detection dimensions, not guaranteed anatomical dimensions.
FeatureMetadata metadata

uint8 PIXELS=0
uint8 METRES=1

# Set explicitly by the producer, including before feature computation.
uint8 unit

# Strictly positive when metadata.status == AVAILABLE.
# Ignore numeric values otherwise.
float64 width
float64 height