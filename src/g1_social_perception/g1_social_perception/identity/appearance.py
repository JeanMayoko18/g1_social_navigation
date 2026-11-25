"""RGB/BGR uint8 image -> proportional upper/lower clothing histograms.

No ROS, image prediction, biometric identification, or implicit colour ordering.
Quality measures image support and caller-supplied reliability, not identity.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class AppearanceConfig:
    # Fractions of the ORIGINAL detector box, not its clipped visible portion.
    upper: Tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.48)
    lower: Tuple[float, float, float, float] = (0.20, 0.52, 0.80, 0.90)
    hue_bins: int = 18
    saturation_bins: int = 4
    value_bins: int = 4
    achromatic_bins: int = 8
    saturation_floor: float = 0.15
    dark_floor: float = 0.12
    min_pixels: int = 64
    min_support: float = 0.70
    max_sample_pixels: int = 4096
    weights: Tuple[float, float] = (0.5, 0.5)

    def __post_init__(self):
        for roi in (self.upper, self.lower):
            if len(roi) != 4 or not all(math.isfinite(x) for x in roi):
                raise ValueError("Invalid region fractions")
            x0, y0, x1, y1 = roi
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("Regions must lie inside [0,1]^2")
        for n in (self.hue_bins, self.saturation_bins, self.value_bins,
                  self.achromatic_bins, self.min_pixels, self.max_sample_pixels):
            if not isinstance(n, int) or n <= 0:
                raise ValueError("Bin and sample counts must be positive integers")
        for q in (self.saturation_floor, self.dark_floor, self.min_support):
            if not math.isfinite(q) or not 0 <= q <= 1:
                raise ValueError("Quality thresholds must be in [0,1]")
        if len(self.weights) != 2 or any(
                not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Two positive region weights required")
        if self.max_sample_pixels < self.min_pixels:
            raise ValueError("Sampling cap must cover min_pixels")

    @property
    def feature_schema(self):
        # Compare only descriptors with exactly the same extraction convention.
        return (
            self.upper, self.lower, self.hue_bins, self.saturation_bins,
            self.value_bins, self.achromatic_bins,
            self.saturation_floor, self.dark_floor,
        )


@dataclass(frozen=True)
class Region:
    histogram: Optional[np.ndarray] = field(repr=False, compare=False)
    quality: float = 0.0
    support: float = 0.0
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    reason: str = "unavailable"

    def __post_init__(self):
        for value in (self.quality, self.support):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Region quality/support must be in [0,1]")
        if self.histogram is None:
            if self.quality != 0:
                raise ValueError("Unavailable histogram must have zero quality")
            return
        hist = np.asarray(self.histogram, dtype=np.float64)
        if (hist.ndim != 1 or hist.size == 0 or not np.isfinite(hist).all() or
                np.any(hist < 0) or not np.isclose(hist.sum(), 1.0)):
            raise ValueError("Histogram must be a normalized nonnegative vector")
        # Immutable backing bytes prevent accidental template contamination.
        object.__setattr__(self, "histogram", np.frombuffer(hist.tobytes(),
                                                         dtype=np.float64))


@dataclass(frozen=True)
class Signature:
    upper: Region
    lower: Region
    schema: tuple


@dataclass(frozen=True)
class Similarity:
    score: Optional[float]
    evidence_quality: float
    common_regions: int
    upper_score: Optional[float]
    lower_score: Optional[float]


def _histogram(rgb, config):
    rgb = rgb.astype(np.float64) / 255.0
    maximum = rgb.max(axis=1)
    minimum = rgb.min(axis=1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta),
                           where=maximum > 0)
    hue = np.zeros_like(delta)
    coloured = delta > 0
    r, g, b = rgb.T
    red = coloured & (maximum == r)
    green = coloured & ~red & (maximum == g)
    blue = coloured & ~red & ~green
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6
    hue[green] = (b[green] - r[green]) / delta[green] + 2
    hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4
    hue /= 6
    chromatic = ((saturation >= config.saturation_floor) &
                 (maximum >= config.dark_floor) & coloured)
    chromatic_size = config.hue_bins * config.saturation_bins * config.value_bins
    indices = chromatic_size + np.minimum(
        (maximum * config.achromatic_bins).astype(int),
        config.achromatic_bins - 1)
    hi = np.minimum((hue * config.hue_bins).astype(int), config.hue_bins - 1)
    si = np.minimum((saturation * config.saturation_bins).astype(int),
                    config.saturation_bins - 1)
    vi = np.minimum((maximum * config.value_bins).astype(int),
                    config.value_bins - 1)
    indices[chromatic] = ((hi * config.saturation_bins + si) *
                         config.value_bins + vi)[chromatic]
    histogram = np.bincount(
        indices, minlength=chromatic_size + config.achromatic_bins
    ).astype(np.float64)
    histogram /= histogram.sum()
    histogram.setflags(write=False)
    return histogram


def extract_signature(image, bbox, *, color_order, config=None,
                      foreground_mask=None, region_reliability=None):
    """Extract from (center_u, center_v, width, height).

    image must correspond to the detector source frame. The caller checks stamps.
    foreground_mask, when provided, is a full-image boolean person mask.
    region_reliability can disable an occluded region: {"lower": 0.0}.
    Without a mask/occlusion signal, foreign pixels inside a box are NOT detected.
    """
    config = config or AppearanceConfig()
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected uint8 image")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Expected non-empty HxWx3 image")
    if color_order not in ("RGB", "BGR"):
        raise ValueError("Explicit RGB or BGR ordering required")
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
        raise ValueError("Invalid bounding box")
    u, v, width, height = map(float, bbox)
    if width <= 0 or height <= 0:
        raise ValueError("Non-positive bounding box dimensions")
    if foreground_mask is not None:
        if (not isinstance(foreground_mask, np.ndarray) or
                foreground_mask.shape != image.shape[:2] or
                foreground_mask.dtype != np.bool_):
            raise ValueError("Mask must be boolean HxW")
    reliability = region_reliability or {}
    if set(reliability) - {"upper", "lower"}:
        raise ValueError("Unknown region")
    for q in reliability.values():
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("Region reliability must be in [0,1]")
    image_height, image_width = image.shape[:2]
    origin_x, origin_y = u - width / 2, v - height / 2
    regions = []
    for name, fractions in (("upper", config.upper), ("lower", config.lower)):
        a, c, b, d = fractions
        fx0, fy0 = origin_x + a * width, origin_y + c * height
        fx1, fy1 = origin_x + b * width, origin_y + d * height
        if not all(math.isfinite(x) for x in (fx0, fy0, fx1, fy1)):
            raise ValueError("Overflowing region geometry")
        rx0, ry0 = math.floor(fx0), math.floor(fy0)
        rx1, ry1 = math.ceil(fx1), math.ceil(fy1)
        x0, x1 = max(0, min(image_width, rx0)), max(0, min(image_width, rx1))
        y0, y1 = max(0, min(image_height, ry0)), max(0, min(image_height, ry1))
        bounds = (x0, y0, x1, y1)
        intended_area = (rx1 - rx0) * (ry1 - ry0)
        pixels = image[y0:y1, x0:x1].reshape(-1, 3)
        if foreground_mask is not None:
            pixels = pixels[foreground_mask[y0:y1, x0:x1].reshape(-1)]
        support = len(pixels) / intended_area if intended_area > 0 else 0.0
        quality = support * reliability.get(name, 1.0)
        if (len(pixels) < config.min_pixels or support < config.min_support or
                quality <= 0):
            regions.append(Region(None, 0.0, support, bounds, "insufficient_support"))
            continue
        if len(pixels) > config.max_sample_pixels:
            idx = np.linspace(0, len(pixels) - 1, config.max_sample_pixels, dtype=int)
            pixels = pixels[idx]
        if color_order == "BGR":
            pixels = pixels[:, ::-1]
        regions.append(Region(_histogram(pixels, config), quality,
                              support, bounds, "available"))
    return Signature(regions[0], regions[1], config.feature_schema)


def compare(first, second, *, weights=(0.5, 0.5), min_region_quality=0.0):
    """Quality-weighted Bhattacharyya similarity; no-common-data -> None."""
    if first.schema != second.schema:
        raise ValueError("Descriptor schemas differ")
    if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Two positive region weights required")
    if not math.isfinite(min_region_quality) or not 0 <= min_region_quality <= 1:
        raise ValueError("Invalid quality threshold")
    scores, pair_weights = [], []
    for a, b, weight in zip((first.upper, first.lower),
                            (second.upper, second.lower), weights):
        quality = min(a.quality, b.quality)
        if (a.histogram is None or b.histogram is None or
                quality <= 0 or quality < min_region_quality):
            scores.append(None)
            pair_weights.append(0.0)
            continue
        if (a.histogram.shape != b.histogram.shape or
                not np.isfinite(a.histogram).all() or
                not np.isfinite(b.histogram).all()):
            raise ValueError("Invalid histogram")
        score = float(np.sqrt(a.histogram * b.histogram).sum())
        scores.append(float(np.clip(score, 0, 1)))
        pair_weights.append(weight * quality)
    denominator = sum(pair_weights)
    score = (sum(w * s for w, s in zip(pair_weights, scores) if s is not None) /
             denominator) if denominator else None
    return Similarity(score, denominator / sum(weights),
                      sum(s is not None for s in scores), *scores)
