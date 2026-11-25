"""Two-region descriptors and bounded, session-local return recognition."""
from .appearance import AppearanceConfig, Signature, extract_signature, compare
from .gallery import Candidate, Decision, GalleryConfig, IdentityGallery

__all__ = [
    "AppearanceConfig", "Signature", "extract_signature", "compare",
    "Candidate", "Decision", "GalleryConfig", "IdentityGallery",
]
