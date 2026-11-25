"""Install perception nodes, shared Python modules and runtime resources."""

import os
from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_NAME = "g1_social_perception"

PACKAGE_DESCRIPTION = (
    "Person observation enrichment, temporal tracking, spatial perception "
    "and social-context interpretation for the Unitree G1. "
    "Includes shared mathematical models and processing-time diagnostics."
)


# =============================================================================
# DATA / RUNTIME RESOURCES
# =============================================================================


def resource_files(directory, patterns):
    """
    Collect runtime resources while preserving their subdirectory layout.

    Receives:
        A package-relative directory and accepted filename patterns.

    Returns:
        setuptools data_files entries under share/<package>/<directory>.

    Python modules are installed separately through find_packages.
    """
    root = Path(directory)

    if not root.is_dir():
        return []

    selected = set()

    for pattern in patterns:
        selected.update(
            path
            for path in root.rglob(pattern)
            if path.is_file()
            and "__pycache__" not in path.parts
        )

    grouped = {}

    for path in sorted(selected):
        destination = os.path.join(
            "share",
            PACKAGE_NAME,
            str(path.parent),
        )

        grouped.setdefault(destination, []).append(str(path))

    return sorted(grouped.items())


DATA_FILES = [
    (
        "share/ament_index/resource_index/packages",
        ["resource/" + PACKAGE_NAME],
    ),
    (
        "share/" + PACKAGE_NAME,
        ["package.xml"],
    ),
]

DATA_FILES += resource_files(
    "launch",
    ("*.py", "*.xml", "*.yaml", "*.yml"),
)

DATA_FILES += resource_files(
    "config",
    ("*.yaml", "*.yml", "*.json"),
)

DATA_FILES += resource_files(
    "rviz",
    ("*.rviz",),
)


# =============================================================================
# OUTPUT / EXECUTABLE ENTRY POINTS
# =============================================================================

# Grouping documents responsibilities; it does not define launch order.

CONSOLE_SCRIPTS = [
    # Sensor and source-message ingress.
    "image_ingress = "
    "g1_social_perception.image_ingress_node:main",

    "person_detection_source = "
    "g1_social_perception.person_detection_source_node:main",

    "person_yolo_identity = "
    "g1_social_perception.person_yolo_identity_node:main",

    "person_yolo_track_source = "
    "g1_social_perception.person_yolo_track_source_node:main",
    
    # Observation geometry and identity processing.
    "person_depth_fusion = "
    "g1_social_perception.person_depth_fusion_node:main",

    "person_frame_transform = "
    "g1_social_perception.person_frame_transform_node:main",

    "person_appearance_extraction = "
    "g1_social_perception.person_appearance_extraction_node:main",

    "person_tracking = "
    "g1_social_perception.person_tracking_node:main",


    "known_person_registry = "
    "g1_social_perception.known_person_registry_node:main",

    "person_botsort_baseline = "
    "g1_social_perception.person_botsort_baseline_node:main",

    "person_botsort_bridge = "
    "g1_social_perception.person_botsort_bridge_node:main",

    "identity_benchmark_recorder = "
    "g1_social_perception.benchmark.identity_benchmark_recorder:main",

    "person_ablation = "
    "g1_social_perception.person_ablation_node:main",


    "person_spatial_tracking = "
    "g1_social_perception.person_spatial_tracking_node:main",


    # Social interpretation.
    "social_zone_generator = "
    "g1_social_perception.social_zone_generator_node:main",

    "social_navigation_decision = "
    "g1_social_perception.social_navigation_decision_node:main",

    # Explanation context.
    "explanation_trigger = "
    "g1_social_perception.explanation_trigger_node:main",

    "explanation_context = "
    "g1_social_perception.explanation_context_node:main",

    # Visualization adapters.
    "person_3d_marker = "
    "g1_social_perception.person_3d_marker_node:main",

    "ros4hri_visualization_adapter = "
    "g1_social_perception.ros4hri_visualization_adapter_node:main",
]


# =============================================================================
# PACKAGE INSTALLATION
# =============================================================================


setup(
    name=PACKAGE_NAME,
    version="0.0.0",

    # Include internal subpackages without installing unrelated directories.
    packages=find_packages(
        include=[
            PACKAGE_NAME,
            PACKAGE_NAME + ".*",
        ],
        exclude=[
            PACKAGE_NAME + ".test",
            PACKAGE_NAME + ".test.*",
            PACKAGE_NAME + ".tests",
            PACKAGE_NAME + ".tests.*",
        ],
    ),

    data_files=DATA_FILES,

    # ROS and system runtime dependencies are declared in package.xml.
    install_requires=["setuptools"],

    zip_safe=False,

    maintainer="jean",
    maintainer_email="icmayoko18@gmail.com",
    description=PACKAGE_DESCRIPTION,
    license="Proprietary",

    entry_points={
        "console_scripts": CONSOLE_SCRIPTS,
    },
)