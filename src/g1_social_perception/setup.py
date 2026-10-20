import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'g1_social_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join(
                "share",
                package_name,
                "config",
            ),
            glob("config/*.yaml"),
        ),
        (
            os.path.join(
                "share",
                package_name,
                "rviz",
            ),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer="jean",
    maintainer_email="icmayoko18@gmail.com",
    description=(
        "Social perception nodes for YOLO and RealSense depth fusion on G1."
    ),
    license="Proprietary",

    entry_points={
        "console_scripts": [
            "person_tracking = "
            "g1_social_perception.person_tracking_node:main",

            "person_depth_fusion = "
            "g1_social_perception.person_depth_fusion_node:main",

            "person_frame_transform = "
            "g1_social_perception.person_frame_transform_node:main",

            "person_spatial_tracking = "
            "g1_social_perception.person_spatial_tracking_node:main",

            "social_zone_generator = "
            "g1_social_perception.social_zone_generator_node:main",

            "social_navigation_decision = "
            "g1_social_perception.social_navigation_decision_node:main",

            "explanation_trigger = "
            "g1_social_perception.explanation_trigger_node:main",

            "explanation_context = "
            "g1_social_perception.explanation_context_node:main",

            "person_3d_marker = "
            "g1_social_perception.person_3d_marker_node:main",

            "image_ingress = "
            "g1_social_perception.image_ingress_node:main",

            "ros4hri_visualization_adapter = "
            "g1_social_perception.ros4hri_visualization_adapter_node:main",
        ],
    },
    

)
