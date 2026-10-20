from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'g1_nav2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        # Package.xml
        (
            'share/' + package_name,
            ['package.xml']
        ),

        # Launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),

        # Params files
        (
            os.path.join(
                'share',
                package_name,
                'params'
            ),
            glob(
                'params/*.yaml'
            )
        ),

        # Behavior Trees
        (
            os.path.join(
                'share',
                package_name,
                'behavior_trees'
            ),
            glob(
                'behavior_trees/*.xml'
            )
        ),

        # Locomotion calibration base parameters
        (
            os.path.join(
                'share',
                package_name,
                'params',
                'locomotion_calibration'
            ),
            glob(
                'params/locomotion_calibration/*.yaml'
            )
        ),

        # Forward lateral compensation profiles
        (
            os.path.join(
                'share',
                package_name,
                'params',
                'locomotion_calibration',
                'profiles',
                'lateral'
            ),
            glob(
                'params/locomotion_calibration/'
                'profiles/lateral/*.yaml'
            )
        ),

        # Forward heading compensation profiles
        (
            os.path.join(
                'share',
                package_name,
                'params',
                'locomotion_calibration',
                'profiles',
                'heading'
            ),
            glob(
                'params/locomotion_calibration/'
                'profiles/heading/*.yaml'
            )
        ),
        
        # ROS distribution overlays: Foxy / Humble
        (
            os.path.join(
                'share',
                package_name,
                'params',
                'distro_overlays'
            ),
            glob(
                'params/distro_overlays/*.yaml'
            )
        ),

        # Perception profiles A / B / C / D
        (
            os.path.join(
                'share',
                package_name,
                'params',
                'perception_profiles'
            ),
            glob(
                'params/perception_profiles/*.yaml'
            )
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jean',
    maintainer_email='icmayoko18@gmail.com',
    description='G1 Nav2 bringup (fake base)',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "map_republisher = g1_nav2.map_republisher:main",
            "amcl_lifecycle_supervisor = g1_nav2.amcl_lifecycle_supervisor:main",
            (
                "navigation_interaction_speech = "
                "g1_nav2."
                "navigation_interaction_speech_node:main"
            ),
            (
                "navigation_goal_gate = "
                "g1_nav2."
                "navigation_goal_gate_node:main"
            ),
        ],
    },
)