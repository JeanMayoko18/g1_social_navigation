from setuptools import find_packages, setup
import os

package_name = "humanoid_nav_bridge"


def collect_data_files(source_directory):
    """
    Recursively collect all files under source_directory and preserve
    the same directory structure inside share/<package_name>/.
    """
    collected = []

    for root, _, filenames in os.walk(source_directory):
        if not filenames:
            continue

        source_files = [
            os.path.join(root, filename)
            for filename in filenames
        ]

        install_directory = os.path.join(
            "share",
            package_name,
            root,
        )

        collected.append(
            (
                install_directory,
                source_files,
            )
        )

    return collected

setup(
    name=package_name,
    version="0.1.0",

    packages=find_packages(exclude=["test"]),

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
            "share/" + package_name + "/launch",
            [
                "launch/humanoid_nav_bridge.launch.py",
                "launch/mujoco_to_gazebo_bridge.launch.py",
                "launch/real_robot_bridge.launch.py",
            ],
        ),
    ]
    + collect_data_files("params"),

    install_requires=["setuptools"],
    zip_safe=True,

    maintainer="Jean Chrysostome Mayoko Biong",
    maintainer_email="icmayoko18@gmail.com",

    description=(
        "ROS 2 bridge for humanoid navigation: "
        "real robot odom -> TF + /odom, "
        "LiDAR cloud -> /scan, Nav2 compatible"
    ),

    license="Apache-2.0",

    entry_points={
        "console_scripts": [
            # Simulation / MuJoCo
            "bridge_node = humanoid_nav_bridge.bridge_node:main",
            "mtog_bridge = humanoid_nav_bridge.mujoco_to_gazebo_node:main",
            "mujoco_to_gazebo_model = "
            "humanoid_nav_bridge.mujoco_to_gazebo_model:main",

            "cloud_y_flip = humanoid_nav_bridge.cloud_y_flip:main",

            # Real robot
            "odom_tf_bridge = "
            "humanoid_nav_bridge.real_odom_tf_bridge:main",

            "g1_odom_publisher = "
            "humanoid_nav_bridge.g1_odom_publisher:main",

            "real_state_to_odom_tf = "
            "humanoid_nav_bridge.real_state_to_odom_tf_node:main",

            "odommodestate_to_odom = "
            "humanoid_nav_bridge.real_odommodestate_to_odom:main",

            "dog_odom_nav_filter = "
            "humanoid_nav_bridge.dog_odom_nav_filter:main",

            "smart_odom_bridge_cmd_gated = "
            "humanoid_nav_bridge.real_smart_odom_bridge_cmd_gated:main",

            "cmd_vel_guarded_artificial_odom = "
            "humanoid_nav_bridge.cmd_vel_guarded_artificial_odom:main",

            "odom_normalizer = "
            "humanoid_nav_bridge.odom_normalizer:main",

            # Scan timestamp fixer
            "scan_restamper = "
            "humanoid_nav_bridge.scan_restamper:main",

            "scan_amcl_gate = "
            "humanoid_nav_bridge.scan_amcl_gate:main",

            "scan_qos_relay = "
            "humanoid_nav_bridge.scan_qos_relay:main",

            # Debug tools
            "cloud_frameid_probe = "
            "humanoid_nav_bridge.cloud_frameid_probe:main",

            "system_logger_node = "
            "humanoid_nav_bridge.system_logger_node:main",

            "fastlio_odom_3d_to_2d = "
            "humanoid_nav_bridge.fastlio_odom_3d_to_2d:main",

            "pc2_restamper = "
            "humanoid_nav_bridge.pc2_restamper:main",

            "scan_timing_monitor = "
            "humanoid_nav_bridge.scan_timing_monitor:main",

            "smart_odom_selector = "
            "humanoid_nav_bridge.smart_odom_selector:main",

            "smart_odom_fusion = "
            "humanoid_nav_bridge.smart_odom_fusion:main",

            "frontvideo_to_image = "
            "humanoid_nav_bridge.frontvideo_to_image:main",

            "camera_info_publisher = "
            "humanoid_nav_bridge.camera_info_publisher:main",

            "livox_to_mola_cloud = "
            "humanoid_nav_bridge.livox_to_mola_cloud:main",

            "mola_odom_2d_bridge = "
            "humanoid_nav_bridge.mola_odom_2d_bridge:main",

            "mola_odom_nav2_bridge = humanoid_nav_bridge.mola_odom_nav2_bridge:main",
        ],
    },
)