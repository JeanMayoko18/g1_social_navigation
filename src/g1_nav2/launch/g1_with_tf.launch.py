#!/usr/bin/env python3
"""
g1_with_tf.launch.py
====================

PURPOSE
-------
This ROS 2 launch file provides a **single entry point** to start the G1 system
together with the full TF tree publication.

It is designed to support both:

1. **Real robot execution**
2. **MuJoCo simulation**

The launch file starts:

- the main RL control executable
  - `rl_real_g1_edu23` in real mode
  - `rl_mujoco` in simulation mode
- `robot_state_publisher`

The main objective is to ensure that the complete robot TF tree
(`pelvis`, legs, arms, torso, etc.) is available in ROS 2.

Without this unified launch file, the user would need to manually start
multiple processes and manually keep their parameters consistent
(network interface, robot type, scene, cmd_vel topic, etc.).


SYSTEM ARCHITECTURE
-------------------

                +-----------------------------+
                |   rl_real_g1_edu23 (real)   |
                |           or                |
                |     rl_mujoco (sim)         |
                +--------------+--------------+
                               |
                               |  publishes /joint_states
                               v
                +-----------------------------+
                |    robot_state_publisher    |
                +--------------+--------------+
                               |
                               v
                           /tf  /tf_static

`robot_state_publisher` uses:
- the robot URDF model
- the `/joint_states` topic

to publish the full kinematic TF tree of the robot.


ROBOT MODEL SOURCE
------------------

The robot model is loaded from the RL zoo directory through the launch
argument:

    zoo_dir

Default example:

    $HOME/g1_ws/src/rl_sar/src/rl_sar_zoo

This launch resolves the URDF path as:

    <zoo_dir>/g1_description/urdf/g1_29dof_rev_1_0_modify_inertia.urdf

This avoids using hard-coded absolute paths and makes the launch easier
to adapt on another PC or workspace.


SUPPORTED MODES
---------------

mode=real
    Starts the real robot executable:

        rl_real_g1_edu23

mode=mujoco
    Starts the MuJoCo simulation executable:

        rl_mujoco

In both cases, `robot_state_publisher` is also launched so that the full TF
tree is available.


MAIN GOAL OF THIS LAUNCH
------------------------

The main goal of this file is to centralize and synchronize:

- the robot description
- the TF publication
- the control executable
- the mode selection (real vs simulation)
- the command velocity topic
- the network interface used by the robot

This makes the setup more robust, more reproducible, and easier to maintain.


IMPORTANT NOTE
--------------

This launch assumes that the RL executables publish:

    /joint_states

That topic is required by `robot_state_publisher` to compute and publish
the articulated TF frames of the robot.

So this launch file alone is not enough:
the executables `rl_real_g1_edu23` and `rl_mujoco` must also publish
joint states correctly.


USAGE EXAMPLES
--------------

Real robot:

    ros2 launch g1_nav2 g1_with_tf.launch.py \
        mode:=real \
        network_interface:=eno1 \
        cmd_vel_topic:=/cmd_vel_rlsar \
        cmd_vel_timeout_sec:=2.0 \
        hw_mode:=0

MuJoCo simulation with 23 DoF:

    ros2 launch g1_nav2 g1_with_tf.launch.py \
        mode:=mujoco \
        scene_name:=scene_23dof \
        cmd_vel_topic:=/cmd_vel \
        cmd_vel_timeout_sec:=0.6

MuJoCo simulation with 29 DoF:

    ros2 launch g1_nav2 g1_with_tf.launch.py \
        mode:=mujoco \
        scene_name:=scene_29dof \
        cmd_vel_topic:=/cmd_vel \
        cmd_vel_timeout_sec:=0.6


MAIN LAUNCH ARGUMENTS
---------------------

mode
    Execution mode: 'real' or 'mujoco'

network_interface
    Network interface used by the real robot (for example eno1)

robot_name
    Robot type/name, usually 'g1'

scene_name
    MuJoCo scene name, for example 'scene_23dof' or 'scene_29dof'

cmd_vel_topic
    Velocity command topic used by the RL controller

cmd_vel_timeout_sec
    Timeout used to invalidate stale cmd_vel commands

navigation_mode
    Enables RL navigation mode

hw_mode
    Hardware mode for the real robot controller

publish_frequency
    TF publication frequency used by robot_state_publisher

zoo_dir
    Root directory of rl_sar_zoo
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node

ETH0_ENV = {
    "ROS_DOMAIN_ID": "0",
    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    "ROS_LOCALHOST_ONLY": "0",
    #"CYCLONEDDS_URI": "/home/unitree/cyclonedds_ws/cyclonedds.xml",
    "CYCLONEDDS_URI": "/home/unitree/cyclonedds_ws/cyclonedds_g1_optimized.xml",
}

WIFI_ENV = {
    "ROS_DOMAIN_ID": "0",
    "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    "ROS_LOCALHOST_ONLY": "0",
    "CYCLONEDDS_URI": "file:///home/unitree/cyclonedds_ws/cyclonedds_wifi.xml",
}

def launch_setup(context, *args, **kwargs):
    # ------------------------------------------------------------------
    # Launch arguments
    #
    # These arguments allow the same launch file to be reused for:
    #   - real robot execution
    #   - MuJoCo simulation
    # while keeping a single centralized configuration.
    # ------------------------------------------------------------------
    mode = LaunchConfiguration("mode")
    network_interface = LaunchConfiguration("network_interface")
    robot_name = LaunchConfiguration("robot_name")
    scene_name = LaunchConfiguration("scene_name")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    cmd_vel_timeout_sec = LaunchConfiguration("cmd_vel_timeout_sec")
    navigation_mode = LaunchConfiguration("navigation_mode")
    hw_mode = LaunchConfiguration("hw_mode")
    publish_frequency = LaunchConfiguration("publish_frequency")
    zoo_dir = LaunchConfiguration("zoo_dir")
    body_tf_frequency = LaunchConfiguration("body_tf_frequency")
    static_tf_frequency = LaunchConfiguration("static_tf_frequency")
    publish_full_body_tf = LaunchConfiguration("publish_full_body_tf")
    nav_tf_frequency = LaunchConfiguration("nav_tf_frequency")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    

    # ------------------------------------------------------------------
    # Robot URDF path
    #
    # This is the robot description used by robot_state_publisher.
    # We directly load the URDF from the source tree instead of using
    # xacro, because for TF publication we only need the kinematic model.
    # ------------------------------------------------------------------
    urdf_path = PathJoinSubstitution([
        zoo_dir,
        "g1_description",
        "urdf",
        "g1_edu23_real_tf.urdf"
    ]).perform(context)

    static_nav_urdf_path = PathJoinSubstitution([
        zoo_dir,
        "g1_description",
        "urdf",
        "g1_nav_static_frames.urdf"
    ]).perform(context)

    static_nav_urdf_path_old = PathJoinSubstitution([
        zoo_dir,
        "g1_description",
        "urdf",
        "g1_nav_minimal_tf.urdf",
    ]).perform(context)

    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description_content = f.read()

    with open(static_nav_urdf_path, "r", encoding="utf-8") as f:
        static_nav_description = f.read()
    # ------------------------------------------------------------------
    # robot_state_publisher
    #
    # Role:
    #   - reads robot_description
    #   - listens to /joint_states
    #   - publishes the full TF tree to /tf and /tf_static
    #
    # This is the key node that makes frames such as pelvis available.
    # ------------------------------------------------------------------
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        additional_env=ETH0_ENV,
        remappings=[
            ("/joint_states", joint_states_topic),
        ],
        arguments=["--ros-args", "--log-level", "warn"],
        parameters=[{
            "robot_description": robot_description_content,
            "publish_frequency": float(
                body_tf_frequency.perform(context)
            ),
            "ignore_timestamp": True,
        }],
        condition=IfCondition(
            publish_full_body_tf
        ),
    )

    rsp_nav_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="g1_nav_tf_publisher",
        output="screen",
        additional_env=ETH0_ENV,
        parameters=[{
            "robot_description": static_nav_description,
            "publish_frequency": float(nav_tf_frequency.perform(context)),
            "ignore_timestamp": True,
        }],
    )

    # ------------------------------------------------------------------
    # Real robot node
    #
    # Equivalent CLI:
    #
    #   ros2 run rl_sar rl_real_g1_edu23 eno1 --ros-args ...
    #
    # The network interface is passed as a positional argument because the
    # executable expects it that way.
    # ------------------------------------------------------------------
    real_node = Node(
        package="rl_sar",
        executable="rl_real_g1_edu23",
        name="rl_real_g1_edu23",
        output="screen",
        arguments=[network_interface.perform(context)],
        additional_env=ETH0_ENV,
        parameters=[{
            "hw_mode": int(hw_mode.perform(context)),
            "navigation_mode": navigation_mode.perform(context).lower() == "true",
            "cmd_vel_timeout_sec": float(cmd_vel_timeout_sec.perform(context)),
            "cmd_vel_topic": cmd_vel_topic.perform(context),
        }],
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'real'"])
        ),
    )

    # ------------------------------------------------------------------
    # MuJoCo simulation node
    #
    # Equivalent CLI:
    #
    #   ros2 run rl_sar rl_mujoco g1 scene_23dof --ros-args ...
    #
    # The executable receives:
    #   - robot_name
    #   - scene_name
    # as positional arguments.
    # ------------------------------------------------------------------
    mujoco_node = Node(
        package="rl_sar",
        executable="rl_mujoco",
        name="rl_mujoco",
        output="screen",
        arguments=[
            robot_name.perform(context),
            scene_name.perform(context),
        ],
        parameters=[{
            "navigation_mode": navigation_mode.perform(context).lower() == "true",
            "cmd_vel_timeout_sec": float(cmd_vel_timeout_sec.perform(context)),
            "cmd_vel_topic": cmd_vel_topic.perform(context),
        }],
        condition=IfCondition(
            PythonExpression(["'", mode, "' == 'mujoco'"])
        ),
    )

    # ------------------------------------------------------------------
    # Final launch nodes returned by OpaqueFunction
    # ------------------------------------------------------------------
    return [
        rsp_node,
        #rsp_nav_node,
        #real_node,
        #mujoco_node,
    ]


def generate_launch_description():
    # ------------------------------------------------------------------
    # Final launch description
    #
    # Only one control node is launched depending on the selected mode:
    #   - real_node   if mode == real
    #   - mujoco_node if mode == mujoco
    #
    # robot_state_publisher is always launched.
    # ------------------------------------------------------------------
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="real"),
        DeclareLaunchArgument("network_interface", default_value="eno1"),
        DeclareLaunchArgument("robot_name", default_value="g1"),
        DeclareLaunchArgument("scene_name", default_value="scene_23dof"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel_rlsar"),
        DeclareLaunchArgument("cmd_vel_timeout_sec", default_value="2.0"),
        DeclareLaunchArgument("navigation_mode", default_value="true"),
        DeclareLaunchArgument("hw_mode", default_value="0"),
        DeclareLaunchArgument("publish_frequency", default_value="100.0"),
        DeclareLaunchArgument("body_tf_frequency", default_value="15.0",),
        DeclareLaunchArgument("static_tf_frequency", default_value="1.0",),
        DeclareLaunchArgument("nav_tf_frequency", default_value="15.0",),
        DeclareLaunchArgument(
            "publish_full_body_tf",
            default_value="true",
            description=(
                "Publish the complete articulated G1 TF tree. "
                "Keep disabled for the lightweight navigation profile."
            ),
        ),
        DeclareLaunchArgument(
            "joint_states_topic",
            default_value="/joint_states_source",
            description=(
                "JointState topic used to update the minimal "
                "navigation TF chain."
            ),
        ),

        DeclareLaunchArgument(
            "zoo_dir",
            default_value=[
                EnvironmentVariable("HOME"),
                "/g1_ws/src/rl_sar/src/rl_sar_zoo"
                #"/g1_ws/src/g1_description"
            ],
            description="Root directory of rl_sar_zoo"
        ),
        OpaqueFunction(function=launch_setup),
    ])