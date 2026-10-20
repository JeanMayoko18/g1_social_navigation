#!/usr/bin/env python3

"""
cmd_vel_qos_adapter.launch.py

This launch file starts the cmd_vel QoS Adapter node.

--------------------------------------------------------------------------------
PURPOSE
--------------------------------------------------------------------------------
The QoS adapter solves DDS reliability mismatches between Nav2 and rl_sar.

Typical issue:
- Nav2 (controller_server / velocity_smoother) publishes /cmd_vel_nav
  with RELIABLE QoS.
- rl_sar subscribes using SystemDefaultsQoS() which may resolve to BEST_EFFORT.
- RELIABLE publisher + BEST_EFFORT subscriber can lead to inconsistent
  behavior depending on RMW implementation and network conditions.

This adapter ensures:
  Nav2 output  --->  QoS Adapter  --->  rl_sar

--------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------
The node creates TWO subscriptions on the same input topic:

  1) RELIABLE subscription
  2) BEST_EFFORT subscription

Why?
Because the publishing PC may use either RELIABLE or BEST_EFFORT.
This makes the system robust across machines and DDS configurations.

Policy:
  - If a RELIABLE message is received, it is preferred.
  - If no RELIABLE message is seen recently, BEST_EFFORT is used.
  - Duplicate commands are filtered.
  - Output is always BEST_EFFORT (safe for rl_sar).

--------------------------------------------------------------------------------
DEFAULT PIPELINE
--------------------------------------------------------------------------------
controller_server  -->  /cmd_vel_nav (RELIABLE)
QoS adapter        -->  /cmd_vel_rlsar (BEST_EFFORT)
rl_sar             -->  subscribes to /cmd_vel_rlsar

So you can wire:
  velocity_smoother (/cmd_vel) -> twist_mux (/cmd_vel_mux) -> qos_adapter -> rl_sar (/cmd_vel_rlsar)

--------------------------------------------------------------------------------
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration



def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    yaw_feedback_odom_topic = LaunchConfiguration("yaw_feedback_odom_topic")

    qos_adapter_node = Node(
        package="cmd_vel_qos_adapter",
        executable="qos_adapter_node",
        name="cmd_vel_qos_adapter",
        output="screen",

        parameters=[
            {
                "use_sim_time": use_sim_time,

                # Input topic from Nav2 controller_server
                # This should be the *raw* navigation command,
                # before velocity_smoother or behavior_server interference.

                # ✅ Input topic from velocity_smoother output not Nav2 controller_server
                # Pipeline:
                #   controller_server -> /cmd_vel_nav
                #   velocity_smoother -> /cmd_vel
                #   twist_mux -> /cmd_vel_mux
                #   qos_adapter_node  -> /cmd_vel_rlsar
                "in_topic": input_topic,

                # Output topic towards rl_sar
                # rl_sar should subscribe to this topic
                # using BEST_EFFORT (default SystemDefaultsQoS on your system).
                "out_topic": output_topic,

                # Time window during which RELIABLE source is preferred.
                # If a RELIABLE message was seen within this duration,
                # BEST_EFFORT messages are ignored.
                "prefer_reliable_window_sec": 0.10, # was 0.25

                # Duplicate suppression window.
                # If the same Twist is received again within this time,
                # it will not be re-published.
                
                #"dedup_window_sec": 0.05,
                
                
                # ✅ Start with no dedup to avoid accidental drops
                # You can re-enable later (0.01..0.05) once stable
                "dedup_window_sec": 0.0,

                #"enable_straight_bias_correction": True,
                "enable_straight_bias_correction": False,
                #"use_yaw_feedback": True,
               
                "use_yaw_feedback": False,
                "yaw_feedback_odom_topic": yaw_feedback_odom_topic,
                "yaw_feedback_kp": 0.10,
                "yaw_feedback_max_correction": 0.025,

                # DimOS-like command adapter
                "cmd_vx_scale": 1.0,
                "cmd_vy_scale": 1.0,
                "cmd_wz_scale": 1.0,

                "cmd_vx_bias": 0.0,
                "cmd_vy_bias": 0.0,
                "cmd_wz_bias": 0.0,
                #"cmd_wz_bias": -0.03,

                # Start with this because the robot drifts left.
                # If it becomes worse, it can be changed to +0.02.
                "straight_wz_bias": -0.04,
                

                # Start WITHOUT feedback loop
                "use_yaw_feedback": False,
                
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Real robot: keep false unless you have /clock.",
        ),
        DeclareLaunchArgument(
            "input_topic",
            default_value="/cmd_vel",
            description="Input Twist topic to adapt QoS from (e.g. /cmd_vel_mux).",
        ),
        DeclareLaunchArgument(
            "output_topic",
            default_value="/cmd_vel_rlsar",
            description="Output Twist topic published as BEST_EFFORT (e.g. /cmd_vel_rlsar).",
        ),
        DeclareLaunchArgument("yaw_feedback_odom_topic", default_value="/dog_odom_filtered", description=""),
        qos_adapter_node,

    ])