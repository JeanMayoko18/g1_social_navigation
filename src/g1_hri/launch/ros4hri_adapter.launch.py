"""
ROS4HRI compatibility launch.

Flow
----

    ROS4HRI / dialogue_manager
              |
              v
         g1_hri manager
              |
              v
          HumanIntent

Speech flow is separate:

    ROS4HRI TTS request
              |
              v
      Common Speech API
              |
              v
        Speech Manager
              |
              v
            Piper

Responsibilities
----------------

This launch owns the ROS4HRI compatibility boundary.

It is valid only on ROS 2 Humble and only when the system resolver
has selected interaction=dialogue for the local host.

It does NOT:
- own CollaborationManager;
- own navigation;
- own NavigationGoalGate;
- own Piper;
- own the final audio backend.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    dialogue_manager_share = get_package_share_directory(
        "dialogue_manager"
    )

    dialogue_manager_launch = os.path.join(
        dialogue_manager_share,
        "launch",
        "dialogue_manager.launch.py",
    )

    dialogue_manager = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            dialogue_manager_launch
        )
    )
    
    tts_adapter = Node(
        package="g1_hri",
        executable="ros4hri_tts_adapter",
        name="ros4hri_tts_adapter",
        output="screen",
        parameters=[
            {
                "action_name": "tts_engine/tts",
                "speech_request_topic": "/explainability/text",
                "speech_status_topic": "/explainability/speech_status",
                "default_language": "en",
                "speech_timeout_sec": 60.0,
            }
        ],
    )

    return LaunchDescription(
        [
            tts_adapter,
            dialogue_manager,
        ]
    )