"""
G1 navigation and social system entry point.

FLOW
====


                    system.launch.py
                           |
                           v
                    system_resolver
                           |
          +----------------+----------------+
          |                |                |
       profile         deployment       interaction
       A/B/C/D         role placement   none/dialogue
          |                |                |
          +----------------+----------------+
                           |
                           v
                      local roles
                           |
          +----------------+----------------+
          |                |                |
      perception       navigation         speech
      social decision  explanation        dialogue
      visualization
                           |
                           v
                    role launch files


                    TEXT PRODUCERS

ExplanationNode --------------------------+
                                          |
NavigationInteractionSpeech --------------+---> /explainability/text
                                          |
NavigationGoalGate -----------------------+
                                          |
future ROS4HRI adapter -------------------+
                                                   |
                                                   v
                                      +-------------------------+
                                      | speech.launch.py        |
                                      |                         |
                                      | PiperTTSNode            |
                                      |   ONE queue             |
                                      |   Piper synthesis       |
                                      |   audio backend         |
                                      +------------+------------+
                                                   |
                                                   v
                                  /explainability/speech_status

Matrix

                    pc_all   robot_all   robot_perception   pc_perception
vision_inference       PC       Robot           PC                PC
social_perception      PC       Robot          Robot               PC
social_decision        PC       Robot           PC                PC
navigation             PC       Robot           PC                PC
explanation / LLM      PC       Robot           PC                PC
speech                 PC       Robot          Robot              Robot
visualization          PC        PC             PC                PC

                   SYSTEM CONFIGURATION
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
    PROFILE          INTERACTION       DEPLOYMENT
    A/B/C/D         none/dialogue      role → host
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                         host=pc                      host=robot
                            │                             │
                     local active roles            local active roles
                            │                             │
                            └──────── ROS 2 ─────────────┘

                                     +
                              rl_sar separately
                              on the G1 robot
IMPORTANT
=========
rl_sar is NOT launched here.

The robot hardware stack must be started independently.

This launch consumes the interfaces provided by the hardware stack
but never owns:
    - LiDAR hardware
    - RealSense hardware
    - IMU hardware
    - canonical odometry
    - hardware TF
    - low-level G1 control
"""

import importlib.util
import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.conditions import IfCondition

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

from launch.substitutions import LaunchConfiguration


def _load_system_resolver():
    package_share = get_package_share_directory("g1_social_bringup")

    resolver_path = os.path.join(
        package_share,
        "launch",
        "system_resolver.py",
    )

    spec = importlib.util.spec_from_file_location(
        "g1_social_bringup_system_resolver",
        resolver_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load system resolver: {resolver_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def _start_after_readiness(
    event,
    context,
    actions,
    label,
    start_on_failure=False,
):
    """
    Start dependent launch actions after the readiness process exits.

    Normal path:
        readiness succeeds -> start dependent subsystem.

    Optional fallback:
        readiness fails -> start dependent subsystem anyway when
        start_on_failure=True.
    """
    del context

    if event.returncode == 0:
        return [
            LogInfo(
                msg=(
                    f"[IMAGE INGRESS READY] {label} readiness passed. "
                    "Starting dependent subsystem."
                )
            ),
            *actions,
        ]

    if start_on_failure:
        return [
            LogInfo(
                msg=(
                    f"[IMAGE INGRESS FALLBACK] {label} readiness FAILED "
                    f"(returncode={event.returncode}). "
                    "Starting dependent subsystem without readiness dependency."
                )
            ),
            *actions,
        ]

    return [
        LogInfo(
            msg=(
                f"[IMAGE INGRESS READY] {label} readiness FAILED "
                f"(returncode={event.returncode}). "
                "Dependent subsystem will NOT be started."
            )
        )
    ]

def _resolve_and_launch(context):
    resolver = _load_system_resolver()

    profile = LaunchConfiguration("profile").perform(context)
    interaction = LaunchConfiguration("interaction").perform(context)
    deployment = LaunchConfiguration("deployment").perform(context)
    host = LaunchConfiguration("host").perform(context)
    ros_distro = LaunchConfiguration("ros_distro").perform(context)
    map_yaml = LaunchConfiguration("map").perform(context)
    language = LaunchConfiguration("language").perform(context).strip().lower()

    if language not in ("en", "es", "fr"):
        raise RuntimeError(
            f"Unsupported language='{language}'. "
            "Expected 'en', 'es' or 'fr'."
        )

    navigation_mode = LaunchConfiguration(
        "navigation_mode"
    ).perform(context)

    controller = LaunchConfiguration(
        "controller"
    ).perform(context)

    camera_transport = LaunchConfiguration(
        "camera_transport"
    ).perform(context)

    navigation_speed = LaunchConfiguration(
        "navigation_speed"
    ).perform(context)

    start_visual_odometry = LaunchConfiguration(
        "start_visual_odometry"
    ).perform(context)

    enable_locomotion_calibration = LaunchConfiguration(
        "enable_locomotion_calibration"
    ).perform(context)

    locomotion_calibration_profile = LaunchConfiguration(
        "locomotion_calibration_profile"
    ).perform(context)

    forward_lateral_compensation_profile = LaunchConfiguration(
        "forward_lateral_compensation_profile"
    ).perform(context)

    forward_heading_compensation_profile = LaunchConfiguration(
        "forward_heading_compensation_profile"
    ).perform(context)

    plan = resolver.resolve_system(
        profile=profile,
        interaction=interaction,
        deployment=deployment,
        host=host,
        ros_distro=ros_distro,
    )
    
    bringup_share = get_package_share_directory(
        "g1_social_bringup"
    )

    vision_is_local = plan.owns("vision_inference")
    social_perception_is_local = plan.owns(
        "social_perception"
    )

    enable_yolo = (
        LaunchConfiguration("enable_yolo")
        .perform(context)
        .strip()
        .lower()
        == "true"
    )

    enable_ros4hri_visualization = (
        LaunchConfiguration(
            "enable_ros4hri_visualization"
        ).perform(context).strip().lower()
        == "true"
    )

    enable_visualization = (
        LaunchConfiguration(
            "enable_visualization"
        ).perform(context).strip().lower()
        == "true"
    )

    enable_social_rviz = (
        LaunchConfiguration("enable_social_rviz")
        .perform(context)
        .strip()
        .lower() == "true"
    )

    enable_raw_image_visualization = (
        LaunchConfiguration(
            "enable_raw_image_visualization"
        ).perform(context).strip().lower()
        == "true"
    )

    camera_host = "robot"

    remote_rgb_required = (
        vision_is_local
        and plan.host != camera_host
    )

    remote_depth_required = (
        social_perception_is_local
        and plan.host != camera_host
    )

    visualization_is_local = plan.owns(
        "visualization"
    )

    visualization_rgb_topic = (
        "/visualization/raw_images_disabled/rgb"
    )

    visualization_depth_topic = (
        "/visualization/raw_images_disabled/depth"
    )

    if (
        visualization_is_local
        and enable_raw_image_visualization
    ):
        if remote_rgb_required:
            visualization_rgb_topic = (
                "/perception/transport/color"
                if camera_transport == "compressed"
                else "/perception/ingress/color/image_raw"
    )
        elif plan.host == camera_host:
            visualization_rgb_topic = (
                "/camera/camera/color/image_raw"
            )

        if remote_depth_required:
            visualization_depth_topic = (
                "/perception/ingress/"
                "aligned_depth/image_raw"
            )
        elif plan.host == camera_host:
            visualization_depth_topic = (
                "/camera/camera/"
                "aligned_depth_to_color/image_raw"
            )

    role_map = ", ".join(
        f"{role}={role_host}"
        for role, role_host in plan.role_hosts.items()
    )

    local_roles = ", ".join(plan.local_roles)

    if not local_roles:
        local_roles = "none"

    pc_audio_fallback_required_old = (
        plan.host == "pc"
        and plan.ros_distro == "humble"
        and not plan.owns("speech")
        and plan.speech_required
    )
    pc_audio_fallback_required = (
        not plan.owns("speech")
    )

    actions = [
        LogInfo(msg="=" * 72),
        LogInfo(msg="G1 SYSTEM PLAN"),
        LogInfo(msg="=" * 72),
        LogInfo(msg=f"Host              : {plan.host}"),
        LogInfo(msg=f"ROS distribution  : {plan.ros_distro}"),
        LogInfo(msg=f"Profile           : {plan.profile}"),
        LogInfo(msg=f"Interaction       : {plan.interaction}"),
        LogInfo(msg=f"Deployment        : {plan.deployment}"),
        LogInfo(
            msg=f"Camera obstacles  : {plan.camera_obstacles}"
        ),
        LogInfo(
            msg=f"Social navigation : {plan.social_navigation}"
        ),
        LogInfo(msg=f"Explanation       : {plan.explanation}"),
        LogInfo(msg=f"Speech required   : {plan.speech_required}"),
        LogInfo(msg=f"Dialogue          : {plan.dialogue}"),
        LogInfo(msg=f"Role placement    : {role_map}"),
        LogInfo(msg=f"Local roles       : {local_roles}"),
        LogInfo(msg="Hardware owner    : rl_sar package"),
        LogInfo(msg="=" * 72),
    ]

    # ---------------------------------------------------------
    # IMAGE TRANSPORT
    # ---------------------------------------------------------
    if camera_transport == "compressed":

        if plan.host == "robot":
            actions.append(
                Node(
                    package="image_transport",
                    executable="republish",
                    name="rgb_network_transport_sender",
                    arguments=[
                        "raw",
                        "compressed",
                    ],
                    remappings=[
                        ("in", "/camera/camera/color/image_raw"),
                        ("out/compressed", "/social_navigation/camera/color/compressed"),
                    ],
                    output="screen",
                )
            )

        elif plan.host == "pc" and remote_rgb_required:
            actions.append(
                Node(
                    package="image_transport",
                    executable="republish",
                    name="rgb_network_transport_receiver",
                    arguments=[
                        "compressed",
                        "raw",
                    ],
                    remappings=[
                        ("in/compressed", "/social_navigation/camera/color/compressed"),
                        ("out", "/perception/transport/color"),
                    ],
                    output="screen",
                )
            )

    # ---------------------------------------------------------
    # IMAGE INGRESS
    # ---------------------------------------------------------
    if remote_rgb_required or remote_depth_required:

        image_ingress_launch = os.path.join(
            bringup_share,
            "launch",
            "image_ingress.launch.py",
        )

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    image_ingress_launch
                ),

                launch_arguments={
                    "enable_rgb": (
                        "true"
                        if remote_rgb_required and camera_transport != "compressed"
                        else "false"
                    ),
                    "enable_depth": (
                        "true"
                        if remote_depth_required
                        else "false"
                    ),
                    "enable_camera_info": (
                        "true"
                        if remote_depth_required
                        else "false"
                    ),
                    #"rgb_input_topic": "/camera/camera/color/image_raw",
                    "rgb_input_topic": (
                        "/perception/transport/color"
                        if camera_transport == "compressed"
                        else "/camera/camera/color/image_raw"
                    ),
                }.items(),


            )
        )


    # ---------------------------------------------------------
    # IMAGE INGRESS READINESS BARRIERS
    # ---------------------------------------------------------
    #
    # Remote image consumers must not start merely because an ingress
    # process exists. They start only after at least one valid message
    # has actually been received on every stream they require.
    #
    # RGB_READY:
    #     valid RGB -> VisionInference may start
    #
    # DEPTH_READY:
    #     valid depth + valid CameraInfo -> SocialPerception may start
    #
    # Local camera consumers do not use these barriers.

    rgb_readiness = None
    depth_readiness = None

    if remote_rgb_required:
        rgb_readiness = Node(
            package="g1_social_bringup",
            executable="wait_for_image_ingress",
            name="wait_for_rgb_ingress",
            output="screen",
            remappings=[
                (
                    "/perception/ingress/color/image_raw",
                    "/perception/transport/color",
                )
            ] if camera_transport == "compressed" else [],
            parameters=[
                {
                    "require_rgb": True,
                    "require_depth": False,
                    "require_camera_info": False,
                    "timeout_sec": 60.0,
                }
            ],
        )

    if remote_depth_required:
        depth_readiness = Node(
            package="g1_social_bringup",
            executable="wait_for_image_ingress",
            name="wait_for_depth_ingress",
            output="screen",
            parameters=[
                {
                    "require_rgb": False,
                    "require_depth": True,
                    "require_camera_info": True,
                    "timeout_sec": 15.0,
                }
            ],
        )

    # ---------------------------------------------------------
    # VISION INFERENCE ROLE
    # ---------------------------------------------------------
    #
    # YOLO placement is independent from downstream social
    # perception placement.
    #
    # In robot_perception, YOLO intentionally remains on the PC
    # while fusion/tracking/zones run on the robot.
    #
    # Step 8D will insert ImageIngress when the RGB producer and
    # YOLO are on different hosts.


    if (
        plan.social_navigation
        and plan.owns("vision_inference")
        and enable_yolo
    ):
        vision_inference_launch = os.path.join(
            bringup_share,
            "launch",
            "vision_inference.launch.py",
        )

        vision_inference_action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                vision_inference_launch
            ),
            launch_arguments={
                "enable_yolo": "true",
                "input_image_topic": (
                    "/perception/transport/color"
                    if remote_rgb_required and camera_transport == "compressed"
                    else (
                        "/perception/ingress/color/image_raw"
                        if remote_rgb_required
                        else "/camera/camera/color/image_raw"
                    )
                ),
            }.items(),
        )

        vision_log_action = LogInfo(
            msg=(
                "Starting local vision inference role "
                f"on host '{plan.host}'."
            )
        )

        if remote_rgb_required:
            actions.append(
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=rgb_readiness,
                        on_exit=lambda event, context: (
                            _start_after_readiness(
                                event,
                                context,
                                [
                                    vision_inference_action,
                                    vision_log_action,
                                ],
                                "RGB",
                                start_on_failure=True,
                            )
                        ),
                    )
                )
            )

            actions.append(rgb_readiness)
        else:
            actions.extend(
                [
                    vision_inference_action,
                    vision_log_action,
                ]
            )

    # ---------------------------------------------------------
    # SOCIAL PERCEPTION ROLE
    # ---------------------------------------------------------
    #
    # This role contains the downstream processing chain only:
    # fusion, tracking, transforms, and social zones.
    #
    # YOLO is not owned by this role.

    if (
        plan.social_navigation
        and plan.owns("social_perception")
    ):
        social_perception_launch = os.path.join(
            bringup_share,
            "launch",
            "social_perception.launch.py",
        )

        social_perception_action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                social_perception_launch
            ),
            launch_arguments={
                "depth_topic": (
                    "/perception/ingress/"
                    "aligned_depth/image_raw"
                    if remote_depth_required
                    else (
                        "/camera/camera/"
                        "aligned_depth_to_color/image_raw"
                    )
                ),
                "language": language,
                "camera_info_topic": (
                    "/perception/ingress/"
                    "aligned_depth/camera_info"
                    if remote_depth_required
                    else (
                        "/camera/camera/"
                        "aligned_depth_to_color/camera_info"
                    )
                ),
            }.items(),
        )

        social_perception_log_action = LogInfo(
            msg=(
                "Starting local social perception role "
                f"on host '{plan.host}'."
            )
        )

        # Start social perception directly.
        #
        # ImageIngress owns the network boundary when depth is remote.
        # The real consumer subscribes directly to the ingress topics;
        # no temporary readiness subscriber is inserted in the data path.
        actions.extend(
            [
                social_perception_action,
                social_perception_log_action,
            ]
        )

    # ---------------------------------------------------------
    # SOCIAL DECISION ROLE
    # ---------------------------------------------------------
    #
    # Profiles C and D use the social decision chain.
    #
    # interaction=none:
    #     deterministic social decision
    #         -> CollaborationManager(hri_enabled=false)
    #         -> CollaborativeDecision
    #
    # interaction=dialogue:
    #     deterministic social decision
    #         -> CollaborationManager(hri_enabled=true)
    #         -> CollaborativeDecision
    #
    # Dialogue-specific HRI/ROS4HRI processes are launched
    # separately and are not owned by this role.

    if (
        plan.social_navigation
        and plan.owns("social_decision")
    ):
        social_decision_launch = os.path.join(
            bringup_share,
            "launch",
            "social_decision.launch.py",
        )

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    social_decision_launch
                ),
                launch_arguments={
                    "interaction": plan.interaction,
                    "host": plan.host,
                    "language": language,
                }.items(),
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting local social decision role "
                    f"on host '{plan.host}' "
                    f"with interaction='{plan.interaction}'."
                )
            )
        )


    # ---------------------------------------------------------
    # DIALOGUE ROLE
    # ---------------------------------------------------------
    #
    # Dialogue is an independent interaction capability.
    #
    # interaction=none:
    #     no HRI manager
    #     no ROS4HRI
    #
    # interaction=dialogue:
    #
    #     CollaborationManager
    #             |
    #             | DialogueRequest
    #             v
    #         HRIManager
    #             |
    #             | HumanIntent
    #             v
    #     CollaborationManager
    #
    # ROS4HRI is a compatibility boundary around the dialogue
    # subsystem. It does not own navigation or final speech.
    #
    # The resolver guarantees that a local dialogue role is
    # started only on a compatible ROS 2 Humble host.

    if (
        plan.dialogue
        and plan.owns("dialogue")
    ):
        hri_share = get_package_share_directory(
            "g1_hri"
        )

        hri_launch = os.path.join(
            hri_share,
            "launch",
            "hri.launch.py",
        )

        ros4hri_adapter_launch = os.path.join(
            hri_share,
            "launch",
            "ros4hri_adapter.launch.py",
        )

        # HRI manager
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    hri_launch
                ),
                launch_arguments={
                    "hri_enabled": "true",
                }.items(),
            )
        )

        # ROS4HRI / dialogue manager
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    ros4hri_adapter_launch
                )
            )
        )


        actions.append(
            LogInfo(
                msg=(
                    "Starting local dialogue role "
                    f"on host '{plan.host}' "
                    "with HRI, ROS4HRI and PC ASR."
                )
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting local dialogue role "
                    f"on host '{plan.host}' "
                    "with HRI and ROS4HRI compatibility."
                )
            )
        )

        # ---------------------------------------------------------
        # PC MICROPHONE ASR
        # ---------------------------------------------------------
        #
        # The dialogue role is owned by the Humble-compatible PC.
        # ASR publishes ROS4HRI LiveSpeech consumed by the existing
        # dialogue pipeline.
        #
        # It is intentionally not part of the collaboration role:
        # collaboration consumes HumanIntent and does not own
        # speech recognition.

        pc_asr_node = Node(
            package="g1_pc_asr",
            executable="pc_asr",
            name="g1_pc_asr",
            output="screen",
            parameters=[
                {
                    "voice_id": "pc_user",
                    "language": language,
                    "model": "small",
                    "device": "cpu",
                    "compute_type": "int8",

                    "sample_rate": 16000,
                    "audio_device": -1,

                    "vad_filter": True,

                    "chunk_ms": 30,
                    "speech_start_rms": 180.0,
                    "speech_end_rms": 100.0,
                    "end_silence_sec": 0.55,
                    "min_speech_sec": 0.20,
                    "max_speech_sec": 8.0,
                    "pre_roll_sec": 0.20,
                }
            ],
        )

        actions.append(pc_asr_node)

    # ---------------------------------------------------------
    # NAVIGATION ROLE
    # ---------------------------------------------------------
    #
    # All profiles use Nav2.
    #
    # Profiles A and B:
    #     client -> Nav2
    #
    # Profiles C and D:
    #     client -> NavigationGoalGate -> Nav2
    #
    # system.launch.py owns the profile-to-gate policy.
    # navigation.launch.py only receives the resolved boolean.

    if plan.owns("navigation"):
        navigation_launch = os.path.join(
            bringup_share,
            "launch",
            "navigation.launch.py",
        )

        enable_navigation_goal_gate = (
            "true"
            if plan.social_navigation
            else "false"
        )

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    navigation_launch
                ),
                launch_arguments={
                    "profile": plan.profile,
                    "ros_distro": plan.ros_distro,
                    "enable_navigation_goal_gate": (
                        enable_navigation_goal_gate
                    ),
                    "map": map_yaml,
                    # Profiles C and D require speech.
                    # Navigation only produces speech requests;
                    # speech.launch.py owns Piper and audio.
                    "enable_navigation_announcements": (
                        "true"
                        if plan.speech_required
                        else "false"
                    ),

                    "navigation_mode": navigation_mode,
                    "controller": controller,
                    "navigation_speed": navigation_speed,
                    "start_visual_odometry": start_visual_odometry,

                    "enable_locomotion_calibration": (
                        enable_locomotion_calibration
                    ),
                    "locomotion_calibration_profile": (
                        locomotion_calibration_profile
                    ),
                    "forward_lateral_compensation_profile": (
                        forward_lateral_compensation_profile
                    ),
                    "forward_heading_compensation_profile": (
                        forward_heading_compensation_profile
                    ),
                    "language": language,

                    # The gate owns goal-reception speech in C/D,
                    # therefore the generic navigation announcer
                    # must not duplicate it.
                    "enable_direct_goal_speech": "false",

                    # Nav2 RViz is the primary navigation visualization
                    # for every profile.
                    #
                    # Profiles C/D additionally start the Social RViz
                    # through the visualization role below.
                    "start_rviz": (
                        "true"
                        if (
                            enable_visualization
                            and plan.owns("visualization")
                        )
                        else "false"
                    ),
                    

                }.items(),
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting local navigation role "
                    f"on host '{plan.host}' | "
                    "navigation_goal_gate="
                    f"{enable_navigation_goal_gate}."
                )
            )
        )

    # ---------------------------------------------------------
    # EXPLANATION ROLE
    # ---------------------------------------------------------
    #
    # Explanation is intrinsic to social profiles C and D.
    #
    # NOTE:
    # Explanation role: LLM/context/explanation generation only.
    # Speech and navigation execution are owned by separate launch roles.

    if (
        plan.explanation
        and plan.owns("explanation")
    ):
        explanation_share = get_package_share_directory(
            "g1_robot_explainability"
        )

        explainability_launch = os.path.join(
            explanation_share,
            "launch",
            "explainability.launch.py",
        )

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    explainability_launch
                ),
                launch_arguments={
                    "runtime_host": plan.host,
                    "ros_distro": plan.ros_distro,
                    "language": language,
                    "enable_explanation_node": "true",
                }.items(),
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting local explanation role "
                    f"on host '{plan.host}'."
                )
            )
        )

    # ---------------------------------------------------------
    # PC AUDIO FALLBACK EXECUTOR
    # ---------------------------------------------------------
    #
    # The PC fallback executor is NOT a speech authority.
    #
    # It is started only when:
    #
    #   - this machine is the Humble PC;
    #   - speech is required by the active profile;
    #   - another host owns the real speech role.
    #
    # Therefore:
    #
    #   PC with local PiperTTS
    #       -> no fallback executor
    #
    #   PC while robot owns PiperTTS
    #       -> fallback executor available
    #
    #   Robot
    #       -> never launch PC fallback executor
    #
    # The executor subscribes ONLY to:
    #
    #   /explainability/pc_audio_fallback/request
    #
    # and never to:
    #
    #   /explainability/text
    #
    # so there remains exactly one speech authority.

    if pc_audio_fallback_required:

        actions.append(
            Node(
                package="g1_robot_explainability",
                executable="pc_audio_fallback",
                name="pc_audio_fallback",
                output="screen",
                parameters=[
                    {
                        "enabled": True,
                        "request_topic": (
                            "/explainability/"
                            "pc_audio_fallback/request"
                        ),
                        "status_topic": (
                            "/explainability/"
                            "pc_audio_fallback/status"
                        ),
                        "audio_player": "aplay",
                    }
                ],
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting passive PC audio fallback executor | "
                    f"host='{plan.host}' | "
                    f"ros_distro='{plan.ros_distro}' | "
                    "local_speech_owner=false."
                )
            )
        )
        
    # ---------------------------------------------------------
    # SPEECH ROLE
    # ---------------------------------------------------------
    #
    # Profiles C and D always require speech.
    #
    # Exactly one host owns the final speech engine.
    # All text producers publish to the common speech API.
    #
    # This role owns:
    #     - one speech queue
    #     - Piper synthesis
    #     - final audio output
    #
    # It does not own explanation generation.

    if (
        plan.speech_required
        and plan.owns("speech")
    ):
        speech_launch = os.path.join(
            bringup_share,
            "launch",
            "speech.launch.py",
        )

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    speech_launch
                ),
                launch_arguments={
                    "runtime_host": plan.host,
                    "ros_distro": plan.ros_distro,
                }.items(),
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting local speech role "
                    f"on host '{plan.host}'."
                )
            )
        )

    # ---------------------------------------------------------
    # VISUALIZATION ROLE
    # ---------------------------------------------------------
    #
    # Visualization is independent from perception execution.
    #
    # Stopping RViz must never stop perception.
    #
    #    | Deployment PC | RGB RViz | Depth RViz |
    #    |---------------|----------|------------|
    #    | `pc_all`      | ingress  | ingress    |
    #    | `pc_perception`| ingress | ingress    |
    #    | `robot_perception` | ingress | disabled |
    #    | `robot_all`   | disabled |   disabled |

    # Raw image visualization is disabled by default. When
    # explicitly enabled, only image streams already available
    # locally for the selected deployment are used.

    if (
        enable_visualization
        and enable_social_rviz
        and plan.owns("visualization")
        and profile in ("C", "D")
    ):
        visualization_launch = os.path.join(
            bringup_share,
            "launch",
            "visualization.launch.py",
        )

        visualization_action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                visualization_launch
            ),
            launch_arguments={
                "enable_visualization": "true",
                "rgb_topic": visualization_rgb_topic,
                "aligned_depth_topic": visualization_depth_topic,
                "enable_ros4hri_visualization": (
                    "true"
                    if enable_ros4hri_visualization
                    else "false"
                ),
            }.items(),
        )

        actions.append(
            TimerAction(
                period=8.0,
                actions=[visualization_action],
            )
        )

        actions.append(
            LogInfo(
                msg=(
                    "Starting visualization role "
                    f"on host '{plan.host}' | "
                    "raw_images="
                    f"{enable_raw_image_visualization} | "
                    f"rgb='{visualization_rgb_topic}' | "
                    f"depth='{visualization_depth_topic}'."
                )
            )
        )

    return actions

def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="A",
                description=(
                    "Navigation/perception profile: "
                    "A | B | C | D"
                ),
            ),

            DeclareLaunchArgument(
                "interaction",
                default_value="none",
                description=(
                    "Human interaction capability: "
                    "none | dialogue"
                ),
            ),

            DeclareLaunchArgument(
                "map",
                default_value=os.path.join(
                    os.path.expanduser("~"),
                    "g1_ws",
                    "maps",
                    "mic.yaml",
                ),
                description="Nav2 map YAML.",
            ),

            DeclareLaunchArgument(
                "deployment",
                default_value="pc_perception",
                description=(
                    "Software role placement: "
                    "pc_all | robot_all | "
                    "robot_perception | pc_perception"
                ),
            ),

            DeclareLaunchArgument(
                "camera_transport",
                default_value="raw",
                choices=["raw", "compressed"],
                description=(
                    "Transport used for remote RGB images. "
                    "'raw' transports sensor_msgs/Image directly; "
                    "'compressed' uses image_transport compressed transport. "
                    "Depth remains raw."
                ),
            ),

            DeclareLaunchArgument(
                "host",
                description=(
                    "Current machine: pc | robot"
                ),
            ),

            DeclareLaunchArgument(
                "enable_social_rviz",
                default_value="true",
                choices=["true", "false"],
                description="Enable Social RViz independently of navigation RViz.",
            ),

            DeclareLaunchArgument(
                "ros_distro",
                default_value="auto",
                description=(
                    "ROS distribution: auto | foxy | humble"
                ),
            ),

            DeclareLaunchArgument(
                "enable_visualization",
                default_value="true",
                description=(
                    "Enable visualization on the host owning the "
                    "visualization role."
                ),
            ),

            DeclareLaunchArgument(
                "enable_raw_image_visualization",
                default_value="false",
                description=(
                    "Allow visualization to consume RGB/depth streams. "
                    "Disabled by default to avoid accidental network traffic."
                ),
            ),

            DeclareLaunchArgument(
                "navigation_mode",
                default_value="nav2",
                choices=["nav2", "slam"],
                description="Navigation mode.",
            ),

            DeclareLaunchArgument(
                "language",
                default_value="en",
                choices=["en", "es", "fr"],
                description=(
                    "System-wide spoken language. "
                    "Propagated to all speech-producing subsystems."
                ),
            ),

            DeclareLaunchArgument(
                "controller",
                default_value="dwb",
                choices=["dwb", "rpp"],
                description="Nav2 controller.",
            ),

            DeclareLaunchArgument(
                "navigation_speed",
                default_value="0.20",
                description="Maximum navigation speed in m/s.",
            ),

            DeclareLaunchArgument(
                "start_visual_odometry",
                default_value="true",
                choices=["true", "false"],
                description="Start visual odometry.",
            ),

            DeclareLaunchArgument(
                "enable_locomotion_calibration",
                default_value="false",
                choices=["true", "false"],
                description="Enable locomotion calibration.",
            ),

            DeclareLaunchArgument(
                "enable_yolo",
                default_value="true",
                choices=["true", "false"],
                description=(
                    "Enable YOLO vision inference when the current host "
                    "owns the vision_inference role."
                ),
            ),

            DeclareLaunchArgument(
                "locomotion_calibration_profile",
                default_value="previous_reference_scaled",
                description="Locomotion calibration profile.",
            ),

            DeclareLaunchArgument(
                "forward_lateral_compensation_profile",
                default_value="previous_reference_scaled",
                description="Forward lateral compensation profile.",
            ),

            DeclareLaunchArgument(
                "forward_heading_compensation_profile",
                default_value="previous_reference_scaled",
                description="Forward heading compensation profile.",
            ),

            DeclareLaunchArgument(
                "enable_ros4hri_visualization",
                default_value="false",
                choices=["true", "false"],
                description=(
                    "Enable ROS4HRI visualization in Social RViz. "
                    "This does not enable dialogue or HRI arbitration."
                ),
            ),

            OpaqueFunction(function=_resolve_and_launch),
        ]
    )