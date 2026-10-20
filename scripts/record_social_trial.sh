#!/usr/bin/env bash

(
set -euo pipefail

# =============================================================================
# G1 SOCIAL NAVIGATION EXPERIMENT RECORDER
#
# Experimental configurations:
#
#   A = LiDAR-only navigation
#   C = LiDAR + social perception/navigation
#   D = LiDAR + social perception/navigation + communication/dialogue
#
# Scenarios:
#
#   S1 = person standing still
#   S2 = person crossing transversely in front of the robot
#   S3 = person approaching the robot
#
# Recorded data:
#
#   A:
#       Navigation, localization, LiDAR, plans, costmaps and velocity commands.
#
#   C:
#       A data
#       + YOLO/person tracking
#       + social perception
#       + social navigation decisions
#       + explainability
#       + speech lifecycle
#
#   D:
#       C data
#       + dialogue
#       + HRI
#       + ROS4HRI tracked identities
#
# Recording must start before the navigation goal is sent.
# =============================================================================


# =============================================================================
# EXPERIMENT SETTINGS
#
# Change these values before each trial.
# =============================================================================

campaign_id="restaurant_social_awareness"

#configuration="A"       # A | C | D
#scenario="S1"           # S1 | S2 | S3
#trial_number="01"

# Experiment selection from command-line arguments.
configuration="${1:-}"
scenario="${2:-}"
trial_number="${3:-}"

if [[ -z "$configuration" || -z "$scenario" || -z "$trial_number" ]]; then
    printf 'Usage: %s <A|C|D> <S1|S2|S3> <trial_number>\n' "$0"
    printf '\n'
    printf 'Examples:\n'
    printf '  %s A S1 01\n' "$0"
    printf '  %s C S2 01\n' "$0"
    printf '  %s D S3 01\n' "$0"
    exit 1
fi

# Used only for configuration D.
communication_initiator="robot"  # human | robot

experiment_root="$HOME/g1_ws/experiments"


# =============================================================================
# VALIDATION
# =============================================================================

case "$configuration" in
    A|C|D)
        ;;
    *)
        printf 'Invalid configuration: %s\n' "$configuration" >&2
        exit 1
        ;;
esac

case "$scenario" in
    S1|S2|S3)
        ;;
    *)
        printf 'Invalid scenario: %s\n' "$scenario" >&2
        exit 1
        ;;
esac

if [[ "$configuration" == "D" ]]; then
    case "$communication_initiator" in
        human|robot)
            ;;
        *)
            printf 'Invalid communication initiator: %s\n' \
                "$communication_initiator" >&2
            exit 1
            ;;
    esac
else
    communication_initiator="none"
fi


# =============================================================================
# CONFIGURATION DESCRIPTION
# =============================================================================

case "$configuration" in
    A)
        configuration_description="lidar_only"
        interaction_mode="none"
        ;;
    C)
        configuration_description="lidar_social"
        interaction_mode="none"
        ;;
    D)
        configuration_description="lidar_social_communication"
        interaction_mode="dialogue"
        ;;
esac


# =============================================================================
# SCENARIO DESCRIPTION
# =============================================================================

case "$scenario" in
    S1)
        scenario_description="standing_person"
        ;;
    S2)
        scenario_description="transverse_crossing"
        ;;
    S3)
        scenario_description="approaching_person"
        ;;
esac


# =============================================================================
# TRIAL DIRECTORY
# =============================================================================

trial_id="${configuration}_${scenario}_R${trial_number}"

trial_directory="${experiment_root}/${campaign_id}/${configuration}/${scenario}/${trial_id}"

bag_directory="${trial_directory}/navigation"

# Protect an existing trial from accidental overwrite.
if [[ -e "$trial_directory" ]]; then
    printf '\n'
    printf 'Trial directory already exists:\n'
    printf '  %s\n' "$trial_directory"
    printf '\n'
    printf 'Choose a new trial number before recording.\n'
    exit 1
fi

mkdir -p "$trial_directory"


# =============================================================================
# TRIAL METADATA
# =============================================================================

cat > "${trial_directory}/trial_metadata.yaml" <<EOF
campaign_id: "${campaign_id}"
trial_id: "${trial_id}"

configuration: "${configuration}"
configuration_description: "${configuration_description}"
interaction_mode: "${interaction_mode}"

scenario: "${scenario}"
scenario_description: "${scenario_description}"

trial_number: "${trial_number}"

planned_communication_initiator: "${communication_initiator}"

recording_started_utc: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ros_distribution: "${ROS_DISTRO:-unknown}"
EOF


# =============================================================================
# RUNTIME TOPIC INVENTORY
#
# Save all topics visible when the experiment starts.
# This provides a runtime trace of which ROS interfaces were available.
# =============================================================================

ros2 topic list -t --include-hidden-topics \
    > "${trial_directory}/available_topics.txt"


# =============================================================================
# COMMON NAVIGATION TOPICS
#
# These topics are recorded for A, C and D so navigation performance can be
# compared directly across all configurations.
# =============================================================================

topics=(
    /tf
    /tf_static

    /scan

    /map
    /map_runtime

    /initialpose
    /goal_pose

    /odom
    /amcl_pose

    /cmd_vel
    /cmd_vel_nav
    /cmd_vel_nav_scaled
    /cmd_vel_mux
    /cmd_vel_rlsar

    /plan
    /plan_smoothed
    /local_plan
    /received_global_plan
    /transformed_global_plan

    /global_costmap/costmap
    /global_costmap/costmap_updates
    /global_costmap/published_footprint

    /local_costmap/costmap
    /local_costmap/costmap_updates
    /local_costmap/published_footprint

    /navigate_to_pose/_action/feedback
    /navigate_to_pose/_action/status

    /navigate_to_pose_internal/_action/feedback
    /navigate_to_pose_internal/_action/status

    /follow_path/_action/status

    /rosout
)


# =============================================================================
# SOCIAL PERCEPTION AND SOCIAL NAVIGATION
#
# Enabled for configurations C and D.
# =============================================================================

if [[ "$configuration" == "C" || "$configuration" == "D" ]]; then
    topics+=(
        /yolo/detections
        /yolo/detections_tracked
        /yolo/detections_3d
        /yolo/detections_3d_base

        /social_perception/tracked_persons
        /social_perception/social_zones
        /social_perception/social_state

        /social_navigation/decision
        /social_navigation/episode_state
        /social_navigation/navigation_state
        /social_navigation/collaborative_decision
    )
fi


# =============================================================================
# EXPLAINABILITY AND SPEECH
#
# Explanation and TTS are intrinsic capabilities of social profiles C and D.
# They are not restricted to dialogue mode.
# =============================================================================

if [[ "$configuration" == "C" || "$configuration" == "D" ]]; then
    topics+=(
        /explainability/decision_request
        /explainability/context
        /explainability/text
        /explainability/speech_status
    )
fi


# =============================================================================
# COMMUNICATION / DIALOGUE / HRI
#
# Configuration D only.
#
# Configuration D corresponds to:
#     interaction:=dialogue
# =============================================================================

if [[ "$configuration" == "D" ]]; then
    topics+=(
        /hri/dialogue_request
        /hri/human_intent

        /dialogue_manager/robot_speech
        /dialogue_manager/closed_captions
        /dialogue_manager/currently_waiting_for_chatbot_response

        /intents

        /humans/persons/tracked
        /humans/bodies/tracked
        /humans/faces/tracked
        /humans/voices/tracked
    )
fi


# =============================================================================
# SAVE REQUESTED TOPICS
# =============================================================================

printf '%s\n' "${topics[@]}" \
    > "${trial_directory}/requested_topics.txt"


# =============================================================================
# RECORDING SUMMARY
# =============================================================================

printf '\n'
printf '====================================================================\n'
printf 'G1 EXPERIMENT RECORDING\n'
printf '====================================================================\n'
printf 'Trial ID       : %s\n' "$trial_id"
printf 'Configuration  : %s\n' "$configuration"
printf 'Description    : %s\n' "$configuration_description"
printf 'Interaction    : %s\n' "$interaction_mode"
printf 'Scenario       : %s\n' "$scenario"
printf 'Scenario desc. : %s\n' "$scenario_description"
printf 'Trial number   : %s\n' "$trial_number"
printf 'Comm initiator : %s\n' "$communication_initiator"
printf 'Bag directory  : %s\n' "$bag_directory"
printf '====================================================================\n'
printf '\n'

printf 'Topics requested: %d\n' "${#topics[@]}"
printf '\n'

printf 'IMPORTANT:\n'
printf '  1. Wait until rosbag reports that recording has started.\n'
printf '  2. Then send the navigation goal.\n'
printf '  3. Run the complete scenario.\n'
printf '  4. Wait until the navigation/social interaction is finished.\n'
printf '  5. Press Ctrl+C once to stop recording cleanly.\n'
printf '\n'


# =============================================================================
# ROSBAG RECORDING
# =============================================================================

ros2 bag record \
    --include-hidden-topics \
    -o "$bag_directory" \
    "${topics[@]}"

)

