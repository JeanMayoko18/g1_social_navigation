
#!/usr/bin/env bash

(
set -euo pipefail

# =============================================================================
# G1 SOCIAL NAVIGATION EXPERIMENT RECORDER
#
# Usage:
#
#   bash scripts/record_experiment.sh \
#       <technical|participant> \
#       <A|C|D> \
#       <S1a|S1b|S2|S3|S4|S5> \
#       <trial_number>
#
# Examples:
#
#   bash scripts/record_experiment.sh technical C S1a 01
#   bash scripts/record_experiment.sh technical D S1a 06
#   bash scripts/record_experiment.sh participant D S5 01
#
# Diagnostics:
#
#   DIAG_RECORD_LIVOX_RAW=0  (default)
#   DIAG_SYSTEM_METRICS=1    (default)
#
# Experimental configurations:
#
#   A = LiDAR navigation reference
#
#   C = Social navigation supervision
#       + deterministic social announcement
#       + no dialogue
#
#   D = Social navigation supervision
#       + announcement
#       + dialogue
#
# Recording must start before the observed events.
# This script records data only. It commands no movement.
# =============================================================================

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

campaign_id="restaurant_social_awareness"

experiment_root="$HOME/g1_ws/experiments"
workspace_root="$HOME/g1_ws"

communication_initiator="${COMMUNICATION_INITIATOR:-robot}"

# Diagnostic switches
record_livox_raw="${DIAG_RECORD_LIVOX_RAW:-0}"
system_metrics="${DIAG_SYSTEM_METRICS:-1}"

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

trial_type="${1:-}"
configuration="${2:-}"
scenario="${3:-}"
trial_number="${4:-}"

if [[ $# -ne 4 ||
      -z "$trial_type" ||
      -z "$configuration" ||
      -z "$scenario" ||
      -z "$trial_number" ]]; then

    printf 'Usage:\n'
    printf '  %s <technical|participant> <A|C|D> <S1a|S1b|S2|S3|S4|S5> <trial_number>\n' "$0"
    exit 1
fi

# =============================================================================
# VALIDATION
# =============================================================================

case "$trial_type" in
    technical|participant) ;;
    *)
        echo "Invalid trial type: $trial_type" >&2
        exit 1
        ;;
esac

case "$configuration" in
    A|C|D) ;;
    *)
        echo "Invalid configuration: $configuration" >&2
        exit 1
        ;;
esac

case "$scenario" in
    S1a|S1b|S2|S3|S4|S5) ;;
    *)
        echo "Invalid scenario: $scenario" >&2
        exit 1
        ;;
esac

if [[ ! "$trial_number" =~ ^[0-9]+$ ]]; then
    echo "Trial number must contain digits only." >&2
    exit 1
fi

case "$record_livox_raw" in
    0|1) ;;
    *)
        echo "DIAG_RECORD_LIVOX_RAW must be 0 or 1." >&2
        exit 1
        ;;
esac

case "$system_metrics" in
    0|1) ;;
    *)
        echo "DIAG_SYSTEM_METRICS must be 0 or 1." >&2
        exit 1
        ;;
esac

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 environment not sourced." >&2
    exit 1
fi

if [[ "$configuration" == "D" ]]; then

    case "$communication_initiator" in
        human|robot) ;;
        *)
            echo "Invalid communication initiator." >&2
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
        configuration_description="lidar_reference"
        interaction_mode="none"
        social_supervision_enabled="false"
        announcement_enabled="false"
        dialogue_enabled="false"
        ;;

    C)
        configuration_description="social_navigation_announcement"
        interaction_mode="none"
        social_supervision_enabled="true"
        announcement_enabled="true"
        dialogue_enabled="false"
        ;;

    D)
        configuration_description="social_navigation_dialogue"
        interaction_mode="dialogue"
        social_supervision_enabled="true"
        announcement_enabled="true"
        dialogue_enabled="true"
        ;;
esac

# =============================================================================
# SCENARIO DESCRIPTION
# =============================================================================

case "$scenario" in

    S1a)
        scenario_description="standing_person_detour_available"
        scenario_goal="validated_detour_while_person_remains_present"
        ;;

    S1b)
        scenario_description="standing_person_no_admissible_detour"
        scenario_goal="remain_waiting_while_conflict_unresolved"
        ;;

    S2)
        scenario_description="transverse_crossing"
        scenario_goal="resume_only_after_confirmed_clear"
        ;;

    S3)
        scenario_description="approaching_person_controlled_stop"
        scenario_goal="detect_dynamic_conflict_and_confirm_physical_stop"
        ;;

    S4)
        scenario_description="observation_loss_during_conflict"
        scenario_goal="observation_lost_must_not_be_treated_as_clear"
        ;;

    S5)
        scenario_description="positive_verbal_response_without_clearance"
        scenario_goal="verbal_confirmation_must_not_authorize_motion_while_conflict_remains"
        ;;
esac

# =============================================================================
# EXPERIMENTAL POLICY
# =============================================================================

resume_policy="${RESUME_POLICY:-validated_detour}"
announcement_policy="${ANNOUNCEMENT_POLICY:-enabled}"

if [[ "$configuration" == "A" ]]; then
    resume_policy="reference_navigation"
    announcement_policy="disabled"
fi

# =============================================================================
# TRIAL DIRECTORY
# =============================================================================

trial_id="${trial_type}_${configuration}_${scenario}_R${trial_number}"

trial_directory="${experiment_root}/${campaign_id}/${trial_type}/${configuration}/${scenario}/${trial_id}"

bag_directory="${trial_directory}/navigation"

if [[ -e "$trial_directory" ]]; then

    printf '\nTrial directory already exists:\n'
    printf '  %s\n' "$trial_directory"
    printf 'Choose a new trial number.\n'

    exit 1
fi

mkdir -p "$trial_directory"

# =============================================================================
# TRIAL METADATA
# =============================================================================

cat > "${trial_directory}/trial_metadata.yaml" <<EOF
campaign_id: "${campaign_id}"
trial_id: "${trial_id}"
trial_type: "${trial_type}"

configuration:
  id: "${configuration}"
  description: "${configuration_description}"

system:
  interaction_mode: "${interaction_mode}"
  social_supervision_enabled: ${social_supervision_enabled}
  announcement_enabled: ${announcement_enabled}
  dialogue_enabled: ${dialogue_enabled}

experimental_policy:
  resume_policy: "${resume_policy}"
  announcement_policy: "${announcement_policy}"

scenario:
  id: "${scenario}"
  description: "${scenario_description}"
  goal: "${scenario_goal}"

trial_number: "${trial_number}"

communication:
  planned_initiator: "${communication_initiator}"

recording:
  started_utc: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ros_distribution: "${ROS_DISTRO:-unknown}"
  record_livox_raw: ${record_livox_raw}
  system_metrics: ${system_metrics}

experimental_setup:
  robot_start_pose: "TO_FILL"
  navigation_goal: "TO_FILL"
  participant_start_marker: "TO_FILL"
  participant_stop_marker: "TO_FILL"
  participant_exit_marker: "TO_FILL"
  configured_robot_speed: "TO_FILL"
  video_id: "TO_FILL"
  video_sync_marker: "TO_FILL"

outcome:
  valid_trial: "TO_FILL_AFTER_TRIAL"
  expected_behavior_observed: "TO_FILL_AFTER_TRIAL"
  intervention_required: "TO_FILL_AFTER_TRIAL"
  termination_reason: "TO_FILL_AFTER_TRIAL"
EOF

# =============================================================================
# SOURCE-CODE TRACEABILITY
# =============================================================================

if command -v git >/dev/null 2>&1; then

    git -C "$workspace_root" rev-parse HEAD \
        > "${trial_directory}/git_commit.txt" \
        2>/dev/null || true

    git -C "$workspace_root" status --short \
        > "${trial_directory}/git_status.txt" \
        2>/dev/null || true

    git -C "$workspace_root" diff \
        > "${trial_directory}/git_diff.patch" \
        2>/dev/null || true
fi

# =============================================================================
# RUNTIME TOPIC AND NODE INVENTORY
# =============================================================================

ros2 topic list -t --include-hidden-topics \
    > "${trial_directory}/available_topics.txt"

ros2 node list \
    > "${trial_directory}/available_nodes.txt" \
    2>&1 || true

# Useful when reconstructing CPU / clock / DDS conditions.
{
    date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
    date +"unix_seconds=%s.%N"
    echo "host=$(hostname)"
    echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
    echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
    echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
    uptime
    free -h
} > "${trial_directory}/host_snapshot.txt" 2>&1 || true

# =============================================================================
# COMMON NAVIGATION TOPICS
# Configurations A, C and D
# =============================================================================

topics=(

    # Goal / navigation
    /goal_pose

    # Nav2 actions
    /navigate_to_pose/_action/goal
    /navigate_to_pose/_action/cancel
    /navigate_to_pose/_action/feedback
    /navigate_to_pose/_action/status
    /navigate_to_pose/_action/result

    /navigate_to_pose_internal/_action/goal
    /navigate_to_pose_internal/_action/cancel
    /navigate_to_pose_internal/_action/feedback
    /navigate_to_pose_internal/_action/status
    /navigate_to_pose_internal/_action/result

    /follow_path/_action/goal
    /follow_path/_action/cancel
    /follow_path/_action/feedback
    /follow_path/_action/status
    /follow_path/_action/result

    # Command chain
    /cmd_vel
    /cmd_vel_nav
    /cmd_vel_nav_scaled
    /cmd_vel_mux
    /cmd_vel_rlsar

    # Localization / odometry
    /odom
    /odom_dog_calibrated
    /dog_odom
    /dog_odom_2d
    /amcl_pose

    # TF
    /tf
    /tf_static

    # Plans
    /plan
    /plan_smoothed
    /local_plan
    /received_global_plan
    /transformed_global_plan

    # Diagnostics
    /rosout
)

# =============================================================================
# NEW: SCAN / TF LATENCY DIAGNOSTICS
# =============================================================================

topics+=(

    # Scan consumed by localization or navigation
    /scan

    # Optional scan pipeline stages
    /scan_raw
    /scan_filtered

    # Simulation or ROS-time clock, if present
    /clock

    # Standard diagnostic aggregators, if present
    /diagnostics
    /diagnostics_agg
)

# =============================================================================
# SOCIAL PERCEPTION AND SOCIAL NAVIGATION
# Configurations C and D
# =============================================================================

if [[ "$configuration" == "C" ||
      "$configuration" == "D" ]]; then

    topics+=(

        /yolo/detections
        /yolo/detections_tracked
        /yolo/detections_3d
        /yolo/detections_3d_base

        /social_perception/tracked_persons
        /social_perception/social_zones
        /social_perception/social_state

        /social_navigation/navigation_goal_context
        /social_navigation/navigation_state
        /social_navigation/robot_motion_state

        /social_navigation/decision
        /social_navigation/episode_state
        /social_navigation/collaborative_decision

        /social_navigation/trajectory_generation_request
        /social_navigation/candidate_trajectory
        /social_navigation/trajectory_social_validation
    )
fi

# =============================================================================
# EXPLAINABILITY AND SPEECH
# Configurations C and D
# =============================================================================

if [[ "$configuration" == "C" ||
      "$configuration" == "D" ]]; then

    topics+=(

        /explainability/decision_request
        /explainability/context
        /explainability/text
        /explainability/speech_status

        /explainability/pc_audio_fallback/request
        /explainability/pc_audio_fallback/status
    )
fi

# =============================================================================
# COMMUNICATION / DIALOGUE / HRI
# Configuration D only
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
        /humans/voices/pc_user/speech
    )
fi

# =============================================================================
# OPTIONAL: RAW LIVOX ACQUISITION
# =============================================================================

if [[ "$record_livox_raw" == "1" ]]; then

    topics+=(
        /livox/lidar
    )

fi

# =============================================================================
# SAVE REQUESTED TOPICS
# =============================================================================

printf '%s\n' "${topics[@]}" \
    > "${trial_directory}/requested_topics.txt"

# =============================================================================
# NEW: HOST PERFORMANCE MEASUREMENTS
#
# These measurements help correlate a TF/scan interruption
# with a possible CPU or memory saturation.
#
# They run on the machine executing this script.
# =============================================================================

vmstat_pid=""
process_monitor_pid=""

cleanup() {

    if [[ -n "$vmstat_pid" ]]; then
        kill "$vmstat_pid" 2>/dev/null || true
        wait "$vmstat_pid" 2>/dev/null || true
    fi

    if [[ -n "$process_monitor_pid" ]]; then
        kill "$process_monitor_pid" 2>/dev/null || true
        wait "$process_monitor_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT

if [[ "$system_metrics" == "1" ]]; then

    if command -v vmstat >/dev/null 2>&1; then

        vmstat -t 1 \
            > "${trial_directory}/host_vmstat_1s.log" &

        vmstat_pid="$!"

    fi

    (
        while true; do

            printf '\n'
            printf '===== %s =====\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

            ps -eo \
                pid,ppid,stat,pcpu,pmem,rss,etime,comm \
                --sort=-pcpu \
                | head -n 31 || true

            sleep 2
        done
    ) > "${trial_directory}/host_processes_2s.log" \
      2>&1 &

    process_monitor_pid="$!"
fi

# =============================================================================
# RECORDING SUMMARY
# =============================================================================

printf '\n'
printf '====================================================================\n'
printf 'G1 SOCIAL NAVIGATION EXPERIMENT\n'
printf '====================================================================\n'

printf 'Trial ID          : %s\n' "$trial_id"
printf 'Trial type        : %s\n' "$trial_type"
printf '\n'

printf 'Configuration     : %s\n' "$configuration"
printf 'Description       : %s\n' "$configuration_description"
printf 'Interaction       : %s\n' "$interaction_mode"

printf 'Social supervision: %s\n' "$social_supervision_enabled"
printf 'Announcement      : %s\n' "$announcement_enabled"
printf 'Dialogue          : %s\n' "$dialogue_enabled"
printf '\n'

printf 'Scenario          : %s\n' "$scenario"
printf 'Scenario desc.    : %s\n' "$scenario_description"
printf 'Scenario goal     : %s\n' "$scenario_goal"
printf '\n'

printf 'Resume policy     : %s\n' "$resume_policy"
printf 'Announcement pol. : %s\n' "$announcement_policy"
printf '\n'

printf 'Trial number      : %s\n' "$trial_number"
printf 'Comm initiator    : %s\n' "$communication_initiator"
printf '\n'

printf 'Trial directory   : %s\n' "$trial_directory"
printf 'Bag directory     : %s\n' "$bag_directory"
printf 'Raw Livox         : %s\n' "$record_livox_raw"
printf 'System metrics    : %s\n' "$system_metrics"

printf '====================================================================\n'
printf '\n'

printf 'Topics requested: %d\n' "${#topics[@]}"
printf '\n'

# =============================================================================
# SCENARIO REMINDER
# =============================================================================

printf 'SCENARIO REMINDER:\n'

case "$scenario" in

    S1a)
        printf '  Person remains stationary in the social region.\n'
        printf '  An admissible alternative path must exist.\n'
        printf '  Expected: validated detour and resume while person remains present.\n'
        ;;

    S1b)
        printf '  Person remains stationary and blocks the admissible passage.\n'
        printf '  Expected: robot remains waiting; conflicting path must not execute.\n'
        ;;

    S2)
        printf '  Person crosses transversely and leaves the monitored region.\n'
        printf '  Expected: CLEAR must be confirmed before resume.\n'
        ;;

    S3)
        printf '  Person approaches to the predefined stop marker.\n'
        printf '  Expected: conflict during navigation and confirmed physical stop.\n'
        ;;

    S4)
        printf '  Observation is deliberately lost during an unresolved conflict.\n'
        printf '  Expected: OBSERVATION_LOST must not be interpreted as CLEAR.\n'
        ;;

    S5)
        printf '  Person responds positively but remains physically in conflict.\n'
        printf '  Expected: verbal response alone must not authorize movement.\n'
        ;;
esac

printf '\n'
printf 'IMPORTANT:\n'
printf '  1. Wait until rosbag reports that recording has started.\n'
printf '  2. Begin the authorized diagnostic observation.\n'
printf '  3. For the initial diagnostic, keep the robot immobile.\n'
printf '  4. Do not change participant behavior during the trial.\n'
printf '  5. Record through the period of social processing.\n'
printf '  6. Press Ctrl+C once to stop recording cleanly.\n'
printf '\n'

# =============================================================================
# ROSBAG RECORDING
# =============================================================================

ros2 bag record \
    --include-hidden-topics \
    -o "$bag_directory" \
    "${topics[@]}"

)


"""
cd ~/g1_ws
source install/setup.bash
source tools/source_g1_env.sh

./scripts/record_experiment.sh technical D S1a 05
bash ~/g1_ws/scripts/record_experiment.sh participant D S1a 01

"""