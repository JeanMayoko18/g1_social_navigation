
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# G1 SOCIAL NAVIGATION - EXPERIMENT RECORDER
#
# Usage:
#   bash scripts/record_experiment.sh technical C S1a 01
#   bash scripts/record_experiment.sh technical D S1a 01
#   bash scripts/record_experiment.sh participant D S5 01
#
# Options:
#   DIAG_RECORD_LIVOX_RAW=0
#   DIAG_SYSTEM_METRICS=1
#   DIAG_RECORD_ROSOUT=1
#   COMMUNICATION_INITIATOR=robot
#
# No motion commands. No navigation nodes launched here.
# ============================================================

campaign_id="restaurant_social_awareness"
workspace_root="$HOME/g1_ws"
experiment_root="$workspace_root/experiments"

trial_type="${1:-}"
configuration="${2:-}"
scenario="${3:-}"
trial_number="${4:-}"

communication_initiator="${COMMUNICATION_INITIATOR:-robot}"
record_livox_raw="${DIAG_RECORD_LIVOX_RAW:-0}"
system_metrics="${DIAG_SYSTEM_METRICS:-1}"
record_rosout="${DIAG_RECORD_ROSOUT:-1}"

# ============================================================
# VALIDATION
# ============================================================

usage() {
    echo "Usage:"
    echo "  $0 <technical|participant> <A|C|D> <S1a|S1b|S2|S3|S4|S5> <number>"
}

if [[ $# -ne 4 ]]; then
    usage
    exit 1
fi

case "$trial_type" in
    technical|participant) ;;
    *) echo "Invalid trial type"; exit 1 ;;
esac

case "$configuration" in
    A|C|D) ;;
    *) echo "Invalid configuration"; exit 1 ;;
esac

case "$scenario" in
    S1a|S1b|S2|S3|S4|S5) ;;
    *) echo "Invalid scenario"; exit 1 ;;
esac

if [[ ! "$trial_number" =~ ^[0-9]+$ ]]; then
    echo "Trial number must be numeric."
    exit 1
fi

for option in \
    "$record_livox_raw" \
    "$system_metrics" \
    "$record_rosout"
do
    if [[ "$option" != 0 && "$option" != 1 ]]; then
        echo "Diagnostic switches must be 0 or 1."
        exit 1
    fi
done

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 is not sourced."
    exit 1
fi

if [[ "$configuration" != D ]]; then
    communication_initiator="none"
elif [[ "$communication_initiator" != robot &&
        "$communication_initiator" != human ]]; then
    echo "Invalid communication initiator."
    exit 1
fi

# ============================================================
# EXPERIMENT DESCRIPTION
# ============================================================

case "$configuration" in
    A)
        configuration_description="lidar_reference"
        interaction_mode="none"
        social_supervision_enabled=false
        announcement_enabled=false
        dialogue_enabled=false
        ;;
    C)
        configuration_description="social_navigation_announcement"
        interaction_mode="none"
        social_supervision_enabled=true
        announcement_enabled=true
        dialogue_enabled=false
        ;;
    D)
        configuration_description="social_navigation_dialogue"
        interaction_mode="dialogue"
        social_supervision_enabled=true
        announcement_enabled=true
        dialogue_enabled=true
        ;;
esac

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

resume_policy="${RESUME_POLICY:-validated_detour}"
announcement_policy="${ANNOUNCEMENT_POLICY:-enabled}"

if [[ "$configuration" == A ]]; then
    resume_policy="reference_navigation"
    announcement_policy="disabled"
fi

# ============================================================
# TRIAL DIRECTORY
# ============================================================

trial_id="${trial_type}_${configuration}_${scenario}_R${trial_number}"

trial_directory="${experiment_root}/${campaign_id}/${trial_type}/${configuration}/${scenario}/${trial_id}"

bag_directory="${trial_directory}/navigation"

if [[ -e "$trial_directory" ]]; then
    echo "Trial already exists:"
    echo "$trial_directory"
    echo "Choose another trial number."
    exit 1
fi

mkdir -p "$trial_directory"

# ============================================================
# METADATA
# ============================================================

cat > "$trial_directory/trial_metadata.yaml" <<EOF
campaign_id: "$campaign_id"
trial_id: "$trial_id"
trial_type: "$trial_type"

configuration:
  id: "$configuration"
  description: "$configuration_description"

system:
  interaction_mode: "$interaction_mode"
  social_supervision_enabled: $social_supervision_enabled
  announcement_enabled: $announcement_enabled
  dialogue_enabled: $dialogue_enabled

experimental_policy:
  resume_policy: "$resume_policy"
  announcement_policy: "$announcement_policy"

scenario:
  id: "$scenario"
  description: "$scenario_description"
  goal: "$scenario_goal"

trial_number: "$trial_number"

communication:
  planned_initiator: "$communication_initiator"

recording:
  started_utc: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ros_distribution: "${ROS_DISTRO:-unknown}"
  record_livox_raw: $record_livox_raw
  system_metrics: $system_metrics

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

# ============================================================
# SOURCE TRACEABILITY
# ============================================================

if command -v git >/dev/null 2>&1; then
    git -C "$workspace_root" rev-parse HEAD \
        > "$trial_directory/git_commit.txt" 2>/dev/null || true

    git -C "$workspace_root" status --short \
        > "$trial_directory/git_status.txt" 2>/dev/null || true

    git -C "$workspace_root" diff \
        > "$trial_directory/git_diff.patch" 2>/dev/null || true
fi

# ============================================================
# ROS INVENTORY
# ============================================================

ros2 topic list --include-hidden-topics \
    > "$trial_directory/available_topics.txt"

ros2 topic list -t --include-hidden-topics \
    > "$trial_directory/available_topic_types.txt"

ros2 node list \
    > "$trial_directory/available_nodes.txt" 2>&1 || true

ros2 service list -t \
    > "$trial_directory/available_services.txt" 2>&1 || true

ros2 action list -t \
    > "$trial_directory/available_actions.txt" 2>&1 || true

ros2 param list \
    > "$trial_directory/available_parameters.txt" 2>&1 || true

{
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
    echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
    echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
    uptime
    free -h
} > "$trial_directory/host_snapshot.txt" 2>&1 || true

# ============================================================
# COMMON NAVIGATION TOPICS: A / C / D
# ============================================================

topics=(
    /goal_pose

    /navigate_to_pose/_action/status
    /navigate_to_pose/_action/feedback

    /navigate_to_pose_internal/_action/status
    /navigate_to_pose_internal/_action/feedback

    /follow_path/_action/status
    /follow_path/_action/feedback

    /cmd_vel
    /cmd_vel_nav
    /cmd_vel_nav_scaled
    /cmd_vel_mux
    /cmd_vel_rlsar

    /odom
    /odom_dog_calibrated
    /dog_odom
    /dog_odom_2d
    /amcl_pose
    /initialpose

    /tf
    /tf_static

    /map
    /scan
    /scan_raw
    /scan_filtered

    /plan
    /plan_smoothed
    /local_plan
    /received_global_plan
    /transformed_global_plan

    /clock
    /diagnostics
    /diagnostics_agg
)

# ============================================================
# SOCIAL SUPERVISION: C / D
# ============================================================

if [[ "$configuration" == C ||
      "$configuration" == D ]]; then

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
        /social_navigation/episode_retirement
        /social_navigation/observation_health

        /social_navigation/collaborative_decision

        /social_navigation/trajectory_generation_request
        /social_navigation/candidate_trajectory
        /social_navigation/trajectory_social_validation

        /explainability/decision_request
        /explainability/context
        /explainability/text
        /explainability/speech_status
        /explainability/speech_cancel_request

        /explainability/pc_audio_fallback/request
        /explainability/pc_audio_fallback/status
    )
fi

# ============================================================
# DIALOGUE / HRI: D
# ============================================================

if [[ "$configuration" == D ]]; then

    topics+=(
        /hri/dialogue_request
        /hri/human_intent
        /hri/asr_listening_enabled

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

# ============================================================
# OPTIONAL TOPICS
# ============================================================

if [[ "$record_livox_raw" == 1 ]]; then
    topics+=(/livox/lidar)
fi

if [[ "$record_rosout" == 1 ]]; then
    topics+=(/rosout)
fi

printf '%s\n' "${topics[@]}" \
    > "$trial_directory/requested_topics.txt"

# ============================================================
# SELECT AVAILABLE TOPICS
#
# Avoid passing nonexistent topic names to rosbag.
# Record using available topic names at script startup.
# ============================================================

declare -A available_topic_set=()

while IFS= read -r topic; do
    [[ -n "$topic" ]] || continue
    available_topic_set["$topic"]=1
done < "$trial_directory/available_topics.txt"

record_topics=()
missing_topics=()

for topic in "${topics[@]}"; do
    if [[ -n "${available_topic_set[$topic]:-}" ]]; then
        record_topics+=("$topic")
    else
        missing_topics+=("$topic")
    fi
done

printf '%s\n' "${record_topics[@]}" \
    > "$trial_directory/recorded_topics_requested.txt"

printf '%s\n' "${missing_topics[@]}" \
    > "$trial_directory/missing_topics_at_start.txt"

if [[ ${#record_topics[@]} -eq 0 ]]; then
    echo "ERROR: no requested ROS topics available."
    exit 1
fi

# Required diagnostic interfaces for C / D.
# Warn only: do not invent missing topics or nodes.

if [[ "$configuration" == C ||
      "$configuration" == D ]]; then

    required_topics=(
        /social_navigation/navigation_goal_context
        /social_navigation/navigation_state
        /social_navigation/collaborative_decision
        /social_navigation/decision
        /social_navigation/trajectory_generation_request
        /social_navigation/candidate_trajectory
        /social_navigation/trajectory_social_validation
        /explainability/speech_status
    )

    for topic in "${required_topics[@]}"; do
        if [[ -z "${available_topic_set[$topic]:-}" ]]; then
            echo "WARNING: diagnostic topic missing: $topic"
        fi
    done
fi

# ============================================================
# SYSTEM METRICS
# ============================================================

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

if [[ "$system_metrics" == 1 ]]; then

    if command -v vmstat >/dev/null 2>&1; then
        vmstat -t 1 \
            > "$trial_directory/host_vmstat_1s.log" &
        vmstat_pid=$!
    fi

    (
        while true; do
            echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
            ps -eo pid,ppid,stat,pcpu,pmem,rss,etime,comm \
                --sort=-pcpu | head -n 31 || true
            sleep 2
        done
    ) > "$trial_directory/host_processes_2s.log" 2>&1 &

    process_monitor_pid=$!
fi

# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

echo
echo "============================================================"
echo "G1 SOCIAL NAVIGATION EXPERIMENT"
echo "============================================================"
echo "Trial         : $trial_id"
echo "Configuration : $configuration"
echo "Interaction   : $interaction_mode"
echo "Scenario      : $scenario"
echo "Initiator     : $communication_initiator"
echo "------------------------------------------------------------"
echo "Directory     : $trial_directory"
echo "Bag           : $bag_directory"
echo "Topics active : ${#record_topics[@]}"
echo "Topics absent : ${#missing_topics[@]}"
echo "Raw Livox     : $record_livox_raw"
echo "System metrics: $system_metrics"
echo "============================================================"
echo
echo "Important:"
echo "  - Start recording before sending the RViz goal."
echo "  - Missing topics are listed in missing_topics_at_start.txt."
echo "  - Only topics available at startup are recorded."
echo "  - Ctrl+C stops the recorder."
echo

# ============================================================
# ROSBAG
# ============================================================

ros2 bag record \
    --include-hidden-topics \
    -o "$bag_directory" \
    "${record_topics[@]}"



"""
cd ~/g1_ws
source install/setup.bash
source tools/source_g1_env.sh

./scripts/record_experiment_old.sh technical D S1a 03

bash scripts/record_experiment.sh participant D S1a 05

bash ~/g1_ws/scripts/record_experiment.sh participant D S1a 01

"""