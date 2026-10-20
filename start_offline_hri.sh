#!/usr/bin/env bash

set -eo pipefail

# ============================================================
# OFFLINE HRI + EXPLAINABILITY + TTS
#
# ROS 2 Humble
# CycloneDDS / isolated localhost domain
#
# No Nav2 or robot motion backend is launched here.
# ============================================================

# ============================================================
# 1. Paths
# ============================================================

G1_WS="$HOME/g1_ws"
ROS4HRI_WS="$HOME/ros4hri_ws"

LOG_DIR="$G1_WS/log/offline_hri"
PROVIDER_LOG="$G1_WS/log/explainability/provider_chain.jsonl"
OFFLINE_ENV="$G1_WS/offline_ros_env.sh"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PROVIDER_LOG")"

export PYTHONUNBUFFERED=1

# ============================================================
# 2. Exclusive launcher lock
# ============================================================

LOCK_FILE="$LOG_DIR/launcher.lock"

exec 9>"$LOCK_FILE"

if ! flock -n 9; then
    echo "ERROR: An offline HRI launcher is already active."
    exit 1
fi

# ============================================================
# 3. Process tracking and cleanup
# ============================================================

PIDS=()
CLEANUP_DONE=false

group_alive() {
    local pgid="$1"

    kill -0 -- "-$pgid" 2>/dev/null
}

signal_groups() {
    local signal="$1"
    local pid

    for pid in "${PIDS[@]}"; do
        if group_alive "$pid"; then
            kill "-$signal" -- "-$pid" 2>/dev/null || true
        fi
    done
}

cleanup() {
    if [[ "$CLEANUP_DONE" == true ]]; then
        return
    fi

    CLEANUP_DONE=true

    trap - EXIT INT TERM

    echo
    echo "=================================================="
    echo " STOPPING OFFLINE HRI CHAIN"
    echo "=================================================="

    # First request a graceful ROS shutdown.
    signal_groups INT

    sleep 2

    # Terminate groups still running.
    signal_groups TERM

    sleep 2

    # Last resort: eliminate remaining offline children.
    signal_groups KILL

    # Reap direct launcher children.
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo "Offline process groups cleaned up."
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ============================================================
# 4. Load provider configuration
# ============================================================

echo "=================================================="
echo " OFFLINE HRI + EXPLAINABILITY + TTS"
echo "=================================================="

echo
echo "[PRECHECK] Provider configuration"

if [[ -f "$G1_WS/.env" ]]; then
    set -a
    source "$G1_WS/.env"
    set +a

    echo "  OK: .env loaded"
else
    echo "  WARNING: .env not found"
fi

if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    echo "  OK: GEMINI_API_KEY available"
else
    echo "  WARNING: GEMINI_API_KEY missing"
fi

# ============================================================
# 5. Offline ROS 2 environment
# ============================================================

echo
echo "[PRECHECK] Offline ROS 2 environment"

if [[ ! -f "$OFFLINE_ENV" ]]; then
    echo "ERROR: Missing $OFFLINE_ENV"
    exit 1
fi

source "$OFFLINE_ENV"

if [[ "${ROS_DOMAIN_ID:-}" != "91" ]]; then
    echo "ERROR: Unexpected ROS_DOMAIN_ID"
    exit 1
fi

if [[ -n "${ROS_LOCALHOST_ONLY:-}" ]]; then
    echo "ERROR: ROS_LOCALHOST_ONLY must be unset."
    exit 1
fi

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
    echo "ERROR: CycloneDDS RMW is not selected."
    exit 1
fi

if [[ -z "${CYCLONEDDS_URI:-}" ]]; then
    echo "ERROR: CYCLONEDDS_URI is missing."
    exit 1
fi

echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  ROS_LOCALHOST_ONLY=unset"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  CYCLONEDDS_URI=configured"

# ============================================================
# 6. Required tools and package paths
# ============================================================

echo
echo "[PRECHECK] Process isolation tools"

for cmd in setsid flock ps timeout ros2; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: Missing command: $cmd"
        exit 1
    fi
done

echo "  OK: Required commands available"

# Verify that ROS4HRI is the authoritative provider.
DIALOGUE_PREFIX="$(ros2 pkg prefix dialogue_manager)"

if [[ "$DIALOGUE_PREFIX" != "$ROS4HRI_WS/install/dialogue_manager" ]]; then
    echo "ERROR: Wrong Dialogue Manager selected."
    echo "  Actual:   $DIALOGUE_PREFIX"
    echo "  Expected: $ROS4HRI_WS/install/dialogue_manager"
    exit 1
fi

echo "  OK: Dialogue Manager from ros4hri_ws"

# Direct executables allow precise process supervision.

EXPLANATION_BIN="$G1_WS/install/g1_robot_explainability/lib/g1_robot_explainability/explanation_node"

TTS_BIN="$G1_WS/install/g1_robot_explainability/lib/g1_robot_explainability/piper_tts"

HRI_BIN="$G1_WS/install/g1_hri/lib/g1_hri/hri_manager_node"

COLLAB_BIN="$G1_WS/install/g1_robot_collaboration/lib/g1_robot_collaboration/collaboration_manager_node"

ASR_BIN="$G1_WS/install/g1_pc_asr/lib/g1_pc_asr/pc_asr"

for executable in \
    "$EXPLANATION_BIN" \
    "$TTS_BIN" \
    "$HRI_BIN" \
    "$COLLAB_BIN" \
    "$ASR_BIN"
do
    if [[ ! -x "$executable" ]]; then
        echo "ERROR: Executable missing:"
        echo "  $executable"
        exit 1
    fi
done

# ============================================================
# 7. DDS graph and orphan protection
# ============================================================

echo
echo "[PRECHECK] ROS 2 DDS discovery"

GRAPH_LOG="$LOG_DIR/dds_precheck.log"

if ! timeout 15s ros2 node list --no-daemon \
    > "$GRAPH_LOG" 2>&1; then

    echo "ERROR: ROS 2 DDS discovery failed."
    cat "$GRAPH_LOG"
    exit 1
fi

echo "  OK: ROS 2 graph query succeeded"

echo
echo "[PRECHECK] Existing offline nodes"

EXISTING_NODES="$(
    grep -E '^/(dialogue_manager|explanation_node|piper_tts|hri_manager|collaboration_manager|g1_pc_asr)$' \
        "$GRAPH_LOG" || true
)"

if [[ -n "$EXISTING_NODES" ]]; then
    echo "ERROR: Existing HRI nodes detected."
    echo
    echo "$EXISTING_NODES"
    echo
    echo "Stop the previous offline processes before launching."
    exit 1
fi

echo "  OK: No conflicting HRI nodes"

# ============================================================
# 8. Startup helpers
# ============================================================

wait_for_log_pattern() {
    local label="$1"
    local pid="$2"
    local logfile="$3"
    local pattern="$4"
    local timeout_sec="${5:-30}"

    local start_sec
    start_sec="$(date +%s)"

    echo "Waiting for $label..."

    while true; do

        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $label exited."
            tail -n 80 "$logfile" 2>/dev/null || true
            return 1
        fi

        if grep -Fq "$pattern" "$logfile" 2>/dev/null; then
            echo "  OK: $label"
            return 0
        fi

        if (( $(date +%s) - start_sec >= timeout_sec )); then
            echo "ERROR: $label initialization timeout."
            tail -n 80 "$logfile" 2>/dev/null || true
            return 1
        fi

        sleep 0.5
    done
}

start_component() {
    local label="$1"
    local logfile="$2"
    local ready_pattern="$3"
    local startup_timeout="$4"

    shift 4

    echo
    echo "Starting $label"

    # Each component runs in a separate session/process group.
    # fd 9 is closed in children so only this launcher owns
    # the exclusive flock.

    setsid -- "$@" 9>&- > "$logfile" 2>&1 &

    local pid=$!

    PIDS+=("$pid")

    # Verify that the tracked PID is the group leader.
    local pgid

    pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"

    if [[ "$pgid" != "$pid" ]]; then
        echo "ERROR: $label process group mismatch."
        echo "  PID=$pid PGID=$pgid"
        return 1
    fi

    echo "  PID=$pid PGID=$pgid"

    wait_for_log_pattern \
        "$label" \
        "$pid" \
        "$logfile" \
        "$ready_pattern" \
        "$startup_timeout"
}

# ============================================================
# 9. Initialize logs
# ============================================================

echo
echo "[PRECHECK] Offline logs"

: > "$LOG_DIR/dialogue_manager.log"
: > "$LOG_DIR/explanation_node.log"
: > "$LOG_DIR/tts.log"
: > "$LOG_DIR/hri_manager.log"
: > "$LOG_DIR/collaboration_manager.log"
: > "$LOG_DIR/pc_asr.log"

touch "$PROVIDER_LOG"

echo "  OK: Logs initialized"

# ============================================================
# 10. Dialogue Manager
# ============================================================

echo
echo "[1/6] Dialogue Manager"

# Keep the official launch file and its parameters.
# The entire launch process tree belongs to one group.

start_component \
    "Dialogue Manager" \
    "$LOG_DIR/dialogue_manager.log" \
    "Dialogue Manager activated" \
    30 \
    ros2 launch \
        dialogue_manager \
        dialogue_manager.launch.py

# ============================================================
# 11. Explanation Node
# ============================================================

echo
echo "[2/6] Explanation Node"

start_component \
    "Explanation Node" \
    "$LOG_DIR/explanation_node.log" \
    "ROBOT EXPLANATION - INITIALIZATION" \
    30 \
    "$EXPLANATION_BIN" \
        --ros-args \
        -p provider:=gemini \
        -p fallback_enabled:=true \
        -p provider_diagnostics_enabled:=true \
        -p default_language:=en

# ============================================================
# 12. Piper TTS
# ============================================================

echo
echo "[3/6] Piper TTS"

start_component \
    "Piper TTS" \
    "$LOG_DIR/tts.log" \
    "COMMON SPEECH MANAGER - PIPER" \
    30 \
    "$TTS_BIN" \
        --ros-args \
        -p ros_distro:=humble \
        -p audio_backend:=aplay

# ============================================================
# 13. HRI Manager
# ============================================================

echo
echo "[4/6] HRI Manager"

start_component \
    "HRI Manager" \
    "$LOG_DIR/hri_manager.log" \
    "HRI Manager started" \
    30 \
    "$HRI_BIN"

# ============================================================
# 14. Collaboration Manager
# ============================================================

echo
echo "[5/6] Collaboration Manager"

start_component \
    "Collaboration Manager" \
    "$LOG_DIR/collaboration_manager.log" \
    "Speech completion barrier" \
    30 \
    "$COLLAB_BIN" \
        --ros-args \
        -p interaction:=dialogue \
        -p language:=en \
        -p dialogue_response_timeout_sec:=10.0

# ============================================================
# 15. PC ASR
# ============================================================

echo
echo "[6/6] PC ASR"

start_component \
    "PC ASR" \
    "$LOG_DIR/pc_asr.log" \
    "PC ASR ready" \
    60 \
    "$ASR_BIN" \
        --ros-args \
        -p voice_id:=pc_user \
        -p language:=en \
        -p model:=small \
        -p device:=cpu \
        -p compute_type:=int8

# ============================================================
# 16. Ready
# ============================================================

echo
echo "=================================================="
echo " OFFLINE HRI CHAIN READY"
echo "=================================================="

echo
echo "Running components:"
echo "  OK: Dialogue Manager"
echo "  OK: Explanation Node"
echo "  OK: Piper TTS"
echo "  OK: HRI Manager"
echo "  OK: Collaboration Manager"
echo "  OK: PC ASR"

echo
echo "Important topics:"

ros2 topic list --no-daemon 2>/dev/null | grep -E \
    '^/intents$|^/hri/human_intent$|^/hri/dialogue_request$|^/explainability/context$|^/explainability/text$|^/explainability/speech_status$|^/humans/voices/tracked$|^/humans/voices/pc_user/speech$' \
    || true

echo
echo "Injection:"
echo "  bash $G1_WS/inject_offline_social.sh"

echo
echo "Press Ctrl+C to stop all offline components."

# ============================================================
# 17. Monitor
# ============================================================

while true; do
    sleep 1

    for pid in "${PIDS[@]}"; do

        if ! kill -0 "$pid" 2>/dev/null; then

            echo
            echo "ERROR: An offline component stopped."
            echo "PID=$pid"

            tail -n 30 \
                "$LOG_DIR/dialogue_manager.log" \
                "$LOG_DIR/explanation_node.log" \
                "$LOG_DIR/tts.log" \
                "$LOG_DIR/hri_manager.log" \
                "$LOG_DIR/collaboration_manager.log" \
                "$LOG_DIR/pc_asr.log"

            exit 1
        fi
    done
done

"""
chmod +x ~/g1_ws/start_offline_hri.sh
cd ~/g1_ws
source /opt/ros/humble/setup.bash
source ~/ros4hri_ws/install/setup.bash
source ~/g1_ws/install/setup.bash

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI:-unset}"

ros2 daemon stop
sleep 2
ros2 daemon start
sleep 2

ros2 node list
bash start_offline_hri.sh

tail -f \
  ~/g1_ws/log/offline_hri/hri_manager.log \
  ~/g1_ws/log/offline_hri/collaboration_manager.log

tail -n 100 ~/g1_ws/log/offline_hri/hri_manager.log
"""
