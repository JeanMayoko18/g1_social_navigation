#!/usr/bin/env bash

set -u

OUTPUT_DIR="${HOME}/g1_ws/test_logs"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${OUTPUT_DIR}/social_test_${TIMESTAMP}.txt"

TOPICS=(
    "/yolo/detections"
    "/social_perception/tracked_persons"
    "/social_perception/social_zones"
    "/social_navigation/decision"
    "/social_navigation/collaborative_decision"
    "/social_navigation/navigation_state"
    "/explainability/context"
    "/explainability/text"
    "/explainability/speech_status"
)

PIDS=()

cleanup() {
    echo
    echo "Stopping capture..."

    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done

    wait 2>/dev/null || true

    echo "Capture saved to:"
    echo "${OUTPUT_FILE}"
}

trap cleanup EXIT INT TERM

{
    echo "============================================================"
    echo "G1 SOCIAL NAVIGATION RUNTIME CAPTURE"
    echo "============================================================"
    echo "Timestamp: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo
} > "${OUTPUT_FILE}"

for topic in "${TOPICS[@]}"; do
    {
        echo
        echo "============================================================"
        echo "TOPIC: ${topic}"
        echo "============================================================"
    } >> "${OUTPUT_FILE}"

    ros2 topic echo "${topic}" \
        2>&1 \
        | sed -u "s|^|[${topic}] |" \
        >> "${OUTPUT_FILE}" &

    PIDS+=("$!")
done

echo "Recording ${#TOPICS[@]} topics."
echo "Output: ${OUTPUT_FILE}"
echo
echo "Run the navigation/person interaction now."
echo "Press Ctrl+C when the test is finished."

wait