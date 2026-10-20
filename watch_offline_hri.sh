#!/usr/bin/env bash

source ~/g1_ws/offline_ros_env.sh
set -e

LOG_DIR="$HOME/g1_ws/log/offline_hri"
PROVIDER_LOG="$HOME/g1_ws/log/explainability/provider_chain.jsonl"

echo "=================================================="
echo " OFFLINE HRI LOG MONITOR"
echo "=================================================="
echo
echo "Watching:"
echo "  Explanation Node"
echo "  Piper TTS"
echo "  HRI Manager"
echo "  Collaboration Manager"
echo "  Provider diagnostics"
echo
echo "Press Ctrl+C to stop monitoring."
echo "=================================================="
echo

touch \
  "$LOG_DIR/explanation_node.log" \
  "$LOG_DIR/tts.log" \
  "$LOG_DIR/hri_manager.log" \
  "$LOG_DIR/collaboration_manager.log" \
  "$PROVIDER_LOG"

tail -n 20 -F \
  "$LOG_DIR/explanation_node.log" \
  "$LOG_DIR/tts.log" \
  "$LOG_DIR/hri_manager.log" \
  "$LOG_DIR/collaboration_manager.log" \
  "$PROVIDER_LOG"

"""
chmod +x ~/g1_ws/watch_offline_hri.sh
bash ~/g1_ws/watch_offline_hri.sh
"""