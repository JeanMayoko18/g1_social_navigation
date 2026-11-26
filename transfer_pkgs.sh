#!/usr/bin/env bash

ROBOT_HOST="unitree@192.168.123.164"
ROBOT_WORKSPACE="/home/unitree/g1_ws"

cd ~/g1_ws || exit 1

echo "[CONFIG] Robot:     $ROBOT_HOST"
echo "[CONFIG] Workspace: $ROBOT_WORKSPACE"

ssh "$ROBOT_HOST" "mkdir -p '$ROBOT_WORKSPACE/src'" || {
    echo "[ERROR] SSH connection failed"
    exit 1
}

cd ~/g1_ws || exit 1

packages=(
    g1_qos_manager
    g1_social_costmap
    g1_nav2
    g1_social_perception
    g1_social_common
    g1_social_bringup
    g1_robot_explainability
    g1_robot_collaboration
    g1_social_msgs
    g1_hri
    g1_pc_asr
    rl_sar
)

echo "===== START $(date) ====="

for package_name in "${packages[@]}"; do
    echo
    echo "===== START $package_name ====="

    rsync \
        -ah \
        --stats \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "src/$package_name/" \
        "$ROBOT_HOST:$ROBOT_WORKSPACE/src/$package_name/"

    status=$?

    echo "===== END $package_name : rsync=$status ====="

    if (( status != 0 )); then
        echo "[ERROR] $package_name failed"
        exit "$status"
    fi
done

echo
echo "===== ALL TRANSFERS FINISHED ====="
echo "===== END $(date) ====="

exit 0

# bash ~/g1_ws/transfer_test.sh > ~/transfer_debug.log 2>&1
#export ROBOT_HOST="unitree@192.168.123.164"
#export ROBOT_WORKSPACE="/home/unitree/g1_ws"

#bash ~/g1_ws/transfer_test.sh