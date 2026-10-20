#!/usr/bin/env bash

source /opt/ros/humble/setup.bash
source "$HOME/g1_ws/install/setup.bash"

set -u

cleanup() {
    echo
    echo "Stopping test processes..."
    jobs -pr | xargs -r kill
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Starting person_depth_fusion ==="
ros2 run g1_social_perception person_depth_fusion \
    --ros-args \
    -p depth_topic:=/perception/ingress/aligned_depth/image_raw \
    -p camera_info_topic:=/perception/ingress/aligned_depth/camera_info &
sleep 5

echo "=== Adding person_tracking ==="
ros2 run g1_social_perception person_tracking &
sleep 5

echo "=== Adding person_frame_transform ==="
ros2 run g1_social_perception person_frame_transform &
sleep 5

echo "=== Adding person_spatial_tracking ==="
ros2 run g1_social_perception person_spatial_tracking &
sleep 5

echo "=== Adding social_zone_generator ==="
ros2 run g1_social_perception social_zone_generator &
sleep 10

echo "=== Test complete ==="
