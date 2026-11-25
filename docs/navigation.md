# Autonomous navigation

For a mapped indoor route, start with [deployment](deployment.md). Choose the occupancy map that matches the physical environment. The workspace includes [maps/g1_map.yaml](../maps/g1_map.yaml) and other map/image pairs.

| Task | Source entry point |
| --- | --- |
| Inspect real-robot AMCL localization | [real_robot_amcl_localization.launch.py](../src/g1_nav2/launch/real_robot_amcl_localization.launch.py) |
| Inspect navigation bringup | [g1_bringup.launch.py](../src/g1_nav2/launch/g1_bringup.launch.py) |
| Build a map on the robot | [slam_mapping_real.launch.py](../src/g1_nav2/launch/slam_mapping_real.launch.py) |
| Inspect locomotion calibration | [locomotion_calibration.launch.py](../src/g1_nav2/launch/locomotion_calibration.launch.py) |

## Check a mapped-navigation session

1. Confirm scan delivery with `ros2 topic hz /scan`. Check the odometry stream selected for this run, e.g. `/dog_odom_2d` for planar DOG or `/odom_dog_calibrated` for calibrated DOG.
2. In RViz, select the map frame and set the initial pose with **2D Pose Estimate**. Check that the observed walls align with the occupancy map.
3. Check `ros2 run tf2_ros tf2_echo map base_footprint` and `ros2 action list -t`. Once localization is stable and the NavigateToPose server is available, select a reachable goal with **Nav2 Goal**.
4. Observe the global path, local obstacles and final robot pose. For a repeatable comparison, record the map, start and goal, odometry configuration, speed and trial identifier.

The project provides a [navigation recording script](../scripts/record_experiment.sh) and a [navigation analysis guide](../scripts/g1_navigation_analysis_v3/README.md).

## Calibration

For the calibration methodology covering timing, locomotion commands and odometry, see [g1_navigation_calibration](https://github.com/JeanMayoko18/g1_navigation_calibration).
