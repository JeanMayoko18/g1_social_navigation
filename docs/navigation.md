# Mapping and autonomous navigation

Start [hardware and sensors](hardware.md), then use the same [system entry point](../src/g1_social_bringup/launch/system.launch.py) for both tasks. Select the task with `navigation_mode`. The examples use profile A and run navigation on the PC. For onboard navigation, use `deployment:=robot_all host:=robot ros_distro:=foxy`. See [deployment](deployment.md) for role placement and RViz.

## Build a map with SLAM Toolbox

On the navigation host, confirm scan delivery with `ros2 topic hz /scan` and odometry TF with `ros2 run tf2_ros tf2_echo odom base_footprint`. On the PC, start mapping:

```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=A interaction:=none \
  deployment:=pc_perception host:=pc ros_distro:=humble \
  navigation_mode:=slam
```

The system includes [navigation.launch.py](../src/g1_social_bringup/launch/navigation.launch.py) on the host that owns navigation. It forwards the mode to [g1_bringup.launch.py](../src/g1_nav2/launch/g1_bringup.launch.py), which selects [slam.launch.py](../src/g1_nav2/launch/slam.launch.py). This child starts `slam_toolbox/async_slam_toolbox_node` in mapping mode. It consumes `/scan`, uses the frames `map`, `odom` and `base_footprint` by default, and builds a grid with 0.05 m cells.

In RViz, set the fixed frame to `map` and display `/map`. As the robot explores, check that the occupancy grid grows with the observed surroundings. To save it while SLAM is running, use a second sourced terminal and choose a fresh map name:

```bash
ros2 run nav2_map_server map_saver_cli -f "$HOME/g1_ws/maps/g1_trial_01"
```

## Navigate with a saved map

End the mapping launch before switching modes. On the host running Nav2, supply the YAML for the saved occupancy grid:

```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=A interaction:=none \
  deployment:=pc_perception host:=pc ros_distro:=humble \
  navigation_mode:=nav2 map:="$HOME/g1_ws/maps/g1_trial_01.yaml"
```

## Check a mapped-navigation session

1. Confirm scan delivery with `ros2 topic hz /scan`. Check the odometry stream selected for this run, e.g. `/dog_odom_2d` for planar DOG or `/odom_dog_calibrated` for calibrated DOG.
2. In RViz, select the map frame and set the initial pose with **2D Pose Estimate**. Check that the observed walls align with the occupancy map.
3. Check `ros2 run tf2_ros tf2_echo map base_footprint` and `ros2 action list -t`. Once localization is stable and the NavigateToPose server is available, select a reachable goal with **Nav2 Goal**.
4. Observe the global path, local obstacles and final robot pose. For a repeatable comparison, record the map, start and goal, odometry configuration, speed and trial identifier.

The project provides a [navigation recording script](../scripts/record_experiment.sh) and a [navigation analysis guide](../scripts/g1_navigation_analysis_v3/README.md).

## Calibration

For the calibration methodology covering timing, locomotion commands and odometry, see [g1_navigation_calibration](https://github.com/JeanMayoko18/g1_navigation_calibration).
