# Start hardware and sensors

Prepare the [environment](environment.md) on the host running each command. The deployment examples place hardware on the G1. For a hardware component hosted on the PC, use Humble, the PC’s wired interface and its local sensor/driver access. The hardware launcher defaults to `network_interface:=eth0`.

## Option 1 · integrated bring-up

```bash
ros2 launch rl_sar hardware.launch.py \
  run_robot_core:=true \
  run_lidar_pipeline:=true \
  run_realsense_pipeline:=true \
  enable_realsense:=true \
  enable_realsense_tf:=true \
  odom_source:=dogc \
  enable_dog_odom_calibration:=true
```

This command uses the launch default `lidar_source:=unitree`. For an integrated Livox MID-360 run, append the following arguments to the same launch command. Source the Livox driver workspace first as shown below.

```text
lidar_source:=livox \
  enable_livox_driver:=true \
  scan_target_frame:=base_footprint \
  scan_min_height:=-0.53 \
  scan_max_height:=0.72
```

If a separate Livox driver already publishes the cloud and IMU, use `enable_livox_driver:=false`.

## Option 2 · separate terminals

Prepare each terminal independently. For external sensor drivers on the project robot, add their overlay before re-sourcing the G1 overlay:

```bash
source ~/ros2_ws/install/setup.bash
source ~/g1_ws/install/setup.bash
```

**1. Livox driver.** Start this terminal only when Livox supplies the scan.
```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

**2. LiDAR pipeline.** With the external Livox driver running:
```bash
ros2 launch rl_sar lidar_pipeline.launch.py \
  lidar_source:=livox \
  enable_livox_driver:=false \
  run_mola_lidar_odometry:=false \
  scan_target_frame:=base_footprint \
  scan_min_height:=-0.53 \
  scan_max_height:=0.72
```

The Livox conversion keeps cloud points between −0.53 m and +0.72 m in the selected frame. This G1 uses a planar `base_footprint` at pelvis height to avoid projecting scans in a frame that follows pelvis roll and pitch. Recheck the height interval when changing the frame origin or sensor mount. For the internal Unitree source, skip terminal 1 and launch:
```bash
ros2 launch rl_sar lidar_pipeline.launch.py \
  lidar_source:=unitree \
  enable_livox_driver:=false \
  run_mola_lidar_odometry:=false
```

**3. RealSense RGB and aligned depth.** Use this terminal if the camera driver is started independently. When the perception pipeline starts the driver, use that single owner.
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  enable_sync:=false
```

**4. Robot core and camera mounting TF.** Use this example for the Livox setup above. For the Unitree setup, change `lidar_source` to `unitree`.
```bash
ros2 launch rl_sar hardware.launch.py \
  run_robot_core:=true \
  run_lidar_pipeline:=false \
  run_realsense_pipeline:=false \
  enable_realsense:=false \
  enable_realsense_tf:=true \
  lidar_source:=livox \
  odom_source:=dogc \
  enable_dog_odom_calibration:=true
```

The parent includes the RealSense child when either the pipeline or the mounting TF is enabled. The camera driver requires both `run_realsense_pipeline` and `enable_realsense`. This split keeps the mounting TF while the external driver owns acquisition.

## Choose odometry

| `odom_source` | Source and scope |
| --- | --- |
| `dog` | Baseline Unitree DOG odometry. |
| `dogc` | Calibrated DOG used in the reported trials. The current correction addresses underestimated longitudinal velocity, Vx. |
| `aocf` | Adaptive Odom Calibration Framework applied to DOG, covering Vx, Vy and yaw rate Wz. |
| `unitree` | Uses `rt/odommodestate`. Tested with LOCO. The project observed no publication on this source with its custom policy. |
| `mola` | Requires a configured, running MOLA odometry source. To start MOLA in the LiDAR pipeline, inspect `run_mola_lidar_odometry` and `mola_config_file`. |
| `dogekf` | Listed in the launch argument description. Setup and trial results are undocumented. |

The parent forwards the source selection to `robot_core.launch.py`. The exact output topics and TF ownership depend on the child configuration. Before starting navigation, check scan delivery (`ros2 topic hz /scan`), odometry (`ros2 topic list -t`), the transform (`ros2 run tf2_ros tf2_echo odom base_footprint`) and the color/aligned-depth streams in the active camera namespace.


For the calibration procedure behind the odometry and locomotion corrections, see [g1_navigation_calibration](https://github.com/JeanMayoko18/g1_navigation_calibration).
