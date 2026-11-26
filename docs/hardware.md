# Shared hardware for mapping and navigation

Use a built workspace on each participating host. For a wired PC, complete the [Ethernet check](network.md). In **each new terminal**, source the [host environment](environment.md). On the PC use Humble; on the robot use Foxy. The examples below place hardware on the G1 and navigation on the PC.

**Robot terminal 1 · locomotion.** Use the wired interface for this host (the project G1 uses `eth0`). For the Unitree LOCO backend:
```bash
ros2 run rl_sar rl_real_g1_edu23_unitree_loco eth0 --real --control-backend=unitree_loco --max-motion-duration=0.0
```
Press **0** for Bring Up, wait for the ready posture, then press **1** for Enter Locomotion. Keep this terminal running. For a custom policy or PC-hosted locomotion, use the [locomotion guide](locomotion.md).

**Robot terminal 2 · shared hardware.** Start one hardware instance. Edit the three selections at the top before running. Use `camera=false` for mapping or ordinary navigation and `camera=true` for a social trial. For Livox, source the installed driver overlay first (on the project robot: `source ~/ros2_ws/install/setup.bash`, then `source ~/g1_ws/install/setup.bash`).
```bash
scan=livox          # livox or unitree
odom=dogc           # see the source table below
camera=false        # false for A; true for C or D
livox_driver=false
if [ "$scan" = livox ]; then livox_driver=true; fi
dog_calibration=false
if [ "$odom" = dogc ]; then dog_calibration=true; fi

ros2 launch rl_sar hardware.launch.py \
  run_robot_core:=true run_lidar_pipeline:=true \
  lidar_source:="$scan" enable_livox_driver:="$livox_driver" \
  scan_target_frame:=base_footprint scan_min_height:=-0.53 scan_max_height:=0.72 \
  odom_source:="$odom" enable_dog_odom_calibration:="$dog_calibration" \
  run_realsense_pipeline:="$camera" enable_realsense:="$camera" enable_realsense_tf:="$camera"
```
If a separate driver already supplies the Livox cloud, set `livox_driver=false` before executing the launch. The Livox height limits are in the project G1’s planar frame at pelvis height. Check these limits for your robot. Changing from A to C/D requires RGB-D: stop and restart terminal 2 with `camera=true`, or use the separate camera procedure in [Hardware](hardware.md). Keep the same hardware running when switching between C and D.

| Odometry choice | Selector input | What the robot must supply |
| --- | --- | --- |
| `dog` | `/dog_odom_2d` | Baseline planar DOG odometry. |
| `dogc` | `/odom_dog_calibrated` | Calibrated DOG. This is the project's tested longitudinal correction. |
| `unitree` | `/odom_unitree` | Unitree mode-state odometry. Observed with LOCO; verify publication with your backend. |
| `aocf` | `/odom_aocf` | Start the configured adaptive DOG producer first. |
| `mola` | `/odom_mola_nav` | Start configured MOLA and its navigation adapter first. |
| `dogekf` | `/odom_dog_ekf` | Requires a running DOG EKF. Listed by the selector; no trial recipe supplied. |

In the inspected robot core, automatic startup of the AOCF, MOLA adapter and DOG EKF nodes is commented out. Selecting one routes its input; confirm the producer delivers samples. Either scan choice feeds `/scan`. All odometry choices feed the same `/odom` and `odom → base_footprint` TF.

**PC · verify the hardware before continuing.** Run each streaming check separately; press Ctrl+C after observing samples. For C/D also check color and aligned depth.
```bash
ros2 topic hz /scan
ros2 topic hz /odom
ros2 run tf2_ros tf2_echo odom base_footprint
```


## Alternative sensor bring-up



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

## Odometry notes

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
