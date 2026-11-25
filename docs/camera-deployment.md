# Choose a G1 launch configuration

Use `g1_social_bringup system.launch.py` as the entry point for mapping,
navigation and social navigation. Start locomotion and hardware first.
Unitree LOCO and the custom RL policy can each run on either the PC or the robot.

## 1. Choose the task

| Task | System arguments | What to start or check |
| --- | --- | --- |
| Create a map | `profile:=A navigation_mode:=slam interaction:=none` | Start the LiDAR, odometry and TF. Disable RealSense, YOLO and visual odometry. Save the map before stopping SLAM. |
| Navigate on a saved map | `profile:=A navigation_mode:=nav2 interaction:=none` | Use the same LiDAR-only hardware. Supply the map YAML and set the initial pose. |
| Navigate with camera obstacles | `profile:=B navigation_mode:=nav2 interaction:=none` | Start the camera obstacle source and verify that the Nav2 costmap receives its observation topic. The current launcher enables YOLO inference only in C/D. |
| Run social navigation | `profile:=C navigation_mode:=nav2` | Start RGB, aligned depth, CameraInfo and the detection source. Choose `interaction:=none` or `dialogue`. |
| Run social navigation with camera obstacles | `profile:=D navigation_mode:=nav2` | Reuse the C hardware and social perception chain. Also verify the camera obstacle input to Nav2. Choose `interaction:=none` or `dialogue`. |

Choose the scan and odometry sources independently of the software deployment.
For `lidar_source:=livox`, start one Livox driver. For `lidar_source:=unitree`,
use the internal cloud. Select `odom_source:=dog`, `dogc`, `aocf`, `unitree`
or `mola` according to the installed hardware configuration. See
[hardware and odometry](hardware.md), [mapping and navigation](navigation.md)
and [social navigation](social-navigation.md) for the commands.

## 2. Choose where the camera and software run

`deployment` assigns software roles. `camera_host` identifies the machine
running the RealSense driver. `host` identifies the terminal executing this
instance of `system.launch.py`. Start the camera driver exactly once on
`camera_host`.

The table covers C/D with compressed RGB and YOLO enabled on its assigned host.

| Deployment | RealSense driver | YOLO | Depth fusion | Where to run `system.launch.py` and why |
| --- | --- | --- | --- | --- |
| `pc_all` | PC | PC | PC | PC only. Camera data stays local. |
| `pc_all` | Robot | PC | PC | Robot for the RGB sender, PC for the receiver and processing. |
| `robot_all` | Robot | Robot | Robot | Robot for processing. A PC launch is optional for visualization. |
| `robot_all` | PC | Robot | Robot | PC for the RGB sender, robot for the receiver and processing. |
| `pc_perception` | PC | PC | PC | PC for processing, robot for speech. RGB stays local on the PC. |
| `pc_perception` | Robot | PC | PC | Robot for acquisition-side RGB transport and speech, PC for perception and navigation. |
| `robot_perception` | Robot | PC | Robot | Robot for fusion and speech, PC for YOLO and navigation. RGB crosses to the PC. Depth stays on the robot. |
| `robot_perception` | PC | PC | Robot | PC for YOLO and navigation, robot for fusion and speech. Depth and CameraInfo cross to the robot, along with RGB for consumers that require it. |

**RGB transport.** `camera_transport:=raw` uses uncompressed remote RGB via
`/perception/ingress/color/image_raw`. `camera_transport:=compressed` uses
an encoder on the camera host and exposes decoded RGB to remote consumers
on `/perception/transport/color`. Only RGB is compressed by this bridge.
Remote fusion uses `/perception/ingress/aligned_depth/image_raw` and
`/perception/ingress/aligned_depth/camera_info`. Local consumers use the
original RealSense topics. The B/D camera obstacle observation source is a
separate Nav2 input and should be checked independently.

**Dialogue.** C and D accept `interaction:=none` and `interaction:=dialogue`.
The host assigned the dialogue role requires ROS 2 Humble. On a Foxy G1,
choose `pc_all`, `pc_perception` or `robot_perception` for dialogue.
With `robot_all` on Foxy, use `interaction:=none`.

## 3. Launch the selected row

Complete the [host environment](environment.md) setup in each terminal.
For Ethernet, use the [wired network setup](network.md), the same
`ROS_DOMAIN_ID` on both hosts and `ROS_LOCALHOST_ONLY=0`.
The map YAML must exist on the machine running Nav2.

For a C/D run, set these values in every participating system-launch terminal.
This example uses a robot-mounted RealSense and PC perception:

```bash
profile=C
interaction=none
deployment=pc_perception
camera_host=robot
transport=compressed
```

Choose the terminal identity on each participating machine:

```bash
# PC terminal
host=pc
distro=humble
```

```bash
# Robot terminal
host=robot
distro=foxy
```

Run this command on each host listed in the selected row:

```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:="$profile" interaction:="$interaction" \
  deployment:="$deployment" host:="$host" ros_distro:="$distro" \
  camera_host:="$camera_host" camera_transport:="$transport" \
  navigation_mode:=nav2 map:="$HOME/g1_ws/maps/mic.yaml" \
  enable_yolo:=true start_visual_odometry:=false
```

For mapping and LiDAR-only navigation, use the dedicated
[profile A commands](navigation.md). Set `enable_yolo:=false`,
`start_visual_odometry:=false` and disable RealSense in hardware bring-up.

## 4. Verify the camera route

The offline Humble test used `profile:=C`, `deployment:=pc_all`, `host:=pc`,
`camera_host:=pc`, `camera_transport:=compressed`,
`identity_backend:=multimodal` and `enable_yolo:=false`.
These subscriptions were observed with the robot powered off:

| Consumer | Observed local subscription |
| --- | --- |
| `/person_appearance_extraction` | `/camera/camera/color/image_raw` |
| `/person_depth_fusion` | `/camera/camera/aligned_depth_to_color/image_raw` |
| `/person_depth_fusion` | `/camera/camera/aligned_depth_to_color/camera_info` |

Keep the launch running and check from a second terminal in the same ROS domain:

```bash
ros2 node info /person_appearance_extraction
ros2 node info /person_depth_fusion
```

The offline test verified the local subscriptions. To verify data delivery,
start acquisition and check `ros2 topic hz` on the selected RGB and depth
topics. To verify 3D detections, also enable a detection source and observe
`/social_perception/detections_3d` with a person in view. The other deployment
rows describe the launch configuration and still require their own runtime tests.
