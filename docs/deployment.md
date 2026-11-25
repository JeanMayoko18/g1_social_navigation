# Launch navigation by deployment

Complete [environment setup](environment.md), [locomotion bring-up](locomotion.md) and [hardware bring-up](hardware.md) first. The tables below describe `g1_social_bringup system.launch.py`. The navigation deployment is independent of the locomotion host. Either the robot or the PC can run Unitree LOCO or the custom locomotion policy.

| Argument | Meaning |
| --- | --- |
| `profile` | Which navigation behaviour to enable (A–D). |
| `deployment` | Where software responsibilities are assigned. |
| `host` | Which machine is running this instance of the launcher. |
| `interaction` | Whether dialogue/HRI is enabled (`none` or `dialogue`). |

| Deployment | PC responsibilities | Robot responsibilities |
| --- | --- | --- |
| `pc_all` | All software roles | Hardware |
| `robot_all` | Optional visualization | All software roles and hardware |
| `pc_perception` | Vision, social perception, decision, navigation and explanation | Speech and hardware |
| `robot_perception` | Vision, decision, navigation, explanation and visualization | Social perception, speech and hardware |

| Profile | pc_all | robot_all | pc_perception | robot_perception |
| --- | --- | --- | --- | --- |
| A, B | PC | Robot; optional PC for RViz | PC | PC |
| C, D | PC | Robot; optional PC for RViz | PC + robot | PC + robot |

For hybrid runs that send RealSense images across Ethernet, the robot compresses the streams and the PC decompresses them. Use `camera_transport:=compressed` on both participating instances. When acquisition and consumption are on the same machine, the network compression step is unnecessary.

For mapping, select `navigation_mode:=slam` at this same system entry point. See [mapping and autonomous navigation](navigation.md) for mapping, map saving and switching to Nav2.

## Profile A · navigation on PC

The recorded PC example uses `pc_perception`. Profile A requires only the PC system launch in this deployment. The map path refers to the host running Nav2.

```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=A \
  interaction:=none \
  deployment:=pc_perception \
  host:=pc \
  ros_distro:=humble \
  enable_yolo:=false \
  language:=es \
  navigation_mode:=nav2 \
  map:="$HOME/g1_ws/maps/mic.yaml" \
  controller:=dwb \
  navigation_speed:=0.12 \
  start_visual_odometry:=true \
  enable_locomotion_calibration:=false \
  forward_lateral_compensation_profile:=affine \
  forward_heading_compensation_profile:=zero \
  enable_visualization:=true \
  enable_ros4hri_visualization:=false \
  enable_raw_image_visualization:=false \
  2>&1 | tee ~/g1_ws/test_logs/profile_A_pc.log
```

The affine and zero profile arguments are retained from the trial command. Locomotion compensation is disabled in this run.

## Profile A · navigation on robot

**Robot · Foxy**
```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=A \
  interaction:=none \
  deployment:=robot_all \
  host:=robot \
  ros_distro:=foxy
```

**PC · Humble, optional RViz**
```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=A \
  interaction:=none \
  deployment:=robot_all \
  host:=pc \
  ros_distro:=humble \
  enable_visualization:=true
```

## Profile C · hybrid, announcements

Use the matching robot and PC blocks for the same encounter. Speech runs on the robot; YOLO inference and navigation run on the PC.
**Robot · Foxy**
```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=C \
  interaction:=none \
  deployment:=pc_perception \
  host:=robot \
  ros_distro:=foxy \
  camera_transport:=compressed \
  enable_yolo:=false \
  navigation_mode:=nav2 \
  enable_visualization:=false \
  2>&1 | tee ~/g1_ws/tts_test.log
```

**PC · Humble**
```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=C \
  interaction:=none \
  collaboration_manager_type:=simple_cd \
  anticipatory_planning_enabled:=false \
  deployment:=pc_perception \
  host:=pc \
  ros_distro:=humble \
  camera_transport:=compressed \
  enable_yolo:=true \
  language:=en \
  navigation_mode:=nav2 \
  map:="$HOME/g1_ws/maps/mic.yaml" \
  controller:=dwb \
  navigation_speed:=0.30 \
  start_visual_odometry:=true \
  enable_locomotion_calibration:=true \
  forward_lateral_compensation_profile:=affine \
  forward_heading_compensation_profile:=zero \
  enable_visualization:=true \
  enable_raw_image_visualization:=false \
  2>&1 | tee ~/g1_ws/test_logs/technical_test0702.log
```

## Profile C · hybrid, dialogue

The PC command here includes the participant demo option `demo_robot_only_single_yes:=true`.

**Robot · Foxy**
```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=C \
  interaction:=dialogue \
  deployment:=pc_perception \
  host:=robot \
  ros_distro:=foxy \
  camera_transport:=compressed \
  identity_backend:=yolo_track \
  enable_yolo:=false \
  navigation_mode:=nav2 \
  enable_visualization:=false \
  2>&1 | tee ~/g1_ws/tts_robot_test12.log
```

**PC · Humble**
```bash
ros2 launch g1_social_bringup system.launch.py \
  profile:=C \
  interaction:=dialogue \
  demo_robot_only_single_yes:=true \
  collaboration_manager_type:=simple_cd \
  anticipatory_planning_enabled:=false \
  deployment:=pc_perception \
  host:=pc \
  ros_distro:=humble \
  camera_transport:=compressed \
  enable_yolo:=true \
  language:=en \
  navigation_mode:=nav2 \
  map:="$HOME/g1_ws/maps/mic.yaml" \
  controller:=dwb \
  navigation_speed:=0.30 \
  start_visual_odometry:=true \
  enable_locomotion_calibration:=true \
  forward_lateral_compensation_profile:=affine \
  forward_heading_compensation_profile:=zero \
  enable_visualization:=true \
  enable_ros4hri_visualization:=false \
  enable_social_rviz:=true \
  enable_raw_image_visualization:=false \
  2>&1 | tee ~/g1_ws/test_logs/participant_dialog_demo_R01.log
```

For another profile or deployment, inspect `ros2 launch g1_social_bringup system.launch.py --show-args` on the target installation. The four recipes above correspond to the supplied operator examples.

