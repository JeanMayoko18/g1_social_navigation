# Launch navigation by deployment

Before selecting a command, use the [launch matrix and camera placement guide](camera-deployment.md) to identify which hosts must run the system launch.

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

For hybrid runs that send RealSense images across Ethernet, the robot compresses RGB and the PC decompresses it; aligned depth uses the depth ingress path. Use `camera_transport:=compressed` on both participating instances. When acquisition and consumption are on the same machine, the network compression step is unnecessary.

For mapping, select `navigation_mode:=slam` at this same system entry point. See [mapping and autonomous navigation](navigation.md) for mapping, map saving and switching to Nav2.

## Select the deployment for the test

The [README runbooks](../README.md#test-the-framework-on-your-g1) give complete PC/robot command pairs. For profile A on the robot, use the same LiDAR-only hardware and A launch arguments, substituting `deployment:=robot_all host:=robot ros_distro:=foxy`. For optional PC visualization, use the same A mode and `deployment:=robot_all host:=pc ros_distro:=humble enable_visualization:=true`. For a PC-owned social stack use `deployment:=pc_all host:=pc ros_distro:=humble` with accessible camera, microphone and speaker resources; when the camera is remote, provide the expected image transport. The paired C/D examples use `pc_perception` with the camera and speaker on the robot. Dialogue requires Humble on the host owning the dialogue role; `robot_all` on a Foxy G1 cannot own ROS4HRI dialogue.
