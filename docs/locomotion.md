# Start locomotion on the robot or PC

Choose one host and one locomotion backend for the session. Both executables can run on either host.

| Launch host | Unitree LOCO | Custom locomotion policy |
| --- | --- | --- |
| G1 onboard computer · ROS 2 Foxy | Supported | Supported |
| PC connected to the G1 · ROS 2 Humble | Supported | Supported |

Prepare the [environment for the selected host](environment.md). Run `ip -br addr` on that host to identify the wired interface used to reach the G1. The commands below use `eth0` as an example. Substitute the actual interface name in either command.

## Unitree LOCO · either host

```bash
ros2 run rl_sar rl_real_g1_edu23_unitree_loco eth0 --real --control-backend=unitree_loco --max-motion-duration=0.0
```

## Custom locomotion policy · either host

Before starting the custom policy from either host, use the G1 controller: press **L2 and R2 simultaneously** and wait for Debug mode to take effect. If there is no audible indication, confirm the Debug-mode indicator on the robot. Then start the executable on the selected host.

```bash
ros2 run rl_sar rl_real_g1_edu23_unitree eth0 --real
```

## Both backends · bring up the robot

1. In the active locomotion terminal, press **0** for **Bring Up**.
2. Wait until bring-up completes and the robot is in its ready posture.
3. Press **1** for **Enter Locomotion**. Keep this terminal available while preparing [hardware and sensors](hardware.md).

