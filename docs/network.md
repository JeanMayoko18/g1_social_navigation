# Connect the G1 and PC over Ethernet

Use this guide when ROS nodes on the PC need to discover nodes on the G1. The addresses and interface names depend on the installation.

1. Connect the cable. On each host, use `ip -br link` and `ip -br -4 addr` to identify the wired interface and its IPv4 prefix. Record the G1 address from the robot.
2. In the PC’s wired IPv4 settings, select a free address on the G1’s wired subnet and use the same prefix length. Keep the two host addresses distinct. A direct link needs no gateway for traffic between these hosts.
3. On the PC, enter the G1 address below. Check that the route reports the wired interface and that the peer responds.

```bash
read -rp "G1 wired IPv4 address: " robot_ip
ip route get "$robot_ip"
ping -c 3 "$robot_ip"
```

4. Source the ROS environment on each host as described in [Host environment](environment.md). The reported project setup uses these settings on both machines:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
printenv CYCLONEDDS_URI
```

If the Cyclone DDS configuration selects an interface explicitly, check that it names the wired interface on that host. Review the [Unitree ROS 2 setup files](../src/unitree_ros2/) before sourcing a machine-specific configuration.

5. With the LiDAR pipeline running on the robot, run `ros2 topic info /scan --verbose` on the PC, then `ros2 topic hz /scan`. The first command checks discovery and endpoint QoS. The second checks sample delivery. If the topic appears without samples, inspect reliability compatibility and confirm that the robot is publishing scans locally.
