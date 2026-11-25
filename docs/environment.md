# Prepare each terminal

## Robot · Foxy

If Conda is active, run `conda deactivate` first. Source the ROS distribution before the workspace overlay.

```bash
cd ~/g1_ws
source /opt/ros/foxy/setup.bash
source ~/g1_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI=file:///home/unitree/cyclonedds_ws/cyclonedds_g1_optimized.xml
test -r /home/unitree/cyclonedds_ws/cyclonedds_g1_optimized.xml && echo "DDS profile found"
```


This robot-specific XML was tuned for higher DDS bandwidth. It must be supplied separately. One robot-only trial used `/home/unitree/cyclonedds_ws/cyclonedds.xml`; select the profile installed and configured for your robot.

## PC · Humble

```bash
cd ~/g1_ws
source /opt/ros/humble/setup.bash
source ~/g1_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
unset CYCLONEDDS_URI
```

For the PC navigation recipes, also run:
```bash
source tools/source_g1_env.sh
unset CYCLONEDDS_URI
mkdir -p ~/g1_ws/test_logs
```

Verify the [Ethernet connection](network.md) before starting a process that talks to the real robot.

