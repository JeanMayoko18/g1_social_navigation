# Install, transfer and build g1_ws

| Where you will run the software | What to prepare |
| --- | --- |
| Robot runs hardware and locomotion | Transfer the sources to the G1. Install the hardware dependencies and build `rl_sar` under Foxy. |
| PC runs locomotion | Install the Unitree dependencies for the PC architecture and build `rl_sar` under Humble. Verify that the selected LOCO or custom-policy executable is installed. |
| PC runs navigation and social processing | Build `g1_social_costmap` on the PC so Nav2 can load the social layer for profiles that use personal-space costs. Include `g1_dwb_critics` when using the project DWB configuration. |
| Robot runs navigation or social processing | Build the packages assigned to the robot under Foxy. If Nav2 runs on the robot, build `g1_social_costmap` there for social navigation. Assign the ROS4HRI dialogue role to a Humble host. |

| Hybrid deployment | Complete the build on both machines. Use the same project revision and build each host's own overlay. |

## 1. Clone the project on the PC

For a new checkout:

```bash
git clone --recurse-submodules https://github.com/JeanMayoko18/g1_social_navigation.git ~/g1_ws
cd ~/g1_ws
```

If the workspace already exists, enter it with `cd ~/g1_ws`. Select the project
branch you intend to deploy before transferring the sources.

## 2. Transfer selected packages from the PC to the robot

The navigation stack can run on ROS 2 mobile robots that provide a compatible
laser scan, odometry, TF frames and velocity command interface. If you cloned
`g1_ws` on your PC and want to run navigation onboard, you can either clone
the repository directly on the robot or transfer selected source packages
from your PC using `transfer_pkgs.sh`.

### 1. Select the packages on the PC

Open `~/g1_ws/transfer_pkgs.sh` and review the `packages` array. Include
`g1_nav2` for navigation and `g1_social_costmap` if the robot will run Nav2
with the social costmap layer. Include `rl_sar` when deploying the Unitree G1
locomotion bridge. For another platform, use its own hardware drivers and
velocity command interface.

### 2. Enter the destination and start the transfer

Complete the [wired network setup](network.md), then run the following
commands in a Bash terminal **on the PC**:

```bash
cd ~/g1_ws
read -rp "Robot SSH destination (user@IP): " ROBOT_HOST
read -rp "Remote workspace path: " ROBOT_WORKSPACE
ROBOT_HOST="$ROBOT_HOST" ROBOT_WORKSPACE="$ROBOT_WORKSPACE" bash transfer_pkgs.sh
```

At the first prompt, enter the robot's **SSH username and reachable IP
address**, for example `unitree@192.168.1xx.16x`. At the second prompt,
enter the **absolute workspace path on the robot**, for example
`/home/unitree/g1_ws`. These values apply to this transfer. Press Enter
at either prompt to use the corresponding G1 default configured in the script.

The helper creates `src/` under the remote workspace and copies each selected
package over SSH using `rsync`. If SSH asks for a password, enter the password
for the robot account. A successful transfer prints `ALL TRANSFERS FINISHED`.
If a package transfer fails, the script reports the package name and stops.

### 3. Verify the sources and build on the robot

Open a terminal **on the robot** and enter the workspace path you supplied
during the transfer. For the example above:

```bash
cd /home/unitree/g1_ws
ls src
```

Verify that the selected packages are present, then follow the build steps
below using a ROS 2 distribution supported by your platform and the framework.
The Unitree G1 setup documented here uses ROS 2 Foxy.

The helper transfers the source packages listed in `packages`. If navigation
uses a saved map, also copy the map YAML and its referenced image to the host
running Nav2. Transfer any workspace-level tools required by your launch
procedure separately.

### Fresh GitHub clone

The development helper `transfer_test.sh` is excluded by `.gitignore`.
For a fresh clone, the following alternative transfers the tracked project
files over SSH. The first `rsync` previews the changes. The second applies
them. Matching files on the robot can be updated.

```bash
cd ~/g1_ws
read -rp "Robot IP address: " robot_ip
ssh "unitree@$robot_ip" 'mkdir -p ~/g1_ws'

git ls-files -z | rsync -an --from0 --files-from=- ./ "unitree@$robot_ip:g1_ws/"
git ls-files -z | rsync -a  --from0 --files-from=- ./ "unitree@$robot_ip:g1_ws/"
```

Provision external packages, untracked policy files and models separately.
Build on the destination machine. The PC and robot use different CPU
architectures and ROS distributions.

## 3. Prepare the build terminal and install dependencies

Use a fresh terminal on the machine you are building. If Conda is active,
run `conda deactivate`. These commands assume the host's ROS distribution
and ROS apt repository are already installed.

**Robot — Foxy**

```bash
cd ~/g1_ws
source /opt/ros/foxy/setup.bash
```

**PC — Humble**

```bash
cd ~/g1_ws
source /opt/ros/humble/setup.bash
```

**On each build host**, install the build tools and native dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake python3-colcon-common-extensions \
  python3-rosdep libyaml-cpp-dev libeigen3-dev libboost-all-dev \
  libspdlog-dev libfmt-dev libtbb-dev liblcm-dev \
  "ros-$ROS_DISTRO-rmw-cyclonedds-cpp"
```

On a fresh checkout, the vendored `rl_sar`, `robot_msgs` and
`robot_joint_controller` packages provide separate ROS 1 and ROS 2 manifests.
Expose their ROS 2 manifests before running `rosdep` and `colcon`:

```bash
cd ~/g1_ws
for pkg in rl_sar robot_msgs robot_joint_controller; do
  dir="src/rl_sar/src/$pkg"
  if [ ! -e "$dir/package.xml" ] && [ ! -L "$dir/package.xml" ]; then
    ln -s package.ros2.xml "$dir/package.xml"
  fi
done
```

Provide `unitree_go` and `unitree_hg` as source packages or source the
installed overlay that supplies them. The project's onboard robot-core
build expects Unitree SDK2 at `/home/unitree/unitree_sdk2`. For a custom
locomotion policy, also provide the model and its matching inference runtime
for the host architecture. See the [rl_sar dependency guide](../src/rl_sar/README.md).

Initialize rosdep once per machine if needed, then install the dependencies
declared by the workspace:

```bash
# Run once if rosdep has never been initialized on this machine:
# sudo rosdep init

rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src --rosdistro "$ROS_DISTRO" -y
colcon list --names-only
```

Review unresolved keys before building the corresponding packages. Depending
on the deployment, external dependencies also include the Livox driver,
RealSense driver, or dialogue packages. Source any installed external
overlays in the build terminal.

## 4. Build rl_sar for real hardware

Run this step on the machine that will execute the locomotion bridge.

If `unitree_go` is present as source and you need to build it separately:

```bash
cd ~/g1_ws
colcon build --symlink-install --packages-up-to unitree_go &&
source install/setup.bash
```

**First hardware build**, including workspace dependencies:

```bash
cd ~/g1_ws
colcon build \
  --symlink-install \
  --packages-up-to rl_sar \
  --cmake-args \
    -DUSE_MUJOCO=OFF \
    -DENABLE_REAL_ROBOT=ON &&
source install/setup.bash
```

**Rebuild only `rl_sar`** after its dependencies are already available:

```bash
cd ~/g1_ws
colcon build \
  --symlink-install \
  --packages-select rl_sar \
  --cmake-args \
    -DUSE_MUJOCO=OFF \
    -DENABLE_REAL_ROBOT=ON &&
source install/setup.bash
```

Verify the executables on the host from which you intend to start locomotion:

```bash
ros2 pkg executables rl_sar | grep rl_real_g1_edu23_unitree
```

For Unitree LOCO, look for `rl_real_g1_edu23_unitree_loco`.
For the custom policy, look for `rl_real_g1_edu23_unitree`.
Continue with [locomotion on the robot or PC](locomotion.md).

## 5. Build navigation, social navigation and DWB critics

Use the following package set for the full software build on a host with
the corresponding dependencies installed. For a split deployment, select
the packages required by the roles assigned to each host.

```bash
cd ~/g1_ws

packages=(
  g1_pc_asr
  g1_social_common
  g1_robot_explainability
  g1_social_bringup
  g1_social_perception
  g1_social_msgs
  g1_social_costmap
  g1_nav2
  g1_robot_collaboration
  g1_hri
  g1_speaker_association
)
```

**First build**, including package dependencies and the project DWB critics:

```bash
colcon build \
  --symlink-install \
  --packages-up-to "${packages[@]}" g1_dwb_critics &&
source install/setup.bash
```

**Incremental build** of the selected packages in the same terminal:

```bash
colcon build \
  --symlink-install \
  --packages-select "${packages[@]}" &&
source install/setup.bash
```

**Rebuild the DWB critics separately** after changing the plugin:

```bash
colcon build \
  --symlink-install \
  --packages-select g1_dwb_critics &&
source install/setup.bash
```

**Verify the navigation plugins on the host running Nav2.**

```bash
ros2 pkg prefix g1_social_costmap
ros2 pkg prefix g1_dwb_critics
ros2 launch g1_social_bringup system.launch.py --show-args
```

`g1_social_costmap` provides `g1_social_costmap::SocialLayer`. When the
selected navigation profile loads this layer, it uses observations from
`/social_perception/social_zones` to add costs around people in the local
costmap. These costs allow the local controller to favor trajectories with
greater interpersonal clearance.

The commands above verify that the packages are installed and the system
entry point is available. After launching the selected profile, check the
effective local costmap configuration:

```bash
ros2 param get /local_costmap/local_costmap plugins
ros2 param get /local_costmap/local_costmap social_layer.enabled
```

For a profile using personal-space costs, expect `social_layer` in the
plugin list and `true` for `social_layer.enabled`.

## 6. Clean rebuild when needed

For ordinary source changes, use the incremental commands above. If an
isolated colcon build contains stale artifacts, remove the generated
directories for the affected packages and repeat their build. For example,
to clean the hardware bridge:

```bash
cd ~/g1_ws && rm -rf -- build/rl_sar install/rl_sar
```

If the affected packages are `robot_msgs`, `unitree_hg` and
`robot_joint_controller`, clean their corresponding `build/<package>` and
`install/<package>` directories before repeating the hardware build.
The same package-specific procedure applies to the social packages and
`g1_dwb_critics`. Keep `log/` available when diagnosing a failure.

After a successful build, follow the [host environment guide](environment.md)
in each new terminal, then choose the [launch configuration](deployment.md).
