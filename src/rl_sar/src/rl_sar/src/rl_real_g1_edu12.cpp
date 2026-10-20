/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * RL_Real_G1_EDU12.cpp
 * ----------------------------------------
 * A safe EDU+ (12 active joints) adapter that keeps a 29-DoF policy interface.
 *
 * Key goals:
 *  1) Allow testing cmd_vel reception (Nav2 / teleop) without touching hardware.
 *  2) Allow testing the whole RL pipeline using a FAKE LowState (no robot required).
 *  3) Allow running on the real robot while masking missing joints (EDU12).
 *
 * IMPORTANT DESIGN NOTE ABOUT YAML:
 *  - base.yaml (g1) is loaded in the constructor, then later your FSM Enter()
 *    calls rl.InitRL("g1/robomimic/locomotion"), which loads policy YAML.
 *  - That policy YAML overwrites rl.params.config_node (because that's how rl_sar is written).
 *  - To avoid "policy YAML breaking hardware binding", this file provides a "base snapshot"
 *    (base_snap_) of critical keys loaded from base.yaml.
 *
 * What we do here:
 *  - For REAL hardware I/O, we do NOT rely on "joint_names/controller_names" from policy.
 *  - For mapping policy index -> hardware motor_state index, we use the *current* rl.params joint_mapping
 *    (this must be the policy joint_mapping from policy YAML).
 *  - If you want "policy YAML not to overwrite base.yaml keys", the clean way is to do a merge
 *    in your FSM Enter() (restore base keys + keep policy-only keys). But even without that merge,
 *    the code below remains safe because:
 *       - GetState_real() initializes all 29 DoF values with defaults.
 *       - Missing joints are never read/written to hardware.
 *       - SetCommand() disables everything then enables only active joints.
 *
 * ----------------------------------------
 * Build-time switches:
 *
 *   -DUSE_ROS -DUSE_ROS2
 *       Enable ROS2 cmd_vel subscriber.
 *
 *   -DFAKE_LOWSTATE_TEST
 *       Use fake lowstate values (no DDS subscription needed to test pipeline).
 *
 *   -DDRY_RUN_NO_LOW_CMD
 *       Never publish LowCmd even on real robot. Good for first safety tests.
 *
 *   -DCMD_VEL_ONLY
 *       Only print cmd_vel reception; do not run RL / do not publish LowCmd.
 * ----------------------------------------
 */

#include "rl_real_g1_edu12.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <array>   // optional (if use std::array for policy_to_*)
#include <cmath>

#if defined(USE_ROS2) && defined(USE_ROS)
#include <std_msgs/msg/empty.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#endif
#include <unordered_set>

#ifndef USE_ROS
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#endif

using namespace std::chrono_literals;

// ============================================================
// SECTION 0 — COMPILE-TIME HARDWARE VALIDATION
// ============================================================
//
// These static assertions verify that the EDU23 hardware mask
// has not been accidentally modified.
//
// IMPORTANT:
// ------------------------------------------------------------
// Even though this wrapper uses a 12-action RL policy,
// the physical robot is STILL an EDU23 platform.
//
// Therefore:
//  - the EDU23 joint layout remains valid,
//  - forbidden joints must remain disabled,
//  - compile-time checks are still required.
//
// These checks prevent accidental modifications of the
// hardware joint mask during future refactors.
//
// ============================================================

static_assert(
    is_edu23_active_joint(0) == true,
    "EDU23 joint mask broken"
);

static_assert(
    is_edu23_active_joint(14) == false,
    "EDU23 joint mask broken"
);
static_assert(
    is_edu23_active_joint(13) == false,
    "Waist roll should remain disabled on EDU23"
);

static_assert(
    is_edu23_active_joint(20) == false,
    "Left wrist pitch should remain disabled on EDU23"
);

static_assert(
    is_edu23_active_joint(27) == false,
    "Right wrist pitch should remain disabled on EDU23"
);

// ============================================================
// SECTION 1 — GLOBAL HELPERS
// ============================================================
//
// These helper functions are intentionally placed at the top
// of the file because they are reused by multiple methods.
//
// Why keep them here?
//  - Avoid duplicated safety logic.
//  - Keep the main control methods readable.
//  - Centralize utility functions used across the wrapper.
//
// Typical call chain:
//
//   BuildLegPolicyToMsTable()
//       └── IsForbiddenEdu23MotorIndex()
//
//   SetCommand()
//       └── IsForbiddenEdu23MotorIndex()
//
//   RunModel()
//       └── clampf()
//
// ============================================================

/**
 * @brief Clamp a floating-point value inside a range.
 *
 * Importance:
 *  - Prevents unstable values from propagating into control.
 *  - Useful for actions, torques, velocities, or gains.
 *
 * Called by:
 *  - RunModel()
 *  - SetCommand()
 *  - Future filtering or stabilization helpers
 */
static inline float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

/**
 * @brief Extract yaw angle from a quaternion.
 *
 * Quaternion format:
 *   [w, x, y, z]
 *
 * Importance:
 *  - Useful for heading estimation.
 *  - Can be used for locomotion stabilization.
 *  - Useful for debugging orientation drift.
 *
 * Called by:
 *  - Optional stabilization logic
 *  - Debugging utilities
 *  - Future heading correction modules
 */
static float GetYawFromQuat(const std::vector<float>& q)
{
    if (q.size() < 4)
        return 0.0f;

    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    const float siny = 2.0f * (w * z + x * y);
    const float cosy = 1.0f - 2.0f * (y * y + z * z);

    return std::atan2(siny, cosy);
}

template<typename Q>
static float GetYawFromQuats(const Q& q)
{
    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    const float siny = 2.0f * (w*z + x*y);
    const float cosy = 1.0f - 2.0f * (y*y + z*z);

    return std::atan2(siny, cosy);
}

/**
 * @brief Detect forbidden EDU23 motor indices.
 *
 * Even though this wrapper uses a 12-DoF leg policy,
 * the official Unitree YAML still contains arm and waist joints.
 *
 * Some EDU23 joints must remain disabled because:
 *  - They do not physically exist on EDU23.
 *  - They are unsafe to command.
 *  - They are mechanically constrained.
 *
 * Forbidden joints typically include:
 *  - Waist roll/pitch
 *  - Wrist pitch/yaw joints
 *
 * Called by:
 *  - BuildLegPolicyToMsTable()
 *  - SetCommand()
 */
static inline bool IsForbiddenEdu23MotorIndex(int ms)
{
    switch (ms)
    {
        case 13: case 14:
        case 20: case 21:
        case 27: case 28:
            return true;
        default:
            return false;
    }
}


/**
 * @brief Rate-limited debug printer helper.
 *
 * Importance:
 *  - Prevents flooding the terminal during real-time loops.
 *  - Keeps logs readable.
 *
 * Called by:
 *  - RobotControl()
 *  - RunModel()
 *  - Safety state transitions
 */

static bool ShouldPrintEvery(const std::string& key, double period_sec)
{
    static std::unordered_map<
        std::string,
        std::chrono::steady_clock::time_point
    > last_prints;

    const auto now = std::chrono::steady_clock::now();
    auto& last = last_prints[key];

    if (last.time_since_epoch().count() == 0)
    {
        last = now;
        return true;
    }

    const double elapsed =
        std::chrono::duration<double>(now - last).count();

    if (elapsed >= period_sec)
    {
        last = now;
        return true;
    }

    return false;
}

// ============================================================
// SECTION 1B — EDU23 JOINT STATE ORDER HELPERS
// ============================================================
//
// IMPORTANT CONCEPT
// ------------------------------------------------------------
// The RL policy controls ONLY 12 leg joints.
//
// However:
//
//   ROS JointState publishing
//   still requires the FULL EDU23 active joint layout.
//
// Why?
//
// Because ROS visualization tools expect:
//
//   /joint_states
//       ↓
//   robot_state_publisher
//       ↓
//   TF tree
//       ↓
//   RViz / URDF consistency
//
// If we only publish the 12 RL-controlled joints:
//
//  - TF becomes incomplete,
//  - RViz visualization breaks,
//  - robot_state_publisher becomes inconsistent,
//  - arm joints disappear,
//  - URDF transforms become invalid.
//
// Therefore:
//
//   RL action space  !=  JointState publishing space
//
// The policy uses:
//   12 joints
//
// The robot visualization still uses:
//   23 active EDU23 joints
//
// ============================================================

/**
 * @brief Build the ordered list of active EDU23 motor indices.
 *
 * This helper generates the motor ordering used for:
 *
 *   sensor_msgs/JointState
 *
 * publication.
 *
 * IMPORTANT:
 * ------------------------------------------------------------
 * This ordering is NOT the RL policy ordering.
 *
 * The RL policy ordering:
 *   - contains only 12 leg joints
 *
 * This ordering:
 *   - contains all active EDU23 joints
 *   - includes arms
 *   - excludes forbidden joints
 *
 * The ordering is used by:
 *
 *   PublishJointStates()
 *       ↓
 *   robot_state_publisher
 *       ↓
 *   TF generation
 *
 * ============================================================
 * EDU23 ACTIVE JOINT LAYOUT
 * ============================================================
 *
 * Included:
 *   0  → 12   : lower body + safe waist
 *   15 → 19   : left arm active joints
 *   22 → 26   : right arm active joints
 *
 * Excluded:
 *   13,14     : forbidden waist joints
 *   20,21     : forbidden left wrist joints
 *   27,28     : forbidden right wrist joints
 *
 * ============================================================
 */
static std::vector<int> BuildEdu23JointStateMotorOrder()
{
    std::vector<int> order;

    /*
     * Reserve memory in advance.
     *
     * This avoids dynamic reallocations during insertion.
     */
    order.reserve(23);

    // ========================================================
    // LOWER BODY + SAFE WAIST JOINTS
    // ========================================================

    for (int ms = 0; ms <= 12; ++ms)
    {
        order.push_back(ms);
    }

    // ========================================================
    // LEFT ARM ACTIVE JOINTS
    // ========================================================

    for (int ms = 15; ms <= 19; ++ms)
    {
        order.push_back(ms);
    }

    // ========================================================
    // RIGHT ARM ACTIVE JOINTS
    // ========================================================

    for (int ms = 22; ms <= 26; ++ms)
    {
        order.push_back(ms);
    }

    return order;
}

//================================================================================================

// ============================================================
// SECTION 2 — CONSTRUCTOR / GLOBAL INITIALIZATION
// ============================================================
//
// The constructor is the main entry point of the wrapper.
//
// IMPORTANT DESIGN PRINCIPLE
// ------------------------------------------------------------
// The constructor should initialize the robot interface safely,
// but it should NOT immediately activate locomotion.
//
// Why?
//
// Because at startup:
//  - DDS communication may still be unstable.
//  - LowState packets may not yet be received.
//  - The robot may still be physically unstable.
//  - The FSM may still be in damping or GetUp mode.
//  - The RL policy should not start before safety checks.
//
// Therefore:
//
//   constructor()
//       ↓
//   initialize infrastructure
//       ↓
//   initialize DDS / ROS
//       ↓
//   initialize FSM
//       ↓
//   initialize low-level command structures
//       ↓
//   start loops
//       ↓
//   wait for FSM locomotion state
//       ↓
//   OnEnterPolicy()
//       ↓
//   RL policy becomes active
//
// This design prevents unsafe startup behavior.
//
// ============================================================

/**
 * @brief Main constructor of the EDU12 wrapper.
 *
 * This constructor is responsible for:
 *
 *  1. Parsing runtime configuration
 *  2. Initializing ROS2 or Unitree-only mode
 *  3. Loading base robot configuration
 *  4. Initializing FSM support
 *  5. Initializing low-level command buffers
 *  6. Initializing RL interface dimensions
 *  7. Initializing DDS communication
 *  8. Starting real-time loops
 *
 * IMPORTANT:
 * ------------------------------------------------------------
 * This constructor DOES NOT load motion.pt directly.
 *
 * The RL model is loaded later through:
 *
 *   OnEnterPolicy()
 *       ↓
 *   InitEdu12PolicyMapping()
 *       ↓
 *   InitRL()
 *
 * This separation is intentional and improves safety.
 *
 * ============================================================
 * REAL-TIME LOOPS CREATED HERE
 * ============================================================
 *
 * loop_control
 * ----------------
 * Calls:
 *   RobotControl()
 *
 * Purpose:
 *   Main robot control loop.
 *
 * Frequency:
 *   dt
 *
 * ------------------------------------------------------------
 *
 * loop_rl
 * ----------------
 * Calls:
 *   RunModel()
 *
 * Purpose:
 *   Executes RL inference periodically.
 *
 * Frequency:
 *   dt * decimation
 *
 * ------------------------------------------------------------
 *
 * loop_keyboard
 * ----------------
 * Calls:
 *   KeyboardInterface()
 *
 * Purpose:
 *   Manual debugging / emergency interaction.
 *
 * ============================================================
 */
// ============================================================
// SECTION 2 — CONSTRUCTOR / GLOBAL INITIALIZATION
// ============================================================
//
// This constructor initializes the complete runtime infrastructure
// required by the EDU12 wrapper.
//
// IMPORTANT CONCEPT
// ------------------------------------------------------------
// EDU12 does NOT mean that the physical robot has only 12 joints.
//
// It means:
//   - the RL policy outputs 12 actions,
//   - these 12 actions control the leg joints,
//   - the physical platform is still EDU23,
//   - arms and waist are handled separately by fixed PD targets.
//
// Therefore:
//   - base_edu23.yaml is still loaded,
//   - EDU23 hardware safety rules still apply,
//   - InitJointNum(kNdofPolicy) initializes the RL interface as 12 DoF,
//   - LowState and LowCmd still use Unitree HG motor indices.
//
// MAIN INITIALIZATION ORDER
// ------------------------------------------------------------
//  1. Parse runtime mode
//  2. Initialize ROS or Unitree-only interface
//  3. Load base EDU23 hardware configuration
//  4. Create FSM
//  5. Initialize safe LowCmd
//  6. Initialize RL buffers in 12-DoF policy space
//  7. Initialize Unitree DDS
//  8. Start control loops
//
// The RL model is NOT loaded directly in the constructor.
// It is loaded later through:
//
//   FSM locomotion state
//       ↓
//   OnEnterPolicy()
//       ↓
//   InitEdu12PolicyMapping()
//       ↓
//   InitRL()
//
// ============================================================

RL_Real_G1_EDU12::RL_Real_G1_EDU12(int argc, char **argv)
{
    std::cout << "[CTOR] begin" << std::endl;

    // ============================================================
    // 0) RUNTIME INTERFACE INITIALIZATION
    // ============================================================
    //
    // Two execution modes are supported:
    //
    // 1) Unitree-only mode:
    //      - built without USE_ROS
    //      - used on the robot for direct DDS execution
    //      - runtime arguments are parsed manually
    //
    //      Example:
    //        ros2 run rl_sar rl_real_g1_edu12_unitree eth0 --dry-run
    //
    // 2) ROS-enabled mode:
    //      - built with USE_ROS and USE_ROS2
    //      - reads ROS parameters
    //      - subscribes directly to /cmd_vel
    //
    // ============================================================

#ifndef USE_ROS
    ParseUnitreeOnlyArgs(argc, argv);
#endif

#if defined(USE_ROS1) && defined(USE_ROS)

    // ------------------------------------------------------------
    // ROS1 cmd_vel subscriber.
    //
    // This path is kept for compatibility, although the current
    // robot workflow mainly uses ROS2 or Unitree-only execution.
    // the subscriber is used for teleoperation / navigation velocity commands.
    // ------------------------------------------------------------

    ros::NodeHandle nh;

    this->cmd_vel_subscriber =
        nh.subscribe<geometry_msgs::Twist>(
            "/cmd_vel",
            10,
            &RL_Real_G1_EDU12::CmdvelCallback,
            this
        );

#elif defined(USE_ROS2) && defined(USE_ROS)

    // ------------------------------------------------------------
    // ROS2 node creation.
    //
    // This node handles:
    //  - /cmd_vel reception,
    //  - /joint_states publishing,
    //  - optional heartbeat monitoring,
    //  - runtime safety parameters.
    // ------------------------------------------------------------

    std::cout << "[CTOR] before ros2_node create" << std::endl;

    ros2_node =
        std::make_shared<rclcpp::Node>(
            "rl_real_g1_edu12_node"
        );

    std::cout << "[CTOR] after ros2_node create" << std::endl;


    // ------------------------------------------------------------
    // JointState publisher.
    //
    // IMPORTANT:
    // The RL policy has only 12 actions, but ROS visualization still
    // needs the active EDU23 joint layout.
    //
    // This publisher is used by:
    //   /joint_states
    //       ↓
    //   robot_state_publisher
    //       ↓
    //   TF tree / RViz
    // ------------------------------------------------------------

    joint_state_pub_ =
        ros2_node->create_publisher<
            sensor_msgs::msg::JointState>(
                "/joint_states",
                rclcpp::QoS(
                    rclcpp::KeepLast(50)
                ).best_effort()
        );
    // Names of active joints only.
    tf_joint_names_ =
        GetEdu23ActiveJointNames();

    RCLCPP_INFO(
        ros2_node->get_logger(),
        "JointState publisher initialized on /joint_states with %zu joints",
        tf_joint_names_.size()
    );

    // ============================================================
    // 0.1) HARDWARE MODE PARAMETERS
    // ============================================================
    //
    // hw_mode:
    //   0 = REAL
    //       Real robot LowState + LowCmd.
    //
    //   1 = FAKE_LOWSTATE
    //       Synthetic state for pipeline tests.
    //
    //   2 = CMD_VEL_ONLY
    //       Only verify velocity command reception.
    //
    //   3 = DRY_RUN
    //       Run logic without publishing LowCmd.
    //
    // ============================================================

    const int hw_mode_param =
        ros2_node->declare_parameter<int>(
            "hw_mode",
            0
        );

    if (hw_mode_param < 0 || hw_mode_param > 3)
    {
        throw std::runtime_error(
            "Invalid hw_mode param: expected 0..3"
        );
    }

    hw_mode_ =
        static_cast<HwMode>(hw_mode_param);

    // Whether LowCmd is really published to the robot.
    // For first real tests, use publish_lowcmd=false.
    publish_lowcmd_ =
        ros2_node->declare_parameter<bool>(
            "publish_lowcmd",
            true
        );
    // In FAKE mode, allow the full RL loop to run if true.
    fake_rl_full_ =
        ros2_node->declare_parameter<bool>(
            "fake_rl_full",
            false
        );
    // If true, only cmd_vel reception is tested.
    cmd_vel_only_ =
        ros2_node->declare_parameter<bool>(
            "cmd_vel_only",
            false
        );

    RCLCPP_INFO(
        ros2_node->get_logger(),
        "HW MODE = %d (0=REAL,1=FAKE,2=CMD_VEL_ONLY,3=DRY_RUN)",
        hw_mode_param
    );

    // ------------------------------------------------------------
    // Compile-time safety gate.
    //
    // REAL mode is allowed only if the binary was explicitly
    // compiled with ENABLE_REAL_ROBOT.
    //
    // This prevents accidental motor command publication during
    // development builds.
    // ------------------------------------------------------------

    if (hw_mode_ == HwMode::REAL)
    {
        #ifndef ENABLE_REAL_ROBOT
                throw std::runtime_error(
                    "REAL robot mode requested but ENABLE_REAL_ROBOT "
                    "is not defined at compile-time"
                );
        #endif
    }

    // ============================================================
    // 0.2) NAVIGATION / CMD_VEL PARAMETERS
    // ============================================================
    //
    // These parameters control how external velocity commands are
    // injected into the RL observation.
    //
    // RunModel() later reads these values through GetFreshCmdVel().
    //
    // ============================================================

    // Maximum allowed age of the latest cmd_vel command.
    // If no fresh cmd_vel is received, the command is forced to zero.
    cmd_vel_timeout_sec_ =
        ros2_node->declare_parameter<double>(
            "cmd_vel_timeout_sec",
            0.3
        );
    // Topic used for velocity commands.
    cmd_vel_topic_ =
        ros2_node->declare_parameter<std::string>(
            "cmd_vel_topic",
            "/cmd_vel"
        );
    // Whether navigation mode starts enabled.
    navigation_mode_param_ =
        ros2_node->declare_parameter<bool>(
            "navigation_mode",
            false
        );

    // If true, navigation mode is always active and cmd_vel overrides
    // joystick commands.
    navigation_mode_force_ =
        ros2_node->declare_parameter<bool>(
            "navigation_mode_force",
            false
        );

    this->control.navigation_mode =
        navigation_mode_param_;

    RCLCPP_INFO(
        ros2_node->get_logger(),
        "navigation_mode_force=%d",
        static_cast<int>(navigation_mode_force_)
    );


    // Initialize last cmd_vel time to current time.
    // This avoids an immediate timeout warning during startup.
    //last_cmd_vel_time_ = ros2_node->now();
    last_cmd_vel_time_ =
        rclcpp::Time(0, 0, RCL_ROS_TIME);

 // ------------------------------------------------------------
    // cmd_vel subscriber
    //
    // The callback stores the command in a mutex-protected variable.
    // ------------------------------------------------------------
    using std::placeholders::_1;

    cmd_vel_subscriber =
        ros2_node->create_subscription<
            geometry_msgs::msg::Twist>(
                cmd_vel_topic_,
                rclcpp::SystemDefaultsQoS(),
                std::bind(
                    &RL_Real_G1_EDU12::CmdvelCallback,
                    this,
                    _1
                )
        );

    // ============================================================
    // 0.3) SAFETY PARAMETERS
    // ============================================================
    //
    // These values are used by:
    //
    //   RobotControl()
    //       ↓
    //   UpdateSafetyState()
    //
    // They prevent RL execution when LowState or heartbeat data
    // becomes stale.
    //
    // ============================================================
    // Maximum allowed age of LowState.
    // If exceeded, the robot enters safety mode.
    lowstate_timeout_sec_ =
        ros2_node->declare_parameter<double>(
            "lowstate_timeout_sec",
            0.20
        );

    // Duration of the RECOVERING state before allowing RL again.
    recovery_duration_sec_ =
        ros2_node->declare_parameter<double>(
            "recovery_duration_sec",
            1.00
        );
    // Maximum duration during which stand-hold is attempted after
    // LowState/laptop heartbeat loss.
    stand_hold_max_sec_ =
        ros2_node->declare_parameter<double>(
            "stand_hold_max_sec",
            2.0
        );
    // PD gains used for stand-hold safety posture.
    // Start with low values in real tests and increase carefully.
    stand_hold_kp_ =
        static_cast<float>(
            ros2_node->declare_parameter<double>(
                "stand_hold_kp",
                35.0
            )
        );

    stand_hold_kd_ =
        static_cast<float>(
            ros2_node->declare_parameter<double>(
                "stand_hold_kd",
                3.0
            )
        );
    // ============================================================
    // 0.4) OPTIONAL LAPTOP HEARTBEAT SAFETY
    // ============================================================
    //
    // If enabled, the robot is allowed to run RL only while
    // heartbeat messages are received from the laptop.
    //
    // This is useful when the robot depends on an external computer
    // for navigation or high-level decision making.
    //
    // ============================================================

    // If enabled, the robot is allowed to run RL only while heartbeat
    // messages are received from the laptop.
    require_laptop_heartbeat_ =
        ros2_node->declare_parameter<bool>(
            "require_laptop_heartbeat",
            false
        );
    
    // Maximum age of laptop heartbeat before entering safety mode.
    laptop_heartbeat_timeout_sec_ =
        ros2_node->declare_parameter<double>(
            "laptop_heartbeat_timeout_sec",
            0.5
        );

    // Topic where the laptop publishes std_msgs/msg/Empty heartbeat.
    laptop_heartbeat_topic_ =
        ros2_node->declare_parameter<std::string>(
            "laptop_heartbeat_topic",
            "/rl_laptop_heartbeat"
        );

    // Heartbeat subscriber.
    // Best effort is acceptable because we only need recent messages,
    // not guaranteed delivery of old messages.
    laptop_heartbeat_sub_ =
        ros2_node->create_subscription<std_msgs::msg::Empty>(
            laptop_heartbeat_topic_,
            rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
            std::bind(
                &RL_Real_G1_EDU12::LaptopHeartbeatCallback,
                this,
                _1
            )
        );

    RCLCPP_INFO(
        ros2_node->get_logger(),
        "Laptop heartbeat safety: required=%d topic=%s timeout=%.3f sec",
        static_cast<int>(require_laptop_heartbeat_),
        laptop_heartbeat_topic_.c_str(),
        laptop_heartbeat_timeout_sec_
    );


#endif  // USE_ROS2 && USE_ROS

    // ============================================================
    // 1) LOAD BASE HARDWARE CONFIGURATION
    // ============================================================
    //
    // The physical robot is still EDU23.
    //
    // Therefore, even in this EDU12 wrapper, the base hardware
    // configuration remains:
    //
    //   base_edu23.yaml
    //
    // This file contains hardware-level information such as:
    //  - active joints,
    //  - motor layout,
    //  - joint names,
    //  - torque limits,
    //  - controller names.
    //
    // The 12-DoF policy configuration is loaded later by InitRL().
    //
    // ============================================================
    // This identifies the robot configuration namespace.
    this->ang_vel_axis = "body";
    //this->robot_name = "g1";
    this->robot_name = "g1_edu12";

   // Load EDU23-specific base configuration.
    // This file should contain hardware-related information such as:
    //   - active joints
    //   - hardware mapping
    //   - torque limits
    //   - joint names
    std::cout << "[CTOR] before ReadYaml" << std::endl;

    this->ReadYaml(
        this->robot_name,
        "base_edu23.yaml"
    );

    std::cout << "[CTOR] after ReadYaml" << std::endl;

    // ------------------------------------------------------------
    // Save a snapshot of base hardware keys.
    //
    // Why?
    //
    // InitRL() later loads the policy YAML and may overwrite
    // params.config_node.
    //
    // This snapshot keeps important hardware information available.
    // ------------------------------------------------------------

    base_snap_.dt =
        this->params.Get<float>(
            "dt",
            0.005f
        );

    base_snap_.decimation =
        this->params.Get<int>(
            "decimation",
            4
        );

    base_snap_.num_of_dofs =
        this->params.Get<int>(
            "num_of_dofs",
            29
        );

    base_snap_.joint_mapping =
        this->params.Get<std::vector<int>>(
            "joint_mapping",
            {}
        );

    base_snap_.default_dof_pos =
        this->params.Get<std::vector<float>>(
            "default_dof_pos",
            {}
        );

    base_snap_.torque_limits =
        this->params.Get<std::vector<float>>(
            "torque_limits",
            {}
        );

    base_snap_.joint_names =
        this->params.Get<std::vector<std::string>>(
            "joint_names",
            {}
        );

    base_snap_.joint_controller_names =
        this->params.Get<std::vector<std::string>>(
            "joint_controller_names",
            {}
        );

    std::cout
        << "[INFO] Base hardware YAML snapshot saved"
        << std::endl;

    // ------------------------------------------------------------
    // Debug print for base joint_mapping.
    //
    // This is not the final EDU12 policy mapping.
    // It only confirms that the base hardware YAML was read.
    // ------------------------------------------------------------

    {
        const auto jm =
            this->params.Get<std::vector<int>>(
                "joint_mapping",
                {}
            );

        std::cout
            << "[DEBUG] joint_mapping AFTER base.yaml size="
            << jm.size()
            << " : ";

        for (int i = 0;
             i < std::min<int>(
                    static_cast<int>(jm.size()),
                    12);
             ++i)
        {
            std::cout << jm[i] << " ";
        }

        std::cout << "..." << std::endl;
    }

    // ============================================================
    // 2) CREATE FINITE STATE MACHINE
    // ============================================================
    //
    // The FSM controls high-level behavior:
    //  - passive mode,
    //  - GetUp,
    //  - locomotion,
    //  - policy activation,
    //  - keyboard transitions.
    //
    // Later, the FSM calls:
    //
    //   OnEnterPolicy()
    //
    // which triggers:
    //
    //   InitEdu12PolicyMapping()
    //
    // ============================================================

    if (FSMManager::GetInstance().IsTypeSupported(
            this->robot_name))
    {
        auto fsm_ptr =
            FSMManager::GetInstance().CreateFSM(
                this->robot_name,
                this
            );

        if (fsm_ptr)
            this->fsm = *fsm_ptr;
    }
    else
    {
        std::cout
            << LOGGER::ERROR
            << "[FSM] No FSM registered for robot: "
            << this->robot_name
            << std::endl;
    }
    
    // ============================================================
    // 3) INITIALIZE CONTROL CONTAINERS
    // ============================================================
    //
    // LowCmd is initialized before any loop starts.
    //
    // This guarantees that all motor commands start in a safe,
    // disabled state.
    //
    // ============================================================
    // Unitree command mode.
    this->mode_pr = Mode::PR;
    // Machine mode is updated later from LowState.
    this->mode_machine = 0;

    // Initialize LowCmd with all motors disabled.
    // This is the safest initial hardware command.
    std::cout << "[CTOR] before InitLowCmd" << std::endl;
    this->InitLowCmd();
    std::cout << "[CTOR] after InitLowCmd" << std::endl;

    // ------------------------------------------------------------
    // Initialize RL interface size.
    //
    // IMPORTANT:
    //
    // In EDU23 wrapper:
    //   kNdofPolicy = 29
    //
    // In EDU12 wrapper:
    //   kNdofPolicy = 12
    //
    // Therefore this call initializes the policy buffers for the
    // official Unitree 12-action leg policy.
    // ------------------------------------------------------------

    this->InitJointNum(this->kNdofPolicy);
    // Initialize output buffers and control structures.
    this->InitOutputs();
    this->InitControl();

#ifndef USE_ROS

    this->control.navigation_mode =
        navigation_mode_param_;

    StartIpcCmdVelReceiver(ipc_cmd_port_);

#endif

    // ============================================================
    // 4) UNITREE DDS INITIALIZATION
    // ============================================================
    //
    // DDS is used for:
    //  - LowState reception,
    //  - IMU reception,
    //  - LowCmd publication.
    //
    // In DRY_RUN:
    //  - DDS LowState can still be received,
    //  - LowCmd is not published if publish_lowcmd_ == false.
    //
    // ============================================================

    const bool use_unitree_dds =
        (hw_mode_ == HwMode::REAL ||
         hw_mode_ == HwMode::DRY_RUN);

    if (use_unitree_dds)
    {
        std::cout << "[CTOR] entering Unitree DDS block" << std::endl;

        if (argc < 2)
        {
            throw std::runtime_error(
                "Unitree mode requires network interface argument, e.g. eth0"
            );
        }

        std::cout << "[CTOR] before ChannelFactory Init" << std::endl;
        ChannelFactory::Instance()->Init(0, argv[1]);
        std::cout << "[CTOR] after ChannelFactory Init" << std::endl;
        // ========================================================
        // 4.1) LOWCMD PUBLISHER
        // ========================================================
        //
        // LowCmd publication is enabled only when publish_lowcmd_
        // is true.
        //
        // In dry-run tests, publish_lowcmd_ should remain false.
        //
        // ========================================================

        if (publish_lowcmd_)
        {
            // MotionSwitcher is used to release any active Unitree high-level
            // motion service before this low-level controller takes over.
            std::cout << "[CTOR] before MotionSwitcher create/init" << std::endl;

            msc_ = std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
            msc_->SetTimeout(5.0f);
            msc_->Init();

            std::cout << "[CTOR] after MotionSwitcher Init" << std::endl;

            std::string form;
            std::string name;

            const int max_attempts = 10;
            int attempts = 0;

            // ----------------------------------------------------
            // Release active high-level motion mode.
            //
            // This prevents conflict between:
            //  - Unitree built-in controller,
            //  - this low-level RL controller.
            // ----------------------------------------------------
            while (msc_->CheckMode(form, name), !name.empty())
            {
                if (msc_->ReleaseMode())
                    break;

                ++attempts;

                if (attempts >= max_attempts)
                {
                    std::cerr << "[FATAL] Unable to release motion mode after "
                            << max_attempts << " attempts" << std::endl;
                    break;
                }

                sleep(3);
            }
            // ------------------------------------------------------------
            // LowCmd publisher
            // Sends motor commands to the robot.
            // ------------------------------------------------------------
            this->lowcmd_publisher.reset(
                new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC)
            );
            this->lowcmd_publisher->InitChannel();

            std::cout << "[CTOR] LowCmd publisher enabled" << std::endl;

        }
        else
        {
            std::cout << "[CTOR] MotionSwitcher skipped: publish_lowcmd=false" << std::endl;
            std::cout << "[CTOR] LowCmd publisher skipped: publish_lowcmd=false" << std::endl;
        }

        // ========================================================
        // 4.2) LOWSTATE SUBSCRIBER
        // ========================================================
        //
        // Calls:
        //   LowStateHandler()
        //
        // LowStateHandler() updates:
        //  - latest motor states,
        //  - IMU state,
        //  - freshness timestamp,
        //  - optional stand-hold posture.
        //
        // ========================================================
        this->lowstate_subscriber.reset(
            new ChannelSubscriber<LowState_>(HG_STATE_TOPIC)
        );
        this->lowstate_subscriber->InitChannel(
            std::bind(
                &RL_Real_G1_EDU12::LowStateHandler,
                this,
                std::placeholders::_1
            ),
            1
        );

        std::cout << "[CTOR] LowState subscriber enabled" << std::endl;

        // ========================================================
        // 4.3) TORSO IMU SUBSCRIBER
        // ========================================================
        //
        // Calls:
        //   ImuTorsoHandler()
        //
        // This provides an optional additional IMU source.
        //
        // ========================================================

        // ------------------------------------------------------------
        // Torso IMU subscriber
        // Optional additional IMU source.
        // ------------------------------------------------------------
        this->imutorso_subscriber.reset(
            new ChannelSubscriber<IMUState_>(HG_IMU_TORSO)
        );
        this->imutorso_subscriber->InitChannel(
            std::bind(
                &RL_Real_G1_EDU12::ImuTorsoHandler,
                this,
                std::placeholders::_1
            ),
            1
        );

        std::cout
            << "[CTOR] IMU subscriber enabled"
            << std::endl;
    }
    else
    {
        std::cout
            << "[CTOR] Unitree DDS skipped: non-real or publish_lowcmd=false"
            << std::endl;
    }

    // ============================================================
    // 5) CONTROL LOOPS 
    // ============================================================
    //
     // Allows switching FSM states from keyboard.
    // Three loops are started:
    //
    // 1) loop_keyboard
    //      calls RL::KeyboardInterface()
    //
    // 2) loop_control
    //      calls RobotControl()
    //
    // 3) loop_rl
    //      calls RunModel()
    //
    // ============================================================
    this->loop_keyboard =
            std::make_shared<LoopFunc>(
                "loop_keyboard",
                0.05,
                std::bind(
                    &RL::KeyboardInterface,
                    static_cast<RL*>(this)
                )
            );
    // ------------------------------------------------------------
    // Robot control loop
    //
    // Runs at dt.
    // Responsible for:
    //   - reading robot state
    //   - running FSM
    //   - publishing LowCmd when safe
    // ------------------------------------------------------------
    this->loop_control =
            std::make_shared<LoopFunc>(
                "loop_control",
                this->params.Get<float>("dt"),
                std::bind(
                    &RL_Real_G1_EDU12::RobotControl,
                    this
                )
            );

    // ------------------------------------------------------------
    // RL inference loop
    //
    // Runs at dt * decimation.
    // Responsible for:
    //   - building observation
    //   - running neural policy
    //   - computing target joint commands
    // ------------------------------------------------------------
    this->loop_rl =
        std::make_shared<LoopFunc>(
            "loop_rl",
            this->params.Get<float>("dt")
                * this->params.Get<int>("decimation"),
            std::bind(
                &RL_Real_G1_EDU12::RunModel,
                this
            )
        );


    // ------------------------------------------------------------
    // Startup safety initialization.
    //
    // The safety state starts as RUNNING.
    // The actual safety watchdog is armed later inside RobotControl()
    // after a startup grace period.
    // ------------------------------------------------------------

    std::cout
        << "[STARTUP FIX] Forcing safety_state_ to RUNNING before loops"
        << std::endl;

    safety_state_ =
        RLSafetyState::RUNNING;

    lowstate_lost_timer_started_ =
        false;

    // ============================================================
    // 6) OPTIONAL PLOTTING
    // ============================================================

    #ifdef PLOT

        this->plot_t =
            std::vector<int>(
                this->plot_size,
                0
            );

        this->plot_real_joint_pos.resize(
            this->params.Get<int>("num_of_dofs")
        );

        this->plot_target_joint_pos.resize(
            this->params.Get<int>("num_of_dofs")
        );

        for (auto& v : this->plot_real_joint_pos)
            v = std::vector<float>(
                this->plot_size,
                0
            );

        for (auto& v : this->plot_target_joint_pos)
            v = std::vector<float>(
                this->plot_size,
                0
            );

        this->loop_plot =
            std::make_shared<LoopFunc>(
                "loop_plot",
                0.002,
                std::bind(
                    &RL_Real_G1_EDU23::Plot,
                    this
                )
            );

        this->loop_plot->start();

    #endif

    // ============================================================
    // 7) OPTIONAL CSV LOGGING
    // ============================================================

    #ifdef CSV_LOGGER

        // Initialize CSV logger for debugging policy outputs,
        // joint states, torques, etc.
        this->CSVInit(this->robot_name);

    #endif

    // ============================================================
    // 8) FINAL STARTUP MESSAGE
    // ============================================================
    std::cout
    << LOGGER::INFO
    << "RL_Real_G1_EDU12 started "
    << "(official Unitree policy: 12 leg actions)"
    << std::endl;

    // Start loops only after all safety parameters, ROS topics,
    // DDS channels, and control containers are initialized.

    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
}

// ----------------------------
// Destructor
// ----------------------------
RL_Real_G1_EDU12::~RL_Real_G1_EDU12()
{
#ifndef USE_ROS
    StopIpcCmdVelReceiver();
#endif

    if (this->loop_keyboard) this->loop_keyboard->shutdown();
    if (this->loop_control)  this->loop_control->shutdown();
    if (this->loop_rl)       this->loop_rl->shutdown();

#ifdef PLOT
    if (this->loop_plot)     this->loop_plot->shutdown();
#endif

    std::cout << LOGGER::INFO << "RL_Real_G1_EDU12 exit" << std::endl;
}

void RL_Real_G1_EDU12::DumpPolicyHardwareComparisonOnce() const
{
    static bool done = false;
    if (done) return;
    done = true;

    LowState_ low_state_copy;
    {
        std::lock_guard<std::mutex> lk(this->lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    const auto& ms_arr = low_state_copy.motor_state();

    std::cout << "\n=== EDU12 POLICY ↔ HARDWARE CHECK ===\n";
    std::cout << "p | ms | default_q | hw_q | hw_dq\n";
    std::cout << "-------------------------------------\n";

    for (int p = 0; p < this->kNdofPolicy; ++p)
    {
        const int ms = this->leg_policy_to_ms_[p];

        std::cout
            << p << " | "
            << ms << " | "
            << this->policy_default_dof_pos_[p] << " | ";

        if (ms >= 0 && ms < static_cast<int>(ms_arr.size()))
        {
            std::cout
                << ms_arr[ms].q() << " | "
                << ms_arr[ms].dq();
        }
        else
        {
            std::cout << "INVALID | INVALID";
        }

        std::cout << "\n";
    }

    std::cout << "=====================================\n\n";
}

// ============================================================
// SECTION 3 — FSM ENTRY POINTS
// ============================================================
//
// This section contains the entry points used by the FSM
// (Finite State Machine).
//
// IMPORTANT DESIGN IDEA
// ------------------------------------------------------------
// The constructor does NOT immediately load the RL policy.
//
// Why?
//
// Because:
//  - The robot may still be initializing.
//  - DDS communication may not yet be stable.
//  - LowState data may not yet be available.
//  - The FSM may still be in GetUp or damping mode.
//
// Therefore:
//
//   constructor()
//       ↓
//   FSM initialization
//       ↓
//   robot stabilization
//       ↓
//   locomotion state entered
//       ↓
//   OnEnterPolicy()
//       ↓
//   InitEdu12PolicyMapping()
//       ↓
//   InitRL()
//       ↓
//   motion.pt loaded
//
// This design is much safer than loading the RL model
// immediately at startup.
//
// ============================================================

/**
 * @brief FSM callback executed when entering the locomotion policy state.
 *
 * Importance:
 *  - This is the TRUE entry point of the RL locomotion system.
 *  - The policy is not active before this method succeeds.
 *  - This method guarantees that mapping and policy loading
 *    happen only once.
 *
 * Why this method exists:
 *  - The constructor should stay lightweight and safe.
 *  - The FSM decides when locomotion is allowed.
 *  - The robot must already be stabilized before enabling RL.
 *
 * Call chain:
 *
 *   FSM locomotion state
 *       ↓
 *   OnEnterPolicy()
 *       ↓
 *   InitEdu12PolicyMapping()
 *       ↓
 *   InitRL()
 *       ↓
 *   BuildLegPolicyToMsTable()
 *
 * Returns:
 *  - true  → policy ready
 *  - false → locomotion state rejected
 */
bool RL_Real_G1_EDU12::OnEnterPolicy(
    const std::string& robot_config_path)
{
    /*
     * Avoid rebuilding mappings multiple times.
     *
     * Once the mapping is ready,
     * the locomotion state can safely reuse it.
     */
    if (this->policy_table_ready_)
    {
        std::cout
            << "[INFO] EDU12 policy already initialized"
            << std::endl;

        return true;
    }

    /*
     * Main initialization sequence.
     *
     * This method:
     *  - loads motion.pt
     *  - loads YAML parameters
     *  - builds the 12DoF → motor mapping
     *  - validates all dimensions
     */
    if (!this->InitEdu12PolicyMapping(robot_config_path))
    {
        std::cerr
            << "[FATAL] Failed to initialize EDU12 policy"
            << std::endl;

        return false;
    }
    
    this->DumpPolicyHardwareComparisonOnce();

    std::cout
        << "[INFO] EDU12 locomotion policy ready"
        << std::endl;

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// ============================================================
// SECTION 4 — POLICY / MOTOR MAPPING
// ============================================================
//
// This section connects the official Unitree 12-action policy
// to the physical Unitree HG motor indices.
//
// IMPORTANT CONCEPT
// ------------------------------------------------------------
// The official Unitree G1 policy does NOT control all robot joints.
//
// It controls only:
//
//   12 leg joints
//
// The remaining joints:
//
//   waist + arms
//
// are not produced by the neural policy. They are held using
// fixed PD targets loaded from the official YAML:
//
//   arm_waist_joint2motor_idx
//   arm_waist_kps
//   arm_waist_kds
//   arm_waist_target
//
// Therefore, this wrapper separates:
//
//   1) RL policy action space
//      - size 12
//      - produced by motion.pt
//
//   2) Unitree motor space
//      - size 35 on HG messages
//      - used by LowState and LowCmd
//
//   3) Fixed PD space
//      - waist + arms
//      - not controlled by RL
//
// ============================================================
//
// Main call chain:
//
//   OnEnterPolicy()
//       ↓
//   InitEdu12PolicyMapping()
//       ↓
//   InitRL()
//       ↓
//   read YAML parameters
//       ↓
//   BuildLegPolicyToMsTable()
//       ↓
//   policy_table_ready_ = true
//
// After policy_table_ready_ becomes true:
//
//   GetState_real()
//       uses leg_policy_to_ms_
//
//   SetCommand()
//       uses leg_policy_to_ms_
//       uses arm_waist_* fixed PD targets
//
// ============================================================

/**
 * @brief Load the official Unitree EDU12 policy configuration.
 *
 * This method is the main policy initialization entry point.
 *
 * It is called by:
 *
 *   OnEnterPolicy()
 *
 * It calls:
 *
 *   InitRL(robot_config_path)
 *       ↓
 *   BuildLegPolicyToMsTable()
 *
 * Responsibilities:
 * ------------------------------------------------------------
 *  1. Load the RL model and YAML through InitRL().
 *  2. Extract the 12-leg policy mapping.
 *  3. Extract leg PD gains.
 *  4. Extract default leg angles.
 *  5. Extract fixed arm/waist PD configuration.
 *  6. Validate and build the final motor mapping.
 *
 * Why InitRL() is called here:
 * ------------------------------------------------------------
 * The constructor loads only the base hardware configuration.
 *
 * The policy YAML is loaded only when the FSM enters the policy
 * state, because locomotion should not be activated during early
 * startup.
 *
 * Expected official YAML fields:
 * ------------------------------------------------------------
 *
 *   leg_joint2motor_idx:
 *     Maps the 12 policy action indices to Unitree motor indices.
 *
 *   kps / kds:
 *     PD gains for the leg joints.
 *
 *   default_angles:
 *     Default leg posture used for state initialization.
 *
 *   arm_waist_joint2motor_idx:
 *     Motor indices for waist and arm joints.
 *
 *   arm_waist_kps / arm_waist_kds:
 *     PD gains for fixed waist/arm targets.
 *
 *   arm_waist_target:
 *     Fixed target posture for waist and arms.
 *
 * Return:
 * ------------------------------------------------------------
 *  - true  if the policy and mapping are valid.
 *  - false if any required configuration is missing or invalid.
 */
bool RL_Real_G1_EDU12::InitEdu12PolicyMapping(
    const std::string& robot_config_path)
{
    try
    {
        std::cout
            << "[INFO] InitEdu12PolicyMapping: loading official Unitree policy"
            << std::endl;

        // --------------------------------------------------------
        // Load policy YAML and model.
        //
        // This call usually loads:
        //  - motion.pt
        //  - num_obs
        //  - num_actions
        //  - observation settings
        //  - scaling factors
        //  - PD gains
        //
        // Important:
        // InitRL() may overwrite params.config_node with the
        // policy-specific YAML. This is expected.
        // --------------------------------------------------------
        if (this->debug_policy_init_)
        {
            std::cout
                << "[DEBUG EDU12] robot_config_path BEFORE InitRL = "
                << robot_config_path
                << std::endl;
        }
        if (this->debug_policy_init_)
        {
            std::cout
                << "[DEBUG EDU12] expected config path = "
                << std::string(POLICY_DIR)
                << "/"
                << robot_config_path
                << "/config.yaml"
                << std::endl;
        }
        this->InitRL(robot_config_path);

        std::cout
            << "[DEBUG EDU12] model_name="
            << this->params.Get<std::string>("model_name", "NONE")
            << std::endl;

        std::cout
            << "[DEBUG EDU12] num_of_dofs="
            << this->params.Get<int>("num_of_dofs", -1)
            << std::endl;

        std::cout
            << "[DEBUG EDU12] observations size="
            << this->params.Get<std::vector<std::string>>(
                "observations", {}
            ).size()
            << std::endl;

        std::cout
            << "[DEBUG EDU12] leg_joint2motor_idx size="
            << this->params.Get<std::vector<int>>(
                "leg_joint2motor_idx", {}
            ).size()
            << std::endl;

        // ========================================================
        // 1) LEG POLICY CONFIGURATION
        // ========================================================
        //
        // The official Unitree policy outputs 12 actions.
        //
        // Each action corresponds to one leg joint.
        //
        // leg_joint2motor_idx_[p] tells which Unitree motor index
        // receives action p.
        //
        // ========================================================

        leg_joint2motor_idx_ =
            this->params.Get<std::vector<int>>(
                "leg_joint2motor_idx",
                {}
            );

        leg_kps_ =
            this->params.Get<std::vector<float>>(
                "kps",
                {}
            );

        leg_kds_ =
            this->params.Get<std::vector<float>>(
                "kds",
                {}
            );

        leg_default_angles_ =
            this->params.Get<std::vector<float>>(
                "default_angles",
                {}
            );
        
        this->walk_ready_dof_pos_ =
            this->params.Get<std::vector<float>>(
                "walk_ready_dof_pos",
                {}
            );
        
        this->stand_hold_dof_pos_ =
            this->params.Get<std::vector<float>>(
                "stand_hold_dof_pos",
                {}
            );

        this->real_stable_stand_dof_pos_ =
            this->params.Get<std::vector<float>>(
                "real_stable_stand_dof_pos",
                {}
            );
        
        this->stand_kps_ =
            this->params.Get<std::vector<float>>(
                "stand_kps",
                {}
            );

        this->stand_kds_ =
            this->params.Get<std::vector<float>>(
                "stand_kds",
                {}
            );

        this->default_dof_pos_ =
            this->params.Get<std::vector<float>>(
                "default_dof_pos",
                {}
            );

        if (this->default_dof_pos_.size() !=
            static_cast<size_t>(this->kNdofPolicy))
        {
            std::cerr << "[FATAL] default_dof_pos invalid size" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] real_stable_stand_dof_pos size="
                << this->real_stable_stand_dof_pos_.size()
                << " expected="
                << this->kNdofPolicy
                << std::endl;

        if (this->real_stable_stand_dof_pos_.size() !=
            static_cast<size_t>(this->kNdofPolicy))
        {
            std::cerr << "[FATAL] real_stable_stand_dof_pos invalid size" << std::endl;
            return false;
        }

        if (this->stand_kps_.size() != static_cast<size_t>(this->kNdofPolicy) ||
                this->stand_kds_.size() != static_cast<size_t>(this->kNdofPolicy))
        {
            std::cerr << "[FATAL] stand_kps/kds invalid size" << std::endl;
            return false;
        }

        if (this->stand_hold_dof_pos_.size() !=
            static_cast<size_t>(this->kNdofPolicy))
        {
            std::cerr
                << "[FATAL] stand_hold_dof_pos size="
                << this->stand_hold_dof_pos_.size()
                << " expected "
                << this->kNdofPolicy
                << std::endl;

            return false;
        }

        // ========================================================
        // 2) FIXED ARM / WAIST CONFIGURATION
        // ========================================================
        //
        // The official policy does not output arm or waist actions.
        //
        // Instead, these joints are held at fixed targets using
        // PD control.
        //
        // This is why arm_waist_target_ is not part of the neural
        // network output.
        //
        // ========================================================

        arm_waist_joint2motor_idx_ =
            this->params.Get<std::vector<int>>(
                "arm_waist_joint2motor_idx",
                {}
            );

        arm_waist_kps_ =
            this->params.Get<std::vector<float>>(
                "arm_waist_kps",
                {}
            );

        arm_waist_kds_ =
            this->params.Get<std::vector<float>>(
                "arm_waist_kds",
                {}
            );

        arm_waist_target_ =
            this->params.Get<std::vector<float>>(
                "arm_waist_target",
                {}
            );
        // ========================================================
        // 3) BUILD FINAL MAPPING
        // ========================================================
        //
        // BuildLegPolicyToMsTable() validates all dimensions and
        // builds the final:
        //
        //   policy action index → Unitree motor index
        //
        // table.
        //
        // If this succeeds, policy_table_ready_ becomes true.
        //
        // ========================================================

        return BuildLegPolicyToMsTable();
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FATAL] InitEdu12PolicyMapping exception: "
            << e.what()
            << std::endl;

        return false;
    }
}

/**
 * @brief Build and validate the 12-action policy-to-motor table.
 *
 * This method is called by:
 *
 *   InitEdu12PolicyMapping()
 *
 * It prepares the mapping used later by:
 *
 *   GetState_real()
 *       to read real joint positions and velocities
 *
 *   SetCommand()
 *       to write motor commands
 *
 * Why this method is separate:
 * ------------------------------------------------------------
 * Loading YAML and validating motor mappings are different tasks.
 *
 * Keeping them separate makes the logic safer and easier to debug:
 *
 *   InitEdu12PolicyMapping()
 *       = load policy-related YAML fields
 *
 *   BuildLegPolicyToMsTable()
 *       = validate dimensions and build lookup tables
 *
 * Mapping built:
 * ------------------------------------------------------------
 *
 *   leg_policy_to_ms_[p] = ms
 *
 * where:
 *
 *   p  = policy action index, from 0 to 11
 *   ms = Unitree HG motor index
 *
 * Example:
 *
 *   leg_policy_to_ms_[0] = 0
 *   leg_policy_to_ms_[1] = 1
 *   ...
 *   leg_policy_to_ms_[11] = 11
 *
 * Safety checks:
 * ------------------------------------------------------------
 *  - leg_joint2motor_idx must have 12 elements.
 *  - leg gains must have 12 elements.
 *  - default leg posture must have 12 elements.
 *  - motor indices must be inside [0, kMsTotal).
 *  - duplicate motor indices are rejected.
 *  - forbidden EDU23 motor indices are rejected.
 *  - arm/waist vector sizes must be consistent.
 *
 * On success:
 * ------------------------------------------------------------
 *  - enabled_ms_mask_ marks all allowed motor channels.
 *  - policy_default_dof_pos_ becomes the 12-leg default posture.
 *  - policy_table_ready_ becomes true.
 */
bool RL_Real_G1_EDU12::BuildLegPolicyToMsTable()
{
    // ------------------------------------------------------------
    // Always start from a safe invalid state.
    //
    // This prevents partially initialized mappings from being used
    // by GetState_real() or SetCommand().
    // ------------------------------------------------------------

    policy_table_ready_ = false;

    leg_policy_to_ms_.fill(-1);
    enabled_ms_mask_.fill(false);

    // ============================================================
    // 1) VALIDATE LEG POLICY CONFIGURATION
    // ============================================================

    if ((int)leg_joint2motor_idx_.size() != kNdofPolicy)
    {
        std::cerr
            << "[FATAL] leg_joint2motor_idx size="
            << leg_joint2motor_idx_.size()
            << " expected "
            << kNdofPolicy
            << std::endl;

        return false;
    }

    if ((int)leg_kps_.size() != kNdofPolicy)
    {
        std::cerr
            << "[FATAL] leg_kps size="
            << leg_kps_.size()
            << " expected "
            << kNdofPolicy
            << std::endl;

        return false;
    }

    if ((int)leg_kds_.size() != kNdofPolicy)
    {
        std::cerr
            << "[FATAL] leg_kds size="
            << leg_kds_.size()
            << " expected "
            << kNdofPolicy
            << std::endl;

        return false;
    }

    if ((int)leg_default_angles_.size() != kNdofPolicy)
    {
        std::cerr
            << "[FATAL] leg_default_angles size="
            << leg_default_angles_.size()
            << " expected "
            << kNdofPolicy
            << std::endl;

        return false;
    }

    // ============================================================
    // 2) BUILD LEG POLICY → MOTOR TABLE
    // ============================================================
    //
    // Each policy output action is associated with one physical
    // motor channel.
    //
    // The official Unitree YAML usually maps:
    //
    //   policy index 0..11 → motor index 0..11
    //
    // But we still validate this instead of assuming it.
    //
    // ============================================================

    std::unordered_set<int> seen_ms;

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        const int ms =
            leg_joint2motor_idx_[p];

        if (ms < 0 || ms >= kMsTotal)
        {
            std::cerr
                << "[FATAL] leg_joint2motor_idx["
                << p
                << "]="
                << ms
                << " out of range 0.."
                << (kMsTotal - 1)
                << std::endl;

            return false;
        }

        if (IsForbiddenEdu23MotorIndex(ms))
        {
            std::cerr
                << "[FATAL] leg_joint2motor_idx["
                << p
                << "] uses forbidden EDU23 motor index "
                << ms
                << std::endl;

            return false;
        }

        if (seen_ms.count(ms))
        {
            std::cerr
                << "[FATAL] Duplicate leg motor index detected: "
                << ms
                << std::endl;

            return false;
        }

        seen_ms.insert(ms);

        leg_policy_to_ms_[p] =
            ms;

        enabled_ms_mask_[ms] =
            true;
    }

    // ============================================================
    // 3) VALIDATE ARM / WAIST FIXED PD CONFIGURATION
    // ============================================================
    //
    // The arm/waist configuration is not part of the RL action
    // vector, but it is still sent to LowCmd as fixed PD targets.
    //
    // All vectors must have the same size because they are accessed
    // using the same index.
    //
    // ============================================================

    const size_t arm_size =
        arm_waist_joint2motor_idx_.size();

    if (arm_waist_kps_.size() != arm_size ||
        arm_waist_kds_.size() != arm_size ||
        arm_waist_target_.size() != arm_size)
    {
        std::cerr
            << "[FATAL] Inconsistent arm/waist vector sizes: "
            << "idx="
            << arm_waist_joint2motor_idx_.size()
            << " kps="
            << arm_waist_kps_.size()
            << " kds="
            << arm_waist_kds_.size()
            << " target="
            << arm_waist_target_.size()
            << std::endl;

        return false;
    }

    // ============================================================
    // 4) ENABLE SAFE ARM / WAIST MOTOR CHANNELS
    // ============================================================
    //
    // Forbidden EDU23 joints are skipped.
    //
    // This allows the official YAML to contain full G1 arm/waist
    // entries while still protecting the EDU23 hardware.
    //
    // ============================================================

    for (size_t i = 0; i < arm_size; ++i)
    {
        const int ms =
            arm_waist_joint2motor_idx_[i];

        if (ms < 0 || ms >= kMsTotal)
        {
            std::cerr
                << "[WARN] Ignoring arm/waist motor index out of range: "
                << ms
                << std::endl;

            continue;
        }

        if (IsForbiddenEdu23MotorIndex(ms))
        {
            std::cerr
                << "[WARN] Ignoring forbidden EDU23 arm/waist motor index: "
                << ms
                << std::endl;

            continue;
        }

        enabled_ms_mask_[ms] =
            true;
    }

    // ============================================================
    // 5) FINALIZE POLICY DEFAULT POSTURE
    // ============================================================
    //
    // GetState_minimal() uses policy_default_dof_pos_ to provide a
    // valid fallback state before LowState is received.
    //
    // For EDU12, this default posture is simply the 12-leg default
    // angle vector from the official YAML.
    //
    // ============================================================

    policy_default_dof_pos_ =
        leg_default_angles_;

    // ============================================================
    // 6) MARK MAPPING READY
    // ============================================================

    policy_table_ready_ =
        true;

    std::cout
        << "[INFO] EDU12 mapping ready: 12 leg actions mapped"
        << std::endl;

    DumpFullMappingOnce();

    return true;
}

/**
 * @brief Print the final 12-action policy mapping once.
 *
 * Called by:
 *
 *   BuildLegPolicyToMsTable()
 *
 * Purpose:
 * ------------------------------------------------------------
 *  - Helps verify that the policy action order matches the motor
 *    order expected by the official Unitree policy.
 *  - Prints only once to avoid log flooding.
 *
 * This is a diagnostic method only.
 */
void RL_Real_G1_EDU12::DumpFullMappingOnce() const
{
    static bool done = false;

    if (done)
        return;

    done = true;

    std::cout
        << "\n=== EDU12 Leg Mapping Debug ===\n";

    std::cout
        << "policy_leg_index | motor_index\n";

    std::cout
        << "-------------------------------\n";

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        std::cout
            << p
            << " | "
            << leg_policy_to_ms_[p]
            << "\n";
    }

    std::cout
        << "===============================\n\n";
}
///////////////////////////////////////////////////////////////////////////////////////////////////////

// ============================================================
// SECTION 5 — LOWSTATE ACQUISITION / ROBOT STATE READING
// ============================================================
//
// This section is responsible for reading the physical robot state
// and converting it into the 12-DoF policy state expected by the
// official Unitree locomotion policy.
//
// IMPORTANT CONCEPT
// ------------------------------------------------------------
// Unitree LowState is expressed in hardware motor space.
//
// The RL policy does NOT consume the full hardware state.
// It consumes only:
//
//   12 leg joint positions
//   12 leg joint velocities
//
// Therefore:
//
//   LowState motor space
//       ↓
//   leg_policy_to_ms_ mapping
//       ↓
//   12-DoF policy state
//
// Main call chain:
//
//   DDS LowState callback
//       ↓
//   LowStateHandler()
//       ↓
//   stores latest LowState safely
//
//   RobotControl()
//       ↓
//   GetState()
//       ↓
//   GetState_real()
//       ↓
//   robot_state.motor_state.q/dq/tau_est
//
// ============================================================

/**
 * @brief Callback executed whenever a new Unitree LowState message arrives.
 *
 * Called by:
 *  - Unitree DDS subscriber initialized in the constructor.
 *
 * Responsibilities:
 *  1. Copy the LowState message into local class storage.
 *  2. Update the LowState freshness timestamp.
 *  3. Mark that at least one LowState has been received.
 *  4. Optionally publish ROS /joint_states.
 *  5. Save a safe posture for stand-hold recovery.
 *
 * Why this method does not run RL:
 * ------------------------------------------------------------
 * DDS callbacks should remain short and deterministic.
 *
 * RL inference is executed separately by:
 *
 *   loop_rl → RunModel()
 *
 * Robot control is executed separately by:
 *
 *   loop_control → RobotControl()
 */

 #if defined(USE_ROS2) && defined(USE_ROS)

/**
 * @brief Publish ROS2 JointState for robot_state_publisher.
 *
 * Called by:
 *  - LowStateHandler()
 *
 * Importance:
 * ------------------------------------------------------------
 * This method publishes the active EDU23 joint states to ROS2.
 *
 * IMPORTANT:
 * ------------------------------------------------------------
 * The RL policy uses only 12 leg actions, but the robot visualization
 * still requires the full active EDU23 joint layout.
 *
 * Therefore this method must NOT use:
 *
 *   leg_policy_to_ms_
 *
 * because that mapping only covers the 12 RL-controlled leg joints.
 *
 * Instead, it uses:
 *
 *   BuildEdu23JointStateMotorOrder()
 *
 * which returns the 23 active EDU23 motor indices in the same order
 * as tf_joint_names_.
 */

void RL_Real_G1_EDU12::PublishJointStates()
{
    if (!this->joint_state_pub_)
        return;

    if (!this->got_lowstate.load(std::memory_order_relaxed))
        return;

    LowState_ low_state_copy;

    {
        std::lock_guard<std::mutex> lk(this->lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    const auto& ms_arr = low_state_copy.motor_state();
    const int ms_size = static_cast<int>(ms_arr.size());

    const std::vector<int> ms_order =
        BuildEdu23JointStateMotorOrder();

    sensor_msgs::msg::JointState js;

    js.header.stamp =
        this->ros2_node->now();

    js.name =
        this->tf_joint_names_;

    js.position.reserve(this->tf_joint_names_.size());
    js.velocity.reserve(this->tf_joint_names_.size());
    js.effort.reserve(this->tf_joint_names_.size());

    for (int ms : ms_order)
    {
        if (ms < 0 || ms >= ms_size)
        {
            js.position.push_back(0.0);
            js.velocity.push_back(0.0);
            js.effort.push_back(0.0);
            continue;
        }

        js.position.push_back(ms_arr[ms].q());
        js.velocity.push_back(ms_arr[ms].dq());
        js.effort.push_back(ms_arr[ms].tau_est());
    }

    if (js.name.size() != js.position.size())
    {
        RCLCPP_WARN_THROTTLE(
            this->ros2_node->get_logger(),
            *this->ros2_node->get_clock(),
            2000,
            "JointState size mismatch: names=%zu positions=%zu",
            js.name.size(),
            js.position.size()
        );
        return;
    }

    this->joint_state_pub_->publish(js);
}

#endif

void RL_Real_G1_EDU12::LowStateHandler(const void *message)
{
    if (!message)
        return;

    const auto now =
        std::chrono::steady_clock::now();

    const auto* low_state =
        static_cast<const LowState_*>(message);

    if (!low_state)
        return;

    /*
     * Copy the DDS message immediately locally.
     *
     * This avoids using a raw DDS pointer after the callback scope.
     */
    LowState_ local_low_state =
        *low_state;

    /*
     * Store the latest LowState safely.
     *
     * GetState_real() later reads this copy from another loop.
     */
    {
        std::lock_guard<std::mutex> lk(lowstate_mtx_);
        this->unitree_low_state =
            local_low_state;
    }

    last_lowstate_tp_ =
        now;

    lowstate_time_initialized_ =
        true;

    got_lowstate.store(
        true,
        std::memory_order_relaxed
    );

    #if defined(USE_ROS2) && defined(USE_ROS)

        /*
        * Publish /joint_states for visualization and TF.
        *
        * This is independent from the RL policy dimension.
        * The policy uses 12 joints, but TF still needs the active
        * EDU23 layout.
        */
        this->PublishJointStates();

    #endif

    /*
     * Stand-hold posture capture.
     *
     * This is used only after the policy mapping is ready and the
     * robot is in a normal running state.
     */
    if (!policy_table_ready_)
        return;

    if (safety_state_ != RLSafetyState::RUNNING)
        return;

    const auto& ms_arr =
        local_low_state.motor_state();

    const int ms_size =
        static_cast<int>(ms_arr.size());

    if (ms_size <= 0)
        return;

    // ------------------------------------------------------------
    // 1) IMU sanity check.
    //
    // If the quaternion is invalid or the robot appears heavily
    // tilted, we do not store this posture as a safe recovery target.
    // ------------------------------------------------------------
    const auto& imu =
        local_low_state.imu_state();

    const float qw = imu.quaternion()[0];
    const float qx = imu.quaternion()[1];
    const float qy = imu.quaternion()[2];
    const float qz = imu.quaternion()[3];

    const float q_norm =
        std::sqrt(
            qw * qw +
            qx * qx +
            qy * qy +
            qz * qz
        );

    if (q_norm < 0.8f || q_norm > 1.2f)
        return;

    // Simple roll/pitch instability proxy.
    // If qx/qy are too large, the robot may already be tilted.
    const float roll_pitch_energy =
        qx * qx + qy * qy;

    if (roll_pitch_energy > 0.15f)
        return;

    // ------------------------------------------------------------
    // 2) Joint velocity sanity check.
    //
    // If a joint is moving too fast, the posture is not considered
    // safe enough for stand-hold recovery.
    // ------------------------------------------------------------

    float max_abs_dq =
        0.0f;

    for (int ms = 0;
         ms < ms_size && ms < kMsTotal;
         ++ms)
    {
        if (!enabled_ms_mask_[ms])
            continue;

        max_abs_dq =
            std::max(
                max_abs_dq,
                std::abs(ms_arr[ms].dq())
            );
    }

    if (max_abs_dq > 8.0f)
        return;

    // ------------------------------------------------------------
    // 3) Save last valid posture for Stand-Hold
    //
    // This posture can later be used by:
    //
    //   SendStandHoldCommandIfPossible()
    //
    // during LowState loss or safety recovery.
    // ------------------------------------------------------------

    {
        std::lock_guard<std::mutex> lk(stand_hold_mtx_);

        for (int ms = 0;
             ms < ms_size && ms < kMsTotal;
             ++ms)
        {
            if (!enabled_ms_mask_[ms])
                continue;

            last_good_ms_q_[ms] =
                ms_arr[ms].q();

            last_good_ms_dq_[ms] =
                ms_arr[ms].dq();

            last_good_ms_valid_[ms] =
                true;
        }

        stand_hold_ready_ =
            true;
    }
}

/**
 * @brief Store the optional torso IMU message.
 *
 * Called by:
 *  - Unitree DDS IMU subscriber.
 *
 * Importance:
 * ------------------------------------------------------------
 * This method simply stores an additional IMU source.
 *
 * The main policy currently reads the IMU from LowState, but keeping
 * this subscriber allows future improvements such as:
 *  - torso/pelvis IMU selection,
 *  - IMU comparison,
 *  - filtering,
 *  - diagnostics.
 */
void RL_Real_G1_EDU12::ImuTorsoHandler(const void *message)
{
    if (!message)
        return;

    this->unitree_imu_torso =
        *(const IMUState_*)message;
}


/**
 * @brief Check whether the latest LowState is still fresh.
 *
 * Called by:
 *  - RobotControl()
 *  - UpdateSafetyState()
 *  - debug logs
 *
 * Importance:
 * ------------------------------------------------------------
 * The RL controller must not run on stale robot state.
 *
 * If LowState becomes too old, the safety state machine can switch
 * to stand-hold or damping mode.
 */
bool RL_Real_G1_EDU12::IsLowStateFresh() const
{
    if (!lowstate_time_initialized_)
        return false;

    const auto now =
        std::chrono::steady_clock::now();

    const double age =
        std::chrono::duration<double>(
            now - last_lowstate_tp_
        ).count();

    return age <= lowstate_timeout_sec_;
}


/**
 * @brief Build a minimal safe 12-DoF robot state.
 *
 * Called by:
 *  - GetState()
 *  - GetState_real()
 *  - RobotControl() during startup
 *  - safety fallback paths
 *
 * Importance:
 * ------------------------------------------------------------
 * This method guarantees that the RL/FSM always receives a valid
 * state vector, even before real LowState data is available.
 *
 * For EDU12:
 *  - motor_state size is 12
 *  - default posture comes from official Unitree default_angles
 *  - if defaults are unavailable, zeros are used
 */
void RL_Real_G1_EDU12::GetState_minimal(
    RobotState<float>* state) const
{
    if (!state)
        return;
    // IMU stable
    state->imu.quaternion =
        {1.0f, 0.0f, 0.0f, 0.0f};

    state->imu.gyroscope =
        {0.0f, 0.0f, 0.0f};

    state->motor_state.resize(
        kNdofPolicy
    );
    
    // If policy defaults available use them, else zeros
    const bool have_default =
        ((int)policy_default_dof_pos_.size()
            == kNdofPolicy);

    for (int i = 0; i < kNdofPolicy; ++i)
    {
        state->motor_state.q[i] =
            have_default
                ? policy_default_dof_pos_[i]
                : 0.0f;

        state->motor_state.dq[i] =
            0.0f;

        state->motor_state.tau_est[i] =
            0.0f;
    }
}

/**
 * @brief Read real robot state during GetUp before policy mapping is ready.
 *
 * Called by:
 *  - RobotControl() during early startup.
 *
 * Why this method exists:
 * ------------------------------------------------------------
 * During GetUp, the policy mapping may not be ready yet.
 *
 * However, using the real IMU can still improve posture awareness.
 *
 * Therefore this method:
 *  - starts from GetState_minimal(),
 *  - copies only IMU data from LowState,
 *  - does NOT read joint mappings.
 */
void RL_Real_G1_EDU12::GetState_real_for_getup(
    RobotState<float>* state)
{
    if (!state)
        return;

    GetState_minimal(state);

    if (!got_lowstate.load(std::memory_order_relaxed))
        return;

    LowState_ low_state_copy;

    {
        std::lock_guard<std::mutex> lk(lowstate_mtx_);
        low_state_copy =
            this->unitree_low_state;
    }

    const auto& imu =
        low_state_copy.imu_state();

    for (int i = 0; i < 4; ++i)
        state->imu.quaternion[i] =
            imu.quaternion()[i];

    for (int i = 0; i < 3; ++i)
        state->imu.gyroscope[i] =
            imu.gyroscope()[i];
}

/**
 * @brief Dispatch the appropriate state-reading method.
 *
 * Called by:
 *  - RobotControl()
 *
 * Method selection:
 * ------------------------------------------------------------
 *  - FAKE_LOWSTATE:
 *      GetState_test()
 *
 *  - REAL:
 *      GetState_real()
 *
 *  - DRY_RUN:
 *      GetState_real()
 *      because we still want to read LowState if available
 *
 *  - CMD_VEL_ONLY:
 *      minimal or real state, depending on debug needs
 */
void RL_Real_G1_EDU12::GetState(
    RobotState<float>* state)
{
    GetState_minimal(state);

    switch (hw_mode_)
    {
        case HwMode::FAKE_LOWSTATE:
            GetState_test(state); // overwrite fake values
            break;

        case HwMode::REAL:
            GetState_real(state); // overwrite only mapped joints + real IMU
            break;

        case HwMode::CMD_VEL_ONLY:
        case HwMode::DRY_RUN:
            GetState_real(state);
            break;

        default:
            // keep minimal
            break;
    }
}

/**
 * @brief Convert Unitree LowState into a 12-DoF policy state.
 *
 * Called by:
 *  - GetState()
 *
 * Importance:
 * ------------------------------------------------------------
 * This is the main method that translates hardware state into
 * RL policy state.
 *
 * It uses:
 *
 *   leg_policy_to_ms_
 *
 * to map:
 *
 *   policy leg index → Unitree motor index
 *
 * It fills:
 *  - state->motor_state.q
 *  - state->motor_state.dq
 *  - state->motor_state.tau_est
 *
 * only for the 12 policy-controlled leg joints.
 */
void RL_Real_G1_EDU12::GetState_real(
    RobotState<float>* state)
{
    if (!state)
        return;

    #ifdef CMD_VEL_ONLY
        // Debug mode: do not read real hardware.
        GetState_minimal(state);
        return;
    #endif

    // Mapping must be ready before reading hardware joints.
    if (!policy_table_ready_)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::cerr << "[ERROR] GetState_real(): mapping not ready\n";
        }

        GetState_minimal(state);
        return;
    }
    // Start from safe default values.
    GetState_minimal(state);

    // If no LowState has arrived yet, keep defaults.
    if (!this->got_lowstate.load(std::memory_order_relaxed))
        return;

    // Copy LowState safely once, then use only the local copy.
    LowState_ low_state_copy;
    {
        std::lock_guard<std::mutex> lk(lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    // Update machine mode.
    if (this->mode_machine != low_state_copy.mode_machine())
    {
        static bool printed = false;

        if (!printed)
        {
            std::cout << "G1 type: "
                      << unsigned(low_state_copy.mode_machine())
                      << std::endl;
            printed = true;
        }

        this->mode_machine = low_state_copy.mode_machine();
    }
    // Parse wireless remote.
    std::memcpy(
        this->remote_data_rx.buff,
        &low_state_copy.wireless_remote()[0],
        40
    );

    this->gamepad.update(this->remote_data_rx.RF_RX);

    this->control.x   = this->gamepad.ly;
    this->control.y   = -this->gamepad.lx;
    this->control.yaw = -this->gamepad.rx;

    // Copy IMU.
    const auto& imu = low_state_copy.imu_state();

    for (int i = 0; i < 4; ++i)
        state->imu.quaternion[i] = imu.quaternion()[i];

    for (int i = 0; i < 3; ++i)
        state->imu.gyroscope[i] = imu.gyroscope()[i];

    // Copy valid motor states using policy -> motor_state mapping.
    const auto& ms_arr = low_state_copy.motor_state();
    const int ms_size = static_cast<int>(ms_arr.size());

    static bool warned_oob = false;

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        const int ms = leg_policy_to_ms_[p];

        // Joint absent on EDU23.
        if (ms < 0)
            continue;

        // Invalid mapping protection.
        if (ms >= ms_size)
        {
            if (!warned_oob)
            {
                warned_oob = true;
                std::cerr << "[ERROR] GetState_real(): policy_to_ms_ out of range. "
                          << "p=" << p
                          << " ms=" << ms
                          << " ms_size=" << ms_size
                          << "\n";
            }
            continue;
        }

        state->motor_state.q[p]       = ms_arr[ms].q();
        state->motor_state.dq[p]      = ms_arr[ms].dq();
        state->motor_state.tau_est[p] = ms_arr[ms].tau_est();
    }
}

// ----------------------------
// GetState_test (FAKE_LOWSTATE_TEST)
// ----------------------------
void RL_Real_G1_EDU12::GetState_test(RobotState<float> *state)
{
    const int ndof = this->params.Get<int>("num_of_dofs", 29);
    const auto jm_policy = this->params.Get<std::vector<int>>("joint_mapping", {});
    state->motor_state.resize(ndof);

    // Initialize fake hw arrays once
    static const int hw_size = 35;

    if (!fake_inited)
    {
        fake_inited = true;
        fake_hw_q.resize(hw_size);
        fake_hw_dq.resize(hw_size);
        fake_hw_tau.resize(hw_size);

        for (int k = 0; k < hw_size; ++k)
        {
            fake_hw_q[k]   = 1000.f + k;
            fake_hw_dq[k]  = 2000.f + k;
            fake_hw_tau[k] = 3000.f + k;
        }

        //this->got_lowstate.store(true);
        this->got_lowstate.store(true, std::memory_order_relaxed);

        std::cout << "[FAKE_LOWSTATE_TEST] Synthetic LowState ready (hw_size=" << hw_size << ")\n";
    }

    // IMU stable
    state->imu.quaternion = {1.f, 0.f, 0.f, 0.f};
    state->imu.gyroscope  = {0.f, 0.f, 0.f};

    // Fill motor_state using policy mapping: policy_i -> hw index jm_policy[i]
    for (int i = 0; i < ndof; ++i)
    {
        int hw = (i < (int)jm_policy.size()) ? jm_policy[i] : -1;
        if (hw < 0 || hw >= hw_size)
        {
            state->motor_state.q[i]       = 0.f;
            state->motor_state.dq[i]      = 0.f;
            state->motor_state.tau_est[i] = 0.f;
        }
        else
        {
            state->motor_state.q[i]       = fake_hw_q[hw];
            state->motor_state.dq[i]      = fake_hw_dq[hw];
            state->motor_state.tau_est[i] = fake_hw_tau[hw];
        }
    }

    // Print mapping check only once
    static bool printed = false;
    if (!printed)
    {
        printed = true;
        std::cout << "\n===== TEST: policy->HW mapping check =====\n";
        std::cout << "i | jm_policy[i] | expected_q | got_q\n";
        for (int i = 0; i < std::min(ndof, 29); ++i)
        {
            int hw = (i < (int)jm_policy.size()) ? jm_policy[i] : -1;
            float expected = (hw >= 0) ? (1000.0f + hw) : -1.0f;
            float got = state->motor_state.q[i];
            std::cout << i << " | " << hw << " | " << expected << " | " << got << "\n";
        }
        std::cout << "=========================================\n\n";
    }
}
// ============================================================
// SECTION 6 — OBSERVATION / RL INPUT PREPARATION
// ============================================================
//
// This section prepares all command-related inputs used by the
// locomotion policy observation.
//
// IMPORTANT DESIGN IDEA
// ------------------------------------------------------------
// This wrapper does NOT directly assemble the final observation
// tensor used by motion.pt.
//
// The final neural observation vector is built later by:
//
//   ComputeObservation()
//
// from the RL base class.
//
// However, before ComputeObservation() can run, this wrapper must
// provide valid runtime data:
//
//   obs.commands
//   obs.dof_pos
//   obs.dof_vel
//   obs.ang_vel
//   obs.base_quat
//   obs.actions
//
// ============================================================
//
// COMMAND FLOW
// ------------------------------------------------------------
//
// ROS2 PATH:
//
//   /cmd_vel
//       ↓
//   CmdvelCallback()
//       ↓
//   this->cmd_vel_
//   this->last_cmd_vel_time_
//       ↓
//   GetFreshCmdVel()
//       ↓
//   RunModel()
//       ↓
//   obs.commands
//       ↓
//   ComputeObservation()
//       ↓
//   motion.pt
//
// ============================================================
//
// UNITREE-ONLY PATH:
//
//   UDP bridge
//       ↓
//   StartIpcCmdVelReceiver()
//       ↓
//   this->ipc_cmd_
//       ↓
//   GetFreshIpcCmdVel()
//       ↓
//   RunModel()
//       ↓
//   obs.commands
//
// ============================================================

#if defined(USE_ROS)

/**
 * @brief Receive external velocity commands from ROS.
 *
 * Called by:
 *  - ROS subscriber created in the constructor.
 *
 * Importance:
 * ------------------------------------------------------------
 * This callback is the main ROS entry point for:
 *  - Nav2,
 *  - teleoperation,
 *  - joystick velocity commands.
 *
 * The command is NOT directly sent to the robot.
 *
 * Instead:
 *
 *   CmdvelCallback()
 *       ↓
 *   stores latest command safely
 *       ↓
 *   GetFreshCmdVel()
 *       ↓
 *   RunModel()
 *       ↓
 *   obs.commands
 *
 * Thread safety:
 * ------------------------------------------------------------
 * RunModel() and the ROS callback may execute concurrently.
 *
 * Therefore:
 *  - cmd_vel_,
 *  - last_cmd_vel_time_,
 *
 * are protected by cmd_vel_mtx_.
 *
 * Why timestamps matter:
 * ------------------------------------------------------------
 * A stale velocity command should never continue influencing
 * locomotion forever.
 *
 * GetFreshCmdVel() later validates:
 *
 *   now - last_cmd_vel_time_
 *
 * before using the command.
 */
void RL_Real_G1_EDU12::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    if (!msg)
        return;

#if defined(USE_ROS2) && defined(USE_ROS)

    // ========================================================
    // ROS2 THREAD-SAFE COMMAND STORAGE
    // ========================================================

    {
        std::lock_guard<std::mutex> lk(
            this->cmd_vel_mtx_
        );

        this->cmd_vel_ =
            *msg;

        this->last_cmd_vel_time_ =
            this->ros2_node->now();
    }

    // ========================================================
    // DEBUG LOGGING
    // ========================================================

    RCLCPP_INFO_THROTTLE(
        this->ros2_node->get_logger(),
        *this->ros2_node->get_clock(),
        200,
        "[CMD_VEL RX] vx=%.3f vy=%.3f wz=%.3f",
        msg->linear.x,
        msg->linear.y,
        msg->angular.z
    );

#else

    // ========================================================
    // ROS1 COMMAND STORAGE
    // ========================================================

    this->cmd_vel_ =
        *msg;

    static int cnt = 0;

    if ((cnt++ % 10) == 0)
    {
        std::cout
            << "[CMD_VEL RX] vx="
            << this->cmd_vel_.linear.x
            << " vy="
            << this->cmd_vel_.linear.y
            << " wz="
            << this->cmd_vel_.angular.z
            << std::endl;
    }

#endif
}

/**
 * @brief Return a fresh velocity command from ROS if available.
 *
 * Called by:
 *  - RunModel()
 *
 * Importance:
 * ------------------------------------------------------------
 * The locomotion policy receives:
 *
 *   vx, vy, wz
 *
 * as part of its observation.
 *
 * These values may originate from:
 *  - Nav2,
 *  - teleoperation,
 *  - joystick,
 *  - external planners.
 *
 * This method ensures:
 *  - the command exists,
 *  - the command is recent enough,
 *  - stale commands are rejected.
 *
 * Return value:
 * ------------------------------------------------------------
 * true:
 *   A valid fresh command is available.
 *
 * false:
 *   No valid command should be used.
 */
bool RL_Real_G1_EDU12::GetFreshCmdVel(
    float& vx,
    float& vy,
    float& wz) const
{
#if defined(USE_ROS2) && defined(USE_ROS)

    geometry_msgs::msg::Twist cv;
    rclcpp::Time last_t;

    {
        std::lock_guard<std::mutex> lk(
            this->cmd_vel_mtx_
        );

        cv =
            this->cmd_vel_;

        last_t =
            this->last_cmd_vel_time_;
    }

    if (last_t.nanoseconds() == 0)
        return false;

    const auto now =
        this->ros2_node->now();

    const double age =
        (now - last_t).seconds();

    if (age > this->cmd_vel_timeout_sec_)
        return false;

    vx =
        static_cast<float>(cv.linear.x);

    vy =
        static_cast<float>(cv.linear.y);

    wz =
        static_cast<float>(cv.angular.z);

    return true;

#elif defined(USE_ROS1) && defined(USE_ROS)

    vx =
        static_cast<float>(
            this->cmd_vel_.linear.x
        );

    vy =
        static_cast<float>(
            this->cmd_vel_.linear.y
        );

    wz =
        static_cast<float>(
            this->cmd_vel_.angular.z
        );

    return true;

#else

    (void)vx;
    (void)vy;
    (void)wz;

    return false;

#endif
}

#endif

#ifndef USE_ROS

/**
 * @brief Parse runtime arguments in Unitree-only mode.
 *
 * Called by:
 *  - constructor
 *
 * Importance:
 * ------------------------------------------------------------
 * Unitree-only mode does not use ROS parameters.
 *
 * Therefore runtime behavior must be configured through:
 *
 *   argv
 *
 * Example:
 *
 *   ros2 run rl_sar rl_real_g1_edu12_unitree eth0 --dry-run
 *
 * Supported options:
 * ------------------------------------------------------------
 *   --real
 *   --dry-run
 *   --udp-port=15000
 */
void RL_Real_G1_EDU12::ParseUnitreeOnlyArgs(int argc, char** argv)
{
    // Default: REAL mode
    hw_mode_ = HwMode::REAL;
    publish_lowcmd_ = true;
    fake_rl_full_ = false;
    cmd_vel_only_ = false;

    cmd_vel_timeout_sec_ = 0.3;
    lowstate_timeout_sec_ = 0.80;
    recovery_duration_sec_ = 1.0;
    stand_hold_max_sec_ = 5.0;

    links_loss_confirm_sec_ = 0.35;

    navigation_mode_param_ = true;
    navigation_mode_force_ = true;

    require_laptop_heartbeat_ = false;

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "--real" || arg == "real" || arg == "0" || arg == "--hw-mode=0")
        {
            hw_mode_ = HwMode::REAL;
            publish_lowcmd_ = true;
            std::cout << "[UNITREE-ONLY] Mode: REAL" << std::endl;
        }
        else if (arg == "--dry-run" || arg == "dry_run" || arg == "dry-run" ||
                 arg == "3" || arg == "--hw-mode=3")
        {
            hw_mode_ = HwMode::DRY_RUN;
            publish_lowcmd_ = false;
            std::cout << "[UNITREE-ONLY] Mode: DRY_RUN" << std::endl;
        }
        else if (arg.rfind("--udp-port=", 0) == 0)
        {
            ipc_cmd_port_ =
                std::stoi(arg.substr(std::string("--udp-port=").size()));

            std::cout << "[UNITREE-ONLY] UDP port: "
                      << ipc_cmd_port_ << std::endl;
        }
        else
        {
            std::cout << "[UNITREE-ONLY] Unknown argument ignored: "
                      << arg << std::endl;
        }
    }

    std::cout << "[UNITREE-ONLY] hw_mode="
              << static_cast<int>(hw_mode_)
              << " publish_lowcmd="
              << static_cast<int>(publish_lowcmd_)
              << " udp_port="
              << ipc_cmd_port_
              << std::endl;
}

/**
 * @brief Start UDP receiver for external velocity commands.
 *
 * Called by:
 *  - constructor
 *
 * Importance:
 * ------------------------------------------------------------
 * Unitree-only mode has no ROS subscriber inside the process.
 *
 * Therefore velocity commands are received through:
 *
 *   UDP localhost bridge
 *
 * Architecture:
 *
 *   Nav2 / ROS2 process
 *       ↓
 *   UDP bridge
 *       ↓
 *   rl_real_g1_edu12_unitree
 *       ↓
 *   RunModel()
 */
void RL_Real_G1_EDU12::StartIpcCmdVelReceiver(int port)
{
    ipc_running_.store(true);

    ipc_cmd_thread_ = std::thread([this, port]()
    {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0)
        {
            std::cerr << "[IPC] Failed to create UDP socket" << std::endl;
            return;
        }

        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(port);

        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        {
            std::cerr << "[IPC] Failed to bind UDP port " << port << std::endl;
            close(sock);
            return;
        }

        std::cout << "[IPC] cmd_vel UDP receiver listening on 127.0.0.1:"
                  << port << std::endl;

        char buffer[256];

        while (ipc_running_.load())
        {
            sockaddr_in src{};
            socklen_t src_len = sizeof(src);

            ssize_t n = recvfrom(
                sock,
                buffer,
                sizeof(buffer) - 1,
                0,
                reinterpret_cast<sockaddr*>(&src),
                &src_len
            );

            if (n > 0)
            {
                buffer[n] = '\0';

                std::istringstream iss(buffer);
                float vx = 0.0f;
                float vy = 0.0f;
                float wz = 0.0f;

                if (iss >> vx >> vy >> wz)
                {
                    std::lock_guard<std::mutex> lk(ipc_cmd_mtx_);

                    ipc_cmd_.vx = vx;
                    ipc_cmd_.vy = vy;
                    ipc_cmd_.wz = wz;
                    ipc_cmd_.stamp = std::chrono::steady_clock::now();
                    ipc_cmd_.initialized = true;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        close(sock);
    });
}

/**
 * @brief Stop UDP velocity command receiver.
 *
 * Called by:
 *  - destructor
 *
 * Importance:
 * ------------------------------------------------------------
 * Ensures clean shutdown of the background UDP thread.
 */
void RL_Real_G1_EDU12::StopIpcCmdVelReceiver()
{
    this->ipc_running_.store(false);

    if (this->ipc_cmd_thread_.joinable())
    {
        this->ipc_cmd_thread_.join();
    }
}

/**
 * @brief Return fresh UDP velocity command if available.
 *
 * Called by:
 *  - RunModel()
 *
 * Importance:
 * ------------------------------------------------------------
 * This is the Unitree-only equivalent of:
 *
 *   GetFreshCmdVel()
 *
 * It provides:
 *
 *   vx, vy, wz
 *
 * to the locomotion policy observation.
 */
bool RL_Real_G1_EDU12::GetFreshIpcCmdVel(float& vx, float& vy, float& wz) const
{
    std::lock_guard<std::mutex> lk(this->ipc_cmd_mtx_);

    if (!this->ipc_cmd_.initialized)
        return false;

    const double age =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - this->ipc_cmd_.stamp
        ).count();

    if (age > this->cmd_vel_timeout_sec_)
        return false;

    vx = this->ipc_cmd_.vx;
    vy = this->ipc_cmd_.vy;
    wz = this->ipc_cmd_.wz;

    return true;
}

#endif

// ============================================================
// SECTION 7 — RL MODEL EXECUTION
// ============================================================
//
// This section executes the neural locomotion policy.
//
// IMPORTANT CONCEPT
// ------------------------------------------------------------
// The official Unitree policy is not called directly from the
// control loop.
//
// Instead, the control architecture is split into two loops:
//
//   loop_control
//       ↓
//   RobotControl()
//
// and:
//
//   loop_rl
//       ↓
//   RunModel()
//
// RobotControl() handles:
//  - FSM execution
//  - safety checks
//  - LowCmd publication
//
// RunModel() handles:
//  - observation field preparation
//  - neural network inference
//  - action post-processing
//
// This separation keeps control logic and model inference isolated.
//
// ============================================================
//
// MAIN CALL CHAIN
// ------------------------------------------------------------
//
//   loop_rl
//       ↓
//   RunModel()
//       ↓
//   update obs fields
//       ↓
//   Forward()
//       ↓
//   ComputeObservation()
//       ↓
//   model->forward()
//       ↓
//   ComputeOutput()
//       ↓
//   output_dof_pos / output_dof_vel / output_dof_tau
//
// Then later:
//
//   loop_control
//       ↓
//   RobotControl()
//       ↓
//   SetCommand()
//
// ============================================================

/**
 * @brief Main RL inference loop.
 *
 * Called by:
 *  - loop_rl
 *
 * This method is responsible for preparing the RL observation and
 * executing the neural network policy.
 *
 * It does NOT directly publish LowCmd.
 *
 * LowCmd publication is handled later by:
 *
 *   RobotControl()
 *       ↓
 *   SetCommand()
 *
 * Responsibilities:
 * ------------------------------------------------------------
 *  1. Stop immediately if safety blocks RL.
 *  2. Update observation fields from robot_state.
 *  3. Read velocity command from joystick, ROS2, or UDP.
 *  4. Write command into obs.commands.
 *  5. Run Forward().
 *  6. Convert policy actions into output joint commands.
 *  7. Push outputs into queues used by the control loop.
 *
 * Important EDU12 behavior:
 * ------------------------------------------------------------
 *  - obs.dof_pos has size 12.
 *  - obs.dof_vel has size 12.
 *  - obs.actions has size 12.
 *  - ComputeOutput() must therefore operate in 12-DoF policy space.
 */

void RL_Real_G1_EDU12::RunModel()
{
    if (!this->policy_table_ready_ || !this->rl_init_done)
    {
        return;
    }
    // ========================================================
    // 0) SAFETY GATE
    // ========================================================
    //
    // If the robot is in REAL mode and the safety state is not
    // RUNNING, the policy must not generate locomotion commands.
    //
    // This prevents stale commands from being reused after a
    // communication or LowState failure.
    //
    // ========================================================

    if (hw_mode_ == HwMode::REAL &&
        safety_state_ != RLSafetyState::RUNNING)
    {
        ClearCmdVelUnsafe();
        this->obs.commands = {0.0f, 0.0f, 0.0f};
        return;
    }

    // ========================================================
    // 1) FAST EXIT CONDITIONS
    // ========================================================
    //
    // These guards prevent inference when the system is not ready.
    //
    // cmd_vel_only_:
    //   Used only to verify velocity command reception.
    //
    // rl_init_done:
    //   Set by InitRL() after the model and configuration are loaded.
    //
    // FAKE_LOWSTATE with fake_rl_full_ == false:
    //   Allows testing mapping without running the model.
    //
    // ========================================================
    if (this->cmd_vel_only_) return;
    if (!this->rl_init_done) return;
    if (hw_mode_ == HwMode::FAKE_LOWSTATE && !fake_rl_full_) return;

    // ========================================================
    // 2) UPDATE BASIC OBSERVATION FIELDS
    // ========================================================
    //
    // robot_state is updated by:
    //
    //   RobotControl()
    //       ↓
    //   GetState()
    //       ↓
    //   GetState_real()
    //
    // RunModel() copies those values into obs.
    //
    // ComputeObservation() later reads obs and builds the final
    // neural network input tensor.
    //
    // ========================================================
    this->episode_length_buf += 1;

    this->obs.ang_vel   = this->robot_state.imu.gyroscope;
    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.dof_pos   = this->robot_state.motor_state.q;
    this->obs.dof_vel   = this->robot_state.motor_state.dq;


    // ========================================================
    // 3) SELECT VELOCITY COMMAND SOURCE
    // ========================================================
    //
    // Default source:
    //   gamepad / joystick / wireless remote
    //
    // Optional override:
    //   ROS2 /cmd_vel
    //
    // Unitree-only override:
    //   UDP command bridge
    //
    // The final selected command is written into:
    //
    //   obs.commands = {vx, vy, wz}
    //
    // ========================================================
    
    float vx = this->control.x;
    float vy = this->control.y;
    float wz = this->control.yaw;

#if defined(USE_ROS) && (defined(USE_ROS1) || defined(USE_ROS2))

    // ========================================================
    // ROS COMMAND SOURCE
    // ========================================================
    //
    // requested_nav_on:
    //   The system is allowed to use /cmd_vel.
    //
    // fresh:
    //   A recent /cmd_vel was actually received.
    //
    // nav_active:
    //   Navigation command is both requested and fresh.
    //
    // Safety rule:
    //   If navigation is requested but no fresh command exists,
    //   the policy receives zero velocity.
    //
    // ========================================================

    bool requested_nav_on =
    #if defined(USE_ROS2)
            (navigation_mode_force_ || this->control.navigation_mode);
    #else
            this->control.navigation_mode;
    #endif

    float vxf = 0.f;
    float vyf = 0.f;
    float wzf = 0.f;

    const bool fresh = GetFreshCmdVel(vxf, vyf, wzf);
    const bool nav_active = requested_nav_on && fresh;

    this->nav_cmd_fresh_ = nav_active;

    if (nav_active)
    {
            // Fresh Nav2/teleop command available.
            vx = vxf;
            vy = vyf;
            wz = wzf;
    }
    else if (requested_nav_on && !fresh)
    {
            // Navigation was requested, but command is stale or absent.
            // Keep the policy in idle locomotion, not in active navigation.
            vx = 0.f;
            vy = 0.f;
            wz = 0.f;
    }
    else
    {
            // Navigation not requested.
            // Keep joystick/gamepad values already stored in vx, vy, wz.
    }

    // Keep control struct coherent for FSM/debug/other code paths.
    this->control.x   = vx;
    this->control.y   = vy;
    this->control.yaw = wz;

#if defined(USE_ROS2)

    double age_dbg = -1.0;

    {
        std::lock_guard<std::mutex> lk(this->cmd_vel_mtx_);

        if (this->last_cmd_vel_time_.nanoseconds() != 0)
        {
            age_dbg =
                (ros2_node->now() - this->last_cmd_vel_time_).seconds();
        }
    }

    RCLCPP_INFO_THROTTLE(
        ros2_node->get_logger(),
        *ros2_node->get_clock(),
        1000 /*ms*/,
        "[NAV] requested=%d active=%d fresh=%d age=%.3fs cmd=[%.2f %.2f %.2f]",
        static_cast<int>(requested_nav_on),
        static_cast<int>(nav_active),
        static_cast<int>(fresh),
        age_dbg,
        vx,
        vy,
        wz
    );

#endif  // USE_ROS2

#endif  // USE_ROS

#ifndef USE_ROS

    // ========================================================
    // UNITREE-ONLY UDP COMMAND SOURCE
    // ========================================================
    //
    // In Unitree-only mode there is no ROS2 subscription inside
    // this process.
    //
    // External commands can be injected using the UDP bridge:
    //
    //   ROS2 Nav2
    //       ↓
    //   ros2_nav_bridge_node
    //       ↓
    //   UDP
    //       ↓
    //   GetFreshIpcCmdVel()
    //
    // ========================================================

    {
        float vxf = 0.0f;
        float vyf = 0.0f;
        float wzf = 0.0f;

        const bool fresh =
            GetFreshIpcCmdVel(
                vxf,
                vyf,
                wzf
            );

        const bool nav_requested =
            navigation_mode_force_ ||
            control.navigation_mode;

        this->nav_cmd_fresh_ = nav_requested && fresh;

        if (nav_requested && fresh)
        {
            // External navigation command is available.
            vx = vxf;
            vy = vyf;
            wz = wzf;
        }
        else if (nav_requested && !fresh)
        {
            // IMPORTANT:
            // Missing IPC/Nav2 command is NOT a locomotion fault.
            // Keep the RL policy alive in idle locomotion.
            vx = 0.0f;
            vy = 0.0f;
            wz = 0.0f;
        }
        else
        {
            // Navigation not requested.
            // Keep joystick/gamepad command already stored in vx, vy, wz.
        }

        this->control.x = vx;
        this->control.y = vy;
        this->control.yaw = wz;

        if (ShouldPrintEvery("ipc_nav", 3.0))
        {
            std::cout
                << "[IPC NAV] requested="
                << static_cast<int>(nav_requested)
                << " fresh="
                << static_cast<int>(fresh)
                << " cmd=["
                << vx
                << " "
                << vy
                << " "
                << wz
                << "] policy=ACTIVE_IDLE_IF_NO_CMD"
                << std::endl;
        }
    }

#endif

    // ========================================================
    // 4) WRITE COMMAND INTO OBSERVATION STRUCTURE
    // ========================================================
    //
    // The official Unitree policy expects command information as
    // part of its observation.
    //
    // Usually:
    //   obs.commands[0] = vx
    //   obs.commands[1] = vy
    //   obs.commands[2] = wz
    //
    // Scaling is handled later by ComputeObservation().
    //
    // ========================================================

    // ------------------------------------------------------------
// DimOS policy command adaptation
// ------------------------------------------------------------
// DimOS G1 policy does not feed raw command directly.
// In DimOS policy.py:
//   command[0] = command[0] * 2
//   command[1] = command[1] * 2
//   command[0] += -0.18
//   command[1] +=  0.00
//   command[2] += -0.09
//
// Here the values are YAML parameters so they can be tuned safely.
    this->obs.commands = {vx, vy, wz};

#ifndef USE_ROS

    if (ShouldPrintEvery("obs_cmd", 3.0))
    {
        std::cout
            << "[OBS.CMD] ["
            << this->obs.commands[0]
            << " "
            << this->obs.commands[1]
            << " "
            << this->obs.commands[2]
            << "]"
            << std::endl;
    }

#endif

#if defined(USE_ROS2) && defined(USE_ROS)

    RCLCPP_INFO_THROTTLE(
        ros2_node->get_logger(),
        *ros2_node->get_clock(),
        1000,
        "[OBS.CMD] [%.2f %.2f %.2f]",
        this->obs.commands[0],
        this->obs.commands[1],
        this->obs.commands[2]
    );

#endif

    // ========================================================
    // 5) RUN POLICY FORWARD PASS
    // ========================================================
    //
    // Forward() internally calls:
    //
    //   ComputeObservation()
    //       ↓
    //   model->forward()
    //
    // The result must be a 12-element action vector.
    //
    // ========================================================
    this->obs.actions = 
        this->Forward();

    // ========================================================
    // 6) CONVERT ACTIONS TO JOINT OUTPUTS
    // ========================================================
    //
    // ComputeOutput() converts raw neural actions into:
    //
    //   output_dof_pos
    //   output_dof_vel
    //   output_dof_tau
    //
    // These outputs are still in 12-DoF policy space.
    //
    // SetCommand() later maps them to Unitree motor indices.
    //
    // ========================================================


    this->ComputeOutput(
        this->obs.actions,
        this->output_dof_pos,
        this->output_dof_vel,
        this->output_dof_tau
    );

    // ========================================================
    // 7) PUSH OUTPUTS TO CONTROL QUEUES
    // ========================================================
    //
    // RobotControl() later consumes these outputs when sending
    // LowCmd.
    //
    // The queues decouple inference timing from control timing.
    //
    // ========================================================

    if (!this->output_dof_pos.empty()) output_dof_pos_queue.push(this->output_dof_pos);
    if (!this->output_dof_vel.empty()) output_dof_vel_queue.push(this->output_dof_vel);
    if (!this->output_dof_tau.empty()) output_dof_tau_queue.push(this->output_dof_tau);

    #ifdef CSV_LOGGER
        const std::vector<float> tau_est = this->robot_state.motor_state.tau_est;
        this->CSVLogger(
            this->output_dof_tau,
            tau_est,
            this->obs.dof_pos,
            this->output_dof_pos,
            this->obs.dof_vel
        );
    #endif
}

/**
 * @brief Execute the neural network forward pass.
 *
 * Called by:
 *  - RunModel()
 *
 * Responsibilities:
 * ------------------------------------------------------------
 *  1. Lock the model mutex.
 *  2. Build the final observation vector.
 *  3. Apply observation history if configured.
 *  4. Execute model->forward().
 *  5. Clamp actions if action limits exist.
 *
 * Why a mutex is used:
 * ------------------------------------------------------------
 * The model may be reset or reloaded during recovery.
 *
 * The mutex prevents inference from running while the model is in
 * an inconsistent state.
 *
 * Observation construction:
 * ------------------------------------------------------------
 * This method calls:
 *
 *   ComputeObservation()
 *
 * which builds the actual input vector expected by motion.pt.
 *
 * For the official Unitree policy, the final observation should be
 * compatible with:
 *
 *   num_obs: 47
 *
 * The typical composition is:
 *
 *   phase information
 *   angular velocity
 *   projected gravity
 *   command
 *   12 dof positions
 *   12 dof velocities
 *   12 previous actions
 *
 * Return:
 * ------------------------------------------------------------
 * A 12-element action vector.
 */

 // ----------------------------
// Forward (unchanged)
// ----------------------------
std::vector<float> RL_Real_G1_EDU12::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    if (!lock.owns_lock())
    {
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    if (this->debug_observation_)
    {
        static int obs_debug_counter = 0;

        if ((obs_debug_counter++ % 500) == 0)
        {
            std::cout << "\n=== EDU12 OBS DEBUG ===\n";
            std::cout << "obs_size=" << clamped_obs.size() << "\n";

            for (size_t i = 0; i < clamped_obs.size(); ++i)
            {
                std::cout
                    << "obs[" << i << "]="
                    << clamped_obs[i]
                    << "\n";
            }

            std::cout << "========================\n";
        }
    }

    std::vector<float> actions;

    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);

        this->history_obs =
            this->history_obs_buf.get_obs_vec(
                this->params.Get<std::vector<int>>(
                    "observations_history"
                )
            );

        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    // ------------------------------------------------------------
    // Debug raw network actions before any clamp.
    // ------------------------------------------------------------
    static int action_debug_counter = 0;
    const bool print_action_debug =
        this->debug_policy_output_ &&
        ((action_debug_counter++ % 500) == 0);

    if(this->debug_transition_)
    {
        if (print_action_debug)
        {
            std::cout << "\n=== EDU12 RAW ACTIONS BEFORE CLAMP ===\n";

            for (size_t i = 0; i < actions.size(); ++i)
            {
                std::cout << "a_raw[" << i << "]=" << actions[i] << "\n";
            }

            std::cout << "====================================\n";
        }
    }

    // ------------------------------------------------------------
    // Existing YAML clamp.
    // This keeps compatibility with rl_sar config.
    // ------------------------------------------------------------
    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() &&
        !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        actions =
            clamp(
                actions,
                this->params.Get<std::vector<float>>("clip_actions_lower"),
                this->params.Get<std::vector<float>>("clip_actions_upper")
            );
    }

    // ------------------------------------------------------------
    // Temporary hard safety clamp for EDU12 debugging.
    //
    // Purpose:
    //  - prevent previous-action feedback from exploding
    //  - keep obs.actions in a safe range
    //  - diagnose whether instability comes from large actions
    //
    // Start with 0.25.
    // If stable, later test 0.50, then 1.00.
    // ------------------------------------------------------------
    const float debug_action_limit = 0.25f;

    for (auto& a : actions)
    {
        if (a > debug_action_limit)
            a = debug_action_limit;

        if (a < -debug_action_limit)
            a = -debug_action_limit;
    }

    // ------------------------------------------------------------
    // Debug final actions after all clamps.
    // These are the values that will become obs.actions next cycle.
    // ------------------------------------------------------------
        if(this->debug_transition_)
    {
        if (print_action_debug)
        {
            std::cout << "\n=== EDU12 FINAL ACTIONS AFTER CLAMP ===\n";

            for (size_t i = 0; i < actions.size(); ++i)
            {
                std::cout << "a_final[" << i << "]=" << actions[i] << "\n";
            }

            std::cout << "======================================\n";
        }
    }
        return actions;
}


// ============================================================
// SECTION 8 — COMMAND GENERATION / LOWCMD OUTPUT
// ============================================================
//
// This section converts the 12-DoF policy output into Unitree
// LowCmd motor commands.
//
// IMPORTANT CONCEPT
// ------------------------------------------------------------
// The neural network does NOT directly command Unitree motor
// indices.
//
// The data flow is:
//
//   motion.pt
//       ↓
//   12 policy actions
//       ↓
//   ComputeOutput()
//       ↓
//   RobotCommand<float> in 12-DoF policy space
//       ↓
//   SetCommand()
//       ↓
//   Unitree LowCmd motor_cmd[35]
//
// For EDU12:
//
//   - The policy controls only the 12 leg joints.
//   - Arms and waist are held using fixed PD targets.
//   - Forbidden EDU23 joints are never commanded.
//   - All motor commands are disabled first before enabling
//     selected channels.
//
// ============================================================
//
// Main call chain:
//
//   RunModel()
//       ↓
//   ComputeOutput()
//       ↓
//   output queues
//
//   RobotControl()
//       ↓
//   StateController()
//       ↓
//   robot_command
//       ↓
//   SetCommand()
//       ↓
//   LowCmd publish
//
// ============================================================

/**
 * @brief Initialize LowCmd in a fully safe disabled state.
 *
 * Called by:
 *  - constructor
 *
 * Importance:
 * ------------------------------------------------------------
 * This method must be called before any DDS publication is possible.
 *
 * It guarantees that all Unitree motor command channels start with:
 *
 *   mode = 0
 *   q    = 0
 *   dq   = 0
 *   kp   = 0
 *   kd   = 0
 *   tau  = 0
 *
 * Why this matters:
 * ------------------------------------------------------------
 * Unitree LowCmd may contain many motor channels.
 *
 * If a previous value remains in memory, it could accidentally be
 * sent to the robot.
 *
 * This method prevents stale or undefined commands.
 */
void RL_Real_G1_EDU12::InitLowCmd()
{
    auto &mc = this->unitree_low_command.motor_cmd();
    const int n = (int)mc.size();
    for (int i = 0; i < n; ++i)
    {
        mc[i].mode() = 0; // SAFE: disabled by default
        mc[i].q() = 0;
        mc[i].kp() = 0;
        mc[i].dq() = 0;
        mc[i].kd() = 0;
        mc[i].tau() = 0;
    }
}

/**
 * @brief Compute CRC32 for Unitree LowCmd.
 *
 * Called by:
 *  - SetCommand()
 *  - SendDampingCommandIfPossible()
 *  - SendStandHoldCommandIfPossible()
 *
 * Importance:
 * ------------------------------------------------------------
 * Unitree low-level messages require a valid CRC before publishing.
 *
 * Without updating CRC:
 *  - the robot may reject the command,
 *  - communication may fail silently,
 *  - LowCmd may not be applied.
 */
uint32_t RL_Real_G1_EDU12::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }
    return CRC32;
}

/**
 * @brief Convert a 12-DoF policy command into Unitree LowCmd.
 *
 * Called by:
 *  - RobotControl()
 *
 * This is the main output method of the wrapper.
 *
 * Responsibilities:
 * ------------------------------------------------------------
 *  1. Reject unsafe modes.
 *  2. Verify mapping is ready.
 *  3. Verify the command vector has 12 elements.
 *  4. Disable all motors first.
 *  5. Enable only mapped leg motors.
 *  6. Apply fixed PD targets to safe arm/waist joints.
 *  7. Compute CRC.
 *  8. Publish LowCmd.
 *
 * Why all motors are disabled first:
 * ------------------------------------------------------------
 * This prevents stale commands from previous control cycles.
 *
 * Only explicitly allowed motors are re-enabled afterward.
 *
 * EDU12 policy behavior:
 * ------------------------------------------------------------
 * The policy controls:
 *
 *   leg_policy_to_ms_[0..11]
 *
 * The fixed PD controller holds:
 *
 *   arm_waist_joint2motor_idx_
 *
 * Forbidden EDU23 joints are skipped.
 */
void RL_Real_G1_EDU12::SetCommand(
    const RobotCommand<float>* command)
{
    // ========================================================
    // 0) BASIC SAFETY GUARDS
    // ========================================================

    if (!command)
        return;

    if (hw_mode_ != HwMode::REAL)
        return;

    if (!publish_lowcmd_)
        return;

    if (!policy_table_ready_)
    {
#if defined(USE_ROS2) && defined(USE_ROS)
        RCLCPP_ERROR_THROTTLE(
            ros2_node->get_logger(),
            *ros2_node->get_clock(),
            2000,
            "SetCommand blocked: policy_table_ready_=false"
        );
#else
        if (ShouldPrintEvery("setcommand_mapping_not_ready", 2.0))
        {
            std::cerr
                << "[ERROR] SetCommand blocked: policy_table_ready_=false"
                << std::endl;
        }
#endif
        return;
    }

    // ========================================================
    // 1) VERIFY 12-DOF POLICY COMMAND SIZE
    // ========================================================
    //
    // The official Unitree policy outputs 12 leg commands.
    //
    // Therefore all command vectors must have size 12.
    //
    // ========================================================

    if ((int)command->motor_command.q.size()   != kNdofPolicy ||
        (int)command->motor_command.dq.size()  != kNdofPolicy ||
        (int)command->motor_command.kp.size()  != kNdofPolicy ||
        (int)command->motor_command.kd.size()  != kNdofPolicy ||
        (int)command->motor_command.tau.size() != kNdofPolicy)
    {
#if defined(USE_ROS2) && defined(USE_ROS)
        RCLCPP_ERROR_THROTTLE(
            ros2_node->get_logger(),
            *ros2_node->get_clock(),
            2000,
            "SetCommand blocked: command vector size mismatch, expected 12"
        );
#else
        if (ShouldPrintEvery("setcommand_size_mismatch", 2.0))
        {
            std::cerr
                << "[FATAL] SetCommand expected 12 leg commands"
                << std::endl;
        }
#endif
        return;
    }

    // ========================================================
    // 2) PREPARE UNITREE LOWCMD HEADER
    // ========================================================

    unitree_low_command.mode_pr() =
        static_cast<uint8_t>(mode_pr);

    unitree_low_command.mode_machine() =
        mode_machine;

    auto& mc =
        unitree_low_command.motor_cmd();

    const int mc_size =
        static_cast<int>(mc.size());

    const bool debug_hold_policy_output = false;
    const float alpha = 0.10f;

    // ========================================================
    // 3) HARD SAFE BASELINE
    // ========================================================
    //
    // Disable every motor before writing new commands.
    //
    // This guarantees deterministic command projection.
    //
    // ========================================================

    for (int ms = 0; ms < mc_size; ++ms)
    {
        auto& m =
            mc[ms];

        m.mode() =
            0;

        m.q() =
            0.0f;

        m.dq() =
            0.0f;

        m.kp() =
            0.0f;

        m.kd() =
            0.0f;

        m.tau() =
            0.0f;
    }

    // ========================================================
    // 4) WRITE 12 RL LEG COMMANDS
    // ========================================================
    //
    // command->motor_command is in policy space:
    //
    //   p = 0..11
    //
    // Unitree LowCmd is in motor space:
    //
    //   ms = 0..34
    //
    // The mapping is:
    //
    //   ms = leg_policy_to_ms_[p]
    //
    // ========================================================

    if (debug_hold_policy_output)
    {
        static int hold_debug_counter = 0;

        if ((hold_debug_counter++ % 500) == 0)
        {
            std::cout
                << "[DEBUG SAFETY] RL policy output DISABLED -> holding default posture"
                << std::endl;
        }

        //if ((int)this->policy_default_dof_pos_.size() != this->kNdofPolicy)
        //    return;

        if (stand_hold_dof_pos_.size() != static_cast<size_t>(this->kNdofPolicy))
        {
            std::cerr
                << "[FATAL] stand_hold_dof_pos invalid size"
                << std::endl;

            return;
        }
    }
    //if (this->policy_default_dof_pos_.size() < (size_t)this->kNdofPolicy)
    //{
     //   std::cerr
    //        << "[FATAL] policy_default_dof_pos_ invalid size"
    //        << std::endl;
    //    return;
    //}



    // ========================================================
    // 4) WRITE 12 LEG COMMANDS
    // ========================================================
    // Legs are controlled by the official 12-action policy.
    // Before a fresh navigation command is received, we reproduce
    // the official Python default_pos_state():
    //   q   = default_angles
    //   dq  = 0
    //   kp  = leg_kps
    //   kd  = leg_kds
    //   tau = 0
    // ========================================================

    const bool lowstate_fresh =
        IsLowStateFresh();

    const bool no_fresh_nav_cmd =
        !this->nav_cmd_fresh_;

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        const int ms =
            leg_policy_to_ms_[p];

        if (ms < 0 || ms >= mc_size)
            continue;

        if (!enabled_ms_mask_[ms])
            continue;

        if (IsForbiddenEdu23MotorIndex(ms))
            continue;

        auto& m =
            mc[ms];

        m.mode() = 1;

        if (!this->locomotion_started_once_)
        {
            // GetUp phase:
            // Use the FSM interpolation command directly.
            m.q()   = command->motor_command.q[p];
            m.dq()  = command->motor_command.dq[p];
            m.kp()  = command->motor_command.kp[p];
            m.kd()  = command->motor_command.kd[p];
            m.tau() = command->motor_command.tau[p];
        }
        else if (debug_hold_policy_output)
        {
            // Manual debug hold.
            m.q()   = this->stand_hold_dof_pos_[p];
            m.dq()  = 0.0f;
            m.kp()  = 30.0f;
            m.kd()  = 2.0f;
            m.tau() = 0.0f;
        }
        else if (!lowstate_fresh)
        {
            // Safety fallback:
            // Keep the official policy default posture with moderate gains.
            m.q()   = this->real_stable_stand_dof_pos_[p];
            m.dq()  = 0.0f;
            m.kp()  = this->leg_kps_[p];
            m.kd()  = this->leg_kds_[p];
            m.tau() = 0.0f;
        }
        else if (no_fresh_nav_cmd)
        {
            // Idle locomotion without fresh Nav2/UDP command.
            // This reproduces the official Python default_pos_state()
            // instead of using hard-coded stiffness.
            m.q()   = command->motor_command.q[p];
            m.dq()  = 0.0f;
            //m.kp()  = this->leg_kps_[p];
            //m.kd()  = this->leg_kds_[p];
            m.kp() = this->stand_kps_[p];
            m.kd() = this->stand_kds_[p];
            m.tau() = 0.0f;
        }
        else
        {
            // Active locomotion:
            // Use the policy output for the 12 leg joints.
            m.q()   = command->motor_command.q[p];
            m.dq()  = 0.0f;
            m.kp()  = this->leg_kps_[p];
            m.kd()  = this->leg_kds_[p];
            m.tau() = 0.0f;
        }
    }


    // ========================================================
    // 5) WRITE FIXED WAIST / ARM PD TARGETS
    // ========================================================
    // The official 12-action policy does not output waist/arm
    // commands. The Python deployment keeps them fixed using:
    //   q   = arm_waist_target
    //   dq  = 0
    //   kp  = arm_waist_kps
    //   kd  = arm_waist_kds
    //   tau = 0
    // ========================================================

    for (size_t i = 0; i < arm_waist_joint2motor_idx_.size(); ++i)
    {
        const int ms =
            arm_waist_joint2motor_idx_[i];

        if (ms < 0 || ms >= mc_size)
            continue;

        if (!enabled_ms_mask_[ms])
            continue;

        if (IsForbiddenEdu23MotorIndex(ms))
            continue;

        if (i >= arm_waist_target_.size() ||
            i >= arm_waist_kps_.size() ||
            i >= arm_waist_kds_.size())
            continue;

        auto& m =
            mc[ms];

        m.mode() = 1;
        m.q()    = arm_waist_target_[i];
        m.dq()   = 0.0f;
        m.kp()   = arm_waist_kps_[i];
        m.kd()   = arm_waist_kds_[i];
        m.tau()  = 0.0f;
    }

    // ========================================================
    // 6) CRC + PUBLISH
    // ========================================================

    unitree_low_command.crc() =
        Crc32Core(
            reinterpret_cast<uint32_t*>(
                &unitree_low_command
            ),
            (sizeof(LowCmd_) >> 2) - 1
        );

    lowcmd_publisher->Write(
        unitree_low_command
    );
    if (this->locomotion_started_once_)
    {
        static int policy_debug_counter = 0;

        if ((policy_debug_counter++ % 1000) == 0)
        {
            std::cout << "\n=== EDU12 POLICY OUTPUT DEBUG ===\n";

            for (int p = 0; p < this->kNdofPolicy; ++p)
            {
                float q_safe =
                    this->policy_default_dof_pos_[p];

                float q_target =
                    command->motor_command.q[p];

                float delta =
                    q_target - q_safe;

                std::cout
                    << "p=" << p
                    << " q_safe=" << q_safe
                    << " q_target=" << q_target
                    << " delta=" << delta
                    << " kp=" << command->motor_command.kp[p]
                    << " kd=" << command->motor_command.kd[p]
                    << " tau=" << command->motor_command.tau[p]
                    << " q_policy=" << command->motor_command.q[p]
                    << " q_sent=" << mc[leg_policy_to_ms_[p]].q()
                    << " real_stable=" << real_stable_stand_dof_pos_[p]
                    << "\n";
            }

            std::cout
                << "================================="
                << std::endl;
        }

    }
}

// ============================================================
// SECTION 9 — REAL-TIME CONTROL LOOP
// ============================================================
//
// This section contains the main real-time control loop.
//
// The key method is:
//
//   RobotControl()
//
// It is called periodically by:
//
//   loop_control
//
// created in the constructor.
//
// ============================================================
//
// GLOBAL EXECUTION FLOW
// ------------------------------------------------------------
//
//   loop_control
//       ↓
//   RobotControl()
//       ↓
//   GetState()
//       ↓
//   StateController()
//       ↓
//   SetCommand()
//
// In parallel:
//
//   loop_rl
//       ↓
//   RunModel()
//       ↓
//   Forward()
//       ↓
//   ComputeOutput()
//
// ============================================================
//
// IMPORTANT DESIGN PRINCIPLE
// ------------------------------------------------------------
// RobotControl() does NOT run neural inference directly.
//
// Neural inference is handled by:
//
//   RunModel()
//
// RobotControl() focuses on:
//
//  - startup sequencing,
//  - safety gating,
//  - state update,
//  - FSM execution,
//  - LowCmd publication.
//
// This separation keeps the policy loop and the command loop
// independent and safer.
//
// ============================================================

/**
 * @brief Main real-time robot control loop.
 *
 * Called by:
 *  - loop_control
 *
 * Responsibilities:
 * ------------------------------------------------------------
 *  1. Keep the FSM alive before the policy is ready.
 *  2. Avoid enabling safety too early during startup.
 *  3. Read the latest robot state.
 *  4. Execute the FSM.
 *  5. Apply safety checks after startup grace period.
 *  6. Publish LowCmd only when it is safe.
 *
 * Startup logic:
 * ------------------------------------------------------------
 *
 * At startup:
 *
 *   policy_table_ready_ == false
 *
 * This means the policy mapping is not ready yet.
 *
 * During this phase:
 *  - the FSM is still allowed to run,
 *  - GetUp can still be executed,
 *  - safety watchdog is not fully armed yet,
 *  - LowCmd publication is blocked until mapping is ready.
 *
 * Once policy_table_ready_ becomes true:
 *  - a startup grace period begins,
 *  - the robot is allowed to stabilize,
 *  - safety monitoring is armed after the grace period.
 *
 * Why a startup grace period is needed:
 * ------------------------------------------------------------
 * Without this delay, the robot could enter LOWSTATE_LOST or
 * RECOVERING immediately after GetUp due to transient startup timing.
 *
 * LowCmd publication:
 * ------------------------------------------------------------
 * LowCmd is published only if:
 *
 *  - hw_mode_ == REAL
 *  - policy_table_ready_ == true
 *  - LowState has been received
 *  - publish_lowcmd_ == true
 *  - safety allows command output
 */

void RL_Real_G1_EDU12::RobotControl()
{
    try{
        #ifdef CMD_VEL_ONLY
            return;
        #endif

        if (hw_mode_ == HwMode::CMD_VEL_ONLY)
            return;

        // ========================================================
        // 1) POLICY NOT READY YET
        // ========================================================
        //
        // At startup, the policy mapping is not ready.
        //
        // This is normal because the policy is loaded later by:
        //
        //   OnEnterPolicy()
        //       ↓
        //   InitEdu12PolicyMapping()
        //
        // During this phase, we still allow the FSM to run.
        //
        // Why?
        //  - Keyboard transitions must still work.
        //  - GetUp may need to run before locomotion.
        //  - The system must not freeze just because the policy has
        //    not yet entered locomotion mode.
        //
        // ========================================================

        if (!policy_table_ready_)
        {
            //std::cout << "[RC DBG] before GetState startup" << std::endl;
        #if defined(USE_ROS2) && defined(USE_ROS)
                RCLCPP_WARN_THROTTLE(
                    ros2_node->get_logger(),
                    *ros2_node->get_clock(),
                    2000,
                    "[STARTUP] policy_table_ready_=false -> FSM allowed, safety not armed yet"
                );
        #endif

            // ----------------------------------------------------
            // Use real IMU if available.
            //
            // Even before the policy mapping is ready, real IMU data
            // can help the FSM during GetUp.
            //
            // We do NOT read mapped joints here because the mapping
            // is not ready yet.
            // ----------------------------------------------------
            if ((hw_mode_ == HwMode::REAL || hw_mode_ == HwMode::DRY_RUN) &&
                    this->got_lowstate.load(std::memory_order_relaxed))
                {
                    this->GetState_real_for_getup(&this->robot_state);
                }
                else
                {
                    // Use minimal state only to keep the FSM alive.
                    this->GetState_minimal(&this->robot_state);
                }
            
            // ----------------------------------------------------
            // During startup/GetUp, navigation commands must not
            // drive the robot.
            // ----------------------------------------------------

            this->control.navigation_mode =
                false;

            // ----------------------------------------------------
            // Run FSM even before policy readiness.
            //
            // This allows the FSM to reach the locomotion state, which
            // then calls OnEnterPolicy().
            // ----------------------------------------------------
            //std::cout << "[RC DBG] before StateController startup" << std::endl;
            this->StateController(&this->robot_state, &this->robot_command);
            //std::cout << "[RC DBG] after StateController startup" << std::endl;
            this->control.ClearInput();
            //std::cout << "[RC DBG] before return startup" << std::endl;

            return;
        }
        // ========================================================
        // 2) FIRST TIME POLICY BECOMES READY
        // ========================================================
        //
        // This block runs once after:
        //
        //   BuildLegPolicyToMsTable()
        //
        // succeeds.
        //
        // It starts a startup grace period before safety watchdog is
        // fully armed.
        //
        // ========================================================

        static bool policy_ready_seen = false;
        static auto policy_ready_tp = std::chrono::steady_clock::now();

        if (!policy_ready_seen)
        {
            policy_ready_seen = true;
            policy_ready_tp = std::chrono::steady_clock::now();
            // When policy becomes ready for the first time, we explicitly start
            // from RUNNING. Safety will be evaluated only after the grace period.
            safety_state_ = RLSafetyState::RUNNING;
            lowstate_lost_timer_started_ = false;
            std::cout << "[STARTUP] Policy ready: safety_state forced to RUNNING during grace period"
                << std::endl;

            #if defined(USE_ROS2) && defined(USE_ROS)
                    RCLCPP_WARN(
                        ros2_node->get_logger(),
                        "[STARTUP] Policy mapping ready. Starting safety grace period."
                    );
            #endif
        }

        const double startup_age =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                - policy_ready_tp
            ).count();

        // Grace period after policy initialization.
        // It gives enough time for GetUp / ready posture to complete.
        const double startup_safety_grace_sec =
            3.0;

        const bool safety_armed =
            startup_age >= startup_safety_grace_sec;

        if (ShouldPrintEvery("runtime_health", 3.0))
        {
            std::cout
                << "[HEALTH]"
                << " lowstate_fresh="
                << static_cast<int>(IsLowStateFresh())
                << " got_lowstate="
                << static_cast<int>(got_lowstate.load())
                << " safety="
                << static_cast<int>(safety_state_)
                << " policy_ready="
                << static_cast<int>(policy_table_ready_)
                << " publish_lowcmd="
                << static_cast<int>(publish_lowcmd_)
                << " mode_machine="
                << static_cast<int>(mode_machine)
                << " safety_armed="
                << static_cast<int>(safety_armed)
                << std::endl;
        }

        // ========================================================
        // 3) SAFETY CHECK AFTER STARTUP GRACE PERIOD
        // ========================================================
        //
        // Safety is not fully active immediately after policy readiness.
        //
        // After the grace period, RobotControl() calls:
        //
        //   UpdateSafetyState()
        //
        // If safety_state_ is not RUNNING:
        //  - external commands are cleared,
        //  - RL command output is blocked,
        //  - stand-hold or damping is sent if possible.
        //
        // ========================================================
        if (hw_mode_ == HwMode::REAL && safety_armed)
        {
            this->UpdateSafetyState();

            if (safety_state_ != RLSafetyState::RUNNING)
            {
                // Stop any external velocity command.
                this->ClearCmdVelUnsafe();

                // Keep internal state stable.
                this->GetState_minimal(&this->robot_state);

                // If LowState was lost recently, try Stand-Hold first.
                if (lowstate_lost_timer_started_)
                {
                    const double lost_age =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - lowstate_lost_start_tp_
                        ).count();

                    if (IsStandHoldReady())
                    {
                        this->SendStandHoldCommandIfPossible();

                        if (ShouldPrintEvery("stand_hold_safety", 1.0))
                        {
                            std::cout << "[RL SAFETY] LOWSTATE_LOST: Stand-Hold active"
                                    << " lost_age=" << lost_age
                                    << std::endl;
                        }
                    }
                    else
                    {
                        this->SendDampingCommandIfPossible();

                        if (ShouldPrintEvery("damping_fallback", 1.0))
                        {
                            std::cerr << "[RL SAFETY] LOWSTATE_LOST: Stand-Hold not ready, using damping fallback"
                                    << " lost_age=" << lost_age
                                    << std::endl;
                        }
                    }
                }
                else
                {
                    if (IsStandHoldReady())
                        this->SendStandHoldCommandIfPossible();
                    else
                        this->SendDampingCommandIfPossible();
                }

            #if defined(USE_ROS2) && defined(USE_ROS)
                        RCLCPP_ERROR_THROTTLE(
                            ros2_node->get_logger(),
                            *ros2_node->get_clock(),
                            500,
                            "[RL SAFETY] Safety armed: LowState/heartbeat unsafe. RL command blocked."
                        );
            #endif
                return;
            }
        }

        // ========================================================
        // 4) NORMAL STATE UPDATE
        // ========================================================
        //
        // At this point:
        //  - the policy mapping is ready,
        //  - startup grace is either active or completed,
        //  - safety allows normal execution.
        //
        // GetState() reads the physical state and converts it into the
        // 12-DoF policy state.
        //
        // ========================================================
        this->GetState(&this->robot_state);

        #if defined(USE_ROS2) && defined(USE_ROS)
            RCLCPP_INFO_THROTTLE(
                ros2_node->get_logger(),
                *ros2_node->get_clock(),
                1000,
                "[DBG] mapping=%d got_lowstate=%d safety=%d armed=%d startup_age=%.2f",
                static_cast<int>(this->policy_table_ready_),
                static_cast<int>(this->got_lowstate.load(std::memory_order_relaxed)),
                static_cast<int>(this->safety_state_),
                static_cast<int>(safety_armed),
                startup_age
            );
        #endif
        // ========================================================
        // 5) RUN FINITE STATE MACHINE
        // ========================================================
        //
        // The FSM updates robot_command according to the current state:
        //
        //  - passive
        //  - GetUp
        //  - locomotion
        //  - recovery
        //
        // In locomotion mode, robot_command is usually populated from
        // the latest RL outputs generated by RunModel().
        //
        // ========================================================
        this->StateController(&this->robot_state, &this->robot_command);
        this->control.ClearInput();

        // ========================================================
        // 6) LOWCMD PUBLICATION GATE
        // ========================================================
        //
        // LowCmd is allowed only if all conditions are true:
        //
        //  - REAL mode
        //  - policy mapping ready
        //  - at least one LowState received
        //  - LowCmd publication enabled
        //  - safety state allows output
        //
        // During startup grace:
        //  - commands may be allowed to finish GetUp.
        //
        // After safety is armed:
        //  - only RUNNING state can publish.
        //
        // ========================================================
        const bool allow_publish =
            hw_mode_ == HwMode::REAL &&
            this->policy_table_ready_ &&
            this->got_lowstate.load(std::memory_order_relaxed) &&
            publish_lowcmd_ &&
            (
                !safety_armed ||
                safety_state_ == RLSafetyState::RUNNING
            );

        if (allow_publish)
        {
            this->SetCommand(&this->robot_command);
        } 

    }
    catch (const std::exception& e)
    {
        std::cerr << "[FATAL] RobotControl exception: "
                  << e.what() << std::endl;
        while (true) sleep(1);
    }
    catch (...)
    {
        std::cerr << "[FATAL] RobotControl unknown exception" << std::endl;
        while (true) sleep(1);
    }
}

// ============================================================
// SECTION 10 — SAFETY MANAGEMENT / RECOVERY
// ============================================================
//
// This section protects the real robot when communication or state
// feedback becomes unsafe.
//
// The safety system is intentionally separated from the RL model.
//
// Why?
// ------------------------------------------------------------
// The neural policy should never be responsible for deciding whether
// the hardware communication is safe.
//
// Safety is handled here through a deterministic state machine.
//
// ============================================================
//
// SAFETY STATE MACHINE
// ------------------------------------------------------------
//
//   RUNNING
//      |
//      | LowState lost or heartbeat lost
//      v
//   LOWSTATE_LOST
//      |
//      | communication restored
//      v
//   RECOVERING
//      |
//      | recovery delay completed
//      v
//   RUNNING
//
// ============================================================
//
// Main call chain:
//
//   RobotControl()
//       ↓
//   UpdateSafetyState()
//       ↓
//   ClearCmdVelUnsafe()
//       ↓
//   SendStandHoldCommandIfPossible()
//       or
//   SendDampingCommandIfPossible()
//
// During recovery:
//
//   UpdateSafetyState()
//       ↓
//   ResetRLAfterRecovery()
//

// ============================================================

/**
 * @brief Clear all external velocity commands immediately.
 *
 * Called by:
 *  - OnEnterGetUp()
 *  - OnEnterLocomotion()
 *  - OnExitLocomotion()
 *  - RobotControl()
 *  - UpdateSafetyState()
 *  - ResetRLAfterRecovery()
 *
 * Importance:
 * ------------------------------------------------------------
 * When the robot enters a safety transition, stale velocity commands
 * must not remain active.
 *
 * This method resets:
 *  - ROS2 cmd_vel storage,
 *  - Unitree-only UDP command storage,
 *  - internal control command values.
 */

void RL_Real_G1_EDU12::ClearCmdVelUnsafe()
{
#if defined(USE_ROS2) && defined(USE_ROS)
    std::lock_guard<std::mutex> lk(this->cmd_vel_mtx_);

    this->cmd_vel_.linear.x  = 0.0;
    this->cmd_vel_.linear.y  = 0.0;
    this->cmd_vel_.linear.z  = 0.0;
    this->cmd_vel_.angular.x = 0.0;
    this->cmd_vel_.angular.y = 0.0;
    this->cmd_vel_.angular.z = 0.0;

    this->last_cmd_vel_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
#endif

#ifndef USE_ROS
    {
        std::lock_guard<std::mutex> lk(this->ipc_cmd_mtx_);
        this->ipc_cmd_.vx = 0.0f;
        this->ipc_cmd_.vy = 0.0f;
        this->ipc_cmd_.wz = 0.0f;
        this->ipc_cmd_.initialized = false;
    }
#endif

    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
    
}

/**
 * @brief Confirm that unsafe links are really lost.
 *
 * Called by:
 *  - UpdateSafetyState()
 *
 * Importance:
 * ------------------------------------------------------------
 * Communication may occasionally drop for a very short time.
 *
 * This helper avoids immediately switching to safety mode because
 * of a single transient glitch.
 *
 * It starts a timer when links become unsafe and returns true only
 * if the unsafe condition persists long enough.
 */
bool RL_Real_G1_EDU12::ConfirmLinksLost(
    bool links_ok,
    const std::chrono::steady_clock::time_point& now)
{
    if (links_ok)
    {
        links_unsafe_timer_started_ = false;
        return false;
    }

    if (!links_unsafe_timer_started_)
    {
        links_unsafe_timer_started_ = true;
        links_unsafe_since_tp_ = now;
        return false;
    }

    const double unsafe_age =
        std::chrono::duration<double>(now - links_unsafe_since_tp_).count();

    return unsafe_age >= links_loss_confirm_sec_;
}

/**
 * @brief Check whether stand-hold posture is available.
 *
 * Called by:
 *  - RobotControl()
 *  - safety fallback logic
 *
 * Importance:
 * ------------------------------------------------------------
 * Stand-hold should only be used if LowStateHandler() has already
 * captured a valid safe posture.
 *
 * If no valid posture has been captured, the safer fallback is
 * damping mode.
 */
bool RL_Real_G1_EDU12::IsStandHoldReady() const
{
    std::lock_guard<std::mutex> lk(stand_hold_mtx_);
    return stand_hold_ready_;
}

/**
 * @brief Reset RL buffers after recovering from a safety interruption.
 *
 * Called by:
 *  - UpdateSafetyState()
 *
 * Importance:
 * ------------------------------------------------------------
 * After a communication loss, the previous RL internal state may be
 * inconsistent with the real robot posture.
 *
 * This method clears:
 *  - velocity commands,
 *  - observation command vector,
 *  - output queues,
 *  - episode counter,
 *  - previous actions,
 *  - observation history if used.
 *
 * Why not always reset the neural model?
 * ------------------------------------------------------------
 * Some recurrent policies are sensitive to reset timing.
 *
 * This wrapper currently resets buffers and keeps model reset
 * disabled unless explicitly enabled by recovery logic.
 */
void RL_Real_G1_EDU12::ResetRLAfterRecovery()
{
    ClearCmdVelUnsafe();

    this->control.x   = 0.0f;
    this->control.y   = 0.0f;
    this->control.yaw = 0.0f;

    this->obs.commands = {0.0f, 0.0f, 0.0f};

    std::vector<float> tmp;

    while (output_dof_pos_queue.try_pop(tmp)) {}
    while (output_dof_vel_queue.try_pop(tmp)) {}
    while (output_dof_tau_queue.try_pop(tmp)) {}

    this->episode_length_buf = 0;

    // Avoid reusing stale policy actions after recovery
    if (!this->obs.actions.empty())
    {
        std::fill(this->obs.actions.begin(), this->obs.actions.end(), 0.0f);
    }

    // Reset observation history if the policy uses temporal observations
    const auto obs_hist = this->params.Get<std::vector<int>>("observations_history", {});
    if (!obs_hist.empty())
    {
        const std::vector<float> fresh_obs = this->ComputeObservation();

        // Buffer was created for one environment, so reset env 0
        this->history_obs_buf.reset({0}, fresh_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(obs_hist);
    }

    if (model && allow_recovery_model_reset_)
    {
        std::cout
            << "[RESET] ResetRLAfterRecovery: RL buffers reset"
            << std::endl;

        /*
         * Model reset is intentionally not forced here.
         *
         * If the model is recurrent and reset is required, enable it
         * only after validating startup behavior on the real robot.
         */
        // model->reset();
    }
    else
    {
        std::cout
            << "[RESET SKIPPED] First startup/GetUp phase: model reset blocked"
            << std::endl;
    }
}

/**
 * @brief Update the safety state machine.
 *
 * Called by:
 *  - RobotControl()
 *
 * This method decides whether the robot can continue running RL.
 *
 * It monitors:
 *  - LowState freshness,
 *  - optional laptop heartbeat freshness.
 *
 * Safety transitions:
 * ------------------------------------------------------------
 *
 * RUNNING → LOWSTATE_LOST
 *  - if LowState or laptop heartbeat is lost long enough.
 *
 * LOWSTATE_LOST → RECOVERING
 *  - if all required links become healthy again.
 *
 * RECOVERING → RUNNING
 *  - after a stabilization delay.
 *
 * RECOVERING → LOWSTATE_LOST
 *  - if communication is lost again during recovery.
 */

 #if defined(USE_ROS2) && defined(USE_ROS)

/**
 * @brief Receive laptop heartbeat messages.
 *
 * Called by:
 *  - ROS2 heartbeat subscriber created in the constructor.
 *
 * Importance:
 * ------------------------------------------------------------
 * This callback is part of the high-level safety layer.
 *
 * The robot may depend on an external laptop for:
 *  - Nav2,
 *  - perception,
 *  - command generation,
 *  - supervision.
 *
 * If the laptop disappears, the robot should not continue running
 * locomotion blindly.
 *
 * This callback updates the timestamp used by:
 *
 *   UpdateSafetyState()
 *
 * Call chain:
 * ------------------------------------------------------------
 *
 *   /rl_laptop_heartbeat
 *       ↓
 *   LaptopHeartbeatCallback()
 *       ↓
 *   last_laptop_heartbeat_tp_
 *       ↓
 *   UpdateSafetyState()
 *       ↓
 *   laptop_fresh check
 *
 * Notes:
 * ------------------------------------------------------------
 * The heartbeat message content is irrelevant.
 *
 * Only the reception time matters.
 */
void RL_Real_G1_EDU12::LaptopHeartbeatCallback(
    const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg;

    this->last_laptop_heartbeat_tp_ =
        std::chrono::steady_clock::now();

    this->laptop_heartbeat_initialized_ =
        true;

    RCLCPP_DEBUG_THROTTLE(
        this->ros2_node->get_logger(),
        *this->ros2_node->get_clock(),
        1000,
        "[HEARTBEAT] Laptop alive"
    );
}

#endif

void RL_Real_G1_EDU12::UpdateSafetyState()
{
        // ------------------------------------------------------------
        // 0) Current time reference
        // ------------------------------------------------------------
    const auto now = std::chrono::steady_clock::now();

        // ------------------------------------------------------------
        // 1) Check Low-Level connection (robot DDS)
        //    - Based on reception of LowState messages
        // ------------------------------------------------------------
    const bool lowstate_fresh = IsLowStateFresh();

        // ------------------------------------------------------------
        // 2) Check High-Level connection (laptop / ROS / Nav2)
        //    - Optional safety layer
        // ------------------------------------------------------------
    bool laptop_fresh = !require_laptop_heartbeat_;

    if (require_laptop_heartbeat_)
    {
        laptop_fresh =
            false;

        if (laptop_heartbeat_initialized_)
        {
            const double laptop_age =
                std::chrono::duration<double>(
                    now - last_laptop_heartbeat_tp_
                ).count();

            laptop_fresh =
                laptop_age <= laptop_heartbeat_timeout_sec_;
        }
    }
    // ------------------------------------------------------------
    // 3) Global link status
    // ------------------------------------------------------------
        const bool links_ok = lowstate_fresh && laptop_fresh;

    // ========================================================
    // STATE: RUNNING (normal RL operation)
    // ========================================================
    if (safety_state_ == RLSafetyState::RUNNING)
    {
        // If ANY link is lost → immediate safety fallback
        if (!links_ok)
        {
            if (!ConfirmLinksLost(
                    links_ok,
                    now))
            {
                return;
            }

            safety_state_ =
                RLSafetyState::LOWSTATE_LOST;
            // Start safety timer (used for stand-hold duration)
            lowstate_lost_start_tp_ =
                now;

            lowstate_lost_timer_started_ =
                true;
            // Stop all commands immediately
            ClearCmdVelUnsafe();

#ifndef USE_ROS
            std::cerr
                << "[SAFETY TRANSITION] RUNNING -> LOWSTATE_LOST"
                << " lowstate_fresh="
                << lowstate_fresh
                << " laptop_fresh="
                << laptop_fresh
                << std::endl;
#endif

#if defined(USE_ROS2) && defined(USE_ROS)
            const char* reason =
                !lowstate_fresh
                    ? "Robot LowState/DDS"
                    : "Laptop heartbeat";

            RCLCPP_ERROR(
                ros2_node->get_logger(),
                "CRITICAL: %s lost. Entering safety mode.",
                reason
            );
#endif
            return;
        }

        ConfirmLinksLost(
            true,
            now
        );
    }

    // ========================================================
    // STATE: LOWSTATE_LOST
    // ========================================================

    if (safety_state_ == RLSafetyState::LOWSTATE_LOST)
    {
        ClearCmdVelUnsafe();

        if (links_ok)
        {
            ConfirmLinksLost(
                true,
                now
            );

            safety_state_ =
                RLSafetyState::RECOVERING;

            lowstate_lost_timer_started_ =
                false;

            recovery_start_tp_ =
                now;

#ifndef USE_ROS
            std::cerr
                << "[SAFETY TRANSITION] LOWSTATE_LOST -> RECOVERING"
                << std::endl;
#endif

#if defined(USE_ROS2) && defined(USE_ROS)
            RCLCPP_WARN(
                ros2_node->get_logger(),
                "Connections restored. Entering RECOVERING."
            );
#endif
        }

        return;
    }

    // ========================================================
    // STATE: RECOVERING
    // ========================================================

    if (safety_state_ == RLSafetyState::RECOVERING)
    {
        ClearCmdVelUnsafe();

        if (!links_ok)
        {
            if (!ConfirmLinksLost(
                    links_ok,
                    now))
            {
                return;
            }

            safety_state_ =
                RLSafetyState::LOWSTATE_LOST;
            
            lowstate_lost_start_tp_ =
                now;

            lowstate_lost_timer_started_ =
                true;

#ifndef USE_ROS
            std::cerr
                << "[SAFETY TRANSITION] RECOVERING -> LOWSTATE_LOST"
                << std::endl;
#endif

#if defined(USE_ROS2) && defined(USE_ROS)
            const char* reason =
                !lowstate_fresh
                    ? "Robot LowState/DDS"
                    : "Laptop heartbeat";

            RCLCPP_ERROR(
                ros2_node->get_logger(),
                "CRITICAL: %s lost again during RECOVERING.",
                reason
            );
#endif
            return;
        }

        ConfirmLinksLost(
            true,
            now
        );

        const double rec_age =
            std::chrono::duration<double>(
                now - recovery_start_tp_
            ).count();

        if (rec_age >= recovery_duration_sec_)
        {
            if (locomotion_started_once_)
            {
                ResetRLAfterRecovery();
            }
            else
            {
                ClearCmdVelUnsafe();

                obs.commands =
                    {0.0f, 0.0f, 0.0f};

                std::cout
                    << "[RECOVERY] First startup recovery: model reset skipped"
                    << std::endl;
            }

            safety_state_ =
                RLSafetyState::RUNNING;

#ifndef USE_ROS
            std::cerr
                << "[SAFETY TRANSITION] RECOVERING -> RUNNING"
                << std::endl;
#endif

#if defined(USE_ROS2) && defined(USE_ROS)
            RCLCPP_INFO(
                ros2_node->get_logger(),
                "Recovery complete. RL policy is RUNNING."
            );
#endif
        }

        return;
    }
}

/**
 * @brief Send light damping command to enabled motors.
 *
 * Called by:
 *  - RobotControl()
 *
 * Used when:
 *  - the robot is unsafe,
 *  - stand-hold posture is not available,
 *  - LowState was lost before a good posture could be captured.
 *
 * Importance:
 * ------------------------------------------------------------
 * Damping mode is safer than sending stale RL commands.
 */

 void RL_Real_G1_EDU12::SendDampingCommandIfPossible()
{
    if (hw_mode_ != HwMode::REAL)
        return;

    if (!publish_lowcmd_)
        return;

    auto& mc = this->unitree_low_command.motor_cmd();
    const int mc_size = static_cast<int>(mc.size());

    for (int ms = 0; ms < mc_size; ++ms)
    {
        auto& m = mc[ms];

        if (ms < static_cast<int>(enabled_ms_mask_.size()) && enabled_ms_mask_[ms])
        {
            m.mode() = 1;
            m.dq()   = 0.0f;
            m.kp()   = 0.0f;
            m.kd()   = 1.0f;   // damping very light to be adjust with precaution
            m.tau()  = 0.0f;
            m.q() = 0.0f;
            // Do not force q here if not sure of the posture targeted of the robot.
        }
        else
        {
            m.mode() = 0;
            m.q()    = 0.0f;
            m.dq()   = 0.0f;
            m.kp()   = 0.0f;
            m.kd()   = 0.0f;
            m.tau()  = 0.0f;
        }
    }

    this->unitree_low_command.crc() =
        this->Crc32Core(
            reinterpret_cast<uint32_t*>(&this->unitree_low_command),
            (sizeof(LowCmd_) >> 2) - 1
        );

    this->lowcmd_publisher->Write(this->unitree_low_command);
}

/**
 * @brief Hold the last valid safe posture.
 *
 * Called by:
 *  - RobotControl()
 *
 * This method is preferred over damping when a safe posture has
 * already been captured by LowStateHandler().
 *
 * It commands:
 *  - q = last_good_ms_q_[ms]
 *  - dq = 0
 *  - kp = stand_hold_kp_
 *  - kd = stand_hold_kd_
 *  - tau = 0
 */
void RL_Real_G1_EDU12::SendStandHoldCommandIfPossible()
{
    if (hw_mode_ != HwMode::REAL)
        return;

    if (!publish_lowcmd_)
        return;

    if (!policy_table_ready_)
        return;

    std::array<float, kMsTotal> target_q{};
    std::array<bool,  kMsTotal> target_valid{};
    bool ready = false;

    // Copy the last valid posture quickly, then release the mutex.
    {
        std::lock_guard<std::mutex> lk(stand_hold_mtx_);
        target_q = last_good_ms_q_;
        target_valid = last_good_ms_valid_;
        ready = stand_hold_ready_;
    }

    if (!ready)
    {
        SendDampingCommandIfPossible();
        return;
    }

    auto& mc = this->unitree_low_command.motor_cmd();
    const int mc_size = static_cast<int>(mc.size());

    this->unitree_low_command.mode_pr() =
        static_cast<uint8_t>(this->mode_pr);

    this->unitree_low_command.mode_machine() =
        this->mode_machine;

    for (int ms = 0; ms < mc_size; ++ms)
    {
        auto& m = mc[ms];

        if (ms < kMsTotal &&
            enabled_ms_mask_[ms] &&
            target_valid[ms])
        {
            m.mode() = 1;
            m.q()    = target_q[ms];
            m.dq()   = 0.0f;
            m.kp()   = stand_hold_kp_;
            m.kd()   = stand_hold_kd_;
            m.tau()  = 0.0f;
        }
        else
        {
            m.mode() = 0;
            m.q()    = 0.0f;
            m.dq()   = 0.0f;
            m.kp()   = 0.0f;
            m.kd()   = 0.0f;
            m.tau()  = 0.0f;
        }
    }

    this->unitree_low_command.crc() =
        this->Crc32Core(
            reinterpret_cast<uint32_t*>(&this->unitree_low_command),
            (sizeof(LowCmd_) >> 2) - 1
        );

    this->lowcmd_publisher->Write(this->unitree_low_command);
}


// ============================================================
// SECTION 11 — FSM MODE TRANSITIONS
// ============================================================
//
// This section contains FSM hooks related to mode transitions.
//
// IMPORTANT:
// ------------------------------------------------------------
// OnEnterPolicy() is NOT defined here.
//
// OnEnterPolicy() belongs to:
//
//   SECTION 3 — FSM POLICY ENTRY POINT
//
// because it is the entry point that loads the policy through:
//
//   OnEnterPolicy()
//       ↓
//   InitEdu12PolicyMapping()
//       ↓
//   InitRL()
//       ↓
//   BuildLegPolicyToMsTable()
//
// This section only contains transition hooks that prepare or clean
// internal state when entering or leaving specific robot modes.
//
// ============================================================
//
// Methods in this section:
//
//   OnEnterGetUp()
//       Called when the FSM enters GetUp.
//
//   OnEnterLocomotion()
//       Called when the FSM enters locomotion.
//
//   OnExitLocomotion()
//       Called when the FSM exits locomotion.
//
// ============================================================

/**
 * @brief Called when the FSM enters GetUp.
 *
 * Called by:
 *  - FSM GetUp state
 *
 * Importance:
 * ------------------------------------------------------------
 * GetUp must not be affected by stale navigation commands.
 *
 * A previous /cmd_vel or UDP velocity command could otherwise remain
 * active and be injected into the locomotion policy immediately after
 * standing up.
 *
 * This method:
 *  - clears all velocity commands,
 *  - disables navigation mode,
 *  - resets internal command values,
 *  - keeps the neural model state untouched.
 *
 * Why the model is not reset here:
 * ------------------------------------------------------------
 * Resetting recurrent or stateful models during GetUp may disturb
 * the transition into stable locomotion.
 */
void RL_Real_G1_EDU12::OnEnterGetUp()
{
    try
    {
        //std::cout << "[GETUP TRACE] ENTER OnEnterGetUp" << std::endl;
        static int getup_count = 0;
        getup_count++;

        std::cout << "[FSM TRACE] OnEnterGetUp called count="
                << getup_count
                << " policy_ready=" << policy_table_ready_
                << " safety_state=" << static_cast<int>(safety_state_)
                << " got_lowstate=" << got_lowstate.load()
                << " lowstate_fresh=" << IsLowStateFresh()
                << std::endl;
                
        // GetUp must not use Nav2/cmd_vel.
        ClearCmdVelUnsafe();
        this->control.navigation_mode = false;
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;

        std::cout << "[GETUP TRACE] EXIT OnEnterGetUp normally" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FATAL] OnEnterGetUp exception: "
            << e.what()
            << std::endl;

        throw;
    }
    catch (...)
    {
        std::cerr
            << "[FATAL] OnEnterGetUp unknown exception"
            << std::endl;

        throw;
    }
}

void RL_Real_G1_EDU12::HoldRealStableStandPose(double hold_time)
{
    const auto& leg_target = this->real_stable_stand_dof_pos_;

    const auto leg_kps =
        this->params.Get<std::vector<float>>("kps", {});

    const auto leg_kds =
        this->params.Get<std::vector<float>>("kds", {});

    if (leg_target.size() != static_cast<size_t>(this->kNdofPolicy))
    {
        std::cerr << "[FATAL] real_stable_stand_dof_pos invalid size" << std::endl;
        return;
    }

    const double dt =
        this->params.Get<double>("control_dt", 0.02);

    const int steps =
        std::max(1, static_cast<int>(hold_time / dt));

    for (int s = 0; s < steps; ++s)
    {
        auto& mc = this->unitree_low_command.motor_cmd();

        for (int p = 0; p < this->kNdofPolicy; ++p)
        {
            const int ms = this->leg_policy_to_ms_[p];

            if (ms < 0 || ms >= static_cast<int>(mc.size()))
                continue;

            auto& m = mc[ms];
            m.mode() = 1;
            m.q() = leg_target[p];
            m.dq() = 0.0f;
            m.kp() = leg_kps[p];
            m.kd() = leg_kds[p];
            m.tau() = 0.0f;
        }

        this->unitree_low_command.crc() =
            this->Crc32Core(
                reinterpret_cast<uint32_t*>(&this->unitree_low_command),
                (sizeof(LowCmd_) >> 2) - 1
            );

        this->lowcmd_publisher->Write(this->unitree_low_command);

        std::this_thread::sleep_for(std::chrono::duration<double>(dt));
    }

    std::cout << "[INFO] Real stable stand pose held before RL" << std::endl;
}

void RL_Real_G1_EDU12::PrintCurrentLegPose(const std::string& tag)
{
    LowState_ low_state_copy;
    {
        std::lock_guard<std::mutex> lk(this->lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    std::cout << "\n" << tag << "\n";

    for (int p = 0; p < this->kNdofPolicy; ++p)
    {
        const int ms = this->leg_policy_to_ms_[p];
        const float q = low_state_copy.motor_state()[ms].q();

        std::cout
            << "p=" << p
            << " ms=" << ms
            << " q_hw=" << q
            << " default=" << this->policy_default_dof_pos_[p]
            << " error=" << q - this->policy_default_dof_pos_[p]
            << std::endl;
    }
}

bool RL_Real_G1_EDU12::MoveToPolicyNominalPose()
{
    const auto leg_target = this->params.Get<std::vector<float>>("default_dof_pos", {});
    const auto leg_kps    = this->params.Get<std::vector<float>>("kps", {});
    const auto leg_kds    = this->params.Get<std::vector<float>>("kds", {});

    const auto arm_target = this->params.Get<std::vector<float>>("arm_waist_target", {});
    const auto arm_kps    = this->params.Get<std::vector<float>>("arm_waist_kps", {});
    const auto arm_kds    = this->params.Get<std::vector<float>>("arm_waist_kds", {});

    if (leg_target.size() != static_cast<size_t>(this->kNdofPolicy) ||
        leg_kps.size()    != static_cast<size_t>(this->kNdofPolicy) ||
        leg_kds.size()    != static_cast<size_t>(this->kNdofPolicy))
    {
        std::cerr << "[FATAL] Invalid leg nominal vectors" << std::endl;
        return false;
    }

    if (arm_target.size() != arm_waist_joint2motor_idx_.size() ||
        arm_kps.size()    != arm_waist_joint2motor_idx_.size() ||
        arm_kds.size()    != arm_waist_joint2motor_idx_.size())
    {
        std::cerr << "[FATAL] Invalid arm/waist nominal vectors" << std::endl;
        return false;
    }

    const double total_time =
        this->params.Get<double>("policy_nominal_transition_time", 8.0);

    const double dt =
        this->params.Get<double>("control_dt", 0.02);

    const int steps =
        std::max(1, static_cast<int>(total_time / dt));

    LowState_ low_state_copy;
    {
        std::lock_guard<std::mutex> lk(this->lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    auto& mc_init =
        this->unitree_low_command.motor_cmd();

    const int mc_size =
        static_cast<int>(mc_init.size());

    const int motor_state_size =
        static_cast<int>(low_state_copy.motor_state().size());

    std::vector<float> leg_init(this->kNdofPolicy, 0.0f);

    for (int p = 0; p < this->kNdofPolicy; ++p)
    {
        const int ms = this->leg_policy_to_ms_[p];

        if (ms < 0 || ms >= motor_state_size)
            continue;

        leg_init[p] =
            low_state_copy.motor_state()[ms].q();
    }

    std::vector<float> arm_init(
        arm_waist_joint2motor_idx_.size(),
        0.0f
    );

    for (size_t i = 0; i < arm_waist_joint2motor_idx_.size(); ++i)
    {
        const int ms =
            arm_waist_joint2motor_idx_[i];

        if (ms < 0 || ms >= motor_state_size)
            continue;

        arm_init[i] =
            low_state_copy.motor_state()[ms].q();
    }

    auto send_nominal_command =
        [&](float alpha)
        {
            auto& mc =
                this->unitree_low_command.motor_cmd();

            for (int p = 0; p < this->kNdofPolicy; ++p)
            {
                const int ms =
                    this->leg_policy_to_ms_[p];

                if (ms < 0 || ms >= static_cast<int>(mc.size()))
                    continue;

                if (!enabled_ms_mask_[ms])
                    continue;

                if (IsForbiddenEdu23MotorIndex(ms))
                    continue;

                auto& m =
                    mc[ms];

                m.mode() = 1;
                m.q() =
                    (1.0f - alpha) * leg_init[p]
                    + alpha * leg_target[p];

                m.dq()  = 0.0f;
                m.kp()  = leg_kps[p];
                m.kd()  = leg_kds[p];
                m.tau() = 0.0f;

                if ((p == 4 || p == 10) && (alpha >= 0.99f))
                {
                    std::cout
                        << "[MOVE NOMINAL] p=" << p
                        << " ms=" << ms
                        << " q_sent=" << m.q()
                        << " kp_sent=" << m.kp()
                        << " kd_sent=" << m.kd()
                        << std::endl;
                }
            }

            for (size_t i = 0; i < arm_waist_joint2motor_idx_.size(); ++i)
            {
                const int ms =
                    arm_waist_joint2motor_idx_[i];

                if (ms < 0 || ms >= static_cast<int>(mc.size()))
                    continue;

                if (!enabled_ms_mask_[ms])
                    continue;

                if (IsForbiddenEdu23MotorIndex(ms))
                    continue;

                auto& m =
                    mc[ms];

                m.mode() = 1;
                m.q() =
                    (1.0f - alpha) * arm_init[i]
                    + alpha * arm_target[i];

                m.dq()  = 0.0f;
                m.kp()  = arm_kps[i];
                m.kd()  = arm_kds[i];
                m.tau() = 0.0f;
            }

            this->unitree_low_command.mode_pr() =
            static_cast<uint8_t>(this->mode_pr);

            this->unitree_low_command.mode_machine() =
                this->mode_machine;

            this->unitree_low_command.crc() =
                this->Crc32Core(
                    reinterpret_cast<uint32_t*>(
                        &this->unitree_low_command
                    ),
                    (sizeof(LowCmd_) >> 2) - 1
                );

            if (this->hw_mode_ == HwMode::REAL &&
                this->publish_lowcmd_)
            {
                this->lowcmd_publisher->Write(
                    this->unitree_low_command
                );
            }
        };

    for (int s = 0; s <= steps; ++s)
    {
        const float alpha =
            static_cast<float>(s) /
            static_cast<float>(steps);

        if (this->debug_transition_ && (s % 20 == 0 || s == steps))
        {
            std::cout
                << "[TRANSITION] step "
                << s
                << "/"
                << steps
                << " alpha="
                << alpha
                << std::endl;
        }

        send_nominal_command(alpha);

        std::this_thread::sleep_for(
            std::chrono::duration<double>(dt)
        );
    }

    const double hold_time =
        this->params.Get<double>("policy_nominal_hold_time", 1.5);

    const int hold_steps =
        std::max(1, static_cast<int>(hold_time / dt));

    for (int s = 0; s < hold_steps; ++s)
    {
        send_nominal_command(1.0f);

        std::this_thread::sleep_for(
            std::chrono::duration<double>(dt)
        );
    }

    std::cout
        << "[INFO] Policy nominal default_angles reached and held"
        << std::endl;

    return true;
}

/**
 * @brief Called when the FSM enters locomotion.
 *
 * Called by:
 *  - FSM locomotion state
 *
 * Importance:
 * ------------------------------------------------------------
 * Locomotion is the state where the robot is allowed to use
 * external velocity commands.
 *
 * This method:
 *  - clears stale commands before locomotion starts,
 *  - enables navigation command usage,
 *  - resets internal command values,
 *  - marks locomotion as started,
 *  - allows recovery logic to reset RL buffers after future failures.
 *
 * Why ClearCmdVelUnsafe() is called first:
 * ------------------------------------------------------------
 * A stale velocity command from GetUp, teleop, Nav2, or UDP bridge
 * must not be reused as the first locomotion command.
 */

void RL_Real_G1_EDU12::OnEnterLocomotion()
{
    // Remove any old/stale cmd_vel before entering RL locomotion.
    this->ClearCmdVelUnsafe();

    // Allow RunModel() to listen to /cmd_vel.
    // RunModel() will still require fresh=true before using it.
    this->control.navigation_mode = true;
    this->locomotion_ramp_time_ = 0.0f;
    this->obs.commands = {0.0f, 0.0f, 0.0f};


    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;

    if (this->hw_mode_ == HwMode::REAL && this->publish_lowcmd_)
    {
        if (!this->MoveToPolicyNominalPose())
        {
            std::cerr << "[FATAL] Failed to move to policy nominal pose" << std::endl;
            return;
        }

        //this->HoldRealStableStandPose(0.8);
        this->PrintCurrentLegPose("[AFTER HOLD BEFORE RL]");

        this->policy_takeover_start_pos_.assign(
            this->kNdofPolicy,
            0.0f
        );

        LowState_ st;
        {
            std::lock_guard<std::mutex> lk(this->lowstate_mtx_);
            st = this->unitree_low_state;
        }

        for (int p = 0; p < this->kNdofPolicy; ++p)
        {
            const int ms = this->leg_policy_to_ms_[p];

            if (ms >= 0 &&
                ms < static_cast<int>(st.motor_state().size()))
            {
                this->policy_takeover_start_pos_[p] =
                    st.motor_state()[ms].q();
            }
        }

        this->locomotion_ramp_time_ = 0.0f;
    }
    else
    {
        std::cout
            << "[DRY-RUN] Skipping MoveToPolicyNominalPose because LowCmd publishing is disabled"
            << std::endl;
    }
    
    if (this->model)
    {
        this->model->reset();
    }
    /*
    * Clear previous action history.
    *
    * The policy observation includes previous actions.
    * If not reset, stale actions can destabilize the policy.
    */
    this->obs.actions =
        std::vector<float>(this->kNdofPolicy, 0.0f);

    this->locomotion_started_once_ = true;
    this->allow_recovery_model_reset_ = true;

    std::cout << "[FSM] Enter Locomotion: policy active" << std::endl;
}

/**
 * @brief Called when the FSM exits locomotion.
 *
 * Called by:
 *  - FSM locomotion state
 *
 * Importance:
 * ------------------------------------------------------------
 * When leaving locomotion, the robot must stop accepting walking
 * commands.
 *
 * This prevents /cmd_vel or UDP commands from leaking into passive,
 * GetUp, recovery, or any other non-locomotion state.
 *
 * This method:
 *  - clears all velocity commands,
 *  - disables navigation mode,
 *  - resets internal command values.
 */
void RL_Real_G1_EDU12::OnExitLocomotion()
{
    // Leaving locomotion: stop Nav2 command usage.
    ClearCmdVelUnsafe();

    this->control.navigation_mode = false;
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

// ============================================================
// SECTION 12 — DEBUG / DIAGNOSTICS / OPTIONAL PLOTTING
// ============================================================
#ifdef PLOT

/**
 * @brief Optional real-time plotting helper.
 *
 * Called by:
 *  - loop_plot, only when PLOT is enabled at compile time.
 *
 * Importance:
 * ------------------------------------------------------------
 * This method is only for debugging.
 *
 * It should never be required for real robot execution.
 *
 * EDU12 behavior:
 * ------------------------------------------------------------
 * The policy controls only 12 leg joints.
 *
 * Therefore this plot visualizes only:
 *
 *   leg_policy_to_ms_[0..11]
 *
 * This avoids confusion between:
 *  - policy space,
 *  - full EDU23 hardware space,
 *  - Unitree motor index space.
 */
void RL_Real_G1_EDU12::Plot()
{
    if (!this->policy_table_ready_)
        return;

    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);

    plt::cla();
    plt::clf();

    const int n =
        this->kNdofPolicy;

    for (int p = 0; p < n; ++p)
    {
        const int ms =
            this->leg_policy_to_ms_[p];

        if (ms < 0 ||
            ms >= static_cast<int>(
                this->unitree_low_state.motor_state().size()))
        {
            continue;
        }

        this->plot_real_joint_pos[p].erase(
            this->plot_real_joint_pos[p].begin()
        );

        this->plot_target_joint_pos[p].erase(
            this->plot_target_joint_pos[p].begin()
        );

        this->plot_real_joint_pos[p].push_back(
            this->unitree_low_state.motor_state()[ms].q()
        );

        this->plot_target_joint_pos[p].push_back(
            this->unitree_low_command.motor_cmd()[ms].q()
        );

        plt::subplot(n, 1, p + 1);

        plt::named_plot(
            "_real_joint_pos",
            this->plot_t,
            this->plot_real_joint_pos[p],
            "r"
        );

        plt::named_plot(
            "_target_joint_pos",
            this->plot_t,
            this->plot_target_joint_pos[p],
            "b"
        );

        plt::xlim(
            this->plot_t.front(),
            this->plot_t.back()
        );
    }

    plt::pause(0.0001);
}

#endif


// ============================================================
// SECTION 13 — MAIN ENTRY POINT
// ============================================================
//
// This section starts the correct runtime depending on build mode.
//
// Supported modes:
//  - ROS1
//  - ROS2
//  - Unitree-only
//
// In Unitree-only mode, the network interface is mandatory.
//
// Example:
//
//   ros2 run rl_sar rl_real_g1_edu12_unitree eth0 --dry-run
//
// ============================================================

#if defined(USE_ROS1) && defined(USE_ROS)
static void signalHandler(int)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
    try
    {
#if !defined(USE_ROS)
        if (argc < 2)
        {
            std::cout
                << LOGGER::ERROR
                << "Usage: "
                << argv[0]
                << " networkInterface [--real|--dry-run] [--udp-port=15000]"
                << std::endl;

            return 1;
        }
#endif

#if defined(USE_ROS1) && defined(USE_ROS)

        signal(SIGINT, signalHandler);

        ros::init(
            argc,
            argv,
            "rl_sar_g1_edu12"
        );

        //std::cout << "[MAIN] UNITREE-ONLY branch reached" << std::endl;

        RL_Real_G1_EDU12 app(
            argc,
            argv
        );

        ros::spin();

#elif defined(USE_ROS2) && defined(USE_ROS)

        std::cout
            << "[MAIN] rclcpp::init"
            << std::endl;

        rclcpp::init(
            argc,
            argv
        );

        std::cout
            << "[MAIN] creating RL_Real_G1_EDU12"
            << std::endl;

        auto app =
            std::make_shared<RL_Real_G1_EDU12>(
                argc,
                argv
            );

        std::cout
            << "[MAIN] spinning ros2_node"
            << std::endl;

        rclcpp::spin(
            app->ros2_node
        );

        std::cout
            << "[MAIN] shutdown"
            << std::endl;

        rclcpp::shutdown();

#else
        //std::cout << "[MAIN] UNITREE-ONLY branch reached" << std::endl;
        RL_Real_G1_EDU12 app(
            argc,
            argv
        );

        //std::cout << "[MAIN] app constructed, entering infinite loop" << std::endl;

        while (true)
        {
            //std::cout << "[MAIN] alive" << std::endl;
            sleep(10);
        }


#endif

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FATAL MAIN] Exception: "
            << e.what()
            << std::endl;

        return 245;
    }
    catch (...)
    {
        std::cerr
            << "[FATAL MAIN] Unknown exception"
            << std::endl;

        return 246;
    }
}