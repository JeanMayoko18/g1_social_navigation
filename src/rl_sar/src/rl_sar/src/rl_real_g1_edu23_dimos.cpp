/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * RL_Real_G1_EDU23.cpp
 * ----------------------------------------
 * A safe EDU+ (23 active joints) adapter that keeps a 29-DoF policy interface.
 *
 * Key goals:
 *  1) Allow testing cmd_vel reception (Nav2 / teleop) without touching hardware.
 *  2) Allow testing the whole RL pipeline using a FAKE LowState (no robot required).
 *  3) Allow running on the real robot while masking missing joints (EDU23).
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

#include "rl_real_g1_edu23.hpp"

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

// ----------------------------
// Helper: clamp
// ----------------------------
static inline float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

// ------------------------------------------------------------
// Helper: Extract yaw (Z-axis rotation) from quaternion
// Accepts std::vector<float> of size >= 4
// Quaternion format: [w, x, y, z]
// ------------------------------------------------------------
static float GetYawFromQuat(const std::vector<float>& q)
{
    if (q.size() < 4)
        return 0.0f;  // safety fallback

    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    const float siny = 2.0f * (w*z + x*y);
    const float cosy = 1.0f - 2.0f * (y*y + z*z);

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


static_assert(is_edu23_active_joint(0) == true, "EDU23 joint mask broken");
static_assert(is_edu23_active_joint(14) == false, "EDU23 joint mask broken");

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

// ----------------------------
// Constructor
// ----------------------------
RL_Real_G1_EDU23::RL_Real_G1_EDU23(int argc, char **argv)
{
    std::cout << "[CTOR] begin" << std::endl;
    // ============================================================
    // 0) ROS INTERFACE INITIALIZATION
    // ============================================================
#ifndef USE_ROS
    ParseUnitreeOnlyArgs(argc, argv);
#endif

#if defined(USE_ROS1) && defined(USE_ROS)

    // ------------------------------------------------------------
    // ROS1 cmd_vel subscriber
    // Used mainly for teleoperation / navigation velocity commands.
    // ------------------------------------------------------------
    ros::NodeHandle nh;

    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>(
        "/cmd_vel",
        10,
        &RL_Real_G1_EDU23::CmdvelCallback,
        this
    );

#elif defined(USE_ROS2) && defined(USE_ROS)

    // ------------------------------------------------------------
    // ROS2 node creation
    // This node handles:
    //   - /cmd_vel reception
    //   - /joint_states publication
    //   - optional laptop heartbeat monitoring
    // ------------------------------------------------------------

    std::cout << "[CTOR] before ros2_node create" << std::endl;

    ros2_node = std::make_shared<rclcpp::Node>(
        "rl_real_g1_edu23_node"
    );

    std::cout << "[CTOR] after ros2_node create" << std::endl;


    // ------------------------------------------------------------
    // JointState publisher
    //
    // Purpose:
    //   Publishes active EDU23 joint positions to /joint_states so
    //   robot_state_publisher can generate the TF tree.
    // ------------------------------------------------------------
    joint_state_pub_ =
        ros2_node->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states",
            rclcpp::QoS(rclcpp::KeepLast(50)).best_effort()
        );

    // Names of active joints only.
    tf_joint_names_ = GetEdu23ActiveJointNames();

    RCLCPP_INFO(
        ros2_node->get_logger(),
        "JointState publisher initialized on /joint_states with %zu joints",
        tf_joint_names_.size()
    );

    // ============================================================
    // 0.1) Hardware mode parameters
    // ============================================================

    // hw_mode:
    //   0 = REAL          → real robot LowCmd/LowState
    //   1 = FAKE          → fake lowstate testing
    //   2 = CMD_VEL_ONLY  → only verify cmd_vel reception
    //   3 = DRY_RUN       → run logic without publishing LowCmd
    const int hw_mode_param =
    ros2_node->declare_parameter<int>("hw_mode", 0);
    

    if (hw_mode_param < 0 || hw_mode_param > 3)
    {
        throw std::runtime_error(
            "Invalid hw_mode param: expected 0..3"
        );
    }

    hw_mode_ = static_cast<HwMode>(hw_mode_param);

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
    // Compile-time safety gate for real robot control
    //
    // This prevents accidental execution on real hardware unless the
    // binary was explicitly compiled with ENABLE_REAL_ROBOT.
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
    // 0.2) Navigation / cmd_vel parameters
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

    this->control.navigation_mode = navigation_mode_param_;

    RCLCPP_INFO(
        ros2_node->get_logger(),
        "navigation_mode_force=%d",
        static_cast<int>(navigation_mode_force_)
    );

    // Initialize last cmd_vel time to current time.
    // This avoids an immediate timeout warning during startup.
    //last_cmd_vel_time_ = ros2_node->now();
    last_cmd_vel_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    // ------------------------------------------------------------
    // cmd_vel subscriber
    //
    // The callback stores the command in a mutex-protected variable.
    // ------------------------------------------------------------
    using std::placeholders::_1;

    cmd_vel_subscriber =
        ros2_node->create_subscription<geometry_msgs::msg::Twist>(
            cmd_vel_topic_,
            rclcpp::SystemDefaultsQoS(),
            std::bind(
                &RL_Real_G1_EDU23::CmdvelCallback,
                this,
                _1
            )
        );

    // ============================================================
    // 0.3) Safety parameters: LowState / recovery / stand-hold
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
    // 0.4) Optional laptop heartbeat safety layer
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
                &RL_Real_G1_EDU23::LaptopHeartbeatCallback,
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

    // This identifies the robot configuration namespace.
    this->ang_vel_axis = "body";
    this->robot_name = "g1";

    // Load EDU23-specific base configuration.
    // This file should contain hardware-related information such as:
    //   - active joints
    //   - hardware mapping
    //   - torque limits
    //   - joint names

    std::cout << "[CTOR] before ReadYaml" << std::endl;
    this->ReadYaml(this->robot_name, "base_edu23.yaml");
    std::cout << "[CTOR] after ReadYaml" << std::endl;

    // ------------------------------------------------------------
    // Save a snapshot of important base.yaml keys.
    //
    // Reason:
    //   Later, InitRL(policy_path) may overwrite params.config_node.
    //   This snapshot preserves critical hardware information.
    // ------------------------------------------------------------
    base_snap_.dt =
        this->params.Get<float>("dt", 0.005f);

    base_snap_.decimation =
        this->params.Get<int>("decimation", 4);

    base_snap_.num_of_dofs =
        this->params.Get<int>("num_of_dofs", 29);

    base_snap_.joint_mapping =
        this->params.Get<std::vector<int>>("joint_mapping", {});

    base_snap_.default_dof_pos =
        this->params.Get<std::vector<float>>("default_dof_pos", {});

    base_snap_.torque_limits =
        this->params.Get<std::vector<float>>("torque_limits", {});

    base_snap_.joint_names =
        this->params.Get<std::vector<std::string>>("joint_names", {});

    base_snap_.joint_controller_names =
        this->params.Get<std::vector<std::string>>(
            "joint_controller_names",
            {}
        );

    std::cout << "[INFO] Base hardware YAML snapshot saved\n";

    // ------------------------------------------------------------
    // Debug: print beginning of joint_mapping after base YAML load.
    // ------------------------------------------------------------
    {
        const auto jm =
            this->params.Get<std::vector<int>>("joint_mapping", {});

        std::cout << "[DEBUG] joint_mapping AFTER base.yaml size="
                  << jm.size()
                  << " : ";

        for (int i = 0; i < std::min<int>(
                                static_cast<int>(jm.size()),
                                12);
             ++i)
        {
            std::cout << jm[i] << " ";
        }

        std::cout << "...\n";
    }


    // ============================================================
    // 2) CREATE FINITE STATE MACHINE
    // ============================================================

    // The FSM is responsible for high-level behavior:
    //   - passive mode
    //   - policy mode
    //   - transitions
    //   - keyboard commands
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
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
        std::cout << LOGGER::ERROR
                  << "[FSM] No FSM registered for robot: "
                  << this->robot_name
                  << std::endl;
    }


    // ============================================================
    // 3) INITIALIZE CONTROL CONTAINERS
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

    // Policy interface remains 29 DoF even if EDU23 has 23 active joints.
    this->InitJointNum(this->kNdofPolicy);

    // Initialize output buffers and control structures.
    this->InitOutputs();
    this->InitControl();

    #ifndef USE_ROS
    this->control.navigation_mode = navigation_mode_param_;
    StartIpcCmdVelReceiver(ipc_cmd_port_);
    #endif

    //const bool use_real_unitree_dds = (hw_mode_ == HwMode::REAL) && publish_lowcmd_;

    //const bool use_real_unitree_dds =
    //(hw_mode_ == HwMode::REAL);

    const bool use_unitree_dds =
    (hw_mode_ == HwMode::REAL || hw_mode_ == HwMode::DRY_RUN);


if (use_unitree_dds)
{
    std::cout << "[CTOR] entering Unitree DDS block" << std::endl;

    if (argc < 2)
    {
        throw std::runtime_error(
            "Unitree mode requires network interface argument, e.g. eth0"
        );
    }
    // ============================================================
    // 4) UNITREE DDS INITIALIZATION
    // ============================================================
    std::cout << "[CTOR] before ChannelFactory Init" << std::endl;
    ChannelFactory::Instance()->Init(0, argv[1]);
    std::cout << "[CTOR] after ChannelFactory Init" << std::endl;

    // ============================================================
    // 5) UNITREE DDS PUB/SUB
    // ============================================================

    if (publish_lowcmd_)
    {
        // ============================================================
        // 5.1) UNITREE MOTION SWITCHER 
        // ============================================================

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

        // ------------------------------------------------------------
        // Release any existing motion mode.
        //
        // This prevents conflicts between:
        //   - Unitree built-in controller
        //   - this RL low-level controller
        // ------------------------------------------------------------
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

        // ============================================================
        // 5.2) LOWCMD PUBLISHER
        // ============================================================
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

    // ============================================================
    // 5.3) LOWSTATE & IMU SUBSCRIBERS
    // ============================================================
    // ------------------------------------------------------------
    // LowState subscriber
    // Receives:
    //   - motor states
    //   - IMU
    //   - wireless remote
    //   - machine mode
    // ------------------------------------------------------------
    this->lowstate_subscriber.reset(
        new ChannelSubscriber<LowState_>(HG_STATE_TOPIC)
    );
    this->lowstate_subscriber->InitChannel(
        std::bind(
            &RL_Real_G1_EDU23::LowStateHandler,
            this,
            std::placeholders::_1
        ),
        1
    );

    std::cout << "[CTOR] LowState subscriber enabled" << std::endl;

    // ------------------------------------------------------------
    // Torso IMU subscriber
    // Optional additional IMU source.
    // ------------------------------------------------------------
    this->imutorso_subscriber.reset(
        new ChannelSubscriber<IMUState_>(HG_IMU_TORSO)
    );
    this->imutorso_subscriber->InitChannel(
        std::bind(
            &RL_Real_G1_EDU23::ImuTorsoHandler,
            this,
            std::placeholders::_1
        ),
        1
    );

    std::cout << "[CTOR] IMU subscriber enabled" << std::endl;
}
else
{
    std::cout << "[CTOR] Unitree DDS skipped: non-real or publish_lowcmd=false"
              << std::endl;
}

    // ============================================================
    // 6) CONTROL LOOPS
    // ============================================================

    // ------------------------------------------------------------
    // Keyboard loop
    //
    // Allows switching FSM states from keyboard.
    // Bound to base RL class interface.
    // ------------------------------------------------------------
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
                &RL_Real_G1_EDU23::RobotControl,
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
                &RL_Real_G1_EDU23::RunModel,
                this
            )
        );
    // Startup must begin in RUNNING.
    // Safety is armed later by RobotControl() after the startup grace period.
    // If we start in LOWSTATE_LOST, the robot will enter RECOVERING immediately
    // after GetUp, which interrupts the posture.
    std::cout << "[STARTUP FIX] Forcing safety_state_ to RUNNING before loops" << std::endl;
    safety_state_ = RLSafetyState::RUNNING;
    lowstate_lost_timer_started_ = false;

    // Start loops only after all safety parameters, ROS topics,
    // DDS channels, and control containers are initialized.
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();


    // ============================================================
    // 7) OPTIONAL PLOTTING
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
    // 8) OPTIONAL CSV LOGGING
    // ============================================================

#ifdef CSV_LOGGER

    // Initialize CSV logger for debugging policy outputs,
    // joint states, torques, etc.
    this->CSVInit(this->robot_name);

#endif


    // ============================================================
    // 9) FINAL STARTUP MESSAGE
    // ============================================================

    std::cout << LOGGER::INFO
              << "RL_Real_G1_EDU23 started "
              << "(policy=29dof, robot=EDU+ 23 active joints)"
              << std::endl;
}

bool RL_Real_G1_EDU23::IsStandHoldReady() const
{
    std::lock_guard<std::mutex> lk(stand_hold_mtx_);
    return stand_hold_ready_;
}
bool RL_Real_G1_EDU23::ConfirmLinksLost(
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

#ifndef USE_ROS
void RL_Real_G1_EDU23::ParseUnitreeOnlyArgs(int argc, char** argv)
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


void RL_Real_G1_EDU23::StartIpcCmdVelReceiver(int port)
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

void RL_Real_G1_EDU23::StopIpcCmdVelReceiver()
{
    ipc_running_.store(false);

    if (ipc_cmd_thread_.joinable())
        ipc_cmd_thread_.join();
}

bool RL_Real_G1_EDU23::GetFreshIpcCmdVel(float& vx, float& vy, float& wz) const
{
    std::lock_guard<std::mutex> lk(ipc_cmd_mtx_);

    if (!ipc_cmd_.initialized)
        return false;

    const double age =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ipc_cmd_.stamp
        ).count();

    if (age > cmd_vel_timeout_sec_)
        return false;

    vx = ipc_cmd_.vx;
    vy = ipc_cmd_.vy;
    wz = ipc_cmd_.wz;

    return true;
}

#endif

// ----------------------------
// FSM hook
// ----------------------------
bool RL_Real_G1_EDU23::OnEnterPolicy(const std::string& robot_config_path)
{

    if (this->policy_table_ready_) {
        return true;
    }

    if (!this->InitEdu23PolicyMapping(robot_config_path))
        return false;

    return true;
}

void RL_Real_G1_EDU23::OnEnterGetUp()
{
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
    // Remove any old/stale cmd_vel before entering RL locomotion.
    //if (this->model)
    //{
    //    this->model->reset();
    //}
    // Do NOT reset the ONNX recurrent state here.
    // Resetting here can disturb the transition after GetUp on the real robot.

    this->control.navigation_mode = false;
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}


void RL_Real_G1_EDU23::OnEnterLocomotion()
{
    // Remove any old/stale cmd_vel before entering RL locomotion.
    ClearCmdVelUnsafe();

    // Allow RunModel() to listen to /cmd_vel.
    // RunModel() will still require fresh=true before using it.
    this->control.navigation_mode = true;

    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;

    locomotion_started_once_ = true;
    allow_recovery_model_reset_ = true;

    std::cout << "\n[EDU23 STABLE LEG MOTORS ms=0..11]\n";

    for (int ms = 0; ms < 12; ++ms)
    {
        const float q =
            this->unitree_low_state.motor_state()[ms].q();

        std::cout
            << "ms=" << ms
            << " q=" << q
            << std::endl;
    }


    std::cout << "[FSM] Enter Locomotion: locomotion_started_once_=true" << std::endl;
}

void RL_Real_G1_EDU23::OnExitLocomotion()
{
    // Leaving locomotion: stop Nav2 command usage.
    ClearCmdVelUnsafe();

    this->control.navigation_mode = false;
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

// ----------------------------
// Destructor
// ----------------------------
RL_Real_G1_EDU23::~RL_Real_G1_EDU23()
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

    std::cout << LOGGER::INFO << "RL_Real_G1_EDU23 exit" << std::endl;
}

// ----------------------------
// DDS callbacks
// ----------------------------
/**
 * @brief Handles incoming LowState from DDS and captures a safe recovery posture.
 *
 * This callback:
 *   1) stores the latest LowState in a thread-safe way,
 *   2) updates the LowState freshness timestamp,
 *   3) optionally publishes JointState for TF,
 *   4) saves the last valid standing posture for Stand-Hold recovery.
 */
void RL_Real_G1_EDU23::LowStateHandler(const void *message)
{
    if (!message)
        return;

    const auto now = std::chrono::steady_clock::now();

    const auto* low_state = static_cast<const LowState_*>(message);
    if (!low_state)
        return;

    // Local copy first.
    // This avoids reading the DDS message pointer after the callback context.
    LowState_ local_low_state = *low_state;

    // Store latest LowState safely for GetState_real(), PublishJointStates(), etc.
    {
        std::lock_guard<std::mutex> lk(lowstate_mtx_);
        this->unitree_low_state = local_low_state;
    }

    last_lowstate_tp_ = now;
    lowstate_time_initialized_ = true;
    got_lowstate.store(true, std::memory_order_relaxed);

#if defined(USE_ROS2) && defined(USE_ROS)
    this->PublishJointStates();
#endif

    // Capture a stand-hold posture only when:
    //   - mapping is ready,
    //   - normal control is running,
    //   - posture looks physically reasonable.
    if (!policy_table_ready_)
        return;

    if (safety_state_ != RLSafetyState::RUNNING)
        return;

    const auto& ms_arr = local_low_state.motor_state();
    const int ms_size = static_cast<int>(ms_arr.size());

    if (ms_size <= 0)
        return;

    // ------------------------------------------------------------
    // 1) IMU sanity check
    // ------------------------------------------------------------
    const auto& imu = local_low_state.imu_state();

    const float qw = imu.quaternion()[0];
    const float qx = imu.quaternion()[1];
    const float qy = imu.quaternion()[2];
    const float qz = imu.quaternion()[3];

    const float q_norm =
        std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);

    if (q_norm < 0.8f || q_norm > 1.2f)
        return;

    // Simple roll/pitch instability proxy.
    // If qx/qy are too large, the robot may already be tilted.
    const float roll_pitch_energy = qx*qx + qy*qy;

    if (roll_pitch_energy > 0.15f)
        return;

    // ------------------------------------------------------------
    // 2) Joint velocity sanity check
    // ------------------------------------------------------------
    float max_abs_dq = 0.0f;

    for (int ms = 0; ms < ms_size && ms < kMsTotal; ++ms)
    {
        if (!enabled_ms_mask_[ms])
            continue;

        max_abs_dq = std::max(
            max_abs_dq,
            std::abs(ms_arr[ms].dq())
        );
    }

    if (max_abs_dq > 8.0f)
        return;

    // ------------------------------------------------------------
    // 3) Save last valid posture for Stand-Hold
    // ------------------------------------------------------------
    {
        std::lock_guard<std::mutex> lk(stand_hold_mtx_);

        for (int ms = 0; ms < ms_size && ms < kMsTotal; ++ms)
        {
            if (!enabled_ms_mask_[ms])
                continue;

            last_good_ms_q_[ms] = ms_arr[ms].q();
            last_good_ms_dq_[ms] = ms_arr[ms].dq();
            last_good_ms_valid_[ms] = true;
        }

        stand_hold_ready_ = true;
    }
}

/**
 * @brief Publish ROS2 JointState for robot_state_publisher (TF tree).
 *
 * Key points:
 *  - Uses a thread-safe copy of LowState
 *  - Uses policy_to_ms_ mapping (correct abstraction layer)
 *  - Handles missing joints safely
 */
#if defined(USE_ROS2) && defined(USE_ROS)
void RL_Real_G1_EDU23::PublishJointStates()
{

    // Safety: publisher must exist
    if (!joint_state_pub_)
        return;

    // Safety: we must have received at least one LowState
    if (!this->got_lowstate.load(std::memory_order_relaxed))
        return;

    // ------------------------------------------------------------
    // Thread-safe copy of LowState
    // This avoids reading shared memory while DDS callback writes
    // ------------------------------------------------------------
    LowState_ low_state_copy;
    {
        std::lock_guard<std::mutex> lk(lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    const auto& ms_arr = low_state_copy.motor_state();
    const int ms_size = static_cast<int>(ms_arr.size());

    // ------------------------------------------------------------
    // Build JointState message
    // ------------------------------------------------------------
    sensor_msgs::msg::JointState js;
    js.header.stamp = ros2_node->now();
    js.name = tf_joint_names_;

    js.position.reserve(tf_joint_names_.size());
    js.velocity.reserve(tf_joint_names_.size());
    js.effort.reserve(tf_joint_names_.size());

    // ------------------------------------------------------------
    // Fill JointState using policy → ms mapping
    // ------------------------------------------------------------
    for (int p = 0; p < kNdofPolicy; ++p)
    {
        // Skip joints that do not exist on EDU23
        if (!is_edu23_active_joint(p))
            continue;

        const int ms = policy_to_ms_[p];

        // Safety: invalid mapping
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

    // Publish TF-compatible joint states
    joint_state_pub_->publish(js);

}
#endif




bool RL_Real_G1_EDU23::IsLowStateFresh() const
{
    if (!lowstate_time_initialized_)
        return false;

    const auto now = std::chrono::steady_clock::now();

    const double age = std::chrono::duration<double>(
        now - last_lowstate_tp_
    ).count();

    return age <= lowstate_timeout_sec_;
}

void RL_Real_G1_EDU23::ClearCmdVelUnsafe()
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


void RL_Real_G1_EDU23::ResetRLAfterRecovery()
{
    ClearCmdVelUnsafe();

    this->control.x   = 0.0f;
    this->control.y   = 0.0f;
    this->control.yaw = 0.0f;

    this->obs.commands = {0.0f, 0.0f, 0.0f};

    //while (!output_dof_pos_queue.empty()) output_dof_pos_queue.pop();
    //while (!output_dof_vel_queue.empty()) output_dof_vel_queue.pop();
    //while (!output_dof_tau_queue.empty()) output_dof_tau_queue.pop();

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


    if (this->model && allow_recovery_model_reset_)
    {
        std::cout << "[RESET] ResetRLAfterRecovery -> ONNX recurrent states reset" << std::endl;
        // Recovery relies on RL buffers/state reset only.
        //this->model->reset();

        std::cout << "[TRACE] ResetRLAfterRecovery called. "
          << "locomotion_started_once_=" << locomotion_started_once_
          << " safety_state=" << static_cast<int>(safety_state_)
          << " lowstate_fresh=" << IsLowStateFresh()
          << " got_lowstate=" << got_lowstate.load()
          << std::endl;
    }
    else
    {
        std::cout << "[RESET SKIPPED] First startup/GetUp phase: model reset blocked" << std::endl;
    }
}

/**
 * @brief Monitors connection health and manages transitions between safety states.
 *
 * This function implements a safety state machine based on:
 *   - Low-level robot state (DDS / LowState)
 *   - High-level controller (laptop heartbeat)
 *
 * Safety logic:
 *   RUNNING  → normal RL execution
 *   LOWSTATE_LOST → Stand-Hold (freeze posture)
 *   RECOVERING → wait before re-enabling RL
 *
 * Key idea:
 *   The robot is ONLY allowed to run RL when ALL links are healthy.
 */


void RL_Real_G1_EDU23::UpdateSafetyState()
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
            laptop_fresh = false;

            if (laptop_heartbeat_initialized_)
            {
                const double laptop_age =
                    std::chrono::duration<double>(
                        now - last_laptop_heartbeat_tp_
                    ).count();

                laptop_fresh = (laptop_age <= laptop_heartbeat_timeout_sec_);
            }
        }

        // ------------------------------------------------------------
        // 3) Global link status
        // ------------------------------------------------------------
        const bool links_ok = lowstate_fresh && laptop_fresh;

        // ============================================================
        // STATE: RUNNING (normal RL operation)
        // ============================================================
        if (safety_state_ == RLSafetyState::RUNNING)
        {
            // If ANY link is lost → immediate safety fallback
            if (!links_ok)
            {
                if (!ConfirmLinksLost(links_ok, now))
                    return;

                safety_state_ = RLSafetyState::LOWSTATE_LOST;

                // Start safety timer (used for stand-hold duration)
                lowstate_lost_start_tp_ = now;
                lowstate_lost_timer_started_ = true;

                // Stop all commands immediately
                ClearCmdVelUnsafe();

                #ifndef USE_ROS
                std::cerr << "[SAFETY TRANSITION] RUNNING -> LOWSTATE_LOST"
                        << " lowstate_fresh=" << lowstate_fresh
                        << " laptop_fresh=" << laptop_fresh
                        << std::endl;
                #endif
                #if defined(USE_ROS2) && defined(USE_ROS)
                            const char* reason =
                                !lowstate_fresh ? "Robot LowState/DDS" : "Laptop heartbeat";

                            RCLCPP_ERROR(
                                ros2_node->get_logger(),
                                "CRITICAL: %s lost. Entering Stand-Hold safety mode.",
                                reason
                            );
                #endif
                return;
            }
            else
            {
                ConfirmLinksLost(true, now);
            }
        }

        // ============================================================
        // STATE: LOWSTATE_LOST (Safety mode active)
        // ============================================================
        if (safety_state_ == RLSafetyState::LOWSTATE_LOST)
        {
            // Always enforce zero command while unsafe
            ClearCmdVelUnsafe();

            // Transition only if BOTH links are restored
            if (links_ok)
            {
                ConfirmLinksLost(true, now);

                safety_state_ = RLSafetyState::RECOVERING;

                // Reset timers
                lowstate_lost_timer_started_ = false;
                recovery_start_tp_ = now;

            #ifndef USE_ROS
                    std::cerr << "[SAFETY TRANSITION] LOWSTATE_LOST -> RECOVERING"
                            << std::endl;
            #endif

            #if defined(USE_ROS2) && defined(USE_ROS)
                    RCLCPP_WARN(
                        ros2_node->get_logger(),
                        "Connections restored. Entering RECOVERING (stand-still)..."
                    );
            #endif
            }

            return;
        }

        // ============================================================
        // STATE: RECOVERING (stabilization phase)
        // ============================================================
        if (safety_state_ == RLSafetyState::RECOVERING)
        {
            // Ensure robot remains still during recovery
            ClearCmdVelUnsafe();


            // If failure happens again → go back to safety immediately
            if (!links_ok)
            {
                if (!ConfirmLinksLost(links_ok, now))
                    return;

                safety_state_ = RLSafetyState::LOWSTATE_LOST;
                
                #ifndef USE_ROS
                    std::cerr << "[SAFETY TRANSITION] RECOVERING -> RUNNING" << std::endl;
                #endif

                lowstate_lost_start_tp_ = now;
                lowstate_lost_timer_started_ = true;

                #ifndef USE_ROS
                        std::cerr << "[SAFETY TRANSITION] RECOVERING -> LOWSTATE_LOST"
                                << " lowstate_fresh=" << lowstate_fresh
                                << " laptop_fresh=" << laptop_fresh
                                << std::endl;
                #endif

                #if defined(USE_ROS2) && defined(USE_ROS)
                            const char* reason =
                                !lowstate_fresh ? "Robot LowState/DDS" : "Laptop heartbeat";

                    RCLCPP_ERROR(
                        ros2_node->get_logger(),
                        "CRITICAL: %s lost again during RECOVERING. Returning to Stand-Hold.",
                        reason
                    );
                #endif

                return;
            }
            else
            {
                ConfirmLinksLost(true, now);
            }

            // Wait a stabilization delay before re-enabling RL
            const double rec_age =
                std::chrono::duration<double>(
                    now - recovery_start_tp_
                ).count();

            if (rec_age >= recovery_duration_sec_)
            {
                // Reset RL buffers and history
                if (locomotion_started_once_)
                {
                    ResetRLAfterRecovery();
                }
                else
                {
                    ClearCmdVelUnsafe();
                    this->obs.commands = {0.0f, 0.0f, 0.0f};
                    std::cout << "[RECOVERY] First startup recovery: model reset skipped" << std::endl;
                }


                // Resume normal operation
                safety_state_ = RLSafetyState::RUNNING;


                #ifndef USE_ROS
                    std::cerr << "[SAFETY TRANSITION] RECOVERING -> RUNNING"
                        << std::endl;
                #endif

                #if defined(USE_ROS2) && defined(USE_ROS)
                            RCLCPP_INFO(
                                ros2_node->get_logger(),
                                "Recovery complete. RL policy is now RUNNING."
                            );
                #endif
            }

            return;
        }
}


void RL_Real_G1_EDU23::SendDampingCommandIfPossible()
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
 * @brief Sends a PD command to hold the last valid posture (Safe Guard).
 */
void RL_Real_G1_EDU23::SendStandHoldCommandIfPossible()
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

// ------------------------------------------------------------
// GetState_minimal (IMPLEMENTÉ)
//   => Stable 29DoF state used for:
//      - DRY_RUN / CMD_VEL_ONLY
//      - safety fallback if mapping not ready
//      - startup before LowState arrives
// ------------------------------------------------------------
void RL_Real_G1_EDU23::GetState_minimal(RobotState<float>* state) const
{
    if (!state) return;

    // IMU stable
    state->imu.quaternion = {1.f, 0.f, 0.f, 0.f};
    state->imu.gyroscope  = {0.f, 0.f, 0.f};

    state->motor_state.resize(kNdofPolicy);

    // If policy defaults available use them, else zeros
    const bool have_def = ((int)policy_default_dof_pos_.size() == kNdofPolicy);

    for (int i = 0; i < kNdofPolicy; ++i)
    {
        state->motor_state.q[i]       = have_def ? policy_default_dof_pos_[i] : 0.f;
        state->motor_state.dq[i]      = 0.f;
        state->motor_state.tau_est[i] = 0.f;
    }
}

void RL_Real_G1_EDU23::GetState_real_for_getup(RobotState<float>* state)
{
    if (!state) return;

    GetState_minimal(state);

    if (!got_lowstate.load(std::memory_order_relaxed))
        return;

    LowState_ low_state_copy;
    {
        std::lock_guard<std::mutex> lk(lowstate_mtx_);
        low_state_copy = this->unitree_low_state;
    }

    const auto& imu = low_state_copy.imu_state();

    for (int i = 0; i < 4; ++i)
        state->imu.quaternion[i] = imu.quaternion()[i];

    for (int i = 0; i < 3; ++i)
        state->imu.gyroscope[i] = imu.gyroscope()[i];
}

void RL_Real_G1_EDU23::ImuTorsoHandler(const void *message)
{
    this->unitree_imu_torso = *(const IMUState_ *)message;
}

/**
 * @brief Build mapping table: policy(29 DoF) → Unitree motor_state index
 *
 * This function constructs the final lookup table:
 *
 *      policy index (0..28)
 *              ↓
 *      hw23 index (0..22)
 *              ↓
 *      motor_state / motor_cmd index (ms)
 *
 * The mapping chain is:
 *
 *   kPolicyToHw23[p]      → which EDU23 joint corresponds to policy joint p
 *   hw23_to_ms_[h]        → which Unitree motor_state index corresponds to hw23 joint h
 *
 * Result:
 *   policy_to_ms_[p]      → final Unitree motor index
 *
 * Safety checks performed:
 *   - hw23_to_ms size must be 23
 *   - policy_default_dof_pos must be size 29
 *   - no out-of-range mapping
 *   - no duplicate motor_state targets
 *
 * If any error is detected, function returns false and mapping is invalid.
 */
// RL_Real_G1_EDU23.cpp
// ------------------------------------------------------------
// BuildPolicyToMsTable()
//   Builds the final mapping:
//      policy joint index (0..28)  -->  Unitree motor_state/motor_cmd index
//
//   This is the ONLY mapping that GetState_real() and SetCommand() should use
//   once validated.
//
//   Inputs required BEFORE calling this function:
//     - hw23_to_ms_            : size 23 (from base_edu23.yaml)
//     - policy_default_dof_pos_: size 29 (from policy YAML default_dof_pos)
//
//   Outputs:
//     - policy_to_hw23_[29] : policy -> hw23 slot (or -1)
//     - policy_to_ms_[29]   : policy -> Unitree motor index (or -1)
//     - policy_table_ready_ : true only if all validations passed
//
//   Notes:
//     - We do NOT validate against motor_state().size() here because it is runtime.
//       That check must be done in GetState_real()/SetCommand() using actual sizes.
// ------------------------------------------------------------
/*
    * The mapping logic is as follows:
    *
    *   For each policy joint index p (0..28):
    *     1) Look up which hw23 joint it corresponds to using joint_mapping[p] and kPolicyToHw23.
    *     2) If hw23 index is -1 → joint does not exist on EDU23 → policy_to_ms_[p] = -1
    *     3) Else look up which motor_state index (ms) corresponds to that hw23 joint using hw23_to_ms_.
    *     4) Validate ms is in range and not duplicated.
    *     5) Set policy_to_ms_[p] = ms
    *
    * After this function, GetState_real() and SetCommand() should ONLY use policy_to_ms_ for indexing.
    *
    * Safety checks:
    *   - hw23_to_ms_ size must be 23
    *   - policy_default_dof_pos_ size must be 29
    *   joint_mapping size must be 29
    *   No out-of-range indices in joint_mapping or hw23_to_ms_
    *   No duplicate motor_state targets (each ms can only be mapped once)
    *
    * If any check fails, the function returns false and the mapping is not ready.
    * Critical: If the mapping is not ready, GetState_real() and SetCommand() must NOT use policy_to_ms_ and should fallback to safe defaults.

*/

bool RL_Real_G1_EDU23::BuildPolicyToMsTable()
{
    // ------------------------------------------------------------
    // 0) Safety default
    // ------------------------------------------------------------
    this->policy_table_ready_ = false;
    this->policy_to_ms_.fill(-1);
    this->enabled_ms_mask_.fill(false);

    // Never enable these on EDU23 (waist roll/pitch, wrist pitch/yaw)
    static const std::unordered_set<int> kForbiddenMs = {13, 14, 20, 21, 27, 28};

    // ------------------------------------------------------------
    // 1) Validate HW params
    // ------------------------------------------------------------
    if (this->ms_total_ <= 0 || this->ms_total_ > kMsTotal) {
        std::cerr << "[FATAL] BuildPolicyToMsTable: ms_total_=" << this->ms_total_
                  << " invalid (1.." << kMsTotal << ")\n";
        return false;
    }

    if ((int)this->hw23_to_ms_.size() != this->hw_active_dofs_) {
        std::cerr << "[FATAL] BuildPolicyToMsTable: hw23_to_ms size="
                  << this->hw23_to_ms_.size()
                  << " expected " << this->hw_active_dofs_ << "\n";
        return false;
    }

    if ((int)this->policy_default_dof_pos_.size() != kNdofPolicy) {
        std::cerr << "[FATAL] BuildPolicyToMsTable: policy_default_dof_pos size="
                  << this->policy_default_dof_pos_.size()
                  << " expected " << kNdofPolicy << "\n";
        return false;
    }

    // ------------------------------------------------------------
    // 2) Read policy joint_mapping (CRITICAL)
    // joint_mapping[p] tells which "hw29 joint index" the policy vector index p refers to.
    // If you ignore this, you command the wrong joints.
    // ------------------------------------------------------------
    const auto jm_policy = this->params.Get<std::vector<int>>("joint_mapping", {});
    if ((int)jm_policy.size() != kNdofPolicy) {
        std::cerr << "[FATAL] BuildPolicyToMsTable: joint_mapping size="
                  << jm_policy.size() << " expected " << kNdofPolicy << "\n";
        return false;
    }

    // ------------------------------------------------------------
    // 3) Define hw29 -> hw23 slot mapping (your kPolicyToHw23, but interpreted as hw29-indexed)
    // hw29_to_hw23[hw29] = hw23 slot, or -1 if absent on EDU23
    // ------------------------------------------------------------
    static constexpr int hw29_to_hw23[kNdofPolicy] = {
        // 0..12 legs+waist_yaw
         0,  1,  2,  3,  4,  5,
         6,  7,  8,  9, 10, 11,
        12,
        // 13,14 absent
        -1, -1,
        // 15..19 left arm
        13, 14, 15, 16, 17,
        // 20,21 absent
        -1, -1,
        // 22..26 right arm
        18, 19, 20, 21, 22,
        // 27,28 absent
        -1, -1
    };

    // ------------------------------------------------------------
    // 4) Build policy p -> ms using joint_mapping:
    // hw29 = jm_policy[p]
    // hw23 = hw29_to_hw23[hw29]
    // ms   = hw23_to_ms[hw23]
    // ------------------------------------------------------------
    std::unordered_map<int, int> seen_ms; // ms -> policy p (duplicate check)
    int active = 0;

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        const int hw29 = jm_policy[p];
        if (hw29 < 0 || hw29 >= kNdofPolicy) {
            std::cerr << "[FATAL] BuildPolicyToMsTable: joint_mapping[" << p
                      << "]=" << hw29 << " out of range 0.." << (kNdofPolicy - 1) << "\n";
            return false;
        }

        const int h23 = hw29_to_hw23[hw29];
        if (h23 < 0) {
            // joint absent on EDU23
            this->policy_to_ms_[p] = -1;
            continue;
        }

        if (h23 >= this->hw_active_dofs_) {
            std::cerr << "[FATAL] BuildPolicyToMsTable: hw23 slot out of range: h23="
                      << h23 << " hw_active_dofs_=" << this->hw_active_dofs_
                      << " (p=" << p << " hw29=" << hw29 << ")\n";
            return false;
        }

        const int ms = this->hw23_to_ms_[h23];

        if (ms < 0 || ms >= this->ms_total_) {
            std::cerr << "[FATAL] BuildPolicyToMsTable: hw23_to_ms[" << h23
                      << "]=" << ms << " out of range 0.." << (this->ms_total_ - 1) << "\n";
            return false;
        }

        if (kForbiddenMs.count(ms)) {
            std::cerr << "[FATAL] BuildPolicyToMsTable: forbidden ms=" << ms
                      << " (p=" << p << " hw29=" << hw29 << " h23=" << h23 << ")\n";
            return false;
        }

        // Duplicate check
        if (seen_ms.find(ms) != seen_ms.end()) {
            std::cerr << "[FATAL] BuildPolicyToMsTable: Duplicate ms=" << ms
                      << " mapped by p=" << seen_ms[ms] << " and p=" << p << "\n";
            return false;
        }

        seen_ms[ms] = p;
        this->policy_to_ms_[p] = ms;
        active++;
    }

    if ((int)seen_ms.size() != this->hw_active_dofs_) {
        std::cerr << "[FATAL] BuildPolicyToMsTable: active joints=" << seen_ms.size()
                  << " expected=" << this->hw_active_dofs_ << "\n";
        return false;
    }

    // ------------------------------------------------------------
    // 5) Build allowlist enabled_ms_mask_ (size kMsTotal=35)
    // ------------------------------------------------------------
    for (const auto& kv : seen_ms) {
        const int ms = kv.first;
        if (0 <= ms && ms < this->ms_total_) enabled_ms_mask_[ms] = true;
    }
    for (int bad : kForbiddenMs) {
        if (0 <= bad && bad < this->ms_total_) enabled_ms_mask_[bad] = false;
    }

    // ------------------------------------------------------------
    // 6) Debug prints
    // ------------------------------------------------------------
    std::cout << "policy_to_ms =\n[\n  ";
    for (int p = 0; p < kNdofPolicy; ++p) {
        std::cout << this->policy_to_ms_[p];
        if (p != kNdofPolicy - 1) std::cout << ",";
        if ((p + 1) % 6 == 0 && p != kNdofPolicy - 1) std::cout << "\n  ";
        else std::cout << " ";
    }
    std::cout << "\n]\n";

    std::cout << "[INFO] BuildPolicyToMsTable: mapped " << seen_ms.size()
              << " joints (expected " << this->hw_active_dofs_ << ")\n";

    this->policy_table_ready_ = true;
    this->DumpFullMappingOnce();
    return true;
}

bool RL_Real_G1_EDU23::InitEdu23PolicyMapping(const std::string& robot_config_path)
{
  try
  {
    std::cout << "[ENTER] InitEdu23PolicyMapping(" << robot_config_path << ")\n";

    // ============================================================
    // 0) Snapshot BASE (EDU23) BEFORE InitRL overwrites params
    // ============================================================
    YAML::Node base_full = this->params.config_node;

    // Your YAML is namespaced: g1: { ... }
    // Keep it robust: if "g1" exists use it, else fallback to root.
    YAML::Node base_node = base_full["g1"] ? base_full["g1"] : base_full;

    auto get_int_from = [&](const YAML::Node& n, const std::string& key, int def) -> int {
      return (n[key] && n[key].IsScalar()) ? n[key].as<int>() : def;
    };
    auto get_vec_int_from = [&](const YAML::Node& n, const std::string& key) -> std::vector<int> {
      return (n[key] && n[key].IsSequence()) ? n[key].as<std::vector<int>>() : std::vector<int>{};
    };
    auto get_vec_float_from = [&](const YAML::Node& n, const std::string& key) -> std::vector<float> {
      return (n[key] && n[key].IsSequence()) ? n[key].as<std::vector<float>>() : std::vector<float>{};
    };

    // ============================================================
    // 1) Extract HW-specific parameters FROM BASE (NOT from this->params after InitRL)
    // ============================================================
    this->hw_active_dofs_ = get_int_from(base_node, "hw_active_dofs", this->kNdofHw23);
    this->ms_total_       = get_int_from(base_node, "ms_total",       this->kMsTotal);

    // hw23_to_ms MUST live in base_edu23.yaml (inside g1:)
    this->hw23_to_ms_     = get_vec_int_from(base_node, "hw23_to_ms");

    if (this->hw_active_dofs_ != this->kNdofHw23)
    {
      std::cout << "[WARN] base hw_active_dofs=" << this->hw_active_dofs_
                << " but kNdofHw23=" << this->kNdofHw23
                << " (ok if you intentionally changed HW active joints)\n";
    }

    if ((int)this->hw23_to_ms_.size() != this->hw_active_dofs_)
    {
      std::cerr << "[FATAL] base hw23_to_ms size=" << this->hw23_to_ms_.size()
                << " expected " << this->hw_active_dofs_
                << " (add/verify hw23_to_ms under g1: in base_edu23.yaml)\n";
      return false;
    }

    // Optional: keep base default posture (HW space, 23) if you store it
    this->base_default_dof_pos_ = get_vec_float_from(base_node, "default_dof_pos");
    if (!this->base_default_dof_pos_.empty() &&
        (int)this->base_default_dof_pos_.size() != this->hw_active_dofs_)
    {
      std::cout << "[WARN] base default_dof_pos size=" << this->base_default_dof_pos_.size()
                << " expected " << this->hw_active_dofs_
                << " (ignored)\n";
      this->base_default_dof_pos_.clear();
    }

    // ============================================================
    // 2) Load POLICY YAML (29 DoF) -> overwrites this->params.config_node
    // ============================================================
    //this->InitRL(robot_config_path);                 // loads policy config.yaml
    std::cout << "[TRACE] Before InitRL()" << std::endl;
    this->InitRL(robot_config_path); // loads policy config.yaml
    std::cout << "[TRACE] After InitRL()" << std::endl;
    YAML::Node policy_full = this->params.config_node;

    // Policy YAML is typically namespaced too (e.g. g1/robomimic/locomotion: {...})
    // But your Params::Get() likely already resolves the correct section via robot_config_path.
    // Still keep a node handle for merging if needed:
    YAML::Node policy_node = policy_full;

    // ============================================================
    // 3) Extract policy arrays into C++ members (source of truth)
    // ============================================================
    this->policy_default_dof_pos_ = this->params.Get<std::vector<float>>("default_dof_pos", {});
    if ((int)this->policy_default_dof_pos_.size() != this->kNdofPolicy)
    {
      std::cerr << "[FATAL] policy default_dof_pos size="
                << this->policy_default_dof_pos_.size()
                << " expected " << this->kNdofPolicy << "\n";
      return false;
    }
    this->use_dimos_policy_ =
    this->params.Get<bool>("use_dimos_policy", false);

    std::cout << "[POLICY MODE] use_dimos_policy_="
            << static_cast<int>(this->use_dimos_policy_)
            << std::endl;

    // Optional debug: policy joint_mapping
    const auto jm_policy = this->params.Get<std::vector<int>>("joint_mapping", {});
    if (!jm_policy.empty() && (int)jm_policy.size() != this->kNdofPolicy)
    {
      std::cerr << "[WARN] policy joint_mapping size=" << jm_policy.size()
                << " expected " << this->kNdofPolicy
                << " (not fatal if you don't use it)\n";
    }

    // ============================================================
    // 4) Merge ONLY RL keys from policy into BASE node (namespaced: g1: or g1_dimos)
    // ============================================================
    static const std::unordered_set<std::string> kPolicyKeys = {
      "model_name",
      "num_observations",
      "observations",
      "observations_history",
      "observations_history_priority",
      "clip_obs",
      "clip_actions_lower",
      "clip_actions_upper",
      "rl_kp",
      "rl_kd",
      "fixed_kp",
      "fixed_kd",
      "action_scale",
      "lin_vel_scale",
      "ang_vel_scale",
      "dof_pos_scale",
      "dof_vel_scale",
      "commands_scale",
      "torque_limits"
    };

    // IMPORTANT:
    // - We merge into base_node (g1 namespace)
    // - We do NOT destroy hw fields (hw23_to_ms, hw_active_dofs, ms_total, etc.)
    // - We force num_of_dofs to policy space (29) for RL/FSM math.
    for (const auto& k : kPolicyKeys)
    {
      if (policy_node[k] && policy_node[k].IsDefined())
        base_node[k] = policy_node[k];
    }

    // Keep policy-space critical arrays for RL logic
    if (policy_node["joint_mapping"] && policy_node["joint_mapping"].IsDefined())
      base_node["joint_mapping"] = policy_node["joint_mapping"];
    if (policy_node["default_dof_pos"] && policy_node["default_dof_pos"].IsDefined())
      base_node["default_dof_pos"] = policy_node["default_dof_pos"];

    // Debug copies (optional)
    base_node["policy_default_dof_pos"] = base_node["default_dof_pos"];
    base_node["policy_joint_mapping"]   = base_node["joint_mapping"];

    // Force RL dimension (FSM/RL arrays use 29)
    base_node["num_of_dofs"] = this->kNdofPolicy;

    // ============================================================
    // 5) Restore BASE as active runtime params (merged) — keep namespace
    // ============================================================
    if (base_full["g1"])
      base_full["g1"] = base_node;
    else
      base_full = base_node;

    this->params.config_node = base_full;

    // ============================================================
    // 6) Build policy→ms lookup table (uses hw23_to_ms_ + kPolicyToHw23)
    // ============================================================
    if (!this->BuildPolicyToMsTable())
    {
      std::cerr << "[FATAL] BuildPolicyToMsTable failed\n";
      return false;
    }

    // ============================================================
    // 7) Debug
    // ============================================================
    std::cout << "[DEBUG] hw_active_dofs=" << this->hw_active_dofs_
              << " ms_total=" << this->ms_total_
              << " hw23_to_ms size=" << this->hw23_to_ms_.size() << "\n";
    std::cout << "[DEBUG] policy_default_dof_pos size=" << this->policy_default_dof_pos_.size() << "\n";
    std::cout << "[DEBUG] policy_table_ready=" << (int)this->policy_table_ready_ << "\n";
    std::cout << "[OK] policy_table_ready_=" << (int)this->policy_table_ready_ << "\n";

    return true;
  }
  catch (const std::exception& e)
  {
    std::cerr << "[FATAL] InitEdu23PolicyMapping exception: " << e.what() << "\n";
    return false;
  }
}


// ------------------------------------------------------------
// GetState dispatch
//   => Toujours un state valide via GetState_minimal()
// ------------------------------------------------------------
void RL_Real_G1_EDU23::GetState(RobotState<float> *state)
{
    GetState_minimal(state);

    switch (hw_mode_)
    {
        case HwMode::FAKE_LOWSTATE:
            GetState_test(state);   // overwrite fake values
            break;

        case HwMode::REAL:
            GetState_real(state);   // overwrite only mapped joints + real IMU
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

// ----------------------------
// GetState_real (EDU23 mask applied)
// ----------------------------
/**
 * @brief Builds the RL state using real robot data.
 *
 * Design:
 *  - Always starts from a safe minimal 29-DoF state.
 *  - Uses a thread-safe copy of LowState.
 *  - Overwrites only mapped and valid EDU23 joints.
 */
void RL_Real_G1_EDU23::GetState_real(RobotState<float>* state)
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
        const int ms = policy_to_ms_[p];

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
void RL_Real_G1_EDU23::GetState_test(RobotState<float> *state)
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

void RL_Real_G1_EDU23::DumpFullMappingOnce() const
{
    static bool done = false;
    if (done) return;
    done = true;

    const auto jm_policy = this->params.Get<std::vector<int>>("joint_mapping", {});

    std::cout << "\n=== Full Mapping Debug ===\n";
    std::cout << "p | hw29 | hw23 | ms\n";
    std::cout << "--------------------------\n";

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        int hw29 = -1;
        int h23  = -1;
        int ms   = policy_to_ms_[p];

        if ((int)jm_policy.size() == kNdofPolicy)
            hw29 = jm_policy[p];

        // reconstruct hw23 from policy_to_ms_ if needed
        if (ms >= 0) {
            // find which hw23 slot maps to this ms
            for (int h = 0; h < hw_active_dofs_; ++h)
                if (hw23_to_ms_[h] == ms)
                    h23 = h;
        }

        std::cout << p << " | "
                  << hw29 << " | "
                  << h23  << " | "
                  << ms   << "\n";
    }

    std::cout << "===========================\n\n";
}


// ----------------------------
// SetCommand (EDU23 mask applied)
// ----------------------------
void RL_Real_G1_EDU23::SetCommand(const RobotCommand<float>* command)
{
    // --------------------------------------------------------------------
    // SAFETY GUARDS
    // --------------------------------------------------------------------
    if (!command) return;

    // Only REAL mode is allowed to publish LowCmd to hardware.
    if (this->hw_mode_ != HwMode::REAL) return;

    // Mapping/allowlist must be ready (policy 29 -> ms 35)
    if (!this->policy_table_ready_) {
    #if defined(USE_ROS2) && defined(USE_ROS)
        RCLCPP_ERROR_THROTTLE(
            ros2_node->get_logger(), *ros2_node->get_clock(), 2000,
            "SetCommand blocked: policy_table_ready_=false"
        );
    #else
        static int cnt = 0;
        if ((cnt++ % 200) == 0)
            std::cerr << "[ERROR] SetCommand blocked: policy_table_ready_=false\n";
    #endif
        return;
    }

    // Guard: policy-space command must be 29 DoF everywhere
    if ((int)command->motor_command.q.size()   != kNdofPolicy ||
        (int)command->motor_command.dq.size()  != kNdofPolicy ||
        (int)command->motor_command.kp.size()  != kNdofPolicy ||
        (int)command->motor_command.kd.size()  != kNdofPolicy ||
        (int)command->motor_command.tau.size() != kNdofPolicy)
    {
    #if defined(USE_ROS2) && defined(USE_ROS)
        RCLCPP_ERROR_THROTTLE(
            ros2_node->get_logger(), *ros2_node->get_clock(), 2000,
            "SetCommand blocked: command vectors size mismatch (expected 29)"
        );
    #else
        static int cnt = 0;
        if ((cnt++ % 200) == 0)
            std::cerr << "[FATAL] SetCommand: command vectors size mismatch (expected 29)\n";
    #endif
        return;
    }
    // --------------------------------------------------------------------
    // HEADER FIELDS (Unitree protocol)
    // --------------------------------------------------------------------

    // Header fields (Unitree protocol)
    this->unitree_low_command.mode_pr()      = static_cast<uint8_t>(this->mode_pr);
    this->unitree_low_command.mode_machine() = this->mode_machine;

    auto& mc = this->unitree_low_command.motor_cmd(); // IDL array size 35 on HG
    const int mc_size = static_cast<int>(mc.size());  // should be 35
    // --------------------------------------------------------------------
    // Step 0: HARD SAFE BASELINE
    // Disable all motors first.
    //
    // Why:
    //  - Prevents stale commands from previous cycles
    //  - Guarantees absent joints stay disabled
    //  - Makes command projection deterministic
    // --------------------------------------------------------------------
    for (int ms = 0; ms < mc_size; ++ms)
    {
          
        auto& m = mc[ms];
        
        // Default fully safe values
        m.mode() = 0;     // disabled
        m.q()    = 0.f;
        m.dq()   = 0.f;
        m.kp()   = 0.f;
        m.kd()   = 0.f;
        m.tau()  = 0.f;
    }

    // --------------------------------------------------------------------
    // Optional: torque clamping
    // - If you have torque limits in policy space, clamp there.
    // - If you only have limits in hardware space, clamp after mapping.
    // Here: we clamp in policy space if available (size 29).
    // --------------------------------------------------------------------

    const auto torque_limits_policy = this->params.Get<std::vector<float>>("torque_limits", {});
    const bool has_policy_limits = ((int)torque_limits_policy.size() == kNdofPolicy);

    // --------------------------------------------------------------------
    // Step 1: PROJECT POLICY COMMANDS (29 DoF) -> HW motor_cmd indices (35) using policy_to_ms_
    //
    // policy_to_ms_[p] gives:
    //  - ms index in unitree motor_cmd array
    //  - or -1 if joint does not exist on EDU23
    //
    // This is the ONLY mapping used here (same as GetState_real()).
    // --------------------------------------------------------------------

    for (int p = 0; p < kNdofPolicy; ++p)
    {
        const int ms = this->policy_to_ms_[p];

        // absent joint or invalid mapping
        if (ms < 0 || ms >= mc_size) continue;

        // strict allowlist (must be size 35)
        if (!enabled_ms_mask_[ms]) continue;

        auto& m = mc[ms];

        // enable this motor channel
        m.mode() = 1;

        // targets
        m.q()  = command->motor_command.q[p];
        m.dq() = command->motor_command.dq[p];
        m.kp() = command->motor_command.kp[p];
        m.kd() = command->motor_command.kd[p];

        float tau = command->motor_command.tau[p];

        // policy torque limits
        if (has_policy_limits) {
            const float lim = std::abs(torque_limits_policy[p]);
            if (lim > 0.f) {
                if (tau >  lim) tau =  lim;
                if (tau < -lim) tau = -lim;
            }
        }

        m.tau() = tau;
    }

    // --------------------------------------------------------------------
    // Step 2: CRC (Cyclic Redundancy Check) + publish
    // --------------------------------------------------------------------
    this->unitree_low_command.crc() =
        this->Crc32Core(
            reinterpret_cast<uint32_t*>(&this->unitree_low_command),
            (sizeof(LowCmd_) >> 2) - 1
        );

    this->lowcmd_publisher->Write(this->unitree_low_command);

}

// ----------------------------
// RobotControl loop
// ----------------------------
void RL_Real_G1_EDU23::RobotControl_old()
{
    #ifdef CMD_VEL_ONLY
        return;
    #endif
            // Runtime guards
        if (hw_mode_ == HwMode::CMD_VEL_ONLY) return;

    // ------------------------------------------------------------
    // 1) Always provide a valid state to the FSM
    //    - if mapping not ready => minimal stable state (29DoF)
    //    - else => full state (real/fake)
    // ------------------------------------------------------------
    if (!this->policy_table_ready_)
    {
#if defined(USE_ROS2) && defined(USE_ROS)
        RCLCPP_WARN_THROTTLE(
            ros2_node->get_logger(), *ros2_node->get_clock(), 2000,
            "[WAIT] policy_table_ready_=false (mapping not initialized yet) -> running FSM with minimal state"
        );
#else
        static int cnt0 = 0;
        if ((cnt0++ % 200) == 0)
            std::cerr << "[WAIT] policy_table_ready_=false -> running FSM with minimal state\n";
            std::cout << "[DBG] got_lowstate=" << (int)this->got_lowstate.load() << "\n";
#endif

        // IMPORTANT: do NOT return here!
        this->GetState_minimal(&this->robot_state);
    }
    else
    {
        this->GetState(&this->robot_state);
    }

    // ------------------------------------------------------------
    // 2) Block ONLY if REAL and no LowState yet
    //    (but still allow FSM in PASSIVE to run if you want)
    //    Here: we allow FSM to run even without LowState, but we won't send commands.
    // ------------------------------------------------------------
#if defined(USE_ROS2) && defined(USE_ROS)
    RCLCPP_INFO_THROTTLE(
        ros2_node->get_logger(), *ros2_node->get_clock(), 1000,
        "[DBG] mapping=%d got_lowstate=%d",
        (int)this->policy_table_ready_,
        (int)this->got_lowstate.load(std::memory_order_relaxed)
    );
#endif

    // ------------------------------------------------------------
    // 3) Always run the FSM (so Num0 works)
    // ------------------------------------------------------------
    this->StateController(&this->robot_state, &this->robot_command);
    this->control.ClearInput();

    // ------------------------------------------------------------
    // 4) Only publish to hardware when everything is ready
    // ------------------------------------------------------------
    if (hw_mode_ == HwMode::REAL &&
        this->policy_table_ready_ &&
        this->got_lowstate.load(std::memory_order_relaxed) &&
        publish_lowcmd_)
    {
        this->SetCommand(&this->robot_command);
    }
}


/**
 * @brief Main real-time robot control loop.
 *
 * Startup logic:
 *   1) Before policy mapping is ready:
 *        - keep FSM alive
 *        - do NOT apply safety watchdog yet
 *
 *   2) When policy becomes ready:
 *        - allow GetUp / Policy Ready to complete
 *        - start a grace timer
 *
 *   3) After grace period:
 *        - enable LowState / heartbeat safety mechanisms
 *
 * This avoids blocking the robot before it reaches the ready standing state.
 */
void RL_Real_G1_EDU23::RobotControl()
{
#ifdef CMD_VEL_ONLY
    return;
#endif

    if (hw_mode_ == HwMode::CMD_VEL_ONLY)
        return;

    // ------------------------------------------------------------
    // 1) POLICY NOT READY YET
    //
    // At startup, policy_table_ready_ is false.
    // This phase must NOT trigger safety fallback, otherwise the FSM
    // may never reach GetUp / policy initialization.
    // ------------------------------------------------------------
    if (!this->policy_table_ready_)
    {
        #if defined(USE_ROS2) && defined(USE_ROS)
            RCLCPP_WARN_THROTTLE(
                ros2_node->get_logger(),
                *ros2_node->get_clock(),
                2000,
                "[STARTUP] policy_table_ready_=false -> FSM allowed, safety not armed yet"
            );
        #endif

            // Use real IMU if LowState is available, even before policy mapping.
            // This improves GetUp stability without requiring policy_to_ms_.
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

            // During startup/GetUp, navigation must stay disabled.
            this->control.navigation_mode = false;

            // Let keyboard/FSM transitions work normally.
            this->StateController(&this->robot_state, &this->robot_command);
            this->control.ClearInput();

            return;
    }

    // ------------------------------------------------------------
    // 2) POLICY JUST BECAME READY
    //
    // The first time policy_table_ready_ becomes true, we start a
    // grace period. During this period, GetUp / ready transition is
    // allowed without watchdog intervention.
    // ------------------------------------------------------------
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
            std::chrono::steady_clock::now() - policy_ready_tp
        ).count();

    // Grace period after policy initialization.
    // It gives enough time for GetUp / ready posture to complete.
    const double startup_safety_grace_sec = 3.0;

    const bool safety_armed =
        startup_age >= startup_safety_grace_sec;

    if (ShouldPrintEvery("runtime_health", 3.0))
    {
        std::cout << "[HEALTH]"
                << " lowstate_fresh=" << static_cast<int>(IsLowStateFresh())
                << " got_lowstate=" << static_cast<int>(got_lowstate.load())
                << " safety=" << static_cast<int>(safety_state_)
                << " policy_ready=" << static_cast<int>(policy_table_ready_)
                << " publish_lowcmd=" << static_cast<int>(publish_lowcmd_)
                << " mode_machine=" << static_cast<int>(mode_machine)
                << " safety_armed=" << static_cast<int>(safety_armed)
                << std::endl;
    }
    // ------------------------------------------------------------
    // 3) SAFETY CHECK ONLY AFTER STARTUP GRACE
    // ------------------------------------------------------------
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

    // ------------------------------------------------------------
    // 4) NORMAL STATE UPDATE
    //
    // During grace period:
    //   - use real state if available
    //   - do not block GetUp / ready transition
    // ------------------------------------------------------------
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

    // ------------------------------------------------------------
    // 5) RUN FSM
    //
    // This is where GetUp / locomotion / passive transitions are
    // executed.
    // ------------------------------------------------------------
    this->StateController(&this->robot_state, &this->robot_command);
    this->control.ClearInput();
    //std::cout << "[KEY] current="
    //      << static_cast<int>(control.current_keyboard)
    //      << " last="
    //      << static_cast<int>(control.last_keyboard)
    //      << std::endl;

    // ------------------------------------------------------------
    // 6) PUBLISH LOWCMD
    //
    // During startup grace:
    //   - allow commands so GetUp can finish
    //
    // After safety is armed:
    //   - publish only if safety_state_ is RUNNING
    // ------------------------------------------------------------
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

// ------------------------------------------------------------------
// Helper (put in .cpp, and declare in header as:
// bool GetFreshCmdVel(float& vx, float& vy, float& wz) const; )
// ------------------------------------------------------------------
#if defined(USE_ROS)
bool RL_Real_G1_EDU23::GetFreshCmdVel(float &vx, float &vy, float &wz) const
{
#if defined(USE_ROS2) && defined(USE_ROS)
    geometry_msgs::msg::Twist cv;
    rclcpp::Time last_t;
    {
        std::lock_guard<std::mutex> lk(this->cmd_vel_mtx_);
        cv     = this->cmd_vel_; 
        last_t = this->last_cmd_vel_time_;
    }

    if (last_t.nanoseconds() == 0) return false;

    const auto now = ros2_node->now();
    const double age = (now - last_t).seconds();
    if (age > this->cmd_vel_timeout_sec_) return false;

    vx = static_cast<float>(cv.linear.x);
    vy = static_cast<float>(cv.linear.y);
    wz = static_cast<float>(cv.angular.z);
    return true;

#elif defined(USE_ROS1) && defined(USE_ROS)
    vx = static_cast<float>(this->cmd_vel.linear.x);
    vy = static_cast<float>(this->cmd_vel.linear.y);
    wz = static_cast<float>(this->cmd_vel.angular.z);
    return true;
#else
    (void)vx; (void)vy; (void)wz;
    return false;
#endif
}
#endif


// ------------------------------------------------------------------
// RunModel() (drop-in replacement)
// ------------------------------------------------------------------
void RL_Real_G1_EDU23::RunModels()
{

    if (safety_state_ != RLSafetyState::RUNNING)
    {
        ClearCmdVelUnsafe();
        this->obs.commands = {0.0f, 0.0f, 0.0f};
        return;
    }

    // ------------------------------------------------------------
    // 0) Early exits
    // ------------------------------------------------------------
    if (this->cmd_vel_only_) return;
    if (!this->rl_init_done) return;
    if (hw_mode_ == HwMode::FAKE_LOWSTATE && !fake_rl_full_) return;

    // ------------------------------------------------------------
    // 1) Update base observation fields
    // ------------------------------------------------------------
    this->episode_length_buf += 1;

    this->obs.ang_vel   = this->robot_state.imu.gyroscope;
    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.dof_pos   = this->robot_state.motor_state.q;
    this->obs.dof_vel   = this->robot_state.motor_state.dq;

    // Default control inputs (joystick / gamepad)
    float vx = this->control.x;
    float vy = this->control.y;
    float wz = this->control.yaw;

    // ------------------------------------------------------------
    // 2) Navigation mode override using /cmd_vel 
    // ------------------------------------------------------------
#if defined(USE_ROS) && (defined(USE_ROS1) || defined(USE_ROS2))

    const bool nav_on =
#if defined(USE_ROS2)
        (navigation_mode_force_ ? true : this->control.navigation_mode);
#else
        this->control.navigation_mode;
#endif

    if (nav_on)
    {
        float vxf = 0.f, vyf = 0.f, wzf = 0.f;
        const bool fresh = GetFreshCmdVel(vxf, vyf, wzf);

        if (fresh)
        {
            vx = vxf;
            vy = vyf;
            wz = wzf;
        }
        else
        {
            // No recent cmd_vel → stop safely
            vx = 0.f;
            vy = 0.f;
            wz = 0.f;
        }

        // Keep control struct coherent
        this->control.x   = vx;
        this->control.y   = vy;
        this->control.yaw = wz;
    }

#endif

#ifndef USE_ROS
    {
        float vxf = 0.0f;
        float vyf = 0.0f;
        float wzf = 0.0f;

        const bool fresh = GetFreshIpcCmdVel(vxf, vyf, wzf);

        if (fresh)
        {
            vx = vxf;
            vy = vyf;
            wz = wzf;
        }
        else
        {
            vx = 0.0f;
            vy = 0.0f;
            wz = 0.0f;
        }

        this->control.x = vx;
        this->control.y = vy;
        this->control.yaw = wz;

        static int cnt = 0;
        if ((cnt++ % 50) == 0)
        {
            std::cout << "[IPC NAV] fresh=" << fresh
                      << " vx=" << vx
                      << " vy=" << vy
                      << " wz=" << wz
                      << std::endl;
        }
    }
#endif

    // ------------------------------------------------------------
    // 3) HEADING STABILIZATION (Closed-loop yaw correction)
    // ------------------------------------------------------------
    // This corrects slow drift when moving forward with wz ≈ 0.
    // Only activates when:
    //  - Forward motion is significant
    //  - No intentional turning command
    // ------------------------------------------------------------

    static float yaw_reference = 0.f;
    static bool  yaw_initialized = false;

    const float yaw_real = GetYawFromQuat(this->robot_state.imu.quaternion);

    if (!yaw_initialized)
    {
        yaw_reference = yaw_real;
        yaw_initialized = true;
    }

    const float forward_threshold = 0.15f;   // minimum vx to activate stabilizer
    const float turn_threshold    = 0.08f;   // if wz bigger than this → intentional turn
    const float kp_heading        = 1.2f;    // proportional gain (tune 0.6 – 1.5)

    if (std::abs(vx) > forward_threshold &&
        std::abs(wz) < turn_threshold)
    {
        // Compute shortest angular error
        float yaw_error = yaw_reference - yaw_real;

        // Normalize error to [-pi, pi]
        while (yaw_error >  M_PI) yaw_error -= 2.0f * M_PI;
        while (yaw_error < -M_PI) yaw_error += 2.0f * M_PI;

        // Apply proportional correction
        wz += kp_heading * yaw_error;
    }
    else
    {
        // If user is turning intentionally,
        // update reference to current yaw
        yaw_reference = yaw_real;
    }

    // ------------------------------------------------------------
    // 4) Feed corrected commands into RL observation
    // ------------------------------------------------------------
    this->obs.commands = {vx, vy, wz};

#if defined(USE_ROS2) && defined(USE_ROS)
    RCLCPP_INFO_THROTTLE(
        ros2_node->get_logger(), *ros2_node->get_clock(), 500,
        "[OBS.CMD corrected] vx=%.2f vy=%.2f wz=%.2f",
        vx, vy, wz
    );
#endif

    // ------------------------------------------------------------
    // 5) Forward policy
    // ------------------------------------------------------------
    this->obs.actions = this->Forward();

    // ------------------------------------------------------------
    // 6) Convert actions → motor targets
    // ------------------------------------------------------------
    this->ComputeOutput(
        this->obs.actions,
        this->output_dof_pos,
        this->output_dof_vel,
        this->output_dof_tau
    );

    if (!this->output_dof_pos.empty())
        output_dof_pos_queue.push(this->output_dof_pos);

    if (!this->output_dof_vel.empty())
        output_dof_vel_queue.push(this->output_dof_vel);

    if (!this->output_dof_tau.empty())
        output_dof_tau_queue.push(this->output_dof_tau);
}

void RL_Real_G1_EDU23::RunModel()
{

    if (hw_mode_ == HwMode::REAL &&
        safety_state_ != RLSafetyState::RUNNING)
    {
        ClearCmdVelUnsafe();
        this->obs.commands = {0.0f, 0.0f, 0.0f};
        return;
    }

    // ----------------------------
    // Fast exits
    // ----------------------------
    if (this->cmd_vel_only_) return;
    if (!this->rl_init_done) return;
    if (hw_mode_ == HwMode::FAKE_LOWSTATE && !fake_rl_full_) return;

    // ----------------------------
    // 1) Update episode + base obs
    // ----------------------------
    this->episode_length_buf += 1;

    this->obs.ang_vel   = this->robot_state.imu.gyroscope;
    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.dof_pos   = this->robot_state.motor_state.q;
    this->obs.dof_vel   = this->robot_state.motor_state.dq;

    // Default (gamepad/joystick)
    float vx = this->control.x;
    float vy = this->control.y;
    float wz = this->control.yaw;


    // ----------------------------
    // 2) Optional navigation mode => /cmd_vel
    // ----------------------------
    #if defined(USE_ROS) && (defined(USE_ROS1) || defined(USE_ROS2))

        // ------------------------------------------------------------
        // requested_nav_on:
        //   Means: the system is allowed/requested to use /cmd_vel.
        //
        // nav_active:
        //   Means: /cmd_vel is actually usable now because it is fresh.
        //
        // Important safety rule:
        //   requested_nav_on=true + fresh=false must NOT be considered
        //   active navigation.
        // ------------------------------------------------------------

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
    {
        float vxf = 0.0f;
        float vyf = 0.0f;
        float wzf = 0.0f;

        const bool fresh = GetFreshIpcCmdVel(vxf, vyf, wzf);

        const bool nav_requested =
            navigation_mode_force_ || this->control.navigation_mode;

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
            std::cout << "[IPC NAV] requested=" << static_cast<int>(nav_requested)
                    << " fresh=" << static_cast<int>(fresh)
                    << " cmd=[" << vx << " " << vy << " " << wz << "]"
                    << " policy=ACTIVE_IDLE_IF_NO_CMD"
                    << std::endl;
        }
    }
    #endif

    // ----------------------------
    // 3) Feed policy
    // ----------------------------
    this->obs.commands = {vx, vy, wz};

    #ifndef USE_ROS
    if (ShouldPrintEvery("obs_cmd", 3.0))
    {
        std::cout << "[OBS.CMD] ["
                << this->obs.commands[0] << " "
                << this->obs.commands[1] << " "
                << this->obs.commands[2] << "]"
                << std::endl;
    }
    #endif

    #if defined(USE_ROS2) && defined(USE_ROS)

    
        RCLCPP_INFO_THROTTLE(
            ros2_node->get_logger(), *ros2_node->get_clock(), 1000 /*ms*/,
            "[OBS.CMD] [%.2f %.2f %.2f]",
            this->obs.commands[0], this->obs.commands[1], this->obs.commands[2]
        );
    #endif

    // ----------------------------
    // 4) Forward + post-processing
    // ----------------------------
    this->obs.actions = this->Forward();

    this->ComputeOutput(
        this->obs.actions,
        this->output_dof_pos,
        this->output_dof_vel,
        this->output_dof_tau
    );

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


// ----------------------------
// Forward (unchanged)
// ----------------------------
std::vector<float> RL_Real_G1_EDU23::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
        return this->obs.actions;

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() &&
        !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions,
                     this->params.Get<std::vector<float>>("clip_actions_lower"),
                     this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    return actions;
}

// ----------------------------
// CRC32
// ----------------------------
uint32_t RL_Real_G1_EDU23::Crc32Core(uint32_t *ptr, uint32_t len)
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

// ----------------------------
// InitLowCmd (SAFE)
// ----------------------------
void RL_Real_G1_EDU23::InitLowCmd()
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

// ----------------------------
// Plot (optional)
// ----------------------------
#ifdef PLOT
void RL_Real_G1_EDU23::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);

    plt::cla();
    plt::clf();

    // NOTE: These plots use raw motor_state index i, not policy mapping.
    // If you need correct mapping, map i->hw_idx before indexing motor_state().
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());

        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state()[i].q());
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd()[i].q());

        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    plt::pause(0.0001);
}
#endif

// ----------------------------
// ROS cmd_vel callback
// ----------------------------

/**
 * @brief Receives /cmd_vel and stores it safely.
 *
 * Key points:
 *  - Thread-safe storage
 *  - Timestamp update
 *  - No duplicate writes (IMPORTANT FIX)
 */
//#if !defined(USE_CMAKE) && defined(USE_ROS)
#if defined(USE_ROS)

/**
 * @brief Receives /cmd_vel and stores it safely.
 *
 * ROS compatibility:
 *   - ROS1: geometry_msgs::Twist::ConstPtr
 *   - ROS2: geometry_msgs::msg::Twist::SharedPtr
 *
 * Key safety points:
 *   - ROS2 uses a mutex-protected cmd_vel_ update.
 *   - ROS2 updates last_cmd_vel_time_ for freshness checking.
 *   - No duplicate write outside the mutex.
 */
void RL_Real_G1_EDU23::CmdvelCallback(
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

    // Thread-safe store: single source of truth.
    {
        std::lock_guard<std::mutex> lk(this->cmd_vel_mtx_);
        this->cmd_vel_ = *msg;
        this->last_cmd_vel_time_ = ros2_node->now();
    }

    // Throttled log: confirms reception.
    RCLCPP_INFO_THROTTLE(
        ros2_node->get_logger(),
        *ros2_node->get_clock(),
        200,
        "[CMD_VEL RX] vx=%.3f vy=%.3f wz=%.3f",
        msg->linear.x,
        msg->linear.y,
        msg->angular.z
    );

#else  // ROS1

    // ROS1 path. If your ROS1 spinner becomes multi-threaded later,
    // protect this with the same mutex style.
    this->cmd_vel_ = *msg;

    static int cnt = 0;
    if ((cnt++ % 10) == 0)
    {
        std::cout << "[CMD_VEL RX] vx=" << this->cmd_vel_.linear.x
                  << " vy=" << this->cmd_vel_.linear.y
                  << " wz=" << this->cmd_vel_.angular.z
                  << std::endl;
    }

#endif
}

#endif

/**
 * @brief Callback triggered when laptop heartbeat is received.
 *
 * This function:
 *   - updates last heartbeat timestamp
 *   - enables heartbeat monitoring
 */

#if defined(USE_ROS2) && defined(USE_ROS)
void RL_Real_G1_EDU23::LaptopHeartbeatCallback(
    const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg;

    // Update heartbeat timestamp
    last_laptop_heartbeat_tp_ = std::chrono::steady_clock::now();

    // Mark heartbeat system as initialized
    laptop_heartbeat_initialized_ = true;

    RCLCPP_DEBUG_THROTTLE(
        ros2_node->get_logger(),
        *ros2_node->get_clock(),
        1000,
        "[HEARTBEAT] Laptop alive"
    );
}
#endif

// ----------------------------
// MAIN
// ----------------------------
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
                std::cout << LOGGER::ERROR
                        << "Usage: " << argv[0]
                        << " networkInterface [--real|--dry-run] [--udp-port=15000]"
                        << std::endl;
                return 1;
            }
        #endif
#if defined(USE_ROS1) && defined(USE_ROS)

        signal(SIGINT, signalHandler);
        ros::init(argc, argv, "rl_sar_g1_edu23");

        RL_Real_G1_EDU23 app(argc, argv);
        ros::spin();

#elif defined(USE_ROS2) && defined(USE_ROS)

        std::cout << "[MAIN] rclcpp::init" << std::endl;
        rclcpp::init(argc, argv);

        std::cout << "[MAIN] creating RL_Real_G1_EDU23" << std::endl;
        auto app = std::make_shared<RL_Real_G1_EDU23>(argc, argv);

        std::cout << "[MAIN] spinning ros2_node" << std::endl;
        rclcpp::spin(app->ros2_node);

        std::cout << "[MAIN] shutdown" << std::endl;
        rclcpp::shutdown();

#else

        RL_Real_G1_EDU23 app(argc, argv);
        while (1) { sleep(10); }

#endif

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FATAL MAIN] Exception: "
                  << e.what()
                  << std::endl;
        return 245;
    }
    catch (...)
    {
        std::cerr << "[FATAL MAIN] Unknown exception"
                  << std::endl;
        return 246;
    }
}