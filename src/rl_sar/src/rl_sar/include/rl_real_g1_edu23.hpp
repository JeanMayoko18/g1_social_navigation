/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_G1_EDU23_HPP
#define RL_REAL_G1_EDU23_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_g1.hpp"
#include "g1_edu23_joints.hpp"


#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>


#ifdef ENABLE_UNITREE_LOCO_BACKEND
#include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#endif

#include <array>
#include <csignal>
#include <cmath>
#include <memory>

#include <string>
#include <vector>
#include <condition_variable>
#include <cstdint>

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>


#if defined(USE_ROS1) && defined(USE_ROS)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#elif defined(USE_ROS2) && defined(USE_ROS)
#include <std_msgs/msg/empty.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#endif


#ifdef PLOT
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

// Unitree HG topics (bare DDS)
static const std::string HG_CMD_TOPIC   = "rt/lowcmd";
static const std::string HG_IMU_TORSO   = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

// ----------------------------
// Gamepad raw mapping (unchanged)
// ----------------------------
typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

typedef struct
{
    uint8_t head[2];
    xKeySwitchUnion btn;
    float lx;
    float rx;
    float ry;
    float L2;
    float ly;

    uint8_t idle[16];
} xRockerBtnDataStruct;

typedef union
{
    xRockerBtnDataStruct RF_RX;
    uint8_t buff[40];
} REMOTE_DATA_RX;

class Button
{
public:
    void update(bool state)
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

class Gamepad
{
public:
    void update(xRockerBtnDataStruct &key_data)
    {
        lx = lx * (1 - smooth) + (std::fabs(key_data.lx) < dead_zone ? 0.0f : key_data.lx) * smooth;
        rx = rx * (1 - smooth) + (std::fabs(key_data.rx) < dead_zone ? 0.0f : key_data.rx) * smooth;
        ry = ry * (1 - smooth) + (std::fabs(key_data.ry) < dead_zone ? 0.0f : key_data.ry) * smooth;
        l2 = l2 * (1 - smooth) + (std::fabs(key_data.L2) < dead_zone ? 0.0f : key_data.L2) * smooth;
        ly = ly * (1 - smooth) + (std::fabs(key_data.ly) < dead_zone ? 0.0f : key_data.ly) * smooth;

        R1.update(key_data.btn.components.R1);
        L1.update(key_data.btn.components.L1);
        start.update(key_data.btn.components.start);
        select.update(key_data.btn.components.select);
        R2.update(key_data.btn.components.R2);
        L2.update(key_data.btn.components.L2);
        F1.update(key_data.btn.components.F1);
        F2.update(key_data.btn.components.F2);
        A.update(key_data.btn.components.A);
        B.update(key_data.btn.components.B);
        X.update(key_data.btn.components.X);
        Y.update(key_data.btn.components.Y);
        up.update(key_data.btn.components.up);
        right.update(key_data.btn.components.right);
        down.update(key_data.btn.components.down);
        left.update(key_data.btn.components.left);
    }

    float lx = 0.f;
    float rx = 0.f;
    float ry = 0.f;
    float l2 = 0.f;
    float ly = 0.f;

    float smooth = 0.03f;
    float dead_zone = 0.01f;

    Button R1, L1, start, select, R2, L2, F1, F2, A, B, X, Y, up, right, down, left;
};

enum class Mode
{
    PR = 0, // Series Control for Pitch/Roll joints
    AB = 1  // Parallel Control for A/B joints
};

enum G1JointIndex {
    LeftHipPitch = 0,
    LeftHipRoll = 1,
    LeftHipYaw = 2,
    LeftKnee = 3,
    LeftAnklePitch = 4,
    LeftAnkleB = 4,
    LeftAnkleRoll = 5,
    LeftAnkleA = 5,
    RightHipPitch = 6,
    RightHipRoll = 7,
    RightHipYaw = 8,
    RightKnee = 9,
    RightAnklePitch = 10,
    RightAnkleB = 10,
    RightAnkleRoll = 11,
    RightAnkleA = 11,
    WaistYaw = 12,
    WaistRoll = 13,        // INVALID on EDU23 (waist locked)
    WaistA = 13,
    WaistPitch = 14,       // INVALID on EDU23 (waist locked)
    WaistB = 14,
    LeftShoulderPitch = 15,
    LeftShoulderRoll = 16,
    LeftShoulderYaw = 17,
    LeftElbow = 18,
    LeftWristRoll = 19,
    LeftWristPitch = 20,   // INVALID on EDU23
    LeftWristYaw = 21,     // INVALID on EDU23
    RightShoulderPitch = 22,
    RightShoulderRoll = 23,
    RightShoulderYaw = 24,
    RightElbow = 25,
    RightWristRoll = 26,
    RightWristPitch = 27,  // INVALID on EDU23
    RightWristYaw = 28     // INVALID on EDU23
};


// ----------------------------
// EDU+ 23DoF joint activation mask
// We keep policy-space indices [0..28] (29-DoF) but only 23 exist.
// Missing: 13,14 (waist locked), 20,21 (L wrist pitch/yaw), 27,28 (R wrist pitch/yaw)
// ----------------------------
//static inline bool is_edu23_active_joint(int policy_joint_index)
//{
//    switch (policy_joint_index)
//    {
//    case 0: case 1: case 2: case 3: case 4: case 5:
//    case 6: case 7: case 8: case 9: case 10: case 11:
//    case 12:
//    case 15: case 16: case 17: case 18: case 19:
//    case 22: case 23: case 24: case 25: case 26:
//        return true;
//    default:
//        return false;
//    }
//}



class RL_Real_G1_EDU23 : public RL
{
public:
    RL_Real_G1_EDU23(int argc, char **argv);
    ~RL_Real_G1_EDU23();

    // ------------------------------------------------------------
    // EDU23 specific initialization (called via polymorphism)
    // ------------------------------------------------------------
    bool OnEnterPolicy(const std::string& robot_config_path) override;

    bool InitEdu23PolicyMapping(const std::string& robot_config_path);

    // Debug (optional)
    void DumpFullMappingOnce() const;


#if defined(USE_ROS2) && defined(USE_ROS)
    void PublishJointStates();
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

    

private:

    // ============================================================
    // Fixed sizes for EDU23 adapter
    // ============================================================
    static constexpr int kNdofPolicy = 29;   // RL policy space
    static constexpr int kNdofHw23   = 23;   // Physical EDU23 joints
    static constexpr int kMsTotal = 35;     // Unitree HG motor channels

    // ============================================================
    // Mapping tables (built once)
    // ============================================================

    std::array<int,  kNdofPolicy> policy_to_hw23_{};   // policy p -> hw23 slot (or -1) [size 29, -1 where absent]
    std::array<int,  kNdofPolicy> policy_to_ms_{};     // policy p -> Unitree motor index (or -1) [size 29, -1 where absent]
    std::array<bool, kMsTotal>     enabled_ms_mask_{};  // allowlist: ms index enabled/disabled
    bool policy_table_ready_{false};

    // From base_edu23.yaml
    std::vector<int>   hw23_to_ms_;              // size 23 (hw23 slot -> Unitree motor index)
    std::vector<float> base_default_dof_pos_;    // size 23 (optional)   
    
    // --- HW descriptors ---
    int hw_active_dofs_ = kNdofHw23;
    int ms_total_ = kMsTotal;


    #if defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    #endif
     std::vector<std::string> tf_joint_names_;

    // From policy YAML  
    // --- posture stable EDU23 -> project into policy space ---
    std::vector<float> hw_default_dof_pos_23_;      // size 23
    std::vector<float> policy_default_dof_pos_;     // size 29 (from policy yaml)

    // policy loaded flag (torch + config.yaml)
    std::atomic<bool> policy_loaded_{false};
    bool dimos_warning_printed_ = false;
    
    bool use_dimos_policy_ = false;

    // Optional freeze buffers
    std::array<float, kNdofPolicy> last_q_{};
    std::array<float, kNdofPolicy> last_dq_{};
    std::array<float, kNdofPolicy> last_tau_{};
    bool have_last_cmd_{false};

    // Build mapping
    bool BuildPolicyToMsTable();

    // ================================
    // Hardware execution mode
    // ================================
    enum class HwMode : uint8_t
    {
        REAL = 0,          // real robot (LowCmd publish)
        FAKE_LOWSTATE = 1, // fake LowState (pipeline test)
        CMD_VEL_ONLY = 2,  // cmd_vel listening only
        DRY_RUN = 3        // compute everything but send nothing
    };

    // ================================
    // Hardware mode (runtime)
    // ================================
    HwMode hw_mode_{HwMode::REAL};
    bool publish_lowcmd_{true};
    bool fake_rl_full_{false};   // false = mapping_only, true = rl_full
    bool cmd_vel_only_{false};

    // Used by ROS cmd_vel and Unitree-only UDP cmd_vel
    double cmd_vel_timeout_sec_{0.6};

    // Navigation mode control, available in both ROS2 and Unitree-only modes
    bool navigation_mode_param_{false};  // param initial
    bool navigation_mode_force_{false};  // force ON




    /* ===============================
     * Fake LowState storage (runtime) not used
     * =============================== */
    std::vector<float> fake_hw_q_;
    std::vector<float> fake_hw_dq_;
    std::vector<float> fake_hw_tau_;
    bool fake_inited_{false};

    // ================================
    // Fake hardware buffers (runtime) used
    // ================================
    std::vector<float> fake_hw_q;
    std::vector<float> fake_hw_dq;
    std::vector<float> fake_hw_tau;
    bool fake_inited = false;


    // RL interface overrides
    std::atomic_bool got_lowstate{false};
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void GetState_real(RobotState<float> *state);
    void GetState_test(RobotState<float> *state);
    void GetState_minimal(RobotState<float> *state) const;
    void GetState_real_for_getup(RobotState<float>* state);

    void OnEnterGetUp() override;
    void OnEnterLocomotion() override;
    void OnExitLocomotion() override;
    
    void SetCommand(const RobotCommand<float> *command) override;

    // loops
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
#ifdef PLOT
    std::shared_ptr<LoopFunc> loop_plot;
#endif

    // main steps
    void RunModel();
    void RunModels();
    void RobotControl();
    void RobotControl_old();

#ifdef PLOT
    void Plot();
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
#endif

    // Unitree interface
    void InitLowCmd();
    uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
    void LowStateHandler(const void *message);
    void ImuTorsoHandler(const void *message);

    //unitree::robot::b2::MotionSwitcherClient msc;
    std::unique_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;
    LowCmd_   unitree_low_command;
    LowState_ unitree_low_state;
    IMUState_ unitree_imu_torso;

    mutable std::mutex stand_hold_mtx_;
    mutable std::mutex lowstate_mtx_;


    #if defined(USE_ROS2) && defined(USE_ROS)
        rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr laptop_heartbeat_sub_;
    #endif

    #if defined(USE_ROS2) && defined(USE_ROS)
        void LaptopHeartbeatCallback(std_msgs::msg::Empty::SharedPtr msg);
    #endif


    Mode mode_pr;
    uint8_t mode_machine;

    Gamepad gamepad;
    REMOTE_DATA_RX remote_data_rx;

    ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<IMUState_> imutorso_subscriber;

    enum class RLSafetyState
    {
        RUNNING,
        LOWSTATE_LOST,
        RECOVERING
    };

    RLSafetyState safety_state_ = RLSafetyState::LOWSTATE_LOST;

    std::chrono::steady_clock::time_point last_lowstate_tp_;
    std::chrono::steady_clock::time_point recovery_start_tp_;
    std::chrono::steady_clock::time_point unsafe_since_tp_;
    std::chrono::steady_clock::time_point links_unsafe_since_tp_;

    double lowstate_timeout_sec_ = 0.20;     // 200 ms
    double recovery_duration_sec_ = 1.00;    // 1 seconde stable avant reprise

    
    bool links_unsafe_timer_started_ = false;
    bool unsafe_timer_started_ = false;
    double links_loss_confirm_sec_ = 0.35;
    double lowstate_loss_confirm_sec_ = 0.35;

    double laptop_loss_confirm_sec_   = 1.0;
    
    // Maximum allowed continuous locomotion duration.
    // 0.0 disables this watchdog.
    double max_motion_duration_sec_{60.0};

    bool lowstate_time_initialized_ = false;
    bool locomotion_started_once_ = false;
    bool allow_recovery_model_reset_ = false;

    std::array<float, kMsTotal> last_good_ms_q_{};
    std::array<float, kMsTotal> last_good_ms_dq_{};
    std::array<bool,  kMsTotal> last_good_ms_valid_{};

    bool stand_hold_ready_ = false;

    double stand_hold_max_sec_ = 2.0;
    float stand_hold_kp_ = 35.0f;
    float stand_hold_kd_ = 3.0f;


    // --- Laptop Monitoring ---
    std::chrono::steady_clock::time_point last_laptop_heartbeat_tp_;
    std::string laptop_heartbeat_topic_ = "/laptop_heartbeat";
    bool laptop_heartbeat_initialized_ = false;
    bool require_laptop_heartbeat_ = false;
    double laptop_heartbeat_timeout_sec_ = 1.0;

    // --- Safety Posture ---
    bool is_robot_calibrated_ = false; 
    const float kKneeValidThreshold = -0.5f; // Threshold to ensure robot isn't collapsed

    std::chrono::steady_clock::time_point lowstate_lost_start_tp_;
    bool lowstate_lost_timer_started_ = false;

    bool IsLowStateFresh() const;
    void UpdateSafetyState();
    void ClearCmdVelUnsafe();

    void DebugMotorIndexProbe(int ms_test);

    void ResetRLAfterRecovery();
    void SendDampingCommandIfPossible();
    void SendStandHoldCommandIfPossible();
    bool IsStandHoldReady() const;
    bool ConfirmLinksLost(
        bool links_ok,
        const std::chrono::steady_clock::time_point& now
    );



// ===============================
// cmd_vel storage + subscription
// ===============================
#if defined(USE_ROS1) && defined(USE_ROS)

    // ROS1: single variable (no mutex needed usually)
    geometry_msgs::Twist cmd_vel_;
    ros::Subscriber cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);

#elif defined(USE_ROS2) && defined(USE_ROS)

    // ROS2: single variable + mutex (callback thread vs control loop)
    geometry_msgs::msg::Twist cmd_vel_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);

    // cmd_vel freshness tracking (ROS2 only)
    // ✅ init “never received” by using nanoseconds==0 check
    rclcpp::Time last_cmd_vel_time_{0, 0, RCL_SYSTEM_TIME};
    std::string cmd_vel_topic_{"/cmd_vel"};

    // thread safety for cmd_vel_ + last_cmd_vel_time_
    mutable std::mutex cmd_vel_mtx_;

    // Helper: returns true only if cmd_vel is fresh
    bool GetFreshCmdVel(float &vx, float &vy, float &wz) const;

#endif



    /* ================================
     * Base hardware YAML snapshot
     * Used to restore real-robot params
     * after policy YAML is loaded
     * ================================ */
    struct BaseYamlSnapshot
    {
        float dt = 0.005f;
        int decimation = 4;
        int num_of_dofs = 29;
        std::vector<int> joint_mapping;
        std::vector<float> default_dof_pos;
        std::vector<float> torque_limits;
        std::vector<std::string> joint_names;
        std::vector<std::string> joint_controller_names;
    };

    BaseYamlSnapshot base_snap_;

    #ifndef USE_ROS
    struct IpcCmdVel
    {
        float vx{0.0f};
        float vy{0.0f};
        float wz{0.0f};
        bool initialized{false};
        std::chrono::steady_clock::time_point stamp{};
    };

    std::thread ipc_cmd_thread_;
    std::atomic<bool> ipc_running_{false};
    mutable std::mutex ipc_cmd_mtx_;
    IpcCmdVel ipc_cmd_;

    int ipc_cmd_port_{15000};

    void ParseUnitreeOnlyArgs(int argc, char** argv);
    void StartIpcCmdVelReceiver(int port);
    void StopIpcCmdVelReceiver();
    bool GetFreshIpcCmdVel(float& vx, float& vy, float& wz) const;
    #endif

    // ============================================================================
    // UNITREE_LOCO BACKEND
    // ============================================================================
    // ----------------------------------------------------------------------------
    // DimOS-style high-level locomotion backend.
    //
    // Purpose:
    //   - Use Unitree internal locomotion controller.
    //   - Send vx/vy/wz through LocoClient::Move().
    //   - Stop motion with LocoClient::StopMove().
    //   - Avoid direct LowCmd motor control.
    //
    // This backend is independent from the RL low-level policy backend.
    // ============================================================================
    // rl_sar supports multiple locomotion backends.
    //
    // LOWCMD_POLICY:
    //   - Current rl_sar behavior.
    //   - Policy generates joint commands.
    //   - Commands are sent directly through LowCmd.
    //
    // UNITREE_LOCO:
    //   - DimOS-style behavior.
    //   - Uses Unitree internal locomotion controller.
    //   - Receives vx/vy/wz commands.
    //   - Calls LocoClient::Move() and StopMove().
    //   - No direct motor control from rl_sar.
    //
    // Reference:
    //   DimOS:
    //     G1HighLevelDdsSdk
    //         -> LocoClient
    //         -> Move()
    //         -> StopMove()
    //         -> Unitree locomotion controller
    // ============================================================================
    
    enum class ControlBackend
    {
        LOWCMD_POLICY,
        UNITREE_LOCO
    };

    // Selected locomotion backend.
    ControlBackend control_backend_ =
        ControlBackend::LOWCMD_POLICY;

    // Human-readable backend name.
    std::string control_backend_name_ =
        "lowcmd_policy";

    bool IsUnitreeLocoBackend() const override
    {
        return control_backend_ == ControlBackend::UNITREE_LOCO;
    }

    #ifdef ENABLE_UNITREE_LOCO_BACKEND

    struct LocoCommand
    {
        float vx{0.0f};
        float vy{0.0f};
        float wz{0.0f};

        bool stop_requested{true};
        bool initialized{false};

        std::uint64_t generation{0};
    };

    // Unitree high-level locomotion client.
    std::unique_ptr<unitree::robot::g1::LocoClient>
        loco_client_;

    // Worker thread used so Move() cannot block the main control loop.
    std::thread loco_command_thread_;

    std::atomic<bool>
        loco_command_thread_running_{false};

    mutable std::mutex
        loco_command_mtx_;

    std::condition_variable
        loco_command_cv_;

    LocoCommand
        latest_loco_command_;

    // SDK call monitoring.
    std::atomic<bool>
        loco_call_in_progress_{false};

    // Backend runtime state.
    bool unitree_loco_motion_enabled_{false};
    bool unitree_loco_standup_done_{false};
    bool loco_safety_locked_{false};
    std::atomic<bool> loco_rpc_fault_latched_{false};
    std::atomic<bool> loco_stop_rpc_confirmed_{false};
    std::atomic<bool> loco_physical_stop_confirmed_{false};

    std::mutex loco_recovery_mtx_;
    bool loco_stillness_timer_started_{false};
    std::chrono::steady_clock::time_point loco_stillness_since_tp_;

    double loco_stop_stable_duration_sec_{0.5};
    float loco_stop_max_joint_dq_{0.15f};
    float loco_stop_max_gyro_{0.10f};
    std::chrono::steady_clock::time_point
    loco_fault_stop_retry_after_tp_{};

    double loco_fault_stop_retry_delay_sec_{0.75};

    float locked_vx_{0.0f};
    float locked_vy_{0.0f};
    float locked_wz_{0.0f};

    // Unitree FSM identifiers.
    static constexpr int FSM_ZERO_TORQUE = 0;
    static constexpr int FSM_DAMP = 1;
    static constexpr int FSM_AI_MODE = 200;
    static constexpr int FSM_SQUAT_STANDUP_TOGGLE = 706;

    // Backend lifecycle.
    bool InitUnitreeLocoBackend();
    bool ShutdownUnitreeLocoBackend();

    void StartLocoCommandThread();
    void StopLocoCommandThread();

    // Non-blocking interface used by the control loop.
    void QueueLocoMove(
        float vx,
        float vy,
        float wz
    );

    void QueueLocoStop();

    // Worker-side synchronous SDK calls.
    bool UnitreeLocoMove(
        float vx,
        float vy,
        float wz
    );

    bool UnitreeLocoStop();

    // Periodic control path.
    void UnitreeLocoControlLoop();
    void RunUnitreeLocoBackend();

    // Unitree locomotion FSM operations.
    bool UnitreeLocoStandUp() override;
    bool UnitreeLocoStandUps() override;
    bool UnitreeLocoEnterWalkReady() override;
    bool UnitreeLocoNoEnterWalkReady() override;
    bool UnitreeLocoEnterWalkReady_old();
    bool IsLocoPhysicallyStopped() const;
    void UpdateLocoPhysicalStopState(const LowState_& low_state);

    int GetCurrentFsmId();

    bool TrySetFsmId(
        int id,
        const std::string& name,
        int sleep_ms
    );

    bool IsUnlockCommand(
        float vx,
        float vy,
        float wz
    ) const;

    #endif  // ENABLE_UNITREE_LOCO_BACKEND

     // ============================================================================

};

#endif // RL_REAL_G1_EDU23_HPP
