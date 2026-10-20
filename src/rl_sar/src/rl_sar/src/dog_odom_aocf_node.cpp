#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/quaternion.hpp"


// ============================================================================
// AOCF shared data structures
// ============================================================================

/**
 * @brief Planar motion increment expressed in the robot body frame.
 *
 * Frame:
 *   F_B = base_footprint
 *
 * These values represent motion during one physical dog_odom sampling interval.
 */
struct BodyIncrement
{
  double dx = 0.0;
  double dy = 0.0;
  double dyaw = 0.0;
};


/**
 * @brief Planar body-frame velocity.
 */
struct BodyVelocity
{
  double vx = 0.0;
  double vy = 0.0;
  double wz = 0.0;
};


/**
 * @brief Current validated navigation command.
 *
 * cmd_vel is never directly integrated as odometry.
 * It is only used as a weak physical consistency reference.
 */
struct CommandState
{
  double vx = 0.0;
  double wz = 0.0;

  bool active = false;
};


/**
 * @brief Motion expected by MPM during one odometry interval.
 */
struct MotionPrediction
{
  BodyIncrement increment;
  BodyVelocity velocity;

  bool valid = false;
};


/**
 * @brief Component-wise confidence estimated by AMCE.
 *
 * 1.0:
 *   fully trust measured dog_odom motion.
 *
 * 0.0:
 *   fully trust the MPM prediction.
 */
struct MotionConfidence
{
  double x = 1.0;
  double y = 1.0;
  double yaw = 1.0;
};


/**
 * @brief Output of TMIR.
 *
 * TMIR is the only module responsible for converting the incoming
 * absolute dog odometry into a local temporal body-frame increment.
 */
struct TmirSample
{
  BodyIncrement measured_increment;

  double raw_yaw = 0.0;
  double previous_raw_yaw = 0.0;

  double dt = 0.0;

  bool valid = false;
};


class DogOdomAocfNode : public rclcpp::Node
{
public:
  DogOdomAocfNode()
  : Node("dog_odom_aocf_node")
  {
    // ========================================================================
    // 1. ROS topics and frames
    //
    // /dog_odom is already converted to planar /dog_odom_2d by
    // dog_odom_adapter_node.
    //
    // Therefore this node starts directly from /dog_odom_2d.
    // ========================================================================

    input_odom_topic_ =
      declare_parameter<std::string>(
        "input_odom_topic",
        "/dog_odom_2d");

    output_odom_topic_ =
      declare_parameter<std::string>(
        "output_odom_topic",
        "/odom_aocf");

    cmd_vel_topic_ =
      declare_parameter<std::string>(
        "cmd_vel_topic",
        "/cmd_vel_rlsar");

    odom_frame_ =
      declare_parameter<std::string>(
        "odom_frame",
        "odom");

    base_frame_ =
      declare_parameter<std::string>(
        "base_frame",
        "base_footprint");


    // ========================================================================
    // 2. AOCF module switches
    //
    // These parameters allow independent ablation of each AOCF module.
    // ========================================================================

    enable_mpm_ =
      declare_parameter<bool>(
        "enable_mpm",
        true);

    enable_amce_ =
      declare_parameter<bool>(
        "enable_amce",
        true);

    enable_afi_ =
      declare_parameter<bool>(
        "enable_afi",
        true);

    enable_ahc_ =
      declare_parameter<bool>(
        "enable_ahc",
        true);

    enable_atc_ =
      declare_parameter<bool>(
        "enable_atc",
        true);


    // ========================================================================
    // 3. Non-core safety mechanisms
    //
    // These mechanisms are intentionally separated from the five core
    // calibration modules.
    // ========================================================================

    enable_increment_limiter_ =
      declare_parameter<bool>(
        "enable_increment_limiter",
        true);

    stationary_hold_enabled_ =
      declare_parameter<bool>(
        "stationary_hold_enabled",
        true);


    // ========================================================================
    // 4. Navigation output timing
    //
    // AOCF computation runs only when a new dog_odom_2d sample arrives.
    //
    // Publication is independent:
    //
    //   latest corrected state
    //            ↓
    //      timer at 45 Hz
    //            ↓
    //       /odom_aocf
    //
    // /odom_aocf is stamped with current ROS time.
    // ========================================================================

    output_frequency_hz_ =
      declare_parameter<double>(
        "output_frequency_hz",
        45.0);

    max_state_age_sec_ =
      declare_parameter<double>(
        "max_state_age_sec",
        0.5);


    // ========================================================================
    // 5. Startup protection
    // ========================================================================

    startup_freeze_sec_ =
      declare_parameter<double>(
        "startup_freeze_sec",
        2.0);

    startup_ramp_sec_ =
      declare_parameter<double>(
        "startup_ramp_sec",
        1.0);


    // ========================================================================
    // 6. Source timing / TMIR parameters
    // ========================================================================

    odom_period_alpha_ =
      declare_parameter<double>(
        "odom_period_alpha",
        0.02);

    odom_gap_factor_ =
      declare_parameter<double>(
        "odom_gap_factor",
        5.0);

    odom_period_init_samples_ =
      declare_parameter<int>(
        "odom_period_init_samples",
        20);


    // ========================================================================
    // 7. Increment safety limiter
    // ========================================================================

    max_step_m_ =
      declare_parameter<double>(
        "max_step_m",
        0.12);

    max_step_yaw_ =
      declare_parameter<double>(
        "max_step_yaw",
        0.12);


    // ========================================================================
    // 8. MPM parameters
    //
    // MPM predicts the motion expected from cmd_vel.
    // ========================================================================

    straight_cmd_vx_min_ =
      declare_parameter<double>(
        "straight_cmd_vx_min",
        0.08);

    straight_cmd_wz_max_ =
      declare_parameter<double>(
        "straight_cmd_wz_max",
        0.08);

    turn_cmd_wz_min_ =
      declare_parameter<double>(
        "turn_cmd_wz_min",
        0.08);

    vx_gain_min_ =
      declare_parameter<double>(
        "vx_gain_min",
        0.35);

    vx_gain_max_ =
      declare_parameter<double>(
        "vx_gain_max",
        1.20);

    wz_gain_min_ =
      declare_parameter<double>(
        "wz_gain_min",
        0.30);

    wz_gain_max_ =
      declare_parameter<double>(
        "wz_gain_max",
        1.30);


    // ========================================================================
    // 9. AMCE parameters
    //
    // AMCE estimates confidence only.
    // It never directly changes an odometry increment.
    // ========================================================================

    amce_alpha_ =
      declare_parameter<double>(
        "amce_alpha",
        0.02);

    mci_x_min_ =
      declare_parameter<double>(
        "mci_x_min",
        0.60);

    mci_y_min_ =
      declare_parameter<double>(
        "mci_y_min",
        0.20);

    mci_yaw_min_ =
      declare_parameter<double>(
        "mci_yaw_min",
        0.05);

    mci_x_sensitivity_ =
      declare_parameter<double>(
        "mci_x_sensitivity",
        8.0);

    mci_y_sensitivity_ =
      declare_parameter<double>(
        "mci_y_sensitivity",
        16.0);

    mci_yaw_sensitivity_ =
      declare_parameter<double>(
        "mci_yaw_sensitivity",
        10.0);


    // ========================================================================
    // 10. AHC parameters
    //
    // AHC acts only on the heading increment.
    // ========================================================================

    straight_yaw_gate_wz_ =
      declare_parameter<double>(
        "straight_yaw_gate_wz",
        0.008);


    straight_lateral_hold_enabled_ =
      declare_parameter<bool>(
        "straight_lateral_hold_enabled",
        true);

    straight_lateral_velocity_max_ =
      declare_parameter<double>(
        "straight_lateral_velocity_max",
        0.04);

    // ========================================================================
    // 11. Stationary hold
    // ========================================================================

    stationary_cmd_vx_max_ =
      declare_parameter<double>(
        "stationary_cmd_vx_max",
        0.02);

    stationary_cmd_wz_max_ =
      declare_parameter<double>(
        "stationary_cmd_wz_max",
        0.02);

    stationary_raw_step_max_ =
      declare_parameter<double>(
        "stationary_raw_step_max",
        0.003);

    stationary_raw_yaw_max_ =
      declare_parameter<double>(
        "stationary_raw_yaw_max",
        0.003);


    // ========================================================================
    // 12. Command validity
    // ========================================================================

    cmd_timeout_sec_ =
      declare_parameter<double>(
        "cmd_timeout_sec",
        0.35);


    // ========================================================================
    // 13. ATC parameters
    //
    // ATC calibrates the published twist only.
    // It never changes the integrated pose.
    // ========================================================================

    filter_output_vx_ =
      declare_parameter<bool>(
        "filter_output_vx",
        true);

    filter_output_vy_ =
      declare_parameter<bool>(
        "filter_output_vy",
        true);

    filter_output_wz_ =
      declare_parameter<bool>(
        "filter_output_wz",
        true);

    ema_vx_alpha_ =
      declare_parameter<double>(
        "ema_vx_alpha",
        0.20);

    ema_vy_alpha_ =
      declare_parameter<double>(
        "ema_vy_alpha",
        0.15);

    ema_wz_alpha_ =
      declare_parameter<double>(
        "ema_wz_alpha",
        0.25);

    max_output_vx_ =
      declare_parameter<double>(
        "max_output_vx",
        0.25);

    max_output_vy_ =
      declare_parameter<double>(
        "max_output_vy",
        0.08);

    max_output_wz_ =
      declare_parameter<double>(
        "max_output_wz",
        0.35);


    // ========================================================================
    // 14. Callback groups
    //
    // dog odometry processing, command reception and periodic publication
    // can run independently.
    // ========================================================================

    odom_callback_group_ =
      create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    cmd_callback_group_ =
      create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    timer_callback_group_ =
      create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);


    // ========================================================================
    // 15. ROS interfaces
    // ========================================================================

    odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(
        output_odom_topic_,
        rclcpp::QoS(10));


    rclcpp::SubscriptionOptions odom_options;

    odom_options.callback_group =
      odom_callback_group_;


    odom_sub_ =
      create_subscription<nav_msgs::msg::Odometry>(
        input_odom_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(
          &DogOdomAocfNode::odomCallback,
          this,
          std::placeholders::_1),
        odom_options);


    rclcpp::SubscriptionOptions cmd_options;

    cmd_options.callback_group =
      cmd_callback_group_;


    cmd_sub_ =
      create_subscription<geometry_msgs::msg::Twist>(
        cmd_vel_topic_,
        rclcpp::QoS(10),
        std::bind(
          &DogOdomAocfNode::cmdCallback,
          this,
          std::placeholders::_1),
        cmd_options);


    // ========================================================================
    // 16. Periodic output timer
    // ========================================================================

    const double safe_frequency =
      std::max(
        output_frequency_hz_,
        1.0);


    output_timer_ =
      create_wall_timer(
        std::chrono::duration<double>(
          1.0 / safe_frequency),
        std::bind(
          &DogOdomAocfNode::publishNavigationState,
          this),
        timer_callback_group_);


    start_time_ =
      get_clock()->now();


    // ========================================================================
    // 17. Startup summary
    // ========================================================================

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "============================================================\n"
      "             ADAPTIVE ODOMETRY CALIBRATION FRAMEWORK\n"
      "============================================================\n"
      " Input odometry       : %s\n"
      " Output odometry      : %s\n"
      " Command reference    : %s\n"
      " Output frequency     : %.2f Hz\n"
      "------------------------------------------------------------\n"
      " TMIR                 : ALWAYS ACTIVE\n"
      " MPM                  : %s\n"
      " AMCE                 : %s\n"
      " AFI                  : %s\n"
      " AHC                  : %s\n"
      " ATC                  : %s\n"
      "------------------------------------------------------------\n"
      " Increment limiter    : %s\n"
      " Stationary hold      : %s\n"
      "============================================================",
      input_odom_topic_.c_str(),
      output_odom_topic_.c_str(),
      cmd_vel_topic_.c_str(),
      output_frequency_hz_,
      enable_mpm_ ? "ENABLED" : "DISABLED",
      enable_amce_ ? "ENABLED" : "DISABLED",
      enable_afi_ ? "ENABLED" : "DISABLED",
      enable_ahc_ ? "ENABLED" : "DISABLED",
      enable_atc_ ? "ENABLED" : "DISABLED",
      enable_increment_limiter_ ? "ENABLED" : "DISABLED",
      stationary_hold_enabled_ ? "ENABLED" : "DISABLED");
  }


private:

  // ==========================================================================
  // COMMON UTILITIES
  // ==========================================================================

  static double wrapAngle(
    const double angle)
  {
    return std::atan2(
      std::sin(angle),
      std::cos(angle));
  }


  static double clamp(
    const double value,
    const double minimum,
    const double maximum)
  {
    return std::max(
      minimum,
      std::min(
        value,
        maximum));
  }


  static double quaternionToYaw(
    const geometry_msgs::msg::Quaternion & q)
  {
    const double siny =
      2.0 *
      (
        q.w * q.z
        + q.x * q.y
      );

    const double cosy =
      1.0
      - 2.0 *
      (
        q.y * q.y
        + q.z * q.z
      );

    return std::atan2(
      siny,
      cosy);
  }


  static geometry_msgs::msg::Quaternion yawToQuaternion(
    const double yaw)
  {
    geometry_msgs::msg::Quaternion q;

    q.x = 0.0;
    q.y = 0.0;

    q.z =
      std::sin(
        yaw * 0.5);

    q.w =
      std::cos(
        yaw * 0.5);

    return q;
  }


  static bool validQuaternion(
    const geometry_msgs::msg::Quaternion & q)
  {
    if (
      !std::isfinite(q.x)
      || !std::isfinite(q.y)
      || !std::isfinite(q.z)
      || !std::isfinite(q.w))
    {
      return false;
    }


    const double norm =
      std::sqrt(
        q.x * q.x
        + q.y * q.y
        + q.z * q.z
        + q.w * q.w);


    return norm > 1e-6;
  }


  // ==========================================================================
  // COMMAND REFERENCE
  //
  // This is not an odometry module.
  //
  // cmd_vel is only maintained as a temporally validated reference used by
  // MPM, AMCE, stationary hold and AHC.
  // ==========================================================================

  void cmdCallback(
    const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(
      command_mutex_);


    last_cmd_vx_ =
      msg->linear.x;

    last_cmd_wz_ =
      msg->angular.z;

    last_cmd_time_ =
      get_clock()->now();

    have_command_ =
      true;
  }


  CommandState getActiveCommand()
  {
    CommandState command;


    std::lock_guard<std::mutex> lock(
      command_mutex_);


    if (!have_command_)
    {
      return command;
    }


    const double age =
      (
        get_clock()->now()
        - last_cmd_time_
      ).seconds();


    if (
      age
      > cmd_timeout_sec_)
    {
      return command;
    }


    command.vx =
      last_cmd_vx_;

    command.wz =
      last_cmd_wz_;

    command.active =
      true;


    return command;
  }


  // ==========================================================================
  // TMIR — TEMPORAL MOTION INCREMENT RECONSTRUCTION
  //
  // ROLE
  // ----
  // TMIR is the interface between absolute dog odometry and the remaining
  // AOCF modules.
  //
  // Responsibilities:
  //
  //   1. establish a local odometry origin;
  //   2. recover physical dt from consecutive source timestamps;
  //   3. detect source discontinuities;
  //   4. compute temporal pose increments;
  //   5. express translational increments in base_footprint.
  //
  // TMIR never uses cmd_vel.
  // TMIR never predicts motion.
  // TMIR never fuses or filters odometry.
  // ==========================================================================

  void normalizeRawPose(
    const nav_msgs::msg::Odometry & msg,
    double & local_x,
    double & local_y,
    double & local_yaw)
  {
    const double raw_x =
      msg.pose.pose.position.x;

    const double raw_y =
      msg.pose.pose.position.y;

    const double raw_yaw =
      quaternionToYaw(
        msg.pose.pose.orientation);


    if (!have_origin_)
    {
      origin_x_ =
        raw_x;

      origin_y_ =
        raw_y;

      origin_yaw_ =
        raw_yaw;

      have_origin_ =
        true;


      local_x =
        0.0;

      local_y =
        0.0;

      local_yaw =
        0.0;


      RCLCPP_INFO(
        get_logger(),
        "[TMIR] Local origin initialized: "
        "x=%.3f y=%.3f yaw=%.3f",
        raw_x,
        raw_y,
        raw_yaw);


      return;
    }


    const double dx =
      raw_x
      - origin_x_;

    const double dy =
      raw_y
      - origin_y_;


    const double c =
      std::cos(
        -origin_yaw_);

    const double s =
      std::sin(
        -origin_yaw_);


    local_x =
      c * dx
      - s * dy;

    local_y =
      s * dx
      + c * dy;

    local_yaw =
      wrapAngle(
        raw_yaw
        - origin_yaw_);
  }


  bool updateNominalOdomPeriod(
    const double period)
  {
    if (
      !std::isfinite(period)
      || period <= 0.0)
    {
      return false;
    }


    if (!have_nominal_odom_period_)
    {
      odom_period_samples_.push_back(
        period);


      if (
        static_cast<int>(
          odom_period_samples_.size())
        < odom_period_init_samples_)
      {
        return false;
      }


      std::sort(
        odom_period_samples_.begin(),
        odom_period_samples_.end());


      const std::size_t n =
        odom_period_samples_.size();


      if (
        n % 2
        == 1)
      {
        nominal_odom_period_ =
          odom_period_samples_[
            n / 2];
      }
      else
      {
        nominal_odom_period_ =
          0.5
          *
          (
            odom_period_samples_[n / 2 - 1]
            +
            odom_period_samples_[n / 2]
          );
      }


      have_nominal_odom_period_ =
        true;

      odom_period_samples_.clear();


      RCLCPP_INFO(
        get_logger(),
        "[TMIR] Nominal dog odometry period: "
        "%.6f s (%.2f Hz)",
        nominal_odom_period_,
        1.0 / nominal_odom_period_);


      return false;
    }


    const bool source_gap =
      period
      >
      nominal_odom_period_
      * odom_gap_factor_;


    if (!source_gap)
    {
      const double alpha =
        clamp(
          odom_period_alpha_,
          0.0,
          1.0);


      nominal_odom_period_ =
        (
          1.0
          - alpha
        )
        * nominal_odom_period_
        +
        alpha
        * period;
    }


    return source_gap;
  }


  TmirSample runTmir(
    const nav_msgs::msg::Odometry & msg)
  {
    TmirSample sample;


    const rclcpp::Time source_stamp(
      msg.header.stamp);


    double local_x;
    double local_y;
    double local_yaw;


    normalizeRawPose(
      msg,
      local_x,
      local_y,
      local_yaw);


    sample.raw_yaw =
      local_yaw;


    if (!have_previous_source_stamp_)
    {
      previous_source_stamp_ =
        source_stamp;

      have_previous_source_stamp_ =
        true;


      previous_raw_x_ =
        local_x;

      previous_raw_y_ =
        local_y;

      previous_raw_yaw_ =
        local_yaw;

      have_previous_raw_pose_ =
        true;


      return sample;
    }


    const double dt =
      (
        source_stamp
        - previous_source_stamp_
      ).seconds();


    previous_source_stamp_ =
      source_stamp;


    const bool source_gap =
      updateNominalOdomPeriod(
        dt);


    if (
      !std::isfinite(dt)
      || dt <= 1e-4
      || source_gap)
    {
      previous_raw_x_ =
        local_x;

      previous_raw_y_ =
        local_y;

      previous_raw_yaw_ =
        local_yaw;


      resetTemporalModules();


      if (source_gap)
      {
        RCLCPP_WARN(
          get_logger(),
          "[TMIR] Source gap detected: dt=%.4f s",
          dt);
      }


      return sample;
    }


    const double heading_previous =
      previous_raw_yaw_;


    const double dx_odom =
      local_x
      - previous_raw_x_;

    const double dy_odom =
      local_y
      - previous_raw_y_;

    const double dyaw =
      wrapAngle(
        local_yaw
        - previous_raw_yaw_);


    previous_raw_x_ =
      local_x;

    previous_raw_y_ =
      local_y;

    previous_raw_yaw_ =
      local_yaw;


    const double c =
      std::cos(
        heading_previous);

    const double s =
      std::sin(
        heading_previous);


    sample.measured_increment.dx =
      c * dx_odom
      + s * dy_odom;


    sample.measured_increment.dy =
      -s * dx_odom
      + c * dy_odom;


    sample.measured_increment.dyaw =
      dyaw;


    sample.previous_raw_yaw =
      heading_previous;


    sample.dt =
      dt;


    sample.valid =
      true;


    return sample;
  }


  // ==========================================================================
  // MPM — MOTION PREDICTION MODEL
  //
  // ROLE
  // ----
  // Predict motion expected from the active command.
  //
  // MPM:
  //   - reads command and dt;
  //   - uses learned G1 response gains;
  //   - returns expected velocity and increment.
  //
  // MPM never changes measured dog odometry.
  // ==========================================================================

  MotionPrediction runMpm(
    const double dt,
    const CommandState & command)
  {
    MotionPrediction prediction;


    if (
      !command.active
      || dt <= 1e-4)
    {
      return prediction;
    }


    prediction.velocity.vx =
      command.vx
      * vx_gain_est_;


    prediction.velocity.vy =
      0.0;


    const bool straight_motion =
      std::abs(command.vx)
        >= straight_cmd_vx_min_
      &&
      std::abs(command.wz)
        <= straight_cmd_wz_max_;


    prediction.velocity.wz =
      straight_motion
      ? 0.0
      : command.wz
        * wz_gain_est_;


    prediction.increment.dx =
      prediction.velocity.vx
      * dt;


    prediction.increment.dy =
      prediction.velocity.vy
      * dt;


    prediction.increment.dyaw =
      prediction.velocity.wz
      * dt;


    prediction.valid =
      true;


    return prediction;
  }


  // ==========================================================================
  // AMCE — ADAPTIVE MOTION CONSISTENCY ESTIMATOR
  //
  // ROLE
  // ----
  // AMCE compares measured and predicted motion.
  //
  // Its only output is component-wise confidence.
  //
  // AMCE does NOT modify dx, dy or dyaw.
  // Fusion belongs exclusively to AFI.
  // ==========================================================================

  void updateResponseModel(
    const BodyVelocity & measured,
    const CommandState & command)
  {
    if (!command.active)
    {
      return;
    }


    const double alpha =
      clamp(
        amce_alpha_,
        0.0,
        1.0);


    if (
      std::abs(command.vx)
      >= straight_cmd_vx_min_)
    {
      double ratio =
        std::abs(measured.vx)
        /
        std::max(
          std::abs(command.vx),
          1e-6);


      ratio =
        clamp(
          ratio,
          vx_gain_min_,
          vx_gain_max_);


      vx_gain_est_ =
        (
          1.0
          - alpha
        )
        * vx_gain_est_
        +
        alpha
        * ratio;
    }


    if (
      std::abs(command.wz)
      >= turn_cmd_wz_min_)
    {
      double ratio =
        std::abs(measured.wz)
        /
        std::max(
          std::abs(command.wz),
          1e-6);


      ratio =
        clamp(
          ratio,
          wz_gain_min_,
          wz_gain_max_);


      wz_gain_est_ =
        (
          1.0
          - alpha
        )
        * wz_gain_est_
        +
        alpha
        * ratio;
    }
  }


  MotionConfidence runAmce(
    const BodyIncrement & measured_increment,
    const MotionPrediction & prediction,
    const double dt,
    const CommandState & command)
  {
    MotionConfidence confidence;


    if (
      !prediction.valid
      || dt <= 1e-4)
    {
      return confidence;
    }


    BodyVelocity measured_velocity;


    measured_velocity.vx =
      measured_increment.dx
      / dt;

    measured_velocity.vy =
      measured_increment.dy
      / dt;

    measured_velocity.wz =
      measured_increment.dyaw
      / dt;


    updateResponseModel(
      measured_velocity,
      command);


    const double alpha =
      clamp(
        amce_alpha_,
        0.0,
        1.0);


    error_vx_ =
      (
        1.0
        - alpha
      )
      * error_vx_
      +
      alpha
      * std::abs(
        measured_velocity.vx
        - prediction.velocity.vx);


    error_vy_ =
      (
        1.0
        - alpha
      )
      * error_vy_
      +
      alpha
      * std::abs(
        measured_velocity.vy
        - prediction.velocity.vy);


    error_wz_ =
      (
        1.0
        - alpha
      )
      * error_wz_
      +
      alpha
      * std::abs(
        measured_velocity.wz
        - prediction.velocity.wz);


    confidence.x =
      clamp(
        std::exp(
          -mci_x_sensitivity_
          * error_vx_),
        mci_x_min_,
        1.0);


    confidence.y =
      clamp(
        std::exp(
          -mci_y_sensitivity_
          * error_vy_),
        mci_y_min_,
        1.0);


    confidence.yaw =
      clamp(
        std::exp(
          -mci_yaw_sensitivity_
          * error_wz_),
        mci_yaw_min_,
        1.0);


    latest_mci_ =
      confidence;


    return confidence;
  }


  // ==========================================================================
  // AFI — ADAPTIVE FUSION OF INCREMENTS
  //
  // ROLE
  // ----
  // AFI is the only module responsible for fusing:
  //
  //   measured TMIR increment
  //          +
  //   predicted MPM increment
  //
  // using AMCE confidence.
  // ==========================================================================

  BodyIncrement runAfi(
    const BodyIncrement & measured,
    const MotionPrediction & prediction,
    const MotionConfidence & confidence)
  {
    if (!prediction.valid)
    {
      return measured;
    }


    BodyIncrement fused;


    fused.dx =
      confidence.x
      * measured.dx
      +
      (
        1.0
        - confidence.x
      )
      * prediction.increment.dx;


    fused.dy =
      confidence.y
      * measured.dy
      +
      (
        1.0
        - confidence.y
      )
      * prediction.increment.dy;


    fused.dyaw =
      confidence.yaw
      * measured.dyaw
      +
      (
        1.0
        - confidence.yaw
      )
      * prediction.increment.dyaw;


    return fused;
  }


  // ==========================================================================
  // AHC — ADAPTIVE HEADING CALIBRATION
  //
  // ROLE
  // ----
  // AHC modifies only the angular increment dyaw.
  //
  // During straight commanded motion:
  //   corrected heading is locked and raw yaw drift is absorbed into
  //   the alignment offset.
  //
  // During actual turning:
  //   raw yaw is realigned with corrected yaw before computing the
  //   new heading increment.
  //
  // AHC never modifies translation.
  // ==========================================================================

  double runAhc(
    const double input_dyaw,
    const double raw_yaw,
    const double corrected_yaw,
    const double dt,
    const CommandState & command)
  {
    if (
      dt <= 1e-4)
    {
      return input_dyaw;
    }


    if (!ahc_alignment_initialized_)
    {
      ahc_raw_to_corrected_offset_ =
        wrapAngle(
          corrected_yaw
          - raw_yaw);


      ahc_alignment_initialized_ =
        true;
    }


    const bool forward_motion =
      command.active
      &&
      std::abs(command.vx)
        >= straight_cmd_vx_min_;


    const bool negligible_turn =
      std::abs(command.wz)
        <= straight_cmd_wz_max_;


    if (
      forward_motion
      && negligible_turn)
    {
      ahc_raw_to_corrected_offset_ =
        wrapAngle(
          corrected_yaw
          - raw_yaw);


      return 0.0;
    }


    const double aligned_raw_yaw =
      wrapAngle(
        raw_yaw
        +
        ahc_raw_to_corrected_offset_);


    const double aligned_dyaw =
      wrapAngle(
        aligned_raw_yaw
        - corrected_yaw);


    const double measured_wz =
      aligned_dyaw
      / dt;


    const double parasitic_wz =
      std::abs(
        straight_yaw_gate_wz_);


    if (
      std::abs(measured_wz)
      <= parasitic_wz)
    {
      return 0.0;
    }


    const double corrected_wz =
      measured_wz
      -
      std::copysign(
        parasitic_wz,
        measured_wz);


    return corrected_wz
      * dt;
  }


  // ==========================================================================
  // ATC — ADAPTIVE TWIST CALIBRATION
  //
  // ROLE
  // ----
  // ATC converts the final corrected body-frame increment into twist.
  //
  // It may smooth and bound vx, vy and wz independently.
  //
  // ATC never changes the corrected pose.
  // ==========================================================================

  BodyVelocity runAtc(
    const BodyIncrement & increment,
    const double dt,
    const bool force_zero)
  {
    BodyVelocity velocity;


    if (
      force_zero
      || dt <= 1e-4)
    {
      return velocity;
    }


    const double raw_vx =
      increment.dx
      / dt;

    const double raw_vy =
      increment.dy
      / dt;

    const double raw_wz =
      increment.dyaw
      / dt;


    if (filter_output_vx_)
    {
      filtered_vx_ =
        ema_vx_alpha_
        * raw_vx
        +
        (
          1.0
          - ema_vx_alpha_
        )
        * filtered_vx_;


      velocity.vx =
        clamp(
          filtered_vx_,
          -max_output_vx_,
          max_output_vx_);
    }
    else
    {
      velocity.vx =
        clamp(
          raw_vx,
          -max_output_vx_,
          max_output_vx_);
    }


    if (filter_output_vy_)
    {
      filtered_vy_ =
        ema_vy_alpha_
        * raw_vy
        +
        (
          1.0
          - ema_vy_alpha_
        )
        * filtered_vy_;


      velocity.vy =
        clamp(
          filtered_vy_,
          -max_output_vy_,
          max_output_vy_);
    }
    else
    {
      velocity.vy =
        clamp(
          raw_vy,
          -max_output_vy_,
          max_output_vy_);
    }


    if (filter_output_wz_)
    {
      filtered_wz_ =
        ema_wz_alpha_
        * raw_wz
        +
        (
          1.0
          - ema_wz_alpha_
        )
        * filtered_wz_;


      velocity.wz =
        clamp(
          filtered_wz_,
          -max_output_wz_,
          max_output_wz_);
    }
    else
    {
      velocity.wz =
        clamp(
          raw_wz,
          -max_output_wz_,
          max_output_wz_);
    }


    return velocity;
  }


  // ==========================================================================
  // SAFETY — INCREMENT LIMITER
  //
  // This is not considered a core AOCF estimation module.
  // It only protects the pipeline from isolated physically impossible jumps.
  // ==========================================================================

  BodyIncrement applyIncrementLimiter(
    const BodyIncrement & input)
  {
    BodyIncrement output =
      input;


    const double step =
      std::hypot(
        output.dx,
        output.dy);


    if (
      step
      > max_step_m_)
    {
      const double scale =
        max_step_m_
        /
        std::max(
          step,
          1e-9);


      output.dx *=
        scale;

      output.dy *=
        scale;
    }


    output.dyaw =
      clamp(
        output.dyaw,
        -max_step_yaw_,
        max_step_yaw_);


    return output;
  }


  // ==========================================================================
  // SAFETY — STATIONARY HOLD
  //
  // Suppresses very small residual odometry motion when the robot has no
  // meaningful active command.
  // ==========================================================================

  bool applyStationaryHold(
    BodyIncrement & increment,
    const CommandState & command)
  {
    const bool command_is_zero =
      !command.active
      ||
      (
        std::abs(command.vx)
          <= stationary_cmd_vx_max_
        &&
        std::abs(command.wz)
          <= stationary_cmd_wz_max_
      );


    const bool measured_motion_is_small =
      std::hypot(
        increment.dx,
        increment.dy)
        <= stationary_raw_step_max_
      &&
      std::abs(
        increment.dyaw)
        <= stationary_raw_yaw_max_;


    if (
      command_is_zero
      && measured_motion_is_small)
    {
      increment =
        BodyIncrement{};


      return true;
    }


    return false;
  }


  // ==========================================================================
  // STARTUP PROTECTION
  // ==========================================================================

  double startupAlpha()
  {
    const double elapsed =
      (
        get_clock()->now()
        - start_time_
      ).seconds();


    if (
      elapsed
      < startup_freeze_sec_)
    {
      return 0.0;
    }


    const double ramp_duration =
      std::max(
        startup_ramp_sec_,
        1e-6);


    if (
      elapsed
      <
      startup_freeze_sec_
      + ramp_duration)
    {
      return
        (
          elapsed
          - startup_freeze_sec_
        )
        /
        ramp_duration;
    }


    return 1.0;
  }


  // ==========================================================================
  // CORRECTED POSE INTEGRATION
  //
  // The final AOCF increment is expressed in base_footprint.
  //
  // It must therefore be rotated into the corrected odom frame before being
  // integrated into x and y.
  // ==========================================================================

  void integrateCorrectedIncrement(
    const BodyIncrement & increment)
  {
    const double c =
      std::cos(
        corrected_yaw_);

    const double s =
      std::sin(
        corrected_yaw_);


    const double dx_odom =
      c * increment.dx
      - s * increment.dy;


    const double dy_odom =
      s * increment.dx
      + c * increment.dy;


    corrected_x_ +=
      dx_odom;


    corrected_y_ +=
      dy_odom;


    corrected_yaw_ =
      wrapAngle(
        corrected_yaw_
        + increment.dyaw);
  }


  // ==========================================================================
  // MAIN AOCF PIPELINE
  //
  // The callback explicitly exposes every module boundary.
  // ==========================================================================

void odomCallback(
  const nav_msgs::msg::Odometry::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }


  // ======================================================================
  // STEP 0 — INPUT VALIDATION
  // ======================================================================

  if (
    !validQuaternion(
      msg->pose.pose.orientation))
  {
    RCLCPP_WARN(
      get_logger(),
      "Dropping /dog_odom_2d sample: invalid quaternion");

    return;
  }


  // ======================================================================
  // STEP 1 — TMIR
  //
  // Reconstruct one measured body-frame temporal increment.
  //
  // Important:
  // msg->header.stamp is already expressed in the common Nav time domain,
  // because dog_odom_adapter_node converted the original Dog timestamp
  // before publishing /dog_odom_2d.
  // ======================================================================

  const TmirSample tmir =
    runTmir(
      *msg);


  if (!tmir.valid)
  {
    /*
     * No valid motion increment is available yet, typically during
     * initialization.
     *
     * We still preserve the current measurement timestamp so that
     * publication never falls back to stamp=0 or to now().
     */
    storeCurrentState(
      BodyVelocity{},
      msg->header.stamp);

    return;
  }


  // ======================================================================
  // STEP 2 — COMMAND SNAPSHOT
  //
  // Obtain one coherent command reference for the complete processing
  // of this odometry sample.
  // ======================================================================

  const CommandState command =
    getActiveCommand();


  // ======================================================================
  // STEP 3 — PRIMARY MEASUREMENT
  //
  // dog_odom remains the primary odometry source.
  // ======================================================================

  BodyIncrement corrected_increment =
    tmir.measured_increment;


  // ======================================================================
  // STEP 4 — MPM
  //
  // If disabled:
  // prediction.valid remains false.
  // ======================================================================

  MotionPrediction prediction;


  if (enable_mpm_)
  {
    prediction =
      runMpm(
        tmir.dt,
        command);
  }


  // ======================================================================
  // STEP 5 — AMCE
  //
  // If disabled, confidence remains neutral.
  // ======================================================================

  MotionConfidence confidence;


  if (
    enable_amce_
    && prediction.valid)
  {
    confidence =
      runAmce(
        corrected_increment,
        prediction,
        tmir.dt,
        command);
  }
  else
  {
    confidence =
      MotionConfidence{};
  }


  // ======================================================================
  // STEP 6 — AFI
  //
  // If disabled, the measured increment passes unchanged.
  // ======================================================================

  if (
    enable_afi_
    && prediction.valid)
  {
    corrected_increment =
      runAfi(
        corrected_increment,
        prediction,
        confidence);
  }

  // ======================================================================
  // STEP 6.5 — STRAIGHT-MOTION LATERAL HOLD
  //
  // Suppress small parasitic lateral motion while the robot is commanded
  // to walk straight. Larger lateral motion is preserved because it may
  // represent a real displacement.
  // ======================================================================

  const bool straight_command =
    command.active
    &&
    std::abs(command.vx) >= straight_cmd_vx_min_
    &&
    std::abs(command.wz) <= straight_cmd_wz_max_;

  if (
    straight_lateral_hold_enabled_
    &&
    straight_command
    &&
    tmir.dt > 1e-4)
  {
    const double measured_lateral_velocity =
      corrected_increment.dy / tmir.dt;

    if (
      std::abs(measured_lateral_velocity)
      <= straight_lateral_velocity_max_)
    {
      corrected_increment.dy = 0.0;
    }
  }


  // ======================================================================
  // STEP 7 — SAFETY INCREMENT LIMITER
  //
  // Independent from the core AOCF modules.
  // ======================================================================

  if (enable_increment_limiter_)
  {
    corrected_increment =
      applyIncrementLimiter(
        corrected_increment);
  }


  // ======================================================================
  // STEP 8 — STATIONARY HOLD
  // ======================================================================

  bool stationary_hold_active =
    false;


  if (stationary_hold_enabled_)
  {
    stationary_hold_active =
      applyStationaryHold(
        corrected_increment,
        command);
  }


  // ======================================================================
  // STEP 9 — AHC
  //
  // If disabled, incoming dyaw is integrated unchanged.
  // ======================================================================

  if (enable_ahc_)
  {
    corrected_increment.dyaw =
      runAhc(
        corrected_increment.dyaw,
        tmir.raw_yaw,
        corrected_yaw_,
        tmir.dt,
        command);
  }


  // ======================================================================
  // STEP 10 — STARTUP FREEZE / RAMP
  // ======================================================================

  const double startup_gain =
    clamp(
      startupAlpha(),
      0.0,
      1.0);


  corrected_increment.dx *=
    startup_gain;


  corrected_increment.dy *=
    startup_gain;


  corrected_increment.dyaw *=
    startup_gain;


  // ======================================================================
  // STEP 11 — POSE INTEGRATION
  //
  // Update the internally maintained corrected pose.
  // ======================================================================

  integrateCorrectedIncrement(
    corrected_increment);


  // ======================================================================
  // STEP 12 — ATC
  //
  // Compute the final corrected velocity associated with this exact
  // odometry measurement.
  // ======================================================================

  BodyVelocity velocity;


  if (enable_atc_)
  {
    velocity =
      runAtc(
        corrected_increment,
        tmir.dt,
        stationary_hold_active);
  }
  else if (
    !stationary_hold_active
    && tmir.dt > 1e-4)
  {
    velocity.vx =
      corrected_increment.dx
      / tmir.dt;


    velocity.vy =
      corrected_increment.dy
      / tmir.dt;


    velocity.wz =
      corrected_increment.dyaw
      / tmir.dt;
  }


  // ======================================================================
  // STEP 13 — STORE COMPLETE CORRECTED STATE
  //
  // This is the ONLY place where the processed state is transferred
  // to latest_state_.
  //
  // The important temporal rule is:
  //
  //   latest_state_.header.stamp = msg->header.stamp
  //
  // Therefore the publication timer republishes the true measurement
  // timestamp instead of inventing a new timestamp with now().
  // ======================================================================

  storeCurrentState(
    velocity,
    msg->header.stamp);
}


  // ==========================================================================
  // BUILD AND STORE COMPLETE CORRECTED STATE
  // ==========================================================================

void storeCurrentState(
  const BodyVelocity & velocity,
  const builtin_interfaces::msg::Time & measurement_stamp)
{
  nav_msgs::msg::Odometry state;


  // Preserve the true measurement timestamp coming from
  // /dog_odom_2d.
  state.header.stamp =
    measurement_stamp;


  state.header.frame_id =
    odom_frame_;


  state.child_frame_id =
    base_frame_;


  state.pose.pose.position.x =
    corrected_x_;


  state.pose.pose.position.y =
    corrected_y_;


  state.pose.pose.position.z =
    0.0;


  state.pose.pose.orientation =
    yawToQuaternion(
      corrected_yaw_);


  state.twist.twist.linear.x =
    velocity.vx;


  state.twist.twist.linear.y =
    velocity.vy;


  state.twist.twist.linear.z =
    0.0;


  state.twist.twist.angular.x =
    0.0;


  state.twist.twist.angular.y =
    0.0;


  state.twist.twist.angular.z =
    velocity.wz;


  setCovariances(
    state);


  // This local time is only used to measure how stale
  // the stored state is.
  const rclcpp::Time now =
    get_clock()->now();


  std::lock_guard<std::mutex> lock(
    state_mutex_);


  latest_state_ =
    state;


  latest_state_update_time_ =
    now;


  have_latest_state_ =
    true;
}


  // ==========================================================================
  // PERIODIC NAVIGATION OUTPUT
  //
  // Important time policy:
  //
  //   dog_odom_2d source stamp
  //        → used internally by TMIR for physical dt
  //
  //   /odom_aocf stamp
  //        → current ROS time
  //
  // This makes output timing independent from the source network cadence.
  // ==========================================================================

  void publishNavigationState()
  {
    nav_msgs::msg::Odometry out;


    rclcpp::Time state_time(
      0,
      0,
      get_clock()->get_clock_type());


    {
      std::lock_guard<std::mutex> lock(
        state_mutex_);


      if (!have_latest_state_)
      {
        return;
      }


      // Copy the latest processed AOCF state.
      //
      // Important:
      // latest_state_.header.stamp already represents the
      // measurement time inherited from /dog_odom_2d.
      //
      // We must preserve this timestamp when publishing.
      out =
        latest_state_;


      // This timestamp is only used to determine whether the
      // internally stored state has become stale.
      state_time =
        latest_state_update_time_;
    }


    const rclcpp::Time now =
      get_clock()->now();


    const double state_age =
      (
        now
        - state_time
      ).seconds();


    // Do NOT replace out.header.stamp with now.
    //
    // out.header.stamp remains the measurement timestamp coming
    // from the source odometry and propagated through AOCF.


    out.header.frame_id =
      odom_frame_;


    out.child_frame_id =
      base_frame_;


    // Never keep publishing old motion when source updates stop.
    if (
      state_age
      > max_state_age_sec_)
    {
      out.twist.twist.linear.x = 0.0;
      out.twist.twist.linear.y = 0.0;
      out.twist.twist.linear.z = 0.0;

      out.twist.twist.angular.x = 0.0;
      out.twist.twist.angular.y = 0.0;
      out.twist.twist.angular.z = 0.0;
    }


    odom_pub_->publish(
      out);
  }

  // ==========================================================================
  // RESET STATEFUL TEMPORAL MODULES AFTER SOURCE DISCONTINUITY
  // ==========================================================================

  void resetTemporalModules()
  {
    ahc_alignment_initialized_ =
      false;


    ahc_raw_to_corrected_offset_ =
      0.0;


    filtered_vx_ =
      0.0;


    filtered_vy_ =
      0.0;


    filtered_wz_ =
      0.0;
  }


  // ==========================================================================
  // NAVIGATION COVARIANCES
  // ==========================================================================

  static void setCovariances(
    nav_msgs::msg::Odometry & out)
  {
    out.pose.covariance.fill(
      0.0);


    out.pose.covariance[0] =
      0.05;

    out.pose.covariance[7] =
      0.05;

    out.pose.covariance[14] =
      999.0;

    out.pose.covariance[21] =
      999.0;

    out.pose.covariance[28] =
      999.0;

    out.pose.covariance[35] =
      0.10;


    out.twist.covariance.fill(
      0.0);


    out.twist.covariance[0] =
      0.10;

    out.twist.covariance[7] =
      0.10;

    out.twist.covariance[14] =
      999.0;

    out.twist.covariance[21] =
      999.0;

    out.twist.covariance[28] =
      999.0;

    out.twist.covariance[35] =
      0.20;
  }


  // ==========================================================================
  // PARAMETERS
  // ==========================================================================

  std::string input_odom_topic_;
  std::string output_odom_topic_;
  std::string cmd_vel_topic_;

  std::string odom_frame_;
  std::string base_frame_;


  bool enable_mpm_;
  bool enable_amce_;
  bool enable_afi_;
  bool enable_ahc_;
  bool enable_atc_;

  bool enable_increment_limiter_;
  bool stationary_hold_enabled_;


  double output_frequency_hz_;
  double max_state_age_sec_;


  double startup_freeze_sec_;
  double startup_ramp_sec_;


  double odom_period_alpha_;
  double odom_gap_factor_;

  int odom_period_init_samples_;


  double max_step_m_;
  double max_step_yaw_;


  double straight_cmd_vx_min_;
  double straight_cmd_wz_max_;
  double turn_cmd_wz_min_;


  double vx_gain_min_;
  double vx_gain_max_;

  double wz_gain_min_;
  double wz_gain_max_;


  double amce_alpha_;

  double mci_x_min_;
  double mci_y_min_;
  double mci_yaw_min_;

  double mci_x_sensitivity_;
  double mci_y_sensitivity_;
  double mci_yaw_sensitivity_;


  double straight_yaw_gate_wz_;


  double stationary_cmd_vx_max_;
  double stationary_cmd_wz_max_;

  double stationary_raw_step_max_;
  double stationary_raw_yaw_max_;


  double cmd_timeout_sec_;


  bool filter_output_vx_;
  bool filter_output_vy_;
  bool filter_output_wz_;

  double ema_vx_alpha_;
  double ema_vy_alpha_;
  double ema_wz_alpha_;

  double max_output_vx_;
  double max_output_vy_;
  double max_output_wz_;


  // ==========================================================================
  // ROS INTERFACES
  // ==========================================================================

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
    odom_pub_;


  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
    odom_sub_;


  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr
    cmd_sub_;


  rclcpp::TimerBase::SharedPtr
    output_timer_;


  rclcpp::CallbackGroup::SharedPtr
    odom_callback_group_;


  rclcpp::CallbackGroup::SharedPtr
    cmd_callback_group_;


  rclcpp::CallbackGroup::SharedPtr
    timer_callback_group_;


  // ==========================================================================
  // COMMAND STATE
  // ==========================================================================

  std::mutex command_mutex_;


  bool have_command_ =
    false;


  double last_cmd_vx_ =
    0.0;


  double last_cmd_wz_ =
    0.0;


  rclcpp::Time last_cmd_time_{
    0,
    0,
    RCL_ROS_TIME};


  // ==========================================================================
  // TMIR STATE
  // ==========================================================================

  bool have_origin_ =
    false;


  double origin_x_ =
    0.0;


  double origin_y_ =
    0.0;


  double origin_yaw_ =
    0.0;


  bool have_previous_raw_pose_ =
    false;


  double previous_raw_x_ =
    0.0;


  double previous_raw_y_ =
    0.0;


  double previous_raw_yaw_ =
    0.0;


  bool have_previous_source_stamp_ =
    false;


  rclcpp::Time previous_source_stamp_{
    0,
    0,
    RCL_ROS_TIME};


  bool have_nominal_odom_period_ =
    false;


  double nominal_odom_period_ =
    0.0;


  std::vector<double>
    odom_period_samples_;


  // ==========================================================================
  // MPM ADAPTIVE RESPONSE STATE
  // ==========================================================================

  double vx_gain_est_ =
    0.60;


  double wz_gain_est_ =
    0.60;


  // ==========================================================================
  // AMCE STATE
  // ==========================================================================

  double error_vx_ =
    0.0;


  double error_vy_ =
    0.0;


  double error_wz_ =
    0.0;


  MotionConfidence latest_mci_;


  // ==========================================================================
  // AHC STATE
  // ==========================================================================

  bool ahc_alignment_initialized_ =
    false;


  double ahc_raw_to_corrected_offset_ =
    0.0;

  bool straight_lateral_hold_enabled_;
  double straight_lateral_velocity_max_;


  // ==========================================================================
  // ATC STATE
  // ==========================================================================

  double filtered_vx_ =
    0.0;


  double filtered_vy_ =
    0.0;


  double filtered_wz_ =
    0.0;


  // ==========================================================================
  // CORRECTED AOCF TRAJECTORY
  // ==========================================================================

  double corrected_x_ =
    0.0;


  double corrected_y_ =
    0.0;


  double corrected_yaw_ =
    0.0;


  // ==========================================================================
  // STARTUP TIME
  // ==========================================================================

  rclcpp::Time start_time_{
    0,
    0,
    RCL_ROS_TIME};


  // ==========================================================================
  // LATEST CORRECTED OUTPUT STATE
  // ==========================================================================

  std::mutex state_mutex_;


  bool have_latest_state_ =
    false;


  nav_msgs::msg::Odometry
    latest_state_;


  rclcpp::Time latest_state_update_time_{
    0,
    0,
    RCL_ROS_TIME};
};


// ============================================================================
// MAIN
// ============================================================================

int main(
  int argc,
  char ** argv)
{
  try
  {
    rclcpp::init(
      argc,
      argv);


    auto node =
      std::make_shared<DogOdomAocfNode>();


    rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(),
      3);


    executor.add_node(
      node);


    executor.spin();


    executor.remove_node(
      node);


    rclcpp::shutdown();


    return 0;
  }

  catch (
    const std::exception & e)
  {
    std::cerr
      << "[dog_odom_aocf_node] Fatal exception: "
      << e.what()
      << std::endl;


    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }


    return 1;
  }

  catch (...)
  {
    std::cerr
      << "[dog_odom_aocf_node] Unknown fatal exception"
      << std::endl;


    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }


    return 1;
  }
}