#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "std_msgs/msg/float64.hpp"

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"

#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2_ros/transform_broadcaster.h"


class MolaOdomAdapter : public rclcpp::Node
{
public:
  MolaOdomAdapter()
  : Node("mola_odom_adapter_node")
  {
    // ============================================================
    // 1. MOLA input and physical 2D odometry output
    // ============================================================

    // Livox: retain source timestamps on both outputs and TF.
    // Unitree: retain the existing offset and navigation timer policy.
    apply_lidar_time_offset_ = declare_parameter<bool>(
      "apply_lidar_time_offset", true);

    RCLCPP_INFO(
      get_logger(), "MOLA timestamp policy: %s",
      apply_lidar_time_offset_
        ? "UNITREE: offset enabled; navigation uses ROS now"
        : "LIVOX: source timestamps preserved; no offset subscription");

    // Raw odometry produced by MOLA.
    input_topic_ =
      declare_parameter<std::string>(
        "input_topic",
        "/lidar_odometry/pose");


    // Internal planar MOLA odometry.
    //
    // This topic keeps the physical MOLA measurement timestamp,
    // converted using the LiDAR -> Nav offset only when enabled.
    //
    // It is intended for:
    //   - physical analysis;
    //   - debugging;
    //   - bag recording;
    //   - velocity reconstruction.
    output_topic_ =
      declare_parameter<std::string>(
        "output_topic",
        "/odom_mola_2d");


    // Navigation frames.
    output_frame_id_ =
      declare_parameter<std::string>(
        "output_frame_id",
        "odom");


    output_child_frame_id_ =
      declare_parameter<std::string>(
        "output_child_frame_id",
        "base_footprint");


    // ============================================================
    // 2. Nav2-facing odometry and TF
    // ============================================================

    // Final odometry topic consumed by Nav2.
    nav_output_topic_ =
      declare_parameter<std::string>(
        "nav_output_topic",
        "/odom");


    // Publish:
    //
    //     odom -> base_footprint
    //
    // for the navigation stack.
    publish_nav_tf_ =
      declare_parameter<bool>(
        "publish_nav_tf",
        true);


    // Periodic publication rate of /odom and navigation TF.
    //
    // The latest available MOLA pose is republished at this rate,
    // independently from the MOLA callback frequency.
    nav_publish_frequency_ =
      declare_parameter<double>(
        "nav_publish_frequency",
        30.0);


    // Maximum age of the last received MOLA state before the twist
    // is considered stale.
    //
    // The pose remains available, but stale velocity is never
    // republished.
    max_state_age_sec_ =
      declare_parameter<double>(
        "max_state_age_sec",
        0.5);


    // ============================================================
    // 3. Twist reconstruction parameters
    // ============================================================

    enable_twist_filter_ =
      declare_parameter<bool>(
        "enable_twist_filter",
        true);


    twist_filter_alpha_ =
      declare_parameter<double>(
        "twist_filter_alpha",
        0.3);


    stationary_linear_threshold_ =
      declare_parameter<double>(
        "stationary_linear_threshold",
        0.0);


    stationary_angular_threshold_ =
      declare_parameter<double>(
        "stationary_angular_threshold",
        0.01);


    // Reject invalid or excessive physical MOLA intervals.
    max_dt_ =
      declare_parameter<double>(
        "max_dt",
        0.5);


    // Velocity limits used only for reconstructed twist.
    max_linear_velocity_ =
      declare_parameter<double>(
        "max_linear_velocity",
        2.0);


    max_angular_velocity_ =
      declare_parameter<double>(
        "max_angular_velocity",
        2.0);


    // ============================================================
    // 4. Publishers
    // ============================================================

    // Physical/internal MOLA odometry.
    //
    // Timestamp:
    //
    //     corrected physical MOLA measurement time
    //
    // This is not the Nav2 compatibility stream.
    mola_odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(
        output_topic_,
        rclcpp::QoS(10));


    // Navigation odometry.
    //
    // Timestamp:
    //
    //     Livox: source time; Unitree: current ROS time
    //
    // This keeps /odom aligned with the dynamic navigation TF.
    nav_odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(
        nav_output_topic_,
        rclcpp::QoS(10));


    // ============================================================
    // 5. TF broadcaster
    //
    // Standard Foxy dynamic TF QoS.
    // ============================================================

    tf_broadcaster_ =
      std::make_shared<
        tf2_ros::TransformBroadcaster>(
          this);


    // ============================================================
    // 6. MOLA subscription
    // ============================================================

    sub_ =
      create_subscription<nav_msgs::msg::Odometry>(
        input_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(
          &MolaOdomAdapter::odomCallback,
          this,
          std::placeholders::_1));


    // ============================================================
    // 7. LiDAR -> Nav time offset
    //
    // The physical MOLA output remains associated with the LiDAR
    // measurement timeline.
    //
    // The time synchronization node publishes:
    //
    //     T_nav = T_lidar + lidar_to_nav_offset
    //
    // We keep this only for the physical/internal MOLA odometry path.
    // Unitree Nav2 outputs use current ROS time; Livox preserves source time.
    // ============================================================

    if (apply_lidar_time_offset_)
    {
      auto time_sync_qos = rclcpp::QoS(rclcpp::KeepLast(1));
      time_sync_qos.reliable();
      time_sync_qos.transient_local();

      lidar_to_nav_offset_sub_ =
        create_subscription<std_msgs::msg::Float64>(
          "/navigation_time/lidar_to_nav_offset",
          time_sync_qos,
          [this](const std_msgs::msg::Float64::SharedPtr msg)
          {
            if (!msg || !std::isfinite(msg->data)) {
              return;
            }
            lidar_to_nav_offset_sec_ = msg->data;
            has_lidar_to_nav_offset_ = true;
          });
    }


    // ============================================================
    // 8. Periodic Nav2 publication timer
    //
    // No custom std::thread.
    // No sleep_until().
    // No future TF prediction.
    //
    // The standard ROS wall timer periodically republishes the latest
    // valid MOLA state with the selected timestamp policy.
    // ============================================================

    const double safe_frequency =
      std::max(
        nav_publish_frequency_,
        1.0);


    nav_timer_ =
      create_wall_timer(
        std::chrono::duration<double>(
          1.0 / safe_frequency),
        std::bind(
          &MolaOdomAdapter::publishNavigationState,
          this));


    // ============================================================
    // 9. Startup information
    // ============================================================

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "=========================================================\n"
      "        MOLA ODOM ADAPTER - SIMPLE NAV MODE\n"
      "=========================================================\n"
      " Input MOLA topic       : %s\n"
      " Physical 2D output     : %s\n"
      " Nav2 odom output       : %s\n"
      " Output frame           : %s\n"
      " Output child frame     : %s\n"
      " Nav TF                 : %s\n"
      " Nav publish frequency  : %.2f Hz\n"
      " Max state age          : %.3f s\n"
      " Twist filter           : %s\n"
      " TF prediction          : disabled\n"
      " Custom publish thread  : disabled\n"
      
      " Physical convention    : native MOLA\n"
      " Navigation convention  : ROS/Nav2 normalized\n"
      " Nav frame conversion   : x,+ | y,- | yaw,-\n"
      "=========================================================",
      input_topic_.c_str(),
      output_topic_.c_str(),
      nav_output_topic_.c_str(),
      output_frame_id_.c_str(),
      output_child_frame_id_.c_str(),
      publish_nav_tf_
        ? "enabled"
        : "disabled",
      nav_publish_frequency_,
      max_state_age_sec_,
      enable_twist_filter_
        ? "enabled"
        : "disabled");
  }


private:

  // ==============================================================
  // Utility functions
  // ==============================================================

  static double wrapAngle(
    const double angle)
  {
    return std::atan2(
      std::sin(
        angle),
      std::cos(
        angle));
  }


  static double quaternionToYaw(
    const geometry_msgs::msg::Quaternion & q)
  {
    const double siny =
      2.0
      *
      (
        q.w * q.z
        +
        q.x * q.y
      );


    const double cosy =
      1.0
      -
      2.0
      *
      (
        q.y * q.y
        +
        q.z * q.z
      );


    return std::atan2(
      siny,
      cosy);
  }


  static geometry_msgs::msg::Quaternion
  yawToQuaternion(
    const double yaw)
  {
    geometry_msgs::msg::Quaternion q;


    q.x =
      0.0;


    q.y =
      0.0;


    q.z =
      std::sin(
        yaw * 0.5);


    q.w =
      std::cos(
        yaw * 0.5);


    return q;
  }


  static double clamp(
    const double value,
    const double min_value,
    const double max_value)
  {
    return std::max(
      min_value,
      std::min(
        value,
        max_value));
  }

  // ==============================================================
  // Convert native MOLA odometry to ROS/Nav2 convention
  //
  // Native MOLA convention observed during calibration:
  //
  //   x_nav   = +x_mola
  //   y_nav   = -y_mola
  //   yaw_nav = -yaw_mola
  //
  // The corresponding planar twist convention is:
  //
  //   vx_nav = +vx_mola
  //   vy_nav = -vy_mola
  //   wz_nav = -wz_mola
  //
  // /odom_mola_2d remains untouched and keeps the native MOLA
  // convention. Only the Nav2-facing state uses this conversion.
  // ==============================================================

  static nav_msgs::msg::Odometry
  convertMolaToNavigationConvention(
    const nav_msgs::msg::Odometry & mola_state)
  {
    nav_msgs::msg::Odometry nav_state =
      mola_state;


    // ------------------------------------------------------------
    // Pose
    // ------------------------------------------------------------

    nav_state.pose.pose.position.x =
      mola_state.pose.pose.position.x;


    nav_state.pose.pose.position.y =
      -mola_state.pose.pose.position.y;


    nav_state.pose.pose.position.z =
      mola_state.pose.pose.position.z;


    const double mola_yaw =
      quaternionToYaw(
        mola_state.pose.pose.orientation);


    const double nav_yaw =
      wrapAngle(
        -mola_yaw);


    nav_state.pose.pose.orientation =
      yawToQuaternion(
        nav_yaw);


    // ------------------------------------------------------------
    // Twist
    // ------------------------------------------------------------

    nav_state.twist.twist.linear.x =
      mola_state.twist.twist.linear.x;


    nav_state.twist.twist.linear.y =
      -mola_state.twist.twist.linear.y;


    nav_state.twist.twist.linear.z =
      mola_state.twist.twist.linear.z;


    nav_state.twist.twist.angular.x =
      -mola_state.twist.twist.angular.x;


    nav_state.twist.twist.angular.y =
      mola_state.twist.twist.angular.y;


    nav_state.twist.twist.angular.z =
      -mola_state.twist.twist.angular.z;


    // ------------------------------------------------------------
    // Covariance
    //
    // Reflection:
    //   position  : [x, y, z]       -> [+x, -y, +z]
    //   rotation  : [rx, ry, rz]    -> [-rx, +ry, -rz]
    //
    // Therefore:
    //   covariance_nav = S * covariance_mola * S^T
    //
    // with:
    //   S = diag(+1, -1, +1, -1, +1, -1)
    // ------------------------------------------------------------

    const double signs[6] = {
      1.0,
      -1.0,
      1.0,
      -1.0,
      1.0,
      -1.0
    };


    for (std::size_t row = 0; row < 6; ++row)
    {
      for (std::size_t column = 0; column < 6; ++column)
      {
        const std::size_t index =
          row * 6 + column;


        nav_state.pose.covariance[index] =
          signs[row]
          *
          mola_state.pose.covariance[index]
          *
          signs[column];


        nav_state.twist.covariance[index] =
          signs[row]
          *
          mola_state.twist.covariance[index]
          *
          signs[column];
      }
    }


    return nav_state;
  }


  // ==============================================================
  // Store latest complete navigation state
  //
  // The MOLA callback updates this snapshot.
  //
  // The Nav2 timer later copies it and republishes it with the
  // selected navigation timestamp policy.
  // ==============================================================

  void storeLatestNavigationState(
    const nav_msgs::msg::Odometry & state)
  {
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
    navigation_state_pending_ = true;
  }


  // ==============================================================
  // MOLA callback
  //
  // Responsibilities:
  //
  // 1. Receive the physical MOLA pose.
  // 2. Convert its timestamp into the common navigation time
  //    reference using the LiDAR clock offset.
  // 3. Project the pose into x/y/yaw.
  // 4. Reconstruct planar body-frame twist.
  // 5. Publish /odom_mola_2d.
  // 6. Store the latest complete state for Nav2.
  //
  // IMPORTANT:
  //
  // This callback does not directly publish:
  //
  //     /odom
  //     odom -> base_footprint
  //
  // Those outputs are handled by the independent ROS timer.
  // ==============================================================

  void odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }


    // ------------------------------------------------------------
    // Wait until the physical LiDAR -> Nav clock relationship is
    // known.
    // ------------------------------------------------------------

    if (apply_lidar_time_offset_ && !has_lidar_to_nav_offset_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for locked LiDAR->Nav time offset. "
        "Dropping MOLA odometry.");

      return;
    }


    // ------------------------------------------------------------
    // Extract planar MOLA pose
    // ------------------------------------------------------------

    const double x =
      msg->pose.pose.position.x;


    const double y =
      msg->pose.pose.position.y;


    const double yaw =
      quaternionToYaw(
        msg->pose.pose.orientation);


    // ------------------------------------------------------------
    // Physical MOLA source timestamp
    // ------------------------------------------------------------

    const rclcpp::Time source_stamp(
      msg->header.stamp);


    // ------------------------------------------------------------
    // Convert physical MOLA/LiDAR timestamp into the common
    // navigation ROS time reference.
    //
    // Livox preserves source time on both odometry outputs and TF.
    // ------------------------------------------------------------

    const rclcpp::Time physical_nav_stamp =
      apply_lidar_time_offset_
      ? source_stamp + rclcpp::Duration::from_seconds(lidar_to_nav_offset_sec_)
      : source_stamp;


    // ------------------------------------------------------------
    // Build physical/internal planar MOLA odometry
    // ------------------------------------------------------------

    nav_msgs::msg::Odometry out;


    out.header.stamp =
      physical_nav_stamp;


    out.header.frame_id =
      output_frame_id_;


    out.child_frame_id =
      output_child_frame_id_;


    // ------------------------------------------------------------
    // Force 2D pose
    // ------------------------------------------------------------

    out.pose.pose.position.x =
      x;


    out.pose.pose.position.y =
      y;


    out.pose.pose.position.z =
      0.0;


    out.pose.pose.orientation =
      yawToQuaternion(
        yaw);


    // ------------------------------------------------------------
    // Preserve pose covariance
    // ------------------------------------------------------------

    out.pose.covariance =
      msg->pose.covariance;


    // z
    out.pose.covariance[14] =
      1e3;


    // roll
    out.pose.covariance[21] =
      1e3;


    // pitch
    out.pose.covariance[28] =
      1e3;


    // ------------------------------------------------------------
    // First MOLA message
    //
    // We need two physical measurements before velocity can be
    // reconstructed.
    // ------------------------------------------------------------

    if (!initialized_)
    {
      prev_x_ =
        x;


      prev_y_ =
        y;


      prev_yaw_ =
        yaw;


      prev_stamp_ =
        source_stamp;


      initialized_ =
        true;


      zeroTwist(
        out);


      setTwistCovariance(
        out);


      mola_odom_pub_->publish(
        out);


      const nav_msgs::msg::Odometry nav_state =
        convertMolaToNavigationConvention(
          out);


      storeLatestNavigationState(
        nav_state);


      return;
    }


    // ------------------------------------------------------------
    // Physical MOLA sampling interval
    //
    // IMPORTANT:
    //
    // dt continues to come from MOLA source timestamps.
    //
    // Therefore, replacing Nav2 timestamps with now() does not alter
    // physical twist reconstruction.
    // ------------------------------------------------------------

    const double dt =
      (
        source_stamp
        -
        prev_stamp_
      ).seconds();


    // ------------------------------------------------------------
    // Invalid or discontinuous MOLA interval
    // ------------------------------------------------------------

    if (
      dt <= 1e-4
      ||
      dt > max_dt_)
    {
      prev_x_ =
        x;


      prev_y_ =
        y;


      prev_yaw_ =
        yaw;


      prev_stamp_ =
        source_stamp;


      filtered_vx_ =
        0.0;


      filtered_vy_ =
        0.0;


      filtered_wz_ =
        0.0;


      zeroTwist(
        out);


      setTwistCovariance(
        out);


      mola_odom_pub_->publish(
        out);


      const nav_msgs::msg::Odometry nav_state =
        convertMolaToNavigationConvention(
          out);


      storeLatestNavigationState(
        nav_state);


      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "Invalid MOLA dt detected: dt=%.3f s. "
        "Temporal reference resynchronized.",
        dt);


      return;
    }


    // ------------------------------------------------------------
    // Pose increment in odom frame
    // ------------------------------------------------------------

    const double dx_odom =
      x
      -
      prev_x_;


    const double dy_odom =
      y
      -
      prev_y_;


    const double dyaw =
      wrapAngle(
        yaw
        -
        prev_yaw_);


    // ------------------------------------------------------------
    // Convert translational increment into previous body frame
    //
    // nav_msgs/Odometry twist should be expressed in child_frame_id.
    // ------------------------------------------------------------

    const double cos_yaw =
      std::cos(
        prev_yaw_);


    const double sin_yaw =
      std::sin(
        prev_yaw_);


    const double dx_body =
      cos_yaw * dx_odom
      +
      sin_yaw * dy_odom;


    const double dy_body =
      -sin_yaw * dx_odom
      +
      cos_yaw * dy_odom;


    // ------------------------------------------------------------
    // Reconstruct raw planar body-frame velocity
    // ------------------------------------------------------------

    double vx =
      dx_body
      /
      dt;


    double vy =
      dy_body
      /
      dt;


    double wz =
      dyaw
      /
      dt;


    // ------------------------------------------------------------
    // Limit velocity spikes
    // ------------------------------------------------------------

    vx =
      clamp(
        vx,
        -max_linear_velocity_,
        max_linear_velocity_);


    vy =
      clamp(
        vy,
        -max_linear_velocity_,
        max_linear_velocity_);


    wz =
      clamp(
        wz,
        -max_angular_velocity_,
        max_angular_velocity_);


    // ------------------------------------------------------------
    // Optional low-pass filtering
    //
    // Only the reconstructed twist is filtered.
    // The MOLA pose itself is never modified.
    // ------------------------------------------------------------

    if (enable_twist_filter_)
    {
      vx =
        twist_filter_alpha_
        *
        vx
        +
        (
          1.0
          -
          twist_filter_alpha_
        )
        *
        filtered_vx_;


      vy =
        twist_filter_alpha_
        *
        vy
        +
        (
          1.0
          -
          twist_filter_alpha_
        )
        *
        filtered_vy_;


      wz =
        twist_filter_alpha_
        *
        wz
        +
        (
          1.0
          -
          twist_filter_alpha_
        )
        *
        filtered_wz_;
    }


    // ------------------------------------------------------------
    // Save filtered state before deadband
    // ------------------------------------------------------------

    filtered_vx_ =
      vx;


    filtered_vy_ =
      vy;


    filtered_wz_ =
      wz;


    // ------------------------------------------------------------
    // Stationary deadbands
    // ------------------------------------------------------------

    const double linear_speed =
      std::hypot(
        vx,
        vy);


    if (
      linear_speed
      <
      stationary_linear_threshold_)
    {
      vx =
        0.0;


      vy =
        0.0;
    }


    if (
      std::abs(
        wz)
      <
      stationary_angular_threshold_)
    {
      wz =
        0.0;
    }


    // ------------------------------------------------------------
    // Store reconstructed twist
    // ------------------------------------------------------------

    out.twist.twist.linear.x =
      vx;


    out.twist.twist.linear.y =
      vy;


    out.twist.twist.linear.z =
      0.0;


    out.twist.twist.angular.x =
      0.0;


    out.twist.twist.angular.y =
      0.0;


    out.twist.twist.angular.z =
      wz;


    setTwistCovariance(
      out);


    // ------------------------------------------------------------
    // Update physical source reference
    // ------------------------------------------------------------

    prev_x_ =
      x;


    prev_y_ =
      y;


    prev_yaw_ =
      yaw;


    prev_stamp_ =
      source_stamp;


    // ------------------------------------------------------------
    // Publish physical/internal MOLA odometry
    // ------------------------------------------------------------

    mola_odom_pub_->publish(
      out);


    // ------------------------------------------------------------
    // Store latest state for Nav2 timer
    // ------------------------------------------------------------

    const nav_msgs::msg::Odometry nav_state =
      convertMolaToNavigationConvention(
        out);


    storeLatestNavigationState(
      nav_state);


    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "MOLA ODOM | "
      "x=%+.3f y=%+.3f yaw=%+.2f deg | "
      "vx=%+.3f vy=%+.3f wz=%+.3f | "
      "dt=%.3f",
      x,
      y,
      yaw * 180.0 / M_PI,
      vx,
      vy,
      wz,
      dt);
  }


  // ==============================================================
  // Periodic Nav2 publication
  //
  // This is deliberately simple.
  //
  // At each timer tick:
  //
  //   1. Copy the latest MOLA state.
  //   2. Preserve source time for Livox; use ROS now for Unitree.
  //   3. Zero stale velocity if necessary.
  //   4. Publish /odom.
  //   5. Publish odom -> base_footprint with exactly the same time.
  //
  // There is:
  //
  //   - no pose prediction;
  //   - no future TF offset;
  //   - no custom publication thread;
  //   - no extrapolation of stale motion.
  // ==============================================================

  void publishNavigationState()
  {
    nav_msgs::msg::Odometry out;


    rclcpp::Time state_update_time(
      0,
      0,
      get_clock()->get_clock_type());


    // ------------------------------------------------------------
    // Atomically copy latest complete MOLA state
    // ------------------------------------------------------------

    {
      std::lock_guard<std::mutex> lock(
        state_mutex_);


      if (!have_latest_state_)
      {
        return;
      }


      // In Livox mode, publish each stored update at most once.
      // Do not repeatedly broadcast the same TF timestamp.
      if (!apply_lidar_time_offset_ && !navigation_state_pending_) {
        return;
      }
      navigation_state_pending_ = false;

      out =
        latest_state_;


      state_update_time =
        latest_state_update_time_;
    }


    // ------------------------------------------------------------
    // Current ROS navigation time
    // ------------------------------------------------------------

    const rclcpp::Time now =
      get_clock()->now();


    // ------------------------------------------------------------
    // Age of latest physical MOLA update
    // ------------------------------------------------------------

    const double state_age_sec =
      (
        now
        -
        state_update_time
      ).seconds();


    // ------------------------------------------------------------
    // Nav2-facing timestamp
    //
    // IMPORTANT:
    //
    // /odom and odom->base_footprint use exactly the same ROS time.
    //
    // Only the Unitree compatibility mode replaces measurement time.
    // ------------------------------------------------------------

    // Livox timestamps are already in the navigation time domain.
    if (apply_lidar_time_offset_) {
      out.header.stamp = now;
    }


    out.header.frame_id =
      output_frame_id_;


    out.child_frame_id =
      output_child_frame_id_;


    // ------------------------------------------------------------
    // Never continue publishing stale velocity
    //
    // If MOLA temporarily stops updating:
    //
    //   - hold the last known pose;
    //   - refresh TF periodically only in Unitree compatibility mode;
    //   - publish zero twist.
    //
    // No motion is invented.
    // ------------------------------------------------------------

    if (
      state_age_sec
      >
      max_state_age_sec_)
    {
      zeroTwist(
        out);


      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Latest MOLA state is stale: age=%.3f s. "
        "Holding pose and publishing zero twist.",
        state_age_sec);
    }


    // ------------------------------------------------------------
    // Publish Nav2 odometry
    // ------------------------------------------------------------

    nav_odom_pub_->publish(
      out);


    // ------------------------------------------------------------
    // Publish matching navigation TF
    //
    // The TF uses:
    //
    //   - exactly the same pose as /odom;
    //   - exactly the same timestamp as /odom.
    // ------------------------------------------------------------

    if (publish_nav_tf_)
    {
      geometry_msgs::msg::TransformStamped tf_msg;


      tf_msg.header.stamp =
        out.header.stamp;


      tf_msg.header.frame_id =
        output_frame_id_;


      tf_msg.child_frame_id =
        output_child_frame_id_;


      tf_msg.transform.translation.x =
        out.pose.pose.position.x;


      tf_msg.transform.translation.y =
        out.pose.pose.position.y;


      tf_msg.transform.translation.z =
        0.0;


      tf_msg.transform.rotation =
        out.pose.pose.orientation;


      tf_broadcaster_->sendTransform(
        tf_msg);
    }
  }


  // ==============================================================
  // Zero planar and non-planar twist
  // ==============================================================

  static void zeroTwist(
    nav_msgs::msg::Odometry & out)
  {
    out.twist.twist.linear.x =
      0.0;


    out.twist.twist.linear.y =
      0.0;


    out.twist.twist.linear.z =
      0.0;


    out.twist.twist.angular.x =
      0.0;


    out.twist.twist.angular.y =
      0.0;


    out.twist.twist.angular.z =
      0.0;
  }


  // ==============================================================
  // Conservative planar twist covariance
  // ==============================================================

  static void setTwistCovariance(
    nav_msgs::msg::Odometry & out)
  {
    out.twist.covariance.fill(
      0.0);


    // vx
    out.twist.covariance[0] =
      0.05;


    // vy
    out.twist.covariance[7] =
      0.05;


    // vz
    out.twist.covariance[14] =
      1e3;


    // wx
    out.twist.covariance[21] =
      1e3;


    // wy
    out.twist.covariance[28] =
      1e3;


    // wz
    out.twist.covariance[35] =
      0.05;
  }


  // ==============================================================
  // Parameters
  // ==============================================================

  std::string input_topic_;

  std::string output_topic_;

  std::string nav_output_topic_;

  std::string output_frame_id_;

  std::string output_child_frame_id_;


  bool apply_lidar_time_offset_;
  bool navigation_state_pending_{false};
  bool publish_nav_tf_;

  bool enable_twist_filter_;


  double nav_publish_frequency_;

  double max_state_age_sec_;

  double twist_filter_alpha_;

  double stationary_linear_threshold_;

  double stationary_angular_threshold_;

  double max_dt_;

  double max_linear_velocity_;

  double max_angular_velocity_;


  // ==============================================================
  // ROS interfaces
  // ==============================================================

  rclcpp::Subscription<
    nav_msgs::msg::Odometry>::SharedPtr
    sub_;


  rclcpp::Publisher<
    nav_msgs::msg::Odometry>::SharedPtr
    mola_odom_pub_;


  rclcpp::Publisher<
    nav_msgs::msg::Odometry>::SharedPtr
    nav_odom_pub_;


  std::shared_ptr<
    tf2_ros::TransformBroadcaster>
    tf_broadcaster_;


  rclcpp::TimerBase::SharedPtr
    nav_timer_;


  rclcpp::Subscription<
    std_msgs::msg::Float64>::SharedPtr
    lidar_to_nav_offset_sub_;


  // ==============================================================
  // Physical MOLA temporal state
  // ==============================================================

  bool initialized_ =
    false;


  double prev_x_ =
    0.0;


  double prev_y_ =
    0.0;


  double prev_yaw_ =
    0.0;


  rclcpp::Time prev_stamp_{
    0,
    0,
    RCL_ROS_TIME};


  // ==============================================================
  // Reconstructed twist filter state
  // ==============================================================

  double filtered_vx_ =
    0.0;


  double filtered_vy_ =
    0.0;


  double filtered_wz_ =
    0.0;


  // ==============================================================
  // Time synchronization state
  // ==============================================================

  double lidar_to_nav_offset_sec_ =
    0.0;


  bool has_lidar_to_nav_offset_ =
    false;


  // ==============================================================
  // Latest complete state shared with Nav2 timer
  // ==============================================================

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


// ================================================================
// Main
// ================================================================

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
      std::make_shared<
        MolaOdomAdapter>();


    rclcpp::spin(
      node);


    rclcpp::shutdown();


    return 0;
  }


  catch (
    const std::exception & e)
  {
    std::cerr
      << "[mola_odom_adapter_node] Fatal exception: "
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
      << "[mola_odom_adapter_node] "
      << "Unknown fatal exception"
      << std::endl;


    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }


    return 1;
  }
}