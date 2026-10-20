#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "geometry_msgs/msg/twist.hpp"

#include "geometry_msgs/msg/quaternion.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"


class DogOdomCalibrationNode : public rclcpp::Node
{
public:
  DogOdomCalibrationNode()
  : Node("dog_odom_calibration_node")
  {
    // ========================================================================
    // TOPICS / FRAMES
    // ========================================================================

    input_topic_ = declare_parameter<std::string>(
      "input_topic",
      "/dog_odom_2d");

    output_topic_ = declare_parameter<std::string>(
      "output_topic",
      "/odom_dog_calibrated");

    odom_frame_ = declare_parameter<std::string>(
      "odom_frame",
      "odom");

    base_frame_ = declare_parameter<std::string>(
      "base_frame",
      "base_footprint");


    // ========================================================================
    // GLOBAL ACTIVATION
    // ========================================================================

    enable_calibration_ = declare_parameter<bool>(
      "enable_calibration",
      true);


    // ========================================================================
    // STRAIGHT-MOTION CALIBRATION
    // ========================================================================

    enable_longitudinal_scaling_ = declare_parameter<bool>(
      "enable_longitudinal_scaling",
      true);

    longitudinal_scale_ = declare_parameter<double>(
      "longitudinal_scale",
      1.168956);

    enable_lateral_scaling_ = declare_parameter<bool>(
      "enable_lateral_scaling",
      false);

    lateral_scale_ = declare_parameter<double>(
      "lateral_scale",
      1.0);


    // ========================================================================
    // PURE-ROTATION CALIBRATION
    // ========================================================================

    enable_yaw_scaling_ = declare_parameter<bool>(
      "enable_yaw_scaling",
      false);

    yaw_scale_ccw_ = declare_parameter<double>(
      "yaw_scale_ccw",
      1.0);

    yaw_scale_cw_ = declare_parameter<double>(
      "yaw_scale_cw",
      1.0);

    maximum_linear_velocity_for_rotation_ =
      declare_parameter<double>(
        "maximum_linear_velocity_for_rotation",
        0.02);

    minimum_angular_velocity_for_rotation_ =
      declare_parameter<double>(
        "minimum_angular_velocity_for_rotation",
        0.03);


    command_topic_ =
      declare_parameter<std::string>(
        "command_topic",
        "/cmd_vel_rlsar");

    command_timeout_s_ =
      declare_parameter<double>(
        "command_timeout_s",
        0.30);
        
    // ========================================================================
    // TWIST
    //
    // The current calibration was estimated from pose increments.
    // Keep this false until Twist calibration is independently validated.
    // ========================================================================

    scale_twist_ = declare_parameter<bool>(
      "scale_twist",
      false);


    // ========================================================================
    // DISCONTINUITY GUARDS
    // ========================================================================

    maximum_translation_increment_m_ =
      declare_parameter<double>(
        "maximum_translation_increment_m",
        0.20);

    maximum_yaw_increment_rad_ =
      declare_parameter<double>(
        "maximum_yaw_increment_rad",
        0.30);


    // ========================================================================
    // DIAGNOSTICS
    // ========================================================================

    enable_latency_diagnostics_ =
      declare_parameter<bool>(
        "enable_latency_diagnostics",
        true);

    computation_latency_topic_ =
      declare_parameter<std::string>(
        "computation_latency_topic",
        "/dog_odom_calibration/computation_latency_ms");

    callback_latency_topic_ =
      declare_parameter<std::string>(
        "callback_latency_topic",
        "/dog_odom_calibration/callback_latency_ms");


    validateParameters();


    // ========================================================================
    // ROS INTERFACES
    // ========================================================================

    publisher_ =
      create_publisher<nav_msgs::msg::Odometry>(
        output_topic_,
        rclcpp::QoS(10));

    subscription_ =
      create_subscription<nav_msgs::msg::Odometry>(
        input_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(
          &DogOdomCalibrationNode::odomCallback,
          this,
          std::placeholders::_1));

/*
    // Accept commands from either best-effort or reliable publishers.
    auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    command_qos.best_effort();
    command_qos.durability_volatile();

    command_subscription_ =
      create_subscription<geometry_msgs::msg::Twist>(
        command_topic_,
        command_qos,
        std::bind(
          &DogOdomCalibrationNode::commandCallback,
          this,
          std::placeholders::_1));


*/

    command_subscription_ =
      create_subscription<geometry_msgs::msg::Twist>(
        command_topic_,
        rclcpp::QoS(10),
        std::bind(
          &DogOdomCalibrationNode::commandCallback,
          this,
          std::placeholders::_1));
          

    if (enable_latency_diagnostics_)
    {
      computation_latency_publisher_ =
        create_publisher<std_msgs::msg::Float64>(
          computation_latency_topic_,
          rclcpp::QoS(10));

      callback_latency_publisher_ =
        create_publisher<std_msgs::msg::Float64>(
          callback_latency_topic_,
          rclcpp::QoS(10));
    }


    RCLCPP_INFO(
      get_logger(),
      "\n"
      "============================================================\n"
      "              DOG ODOM CALIBRATION NODE\n"
      "============================================================\n"
      " Input topic                       : %s\n"
      " Output topic                      : %s\n"
      " Calibration enabled               : %s\n"
      "\n"
      " STRAIGHT MOTION\n"
      "------------------------------------------------------------\n"
      " Longitudinal scaling              : %s\n"
      " Longitudinal scale                : %.6f\n"
      " Lateral scaling                   : %s\n"
      " Lateral scale                     : %.6f\n"
      "\n"
      " PURE ROTATION\n"
      "------------------------------------------------------------\n"
      " Yaw scaling                       : %s\n"
      " CCW yaw scale                     : %.6f\n"
      " CW yaw scale                      : %.6f\n"
      " Maximum rotation linear velocity  : %.4f m/s\n"
      " Minimum rotation angular velocity : %.4f rad/s\n"
      "\n"
      " OUTPUT\n"
      "------------------------------------------------------------\n"
      " Twist scaling                     : %s\n"
      " Publish TF                        : NO\n"
      "============================================================",
      input_topic_.c_str(),
      output_topic_.c_str(),
      enable_calibration_ ? "yes" : "no",
      enable_longitudinal_scaling_ ? "yes" : "no",
      longitudinal_scale_,
      enable_lateral_scaling_ ? "yes" : "no",
      lateral_scale_,
      enable_yaw_scaling_ ? "yes" : "no",
      yaw_scale_ccw_,
      yaw_scale_cw_,
      maximum_linear_velocity_for_rotation_,
      minimum_angular_velocity_for_rotation_,
      scale_twist_ ? "yes" : "no");
  }


private:
  struct Pose2D
  {
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
  };


  struct BodyIncrement
  {
    double longitudinal{0.0};
    double lateral{0.0};
    double yaw{0.0};
  };


  // ==========================================================================
  // VALIDATION
  // ==========================================================================

  void validateParameters() const
  {
    validatePositiveFinite(
      longitudinal_scale_,
      "longitudinal_scale");

    validatePositiveFinite(
      lateral_scale_,
      "lateral_scale");

    validatePositiveFinite(
      yaw_scale_ccw_,
      "yaw_scale_ccw");

    validatePositiveFinite(
      yaw_scale_cw_,
      "yaw_scale_cw");

    validatePositiveFinite(
      maximum_translation_increment_m_,
      "maximum_translation_increment_m");

    validatePositiveFinite(
      maximum_yaw_increment_rad_,
      "maximum_yaw_increment_rad");

    validateNonNegativeFinite(
      maximum_linear_velocity_for_rotation_,
      "maximum_linear_velocity_for_rotation");

    validateNonNegativeFinite(
      minimum_angular_velocity_for_rotation_,
      "minimum_angular_velocity_for_rotation");

    validatePositiveFinite(
      command_timeout_s_,
      "command_timeout_s");
    if (input_topic_.empty())
    {
      throw std::runtime_error(
        "input_topic must not be empty.");
    }

    if (output_topic_.empty())
    {
      throw std::runtime_error(
        "output_topic must not be empty.");
    }

    if (odom_frame_.empty())
    {
      throw std::runtime_error(
        "odom_frame must not be empty.");
    }

    if (base_frame_.empty())
    {
      throw std::runtime_error(
        "base_frame must not be empty.");
    }
  }


  static void validatePositiveFinite(
    const double value,
    const std::string & name)
  {
    if (!std::isfinite(value) || value <= 0.0)
    {
      throw std::runtime_error(
        name + " must be positive and finite.");
    }
  }


  static void validateNonNegativeFinite(
    const double value,
    const std::string & name)
  {
    if (!std::isfinite(value) || value < 0.0)
    {
      throw std::runtime_error(
        name + " must be non-negative and finite.");
    }
  }


  // ==========================================================================
  // GEOMETRY
  // ==========================================================================

  static double normalizeAngle(const double angle)
  {
    return std::atan2(
      std::sin(angle),
      std::cos(angle));
  }


  static double quaternionToYaw(
    const geometry_msgs::msg::Quaternion & quaternion)
  {
    const double sin_yaw =
      2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y);

    const double cos_yaw =
      1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z);

    return std::atan2(
      sin_yaw,
      cos_yaw);
  }


  static geometry_msgs::msg::Quaternion yawToQuaternion(
    const double yaw)
  {
    geometry_msgs::msg::Quaternion quaternion;

    quaternion.x = 0.0;
    quaternion.y = 0.0;
    quaternion.z = std::sin(0.5 * yaw);
    quaternion.w = std::cos(0.5 * yaw);

    return quaternion;
  }


  static Pose2D extractPose(
    const nav_msgs::msg::Odometry & message)
  {
    Pose2D pose;

    pose.x = message.pose.pose.position.x;
    pose.y = message.pose.pose.position.y;

    pose.yaw = quaternionToYaw(
      message.pose.pose.orientation);

    return pose;
  }


  static bool isFinitePose(const Pose2D & pose)
  {
    return (
      std::isfinite(pose.x)
      && std::isfinite(pose.y)
      && std::isfinite(pose.yaw));
  }


  // ==========================================================================
  // MOTION CLASSIFICATION
  // ==========================================================================

  bool isPureRotation() const
  {
    if (!command_received_)
    {
      return false;
    }

    const double command_age_s =
      (now() - last_command_time_).seconds();

    if (
      command_age_s < 0.0
      || command_age_s > command_timeout_s_)
    {
      return false;
    }

    const double commanded_linear_speed =
      std::hypot(
        last_command_vx_,
        last_command_vy_);

    return (
      commanded_linear_speed
        <= maximum_linear_velocity_for_rotation_
      &&
      std::abs(last_command_wz_)
        >= minimum_angular_velocity_for_rotation_);
  }

  // ==========================================================================
  // RELATIVE MOTION
  // ==========================================================================

  static BodyIncrement computeRelativeIncrement(
    const Pose2D & previous_pose,
    const Pose2D & current_pose)
  {
    const double delta_x =
      current_pose.x - previous_pose.x;

    const double delta_y =
      current_pose.y - previous_pose.y;

    const double delta_yaw =
      normalizeAngle(
        current_pose.yaw - previous_pose.yaw);

    const double reference_yaw =
      normalizeAngle(
        previous_pose.yaw + 0.5 * delta_yaw);

    const double cosine =
      std::cos(reference_yaw);

    const double sine =
      std::sin(reference_yaw);

    BodyIncrement increment;

    increment.longitudinal =
      cosine * delta_x
      + sine * delta_y;

    increment.lateral =
      -sine * delta_x
      + cosine * delta_y;

    increment.yaw = delta_yaw;

    return increment;
  }


  bool validateRawIncrement(
    const BodyIncrement & increment) const
  {
    const double translation =
      std::hypot(
        increment.longitudinal,
        increment.lateral);

    return (
      translation <= maximum_translation_increment_m_
      && std::abs(increment.yaw)
        <= maximum_yaw_increment_rad_);
  }


  BodyIncrement applyCalibration(
    const BodyIncrement & raw_increment,
    const bool pure_rotation) const
  {
    BodyIncrement calibrated =
      raw_increment;

    if (pure_rotation)
    {
      /*
       * Straight-motion scales are deliberately not applied during
       * pure rotation. This prevents longitudinal_scale from amplifying
       * the observed rotational translation drift.
       */
      if (enable_yaw_scaling_)
      {
        if (raw_increment.yaw >= 0.0)
        {
          calibrated.yaw *= yaw_scale_ccw_;
        }
        else
        {
          calibrated.yaw *= yaw_scale_cw_;
        }
      }

      return calibrated;
    }

    if (enable_longitudinal_scaling_)
    {
      calibrated.longitudinal *=
        longitudinal_scale_;
    }

    if (enable_lateral_scaling_)
    {
      calibrated.lateral *=
        lateral_scale_;
    }

    /*
     * Yaw scaling is intentionally not applied to general curved or
     * straight motion. The current yaw factors belong to pure-rotation
     * physical validation campaigns.
     */

    return calibrated;
  }


  static Pose2D integrateIncrement(
    const Pose2D & previous_pose,
    const BodyIncrement & increment)
  {
    const double reference_yaw =
      normalizeAngle(
        previous_pose.yaw + 0.5 * increment.yaw);

    const double cosine =
      std::cos(reference_yaw);

    const double sine =
      std::sin(reference_yaw);

    Pose2D result = previous_pose;

    result.x +=
      cosine * increment.longitudinal
      - sine * increment.lateral;

    result.y +=
      sine * increment.longitudinal
      + cosine * increment.lateral;

    result.yaw = normalizeAngle(
      previous_pose.yaw + increment.yaw);

    return result;
  }


  // ==========================================================================
  // OUTPUT
  // ==========================================================================

  nav_msgs::msg::Odometry buildCalibratedMessage(
    const nav_msgs::msg::Odometry & input,
    const Pose2D & calibrated_pose,
    const bool pure_rotation) const
  {
    nav_msgs::msg::Odometry output = input;

    output.header.frame_id = odom_frame_;
    output.child_frame_id = base_frame_;

    output.pose.pose.position.x = calibrated_pose.x;
    output.pose.pose.position.y = calibrated_pose.y;
    output.pose.pose.position.z = 0.0;

    output.pose.pose.orientation =
      yawToQuaternion(calibrated_pose.yaw);

    if (scale_twist_)
    {
      if (pure_rotation)
      {
        if (enable_yaw_scaling_)
        {
          if (output.twist.twist.angular.z >= 0.0)
          {
            output.twist.twist.angular.z *=
              yaw_scale_ccw_;
          }
          else
          {
            output.twist.twist.angular.z *=
              yaw_scale_cw_;
          }
        }
      }
      else
      {
        if (enable_longitudinal_scaling_)
        {
          output.twist.twist.linear.x *=
            longitudinal_scale_;
        }

        if (enable_lateral_scaling_)
        {
          output.twist.twist.linear.y *=
            lateral_scale_;
        }
      }
    }

    output.twist.twist.linear.z = 0.0;
    output.twist.twist.angular.x = 0.0;
    output.twist.twist.angular.y = 0.0;

    return output;
  }


  nav_msgs::msg::Odometry buildPassthroughMessage(
    const nav_msgs::msg::Odometry & input) const
  {
    nav_msgs::msg::Odometry output = input;

    /*
     * Only canonical frame names are imposed.
     * Pose, Twist, covariance and timestamp remain unchanged.
     */
    output.header.frame_id = odom_frame_;
    output.child_frame_id = base_frame_;

    return output;
  }


  // ==========================================================================
  // DIAGNOSTICS
  // ==========================================================================

  void publishOutput(
    const nav_msgs::msg::Odometry & output,
    const std::chrono::steady_clock::time_point & callback_start,
    const std::chrono::steady_clock::time_point & computation_end)
  {
    const double computation_latency_ms =
      std::chrono::duration<double, std::milli>(
        computation_end - callback_start).count();

    publisher_->publish(output);

    const auto callback_end =
      std::chrono::steady_clock::now();

    const double callback_latency_ms =
      std::chrono::duration<double, std::milli>(
        callback_end - callback_start).count();

    if (!enable_latency_diagnostics_)
    {
      return;
    }

    std_msgs::msg::Float64 computation_message;
    computation_message.data = computation_latency_ms;

    computation_latency_publisher_->publish(
      computation_message);

    std_msgs::msg::Float64 callback_message;
    callback_message.data = callback_latency_ms;

    callback_latency_publisher_->publish(
      callback_message);
  }


  // ==========================================================================
  // CALLBACK
  // ==========================================================================

  void commandCallback(
    const geometry_msgs::msg::Twist::SharedPtr message)
  {
    if (!message)
    {
      return;
    }

    last_command_vx_ = message->linear.x;
    last_command_vy_ = message->linear.y;
    last_command_wz_ = message->angular.z;

    last_command_time_ = now();
    command_received_ = true;
  }

  void odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr message)
  {
    if (!message)
    {
      return;
    }

    const auto callback_start =
      std::chrono::steady_clock::now();

    const Pose2D current_raw_pose =
      extractPose(*message);

    if (!isFinitePose(current_raw_pose))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Non-finite DOG pose received. Message dropped.");

      return;
    }


    // ========================================================================
    // TRUE PASSTHROUGH
    // ========================================================================

    if (!enable_calibration_)
    {
      const auto output =
        buildPassthroughMessage(*message);

      const auto computation_end =
        std::chrono::steady_clock::now();

      publishOutput(
        output,
        callback_start,
        computation_end);

      return;
    }


    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    if (!initialized_)
    {
      previous_raw_pose_ = current_raw_pose;
      calibrated_pose_ = current_raw_pose;
      initialized_ = true;

      const bool pure_rotation =
        isPureRotation();

      const auto output =
        buildCalibratedMessage(
          *message,
          calibrated_pose_,
          pure_rotation);

      const auto computation_end =
        std::chrono::steady_clock::now();

      publishOutput(
        output,
        callback_start,
        computation_end);

      RCLCPP_INFO(
        get_logger(),
        "Calibration initialized at "
        "x=%+.3f y=%+.3f yaw=%+.2f deg",
        calibrated_pose_.x,
        calibrated_pose_.y,
        calibrated_pose_.yaw * 180.0 / PI_);

      return;
    }


    // ========================================================================
    // RELATIVE INCREMENT
    // ========================================================================

    const Pose2D previous_raw_pose =
      previous_raw_pose_;

    const BodyIncrement raw_increment =
      computeRelativeIncrement(
        previous_raw_pose,
        current_raw_pose);

    if (!validateRawIncrement(raw_increment))
    {
      previous_raw_pose_ =
        current_raw_pose;

      const bool pure_rotation =
        isPureRotation();

      const auto output =
        buildCalibratedMessage(
          *message,
          calibrated_pose_,
          pure_rotation);

      const auto computation_end =
        std::chrono::steady_clock::now();

      publishOutput(
        output,
        callback_start,
        computation_end);

      RCLCPP_WARN(
        get_logger(),
        "DOG discontinuity rejected: "
        "translation=%+.4f m yaw=%+.3f deg",
        std::hypot(
          raw_increment.longitudinal,
          raw_increment.lateral),
        raw_increment.yaw * 180.0 / PI_);

      return;
    }


    // ========================================================================
    // MOTION REGIME AND CALIBRATION
    // ========================================================================

    const bool pure_rotation =
      isPureRotation();

    const BodyIncrement calibrated_increment =
      applyCalibration(
        raw_increment,
        pure_rotation);

    calibrated_pose_ =
      integrateIncrement(
        calibrated_pose_,
        calibrated_increment);

    previous_raw_pose_ =
      current_raw_pose;


    // ========================================================================
    // PUBLISH
    // ========================================================================

    const auto output =
      buildCalibratedMessage(
        *message,
        calibrated_pose_,
        pure_rotation);

    const auto computation_end =
      std::chrono::steady_clock::now();

    publishOutput(
      output,
      callback_start,
      computation_end);


    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "DOG CAL | regime=%s | "
      "raw=(long=%+.4f lat=%+.4f yaw=%+.3fdeg) | "
      "cal=(long=%+.4f lat=%+.4f yaw=%+.3fdeg)",
      pure_rotation ? "PURE_ROTATION" : "GENERAL_MOTION",
      raw_increment.longitudinal,
      raw_increment.lateral,
      raw_increment.yaw * 180.0 / PI_,
      calibrated_increment.longitudinal,
      calibrated_increment.lateral,
      calibrated_increment.yaw * 180.0 / PI_);
  }


  // ==========================================================================
  // PARAMETERS
  // ==========================================================================

  static constexpr double PI_ =
    3.14159265358979323846;

  std::string input_topic_;
  std::string output_topic_;
  std::string odom_frame_;
  std::string base_frame_;

  bool enable_calibration_{true};

  bool enable_longitudinal_scaling_{true};
  double longitudinal_scale_{1.168956};

  bool enable_lateral_scaling_{false};
  double lateral_scale_{1.0};

  bool enable_yaw_scaling_{false};
  double yaw_scale_ccw_{1.0};
  double yaw_scale_cw_{1.0};

  double maximum_linear_velocity_for_rotation_{0.02};
  double minimum_angular_velocity_for_rotation_{0.03};

  bool scale_twist_{false};

  double maximum_translation_increment_m_{0.20};
  double maximum_yaw_increment_rad_{0.30};

  std::string command_topic_;
  double command_timeout_s_{0.30};

  bool command_received_{false};

  double last_command_vx_{0.0};
  double last_command_vy_{0.0};
  double last_command_wz_{0.0};

  rclcpp::Time last_command_time_{0, 0};

  rclcpp::Subscription<
    geometry_msgs::msg::Twist
  >::SharedPtr command_subscription_;

  bool enable_latency_diagnostics_{true};
  std::string computation_latency_topic_;
  std::string callback_latency_topic_;

  bool initialized_{false};
  Pose2D previous_raw_pose_;
  Pose2D calibrated_pose_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
    subscription_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
    publisher_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
    computation_latency_publisher_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr
    callback_latency_publisher_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<DogOdomCalibrationNode>());

  rclcpp::shutdown();

  return 0;
}