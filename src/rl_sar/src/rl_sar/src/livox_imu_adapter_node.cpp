#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"


class LivoxImuAdapterNode : public rclcpp::Node
{
public:
  // ==========================================================================
  // CONSTRUCTION
  // ==========================================================================

  LivoxImuAdapterNode()
  : Node("livox_imu_adapter_node")
  {
    declareAndLoadParameters();
    validateParameters();
    createRosInterfaces();
    reportConfiguration();
  }


private:
  // ==========================================================================
  // PARAMETER INITIALIZATION
  // ==========================================================================

  void declareAndLoadParameters()
  {
    input_topic_ =
      declare_parameter<std::string>(
        "input_topic",
        "/livox/imu");

    output_topic_ =
      declare_parameter<std::string>(
        "output_topic",
        "/imu_livox_corrected");

    expected_frame_id_ =
      declare_parameter<std::string>(
        "expected_frame_id",
        "livox_frame");

    angular_velocity_variance_x_ =
      declare_parameter<double>(
        "angular_velocity_variance_x",
        0.01);

    angular_velocity_variance_y_ =
      declare_parameter<double>(
        "angular_velocity_variance_y",
        0.01);

    angular_velocity_variance_z_ =
      declare_parameter<double>(
        "angular_velocity_variance_z",
        0.01);

    reject_unexpected_frame_ =
      declare_parameter<bool>(
        "reject_unexpected_frame",
        true);
  }


  // ==========================================================================
  // PARAMETER VALIDATION
  // ==========================================================================

  void validateParameters() const
  {
    validateNonEmptyString(
      input_topic_,
      "input_topic");

    validateNonEmptyString(
      output_topic_,
      "output_topic");

    validateNonEmptyString(
      expected_frame_id_,
      "expected_frame_id");

    validatePositiveFinite(
      angular_velocity_variance_x_,
      "angular_velocity_variance_x");

    validatePositiveFinite(
      angular_velocity_variance_y_,
      "angular_velocity_variance_y");

    validatePositiveFinite(
      angular_velocity_variance_z_,
      "angular_velocity_variance_z");
  }


  static void validateNonEmptyString(
    const std::string & value,
    const std::string & parameter_name)
  {
    if (value.empty())
    {
      throw std::runtime_error(
        parameter_name + " must not be empty.");
    }
  }


  static void validatePositiveFinite(
    const double value,
    const std::string & parameter_name)
  {
    if (
      !std::isfinite(value)
      || value <= 0.0)
    {
      throw std::runtime_error(
        parameter_name
        + " must be positive and finite.");
    }
  }


  // ==========================================================================
  // ROS INTERFACES
  // ==========================================================================

  void createRosInterfaces()
  {
    publisher_ =
      create_publisher<sensor_msgs::msg::Imu>(
        output_topic_,
        rclcpp::SensorDataQoS());

    subscription_ =
      create_subscription<sensor_msgs::msg::Imu>(
        input_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(
          &LivoxImuAdapterNode::imuCallback,
          this,
          std::placeholders::_1));
  }


  // ==========================================================================
  // INPUT VALIDATION
  // ==========================================================================

  bool validateInputMessage(
    const sensor_msgs::msg::Imu & message)
  {
    const std::string normalized_frame =
      normalizeFrameId(
        message.header.frame_id);

    const std::string normalized_expected_frame =
      normalizeFrameId(
        expected_frame_id_);

    if (
      normalized_frame
      != normalized_expected_frame)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Unexpected IMU frame | received='%s' expected='%s'",
        message.header.frame_id.c_str(),
        expected_frame_id_.c_str());

      if (reject_unexpected_frame_)
      {
        return false;
      }
    }

    const auto & angular_velocity =
      message.angular_velocity;

    if (
      !std::isfinite(angular_velocity.x)
      || !std::isfinite(angular_velocity.y)
      || !std::isfinite(angular_velocity.z))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Non-finite angular velocity received. Message dropped.");

      return false;
    }

    return true;
  }


  static std::string normalizeFrameId(
    const std::string & frame_id)
  {
    if (
      !frame_id.empty()
      && frame_id.front() == '/')
    {
      return frame_id.substr(1);
    }

    return frame_id;
  }


  // ==========================================================================
  // OUTPUT MESSAGE CONSTRUCTION
  // ==========================================================================

  sensor_msgs::msg::Imu buildOutputMessage(
    const sensor_msgs::msg::Imu & input) const
  {
    sensor_msgs::msg::Imu output = input;

    /*
     * Preserve the original timestamp.
     *
     * Keep the sensor frame instead of pretending that the measurements are
     * already expressed in base_footprint. robot_localization will transform
     * the angular velocity vector using:
     *
     *     base_footprint <-> livox_frame
     */
    output.header.frame_id =
      normalizeFrameId(
        expected_frame_id_);


    // ------------------------------------------------------------------------
    // Orientation
    //
    // The Livox driver publishes an identity quaternion without a meaningful
    // orientation estimate. According to sensor_msgs/Imu conventions,
    // covariance[0] = -1 means that this measurement is unavailable.
    // ------------------------------------------------------------------------

    output.orientation.x = 0.0;
    output.orientation.y = 0.0;
    output.orientation.z = 0.0;
    output.orientation.w = 1.0;

    output.orientation_covariance.fill(0.0);
    output.orientation_covariance[0] = -1.0;


    // ------------------------------------------------------------------------
    // Angular velocity
    //
    // Preserve the complete raw vector in livox_frame. The TF rotation,
    // including the approximately 180-degree roll, must be applied by
    // robot_localization.
    // ------------------------------------------------------------------------

    output.angular_velocity_covariance.fill(0.0);

    output.angular_velocity_covariance[0] =
      angular_velocity_variance_x_;

    output.angular_velocity_covariance[4] =
      angular_velocity_variance_y_;

    output.angular_velocity_covariance[8] =
      angular_velocity_variance_z_;


    // ------------------------------------------------------------------------
    // Linear acceleration
    //
    // The current Livox acceleration values appear not to use the SI units
    // required by sensor_msgs/Imu. Mark the measurement as unavailable.
    // ------------------------------------------------------------------------

    output.linear_acceleration.x = 0.0;
    output.linear_acceleration.y = 0.0;
    output.linear_acceleration.z = 0.0;

    output.linear_acceleration_covariance.fill(0.0);
    output.linear_acceleration_covariance[0] = -1.0;

    return output;
  }


  // ==========================================================================
  // CALLBACK
  // ==========================================================================

  void imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr message)
  {
    if (!message)
    {
      return;
    }

    received_message_count_++;

    if (!validateInputMessage(*message))
    {
      rejected_message_count_++;
      return;
    }

    const auto output =
      buildOutputMessage(*message);

    publisher_->publish(output);

    published_message_count_++;

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      5000,
      "Livox IMU adapter | received=%lu rejected=%lu published=%lu | "
      "w=(%+.4f, %+.4f, %+.4f) rad/s",
      static_cast<unsigned long>(
        received_message_count_),
      static_cast<unsigned long>(
        rejected_message_count_),
      static_cast<unsigned long>(
        published_message_count_),
      output.angular_velocity.x,
      output.angular_velocity.y,
      output.angular_velocity.z);
  }


  // ==========================================================================
  // STARTUP REPORT
  // ==========================================================================

  void reportConfiguration() const
  {
    RCLCPP_INFO(
      get_logger(),
      "\n"
      "============================================================\n"
      "                 LIVOX IMU ADAPTER NODE\n"
      "============================================================\n"
      " Input topic                  : %s\n"
      " Output topic                 : %s\n"
      " Expected frame               : %s\n"
      " Reject unexpected frame      : %s\n"
      "\n"
      " ANGULAR VELOCITY COVARIANCE\n"
      "------------------------------------------------------------\n"
      " Variance X                   : %.6f\n"
      " Variance Y                   : %.6f\n"
      " Variance Z                   : %.6f\n"
      "\n"
      " MEASUREMENT POLICY\n"
      "------------------------------------------------------------\n"
      " Orientation                  : unavailable\n"
      " Angular velocity             : preserved in sensor frame\n"
      " Linear acceleration          : unavailable\n"
      " Timestamp                    : preserved\n"
      " Frame transformation         : delegated to robot_localization\n"
      "============================================================",
      input_topic_.c_str(),
      output_topic_.c_str(),
      expected_frame_id_.c_str(),
      reject_unexpected_frame_ ? "yes" : "no",
      angular_velocity_variance_x_,
      angular_velocity_variance_y_,
      angular_velocity_variance_z_);
  }


  // ==========================================================================
  // PARAMETERS
  // ==========================================================================

  std::string input_topic_;
  std::string output_topic_;
  std::string expected_frame_id_;

  double angular_velocity_variance_x_{0.01};
  double angular_velocity_variance_y_{0.01};
  double angular_velocity_variance_z_{0.01};

  bool reject_unexpected_frame_{true};


  // ==========================================================================
  // RUNTIME STATE
  // ==========================================================================

  std::size_t received_message_count_{0};
  std::size_t rejected_message_count_{0};
  std::size_t published_message_count_{0};


  // ==========================================================================
  // ROS OBJECTS
  // ==========================================================================

  rclcpp::Subscription<
    sensor_msgs::msg::Imu
  >::SharedPtr subscription_;

  rclcpp::Publisher<
    sensor_msgs::msg::Imu
  >::SharedPtr publisher_;
};


// =============================================================================
// MAIN
// =============================================================================

int main(
  int argc,
  char ** argv)
{
  rclcpp::init(
    argc,
    argv);

  const auto node =
    std::make_shared<LivoxImuAdapterNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}