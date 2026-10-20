#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "geometry_msgs/msg/quaternion.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"


class DogOdomAdapter : public rclcpp::Node
{
public:
  // ==========================================================================
  // CONSTRUCTION / DESTRUCTION
  // ==========================================================================

  DogOdomAdapter()
  : Node("dog_odom_adapter_node")
  {
    // ========================================================================
    // INPUT / CONFIGURATION
    // ========================================================================

    input_topic_ =
      declare_parameter<std::string>(
        "input_topic",
        "/dog_odom");

    output_topic_ =
      declare_parameter<std::string>(
        "output_topic",
        "/dog_odom_2d");

    output_frame_id_ =
      declare_parameter<std::string>(
        "output_frame_id",
        "odom");

    output_child_frame_id_ =
      declare_parameter<std::string>(
        "output_child_frame_id",
        "base_footprint");

    preserve_source_twist_ =
      declare_parameter<bool>(
        "preserve_source_twist",
        true);


    // ========================================================================
    // ROS INTERFACES
    // ========================================================================

    pub_ =
      create_publisher<nav_msgs::msg::Odometry>(
        output_topic_,
        rclcpp::QoS(10));

    sub_ =
      create_subscription<nav_msgs::msg::Odometry>(
        input_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(
          &DogOdomAdapter::odomCallback,
          this,
          std::placeholders::_1));


    rclcpp::QoS time_sync_qos(
      rclcpp::KeepLast(1));

    time_sync_qos.reliable();
    time_sync_qos.transient_local();


    dog_to_nav_offset_sub_ =
      create_subscription<std_msgs::msg::Float64>(
        "/navigation_time/dog_to_nav_offset",
        time_sync_qos,
        [this](
          const std_msgs::msg::Float64::SharedPtr msg)
        {
          if (!msg)
          {
            return;
          }

          dog_to_nav_offset_sec_ =
            msg->data;

          has_dog_to_nav_offset_ =
            true;
        });


    // ========================================================================
    // STARTUP REPORTING
    // ========================================================================

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "=========================================================\n"
      "                DOG ODOM ADAPTER\n"
      "=========================================================\n"
      " Input topic           : %s\n"
      " Output topic          : %s\n"
      " Output frame          : %s\n"
      " Output child frame    : %s\n"
      " Preserve source twist : %s\n"
      " Publish TF            : NO\n"
      "=========================================================",
      input_topic_.c_str(),
      output_topic_.c_str(),
      output_frame_id_.c_str(),
      output_child_frame_id_.c_str(),
      preserve_source_twist_
        ? "yes"
        : "no");
  }


  ~DogOdomAdapter() override = default;


private:
  // ==========================================================================
  // PROCESSING / UTILITIES
  // ==========================================================================

  static double quaternionToYaw(
    const geometry_msgs::msg::Quaternion & q)
  {
    /*
     * Input:
     *     Quaternion representing the source orientation.
     *
     * Processing:
     *     Extracts the planar yaw angle from the quaternion.
     *
     * Output:
     *     Yaw angle in radians.
     *
     * Used by:
     *     odomCallback().
     *
     * Side effects:
     *     None.
     */

    const double siny =
      2.0 * (
        q.w * q.z +
        q.x * q.y);

    const double cosy =
      1.0 -
      2.0 * (
        q.y * q.y +
        q.z * q.z);

    return std::atan2(
      siny,
      cosy);
  }


  static geometry_msgs::msg::Quaternion yawToQuaternion(
    const double yaw)
  {
    /*
     * Input:
     *     Planar yaw angle in radians.
     *
     * Processing:
     *     Builds a quaternion containing yaw only.
     *
     * Output:
     *     Planar quaternion.
     *
     * Used by:
     *     odomCallback().
     *
     * Side effects:
     *     None.
     */

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


  // ==========================================================================
  // CALLBACKS
  // ==========================================================================

  void odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    /*
     * Input:
     *     Raw DOG odometry message.
     *
     * Processing:
     *     1. Validates the input message.
     *     2. Waits until a locked DOG->Nav time offset is available.
     *     3. Converts the DOG source timestamp into the navigation time domain.
     *     4. Projects the pose onto the planar navigation model.
     *     5. Preserves or clears the source twist according to configuration.
     *     6. Preserves covariance while marking removed 3D dimensions as
     *        highly uncertain.
     *
     * Output:
     *     Publishes planar navigation odometry on output_topic_.
     *
     * Used by:
     *     ROS subscription to input_topic_.
     *
     * Side effects:
     *     Publishes nav_msgs/msg/Odometry and diagnostic logs.
     */

    // ========================================================================
    // INPUT / VALIDATION
    // ========================================================================

    if (!msg)
    {
      return;
    }


    if (!has_dog_to_nav_offset_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for locked Dog->Nav time offset. "
        "Dropping dog odometry.");

      return;
    }


    // ========================================================================
    // SOURCE DATA EXTRACTION
    // ========================================================================

    const double x =
      msg->pose.pose.position.x;

    const double y =
      msg->pose.pose.position.y;

    const double yaw =
      quaternionToYaw(
        msg->pose.pose.orientation);


    // ========================================================================
    // TIME-DOMAIN CONVERSION
    // ========================================================================

    /*
     * Convert the original DOG timestamp into the navigation ROS-time domain.
     *
     * The physical measurement instant is preserved conceptually, while its
     * numerical timestamp representation is shifted by the locked DOG->Nav
     * offset supplied by navigation_time_sync_node.
     */

    const rclcpp::Time dog_source_stamp(
      msg->header.stamp);

    const rclcpp::Time dog_nav_stamp =
      dog_source_stamp
      + rclcpp::Duration::from_seconds(
          dog_to_nav_offset_sec_);


    // ========================================================================
    // RESULT OBJECT
    // ========================================================================

    nav_msgs::msg::Odometry out;


    // ------------------------------------------------------------------------
    // Header / frames
    // ------------------------------------------------------------------------

    out.header.stamp =
      dog_nav_stamp;

    out.header.frame_id =
      output_frame_id_;

    out.child_frame_id =
      output_child_frame_id_;


    // ------------------------------------------------------------------------
    // Planar pose
    // ------------------------------------------------------------------------

    out.pose.pose.position.x =
      x;

    out.pose.pose.position.y =
      y;

    out.pose.pose.position.z =
      0.0;

    out.pose.pose.orientation =
      yawToQuaternion(
        yaw);


    // ------------------------------------------------------------------------
    // Pose covariance
    // ------------------------------------------------------------------------

    out.pose.covariance =
      msg->pose.covariance;

    // Dimensions removed by the 2D projection are marked as highly uncertain.
    out.pose.covariance[14] =
      1e3;  // z

    out.pose.covariance[21] =
      1e3;  // roll

    out.pose.covariance[28] =
      1e3;  // pitch


    // ------------------------------------------------------------------------
    // Twist
    // ------------------------------------------------------------------------

    if (preserve_source_twist_)
    {
      out.twist =
        msg->twist;

      // Enforce planar twist.
      out.twist.twist.linear.z =
        0.0;

      out.twist.twist.angular.x =
        0.0;

      out.twist.twist.angular.y =
        0.0;
    }
    else
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


    // ========================================================================
    // OUTPUT
    // ========================================================================

    pub_->publish(
      out);


    // ========================================================================
    // REPORTING
    // ========================================================================

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "DOG ODOM 2D | "
      "x=%+.3f y=%+.3f yaw=%+.2f deg | "
      "source_stamp=%.9f | nav_stamp=%.9f | offset=%+.9f s",
      x,
      y,
      yaw * 180.0 / M_PI,
      dog_source_stamp.seconds(),
      dog_nav_stamp.seconds(),
      dog_to_nav_offset_sec_);
  }


  // ==========================================================================
  // PARAMETERS / CONFIGURATION
  // ==========================================================================

  std::string input_topic_;

  std::string output_topic_;

  std::string output_frame_id_;

  std::string output_child_frame_id_;

  bool preserve_source_twist_{
    true};


  // ==========================================================================
  // RUNTIME STATE
  // ==========================================================================

  double dog_to_nav_offset_sec_{
    0.0};

  bool has_dog_to_nav_offset_{
    false};


  // ==========================================================================
  // ROS INTERFACES
  // ==========================================================================

  rclcpp::Subscription<
    nav_msgs::msg::Odometry>::SharedPtr
    sub_;

  rclcpp::Publisher<
    nav_msgs::msg::Odometry>::SharedPtr
    pub_;

  rclcpp::Subscription<
    std_msgs::msg::Float64>::SharedPtr
    dog_to_nav_offset_sub_;
};


// ============================================================================
// PROGRAM ENTRY POINT
// ============================================================================

int main(
  int argc,
  char ** argv)
{
  try
  {
    // ========================================================================
    // ROS INITIALIZATION
    // ========================================================================

    rclcpp::init(
      argc,
      argv);


    // ========================================================================
    // NODE CREATION
    // ========================================================================

    auto node =
      std::make_shared<DogOdomAdapter>();


    // ========================================================================
    // EXECUTION
    // ========================================================================

    rclcpp::spin(
      node);


    // ========================================================================
    // SHUTDOWN
    // ========================================================================

    rclcpp::shutdown();

    return 0;
  }
  catch (const std::exception & e)
  {
    std::cerr
      << "[dog_odom_adapter_node] Fatal exception: "
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
      << "[dog_odom_adapter_node] Unknown fatal exception"
      << std::endl;

    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }

    return 1;
  }
}