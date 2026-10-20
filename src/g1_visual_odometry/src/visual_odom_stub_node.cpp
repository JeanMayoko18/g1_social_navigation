/**
 * File: visual_odom_stub_node.cpp
 *
 * Role:
 * -----
 * This node is a minimal visual odometry placeholder for the G1 perception stack.
 *
 * It does not compute real visual odometry yet. Instead, it provides the ROS 2
 * interfaces and frame conventions that will later be reused by a real
 * monocular / visual-inertial odometry pipeline.
 *
 * How it works:
 * -------------
 * 1. Subscribes to:
 *      - /camera/image_raw
 *      - /camera/camera_info
 *      - /dog_imu_raw
 *
 * 2. Tracks whether the required sensor streams are alive.
 *
 * 3. Publishes a placeholder odometry message on:
 *      - /odom/visual
 *
 * 4. Optionally publishes the TF transform:
 *      - odom -> base_footprint
 *
 * Current behavior:
 * -----------------
 * - The node keeps a zero-motion pose by default.
 * - It is meant to validate topic connectivity, timing, TF conventions,
 *   and Nav2 integration before a real VO/VIO backend is inserted.
 *
 * Future extension:
 * -----------------
 * Replace the internal placeholder pose update with:
 *   - ORB-SLAM3
 *   - VINS-Fusion
 *   - RTAB-Map visual odometry
 *   - custom vision / VIO pipeline
 */

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2_ros/transform_broadcaster.h"

class VisualOdomStubNode : public rclcpp::Node
{
public:
  VisualOdomStubNode()
  : Node("visual_odom_stub_node"),
    got_image_(false),
    got_camera_info_(false),
    got_imu_(false),
    x_(0.0),
    y_(0.0),
    yaw_(0.0)
  {
    // ------------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------------
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/dog_imu_raw");

    odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/odom/visual");

    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_footprint");

    publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 20.0);

    // ------------------------------------------------------------------------
    // QoS
    // ------------------------------------------------------------------------
    // Sensor topics are usually best handled with sensor data QoS.
    auto sensor_qos = rclcpp::SensorDataQoS();

    // ------------------------------------------------------------------------
    // Subscribers
    // ------------------------------------------------------------------------
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      sensor_qos,
      std::bind(&VisualOdomStubNode::image_callback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      sensor_qos,
      std::bind(&VisualOdomStubNode::camera_info_callback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      sensor_qos,
      std::bind(&VisualOdomStubNode::imu_callback, this, std::placeholders::_1));

    // ------------------------------------------------------------------------
    // Publisher
    // ------------------------------------------------------------------------
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);

    // ------------------------------------------------------------------------
    // TF broadcaster
    // ------------------------------------------------------------------------
    if (publish_tf_) {
      tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    }

    // ------------------------------------------------------------------------
    // Timer
    // ------------------------------------------------------------------------
    auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&VisualOdomStubNode::publish_outputs, this));

    RCLCPP_INFO(this->get_logger(), "VisualOdomStubNode started.");
    RCLCPP_INFO(this->get_logger(), "  image_topic      : %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  camera_info_topic: %s", camera_info_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  imu_topic        : %s", imu_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  odom_topic       : %s", odom_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  frames           : %s -> %s", odom_frame_.c_str(), base_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  publish_tf       : %s", publish_tf_ ? "true" : "false");
  }

private:
  // --------------------------------------------------------------------------
  // Callbacks
  // --------------------------------------------------------------------------
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    (void)msg;
    got_image_ = true;
  }

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    (void)msg;
    got_camera_info_ = true;
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    (void)msg;
    got_imu_ = true;

    // Placeholder behavior:
    // For now we do not integrate motion.
    // Later, this callback can feed a real VIO backend.
    last_imu_stamp_ = msg->header.stamp;
  }

  // --------------------------------------------------------------------------
  // Periodic output publication
  // --------------------------------------------------------------------------
  void publish_outputs()
  {
    // Only publish when all required sensor streams are available.
    if (!got_image_ || !got_camera_info_ || !got_imu_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "Waiting for all inputs: image=%s camera_info=%s imu=%s",
        got_image_ ? "yes" : "no",
        got_camera_info_ ? "yes" : "no",
        got_imu_ ? "yes" : "no");
      return;
    }

    // ------------------------------------------------------------------------
    // Publish placeholder odometry
    // ------------------------------------------------------------------------
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = this->now();
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_;

    odom_msg.pose.pose.position.x = x_;
    odom_msg.pose.pose.position.y = y_;
    odom_msg.pose.pose.position.z = 0.0;

    // Zero-yaw quaternion
    odom_msg.pose.pose.orientation.x = 0.0;
    odom_msg.pose.pose.orientation.y = 0.0;
    odom_msg.pose.pose.orientation.z = 0.0;
    odom_msg.pose.pose.orientation.w = 1.0;

    // Conservative placeholder covariance
    odom_msg.pose.covariance[0] = 0.05;   // x
    odom_msg.pose.covariance[7] = 0.05;   // y
    odom_msg.pose.covariance[35] = 0.10;  // yaw

    odom_msg.twist.twist.linear.x = 0.0;
    odom_msg.twist.twist.linear.y = 0.0;
    odom_msg.twist.twist.angular.z = 0.0;

    odom_msg.twist.covariance[0] = 0.10;
    odom_msg.twist.covariance[7] = 0.10;
    odom_msg.twist.covariance[35] = 0.20;

    odom_pub_->publish(odom_msg);

    // ------------------------------------------------------------------------
    // Publish TF: odom -> base_footprint
    // ------------------------------------------------------------------------
    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = odom_msg.header.stamp;
      tf_msg.header.frame_id = odom_frame_;
      tf_msg.child_frame_id = base_frame_;

      tf_msg.transform.translation.x = x_;
      tf_msg.transform.translation.y = y_;
      tf_msg.transform.translation.z = 0.0;

      tf_msg.transform.rotation.x = 0.0;
      tf_msg.transform.rotation.y = 0.0;
      tf_msg.transform.rotation.z = 0.0;
      tf_msg.transform.rotation.w = 1.0;

      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

private:
  // --------------------------------------------------------------------------
  // Parameters
  // --------------------------------------------------------------------------
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  bool publish_tf_;
  double publish_rate_hz_;

  // --------------------------------------------------------------------------
  // ROS interfaces
  // --------------------------------------------------------------------------
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  // --------------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------------
  bool got_image_;
  bool got_camera_info_;
  bool got_imu_;

  double x_;
  double y_;
  double yaw_;

  builtin_interfaces::msg::Time last_imu_stamp_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisualOdomStubNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}