/**
 * File: visual_inertial_buffer_node.cpp
 *
 * Role:
 * -----
 * This node prepares synchronized visual-inertial data packets for a future
 * visual odometry / visual-inertial odometry backend.
 *
 * It subscribes to:
 *   - /camera/image_raw
 *   - /camera/camera_info
 *   - /dog_imu_raw
 *
 * and maintains:
 *   - a rolling IMU buffer
 *   - the latest camera intrinsics
 *
 * For every incoming image, the node extracts the IMU samples contained in a
 * configurable temporal window around the image timestamp and reports whether
 * the packet is usable for a future VO/VIO algorithm.
 *
 * How it works:
 * -------------
 * 1. IMU messages are continuously stored in a bounded deque.
 *
 * 2. CameraInfo messages are cached so that intrinsics are always available.
 *
 * 3. For each image:
 *      - the node reads the image timestamp
 *      - selects IMU samples in [image_time - pre_window, image_time + post_window]
 *      - checks whether enough IMU data is available
 *      - logs a synchronized packet summary
 *
 * 4. This node does not yet estimate pose.
 *    It is a preparation layer for integrating a real backend such as:
 *      - ORB-SLAM3
 *      - VINS-Fusion
 *      - RTAB-Map VO/VIO
 *      - a custom VIO frontend
 */

#include <deque>
#include <memory>
#include <string>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"

class VisualInertialBufferNode : public rclcpp::Node
{
public:
  VisualInertialBufferNode()
  : Node("visual_inertial_buffer_node")
  {
    // ------------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------------
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/dog_imu_raw");

    imu_buffer_duration_s_ = this->declare_parameter<double>("imu_buffer_duration_s", 5.0);
    imu_pre_window_s_ = this->declare_parameter<double>("imu_pre_window_s", 0.10);
    imu_post_window_s_ = this->declare_parameter<double>("imu_post_window_s", 0.02);
    min_imu_samples_per_image_ = this->declare_parameter<int>("min_imu_samples_per_image", 3);

    // ------------------------------------------------------------------------
    // Subscribers
    // ------------------------------------------------------------------------
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&VisualInertialBufferNode::image_callback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&VisualInertialBufferNode::camera_info_callback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&VisualInertialBufferNode::imu_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "VisualInertialBufferNode started.");
    RCLCPP_INFO(this->get_logger(), "  image_topic             : %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  camera_info_topic       : %s", camera_info_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  imu_topic               : %s", imu_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  imu_buffer_duration_s   : %.3f", imu_buffer_duration_s_);
    RCLCPP_INFO(this->get_logger(), "  imu_pre_window_s        : %.3f", imu_pre_window_s_);
    RCLCPP_INFO(this->get_logger(), "  imu_post_window_s       : %.3f", imu_post_window_s_);
    RCLCPP_INFO(this->get_logger(), "  min_imu_samples/image   : %d", min_imu_samples_per_image_);
  }

private:
  struct ImuStamped
  {
    rclcpp::Time stamp;
    sensor_msgs::msg::Imu msg;
  };

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    latest_camera_info_ = *msg;
    has_camera_info_ = true;
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const rclcpp::Time t(msg->header.stamp);

    ImuStamped sample;
    sample.stamp = t;
    sample.msg = *msg;

    imu_buffer_.push_back(sample);
    trim_imu_buffer();

    imu_count_++;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    image_count_++;

    if (!has_camera_info_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "No camera_info received yet. Cannot build a valid visual-inertial packet.");
      return;
    }

    if (imu_buffer_.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "IMU buffer is empty. Cannot build a visual-inertial packet.");
      return;
    }

    const rclcpp::Time image_time(msg->header.stamp);

    const rclcpp::Time t_min = image_time - rclcpp::Duration::from_seconds(imu_pre_window_s_);
    const rclcpp::Time t_max = image_time + rclcpp::Duration::from_seconds(imu_post_window_s_);

    int imu_samples_in_window = 0;

    for (const auto &sample : imu_buffer_) {
      if (sample.stamp >= t_min && sample.stamp <= t_max) {
        imu_samples_in_window++;
      }
    }

    const bool packet_ok = imu_samples_in_window >= min_imu_samples_per_image_;
    const double image_age = (this->now() - image_time).seconds();

    RCLCPP_INFO(
      this->get_logger(),
      "[VI packet] image_count=%zu imu_count=%zu img_stamp=%.6f img_age=%.3fs "
      "imu_window=[%.3fms, +%.3fms] imu_samples=%d packet_ok=%s",
      image_count_,
      imu_count_,
      image_time.seconds(),
      image_age,
      imu_pre_window_s_ * 1000.0,
      imu_post_window_s_ * 1000.0,
      imu_samples_in_window,
      packet_ok ? "yes" : "no");

    if (!packet_ok) {
      RCLCPP_WARN(
        this->get_logger(),
        "Not enough IMU samples around image timestamp. Expected at least %d, got %d.",
        min_imu_samples_per_image_,
        imu_samples_in_window);
    }
  }

  void trim_imu_buffer()
  {
    if (imu_buffer_.empty()) {
      return;
    }

    const rclcpp::Time newest = imu_buffer_.back().stamp;
    const rclcpp::Time oldest_allowed =
      newest - rclcpp::Duration::from_seconds(imu_buffer_duration_s_);

    while (!imu_buffer_.empty() && imu_buffer_.front().stamp < oldest_allowed) {
      imu_buffer_.pop_front();
    }
  }

private:
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string imu_topic_;

  double imu_buffer_duration_s_;
  double imu_pre_window_s_;
  double imu_post_window_s_;
  int min_imu_samples_per_image_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  std::deque<ImuStamped> imu_buffer_;
  sensor_msgs::msg::CameraInfo latest_camera_info_;
  bool has_camera_info_{false};

  size_t image_count_{0};
  size_t imu_count_{0};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisualInertialBufferNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}