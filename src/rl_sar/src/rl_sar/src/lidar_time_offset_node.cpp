#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <chrono>
#include <algorithm>
#include <iostream>
#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float64.hpp"


class LidarTimeOffsetNode : public rclcpp::Node
{
public:
  LidarTimeOffsetNode()
  : Node("lidar_time_offset_node")
  {
    // ============================================================
    // Parameters
    //
    // The launch file is the only authority that selects the
    // actual LiDAR source.
    //
    // This node does not know whether the input is Livox, Unitree,
    // or another PointCloud2 source.
    // ============================================================

    input_topic_ =
      declare_parameter<std::string>(
        "input_topic",
        "");

    output_topic_ =
      declare_parameter<std::string>(
        "output_topic",
        "");

    offset_topic_ =
      declare_parameter<std::string>(
        "offset_topic",
        "/navigation_time/lidar_to_nav_offset");

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

    if (offset_topic_.empty())
    {
      throw std::runtime_error(
        "offset_topic must not be empty.");
    }

    // ============================================================
    // QoS
    //
    // PointCloud2:
    //   SensorDataQoS, compatible with the LiDAR pipeline.
    //
    // Offset:
    //   navigation_time_sync_node publishes the locked offsets
    //   as reliable + transient_local, therefore a late-starting
    //   restamper immediately receives the latest locked value.
    // ============================================================

    const auto sensor_qos =
      rclcpp::SensorDataQoS();

    rclcpp::QoS offset_qos(
      rclcpp::KeepLast(1));

    offset_qos.reliable();
    offset_qos.transient_local();

    // ============================================================
    // Publisher
    // ============================================================

    cloud_pub_ =
      create_publisher<
        sensor_msgs::msg::PointCloud2>(
          output_topic_,
          sensor_qos);

    // ============================================================
    // Offset subscription
    // ============================================================

    offset_sub_ =
      create_subscription<
        std_msgs::msg::Float64>(
          offset_topic_,
          offset_qos,
          std::bind(
            &LidarTimeOffsetNode::offsetCallback,
            this,
            std::placeholders::_1));

    // ============================================================
    // LiDAR subscription
    // ============================================================

    cloud_sub_ =
      create_subscription<
        sensor_msgs::msg::PointCloud2>(
          input_topic_,
          sensor_qos,
          std::bind(
            &LidarTimeOffsetNode::cloudCallback,
            this,
            std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "=========================================================\n"
      "             LIDAR TIME OFFSET NODE\n"
      "=========================================================\n"
      " Input cloud  : %s\n"
      " Output cloud : %s\n"
      " Offset topic : %s\n"
      " Policy       : stamp_out = stamp_in + LiDAR->Nav offset\n"
      "=========================================================",
      input_topic_.c_str(),
      output_topic_.c_str(),
      offset_topic_.c_str());
  }


private:
  // ==============================================================
  // Offset callback
  // ==============================================================

  void offsetCallback(
    const std_msgs::msg::Float64::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    if (!std::isfinite(msg->data))
    {
      RCLCPP_WARN(
        get_logger(),
        "Ignoring non-finite LiDAR -> Nav offset.");

      return;
    }

    {
      std::lock_guard<std::mutex> lock(
        mutex_);

      lidar_to_nav_offset_sec_ =
        msg->data;

      offset_available_ =
        true;
    }

    RCLCPP_INFO(
      get_logger(),
      "LiDAR -> Nav offset received: %+.9f s",
      msg->data);
  }


  // ==============================================================
  // Cloud callback
  //
  // Convert only the timestamp domain:
  //
  //   T_nav = T_lidar + lidar_to_nav_offset
  //
  // Geometry, frame_id, fields and PointCloud2 data remain exactly
  // unchanged.
  //
  // No cloud is published before a valid locked offset has been
  // received from navigation_time_sync_node.
  // ==============================================================

void cloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!msg)
  {
    return;
  }

  double offset_sec = 0.0;
  bool offset_available = false;

  {
    std::lock_guard<std::mutex> lock(
      mutex_);

    offset_available =
      offset_available_;

    offset_sec =
      lidar_to_nav_offset_sec_;
  }

  if (!offset_available)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "LiDAR -> Nav offset not available yet; "
      "PointCloud2 publication is withheld.");

    return;
  }

  const rclcpp::Time source_stamp(
    msg->header.stamp);

  if (
    source_stamp.nanoseconds()
    <= 0)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Received PointCloud2 with invalid zero timestamp; "
      "cloud rejected.");

    return;
  }

  if (!std::isfinite(offset_sec))
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Current LiDAR -> Nav offset is non-finite; "
      "cloud rejected.");

    return;
  }

  const int64_t offset_nanoseconds =
    static_cast<int64_t>(
      std::llround(
        offset_sec * 1.0e9));

  const rclcpp::Duration offset_duration{
      std::chrono::nanoseconds(offset_nanoseconds)
  };

  const rclcpp::Time corrected_stamp =
    source_stamp
    + offset_duration;

  auto output =
    std::make_unique<
      sensor_msgs::msg::PointCloud2>(
        *msg);

  const int64_t corrected_ns =
    corrected_stamp.nanoseconds();

  output->header.stamp.sec =
    static_cast<int32_t>(
      corrected_ns / 1000000000LL);

  output->header.stamp.nanosec =
    static_cast<uint32_t>(
      corrected_ns % 1000000000LL);

  cloud_pub_->publish(
    std::move(output));
}




  // ==============================================================
  // Parameters
  // ==============================================================

  std::string input_topic_;
  std::string output_topic_;
  std::string offset_topic_;


  // ==============================================================
  // State
  // ==============================================================

  std::mutex mutex_;

  bool offset_available_{
    false};

  double lidar_to_nav_offset_sec_{
    0.0};


  // ==============================================================
  // ROS interfaces
  // ==============================================================

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    cloud_sub_;

  rclcpp::Subscription<
    std_msgs::msg::Float64>::SharedPtr
    offset_sub_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    cloud_pub_;
};


int main(
  int argc,
  char ** argv)
{
  try
  {
    rclcpp::init(
      argc,
      argv);

    const auto node =
      std::make_shared<
        LidarTimeOffsetNode>();

    rclcpp::spin(
      node);

    rclcpp::shutdown();

    return 0;
  }
  catch (const std::exception & exception)
  {
    std::cerr
      << "[lidar_time_offset_node] Fatal exception: "
      << exception.what()
      << std::endl;

    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }

    return 1;
  }
}