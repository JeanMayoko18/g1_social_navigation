#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

class CloudStampRelayNode : public rclcpp::Node
{
public:
  CloudStampRelayNode()
  : Node("cloud_stamp_relay_node"),
    received_count_(0),
    published_count_(0)
  {
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic",
      "/utlidar/cloud_livox_mid360");

    output_topic_ = this->declare_parameter<std::string>(
      "output_topic",
      "/cloud_livox_now");

    log_every_n_ = this->declare_parameter<int>(
      "log_every_n",
      100);

    /*
     * Livox point clouds are typically published using sensor-data QoS:
     * BEST_EFFORT reliability and VOLATILE durability.
     */
    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(5);

    publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        output_topic_,
        qos);

    subscription_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_,
        qos,
        std::bind(
          &CloudStampRelayNode::cloud_callback,
          this,
          std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "Cloud stamp relay started");

    RCLCPP_INFO(
      this->get_logger(),
      "Input topic: %s",
      input_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "Output topic: %s",
      output_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "Only header.stamp will be replaced with the current ROS time");
  }

private:
  void cloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    ++received_count_;

    /*
     * Copy the complete PointCloud2 message:
     * - header
     * - fields
     * - dimensions
     * - point layout
     * - raw point data
     * - density information
     *
     * No geometric transformation, filtering, deskewing, axis inversion,
     * frame replacement, or point modification is applied.
     */
    auto output =
      std::make_unique<sensor_msgs::msg::PointCloud2>(*msg);

    const rclcpp::Time input_stamp(msg->header.stamp);
    const rclcpp::Time output_stamp =
      this->get_clock()->now();

    /*
     * ROS2 Foxy does not provide rclcpp::Time::to_msg() in all releases.
     * Assign the timestamp fields explicitly for full Foxy compatibility.
     */
    const int64_t output_nanoseconds =
      output_stamp.nanoseconds();

    output->header.stamp.sec =
      static_cast<int32_t>(
        output_nanoseconds / 1000000000LL);

    output->header.stamp.nanosec =
      static_cast<uint32_t>(
        output_nanoseconds % 1000000000LL);

    /*
     * Preserve the original frame_id.
     */
    output->header.frame_id =
      msg->header.frame_id;

    publisher_->publish(std::move(output));
    ++published_count_;

    if (
      log_every_n_ > 0 &&
      published_count_ %
        static_cast<uint64_t>(log_every_n_) == 0)
    {
      const double original_age =
        (output_stamp - input_stamp).seconds();

      RCLCPP_INFO(
        this->get_logger(),
        "Cloud relay | rx=%lu tx=%lu | "
        "original_age=%.3f s | frame=%s",
        static_cast<unsigned long>(received_count_),
        static_cast<unsigned long>(published_count_),
        original_age,
        msg->header.frame_id.c_str());
    }
  }

  std::string input_topic_;
  std::string output_topic_;

  int log_every_n_;

  uint64_t received_count_;
  uint64_t published_count_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  try
  {
    rclcpp::init(argc, argv);

    auto node =
      std::make_shared<CloudStampRelayNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
  }
  catch (const std::exception & e)
  {
    std::cerr
      << "[cloud_stamp_relay_node] Fatal exception: "
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
      << "[cloud_stamp_relay_node] Unknown fatal exception"
      << std::endl;

    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }

    return 1;
  }
}