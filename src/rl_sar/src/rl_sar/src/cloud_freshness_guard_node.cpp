#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

class CloudFreshnessGuardNode : public rclcpp::Node
{
public:
  CloudFreshnessGuardNode()
  : Node("cloud_freshness_guard_node")
  {
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic",
      "/navigation_time/lidar_cloud");

    output_topic_ = this->declare_parameter<std::string>(
      "output_topic",
      "/navigation_time/lidar_cloud_fresh");

    warning_age_sec_ = this->declare_parameter<double>(
      "warning_age_sec",
      0.50);

    drop_age_sec_ = this->declare_parameter<double>(
      "drop_age_sec",
      1.00);

    future_tolerance_sec_ = this->declare_parameter<double>(
      "future_tolerance_sec",
      0.10);

    if (warning_age_sec_ <= 0.0) {
      throw std::runtime_error(
        "warning_age_sec must be > 0");
    }

    if (drop_age_sec_ <= warning_age_sec_) {
      throw std::runtime_error(
        "drop_age_sec must be > warning_age_sec");
    }

    if (future_tolerance_sec_ < 0.0) {
      throw std::runtime_error(
        "future_tolerance_sec must be >= 0");
    }

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
          &CloudFreshnessGuardNode::cloudCallback,
          this,
          std::placeholders::_1));

    RCLCPP_INFO(
      this->get_logger(),
      "[CLOUD FRESHNESS] "
      "input=%s | output=%s | warn=%.3fs | drop=%.3fs | future=%.3fs",
      input_topic_.c_str(),
      output_topic_.c_str(),
      warning_age_sec_,
      drop_age_sec_,
      future_tolerance_sec_);
  }

private:
  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    ++received_;

    if (!msg) {
      return;
    }

    const int64_t stamp_ns =
      static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
      static_cast<int64_t>(msg->header.stamp.nanosec);

    if (stamp_ns <= 0) {
      ++dropped_invalid_;

      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "[CLOUD FRESHNESS] DROP_INVALID_STAMP");

      return;
    }

    const int64_t now_ns =
      this->get_clock()->now().nanoseconds();

    const double age_sec =
      static_cast<double>(
        now_ns - stamp_ns) / 1.0e9;

    if (age_sec < -future_tolerance_sec_) {
      ++dropped_future_;

      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "[CLOUD FRESHNESS] DROP_FUTURE | age=%.3fs",
        age_sec);

      return;
    }

    if (age_sec > drop_age_sec_) {
      ++dropped_stale_;

      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "[CLOUD FRESHNESS] DROP_STALE | "
        "age=%.3fs | drop_limit=%.3fs | "
        "received=%lu | passed=%lu | dropped_stale=%lu",
        age_sec,
        drop_age_sec_,
        static_cast<unsigned long>(received_),
        static_cast<unsigned long>(passed_),
        static_cast<unsigned long>(dropped_stale_));

      return;
    }

    if (age_sec > warning_age_sec_) {
      ++passed_warning_;

      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "[CLOUD FRESHNESS] PASS_LATE | "
        "age=%.3fs | warning=%.3fs | drop=%.3fs",
        age_sec,
        warning_age_sec_,
        drop_age_sec_);
    }

    ++passed_;

    /*
     * Critical rule:
     *
     * The timestamp is NEVER modified here.
     *
     * This node is not a synchronizer and does not make old data look new.
     * It only rejects measurements that are already unusably stale.
     */
    publisher_->publish(*msg);
  }

  std::string input_topic_;
  std::string output_topic_;

  double warning_age_sec_{0.50};
  double drop_age_sec_{1.00};
  double future_tolerance_sec_{0.10};

  uint64_t received_{0};
  uint64_t passed_{0};
  uint64_t passed_warning_{0};
  uint64_t dropped_stale_{0};
  uint64_t dropped_future_{0};
  uint64_t dropped_invalid_{0};

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

  rclcpp::Publisher<
    sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};


int main(
  int argc,
  char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<CloudFreshnessGuardNode>());

  rclcpp::shutdown();

  return 0;
}