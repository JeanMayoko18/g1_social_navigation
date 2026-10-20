/**
 * File: visual_odom_monitor_node.cpp
 *
 * Role:
 * -----
 * This node monitors the health of the visual-inertial input pipeline
 * for future visual odometry / visual-inertial odometry integration.
 *
 * It subscribes to:
 *   - /camera/image_raw
 *   - /camera/camera_info
 *   - /dog_imu_raw
 *
 * and periodically reports:
 *   - observed image rate
 *   - observed camera_info rate
 *   - observed IMU rate
 *   - time since last message for each stream
 *   - whether the pipeline appears usable for a future VO/VIO backend
 *
 * How it works:
 * -------------
 * 1. Each subscriber records:
 *      - message count
 *      - timestamp of last received message
 *      - arrival intervals
 *
 * 2. A timer periodically computes approximate rates from arrival intervals.
 *
 * 3. The node prints a diagnostic summary indicating whether:
 *      - image stream is alive
 *      - camera_info stream is alive
 *      - IMU stream is alive
 *      - the complete pipeline is ready for VO/VIO experiments
 *
 * Current purpose:
 * ----------------
 * This node is intended as a diagnostic tool before integrating a real
 * visual odometry or visual-inertial odometry backend.
 */

#include <deque>
#include <memory>
#include <numeric>
#include <string>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"

class VisualOdomMonitorNode : public rclcpp::Node
{
public:
  VisualOdomMonitorNode()
  : Node("visual_odom_monitor_node")
  {
    // ------------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------------
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/dog_imu_raw");

    report_period_s_ = this->declare_parameter<double>("report_period_s", 2.0);
    stale_timeout_s_ = this->declare_parameter<double>("stale_timeout_s", 1.0);
    max_samples_ = this->declare_parameter<int>("max_samples", 100);

    // ------------------------------------------------------------------------
    // QoS
    // ------------------------------------------------------------------------
    auto sensor_qos = rclcpp::SensorDataQoS();

    // ------------------------------------------------------------------------
    // Subscribers
    // ------------------------------------------------------------------------
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      sensor_qos,
      std::bind(&VisualOdomMonitorNode::image_callback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      sensor_qos,
      std::bind(&VisualOdomMonitorNode::camera_info_callback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      sensor_qos,
      std::bind(&VisualOdomMonitorNode::imu_callback, this, std::placeholders::_1));

    // ------------------------------------------------------------------------
    // Timer
    // ------------------------------------------------------------------------
    auto period = std::chrono::duration<double>(std::max(0.1, report_period_s_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&VisualOdomMonitorNode::report_status, this));

    RCLCPP_INFO(this->get_logger(), "VisualOdomMonitorNode started.");
    RCLCPP_INFO(this->get_logger(), "  image_topic      : %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  camera_info_topic: %s", camera_info_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  imu_topic        : %s", imu_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  report_period_s  : %.2f", report_period_s_);
    RCLCPP_INFO(this->get_logger(), "  stale_timeout_s  : %.2f", stale_timeout_s_);
    RCLCPP_INFO(this->get_logger(), "  max_samples      : %d", max_samples_);
  }

private:
  // ==========================================================================
  // Callbacks
  // ==========================================================================
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    (void)msg;
    update_stream_state(image_state_);
  }

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    (void)msg;
    update_stream_state(camera_info_state_);
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    (void)msg;
    update_stream_state(imu_state_);
  }

  // ==========================================================================
  // Stream state handling
  // ==========================================================================
  struct StreamState
  {
    size_t msg_count{0};
    bool has_msg{false};
    rclcpp::Time last_arrival_time{0, 0, RCL_ROS_TIME};
    std::deque<double> dt_samples;
  };

  void update_stream_state(StreamState &state)
  {
    const auto now = this->now();

    if (state.has_msg) {
      const double dt = (now - state.last_arrival_time).seconds();

      // Keep only positive intervals and ignore pathological zeros/negatives.
      if (dt > 0.0) {
        state.dt_samples.push_back(dt);
        while (static_cast<int>(state.dt_samples.size()) > max_samples_) {
          state.dt_samples.pop_front();
        }
      }
    }

    state.last_arrival_time = now;
    state.has_msg = true;
    state.msg_count++;
  }

  double estimate_rate_hz(const StreamState &state) const
  {
    if (state.dt_samples.empty()) {
      return 0.0;
    }

    const double sum =
      std::accumulate(state.dt_samples.begin(), state.dt_samples.end(), 0.0);

    if (sum <= 0.0) {
      return 0.0;
    }

    const double mean_dt = sum / static_cast<double>(state.dt_samples.size());
    if (mean_dt <= 0.0) {
      return 0.0;
    }

    return 1.0 / mean_dt;
  }

  double age_seconds(const StreamState &state) const
  {
    if (!state.has_msg) {
      return 1e9;
    }
    return (this->now() - state.last_arrival_time).seconds();
  }

  bool is_alive(const StreamState &state) const
  {
    return state.has_msg && age_seconds(state) <= stale_timeout_s_;
  }

  // ==========================================================================
  // Reporting
  // ==========================================================================
  void report_status()
  {
    const double image_rate = estimate_rate_hz(image_state_);
    const double camera_info_rate = estimate_rate_hz(camera_info_state_);
    const double imu_rate = estimate_rate_hz(imu_state_);

    const double image_age = age_seconds(image_state_);
    const double camera_info_age = age_seconds(camera_info_state_);
    const double imu_age = age_seconds(imu_state_);

    const bool image_alive = is_alive(image_state_);
    const bool camera_info_alive = is_alive(camera_info_state_);
    const bool imu_alive = is_alive(imu_state_);

    // Very simple readiness heuristic for early VO/VIO experiments.
    const bool ready_for_vo =
      image_alive && camera_info_alive && image_rate >= 5.0;

    const bool ready_for_vio =
      ready_for_vo && imu_alive && imu_rate >= 20.0;

    RCLCPP_INFO(
      this->get_logger(),
      "[Pipeline status] "
      "image: alive=%s rate=%.2fHz age=%.3fs count=%zu | "
      "camera_info: alive=%s rate=%.2fHz age=%.3fs count=%zu | "
      "imu: alive=%s rate=%.2fHz age=%.3fs count=%zu | "
      "ready_for_vo=%s ready_for_vio=%s",
      image_alive ? "yes" : "no",
      image_rate,
      image_age,
      image_state_.msg_count,
      camera_info_alive ? "yes" : "no",
      camera_info_rate,
      camera_info_age,
      camera_info_state_.msg_count,
      imu_alive ? "yes" : "no",
      imu_rate,
      imu_age,
      imu_state_.msg_count,
      ready_for_vo ? "yes" : "no",
      ready_for_vio ? "yes" : "no");

    // Additional warning if image rate is the main bottleneck.
    if (image_alive && image_rate < 5.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Image stream is alive but slow (%.2f Hz). This may be insufficient for stable visual odometry.",
        image_rate);
    }

    if (!image_alive || !camera_info_alive || !imu_alive) {
      RCLCPP_WARN(
        this->get_logger(),
        "One or more required streams are stale or missing.");
    }
  }

private:
  // --------------------------------------------------------------------------
  // Parameters
  // --------------------------------------------------------------------------
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string imu_topic_;
  double report_period_s_;
  double stale_timeout_s_;
  int max_samples_;

  // --------------------------------------------------------------------------
  // ROS interfaces
  // --------------------------------------------------------------------------
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // --------------------------------------------------------------------------
  // Internal state
  // --------------------------------------------------------------------------
  StreamState image_state_;
  StreamState camera_info_state_;
  StreamState imu_state_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisualOdomMonitorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}