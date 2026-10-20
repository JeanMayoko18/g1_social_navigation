#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace std::chrono_literals;


/*
 * IMAGE INGRESS READINESS BARRIER
 *
 * FLOW
 * ====
 *
 * Remote camera
 *      |
 *      v
 * ImageIngress
 *      |
 *      +---- RGB image -------------------+
 *      |                                  |
 *      +---- aligned depth image ---------+--> WaitForImageIngress
 *      |                                  |          |
 *      +---- aligned depth CameraInfo ----+          |
 *                                                    v
 *                                      all required streams received
 *                                                    |
 *                                                    v
 *                                             process exits 0
 *
 * Responsibility
 * ==============
 *
 * This process is a startup readiness barrier.
 *
 * It waits until every requested ingress stream has delivered at least one
 * valid ROS message. It does not republish images, perform perception,
 * synchronize streams, or own camera hardware.
 *
 * Exit status:
 *   0 -> all required streams are ready
 *   1 -> timeout or invalid configuration
 */


class WaitForImageIngress : public rclcpp::Node
{
public:
  WaitForImageIngress()
  : Node("wait_for_image_ingress"),
    start_time_(std::chrono::steady_clock::now())
  {
    require_rgb_ =
      this->declare_parameter<bool>("require_rgb", false);

    require_depth_ =
      this->declare_parameter<bool>("require_depth", false);

    require_camera_info_ =
      this->declare_parameter<bool>("require_camera_info", false);

    rgb_topic_ =
      this->declare_parameter<std::string>(
      "rgb_topic",
      "/perception/ingress/color/image_raw");

    depth_topic_ =
      this->declare_parameter<std::string>(
      "depth_topic",
      "/perception/ingress/aligned_depth/image_raw");

    camera_info_topic_ =
      this->declare_parameter<std::string>(
      "camera_info_topic",
      "/perception/ingress/aligned_depth/camera_info");

    timeout_sec_ =
      this->declare_parameter<double>("timeout_sec", 15.0);

    rgb_ready_ = !require_rgb_;
    depth_ready_ = !require_depth_;
    camera_info_ready_ = !require_camera_info_;

    /*
     * Image streams use sensor-data QoS.
     *
     * BEST_EFFORT subscribers remain compatible with both BEST_EFFORT
     * camera publishers and RELIABLE ingress publishers.
     */
    const auto image_qos = rclcpp::SensorDataQoS();

    /*
    * ImageIngress republishes its local readiness streams using a stable
    * RELIABLE / VOLATILE / KEEP_LAST(5) contract.
    *
    * This barrier subscribes to ImageIngress outputs, not directly to the
    * RealSense hardware topics. Match the ingress contract explicitly so
    * startup readiness does not depend on DDS reliability negotiation.
    */
    const auto ingress_qos =
      rclcpp::QoS(rclcpp::KeepLast(5))
      .reliable()
      .durability_volatile();

    if (require_rgb_) {
      rgb_sub_ =
        this->create_subscription<sensor_msgs::msg::Image>(
        rgb_topic_,
        ingress_qos,
        std::bind(
          &WaitForImageIngress::rgb_callback,
          this,
          std::placeholders::_1));
    }

    if (require_depth_) {
      depth_sub_ =
        this->create_subscription<sensor_msgs::msg::Image>(
        depth_topic_,
        ingress_qos,
        std::bind(
          &WaitForImageIngress::depth_callback,
          this,
          std::placeholders::_1));
    }

    if (require_camera_info_) {
      camera_info_sub_ =
        this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_,
        ingress_qos,
        std::bind(
          &WaitForImageIngress::camera_info_callback,
          this,
          std::placeholders::_1));
    }

    timer_ =
      this->create_wall_timer(
      100ms,
      std::bind(
        &WaitForImageIngress::check_state,
        this));

    RCLCPP_INFO(
      this->get_logger(),
      "[IMAGE INGRESS READY] WAITING | "
      "RGB=%s | DEPTH=%s | CAMERA_INFO=%s | TIMEOUT=%.1fs",
      require_rgb_ ? "true" : "false",
      require_depth_ ? "true" : "false",
      require_camera_info_ ? "true" : "false",
      timeout_sec_);
  }

  int exit_code() const
  {
    return exit_code_;
  }

  bool completed() const
  {
    return completed_;
  }

private:
  static bool valid_image(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    return
      msg &&
      msg->width > 0 &&
      msg->height > 0 &&
      !msg->data.empty();
  }

  static bool valid_camera_info(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    return
      msg &&
      msg->width > 0 &&
      msg->height > 0 &&
      msg->k[0] > 0.0 &&
      msg->k[4] > 0.0;
  }

  void rgb_callback(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (!rgb_ready_ && valid_image(msg)) {
      rgb_ready_ = true;

      RCLCPP_INFO(
        this->get_logger(),
        "[IMAGE INGRESS READY] RGB received.");
    }
  }

  void depth_callback(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (!depth_ready_ && valid_image(msg)) {
      depth_ready_ = true;

      RCLCPP_INFO(
        this->get_logger(),
        "[IMAGE INGRESS READY] DEPTH received.");
    }
  }

  void camera_info_callback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (!camera_info_ready_ && valid_camera_info(msg)) {
      camera_info_ready_ = true;

      RCLCPP_INFO(
        this->get_logger(),
        "[IMAGE INGRESS READY] CAMERA_INFO received.");
    }
  }

  bool all_ready() const
  {
    return
      rgb_ready_ &&
      depth_ready_ &&
      camera_info_ready_;
  }

  void check_state()
  {
    if (completed_) {
      return;
    }

    if (all_ready()) {
      completed_ = true;
      exit_code_ = EXIT_SUCCESS;

      RCLCPP_INFO(
        this->get_logger(),
        "[IMAGE INGRESS READY] READY | "
        "All required ingress streams are receiving valid data.");

      rclcpp::shutdown();
      return;
    }

    const auto now =
      std::chrono::steady_clock::now();

    const double elapsed_sec =
      std::chrono::duration<double>(
      now - start_time_).count();

    if (elapsed_sec >= timeout_sec_) {
      completed_ = true;
      exit_code_ = EXIT_FAILURE;

      RCLCPP_ERROR(
        this->get_logger(),
        "[IMAGE INGRESS READY] TIMEOUT | "
        "RGB=%s | DEPTH=%s | CAMERA_INFO=%s",
        rgb_ready_ ? "true" : "false",
        depth_ready_ ? "true" : "false",
        camera_info_ready_ ? "true" : "false");

      rclcpp::shutdown();
    }
  }

  bool require_rgb_{false};
  bool require_depth_{false};
  bool require_camera_info_{false};

  bool rgb_ready_{false};
  bool depth_ready_{false};
  bool camera_info_ready_{false};

  bool completed_{false};
  int exit_code_{EXIT_FAILURE};

  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;

  double timeout_sec_{15.0};

  std::chrono::steady_clock::time_point start_time_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
    rgb_sub_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
    depth_sub_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
    camera_info_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node =
    std::make_shared<WaitForImageIngress>();

  rclcpp::spin(node);

  const int exit_code =
    node->completed()
    ? node->exit_code()
    : EXIT_FAILURE;

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  return exit_code;
}