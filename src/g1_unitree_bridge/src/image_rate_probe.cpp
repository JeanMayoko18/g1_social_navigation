#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"


class ImageRateProbe : public rclcpp::Node
{
public:
  // ==========================================================================
  // NODE CONSTRUCTOR
  //
  // Responsibility:
  //   Initialize the image-rate diagnostic node.
  //
  // Initialization sequence:
  //   1. Load runtime parameters.
  //   2. Create the ROS 2 image subscriber.
  //   3. Initialize timing state.
  //   4. Report the effective configuration.
  //
  // Input:
  //   /camera/front/image_raw
  //
  // Output:
  //   Periodic diagnostic logs describing the effective image reception rate.
  // ==========================================================================
  ImageRateProbe()
  : Node("image_rate_probe")
  {
    // ------------------------------------------------------------------------
    // CONSTRUCTOR STEP 1 - LOAD RUNTIME PARAMETERS
    // ------------------------------------------------------------------------

    input_topic_ =
      this->declare_parameter<std::string>(
      "input_topic",
      "/camera/front/image_raw");

    report_period_s_ =
      this->declare_parameter<double>(
      "report_period_s",
      5.0);


    // ------------------------------------------------------------------------
    // CONSTRUCTOR STEP 2 - CREATE SENSOR IMAGE SUBSCRIBER
    //
    // SensorDataQoS is used to match the standard camera publisher behavior.
    // ------------------------------------------------------------------------

    subscription_ =
      this->create_subscription<sensor_msgs::msg::Image>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(
        &ImageRateProbe::image_callback,
        this,
        std::placeholders::_1));


    // ------------------------------------------------------------------------
    // CONSTRUCTOR STEP 3 - INITIALIZE RUNTIME TIMING STATE
    // ------------------------------------------------------------------------

    start_time_ =
      this->now();

    last_report_time_ =
      start_time_;


    // ------------------------------------------------------------------------
    // CONSTRUCTOR STEP 4 - REPORT EFFECTIVE CONFIGURATION
    // ------------------------------------------------------------------------

    log_initialization_summary();
  }


private:
  // ==========================================================================
  // INPUT / COLLECTION - STANDARD ROS IMAGE STREAM
  // ==========================================================================

  // --------------------------------------------------------------------------
  // RECEIVE STANDARD ROS IMAGE
  //
  // Input:
  //   sensor_msgs/msg/Image
  //
  // Processing:
  //   1. Count the received frame.
  //   2. Store the latest image metadata.
  //   3. Periodically calculate the effective reception frequency.
  //
  // Output:
  //   Runtime diagnostic log through report_statistics().
  // --------------------------------------------------------------------------
  void image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg)
  {
    received_frames_++;

    latest_width_ =
      msg->width;

    latest_height_ =
      msg->height;

    latest_payload_bytes_ =
      msg->data.size();

    const auto now =
      this->now();

    const double elapsed =
      (now - last_report_time_).seconds();

    if (elapsed < report_period_s_) {
      return;
    }

    const size_t frames_since_last_report =
      received_frames_ -
      previous_received_frames_;

    const double reception_rate_hz =
      static_cast<double>(
        frames_since_last_report) /
      elapsed;

    report_statistics(
      reception_rate_hz);

    previous_received_frames_ =
      received_frames_;

    last_report_time_ =
      now;
  }


  // ==========================================================================
  // OUTPUT / REPORTING - TERMINAL DIAGNOSTICS
  // ==========================================================================

  // --------------------------------------------------------------------------
  // LOG A VISUAL SECTION HEADER
  //
  // Responsibility:
  //   Separate major runtime phases in the terminal.
  // --------------------------------------------------------------------------
  void log_section(
    const std::string &title)
  {
    const std::string separator(
      LOG_SECTION_WIDTH,
      '=');

    RCLCPP_INFO(
      this->get_logger(),
      "\n%s",
      separator.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "  %s",
      title.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "%s",
      separator.c_str());
  }


  // --------------------------------------------------------------------------
  // LOG A VISUAL SECTION FOOTER
  // --------------------------------------------------------------------------
  void log_section_footer()
  {
    const std::string separator(
      LOG_SECTION_WIDTH,
      '-');

    RCLCPP_INFO(
      this->get_logger(),
      "%s\n",
      separator.c_str());
  }


  // --------------------------------------------------------------------------
  // LOG NODE INITIALIZATION SUMMARY
  //
  // Responsibility:
  //   Report the effective image probe configuration once at startup.
  // --------------------------------------------------------------------------
  void log_initialization_summary()
  {
    log_section(
      "IMAGE RATE PROBE - INITIALIZATION");

    RCLCPP_INFO(
      this->get_logger(),
      "  Input topic          : %s",
      input_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "  QoS profile          : SensorDataQoS / BEST_EFFORT");

    RCLCPP_INFO(
      this->get_logger(),
      "  Report period        : %.1f s",
      report_period_s_);

    RCLCPP_INFO(
      this->get_logger(),
      "  Probe state          : READY");

    log_section_footer();
  }


  // --------------------------------------------------------------------------
  // REPORT IMAGE RECEPTION STATISTICS
  //
  // Input:
  //   reception_rate_hz - measured rate over the last reporting interval.
  //
  // Reports:
  //   - effective reception rate
  //   - cumulative received frames
  //   - latest image resolution
  //   - latest image payload size
  // --------------------------------------------------------------------------
  void report_statistics(
    const double reception_rate_hz)
  {
    const double payload_mb =
      static_cast<double>(
        latest_payload_bytes_) /
      (1024.0 * 1024.0);

    RCLCPP_INFO(
      this->get_logger(),
      "[IMAGE PROBE] RATE=%.2f Hz | "
      "TOTAL=%zu | "
      "RES=%ux%u | "
      "PAYLOAD=%.2f MB",
      reception_rate_hz,
      received_frames_,
      latest_width_,
      latest_height_,
      payload_mb);
  }


private:
  // ==========================================================================
  // INTERNAL NODE STATE
  // ==========================================================================

  // --------------------------------------------------------------------------
  // LOGGING CONFIGURATION
  // --------------------------------------------------------------------------

  static constexpr std::size_t
  LOG_SECTION_WIDTH = 82;


  // --------------------------------------------------------------------------
  // RUNTIME CONFIGURATION
  // --------------------------------------------------------------------------

  std::string input_topic_;

  double report_period_s_{5.0};


  // --------------------------------------------------------------------------
  // ROS 2 INTERFACES
  // --------------------------------------------------------------------------

  rclcpp::Subscription<
    sensor_msgs::msg::Image>::SharedPtr
  subscription_;


  // --------------------------------------------------------------------------
  // IMAGE METADATA RESULT STATE
  // --------------------------------------------------------------------------

  uint32_t latest_width_{0};
  uint32_t latest_height_{0};

  size_t latest_payload_bytes_{0};


  // --------------------------------------------------------------------------
  // RUNTIME DIAGNOSTICS AND PERFORMANCE COUNTERS
  // --------------------------------------------------------------------------

  size_t received_frames_{0};
  size_t previous_received_frames_{0};

  rclcpp::Time start_time_;
  rclcpp::Time last_report_time_;
};


// ============================================================================
// NODE ORCHESTRATION / PROCESS ENTRY POINT
//
// Responsibility:
//   Initialize ROS 2, instantiate ImageRateProbe and process image callbacks.
//
// Execution model:
//   Single-threaded executor.
//
// The probe performs lightweight diagnostics only, so a MultiThreadedExecutor
// is not required.
// ============================================================================

int main(
  int argc,
  char **argv)
{
  rclcpp::init(
    argc,
    argv);

  const auto node =
    std::make_shared<ImageRateProbe>();

  RCLCPP_INFO(
    node->get_logger(),
    "[EXECUTOR] ImageRateProbe started with SingleThreadedExecutor.");

  rclcpp::spin(
    node);

  rclcpp::shutdown();

  return 0;
}