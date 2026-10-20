#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2_ros/transform_broadcaster.h"


class G1NavOdomAdapter : public rclcpp::Node
{
public:
  G1NavOdomAdapter()
  : Node("g1_nav_odom_adapter_node")
  {
    // ============================================================
    // 1. Parameters
    // ============================================================

    input_topic_ =
      declare_parameter<std::string>(
        "input_topic",
        "/odom_direct");

    output_topic_ =
      declare_parameter<std::string>(
        "output_topic",
        "/odom");

    odom_frame_ =
      declare_parameter<std::string>(
        "odom_frame",
        "odom");

    base_frame_ =
      declare_parameter<std::string>(
        "base_frame",
        "base_footprint");

    publish_rate_hz_ =
      declare_parameter<double>(
        "publish_rate",
        50.0);

    publish_tf_ =
      declare_parameter<bool>(
        "publish_tf",
        true);

    // Only used for diagnostics.
    //
    // IMPORTANT:
    // A stale state no longer stops publication.
    max_state_age_sec_ =
      declare_parameter<double>(
        "max_state_age_sec",
        0.25);


    // ============================================================
    // 2. Final Nav2 odometry publisher
    // ============================================================

    rclcpp::QoS odom_qos(
      rclcpp::KeepLast(10));

    odom_qos.reliable();
    odom_qos.durability_volatile();

    odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(
        output_topic_,
        odom_qos);


    // ============================================================
    // 3. TF broadcaster
    //
    // This node is the unique owner of:
    //
    //     odom -> base_footprint
    // ============================================================

    if (publish_tf_)
    {
      tf_broadcaster_ =
        std::make_shared<tf2_ros::TransformBroadcaster>(
          this);
    }


    // ============================================================
    // 4. Subscribe to high-rate /odom_direct
    //
    // The callback only stores the newest state.
    //
    // Publication is completely independent and happens from the
    // 50 Hz timer below.
    // ============================================================

    rclcpp::QoS input_qos(
      rclcpp::KeepLast(50));

    input_qos.reliable();
    input_qos.durability_volatile();

    odom_sub_ =
      create_subscription<nav_msgs::msg::Odometry>(
        input_topic_,
        input_qos,
        std::bind(
          &G1NavOdomAdapter::odomCallback,
          this,
          std::placeholders::_1));


    // ============================================================
    // 5. Independent fixed-rate publication
    //
    // This reproduces the important behavior of the reference
    // implementation:
    //
    //   latest physical state
    //           +
    //   independent ROS timer
    //           ↓
    //   /odom + TF continuously at 50 Hz
    //
    // The timer never stops merely because one new input sample
    // arrived late.
    // ============================================================

    const double safe_rate =
      std::max(
        publish_rate_hz_,
        1.0);

    publish_timer_ =
      create_wall_timer(
        std::chrono::duration<double>(
          1.0 / safe_rate),
        std::bind(
          &G1NavOdomAdapter::publishNavigationState,
          this));


    // ============================================================
    // 6. Startup information
    // ============================================================

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "=========================================================\n"
      "             G1 NAV ODOM ADAPTER\n"
      "=========================================================\n"
      " Input topic        : %s\n"
      " Output topic       : %s\n"
      " Odom frame         : %s\n"
      " Base frame         : %s\n"
      " Publish rate       : %.2f Hz\n"
      " Publish TF         : %s\n"
      " Stale warning      : %.3f s\n"
      "\n"
      " Publication policy:\n"
      "   - keep latest /odom_direct state\n"
      "   - publish continuously at fixed rate\n"
      "   - stamp final /odom with ROS now()\n"
      "   - stamp TF with the exact same ROS now()\n"
      "   - never stop TF because of a temporary input gap\n"
      "=========================================================",
      input_topic_.c_str(),
      output_topic_.c_str(),
      odom_frame_.c_str(),
      base_frame_.c_str(),
      publish_rate_hz_,
      publish_tf_ ? "yes" : "no",
      max_state_age_sec_);
  }


private:

  // ==============================================================
  // Receive latest physical state
  // ==============================================================

  void odomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    const rclcpp::Time reception_time =
      get_clock()->now();

    {
      std::lock_guard<std::mutex> lock(
        state_mutex_);

      latest_state_ =
        *msg;

      latest_state_receive_time_ =
        reception_time;

      have_latest_state_ =
        true;
    }
  }


  // ==============================================================
  // Fixed-rate Nav2 publication
  //
  // This function deliberately decouples navigation TF publication
  // from the arrival timing of /odom_direct.
  //
  // As long as at least one valid state has been received:
  //
  //   - the latest pose remains available;
  //   - /odom continues at the requested rate;
  //   - odom -> base_footprint continues at the requested rate.
  //
  // This matches the logic used by the reference implementation.
  // ==============================================================

  void publishNavigationState()
  {
    nav_msgs::msg::Odometry out;

    rclcpp::Time state_receive_time(
      0,
      0,
      get_clock()->get_clock_type());


    // ------------------------------------------------------------
    // Copy latest complete state atomically
    // ------------------------------------------------------------

    {
      std::lock_guard<std::mutex> lock(
        state_mutex_);

      if (!have_latest_state_)
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for first message on %s.",
          input_topic_.c_str());

        return;
      }

      out =
        latest_state_;

      state_receive_time =
        latest_state_receive_time_;
    }


    // ============================================================
    // Current ROS navigation time
    //
    // /odommodestate has no usable source timestamp.
    //
    // Therefore the final navigation-facing state is stamped at
    // publication time, exactly like the reference implementation.
    // ============================================================

    const rclcpp::Time now =
      get_clock()->now();


    const double state_age_sec =
      (
        now
        -
        state_receive_time
      ).seconds();


    // ============================================================
    // Diagnostic only
    //
    // IMPORTANT:
    //
    // We DO NOT return when the source becomes temporarily stale.
    //
    // The latest pose continues to be published so that TF remains
    // continuous for Nav2.
    //
    // Since we do not integrate the twist here, the pose itself does
    // not artificially move during the freeze.
    // ============================================================

    if (
      state_age_sec
      > max_state_age_sec_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "/odom_direct input gap | "
        "reception_age=%.3f s | "
        "holding latest pose while keeping Nav2 TF alive.",
        state_age_sec);
    }


    // ============================================================
    // Final navigation message
    //
    // The physical state comes from the latest /odom_direct message.
    //
    // The final navigation timestamp is the publication time.
    // ============================================================

    out.header.stamp =
      now;

    out.header.frame_id =
      odom_frame_;

    out.child_frame_id =
      base_frame_;


    // ============================================================
    // Publish /odom
    // ============================================================

    odom_pub_->publish(
      out);


    // ============================================================
    // Publish matching TF
    //
    // Exactly the same timestamp and pose as /odom.
    // ============================================================

    if (
      publish_tf_
      &&
      tf_broadcaster_)
    {
      geometry_msgs::msg::TransformStamped tf_msg;

      tf_msg.header.stamp =
        now;

      tf_msg.header.frame_id =
        odom_frame_;

      tf_msg.child_frame_id =
        base_frame_;

      tf_msg.transform.translation.x =
        out.pose.pose.position.x;

      tf_msg.transform.translation.y =
        out.pose.pose.position.y;

      tf_msg.transform.translation.z =
        out.pose.pose.position.z;

      tf_msg.transform.rotation =
        out.pose.pose.orientation;

      tf_broadcaster_->sendTransform(
        tf_msg);
    }


    // ============================================================
    // Diagnostics
    // ============================================================

    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "NAV ODOM | "
      "x=%+.3f y=%+.3f | "
      "vx=%+.3f vy=%+.3f wz=%+.3f | "
      "stamp=%.9f | "
      "source_rx_age=%.4f s%s",
      out.pose.pose.position.x,
      out.pose.pose.position.y,
      out.twist.twist.linear.x,
      out.twist.twist.linear.y,
      out.twist.twist.angular.z,
      now.seconds(),
      state_age_sec,
      state_age_sec > max_state_age_sec_
        ? " | HOLD"
        : "");
  }


  // ==============================================================
  // Parameters
  // ==============================================================

  std::string input_topic_;

  std::string output_topic_;

  std::string odom_frame_;

  std::string base_frame_;


  double publish_rate_hz_{
    50.0};


  bool publish_tf_{
    true};


  double max_state_age_sec_{
    0.25};


  // ==============================================================
  // Latest source state
  // ==============================================================

  std::mutex state_mutex_;


  nav_msgs::msg::Odometry
    latest_state_;


  rclcpp::Time latest_state_receive_time_{
    0,
    0,
    RCL_ROS_TIME};


  bool have_latest_state_{
    false};


  // ==============================================================
  // ROS interfaces
  // ==============================================================

  rclcpp::Subscription<
    nav_msgs::msg::Odometry>::SharedPtr
    odom_sub_;


  rclcpp::Publisher<
    nav_msgs::msg::Odometry>::SharedPtr
    odom_pub_;


  std::shared_ptr<
    tf2_ros::TransformBroadcaster>
    tf_broadcaster_;


  rclcpp::TimerBase::SharedPtr
    publish_timer_;
};


// ============================================================================
// Main
// ============================================================================

int main(
  int argc,
  char ** argv)
{
  try
  {
    rclcpp::init(
      argc,
      argv);

    auto node =
      std::make_shared<
        G1NavOdomAdapter>();

    rclcpp::spin(
      node);

    rclcpp::shutdown();

    return 0;
  }
  catch (
    const std::exception & e)
  {
    std::cerr
      << "[g1_nav_odom_adapter_node] Fatal exception: "
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
      << "[g1_nav_odom_adapter_node] "
      << "Unknown fatal exception"
      << std::endl;

    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }

    return 1;
  }
}