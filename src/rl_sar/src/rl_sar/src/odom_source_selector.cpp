#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"


class OdomSourceSelector : public rclcpp::Node
{
public:
    OdomSourceSelector()
    : Node("odom_source_selector")
    {
        this->declare_parameter<std::string>(
            "input_topic",
            "/odom_aocf"
        );

        this->declare_parameter<std::string>(
            "output_topic",
            "/odom"
        );

        this->declare_parameter<std::string>(
            "odom_frame",
            "odom"
        );

        this->declare_parameter<std::string>(
            "base_frame",
            "base_footprint"
        );

        this->declare_parameter<bool>(
            "publish_tf",
            true
        );

        this->declare_parameter<bool>(
            "preserve_input_stamp",
            true
        );

        input_topic_ = this->get_parameter(
            "input_topic"
        ).as_string();

        output_topic_ = this->get_parameter(
            "output_topic"
        ).as_string();

        odom_frame_ = this->get_parameter(
            "odom_frame"
        ).as_string();

        base_frame_ = this->get_parameter(
            "base_frame"
        ).as_string();

        publish_tf_ = this->get_parameter(
            "publish_tf"
        ).as_bool();

        preserve_input_stamp_ = this->get_parameter(
            "preserve_input_stamp"
        ).as_bool();

        auto qos = rclcpp::QoS(
            rclcpp::KeepLast(10)
        );

        qos.reliable();
        qos.durability_volatile();

        odom_publisher_ =
            this->create_publisher<nav_msgs::msg::Odometry>(
                output_topic_,
                qos
            );

        tf_broadcaster_ =
            std::make_unique<tf2_ros::TransformBroadcaster>(
                *this
            );

        odom_subscription_ =
            this->create_subscription<nav_msgs::msg::Odometry>(
                input_topic_,
                qos,
                std::bind(
                    &OdomSourceSelector::odom_callback,
                    this,
                    std::placeholders::_1
                )
            );

        RCLCPP_INFO(
            this->get_logger(),
            "Odometry source selector started"
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Input topic: %s",
            input_topic_.c_str()
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Output topic: %s",
            output_topic_.c_str()
        );

        RCLCPP_INFO(
            this->get_logger(),
            "TF authority: %s -> %s",
            odom_frame_.c_str(),
            base_frame_.c_str()
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Publish TF: %s",
            publish_tf_ ? "true" : "false"
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Preserve input stamp: %s",
            preserve_input_stamp_ ? "true" : "false"
        );
    }

private:
    void odom_callback(
        const nav_msgs::msg::Odometry::SharedPtr input_message
    )
    {
        auto output_message = *input_message;

        if (!preserve_input_stamp_) {
            output_message.header.stamp = this->now();
        }

        output_message.header.frame_id = odom_frame_;
        output_message.child_frame_id = base_frame_;

        odom_publisher_->publish(output_message);

        if (publish_tf_) {
            geometry_msgs::msg::TransformStamped transform;

            transform.header.stamp =
                output_message.header.stamp;

            transform.header.frame_id =
                odom_frame_;

            transform.child_frame_id =
                base_frame_;

            transform.transform.translation.x =
                output_message.pose.pose.position.x;

            transform.transform.translation.y =
                output_message.pose.pose.position.y;

            transform.transform.translation.z =
                output_message.pose.pose.position.z;

            transform.transform.rotation =
                output_message.pose.pose.orientation;

            tf_broadcaster_->sendTransform(transform);
        }

        message_count_++;

        if (message_count_ % 500 == 0) {
            RCLCPP_INFO(
                this->get_logger(),
                "Forwarded %zu odometry messages from %s",
                message_count_,
                input_topic_.c_str()
            );
        }
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string odom_frame_;
    std::string base_frame_;

    bool publish_tf_;
    bool preserve_input_stamp_;

    std::size_t message_count_{0};

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
        odom_publisher_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
        odom_subscription_;

    std::unique_ptr<tf2_ros::TransformBroadcaster>
        tf_broadcaster_;
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
        OdomSourceSelector>();

    rclcpp::spin(
      node);

    rclcpp::shutdown();

    return 0;
  }
  catch (
    const std::exception & e)
  {
    std::cerr
      << "[odom_source_selector] Fatal exception: "
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
      << "[odom_source_selector] "
      << "Unknown fatal exception"
      << std::endl;

    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }

    return 1;
  }
}
