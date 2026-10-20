/*
 * pc2_restamper_node.cpp
 *
 * PURPOSE
 * -------
 * This node receives the raw Livox MID360 PointCloud2 stream and republishes it
 * into two different output topics:
 *
 *   1) /cloud_now
 *      - Timestamp is replaced with the current ROS time.
 *      - Intended for Nav2, pointcloud_to_laserscan, costmaps, and TF-sensitive
 *        visualization pipelines.
 *
 *   2) /cloud_fastlio
 *      - Original LiDAR timestamp is preserved.
 *      - Intended for FAST-LIO, because FAST-LIO must synchronize LiDAR clouds
 *        with IMU measurements using their original sensor timestamps.
 *
 * WHY TWO OUTPUTS?
 * ----------------
 * Nav2 often needs clouds stamped with the current ROS time to avoid TF errors:
 *
 *   - extrapolation into the past
 *   - message filter drops
 *   - unstable scan generation
 *
 * FAST-LIO, however, must NOT receive restamped clouds, because restamping breaks
 * LiDAR/IMU synchronization. If the LiDAR cloud timestamp is shifted while the
 * IMU keeps its original timestamp, FAST-LIO may receive both topics but never
 * publish /Odometry or /cloud_registered.
 *
 * PIPELINE
 * --------
 *
 *   /utlidar/cloud_livox_mid360
 *              |
 *              v
 *     pc2_restamper_node
 *          /        \
 *         /          \
 *        v            v
 *   /cloud_now    /cloud_fastlio
 *   restamped     original stamp
 *   for Nav2      for FAST-LIO
 */

#include <memory>
#include <string>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>


class Pc2RestamperNode : public rclcpp::Node
{
public:
    Pc2RestamperNode()
    : Node("pc2_restamper_node")
    {
        /*
         * Input raw PointCloud2 topic.
         *
         * On the Unitree G1, this is the raw Livox MID360 PointCloud2 topic.
         */
        input_topic_ =
            declare_parameter<std::string>(
                "input_cloud_topic",
                "/utlidar/cloud_livox_mid360"
            );

        /*
         * Output topic for Nav2 / scan generation.
         *
         * This cloud is restamped with the current ROS time.
         */
        output_now_topic_ =
            declare_parameter<std::string>(
                "output_cloud_now_topic",
                "/cloud_now"
            );
        
        output_now_frame_id_ =
            declare_parameter<std::string>(
                "output_now_frame_id",
                "livox_2d"
            );

        /*
         * Output topic for FAST-LIO.
         *
         * This cloud preserves the original LiDAR timestamp.
         */
        output_fastlio_topic_ =
            declare_parameter<std::string>(
                "output_cloud_fastlio_topic",
                "/cloud_fastlio"
            );

        /*
         * Subscriber QoS.
         *
         * The raw Livox PointCloud2 publisher was observed as RELIABLE.
         * Therefore, this subscriber is also RELIABLE to guarantee QoS
         * compatibility with /utlidar/cloud_livox_mid360.
         */
        rclcpp::QoS sub_qos(rclcpp::KeepLast(1));
        sub_qos.reliable();
        sub_qos.durability_volatile();

        /*
         * Publisher QoS for /cloud_now.
         *
         * BEST_EFFORT is acceptable for high-rate sensor streams used by Nav2
         * and pointcloud_to_laserscan. If one cloud is dropped, the next one
         * arrives quickly.
         */
        rclcpp::QoS pub_now_qos(rclcpp::KeepLast(1));
        pub_now_qos.best_effort();
        pub_now_qos.durability_volatile();

        /*
         * Publisher QoS for /cloud_fastlio.
         *
         * FAST-LIO commonly uses sensor-data style QoS, usually BEST_EFFORT.
         * Publishing /cloud_fastlio as BEST_EFFORT makes it compatible with
         * FAST-LIO when its PointCloud2 subscriber uses SensorDataQoS.
         */
        rclcpp::QoS pub_fastlio_qos(rclcpp::KeepLast(1));
        pub_fastlio_qos.best_effort();
        pub_fastlio_qos.durability_volatile();

        /*
         * Publisher for the Nav2 / scan pipeline.
         */
        pub_now_ =
            create_publisher<sensor_msgs::msg::PointCloud2>(
                output_now_topic_,
                pub_now_qos
            );

        /*
         * Publisher for FAST-LIO.
         */
        pub_fastlio_ =
            create_publisher<sensor_msgs::msg::PointCloud2>(
                output_fastlio_topic_,
                pub_fastlio_qos
            );

        /*
        * Enable/disable publication of each output stream.
        */
        publish_cloud_now_ =
            declare_parameter<bool>(
                "publish_cloud_now",
                true
            );

        publish_cloud_fastlio_ =
            declare_parameter<bool>(
                "publish_cloud_fastlio",
                true
            );

        /*
         * Subscriber to the raw Livox cloud.
         */
        sub_ =
            create_subscription<sensor_msgs::msg::PointCloud2>(
                input_topic_,
                sub_qos,
                std::bind(
                    &Pc2RestamperNode::cloudCallback,
                    this,
                    std::placeholders::_1
                )
            );

        RCLCPP_INFO(
            get_logger(),
            "\n"
            "========================================\n"
            " pc2_restamper_node configuration\n"
            "========================================\n"
            " input_cloud_topic        : %s\n"
            " nav2_output_topic        : %s\n"
            " nav2_stamp_policy        : current ROS time\n"
            " nav2_cloud_frame_id      : %s\n"
            " fastlio_output_topic     : %s\n"
            " fastlio_stamp_policy     : original LiDAR time\n"
            " fastlio_frame_policy     : original LiDAR frame\n"
            " publish_cloud_now        : %s\n"
            " publish_cloud_fastlio    : %s\n"
            "========================================",
            input_topic_.c_str(),
            output_now_topic_.c_str(),
            output_now_frame_id_.c_str(),
            output_fastlio_topic_.c_str(),
            publish_cloud_now_ ? "true" : "false",
            publish_cloud_fastlio_ ? "true" : "false"
        );

    }

private:
    /*
     * Callback executed for each incoming PointCloud2 message.
     *
     * The point cloud data itself is never modified.
     * Only the timestamp policy changes depending on the output topic.
     */

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!msg)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "Received a null PointCloud2 message. Ignoring it."
            );
            return;
        }

        /*
        * Output 1: /cloud_now
        * Used by Nav2 / pointcloud_to_laserscan.
        * Stamp is current ROS time and frame is forced to a stable 2D LiDAR frame.
        */
        if (publish_cloud_now_)
        {
            sensor_msgs::msg::PointCloud2 cloud_now = *msg;
            cloud_now.header.stamp = now();
            cloud_now.header.frame_id = output_now_frame_id_;
            pub_now_->publish(cloud_now);
        }

        /*
        * Output 2: /cloud_fastlio
        * Used by FAST-LIO.
        * Stamp and frame are preserved exactly as received.
        */
        if (publish_cloud_fastlio_)
        {
            sensor_msgs::msg::PointCloud2 cloud_fastlio = *msg;
            cloud_fastlio.header.stamp = msg->header.stamp;
            cloud_fastlio.header.frame_id = msg->header.frame_id;
            pub_fastlio_->publish(cloud_fastlio);
        }
    }

private:
    std::string input_topic_;
    std::string output_now_topic_;
    std::string output_fastlio_topic_;
    std::string output_now_frame_id_;

    bool publish_cloud_now_;
    bool publish_cloud_fastlio_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_now_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_fastlio_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};


int main(int argc, char ** argv)
{
    try
    {
        rclcpp::init(argc, argv);

        auto node =
            std::make_shared<Pc2RestamperNode>();

        rclcpp::spin(node);

        rclcpp::shutdown();

        return 0;
    }
    catch (const std::exception & e)
    {
        std::cerr
            << "[pc2_restamper_node] Fatal exception: "
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
            << "[pc2_restamper_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}