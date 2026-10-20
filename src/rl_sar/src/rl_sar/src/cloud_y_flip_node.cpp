/*
 * cloud_y_flip_node.cpp
 *
 * PURPOSE
 * -------
 * This node corrects the lateral Y-axis direction of a PointCloud2 stream.
 *
 * Pipeline:
 *
 *   /cloud_now
 *       ↓
 *   cloud_y_flip_node
 *       ↓
 *   /cloud_fixed
 *
 * WHY
 * ---
 * On the tested G1/Livox setup, the cloud appears mirrored laterally:
 *
 *   - forward/backward is correct
 *   - left/right is inverted
 *
 * This node applies:
 *
 *   x' =  x
 *   y' = -y
 *   z' =  z
 *
 * before the cloud is converted into LaserScan.
 *
 * This is important because if Y is not corrected, obstacles on the
 * left may appear on the right in Nav2 costmaps.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstring>
#include <string>

class CloudYFlipNode : public rclcpp::Node
{
public:
    CloudYFlipNode()
    : Node("cloud_y_flip_node")
    {
        /*
         * Input restamped cloud.
         */
        input_topic_ =
            declare_parameter<std::string>(
                "input_cloud_topic",
                "/cloud_now"
            );

        /*
         * Output corrected cloud.
         */
        output_topic_ =
            declare_parameter<std::string>(
                "output_cloud_topic",
                "/cloud_fixed"
            );

        /*
         * Sensor stream QoS.
         *
         * BEST_EFFORT is acceptable because only the latest cloud matters.
         */
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        pub_ =
            create_publisher<sensor_msgs::msg::PointCloud2>(
                output_topic_,
                qos
            );

        sub_ =
            create_subscription<sensor_msgs::msg::PointCloud2>(
                input_topic_,
                qos,
                std::bind(
                    &CloudYFlipNode::callback,
                    this,
                    std::placeholders::_1
                )
            );

        RCLCPP_INFO(
            get_logger(),
            "[cloud_y_flip_node] %s -> %s | applying y' = -y",
            input_topic_.c_str(),
            output_topic_.c_str()
        );
    }

private:
    /*
     * Find the byte offset of a PointCloud2 field.
     *
     * PointCloud2 stores points as raw binary data.
     * The fields array tells us where x/y/z are located inside each point.
     */
    int find_field_offset(
        const sensor_msgs::msg::PointCloud2 & msg,
        const std::string & field_name)
    {
        for (const auto & field : msg.fields)
        {
            if (field.name == field_name)
                return static_cast<int>(field.offset);
        }

        return -1;
    }

    /*
     * Callback executed for each incoming cloud.
     *
     * It copies the full cloud and modifies only the Y coordinate
     * of each point.
     */
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!msg)
            return;

        const int y_offset =
            find_field_offset(*msg, "y");

        if (y_offset < 0)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "PointCloud2 does not contain field 'y'"
            );
            return;
        }

        /*
         * Copy entire message.
         *
         * This preserves:
         *   - header
         *   - frame_id
         *   - x/z/intensity/ring/time fields
         *   - point layout
         */
        sensor_msgs::msg::PointCloud2 out = *msg;
        //out.header.frame_id = "nav_laser_fixed";
        out.header.frame_id = msg->header.frame_id; 

        auto & data = out.data;

        const size_t point_step =
            static_cast<size_t>(out.point_step);

        if (point_step == 0)
            return;

        const size_t point_count =
            data.size() / point_step;

        /*
         * For each point:
         *
         *   read y
         *   y = -y
         *   write y back
         */
        for (size_t i = 0; i < point_count; ++i)
        {
            const size_t base =
                i * point_step;

            float y = 0.0f;

            std::memcpy(
                &y,
                &data[base + y_offset],
                sizeof(float)
            );

            y = -y;

            std::memcpy(
                &data[base + y_offset],
                &y,
                sizeof(float)
            );
        }
        /*
        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "Publishing /cloud_fixed frame_id=%s",
            out.header.frame_id.c_str()
        );
        */
        pub_->publish(out);
    }

private:
    std::string input_topic_;
    std::string output_topic_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
    try
    {
        /*
         * Initialize ROS 2.
         */
        rclcpp::init(argc, argv);

        /*
         * Create node.
         */
        auto node =
            std::make_shared<CloudYFlipNode>();

        /*
         * Spin ROS 2 executor.
         */
        rclcpp::spin(node);

        /*
         * Shutdown ROS 2 cleanly.
         */
        rclcpp::shutdown();

        return 0;
    }
    catch (const std::exception & e)
    {
        /*
         * Catch standard C++ exceptions.
         */
        std::cerr
            << "[cloud_y_flip_node] Fatal exception: "
            << e.what()
            << std::endl;

        /*
         * Shutdown ROS 2 if still active.
         */
        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
    catch (...)
    {
        /*
         * Catch unknown exceptions.
         */
        std::cerr
            << "[cloud_y_flip_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}