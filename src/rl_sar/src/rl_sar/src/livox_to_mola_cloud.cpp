/*
 * livox_to_mola_cloud_node.cpp
 *
 * Converts Livox MID-360 PointCloud2:
 *   x, y, z, intensity, ring, time
 *   point_step = 22
 *
 * into a standard PointCloud2 for MOLA:
 *   x, y, z, intensity
 *   point_step = 16
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

class LivoxToMolaCloudNode : public rclcpp::Node
{
public:
    LivoxToMolaCloudNode()
    : Node("livox_to_mola_cloud_node")
    {
        input_topic_ = declare_parameter<std::string>(
            "input_cloud_topic", "/cloud_fixed");

        output_topic_ = declare_parameter<std::string>(
            "output_cloud_topic", "/mola/cloud_xyz_i");

        output_frame_id_ = declare_parameter<std::string>(
            "output_frame_id", "mola_livox_frame");

        max_points_ = declare_parameter<int>("max_points", 0);
        //skip_n_ = std::max(1, declare_parameter<int>("skip_n", 1));

        const int skip_n_param =
            static_cast<int>(
                declare_parameter<int>("skip_n", 1)
            );

        skip_n_ = std::max<int>(1, skip_n_param);

        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_, qos);

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            qos,
            std::bind(
                &LivoxToMolaCloudNode::cloudCallback,
                this,
                std::placeholders::_1));

        RCLCPP_INFO(
            get_logger(),
            "[livox_to_mola_cloud_node] %s -> %s | output_frame=%s",
            input_topic_.c_str(),
            output_topic_.c_str(),
            output_frame_id_.c_str());
    }

private:
    int findFieldOffset(
        const sensor_msgs::msg::PointCloud2 & msg,
        const std::string & name) const
    {
        for (const auto & field : msg.fields)
        {
            if (field.name == name)
                return static_cast<int>(field.offset);
        }
        return -1;
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!msg)
            return;

        const int x_off = findFieldOffset(*msg, "x");
        const int y_off = findFieldOffset(*msg, "y");
        const int z_off = findFieldOffset(*msg, "z");
        const int i_off = findFieldOffset(*msg, "intensity");

        if (x_off < 0 || y_off < 0 || z_off < 0)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Input cloud missing required x/y/z fields");
            return;
        }

        const size_t point_step = msg->point_step;
        const size_t total_points =
            static_cast<size_t>(msg->width) *
            static_cast<size_t>(msg->height);

        if (point_step == 0 || total_points == 0)
            return;

        std::vector<uint8_t> out_data;
        out_data.reserve(total_points * 16);

        uint32_t count = 0;

        for (size_t idx = 0; idx < total_points; idx += skip_n_)
        {
            if (max_points_ > 0 &&
                count >= static_cast<uint32_t>(max_points_))
            {
                break;
            }

            const size_t base = idx * point_step;

            if (base + point_step > msg->data.size())
                break;

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float intensity = 0.0f;

            std::memcpy(&x, &msg->data[base + x_off], sizeof(float));
            std::memcpy(&y, &msg->data[base + y_off], sizeof(float));
            std::memcpy(&z, &msg->data[base + z_off], sizeof(float));

            if (i_off >= 0)
                std::memcpy(&intensity, &msg->data[base + i_off], sizeof(float));

            if (!std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z) ||
                !std::isfinite(intensity))
            {
                continue;
            }

            const size_t old_size = out_data.size();
            out_data.resize(old_size + 16);

            std::memcpy(out_data.data() + old_size + 0,  &x, sizeof(float));
            std::memcpy(out_data.data() + old_size + 4,  &y, sizeof(float));
            std::memcpy(out_data.data() + old_size + 8,  &z, sizeof(float));
            std::memcpy(out_data.data() + old_size + 12, &intensity, sizeof(float));

            count++;
        }

        sensor_msgs::msg::PointCloud2 out;
        out.header = msg->header;

        if (!output_frame_id_.empty())
            out.header.frame_id = output_frame_id_;

        out.height = 1;
        out.width = count;
        out.is_bigendian = false;
        out.is_dense = true;

        out.point_step = 16;
        out.row_step = out.point_step * count;
        out.data = std::move(out_data);

        out.fields.clear();
        out.fields.push_back(makeField("x", 0));
        out.fields.push_back(makeField("y", 4));
        out.fields.push_back(makeField("z", 8));
        // IMPORTANT:
        // Keep point_step = 16 and keep intensity in the binary data,
        // but do NOT declare the intensity field.
        // This reproduces the Python version that worked with MOLA.
        //out.fields.push_back(makeField("intensity", 12));

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "[MOLA CLOUD] input=%zu output=%u point_step=%u fields=%zu frame=%s",
            total_points,
            out.width,
            out.point_step,
            out.fields.size(),
            out.header.frame_id.c_str()
        );
        pub_->publish(out);
    }

    sensor_msgs::msg::PointField makeField(
        const std::string & name,
        uint32_t offset) const
    {
        sensor_msgs::msg::PointField field;
        field.name = name;
        field.offset = offset;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.count = 1;
        return field;
    }

private:
    std::string input_topic_;
    std::string output_topic_;
    std::string output_frame_id_;

    int max_points_{0};
    int skip_n_{1};

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
            std::make_shared<LivoxToMolaCloudNode>();

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
            << "[livox_to_mola_cloud_node] Fatal exception: "
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
            << "[livox_to_mola_cloud_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}