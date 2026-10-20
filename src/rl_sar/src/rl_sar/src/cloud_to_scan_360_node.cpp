/*
 * cloud_to_scan_360_node.cpp
 *
 * PURPOSE
 * -------
 * Convert a PointCloud2 into a deterministic 360-beam LaserScan.
 *
 * This node is intended to replace pointcloud_to_laserscan on ROS 2 Foxy
 * when the official node publishes inconsistent LaserScan geometry.
 *
 * Pipeline:
 *
 *   /cloud_fixed
 *       ↓
 *   cloud_to_scan_360_node
 *       ↓
 *   /scan_robot
 *
 * The node always publishes:
 *
 *   - 360 ranges
 *   - angle_min = -pi
 *   - angle_increment = 1 degree
 *   - angle_max = -pi + 359 degrees
 *
 * For each angular bin, the closest valid point is kept.
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

class CloudToScan360Node : public rclcpp::Node
{
public:
    CloudToScan360Node()
    : Node("cloud_to_scan_360_node")
    {
        input_cloud_topic_ = declare_parameter<std::string>(
            "input_cloud_topic", "/cloud_fixed");

        output_scan_topic_ = declare_parameter<std::string>(
            "output_scan_topic", "/scan_robot");

        output_frame_id_ = declare_parameter<std::string>(
            "output_frame_id", "nav_laser_fixed");

        range_min_ = declare_parameter<double>("range_min", 0.45);
        range_max_ = declare_parameter<double>("range_max", 8.0);

        min_height_ = declare_parameter<double>("min_height", -1.0);
        max_height_ = declare_parameter<double>("max_height", 1.0);

        scan_time_ = declare_parameter<double>("scan_time", 0.1);
        use_node_time_ = declare_parameter<bool>("use_node_time", true);

        use_inf_ = declare_parameter<bool>("use_inf", true);
        invalid_range_value_ = declare_parameter<double>("invalid_range_value", 8.0);

        target_beams_ = declare_parameter<int>("target_beams", 360);

        if (target_beams_ < 2)
        {
            RCLCPP_WARN(get_logger(), "target_beams < 2, forcing 360");
            target_beams_ = 360;
        }

        angle_min_ = -M_PI;
        angle_increment_ = 2.0 * M_PI / static_cast<double>(target_beams_);
        angle_max_ = angle_min_ + static_cast<double>(target_beams_ - 1) * angle_increment_;

        rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
        sensor_qos.best_effort();
        sensor_qos.durability_volatile();

        pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
            output_scan_topic_,
            sensor_qos);

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_cloud_topic_,
            sensor_qos,
            std::bind(&CloudToScan360Node::callback, this, std::placeholders::_1));

        RCLCPP_INFO(
            get_logger(),
            "\n"
            "========================================\n"
            " cloud_to_scan_360_node configuration\n"
            "========================================\n"
            " input_cloud_topic : %s\n"
            " output_scan_topic : %s\n"
            " output_frame_id   : %s\n"
            " target_beams      : %d\n"
            " angle_min         : %.6f\n"
            " angle_max         : %.6f\n"
            " angle_increment   : %.9f\n"
            " range_min         : %.3f\n"
            " range_max         : %.3f\n"
            " min_height        : %.3f\n"
            " max_height        : %.3f\n"
            " use_node_time     : %s\n"
            " use_inf           : %s\n"
            "========================================",
            input_cloud_topic_.c_str(),
            output_scan_topic_.c_str(),
            output_frame_id_.c_str(),
            target_beams_,
            angle_min_,
            angle_max_,
            angle_increment_,
            range_min_,
            range_max_,
            min_height_,
            max_height_,
            use_node_time_ ? "true" : "false",
            use_inf_ ? "true" : "false");
    }

private:
    int fieldOffset(
        const sensor_msgs::msg::PointCloud2 & msg,
        const std::string & name) const
    {
        for (const auto & field : msg.fields)
        {
            if (field.name == name)
            {
                return static_cast<int>(field.offset);
            }
        }

        return -1;
    }

    bool readFloatField(
        const sensor_msgs::msg::PointCloud2 & msg,
        size_t base,
        int offset,
        float & value) const
    {
        if (offset < 0)
        {
            return false;
        }

        const size_t idx = base + static_cast<size_t>(offset);

        if (idx + sizeof(float) > msg.data.size())
        {
            return false;
        }

        std::memcpy(&value, &msg.data[idx], sizeof(float));
        return std::isfinite(value);
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!msg)
        {
            return;
        }

        const int x_offset = fieldOffset(*msg, "x");
        const int y_offset = fieldOffset(*msg, "y");
        const int z_offset = fieldOffset(*msg, "z");

        if (x_offset < 0 || y_offset < 0 || z_offset < 0)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "PointCloud2 must contain x, y and z fields");
            return;
        }

        if (msg->point_step == 0 || msg->data.empty())
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Dropping empty PointCloud2");
            return;
        }

        sensor_msgs::msg::LaserScan scan;

        //scan.header.stamp = use_node_time_ ? now() : msg->header.stamp;
        if (use_node_time_)
        {
            auto t = this->get_clock()->now();

            scan.header.stamp.sec =
                static_cast<int32_t>(t.nanoseconds() / 1000000000LL);

            scan.header.stamp.nanosec =
                static_cast<uint32_t>(t.nanoseconds() % 1000000000LL);
        }
        else
        {
            scan.header.stamp = msg->header.stamp;
        }


        scan.header.frame_id = output_frame_id_;

        scan.angle_min = angle_min_;
        scan.angle_max = angle_max_;
        scan.angle_increment = angle_increment_;
        scan.time_increment = 0.0;
        scan.scan_time = scan_time_;
        scan.range_min = range_min_;
        scan.range_max = range_max_;

        const float empty_value =
            use_inf_
            ? std::numeric_limits<float>::infinity()
            : static_cast<float>(invalid_range_value_);

        scan.ranges.assign(static_cast<size_t>(target_beams_), empty_value);
        scan.intensities.clear();

        const size_t point_count = msg->data.size() / msg->point_step;

        size_t valid_points = 0;
        size_t used_points = 0;

        for (size_t i = 0; i < point_count; ++i)
        {
            const size_t base = i * msg->point_step;

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            if (!readFloatField(*msg, base, x_offset, x) ||
                !readFloatField(*msg, base, y_offset, y) ||
                !readFloatField(*msg, base, z_offset, z))
            {
                continue;
            }

            if (z < min_height_ || z > max_height_)
            {
                continue;
            }

            const double range = std::sqrt(
                static_cast<double>(x) * static_cast<double>(x) +
                static_cast<double>(y) * static_cast<double>(y));

            if (!std::isfinite(range) || range < range_min_ || range > range_max_)
            {
                continue;
            }

            valid_points++;

            double angle = std::atan2(static_cast<double>(y), static_cast<double>(x));

            while (angle < angle_min_)
            {
                angle += 2.0 * M_PI;
            }

            while (angle > M_PI)
            {
                angle -= 2.0 * M_PI;
            }

            int bin = static_cast<int>(
                std::floor((angle - angle_min_) / angle_increment_ + 0.5));

            if (bin < 0 || bin >= target_beams_)
            {
                continue;
            }

            auto & current = scan.ranges[static_cast<size_t>(bin)];

            if (!std::isfinite(current) || range < static_cast<double>(current))
            {
                current = static_cast<float>(range);
                used_points++;
            }
        }

        if (valid_points == 0)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "No valid points after filtering. Publishing empty scan.");
        }

        pub_->publish(scan);

        RCLCPP_DEBUG(
            get_logger(),
            "Published scan: beams=%zu, target_beams=%d, valid_points=%zu, used_points=%zu",
            scan.ranges.size(),
            target_beams_,
            valid_points,
            used_points);
    }

private:
    std::string input_cloud_topic_;
    std::string output_scan_topic_;
    std::string output_frame_id_;

    double range_min_;
    double range_max_;
    double min_height_;
    double max_height_;
    double scan_time_;
    bool use_node_time_;
    bool use_inf_;
    double invalid_range_value_;

    int target_beams_;
    double angle_min_;
    double angle_max_;
    double angle_increment_;

    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
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
            std::make_shared<CloudToScan360Node>();

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
            << "[cloud_to_scan_360_node] Fatal exception: "
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
            << "[cloud_to_scan_360_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}