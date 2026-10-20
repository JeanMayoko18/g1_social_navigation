/*
 * Processing pipeline
 * -------------------
 *
 * Input:
 *   Topic : /utlidar/cloud_livox_mid360
 *   Frame : livox_frame
 *   Stamp : LiDAR timestamp provided by the Unitree DDS publisher
 *
 * auto_offset:
 *   - Estimate the time offset between the LiDAR clock and the ROS clock.
 *   - Smooth the measured offset using an exponential moving average (EMA)
 *     to reject timestamp jitter.
 *   - Reconstruct a TF-compatible cloud reference time.
 *
 * Deskew:
 *   - Transform each point individually from livox_frame to nav_laser_fixed.
 *   - Each point uses its own acquisition time
 *     (cloud reference time + point.time).
 *
 * Geometry:
 *   - Each point is transformed directly from livox_frame
 *     to nav_laser_fixed using the corresponding TF bin.
 *   - No additional axis reflection or yaw correction is applied.
 *
 * Output:
 *   Topic : /cloud_nav_deskewed
 *   Frame : nav_laser_fixed
 *   Stamp : Current ROS time
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

class CloudNavStabilizerNode : public rclcpp::Node
{
public:
    CloudNavStabilizerNode()
    : Node("cloud_nav_stabilizer_node"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
    {
        /*
         * Input / output topics.
         *
         * The input cloud comes from the real Livox LiDAR.
         * On the G1, this LiDAR frame moves with the head / torso.
         *
         * The output cloud is expressed in a navigation-stable frame.
         */
        input_topic_ = declare_parameter<std::string>(
            "input_cloud_topic",
            "/utlidar/cloud_livox_mid360"
        );

        output_topic_ = declare_parameter<std::string>(
            "output_cloud_topic",
            "/cloud_nav_deskewed"
        );

        /*
         * TF frames.
         *
         * source_frame:
         *   Real LiDAR frame from the input PointCloud2.
         *
         * target_frame:
         *   Navigation-stable frame attached to base_footprint.
         */
        source_frame_ = declare_parameter<std::string>(
            "source_frame",
            "livox_frame"
        );

        target_frame_ = declare_parameter<std::string>(
            "target_frame",
            "nav_laser_fixed"
        );

        /*
         * Livox PointCloud2 has a per-point "time" field.
         *
         * On your G1, this field was measured as nanoseconds:
         *   min/max ≈ 5000 ns -> 100320000 ns
         *
         * Therefore:
         *   seconds = time * 1e-9
         */
        time_scale_ = declare_parameter<double>(
            "time_scale",
            1.0e-9
        );

        /*
         * Optional TF time offset.
         *
         * Keep 0.0 first.
         * Use this only if you later detect a systematic time shift.
         */
        tf_time_offset_sec_ = declare_parameter<double>(
            "tf_time_offset_sec",
            0.0
        );

        time_origin_mode_ = declare_parameter<std::string>(
            "time_origin_mode",
            "auto_offset"
        );

        offset_filter_alpha_ = declare_parameter<double>(
            "offset_filter_alpha",
            0.02
        );

        cloud_time_offset_sec_ = declare_parameter<double>(
            "cloud_time_offset_sec",
            0.0
        );

        /*
         * Number of TF lookup bins.
         *
         * Instead of doing one TF lookup per point, the node samples TF
         * at several times across the cloud duration and assigns each point
         * to the nearest time bin.
         *
         * This is much lighter than 20k TF lookups per cloud.
         */
        bins_ = declare_parameter<int>(
            "deskew_bins",
            40
        );

        bins_ = std::max(1, bins_);

        /*
         * QoS.
         *
         * Livox clouds are high-rate and large.
         * Keep small queue to avoid accumulating old clouds.
         */
        rclcpp::QoS qos(rclcpp::KeepLast(2));
        qos.best_effort();

        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_,
            qos
        );

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            qos,
            std::bind(
                &CloudNavStabilizerNode::callback,
                this,
                std::placeholders::_1
            )
        );

        RCLCPP_INFO(
            get_logger(),
            "\n"
            "========================================\n"
            " cloud_nav_stabilizer_node configuration\n"
            "========================================\n"
            " input_cloud_topic        : %s\n"
            " output_cloud_topic       : %s\n"
            " source_frame             : %s\n"
            " target_frame             : %s\n"
            " point_time_field         : time\n"
            " point_time_unit          : nanoseconds\n"
            " time_field_scale         : %.1e\n"
            " point_time_policy        : cloud stamp + point.time\n"
            " tf_time_offset_sec       : %.6f\n"
            " deskew_bins              : %d\n"
            " deskew_policy            : TF lookup per time bin\n"
            " output_frame_policy      : target_frame\n"
            " time_origin_mode         : %s\n"
            " offset_filter_alpha      : %.3f\n"
            " cloud_time_offset_sec    : %.6f\n"
            " output_stamp_policy      : current ROS time\n"
            "========================================",
            input_topic_.c_str(),
            output_topic_.c_str(),
            source_frame_.c_str(),
            target_frame_.c_str(),
            time_scale_,
            tf_time_offset_sec_,
            bins_,
            time_origin_mode_.c_str(),
            offset_filter_alpha_,
            cloud_time_offset_sec_
        );
    }

private:
    int find_field_offset(
        const sensor_msgs::msg::PointCloud2 & msg,
        const std::string & name)
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

    float read_float(
        const std::vector<uint8_t> & data,
        size_t index)
    {
        float value = 0.0f;
        std::memcpy(&value, &data[index], sizeof(float));
        return value;
    }

    void write_float(
        std::vector<uint8_t> & data,
        size_t index,
        float value)
    {
        std::memcpy(&data[index], &value, sizeof(float));
    }

    struct TfData
    {
        tf2::Matrix3x3 R;
        tf2::Vector3 t;
        bool valid{false};
    };

    TfData to_tf_data(
        const geometry_msgs::msg::TransformStamped & tf_msg)
    {
        TfData out;

        const auto & q = tf_msg.transform.rotation;
        tf2::Quaternion quat(q.x, q.y, q.z, q.w);
        out.R = tf2::Matrix3x3(quat);

        const auto & tr = tf_msg.transform.translation;
        out.t = tf2::Vector3(tr.x, tr.y, tr.z);

        out.valid = true;
        return out;
    }

    bool get_cloud_time_range(
        const sensor_msgs::msg::PointCloud2 & msg,
        int x_offset,
        int y_offset,
        int z_offset,
        int time_offset,
        double & min_time_sec,
        double & max_time_sec)
    {
        const auto & data = msg.data;
        const size_t point_step = static_cast<size_t>(msg.point_step);

        if (point_step == 0)
        {
            return false;
        }

        const size_t point_count = data.size() / point_step;

        min_time_sec = 1e30;
        max_time_sec = -1e30;

        for (size_t i = 0; i < point_count; ++i)
        {
            const size_t base = i * point_step;

            const float x = read_float(data, base + x_offset);
            const float y = read_float(data, base + y_offset);
            const float z = read_float(data, base + z_offset);
            const float point_time_raw = read_float(data, base + time_offset);

            if (!std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z) ||
                !std::isfinite(point_time_raw))
            {
                continue;
            }

            /*
             * Ignore zero / invalid points.
             */
            if ((x * x + y * y + z * z) < 1e-6f)
            {
                continue;
            }

            const double point_time_sec =
                static_cast<double>(point_time_raw) * time_scale_;

            min_time_sec = std::min(min_time_sec, point_time_sec);
            max_time_sec = std::max(max_time_sec, point_time_sec);
        }

        return max_time_sec > min_time_sec;
    }

    bool build_tf_bins(
        const rclcpp::Time & cloud_stamp,
        double min_time_sec,
        double max_time_sec,
        std::vector<TfData> & tf_bins)
    {
        tf_bins.clear();
        tf_bins.resize(static_cast<size_t>(bins_));

        /*
        * Get the latest available TF once.
        *
        * This TF is used for two purposes:
        *   1. It gives the newest timestamp currently available in the TF buffer.
        *   2. It is used as a safe fallback if an exact bin lookup fails.
        *
        * This avoids repeatedly asking TF for a time that is slightly in the future.
        */
        geometry_msgs::msg::TransformStamped latest_tf_msg;

        try
        {
            latest_tf_msg =
                tf_buffer_.lookupTransform(
                    target_frame_,
                    source_frame_,
                    tf2::TimePointZero
                );
        }
        catch (const tf2::TransformException & ex)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                3000,
                "TF tree not ready yet (%s <- %s). Dropping startup cloud: %s",
                target_frame_.c_str(),
                source_frame_.c_str(),
                ex.what()
            );

            return false;
        }

        const TfData latest_tf = to_tf_data(latest_tf_msg);
        const rclcpp::Time latest_tf_stamp(latest_tf_msg.header.stamp);

        const rclcpp::Time now_t = now();
        const rclcpp::Time req_min =
            cloud_stamp + rclcpp::Duration::from_seconds(min_time_sec + tf_time_offset_sec_);
        const rclcpp::Time req_max =
            cloud_stamp + rclcpp::Duration::from_seconds(max_time_sec + tf_time_offset_sec_);

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "TF_TIME | latest_tf_age=%.3f | req_min_age=%.3f | req_max_age=%.3f | req_max_minus_latest=%.3f",
            (now_t - latest_tf_stamp).seconds(),
            (now_t - req_min).seconds(),
            (now_t - req_max).seconds(),
            (req_max - latest_tf_stamp).seconds()
        );

        bool has_any_valid_tf = false;
        TfData previous_valid_tf;

        size_t exact_ok_count = 0;
        size_t clamped_to_latest_count = 0;
        size_t latest_fallback_count = 0;
        size_t previous_fallback_count = 0;

        for (int b = 0; b < bins_; ++b)
        {
            const double alpha =
                (bins_ == 1)
                    ? 0.0
                    : static_cast<double>(b) / static_cast<double>(bins_ - 1);

            const double relative_time_sec =
                min_time_sec +
                alpha * (max_time_sec - min_time_sec) +
                tf_time_offset_sec_;

            rclcpp::Time tf_time =
                cloud_stamp + rclcpp::Duration::from_seconds(relative_time_sec);

            /*
            * Critical production rule:
            *
            * Never request a TF newer than the newest TF currently available.
            * If the computed cloud/bin time is slightly in the future, clamp it
            * to the latest TF stamp before calling lookupTransform().
            *
            * This prevents normal navigation-time logs such as:
            *   "Lookup would require extrapolation into the future"
            */
            if (tf_time > latest_tf_stamp)
            {
                tf_time = latest_tf_stamp;
                clamped_to_latest_count++;
            }

            const size_t idx = static_cast<size_t>(b);

            try
            {
                const auto tf_msg =
                    tf_buffer_.lookupTransform(
                        target_frame_,
                        source_frame_,
                        tf_time,
                        rclcpp::Duration::from_seconds(0.02)
                    );

                tf_bins[idx] = to_tf_data(tf_msg);
                previous_valid_tf = tf_bins[idx];

                has_any_valid_tf = true;
                exact_ok_count++;
            }
            catch (const tf2::TransformException & ex)
            {
                /*
                * Safe fallback:
                * If the exact lookup still fails, use the latest TF already
                * obtained at the beginning of this function.
                */
                if (latest_tf.valid)
                {
                    tf_bins[idx] = latest_tf;
                    previous_valid_tf = latest_tf;

                    has_any_valid_tf = true;
                    latest_fallback_count++;

                    continue;
                }

                /*
                * Last fallback:
                * Reuse the previous valid bin if available.
                */
                if (previous_valid_tf.valid)
                {
                    tf_bins[idx] = previous_valid_tf;
                    previous_fallback_count++;

                    continue;
                }

                RCLCPP_WARN_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    2000,
                    "TF lookup failed %s <- %s at clamped cloud-relative time %.6f s: %s",
                    target_frame_.c_str(),
                    source_frame_.c_str(),
                    relative_time_sec,
                    ex.what()
                );
            }
        }

        /*
        * Backward fill:
        * If the first bins failed before a valid bin existed, reuse the first
        * valid TF found later.
        */
        TfData next_valid_tf;
        bool has_next_valid_tf = false;

        for (int b = bins_ - 1; b >= 0; --b)
        {
            const size_t idx = static_cast<size_t>(b);

            if (tf_bins[idx].valid)
            {
                next_valid_tf = tf_bins[idx];
                has_next_valid_tf = true;
                continue;
            }

            if (has_next_valid_tf)
            {
                tf_bins[idx] = next_valid_tf;
                previous_fallback_count++;
            }
        }

        if (!has_any_valid_tf)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "No usable TF found for cloud %s <- %s. Dropping cloud.",
                target_frame_.c_str(),
                source_frame_.c_str()
            );

            return false;
        }

        RCLCPP_DEBUG_THROTTLE(
            get_logger(),
            *get_clock(),
            5000,
            "TF bins built: exact=%zu clamped_to_latest=%zu latest_fallback=%zu reused_neighbor=%zu total=%d",
            exact_ok_count,
            clamped_to_latest_count,
            latest_fallback_count,
            previous_fallback_count,
            bins_
        );

        return true;
    }

    void callback(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!msg)
        {
            return;
        }

        const int x_offset = find_field_offset(*msg, "x");
        const int y_offset = find_field_offset(*msg, "y");
        const int z_offset = find_field_offset(*msg, "z");
        const int time_offset = find_field_offset(*msg, "time");

        if (x_offset < 0 ||
            y_offset < 0 ||
            z_offset < 0)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "PointCloud2 missing required x/y/z fields"
            );

            return;
        }

        const bool has_point_time = (time_offset >= 0);

        double min_time_sec = 0.0;
        double max_time_sec = 0.0;

        if (has_point_time)
        {
            if (!get_cloud_time_range(
                    *msg,
                    x_offset,
                    y_offset,
                    z_offset,
                    time_offset,
                    min_time_sec,
                    max_time_sec))
            {
                RCLCPP_WARN_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    2000,
                    "Invalid or empty point time range"
                );

                return;
            }
        }
        else
        {
            min_time_sec = 0.0;
            max_time_sec = 0.0;

            RCLCPP_INFO_ONCE(
                get_logger(),
                "PointCloud2 has no per-point time field; using header.stamp"
            );
        }

        //const rclcpp::Time cloud_stamp(msg->header.stamp);
        //const rclcpp::Time cloud_stamp =
        //    now() - rclcpp::Duration::from_seconds(max_time_sec);

        const rclcpp::Time lidar_stamp(msg->header.stamp);
        const rclcpp::Time cb_start = now();
        rclcpp::Time cloud_stamp;

        if (time_origin_mode_ == "lidar_stamp")
        {
            cloud_stamp = lidar_stamp;
        }
        else if (time_origin_mode_ == "lidar_stamp_plus_offset")
        {
            cloud_stamp =
                lidar_stamp +
                rclcpp::Duration::from_seconds(cloud_time_offset_sec_);
        }
        else if (time_origin_mode_ == "ros_now")
        {
            cloud_stamp =
                now() - rclcpp::Duration::from_seconds(max_time_sec);
        }
        else if (time_origin_mode_ == "auto_offset")
        {
            //const double measured_offset_sec =
            //    (now() - lidar_stamp).seconds();

                const double measured_offset_sec =
                    ((now() - rclcpp::Duration::from_seconds(max_time_sec)) - lidar_stamp).seconds();
                    
            if (!offset_initialized_)
            {
                filtered_offset_sec_ = measured_offset_sec;
                offset_initialized_ = true;
            }
            else
            {
                filtered_offset_sec_ =
                    (1.0 - offset_filter_alpha_) * filtered_offset_sec_
                    + offset_filter_alpha_ * measured_offset_sec;
            }

            RCLCPP_DEBUG_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "time_origin=auto_offset | filtered_offset=%.6f s | measured_offset=%.6f s | cloud_duration=%.6f s",
                filtered_offset_sec_,
                measured_offset_sec,
                max_time_sec - min_time_sec
            );

            cloud_stamp =
                lidar_stamp +
                rclcpp::Duration::from_seconds(filtered_offset_sec_);
        }
        else
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Unknown time_origin_mode '%s', falling back to ros_now",
                time_origin_mode_.c_str()
            );

            cloud_stamp =
                now() - rclcpp::Duration::from_seconds(max_time_sec);
        }

        std::vector<TfData> tf_bins;
        const rclcpp::Time cloud_begin =
            cloud_stamp + rclcpp::Duration::from_seconds(min_time_sec);

        const rclcpp::Time cloud_end =
            cloud_stamp + rclcpp::Duration::from_seconds(max_time_sec);

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "CLOUD_TIME | now_age_lidar=%.3f | cloud_begin_age=%.3f | cloud_end_age=%.3f | duration=%.3f | min_t=%.6f max_t=%.6f",
            (cb_start - lidar_stamp).seconds(),
            (cb_start - cloud_begin).seconds(),
            (cb_start - cloud_end).seconds(),
            max_time_sec - min_time_sec,
            min_time_sec,
            max_time_sec
        );

        if (!build_tf_bins(
                cloud_stamp,
                min_time_sec,
                max_time_sec,
                tf_bins))
        {
            return;
        }

        sensor_msgs::msg::PointCloud2 out = *msg;

        /*
         * This is now valid because the points are really transformed
         * into target_frame below.
         */
        out.header.frame_id = target_frame_;
        out.header.stamp = get_clock()->now();

        auto & data = out.data;
        const size_t point_step = static_cast<size_t>(out.point_step);
        const size_t point_count = data.size() / point_step;

        size_t transformed_count = 0;

        for (size_t i = 0; i < point_count; ++i)
        {
            const size_t base = i * point_step;

            float x = read_float(data, base + x_offset);
            float y = read_float(data, base + y_offset);
            float z = read_float(data, base + z_offset);

            
            if (!std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z))
            {
                continue;
            }

            int bin = 0;

            if (has_point_time)
            {
                const float point_time_raw =
                    read_float(data, base + time_offset);

                if (!std::isfinite(point_time_raw))
                {
                    continue;
                }

                const double point_time_sec =
                    static_cast<double>(point_time_raw) * time_scale_;

                double ratio =
                    (point_time_sec - min_time_sec) /
                    (max_time_sec - min_time_sec);

                ratio = std::max(0.0, std::min(1.0, ratio));

                bin =
                    std::max(
                        0,
                        std::min(
                            bins_ - 1,
                            static_cast<int>(
                                std::round(
                                    ratio * static_cast<double>(bins_ - 1)
                                )
                            )
                        )
                    );
            }

            const TfData & T = tf_bins[static_cast<size_t>(bin)];

            if (!T.valid)
            {
                continue;
            }

            /*
            * Transform the point directly from source_frame
            * into target_frame using the TF corresponding to
            * its acquisition-time bin.
            */
            const tf2::Vector3 p(x, y, z);
            const tf2::Vector3 q = T.R * p + T.t;

            write_float(
                data,
                base + x_offset,
                static_cast<float>(q.x())
            );

            write_float(
                data,
                base + y_offset,
                static_cast<float>(q.y())
            );

            write_float(
                data,
                base + z_offset,
                static_cast<float>(q.z())
            );

            transformed_count++;
        }

        const rclcpp::Time cb_end = now();

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "CLOUD_PROC | input_age=%.3f | output_stamp_age=%.3f | processing=%.3f | points=%zu | transformed=%zu",
            (cb_end - lidar_stamp).seconds(),
            (cb_end - rclcpp::Time(out.header.stamp)).seconds(),
            (cb_end - cb_start).seconds(),
            point_count,
            transformed_count
        );

        pub_->publish(out);

        RCLCPP_DEBUG(
            get_logger(),
            "Published deskewed cloud: %zu / %zu points | time %.6f -> %.6f s",
            transformed_count,
            point_count,
            min_time_sec,
            max_time_sec
        );
    }

private:
    std::string input_topic_;
    std::string output_topic_;

    std::string source_frame_;
    std::string target_frame_;

    double time_scale_{1.0e-9};
    double tf_time_offset_sec_{0.0};
    int bins_{40};

    std::string time_origin_mode_{"auto_offset"};
    double cloud_time_offset_sec_{0.0};
    double offset_filter_alpha_{0.02};

    bool offset_initialized_{false};
    double filtered_offset_sec_{0.0};

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
    try
    {
        rclcpp::init(argc, argv);

        auto node =
            std::make_shared<CloudNavStabilizerNode>();

        rclcpp::spin(node);

        rclcpp::shutdown();

        return 0;
    }
    catch (const std::exception & e)
    {
        std::cerr
            << "[cloud_nav_stabilizer_node] Fatal exception: "
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
            << "[cloud_nav_stabilizer_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}