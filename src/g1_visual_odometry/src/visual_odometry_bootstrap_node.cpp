/**
 * File: visual_odometry_bootstrap_node.cpp
 *
 * Role:
 * -----
 * This node provides a lightweight bootstrap visual odometry estimate for the
 * Unitree G1 pipeline.
 *
 * It is NOT a full SLAM or production-grade VIO backend.
 * Its purpose is to replace the previous odometry stub with a simple, real,
 * image-driven planar odometry estimate that can be used for integration tests.
 *
 * Inputs:
 *   - /camera/image_raw      : monocular image stream
 *   - /camera/camera_info    : camera intrinsics
 *   - /dog_imu_raw           : IMU orientation
 *
 * Outputs:
 *   - /odom/visual           : nav_msgs/Odometry
 *   - TF: odom -> base_footprint
 *
 * How it works:
 * -------------
 * 1. The node converts incoming images to grayscale.
 * 2. It detects good visual features in the previous image.
 * 3. It tracks them in the current image using Lucas-Kanade optical flow.
 * 4. It computes the median image displacement (pixel flow).
 * 5. It converts the median forward pixel displacement into an approximate
 *    planar metric displacement using a tunable meters_per_pixel factor.
 * 6. Yaw is taken from the IMU and used as the robot heading.
 * 7. The node integrates planar pose over time and publishes odometry + TF.
 *
 * Important limitations:
 * ----------------------
 * - Monocular vision alone does not provide absolute metric scale.
 * - The parameter meters_per_pixel is only a rough approximation.
 * - At low image rate, motion estimates may be noisy.
 * - This node is meant for pipeline integration and early experiments,
 *   not for final navigation quality.
 *
 * Robustness additions:
 * ---------------------
 * - The node checks that consecutive images have the same resolution.
 * - It resets tracking safely when the image size changes.
 * - It catches OpenCV and standard exceptions in the image callback so the
 *   process does not abort on corrupted or inconsistent frames.
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

class VisualOdometryBootstrapNode : public rclcpp::Node
{
public:
  VisualOdometryBootstrapNode()
  : Node("visual_odometry_bootstrap_node"),
    sensor_qos_(rclcpp::SensorDataQoS())
    //,tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this))
  {
    // ------------------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------------------
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    imu_topic_ = this->declare_parameter<std::string>("imu_topic", "/dog_imu_raw");
    odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/odom/visual");

    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_footprint");

    publish_tf_ = this->declare_parameter<bool>("publish_tf", false);

    max_features_ = this->declare_parameter<int>("max_features", 100);
    quality_level_ = this->declare_parameter<double>("quality_level", 0.01);
    min_distance_ = this->declare_parameter<double>("min_distance", 8.0);

    meters_per_pixel_ = this->declare_parameter<double>("meters_per_pixel", 0.0025);
    min_tracked_points_ = this->declare_parameter<int>("min_tracked_points", 10);
    max_dt_s_ = this->declare_parameter<double>("max_dt_s", 0.8);

    if (publish_tf_) {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

    // ------------------------------------------------------------------------
    // Publisher
    // ------------------------------------------------------------------------
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);

    // ------------------------------------------------------------------------
    // Subscribers
    // ------------------------------------------------------------------------
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      sensor_qos_,
      std::bind(&VisualOdometryBootstrapNode::image_callback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      sensor_qos_,
      std::bind(&VisualOdometryBootstrapNode::camera_info_callback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      sensor_qos_,
      std::bind(&VisualOdometryBootstrapNode::imu_callback, this, std::placeholders::_1));

    // ------------------------------------------------------------------------
    // Startup logs
    // ------------------------------------------------------------------------
    RCLCPP_INFO(this->get_logger(), "VisualOdometryBootstrapNode started.");
    RCLCPP_INFO(this->get_logger(), "  image_topic         : %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  camera_info_topic   : %s", camera_info_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  imu_topic           : %s", imu_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  odom_topic          : %s", odom_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  odom_frame          : %s", odom_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  base_frame          : %s", base_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  publish_tf          : %s", publish_tf_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  max_features        : %d", max_features_);
    RCLCPP_INFO(this->get_logger(), "  min_tracked_points  : %d", min_tracked_points_);
    RCLCPP_INFO(this->get_logger(), "  max_dt_s            : %.3f", max_dt_s_);
    RCLCPP_INFO(this->get_logger(), "  meters_per_pixel    : %.6f", meters_per_pixel_);
  }

private:
  // --------------------------------------------------------------------------
  // Utility: extract yaw from quaternion
  // --------------------------------------------------------------------------
  static double quat_to_yaw(const geometry_msgs::msg::Quaternion & q)
  {
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  // --------------------------------------------------------------------------
  // Utility: robust median of a vector
  // --------------------------------------------------------------------------
  static double median(std::vector<double> values)
  {
    if (values.empty()) {
      return 0.0;
    }

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];

    if (values.size() % 2 == 0) {
      std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
      med = 0.5 * (med + values[mid - 1]);
    }

    return med;
  }

  // --------------------------------------------------------------------------
  // Normalize angle to [-pi, pi]
  // --------------------------------------------------------------------------
  double normalize_angle(double a) const
  {
    while (a > M_PI) {
      a -= 2.0 * M_PI;
    }
    while (a < -M_PI) {
      a += 2.0 * M_PI;
    }
    return a;
  }

  // --------------------------------------------------------------------------
  // Camera info callback
  // --------------------------------------------------------------------------
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    latest_camera_info_ = *msg;
    has_camera_info_ = true;
  }

  // --------------------------------------------------------------------------
  // IMU callback
  // We only use yaw from the IMU orientation in this bootstrap version.
  // --------------------------------------------------------------------------
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    latest_yaw_ = quat_to_yaw(msg->orientation);
    latest_imu_stamp_ = rclcpp::Time(msg->header.stamp);
    has_imu_ = true;
  }

  // --------------------------------------------------------------------------
  // Convert ROS image to grayscale OpenCV image
  // Supported encodings:
  //   - mono8
  //   - bgr8
  //   - rgb8
  // --------------------------------------------------------------------------
  bool image_to_gray(const sensor_msgs::msg::Image & msg, cv::Mat & gray_out)
  {
    if (msg.encoding != "bgr8" && msg.encoding != "rgb8" && msg.encoding != "mono8") {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Unsupported image encoding: %s", msg.encoding.c_str());
      return false;
    }

    if (msg.height == 0 || msg.width == 0 || msg.data.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Received empty image.");
      return false;
    }

    if (msg.encoding == "mono8") {
      cv::Mat mono(
        static_cast<int>(msg.height),
        static_cast<int>(msg.width),
        CV_8UC1,
        const_cast<unsigned char *>(msg.data.data()),
        static_cast<size_t>(msg.step));
      gray_out = mono.clone();
      return true;
    }

    cv::Mat color(
      static_cast<int>(msg.height),
      static_cast<int>(msg.width),
      CV_8UC3,
      const_cast<unsigned char *>(msg.data.data()),
      static_cast<size_t>(msg.step));

    cv::Mat color_copy = color.clone();

    if (msg.encoding == "bgr8") {
      cv::cvtColor(color_copy, gray_out, cv::COLOR_BGR2GRAY);
    } else {
      cv::cvtColor(color_copy, gray_out, cv::COLOR_RGB2GRAY);
    }

    return true;
  }

  // --------------------------------------------------------------------------
  // Image callback
  // Main bootstrap visual odometry logic
  // --------------------------------------------------------------------------
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      // We need camera intrinsics and IMU before producing odometry
      if (!has_camera_info_ || !has_imu_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 3000,
          "Waiting for camera_info and imu before producing visual odometry.");
        return;
      }

      cv::Mat gray;
      if (!image_to_gray(*msg, gray)) {
        return;
      }

      const rclcpp::Time current_stamp(msg->header.stamp);

      // First valid frame: initialize tracking state and publish zero odometry
      if (!has_prev_image_) {
        prev_gray_ = gray.clone();
        prev_stamp_ = current_stamp;
        has_prev_image_ = true;

        publish_current_odom(current_stamp, 0.0, 0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Initialized first image for visual odometry.");
        return;
      }

      // Defensive checks against corrupted or inconsistent decoded frames
      if (prev_gray_.empty() || gray.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty image detected. Resetting visual tracking state.");
        prev_gray_ = gray.clone();
        prev_stamp_ = current_stamp;
        has_prev_image_ = true;
        publish_current_odom(current_stamp, 0.0, 0.0, 0.0);
        return;
      }

      // Important: Lucas-Kanade optical flow requires same image dimensions
      if (prev_gray_.size() != gray.size()) {
        RCLCPP_WARN(
          this->get_logger(),
          "Image size changed from %dx%d to %dx%d. Resetting visual tracking state.",
          prev_gray_.cols, prev_gray_.rows,
          gray.cols, gray.rows);

        prev_gray_ = gray.clone();
        prev_stamp_ = current_stamp;
        has_prev_image_ = true;
        publish_current_odom(current_stamp, 0.0, 0.0, 0.0);
        return;
      }

      const double dt = (current_stamp - prev_stamp_).seconds();

      // If the time gap is invalid or too large, we reset reference frame
      if (dt <= 0.0 || dt > max_dt_s_) {
        RCLCPP_WARN(
          this->get_logger(),
          "Invalid dt=%.6f s. Resetting previous frame reference.",
          dt);

        prev_gray_ = gray.clone();
        prev_stamp_ = current_stamp;
        publish_current_odom(current_stamp, 0.0, 0.0, 0.0);
        return;
      }

      // Detect corners/features on the previous image
      std::vector<cv::Point2f> prev_pts;
      cv::goodFeaturesToTrack(
        prev_gray_,
        prev_pts,
        max_features_,
        quality_level_,
        min_distance_);

      if (static_cast<int>(prev_pts.size()) < min_tracked_points_) {
        RCLCPP_WARN(
          this->get_logger(),
          "Not enough features in previous image: %zu < %d",
          prev_pts.size(),
          min_tracked_points_);

        prev_gray_ = gray.clone();
        prev_stamp_ = current_stamp;
        publish_current_odom(current_stamp, 0.0, 0.0, 0.0);
        return;
      }

      // Track features from previous image to current image
      std::vector<cv::Point2f> curr_pts;
      std::vector<unsigned char> status;
      std::vector<float> err;

      cv::calcOpticalFlowPyrLK(
        prev_gray_,
        gray,
        prev_pts,
        curr_pts,
        status,
        err);

      // Keep only successfully tracked points
      std::vector<double> dxs;
      std::vector<double> dys;
      dxs.reserve(prev_pts.size());
      dys.reserve(prev_pts.size());

      for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i]) {
          continue;
        }
        dxs.push_back(static_cast<double>(curr_pts[i].x - prev_pts[i].x));
        dys.push_back(static_cast<double>(curr_pts[i].y - prev_pts[i].y));
      }

      if (static_cast<int>(dxs.size()) < min_tracked_points_) {
        RCLCPP_WARN(
          this->get_logger(),
          "Not enough tracked points after LK: %zu < %d",
          dxs.size(),
          min_tracked_points_);

        prev_gray_ = gray.clone();
        prev_stamp_ = current_stamp;
        publish_current_odom(current_stamp, 0.0, 0.0, 0.0);
        return;
      }

      // Robust median flow estimate
      const double median_dx_px = median(dxs);
      const double median_dy_px = median(dys);

      // Approximate planar motion from image displacement.
      // This is a rough bootstrap heuristic, not a true metric VO model.
      const double forward_m = -median_dy_px * meters_per_pixel_;
      const double lateral_m = -median_dx_px * meters_per_pixel_ * 0.25;

      // Use IMU yaw as heading
      const double yaw = latest_yaw_;
      const double cos_yaw = std::cos(yaw);
      const double sin_yaw = std::sin(yaw);

      // Transform local planar motion into odom/world frame
      const double dx_world = forward_m * cos_yaw - lateral_m * sin_yaw;
      const double dy_world = forward_m * sin_yaw + lateral_m * cos_yaw;

      // Integrate state
      x_ += dx_world;
      y_ += dy_world;
      yaw_ = yaw;

      // Approximate velocities
      const double vx = dx_world / dt;
      const double vy = dy_world / dt;
      const double wz = normalize_angle(yaw_ - prev_yaw_) / dt;
      prev_yaw_ = yaw_;

      publish_current_odom(current_stamp, vx, vy, wz);

      RCLCPP_INFO(
        this->get_logger(),
        "Published visual odom | x=%.3f y=%.3f yaw=%.3f vx=%.3f vy=%.3f wz=%.3f tracked=%zu",
        x_, y_, yaw_, vx, vy, wz, dxs.size());

      // Update reference image
      prev_gray_ = gray.clone();
      prev_stamp_ = current_stamp;
    }
    catch (const cv::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "OpenCV exception in image_callback: %s", e.what());

      // Reset state using the current image if possible
      cv::Mat gray;
      if (image_to_gray(*msg, gray) && !gray.empty()) {
        prev_gray_ = gray.clone();
        prev_stamp_ = rclcpp::Time(msg->header.stamp);
        has_prev_image_ = true;
      }
    }
    catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "std::exception in image_callback: %s", e.what());
    }
    catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Unknown exception in image_callback.");
    }
  }

  // --------------------------------------------------------------------------
  // Publish odometry and optional TF
  // --------------------------------------------------------------------------
  void publish_current_odom(const rclcpp::Time & stamp, double vx, double vy, double wz)
  {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;

    odom.pose.pose.position.x = x_;
    odom.pose.pose.position.y = y_;
    odom.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw_);

    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.angular.z = wz;

    // Simple fixed covariance placeholders for bootstrap stage
    odom.pose.covariance[0] = 0.10;
    odom.pose.covariance[7] = 0.10;
    odom.pose.covariance[35] = 0.20;

    odom.twist.covariance[0] = 0.20;
    odom.twist.covariance[7] = 0.20;
    odom.twist.covariance[35] = 0.30;

    odom_pub_->publish(odom);

    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = stamp;
      tf_msg.header.frame_id = odom_frame_;
      tf_msg.child_frame_id = base_frame_;

      tf_msg.transform.translation.x = x_;
      tf_msg.transform.translation.y = y_;
      tf_msg.transform.translation.z = 0.0;

      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_msg.transform.rotation.w = q.w();

      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

private:
  // Topics / frames
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string odom_frame_;
  std::string base_frame_;

  // Parameters
  bool publish_tf_{true};
  int max_features_{100};
  double quality_level_{0.01};
  double min_distance_{8.0};
  double meters_per_pixel_{0.0025};
  int min_tracked_points_{10};
  double max_dt_s_{0.8};

  // QoS
  rclcpp::QoS sensor_qos_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Cached inputs
  sensor_msgs::msg::CameraInfo latest_camera_info_;
  bool has_camera_info_{false};

  bool has_imu_{false};
  double latest_yaw_{0.0};
  rclcpp::Time latest_imu_stamp_{0, 0, RCL_ROS_TIME};

  // Visual tracking state
  cv::Mat prev_gray_;
  bool has_prev_image_{false};
  rclcpp::Time prev_stamp_{0, 0, RCL_ROS_TIME};

  // Integrated planar state
  double x_{0.0};
  double y_{0.0};
  double yaw_{0.0};
  double prev_yaw_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<VisualOdometryBootstrapNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception & e) {
    std::fprintf(stderr, "visual_odometry_bootstrap_node exception: %s\n", e.what());
  }
  catch (...) {
    std::fprintf(stderr, "visual_odometry_bootstrap_node unknown exception\n");
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}