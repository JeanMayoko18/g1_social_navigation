#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"


class NavigationTimeSyncNode : public rclcpp::Node
{
public:
  NavigationTimeSyncNode()
  : Node("navigation_time_sync_node")
  {
    // ============================================================
    // 1. Input and output topics
    // ============================================================

    lidar_topic_ = declare_parameter<std::string>(
      "lidar_topic",
      "/utlidar/cloud_livox_mid360");

    dog_odom_topic_ = declare_parameter<std::string>(
      "dog_odom_topic",
      "/utlidar/dog_odom");

    lidar_to_nav_offset_topic_ =
      declare_parameter<std::string>(
        "lidar_to_nav_offset_topic",
        "/navigation_time/lidar_to_nav_offset");

    dog_to_nav_offset_topic_ =
      declare_parameter<std::string>(
        "dog_to_nav_offset_topic",
        "/navigation_time/dog_to_nav_offset");

    dog_to_lidar_offset_topic_ =
      declare_parameter<std::string>(
        "dog_to_lidar_offset_topic",
        "/navigation_time/dog_to_lidar_offset");

    sync_status_topic_ =
      declare_parameter<std::string>(
        "sync_status_topic",
        "/navigation_time/sync_status");


    // ============================================================
    // 2. LiDAR timestamp interpretation
    // ============================================================
    //
    // Supported values:
    //
    //   auto
    //     Use the per-point "time" field when available.
    //     Otherwise, use header.stamp.
    //
    //   point_time
    //     Require a valid per-point "time" field.
    //
    //   header_stamp
    //     Always use header.stamp as the cloud timestamp.
    //
    // Livox ROS Driver2 usually publishes PointCloud2 messages
    // without a per-point "time" field, so "auto" is appropriate.
    // ============================================================

    lidar_timestamp_mode_ =
      declare_parameter<std::string>(
        "lidar_timestamp_mode",
        "auto");

    if (
      lidar_timestamp_mode_ != "auto"
      && lidar_timestamp_mode_ != "point_time"
      && lidar_timestamp_mode_ != "header_stamp")
    {
      throw std::runtime_error(
        "Invalid lidar_timestamp_mode: "
        + lidar_timestamp_mode_);
    }


    // ============================================================
    // 3. Synchronization parameters
    // ============================================================

    // Number of observations required before locking an offset.
    warmup_samples_ =
      declare_parameter<int>(
        "warmup_samples",
        100);

    // Maximum number of recent observations retained.
    history_size_ =
      declare_parameter<int>(
        "history_size",
        500);

    // Maximum accepted median absolute deviation before an offset
    // is considered stable.
    stable_mad_threshold_sec_ =
      declare_parameter<double>(
        "stable_mad_threshold_sec",
        0.005);

    // Difference between the old and new halves of the Dog->LiDAR
    // history above which temporal drift is reported.
    drift_detection_threshold_sec_ =
      declare_parameter<double>(
        "drift_detection_threshold_sec",
        0.01);

    // Minimum number of Dog->LiDAR observations required before
    // drift monitoring becomes active.
    min_drift_samples_ =
      declare_parameter<int>(
        "min_drift_samples",
        100);

    // Frequency of status and diagnostic publication.
    diagnostics_frequency_hz_ =
      declare_parameter<double>(
        "diagnostics_frequency_hz",
        1.0);


    // ============================================================
    // 4. Parameter validation
    // ============================================================

    if (warmup_samples_ <= 0)
    {
      throw std::runtime_error(
        "warmup_samples must be greater than zero.");
    }

    if (history_size_ < warmup_samples_)
    {
      throw std::runtime_error(
        "history_size must be greater than or equal to "
        "warmup_samples.");
    }

    if (min_drift_samples_ <= 1)
    {
      throw std::runtime_error(
        "min_drift_samples must be greater than one.");
    }

    if (history_size_ < min_drift_samples_)
    {
      throw std::runtime_error(
        "history_size must be greater than or equal to "
        "min_drift_samples.");
    }

    if (stable_mad_threshold_sec_ < 0.0)
    {
      throw std::runtime_error(
        "stable_mad_threshold_sec cannot be negative.");
    }

    if (drift_detection_threshold_sec_ < 0.0)
    {
      throw std::runtime_error(
        "drift_detection_threshold_sec cannot be negative.");
    }


    // ============================================================
    // 5. QoS profiles
    // ============================================================

    const auto sensor_qos =
      rclcpp::SensorDataQoS();

    rclcpp::QoS latched_qos(
      rclcpp::KeepLast(1));

    latched_qos.reliable();
    latched_qos.transient_local();


    // ============================================================
    // 6. Publishers
    // ============================================================

    lidar_to_nav_pub_ =
      create_publisher<std_msgs::msg::Float64>(
        lidar_to_nav_offset_topic_,
        latched_qos);

    dog_to_nav_pub_ =
      create_publisher<std_msgs::msg::Float64>(
        dog_to_nav_offset_topic_,
        latched_qos);

    dog_to_lidar_pub_ =
      create_publisher<std_msgs::msg::Float64>(
        dog_to_lidar_offset_topic_,
        latched_qos);

    sync_status_pub_ =
      create_publisher<std_msgs::msg::String>(
        sync_status_topic_,
        latched_qos);


    // ============================================================
    // 7. Subscriptions
    // ============================================================

    lidar_sub_ =
      create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_,
        sensor_qos,
        std::bind(
          &NavigationTimeSyncNode::lidarCallback,
          this,
          std::placeholders::_1));

    dog_odom_sub_ =
      create_subscription<nav_msgs::msg::Odometry>(
        dog_odom_topic_,
        sensor_qos,
        std::bind(
          &NavigationTimeSyncNode::dogOdomCallback,
          this,
          std::placeholders::_1));


    // ============================================================
    // 8. Diagnostic timer
    // ============================================================

    const double safe_frequency_hz =
      std::max(
        diagnostics_frequency_hz_,
        0.1);

    diagnostics_timer_ =
      create_wall_timer(
        std::chrono::duration<double>(
          1.0 / safe_frequency_hz),
        std::bind(
          &NavigationTimeSyncNode::publishDiagnostics,
          this));


    // ============================================================
    // 9. Startup log
    // ============================================================

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "=========================================================\n"
      "           NAVIGATION TIME SYNC NODE\n"
      "=========================================================\n"
      " LiDAR topic               : %s\n"
      " Dog odom topic            : %s\n"
      " LiDAR timestamp mode      : %s\n"
      " Warmup samples            : %d\n"
      " History size              : %d\n"
      " Stable MAD threshold      : %.6f s\n"
      " Drift detection threshold : %.6f s\n"
      " Minimum drift samples     : %d\n"
      " Diagnostics frequency     : %.3f Hz\n"
      "=========================================================",
      lidar_topic_.c_str(),
      dog_odom_topic_.c_str(),
      lidar_timestamp_mode_.c_str(),
      warmup_samples_,
      history_size_,
      stable_mad_threshold_sec_,
      drift_detection_threshold_sec_,
      min_drift_samples_,
      safe_frequency_hz);
  }


private:
  // ==============================================================
  // Statistical utilities
  // ==============================================================

  static double median(
    std::vector<double> values)
  {
    if (values.empty())
    {
      return 0.0;
    }

    const size_t middle =
      values.size() / 2;

    std::nth_element(
      values.begin(),
      values.begin() + middle,
      values.end());

    double result =
      values[middle];

    if ((values.size() % 2) == 0)
    {
      const auto lower_max =
        std::max_element(
          values.begin(),
          values.begin() + middle);

      result =
        (
          result
          + *lower_max
        ) * 0.5;
    }

    return result;
  }


  static double medianAbsoluteDeviation(
    const std::vector<double> & values,
    const double center)
  {
    if (values.empty())
    {
      return 0.0;
    }

    std::vector<double> deviations;

    deviations.reserve(
      values.size());

    for (const double value : values)
    {
      deviations.push_back(
        std::abs(
          value - center));
    }

    return median(
      deviations);
  }


  template<typename T>
  static void trimDeque(
    std::deque<T> & buffer,
    const size_t max_size)
  {
    while (buffer.size() > max_size)
    {
      buffer.pop_front();
    }
  }


  // ==============================================================
  // PointCloud2 utilities
  // ==============================================================

  static int findFieldOffset(
    const sensor_msgs::msg::PointCloud2 & msg,
    const std::string & field_name)
  {
    for (const auto & field : msg.fields)
    {
      if (field.name == field_name)
      {
        return static_cast<int>(
          field.offset);
      }
    }

    return -1;
  }


  static float readFloat32(
    const std::vector<uint8_t> & data,
    const size_t index)
  {
    float value = 0.0f;

    std::memcpy(
      &value,
      &data[index],
      sizeof(float));

    return value;
  }


  // ==============================================================
  // Determine the maximum relative acquisition time in a cloud.
  //
  // For PointCloud2 messages with a valid float32 "time" field,
  // this function interprets the field as nanoseconds and returns
  // the latest point time in seconds.
  //
  // For Livox ROS Driver2 clouds without a "time" field:
  //
  //   auto         -> use header.stamp
  //   header_stamp -> use header.stamp
  //   point_time   -> reject the cloud
  // ==============================================================

  bool getMaxPointTimeSec(
    const sensor_msgs::msg::PointCloud2 & msg,
    double & max_time_sec) const
  {
    max_time_sec =
      0.0;

    if (lidar_timestamp_mode_ == "header_stamp")
    {
      return true;
    }

    const int time_offset =
      findFieldOffset(
        msg,
        "time");

    if (time_offset < 0)
    {
      if (lidar_timestamp_mode_ == "point_time")
      {
        return false;
      }

      // Auto mode falls back to header.stamp.
      return true;
    }

    const size_t point_step =
      static_cast<size_t>(
        msg.point_step);

    if (
      point_step == 0
      || static_cast<size_t>(time_offset) + sizeof(float)
        > point_step)
    {
      return lidar_timestamp_mode_ == "auto";
    }

    const size_t point_count =
      msg.data.size() / point_step;

    double detected_max_time_sec =
      -1.0;

    for (size_t i = 0; i < point_count; ++i)
    {
      const size_t index =
        i * point_step
        + static_cast<size_t>(time_offset);

      if (
        index + sizeof(float)
        > msg.data.size())
      {
        break;
      }

      const float raw_point_time =
        readFloat32(
          msg.data,
          index);

      if (!std::isfinite(raw_point_time))
      {
        continue;
      }

      // This implementation assumes the Livox-compatible convention
      // where the per-point relative time is expressed in nanoseconds.
      const double point_time_sec =
        static_cast<double>(
          raw_point_time)
        * 1.0e-9;

      if (point_time_sec < 0.0)
      {
        continue;
      }

      detected_max_time_sec =
        std::max(
          detected_max_time_sec,
          point_time_sec);
    }

    if (detected_max_time_sec >= 0.0)
    {
      max_time_sec =
        detected_max_time_sec;

      return true;
    }

    if (lidar_timestamp_mode_ == "auto")
    {
      max_time_sec =
        0.0;

      return true;
    }

    return false;
  }


  // ==============================================================
  // LiDAR callback
  //
  // Clock convention:
  //
  //   T_nav = T_lidar + lidar_to_nav_offset
  //
  // With the local ROS clock used as the navigation clock:
  //
  //   lidar_to_nav_offset
  //     = ROS reception time - LiDAR end-of-cloud time
  //
  // If no per-point time is available, the end-of-cloud time is
  // approximated by header.stamp.
  // ==============================================================

  void lidarCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    const rclcpp::Time reception_time =
      get_clock()->now();

    const rclcpp::Time source_stamp(
      msg->header.stamp);

    double max_point_time_sec =
      0.0;

    if (
      !getMaxPointTimeSec(
        *msg,
        max_point_time_sec))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Cannot estimate LiDAR offset: invalid PointCloud2 "
        "timing data for timestamp mode '%s'.",
        lidar_timestamp_mode_.c_str());

      return;
    }

    const rclcpp::Time lidar_end_stamp =
      source_stamp
      + rclcpp::Duration::from_seconds(
          max_point_time_sec);

    const double lidar_clock_offset_sec =
      (
        reception_time
        - lidar_end_stamp
      ).seconds();

    if (!std::isfinite(lidar_clock_offset_sec))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring non-finite LiDAR clock offset.");

      return;
    }

    std::lock_guard<std::mutex> lock(
      mutex_);

    // Continuous history used for Dog->LiDAR diagnostics and drift.
    lidar_clock_offset_samples_.push_back(
      lidar_clock_offset_sec);

    trimDeque(
      lidar_clock_offset_samples_,
      static_cast<size_t>(
        history_size_));

    // Locking history for LiDAR->Nav.
    //
    // Once locked, the original stable estimate is retained.
    if (!lidar_to_nav_offset_locked_)
    {
      lidar_to_nav_offset_samples_.push_back(
        lidar_clock_offset_sec);

      trimDeque(
        lidar_to_nav_offset_samples_,
        static_cast<size_t>(
          history_size_));

      tryLockLidarToNavOffset();
    }
  }


  // ==============================================================
  // Dog odometry callback
  //
  // Clock convention:
  //
  //   T_nav = T_dog + dog_to_nav_offset
  //
  // With the local ROS clock used as navigation time:
  //
  //   dog_to_nav_offset
  //     = ROS reception time - Dog source timestamp
  // ==============================================================

  void dogOdomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    const rclcpp::Time reception_time =
      get_clock()->now();

    const rclcpp::Time dog_stamp(
      msg->header.stamp);

    const double dog_clock_offset_sec =
      (
        reception_time
        - dog_stamp
      ).seconds();

    if (!std::isfinite(dog_clock_offset_sec))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring non-finite Dog clock offset.");

      return;
    }

    std::lock_guard<std::mutex> lock(
      mutex_);

    // Continuous history used for Dog->LiDAR diagnostics.
    dog_clock_offset_samples_.push_back(
      dog_clock_offset_sec);

    trimDeque(
      dog_clock_offset_samples_,
      static_cast<size_t>(
        history_size_));

    // Dog->Nav locking uses the same observations.
    if (!dog_to_nav_offset_locked_)
    {
      tryLockDogToNavOffset();
    }
  }


  // ==============================================================
  // Lock LiDAR -> Nav offset
  // ==============================================================

  void tryLockLidarToNavOffset()
  {
    if (
      lidar_to_nav_offset_locked_
      || static_cast<int>(
        lidar_to_nav_offset_samples_.size())
        < warmup_samples_)
    {
      return;
    }

    const std::vector<double> samples(
      lidar_to_nav_offset_samples_.begin(),
      lidar_to_nav_offset_samples_.end());

    const double center =
      median(
        samples);

    const double mad =
      medianAbsoluteDeviation(
        samples,
        center);

    if (mad <= stable_mad_threshold_sec_)
    {
      lidar_to_nav_offset_sec_ =
        center;

      lidar_to_nav_offset_locked_ =
        true;

      RCLCPP_INFO(
        get_logger(),
        "LiDAR -> Nav time offset LOCKED | "
        "offset=%+.9f s | MAD=%.9f s | samples=%zu",
        lidar_to_nav_offset_sec_,
        mad,
        samples.size());

      publishLockedOffsets();
    }
    else
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "LiDAR -> Nav offset not stable yet | "
        "median=%+.9f s | MAD=%.9f s | samples=%zu",
        center,
        mad,
        samples.size());
    }
  }


  // ==============================================================
  // Lock Dog -> Nav offset
  // ==============================================================

  void tryLockDogToNavOffset()
  {
    if (
      dog_to_nav_offset_locked_
      || static_cast<int>(
        dog_clock_offset_samples_.size())
        < warmup_samples_)
    {
      return;
    }

    const std::vector<double> samples(
      dog_clock_offset_samples_.begin(),
      dog_clock_offset_samples_.end());

    const double center =
      median(
        samples);

    const double mad =
      medianAbsoluteDeviation(
        samples,
        center);

    if (mad <= stable_mad_threshold_sec_)
    {
      dog_to_nav_offset_sec_ =
        center;

      dog_to_nav_offset_locked_ =
        true;

      RCLCPP_INFO(
        get_logger(),
        "Dog -> Nav time offset LOCKED | "
        "offset=%+.9f s | MAD=%.9f s | samples=%zu",
        dog_to_nav_offset_sec_,
        mad,
        samples.size());

      publishLockedOffsets();
    }
    else
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Dog -> Nav offset not stable yet | "
        "median=%+.9f s | MAD=%.9f s | samples=%zu",
        center,
        mad,
        samples.size());
    }
  }


  // ==============================================================
  // Lock Dog -> LiDAR offset
  //
  // Since:
  //
  //   T_nav = T_dog   + DogToNav
  //   T_nav = T_lidar + LidarToNav
  //
  // Then:
  //
  //   T_lidar = T_dog + DogToNav - LidarToNav
  //
  // Therefore:
  //
  //   DogToLidar = DogToNav - LidarToNav
  //
  // The observations are computed from continuously updated clock
  // medians in publishDiagnostics().
  // ==============================================================

  void tryLockDogToLidarOffset()
  {
    if (
      dog_to_lidar_offset_locked_
      || static_cast<int>(
        dog_to_lidar_offset_samples_.size())
        < warmup_samples_)
    {
      return;
    }

    const std::vector<double> samples(
      dog_to_lidar_offset_samples_.begin(),
      dog_to_lidar_offset_samples_.end());

    const double center =
      median(
        samples);

    const double mad =
      medianAbsoluteDeviation(
        samples,
        center);

    if (mad <= stable_mad_threshold_sec_)
    {
      dog_to_lidar_offset_sec_ =
        center;

      dog_to_lidar_offset_locked_ =
        true;

      RCLCPP_INFO(
        get_logger(),
        "Dog -> LiDAR time offset LOCKED | "
        "offset=%+.9f s | MAD=%.9f s | samples=%zu",
        dog_to_lidar_offset_sec_,
        mad,
        samples.size());

      publishLockedOffsets();
    }
    else
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Dog -> LiDAR offset not stable yet | "
        "median=%+.9f s | MAD=%.9f s | samples=%zu",
        center,
        mad,
        samples.size());
    }
  }


  // ==============================================================
  // Drift monitoring
  //
  // This function compares:
  //
  //   median(older half)
  //
  // with:
  //
  //   median(newer half)
  //
  // of the Dog->LiDAR history.
  //
  // It does not change any locked offset. It only updates the
  // diagnostic state.
  // ==============================================================

  void updateDriftStatus()
  {
    if (
      static_cast<int>(
        dog_to_lidar_offset_samples_.size())
      < min_drift_samples_)
    {
      drift_detected_ =
        false;

      observed_drift_sec_ =
        0.0;

      return;
    }

    const size_t sample_count =
      dog_to_lidar_offset_samples_.size();

    const size_t middle =
      sample_count / 2;

    if (middle == 0)
    {
      drift_detected_ =
        false;

      observed_drift_sec_ =
        0.0;

      return;
    }

    std::vector<double> older_samples;
    std::vector<double> newer_samples;

    older_samples.reserve(
      middle);

    newer_samples.reserve(
      sample_count - middle);

    for (size_t i = 0; i < sample_count; ++i)
    {
      if (i < middle)
      {
        older_samples.push_back(
          dog_to_lidar_offset_samples_[i]);
      }
      else
      {
        newer_samples.push_back(
          dog_to_lidar_offset_samples_[i]);
      }
    }

    const double older_median =
      median(
        older_samples);

    const double newer_median =
      median(
        newer_samples);

    observed_drift_sec_ =
      newer_median
      - older_median;

    drift_detected_ =
      std::abs(
        observed_drift_sec_)
      > drift_detection_threshold_sec_;
  }


  // ==============================================================
  // Publish locked offsets
  //
  // The publishers use transient-local durability, so a subscriber
  // joining later receives the most recently published value.
  // ==============================================================

  void publishLockedOffsets()
  {
    if (lidar_to_nav_offset_locked_)
    {
      std_msgs::msg::Float64 message;

      message.data =
        lidar_to_nav_offset_sec_;

      lidar_to_nav_pub_->publish(
        message);
    }

    if (dog_to_nav_offset_locked_)
    {
      std_msgs::msg::Float64 message;

      message.data =
        dog_to_nav_offset_sec_;

      dog_to_nav_pub_->publish(
        message);
    }

    if (dog_to_lidar_offset_locked_)
    {
      std_msgs::msg::Float64 message;

      message.data =
        dog_to_lidar_offset_sec_;

      dog_to_lidar_pub_->publish(
        message);
    }
  }


  // ==============================================================
  // Periodic diagnostics
  //
  // Every diagnostic cycle:
  //
  // 1. Compute the current Dog->ROS median.
  // 2. Compute the current LiDAR->ROS median.
  // 3. Derive one Dog->LiDAR observation.
  // 4. Try to lock Dog->LiDAR.
  // 5. Update drift status.
  // 6. Publish the global synchronization status.
  // ==============================================================

  void publishDiagnostics()
  {
    std::lock_guard<std::mutex> lock(
      mutex_);

    // Republish already locked values.
    publishLockedOffsets();


    // ------------------------------------------------------------
    // LiDAR -> Nav locking statistics
    // ------------------------------------------------------------

    double lidar_to_nav_median =
      0.0;

    double lidar_to_nav_mad =
      0.0;

    if (!lidar_to_nav_offset_samples_.empty())
    {
      const std::vector<double> values(
        lidar_to_nav_offset_samples_.begin(),
        lidar_to_nav_offset_samples_.end());

      lidar_to_nav_median =
        median(
          values);

      lidar_to_nav_mad =
        medianAbsoluteDeviation(
          values,
          lidar_to_nav_median);
    }


    // ------------------------------------------------------------
    // Continuous Dog and LiDAR clock statistics
    // ------------------------------------------------------------

    double dog_clock_median =
      0.0;

    double dog_clock_mad =
      0.0;

    double lidar_clock_median =
      0.0;

    double lidar_clock_mad =
      0.0;

    const bool dog_clock_available =
      !dog_clock_offset_samples_.empty();

    const bool lidar_clock_available =
      !lidar_clock_offset_samples_.empty();

    if (dog_clock_available)
    {
      const std::vector<double> values(
        dog_clock_offset_samples_.begin(),
        dog_clock_offset_samples_.end());

      dog_clock_median =
        median(
          values);

      dog_clock_mad =
        medianAbsoluteDeviation(
          values,
          dog_clock_median);
    }

    if (lidar_clock_available)
    {
      const std::vector<double> values(
        lidar_clock_offset_samples_.begin(),
        lidar_clock_offset_samples_.end());

      lidar_clock_median =
        median(
          values);

      lidar_clock_mad =
        medianAbsoluteDeviation(
          values,
          lidar_clock_median);
    }


    // ------------------------------------------------------------
    // Dog -> LiDAR observation
    //
    // DogToLidar
    //   = DogToNav - LidarToNav
    //
    // With the local ROS clock as Nav:
    //
    //   DogToLidar
    //     = (ROS - Dog) - (ROS - LiDAR)
    // ------------------------------------------------------------

    double observed_dog_to_lidar_offset_sec =
      0.0;

    if (
      dog_clock_available
      && lidar_clock_available)
    {
      observed_dog_to_lidar_offset_sec =
        dog_clock_median
        - lidar_clock_median;

      if (
        std::isfinite(
          observed_dog_to_lidar_offset_sec))
      {
        dog_to_lidar_offset_samples_.push_back(
          observed_dog_to_lidar_offset_sec);

        trimDeque(
          dog_to_lidar_offset_samples_,
          static_cast<size_t>(
            history_size_));

        if (!dog_to_lidar_offset_locked_)
        {
          tryLockDogToLidarOffset();
        }

        updateDriftStatus();
      }
      else
      {
        drift_detected_ =
          false;

        observed_drift_sec_ =
          0.0;
      }
    }
    else
    {
      drift_detected_ =
        false;

      observed_drift_sec_ =
        0.0;
    }


    // ------------------------------------------------------------
    // Global synchronization state
    // ------------------------------------------------------------

    std_msgs::msg::String status_message;

    if (
      !lidar_to_nav_offset_locked_
      || !dog_to_nav_offset_locked_
      || !dog_to_lidar_offset_locked_)
    {
      status_message.data =
        "COLLECTING";
    }
    else if (drift_detected_)
    {
      status_message.data =
        "DRIFT_DETECTED";
    }
    else
    {
      status_message.data =
        "OFFSET_STABLE";
    }

    sync_status_pub_->publish(
      status_message);


    // ------------------------------------------------------------
    // Diagnostic log
    // ------------------------------------------------------------

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "TIME SYNC | status=%s\n"
      "  LiDAR->Nav : locked=%s "
      "offset=%+.9f median=%+.9f MAD=%.9f "
      "lock_samples=%zu clock_samples=%zu\n"
      "  Dog->Nav   : locked=%s "
      "offset=%+.9f median=%+.9f MAD=%.9f "
      "samples=%zu\n"
      "  Dog->LiDAR : locked=%s "
      "offset=%+.9f observed=%+.9f "
      "drift=%+.9f samples=%zu\n"
      "  LiDAR clock: median=%+.9f MAD=%.9f samples=%zu",
      status_message.data.c_str(),

      lidar_to_nav_offset_locked_
        ? "yes"
        : "no",

      lidar_to_nav_offset_sec_,
      lidar_to_nav_median,
      lidar_to_nav_mad,
      lidar_to_nav_offset_samples_.size(),
      lidar_clock_offset_samples_.size(),

      dog_to_nav_offset_locked_
        ? "yes"
        : "no",

      dog_to_nav_offset_sec_,
      dog_clock_median,
      dog_clock_mad,
      dog_clock_offset_samples_.size(),

      dog_to_lidar_offset_locked_
        ? "yes"
        : "no",

      dog_to_lidar_offset_sec_,
      observed_dog_to_lidar_offset_sec,
      observed_drift_sec_,
      dog_to_lidar_offset_samples_.size(),

      lidar_clock_median,
      lidar_clock_mad,
      lidar_clock_offset_samples_.size());
  }


  // ==============================================================
  // Parameters
  // ==============================================================

  std::string lidar_topic_;
  std::string dog_odom_topic_;

  std::string lidar_timestamp_mode_;

  std::string lidar_to_nav_offset_topic_;
  std::string dog_to_nav_offset_topic_;
  std::string dog_to_lidar_offset_topic_;
  std::string sync_status_topic_;

  int warmup_samples_{100};
  int history_size_{500};
  int min_drift_samples_{100};

  double stable_mad_threshold_sec_{0.005};
  double drift_detection_threshold_sec_{0.01};
  double diagnostics_frequency_hz_{1.0};


  // ==============================================================
  // Synchronization state
  // ==============================================================

  std::mutex mutex_;

  // LiDAR->Nav observations retained until the offset is locked.
  std::deque<double>
    lidar_to_nav_offset_samples_;

  // Continuous Dog->ROS clock observations.
  std::deque<double>
    dog_clock_offset_samples_;

  // Continuous LiDAR->ROS clock observations.
  std::deque<double>
    lidar_clock_offset_samples_;

  // Dog->LiDAR observations generated by the diagnostic timer.
  std::deque<double>
    dog_to_lidar_offset_samples_;


  // Independent lock states.
  bool lidar_to_nav_offset_locked_{
    false};

  bool dog_to_nav_offset_locked_{
    false};

  bool dog_to_lidar_offset_locked_{
    false};

  bool drift_detected_{
    false};


  // Locked offsets.
  double lidar_to_nav_offset_sec_{
    0.0};

  double dog_to_nav_offset_sec_{
    0.0};

  double dog_to_lidar_offset_sec_{
    0.0};


  // Diagnostic drift between the old and new history halves.
  double observed_drift_sec_{
    0.0};


  // ==============================================================
  // ROS interfaces
  // ==============================================================

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr
    lidar_sub_;

  rclcpp::Subscription<
    nav_msgs::msg::Odometry>::SharedPtr
    dog_odom_sub_;

  rclcpp::Publisher<
    std_msgs::msg::Float64>::SharedPtr
    lidar_to_nav_pub_;

  rclcpp::Publisher<
    std_msgs::msg::Float64>::SharedPtr
    dog_to_nav_pub_;

  rclcpp::Publisher<
    std_msgs::msg::Float64>::SharedPtr
    dog_to_lidar_pub_;

  rclcpp::Publisher<
    std_msgs::msg::String>::SharedPtr
    sync_status_pub_;

  rclcpp::TimerBase::SharedPtr
    diagnostics_timer_;
};


int main(
  int argc,
  char ** argv)
{
  try
  {
    rclcpp::init(
      argc,
      argv);

    const auto node =
      std::make_shared<
        NavigationTimeSyncNode>();

    rclcpp::spin(
      node);

    rclcpp::shutdown();

    return 0;
  }
  catch (const std::exception & exception)
  {
    std::cerr
      << "[navigation_time_sync_node] Fatal exception: "
      << exception.what()
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
      << "[navigation_time_sync_node] "
      << "Unknown fatal exception"
      << std::endl;

    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }

    return 1;
  }
}