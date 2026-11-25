
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_filters/costmap_filter.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_msgs/msg/costmap_filter_info.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace g1_social_costmap
{

class DirectionalKeepoutFilter
  : public nav2_costmap_2d::CostmapFilter
{
public:
  void initializeFilter(
    const std::string & filter_info_topic) override
  {
    std::lock_guard<mutex_t> lock(*getMutex());

    auto node = node_.lock();
    if (!node) {
      throw std::runtime_error("Costmap lifecycle node unavailable");
    }

    declareParameter(
      "confirmation_topic",
      rclcpp::ParameterValue(std::string("")));

    node->get_parameter(
      name_ + ".confirmation_topic", confirmation_topic_);

    if (confirmation_topic_.empty()) {
      throw std::runtime_error(
        "Directional keepout confirmation_topic is required");
    }

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .reliable().transient_local();

    confirmation_pub_ =
      node->create_publisher<std_msgs::msg::String>(
        confirmation_topic_, rclcpp::QoS(10).reliable());

    info_sub_ =
      node->create_subscription<
        nav2_msgs::msg::CostmapFilterInfo>(
        filter_info_topic,
        qos,
        [this](
          nav2_msgs::msg::CostmapFilterInfo::SharedPtr msg)
        {
          std::lock_guard<mutex_t> guard(*getMutex());

          mask_sub_.reset();
          mask_.reset();
          last_confirmation_stamp_ = 0;

          if (
            msg->type != 0 ||
            msg->filter_mask_topic.empty() ||
            msg->base != 0.0f ||
            msg->multiplier != 1.0f)
          {
            RCLCPP_ERROR(
              logger_, "Invalid keepout filter information");
            return;
          }

          auto node = node_.lock();
          if (!node) {
            return;
          }

          mask_sub_ =
            node->create_subscription<
              nav_msgs::msg::OccupancyGrid>(
              msg->filter_mask_topic,
              rclcpp::QoS(rclcpp::KeepLast(1))
                .reliable().transient_local(),
              [this](
                nav_msgs::msg::OccupancyGrid::SharedPtr grid)
              {
                std::lock_guard<mutex_t> mask_guard(
                  *getMutex());

                const auto count =
                  static_cast<std::size_t>(grid->info.width) *
                  static_cast<std::size_t>(grid->info.height);

                if (
                  count == 0 ||
                  grid->data.size() != count ||
                  grid->header.frame_id.empty() ||
                  grid->info.resolution <= 0.0f)
                {
                  mask_.reset();
                  return;
                }

                mask_ = grid;
              });
        });

    RCLCPP_INFO(
      logger_,
      "DirectionalKeepoutFilter initialized: %s",
      confirmation_topic_.c_str());
  }

  void resetFilter() override
  {
    std::lock_guard<mutex_t> lock(*getMutex());
    info_sub_.reset();
    mask_sub_.reset();
    mask_.reset();
    confirmation_pub_.reset();
    last_confirmation_stamp_ = 0;
  }

  void process(
    nav2_costmap_2d::Costmap2D & master,
    int min_i,
    int min_j,
    int max_i,
    int max_j,
    const geometry_msgs::msg::Pose2D &) override
  {
    std::lock_guard<mutex_t> lock(*getMutex());

    if (!mask_ || !confirmation_pub_) {
      return;
    }

    auto node = node_.lock();
    if (!node) {
      return;
    }

    const auto stamp = rclcpp::Time(mask_->header.stamp);
    const double age = (node->now() - stamp).seconds();

    if (age < -0.1 || age > 1.0) {
      return;
    }

    tf2::Transform transform;
    transform.setIdentity();

    const auto costmap_frame =
      layered_costmap_->getGlobalFrameID();

    if (mask_->header.frame_id != costmap_frame) {
      try {
        auto tf = tf_->lookupTransform(
          mask_->header.frame_id,
          costmap_frame,
          tf2::TimePointZero,
          transform_tolerance_);

        tf2::fromMsg(tf.transform, transform);
      } catch (const tf2::TransformException &) {
        return;
      }
    }

    nav2_costmap_2d::Costmap2D mask_costmap(*mask_);

    const int width = static_cast<int>(master.getSizeInCellsX());
    const int height = static_cast<int>(master.getSizeInCellsY());

    const int start_x = std::max(0, min_i);
    const int start_y = std::max(0, min_j);
    const int end_x = std::min(width, max_i);
    const int end_y = std::min(height, max_j);

    std::size_t restricted_cells_processed = 0;

    for (int y = start_y; y < end_y; ++y) {
      for (int x = start_x; x < end_x; ++x) {
        double wx = 0.0;
        double wy = 0.0;

        master.mapToWorld(
          static_cast<unsigned int>(x),
          static_cast<unsigned int>(y),
          wx, wy);

        const tf2::Vector3 point =
          transform * tf2::Vector3(wx, wy, 0.0);

        unsigned int mx = 0;
        unsigned int my = 0;

        if (!mask_costmap.worldToMap(
            point.x(), point.y(), mx, my))
        {
          continue;
        }

        const unsigned char mask_cost =
          mask_costmap.getCost(mx, my);

        if (mask_cost == nav2_costmap_2d::NO_INFORMATION ||
            mask_cost == nav2_costmap_2d::FREE_SPACE)
        {
          continue;
        }

        const unsigned int cx =
          static_cast<unsigned int>(x);
        const unsigned int cy =
          static_cast<unsigned int>(y);

        const unsigned char existing =
          master.getCost(cx, cy);

        if (
          existing == nav2_costmap_2d::NO_INFORMATION ||
          mask_cost > existing)
        {
          master.setCost(cx, cy, mask_cost);
        }

        ++restricted_cells_processed;
      }
    }

    // No claim of successful application if the filtered
    // costmap window contains no restricted mask cell.
    if (restricted_cells_processed == 0) {
      return;
    }

    const std::int64_t stamp_ns =
      static_cast<std::int64_t>(mask_->header.stamp.sec)
        * 1000000000LL +
      static_cast<std::int64_t>(mask_->header.stamp.nanosec);

    if (stamp_ns <= 0) {
      return;
    }

    // FNV-1a fingerprint over the exact mask bytes.
    // A separate coordinator will associate this fingerprint
    // with the producer's stronger mask_id.
    std::uint64_t fingerprint =
      14695981039346656037ULL;

    for (const auto value : mask_->data) {
      fingerprint ^=
        static_cast<std::uint8_t>(value);
      fingerprint *= 1099511628211ULL;
    }

    if (stamp_ns == last_confirmation_stamp_) {
      return;
    }

    last_confirmation_stamp_ = stamp_ns;

    std::ostringstream hash;
    hash << std::hex << std::setfill('0')
         << std::setw(16) << fingerprint;

    std_msgs::msg::String evidence;

    std::ostringstream payload;
    payload
      << "{\"version\":1,"
      << "\"producer\":\"directional_keepout_filter\","
      << "\"mask_stamp_ns\":" << stamp_ns << ","
      << "\"mask_fingerprint\":\"" << hash.str() << "\","
      << "\"restricted_cells_processed\":"
      << restricted_cells_processed << "}";

    evidence.data = payload.str();
    confirmation_pub_->publish(evidence);
  }

private:
  std::string confirmation_topic_;

  std::int64_t last_confirmation_stamp_{0};

  nav_msgs::msg::OccupancyGrid::SharedPtr mask_;

  rclcpp::Subscription<
    nav2_msgs::msg::CostmapFilterInfo>::SharedPtr info_sub_;

  rclcpp::Subscription<
    nav_msgs::msg::OccupancyGrid>::SharedPtr mask_sub_;

  rclcpp::Publisher<
    std_msgs::msg::String>::SharedPtr confirmation_pub_;
};

}  // namespace g1_social_costmap

PLUGINLIB_EXPORT_CLASS(
  g1_social_costmap::DirectionalKeepoutFilter,
  nav2_costmap_2d::Layer)
