// Copyright 2026 Jean Mayoko
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "g1_social_costmap/social_layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cctype>

#include "geometry_msgs/msg/pose_stamped.hpp"

#include "pluginlib/class_list_macros.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <memory>

namespace g1_social_costmap
{



namespace
{

template<typename NodeT>
std::shared_ptr<NodeT> resolveLifecycleNode(
  const std::shared_ptr<NodeT> & node)
{
  return node;
}


template<typename NodeT>
std::shared_ptr<NodeT> resolveLifecycleNode(
  const std::weak_ptr<NodeT> & node)
{
  return node.lock();
}

}  // namespace

SocialLayer::SocialLayer()
: nav2_costmap_2d::CostmapLayer()
{
}


// ============================================================================
// INITIALIZATION
// ============================================================================

void SocialLayer::onInitialize()
{
  const auto node =
    resolveLifecycleNode(node_);

  if (!node) {
    throw std::runtime_error(
            "SocialLayer could not lock its lifecycle node.");
  }

  current_ = false;

  loadParameters();

  if (
    input_message_type_ !=
    "social_zone_array")
  {
    throw std::runtime_error(
            "SocialLayer currently supports only "
            "input_message_type='social_zone_array'. Received: " +
            input_message_type_);
  }

  validateSocialCostParameters(
    cost_parameters_);

  matchSize();

  createSubscription();

  current_ = true;

  RCLCPP_INFO(
    node->get_logger(),
    "SocialLayer initialized: topic=%s, frame=%s, "
    "timeout=%.3f s, QoS=%s/%d",
    social_zones_topic_.c_str(),
    layered_costmap_->getGlobalFrameID().c_str(),
    observation_timeout_sec_,
    subscription_qos_reliability_.c_str(),
    subscription_qos_depth_);
}


void SocialLayer::loadParameters()
{
  const auto node =
    resolveLifecycleNode(node_);

  if (!node) {
    throw std::runtime_error(
            "SocialLayer could not access its lifecycle node.");
  }

  declareParameter(
    "enabled",
    rclcpp::ParameterValue(
      true));

  declareParameter(
    "input_message_type",
    rclcpp::ParameterValue(
      std::string(
        "social_zone_array")));

  declareParameter(
    "social_zones_topic",
    rclcpp::ParameterValue(
      std::string(
        "/social_perception/social_zones")));

  declareParameter(
    "observation_timeout_sec",
    rclcpp::ParameterValue(
      0.80));

  declareParameter(
    "transform_tolerance_sec",
    rclcpp::ParameterValue(
      0.20));

  declareParameter(
    "include_tentative_tracks",
    rclcpp::ParameterValue(
      false));

  declareParameter(
    "include_lost_tracks",
    rclcpp::ParameterValue(
      false));

  declareParameter(
    "lethal_person_radius_m",
    rclcpp::ParameterValue(
      0.35));

  declareParameter(
    "maximum_social_cost",
    rclcpp::ParameterValue(
      252));

  declareParameter(
    "minimum_social_cost",
    rclcpp::ParameterValue(
      10));

  declareParameter(
    "gaussian_cutoff",
    rclcpp::ParameterValue(
      3.0));

  declareParameter(
    "social_cost_forward_only",
    rclcpp::ParameterValue(false));

  declareParameter(
    "subscription_qos_depth",
    rclcpp::ParameterValue(
      10));

  declareParameter(
    "subscription_qos_reliability",
    rclcpp::ParameterValue(
      std::string(
        "best_effort")));

  int maximum_social_cost = 252;
  int minimum_social_cost = 10;

  node->get_parameter(
    name_ + ".enabled",
    enabled_);

  node->get_parameter(
    name_ + ".input_message_type",
    input_message_type_);

  node->get_parameter(
    name_ + ".social_zones_topic",
    social_zones_topic_);

  node->get_parameter(
    name_ + ".observation_timeout_sec",
    observation_timeout_sec_);

  node->get_parameter(
    name_ + ".transform_tolerance_sec",
    transform_tolerance_sec_);

  node->get_parameter(
    name_ + ".include_tentative_tracks",
    include_tentative_tracks_);

  node->get_parameter(
    name_ + ".include_lost_tracks",
    include_lost_tracks_);

  node->get_parameter(
    name_ + ".lethal_person_radius_m",
    cost_parameters_.lethal_person_radius);

  node->get_parameter(
    name_ + ".maximum_social_cost",
    maximum_social_cost);

  node->get_parameter(
    name_ + ".minimum_social_cost",
    minimum_social_cost);

  node->get_parameter(
    name_ + ".gaussian_cutoff",
    cost_parameters_.gaussian_cutoff);

  node->get_parameter(
    name_ + ".social_cost_forward_only",
    social_cost_forward_only_);

  node->get_parameter(
    name_ + ".subscription_qos_depth",
    subscription_qos_depth_);

  node->get_parameter(
    name_ + ".subscription_qos_reliability",
    subscription_qos_reliability_);

  if (input_message_type_.empty()) {
    throw std::runtime_error(
            "SocialLayer input_message_type cannot be empty.");
  }

  if (social_zones_topic_.empty()) {
    throw std::runtime_error(
            "SocialLayer social_zones_topic cannot be empty.");
  }

  if (
    !std::isfinite(
      observation_timeout_sec_)
    ||
    observation_timeout_sec_ <= 0.0)
  {
    throw std::runtime_error(
            "SocialLayer observation_timeout_sec must be positive.");
  }

  if (
    !std::isfinite(
      transform_tolerance_sec_)
    ||
    transform_tolerance_sec_ < 0.0)
  {
    throw std::runtime_error(
            "SocialLayer transform_tolerance_sec must be non-negative.");
  }

  if (subscription_qos_depth_ < 1) {
    throw std::runtime_error(
            "SocialLayer subscription_qos_depth must be positive.");
  }

  if (
    maximum_social_cost < 1
    ||
    maximum_social_cost >
    static_cast<int>(
      MAXIMUM_SOCIAL_COST_LIMIT))
  {
    throw std::runtime_error(
            "SocialLayer maximum_social_cost must be in [1, 252].");
  }

  if (
    minimum_social_cost < 1
    ||
    minimum_social_cost >
    maximum_social_cost)
  {
    throw std::runtime_error(
            "SocialLayer minimum_social_cost must be in "
            "[1, maximum_social_cost].");
  }

  cost_parameters_.maximum_social_cost =
    static_cast<std::uint8_t>(
    maximum_social_cost);

  cost_parameters_.minimum_social_cost =
    static_cast<std::uint8_t>(
    minimum_social_cost);

  std::transform(
    subscription_qos_reliability_.begin(),
    subscription_qos_reliability_.end(),
    subscription_qos_reliability_.begin(),
    [](const unsigned char character) {
      return static_cast<char>(
        std::tolower(
          character));
    });

  if (
    subscription_qos_reliability_ !=
    "best_effort"
    &&
    subscription_qos_reliability_ !=
    "reliable")
  {
    throw std::runtime_error(
            "SocialLayer subscription_qos_reliability must be "
            "'best_effort' or 'reliable'.");
  }
}


void SocialLayer::createSubscription()
{
  const auto node =
    resolveLifecycleNode(node_);

  if (!node) {
    throw std::runtime_error(
            "SocialLayer could not create its subscription.");
  }

  rclcpp::QoS qos(
    rclcpp::KeepLast(
      static_cast<std::size_t>(
        subscription_qos_depth_)));

  qos.durability_volatile();

  if (
    subscription_qos_reliability_ ==
    "reliable")
  {
    qos.reliable();
  } else {
    qos.best_effort();
  }

  social_zones_subscription_ =
    node->create_subscription<SocialZoneArray>(
    social_zones_topic_,
    qos,
    std::bind(
      &SocialLayer::socialZonesCallback,
      this,
      std::placeholders::_1));
}


void SocialLayer::matchSize()
{
  CostmapLayer::matchSize();
}


// ============================================================================
// INPUT
// ============================================================================

void SocialLayer::socialZonesCallback(
  const SocialZoneArray::SharedPtr message)
{
  const auto node =
    resolveLifecycleNode(node_);

  if (!node || !message) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(
      message_mutex_);

    latest_message_ =
      message;

    latest_message_receipt_time_ =
      node->now();
  }

  current_ = true;
}


// ============================================================================
// TRACK FILTERING
// ============================================================================

bool SocialLayer::shouldIncludeTrackingState(
  const std::string & tracking_state) const
{
  if (
    tracking_state.empty()
    ||
    tracking_state == "TRACKED")
  {
    return true;
  }

  if (
    tracking_state == "TENTATIVE")
  {
    return include_tentative_tracks_;
  }

  if (
    tracking_state == "LOST")
  {
    return include_lost_tracks_;
  }

  return false;
}


// ============================================================================
// TRANSFORMATION
// ============================================================================

bool SocialLayer::prepareZoneInCostmapFrame(
  const g1_social_msgs::msg::SocialZone & source_zone,
  const std_msgs::msg::Header & source_header,
  ActiveZone & target_zone)
{
  const auto node =
    resolveLifecycleNode(node_);

  if (!node) {
    return false;
  }

  if (
    !shouldIncludeTrackingState(
      source_zone.tracking_state))
  {
    return false;
  }

  if (source_header.frame_id.empty()) {
    RCLCPP_WARN(
      node->get_logger(),
      "Ignoring SocialZoneArray with an empty frame_id.");

    return false;
  }

  geometry_msgs::msg::PoseStamped source_pose;
  geometry_msgs::msg::PoseStamped target_pose;

  source_pose.header =
    source_header;

  source_pose.pose.position =
    source_zone.center;

  tf2::Quaternion orientation;

  orientation.setRPY(
    0.0,
    0.0,
    static_cast<double>(
      source_zone.heading));

  source_pose.pose.orientation =
    tf2::toMsg(
    orientation);

  try {
    tf_->transform(
      source_pose,
      target_pose,
      layered_costmap_->getGlobalFrameID(),
      tf2::durationFromSec(
        transform_tolerance_sec_));
  } catch (
    const tf2::TransformException & error)
  {
    RCLCPP_WARN(
      node->get_logger(),
      "Could not transform social zone for person '%s' "
      "from '%s' to '%s': %s",
      source_zone.person_id.c_str(),
      source_header.frame_id.c_str(),
      layered_costmap_->getGlobalFrameID().c_str(),
      error.what());

    return false;
  }

  SocialZoneGeometry geometry;

  geometry.center_x =
    target_pose.pose.position.x;

  geometry.center_y =
    target_pose.pose.position.y;

  geometry.heading =
    tf2::getYaw(
    target_pose.pose.orientation);

  geometry.radius_longitudinal =
    static_cast<double>(
    source_zone.radius_longitudinal);

  geometry.radius_lateral =
    static_cast<double>(
    source_zone.radius_lateral);

  try {
    target_zone.prepared =
      prepareSocialZone(
      geometry,
      cost_parameters_);
  } catch (
    const std::invalid_argument & error)
  {
    RCLCPP_WARN(
      node->get_logger(),
      "Ignoring invalid social zone for person '%s': %s",
      source_zone.person_id.c_str(),
      error.what());

    return false;
  }

  // Cover both the Gaussian field and the physical exclusion disk.
  target_zone.influence_radius = std::max(
    cost_parameters_.lethal_person_radius,
    cost_parameters_.gaussian_cutoff *
    std::max(
      geometry.radius_longitudinal,
      geometry.radius_lateral));

  return true;
}


// ============================================================================
// BOUNDS
// ============================================================================

void SocialLayer::includeBounds(
  const double zone_minimum_x,
  const double zone_minimum_y,
  const double zone_maximum_x,
  const double zone_maximum_y,
  double * minimum_x,
  double * minimum_y,
  double * maximum_x,
  double * maximum_y)
{
  *minimum_x =
    std::min(
    *minimum_x,
    zone_minimum_x);

  *minimum_y =
    std::min(
    *minimum_y,
    zone_minimum_y);

  *maximum_x =
    std::max(
    *maximum_x,
    zone_maximum_x);

  *maximum_y =
    std::max(
    *maximum_y,
    zone_maximum_y);
}


void SocialLayer::updateBounds(
  const double robot_x,
  const double robot_y,
  const double robot_yaw,
  double * minimum_x,
  double * minimum_y,
  double * maximum_x,
  double * maximum_y)
{

  // Cache the robot pose in the same frame used by costmap cells.
  robot_pose_valid_ =
    std::isfinite(robot_x) &&
    std::isfinite(robot_y) &&
    std::isfinite(robot_yaw);

  if (robot_pose_valid_) {
    robot_position_x_ = robot_x;
    robot_position_y_ = robot_y;

    robot_heading_cos_ = std::cos(robot_yaw);
    robot_heading_sin_ = std::sin(robot_yaw);
  }

  if (
    minimum_x == nullptr
    ||
    minimum_y == nullptr
    ||
    maximum_x == nullptr
    ||
    maximum_y == nullptr)
  {
    return;
  }

  const auto node =
    resolveLifecycleNode(node_);

  if (!node) {
    current_ = false;
    return;
  }

  // Previous bounds must be included once so that removed or moved zones are
  // cleared from the master costmap.
  if (previous_bounds_valid_) {
    includeBounds(
      previous_minimum_x_,
      previous_minimum_y_,
      previous_maximum_x_,
      previous_maximum_y_,
      minimum_x,
      minimum_y,
      maximum_x,
      maximum_y);
  }

  active_zones_.clear();

  if (!enabled_) {
    previous_bounds_valid_ = false;
    current_ = true;
    return;
  }

  SocialZoneArray::SharedPtr message;
  rclcpp::Time receipt_time(
    0,
    0,
    node->get_clock()->get_clock_type());

  {
    std::lock_guard<std::mutex> lock(
      message_mutex_);

    message =
      latest_message_;

    if (message) {
      receipt_time =
        latest_message_receipt_time_;
    }
  }

  if (!message) {
    previous_bounds_valid_ = false;
    current_ = true;
    return;
  }

  const double message_age_sec =
    (
    node->now() -
    receipt_time
    ).seconds();

  if (
    !std::isfinite(
      message_age_sec)
    ||
    message_age_sec >
    observation_timeout_sec_)
  {
    previous_bounds_valid_ = false;
    current_ = true;
    return;
  }

  double current_minimum_x =
    std::numeric_limits<double>::infinity();

  double current_minimum_y =
    std::numeric_limits<double>::infinity();

  double current_maximum_x =
    -std::numeric_limits<double>::infinity();

  double current_maximum_y =
    -std::numeric_limits<double>::infinity();

  for (const auto & source_zone : message->zones) {
    ActiveZone active_zone;

    if (
      !prepareZoneInCostmapFrame(
        source_zone,
        message->header,
        active_zone))
    {
      continue;
    }

    const double center_x =
      active_zone.prepared.center_x;

    const double center_y =
      active_zone.prepared.center_y;

    const double radius =
      active_zone.influence_radius;

    current_minimum_x =
      std::min(
      current_minimum_x,
      center_x - radius);

    current_minimum_y =
      std::min(
      current_minimum_y,
      center_y - radius);

    current_maximum_x =
      std::max(
      current_maximum_x,
      center_x + radius);

    current_maximum_y =
      std::max(
      current_maximum_y,
      center_y + radius);

    active_zones_.push_back(
      std::move(
        active_zone));
  }

  if (active_zones_.empty()) {
    previous_bounds_valid_ = false;
    current_ = true;
    return;
  }

  includeBounds(
    current_minimum_x,
    current_minimum_y,
    current_maximum_x,
    current_maximum_y,
    minimum_x,
    minimum_y,
    maximum_x,
    maximum_y);

  previous_minimum_x_ =
    current_minimum_x;

  previous_minimum_y_ =
    current_minimum_y;

  previous_maximum_x_ =
    current_maximum_x;

  previous_maximum_y_ =
    current_maximum_y;

  previous_bounds_valid_ = true;
  current_ = true;
}


// ============================================================================
// COSTMAP UPDATE
// ============================================================================

void SocialLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  const int minimum_i,
  const int minimum_j,
  const int maximum_i,
  const int maximum_j)
{
  if (!enabled_) {
    return;
  }

  resetMaps();

  const int map_size_x =
    static_cast<int>(
    getSizeInCellsX());

  const int map_size_y =
    static_cast<int>(
    getSizeInCellsY());

  const int bounded_minimum_i =
    std::clamp(
    minimum_i,
    0,
    map_size_x);

  const int bounded_minimum_j =
    std::clamp(
    minimum_j,
    0,
    map_size_y);

  const int bounded_maximum_i =
    std::clamp(
    maximum_i,
    0,
    map_size_x);

  const int bounded_maximum_j =
    std::clamp(
    maximum_j,
    0,
    map_size_y);

  for (const auto & active_zone : active_zones_) {
    const double center_x =
      active_zone.prepared.center_x;

    const double center_y =
      active_zone.prepared.center_y;

    const double radius =
      active_zone.influence_radius;

    int zone_minimum_i = 0;
    int zone_minimum_j = 0;
    int zone_maximum_i = 0;
    int zone_maximum_j = 0;

    worldToMapEnforceBounds(
      center_x - radius,
      center_y - radius,
      zone_minimum_i,
      zone_minimum_j);

    worldToMapEnforceBounds(
      center_x + radius,
      center_y + radius,
      zone_maximum_i,
      zone_maximum_j);

    const int start_i =
      std::max(
      bounded_minimum_i,
      zone_minimum_i);

    const int start_j =
      std::max(
      bounded_minimum_j,
      zone_minimum_j);

    const int end_i =
      std::min(
      bounded_maximum_i,
      zone_maximum_i + 1);

    const int end_j =
      std::min(
      bounded_maximum_j,
      zone_maximum_j + 1);

    for (
      int map_y = start_j;
      map_y < end_j;
      ++map_y)
    {
      for (
        int map_x = start_i;
        map_x < end_i;
        ++map_x)
      {
        double world_x = 0.0;
        double world_y = 0.0;

        mapToWorld(
          static_cast<unsigned int>(
            map_x),
          static_cast<unsigned int>(
            map_y),
          world_x,
          world_y);

        const std::uint8_t new_cost =
          computeSocialCost(
          world_x,
          world_y,
          active_zone.prepared);


        // Clip only the non-lethal social field.
        // Keep the physical exclusion disk around each tracked person.
        if (
          social_cost_forward_only_ &&
          robot_pose_valid_ &&
          new_cost != LETHAL_OBSTACLE_COST)
        {
          const double delta_x =
            world_x - robot_position_x_;

          const double delta_y =
            world_y - robot_position_y_;

          // Project the cell onto the robot's forward axis:
          //   positive: ahead
          //   negative: behind
          const double forward_x =
            robot_heading_cos_ * delta_x +
            robot_heading_sin_ * delta_y;

          if (forward_x < 0.0) {
            continue;
          }
        }


        if (
          new_cost ==
          FREE_SPACE_COST)
        {
          continue;
        }

        const std::uint8_t previous_cost =
          getCost(
          static_cast<unsigned int>(
            map_x),
          static_cast<unsigned int>(
            map_y));

        if (new_cost > previous_cost) {
          setCost(
            static_cast<unsigned int>(
              map_x),
            static_cast<unsigned int>(
              map_y),
            new_cost);
        }
      }
    }
  }

  updateWithMax(
    master_grid,
    bounded_minimum_i,
    bounded_minimum_j,
    bounded_maximum_i,
    bounded_maximum_j);
}


// ============================================================================
// RESET
// ============================================================================

void SocialLayer::reset()
{
  {
    std::lock_guard<std::mutex> lock(
      message_mutex_);

    latest_message_.reset();

    latest_message_receipt_time_ =
      rclcpp::Time(
      0,
      0,
      RCL_ROS_TIME);
  }

  active_zones_.clear();

  previous_bounds_valid_ = false;

  resetMaps();

  current_ = true;
}


bool SocialLayer::isClearable()
{
  return true;
}

}  // namespace g1_social_costmap


PLUGINLIB_EXPORT_CLASS(
  g1_social_costmap::SocialLayer,
  nav2_costmap_2d::Layer)
