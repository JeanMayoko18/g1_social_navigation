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

#ifndef G1_SOCIAL_COSTMAP__SOCIAL_LAYER_HPP_
#define G1_SOCIAL_COSTMAP__SOCIAL_LAYER_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "g1_social_msgs/msg/social_zone_array.hpp"

#include "nav2_costmap_2d/costmap_layer.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/subscription.hpp"

#include "g1_social_costmap/social_cost.hpp"

namespace g1_social_costmap
{

class SocialLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  SocialLayer();

  ~SocialLayer() override = default;

  /**
   * Declare parameters and create the social-zone subscription.
   */
  void onInitialize() override;

  /**
   * Resize this layer to match the master costmap.
   */
  void matchSize() override;

  /**
   * Extend the update window with current and previous social-zone bounds.
   */
  void updateBounds(
    double robot_x,
    double robot_y,
    double robot_yaw,
    double * minimum_x,
    double * minimum_y,
    double * maximum_x,
    double * maximum_y) override;

  /**
   * Rasterize current social zones and merge them into the master costmap.
   */
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int minimum_i,
    int minimum_j,
    int maximum_i,
    int maximum_j) override;

  /**
   * Clear all internal observations and costs.
   */
  void reset() override;

  /**
   * This layer can be cleared by Nav2.
   */
  bool isClearable();

private:
  struct ActiveZone
  {
    PreparedSocialZone prepared;
    double influence_radius{0.0};
  };

  using SocialZoneArray =
    g1_social_msgs::msg::SocialZoneArray;

  /**
   * Store the latest social-zone observation.
   */
  void socialZonesCallback(
    const SocialZoneArray::SharedPtr message);

  /**
   * Convert one incoming SocialZone into the costmap frame.
   */
  bool prepareZoneInCostmapFrame(
    const g1_social_msgs::msg::SocialZone & source_zone,
    const std_msgs::msg::Header & source_header,
    ActiveZone & target_zone);

  /**
   * Return true when one tracking state may affect navigation.
   */
  bool shouldIncludeTrackingState(
    const std::string & tracking_state) const;

  /**
   * Include one rectangular world-space area in update bounds.
   */
  static void includeBounds(
    double zone_minimum_x,
    double zone_minimum_y,
    double zone_maximum_x,
    double zone_maximum_y,
    double * minimum_x,
    double * minimum_y,
    double * maximum_x,
    double * maximum_y);

  /**
   * Read all layer parameters.
   */
  void loadParameters();

  /**
   * Create the subscription with the configured QoS.
   */
  void createSubscription();

  // ==========================================================================
  // CONFIGURATION
  // ==========================================================================

  std::string input_message_type_;
  std::string social_zones_topic_;

  double observation_timeout_sec_{0.80};
  double transform_tolerance_sec_{0.20};

  bool include_tentative_tracks_{false};
  bool include_lost_tracks_{false};
  
  // Restrict non-lethal social costs to the robot's forward half-plane.
  // Physical person exclusion remains active in every direction.
  bool social_cost_forward_only_{false};

  SocialCostParameters cost_parameters_;

  int subscription_qos_depth_{10};
  std::string subscription_qos_reliability_{"best_effort"};

  // ==========================================================================
  // INPUT STATE
  // ==========================================================================

  std::mutex message_mutex_;

  SocialZoneArray::SharedPtr latest_message_;

  rclcpp::Time latest_message_receipt_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<SocialZoneArray>::SharedPtr
    social_zones_subscription_;

  // ==========================================================================
  // CURRENT COSTMAP UPDATE
  // ==========================================================================

  std::vector<ActiveZone> active_zones_;

  // Robot pose expressed in the costmap frame.
  // Updated once per costmap cycle by updateBounds().
  double robot_position_x_{0.0};
  double robot_position_y_{0.0};
  double robot_heading_cos_{1.0};
  double robot_heading_sin_{0.0};

  // If the pose is invalid, preserve the full social field.
  bool robot_pose_valid_{false};

  bool previous_bounds_valid_{false};

  double previous_minimum_x_{0.0};
  double previous_minimum_y_{0.0};
  double previous_maximum_x_{0.0};
  double previous_maximum_y_{0.0};
};

}  // namespace g1_social_costmap

#endif  // G1_SOCIAL_COSTMAP__SOCIAL_LAYER_HPP_
