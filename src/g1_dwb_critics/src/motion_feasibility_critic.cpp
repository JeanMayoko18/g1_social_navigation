#include "g1_dwb_critics/motion_feasibility_critic.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace g1_dwb_critics
{

void MotionFeasibilityCritic::declareDoubleParameter(
  const std::string & parameter_name,
  double default_value,
  double & destination)
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("DWB node is not available");
  }

  const std::string full_name =
    dwb_plugin_name_ + "." + name_ + "." + parameter_name;

  if (!node->has_parameter(full_name)) {
    node->declare_parameter(
      full_name,
      rclcpp::ParameterValue(default_value));
  }

  node->get_parameter(full_name, destination);
}


void MotionFeasibilityCritic::declareBoolParameter(
  const std::string & parameter_name,
  bool default_value,
  bool & destination)
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("DWB node is not available");
  }

  const std::string full_name =
    dwb_plugin_name_ + "." + name_ + "." + parameter_name;

  if (!node->has_parameter(full_name)) {
    node->declare_parameter(
      full_name,
      rclcpp::ParameterValue(default_value));
  }

  node->get_parameter(full_name, destination);
}


void MotionFeasibilityCritic::onInit()
{
  declareDoubleParameter(
    "minimum_effective_linear_speed",
    0.11,
    minimum_effective_linear_speed_);

  declareDoubleParameter(
    "linear_zero_threshold",
    0.01,
    linear_zero_threshold_);

  declareDoubleParameter(
    "angular_zero_threshold",
    0.02,
    angular_zero_threshold_);

  declareDoubleParameter(
    "final_rotation_distance",
    0.15,
    final_rotation_distance_);

  declareDoubleParameter(
    "path_heading_lookahead_distance",
    0.30,
    path_heading_lookahead_distance_);

  declareDoubleParameter(
    "large_heading_error",
    0.35,
    large_heading_error_);

  declareDoubleParameter(
    "pure_rotation_penalty",
    50.0,
    pure_rotation_penalty_);

  declareDoubleParameter(
    "necessary_rotation_penalty",
    5.0,
    necessary_rotation_penalty_);

  declareDoubleParameter(
    "curvature_penalty",
    10.0,
    curvature_penalty_);

  declareDoubleParameter(
    "direction_change_penalty",
    20.0,
    direction_change_penalty_);

  declareBoolParameter(
    "reject_low_speed_translation",
    true,
    reject_low_speed_translation_);

  if (minimum_effective_linear_speed_ <= linear_zero_threshold_) {
    throw std::runtime_error(
      "minimum_effective_linear_speed must be greater than "
      "linear_zero_threshold");
  }

  if (final_rotation_distance_ < 0.0) {
    throw std::runtime_error(
      "final_rotation_distance must be non-negative");
  }

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("DWB node is not available");
  }

  RCLCPP_INFO(
      node->get_logger(),
    "%s initialized | minimum_effective_linear_speed=%.3f m/s | "
    "final_rotation_distance=%.3f m | "
    "large_heading_error=%.3f rad",
    name_.c_str(),
    minimum_effective_linear_speed_,
    final_rotation_distance_,
    large_heading_error_);
}


void MotionFeasibilityCritic::reset()
{
  goal_distance_ = 0.0;
  path_heading_error_ = 0.0;
  previous_selected_wz_ = 0.0;
}


double MotionFeasibilityCritic::normalizeAngle(double angle)
{
  return std::atan2(
    std::sin(angle),
    std::cos(angle));
}


double MotionFeasibilityCritic::calculatePathHeadingError(
  const geometry_msgs::msg::Pose2D & pose,
  const nav_2d_msgs::msg::Path2D & global_plan) const
{
  if (global_plan.poses.empty()) {
    return 0.0;
  }

  const geometry_msgs::msg::Pose2D * target_pose =
    &global_plan.poses.back();

  for (const auto & candidate : global_plan.poses) {
    const double dx = candidate.x - pose.x;
    const double dy = candidate.y - pose.y;
    const double distance = std::hypot(dx, dy);

    if (distance >= path_heading_lookahead_distance_) {
      target_pose = &candidate;
      break;
    }
  }

  const double dx = target_pose->x - pose.x;
  const double dy = target_pose->y - pose.y;

  if (std::hypot(dx, dy) <= 1.0e-6) {
    return 0.0;
  }

  const double path_heading = std::atan2(dy, dx);

  return std::abs(
    normalizeAngle(path_heading - pose.theta));
}


bool MotionFeasibilityCritic::prepare(
  const geometry_msgs::msg::Pose2D & pose,
  const nav_2d_msgs::msg::Twist2D &,
  const geometry_msgs::msg::Pose2D & goal,
  const nav_2d_msgs::msg::Path2D & global_plan)
{
  goal_distance_ = std::hypot(
    goal.x - pose.x,
    goal.y - pose.y);

  path_heading_error_ = calculatePathHeadingError(
    pose,
    global_plan);

  return true;
}


double MotionFeasibilityCritic::scoreTrajectory(
  const dwb_msgs::msg::Trajectory2D & trajectory)
{
  const double vx = trajectory.velocity.x;
  const double vy = trajectory.velocity.y;
  const double wz = trajectory.velocity.theta;

  const double linear_speed = std::hypot(vx, vy);
  const double angular_speed = std::abs(wz);

  const bool has_translation =
    linear_speed > linear_zero_threshold_;

  const bool has_rotation =
    angular_speed > angular_zero_threshold_;

  /*
   * Reject commands that request translation below the physical G1
   * locomotion deadzone. A negative score marks the trajectory invalid.
   */
  if (
    reject_low_speed_translation_
    && has_translation
    && linear_speed < minimum_effective_linear_speed_)
  {
    return -1.0;
  }

  if (!has_rotation) {
    return 0.0;
  }

  double score = 0.0;

  /*
   * Preserve final in-place orientation control near the goal.
   * Direction-change cost remains active to reduce oscillation.
   */
  const bool near_goal =
    goal_distance_ <= final_rotation_distance_;

  if (!near_goal) {
    if (!has_translation) {
      /*
       * Pure rotation is heavily penalized when the current path already
       * lies reasonably close to the robot heading. When the heading error
       * is large, the lower necessary-rotation penalty is used.
       */
      const double alignment_ratio = std::min(
        path_heading_error_ /
        std::max(large_heading_error_, 1.0e-6),
        1.0);

      const double rotation_weight =
        pure_rotation_penalty_
        + alignment_ratio *
        (
          necessary_rotation_penalty_
          - pure_rotation_penalty_
        );

      score += rotation_weight * angular_speed;
    } else {
      /*
       * Penalize tight arcs. The ratio increases when angular velocity is
       * high relative to executable forward velocity.
       */
      score += curvature_penalty_ *
        angular_speed /
        std::max(linear_speed, minimum_effective_linear_speed_);
    }
  }

  /*
   * Penalize an immediate switch between clockwise and
   * counter-clockwise commands.
   */
  if (
    std::abs(previous_selected_wz_) > angular_zero_threshold_
    && wz * previous_selected_wz_ < 0.0)
  {
    score += direction_change_penalty_ *
      (
        angular_speed
        + std::abs(previous_selected_wz_)
      );
  }

  return score;
}


void MotionFeasibilityCritic::debrief(
  const nav_2d_msgs::msg::Twist2D & selected_velocity)
{
  previous_selected_wz_ = selected_velocity.theta;
}

}  // namespace g1_dwb_critics


PLUGINLIB_EXPORT_CLASS(
  g1_dwb_critics::MotionFeasibilityCritic,
  dwb_core::TrajectoryCritic)