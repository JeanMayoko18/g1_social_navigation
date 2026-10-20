#ifndef G1_DWB_CRITICS__MOTION_FEASIBILITY_CRITIC_HPP_
#define G1_DWB_CRITICS__MOTION_FEASIBILITY_CRITIC_HPP_

#include <string>

#include "dwb_core/trajectory_critic.hpp"
#include "dwb_msgs/msg/trajectory2_d.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav_2d_msgs/msg/path2_d.hpp"
#include "nav_2d_msgs/msg/twist2_d.hpp"

namespace g1_dwb_critics
{

class MotionFeasibilityCritic : public dwb_core::TrajectoryCritic
{
public:
  MotionFeasibilityCritic() = default;
  ~MotionFeasibilityCritic() override = default;

  void onInit() override;

  void reset() override;

  bool prepare(
    const geometry_msgs::msg::Pose2D & pose,
    const nav_2d_msgs::msg::Twist2D & velocity,
    const geometry_msgs::msg::Pose2D & goal,
    const nav_2d_msgs::msg::Path2D & global_plan) override;

  double scoreTrajectory(
    const dwb_msgs::msg::Trajectory2D & trajectory) override;

  void debrief(
    const nav_2d_msgs::msg::Twist2D & selected_velocity) override;

private:
  void declareDoubleParameter(
    const std::string & parameter_name,
    double default_value,
    double & destination);

  void declareBoolParameter(
    const std::string & parameter_name,
    bool default_value,
    bool & destination);

  static double normalizeAngle(double angle);

  double calculatePathHeadingError(
    const geometry_msgs::msg::Pose2D & pose,
    const nav_2d_msgs::msg::Path2D & global_plan) const;

  double minimum_effective_linear_speed_{0.11};
  double linear_zero_threshold_{0.01};
  double angular_zero_threshold_{0.02};

  double final_rotation_distance_{0.15};
  double path_heading_lookahead_distance_{0.30};
  double large_heading_error_{0.35};

  double pure_rotation_penalty_{50.0};
  double necessary_rotation_penalty_{5.0};
  double curvature_penalty_{10.0};
  double direction_change_penalty_{20.0};

  bool reject_low_speed_translation_{true};

  double goal_distance_{0.0};
  double path_heading_error_{0.0};
  double previous_selected_wz_{0.0};
};

}  // namespace g1_dwb_critics

#endif  // G1_DWB_CRITICS__MOTION_FEASIBILITY_CRITIC_HPP_