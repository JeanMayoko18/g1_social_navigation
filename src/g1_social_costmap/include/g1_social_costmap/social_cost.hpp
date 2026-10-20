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

#ifndef G1_SOCIAL_COSTMAP__SOCIAL_COST_HPP_
#define G1_SOCIAL_COSTMAP__SOCIAL_COST_HPP_

#include <cstdint>

namespace g1_social_costmap
{

// ============================================================================
// NAV2 COST VALUES
// ============================================================================
//
// These values match the standard nav2_costmap_2d cost representation while
// keeping this mathematical module independent from Nav2 headers.
// ============================================================================

inline constexpr std::uint8_t FREE_SPACE_COST = 0U;
inline constexpr std::uint8_t MAXIMUM_SOCIAL_COST_LIMIT = 252U;
inline constexpr std::uint8_t INSCRIBED_OBSTACLE_COST = 253U;
inline constexpr std::uint8_t LETHAL_OBSTACLE_COST = 254U;
inline constexpr std::uint8_t NO_INFORMATION_COST = 255U;


// ============================================================================
// INPUT MODELS
// ============================================================================

struct SocialZoneGeometry
{
  /**
   * Center of the social zone in the costmap frame.
   */
  double center_x{0.0};
  double center_y{0.0};

  /**
   * Zone orientation in radians.
   *
   * The longitudinal axis follows this heading.
   */
  double heading{0.0};

  /**
   * Longitudinal and lateral ellipse radii in meters.
   *
   * In the current g1_social_msgs/SocialZone model, the longitudinal radius
   * is symmetric in front of and behind the person.
   */
  double radius_longitudinal{0.0};
  double radius_lateral{0.0};
};


struct SocialCostParameters
{
  /**
   * Circular physical exclusion radius around the person.
   *
   * Cells inside this radius receive LETHAL_OBSTACLE_COST.
   */
  double lethal_person_radius{0.35};

  /**
   * Maximum non-lethal social cost.
   *
   * This value must remain below INSCRIBED_OBSTACLE_COST.
   */
  std::uint8_t maximum_social_cost{MAXIMUM_SOCIAL_COST_LIMIT};

  /**
   * Minimum non-zero cost applied inside the Gaussian cutoff.
   */
  std::uint8_t minimum_social_cost{10U};

  /**
   * Maximum normalized elliptical distance evaluated by the layer.
   *
   * A value of 1.0 represents the declared ellipse boundary.
   * A value of 3.0 extends the Gaussian influence to three normalized radii.
   */
  double gaussian_cutoff{3.0};
};


// ============================================================================
// PREPARED ZONE
// ============================================================================
//
// A zone is prepared once per incoming social-zone observation.
//
// This avoids recalculating trigonometric values and inverse squared radii for
// every costmap cell.
// ============================================================================

struct PreparedSocialZone
{
  double center_x{0.0};
  double center_y{0.0};

  double cos_heading{1.0};
  double sin_heading{0.0};

  double inverse_longitudinal_radius_squared{0.0};
  double inverse_lateral_radius_squared{0.0};

  double lethal_person_radius_squared{0.0};
  double gaussian_cutoff_squared{0.0};

  std::uint8_t maximum_social_cost{MAXIMUM_SOCIAL_COST_LIMIT};
  std::uint8_t minimum_social_cost{10U};
};


// ============================================================================
// VALIDATION
// ============================================================================

/**
 * Validate one social-zone geometry.
 *
 * Throws:
 *   std::invalid_argument when one value is non-finite or when one radius is
 *   not strictly positive.
 */
void validateSocialZoneGeometry(
  const SocialZoneGeometry & geometry);


/**
 * Validate social-cost parameters.
 *
 * Throws:
 *   std::invalid_argument when the configuration is invalid.
 */
void validateSocialCostParameters(
  const SocialCostParameters & parameters);


// ============================================================================
// PREPARATION
// ============================================================================

/**
 * Validate and prepare one social zone for repeated cell evaluation.
 */
PreparedSocialZone prepareSocialZone(
  const SocialZoneGeometry & geometry,
  const SocialCostParameters & parameters);


// ============================================================================
// GEOMETRY
// ============================================================================

/**
 * Compute the squared Euclidean distance from a point to the person.
 */
double euclideanDistanceSquared(
  double point_x,
  double point_y,
  const PreparedSocialZone & zone) noexcept;


/**
 * Compute the normalized squared elliptical distance.
 *
 * The point is first transformed into the oriented social-zone frame:
 *
 *   longitudinal =
 *       cos(theta) * dx + sin(theta) * dy
 *
 *   lateral =
 *      -sin(theta) * dx + cos(theta) * dy
 *
 * The returned value is:
 *
 *   d² =
 *       longitudinal² / radius_longitudinal²
 *       +
 *       lateral² / radius_lateral²
 *
 * Interpretation:
 *
 *   d² < 1.0 : inside the declared social ellipse
 *   d² = 1.0 : on the ellipse boundary
 *   d² > 1.0 : outside the ellipse
 */
double normalizedEllipticalDistanceSquared(
  double point_x,
  double point_y,
  const PreparedSocialZone & zone) noexcept;


// ============================================================================
// COST
// ============================================================================

/**
 * Compute the cost associated with one point.
 *
 * Rules:
 *
 *   1. Non-finite point coordinates return FREE_SPACE_COST.
 *   2. The physical exclusion radius receives LETHAL_OBSTACLE_COST.
 *   3. Points outside gaussian_cutoff receive FREE_SPACE_COST.
 *   4. Remaining points receive a Gaussian social cost:
 *
 *        cost = maximum_social_cost * exp(-0.5 * d²)
 *
 *   5. A non-zero cost is clamped between minimum_social_cost and
 *      maximum_social_cost.
 */
std::uint8_t computeSocialCost(
  double point_x,
  double point_y,
  const PreparedSocialZone & zone) noexcept;

}  // namespace g1_social_costmap

#endif  // G1_SOCIAL_COSTMAP__SOCIAL_COST_HPP_
