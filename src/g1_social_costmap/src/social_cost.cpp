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

#include "g1_social_costmap/social_cost.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace g1_social_costmap
{

namespace
{

bool isFinite(
  const double value) noexcept
{
  return std::isfinite(value);
}


void requireFinite(
  const double value,
  const char * const label)
{
  if (!isFinite(value)) {
    throw std::invalid_argument(
            std::string(label) +
            " must be finite.");
  }
}


void requireStrictlyPositive(
  const double value,
  const char * const label)
{
  requireFinite(
    value,
    label);

  if (value <= 0.0) {
    throw std::invalid_argument(
            std::string(label) +
            " must be strictly positive.");
  }
}


void requireNonNegative(
  const double value,
  const char * const label)
{
  requireFinite(
    value,
    label);

  if (value < 0.0) {
    throw std::invalid_argument(
            std::string(label) +
            " must be non-negative.");
  }
}

}  // namespace


// ============================================================================
// VALIDATION
// ============================================================================

void validateSocialZoneGeometry(
  const SocialZoneGeometry & geometry)
{
  requireFinite(
    geometry.center_x,
    "Social-zone center_x");

  requireFinite(
    geometry.center_y,
    "Social-zone center_y");

  requireFinite(
    geometry.heading,
    "Social-zone heading");

  requireStrictlyPositive(
    geometry.radius_longitudinal,
    "Social-zone longitudinal radius");

  requireStrictlyPositive(
    geometry.radius_lateral,
    "Social-zone lateral radius");
}


void validateSocialCostParameters(
  const SocialCostParameters & parameters)
{
  requireNonNegative(
    parameters.lethal_person_radius,
    "Lethal person radius");

  requireStrictlyPositive(
    parameters.gaussian_cutoff,
    "Gaussian cutoff");

  if (
    parameters.maximum_social_cost >
    MAXIMUM_SOCIAL_COST_LIMIT)
  {
    throw std::invalid_argument(
            "Maximum social cost must be less than or equal to 252.");
  }

  if (
    parameters.maximum_social_cost ==
    FREE_SPACE_COST)
  {
    throw std::invalid_argument(
            "Maximum social cost must be greater than zero.");
  }

  if (
    parameters.minimum_social_cost ==
    FREE_SPACE_COST)
  {
    throw std::invalid_argument(
            "Minimum social cost must be greater than zero.");
  }

  if (
    parameters.minimum_social_cost >
    parameters.maximum_social_cost)
  {
    throw std::invalid_argument(
            "Minimum social cost must not exceed maximum social cost.");
  }
}


// ============================================================================
// PREPARATION
// ============================================================================

PreparedSocialZone prepareSocialZone(
  const SocialZoneGeometry & geometry,
  const SocialCostParameters & parameters)
{
  validateSocialZoneGeometry(
    geometry);

  validateSocialCostParameters(
    parameters);

  PreparedSocialZone prepared;

  prepared.center_x =
    geometry.center_x;

  prepared.center_y =
    geometry.center_y;

  prepared.cos_heading =
    std::cos(
    geometry.heading);

  prepared.sin_heading =
    std::sin(
    geometry.heading);

  prepared.inverse_longitudinal_radius_squared =
    1.0 /
    (
    geometry.radius_longitudinal *
    geometry.radius_longitudinal
    );

  prepared.inverse_lateral_radius_squared =
    1.0 /
    (
    geometry.radius_lateral *
    geometry.radius_lateral
    );

  prepared.lethal_person_radius_squared =
    parameters.lethal_person_radius *
    parameters.lethal_person_radius;

  prepared.gaussian_cutoff_squared =
    parameters.gaussian_cutoff *
    parameters.gaussian_cutoff;

  prepared.maximum_social_cost =
    parameters.maximum_social_cost;

  prepared.minimum_social_cost =
    parameters.minimum_social_cost;

  return prepared;
}


// ============================================================================
// GEOMETRY
// ============================================================================

double euclideanDistanceSquared(
  const double point_x,
  const double point_y,
  const PreparedSocialZone & zone) noexcept
{
  const double delta_x =
    point_x -
    zone.center_x;

  const double delta_y =
    point_y -
    zone.center_y;

  return
    delta_x * delta_x
    +
    delta_y * delta_y
  ;
}


double normalizedEllipticalDistanceSquared(
  const double point_x,
  const double point_y,
  const PreparedSocialZone & zone) noexcept
{
  const double delta_x =
    point_x -
    zone.center_x;

  const double delta_y =
    point_y -
    zone.center_y;

  const double longitudinal =
    zone.cos_heading * delta_x
    +
    zone.sin_heading * delta_y;

  const double lateral =
    -zone.sin_heading * delta_x
    +
    zone.cos_heading * delta_y;

  return
    longitudinal *
    longitudinal *
    zone.inverse_longitudinal_radius_squared
    +
    lateral *
    lateral *
    zone.inverse_lateral_radius_squared
  ;
}


// ============================================================================
// COST
// ============================================================================

std::uint8_t computeSocialCost(
  const double point_x,
  const double point_y,
  const PreparedSocialZone & zone) noexcept
{
  if (
    !isFinite(point_x)
    ||
    !isFinite(point_y))
  {
    return FREE_SPACE_COST;
  }

  const double physical_distance_squared =
    euclideanDistanceSquared(
    point_x,
    point_y,
    zone);

  // Keep a hard physical exclusion area around the person.
  if (
    physical_distance_squared <=
    zone.lethal_person_radius_squared)
  {
    return LETHAL_OBSTACLE_COST;
  }

  const double normalized_distance_squared =
    normalizedEllipticalDistanceSquared(
    point_x,
    point_y,
    zone);

  const double cutoff_tolerance =
    16.0
    *
    std::numeric_limits<double>::epsilon()
    *
    std::max(
    1.0,
    zone.gaussian_cutoff_squared);

  // Cells outside the configured Gaussian support do not contribute
  // any social cost.
  if (
    !isFinite(
      normalized_distance_squared)
    ||
    normalized_distance_squared >=
    (
      zone.gaussian_cutoff_squared
      -
      cutoff_tolerance
    ))
  {
    return FREE_SPACE_COST;
  }

  const double unbounded_cost =
    static_cast<double>(
    zone.maximum_social_cost)
    *
    std::exp(
    -0.5 *
    normalized_distance_squared);

  if (
    !isFinite(
      unbounded_cost)
    ||
    unbounded_cost <
    static_cast<double>(
      zone.minimum_social_cost))
  {
    return FREE_SPACE_COST;
  }

  const double rounded_cost =
    std::round(
    unbounded_cost);

  const double final_cost =
    std::clamp(
    rounded_cost,
    static_cast<double>(
      zone.minimum_social_cost),
    static_cast<double>(
      zone.maximum_social_cost));

  return static_cast<std::uint8_t>(
    final_cost);
}

}  // namespace g1_social_costmap
