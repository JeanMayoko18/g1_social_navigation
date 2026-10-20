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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "g1_social_costmap/social_cost.hpp"

namespace g1_social_costmap
{

namespace
{

constexpr double PI =
  3.14159265358979323846;

constexpr double TOLERANCE =
  1.0e-9;


SocialZoneGeometry createDefaultGeometry()
{
  SocialZoneGeometry geometry;

  geometry.center_x = 0.0;
  geometry.center_y = 0.0;
  geometry.heading = 0.0;

  geometry.radius_longitudinal = 1.20;
  geometry.radius_lateral = 0.75;

  return geometry;
}


SocialCostParameters createDefaultParameters()
{
  SocialCostParameters parameters;

  parameters.lethal_person_radius = 0.35;
  parameters.maximum_social_cost = 252U;
  parameters.minimum_social_cost = 10U;
  parameters.gaussian_cutoff = 3.0;

  return parameters;
}

}  // namespace


// ============================================================================
// GEOMETRY VALIDATION
// ============================================================================

TEST(
  SocialCostValidation,
  RejectsZeroLongitudinalRadius)
{
  auto geometry =
    createDefaultGeometry();

  geometry.radius_longitudinal = 0.0;

  EXPECT_THROW(
    validateSocialZoneGeometry(
      geometry),
    std::invalid_argument);
}


TEST(
  SocialCostValidation,
  RejectsNegativeLateralRadius)
{
  auto geometry =
    createDefaultGeometry();

  geometry.radius_lateral = -0.1;

  EXPECT_THROW(
    validateSocialZoneGeometry(
      geometry),
    std::invalid_argument);
}


TEST(
  SocialCostValidation,
  RejectsNonFiniteHeading)
{
  auto geometry =
    createDefaultGeometry();

  geometry.heading =
    std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(
    validateSocialZoneGeometry(
      geometry),
    std::invalid_argument);
}


// ============================================================================
// PARAMETER VALIDATION
// ============================================================================

TEST(
  SocialCostValidation,
  RejectsNegativeLethalRadius)
{
  auto parameters =
    createDefaultParameters();

  parameters.lethal_person_radius = -0.1;

  EXPECT_THROW(
    validateSocialCostParameters(
      parameters),
    std::invalid_argument);
}


TEST(
  SocialCostValidation,
  RejectsMaximumCostAboveNav2Limit)
{
  auto parameters =
    createDefaultParameters();

  parameters.maximum_social_cost = 253U;

  EXPECT_THROW(
    validateSocialCostParameters(
      parameters),
    std::invalid_argument);
}


TEST(
  SocialCostValidation,
  RejectsMinimumCostAboveMaximum)
{
  auto parameters =
    createDefaultParameters();

  parameters.minimum_social_cost = 200U;
  parameters.maximum_social_cost = 100U;

  EXPECT_THROW(
    validateSocialCostParameters(
      parameters),
    std::invalid_argument);
}


TEST(
  SocialCostValidation,
  RejectsZeroGaussianCutoff)
{
  auto parameters =
    createDefaultParameters();

  parameters.gaussian_cutoff = 0.0;

  EXPECT_THROW(
    validateSocialCostParameters(
      parameters),
    std::invalid_argument);
}


// ============================================================================
// PREPARATION
// ============================================================================

TEST(
  SocialCostPreparation,
  PrecomputesExpectedValues)
{
  const auto geometry =
    createDefaultGeometry();

  const auto parameters =
    createDefaultParameters();

  const auto prepared =
    prepareSocialZone(
    geometry,
    parameters);

  EXPECT_NEAR(
    prepared.center_x,
    geometry.center_x,
    TOLERANCE);

  EXPECT_NEAR(
    prepared.center_y,
    geometry.center_y,
    TOLERANCE);

  EXPECT_NEAR(
    prepared.cos_heading,
    1.0,
    TOLERANCE);

  EXPECT_NEAR(
    prepared.sin_heading,
    0.0,
    TOLERANCE);

  EXPECT_NEAR(
    prepared.lethal_person_radius_squared,
    0.35 * 0.35,
    TOLERANCE);

  EXPECT_NEAR(
    prepared.gaussian_cutoff_squared,
    9.0,
    TOLERANCE);
}


// ============================================================================
// NORMALIZED ELLIPTICAL DISTANCE
// ============================================================================

TEST(
  SocialCostGeometry,
  ReturnsOneAtLongitudinalBoundary)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const double distance_squared =
    normalizedEllipticalDistanceSquared(
    1.20,
    0.0,
    prepared);

  EXPECT_NEAR(
    distance_squared,
    1.0,
    TOLERANCE);
}


TEST(
  SocialCostGeometry,
  ReturnsOneAtLateralBoundary)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const double distance_squared =
    normalizedEllipticalDistanceSquared(
    0.0,
    0.75,
    prepared);

  EXPECT_NEAR(
    distance_squared,
    1.0,
    TOLERANCE);
}


TEST(
  SocialCostGeometry,
  AppliesZoneHeading)
{
  auto geometry =
    createDefaultGeometry();

  geometry.heading =
    PI / 2.0;

  const auto prepared =
    prepareSocialZone(
    geometry,
    createDefaultParameters());

  const double distance_squared =
    normalizedEllipticalDistanceSquared(
    0.0,
    1.20,
    prepared);

  EXPECT_NEAR(
    distance_squared,
    1.0,
    TOLERANCE);
}


TEST(
  SocialCostGeometry,
  IsSymmetricAroundZoneCenter)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const double positive =
    normalizedEllipticalDistanceSquared(
    0.8,
    0.3,
    prepared);

  const double negative =
    normalizedEllipticalDistanceSquared(
    -0.8,
    -0.3,
    prepared);

  EXPECT_NEAR(
    positive,
    negative,
    TOLERANCE);
}


// ============================================================================
// COST CALCULATION
// ============================================================================

TEST(
  SocialCostCalculation,
  ReturnsLethalCostAtPersonCenter)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  EXPECT_EQ(
    computeSocialCost(
      0.0,
      0.0,
      prepared),
    LETHAL_OBSTACLE_COST);
}


TEST(
  SocialCostCalculation,
  ReturnsLethalCostInsidePhysicalRadius)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  EXPECT_EQ(
    computeSocialCost(
      0.20,
      0.0,
      prepared),
    LETHAL_OBSTACLE_COST);
}


TEST(
  SocialCostCalculation,
  ReturnsNonLethalCostOutsidePhysicalRadius)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const std::uint8_t cost =
    computeSocialCost(
    0.50,
    0.0,
    prepared);

  EXPECT_GT(
    cost,
    FREE_SPACE_COST);

  EXPECT_LE(
    cost,
    MAXIMUM_SOCIAL_COST_LIMIT);
}


TEST(
  SocialCostCalculation,
  LongitudinalInfluenceExceedsLateralInfluence)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const std::uint8_t longitudinal_cost =
    computeSocialCost(
    0.90,
    0.0,
    prepared);

  const std::uint8_t lateral_cost =
    computeSocialCost(
    0.0,
    0.90,
    prepared);

  EXPECT_GT(
    longitudinal_cost,
    lateral_cost);
}


TEST(
  SocialCostCalculation,
  CostDecreasesWithEllipticalDistance)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const std::uint8_t near_cost =
    computeSocialCost(
    0.50,
    0.0,
    prepared);

  const std::uint8_t far_cost =
    computeSocialCost(
    1.50,
    0.0,
    prepared);

  EXPECT_GT(
    near_cost,
    far_cost);
}


TEST(
  SocialCostCalculation,
  ReturnsFreeSpaceAtGaussianCutoff)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  const double cutoff_point =
    3.0 * 1.20;

  EXPECT_EQ(
    computeSocialCost(
      cutoff_point,
      0.0,
      prepared),
    FREE_SPACE_COST);
}


TEST(
  SocialCostCalculation,
  ReturnsFreeSpaceOutsideGaussianCutoff)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  EXPECT_EQ(
    computeSocialCost(
      4.0,
      0.0,
      prepared),
    FREE_SPACE_COST);
}


TEST(
  SocialCostCalculation,
  ReturnsFreeSpaceForNonFinitePoint)
{
  const auto prepared =
    prepareSocialZone(
    createDefaultGeometry(),
    createDefaultParameters());

  EXPECT_EQ(
    computeSocialCost(
      std::numeric_limits<double>::quiet_NaN(),
      0.0,
      prepared),
    FREE_SPACE_COST);
}


TEST(
  SocialCostCalculation,
  ProducesSameCostAfterEquivalentRotation)
{
  auto unrotated_geometry =
    createDefaultGeometry();

  auto rotated_geometry =
    createDefaultGeometry();

  rotated_geometry.heading =
    PI / 2.0;

  const auto unrotated =
    prepareSocialZone(
    unrotated_geometry,
    createDefaultParameters());

  const auto rotated =
    prepareSocialZone(
    rotated_geometry,
    createDefaultParameters());

  const std::uint8_t unrotated_cost =
    computeSocialCost(
    0.80,
    0.20,
    unrotated);

  const std::uint8_t rotated_cost =
    computeSocialCost(
    -0.20,
    0.80,
    rotated);

  EXPECT_EQ(
    unrotated_cost,
    rotated_cost);
}

}  // namespace g1_social_costmap
