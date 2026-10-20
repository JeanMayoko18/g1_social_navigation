cmake_minimum_required(VERSION 3.5)

project(g1_social_costmap)


# =============================================================================
# COMPILER
# =============================================================================

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )
endif()


# =============================================================================
# DEPENDENCIES
# =============================================================================

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)

find_package(pluginlib REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

find_package(g1_social_msgs REQUIRED)


# =============================================================================
# LIBRARY
# =============================================================================

add_library(
  g1_social_costmap_layer
  SHARED

  src/social_cost.cpp
  src/social_layer.cpp
)

target_include_directories(
  g1_social_costmap_layer
  PUBLIC

  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  g1_social_costmap_layer

  rclcpp
  rclcpp_lifecycle

  nav2_costmap_2d
  nav2_util

  pluginlib

  tf2
  tf2_ros
  tf2_geometry_msgs

  geometry_msgs
  nav_msgs
  visualization_msgs
  diagnostic_msgs

  g1_social_msgs
)

pluginlib_export_plugin_description_file(
  nav2_costmap_2d
  plugins.xml
)


# =============================================================================
# INSTALLATION
# =============================================================================

install(
  TARGETS
    g1_social_costmap_layer

  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY
    include/

  DESTINATION include
)

install(
  FILES
    plugins.xml

  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY
    config/

  DESTINATION share/${PROJECT_NAME}/config
)


# =============================================================================
# TESTS
# =============================================================================

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_lint_common REQUIRED)

  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(
    test_social_cost
    test/test_social_cost.cpp
  )

  if(TARGET test_social_cost)
    target_link_libraries(
      test_social_cost
      g1_social_costmap_layer
    )

    ament_target_dependencies(
      test_social_cost
      geometry_msgs
    )
  endif()
endif()



# =============================================================================
# PACKAGE EXPORT
# =============================================================================

ament_export_include_directories(
  include
)

ament_export_libraries(
  g1_social_costmap_layer
)

ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  nav2_costmap_2d
  nav2_util
  pluginlib
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  diagnostic_msgs
  g1_social_msgs
)

ament_package()