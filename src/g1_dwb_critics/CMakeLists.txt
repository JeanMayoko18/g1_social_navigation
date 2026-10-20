cmake_minimum_required(VERSION 3.8)
project(g1_dwb_critics)

# Foxy and the current plugin implementation are compatible with C++14.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )
endif()

# =============================================================================
# Dependencies
# =============================================================================

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)

find_package(dwb_core REQUIRED)
find_package(dwb_msgs REQUIRED)

find_package(nav_2d_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)


# =============================================================================
# G1 DWB critic library
# =============================================================================

add_library(
  ${PROJECT_NAME}
  SHARED
  src/motion_feasibility_critic.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  pluginlib
  dwb_core
  dwb_msgs
  nav_2d_msgs
  geometry_msgs
  nav2_costmap_2d
  nav2_util
)


# =============================================================================
# Pluginlib registration
# =============================================================================

pluginlib_export_plugin_description_file(
  dwb_core
  g1_dwb_critics.xml
)


# =============================================================================
# Installation
# =============================================================================

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES
    g1_dwb_critics.xml
  DESTINATION
    share/${PROJECT_NAME}
)


# =============================================================================
# Package exports
# =============================================================================

ament_export_targets(
  export_${PROJECT_NAME}
  HAS_LIBRARY_TARGET
)

ament_export_dependencies(
  rclcpp
  pluginlib
  dwb_core
  dwb_msgs
  nav_2d_msgs
  geometry_msgs
  nav2_costmap_2d
  nav2_util
)


# =============================================================================
# Tests and linting
# =============================================================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Keep these enabled until copyright headers and cpplint requirements
  # are fully configured for this package.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()


# =============================================================================
# Ament package
# =============================================================================

ament_package()