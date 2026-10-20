cmake_minimum_required(VERSION 3.8)

project(g1_unitree_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# ROS 2 dependencies
# -----------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(unitree_go REQUIRED)

# -----------------------------------------------------------------------------
# FFmpeg dependencies
# -----------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)

pkg_check_modules(
  FFMPEG REQUIRED
  libavcodec
  libavutil
  libswscale
)

# -----------------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------------
add_executable(
  front_video_bridge
  src/front_video_bridge.cpp
)

ament_target_dependencies(
  front_video_bridge
  rclcpp
  sensor_msgs
  unitree_go
)

add_executable(
  image_rate_probe
  src/image_rate_probe.cpp
)

ament_target_dependencies(
  image_rate_probe
  rclcpp
  sensor_msgs
)

target_include_directories(
  front_video_bridge
  PRIVATE
  ${FFMPEG_INCLUDE_DIRS}
)

target_link_directories(
  front_video_bridge
  PRIVATE
  ${FFMPEG_LIBRARY_DIRS}
)

target_link_libraries(
  front_video_bridge
  ${FFMPEG_LIBRARIES}
)




# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(
  TARGETS
  front_video_bridge
  image_rate_probe
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()