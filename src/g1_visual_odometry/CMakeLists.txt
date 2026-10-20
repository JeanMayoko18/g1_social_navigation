cmake_minimum_required(VERSION 3.8)
project(g1_visual_odometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(visual_odometry_bootstrap_node
  src/visual_odometry_bootstrap_node.cpp
)

ament_target_dependencies(visual_odometry_bootstrap_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
)

add_executable(visual_odom_stub_node
  src/visual_odom_stub_node.cpp
)
add_executable(visual_odom_monitor_node
  src/visual_odom_monitor_node.cpp
)

add_executable(visual_inertial_buffer_node
  src/visual_inertial_buffer_node.cpp
)

ament_target_dependencies(visual_inertial_buffer_node
  rclcpp
  sensor_msgs
)
ament_target_dependencies(visual_odom_monitor_node
  rclcpp
  sensor_msgs
)

ament_target_dependencies(visual_odom_stub_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
)


target_include_directories(visual_odometry_bootstrap_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visual_odometry_bootstrap_node ${OpenCV_LIBS})

install(TARGETS
  visual_inertial_buffer_node
  visual_odom_stub_node
  visual_odom_monitor_node
  visual_odometry_bootstrap_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()