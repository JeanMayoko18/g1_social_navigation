
cmake_minimum_required(VERSION 3.8)
project(g1_social_msgs)

# =============================================================================
# COMPILER OPTIONS
# =============================================================================

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# DEPENDENCIES
# =============================================================================

find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# =============================================================================
# MESSAGE CATEGORIES
# =============================================================================

set(MSG_CATEGORIES
  perception
  navigation
  trajectory
  collaboration
  explainability
)

# ROSIDL expects interfaces under msg/ or srv/.
# Create a flat virtual msg/ directory in the build tree.
# Original files stay organized in the source tree.

set(MSG_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/msg")

file(MAKE_DIRECTORY "${MSG_BUILD_DIR}")

# Remove only previously generated symbolic links.
# This makes CMake reconfiguration idempotent.
file(GLOB OLD_MSG_LINKS
  "${MSG_BUILD_DIR}/*.msg"
)

foreach(OLD_LINK IN LISTS OLD_MSG_LINKS)
  if(IS_SYMLINK "${OLD_LINK}")
    file(REMOVE "${OLD_LINK}")
  endif()
endforeach()

set(MSG_FILES "")
set(MSG_NAMES "")

# =============================================================================
# DISCOVER MESSAGES AND CREATE SYMLINKS
# =============================================================================

foreach(CATEGORY IN LISTS MSG_CATEGORIES)

  set(CATEGORY_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/msg/${CATEGORY}"
  )

  if(NOT IS_DIRECTORY "${CATEGORY_DIR}")
    message(FATAL_ERROR
      "Missing message category: ${CATEGORY_DIR}"
    )
  endif()

  file(GLOB CATEGORY_MSG_FILES
    "${CATEGORY_DIR}/*.msg"
  )

  list(SORT CATEGORY_MSG_FILES)

  foreach(MSG_SOURCE IN LISTS CATEGORY_MSG_FILES)

    get_filename_component(MSG_NAME "${MSG_SOURCE}" NAME)

    # Message names must be globally unique.
    list(FIND MSG_NAMES "${MSG_NAME}" DUPLICATE_INDEX)

    if(NOT DUPLICATE_INDEX EQUAL -1)
      message(FATAL_ERROR
        "Duplicate message name: ${MSG_NAME}"
      )
    endif()

    list(APPEND MSG_NAMES "${MSG_NAME}")

    set(MSG_LINK "${MSG_BUILD_DIR}/${MSG_NAME}")

    if(EXISTS "${MSG_LINK}" OR IS_SYMLINK "${MSG_LINK}")
      message(FATAL_ERROR
        "Cannot create message link: ${MSG_LINK} already exists"
      )
    endif()

    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "${MSG_SOURCE}"
        "${MSG_LINK}"
      RESULT_VARIABLE LINK_RESULT
    )

    if(NOT LINK_RESULT EQUAL 0)
      message(FATAL_ERROR
        "Failed to create symbolic link: ${MSG_LINK}"
      )
    endif()

    # ROSIDL absolute-base:relative-path syntax.
    # Keeps generated ROS namespaces under msg/.
    list(APPEND MSG_FILES
      "${CMAKE_CURRENT_BINARY_DIR}:msg/${MSG_NAME}"
    )

    message(STATUS
      "[${CATEGORY}] ${MSG_NAME}"
    )

  endforeach()

endforeach()

list(SORT MSG_FILES)

list(LENGTH MSG_FILES MSG_COUNT)
message(STATUS "Discovered ${MSG_COUNT} ROS messages")

if(MSG_COUNT EQUAL 0)
  message(FATAL_ERROR "No ROS messages found")
endif()

# =============================================================================
# SERVICES
# =============================================================================

set(SRV_FILES
  "srv/SimulateHumanIntent.srv"
)

# =============================================================================
# ROS INTERFACES
# =============================================================================

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  ${SRV_FILES}

  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    std_msgs
    nav_msgs
)

# =============================================================================
# TESTING
# =============================================================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

# =============================================================================
# EXPORTS
# =============================================================================

ament_export_dependencies(
  rosidl_default_runtime
)

# =============================================================================
# PACKAGE
# =============================================================================

ament_package()
