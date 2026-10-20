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
# ROS INTERFACES
# =============================================================================

rosidl_generate_interfaces(${PROJECT_NAME}

  # ---------------------------------------------------------------------------
  # PERCEPTION / HUMAN TRACKING
  # ---------------------------------------------------------------------------

  "msg/TrackedPerson.msg"
  "msg/TrackedPersonArray.msg"

  "msg/SocialZone.msg"
  "msg/SocialZoneArray.msg"

  "msg/SpeakerAssociation.msg"


  # ---------------------------------------------------------------------------
  # SOCIAL NAVIGATION
  # ---------------------------------------------------------------------------

  "msg/SocialNavigationDecision.msg"
  "msg/SocialState.msg"

  "msg/NavigationState.msg"
  "msg/NavigationControl.msg"
  "msg/NavigationGoalContext.msg"

  "msg/RobotMotionState.msg"


  # ---------------------------------------------------------------------------
  # TRAJECTORY GENERATION / VALIDATION
  # ---------------------------------------------------------------------------

  "msg/TrajectoryGenerationRequest.msg"
  "msg/CandidateTrajectory.msg"
  "msg/TrajectorySocialValidation.msg"


  # ---------------------------------------------------------------------------
  # HUMAN-ROBOT COLLABORATION
  # ---------------------------------------------------------------------------

  "msg/HumanIntent.msg"
  "msg/DialogueRequest.msg"
  "msg/CollaborativeDecision.msg"


  # ---------------------------------------------------------------------------
  # EXPLAINABILITY / SPEECH
  # ---------------------------------------------------------------------------

  "msg/ExplanationContext.msg"
  "msg/ExplanationText.msg"
  "msg/SpeechStatus.msg"
  "msg/SpeechCancelRequest.msg"


  # ---------------------------------------------------------------------------
  # SERVICES
  # ---------------------------------------------------------------------------

  "srv/SimulateHumanIntent.srv"


  # ---------------------------------------------------------------------------
  # INTERFACE DEPENDENCIES
  # ---------------------------------------------------------------------------

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

  # Skip copyright check until copyright/license headers are added.
  set(ament_cmake_copyright_FOUND TRUE)

  # Skip cpplint until the package is managed as required by the linter.
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