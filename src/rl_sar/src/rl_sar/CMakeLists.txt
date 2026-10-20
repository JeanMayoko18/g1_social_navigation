cmake_minimum_required(VERSION 3.10)

project(rl_sar VERSION 4.0.0 LANGUAGES CXX)

# ============================================================
# Global configuration
# ============================================================

set(BUILD_WITH_ROS2 OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SYSTEM_TYPE ${CMAKE_SYSTEM_NAME})

set(USE_CMAKE OFF CACHE BOOL "Use Cmake build system")

if(USE_CMAKE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()


# ============================================================
# ROS mode detection
# ============================================================

if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "noetic")

        add_compile_definitions(USE_ROS1)
        set(BUILD_WITH_ROS2 OFF)

    elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        set(BUILD_WITH_ROS2 ON)

    endif()

else()

    message(WARNING "ROS_DISTRO not set, assuming non-ROS mode")
    add_compile_definitions(USE_CMAKE)

endif()


# ============================================================
# Build options
# ============================================================

set(USE_MUJOCO OFF CACHE BOOL "Build with MuJoCo simulator support")


# ============================================================
# Robot-core build selection
#
# ARM64 robot:
#     BUILD_ROBOT_CORE=ON by default
#
# x86_64 PC:
#     BUILD_ROBOT_CORE=OFF by default
# ============================================================

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(DEFAULT_BUILD_ROBOT_CORE ON)
else()
    set(DEFAULT_BUILD_ROBOT_CORE OFF)
endif()

option(
    BUILD_ROBOT_CORE
    "Build targets requiring the onboard Unitree robot SDK"
    ${DEFAULT_BUILD_ROBOT_CORE}
)

message(STATUS "BUILD_ROBOT_CORE = ${BUILD_ROBOT_CORE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)

if(UPPER_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_options(-g)
endif()


# ============================================================
# Project paths
# ============================================================

get_filename_component(
    PROJECT_ROOT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE
)

add_definitions(
    -DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_definitions(
    -DPOLICY_DIR="${PROJECT_ROOT_DIR}/policy"
)

add_definitions(
    -DBOOST_BIND_GLOBAL_PLACEHOLDERS
)


# ============================================================
# ROS dependencies
# ============================================================

if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "noetic")

        find_package(catkin REQUIRED COMPONENTS
            controller_manager
            genmsg
            joint_state_controller
            robot_state_publisher
            roscpp
            gazebo_ros
            std_msgs
            tf
            geometry_msgs
            robot_msgs
            robot_joint_controller
            rospy
        )

        catkin_package(
            CATKIN_DEPENDS
            robot_joint_controller
            rospy
        )

    elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        find_package(ament_cmake REQUIRED)

        find_package(joint_state_broadcaster REQUIRED)
        find_package(robot_state_publisher REQUIRED)

        find_package(rclcpp REQUIRED)
        find_package(rclpy REQUIRED)
        find_package(rclcpp_action REQUIRED)

        find_package(std_msgs REQUIRED)
        find_package(std_srvs REQUIRED)

        find_package(geometry_msgs REQUIRED)
        find_package(sensor_msgs REQUIRED)
        find_package(nav_msgs REQUIRED)
        find_package(nav2_msgs REQUIRED)

        find_package(tf2 REQUIRED)
        find_package(tf2_ros REQUIRED)
        find_package(tf2_msgs REQUIRED)

        find_package(robot_msgs REQUIRED)
        find_package(robot_joint_controller REQUIRED)

        find_package(unitree_hg REQUIRED)
        find_package(unitree_go REQUIRED)

        if(USE_MUJOCO)
            find_package(gazebo_msgs REQUIRED)
            find_package(gazebo_ros REQUIRED)
        endif()

    endif()

    if(USE_MUJOCO)
        find_package(gazebo REQUIRED)
        add_compile_options(${GAZEBO_CXX_FLAGS})
    endif()

endif()


# ============================================================
# Inference Runtime
# ============================================================

set(
    INFERENCE_RUNTIME_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../library/inference_runtime"
)

message(
    STATUS
    "INFERENCE_RUNTIME_DIR = ${INFERENCE_RUNTIME_DIR}"
)


# ============================================================
# ONNX Runtime
# ============================================================

set(USE_ONNX "OFF")
set(ONNX_RUNTIME_LIB "")

set(
    ONNX_RUNTIME_DIR
    "${INFERENCE_RUNTIME_DIR}/onnxruntime"
)

message(
    STATUS
    "ONNX_RUNTIME_DIR = ${ONNX_RUNTIME_DIR}"
)


# ------------------------------------------------------------
# Detect Jetson platform
# ------------------------------------------------------------

set(IS_JETSON FALSE)

if(EXISTS "/etc/nv_tegra_release")

    set(IS_JETSON TRUE)

    message(
        STATUS
        "Jetson platform detected - ONNX Runtime allowed if aarch64 runtime exists"
    )

endif()


if(
    EXISTS "${ONNX_RUNTIME_DIR}/include"
    AND
    EXISTS "${ONNX_RUNTIME_DIR}/lib"
)

    set(USE_ONNX "ON")

    add_compile_definitions(USE_ONNX)

    message(
        STATUS
        "Found ONNX Runtime at: ${ONNX_RUNTIME_DIR}"
    )


    if(EXISTS "${ONNX_RUNTIME_DIR}/VERSION_NUMBER")

        file(
            READ
            "${ONNX_RUNTIME_DIR}/VERSION_NUMBER"
            ONNX_RUNTIME_VERSION
        )

        string(
            STRIP
            "${ONNX_RUNTIME_VERSION}"
            ONNX_RUNTIME_VERSION
        )

    endif()


    include_directories(
        "${ONNX_RUNTIME_DIR}/include"
    )


    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

        set(LIB_EXT "so")

    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")

        set(LIB_EXT "dylib")

    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")

        set(LIB_EXT "lib")

    endif()


    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")

        set(
            LIB_NAMES
            onnxruntime.lib
        )

    else()

        set(
            LIB_NAMES
            libonnxruntime.${LIB_EXT}
            onnxruntime.${LIB_EXT}
        )

    endif()


    find_library(
        ONNX_RUNTIME_LIB_PATH

        NAMES
        ${LIB_NAMES}

        PATHS
        "${ONNX_RUNTIME_DIR}/lib"

        NO_DEFAULT_PATH
    )


    if(ONNX_RUNTIME_LIB_PATH)

        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")

            add_library(
                onnxruntime_prebuilt
                STATIC
                IMPORTED
                GLOBAL
            )

        else()

            add_library(
                onnxruntime_prebuilt
                SHARED
                IMPORTED
                GLOBAL
            )

        endif()


        set_target_properties(
            onnxruntime_prebuilt

            PROPERTIES

            IMPORTED_LOCATION
                "${ONNX_RUNTIME_LIB_PATH}"

            INTERFACE_INCLUDE_DIRECTORIES
                "${ONNX_RUNTIME_DIR}/include"
        )


        set(
            ONNX_RUNTIME_LIB
            onnxruntime_prebuilt
        )


        message(
            STATUS
            "ONNX Runtime library: ${ONNX_RUNTIME_LIB_PATH}"
        )

    else()

        message(
            WARNING
            "ONNX Runtime library not found"
        )

        set(
            USE_ONNX
            "OFF"
        )

    endif()

else()

    message(
        WARNING
        "ONNX Runtime not found. Please run: bash download_inference_runtime.sh onnx"
    )

endif()


# ============================================================
# LibTorch
# ============================================================

set(USE_TORCH "OFF")
set(TORCH_LIBRARIES "")

set(
    LIBTORCH_DIR
    "${INFERENCE_RUNTIME_DIR}/libtorch"
)


if(
    EXISTS "${LIBTORCH_DIR}/include"
    AND
    EXISTS "${LIBTORCH_DIR}/lib"
)

    set(USE_TORCH "ON")

    add_compile_definitions(USE_TORCH)


    message(
        STATUS
        "Found LibTorch at: ${LIBTORCH_DIR}"
    )


    if(CMAKE_VERSION VERSION_LESS "3.18")

        set(
            TORCH_CUDA_ARCH_LIST
            "5.3;6.2;7.2;8.7"
            CACHE STRING
            "CUDA architectures"
        )

        set(
            ENV{TORCH_CUDA_ARCH_LIST}
            "5.3;6.2;7.2;8.7"
        )

        message(
            STATUS
            "CMake ${CMAKE_VERSION} detected - using common Jetson CUDA architectures"
        )

    endif()


    set(
        CMAKE_PREFIX_PATH
        "${LIBTORCH_DIR};${CMAKE_PREFIX_PATH}"
    )


    find_package(
        Torch
        QUIET
    )


    if(Torch_FOUND)

        message(
            STATUS
            "LibTorch version: ${Torch_VERSION}"
        )

        message(
            STATUS
            "Torch libraries: ${TORCH_LIBRARIES}"
        )

    else()

        set(
            USE_TORCH
            "OFF"
        )

        message(
            WARNING
            "LibTorch found but find_package(Torch) failed"
        )

    endif()

else()

    message(
        WARNING
        "LibTorch not found. Please run: bash download_inference_runtime.sh libtorch"
    )

endif()


# ============================================================
# Inference RPATH
# ============================================================

if(USE_TORCH OR USE_ONNX)

    set(
        CMAKE_SKIP_BUILD_RPATH
        FALSE
    )

    set(
        CMAKE_BUILD_WITH_INSTALL_RPATH
        TRUE
    )

    set(
        CMAKE_INSTALL_RPATH_USE_LINK_PATH
        FALSE
    )


    set(
        PREBUILT_LIB_PATHS
        ""
    )


    if(USE_TORCH)

        list(
            APPEND
            PREBUILT_LIB_PATHS
            "${LIBTORCH_DIR}/lib"
        )

    endif()


    if(USE_ONNX)

        list(
            APPEND
            PREBUILT_LIB_PATHS
            "${ONNX_RUNTIME_DIR}/lib"
        )

    endif()


    set(
        CMAKE_INSTALL_RPATH
        "${CMAKE_INSTALL_RPATH};${PREBUILT_LIB_PATHS}"
    )


    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")

        set(
            CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags"
        )

        set(
            CMAKE_SHARED_LINKER_FLAGS
            "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags"
        )

    endif()


    message(
        STATUS
        "Configured RPATH:"
    )

    foreach(path ${PREBUILT_LIB_PATHS})

        message(
            STATUS
            "  - ${path}"
        )

    endforeach()

endif()


# ============================================================
# Common dependencies
# ============================================================

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

find_package(
    Python3
    COMPONENTS
    Interpreter
    Development
    REQUIRED
)


if(Python3_LIBRARIES)

    get_filename_component(
        PYTHON_LIB_DIR
        "${Python3_LIBRARIES}"
        DIRECTORY
    )

    set(
        CMAKE_INSTALL_RPATH
        "${CMAKE_INSTALL_RPATH};${PYTHON_LIB_DIR}"
    )

    message(
        STATUS
        "Added Python library path to RPATH: ${PYTHON_LIB_DIR}"
    )

endif()


# ============================================================
# YAML
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")

    set(
        YAML_CPP_LIBRARIES
        yaml-cpp
    )

    link_directories(
        /usr/local/lib
    )

    include_directories(
        ${YAML_CPP_INCLUDE_DIR}
    )

elseif(SYSTEM_TYPE STREQUAL "Darwin")

    find_library(
        YAML_CPP_LIB
        yaml-cpp
        REQUIRED
    )

    set(
        YAML_CPP_LIBRARIES
        ${YAML_CPP_LIB}
    )

    link_directories(
        /opt/homebrew/lib
    )


    if(YAML_CPP_INCLUDE_DIR)

        include_directories(
            ${YAML_CPP_INCLUDE_DIR}
        )

    endif()


    find_library(
        OMP_LIB
        NAMES omp
        PATHS
        /opt/homebrew/lib
        /opt/homebrew/opt/libomp/lib
    )


    if(OMP_LIB)

        message(
            STATUS
            "Found OpenMP library: ${OMP_LIB}"
        )

        set(
            OPENMP_LIBRARIES
            ${OMP_LIB}
        )

        get_filename_component(
            OMP_LIB_DIR
            "${OPENMP_LIBRARIES}"
            DIRECTORY
        )

        set(
            CMAKE_INSTALL_RPATH
            "${CMAKE_INSTALL_RPATH};${OMP_LIB_DIR}"
        )

        message(
            STATUS
            "Added OpenMP to RPATH: ${OMP_LIB_DIR}"
        )

    else()

        message(
            WARNING
            "OpenMP not found. Please install: brew install libomp"
        )

    endif()

endif()


# ============================================================
# Architecture
# ============================================================

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")

    message(
        STATUS
        "Current system architecture: ${CMAKE_SYSTEM_PROCESSOR}"
    )

    set(
        ARCH_DIR
        "aarch64"
    )

    set(
        UNITREE_LIB
        "libunitree_legged_sdk_arm64"
    )

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")

    message(
        STATUS
        "Current system architecture: ${CMAKE_SYSTEM_PROCESSOR}"
    )

    set(
        ARCH_DIR
        "x86_64"
    )

    set(
        UNITREE_LIB
        "libunitree_legged_sdk_amd64"
    )

else()

    message(
        FATAL_ERROR
        "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}"
    )

endif()


# ============================================================
# Unitree SDK2
# ============================================================


# ------------------------------------------------------------
# RL_SAR vendored SDK
#
# Repository-local SDK.
# Kept separate from the official onboard robot SDK.
# ------------------------------------------------------------

set(
    RLSAR_UNITREE_SDK2_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_sdk2"
)

set(
    RLSAR_UNITREE_SDK2_LIB_DIR
    "${RLSAR_UNITREE_SDK2_DIR}/thirdparty/lib/${ARCH_DIR}"
)

set(
    RLSAR_UNITREE_SDK2_CORE_LIB
    "${RLSAR_UNITREE_SDK2_DIR}/lib/${ARCH_DIR}/libunitree_sdk2.a"
)

link_directories(
    ${RLSAR_UNITREE_SDK2_LIB_DIR}
)


if(NOT TARGET unitree_sdk2_rlsar)

    add_library(
        unitree_sdk2_rlsar
        STATIC
        IMPORTED
    )

    set_target_properties(
        unitree_sdk2_rlsar

        PROPERTIES

        IMPORTED_LOCATION
            "${RLSAR_UNITREE_SDK2_CORE_LIB}"

        INTERFACE_INCLUDE_DIRECTORIES
            "${RLSAR_UNITREE_SDK2_DIR}/include;${RLSAR_UNITREE_SDK2_DIR}/thirdparty/include/ddscxx"
    )

endif()


# ------------------------------------------------------------
# Official onboard Unitree SDK
#
# Exists only on the robot:
#
#   /home/unitree/unitree_sdk2
#
# It must never be evaluated during the PC build.
# ------------------------------------------------------------

if(BUILD_ROBOT_CORE)

    set(
        UNITREE_SDK2_DIR
        "/home/unitree/unitree_sdk2"
    )

    set(
        UNITREE_SDK2_LIB_DIR
        "${UNITREE_SDK2_DIR}/thirdparty/lib/${ARCH_DIR}"
    )

    set(
        UNITREE_SDK2_CORE_LIB
        "${UNITREE_SDK2_DIR}/lib/${ARCH_DIR}/libunitree_sdk2.a"
    )


    if(NOT TARGET unitree_sdk2)

        add_library(
            unitree_sdk2
            STATIC
            IMPORTED
        )

        set_target_properties(
            unitree_sdk2

            PROPERTIES

            IMPORTED_LOCATION
                "${UNITREE_SDK2_CORE_LIB}"

            INTERFACE_INCLUDE_DIRECTORIES
                "${UNITREE_SDK2_DIR}/include;${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx"
        )

    endif()


    # --------------------------------------------------------
    # Official Unitree SDK target used by LocoClient
    # --------------------------------------------------------

    set(
        ROBOT_UNITREE_SDK2_DIR
        "/home/unitree/unitree_sdk2"
    )

    set(
        ROBOT_UNITREE_SDK2_INCLUDE_DIR
        "${ROBOT_UNITREE_SDK2_DIR}/include"
    )

    set(
        ROBOT_UNITREE_SDK2_LIB_DIR
        "${ROBOT_UNITREE_SDK2_DIR}/lib/${ARCH_DIR}"
    )

    set(
        ROBOT_UNITREE_SDK2_THIRDPARTY_LIB_DIR
        "${ROBOT_UNITREE_SDK2_DIR}/thirdparty/lib/${ARCH_DIR}"
    )

    set(
        ROBOT_UNITREE_SDK2_CORE_LIB
        "${ROBOT_UNITREE_SDK2_LIB_DIR}/libunitree_sdk2.a"
    )


    if(NOT TARGET unitree_sdk2_robot)

        add_library(
            unitree_sdk2_robot
            STATIC
            IMPORTED
        )

        set_target_properties(
            unitree_sdk2_robot

            PROPERTIES

            IMPORTED_LOCATION
                "${ROBOT_UNITREE_SDK2_CORE_LIB}"

            INTERFACE_INCLUDE_DIRECTORIES
                "${ROBOT_UNITREE_SDK2_DIR}/include;${ROBOT_UNITREE_SDK2_DIR}/thirdparty/include/ddscxx"
        )

    endif()

endif()


# ============================================================
# Unitree legged SDK
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")

    add_library(
        unitree_legged_sdk
        SHARED
        IMPORTED
    )

    set_target_properties(
        unitree_legged_sdk

        PROPERTIES

        INTERFACE_INCLUDE_DIRECTORIES
            "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_legged_sdk/include"

        IMPORTED_LOCATION
            "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_legged_sdk/lib/${UNITREE_LIB}.so"
    )

    set(
        UNITREE_A1_LIBS
        Threads::Threads
        unitree_legged_sdk
        lcm
    )


    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        file(
            GLOB
            GLOB_UNITREE_LEGGED_SDK

            "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_legged_sdk/lib/${UNITREE_LIB}.so"
        )

        install(
            FILES
            ${GLOB_UNITREE_LEGGED_SDK}
            DESTINATION lib/
        )

    endif()

endif()


# ============================================================
# L4W4 SDK
# ============================================================

add_library(
    l4w4_sdk
    INTERFACE
)

target_include_directories(
    l4w4_sdk

    INTERFACE

    library/thirdparty/robot_sdk/zhinao/l4w4_sdk
)

target_link_libraries(
    l4w4_sdk

    INTERFACE

    Threads::Threads
)


# ============================================================
# Lite3 Motion SDK
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")

    add_library(
        lite3_motionsdk
        SHARED
        IMPORTED
    )

    set_target_properties(
        lite3_motionsdk

        PROPERTIES

        INTERFACE_INCLUDE_DIRECTORIES
            "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/Lite3_MotionSDK/include;${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/Lite3_MotionSDK/include/common"

        IMPORTED_LOCATION
            "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/Lite3_MotionSDK/lib/libdeeprobotics_legged_sdk_${ARCH_DIR}.so"
    )

    set(
        LITE3_REAL_LIBS
        Threads::Threads
        lite3_motionsdk
    )


    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        file(
            GLOB
            GLOB_LITE3_SDK_SO

            "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/Lite3_MotionSDK/lib/libdeeprobotics_legged_sdk_${ARCH_DIR}.so"
        )

        install(
            FILES
            ${GLOB_LITE3_SDK_SO}
            DESTINATION lib/
        )

    endif()

endif()


# ============================================================
# Retroid Gamepad
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")

    set(
        GAMEPAD_SRC

        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/gamepad/src/gamepad.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/gamepad/src/retroid_gamepad.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/gamepad/src/udp_receiver.cpp
    )

endif()


# ============================================================
# Global includes
# ============================================================

if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "noetic")

        include_directories(
            ${catkin_INCLUDE_DIRS}
        )

    endif()

endif()


include_directories(
    include

    library/core/matplotlibcpp
    library/core/observation_buffer
    library/core/rl_sdk
    library/core/loop
    library/core/fsm
    library/core/vector_math
    library/core/motion_loader
    library/core/inference_runtime
    library/core/logger

    fsm_robot
)


# ============================================================
# Inference runtime library
# ============================================================

add_library(
    inference_runtime
    library/core/inference_runtime/inference_runtime.cpp
)


if(USE_TORCH)

    target_link_libraries(
        inference_runtime

        PUBLIC

        ${TORCH_LIBRARIES}
    )

    target_compile_definitions(
        inference_runtime

        PRIVATE

        USE_TORCH
    )


    if(
        SYSTEM_TYPE STREQUAL "Darwin"
        AND
        OPENMP_LIBRARIES
    )

        target_link_libraries(
            inference_runtime

            PUBLIC

            ${OPENMP_LIBRARIES}
        )

    endif()

endif()


if(USE_ONNX)

    target_link_libraries(
        inference_runtime

        PUBLIC

        ${ONNX_RUNTIME_LIB}
    )

    target_compile_definitions(
        inference_runtime

        PRIVATE

        USE_ONNX
    )

endif()


if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        install(
            TARGETS
            inference_runtime

            DESTINATION
            lib/${PROJECT_NAME}
        )

    endif()

endif()


# ============================================================
# Motion loader
# ============================================================

add_library(
    motion_loader
    library/core/motion_loader/motion_loader.cpp
)


# ============================================================
# RL SDK
# ============================================================

add_library(
    rl_sdk
    library/core/rl_sdk/rl_sdk.cpp
)

target_link_libraries(
    rl_sdk

    PUBLIC

    motion_loader
    inference_runtime

    Python3::Python
    Python3::Module

    TBB::tbb
)


find_package(
    Python3
    COMPONENTS
    NumPy
)


if(Python3_NumPy_FOUND)

    target_link_libraries(
        rl_sdk

        PUBLIC

        Python3::NumPy
    )

else()

    target_compile_definitions(
        rl_sdk

        PUBLIC

        WITHOUT_NUMPY
    )

endif()


if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        install(
            TARGETS
            rl_sdk

            DESTINATION
            lib/${PROJECT_NAME}
        )

    endif()

endif()


# ============================================================
# Observation buffer
# ============================================================

add_library(
    observation_buffer
    library/core/observation_buffer/observation_buffer.cpp
)


if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        install(
            TARGETS
            observation_buffer

            DESTINATION
            lib/${PROJECT_NAME}
        )

    endif()

endif()


# ============================================================
# Legacy simulator target
# ============================================================

if(NOT USE_CMAKE AND USE_MUJOCO)

    if(FALSE)

        target_link_libraries(
            rl_sim

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}
            Threads::Threads
        )


        if($ENV{ROS_DISTRO} MATCHES "noetic")

            target_link_libraries(
                rl_sim
                ${catkin_LIBRARIES}
            )

        elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

            ament_target_dependencies(
                rl_sim

                joint_state_broadcaster
                robot_state_publisher

                rclcpp

                gazebo_ros
                gazebo_msgs

                std_msgs
                std_srvs
                geometry_msgs

                robot_msgs
                robot_joint_controller

                rclpy
            )

            install(
                TARGETS
                rl_sim

                DESTINATION
                lib/${PROJECT_NAME}
            )

        endif()

    endif()

endif()


# ============================================================
# Legacy robot targets
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")

    # --------------------------------------------------------
    # A1
    # --------------------------------------------------------

    if(FALSE)

        target_link_libraries(
            rl_real_a1

            ${UNITREE_A1_LIBS}

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_link_libraries(
                    rl_real_a1
                    ${catkin_LIBRARIES}
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                ament_target_dependencies(
                    rl_real_a1

                    rclcpp
                    geometry_msgs
                )

                install(
                    TARGETS
                    rl_real_a1

                    DESTINATION
                    lib/${PROJECT_NAME}
                )

            endif()

        endif()

    endif()


    # --------------------------------------------------------
    # Lite3
    # --------------------------------------------------------

    if(FALSE)

        target_link_libraries(
            rl_real_lite3

            ${LITE3_REAL_LIBS}

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}
        )

        target_include_directories(
            rl_real_lite3

            PRIVATE

            ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/deeprobotics/gamepad/include
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_link_libraries(
                    rl_real_lite3
                    ${catkin_LIBRARIES}
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                ament_target_dependencies(
                    rl_real_lite3

                    rclcpp
                    geometry_msgs
                )

                install(
                    TARGETS
                    rl_real_lite3

                    DESTINATION
                    lib/${PROJECT_NAME}
                )

            endif()

        endif()

    endif()


    # --------------------------------------------------------
    # Go2
    # --------------------------------------------------------

    if(FALSE)

        target_link_libraries(
            rl_real_go2

            unitree_sdk2

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}
        )

        set_target_properties(
            rl_real_go2

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_sdk2/thirdparty/lib/${ARCH_DIR}"
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_link_libraries(
                    rl_real_go2
                    ${catkin_LIBRARIES}
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                ament_target_dependencies(
                    rl_real_go2

                    rclcpp
                    geometry_msgs
                )

                install(
                    TARGETS
                    rl_real_go2

                    DESTINATION
                    lib/${PROJECT_NAME}
                )

            endif()

        endif()

    endif()


    # --------------------------------------------------------
    # Legacy G1
    # --------------------------------------------------------

    if(FALSE)

        target_link_libraries(
            rl_real_g1

            unitree_sdk2

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}
        )

        set_target_properties(
            rl_real_g1

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_sdk2/thirdparty/lib/${ARCH_DIR}"
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_link_libraries(
                    rl_real_g1
                    ${catkin_LIBRARIES}
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                ament_target_dependencies(
                    rl_real_g1

                    rclcpp
                    geometry_msgs
                )

                install(
                    TARGETS
                    rl_real_g1

                    DESTINATION
                    lib/${PROJECT_NAME}
                )

            endif()

        endif()

    endif()

endif()


# ============================================================
# Linux ROS2 executables
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")


    # ========================================================
    # Robot-only RL_SAR / Unitree executables
    # ========================================================

    if(BUILD_ROBOT_CORE)


        # ----------------------------------------------------
        # ROS2 EDU23
        # ----------------------------------------------------

        add_executable(
            rl_real_g1_edu23
            src/rl_real_g1_edu23.cpp
        )

        target_compile_definitions(
            rl_real_g1_edu23

            PRIVATE

            USE_ROS
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_compile_definitions(
                    rl_real_g1_edu23

                    PRIVATE

                    USE_ROS1
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                target_compile_definitions(
                    rl_real_g1_edu23

                    PRIVATE

                    USE_ROS2
                )

            endif()

        endif()


        target_include_directories(
            rl_real_g1_edu23

            BEFORE PRIVATE

            ${UNITREE_SDK2_DIR}/include
            ${UNITREE_SDK2_DIR}/thirdparty/include
            ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
        )


        target_link_libraries(
            rl_real_g1_edu23

            unitree_sdk2

            ${UNITREE_SDK2_LIB_DIR}/libddscxx.so
            ${UNITREE_SDK2_LIB_DIR}/libddsc.so

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}

            Threads::Threads
        )


        # ----------------------------------------------------
        # ROS2 EDU12
        # ----------------------------------------------------

        add_executable(
            rl_real_g1_edu12
            src/rl_real_g1_edu12.cpp
        )

        target_compile_definitions(
            rl_real_g1_edu12

            PRIVATE

            USE_ROS
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_compile_definitions(
                    rl_real_g1_edu12

                    PRIVATE

                    USE_ROS1
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                target_compile_definitions(
                    rl_real_g1_edu12

                    PRIVATE

                    USE_ROS2
                )

            endif()

        endif()


        target_include_directories(
            rl_real_g1_edu12

            BEFORE PRIVATE

            ${UNITREE_SDK2_DIR}/include
            ${UNITREE_SDK2_DIR}/thirdparty/include
            ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
        )


        target_link_libraries(
            rl_real_g1_edu12

            unitree_sdk2

            ${UNITREE_SDK2_LIB_DIR}/libddscxx.so
            ${UNITREE_SDK2_LIB_DIR}/libddsc.so

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}

            Threads::Threads
        )


        # ----------------------------------------------------
        # Unitree-only EDU23
        # ----------------------------------------------------

        add_executable(
            rl_real_g1_edu23_unitree
            src/rl_real_g1_edu23.cpp
        )

        target_compile_definitions(
            rl_real_g1_edu23_unitree

            PRIVATE

            ENABLE_REAL_ROBOT
        )


        target_include_directories(
            rl_real_g1_edu23_unitree

            BEFORE PRIVATE

            /home/unitree/unitree_sdk2/include
            /home/unitree/unitree_sdk2/thirdparty/include/ddscxx
        )


        target_link_libraries(
            rl_real_g1_edu23_unitree

            /home/unitree/unitree_sdk2/lib/aarch64/libunitree_sdk2.a

            /home/unitree/unitree_sdk2/thirdparty/lib/aarch64/libddscxx.so
            /home/unitree/unitree_sdk2/thirdparty/lib/aarch64/libddsc.so

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}

            Threads::Threads
        )


        # ----------------------------------------------------
        # Unitree-only EDU12
        # ----------------------------------------------------

        add_executable(
            rl_real_g1_edu12_unitree
            src/rl_real_g1_edu12.cpp
        )

        target_compile_definitions(
            rl_real_g1_edu12_unitree

            PRIVATE

            ENABLE_REAL_ROBOT
        )


        target_include_directories(
            rl_real_g1_edu12_unitree

            BEFORE PRIVATE

            ${RLSAR_UNITREE_SDK2_DIR}/include
            ${RLSAR_UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
        )


        target_link_libraries(
            rl_real_g1_edu12_unitree

            unitree_sdk2_rlsar

            ${RLSAR_UNITREE_SDK2_LIB_DIR}/libddscxx.so
            ${RLSAR_UNITREE_SDK2_LIB_DIR}/libddsc.so

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}

            Threads::Threads
        )


        # ----------------------------------------------------
        # Unitree LocoClient backend
        # ----------------------------------------------------

        add_executable(
            rl_real_g1_edu23_unitree_loco
            src/rl_real_g1_edu23.cpp
        )

        target_compile_definitions(
            rl_real_g1_edu23_unitree_loco

            PRIVATE

            ENABLE_REAL_ROBOT
            ENABLE_UNITREE_LOCO_BACKEND
        )


        target_include_directories(
            rl_real_g1_edu23_unitree_loco

            PRIVATE

            ${ROBOT_UNITREE_SDK2_INCLUDE_DIR}
            ${ROBOT_UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
        )


        target_link_libraries(
            rl_real_g1_edu23_unitree_loco

            unitree_sdk2_robot

            ${ROBOT_UNITREE_SDK2_LIB_DIR}/libunitree_sdk2.a

            ${ROBOT_UNITREE_SDK2_THIRDPARTY_LIB_DIR}/libddscxx.so
            ${ROBOT_UNITREE_SDK2_THIRDPARTY_LIB_DIR}/libddsc.so

            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}

            Threads::Threads
        )


        # ----------------------------------------------------
        # Release Unitree mode
        # ----------------------------------------------------

        add_executable(
            unitree_release_mode_reset
            src/unitree_release_mode_reset.cpp
        )


        target_include_directories(
            unitree_release_mode_reset

            PRIVATE

            ${RLSAR_UNITREE_SDK2_DIR}/include
            ${RLSAR_UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
        )


        target_link_libraries(
            unitree_release_mode_reset

            unitree_sdk2_rlsar

            ${RLSAR_UNITREE_SDK2_LIB_DIR}/libddscxx.so
            ${RLSAR_UNITREE_SDK2_LIB_DIR}/libddsc.so

            Threads::Threads
        )

    endif()


    # ========================================================
    # Portable ROS2 / bridge / navigation / LiDAR executables
    # ========================================================

    add_executable(
        ros2_nav_bridge_node
        src/ros2_nav_bridge_node.cpp
    )

    add_executable(
        topics_udp_sender_node
        src/topics_udp_sender_node.cpp
    )

    add_executable(
        topics_udp_receiver_node
        src/topics_udp_receiver_node.cpp
    )

    add_executable(
        cloud_to_scan_360_node
        src/cloud_to_scan_360_node.cpp
    )

    add_executable(
        cloud_fov_checker
        src/cloud_fov_checker.cpp
    )

    add_executable(
        pc2_restamper_node
        src/pc2_restamper_node.cpp
    )

    add_executable(
        cloud_y_flip_node
        src/cloud_y_flip_node.cpp
    )

    add_executable(
        livox_to_mola_cloud
        src/livox_to_mola_cloud.cpp
    )

    add_executable(
        mola_cloud_udp_sender_node
        src/mola_cloud_udp_sender_node.cpp
    )

    add_executable(
        mola_cloud_udp_receiver_node
        src/mola_cloud_udp_receiver_node.cpp
    )

    add_executable(
        cloud_nav_stabilizer_node
        src/cloud_nav_stabilizer_node.cpp
    )

    add_executable(
        navigation_time_sync_node
        src/navigation_time_sync_node.cpp
    )

    add_executable(
        mola_odom_adapter_node
        src/mola_odom_adapter_node.cpp
    )

    add_executable(
        dog_odom_adapter_node
        src/dog_odom_adapter_node.cpp
    )

    add_executable(
        dog_odom_aocf_node
        src/dog_odom_aocf_node.cpp
    )

    add_executable(
        dog_odom_calibration_node
        src/dog_odom_calibration_node.cpp
    )

    add_executable(
        cloud_stamp_relay_node
        src/cloud_stamp_relay_node.cpp
    )

    add_executable(
        odom_source_selector
        src/odom_source_selector.cpp
    )

    add_executable(
        lidar_time_offset_node
        src/lidar_time_offset_node.cpp
    )

    add_executable(
        g1_nav_odom_adapter_node
        src/g1_nav_odom_adapter_node.cpp
    )

    add_executable(
        cloud_freshness_guard_node
        src/cloud_freshness_guard_node.cpp
    )

    add_executable(
        livox_imu_adapter_node
        src/livox_imu_adapter_node.cpp
    )


    # ========================================================
    # Robot-only ROS2 utility executables
    # ========================================================

    if(BUILD_ROBOT_CORE)

        add_executable(
            lowstate_to_joint_states_node
            src/lowstate_to_joint_states_node.cpp
        )

        add_executable(
            g1_odom_publisher
            src/g1_odom_publisher.cpp
        )

        add_executable(
            g1_odom_unitree_sender
            src/g1_odom_unitree_sender.cpp
        )


        # ----------------------------------------------------
        # g1_odom_unitree_sender
        # ----------------------------------------------------

        target_include_directories(
            g1_odom_unitree_sender

            BEFORE PRIVATE

            /home/unitree/unitree_sdk2/include
            /home/unitree/unitree_sdk2/thirdparty/include/ddscxx
        )


        target_link_libraries(
            g1_odom_unitree_sender

            /home/unitree/unitree_sdk2/lib/aarch64/libunitree_sdk2.a
            /home/unitree/unitree_sdk2/thirdparty/lib/aarch64/libddscxx.so
            /home/unitree/unitree_sdk2/thirdparty/lib/aarch64/libddsc.so

            Threads::Threads
        )


        set_target_properties(
            g1_odom_unitree_sender

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};/home/unitree/unitree_sdk2/thirdparty/lib/aarch64"
        )


        # ----------------------------------------------------
        # LowState -> JointState
        # ----------------------------------------------------

        target_include_directories(
            lowstate_to_joint_states_node

            BEFORE PRIVATE

            ${UNITREE_SDK2_DIR}/include
            ${UNITREE_SDK2_DIR}/thirdparty/include
            ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
        )


        target_link_libraries(
            lowstate_to_joint_states_node

            unitree_sdk2

            ${UNITREE_SDK2_LIB_DIR}/libddscxx.so
            ${UNITREE_SDK2_LIB_DIR}/libddsc.so

            ${YAML_CPP_LIBRARIES}

            Threads::Threads
        )

    endif()


    # ========================================================
    # ROS dependencies for executable targets
    # ========================================================

    if(NOT USE_CMAKE)

        if($ENV{ROS_DISTRO} MATCHES "noetic")

            if(BUILD_ROBOT_CORE)

                target_link_libraries(
                    rl_real_g1_edu23
                    ${catkin_LIBRARIES}
                )

                target_link_libraries(
                    rl_real_g1_edu12
                    ${catkin_LIBRARIES}
                )

            endif()


        elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")


            # ------------------------------------------------
            # Robot-only ROS dependencies
            # ------------------------------------------------

            if(BUILD_ROBOT_CORE)

                ament_target_dependencies(
                    rl_real_g1_edu23

                    rclcpp
                    geometry_msgs
                    sensor_msgs
                    std_msgs
                )


                ament_target_dependencies(
                    rl_real_g1_edu12

                    rclcpp
                    geometry_msgs
                    sensor_msgs
                    std_msgs
                )


                ament_target_dependencies(
                    lowstate_to_joint_states_node

                    rclcpp
                    sensor_msgs
                    unitree_hg
                )


                ament_target_dependencies(
                    g1_odom_publisher

                    rclcpp
                    nav_msgs
                    geometry_msgs
                    tf2_ros
                    tf2_msgs
                )

            endif()


            # ------------------------------------------------
            # Portable bridge / UDP dependencies
            # ------------------------------------------------

            ament_target_dependencies(
                ros2_nav_bridge_node

                rclcpp
                geometry_msgs
            )


            ament_target_dependencies(
                topics_udp_sender_node

                rclcpp
                rclcpp_action

                nav_msgs
                sensor_msgs
                geometry_msgs
                tf2_msgs
                nav2_msgs
            )


            ament_target_dependencies(
                topics_udp_receiver_node

                rclcpp
                rclcpp_action

                nav2_msgs
                geometry_msgs
                nav_msgs
                sensor_msgs
                tf2_msgs
            )


            # ------------------------------------------------
            # Point cloud / LiDAR
            # ------------------------------------------------

            ament_target_dependencies(
                cloud_to_scan_360_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                cloud_fov_checker

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                cloud_nav_stabilizer_node

                rclcpp
                sensor_msgs
                geometry_msgs
                tf2
                tf2_ros
            )


            ament_target_dependencies(
                cloud_stamp_relay_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                pc2_restamper_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                cloud_y_flip_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                livox_to_mola_cloud

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                mola_cloud_udp_sender_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                mola_cloud_udp_receiver_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                navigation_time_sync_node

                rclcpp
                sensor_msgs
                nav_msgs
                std_msgs
            )


            ament_target_dependencies(
                lidar_time_offset_node

                rclcpp
                sensor_msgs
                std_msgs
            )


            ament_target_dependencies(
                cloud_freshness_guard_node

                rclcpp
                sensor_msgs
            )


            ament_target_dependencies(
                livox_imu_adapter_node

                rclcpp
                sensor_msgs
            )


            # ------------------------------------------------
            # Odometry
            # ------------------------------------------------

            ament_target_dependencies(
                mola_odom_adapter_node

                rclcpp
                nav_msgs
                geometry_msgs
                tf2_ros
                tf2_msgs
            )


            ament_target_dependencies(
                dog_odom_adapter_node

                rclcpp
                nav_msgs
                geometry_msgs
                tf2_ros
                tf2_msgs
            )


            ament_target_dependencies(
                dog_odom_aocf_node

                rclcpp
                nav_msgs
                geometry_msgs
                tf2_ros
                tf2_msgs
            )


            ament_target_dependencies(
                dog_odom_calibration_node

                rclcpp
                nav_msgs
                geometry_msgs
            )


            ament_target_dependencies(
                g1_nav_odom_adapter_node

                rclcpp
                nav_msgs
                geometry_msgs
                tf2_ros
                tf2_msgs
            )


            ament_target_dependencies(
                odom_source_selector

                rclcpp
                nav_msgs
                geometry_msgs
                tf2_ros
            )

        endif()

    endif()


    target_link_libraries(
        g1_nav_odom_adapter_node

        tf2_ros::tf2_ros
    )


    # ========================================================
    # Robot feature flags
    # ========================================================

    option(
        ENABLE_REAL_ROBOT
        "Enable real robot hardware access"
        OFF
    )

    option(
        EDU23_CMD_VEL_ONLY
        "Build EDU23 in cmd_vel only mode"
        OFF
    )

    option(
        EDU23_FAKE_LOWSTATE
        "Build EDU23 with fake lowstate test"
        OFF
    )

    option(
        EDU23_DRY_RUN_NO_LOW_CMD
        "Build EDU23 with no LowCmd publishing"
        OFF
    )


    if(BUILD_ROBOT_CORE)

        if(EDU23_CMD_VEL_ONLY)

            target_compile_definitions(
                rl_real_g1_edu23

                PRIVATE

                CMD_VEL_ONLY
            )

        endif()


        if(EDU23_FAKE_LOWSTATE)

            target_compile_definitions(
                rl_real_g1_edu23

                PRIVATE

                FAKE_LOWSTATE_TEST
            )

        endif()


        if(ENABLE_REAL_ROBOT)

            target_compile_definitions(
                rl_real_g1_edu23

                PRIVATE

                ENABLE_REAL_ROBOT
            )


            target_compile_definitions(
                rl_real_g1_edu12

                PRIVATE

                ENABLE_REAL_ROBOT
            )

        endif()


        # ----------------------------------------------------
        # Robot RPATH
        # ----------------------------------------------------

        set_target_properties(
            rl_real_g1_edu23

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};${UNITREE_SDK2_LIB_DIR}"
        )


        set_target_properties(
            rl_real_g1_edu12

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};${UNITREE_SDK2_LIB_DIR}"
        )


        set_target_properties(
            rl_real_g1_edu23_unitree

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};/home/unitree/unitree_sdk2/thirdparty/lib/aarch64;/home/unitree/g1_ws/src/rl_sar/library/inference_runtime/onnxruntime/lib"
        )


        set_target_properties(
            rl_real_g1_edu23_unitree_loco

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};${ROBOT_UNITREE_SDK2_LIB_DIR};${ROBOT_UNITREE_SDK2_THIRDPARTY_LIB_DIR}"
        )


        set_target_properties(
            unitree_release_mode_reset

            PROPERTIES

            INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH};${RLSAR_UNITREE_SDK2_LIB_DIR}"
        )

    endif()


    message(
        STATUS
        "RLSAR_UNITREE_SDK2_LIB_DIR = ${RLSAR_UNITREE_SDK2_LIB_DIR}"
    )


    # ========================================================
    # Portable/common installation
    # ========================================================

    install(
        TARGETS

        ros2_nav_bridge_node

        topics_udp_sender_node
        topics_udp_receiver_node

        pc2_restamper_node
        cloud_y_flip_node

        livox_to_mola_cloud

        mola_cloud_udp_sender_node
        mola_cloud_udp_receiver_node

        cloud_to_scan_360_node
        cloud_fov_checker
        cloud_nav_stabilizer_node

        navigation_time_sync_node

        mola_odom_adapter_node

        dog_odom_adapter_node
        dog_odom_aocf_node
        dog_odom_calibration_node

        livox_imu_adapter_node

        g1_nav_odom_adapter_node

        lidar_time_offset_node

        cloud_freshness_guard_node
        cloud_stamp_relay_node

        odom_source_selector

        DESTINATION
        lib/${PROJECT_NAME}
    )


    # ========================================================
    # Robot-only installation
    # ========================================================

    if(BUILD_ROBOT_CORE)

        install(
            TARGETS

            rl_real_g1_edu23
            rl_real_g1_edu12

            rl_real_g1_edu12_unitree
            rl_real_g1_edu23_unitree
            rl_real_g1_edu23_unitree_loco

            unitree_release_mode_reset

            lowstate_to_joint_states_node

            g1_odom_publisher
            g1_odom_unitree_sender

            DESTINATION
            lib/${PROJECT_NAME}
        )

    endif()


    install(
        DIRECTORY
        launch

        DESTINATION
        share/${PROJECT_NAME}
    )

endif()


# ============================================================
# Legacy L4W4 executable
# ============================================================

if(SYSTEM_TYPE STREQUAL "Linux")

    if(FALSE)

        target_link_libraries(
            rl_real_l4w4

            l4w4_sdk
            rl_sdk
            observation_buffer

            ${YAML_CPP_LIBRARIES}
        )


        if(NOT USE_CMAKE)

            if($ENV{ROS_DISTRO} MATCHES "noetic")

                target_link_libraries(
                    rl_real_l4w4
                    ${catkin_LIBRARIES}
                )

            elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

                ament_target_dependencies(
                    rl_real_l4w4

                    rclcpp
                    geometry_msgs
                )

                install(
                    TARGETS
                    rl_real_l4w4

                    DESTINATION
                    lib/${PROJECT_NAME}
                )

            endif()

        endif()

    endif()

endif()


# ============================================================
# Actuator net Python executable
# ============================================================

if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "noetic")

        catkin_install_python(
            PROGRAMS

            scripts/actuator_net.py

            DESTINATION
            ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

    elseif($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        install(
            PROGRAMS

            scripts/actuator_net.py

            DESTINATION
            lib/${PROJECT_NAME}
        )

    endif()

endif()


# ============================================================
# Shared ROS resources
# ============================================================

if(NOT USE_CMAKE)

    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")

        install(
            DIRECTORY
            launch
            worlds

            DESTINATION
            share/${PROJECT_NAME}
        )

        install(
            DIRECTORY
            ${PROJECT_ROOT_DIR}/policy

            DESTINATION
            share/${PROJECT_NAME}
        )

    endif()

endif()


# ============================================================
# MuJoCo
# ============================================================

if(USE_MUJOCO)

    add_compile_definitions(
        USE_MUJOCO
    )

    set(
        MUJOCO_DIR
        "${PROJECT_ROOT_DIR}/library/mujoco"
    )


    # --------------------------------------------------------
    # Paths / includes / libraries
    # --------------------------------------------------------

    if(SYSTEM_TYPE STREQUAL "Darwin")

        if(
            NOT EXISTS
            "${MUJOCO_DIR}/Headers/mujoco.h"
        )

            message(
                FATAL_ERROR
                "MuJoCo not found. Please run: bash scripts/download_mujoco.sh"
            )

        endif()


        set(
            MUJOCO_COMPAT_INCLUDE_DIR
            "${CMAKE_BINARY_DIR}/mujoco_include"
        )

        file(
            MAKE_DIRECTORY
            "${MUJOCO_COMPAT_INCLUDE_DIR}/mujoco"
        )

        file(
            GLOB
            MUJOCO_HEADERS

            "${MUJOCO_DIR}/Headers/*.h"
        )

        file(
            COPY
            ${MUJOCO_HEADERS}

            DESTINATION
            "${MUJOCO_COMPAT_INCLUDE_DIR}/mujoco"
        )

        set(
            MUJOCO_INCLUDE_DIR
            "${MUJOCO_COMPAT_INCLUDE_DIR}"
        )

        set(
            MUJOCO_LIB_DIR
            "${MUJOCO_DIR}/Versions/Current"
        )

    else()

        if(
            NOT EXISTS
            "${MUJOCO_DIR}/include/mujoco/mujoco.h"
        )

            message(
                FATAL_ERROR
                "MuJoCo not found. Please run: bash scripts/download_mujoco.sh"
            )

        endif()


        set(
            MUJOCO_INCLUDE_DIR
            "${MUJOCO_DIR}/include"
        )

        set(
            MUJOCO_LIB_DIR
            "${MUJOCO_DIR}/lib"
        )

    endif()


    set(
        CMAKE_INSTALL_RPATH
        "${CMAKE_INSTALL_RPATH};${MUJOCO_LIB_DIR}"
    )

    message(
        STATUS
        "Added MuJoCo to RPATH: ${MUJOCO_LIB_DIR}"
    )


    find_package(
        glfw3
        REQUIRED
    )


    if(SYSTEM_TYPE STREQUAL "Linux")

        set(
            MUJOCO_LIB_NAME
            "libmujoco.so.3.2.7"
        )

    elseif(SYSTEM_TYPE STREQUAL "Darwin")

        set(
            MUJOCO_LIB_NAME
            "libmujoco.3.2.7.dylib"
        )

    elseif(SYSTEM_TYPE STREQUAL "Windows")

        set(
            MUJOCO_LIB_NAME
            "mujoco.dll"
        )

    else()

        message(
            FATAL_ERROR
            "Unsupported platform: ${SYSTEM_TYPE}"
        )

    endif()


    add_library(
        mujoco_prebuilt
        SHARED
        IMPORTED
        GLOBAL
    )

    set_target_properties(
        mujoco_prebuilt

        PROPERTIES

        IMPORTED_LOCATION
            "${MUJOCO_LIB_DIR}/${MUJOCO_LIB_NAME}"

        INTERFACE_INCLUDE_DIRECTORIES
            "${MUJOCO_INCLUDE_DIR}"

        IMPORTED_NO_SONAME
            TRUE
    )


    # --------------------------------------------------------
    # MuJoCo simulation support library
    # --------------------------------------------------------

    file(
        GLOB
        MUJOCO_SIMULATE_SRC

        library/thirdparty/mujoco_simulate/*.cc
    )

    set(
        JOYSTICK_SRC
        library/thirdparty/joystick/joystick.cc
    )


    if(SYSTEM_TYPE STREQUAL "Darwin")

        file(
            GLOB
            MUJOCO_SIMULATE_MM

            library/thirdparty/mujoco_simulate/*.mm
        )

        list(
            APPEND
            MUJOCO_SIMULATE_SRC

            ${MUJOCO_SIMULATE_MM}
        )


        set_source_files_properties(
            ${MUJOCO_SIMULATE_MM}

            PROPERTIES

            COMPILE_FLAGS
                "-Wno-deprecated-declarations"
        )

    endif()


    add_library(
        mujoco_sim_support
        STATIC

        ${MUJOCO_SIMULATE_SRC}
        ${JOYSTICK_SRC}
    )


    target_include_directories(
        mujoco_sim_support

        PUBLIC

        ${MUJOCO_INCLUDE_DIR}

        library/thirdparty/mujoco_simulate
        library/thirdparty/joystick
    )


    target_link_libraries(
        mujoco_sim_support

        PUBLIC

        Threads::Threads

        mujoco_prebuilt

        glfw
    )


    if(SYSTEM_TYPE STREQUAL "Darwin")

        if(OPENMP_LIBRARIES)

            target_link_libraries(
                mujoco_sim_support

                PUBLIC

                ${OPENMP_LIBRARIES}
            )

        endif()


        target_link_libraries(
            mujoco_sim_support

            PUBLIC

            "-framework CoreVideo"
            "-framework OpenGL"
            "-framework AppKit"
        )

    endif()


    # --------------------------------------------------------
    # MuJoCo executables
    # --------------------------------------------------------

    add_executable(
        rl_sim_mujoco
        src/rl_sim_mujoco.cpp
    )


    target_link_libraries(
        rl_sim_mujoco

        rl_sdk
        observation_buffer

        ${YAML_CPP_LIBRARIES}

        mujoco_sim_support
    )


    add_executable(
        rl_mujoco
        src/rl_mujoco.cpp
    )


    target_link_libraries(
        rl_mujoco

        rl_sdk
        observation_buffer

        ${YAML_CPP_LIBRARIES}

        mujoco_sim_support
    )


    if(
        NOT USE_CMAKE
        AND
        $ENV{ROS_DISTRO} MATCHES "foxy|humble"
    )

        ament_target_dependencies(
            rl_mujoco

            rclcpp
            geometry_msgs
        )

    endif()


    # --------------------------------------------------------
    # MuJoCo installation
    # --------------------------------------------------------

    if(
        NOT USE_CMAKE
        AND
        $ENV{ROS_DISTRO} MATCHES "foxy|humble"
    )

        install(
            TARGETS

            rl_sim_mujoco
            rl_mujoco

            DESTINATION
            lib/${PROJECT_NAME}
        )

    else()

        install(
            TARGETS

            rl_sim_mujoco
            rl_mujoco

            DESTINATION
            bin
        )

    endif()


    if(
        NOT USE_CMAKE
        AND
        $ENV{ROS_DISTRO} MATCHES "foxy|humble"
    )

        ament_target_dependencies(
            rl_sim_mujoco

            rclcpp
            geometry_msgs
        )

    endif()


    message(
        STATUS
        "MuJoCo configured: ${MUJOCO_DIR}"
    )

endif()


# ============================================================
# Tests
# ============================================================

# add_executable(
#     test_observation_buffer
#     test/test_observation_buffer.cpp
# )
#
# target_link_libraries(
#     test_observation_buffer
#     observation_buffer
# )
#
# install(
#     TARGETS
#     test_observation_buffer
#     DESTINATION lib/${PROJECT_NAME}
# )


# add_executable(
#     test_vector_math
#     test/test_vector_math.cpp
# )
#
# install(
#     TARGETS
#     test_vector_math
#     DESTINATION lib/${PROJECT_NAME}
# )


# add_executable(
#     test_inference_stability
#     test/test_inference_stability.cpp
# )
#
# target_link_libraries(
#     test_inference_stability
#     inference_runtime
#     ${TORCH_LIBRARIES}
#     ${ONNX_RUNTIME_LIB}
# )
#
# install(
#     TARGETS
#     test_inference_stability
#     DESTINATION lib/${PROJECT_NAME}
# )


# ============================================================
# Ament package
# ============================================================

if(BUILD_WITH_ROS2)

    ament_package()

endif()


# ============================================================
# Build summary
# ============================================================

message(STATUS "========================================")
message(STATUS "       RL_SAR Build Configuration       ")
message(STATUS "========================================")

message(
    STATUS
    "Project: ${PROJECT_NAME} v${PROJECT_VERSION}"
)

message(
    STATUS
    "System: ${SYSTEM_TYPE}"
)

message(
    STATUS
    "Architecture: ${CMAKE_SYSTEM_PROCESSOR}"
)

message(
    STATUS
    "C++ Standard: ${CMAKE_CXX_STANDARD}"
)

message(
    STATUS
    "Build Type: ${CMAKE_BUILD_TYPE}"
)

message(
    STATUS
    "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

message(
    STATUS
    "USE_CMAKE: ${USE_CMAKE}"
)

message(
    STATUS
    "BUILD_ROBOT_CORE: ${BUILD_ROBOT_CORE}"
)

message(STATUS "----------------------------------------")
message(STATUS "Backend Configuration:")

message(
    STATUS
    "USE_TORCH: ${USE_TORCH}"
)

message(
    STATUS
    "USE_ONNX: ${USE_ONNX}"
)


if(USE_TORCH)

    if(Torch_VERSION)

        message(
            STATUS
            "LibTorch version: ${Torch_VERSION}"
        )

    else()

        message(
            STATUS
            "LibTorch: Available"
        )

    endif()


    if(IS_JETSON)

        message(
            STATUS
            "LibTorch: Jetson device detected"
        )

        if(JETPACK_VERSION)

            message(
                STATUS
                "JetPack version: ${JETPACK_VERSION}"
            )

        endif()

    endif()

endif()


if(USE_ONNX)

    if(ONNX_RUNTIME_VERSION)

        message(
            STATUS
            "ONNX Runtime version: ${ONNX_RUNTIME_VERSION}"
        )

    elseif(onnxruntime_VERSION)

        message(
            STATUS
            "ONNX Runtime version: ${onnxruntime_VERSION}"
        )

    else()

        message(
            STATUS
            "ONNX Runtime: Available"
        )

    endif()

endif()


message(STATUS "----------------------------------------")
message(STATUS "Simulator Configuration:")

message(
    STATUS
    "USE_MUJOCO: ${USE_MUJOCO}"
)

if(USE_MUJOCO)

    message(
        STATUS
        "MuJoCo: Enabled"
    )

endif()

message(STATUS "========================================")