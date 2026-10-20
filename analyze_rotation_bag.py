#!/usr/bin/env python3

import argparse
import math
from collections import defaultdict

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# =============================================================================
# CONFIGURATION
# =============================================================================

COMMAND_TOPIC_PRIORITY = [
    "/cmd_vel_rlsar",
    "/cmd_vel_mux",
    "/cmd_vel_nav_scaled",
    "/cmd_vel",
]

ODOMETRY_TOPICS = [
    "/dog_odom_2d",
    "/odom_dog_calibrated",
    "/odom",
]

IMU_TOPICS = [
    "/dog_imu_raw",
    "/livox/imu",
]

ROTATION_THRESHOLD_RAD_S = 0.02
LINEAR_THRESHOLD_M_S = 0.02

MINIMUM_INTERVAL_DURATION_S = 0.50
INTERVAL_GAP_TOLERANCE_S = 0.30

# Stationary data before each rotation are used to estimate gyro bias.
BIAS_WINDOW_DURATION_S = 3.0
BIAS_WINDOW_MARGIN_S = 0.10

# Gyro values smaller than this after bias removal are considered noise.
GYRO_SIGNIFICANCE_THRESHOLD_RAD_S = 0.005

# Odom yaw increments smaller than this value are ignored by the
# wrong-direction metric.
ODOM_YAW_INCREMENT_THRESHOLD_DEG = 0.05


# =============================================================================
# GEOMETRY
# =============================================================================

def quaternion_to_yaw(quaternion):
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )

    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )


def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


# =============================================================================
# BAG READING
# =============================================================================

def load_bag(bag_path):
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(
        storage_options,
        converter_options,
    )

    topic_types = {
        information.name: information.type
        for information in reader.get_all_topics_and_types()
    }

    messages = defaultdict(list)

    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()

        if topic not in topic_types:
            continue

        message_type = get_message(
            topic_types[topic]
        )

        message = deserialize_message(
            serialized_data,
            message_type,
        )

        messages[topic].append(
            (
                timestamp_ns * 1.0e-9,
                message,
            )
        )

    return messages, topic_types


# =============================================================================
# COMMAND ANALYSIS
# =============================================================================

def select_command_topic(messages):
    for topic in COMMAND_TOPIC_PRIORITY:
        if messages.get(topic):
            return topic

    return None


def extract_rotation_intervals(command_samples):
    active_samples = []

    for timestamp, message in command_samples:
        linear_speed = math.hypot(
            message.linear.x,
            message.linear.y,
        )

        angular_speed = message.angular.z

        pure_rotation = (
            abs(angular_speed) >= ROTATION_THRESHOLD_RAD_S
            and linear_speed <= LINEAR_THRESHOLD_M_S
        )

        if pure_rotation:
            active_samples.append(
                (
                    timestamp,
                    angular_speed,
                )
            )

    if not active_samples:
        return []

    intervals = []
    current_interval = [active_samples[0]]

    for sample in active_samples[1:]:
        previous_timestamp = current_interval[-1][0]
        current_timestamp = sample[0]

        previous_sign = math.copysign(
            1.0,
            current_interval[-1][1],
        )

        current_sign = math.copysign(
            1.0,
            sample[1],
        )

        same_interval = (
            current_timestamp - previous_timestamp
            <= INTERVAL_GAP_TOLERANCE_S
            and previous_sign == current_sign
        )

        if same_interval:
            current_interval.append(sample)
        else:
            intervals.append(current_interval)
            current_interval = [sample]

    intervals.append(current_interval)

    valid_intervals = []

    for interval in intervals:
        duration = (
            interval[-1][0]
            - interval[0][0]
        )

        if duration >= MINIMUM_INTERVAL_DURATION_S:
            valid_intervals.append(interval)

    return valid_intervals


def integrate_commanded_yaw(interval):
    integrated_yaw = 0.0

    for index in range(1, len(interval)):
        previous_time, previous_wz = interval[index - 1]
        current_time, current_wz = interval[index]

        dt = current_time - previous_time

        if dt <= 0.0:
            continue

        average_wz = 0.5 * (
            previous_wz + current_wz
        )

        integrated_yaw += average_wz * dt

    return integrated_yaw


# =============================================================================
# ODOMETRY EXTRACTION
# =============================================================================

def extract_odometry_samples(
    samples,
    start_time,
    end_time,
):
    extracted = []

    for timestamp, message in samples:
        if timestamp < start_time or timestamp > end_time:
            continue

        pose = message.pose.pose

        extracted.append(
            (
                timestamp,
                pose.position.x,
                pose.position.y,
                quaternion_to_yaw(
                    pose.orientation
                ),
            )
        )

    return extracted


def extract_tf_samples(
    samples,
    start_time,
    end_time,
    parent_frame="odom",
    child_frame="base_footprint",
):
    extracted = []

    for timestamp, message in samples:
        if timestamp < start_time or timestamp > end_time:
            continue

        for transform in message.transforms:
            parent = transform.header.frame_id.lstrip("/")
            child = transform.child_frame_id.lstrip("/")

            if parent != parent_frame:
                continue

            if child != child_frame:
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            extracted.append(
                (
                    timestamp,
                    translation.x,
                    translation.y,
                    quaternion_to_yaw(rotation),
                )
            )

    return extracted


# =============================================================================
# ODOMETRY ANALYSIS
# =============================================================================

def analyze_pose_samples(
    samples,
    command_sign,
):
    if len(samples) < 2:
        return None

    start_time, start_x, start_y, _ = samples[0]
    end_time, end_x, end_y, _ = samples[-1]

    cumulative_yaw = 0.0
    traveled_distance = 0.0

    wrong_direction_count = 0
    significant_increment_count = 0

    maximum_translation_step = 0.0
    maximum_yaw_step = 0.0

    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]

        delta_x = current[1] - previous[1]
        delta_y = current[2] - previous[2]

        translation_step = math.hypot(
            delta_x,
            delta_y,
        )

        yaw_step = normalize_angle(
            current[3] - previous[3]
        )

        cumulative_yaw += yaw_step
        traveled_distance += translation_step

        maximum_translation_step = max(
            maximum_translation_step,
            translation_step,
        )

        maximum_yaw_step = max(
            maximum_yaw_step,
            abs(yaw_step),
        )

        if abs(math.degrees(yaw_step)) >= (
            ODOM_YAW_INCREMENT_THRESHOLD_DEG
        ):
            significant_increment_count += 1

            if yaw_step * command_sign < 0.0:
                wrong_direction_count += 1

    net_translation = math.hypot(
        end_x - start_x,
        end_y - start_y,
    )

    duration = end_time - start_time

    frequency_hz = 0.0

    if duration > 0.0:
        frequency_hz = (
            len(samples) - 1
        ) / duration

    wrong_direction_percentage = 0.0

    if significant_increment_count > 0:
        wrong_direction_percentage = (
            100.0
            * wrong_direction_count
            / significant_increment_count
        )

    return {
        "sample_count": len(samples),
        "duration_s": duration,
        "frequency_hz": frequency_hz,
        "yaw_change_rad": cumulative_yaw,
        "yaw_change_deg": math.degrees(cumulative_yaw),
        "net_translation_m": net_translation,
        "traveled_distance_m": traveled_distance,
        "maximum_translation_step_m": maximum_translation_step,
        "maximum_yaw_step_deg": math.degrees(maximum_yaw_step),
        "wrong_direction_percentage": wrong_direction_percentage,
    }


# =============================================================================
# IMU EXTRACTION AND ANALYSIS
# =============================================================================

def extract_imu_samples(
    samples,
    start_time,
    end_time,
):
    extracted = []

    for timestamp, message in samples:
        if timestamp < start_time or timestamp > end_time:
            continue

        if not hasattr(message, "angular_velocity"):
            continue

        wz = message.angular_velocity.z

        if not math.isfinite(wz):
            continue

        extracted.append(
            (
                timestamp,
                wz,
            )
        )

    return extracted


def calculate_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_standard_deviation(values):
    if len(values) < 2:
        return 0.0

    mean_value = calculate_mean(values)

    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / (len(values) - 1)

    return math.sqrt(variance)


def estimate_gyro_bias(
    samples,
    rotation_start_time,
):
    bias_start_time = (
        rotation_start_time
        - BIAS_WINDOW_DURATION_S
    )

    bias_end_time = (
        rotation_start_time
        - BIAS_WINDOW_MARGIN_S
    )

    stationary_samples = extract_imu_samples(
        samples,
        bias_start_time,
        bias_end_time,
    )

    values = [
        wz
        for _, wz in stationary_samples
    ]

    if not values:
        return {
            "available": False,
            "sample_count": 0,
            "bias_rad_s": 0.0,
            "standard_deviation_rad_s": 0.0,
        }

    return {
        "available": True,
        "sample_count": len(values),
        "bias_rad_s": calculate_mean(values),
        "standard_deviation_rad_s":
            calculate_standard_deviation(values),
    }


def analyze_imu_samples(
    samples,
    command_sign,
    bias_rad_s,
):
    if len(samples) < 2:
        return None

    integrated_yaw = 0.0

    wrong_direction_count = 0
    significant_sample_count = 0

    corrected_values = []

    for _, raw_wz in samples:
        corrected_wz = raw_wz - bias_rad_s
        corrected_values.append(corrected_wz)

        if abs(corrected_wz) >= (
            GYRO_SIGNIFICANCE_THRESHOLD_RAD_S
        ):
            significant_sample_count += 1

            if corrected_wz * command_sign < 0.0:
                wrong_direction_count += 1

    for index in range(1, len(samples)):
        previous_time, previous_raw_wz = samples[index - 1]
        current_time, current_raw_wz = samples[index]

        dt = current_time - previous_time

        if dt <= 0.0:
            continue

        previous_corrected_wz = (
            previous_raw_wz - bias_rad_s
        )

        current_corrected_wz = (
            current_raw_wz - bias_rad_s
        )

        average_corrected_wz = 0.5 * (
            previous_corrected_wz
            + current_corrected_wz
        )

        integrated_yaw += (
            average_corrected_wz * dt
        )

    duration = samples[-1][0] - samples[0][0]

    frequency_hz = 0.0

    if duration > 0.0:
        frequency_hz = (
            len(samples) - 1
        ) / duration

    wrong_direction_percentage = 0.0

    if significant_sample_count > 0:
        wrong_direction_percentage = (
            100.0
            * wrong_direction_count
            / significant_sample_count
        )

    return {
        "sample_count": len(samples),
        "duration_s": duration,
        "frequency_hz": frequency_hz,
        "mean_corrected_wz_rad_s":
            calculate_mean(corrected_values),
        "standard_deviation_rad_s":
            calculate_standard_deviation(corrected_values),
        "maximum_absolute_wz_rad_s":
            max(abs(value) for value in corrected_values),
        "integrated_yaw_rad": integrated_yaw,
        "integrated_yaw_deg": math.degrees(integrated_yaw),
        "wrong_direction_percentage":
            wrong_direction_percentage,
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_odometry_result(
    name,
    result,
    commanded_yaw,
):
    print("  {}".format(name))

    if result is None:
        print("    Status                       : insufficient data")
        return

    yaw_ratio = float("nan")

    if abs(commanded_yaw) > 1.0e-6:
        yaw_ratio = (
            result["yaw_change_rad"]
            / commanded_yaw
        )

    print(
        "    Samples                      : {}".format(
            result["sample_count"]
        )
    )
    print(
        "    Frequency                    : {:.2f} Hz".format(
            result["frequency_hz"]
        )
    )
    print(
        "    Yaw change                   : {:+.3f} deg".format(
            result["yaw_change_deg"]
        )
    )
    print(
        "    Yaw/command ratio            : {:+.3f}".format(
            yaw_ratio
        )
    )
    print(
        "    Net translation              : {:.4f} m".format(
            result["net_translation_m"]
        )
    )
    print(
        "    Cumulative translation       : {:.4f} m".format(
            result["traveled_distance_m"]
        )
    )
    print(
        "    Maximum translation step     : {:.4f} m".format(
            result["maximum_translation_step_m"]
        )
    )
    print(
        "    Maximum yaw step             : {:.3f} deg".format(
            result["maximum_yaw_step_deg"]
        )
    )
    print(
        "    Opposite-direction increments: {:.1f} %".format(
            result["wrong_direction_percentage"]
        )
    )


def print_imu_result(
    name,
    bias_result,
    imu_result,
    commanded_yaw,
):
    print("  {}".format(name))

    if imu_result is None:
        print("    Status                       : insufficient data")
        return

    integrated_yaw_ratio = float("nan")

    if abs(commanded_yaw) > 1.0e-6:
        integrated_yaw_ratio = (
            imu_result["integrated_yaw_rad"]
            / commanded_yaw
        )

    if bias_result["available"]:
        print(
            "    Bias samples                 : {}".format(
                bias_result["sample_count"]
            )
        )
        print(
            "    Estimated stationary bias    : {:+.6f} rad/s".format(
                bias_result["bias_rad_s"]
            )
        )
        print(
            "    Stationary gyro std. dev.    : {:.6f} rad/s".format(
                bias_result["standard_deviation_rad_s"]
            )
        )
    else:
        print(
            "    Stationary bias              : unavailable; using 0 rad/s"
        )

    print(
        "    Rotation samples             : {}".format(
            imu_result["sample_count"]
        )
    )
    print(
        "    Frequency                    : {:.2f} Hz".format(
            imu_result["frequency_hz"]
        )
    )
    print(
        "    Mean bias-corrected wz       : {:+.6f} rad/s".format(
            imu_result["mean_corrected_wz_rad_s"]
        )
    )
    print(
        "    Rotation gyro std. dev.      : {:.6f} rad/s".format(
            imu_result["standard_deviation_rad_s"]
        )
    )
    print(
        "    Maximum absolute wz          : {:.6f} rad/s".format(
            imu_result["maximum_absolute_wz_rad_s"]
        )
    )
    print(
        "    Integrated gyro yaw          : {:+.3f} deg".format(
            imu_result["integrated_yaw_deg"]
        )
    )
    print(
        "    Gyro yaw/command ratio       : {:+.3f}".format(
            integrated_yaw_ratio
        )
    )
    print(
        "    Opposite-direction samples   : {:.1f} %".format(
            imu_result["wrong_direction_percentage"]
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze G1 pure-rotation commands, odometry and IMU "
            "measurements stored in a ROS 2 bag."
        )
    )

    parser.add_argument(
        "bag",
        help="ROS 2 bag directory.",
    )

    args = parser.parse_args()

    messages, topic_types = load_bag(
        args.bag
    )

    command_topic = select_command_topic(
        messages
    )

    if command_topic is None:
        available_command_topics = sorted(
            topic
            for topic in messages
            if "cmd_vel" in topic
            and messages.get(topic)
        )

        raise RuntimeError(
            "No command messages were recorded. "
            "Available populated command topics: {}".format(
                available_command_topics
            )
        )

    intervals = extract_rotation_intervals(
        messages[command_topic]
    )

    print(
        "Analyzed command topic : {}".format(
            command_topic
        )
    )
    print(
        "Detected intervals     : {}".format(
            len(intervals)
        )
    )

    if not intervals:
        print(
            "No sufficiently long pure-rotation interval was detected."
        )
        return

    for interval_index, interval in enumerate(
        intervals,
        start=1,
    ):
        start_time = interval[0][0]
        end_time = interval[-1][0]

        average_wz = calculate_mean(
            [
                wz
                for _, wz in interval
            ]
        )

        command_sign = (
            1.0 if average_wz > 0.0 else -1.0
        )

        direction = (
            "LEFT / CCW"
            if command_sign > 0.0
            else "RIGHT / CW"
        )

        commanded_yaw = integrate_commanded_yaw(
            interval
        )

        print()
        print("=" * 72)
        print(
            "ROTATION {} — {}".format(
                interval_index,
                direction,
            )
        )
        print("=" * 72)

        print(
            "Command duration              : {:.3f} s".format(
                end_time - start_time
            )
        )
        print(
            "Mean commanded wz             : {:+.4f} rad/s".format(
                average_wz
            )
        )
        print(
            "Integrated commanded yaw      : {:+.3f} deg".format(
                math.degrees(commanded_yaw)
            )
        )

        print()
        print("ODOMETRY RESULTS")
        print("-" * 72)

        for topic in ODOMETRY_TOPICS:
            pose_samples = extract_odometry_samples(
                messages.get(topic, []),
                start_time,
                end_time,
            )

            result = analyze_pose_samples(
                pose_samples,
                command_sign,
            )

            print_odometry_result(
                topic,
                result,
                commanded_yaw,
            )

        tf_samples = extract_tf_samples(
            messages.get("/tf", []),
            start_time,
            end_time,
        )

        tf_result = analyze_pose_samples(
            tf_samples,
            command_sign,
        )

        print_odometry_result(
            "TF odom -> base_footprint",
            tf_result,
            commanded_yaw,
        )

        print()
        print("IMU RESULTS")
        print("-" * 72)

        for topic in IMU_TOPICS:
            bias_result = estimate_gyro_bias(
                messages.get(topic, []),
                start_time,
            )

            rotation_samples = extract_imu_samples(
                messages.get(topic, []),
                start_time,
                end_time,
            )

            imu_result = analyze_imu_samples(
                rotation_samples,
                command_sign,
                bias_result["bias_rad_s"],
            )

            print_imu_result(
                topic,
                bias_result,
                imu_result,
                commanded_yaw,
            )


if __name__ == "__main__":
    main()

'''
ros2 bag record \
  /cmd_vel \
  /cmd_vel_rlsar \
  /cmd_vel_nav_scaled \
  /dog_odom_2d \
  /odom_dog_calibrated \
  /odom \
  /dog_imu_raw \
  /livox/imu \
  /amcl_pose \
  /tf

  timeout 10s ros2 topic pub -r 20 \
  /cmd_vel \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.12}}"

ros2 topic pub --once \
  /cmd_vel_rlsar \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

'''