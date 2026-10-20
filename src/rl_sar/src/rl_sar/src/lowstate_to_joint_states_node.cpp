
#include "g1_edu23_joints.hpp"

#include <unitree_hg/msg/low_state.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * lowstate_to_joint_states_node
 *
 * ROBOT SIDE - Unitree eth0 DDS
 *
 * Input:
 *   /lowstate
 *
 * Output:
 *   /joint_states_source
 *
 * Purpose:
 *   Convert Unitree LowState into JointState for
 *   robot_state_publisher.
 *
 * Reliability:
 *   - Reliable LowState subscription, KEEP_LAST(1).
 *   - Reliable JointState publication, KEEP_LAST(5).
 *   - Latest articulated state is retained.
 *   - Publication timer runs at 20 Hz by default.
 *   - Each received state is published at most once.
 *   - Stale states are never republished.
 *
 * Important:
 *   This node publishes measurements only.
 *   It does not invent joint positions when LowState
 *   communication is interrupted.
 */

static YAML::Node find_key_recursive(
    const YAML::Node & node,
    const std::string & key)
{
    if (!node)
        return YAML::Node();

    if (node.IsMap())
    {
        if (node[key])
            return node[key];

        for (auto it = node.begin(); it != node.end(); ++it)
        {
            YAML::Node found =
                find_key_recursive(it->second, key);

            if (found)
                return found;
        }
    }

    if (node.IsSequence())
    {
        for (std::size_t i = 0; i < node.size(); ++i)
        {
            YAML::Node found =
                find_key_recursive(node[i], key);

            if (found)
                return found;
        }
    }

    return YAML::Node();
}

class LowStateToJointStates : public rclcpp::Node
{
public:

    LowStateToJointStates()
        : Node("lowstate_to_joint_states_node")
    {
        // --------------------------------------------------
        // Parameters
        // --------------------------------------------------

        output_topic_ = declare_parameter<std::string>(
            "output_topic",
            "/joint_states_source"
        );

        lowstate_topic_ = declare_parameter<std::string>(
            "lowstate_topic",
            "/lowstate"
        );

        base_yaml_path_ = declare_parameter<std::string>(
            "base_yaml_path",
            "/home/unitree/g1_ws/src/rl_sar/"
            "policy/g1/base_edu23.yaml"
        );

        policy_yaml_path_ = declare_parameter<std::string>(
            "policy_yaml_path",
            "/home/unitree/g1_ws/src/rl_sar/"
            "policy/g1/robomimic/locomotion/config.yaml"
        );

        // Default and launch configuration: 20 Hz.
        joint_pub_rate_hz_ = declare_parameter<double>(
            "joint_pub_rate_hz",
            20.0
        );

        // Maximum age of an accepted measurement.
        max_joint_age_sec_ = declare_parameter<double>(
            "max_joint_age_sec",
            0.2
        );

        if (joint_pub_rate_hz_ <= 0.0)
        {
            throw std::runtime_error(
                "joint_pub_rate_hz must be greater than zero"
            );
        }

        if (max_joint_age_sec_ <= 0.0)
        {
            throw std::runtime_error(
                "max_joint_age_sec must be greater than zero"
            );
        }

        // --------------------------------------------------
        // EDU23 joint configuration
        // --------------------------------------------------

        joint_names_ = GetEdu23ActiveJointNames();

        load_mapping_from_yaml();

        // --------------------------------------------------
        // JointState publisher
        // RELIABLE / KEEP_LAST(5)
        // --------------------------------------------------

        rclcpp::QoS joint_qos(rclcpp::KeepLast(5));

        joint_qos.reliable();
        joint_qos.durability_volatile();

        joint_pub_ =
            create_publisher<sensor_msgs::msg::JointState>(
                output_topic_,
                joint_qos
            );

        // --------------------------------------------------
        // Unitree LowState subscriber
        // RELIABLE / KEEP_LAST(1)
        // --------------------------------------------------

        rclcpp::QoS lowstate_qos(rclcpp::KeepLast(1));

        lowstate_qos.reliable();
        lowstate_qos.durability_volatile();

        lowstate_sub_ =
            create_subscription<unitree_hg::msg::LowState>(
                lowstate_topic_,
                lowstate_qos,
                std::bind(
                    &LowStateToJointStates::LowStateCallback,
                    this,
                    std::placeholders::_1
                )
            );

        // --------------------------------------------------
        // JointState publication timer
        //
        // Created AFTER reading joint_pub_rate_hz_.
        // --------------------------------------------------

        joint_timer_ = create_wall_timer(
            std::chrono::duration<double>(
                1.0 / joint_pub_rate_hz_
            ),
            std::bind(
                &LowStateToJointStates::PublishLatestJointState,
                this
            )
        );

        RCLCPP_INFO(
            get_logger(),
            "LowStateToJointStates started | "
            "input=%s | output=%s | joints=%zu | "
            "rate=%.1f Hz | max_age=%.3f s",
            lowstate_topic_.c_str(),
            output_topic_.c_str(),
            joint_names_.size(),
            joint_pub_rate_hz_,
            max_joint_age_sec_
        );
    }

private:

    // ------------------------------------------------------
    // Load EDU23 mapping
    // ------------------------------------------------------

    void load_mapping_from_yaml()
    {
        YAML::Node base =
            YAML::LoadFile(base_yaml_path_);

        YAML::Node policy =
            YAML::LoadFile(policy_yaml_path_);

        YAML::Node jm_node =
            find_key_recursive(policy, "joint_mapping");

        if (!jm_node)
        {
            throw std::runtime_error(
                "Missing key 'joint_mapping' in policy YAML: "
                + policy_yaml_path_
            );
        }

        YAML::Node hw_node =
            find_key_recursive(base, "hw23_to_ms");

        if (!hw_node)
        {
            throw std::runtime_error(
                "Missing key 'hw23_to_ms' in base YAML: "
                + base_yaml_path_
            );
        }

        joint_mapping_ =
            jm_node.as<std::vector<int>>();

        hw23_to_ms_ =
            hw_node.as<std::vector<int>>();

        if (joint_mapping_.size() != 29)
        {
            throw std::runtime_error(
                "joint_mapping must have 29 entries, got "
                + std::to_string(joint_mapping_.size())
            );
        }

        if (hw23_to_ms_.size() != 23)
        {
            throw std::runtime_error(
                "hw23_to_ms must have 23 entries, got "
                + std::to_string(hw23_to_ms_.size())
            );
        }

        RCLCPP_INFO(
            get_logger(),
            "Loaded mapping | joint_mapping=%zu | "
            "hw23_to_ms=%zu",
            joint_mapping_.size(),
            hw23_to_ms_.size()
        );
    }

    // ------------------------------------------------------
    // Receive LowState and convert to JointState
    //
    // Does not publish directly.
    // ------------------------------------------------------

    void LowStateCallback(
        const unitree_hg::msg::LowState::SharedPtr msg)
    {
        if (!msg)
            return;

        const auto & motor_states = msg->motor_state;

        const rclcpp::Time now_t = this->now();

        sensor_msgs::msg::JointState js;

        js.header.stamp = now_t;

        js.name = joint_names_;

        js.position.reserve(joint_names_.size());
        js.velocity.reserve(joint_names_.size());
        js.effort.reserve(joint_names_.size());

        /**
         * EDU23 mapping:
         *
         *   hw29 index p
         *       -> kPolicyToHw23[p]
         *       -> EDU23 slot
         *       -> hw23_to_ms_[slot]
         *       -> Unitree motor_state index
         *
         * Preserves the mapping used by the source
         * file provided.
         */

        for (int p = 0; p < 29; ++p)
        {
            if (!is_edu23_active_joint(p))
                continue;

            const int hw29 = p;

            if (hw29 < 0 || hw29 >= 29)
            {
                push_zero(js);
                continue;
            }

            const int h23 = kPolicyToHw23[hw29];

            if (
                h23 < 0 ||
                h23 >= static_cast<int>(
                    hw23_to_ms_.size()
                )
            )
            {
                push_zero(js);
                continue;
            }

            const int ms_idx = hw23_to_ms_[h23];

            if (
                ms_idx < 0 ||
                ms_idx >= static_cast<int>(
                    motor_states.size()
                )
            )
            {
                push_zero(js);
                continue;
            }

            js.position.push_back(
                motor_states[ms_idx].q
            );

            js.velocity.push_back(
                motor_states[ms_idx].dq
            );

            js.effort.push_back(
                motor_states[ms_idx].tau_est
            );
        }

        // --------------------------------------------------
        // Validate joint arrays
        // --------------------------------------------------

        if (
            js.position.size() != js.name.size() ||
            js.velocity.size() != js.name.size() ||
            js.effort.size() != js.name.size()
        )
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "JointState size mismatch: "
                "names=%zu positions=%zu "
                "velocities=%zu efforts=%zu",
                js.name.size(),
                js.position.size(),
                js.velocity.size(),
                js.effort.size()
            );

            return;
        }

        // --------------------------------------------------
        // Retain only the latest received state
        // --------------------------------------------------

        latest_joint_state_ = std::move(js);

        latest_joint_rx_time_ = now_t;

        has_joint_state_ = true;

        ++latest_joint_sequence_;
    }

    // ------------------------------------------------------
    // Publish at configured rate
    //
    // Called by wall timer.
    // ------------------------------------------------------

    void PublishLatestJointState()
    {
        if (!has_joint_state_)
            return;

        // Do not publish the same received state twice.
        if (
            latest_joint_sequence_ ==
            last_published_joint_sequence_
        )
        {
            return;
        }

        const rclcpp::Time now_t = this->now();

        const double age =
            (now_t - latest_joint_rx_time_).seconds();

        // No publication of stale or future-dated data.
        if (age < 0.0 || age > max_joint_age_sec_)
        {
            return;
        }

        // Preserve the acquisition timestamp.
        joint_pub_->publish(latest_joint_state_);

        last_published_joint_sequence_ =
            latest_joint_sequence_;
    }

    // ------------------------------------------------------
    // Zero fallback for the existing EDU23 mapping
    // ------------------------------------------------------

    static void push_zero(
        sensor_msgs::msg::JointState & js)
    {
        js.position.push_back(0.0);
        js.velocity.push_back(0.0);
        js.effort.push_back(0.0);
    }

private:

    // Configuration
    std::string output_topic_;
    std::string lowstate_topic_;

    std::string base_yaml_path_;
    std::string policy_yaml_path_;

    double joint_pub_rate_hz_ = 20.0;
    double max_joint_age_sec_ = 0.2;

    // Joint mappings
    std::vector<std::string> joint_names_;

    std::vector<int> joint_mapping_;
    std::vector<int> hw23_to_ms_;

    // Latest articulated state
    sensor_msgs::msg::JointState latest_joint_state_;

    rclcpp::Time latest_joint_rx_time_;

    bool has_joint_state_ = false;

    // Used to prevent duplicate publication
    std::uint64_t latest_joint_sequence_ = 0;
    std::uint64_t last_published_joint_sequence_ = 0;

    // ROS interfaces
    rclcpp::Publisher<
        sensor_msgs::msg::JointState
    >::SharedPtr joint_pub_;

    rclcpp::Subscription<
        unitree_hg::msg::LowState
    >::SharedPtr lowstate_sub_;

    rclcpp::TimerBase::SharedPtr joint_timer_;
};

int main(int argc, char ** argv)
{
    try
    {
        rclcpp::init(argc, argv);

        auto node =
            std::make_shared<LowStateToJointStates>();

        // Single-threaded executor.
        rclcpp::spin(node);

        rclcpp::shutdown();

        return 0;
    }
    catch (const std::exception & e)
    {
        std::cerr
            << "[lowstate_to_joint_states_node] "
            << "Fatal exception: "
            << e.what()
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
    catch (...)
    {
        std::cerr
            << "[lowstate_to_joint_states_node] "
            << "Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}
