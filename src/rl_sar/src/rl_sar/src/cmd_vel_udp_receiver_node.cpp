// src/cmd_vel_udp_receiver_node.cpp

/*
 * Copyright (c) 2026
 *
 * UDP -> ROS 2 receiver for G1 velocity commands.
 *
 * Expected UDP packet format:
 *
 *   "vx vy wz\n"
 *
 * Example:
 *
 *   "0.120 0.000 -0.050\n"
 *
 * The node publishes geometry_msgs/msg/Twist on the configured
 * ROS 2 output topic.
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std::chrono_literals;


class CmdVelUdpReceiverNode : public rclcpp::Node
{
public:
    CmdVelUdpReceiverNode()
    : Node("cmd_vel_udp_receiver")
    {
        // ------------------------------------------------------------
        // Parameters
        // ------------------------------------------------------------
        output_topic_ = declare_parameter<std::string>(
            "output_topic",
            "/cmd_vel_rlsar"
        );

        udp_bind_address_ = declare_parameter<std::string>(
            "udp_bind_address",
            "127.0.0.1"
        );

        udp_port_ = declare_parameter<int>(
            "udp_port",
            15000
        );

        command_timeout_sec_ = declare_parameter<double>(
            "command_timeout_sec",
            0.35
        );

        publish_zero_on_timeout_ = declare_parameter<bool>(
            "publish_zero_on_timeout",
            true
        );

        receive_poll_period_ms_ = declare_parameter<int>(
            "receive_poll_period_ms",
            5
        );

        // Optional safety limits.
        max_abs_linear_x_ = declare_parameter<double>(
            "max_abs_linear_x",
            1.0
        );

        max_abs_linear_y_ = declare_parameter<double>(
            "max_abs_linear_y",
            1.0
        );

        max_abs_angular_z_ = declare_parameter<double>(
            "max_abs_angular_z",
            2.0
        );

        validate_parameters();

        // ------------------------------------------------------------
        // ROS publisher
        // ------------------------------------------------------------
        publisher_ = create_publisher<geometry_msgs::msg::Twist>(
            output_topic_,
            rclcpp::QoS(rclcpp::KeepLast(10)).reliable()
        );

        // ------------------------------------------------------------
        // UDP socket
        // ------------------------------------------------------------
        socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);

        if (socket_fd_ < 0)
        {
            throw std::runtime_error(
                std::string("Failed to create UDP socket: ")
                + std::strerror(errno)
            );
        }

        int reuse_address = 1;

        if (setsockopt(
                socket_fd_,
                SOL_SOCKET,
                SO_REUSEADDR,
                &reuse_address,
                sizeof(reuse_address)) < 0)
        {
            RCLCPP_WARN(
                get_logger(),
                "Failed to set SO_REUSEADDR: %s",
                std::strerror(errno)
            );
        }

        sockaddr_in bind_address{};
        bind_address.sin_family = AF_INET;
        bind_address.sin_port = htons(
            static_cast<uint16_t>(udp_port_)
        );

        if (inet_pton(
                AF_INET,
                udp_bind_address_.c_str(),
                &bind_address.sin_addr) != 1)
        {
            close_socket();

            throw std::runtime_error(
                "Invalid udp_bind_address: "
                + udp_bind_address_
            );
        }

        if (bind(
                socket_fd_,
                reinterpret_cast<sockaddr*>(&bind_address),
                sizeof(bind_address)) < 0)
        {
            const std::string error_message =
                std::string("Failed to bind UDP socket on ")
                + udp_bind_address_
                + ":"
                + std::to_string(udp_port_)
                + ": "
                + std::strerror(errno);

            close_socket();
            throw std::runtime_error(error_message);
        }

        // ------------------------------------------------------------
        // Configure non-blocking reception
        // ------------------------------------------------------------
        const int current_flags = fcntl(
            socket_fd_,
            F_GETFL,
            0
        );

        if (current_flags < 0)
        {
            const std::string error_message =
                std::string("Failed to read socket flags: ")
                + std::strerror(errno);

            close_socket();
            throw std::runtime_error(error_message);
        }

        if (fcntl(
                socket_fd_,
                F_SETFL,
                current_flags | O_NONBLOCK) < 0)
        {
            const std::string error_message =
                std::string("Failed to make UDP socket non-blocking: ")
                + std::strerror(errno);

            close_socket();
            throw std::runtime_error(error_message);
        }

        // ------------------------------------------------------------
        // Timers
        // ------------------------------------------------------------
        receive_timer_ = create_wall_timer(
            std::chrono::milliseconds(receive_poll_period_ms_),
            std::bind(
                &CmdVelUdpReceiverNode::receive_packets,
                this
            )
        );

        watchdog_timer_ = create_wall_timer(
            50ms,
            std::bind(
                &CmdVelUdpReceiverNode::check_command_timeout,
                this
            )
        );

        RCLCPP_INFO(
            get_logger(),
            "UDP cmd_vel receiver started"
        );

        RCLCPP_INFO(
            get_logger(),
            "UDP bind address: %s:%d",
            udp_bind_address_.c_str(),
            udp_port_
        );

        RCLCPP_INFO(
            get_logger(),
            "ROS output topic: %s",
            output_topic_.c_str()
        );

        RCLCPP_INFO(
            get_logger(),
            "Command timeout: %.3f s",
            command_timeout_sec_
        );
    }


    ~CmdVelUdpReceiverNode() override
    {
        close_socket();
    }


private:
    void validate_parameters()
    {
        if (udp_port_ <= 0 || udp_port_ > 65535)
        {
            throw std::runtime_error(
                "udp_port must be between 1 and 65535"
            );
        }

        if (command_timeout_sec_ <= 0.0)
        {
            throw std::runtime_error(
                "command_timeout_sec must be greater than zero"
            );
        }

        if (receive_poll_period_ms_ <= 0)
        {
            throw std::runtime_error(
                "receive_poll_period_ms must be greater than zero"
            );
        }

        if (max_abs_linear_x_ <= 0.0 ||
            max_abs_linear_y_ <= 0.0 ||
            max_abs_angular_z_ <= 0.0)
        {
            throw std::runtime_error(
                "Safety limits must be greater than zero"
            );
        }
    }


    void receive_packets()
    {
        if (socket_fd_ < 0)
        {
            return;
        }

        // Drain all currently available UDP packets.
        while (true)
        {
            std::array<char, 256> buffer{};

            sockaddr_in sender_address{};
            socklen_t sender_address_length =
                sizeof(sender_address);

            const ssize_t received_size = recvfrom(
                socket_fd_,
                buffer.data(),
                buffer.size() - 1,
                0,
                reinterpret_cast<sockaddr*>(&sender_address),
                &sender_address_length
            );

            if (received_size < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    return;
                }

                if (errno == EINTR)
                {
                    continue;
                }

                RCLCPP_ERROR_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    2000,
                    "UDP recvfrom failed: %s",
                    std::strerror(errno)
                );

                return;
            }

            if (received_size == 0)
            {
                continue;
            }

            buffer[static_cast<std::size_t>(received_size)] =
                '\0';

            const std::string packet(
                buffer.data(),
                static_cast<std::size_t>(received_size)
            );

            process_packet(packet);
        }
    }


    void process_packet(const std::string& packet)
    {
        double vx = 0.0;
        double vy = 0.0;
        double wz = 0.0;

        std::istringstream stream(packet);

        if (!(stream >> vx >> vy >> wz))
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Rejected malformed UDP cmd_vel packet: '%s'",
                packet.c_str()
            );

            return;
        }

        // Reject unexpected extra non-whitespace content.
        std::string unexpected_token;

        if (stream >> unexpected_token)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Rejected UDP packet with extra data: '%s'",
                packet.c_str()
            );

            return;
        }

        if (!std::isfinite(vx) ||
            !std::isfinite(vy) ||
            !std::isfinite(wz))
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Rejected non-finite UDP cmd_vel packet"
            );

            return;
        }

        // Clamp unexpected values as a final safety barrier.
        vx = std::clamp(
            vx,
            -max_abs_linear_x_,
            max_abs_linear_x_
        );

        vy = std::clamp(
            vy,
            -max_abs_linear_y_,
            max_abs_linear_y_
        );

        wz = std::clamp(
            wz,
            -max_abs_angular_z_,
            max_abs_angular_z_
        );

        geometry_msgs::msg::Twist command;
        command.linear.x = vx;
        command.linear.y = vy;
        command.linear.z = 0.0;

        command.angular.x = 0.0;
        command.angular.y = 0.0;
        command.angular.z = wz;

        publisher_->publish(command);

        last_valid_command_time_ =
            std::chrono::steady_clock::now();

        has_received_valid_command_ = true;

        if (timeout_active_)
        {
            timeout_active_ = false;

            RCLCPP_INFO(
                get_logger(),
                "UDP cmd_vel stream restored"
            );
        }

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            500,
            "[UDP RX CMD] vx=%.3f vy=%.3f wz=%.3f",
            vx,
            vy,
            wz
        );
    }

    void check_command_timeout()
    {
        // Do not publish a stop command before the first valid packet.
        if (!has_received_valid_command_)
        {
            return;
        }

        const auto now = std::chrono::steady_clock::now();

        const double elapsed_sec =
            std::chrono::duration<double>(
                now - last_valid_command_time_
            ).count();

        if (elapsed_sec <= command_timeout_sec_)
        {
            if (timeout_active_)
            {
                timeout_active_ = false;

                RCLCPP_INFO(
                    get_logger(),
                    "UDP cmd_vel stream restored"
                );
            }

            return;
        }
        // Publish zero only once per timeout event.
        if (!timeout_active_)
        {
            timeout_active_ = true;

            RCLCPP_ERROR(
                get_logger(),
                "UDP cmd_vel timeout after %.3f s. "
                "Publishing continuous stop commands.",
                elapsed_sec
            );
        }

        if (publish_zero_on_timeout_)
        {
            publish_stop_command();
        }
    }


    void publish_stop_command()
    {
        geometry_msgs::msg::Twist stop_command;
        publisher_->publish(stop_command);
    }


    void close_socket()
    {
        if (socket_fd_ >= 0)
        {
            close(socket_fd_);
            socket_fd_ = -1;
        }
    }


private:
    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    std::string output_topic_;
    std::string udp_bind_address_;

    int udp_port_{15000};
    int receive_poll_period_ms_{5};

    double command_timeout_sec_{0.35};
    double max_abs_linear_x_{1.0};
    double max_abs_linear_y_{1.0};
    double max_abs_angular_z_{2.0};

    bool publish_zero_on_timeout_{true};

    // ------------------------------------------------------------
    // ROS resources
    // ------------------------------------------------------------
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr
        publisher_;

    rclcpp::TimerBase::SharedPtr receive_timer_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;

    // ------------------------------------------------------------
    // UDP resources
    // ------------------------------------------------------------
    int socket_fd_{-1};

    // ------------------------------------------------------------
    // Watchdog state
    // ------------------------------------------------------------
    std::chrono::steady_clock::time_point
        last_valid_command_time_{};

    bool has_received_valid_command_{false};
    bool timeout_active_{false};
};


int main(int argc, char** argv)
{
    try
    {
        rclcpp::init(argc, argv);

        auto node =
            std::make_shared<CmdVelUdpReceiverNode>();

        rclcpp::spin(node);

        rclcpp::shutdown();
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "[cmd_vel_udp_receiver_node] Fatal exception: "
            << exception.what()
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}
