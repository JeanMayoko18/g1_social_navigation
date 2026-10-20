// src/ros2_nav_bridge_node.cpp

/*
 * Copyright (c) 2026
 *
 * ros2_nav_bridge_node.cpp
 * ------------------------------------------------------------
 * PURPOSE
 * ------------------------------------------------------------
 *
 * This node bridges ROS2 navigation commands (/cmd_vel)
 * to a lightweight UDP transport used by the Unitree-only
 * RL locomotion process.
 *
 * Architecture:
 *
 *   +--------------------+
 *   | Nav2 / Teleop ROS2 |
 *   +---------+----------+
 *             |
 *             | geometry_msgs/msg/Twist
 *             v
 *   +---------------------------+
 *   | ros2_nav_bridge_node      |
 *   |   ROS2  --> UDP bridge    |
 *   +-------------+-------------+
 *                 |
 *                 | UDP localhost
 *                 | "vx vy wz"
 *                 v
 *   +---------------------------+
 *   | rl_real_g1_edu23_unitree  |
 *   |   Unitree DDS process     |
 *   +---------------------------+
 *
 * ------------------------------------------------------------
 * WHY THIS NODE EXISTS
 * ------------------------------------------------------------
 *
 * The project intentionally separates:
 *
 *   1) ROS2 / Nav2 / DDS middleware
 *   2) Unitree DDS low-level control
 *
 * into TWO independent processes.
 *
 * This avoids:
 *   - DDS conflicts
 *   - ROS2 executor blocking
 *   - CycloneDDS/FastDDS incompatibilities
 *   - crashes propagating between navigation and locomotion
 *
 * The Unitree process receives velocity commands through UDP,
 * which is:
 *
 *   - lightweight
 *   - local
 *   - fast
 *   - independent from ROS middleware
 *
 * ------------------------------------------------------------
 * UDP MESSAGE FORMAT
 * ------------------------------------------------------------
 *
 * Each UDP packet contains:
 *
 *     "vx vy wz\n"
 *
 * Example:
 *
 *     "0.30 0.00 -0.25\n"
 *
 * where:
 *
 *   vx = forward linear velocity
 *   vy = lateral linear velocity
 *   wz = angular velocity around Z
 *
 * ------------------------------------------------------------
 * DEFAULT PARAMETERS
 * ------------------------------------------------------------
 *
 * cmd_vel_topic : "/cmd_vel_rlsar"
 * udp_host      : "127.0.0.1"
 * udp_port      : 15000
 *
 * ------------------------------------------------------------
 * EXAMPLE
 * ------------------------------------------------------------
 *
 * Terminal 1:
 *
 *   ros2 run rl_sar ros2_nav_bridge_node
 *
 * Terminal 2:
 *
 *   ros2 run rl_sar rl_real_g1_edu23_unitree eth0 --real
 *
 * ------------------------------------------------------------
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

// ------------------------------------------------------------
// Linux networking
// ------------------------------------------------------------
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

// ------------------------------------------------------------
// Standard utilities
// ------------------------------------------------------------
#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>


/**
 * @class Ros2NavBridgeNode
 *
 * @brief ROS2 → UDP bridge for navigation velocity commands.
 *
 * Responsibilities:
 *
 *   - Subscribe to a ROS2 cmd_vel topic
 *   - Convert Twist into compact UDP text message
 *   - Send the message to the Unitree-only locomotion process
 *
 * Threading:
 *
 *   - ROS2 callback thread only
 *   - No additional worker threads needed
 *
 * Safety:
 *
 *   - If UDP transmission fails, commands are simply lost
 *   - The Unitree process already has a timeout watchdog
 *     and will stop safely if packets stop arriving
 */
class Ros2NavBridgeNode : public rclcpp::Node
{
public:

    /**
     * @brief Constructor.
     *
     * Initializes:
     *   - ROS2 parameters
     *   - UDP socket
     *   - UDP target address
     *   - ROS2 cmd_vel subscriber
     */
    Ros2NavBridgeNode()
    : Node("ros2_nav_bridge_node")
    {
        // ------------------------------------------------------------
        // Parameter: ROS2 cmd_vel topic
        // ------------------------------------------------------------
        cmd_vel_topic_ =
            declare_parameter<std::string>(
                "cmd_vel_topic",
                "/cmd_vel_rlsar"
            );

        // ------------------------------------------------------------
        // Parameter: UDP target IP
        //
        // Usually localhost because both processes run
        // on the same computer.
        // ------------------------------------------------------------
        udp_host_ =
            declare_parameter<std::string>(
                "udp_host",
                "127.0.0.1"
            );

        // ------------------------------------------------------------
        // Parameter: UDP target port
        // ------------------------------------------------------------
        udp_port_ =
            declare_parameter<int>(
                "udp_port",
                15000
            );

        // ------------------------------------------------------------
        // Create UDP socket
        // ------------------------------------------------------------
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);

        if (sock_ < 0)
        {
            throw std::runtime_error(
                std::string("Failed to create UDP socket: ")
                + std::strerror(errno)
            );
        }

        // ------------------------------------------------------------
        // Configure destination address
        // ------------------------------------------------------------
        target_addr_.sin_family = AF_INET;

        target_addr_.sin_port =
            htons(static_cast<uint16_t>(udp_port_));

        // Convert IPv4 string → binary address
        if (inet_pton(
                AF_INET,
                udp_host_.c_str(),
                &target_addr_.sin_addr) != 1)
        {
            throw std::runtime_error(
                "Invalid udp_host IP address: " + udp_host_
            );
        }

        // ------------------------------------------------------------
        // Create ROS2 subscriber
        //
        // Receives velocity commands from:
        //   - Nav2
        //   - teleop
        //   - joystick bridge
        // ------------------------------------------------------------

        //sub_ =
        //    create_subscription<geometry_msgs::msg::Twist>(
        //        cmd_vel_topic_,
        //        rclcpp::SystemDefaultsQoS(),
        //        std::bind(
        //            &Ros2NavBridgeNode::CmdVelCallback,
        //            this,
        //            std::placeholders::_1
        //        )
        //    );

        sub_reliable_ =
            create_subscription<geometry_msgs::msg::Twist>(
                cmd_vel_topic_,
                rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
                std::bind(&Ros2NavBridgeNode::CmdVelCallback, this, std::placeholders::_1)
            );

/*
        sub_best_effort_ =
            create_subscription<geometry_msgs::msg::Twist>(
                cmd_vel_topic_,
                rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
                std::bind(&Ros2NavBridgeNode::CmdVelCallback, this, std::placeholders::_1)
            );
*/

        // ------------------------------------------------------------
        // Startup log
        // ------------------------------------------------------------
        RCLCPP_INFO(
            get_logger(),
            "ROS2 nav bridge started"
        );

        RCLCPP_INFO(
            get_logger(),
            "cmd_vel_topic = %s",
            cmd_vel_topic_.c_str()
        );

        RCLCPP_INFO(
            get_logger(),
            "UDP target = %s:%d",
            udp_host_.c_str(),
            udp_port_
        );
    }

    /**
     * @brief Destructor.
     *
     * Closes the UDP socket cleanly.
     */
    ~Ros2NavBridgeNode() override
    {
        if (sock_ >= 0)
        {
            close(sock_);
            sock_ = -1;
        }
    }

private:

    /**
     * @brief Callback executed whenever a new Twist arrives.
     *
     * Converts:
     *
     *   geometry_msgs/msg/Twist
     *
     * into:
     *
     *   "vx vy wz\n"
     *
     * and sends it over UDP.
     *
     * @param msg Incoming ROS2 velocity command.
     */
    void CmdVelCallback(
        const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        if (!msg)
            return;

        // ------------------------------------------------------------
        // Build compact text packet
        //
        // Format:
        //   "vx vy wz\n"
        // ------------------------------------------------------------
        std::ostringstream oss;

        oss << msg->linear.x << " "
            << msg->linear.y << " "
            << msg->angular.z << "\n";

        const std::string data = oss.str();

        // ------------------------------------------------------------
        // Send UDP packet
        // ------------------------------------------------------------
        const ssize_t sent =
            sendto(
                sock_,
                data.c_str(),
                data.size(),
                0,
                reinterpret_cast<sockaddr*>(&target_addr_),
                sizeof(target_addr_)
            );

        // ------------------------------------------------------------
        // Error handling
        // ------------------------------------------------------------
        if (sent < 0)
        {
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "UDP send failed: %s",
                std::strerror(errno)
            );

            return;
        }

        // ------------------------------------------------------------
        // Debug log (throttled)
        // ------------------------------------------------------------
        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            300,
            "[UDP CMD] vx=%.3f vy=%.3f wz=%.3f",
            msg->linear.x,
            msg->linear.y,
            msg->angular.z
        );
    }

private:

    // ------------------------------------------------------------
    // ROS2 parameters
    // ------------------------------------------------------------

    /// ROS2 cmd_vel topic name
    std::string cmd_vel_topic_;

    /// UDP target IP address
    std::string udp_host_;

    /// UDP target port
    int udp_port_{15000};

    // ------------------------------------------------------------
    // UDP socket resources
    // ------------------------------------------------------------

    /// UDP socket file descriptor
    int sock_{-1};

    /// Destination socket address
    sockaddr_in target_addr_{};

    // ------------------------------------------------------------
    // ROS2 subscriber
    // ------------------------------------------------------------

    //rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_reliable_;
    //rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_best_effort_;
};


/**
 * @brief Main entry point.
 */
int main(int argc, char** argv)
{
    try
    {
        // ------------------------------------------------------------
        // Initialize ROS2
        // ------------------------------------------------------------
        rclcpp::init(argc, argv);

        // ------------------------------------------------------------
        // Create node
        // ------------------------------------------------------------
        auto node =
            std::make_shared<Ros2NavBridgeNode>();

        // ------------------------------------------------------------
        // Spin ROS2 executor
        // ------------------------------------------------------------
        rclcpp::spin(node);

        // ------------------------------------------------------------
        // Shutdown ROS2
        // ------------------------------------------------------------
        rclcpp::shutdown();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[ros2_nav_bridge_node] Fatal exception: "
            << e.what()
            << std::endl;

        if (rclcpp::ok())
            rclcpp::shutdown();

        return 1;
    }
}