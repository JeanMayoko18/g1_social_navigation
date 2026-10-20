/*
 * mola_cloud_udp_sender_node.cpp
 *
 * Sends the reduced MOLA PointCloud2 over UDP.
 *
 * Input:
 *   /mola/cloud_xyz_i
 *
 * UDP:
 *   127.0.0.1:16002
 *
 * This node must run with the ETH0 / Unitree DDS environment.
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

struct PacketHeader
{
    uint32_t msg_id;
    uint32_t fragment_id;
    uint32_t fragment_count;
    uint32_t payload_size;
    uint32_t total_size;
};

class MolaCloudUdpSender : public rclcpp::Node
{
public:
    MolaCloudUdpSender()
    : Node("mola_cloud_udp_sender_node")
    {
        input_cloud_topic_ = declare_parameter<std::string>(
            "input_cloud_topic", "/mola/cloud_xyz_i");

        udp_host_ = declare_parameter<std::string>(
            "udp_host", "127.0.0.1");

        udp_port_ = declare_parameter<int>(
            "udp_port", 16002);

        sock_ = socket(AF_INET, SOCK_DGRAM, 0);

        if (sock_ < 0)
            throw std::runtime_error("Failed to create UDP socket");

        target_addr_.sin_family = AF_INET;
        target_addr_.sin_port = htons(static_cast<uint16_t>(udp_port_));

        if (inet_pton(AF_INET, udp_host_.c_str(), &target_addr_.sin_addr) != 1)
            throw std::runtime_error("Invalid UDP host");

        /*
         * The input cloud is already reduced by livox_to_mola_cloud_node.
         * We use BEST_EFFORT because this is high-rate sensor data.
         */
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_cloud_topic_,
            qos,
            std::bind(
                &MolaCloudUdpSender::callback,
                this,
                std::placeholders::_1));

        RCLCPP_INFO(
            get_logger(),
            "[mola_cloud_udp_sender_node] %s -> UDP %s:%d",
            input_cloud_topic_.c_str(),
            udp_host_.c_str(),
            udp_port_);
    }

    ~MolaCloudUdpSender() override
    {
        if (sock_ >= 0)
        {
            close(sock_);
            sock_ = -1;
        }
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!msg)
            return;

        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
        rclcpp::SerializedMessage serialized_msg;

        serializer.serialize_message(msg.get(), &serialized_msg);

        const uint8_t * data =
            serialized_msg.get_rcl_serialized_message().buffer;

        const size_t total_size = serialized_msg.size();
        const uint32_t msg_id = ++msg_counter_;

        const uint32_t fragment_count =
            static_cast<uint32_t>(
                (total_size + MAX_UDP_FRAGMENT_PAYLOAD - 1) /
                MAX_UDP_FRAGMENT_PAYLOAD);

        for (uint32_t fragment_id = 0;
             fragment_id < fragment_count;
             ++fragment_id)
        {
            const size_t offset =
                static_cast<size_t>(fragment_id) *
                MAX_UDP_FRAGMENT_PAYLOAD;

            const size_t remaining = total_size - offset;

            const size_t payload_size =
                std::min(MAX_UDP_FRAGMENT_PAYLOAD, remaining);

            PacketHeader header;
            header.msg_id = msg_id;
            header.fragment_id = fragment_id;
            header.fragment_count = fragment_count;
            header.payload_size = static_cast<uint32_t>(payload_size);
            header.total_size = static_cast<uint32_t>(total_size);

            std::vector<uint8_t> packet(
                sizeof(PacketHeader) + payload_size);

            std::memcpy(packet.data(), &header, sizeof(PacketHeader));
            std::memcpy(
                packet.data() + sizeof(PacketHeader),
                data + offset,
                payload_size);

            const ssize_t sent =
                sendto(
                    sock_,
                    packet.data(),
                    packet.size(),
                    0,
                    reinterpret_cast<sockaddr*>(&target_addr_),
                    sizeof(target_addr_));

            if (sent < 0)
            {
                RCLCPP_ERROR_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    1000,
                    "UDP send failed: %s",
                    std::strerror(errno));
            }
        }

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "[MOLA CLOUD TX] size=%zu bytes fragments=%u",
            total_size,
            fragment_count);
    }

private:
    std::string input_cloud_topic_;
    std::string udp_host_;
    int udp_port_{16002};

    int sock_{-1};
    sockaddr_in target_addr_{};

    uint32_t msg_counter_{0};

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

    static constexpr size_t MAX_UDP_FRAGMENT_PAYLOAD = 1200;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MolaCloudUdpSender>());
    rclcpp::shutdown();
    return 0;
}