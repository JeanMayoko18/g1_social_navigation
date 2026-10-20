/*
 * mola_cloud_udp_receiver_node.cpp
 *
 * Receives reduced MOLA PointCloud2 from UDP and republishes it
 * on the Wi-Fi DDS side.
 *
 * Output:
 *   /repub/mola/cloud_xyz_i
 *
 * UDP:
 *   port 16002
 *
 * This node must run with the WIFI / wlan0 CycloneDDS environment.
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <map>
#include <thread>
#include <utility>
#include <vector>

struct PacketHeader
{
    uint32_t msg_id;
    uint32_t fragment_id;
    uint32_t fragment_count;
    uint32_t payload_size;
    uint32_t total_size;
};

struct ReassemblyBuffer
{
    uint32_t total_size{0};
    uint32_t fragment_count{0};
    uint32_t received_count{0};

    std::vector<uint8_t> data;
    std::vector<bool> received;
};

class MolaCloudUdpReceiver : public rclcpp::Node
{
public:
    MolaCloudUdpReceiver()
    : Node("mola_cloud_udp_receiver_node")
    {
        output_cloud_topic_ = declare_parameter<std::string>(
            "output_cloud_topic", "/repub/mola/cloud_xyz_i");

        udp_port_ = declare_parameter<int>(
            "udp_port", 16002);

        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            output_cloud_topic_,
            qos);

        sock_ = socket(AF_INET, SOCK_DGRAM, 0);

        if (sock_ < 0)
            throw std::runtime_error("Failed to create UDP socket");

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(udp_port_));

        if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
            throw std::runtime_error("Failed to bind UDP socket");

        running_ = true;

        worker_ = std::thread(
            [this]()
            {
                receiveLoop();
            });

        RCLCPP_INFO(
            get_logger(),
            "[mola_cloud_udp_receiver_node] UDP %d -> %s",
            udp_port_,
            output_cloud_topic_.c_str());
    }

    ~MolaCloudUdpReceiver() override
    {
        running_ = false;

        if (sock_ >= 0)
        {
            close(sock_);
            sock_ = -1;
        }

        if (worker_.joinable())
            worker_.join();
    }

private:
    void receiveLoop()
    {
        std::vector<uint8_t> buffer(65535);

        while (running_ && rclcpp::ok())
        {
            const ssize_t received =
                recvfrom(
                    sock_,
                    buffer.data(),
                    buffer.size(),
                    0,
                    nullptr,
                    nullptr);

            if (received <= static_cast<ssize_t>(sizeof(PacketHeader)))
                continue;

            PacketHeader header;
            std::memcpy(&header, buffer.data(), sizeof(PacketHeader));

            if (header.fragment_count == 0 ||
                header.fragment_id >= header.fragment_count ||
                header.payload_size == 0 ||
                header.total_size == 0)
            {
                continue;
            }

            const size_t expected_size =
                sizeof(PacketHeader) + header.payload_size;

            if (expected_size != static_cast<size_t>(received))
                continue;

            const uint8_t * payload =
                buffer.data() + sizeof(PacketHeader);

            auto & rb = buffers_[header.msg_id];

            if (rb.data.empty())
            {
                rb.total_size = header.total_size;
                rb.fragment_count = header.fragment_count;
                rb.received_count = 0;

                rb.data.resize(header.total_size);
                rb.received.resize(header.fragment_count, false);
            }

            if (rb.total_size != header.total_size ||
                rb.fragment_count != header.fragment_count)
            {
                buffers_.erase(header.msg_id);
                continue;
            }

            const size_t offset =
                static_cast<size_t>(header.fragment_id) *
                MAX_UDP_FRAGMENT_PAYLOAD;

            if (offset + header.payload_size > rb.data.size())
            {
                buffers_.erase(header.msg_id);
                continue;
            }

            if (!rb.received[header.fragment_id])
            {
                std::memcpy(
                    rb.data.data() + offset,
                    payload,
                    header.payload_size);

                rb.received[header.fragment_id] = true;
                rb.received_count++;
            }

            if (rb.received_count != rb.fragment_count)
                continue;

            publishCloud(rb.data.data(), rb.total_size);

            buffers_.erase(header.msg_id);
        }
    }

    void publishCloud(const uint8_t * payload, uint32_t payload_size)
    {
        rclcpp::SerializedMessage serialized_msg(payload_size);

        auto & rcl_serialized =
            serialized_msg.get_rcl_serialized_message();

        std::memcpy(
            rcl_serialized.buffer,
            payload,
            payload_size);

        rcl_serialized.buffer_length = payload_size;

        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
        sensor_msgs::msg::PointCloud2 msg;

        serializer.deserialize_message(&serialized_msg, &msg);

        pub_->publish(msg);

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "[MOLA CLOUD RX] republishing %s | width=%u point_step=%u",
            output_cloud_topic_.c_str(),
            msg.width,
            msg.point_step);
    }

private:
    std::string output_cloud_topic_;
    int udp_port_{16002};

    int sock_{-1};
    std::atomic<bool> running_{false};
    std::thread worker_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    std::map<uint32_t, ReassemblyBuffer> buffers_;

    static constexpr size_t MAX_UDP_FRAGMENT_PAYLOAD = 1200;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MolaCloudUdpReceiver>());
    rclcpp::shutdown();
    return 0;
}