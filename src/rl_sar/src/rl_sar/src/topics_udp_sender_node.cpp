/*
 * topics_udp_sender_node.cpp
 *
 * PURPOSE
 * -------
 * This node runs on the robot high-level PC with the Unitree DDS configuration.
 *
 * It subscribes to robot sensor topics that are only visible through the
 * Unitree / eth0 DDS environment, for example:
 *
 *   - /dog_odom
 *   - /dog_imu_raw
 *
 * Then it serializes each ROS 2 message and sends it through UDP to another
 * local process: topics_udp_receiver_node.
 *
 * ARCHITECTURE
 * ------------
 *
 *   Unitree DDS / eth0
 *        |
 *        | /dog_odom, /dog_imu_raw
 *        v
 *   topics_udp_sender_node
 *        |
 *        | UDP localhost:16001
 *        v
 *   topics_udp_receiver_node
 *        |
 *        | ROS 2 DDS / wlan0
 *        v
 *   PC Nav2
 *
 * WHY TWO PROCESSES?
 * ------------------
 * The Unitree topics are visible only with the original CycloneDDS file:
 *
 *   /home/unitree/cyclonedds_ws/cyclonedds.xml
 *
 * which forces eth0.
 *
 * But the external PC communicates with the robot through Wi-Fi, wlan0.
 *
 * A single ROS 2 process cannot safely use two different CycloneDDS
 * configurations at the same time. Therefore, we split the bridge into:
 *
 *   1. Sender  : DDS eth0  -> UDP
 *   2. Receiver: UDP       -> DDS wlan0
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cmath>
#include <limits>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <atomic>
#include <map>
#include <thread>
#include <utility>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>


/*
 * Each transported topic receives a numeric ID.
 *
 * This avoids sending long topic names in every UDP packet.
 * The receiver uses the same IDs to know how to deserialize the payload.
 */

enum TopicId : uint32_t
{
    DOG_ODOM = 1,
    DOG_IMU = 2,
    LIVOX_CLOUD = 3,  // reserved/debug only, disabled in production
    TF_DYNAMIC = 4,
    TF_STATIC = 5,
    LASER_SCAN = 6,
    JOINT_STATES = 7,
    DOG_ODOM_FILTERED = 8,
    ODOM_MAIN = 9,
    ODOM_FASTLIO = 10,
    MAP_TOPIC = 11,
    MOLA_ODOM = 12,
    
    GLOBAL_COSTMAP = 13,
    LOCAL_COSTMAP = 14,

    // Domain 10 -> Domain 0
    GOAL_POSE = 100,
    INITIAL_POSE = 101
};




/*
 * Header placed before every UDP fragment.
 *
 * A complete ROS 2 message may be split into several UDP packets.
 *
 * topic_id:
 *   identifies the original ROS 2 topic/message type.
 *
 * msg_id:
 *   unique ID of the complete serialized ROS 2 message.
 *
 * fragment_id:
 *   index of this UDP fragment.
 *
 * fragment_count:
 *   total number of fragments for the complete message.
 *
 * payload_size:
 *   size in bytes of this fragment payload.
 *
 * total_size:
 *   total size in bytes of the complete serialized ROS 2 message.
 */
struct PacketHeader
{
    uint32_t topic_id;
    uint32_t msg_id;
    uint32_t fragment_id;
    uint32_t fragment_count;
    uint32_t payload_size;
    uint32_t total_size;
};

struct ReassemblyBuffer
{
    uint32_t topic_id{0};
    uint32_t total_size{0};
    uint32_t fragment_count{0};
    uint32_t received_count{0};

    std::vector<uint8_t> data;
    std::vector<bool> received;
};

class TopicsUdpSender : public rclcpp::Node
{
public:
    TopicsUdpSender()
    : Node("topics_udp_sender_node")
    {
        /*
         * UDP destination.
         *
         * Default is localhost because both sender and receiver run on the
         * same robot high-level PC.
         */
        udp_host_ = declare_parameter<std::string>("udp_host", "127.0.0.1");
        udp_port_ = declare_parameter<int>("udp_port", 16001);

        scan_frame_id_ = declare_parameter<std::string>("scan_frame_id", "livox_frame");
        angle_min_ = declare_parameter<double>("angle_min", -3.14159);
        angle_max_ = declare_parameter<double>("angle_max", 3.14159);
        angle_increment_ = declare_parameter<double>("angle_increment", 0.0087);
        range_min_ = declare_parameter<double>("range_min", 0.20);
        range_max_ = declare_parameter<double>("range_max", 30.0);
        min_height_ = declare_parameter<double>("min_height", -0.20);
        max_height_ = declare_parameter<double>("max_height", 0.20);
        flip_y_ = declare_parameter<bool>("flip_y", true);
        cloud_decimation_ = declare_parameter<int>("cloud_decimation", 1);

        bridge_dog_odom_ = declare_parameter<bool>("bridge_dog_odom", false);
        bridge_odom_ = declare_parameter<bool>("bridge_odom", false);
        bridge_imu_ = declare_parameter<bool>("bridge_imu", false);
        bridge_scan_ = declare_parameter<bool>("bridge_scan", false);
        bridge_tf_ = declare_parameter<bool>("bridge_tf", false);
        bridge_joint_states_ = declare_parameter<bool>("bridge_joint_states", false);
        bridge_map_ = declare_parameter<bool>("bridge_map", false);
        bridge_global_costmap_ =
            declare_parameter<bool>(
                "bridge_global_costmap",
                true
            );

        bridge_local_costmap_ =
            declare_parameter<bool>(
                "bridge_local_costmap",
                true
            );

        /*
         * Create UDP socket.
         */
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);

        if (sock_ < 0)
        {
            throw std::runtime_error("Failed to create UDP socket");
        }

        command_udp_port_ =
            declare_parameter<int>(
                "command_udp_port",
                16003
        );

        /*
         * Configure UDP destination address.
         */
        target_addr_.sin_family = AF_INET;
        target_addr_.sin_port = htons(static_cast<uint16_t>(udp_port_));

        if (inet_pton(AF_INET, udp_host_.c_str(), &target_addr_.sin_addr) != 1)
        {
            throw std::runtime_error("Invalid UDP host address");
        }

        if (bridge_odom_)
        {
            /*
            * Subscribe to /dog_odom.
            *
            * QoS:
            *   best_effort is usually more compatible with high-frequency robot
            *   state topics and avoids blocking if packets are lost.
            */
            
            odom_main_sub_ =
                create_subscription<nav_msgs::msg::Odometry>(
                    "/odom",
                    rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
                    [this](const nav_msgs::msg::Odometry::SharedPtr msg)
                    {
                        send_ros_message<nav_msgs::msg::Odometry>(
                            *msg,
                            ODOM_MAIN
                        );
                    }
                );
            
            RCLCPP_INFO(
                get_logger(),
                "Subscribed to: /odom"
            );
        }

        if (bridge_dog_odom_)
        {
            /*
            * Subscribe to /dog_odom.
            *
            * QoS:
            *   best_effort is usually more compatible with high-frequency robot
            *   state topics and avoids blocking if packets are lost.
            */
            odom_sub_ =
                create_subscription<nav_msgs::msg::Odometry>(
                    "/dog_odom",
                    rclcpp::QoS(rclcpp::KeepLast(20)).best_effort(),
                    [this](const nav_msgs::msg::Odometry::SharedPtr msg)
                    {
                        send_ros_message<nav_msgs::msg::Odometry>(
                            *msg,
                            DOG_ODOM
                        );
                    }
                );
            
            RCLCPP_INFO(
                get_logger(),
                "Subscribed to: /dog_odom"
            );
        }

        if (bridge_imu_)
        {
            /*
            * Subscribe to /dog_imu_raw.
            *
            * IMU is usually higher frequency, so we use a larger queue.
            */
            imu_sub_ =
                create_subscription<sensor_msgs::msg::Imu>(
                    "/dog_imu_raw",
                    rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
                    [this](const sensor_msgs::msg::Imu::SharedPtr msg)
                    {
                        send_ros_message<sensor_msgs::msg::Imu>(
                            *msg,
                            DOG_IMU
                        );
                    }
                );
            RCLCPP_INFO(
                get_logger(),
                "Subscribed to: /dog_imu_raw"
            );
        }

        if (bridge_scan_)
        {
            /*
            * Subscribe to /scan_robot.
            *
            * This topic should be generated on the robot from
            * /utlidar/cloud_livox_mid360 using pointcloud_to_laserscan.
            * /utlidar/cloud_livox_mid360 → internal cloud-to-scan conversion → UDP LASER_SCAN
            *
            * Compared to the full PointCloud2, LaserScan is much lighter and is
            * therefore preferred for Wi-Fi transport and Nav2/AMCL.
            */
            
            scan_sub_ =
                create_subscription<sensor_msgs::msg::LaserScan>(
                    "/scan",
                    rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
                    [this](const sensor_msgs::msg::LaserScan::SharedPtr msg)
                    {
                        send_ros_message<sensor_msgs::msg::LaserScan>(
                            *msg,
                            LASER_SCAN
                        );
                    }
                );
            RCLCPP_INFO(
                get_logger(),
                "Subscribed to: /utlidar/cloud_livox_mid360 (internal LaserScan)"
            );
        }

        if (bridge_tf_)
        {
            
            tf_sub_reliable_ =
                create_subscription<tf2_msgs::msg::TFMessage>(
                    "/tf",
                    rclcpp::QoS(rclcpp::KeepLast(100)).reliable(),
                    [this](const tf2_msgs::msg::TFMessage::SharedPtr msg)
                    {
                        send_ros_message<tf2_msgs::msg::TFMessage>(
                            *msg,
                            TF_DYNAMIC
                        );
                    }
                );

            tf_static_sub_ =
                create_subscription<tf2_msgs::msg::TFMessage>(
                    "/tf_static",
                    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
                    [this](const tf2_msgs::msg::TFMessage::SharedPtr msg)
                    {
                        send_ros_message<tf2_msgs::msg::TFMessage>(
                            *msg,
                            TF_STATIC
                        );
                    }
                );
            RCLCPP_INFO(
                get_logger(),
                "Subscribed to: /tf, /tf_static"
            );
        }

        if (bridge_joint_states_)
        {
            joint_states_sub_ =
                create_subscription<sensor_msgs::msg::JointState>(
                    "/joint_states_source",
                    rclcpp::QoS(rclcpp::KeepLast(20)).best_effort(),
                    [this](const sensor_msgs::msg::JointState::SharedPtr msg)
                    {
                        send_ros_message<sensor_msgs::msg::JointState>(
                            *msg,
                            JOINT_STATES
                        );
                    }
                );
            RCLCPP_INFO(
                get_logger(),
                "Subscribed to: /joint_states_source"
            );
        }

        if (bridge_map_)
        {
            auto map_qos =
            rclcpp::QoS(rclcpp::KeepLast(1))
            .reliable()
            .transient_local();

            map_sub_ =
                create_subscription<nav_msgs::msg::OccupancyGrid>(
                    "/map_runtime",
                    map_qos,
                    [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
                    {
                        send_ros_message<
                            nav_msgs::msg::OccupancyGrid>(
                                *msg,
                                MAP_TOPIC);
                    });
        RCLCPP_INFO(
            get_logger(),
            "Subscribed to: /map_runtime"
        );
        }

        if (bridge_global_costmap_)
        {
            global_costmap_sub_ =
                create_subscription<nav_msgs::msg::OccupancyGrid>(
                    "/global_costmap/costmap",
                    rclcpp::QoS(rclcpp::KeepLast(1))
                        .reliable()
                        .transient_local(),
                    [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
                    {
                        send_ros_message<nav_msgs::msg::OccupancyGrid>(
                            *msg,
                            GLOBAL_COSTMAP
                        );
                    }
                );
        }

        if (bridge_local_costmap_)
        {
            local_costmap_sub_ =
                create_subscription<nav_msgs::msg::OccupancyGrid>(
                    "/local_costmap/costmap",
                    rclcpp::QoS(rclcpp::KeepLast(1))
                        .reliable()
                        .transient_local(),
                    [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
                    {
                        send_ros_message<nav_msgs::msg::OccupancyGrid>(
                            *msg,
                            LOCAL_COSTMAP
                        );
                    }
                );
        }

        initial_pose_pub_ =
            create_publisher<
                geometry_msgs::msg::
                    PoseWithCovarianceStamped>(
                        "/initialpose",
                        rclcpp::QoS(
                            rclcpp::KeepLast(10)
                        ).reliable()
        );
        navigate_to_pose_client_ =
            rclcpp_action::create_client<
                nav2_msgs::action::NavigateToPose>(
                    this,
                    "/navigate_to_pose"
                );

        command_sock_ =
            socket(
                AF_INET,
                SOCK_DGRAM,
                0
            );

        if (command_sock_ < 0)
        {
            throw std::runtime_error(
                "Failed to create command UDP receive socket"
            );
        }


        sockaddr_in command_addr{};

        command_addr.sin_family =
            AF_INET;

        command_addr.sin_addr.s_addr =
            htonl(
                INADDR_LOOPBACK
            );

        command_addr.sin_port =
            htons(
                static_cast<uint16_t>(
                    command_udp_port_
                )
            );


        if (
            bind(
                command_sock_,
                reinterpret_cast<sockaddr *>(
                    &command_addr
                ),
                sizeof(command_addr)
            )
            < 0
        )
        {
            throw std::runtime_error(
                "Failed to bind command UDP socket"
            );
        }

        command_running_.store(
            true
        );

        command_thread_ =
            std::thread(
                &TopicsUdpSender::command_receive_loop,
                this
            );
        /*
        * Full PointCloud2 bridge disabled in normal navigation mode.
        *
        * The complete /utlidar/cloud_livox_mid360 topic is very heavy over Wi-Fi.
        * For Nav2/AMCL, we prefer sending /scan_robot instead.
        *
        * Re-enable this block only for debugging or RViz cloud visualization.
        */
        auto qos = rclcpp::SensorDataQoS();
        qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

        RCLCPP_INFO(get_logger(), "topics_udp_sender_node started");
        RCLCPP_INFO(get_logger(), "UDP target: %s:%d", udp_host_.c_str(), udp_port_);
    }

    ~TopicsUdpSender() override
    {
        /*
         * Close UDP socket when the node stops.
         */
        if (sock_ >= 0)
        {
            close(sock_);
            sock_ = -1;
        }

        command_running_.store(
            false
        );

        if (command_sock_ >= 0)
        {
            shutdown(
                command_sock_,
                SHUT_RDWR
            );

            close(
                command_sock_
            );

            command_sock_ =
                -1;
        }

        if (
            command_thread_.joinable()
        )
        {
            command_thread_.join();
        }
    }

private:

    void handle_command_message(
        const ReassemblyBuffer & rb)
    {
        switch (rb.topic_id)
        {
            case GOAL_POSE:
            {
                geometry_msgs::msg::PoseStamped goal_pose;

                deserialize_message(
                    rb.data.data(),
                    rb.total_size,
                    goal_pose
                );

                send_navigation_goal(
                    goal_pose
                );

                break;
            }


            case INITIAL_POSE:
            {
                geometry_msgs::msg::
                    PoseWithCovarianceStamped initial_pose;

                deserialize_message(
                    rb.data.data(),
                    rb.total_size,
                    initial_pose
                );

                initial_pose_pub_->publish(
                    initial_pose
                );

                RCLCPP_INFO(
                    get_logger(),
                    "Initial pose received from RViz bridge"
                );

                break;
            }


            default:
                break;
        }
    }

    void command_receive_loop()
    {
        std::vector<uint8_t> buffer(65535);

        while (
            command_running_.load()
            &&
            rclcpp::ok()
        )
        {
            const ssize_t received =
                recvfrom(
                    command_sock_,
                    buffer.data(),
                    buffer.size(),
                    0,
                    nullptr,
                    nullptr
                );

            if (received <= 0)
            {
                if (!command_running_.load())
                {
                    break;
                }

                continue;
            }

            if (
                received
                <=
                static_cast<ssize_t>(
                    sizeof(PacketHeader)
                )
            )
            {
                continue;
            }


            PacketHeader header{};

            std::memcpy(
                &header,
                buffer.data(),
                sizeof(PacketHeader)
            );


            if (
                header.fragment_count == 0
                ||
                header.fragment_id >= header.fragment_count
                ||
                header.payload_size == 0
                ||
                header.total_size == 0
            )
            {
                continue;
            }


            const size_t expected_size =
                sizeof(PacketHeader)
                +
                header.payload_size;

            if (
                expected_size
                !=
                static_cast<size_t>(
                    received
                )
            )
            {
                continue;
            }


            const auto key =
                std::make_pair(
                    header.topic_id,
                    header.msg_id
                );


            auto & rb =
                command_reassembly_buffers_[key];


            if (rb.data.empty())
            {
                rb.topic_id =
                    header.topic_id;

                rb.total_size =
                    header.total_size;

                rb.fragment_count =
                    header.fragment_count;

                rb.received_count =
                    0;

                rb.data.resize(
                    header.total_size
                );

                rb.received.assign(
                    header.fragment_count,
                    false
                );
            }


            if (
                rb.total_size
                    !=
                    header.total_size
                ||
                rb.fragment_count
                    !=
                    header.fragment_count
            )
            {
                command_reassembly_buffers_
                    .erase(key);

                continue;
            }


            const size_t offset =
                static_cast<size_t>(
                    header.fragment_id
                )
                *
                MAX_UDP_FRAGMENT_PAYLOAD;


            if (
                offset
                +
                header.payload_size
                >
                rb.data.size()
            )
            {
                command_reassembly_buffers_
                    .erase(key);

                continue;
            }


            if (
                !rb.received[
                    header.fragment_id
                ]
            )
            {
                std::memcpy(
                    rb.data.data()
                        +
                        offset,
                    buffer.data()
                        +
                        sizeof(PacketHeader),
                    header.payload_size
                );

                rb.received[
                    header.fragment_id
                ] = true;

                rb.received_count++;
            }


            if (
                rb.received_count
                !=
                rb.fragment_count
            )
            {
                continue;
            }


            handle_command_message(
                rb
            );


            command_reassembly_buffers_
                .erase(key);
        }
    }

    template<typename MsgT>
    void deserialize_message(
        const uint8_t * payload,
        const uint32_t payload_size,
        MsgT & msg)
    {
        rclcpp::SerializedMessage
            serialized_msg(
                payload_size
            );

        auto & rcl_serialized =
            serialized_msg
                .get_rcl_serialized_message();


        std::memcpy(
            rcl_serialized.buffer,
            payload,
            payload_size
        );

        rcl_serialized.buffer_length =
            payload_size;


        rclcpp::Serialization<MsgT>
            serializer;


        serializer.deserialize_message(
            &serialized_msg,
            &msg
        );
    }
    /*
     * Serialize one ROS 2 message and send it over UDP.
     *
     * Template parameter MsgT:
     *   allows the same function to handle Odometry, Imu, PointCloud2,
     *   TFMessage, etc.
     *
     * IMPORTANT
     * ---------
     * Some ROS 2 messages, especially PointCloud2, are too large to fit
     * into a single UDP datagram.
     *
     * Therefore, this function does not send:
     *
     *   [ PacketHeader ][ full serialized ROS 2 payload ]
     *
     * in one UDP packet anymore.
     *
     * Instead, it sends:
     *
     *   [ PacketHeader ][ fragment 0 ]
     *   [ PacketHeader ][ fragment 1 ]
     *   [ PacketHeader ][ fragment 2 ]
     *   ...
     *
     * The receiver uses:
     *
     *   - topic_id
     *   - msg_id
     *   - fragment_id
     *   - fragment_count
     *
     * to reconstruct the original serialized ROS 2 message.
     */
    template<typename MsgT>
    void send_ros_message(
        const MsgT & msg,
        uint32_t topic_id)
    {
        /*
         * ROS 2 serializer for the selected message type.
         */
        rclcpp::Serialization<MsgT> serializer;

        /*
         * Buffer that will contain the complete serialized ROS 2 message.
         */
        rclcpp::SerializedMessage serialized_msg;

        /*
         * Convert the typed ROS 2 message into binary CDR representation.
         */
        serializer.serialize_message(&msg, &serialized_msg);

        /*
         * Pointer to the serialized binary payload.
         */
        const uint8_t * serialized_data =
            serialized_msg.get_rcl_serialized_message().buffer;

        /*
         * Total size of the complete serialized ROS 2 message.
         */
        const size_t total_size =
            serialized_msg.size();

        /*
         * Generate a unique ID for this complete ROS 2 message.
         *
         * All fragments produced from this message will carry the same msg_id.
         */
        const uint32_t msg_id =
            ++msg_counter_;

        /*
         * Compute how many UDP fragments are needed.
         *
         * Example:
         *   total_size = 3000 bytes
         *   MAX_UDP_FRAGMENT_PAYLOAD = 1200 bytes
         *
         *   fragment_count = 3
         */
        const uint32_t fragment_count =
            static_cast<uint32_t>(
                (total_size + MAX_UDP_FRAGMENT_PAYLOAD - 1)
                / MAX_UDP_FRAGMENT_PAYLOAD
            );

        /*
         * Send each fragment independently.
         */
        for (uint32_t fragment_id = 0;
             fragment_id < fragment_count;
             ++fragment_id)
        {
            /*
             * Offset of this fragment inside the complete serialized payload.
             */
            const size_t offset =
                static_cast<size_t>(fragment_id)
                * MAX_UDP_FRAGMENT_PAYLOAD;

            /*
             * Number of bytes still available from this offset.
             */
            const size_t remaining =
                total_size - offset;

            /*
             * Payload size for this fragment.
             *
             * All fragments except the last usually have size
             * MAX_UDP_FRAGMENT_PAYLOAD.
             */
            const size_t fragment_payload_size =
                std::min(
                    MAX_UDP_FRAGMENT_PAYLOAD,
                    remaining
                );

            /*
             * Build packet header for this fragment.
             *
             * Header fields:
             *
             *   topic_id:
             *     identifies the ROS 2 topic/message type.
             *
             *   msg_id:
             *     identifies the original complete ROS 2 message.
             *
             *   fragment_id:
             *     index of this fragment.
             *
             *   fragment_count:
             *     total number of fragments for this message.
             *
             *   payload_size:
             *     number of bytes carried by this UDP fragment.
             *
             *   total_size:
             *     size of the complete serialized ROS 2 message.
             */
            PacketHeader header;
            header.topic_id = topic_id;
            header.msg_id = msg_id;
            header.fragment_id = fragment_id;
            header.fragment_count = fragment_count;
            header.payload_size =
                static_cast<uint32_t>(fragment_payload_size);
            header.total_size =
                static_cast<uint32_t>(total_size);

            /*
             * Build UDP packet:
             *
             *   [ PacketHeader ][ serialized payload fragment ]
             */
            std::vector<uint8_t> packet(
                sizeof(PacketHeader) + fragment_payload_size
            );

            /*
             * Copy header into packet.
             */
            std::memcpy(
                packet.data(),
                &header,
                sizeof(PacketHeader)
            );

            /*
             * Copy only the current fragment of the serialized ROS 2 payload.
             */
            std::memcpy(
                packet.data() + sizeof(PacketHeader),
                serialized_data + offset,
                fragment_payload_size
            );

            /*
             * Send this fragment over UDP.
             */
            const ssize_t sent =
                sendto(
                    sock_,
                    packet.data(),
                    packet.size(),
                    0,
                    reinterpret_cast<sockaddr*>(&target_addr_),
                    sizeof(target_addr_)
                );

            /*
             * UDP is connectionless.
             *
             * If one fragment is lost, the receiver will not be able to
             * reconstruct that particular ROS 2 message, but future messages
             * can still be received normally.
             */
            if (sent < 0)
            {
                RCLCPP_ERROR_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    1000,
                    "UDP send failed for topic_id=%u msg_id=%u fragment=%u/%u: %s",
                    topic_id,
                    msg_id,
                    fragment_id + 1,
                    fragment_count,
                    std::strerror(errno)
                );
            }
        }
    }

    void send_navigation_goal(
        const geometry_msgs::msg::PoseStamped & pose)
    {
        if (
            !navigate_to_pose_client_
                ->wait_for_action_server(
                    std::chrono::milliseconds(
                        200
                    )
                )
        )
        {
            RCLCPP_ERROR(
                get_logger(),
                "NavigateToPose action server unavailable"
            );

            return;
        }


        nav2_msgs::action::NavigateToPose::Goal
            goal;


        goal.pose =
            pose;


        RCLCPP_INFO(
            get_logger(),
            "Sending navigation goal from RViz | "
            "x=%.3f y=%.3f",
            pose.pose.position.x,
            pose.pose.position.y
        );


        auto options =
            rclcpp_action::
                Client<
                    nav2_msgs::action::
                        NavigateToPose
                >::SendGoalOptions();


        options.goal_response_callback =
            [this](auto future)
            {
                auto handle =
                    future.get();

                if (!handle)
                {
                    RCLCPP_ERROR(
                        get_logger(),
                        "Navigation goal rejected"
                    );
                }
                else
                {
                    RCLCPP_INFO(
                        get_logger(),
                        "Navigation goal accepted"
                    );
                }
            };


        options.result_callback =
            [this](const auto & result)
            {
                RCLCPP_INFO(
                    get_logger(),
                    "Navigation goal finished | code=%d",
                    static_cast<int>(
                        result.code
                    )
                );
            };


        navigate_to_pose_client_
            ->async_send_goal(
                goal,
                options
            );
    }

    void CloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        /*
        * Convert one Livox PointCloud2 message into a lightweight LaserScan
        * and send it over UDP.
        *
        * Why this conversion is done inside the sender
        * ---------------------------------------------
        * The complete /utlidar/cloud_livox_mid360 message is too heavy for
        * continuous Wi-Fi transport. A single cloud can contain around 20k points
        * and hundreds of kilobytes of binary data.
        *
        * Nav2 and AMCL do not need the full 3D cloud for 2D navigation. They only
        * need a planar scan. Therefore, this node converts the cloud locally on
        * the robot, before UDP transport:
        *
        *   /utlidar/cloud_livox_mid360
        *        -> height filtering
        *        -> optional Y-axis flip
        *        -> polar projection
        *        -> LaserScan
        *        -> UDP LASER_SCAN
        *
        * Importance of the Y-axis flip
        * -----------------------------
        * On the tested G1/Livox setup, the raw cloud showed a lateral mirror
        * effect: forward/backward was correct, but left/right was inverted.
        *
        * If this is not corrected before scan generation, obstacles located on
        * the robot's left side may appear on the right side in Nav2, and vice
        * versa. This can produce wrong local costmaps, unsafe avoidance behavior,
        * and incorrect AMCL scan matching.
        *
        * The flip_y parameter applies:
        *
        *   x' = x
        *   y' = -y
        *   z' = z
        *
        * before computing angle = atan2(y, x).
        *
        * Timestamp policy
        * ----------------
        * The generated LaserScan is restamped with the sender node clock. This
        * replaces the previous pc2_restamper + scan_restamper chain and helps
        * avoid TF message filter errors caused by LiDAR hardware timestamps.
        *
        * Limitation
        * ----------
        * This is a lightweight internal converter, not a full replacement for the
        * official pointcloud_to_laserscan package. It does not perform TF-based
        * cloud transformation. It assumes that the cloud is already expressed in
        * the desired LiDAR frame and that the PC side has the required static TF
        * between base_frame and scan_frame_id.
        */
        if (!msg)
            return;

        if (cloud_decimation_ > 1)
        {
            cloud_counter_++;

            if ((cloud_counter_ % cloud_decimation_) != 0)
                return;
        }

        const auto find_offset =
            [](const sensor_msgs::msg::PointCloud2 & cloud,
            const std::string & name) -> int
            {
                for (const auto & field : cloud.fields)
                {
                    if (field.name == name)
                        return static_cast<int>(field.offset);
                }

                return -1;
            };

        const int x_offset = find_offset(*msg, "x");
        const int y_offset = find_offset(*msg, "y");
        const int z_offset = find_offset(*msg, "z");

        if (x_offset < 0 || y_offset < 0 || z_offset < 0)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "PointCloud2 does not contain x/y/z fields"
            );
            return;
        }

        const int beam_count =
            static_cast<int>(
                std::ceil((angle_max_ - angle_min_) / angle_increment_)
            );

        if (beam_count <= 0)
            return;

        sensor_msgs::msg::LaserScan scan;

        scan.header = msg->header;

        /*
        * Restamp the scan with current ROS time.
        * This replaces the old pc2_restamper + scan_restamper behavior.
        */
        scan.header.stamp = get_clock()->now();

        /*
        * Keep the scan in the LiDAR frame.
        * The PC/Nav2 side must have a valid TF between base_frame and livox_frame.
        */
        scan.header.frame_id = scan_frame_id_;

        scan.angle_min = angle_min_;
        scan.angle_max = angle_max_;
        scan.angle_increment = angle_increment_;
        scan.time_increment = 0.0;
        scan.scan_time = 0.1;
        scan.range_min = range_min_;
        scan.range_max = range_max_;

        scan.ranges.assign(
            beam_count,
            std::numeric_limits<float>::infinity()
        );

        scan.intensities.assign(
            beam_count,
            0.0f
        );

        const size_t point_step = msg->point_step;
        const size_t point_count = msg->data.size() / point_step;

        for (size_t i = 0; i < point_count; ++i)
        {
            const size_t base = i * point_step;

            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            std::memcpy(&x, &msg->data[base + x_offset], sizeof(float));
            std::memcpy(&y, &msg->data[base + y_offset], sizeof(float));
            std::memcpy(&z, &msg->data[base + z_offset], sizeof(float));

            if (!std::isfinite(x) ||
                !std::isfinite(y) ||
                !std::isfinite(z))
            {
                continue;
            }

            if (flip_y_)
                y = -y;

            if (z < min_height_ || z > max_height_)
                continue;

            const double range =
                std::sqrt(
                    static_cast<double>(x) * static_cast<double>(x) +
                    static_cast<double>(y) * static_cast<double>(y)
                );

            if (range < range_min_ || range > range_max_)
                continue;

            const double angle =
                std::atan2(
                    static_cast<double>(y),
                    static_cast<double>(x)
                );

            if (angle < angle_min_ || angle > angle_max_)
                continue;

            const int index =
                static_cast<int>(
                    (angle - angle_min_) / angle_increment_
                );

            if (index < 0 || index >= beam_count)
                continue;

            if (range < scan.ranges[index])
            {
                scan.ranges[index] =
                    static_cast<float>(range);
            }
        }
        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "[SCAN TX] cloud received, beam_count=%d, ranges=%zu",
            beam_count,
            scan.ranges.size()
        );
        send_ros_message<sensor_msgs::msg::LaserScan>(
            scan,
            LASER_SCAN
        );
    }

private:
    /*
     * UDP configuration.
     */
    std::string udp_host_;
    int udp_port_{16001};

    /*
     * UDP socket.
     */
    int sock_{-1};
    sockaddr_in target_addr_{};

    /*
     * ROS 2 subscriptions.
     */
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    //rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_reliable_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_best_effort_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr dog_odom_filtered_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_main_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_fastlio_sub_;

    std::string scan_frame_id_{"livox_frame"};

    double angle_min_{-3.14159};
    double angle_max_{3.14159};
    double angle_increment_{0.0087};
    double range_min_{0.20};
    double range_max_{30.0};
    double min_height_{-0.20};
    double max_height_{0.20};

    bool flip_y_{true};

    int cloud_decimation_{1};
    int cloud_counter_{0};

    /* 
    Bridge management
    */
    bool bridge_odom_{false};
    bool bridge_dog_odom_{false};
    bool bridge_imu_{false};
    bool bridge_scan_{false};
    bool bridge_tf_{false};
    bool bridge_joint_states_{false};
    bool bridge_map_{false};

    int command_udp_port_{
        16003
    };

    int command_sock_{
        -1
    };

    std::atomic<bool>
        command_running_{
            false
        };

    std::thread
        command_thread_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
        global_costmap_sub_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
        local_costmap_sub_;

    bool bridge_global_costmap_{true};
    bool bridge_local_costmap_{true};

    std::map<
        std::pair<uint32_t, uint32_t>,
        ReassemblyBuffer
    > command_reassembly_buffers_;


    rclcpp::Publisher<
        geometry_msgs::msg::
            PoseWithCovarianceStamped
    >::SharedPtr initial_pose_pub_;


    rclcpp_action::Client<
        nav2_msgs::action::
            NavigateToPose
    >::SharedPtr navigate_to_pose_client_;

    /*
        * Incremental message counter.
        *
        * Each serialized ROS 2 message receives a unique ID.
        * This allows the receiver to group UDP fragments that belong
        * to the same original ROS 2 message.
    */
    uint32_t msg_counter_{0};

    /*
        * Maximum payload size per UDP fragment.
        *
        * We keep this small enough to avoid IP fragmentation, especially over Wi-Fi.
        * This is important for large messages such as PointCloud2.
    */
    static constexpr size_t MAX_UDP_FRAGMENT_PAYLOAD = 1200;
};


int main(int argc, char ** argv)
{
    try
    {
        /*
         * Initialize ROS 2.
         */
        rclcpp::init(argc, argv);

        /*
         * Create node.
         */
        auto node =
            std::make_shared<TopicsUdpSender>();

        /*
         * Spin ROS 2 executor.
         */
        rclcpp::spin(node);

        /*
         * Shutdown ROS 2 cleanly.
         */
        rclcpp::shutdown();

        return 0;
    }
    catch (const std::exception & e)
    {
        /*
         * Catch standard C++ exceptions.
         */
        std::cerr
            << "[topics_udp_sender_node] Fatal exception: "
            << e.what()
            << std::endl;

        /*
         * Shutdown ROS 2 if still active.
         */
        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
    catch (...)
    {
        /*
         * Catch unknown exceptions.
         */
        std::cerr
            << "[topics_udp_sender_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}