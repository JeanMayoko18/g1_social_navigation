/*
 * topics_udp_receiver_node.cpp
 *
 * PURPOSE
 * -------
 * This node runs on the robot high-level PC with the Wi-Fi ROS 2 DDS
 * configuration.
 *
 * It receives UDP packets from topics_udp_sender_node, deserializes the
 * contained ROS 2 messages, and republishes them on Wi-Fi-visible topics:
 *
 *   /repub/dog_odom
 *   /repub/dog_imu_raw
 *
 * ARCHITECTURE
 * ------------
 *
 *   topics_udp_sender_node
 *        |
 *        | UDP localhost:16001
 *        v
 *   topics_udp_receiver_node
 *        |
 *        | ROS 2 DDS / wlan0
 *        v
 *   External PC / Nav2
 *
 * IMPORTANT
 * ---------
 * This node must be launched with the Wi-Fi CycloneDDS configuration:
 *
 *   CYCLONEDDS_URI=file:///home/unitree/cyclonedds_ws/cyclonedds_wifi.xml
 */

 /*
 * LIGHTWEIGHT NAVIGATION ARCHITECTURE
 * ----------------------------------
 *
 * Previous architecture:
 *
 *   Livox PointCloud2
 *          ↓
 *       UDP bridge
 *          ↓
 *      External PC
 *          ↓
 * pointcloud_to_laserscan
 *          ↓
 *        Nav2
 *
 * This architecture generated:
 *
 *   - very high Wi-Fi bandwidth usage,
 *   - UDP fragmentation,
 *   - packet losses,
 *   - unstable navigation latency.
 *
 * The new architecture performs the conversion directly
 * on the robot:
 *
 *   Livox PointCloud2
 *          ↓
 *   onboard LaserScan conversion
 *          ↓
 *        UDP bridge
 *          ↓
 *      /repub/scan
 *          ↓
 *         Nav2
 *
 * Advantages:
 *
 *   - dramatically lower bandwidth,
 *   - much more stable Wi-Fi DDS,
 *   - lower CPU usage on external PC,
 *   - lower latency,
 *   - simpler Nav2 integration.
 *
 * This is the recommended production architecture.
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <iostream>
/*
 * LaserScan message.
 *
 * We transport LaserScan instead of the original PointCloud2
 * because:
 *
 *   - LaserScan is MUCH lighter over Wi-Fi
 *   - Nav2 and AMCL are fundamentally 2D
 *   - UDP transport becomes far more stable
 *   - DDS traffic is drastically reduced
 *
 * The robot-side pipeline already converts:
 *
 *   /utlidar/cloud_livox_mid360
 *          ↓
 *   pc2_restamper_node
 *          ↓
 *   cloud_y_flip_node
 *          ↓
 *   pointcloud_to_laserscan_node
 *          ↓
 *        /scan_robot
 *
 * Therefore the receiver only republishes the final scan.
 */
#include <sensor_msgs/msg/laser_scan.hpp>

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

struct PacketHeader
{
    uint32_t topic_id;
    uint32_t msg_id;
    uint32_t fragment_id;
    uint32_t fragment_count;
    uint32_t payload_size;
    uint32_t total_size;
};

/*
 * Temporary buffer used to reconstruct one complete ROS 2 message
 * from several UDP fragments.
 */
struct ReassemblyBuffer
{
    /*
     * Topic/message type transported by this group of fragments.
     */
    uint32_t topic_id{0};

    /*
     * Total size of the complete serialized ROS 2 message.
     */
    uint32_t total_size{0};

    /*
     * Number of fragments expected for this message.
     */
    uint32_t fragment_count{0};

    /*
     * Number of fragments already received.
     */
    uint32_t received_count{0};

    /*
     * Buffer containing the reconstructed serialized ROS 2 message.
     */
    std::vector<uint8_t> data;

    /*
     * Boolean flags used to avoid counting duplicate fragments twice.
     */
    std::vector<bool> received;
};


class TopicsUdpReceiver : public rclcpp::Node
{
public:
    using NavigateToPose =
        nav2_msgs::action::NavigateToPose;

    using GoalHandleNavigateToPose =
        rclcpp_action::ServerGoalHandle<NavigateToPose>;

    TopicsUdpReceiver()
    : Node("topics_udp_receiver_node")
    {
        /*
         * UDP port on which this node listens.
         *
         * It must match the sender udp_port.
         */
        udp_port_ = declare_parameter<int>("udp_port", 16001);

        command_udp_host_ =
            declare_parameter<std::string>(
                "command_udp_host",
                "127.0.0.1"
            );

        command_udp_port_ =
            declare_parameter<int>(
                "command_udp_port",
                16003
            );

        bridge_goal_ =
            declare_parameter<bool>(
                "bridge_goal",
                true
            );

        bridge_initialpose_ =
            declare_parameter<bool>(
                "bridge_initialpose",
                true
            );
        
        bridge_odom_ = declare_parameter<bool>("bridge_odom", false);
        bridge_imu_ = declare_parameter<bool>("bridge_imu", false);
        bridge_scan_ = declare_parameter<bool>("bridge_scan", false);
        bridge_tf_ = declare_parameter<bool>("bridge_tf", false);
        bridge_joint_states_ = declare_parameter<bool>("bridge_joint_states", false);
        bridge_map_ = declare_parameter<bool>("bridge_map", false);
        bridge_mola_odom_ = declare_parameter<bool>("bridge_mola_odom", false);
        bridge_dog_odom_ = declare_parameter<bool>("bridge_dog_odom", false);
        /*
        * LaserScan publisher.
        *
        * This is the lightweight 2D LiDAR representation generated
        * on the robot side from the original Livox PointCloud2.
        *
        * IMPORTANT
        * ---------
        * We intentionally DO NOT republish the complete
        * /utlidar/cloud_livox_mid360 point cloud during normal
        * navigation because:
        *
        *   - the cloud is extremely heavy,
        *   - Wi-Fi bandwidth becomes saturated,
        *   - Nav2 only needs 2D obstacle information,
        *   - packet fragmentation increases UDP losses.
        *
        * Instead, the sender node performs:
        *
        *   PointCloud2 -> LaserScan
        *
        * and only the compact scan is transported over Wi-Fi.
        *
        * Final topic visible from the external PC:
        *
        *   /repub/scan
        *
        * This topic is intended for:
        *
        *   - AMCL
        *   - Nav2 costmaps
        *   - SLAM Toolbox
        *   - RViz 2D visualization
        */

        command_sock_ =
            socket(
                AF_INET,
                SOCK_DGRAM,
                0
            );

        if (command_sock_ < 0)
        {
            throw std::runtime_error(
                "Failed to create command UDP socket"
            );
        }

        command_target_addr_.sin_family =
            AF_INET;

        command_target_addr_.sin_port =
            htons(
                static_cast<uint16_t>(
                    command_udp_port_
                )
            );

        if (
            inet_pton(
                AF_INET,
                command_udp_host_.c_str(),
                &command_target_addr_.sin_addr
            )
            != 1
        )
        {
            throw std::runtime_error(
                "Invalid command UDP host"
            );
        }

        if (bridge_goal_)
        {
            navigate_to_pose_server_ =
                rclcpp_action::create_server<
                    NavigateToPose>(
                        this,
                        "/navigate_to_pose",

                        [this](
                            const rclcpp_action::GoalUUID &,
                            std::shared_ptr<
                                const NavigateToPose::Goal> goal)
                        {
                            RCLCPP_INFO(
                                get_logger(),
                                "RViz NavigateToPose goal received | "
                                "x=%.3f y=%.3f",
                                goal->pose.pose.position.x,
                                goal->pose.pose.position.y
                            );

                            return
                                rclcpp_action::
                                    GoalResponse::
                                        ACCEPT_AND_EXECUTE;
                        },

                        [this](
                            const std::shared_ptr<
                                GoalHandleNavigateToPose>)
                        {
                            RCLCPP_INFO(
                                get_logger(),
                                "RViz requested goal cancellation"
                            );

                            return
                                rclcpp_action::
                                    CancelResponse::
                                        ACCEPT;
                        },

                        [this](
                            const std::shared_ptr<
                                GoalHandleNavigateToPose>
                                    goal_handle)
                        {
                            const auto goal =
                                goal_handle->get_goal();

                            send_command_message<
                                geometry_msgs::msg::PoseStamped>(
                                    goal->pose,
                                    GOAL_POSE
                                );

                            RCLCPP_INFO(
                                get_logger(),
                                "Navigation goal forwarded through UDP"
                            );

                            /*
                            * For this first implementation we acknowledge
                            * successful forwarding immediately.
                            *
                            * Later we can bridge feedback/result from the
                            * real Nav2 action if needed.
                            */
                            auto result =
                                std::make_shared<
                                    NavigateToPose::Result>();

                            goal_handle->succeed(
                                result
                            );
                        }
                    );
        }

        if (bridge_dog_odom_)
        {
            /*
            * Publishers visible from the external PC through Wi-Fi DDS.
            */
            odom_pub_ =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/dog_odom",
                    rclcpp::QoS(rclcpp::KeepLast(20)).reliable()
                );
            
            RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /dog_odom\n"

            );
        }

        if (bridge_mola_odom_)
        {
            /*
            * Publishers visible from the external PC through Wi-Fi DDS.
            */
            mola_odom_pub_ =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/mola_odom_2d",
                    rclcpp::QoS(rclcpp::KeepLast(20)).reliable()
                );
            
            RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /mola_odom_2d\n"

            );
        }

        if (bridge_odom_)
        {
            /*
            * Publishers visible from the external PC through Wi-Fi DDS.
            */
            odom_pub_ =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/dog_odom",
                    rclcpp::QoS(rclcpp::KeepLast(20)).reliable()
                );
            mola_odom_pub_ =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/mola_odom_2d",
                    rclcpp::QoS(rclcpp::KeepLast(20)).reliable()
                );
            odom_pub_artificial =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/odom",
                    rclcpp::QoS(rclcpp::KeepLast(20)).best_effort()
                );
            odom_pub_filter_ =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/dog_odom_filtered",
                    rclcpp::QoS(rclcpp::KeepLast(20)).best_effort()
                );

            odom_pub_fastlio_ =
                create_publisher<nav_msgs::msg::Odometry>(
                    "/odom_fastlio",
                    rclcpp::QoS(rclcpp::KeepLast(20)).best_effort()
                );
            
            RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /dog_odom\n"
                "  - /mola_odom_2d\n"
                "  - /odom\n"
                "  - /odom_fastlio\n"
                "  - /dog_odom_filtered"

            );
        }

        if (bridge_imu_)
        {
            imu_pub_ =
                create_publisher<sensor_msgs::msg::Imu>(
                    "/dog_imu_raw",
                    rclcpp::QoS(rclcpp::KeepLast(50)).best_effort()
                );
            RCLCPP_INFO(
                    get_logger(),
                    "  - /dog_imu_raw"
                );
        }

        if (bridge_scan_)
        {
            scan_pub_ =
                create_publisher<sensor_msgs::msg::LaserScan>(
                    "/scan",
                    rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()
                );
            RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /scan\n"
            );
        }
        if (bridge_joint_states_)
        {
            joint_states_pub_ =
                create_publisher<sensor_msgs::msg::JointState>(
                    "/joint_states",
                    rclcpp::QoS(rclcpp::KeepLast(20)).reliable()
                );
            RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /joint_states"
            );
        }
        if (bridge_map_)
        {
            map_pub_ =
                create_publisher<nav_msgs::msg::OccupancyGrid>(
                    "/map",
                    rclcpp::QoS(rclcpp::KeepLast(1))
                        .reliable()
                        .transient_local());
            
            RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /map\n"
            );
        }
        if (bridge_global_costmap_)
        {
            global_costmap_pub_ =
                create_publisher<nav_msgs::msg::OccupancyGrid>(
                    "/global_costmap/costmap",
                    rclcpp::QoS(rclcpp::KeepLast(1))
                        .reliable()
                        .transient_local()
                );
        }

        if (bridge_local_costmap_)
        {
            local_costmap_pub_ =
                create_publisher<nav_msgs::msg::OccupancyGrid>(
                    "/local_costmap/costmap",
                    rclcpp::QoS(rclcpp::KeepLast(1))
                        .reliable()
                        .transient_local()
                );
        }
        if (bridge_tf_)
        {
            tf_pub_ =
                create_publisher<tf2_msgs::msg::TFMessage>(
                    "/tf",
                    rclcpp::QoS(rclcpp::KeepLast(100)).reliable()
                );

            tf_static_pub_ =
                create_publisher<tf2_msgs::msg::TFMessage>(
                    "/tf_static",
                    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local()
                );
            
                RCLCPP_INFO(
                get_logger(),
                "Publishing:\n"
                "  - /tf\n"
                "  - /tf_static\n"
            );
        }


        if (bridge_initialpose_)
        {
            initial_pose_sub_ =
                create_subscription<
                    geometry_msgs::msg::PoseWithCovarianceStamped>(
                        "/initialpose",
                        rclcpp::QoS(
                            rclcpp::KeepLast(10)
                        ).reliable(),
                        [this](
                            const geometry_msgs::msg::
                                PoseWithCovarianceStamped::SharedPtr msg)
                        {
                            send_command_message<
                                geometry_msgs::msg::
                                    PoseWithCovarianceStamped>(
                                        *msg,
                                        INITIAL_POSE
                                    );
                        }
                    );

            RCLCPP_INFO(
                get_logger(),
                "Command bridge: /initialpose -> UDP port %d",
                command_udp_port_
            );
        }

        /*
         * Create UDP socket.
         */
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);

        if (sock_ < 0)
        {
            throw std::runtime_error("Failed to create UDP socket");
        }

        /*
         * Bind socket to all local interfaces.
         *
         * Since sender and receiver usually communicate through localhost,
         * INADDR_ANY is enough.
         */
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(udp_port_));

        if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        {
            throw std::runtime_error("Failed to bind UDP socket");
        }

        /*
         * Start background UDP receive thread.
         *
         * We do not block the ROS 2 executor with recvfrom().
         */
        running_ = true;

        worker_ =
            std::thread(
                [this]()
                {
                    receive_loop();
                }
            );

        RCLCPP_INFO(get_logger(), "topics_udp_receiver_node started");
        RCLCPP_INFO(get_logger(), "Listening on UDP port: %d", udp_port_);
        //RCLCPP_INFO(get_logger(), "Publishing: /repub/dog_odom, /repub/dog_imu_raw, /repub/utlidar/cloud_livox_mid360, /repub/tf, /repub/tf_static");
    }

    ~TopicsUdpReceiver() override
    {
        /*
         * Stop receive loop.
         */
        running_ = false;

        /*
         * Closing the socket helps unblock recvfrom().
         */
        if (sock_ >= 0)
        {
            close(sock_);
            sock_ = -1;
        }

        /*
         * Wait for the worker thread to finish.
         */
        if (worker_.joinable())
        {
            worker_.join();
        }

        if (command_sock_ >= 0)
        {
            close(command_sock_);
            command_sock_ = -1;
        }
    }

private:
    /*
     * Main UDP receive loop.
     *
     * Packet format:
     *
     *   [ PacketHeader ][ serialized ROS 2 payload fragment ]
     *
     * IMPORTANT
     * ---------
     * A complete ROS 2 message may be larger than the maximum safe UDP
     * datagram size, especially for PointCloud2.
     *
     * Therefore, the sender may split one serialized ROS 2 message into
     * several UDP fragments:
     *
     *   [ PacketHeader ][ fragment 0 ]
     *   [ PacketHeader ][ fragment 1 ]
     *   [ PacketHeader ][ fragment 2 ]
     *   ...
     *
     * The header tells us:
     *
     *   - which topic/message type this packet represents
     *   - which original ROS 2 message this fragment belongs to
     *   - the index of the current fragment
     *   - the total number of fragments
     *   - how many bytes are carried by this fragment
     *   - the total size of the complete serialized ROS 2 message
     */
    
    void receive_loop()
    {
        /*
         * UDP receive buffer.
         *
         * 65535 bytes is the theoretical maximum UDP datagram size.
         * In practice, our sender uses much smaller fragments, around 1200 bytes,
         * to avoid IP fragmentation over Wi-Fi.
         */
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
                    nullptr
                );

            /*
             * If socket was closed during shutdown, recvfrom may return <= 0.
             */
            if (received <= 0)
            {
                continue;
            }

            /*
             * We need at least a complete PacketHeader.
             */
            if (received <= static_cast<ssize_t>(sizeof(PacketHeader)))
            {
                continue;
            }

            PacketHeader header;

            std::memcpy(
                &header,
                buffer.data(),
                sizeof(PacketHeader)
            );

            /*
             * Basic sanity checks.
             *
             * These checks protect the receiver from malformed packets,
             * corrupted headers, or incompatible sender/receiver versions.
             */
            if (header.fragment_count == 0)
            {
                continue;
            }

            if (header.fragment_id >= header.fragment_count)
            {
                continue;
            }

            if (header.payload_size == 0)
            {
                continue;
            }

            if (header.total_size == 0)
            {
                continue;
            }

            /*
             * Validate UDP datagram size.
             *
             * The actual received datagram must contain exactly:
             *
             *   sizeof(PacketHeader) + header.payload_size
             */
            const size_t expected_packet_size =
                sizeof(PacketHeader) + header.payload_size;

            if (expected_packet_size != static_cast<size_t>(received))
            {
                continue;
            }

            /*
             * Pointer to the payload fragment located after the header.
             */
            const uint8_t * payload =
                buffer.data() + sizeof(PacketHeader);

            /*
             * Reassembly key.
             *
             * The pair (topic_id, msg_id) uniquely identifies one complete
             * serialized ROS 2 message being reconstructed.
             */
            const auto key =
                std::make_pair(
                    header.topic_id,
                    header.msg_id
                );

            /*
             * Get or create the reassembly buffer for this message.
             */
            auto & rb =
                reassembly_buffers_[key];

            /*
             * First fragment received for this message:
             *
             * initialize the reassembly buffer.
             */
            if (rb.data.empty())
            {
                rb.topic_id = header.topic_id;
                rb.total_size = header.total_size;
                rb.fragment_count = header.fragment_count;
                rb.received_count = 0;

                rb.data.resize(header.total_size);
                rb.received.resize(header.fragment_count, false);
            }

            /*
             * Validate that all fragments for the same message agree on
             * total size and fragment count.
             */
            if (rb.total_size != header.total_size ||
                rb.fragment_count != header.fragment_count)
            {
                reassembly_buffers_.erase(key);
                continue;
            }

            /*
             * Compute the offset of this fragment inside the complete
             * serialized ROS 2 payload.
             *
             * This must match the sender-side MAX_UDP_FRAGMENT_PAYLOAD.
             */
            const size_t offset =
                static_cast<size_t>(header.fragment_id) *
                MAX_UDP_FRAGMENT_PAYLOAD;

            /*
             * Validate that the fragment fits inside the destination buffer.
             */
            if (offset + header.payload_size > rb.data.size())
            {
                reassembly_buffers_.erase(key);
                continue;
            }

            /*
             * Copy this fragment only if it has not already been received.
             *
             * UDP can theoretically deliver duplicate packets.
             */
            if (!rb.received[header.fragment_id])
            {
                std::memcpy(
                    rb.data.data() + offset,
                    payload,
                    header.payload_size
                );

                rb.received[header.fragment_id] = true;
                rb.received_count++;
            }

            /*
             * If not all fragments have arrived yet, wait for more packets.
             */
            if (rb.received_count != rb.fragment_count)
            {
                continue;
            }

            /*
             * At this point, the full serialized ROS 2 message has been
             * reconstructed in rb.data.
             *
             * We can now deserialize and republish according to topic_id.
             */
            switch (rb.topic_id)
            {
                case DOG_ODOM:
                    if (bridge_dog_odom_)
                        publish_deserialized<nav_msgs::msg::Odometry>(
                            rb.data.data(), rb.total_size, odom_pub_);
                    break;
                
                case MOLA_ODOM:
                    if (bridge_mola_odom_)
                        publish_deserialized<nav_msgs::msg::Odometry>(
                            rb.data.data(), rb.total_size, mola_odom_pub_);
                    break;

                case DOG_ODOM_FILTERED:
                    if (bridge_odom_)
                        publish_deserialized<nav_msgs::msg::Odometry>(
                            rb.data.data(), rb.total_size, odom_pub_filter_);
                    break;

                case ODOM_MAIN:
                    if (bridge_odom_)
                        publish_deserialized<nav_msgs::msg::Odometry>(
                            rb.data.data(), rb.total_size, odom_pub_artificial);
                    break;
                    

                case ODOM_FASTLIO:
                    if (bridge_odom_)
                        publish_deserialized<nav_msgs::msg::Odometry>(
                            rb.data.data(), rb.total_size, odom_pub_fastlio_);
                    break;

                case DOG_IMU:
                    if (bridge_imu_)
                        publish_deserialized<sensor_msgs::msg::Imu>(
                            rb.data.data(), rb.total_size, imu_pub_);
                    break;

                case LASER_SCAN:
                    RCLCPP_INFO_THROTTLE(
                        get_logger(),
                        *get_clock(),
                        1000,
                        "[SCAN RX] republishing /repub/scan"
                    );
                    if (bridge_scan_)
                        publish_deserialized<sensor_msgs::msg::LaserScan>(
                            rb.data.data(), rb.total_size, scan_pub_);
                    break;

                case TF_DYNAMIC:
                    if (bridge_tf_)
                        publish_deserialized<tf2_msgs::msg::TFMessage>(
                            rb.data.data(), rb.total_size, tf_pub_);
                    break;

                case TF_STATIC:
                    if (bridge_tf_)
                        publish_deserialized<tf2_msgs::msg::TFMessage>(
                            rb.data.data(), rb.total_size, tf_static_pub_);
                    break;

                case JOINT_STATES:
                    if (bridge_joint_states_)
                        publish_deserialized<sensor_msgs::msg::JointState>(
                            rb.data.data(), rb.total_size, joint_states_pub_);
                    break;

                case MAP_TOPIC:
                    if (bridge_map_)
                        publish_deserialized<
                            nav_msgs::msg::OccupancyGrid>(
                                rb.data.data(),
                                rb.total_size,
                                map_pub_);
                    break;
                
                case GLOBAL_COSTMAP:
                    if (bridge_global_costmap_)
                        publish_deserialized<nav_msgs::msg::OccupancyGrid>(
                            rb.data.data(),
                            rb.total_size,
                            global_costmap_pub_
                        );
                    break;

                case LOCAL_COSTMAP:
                    if (bridge_local_costmap_)
                        publish_deserialized<nav_msgs::msg::OccupancyGrid>(
                            rb.data.data(),
                            rb.total_size,
                            local_costmap_pub_
                        );
                    break;

                default:
                    /*
                     * Unknown topic_id.
                     * Ignore reconstructed message.
                     */
                    break;
            }

            /*
             * Remove the buffer once the message has been published.
             */
            reassembly_buffers_.erase(key);
        }
    }

    template<typename MsgT>
    void send_command_message(
        const MsgT & msg,
        const uint32_t topic_id)
    {
        rclcpp::Serialization<MsgT>
            serializer;

        rclcpp::SerializedMessage
            serialized_msg;

        serializer.serialize_message(
            &msg,
            &serialized_msg
        );

        const uint8_t * serialized_data =
            serialized_msg
                .get_rcl_serialized_message()
                .buffer;

        const size_t total_size =
            serialized_msg.size();


        const uint32_t msg_id =
            ++command_msg_counter_;


        const uint32_t fragment_count =
            static_cast<uint32_t>(
                (
                    total_size
                    +
                    MAX_UDP_FRAGMENT_PAYLOAD
                    -
                    1
                )
                /
                MAX_UDP_FRAGMENT_PAYLOAD
            );


        for (
            uint32_t fragment_id = 0;
            fragment_id < fragment_count;
            ++fragment_id)
        {
            const size_t offset =
                static_cast<size_t>(
                    fragment_id
                )
                *
                MAX_UDP_FRAGMENT_PAYLOAD;


            const size_t remaining =
                total_size
                -
                offset;


            const size_t fragment_payload_size =
                std::min(
                    MAX_UDP_FRAGMENT_PAYLOAD,
                    remaining
                );


            PacketHeader header{};

            header.topic_id =
                topic_id;

            header.msg_id =
                msg_id;

            header.fragment_id =
                fragment_id;

            header.fragment_count =
                fragment_count;

            header.payload_size =
                static_cast<uint32_t>(
                    fragment_payload_size
                );

            header.total_size =
                static_cast<uint32_t>(
                    total_size
                );


            std::vector<uint8_t>
                packet(
                    sizeof(PacketHeader)
                    +
                    fragment_payload_size
                );


            std::memcpy(
                packet.data(),
                &header,
                sizeof(PacketHeader)
            );


            std::memcpy(
                packet.data()
                    +
                    sizeof(PacketHeader),
                serialized_data
                    +
                    offset,
                fragment_payload_size
            );


            sendto(
                command_sock_,
                packet.data(),
                packet.size(),
                0,
                reinterpret_cast<sockaddr *>(
                    &command_target_addr_
                ),
                sizeof(command_target_addr_)
            );
        }
    }

    /*
     * Convert a serialized ROS 2 payload back into a typed message
     * and publish it.
     */
    template<typename MsgT>
    void publish_deserialized(
        const uint8_t * payload,
        uint32_t payload_size,
        typename rclcpp::Publisher<MsgT>::SharedPtr publisher)
    {
        /*
         * Allocate a SerializedMessage buffer large enough to hold the payload.
         */
        rclcpp::SerializedMessage serialized_msg(payload_size);

        auto & rcl_serialized =
            serialized_msg.get_rcl_serialized_message();

        /*
         * Copy UDP payload into ROS 2 serialized message buffer.
         */
        std::memcpy(
            rcl_serialized.buffer,
            payload,
            payload_size
        );

        rcl_serialized.buffer_length = payload_size;

        /*
         * Deserialize binary CDR payload into typed ROS 2 message.
         */
        rclcpp::Serialization<MsgT> serializer;
        MsgT msg;

        serializer.deserialize_message(
            &serialized_msg,
            &msg
        );

        /*
         * Publish on Wi-Fi-visible ROS 2 topic.
         */
        publisher->publish(msg);
    }

private:
    /*
     * UDP resources.
     */
    int udp_port_{16001};
    int sock_{-1};

    /*
     * Background receive thread.
     */
    std::atomic<bool> running_{false};
    std::thread worker_;

    /*
     * ROS 2 publishers.
     */
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr mola_odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_filter_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_fastlio_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_artificial;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
    rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
    /*
    * Republishes the lightweight 2D scan generated from the
    * Livox PointCloud2 on the robot side.
    *
    * Topic:
    *   /repub/scan
    */
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;

    std::string command_udp_host_{
        "127.0.0.1"
    };

    rclcpp_action::Server<
            NavigateToPose
        >::SharedPtr navigate_to_pose_server_;

    int command_udp_port_{
        16003
    };

    int command_sock_{
        -1
    };

    sockaddr_in command_target_addr_{};

    uint32_t command_msg_counter_{
        0
    };


    bool bridge_goal_{
        true
    };

    bool bridge_initialpose_{
        true
    };

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
    global_costmap_pub_;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
        local_costmap_pub_;

    bool bridge_global_costmap_{true};
    bool bridge_local_costmap_{true};


    rclcpp::Subscription<
        geometry_msgs::msg::
            PoseWithCovarianceStamped
    >::SharedPtr initial_pose_sub_;

    /* 
    Bridge management
    */
    bool bridge_odom_{false};
    bool bridge_dog_odom_{true};
    bool bridge_mola_odom_{true};
    bool bridge_imu_{false};
    bool bridge_scan_{false};
    bool bridge_tf_{false};
    bool bridge_joint_states_{false};
    bool bridge_map_{false};

    /*
        * Reassembly buffers indexed by:
        *
        *   (topic_id, msg_id)
        *
        * This allows several messages from different topics to be reconstructed
        * independently.
    */
    std::map<std::pair<uint32_t, uint32_t>, ReassemblyBuffer> reassembly_buffers_;

    /*
        * Must match the sender-side fragment payload size.
        *
        * It is used to compute where each fragment belongs in the reconstructed
        * serialized ROS 2 message.
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
            std::make_shared<TopicsUdpReceiver>();

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
            << "[topics_udp_receiver_node] Fatal exception: "
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
            << "[topics_udp_receiver_node] Unknown fatal exception"
            << std::endl;

        if (rclcpp::ok())
        {
            rclcpp::shutdown();
        }

        return 1;
    }
}