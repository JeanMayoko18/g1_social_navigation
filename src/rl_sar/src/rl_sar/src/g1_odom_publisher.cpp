#include <arpa/inet.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <chrono>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2_ros/transform_broadcaster.h"


#pragma pack(push, 1)

struct OdomPacket
{
  std::uint32_t magic;

  double x;
  double y;
  double z;

  double vx;
  double vy;
  double vz;

  double yaw_speed;

  double qw;
  double qx;
  double qy;
  double qz;

  std::uint64_t sequence;
};

#pragma pack(pop)


class G1OdomPublisher : public rclcpp::Node
{
public:
  G1OdomPublisher()
  : Node("g1_odom_publisher")
  {
    // ============================================================
    // Parameters
    // ============================================================

    udp_port_ =
      declare_parameter<int>(
        "udp_port",
        17001);

    output_topic_ =
      declare_parameter<std::string>(
        "output_topic",
        "/odom");

    odom_frame_ =
      declare_parameter<std::string>(
        "odom_frame",
        "odom");

    base_frame_ =
      declare_parameter<std::string>(
        "base_frame",
        "base_footprint");

    publish_tf_ =
      declare_parameter<bool>(
        "publish_tf",
        true);

    zero_local_origin_ =
      declare_parameter<bool>(
        "zero_local_origin",
        true);

    publish_rate_ =
      declare_parameter<double>(
        "publish_rate",
        50.0);
        
    // ============================================================
    // ROS publisher
    // ============================================================

    rclcpp::QoS qos(
      rclcpp::KeepLast(10));

    qos.reliable();
    qos.durability_volatile();

    odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(
        output_topic_,
        qos);


    // ============================================================
    // TF broadcaster
    // ============================================================

    if (publish_tf_)
    {
      tf_broadcaster_ =
        std::make_shared<
          tf2_ros::TransformBroadcaster>(
            this);
    }


    // ============================================================
    // UDP socket
    // ============================================================

    socket_fd_ =
      ::socket(
        AF_INET,
        SOCK_DGRAM,
        0);

    if (socket_fd_ < 0)
    {
      throw std::runtime_error(
        "Failed to create UDP socket");
    }


    sockaddr_in address{};

    address.sin_family =
      AF_INET;

    address.sin_addr.s_addr =
      htonl(
        INADDR_LOOPBACK);

    address.sin_port =
      htons(
        static_cast<std::uint16_t>(
          udp_port_));


    if (
      ::bind(
        socket_fd_,
        reinterpret_cast<sockaddr *>(
          &address),
        sizeof(address))
      < 0)
    {
      throw std::runtime_error(
        "Failed to bind UDP socket on port "
        +
        std::to_string(
          udp_port_));
    }

    // ============================================================
    // Dedicated UDP reception thread
    //
    // Important:
    //
    // There is NO ROS timer anymore.
    //
    // Each valid incoming odometry packet is immediately converted
    // and published as:
    //
    //   /odom
    //   odom -> base_footprint
    //
    // This removes the intermediate:
    //
    //   latest_state -> timer -> publication
    //
    // path that previously showed publication gaps.
    // ============================================================

    receiver_thread_ =
      std::thread(
        &G1OdomPublisher::receiverLoop,
        this);

    publisher_thread_ =
      std::thread(
        &G1OdomPublisher::publisherLoop,
        this);

    RCLCPP_INFO(
      get_logger(),
      "\n"
      "=========================================================\n"
      "      G1 SIMPLE ROS ODOM UDP RECEIVER\n"
      "=========================================================\n"
      " UDP port          : %d\n"
      " Output topic      : %s\n"
      " Odom frame        : %s\n"
      " Base frame        : %s\n"
      " Publish TF        : %s\n"
      " Zero local origin : %s\n"
      "\n"
      " Publication mode:\n"
      "   UDP receive -> immediate /odom + TF publication\n"
      "   No ROS timer\n"
      "   No latest-state buffer\n"
      "=========================================================",
      udp_port_,
      output_topic_.c_str(),
      odom_frame_.c_str(),
      base_frame_.c_str(),
      publish_tf_ ? "YES" : "NO",
      zero_local_origin_ ? "YES" : "NO");
  }


  ~G1OdomPublisher()
  {
    running_.store(
      false);

    if (socket_fd_ >= 0)
    {
      ::shutdown(
        socket_fd_,
        SHUT_RDWR);

      ::close(
        socket_fd_);

      socket_fd_ =
        -1;
    }

    if (receiver_thread_.joinable())
    {
      receiver_thread_.join();
    }

    if (publisher_thread_.joinable())
    {
      publisher_thread_.join();
    }
  }


private:

  // ==============================================================
  // Utilities
  // ==============================================================

  static double wrapAngle(
    const double angle)
  {
    return std::atan2(
      std::sin(angle),
      std::cos(angle));
  }


  static double quaternionToYaw(
    const double qw,
    const double qx,
    const double qy,
    const double qz)
  {
    const double siny =
      2.0 *
      (
        qw * qz
        +
        qx * qy
      );

    const double cosy =
      1.0 -
      2.0 *
      (
        qy * qy
        +
        qz * qz
      );

    return std::atan2(
      siny,
      cosy);
  }


  static geometry_msgs::msg::Quaternion
  yawToQuaternion(
    const double yaw)
  {
    geometry_msgs::msg::Quaternion q;

    q.x =
      0.0;

    q.y =
      0.0;

    q.z =
      std::sin(
        yaw * 0.5);

    q.w =
      std::cos(
        yaw * 0.5);

    return q;
  }


  // ==============================================================
  // UDP reception
  // ==============================================================

  void receiverLoop()
  {
    while (
      running_.load())
    {
      OdomPacket packet{};


      const ssize_t received =
        ::recv(
          socket_fd_,
          &packet,
          sizeof(packet),
          0);


      if (
        !running_.load())
      {
        break;
      }


      if (
        received
        !=
        static_cast<ssize_t>(
          sizeof(packet)))
      {
        continue;
      }


      if (
        packet.magic
        !=
        kPacketMagic)
      {
        continue;
      }


      // ----------------------------------------------------------
      // Reject invalid numeric values
      // ----------------------------------------------------------

      if (
        !packetIsValid(
          packet))
      {
        continue;
      }


      // ----------------------------------------------------------
      // Publish immediately.
      //
      // No intermediate timer.
      // ----------------------------------------------------------
      
      {
        std::lock_guard<std::mutex> lock(packet_mutex_);
        latest_packet_ = packet;
        have_packet_ = true;
      }
    }
  }


  // ==============================================================
  // Packet validation
  // ==============================================================

  static bool packetIsValid(
    const OdomPacket & packet)
  {
    const double values[] = {
      packet.x,
      packet.y,
      packet.z,
      packet.vx,
      packet.vy,
      packet.vz,
      packet.yaw_speed,
      packet.qw,
      packet.qx,
      packet.qy,
      packet.qz
    };


    for (
      const double value :
      values)
    {
      if (
        !std::isfinite(
          value))
      {
        return false;
      }
    }


    if (
      std::abs(packet.x)
        > 1000.0
      ||
      std::abs(packet.y)
        > 1000.0
      ||
      std::abs(packet.z)
        > 1000.0)
    {
      return false;
    }


    const double qnorm =
      std::sqrt(
        packet.qw * packet.qw
        +
        packet.qx * packet.qx
        +
        packet.qy * packet.qy
        +
        packet.qz * packet.qz);


    return (
      qnorm >= 0.5
      &&
      qnorm <= 1.5
    );
  }


  // ==============================================================
  // Convert and publish one received packet
  // ==============================================================
  void publisherLoop()
  {
    using clock = std::chrono::steady_clock;

    const auto period =
      std::chrono::duration<double>(
        1.0 / publish_rate_);

    auto next =
      clock::now();

    while (running_.load())
    {
      next +=
        std::chrono::duration_cast<clock::duration>(
          period);

      OdomPacket packet{};
      bool available = false;

      {
        std::lock_guard<std::mutex> lock(packet_mutex_);

        if (have_packet_)
        {
          packet = latest_packet_;
          available = true;
        }
      }

      if (available)
      {
        publishPacket(packet);
      }

      std::this_thread::sleep_until(next);

      // Avoid trying to "catch up" with a publication burst
      // after a long scheduling delay.
      const auto now = clock::now();

      if (now > next + std::chrono::milliseconds(100))
      {
        next = now;
      }
    }
  }

  void publishPacket(
    const OdomPacket & packet)
  {
    // ------------------------------------------------------------
    // Raw yaw
    // ------------------------------------------------------------

    const double raw_yaw =
      quaternionToYaw(
        packet.qw,
        packet.qx,
        packet.qy,
        packet.qz);


    // ------------------------------------------------------------
    // Initialize local navigation origin once
    // ------------------------------------------------------------

    if (
      zero_local_origin_
      &&
      !have_origin_)
    {
      origin_x_ =
        packet.x;

      origin_y_ =
        packet.y;

      origin_yaw_ =
        raw_yaw;

      have_origin_ =
        true;


      RCLCPP_INFO(
        get_logger(),
        "Local odometry origin set | "
        "x=%+.4f y=%+.4f yaw=%+.2f deg",
        origin_x_,
        origin_y_,
        origin_yaw_
          * 180.0
          / M_PI);
    }


    // ------------------------------------------------------------
    // Compute local pose
    // ------------------------------------------------------------

    double x =
      packet.x;

    double y =
      packet.y;

    double yaw =
      raw_yaw;


    if (
      zero_local_origin_
      &&
      have_origin_)
    {
      const double dx =
        packet.x
        -
        origin_x_;

      const double dy =
        packet.y
        -
        origin_y_;


      const double c =
        std::cos(
          -origin_yaw_);

      const double s =
        std::sin(
          -origin_yaw_);


      x =
        c * dx
        -
        s * dy;

      y =
        s * dx
        +
        c * dy;

      yaw =
        wrapAngle(
          raw_yaw
          -
          origin_yaw_);
    }


    // ------------------------------------------------------------
    // One exact ROS timestamp for both /odom and TF
    // ------------------------------------------------------------

    const rclcpp::Time now =
      get_clock()->now();


    const auto orientation =
      yawToQuaternion(
        yaw);


    // ============================================================
    // /odom
    // ============================================================

    nav_msgs::msg::Odometry odom;


    odom.header.stamp =
      now;

    odom.header.frame_id =
      odom_frame_;

    odom.child_frame_id =
      base_frame_;


    odom.pose.pose.position.x =
      x;

    odom.pose.pose.position.y =
      y;

    odom.pose.pose.position.z =
      0.0;

    odom.pose.pose.orientation =
      orientation;


    odom.twist.twist.linear.x =
      packet.vx;

    odom.twist.twist.linear.y =
      packet.vy;

    odom.twist.twist.linear.z =
      0.0;

    odom.twist.twist.angular.x =
      0.0;

    odom.twist.twist.angular.y =
      0.0;

    odom.twist.twist.angular.z =
      packet.yaw_speed;


    // ------------------------------------------------------------
    // Covariances
    // ------------------------------------------------------------

    odom.pose.covariance[0] =
      0.01;

    odom.pose.covariance[7] =
      0.01;

    odom.pose.covariance[14] =
      1e3;

    odom.pose.covariance[21] =
      1e3;

    odom.pose.covariance[28] =
      1e3;

    odom.pose.covariance[35] =
      0.05;


    odom.twist.covariance =
      odom.pose.covariance;


    // ------------------------------------------------------------
    // Publish odometry immediately
    // ------------------------------------------------------------

    odom_pub_->publish(
      odom);


    // ============================================================
    // Matching TF
    // ============================================================

    if (
      publish_tf_
      &&
      tf_broadcaster_)
    {
      geometry_msgs::msg::TransformStamped tf;


      tf.header.stamp =
        now;

      tf.header.frame_id =
        odom_frame_;

      tf.child_frame_id =
        base_frame_;


      tf.transform.translation.x =
        x;

      tf.transform.translation.y =
        y;

      tf.transform.translation.z =
        0.0;

      tf.transform.rotation =
        orientation;


      tf_broadcaster_->sendTransform(
        tf);
    }
  }


  // ==============================================================
  // Constants
  // ==============================================================

  static constexpr std::uint32_t
    kPacketMagic =
      0x47314F44;


  // ==============================================================
  // Parameters
  // ==============================================================

  int udp_port_{
    17001};


  std::string output_topic_;

  std::string odom_frame_;

  std::string base_frame_;


  bool publish_tf_{
    true};


  bool zero_local_origin_{
    true};


  // ==============================================================
  // UDP
  // ==============================================================

  int socket_fd_{
    -1};


  std::atomic<bool>
    running_{
      true};


  std::thread
    receiver_thread_;


  // ==============================================================
  // Local navigation origin
  // ==============================================================

  bool have_origin_{
    false};


  double origin_x_{
    0.0};


  double origin_y_{
    0.0};


  double origin_yaw_{
    0.0};

  double publish_rate_{50.0};

  std::mutex packet_mutex_;

  OdomPacket latest_packet_{};

  bool have_packet_{false};

  std::thread publisher_thread_;


  // ==============================================================
  // ROS interfaces
  // ==============================================================

  rclcpp::Publisher<
    nav_msgs::msg::Odometry>::SharedPtr
    odom_pub_;


  std::shared_ptr<
    tf2_ros::TransformBroadcaster>
    tf_broadcaster_;
};


// ============================================================================
// Main
// ============================================================================

int main(
  int argc,
  char ** argv)
{
  try
  {
    rclcpp::init(
      argc,
      argv);


    auto node =
      std::make_shared<
        G1OdomPublisher>();


    rclcpp::spin(
      node);


    rclcpp::shutdown();


    return 0;
  }


  catch (
    const std::exception & e)
  {
    std::cerr
      << "[g1_odom_publisher] Fatal exception: "
      << e.what()
      << std::endl;


    if (
      rclcpp::ok())
    {
      rclcpp::shutdown();
    }


    return 1;
  }
}