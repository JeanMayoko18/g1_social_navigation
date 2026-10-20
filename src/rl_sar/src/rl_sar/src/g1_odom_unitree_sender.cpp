#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/idl/go2/SportModeState_.hpp"


// ============================================================================
// Native Unitree message type
// ============================================================================

using SportModeState =
  unitree_go::msg::dds_::SportModeState_;


// ============================================================================
// UDP packet exchanged with the ROS-only odometry publisher
//
// IMPORTANT:
//
// This structure is deliberately simple and contains only the state required
// by the navigation odometry node.
//
// No ROS types are used here.
//
// Architecture:
//
//   Unitree DDS process
//          |
//          | OdomPacket over localhost UDP
//          v
//   ROS2 odometry publisher
//
// Keeping the two DDS implementations in separate processes avoids mixing:
//
//   - Unitree SDK CycloneDDS
//   - ROS 2 Foxy CycloneDDS
//
// inside the same executable.
// ============================================================================

#pragma pack(push, 1)

struct OdomPacket
{
  // ------------------------------------------------------------------------
  // Packet identifier
  //
  // ASCII equivalent:
  //
  //   G1OD
  //
  // Allows the receiver to reject unrelated UDP packets.
  // ------------------------------------------------------------------------

  std::uint32_t magic;


  // ------------------------------------------------------------------------
  // Native Unitree position
  // ------------------------------------------------------------------------

  double x;
  double y;
  double z;


  // ------------------------------------------------------------------------
  // Native Unitree velocity
  // ------------------------------------------------------------------------

  double vx;
  double vy;
  double vz;


  // ------------------------------------------------------------------------
  // Native yaw velocity
  // ------------------------------------------------------------------------

  double yaw_speed;


  // ------------------------------------------------------------------------
  // Native Unitree IMU quaternion
  //
  // Convention:
  //
  //   [w, x, y, z]
  // ------------------------------------------------------------------------

  double qw;
  double qx;
  double qy;
  double qz;


  // ------------------------------------------------------------------------
  // Monotonic packet sequence
  //
  // Useful for detecting:
  //
  //   - packet drops
  //   - reordered packets
  //   - native DDS interruptions
  // ------------------------------------------------------------------------

  std::uint64_t sequence;
};

#pragma pack(pop)


// ============================================================================
// G1OdomUnitreeSender
// ============================================================================

class G1OdomUnitreeSender
{
public:

  G1OdomUnitreeSender(
    const std::string & network_interface,
    const std::string & udp_host,
    const int udp_port)
  :
    network_interface_(
      network_interface),

    udp_host_(
      udp_host),

    udp_port_(
      udp_port)
  {
    // ======================================================================
    // 1. Initialize Unitree native DDS
    //
    // This executable is Unitree-only.
    //
    // There is intentionally:
    //
    //   - no rclcpp
    //   - no ROS publisher
    //   - no ROS subscriber
    //   - no tf2
    //
    // This guarantees that only the Unitree SDK CycloneDDS implementation
    // is loaded into this process.
    // ======================================================================

    std::cout
      << "[G1 Odom Sender] Initializing Unitree DDS on interface: "
      << network_interface_
      << std::endl;


    unitree::robot::ChannelFactory::Instance()->Init(
      0,
      network_interface_);


    // ======================================================================
    // 2. Create UDP socket
    //
    // The destination is normally:
    //
    //   127.0.0.1:17001
    //
    // Therefore no network transport outside the robot is involved.
    // ======================================================================

    socket_fd_ =
      ::socket(
        AF_INET,
        SOCK_DGRAM,
        0);


    if (
      socket_fd_
      < 0)
    {
      throw std::runtime_error(
        "Failed to create UDP socket");
    }


    std::memset(
      &destination_,
      0,
      sizeof(destination_));


    destination_.sin_family =
      AF_INET;


    destination_.sin_port =
      htons(
        static_cast<std::uint16_t>(
          udp_port_));


    const int address_result =
      inet_pton(
        AF_INET,
        udp_host_.c_str(),
        &destination_.sin_addr);


    if (
      address_result
      != 1)
    {
      throw std::runtime_error(
        "Invalid UDP destination address: "
        +
        udp_host_);
    }


    // ======================================================================
    // 3. Create native Unitree DDS subscriber
    //
    // Source:
    //
    //   rt/odommodestate
    //
    // This is the same native DDS source used by the reference architecture.
    //
    // The installed SDK uses:
    //
    //   InitChannel(...)
    //
    // and NOT:
    //
    //   Init(...)
    // ======================================================================

    odom_subscriber_ =
      std::make_unique<
        unitree::robot::ChannelSubscriber<
          SportModeState>>(
            "rt/odommodestate");


    odom_subscriber_->InitChannel(
      std::bind(
        &G1OdomUnitreeSender::odomCallback,
        this,
        std::placeholders::_1),
      10);


    // ======================================================================
    // 4. Startup information
    // ======================================================================

    std::cout
      << "\n"
      << "============================================================\n"
      << "           G1 UNITREE ODOM UDP SENDER\n"
      << "============================================================\n"
      << " DDS source       : rt/odommodestate\n"
      << " Network interface: " << network_interface_ << "\n"
      << " UDP destination  : "
      << udp_host_
      << ":"
      << udp_port_
      << "\n"
      << " Packet size      : "
      << sizeof(OdomPacket)
      << " bytes\n"
      << "\n"
      << " Architecture:\n"
      << "   Unitree native DDS\n"
      << "       -> validate state\n"
      << "       -> UDP localhost\n"
      << "       -> ROS-only odometry publisher\n"
      << "============================================================\n"
      << std::endl;
  }


  // ==========================================================================
  // Destructor
  // ==========================================================================

  ~G1OdomUnitreeSender()
  {
    running_.store(
      false);


    if (
      socket_fd_
      >= 0)
    {
      ::close(
        socket_fd_);


      socket_fd_ =
        -1;
    }
  }


  // ==========================================================================
  // Main loop
  //
  // The actual data flow runs from the native DDS callback.
  //
  // This loop only prints diagnostics once per second.
  // ==========================================================================

  void spin()
  {
    std::uint64_t previous_received =
      0;


    std::uint64_t previous_sent =
      0;


    while (
      running_.load())
    {
      std::this_thread::sleep_for(
        std::chrono::seconds(1));


      const std::uint64_t received =
        received_count_.load();


      const std::uint64_t sent =
        sent_count_.load();


      const std::uint64_t invalid =
        invalid_count_.load();


      const std::uint64_t send_errors =
        send_error_count_.load();


      const std::uint64_t current_sequence =
        sequence_.load();


      const std::uint64_t received_rate =
        received
        -
        previous_received;


      const std::uint64_t sent_rate =
        sent
        -
        previous_sent;


      previous_received =
        received;


      previous_sent =
        sent;


      std::cout
        << "[G1 ODOM UDP] "
        << "DDS="
        << received_rate
        << " Hz"
        << " | UDP="
        << sent_rate
        << " Hz"
        << " | sequence="
        << current_sequence
        << " | total_rx="
        << received
        << " | total_tx="
        << sent
        << " | invalid="
        << invalid
        << " | send_errors="
        << send_errors
        << std::endl;
    }
  }


private:

  // ==========================================================================
  // Validate packet contents
  //
  // This protects the ROS navigation process from known startup garbage.
  //
  // Earlier observations showed invalid transient Unitree states with:
  //
  //   - huge positions
  //   - invalid quaternion values
  //
  // These samples must never be forwarded.
  // ==========================================================================

  bool packetIsValid(
    const OdomPacket & packet) const
  {
    // ------------------------------------------------------------------------
    // Check every floating-point field
    // ------------------------------------------------------------------------

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


    // ------------------------------------------------------------------------
    // Reject obviously corrupted position values
    // ------------------------------------------------------------------------

    constexpr double max_abs_position =
      1000.0;


    if (
      std::abs(
        packet.x)
      > max_abs_position
      ||
      std::abs(
        packet.y)
      > max_abs_position
      ||
      std::abs(
        packet.z)
      > max_abs_position)
    {
      return false;
    }


    // ------------------------------------------------------------------------
    // Validate quaternion norm
    //
    // Native Unitree convention:
    //
    //   [w, x, y, z]
    // ------------------------------------------------------------------------

    const double quaternion_norm =
      std::sqrt(
        packet.qw
          * packet.qw
        +
        packet.qx
          * packet.qx
        +
        packet.qy
          * packet.qy
        +
        packet.qz
          * packet.qz);


    constexpr double min_quaternion_norm =
      0.5;


    constexpr double max_quaternion_norm =
      1.5;


    if (
      quaternion_norm
      < min_quaternion_norm
      ||
      quaternion_norm
      > max_quaternion_norm)
    {
      return false;
    }


    return true;
  }


  // ==========================================================================
  // Native Unitree DDS callback
  //
  // Expected native rate:
  //
  //   approximately 500 Hz
  //
  // The callback performs only:
  //
  //   1. copy native state
  //   2. validate state
  //   3. send small UDP packet
  //
  // No ROS operation occurs here.
  //
  // This keeps the callback lightweight and independent from the ROS
  // executor.
  // ==========================================================================

  void odomCallback(
    const void * message)
  {
    if (
      message
      ==
      nullptr)
    {
      return;
    }


    received_count_.fetch_add(
      1,
      std::memory_order_relaxed);


    const std::uint64_t callback_count =
      callback_count_.fetch_add(
        1,
        std::memory_order_relaxed)
      +
      1;


    // Native source is approximately 500 Hz.
    // Forward only one message out of ten.
    if (
      callback_count
      % kSendEveryNMessages
      != 0)
    {
      return;
    }


    const auto * msg =
      static_cast<
        const SportModeState *>(
          message);


    OdomPacket packet{};


    // ------------------------------------------------------------------------
    // Packet signature
    // ------------------------------------------------------------------------

    packet.magic =
      kPacketMagic;


    // ------------------------------------------------------------------------
    // Native Unitree position
    // ------------------------------------------------------------------------

    packet.x =
      static_cast<double>(
        msg->position()[0]);


    packet.y =
      static_cast<double>(
        msg->position()[1]);


    packet.z =
      static_cast<double>(
        msg->position()[2]);


    // ------------------------------------------------------------------------
    // Native Unitree velocity
    // ------------------------------------------------------------------------

    packet.vx =
      static_cast<double>(
        msg->velocity()[0]);


    packet.vy =
      static_cast<double>(
        msg->velocity()[1]);


    packet.vz =
      static_cast<double>(
        msg->velocity()[2]);


    // ------------------------------------------------------------------------
    // Native Unitree yaw velocity
    // ------------------------------------------------------------------------

    packet.yaw_speed =
      static_cast<double>(
        msg->yaw_speed());


    // ------------------------------------------------------------------------
    // Native Unitree IMU orientation
    //
    // Convention:
    //
    //   quaternion[0] = w
    //   quaternion[1] = x
    //   quaternion[2] = y
    //   quaternion[3] = z
    // ------------------------------------------------------------------------

    packet.qw =
      static_cast<double>(
        msg->imu_state()
          .quaternion()[0]);


    packet.qx =
      static_cast<double>(
        msg->imu_state()
          .quaternion()[1]);


    packet.qy =
      static_cast<double>(
        msg->imu_state()
          .quaternion()[2]);


    packet.qz =
      static_cast<double>(
        msg->imu_state()
          .quaternion()[3]);


    // ------------------------------------------------------------------------
    // Monotonic sequence
    //
    // Increment the sequence for every selected packet candidate.
    // At an input rate of approximately 500 Hz and with a 1/10
    // decimation factor, this sequence advances at approximately 50 Hz.
    //
    // The receiver can use this later to detect missing native states.
    // ------------------------------------------------------------------------

    packet.sequence =
      sequence_.fetch_add(
        1,
        std::memory_order_relaxed)
      +
      1;


    // ------------------------------------------------------------------------
    // Validate packet
    // ------------------------------------------------------------------------

    if (
      !packetIsValid(
        packet))
    {
      invalid_count_.fetch_add(
        1,
        std::memory_order_relaxed);


      return;
    }


    // ------------------------------------------------------------------------
    // Send packet through localhost UDP
    //
    // UDP is intentionally used instead of another ROS topic:
    //
    //   - no extra ROS DDS path
    //   - no serialization through rmw
    //   - Unitree DDS remains isolated
    //   - receiver always keeps only the newest physical state
    // ------------------------------------------------------------------------

    const ssize_t result =
      ::sendto(
        socket_fd_,
        &packet,
        sizeof(packet),
        MSG_DONTWAIT,
        reinterpret_cast<
          const sockaddr *>(
            &destination_),
        sizeof(destination_));


    if (
      result
      ==
      static_cast<ssize_t>(
        sizeof(packet)))
    {
      sent_count_.fetch_add(
        1,
        std::memory_order_relaxed);
    }
    else
    {
      send_error_count_.fetch_add(
        1,
        std::memory_order_relaxed);
    }
  }


  // ==========================================================================
  // Constants
  // ==========================================================================

  static constexpr std::uint32_t
    kPacketMagic =
      0x47314F44;


  // ==========================================================================
  // Configuration
  // ==========================================================================

  std::string network_interface_;

  std::string udp_host_;

  int udp_port_{
    17001};


  // ==========================================================================
  // UDP
  // ==========================================================================

  int socket_fd_{
    -1};


  sockaddr_in destination_{};


  // ==========================================================================
  // Unitree DDS
  // ==========================================================================

  std::unique_ptr<
    unitree::robot::ChannelSubscriber<
      SportModeState>>
    odom_subscriber_;


  // ==========================================================================
  // Runtime state
  // ==========================================================================

  std::atomic<bool>
    running_{
      true};


  // ==========================================================================
  // Diagnostics
  // ==========================================================================

  std::atomic<std::uint64_t>
    sequence_{
      0};


  std::atomic<std::uint64_t>
    received_count_{
      0};


  std::atomic<std::uint64_t>
    sent_count_{
      0};


  std::atomic<std::uint64_t>
    invalid_count_{
      0};


  std::atomic<std::uint64_t>
    send_error_count_{
      0};
  
  static constexpr std::uint64_t
    kSendEveryNMessages = 10;

  std::atomic<std::uint64_t>
    callback_count_{0};
};


// ============================================================================
// Main
//
// Usage:
//
//   g1_odom_unitree_sender eth0
//
// Optional future extension:
//
//   g1_odom_unitree_sender eth0 127.0.0.1 17001
//
// At the moment the destination remains local to the robot.
// ============================================================================

int main(
  int argc,
  char ** argv)
{
  try
  {
    // ------------------------------------------------------------------------
    // Network interface
    // ------------------------------------------------------------------------

    std::string network_interface =
      "eth0";


    if (
      argc
      > 1)
    {
      network_interface =
        argv[1];
    }


    // ------------------------------------------------------------------------
    // UDP configuration
    //
    // Keep the bridge completely local to the robot.
    // ------------------------------------------------------------------------

    const std::string udp_host =
      "127.0.0.1";


    constexpr int udp_port =
      17001;


    // ------------------------------------------------------------------------
    // Start sender
    // ------------------------------------------------------------------------

    G1OdomUnitreeSender sender(
      network_interface,
      udp_host,
      udp_port);


    sender.spin();


    return 0;
  }


  catch (
    const std::exception & e)
  {
    std::cerr
      << "[g1_odom_unitree_sender] Fatal exception: "
      << e.what()
      << std::endl;


    return 1;
  }


  catch (...)
  {
    std::cerr
      << "[g1_odom_unitree_sender] "
      << "Unknown fatal exception"
      << std::endl;


    return 1;
  }
}