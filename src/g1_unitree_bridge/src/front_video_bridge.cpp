#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "unitree_go/msg/go2_front_video_data.hpp"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}


// =============================================================================
// FRONT VIDEO BRIDGE NODE
//
// Responsibility:
//   Convert the proprietary Unitree front-camera H.264 stream into standard
//   ROS 2 camera topics.
//
// Input:
//   /frontvideostream
//   unitree_go/msg/Go2FrontVideoData
//
// Processing:
//   Unitree H.264 chunks
//        -> FFmpeg parser
//        -> H.264 decoder
//        -> decoded AVFrame
//        -> libswscale BGR24 conversion
//        -> latest-frame buffer
//
// Output:
//   /camera/front/image_raw
//   sensor_msgs/msg/Image
//
//   /camera/front/camera_info
//   sensor_msgs/msg/CameraInfo
//
// Execution model:
//   The node uses separate callback groups for:
//     - H.264 reception/decoding
//     - periodic ROS image publication
//
//   A MultiThreadedExecutor allows both paths to execute independently.
// =============================================================================

class FrontVideoBridge : public rclcpp::Node
{
public:
  // ===========================================================================
  // NODE CONSTRUCTOR
  //
  // Responsibility:
  //   Initialize the complete runtime state of the bridge.
  //
  // Initialization sequence:
  //   1. Load configuration parameters.
  //   2. Create execution callback groups.
  //   3. Configure ROS 2 QoS profiles.
  //   4. Create standard ROS 2 publishers.
  //   5. Create Unitree video subscriber.
  //   6. Initialize FFmpeg H.264 decoder resources.
  //   7. Create periodic publication timer.
  //   8. Initialize diagnostics.
  //
  // Side effects:
  //   Allocates ROS and FFmpeg runtime resources.
  // ===========================================================================
  FrontVideoBridge()
  : Node("front_video_bridge"),
    rx_msgs_(0),
    decoded_frames_(0),
    published_frames_(0),
    bad_packets_(0),
    consecutive_decode_errors_(0)
  {
    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 1 - LOAD NODE CONFIGURATION PARAMETERS
    // -------------------------------------------------------------------------

    out_topic_ = this->declare_parameter<std::string>(
      "out_topic",
      "/camera/front/image_raw");

    camera_info_topic_ = this->declare_parameter<std::string>(
      "camera_info_topic",
      "/camera/front/camera_info");

    frame_id_ = this->declare_parameter<std::string>(
      "frame_id",
      "front_camera");

    publish_rate_ = this->declare_parameter<double>(
      "publish_rate",
      10.0);

    log_period_s_ = this->declare_parameter<double>(
      "log_period_s",
      2.0);

    width_param_ = this->declare_parameter<int>(
      "width",
      1280);

    height_param_ = this->declare_parameter<int>(
      "height",
      720);

    fx_ = this->declare_parameter<double>(
      "fx",
      900.0);

    fy_ = this->declare_parameter<double>(
      "fy",
      900.0);

    cx_ = this->declare_parameter<double>(
      "cx",
      640.0);

    cy_ = this->declare_parameter<double>(
      "cy",
      360.0);

    max_consecutive_decode_errors_ =
      this->declare_parameter<int>(
      "max_consecutive_decode_errors",
      20);


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 2 - CREATE EXECUTION CALLBACK GROUPS
    //
    // The H.264 decoder may require significantly more processing time than
    // the periodic publication callback.
    //
    // Separate callback groups prevent decoder work from blocking the
    // publication timer when using a MultiThreadedExecutor.
    // -------------------------------------------------------------------------

    decode_callback_group_ =
      this->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    publish_callback_group_ =
      this->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 3 - CONFIGURE ROS 2 QoS PROFILES
    // -------------------------------------------------------------------------

    // Standard sensor-data QoS for ROS camera output topics.
    auto sensor_qos =
      rclcpp::SensorDataQoS();

    // Keep only a small number of compressed video messages.
    //
    // A large queue would allow stale H.264 chunks to accumulate when the
    // decoder temporarily falls behind.
    rclcpp::QoS video_qos(
      rclcpp::KeepLast(5));

    video_qos.reliability(
      RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    video_qos.durability(
      RMW_QOS_POLICY_DURABILITY_VOLATILE);


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 4 - CREATE STANDARD ROS 2 OUTPUT PUBLISHERS
    // -------------------------------------------------------------------------

    image_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>(
      out_topic_,
      sensor_qos);

    camera_info_pub_ =
      this->create_publisher<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_,
      sensor_qos);


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 5 - CREATE UNITREE VIDEO INPUT SUBSCRIBER
    //
    // Input:
    //   /frontvideostream
    //
    // Type:
    //   unitree_go/msg/Go2FrontVideoData
    //
    // Execution:
    //   Assigned to decode_callback_group_ so decoding can execute
    //   independently from the publication timer.
    // -------------------------------------------------------------------------

    rclcpp::SubscriptionOptions subscription_options;

    subscription_options.callback_group =
      decode_callback_group_;

    video_sub_ =
      this->create_subscription<
      unitree_go::msg::Go2FrontVideoData>(
      "/frontvideostream",
      video_qos,
      std::bind(
        &FrontVideoBridge::video_callback,
        this,
        std::placeholders::_1),
      subscription_options);


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 6 - INITIALIZE H.264 / FFMPEG DECODER
    // -------------------------------------------------------------------------

    init_decoder();


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 7 - CREATE PERIODIC IMAGE PUBLICATION TIMER
    //
    // Decoding runs continuously at the incoming stream rate.
    //
    // Publication is deliberately bounded by publish_rate_ so downstream
    // perception components such as YOLO do not need to consume every
    // decoded camera frame.
    // -------------------------------------------------------------------------

    const auto timer_period =
      std::chrono::duration<double>(
      1.0 / std::max(1.0, publish_rate_));

    publish_timer_ =
      this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        timer_period),
      std::bind(
        &FrontVideoBridge::publish_latest,
        this),
      publish_callback_group_);


    // -------------------------------------------------------------------------
    // CONSTRUCTOR STEP 8 - INITIALIZE DIAGNOSTICS AND REPORT NODE STATUS
    // -------------------------------------------------------------------------

    last_log_time_ =
      this->now();

    log_initialization_summary();
  }


  // ===========================================================================
  // NODE DESTRUCTOR
  //
  // Responsibility:
  //   Release all FFmpeg resources allocated during initialization and
  //   runtime decoding.
  //
  // Side effects:
  //   Frees codec, parser, packet, frame and libswscale contexts.
  // ===========================================================================
  ~FrontVideoBridge() override
  {
    cleanup_decoder();
  }


private:
  // ===========================================================================
  // DECODER RESOURCE MANAGEMENT
  //
  // This section owns the complete lifecycle of the FFmpeg H.264 decoder:
  // initialization, cleanup and runtime recovery.
  // ===========================================================================


  // ---------------------------------------------------------------------------
  // INITIALIZE H.264 DECODER RESOURCES
  //
  // Called by:
  //   FrontVideoBridge constructor.
  //
  // Responsibility:
  //   Allocate and initialize:
  //     - H.264 codec
  //     - codec context
  //     - H.264 parser
  //     - reusable AVFrame
  //     - reusable AVPacket
  //
  // Throws:
  //   std::runtime_error if a mandatory decoder resource cannot be created.
  // ---------------------------------------------------------------------------
  void init_decoder()
  {
    // ---------------------------------------------------------------------------
    // CONFIGURE FFMPEG RUNTIME LOGGING
    //
    // FFmpeg can generate a very large amount of H.264 concealment diagnostics.
    // These messages are useful during codec debugging but make the normal ROS 2
    // runtime output unreadable.
    //
    // Only fatal FFmpeg failures are displayed during normal bridge operation.
    // The bridge exposes its own structured diagnostics separately.
    // ---------------------------------------------------------------------------

    av_log_set_level(
      AV_LOG_FATAL);

    const AVCodec *codec =
      avcodec_find_decoder(
      AV_CODEC_ID_H264);

    if (!codec) {
      throw std::runtime_error(
              "Could not find H.264 decoder");
    }

    codec_ctx_ =
      avcodec_alloc_context3(
      codec);

    if (!codec_ctx_) {
      throw std::runtime_error(
              "Could not allocate AVCodecContext");
    }

    if (
      avcodec_open2(
        codec_ctx_,
        codec,
        nullptr) < 0)
    {
      throw std::runtime_error(
              "Could not open H.264 decoder");
    }

    parser_ctx_ =
      av_parser_init(
      AV_CODEC_ID_H264);

    if (!parser_ctx_) {
      throw std::runtime_error(
              "Could not initialize H.264 parser");
    }

    decoded_frame_ =
      av_frame_alloc();

    if (!decoded_frame_) {
      throw std::runtime_error(
              "Could not allocate AVFrame");
    }

    packet_ =
      av_packet_alloc();

    if (!packet_) {
      throw std::runtime_error(
              "Could not allocate AVPacket");
    }
  }


  // ---------------------------------------------------------------------------
  // RELEASE H.264 DECODER RESOURCES
  //
  // Called by:
  //   FrontVideoBridge destructor.
  //
  // Responsibility:
  //   Safely release all FFmpeg and libswscale resources.
  //
  // Side effects:
  //   Internal pointers are reset to nullptr after release.
  // ---------------------------------------------------------------------------
  void cleanup_decoder()
  {
    if (sws_ctx_) {
      sws_freeContext(
        sws_ctx_);

      sws_ctx_ =
        nullptr;
    }

    if (parser_ctx_) {
      av_parser_close(
        parser_ctx_);

      parser_ctx_ =
        nullptr;
    }

    if (decoded_frame_) {
      av_frame_free(
        &decoded_frame_);

      decoded_frame_ =
        nullptr;
    }

    if (packet_) {
      av_packet_free(
        &packet_);

      packet_ =
        nullptr;
    }

    if (codec_ctx_) {
      avcodec_free_context(
        &codec_ctx_);

      codec_ctx_ =
        nullptr;
    }
  }


  // ---------------------------------------------------------------------------
  // RESET DECODER AFTER PERSISTENT STREAM ERRORS
  //
  // Called by:
  //   maybe_recover_decoder().
  //
  // Responsibility:
  //   Flush buffered decoder state after too many consecutive H.264 errors.
  /*
      START
        │
        ▼
      UNSYNCHRONIZED
        │
        ├── type 1 ───────────────> discard
        ├── type 1 ───────────────> discard
        ├── type 1 ───────────────> discard
        │
        └── SPS + PPS + IDR
                  │
                  ▼
            SYNCHRONIZED
                  │
                  ├── P frames
                  ├── P frames
                  ├── IDR
                  └── ...
    */

  // Goal:
  //   Allow recovery when a later valid GOP/keyframe arrives.
  // ---------------------------------------------------------------------------

  void flush_decoder()
  {
    if (codec_ctx_) {
      avcodec_flush_buffers(
        codec_ctx_);
    }

    h264_synchronized_ =
      false;

    consecutive_decode_errors_ =
      0;

    RCLCPP_WARN(
      this->get_logger(),
      "Decoder flushed. Waiting for next SPS + PPS + IDR access unit.");
  }


  // ===========================================================================
  // INPUT / COLLECTION - UNITREE H.264 VIDEO STREAM
  // ===========================================================================


  // ---------------------------------------------------------------------------
  // RECEIVE, PARSE AND DECODE ONE UNITREE VIDEO CHUNK
  //
  // Input:
  //   unitree_go::msg::Go2FrontVideoData
  //
  // Source:
  //   /frontvideostream
  //
  // Processing:
  //   1. Extract video720p bytes.
  //   2. Feed data into the persistent H.264 parser.
  //   3. Submit complete packets to the FFmpeg decoder.
  //   4. Retrieve decoded video frames.
  //   5. Forward decoded frames to the BGR conversion stage.
  //
  // Output:
  //   Updates the latest-frame result buffer through
  //   store_latest_frame_bgr().
  //
  // Side effects:
  //   Updates reception, decoding and error statistics.
  // ---------------------------------------------------------------------------
  void video_callback(
    const unitree_go::msg::Go2FrontVideoData::SharedPtr msg)
  {
    rx_msgs_++;

    if (msg->video720p.empty()) {
      return;
    }

    // ---------------------------------------------------------------------------
    // EXTRACT ANNEX-B H.264 PAYLOAD FROM UNITREE MESSAGE
    //
    // Unitree prepends proprietary bytes before the actual Annex-B H.264 stream.
    // The bridge searches for the first 00 00 00 01 start code and ignores
    // everything before it.
    //
    // Observed structure:
    //
    //   [Unitree proprietary prefix][00 00 00 01][H.264 NAL unit]
    /*
                      video720p
                  │
                  ├── bytes 0..3 : Unitree prefix      X discard
                  │
                  └── bytes 4..  : 00 00 00 01 ...    ✓
                                        │
                                        ▼
                                av_parser_parse2()
                                        │
                                        ▼
                                avcodec decoder
    */
    // This avoids feeding Unitree-specific bytes into the FFmpeg parser.
    // ---------------------------------------------------------------------------

    const auto &payload =
      msg->video720p;

    size_t h264_offset =
      payload.size();

    for (size_t i = 0; i + 4 <= payload.size(); ++i) {
      if (
        payload[i] == 0x00 &&
        payload[i + 1] == 0x00 &&
        payload[i + 2] == 0x00 &&
        payload[i + 3] == 0x01)
      {
        h264_offset =
          i;

        break;
      }
    }

    if (h264_offset == payload.size()) {
      bad_packets_++;
      consecutive_decode_errors_++;

      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Discarding Unitree video payload: "
        "no Annex-B H.264 start code found.");

      maybe_recover_decoder();

      return;
    }


    // ---------------------------------------------------------------------------
    // TEMPORARY H.264 NAL SEQUENCE DIAGNOSTIC
    //
    // Goal:
    //   Inspect every Annex-B NAL unit contained in each Unitree payload.
    //
    // Important NAL types:
    //   1 = non-IDR slice
    //   5 = IDR frame
    //   6 = SEI
    //   7 = SPS
    //   8 = PPS
    // ---------------------------------------------------------------------------

    /*
        if (h264_probe_count_ < 120) {
      h264_probe_count_++;

      size_t search_offset =
        h264_offset;

      size_t nal_index =
        0;

      while (search_offset + 4 < payload.size()) {
        size_t start =
          payload.size();

        for (
          size_t i = search_offset;
          i + 4 <= payload.size();
          ++i)
        {
          if (
            payload[i] == 0x00 &&
            payload[i + 1] == 0x00 &&
            payload[i + 2] == 0x00 &&
            payload[i + 3] == 0x01)
          {
            start =
              i;

            break;
          }
        }

        if (
          start == payload.size() ||
          start + 4 >= payload.size())
        {
          break;
        }

        const int nal_type =
          payload[start + 4] & 0x1F;

        RCLCPP_INFO(
          this->get_logger(),
          "H264 NAL: probe=%zu rx_msg=%zu "
          "nal=%zu offset=%zu payload_size=%zu type=%d",
          h264_probe_count_,
          rx_msgs_,
          nal_index,
          start,
          payload.size(),
          nal_type);

        nal_index++;

        search_offset =
          start + 4;
      }
    }
    */


    // ---------------------------------------------------------------------------
    // DETECT H.264 DECODER SYNCHRONIZATION POINT
    //
    // Before the first valid key access unit is received, discard inter-predicted
    // frames. The first SPS + PPS + IDR access unit establishes a clean decoder
    // starting point.
    // ---------------------------------------------------------------------------

    bool has_sps = false;
    bool has_pps = false;
    bool has_idr = false;

    size_t search_offset =
      h264_offset;

    while (search_offset + 4 < payload.size()) {
      size_t start =
        payload.size();

      for (
        size_t i = search_offset;
        i + 4 <= payload.size();
        ++i)
      {
        if (
          payload[i] == 0x00 &&
          payload[i + 1] == 0x00 &&
          payload[i + 2] == 0x00 &&
          payload[i + 3] == 0x01)
        {
          start =
            i;

          break;
        }
      }

      if (
        start == payload.size() ||
        start + 4 >= payload.size())
      {
        break;
      }

      const int nal_type =
        payload[start + 4] & 0x1F;

      if (nal_type == 7) {
        has_sps = true;
      } else if (nal_type == 8) {
        has_pps = true;
      } else if (nal_type == 5) {
        has_idr = true;
      }

      search_offset =
        start + 4;
    }

    if (!h264_synchronized_) {
      if (!(has_sps && has_pps && has_idr)) {
        unsynchronized_drops_++;
        return;
      }

      h264_synchronized_ =
        true;

      log_h264_synchronization();
    }

    // ---------------------------------------------------------------------------
    // PREPARE CLEAN H.264 PAYLOAD FOR FFMPEG
    //
    // Only the Annex-B portion is forwarded to av_parser_parse2().
    // ---------------------------------------------------------------------------

    const uint8_t *data =
      reinterpret_cast<const uint8_t *>(
      payload.data() + h264_offset);

    int data_size =
      static_cast<int>(
      payload.size() - h264_offset);


    // ---------------------------------------------------------------------------
    // PARSE AND DECODE CLEAN H.264 PAYLOAD
    // ---------------------------------------------------------------------------

    while (data_size > 0) {
      uint8_t *parsed_data =
        nullptr;

      int parsed_size =
        0;

      const int consumed_bytes =
        av_parser_parse2(
        parser_ctx_,
        codec_ctx_,
        &parsed_data,
        &parsed_size,
        data,
        data_size,
        AV_NOPTS_VALUE,
        AV_NOPTS_VALUE,
        0);

      if (consumed_bytes < 0) {
        bad_packets_++;
        consecutive_decode_errors_++;

        maybe_recover_decoder();

        break;
      }

      data +=
        consumed_bytes;

      data_size -=
        consumed_bytes;

      if (parsed_size <= 0) {
        continue;
      }

      packet_->data =
        parsed_data;

      packet_->size =
        parsed_size;

      const int send_result =
        avcodec_send_packet(
        codec_ctx_,
        packet_);

      if (send_result < 0) {
        bad_packets_++;
        consecutive_decode_errors_++;

        maybe_recover_decoder();

        continue;
      }

      bool decoded_any_frame =
        false;

      while (true) {
        const int receive_result =
          avcodec_receive_frame(
          codec_ctx_,
          decoded_frame_);

        if (
          receive_result == AVERROR(EAGAIN) ||
          receive_result == AVERROR_EOF)
        {
          break;
        }

        if (receive_result < 0) {
          bad_packets_++;
          consecutive_decode_errors_++;

          maybe_recover_decoder();

          break;
        }

        decoded_any_frame =
          true;

        consecutive_decode_errors_ =
          0;

        decoded_frames_++;

        store_latest_frame_bgr(
          decoded_frame_);
      }

      if (decoded_any_frame) {
        consecutive_decode_errors_ =
          0;
      }
    }

    maybe_log();
  }


  // ---------------------------------------------------------------------------
  // EVALUATE DECODER RECOVERY POLICY
  //
  // Called by:
  //   video_callback() after parsing or decoding failures.
  //
  // Responsibility:
  //   Flush decoder state only when the configured number of consecutive
  //   errors has been reached.
  // ---------------------------------------------------------------------------
  void maybe_recover_decoder()
  {
    if (
      consecutive_decode_errors_ >=
      max_consecutive_decode_errors_)
    {
      flush_decoder();
    }
  }


  // ===========================================================================
  // PROCESSING - DECODED FRAME CONVERSION AND RESULT BUFFERING
  // ===========================================================================


  // ---------------------------------------------------------------------------
  // CONVERT DECODED FFMPEG FRAME TO BGR24
  //
  // Input:
  //   AVFrame generated by the H.264 decoder.
  //
  // Processing:
  //   Convert the decoder-native pixel format to AV_PIX_FMT_BGR24 using
  //   libswscale.
  //
  // Output:
  //   Stores the latest BGR frame, width and height in the shared result
  //   buffer.
  //
  // Thread safety:
  //   Shared frame state is protected by frame_mutex_.
  // ---------------------------------------------------------------------------
  void store_latest_frame_bgr(
    AVFrame *frame)
  {
    if (
      !frame ||
      frame->width <= 0 ||
      frame->height <= 0)
    {
      return;
    }

    const int width =
      frame->width;

    const int height =
      frame->height;

    if (width == 1280 && height == 720) {
      decoded_720p_frames_++;
    } else if (width == 640 && height == 360) {
      decoded_360p_frames_++;
    } else {
      decoded_other_frames_++;
    }

    sws_ctx_ =
      sws_getCachedContext(
      sws_ctx_,
      width,
      height,
      static_cast<AVPixelFormat>(
        frame->format),
      width,
      height,
      AV_PIX_FMT_BGR24,
      SWS_BILINEAR,
      nullptr,
      nullptr,
      nullptr);

    if (!sws_ctx_) {
      bad_packets_++;

      RCLCPP_WARN(
        this->get_logger(),
        "Could not initialize libswscale context.");

      return;
    }

    std::vector<uint8_t> bgr_frame(
      static_cast<size_t>(
        width * height * 3));

    uint8_t *destination_data[4] = {
      bgr_frame.data(),
      nullptr,
      nullptr,
      nullptr
    };

    int destination_linesize[4] = {
      width * 3,
      0,
      0,
      0
    };

    const int converted_height =
      sws_scale(
      sws_ctx_,
      frame->data,
      frame->linesize,
      0,
      height,
      destination_data,
      destination_linesize);

    if (converted_height <= 0) {
      bad_packets_++;

      RCLCPP_WARN(
        this->get_logger(),
        "libswscale failed to convert decoded frame.");

      return;
    }

    {
      std::lock_guard<std::mutex> lock(
        frame_mutex_);

      latest_bgr_ =
        std::move(
        bgr_frame);

      latest_width_ =
        width;

      latest_height_ =
        height;

      has_frame_ =
        true;
    }
  }


  // ===========================================================================
  // OUTPUT / REPORTING - STANDARD ROS 2 CAMERA TOPICS
  // ===========================================================================


  // ---------------------------------------------------------------------------
  // PUBLISH LATEST AVAILABLE IMAGE AND CAMERA METADATA
  //
  // Input:
  //   Latest decoded BGR frame stored in the internal result buffer.
  //
  // Processing:
  //   1. Copy latest frame under mutex protection.
  //   2. Construct sensor_msgs/Image.
  //   3. Construct sensor_msgs/CameraInfo.
  //   4. Apply identical timestamp and frame_id.
  //
  // Output:
  //   Publishes:
  //     - /camera/front/image_raw
  //     - /camera/front/camera_info
  //
  // Called by:
  //   publish_timer_ at publish_rate_.
  //
  // Side effects:
  //   Updates published_frames_ statistics.
  // ---------------------------------------------------------------------------
  void publish_latest()
  {
    std::vector<uint8_t> local_bgr;

    int width =
      0;

    int height =
      0;

    {
      std::lock_guard<std::mutex> lock(
        frame_mutex_);

      if (!has_frame_) {
        return;
      }

      local_bgr =
        latest_bgr_;

      width =
        latest_width_;

      height =
        latest_height_;
    }

    const auto timestamp =
      this->now();


    // -------------------------------------------------------------------------
    // OUTPUT STEP 1 - BUILD AND PUBLISH sensor_msgs/Image
    // -------------------------------------------------------------------------

    sensor_msgs::msg::Image image_msg;

    image_msg.header.stamp =
      timestamp;

    image_msg.header.frame_id =
      frame_id_;

    image_msg.height =
      static_cast<uint32_t>(
      height);

    image_msg.width =
      static_cast<uint32_t>(
      width);

    image_msg.encoding =
      "bgr8";

    image_msg.is_bigendian =
      0;

    image_msg.step =
      static_cast<
      sensor_msgs::msg::Image::_step_type>(
      width * 3);

    image_msg.data =
      std::move(
      local_bgr);

    image_pub_->publish(
      image_msg);


    // -------------------------------------------------------------------------
    // OUTPUT STEP 2 - BUILD AND PUBLISH sensor_msgs/CameraInfo
    //
    // NOTE:
    //   Current intrinsic parameters are initial/test values.
    //   They must be replaced with calibrated camera intrinsics before using
    //   CameraInfo for metric 3D localization.
    // -------------------------------------------------------------------------

    sensor_msgs::msg::CameraInfo camera_info_msg;

    camera_info_msg.header =
      image_msg.header;

    camera_info_msg.width =
      static_cast<uint32_t>(
      width);

    camera_info_msg.height =
      static_cast<uint32_t>(
      height);

    camera_info_msg.distortion_model =
      "plumb_bob";

    camera_info_msg.d = {
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    };

    camera_info_msg.k = {
      fx_, 0.0, cx_,
      0.0, fy_, cy_,
      0.0, 0.0, 1.0
    };

    camera_info_msg.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0
    };

    camera_info_msg.p = {
      fx_, 0.0, cx_, 0.0,
      0.0, fy_, cy_, 0.0,
      0.0, 0.0, 1.0, 0.0
    };

    camera_info_pub_->publish(
      camera_info_msg);

    published_frames_++;

    maybe_log();
  }


  // ===========================================================================
  // MONITORING / DIAGNOSTICS
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // LOG A VISUAL SECTION HEADER
  //
  // Responsibility:
  //   Improve terminal readability by visually separating major runtime phases.
  //
  // Called by:
  //   Initialization, synchronization and diagnostic reporting functions.
  // ---------------------------------------------------------------------------
  void log_section(
    const std::string &title)
  {
    const std::string separator(
      LOG_SECTION_WIDTH,
      '=');

    RCLCPP_INFO(
      this->get_logger(),
      "\n%s",
      separator.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "  %s",
      title.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "%s",
      separator.c_str());
  }


  // ---------------------------------------------------------------------------
  // LOG A VISUAL SECTION FOOTER
  //
  // Responsibility:
  //   Close a visual runtime section with a consistent separator.
  // ---------------------------------------------------------------------------
  void log_section_footer()
  {
    const std::string separator(
      LOG_SECTION_WIDTH,
      '-');

    RCLCPP_INFO(
      this->get_logger(),
      "%s\n",
      separator.c_str());
  }


  // ---------------------------------------------------------------------------
  // LOG INITIALIZATION SUMMARY
  //
  // Responsibility:
  //   Report the effective runtime configuration of the front video bridge.
  //
  // Called by:
  //   FrontVideoBridge constructor after successful initialization.
  // ---------------------------------------------------------------------------
  void log_initialization_summary()
  {
    log_section("FRONT VIDEO BRIDGE - INITIALIZATION");

    RCLCPP_INFO(
      this->get_logger(),
      "  Input topic          : /frontvideostream");

    RCLCPP_INFO(
      this->get_logger(),
      "  Image output         : %s",
      out_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "  CameraInfo output    : %s",
      camera_info_topic_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "  Frame ID             : %s",
      frame_id_.c_str());

    RCLCPP_INFO(
      this->get_logger(),
      "  Publication rate     : %.1f Hz",
      publish_rate_);

    RCLCPP_INFO(
      this->get_logger(),
      "  Max decode errors    : %d",
      max_consecutive_decode_errors_);

    RCLCPP_INFO(
      this->get_logger(),
      "  H.264 decoder        : FFmpeg / READY");

    log_section_footer();
  }


  // ---------------------------------------------------------------------------
  // LOG H.264 SYNCHRONIZATION SUMMARY
  //
  // Responsibility:
  //   Report when the H.264 stream reaches a valid decoding synchronization
  //   point containing SPS, PPS and IDR NAL units.
  //
  // Called by:
  //   video_callback() when the stream becomes synchronized.
  // ---------------------------------------------------------------------------
  void log_h264_synchronization()
  {
    log_section("H.264 STREAM - SYNCHRONIZED");

    RCLCPP_INFO(
      this->get_logger(),
      "  Synchronization msg  : %zu",
      rx_msgs_);

    RCLCPP_INFO(
      this->get_logger(),
      "  SPS                  : detected");

    RCLCPP_INFO(
      this->get_logger(),
      "  PPS                  : detected");

    RCLCPP_INFO(
      this->get_logger(),
      "  IDR                  : detected");

    RCLCPP_INFO(
      this->get_logger(),
      "  Startup drops        : %zu",
      unsynchronized_drops_);

    RCLCPP_INFO(
      this->get_logger(),
      "  Decoder state        : READY");

    log_section_footer();
  }

  // ---------------------------------------------------------------------------
  // PERIODICALLY REPORT BRIDGE RUNTIME STATISTICS
  //
  // Reports:
  //   rx_msgs_                    - Unitree video messages received.
  //   decoded_frames_             - H.264 frames successfully decoded.
  //   published_frames_           - ROS Image frames published.
  //   bad_packets_                - parser / decoder / conversion failures.
  //   consecutive_decode_errors_  - current decoder failure streak.
  //   unsynchronized_drops_        - valid H.264 payloads intentionally dropped
  //                                  while waiting for SPS + PPS + IDR.
  //
  // Called by:
  //   video_callback()
  //   publish_latest()
  //
  // Rate limiting:
  //   Controlled by log_period_s_.
  // ---------------------------------------------------------------------------
  void maybe_log()
  {
    const auto now =
      this->now();

    if (
      (now - last_log_time_).seconds() <
      log_period_s_)
    {
      return;
    }

    last_log_time_ =
      now;


    RCLCPP_INFO(
      this->get_logger(),
      "[VIDEO] RX_MSG=%zu | DECODED=%zu | PUBLISHED=%zu | "
      "720P=%zu | 360P=%zu | OTHER=%zu | "
      "STARTUP_DROPS=%zu | BAD=%zu | ERRORS=%d",
      rx_msgs_,
      decoded_frames_,
      published_frames_,
      decoded_720p_frames_,
      decoded_360p_frames_,
      decoded_other_frames_,
      unsynchronized_drops_,
      bad_packets_,
      consecutive_decode_errors_);

  }


private:
  // ===========================================================================
  // INTERNAL NODE STATE
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // H.264 STREAM SYNCHRONIZATION STATE
  //
  // The decoder must not start from arbitrary P-frames because they depend on
  // reference frames that existed before the bridge started.
  //
  // Decoding begins only after receiving an access unit containing:
  //   SPS (type 7) + PPS (type 8) + IDR (type 5).
  // ---------------------------------------------------------------------------

  bool h264_synchronized_{false};

  size_t unsynchronized_drops_{0};

  // ---------------------------------------------------------------------------
  // CONFIGURATION - ROS TOPICS, FRAME AND RUNTIME PARAMETERS
  // ---------------------------------------------------------------------------

  std::string out_topic_;
  std::string camera_info_topic_;
  std::string frame_id_;

  double publish_rate_;
  double log_period_s_;

  int width_param_;
  int height_param_;

  double fx_;
  double fy_;
  double cx_;
  double cy_;

  int max_consecutive_decode_errors_;


  // ---------------------------------------------------------------------------
  // ROS 2 INTERFACES - PUBLISHERS, SUBSCRIBER AND TIMER
  // ---------------------------------------------------------------------------

  rclcpp::Publisher<
    sensor_msgs::msg::Image>::SharedPtr
  image_pub_;

  rclcpp::Publisher<
    sensor_msgs::msg::CameraInfo>::SharedPtr
  camera_info_pub_;

  rclcpp::Subscription<
    unitree_go::msg::Go2FrontVideoData>::SharedPtr
  video_sub_;

  rclcpp::TimerBase::SharedPtr
  publish_timer_;


  // ---------------------------------------------------------------------------
  // ROS 2 EXECUTION CONTROL - CALLBACK GROUPS
  //
  // Decoding and publication use separate callback groups so they can run on
  // different worker threads under the MultiThreadedExecutor.
  // ---------------------------------------------------------------------------

  rclcpp::CallbackGroup::SharedPtr
  decode_callback_group_;

  rclcpp::CallbackGroup::SharedPtr
  publish_callback_group_;


  // ---------------------------------------------------------------------------
  // FFMPEG PROCESSING STATE - H.264 DECODER AND PIXEL CONVERSION
  // ---------------------------------------------------------------------------

  AVCodecContext *codec_ctx_{
    nullptr};

  AVCodecParserContext *parser_ctx_{
    nullptr};

  AVFrame *decoded_frame_{
    nullptr};

  AVPacket *packet_{
    nullptr};

  SwsContext *sws_ctx_{
    nullptr};


  // ---------------------------------------------------------------------------
  // RESULT OBJECT - LATEST DECODED BGR FRAME
  //
  // The decoder updates this state whenever a valid frame is obtained.
  //
  // The publisher reads the latest available frame independently at the
  // configured publication rate.
  // ---------------------------------------------------------------------------

  std::mutex frame_mutex_;

  std::vector<uint8_t>
  latest_bgr_;

  int latest_width_{0};
  int latest_height_{0};

  bool has_frame_{false};

  // ---------------------------------------------------------------------------
  // RUNTIME DIAGNOSTICS AND PERFORMANCE COUNTERS
  // ---------------------------------------------------------------------------

  size_t rx_msgs_;
  size_t decoded_frames_;
  size_t published_frames_;
  size_t bad_packets_;

  int consecutive_decode_errors_;

  rclcpp::Time
  last_log_time_;

  static constexpr std::size_t LOG_SECTION_WIDTH = 82;
  size_t decoded_720p_frames_{0};
  size_t decoded_360p_frames_{0};
  size_t decoded_other_frames_{0};
  // ---------------------------------------------------------------------------
  // TEMPORARY H.264 PAYLOAD DIAGNOSTIC STATE
  // ---------------------------------------------------------------------------

  // Counts only non-empty 720p payloads inspected by the H.264 probe.
  // Independent from rx_msgs_ so empty Unitree messages cannot consume the
  // diagnostic window.
  // ---------------------------------------------------------------------------

  size_t h264_probe_count_{0};

};


// =============================================================================
// NODE ORCHESTRATION / PROCESS ENTRY POINT
//
// Responsibility:
//   Initialize ROS 2, instantiate FrontVideoBridge and execute its callbacks.
//
// Execution model:
//   MultiThreadedExecutor with two worker threads.
//
// Worker responsibilities:
//   - Thread 1 can process Unitree H.264 reception and decoding.
//   - Thread 2 can execute periodic standard ROS image publication.
//
// This prevents expensive video decoding from blocking the publication timer.
// =============================================================================

int main(
  int argc,
  char **argv)
{
  rclcpp::init(
    argc,
    argv);

  try {
    const auto node =
      std::make_shared<FrontVideoBridge>();

    rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(),
      2);

    executor.add_node(
      node);

    RCLCPP_INFO(
      node->get_logger(),
      "\n[EXECUTOR] MultiThreadedExecutor started with 2 worker threads.\n");

    executor.spin();

  } catch (
    const std::exception &error)
  {
    std::fprintf(
      stderr,
      "front_video_bridge exception: %s\n",
      error.what());
  }

  rclcpp::shutdown();

  return 0;
}