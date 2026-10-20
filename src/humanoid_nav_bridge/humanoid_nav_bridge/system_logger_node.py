#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SystemLoggerNode(Node):
    def __init__(self):
        super().__init__("system_logger_node")

        self.sub = self.create_subscription(
            String,
            "/system/log_event",
            self.on_log_event,
            10
        )

        self.pub = self.create_publisher(
            String,
            "/system/log",
            10
        )

        self.get_logger().info("SystemLoggerNode started: /system/log_event -> /system/log")

    def on_log_event(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            data = {
                "level": "info",
                "source": "unknown",
                "message": msg.data
            }

        level = str(data.get("level", "info")).lower()
        source = str(data.get("source", "unknown"))
        message = str(data.get("message", ""))

        text = f"[{source}] {message}"

        if level == "warn" or level == "warning":
            self.get_logger().warn(text)
        elif level == "error":
            self.get_logger().error(text)
        elif level == "debug":
            self.get_logger().debug(text)
        else:
            self.get_logger().info(text)

        out = String()
        out.data = json.dumps({
            "level": level,
            "source": source,
            "message": message,
            "stamp_sec": self.get_clock().now().nanoseconds * 1e-9
        })

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SystemLoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()