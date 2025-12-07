import rclpy
from rclpy.node import Node
from custom_msgs.msg import AcquisitionMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class AcquisitionNode(Node):
    def __init__(self):
        super().__init__('acquisition_node')
        # self.get_logger().info("Entering Callback")
        # publish the custom message type (AcquisitionMsg) on /image_raw
        self.publisher = self.create_publisher(AcquisitionMsg, '/image_raw', 10)
        self.bridge = CvBridge()
        # self.get_logger().info("Entering Callback")
        # Webcam capture from cv2
        self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

        # Force MJPEG
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Reasonable resolution & FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)


        
        # Set counter
        self.global_index = 0
        # self.get_logger().info("Entering Callback")
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        # self.get_logger().info("Entering Callback")
        # Publish at ~30Hz
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        # self.get_logger().info("Entering Callback")
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to capture frame")
            return

        # self.get_logger().info("Acquired image")
        img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg = AcquisitionMsg()
        msg.image = img
        msg.global_idx = self.global_index
        self.global_index += 1
        # publish the custom message
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AcquisitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # graceful shutdown on Ctrl-C
        node.get_logger().info('Interrupted by user, shutting down.')
    finally:
        # release camera and cleanup
        try:
            if hasattr(node, 'cap') and node.cap is not None:
                node.cap.release()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()