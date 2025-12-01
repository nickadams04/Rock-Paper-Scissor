import rclpy
from rclpy.node import Node
from custom_msgs.msg import AcquisitionMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class AcquisitionNode(Node):
    def __init__(self):
        super().__init__('acquisition_node')
        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        
        # Webcam capture from cv2
        self.cap = cv2.VideoCapture(0)
        
        # Set counter
        self.global_index = 0
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Publish at ~30Hz
        self.timer = self.create_timer(1/30.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg = AcquisitionMsg()
        msg.image = img
        msg.global_index = self.global_index
        self.global_index += 1
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AcquisitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()