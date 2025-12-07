import rclpy
from rclpy.node import Node
from custom_msgs.msg import AcquisitionMsg, GestureMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_gesture = None
        self.latest_annotated = None  # keep last annotated for display

        # Correct subscription signatures: (msg_type, topic, callback, qos)
        self.subscriber_image = self.create_subscription(
            AcquisitionMsg, '/image_raw', self.acq_callback, 10
        )
        self.subscriber_gesture = self.create_subscription(
            GestureMsg, '/gesture', self.gesture_callback, 10
        )

        # Publisher for annotated image
        self.publisher_annotated = self.create_publisher(Image, '/image_annotated', 10)

        # UI timer to refresh imshow window ~30 FPS
        self.ui_timer = self.create_timer(1/30.0, self.display_callback)
        cv2.namedWindow('Annotated', cv2.WINDOW_NORMAL)

    def acq_callback(self, msg: AcquisitionMsg):
        # Convert ROS Image to OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV2: {e}')
            return

        self.latest_frame = cv_image
        # Try to annotate if we have a gesture already
        self.publish_annotated()

    def gesture_callback(self, msg: GestureMsg):
        self.latest_gesture = msg
        # Try to annotate if we have a frame already
        self.publish_annotated()

    def publish_annotated(self):
        if self.latest_frame is None or self.latest_gesture is None:
            return

        frame = self.latest_frame.copy()
        gesture = self.latest_gesture

        # Draw landmarks if present
        if getattr(gesture, 'detected', False) and getattr(gesture, 'landmarks_x', None) and getattr(gesture, 'landmarks_y', None):
            for x, y in zip(gesture.landmarks_x, gesture.landmarks_y):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw bbox if available
        try:
            x_min = int(getattr(gesture, 'x_min', 0))
            y_min = int(getattr(gesture, 'y_min', 0))
            x_max = int(getattr(gesture, 'x_max', 0))
            y_max = int(getattr(gesture, 'y_max', 0))
            if x_max > x_min and y_max > y_min:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        except Exception as e:
            self.get_logger().warn(f'Failed to draw bbox: {e}')
            
        # Draw prediction
        if getattr(gesture, 'detected', False):
            gesture_idx = getattr(gesture, 'player_gesture', 3)
            gesture_text = [
                'Rock', 'Paper', 'Scissors', 'Unknown'
            ][gesture_idx]
            confidence = getattr(gesture, 'confidence', 0.0)
            text = f'{gesture_text}: {confidence:.2f}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # Store and publish annotated image
        self.latest_annotated = frame
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_annotated.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')

    def display_callback(self):
        # Show the latest annotated frame; fall back to raw
        frame = self.latest_annotated if self.latest_annotated is not None else self.latest_frame
        if frame is None:
            return
        try:
            cv2.imshow('Annotated', frame)
            # waitKey must be called to update window; 1 ms is enough
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Closing visualization window.')
                cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().warn(f'Failed to imshow: {e}')

    def destroy_node(self):
        # Ensure window is closed on shutdown
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user, shutting down.')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()