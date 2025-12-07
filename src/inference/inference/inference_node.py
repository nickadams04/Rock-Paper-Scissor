import time
import rclpy
from rclpy.node import Node
from custom_msgs.msg import AcquisitionMsg, GestureMsg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options


from mediapipe import Image as MPImage
from mediapipe import ImageFormat as MPImageFormat



class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
       
        self.model_path = '/home/nicka/Projects/Rock-Paper-Scissor/src/inference/models/hand_landmarker.task'
       
       # bridge for ROS Image <-> OpenCV conversions
        self.bridge = CvBridge()
        
        # try to load the MediaPipe hand landmarker model
        try:
            base_options = mp_base_options.BaseOptions(model_asset_path=self.model_path)
            opts = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                running_mode=vision.RunningMode.IMAGE
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(opts)
            self.get_logger().info(f'Loaded hand landmarker model: {self.model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load hand landmarker model: {e}')
            self.hand_landmarker = None
        
        # subscription
        self.subscriber = self.create_subscription(AcquisitionMsg, '/image_raw', self.acq_callback, 10)
        self.publisher = self.create_publisher(GestureMsg, '/gesture', 10)
       
    def acq_callback(self, msg: AcquisitionMsg):
        # basic logging
        # self.get_logger().info(f'Got frame {getattr(msg, "global_idx", "unknown")}')        
        image_msg = msg.image
        try:
            # convert ROS image to OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS Image to CV2: {e}')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return
        
        if self.hand_landmarker is None:
            self.get_logger().warning('Hand landmarker not initialized, skipping inference')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return
        
        # MediaPipe expects RGB data for ImageFormat.SRGB
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # preferred path when mediapipe.tasks.python.vision.core.Image is available
        start_time = time.time()
        mp_image = MPImage(image_format=MPImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)
        elapsed_time = time.time() - start_time
        self.get_logger().info(f'Inference time: {elapsed_time:.4f} seconds')        
        # Log the detection results
        if results.hand_landmarks:
            pass
            # self.get_logger().info(f'Detected {len(results.hand_landmarks)} hand(s)')
            # for idx, hand_landmarks in enumerate(results.hand_landmarks):
            #     self.get_logger().info(f'Hand {idx}: {len(hand_landmarks)} landmarks detected')
            #     self.get_logger().info(f'L0: {hand_landmarks[0].x}')
        else:
            self.get_logger().info('No hands detected')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return 
            
        hand_landmarks = results.hand_landmarks[0]
        
        if len(hand_landmarks)==0:
            self.get_logger().info('No keypoints detected')
            out = GestureMsg()
            out.detected = False
            self.publisher.publish(out)
            return 
        else:
            self.get_logger().info(f'{len(hand_landmarks)}')
            
        # De-normalize landmark coordinates
        h, w = cv_image.shape[:2]
        for landmark in hand_landmarks:
            landmark.x = int(landmark.x * w)
            landmark.y = int(landmark.y * h)        
        # Bounding Box estimation
        max_x = 1.1 * max([l.x for l in hand_landmarks])
        max_y = 1.1 * max([l.x for l in hand_landmarks])
        min_x = 0.9 * min([l.x for l in hand_landmarks])
        min_y = 0.9 * min([l.x for l in hand_landmarks])
        
        # self.get_logger().info(f'Bbox:\tmx={min_x}\tMx={max_x}\tmy={min_y}\tMy={max_y}')
        # self.get_logger().info(f'Area: {(max_x - min_x)*(max_y - min_y)}')
            
        out_msg = GestureMsg()
        out_msg.landmarks_x = [l.x for l in hand_landmarks]
        out_msg.landmarks_y = [l.y for l in hand_landmarks]
        
        out_msg.x_max = max_x
        out_msg.x_min = min_x
        out_msg.y_max = max_y
        out_msg.y_min = min_y
        
        out_msg.detected = True
        
        self.publisher.publish(out_msg)
        
                

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # graceful shutdown on Ctrl-C
        node.get_logger().info('Interrupted by user, shutting down.')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()